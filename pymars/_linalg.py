"""Linear algebra of the fit: Gram-Schmidt, the collinearity rules, least squares
and the updates of the R factor in the pruning pass.

Spec: ``docs/algorithm.md``, section "Linear algebra contract" (LA-1 to LA-7),
and the rules that call it: FWD-3, FWD-4 and FWD-11 (forward pass), PRUNE-3,
PRUNE-8 and PRUNE-9 (pruning pass), W-3 to W-5 (weights). The plan's section
"Fast path" sets the methods.

Weights. Weighted least squares works on rescaled rows: with case weights w, a
column a becomes ã = √w·a, so the weighted inner product ⟨a, b⟩_w = Σ w_i a_i b_i
is the plain dot product ãᵀb̃. A function with a ``w`` argument takes unscaled
columns and rescales them itself; ``w=None`` means w_i = 1 exactly (W-5). A row
with zero weight adds nothing to a sum, and the tests of constant data skip it,
since such a row is not a case (W-3). ``orthogonalize`` and ``gram_schmidt``
work in the rescaled space, where Q (n x r) has orthonormal columns.

Numerics. All arithmetic is float64 and no function writes into its inputs.
Every tolerance is relative to a scale that its rule names, and a test whether
data are constant or zero compares the values exactly ("Conventions for all
rules"). Memory is O(n·(M + K)) for n rows, M columns and K responses.

Public functions:

- ``orthogonalize``, ``gram_schmidt``: Gram-Schmidt applied twice (LA-1).
- ``collinearity_tolerance``, ``collinearity_ratio``, ``knot_rejected``: the
  collinearity test of a candidate knot (LA-3).
- ``weighted_variances``, ``pair_search``: the kind of a search (LA-7).
- ``independent_columns``, ``lm_fit``: least squares with the dependent columns
  of LA-4, the analogue of R's ``lm.fit`` (FWD-11, PRUNE-8).
- ``r_factor``, ``prefix_rss``, ``drop_costs``, ``move_column``: the R factor of
  the pruning pass and its updates (PRUNE-3, PRUNE-9).
"""

from __future__ import annotations

import math
import operator
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import scipy.linalg

__all__ = [
    "COLLINEARITY_LAST_EARLY_STEP",
    "COLLINEARITY_TOL_EARLY",
    "COLLINEARITY_TOL_LATE",
    "LM_TOL",
    "PAIR_SEARCH_FACTOR",
    "GramSchmidt",
    "LmFit",
    "RFactor",
    "collinearity_ratio",
    "collinearity_tolerance",
    "drop_costs",
    "gram_schmidt",
    "independent_columns",
    "knot_rejected",
    "lm_fit",
    "move_column",
    "orthogonalize",
    "pair_search",
    "prefix_rss",
    "r_factor",
    "weighted_variances",
]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

#: LA-4: a column is dependent when the norm of its part orthogonal to the kept
#: earlier columns is less than LM_TOL times its norm (not centered).
LM_TOL = 1e-7
#: LA-3: the collinearity tolerance tau in forward steps 1 to 7 (counted from 1).
COLLINEARITY_TOL_EARLY = 0.01
#: LA-3: the tolerance tau from forward step 8 on.
COLLINEARITY_TOL_LATE = 1e-5
#: LA-3: the last forward step that uses COLLINEARITY_TOL_EARLY.
COLLINEARITY_LAST_EARLY_STEP = 7
#: LA-7: a pair search needs A_w ≥ PAIR_SEARCH_FACTOR·Π sigma_v².
PAIR_SEARCH_FACTOR = 0.01


class GramSchmidt(NamedTuple):
    """The result of ``gram_schmidt`` for a new column v against Q."""

    q: FloatArray | None
    """The new orthonormal column v⊥/‖v⊥‖, or None when v⊥ is exactly 0."""
    coef: FloatArray
    """Qᵀv, shape (r,): the coefficients of v on the columns of Q."""
    norm: float
    """‖v⊥‖, the norm of the part of v orthogonal to Q."""


class LmFit(NamedTuple):
    """The result of ``lm_fit``."""

    coef: FloatArray
    """Shape (M,) or (M, K); 0.0 for a dependent column (LA-4)."""
    kept: BoolArray
    """Shape (M,): True for the columns that LA-4 keeps."""
    residuals: FloatArray
    """Y - A·coef, unscaled, with the shape of Y."""
    rss: float
    """Σ_k Σ_i w_i·residuals_ik², the weighted RSS of ``coef`` (PRUNE-8)."""


class RFactor(NamedTuple):
    """The result of ``r_factor``: A·diag(√w) = QR is never formed as Q."""

    R: FloatArray
    """Shape (M, M), upper triangular: the R factor of the rescaled columns."""
    Z: FloatArray
    """Shape (M, K): Qᵀ(√w·Y), the responses in the basis Q."""
    rss: float
    """The weighted RSS of Y on all M columns, summed over the responses."""


def _weights(w: npt.ArrayLike | None, n: int) -> FloatArray:
    """Return w as a new float64 array, or ones when w is None (W-5)."""
    if w is None:
        return np.ones(n)
    w = np.array(w, dtype=np.float64)
    if w.shape != (n,):
        raise ValueError(f"w must have shape ({n},), not {w.shape}")
    if not (np.all(np.isfinite(w)) and np.all(w >= 0.0)):
        raise ValueError("w must be finite and nonnegative")
    return w


def _matrix(A: npt.ArrayLike, name: str) -> FloatArray:
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"{name} must be 2-D, not {A.ndim}-D")
    return A


def _columns(Y: npt.ArrayLike, n: int, name: str) -> tuple[FloatArray, bool]:
    """Return Y as an (n, K) array and whether it was 1-D."""
    Y = np.asarray(Y, dtype=np.float64)
    one_d = Y.ndim == 1
    if one_d:
        Y = Y[:, None]
    if Y.ndim != 2 or Y.shape[0] != n:
        raise ValueError(f"{name} must have shape ({n},) or ({n}, K)")
    return Y, one_d


def orthogonalize(Q: npt.ArrayLike, V: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Remove from V its part in the span of Q, by Gram-Schmidt applied twice.

    Classical Gram-Schmidt with one reorthogonalization: the second pass removes
    what rounding left in the span after the first, so the result is orthogonal
    to Q to working precision [plan: Fast path, "orthogonalize it twice"]. When Q
    is an orthonormal basis of the √w-scaled columns of B and V = √w·Y, the
    result is √w times the weighted least-squares residual of Y on B, so its
    squared norm is the RSS of LA-1.

    Args:
        Q: shape (n, r) with orthonormal columns; r may be 0.
        V: shape (n,) or (n, k), in the rescaled space.

    Returns:
        (V_perp, C): V_perp = V - Q·C, with the shape of V, and C = QᵀV summed
        over both passes, shape (r,) or (r, k).

    Complexity: O(n·r·k) time and O(n·k) memory.
    """
    Q = _matrix(Q, "Q")
    V = np.asarray(V, dtype=np.float64)
    C = Q.T @ V
    V1 = V - Q @ C
    C2 = Q.T @ V1
    return V1 - Q @ C2, C + C2


def gram_schmidt(Q: npt.ArrayLike, v: npt.ArrayLike) -> GramSchmidt:
    """Orthonormalize a new column v against Q (Gram-Schmidt applied twice).

    The forward pass appends ``q`` to its basis Q of the √w-scaled columns of
    B [plan: Fast path]. With v = √w·(b·x), ``norm``² is A_w of LA-7, the
    weighted RSS of b·x regressed on B. The function does not decide whether v
    is dependent: FWD-11 applies LA-4 with ``independent_columns`` when the pass
    stops, and the rules of the search (LA-3, LA-7) keep a chosen column away
    from the span.

    Args:
        Q: shape (n, r) with orthonormal columns; r may be 0.
        v: shape (n,), in the rescaled space.

    Returns:
        GramSchmidt(q, coef, norm); q is None only when v⊥ is exactly 0.

    Complexity: O(n·r) time and O(n) memory.
    """
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError(f"v must be 1-D, not {v.ndim}-D")
    perp, coef = orthogonalize(Q, v)
    norm = float(scipy.linalg.norm(perp))
    return GramSchmidt(perp / norm if norm > 0.0 else None, coef, norm)


def collinearity_tolerance(step: int) -> float:
    """Return the tolerance tau of LA-3 for forward step ``step``, counted from 1.

    tau is 0.01 while the pass has taken at most 6 steps, so in steps 1 to 7,
    and 1e-5 from step 8 on (LA-3, a quirk of earth that pymars copies).

    Complexity: O(1).
    """
    step = operator.index(step)
    if step < 1:
        raise ValueError(f"step counts from 1, not {step}")
    if step <= COLLINEARITY_LAST_EARLY_STEP:
        return COLLINEARITY_TOL_EARLY
    return COLLINEARITY_TOL_LATE


def collinearity_ratio(
    Q: npt.ArrayLike, h: npt.ArrayLike, w: npt.ArrayLike | None = None
) -> float:
    """Return rho = ‖h - P_G h‖_w² / ‖h - h̄‖_w², 1 - R² of h regressed on G (LA-3).

    h̄ is the weighted mean of h, with N = math.fsum(w) (W-4). G holds the
    current columns in a single-hinge search, and also b·x in a pair search
    (LA-7); it must contain the intercept, so rho ≤ 1 up to rounding. When h is
    constant over the cases (Conventions: all its values with positive weight
    are equal), the ratio is 0/0 and the function returns 0.0, so a constant h
    is rejected like any rho < tau (``knot_rejected``). h is first multiplied by a
    power of 2, which changes no bit of rho and keeps the sums of squares in the
    range of float64. The fast scan computes rho for every knot from sums (plan:
    Fast path); this function computes it from the explicit column.

    Args:
        Q: shape (n, r), an orthonormal basis of the √w-scaled columns of G.
        h: shape (n,), the unscaled candidate column, such as b·(x - t)₊.
        w: shape (n,), the case weights, or None for unit weights.

    Returns:
        rho, a float in [0, 1] up to rounding.

    Complexity: O(n·r) time and O(n) memory.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.ndim != 1:
        raise ValueError(f"h must be 1-D, not {h.ndim}-D")
    wv = _weights(w, h.shape[0])
    cases = h if w is None else h[wv > 0.0]
    if cases.size == 0 or np.all(cases == cases[0]):
        return 0.0
    hs = np.ldexp(h, -int(np.frexp(np.max(np.abs(h)))[1]))
    mean = float(wv @ hs) / math.fsum(wv)
    d = hs - mean
    centered = float(wv @ (d * d))
    if centered == 0.0:  # only by underflow with extreme weights (EDGE-6)
        return 0.0
    perp, _ = orthogonalize(Q, np.sqrt(wv) * hs)
    return float(perp @ perp) / centered


def knot_rejected(ratio: npt.ArrayLike, step: int) -> bool | npt.NDArray[np.bool_]:
    """Apply the collinearity test of LA-3: True where rho < tau(step).

    A ratio of 0.0 marks a constant hinge column (``collinearity_ratio``), so
    it is rejected too. A ratio equal to tau is kept. ``ratio`` may be one value
    or an array of ratios, one per knot.

    Complexity: O(len(ratio)).
    """
    rejected = np.asarray(ratio, dtype=np.float64) < collinearity_tolerance(step)
    return bool(rejected) if rejected.ndim == 0 else rejected


def weighted_variances(X: npt.ArrayLike, w: npt.ArrayLike | None = None) -> FloatArray:
    """Return sigma_v² for every column v of X, with divisor N (Notation; LA-7).

    sigma_v² = Σ w_i (x_iv - x̄_v)² / N, where x̄_v is the weighted mean and
    N = math.fsum(w) (W-4). A column that is constant over the cases
    (Conventions: all its values with positive weight are equal) gets exactly
    0.0. Rows with zero weight are left out (W-3).

    Args:
        X: shape (n, p).
        w: shape (n,), or None for unit weights.

    Returns:
        A new float64 array of shape (p,).

    Complexity: O(n·p) time and memory.
    """
    X = _matrix(X, "X")
    wv = _weights(w, X.shape[0])
    if w is not None:
        keep = wv > 0.0
        X, wv = X[keep], wv[keep]
    if X.shape[0] == 0:
        raise ValueError("X has no row with positive weight")
    N = math.fsum(wv)
    D = X - (wv @ X) / N
    var = (wv @ (D * D)) / N
    var[(X[:1] == X).all(axis=0)] = 0.0
    return var


def pair_search(resid_ss: float, variances: npt.ArrayLike) -> bool:
    """Decide the kind of a search for a parent b and a covariate x (LA-7).

    pymars runs a pair search when A_w ≥ 0.01·Π_{v∈V} sigma_v², and a single-hinge
    search otherwise, and always when some v ∈ V is constant (LA-7, a pymars
    departure that makes the rule independent of the units of x).

    Args:
        resid_ss: A_w, the weighted RSS of b·x regressed on the current columns,
            for example ``gram_schmidt(Q, sqrt(w) * b * x).norm ** 2``.
        variances: sigma_v² from ``weighted_variances`` for the covariates V of the
            term b·x (those of the parent and x); 0.0 marks a constant one.

    Returns:
        True for a pair search, False for a single-hinge search.

    Complexity: O(len(variances)).
    """
    v = np.asarray(variances, dtype=np.float64).ravel()
    if v.size == 0:
        raise ValueError("variances must hold sigma² of at least the covariate x")
    if np.any(v == 0.0):
        return False
    return bool(resid_ss >= PAIR_SEARCH_FACTOR * math.prod(v.tolist()))


def _independent(As: FloatArray) -> BoolArray:
    """Return the columns of As that LA-4 keeps, taken in order.

    Each column is orthogonalized twice against the kept earlier ones. It is
    dependent when it is exactly 0, or when the norm of its orthogonal part is
    less than LM_TOL times its norm. Each column is first multiplied by a power
    of 2, which changes neither the test nor the kept set.

    Complexity: O(n·M·r) time for rank r, and O(n·r) memory.
    """
    n, M = As.shape
    Q = np.empty((n, min(n, M)))
    kept = np.zeros(M, dtype=bool)
    r = 0
    for j in range(M):
        a = As[:, j]
        if not np.any(a) or r == n:
            continue
        a = np.ldexp(a, -int(np.frexp(np.max(np.abs(a)))[1]))
        perp, _ = orthogonalize(Q[:, :r], a)
        norm = scipy.linalg.norm(perp)
        if norm < LM_TOL * scipy.linalg.norm(a):
            continue
        Q[:, r] = perp / norm
        r += 1
        kept[j] = True
    return kept


def independent_columns(A: npt.ArrayLike, w: npt.ArrayLike | None = None) -> BoolArray:
    """Return which columns of A are independent by the rule of LA-4.

    The columns are taken in their order. A column is dependent when the w-norm
    of its part orthogonal to the kept earlier columns is less than 1e-7 times
    its w-norm, not centered; a column of zeros is dependent too, as in R's
    ``lm.fit``. FWD-11 applies this rule to the √w-scaled columns of the forward
    terms when the pass stops.

    Args:
        A: shape (n, M), unscaled columns.
        w: shape (n,), or None for unit weights.

    Returns:
        A bool array of shape (M,), True for a kept column.

    Complexity: O(n·M·r) time for rank r, and O(n·M) memory.
    """
    A = _matrix(A, "A")
    return _independent(np.sqrt(_weights(w, A.shape[0]))[:, None] * A)


def lm_fit(A: npt.ArrayLike, Y: npt.ArrayLike, w: npt.ArrayLike | None = None) -> LmFit:
    """Weighted least squares with the dependent columns of LA-4 set to 0.

    This is the analogue of R's ``lm.fit`` and ``lm.wfit`` at their default
    tolerance, whose coefficient NA is 0.0 here (LA-4, PRUNE-8). The kept
    columns come from ``independent_columns``; the coefficients on them come
    from a Householder QR of the √w-scaled kept columns and a triangular solve.
    The residuals are those of the unscaled data, and ``rss`` is the weighted
    RSS of the returned coefficients, as PRUNE-8 defines the final ``rss``.

    Args:
        A: shape (n, M), the unscaled columns, such as the selected columns B_S.
        Y: shape (n,) or (n, K).
        w: shape (n,), or None for unit weights.

    Returns:
        LmFit(coef, kept, residuals, rss); coef has shape (M,) for a 1-D Y and
        (M, K) otherwise.

    Complexity: O(n·M·(M + K)) time and O(n·(M + K)) memory.
    """
    A = _matrix(A, "A")
    n, M = A.shape
    Y2, one_d = _columns(Y, n, "Y")
    wv = _weights(w, n)
    sw = np.sqrt(wv)[:, None]
    As = sw * A
    kept = _independent(As)
    coef = np.zeros((M, Y2.shape[1]))
    if kept.any():
        Qk, Rk = np.linalg.qr(As[:, kept])
        coef[kept] = scipy.linalg.solve_triangular(Rk, Qk.T @ (sw * Y2))
    residuals = Y2 - A @ coef
    rss = float(np.sum(wv[:, None] * np.square(residuals)))
    if one_d:
        return LmFit(coef[:, 0], kept, residuals[:, 0], rss)
    return LmFit(coef, kept, residuals, rss)


def r_factor(
    A: npt.ArrayLike, Y: npt.ArrayLike, w: npt.ArrayLike | None = None
) -> RFactor:
    """Return the R factor of the √w-scaled columns, the responses Z and the RSS.

    One Householder QR of [√w·A, √w·Y] gives R, Z = Qᵀ(√w·Y) and, from its last
    rows, the RSS of Y on all columns; the n x M factor Q is never formed. The
    pruning pass then works on R and Z alone (PRUNE-9). The columns must be
    linearly independent, as the kept forward terms are (FWD-11, PRUNE-2), so
    that R is nonsingular; ``prefix_rss`` and ``drop_costs`` rely on it.

    Args:
        A: shape (n, M) with M ≤ n, the unscaled columns in the working order.
        Y: shape (n,) or (n, K).
        w: shape (n,), or None for unit weights.

    Returns:
        RFactor(R, Z, rss) with R of shape (M, M) and Z of shape (M, K).

    Complexity: O(n·(M + K)²) time and O(n·(M + K)) memory.
    """
    A = _matrix(A, "A")
    n, M = A.shape
    if n < M:
        raise ValueError(f"A has {M} columns and only {n} rows")
    Y2, _ = _columns(Y, n, "Y")
    sw = np.sqrt(_weights(w, n))[:, None]
    F = np.linalg.qr(np.hstack([sw * A, sw * Y2]), mode="r")
    return RFactor(F[:M, :M], F[:M, M:], float(np.sum(np.square(F[M:, M:]))))


def prefix_rss(Z: npt.ArrayLike, rss: float) -> FloatArray:
    """Return the RSS of each prefix of the working order (LA-1, PRUNE-3).

    Entry m - 1 is the RSS of the first m columns, m = 1, …, M: the full RSS
    plus Σ_{j ≥ m} ‖Z_j‖² over the later rows of Z, summed over the responses.
    The sum has no cancellation, since every term is at least 0.

    Args:
        Z: shape (M,) or (M, K), from ``r_factor`` or ``move_column``.
        rss: the RSS on all M columns, from ``r_factor``.

    Returns:
        A new float64 array of shape (M,).

    Complexity: O(M·K).
    """
    Z = np.asarray(Z, dtype=np.float64)
    z2 = np.square(Z) if Z.ndim == 1 else np.sum(np.square(Z), axis=1)
    out = np.full(z2.shape[0], float(rss))
    out[:-1] += np.cumsum(z2[::-1])[::-1][1:]
    return out


def drop_costs(R: npt.ArrayLike, Z: npt.ArrayLike, pos: int) -> FloatArray:
    """Return the RSS increase when each of the first ``pos`` columns is removed.

    Entry i is RSS(first pos columns without column i) - RSS(first pos
    columns), summed over the responses, for i = 0, …, pos - 1; step 1 of the
    PRUNE-3 stages compares these for the positions after the intercept. With
    R_p the leading pos x pos block of R and β = R_p⁻¹ Z[:pos], the increase is
    Σ_k β_ik² / [(R_pᵀR_p)⁻¹]_ii, and [(R_pᵀR_p)⁻¹]_ii is the squared norm of
    row i of R_p⁻¹. R_p must be nonsingular (``r_factor``).

    Args:
        R: shape (M, M), upper triangular.
        Z: shape (M,) or (M, K).
        pos: the number of leading columns, 1 ≤ pos ≤ M.

    Returns:
        A new float64 array of shape (pos,), each entry at least 0.

    Complexity: O(pos³ + pos²·K).
    """
    R = _matrix(R, "R")
    Z = np.asarray(Z, dtype=np.float64)
    pos = operator.index(pos)
    if not 1 <= pos <= R.shape[1]:
        raise ValueError(f"pos must be in 1..{R.shape[1]}, not {pos}")
    Rinv = scipy.linalg.solve_triangular(R[:pos, :pos], np.eye(pos))
    beta = Rinv @ Z[:pos]
    num = np.square(beta) if beta.ndim == 1 else np.sum(np.square(beta), axis=1)
    return num / np.sum(np.square(Rinv), axis=1)


def move_column(
    R: npt.ArrayLike, Z: npt.ArrayLike, i: int, j: int
) -> tuple[FloatArray, FloatArray]:
    """Return R and Z for the order with column i moved to position j.

    The columns between positions i and j shift by one place toward i, as in
    step 2 of a PRUNE-3 stage (move the removed term to position pos). Only rows
    min(i, j) to max(i, j) change: one Householder QR of that block restores the
    triangle, and its orthogonal factor rotates the same rows of Z, so the RSS of
    every prefix stays exact (PRUNE-9, a downdate in place of a refit).

    Args:
        R: shape (M, M), upper triangular.
        Z: shape (M,) or (M, K).
        i: the position of the column to move, 0 ≤ i < M.
        j: its new position, 0 ≤ j < M.

    Returns:
        (R', Z'), new arrays with the shapes of R and Z.

    Complexity: O(h²·(M + K)) with h = |i - j| + 1.
    """
    R = _matrix(R, "R")
    Z = np.asarray(Z, dtype=np.float64)
    M = R.shape[1]
    i, j = operator.index(i), operator.index(j)
    if not (0 <= i < M and 0 <= j < M):
        raise ValueError(f"i and j must be in 0..{M - 1}, not {i} and {j}")
    order = list(range(M))
    order.insert(j, order.pop(i))
    R2 = R[:, order]
    Z2 = Z.copy()
    lo, hi = min(i, j), max(i, j)
    if hi > lo:
        Qb, Rb = np.linalg.qr(R2[lo : hi + 1, lo:])
        R2[lo : hi + 1, lo:] = Rb
        Z2[lo : hi + 1] = Qb.T @ Z[lo : hi + 1]
    return R2, Z2
