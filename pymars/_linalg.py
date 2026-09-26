"""Linear algebra of the fit: Gram-Schmidt, the collinearity rules, least squares
and the updates of the R factor in the pruning pass.

Spec: ``docs/algorithm.md``, "Linear algebra contract" (LA-1 to LA-7), and the
rules that call it: FWD-3, FWD-4 and FWD-11 (forward pass), PRUNE-3, PRUNE-8 and
PRUNE-9 (pruning pass), W-3 to W-5 (weights) and EDGE-6. The plan's section
"Fast path" sets the methods.

Weights. Weighted least squares works on rescaled rows: with case weights w, a
column a becomes √w·a, so that ⟨a, b⟩_w = Σ w_i a_i b_i is a plain dot product.
A function with a ``w`` argument takes unscaled columns and rescales them;
``w=None`` means w_i = 1 exactly (W-5). A row with zero weight adds nothing to
a sum, and the tests of constant data leave it out, since it is not a case
(W-3). ``orthogonalize`` and ``gram_schmidt`` work in the rescaled space, where
Q (n x r) has orthonormal columns.

Numerics. float64 throughout; no function writes into its inputs; every
tolerance is relative to a scale that its rule names, and a test whether data
are constant or zero compares the values exactly ("Conventions for all rules").
Y enters only through products and sums, so multiplying Y by a power of 2, as
EDGE-6 does, changes no bit of a result beyond that factor. Memory is
O(n·(M + K)) for n rows, M columns and K responses.

Public functions, for ``_scan``, ``_forward``, ``_pruning`` and ``_core``:

- ``orthogonalize``, ``gram_schmidt``: Gram-Schmidt applied twice (LA-1).
- ``collinearity_tolerance``, ``collinearity_ratio``, ``knot_rejected``: the
  collinearity test of a candidate knot (LA-3).
- ``weighted_variances``, ``pair_search``: the kind of a search (LA-7).
- ``independent_columns``, ``lm_fit``: least squares with the dependent columns
  of LA-4, the analogue of R's ``lm.fit`` (FWD-11, PRUNE-8).
- ``r_factor``, ``prefix_rss``, ``drop_costs``, ``move_column``: the R factor of
  the pruning pass and its downdates (PRUNE-3, PRUNE-9).
"""

from __future__ import annotations

import math
import operator
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import scipy.linalg

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

#: LA-4: a column is dependent when the norm of its part orthogonal to the kept
#: earlier columns is less than LM_TOL times its norm, not centered.
LM_TOL = 1e-7
#: LA-3: the collinearity tolerance tau in forward steps 1 to 7.
COLLINEARITY_TOL_EARLY = 0.01
#: LA-3: tau from forward step 8 on.
COLLINEARITY_TOL_LATE = 1e-5
#: LA-3: the last forward step, counted from 1, with COLLINEARITY_TOL_EARLY.
COLLINEARITY_LAST_EARLY_STEP = 7
#: LA-7: a pair search needs A_w ≥ PAIR_SEARCH_FACTOR·Π sigma_v².
PAIR_SEARCH_FACTOR = 0.01


class GramSchmidt(NamedTuple):
    """``gram_schmidt(Q, v)``: ``q`` = v⊥/‖v⊥‖, or None when v⊥ is exactly 0;
    ``coef`` = Qᵀv, shape (r,); ``norm`` = ‖v⊥‖."""

    q: FloatArray | None
    coef: FloatArray
    norm: float


class LmFit(NamedTuple):
    """``lm_fit(A, Y, w)``: ``coef``, shape (M,) or (M, K), 0.0 for a dependent
    column; ``kept``, shape (M,), True for a column that LA-4 keeps;
    ``residuals`` = Y - A·coef, unscaled, with the shape of Y; ``rss`` =
    Σ_k Σ_i w_i·residuals_ik² (PRUNE-8)."""

    coef: FloatArray
    kept: BoolArray
    residuals: FloatArray
    rss: float


class RFactor(NamedTuple):
    """``r_factor(A, Y, w)``: with √w·A = QR, ``R`` (M, M) upper triangular;
    ``Z`` = Qᵀ(√w·Y), shape (M, K); ``rss``, the weighted RSS of Y on all M
    columns, summed over the responses."""

    R: FloatArray
    Z: FloatArray
    rss: float


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


def _responses(Y: npt.ArrayLike, n: int) -> tuple[FloatArray, bool]:
    """Return Y as an (n, K) array, and whether it was 1-D."""
    Y = np.asarray(Y, dtype=np.float64)
    one_d = Y.ndim == 1
    if one_d:
        Y = Y[:, None]
    if Y.ndim != 2 or Y.shape[0] != n:
        raise ValueError(f"Y must have shape ({n},) or ({n}, K)")
    return Y, one_d


def orthogonalize(Q: npt.ArrayLike, V: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Remove from V its part in the span of Q, by Gram-Schmidt applied twice.

    The second pass removes what rounding left in the span after the first, so
    the result is orthogonal to Q to working precision [plan: Fast path,
    "orthogonalize it twice"]. With Q an orthonormal basis of the √w-scaled
    columns of B and V = √w·Y, the result is √w times the residual of Y on B,
    and its squared norm is RSS(B) of LA-1.

    Q is (n, r) with orthonormal columns, r ≥ 0; V is (n,) or (n, k). Returns
    (V - Q·C, C) with C = QᵀV summed over both passes, of shape (r,) or (r, k).
    Complexity: O(n·r·k) time, O(n·k) memory.
    """
    Q = _matrix(Q, "Q")
    V = np.asarray(V, dtype=np.float64)
    C = Q.T @ V
    V1 = V - Q @ C
    C2 = Q.T @ V1
    return V1 - Q @ C2, C + C2


def gram_schmidt(Q: npt.ArrayLike, v: npt.ArrayLike) -> GramSchmidt:
    """Orthonormalize a new column v, shape (n,), against Q (applied twice).

    The forward pass appends ``q`` to its basis of the √w-scaled columns of B
    [plan: Fast path]. With v = √w·(b·x), ``norm``² is A_w of LA-7, the weighted
    RSS of b·x regressed on B. The function does not decide whether v is
    dependent: FWD-11 applies LA-4 (``independent_columns``) when the pass
    stops. Complexity: O(n·r) time, O(n) memory.
    """
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError(f"v must be 1-D, not {v.ndim}-D")
    perp, coef = orthogonalize(Q, v)
    norm = float(scipy.linalg.norm(perp))
    return GramSchmidt(perp / norm if norm > 0.0 else None, coef, norm)


def collinearity_tolerance(step: int) -> float:
    """Return tau of LA-3 for forward step ``step``, counted from 1.

    tau = 0.01 while the pass has taken at most 6 steps, so in steps 1 to 7,
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
    """Return rho = ‖h - P_G h‖_w² / ‖h - h̄‖_w², 1 - R² of h on G (LA-3).

    Q (n, r) is an orthonormal basis of the √w-scaled columns of G: the current
    columns in a single-hinge search, and b·x too in a pair search (LA-7). G
    must contain the intercept, so rho ≤ 1 up to rounding. h (n,) is the
    unscaled hinge column, h̄ its weighted mean with N = math.fsum(w) (W-4).
    For an h that is constant over the cases (Conventions: all its values with
    positive weight are equal) the ratio is 0/0, and the function returns 0.0,
    so that ``knot_rejected`` rejects it. h is first multiplied by a power of 2,
    which changes no bit of rho and keeps the sums of squares in range. The
    fast scan computes rho for all knots from sums (plan: Fast path); this is
    the explicit form, for the chosen column and for tests.
    Complexity: O(n·r) time, O(n) memory.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.ndim != 1:
        raise ValueError(f"h must be 1-D, not {h.ndim}-D")
    wv = _weights(w, h.shape[0])
    cases = h if w is None else h[wv > 0.0]
    if cases.size == 0 or np.all(cases == cases[0]):
        return 0.0
    hs = np.ldexp(h, -int(np.frexp(np.max(np.abs(h)))[1]))
    d = hs - float(wv @ hs) / math.fsum(wv)
    centered = float(wv @ (d * d))
    if centered == 0.0:  # only by underflow, with weights that EDGE-6 rejects
        return 0.0
    perp, _ = orthogonalize(Q, np.sqrt(wv) * hs)
    return float(perp @ perp) / centered


def knot_rejected(ratio: npt.ArrayLike, step: int) -> bool | npt.NDArray[np.bool_]:
    """Apply LA-3 in forward step ``step``: True where rho < tau(step).

    ``ratio`` is one rho or an array of them, one per knot; 0.0 marks a
    constant hinge column. A ratio equal to tau is kept. Complexity: O(len).
    """
    rejected = np.asarray(ratio, dtype=np.float64) < collinearity_tolerance(step)
    return bool(rejected) if rejected.ndim == 0 else rejected


def weighted_variances(X: npt.ArrayLike, w: npt.ArrayLike | None = None) -> FloatArray:
    """Return sigma_v² = Σ w_i (x_iv - x̄_v)² / N for each column v (LA-7).

    X is (n, p); x̄_v is the weighted mean, and N = math.fsum(w) over the rows
    with positive weight (Notation, W-3, W-4). A column that is constant over
    those rows (Conventions) gets exactly 0.0. Complexity: O(n·p).
    """
    X = _matrix(X, "X")
    wv = _weights(w, X.shape[0])
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
    """Return True for a pair search, False for a single-hinge search (LA-7).

    pymars runs a pair search for a parent b and a covariate x when
    A_w ≥ 0.01·Π_{v∈V} sigma_v², and a single-hinge search otherwise, and
    always when some v ∈ V is constant (a pymars departure: the rule does not
    depend on the units of x). ``resid_ss`` is A_w, the weighted RSS of b·x on
    the current columns (``gram_schmidt(Q, √w·b·x).norm ** 2``); ``variances``
    holds sigma_v² from ``weighted_variances`` for the covariates V of b·x, the
    parent's and x, where 0.0 marks a constant covariate.
    Complexity: O(len(variances)).
    """
    v = np.asarray(variances, dtype=np.float64).ravel()
    if v.size == 0:
        raise ValueError("variances must hold sigma² of at least the covariate x")
    if np.any(v == 0.0):
        return False
    return bool(resid_ss >= PAIR_SEARCH_FACTOR * math.prod(v.tolist()))


def _independent(As: FloatArray) -> BoolArray:
    """Apply LA-4 to the columns of As, which are already √w-scaled, in order.

    Each column is orthogonalized twice against the kept earlier ones.
    Complexity: O(n·M·r) time for rank r, O(n·r) memory.
    """
    n, M = As.shape
    Q = np.empty((n, min(n, M)))
    kept = np.zeros(M, dtype=bool)
    r = 0
    for j in range(M):
        a = As[:, j]
        if r == n or not np.any(a):
            continue
        perp, _ = orthogonalize(Q[:, :r], a)
        norm = scipy.linalg.norm(perp)
        if norm < LM_TOL * scipy.linalg.norm(a):
            continue
        Q[:, r] = perp / norm
        r += 1
        kept[j] = True
    return kept


def independent_columns(A: npt.ArrayLike, w: npt.ArrayLike | None = None) -> BoolArray:
    """Return which columns of A (n, M) LA-4 keeps, True for a kept column.

    The columns are taken in their order. A column is dependent when the w-norm
    of its part orthogonal to the kept earlier columns is less than 1e-7 times
    its w-norm, not centered. A column that is 0 on every case is dependent
    too, as in R's ``lm.fit`` (the strict test alone would keep it, since 0 is
    not less than 0). FWD-11 applies this rule to the forward terms when the
    pass stops. Complexity: O(n·M·r) time for rank r, O(n·M) memory.
    """
    A = _matrix(A, "A")
    return _independent(np.sqrt(_weights(w, A.shape[0]))[:, None] * A)


def lm_fit(A: npt.ArrayLike, Y: npt.ArrayLike, w: npt.ArrayLike | None = None) -> LmFit:
    """Weighted least squares of Y on A, with LA-4's dependent columns at 0.

    The analogue of R's ``lm.fit`` and ``lm.wfit`` at their default tolerance,
    whose NA is 0.0 here (LA-4, PRUNE-8). ``independent_columns`` picks the
    kept columns; a Householder QR of their √w-scaled values and a triangular
    solve give the coefficients. A is (n, M), Y (n,) or (n, K), and ``coef``
    has shape (M,) or (M, K) to match. ``rss`` is the weighted RSS of the
    returned coefficients, as PRUNE-8 defines the final rss.
    Complexity: O(n·M·(M + K)) time, O(n·(M + K)) memory.
    """
    A = _matrix(A, "A")
    n, M = A.shape
    Y2, one_d = _responses(Y, n)
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
    """Return R, Z and the full RSS for the columns A in the working order.

    A Householder QR of √w·A gives Q and R; ``orthogonalize`` gives
    Z = Qᵀ(√w·Y) and the residual, whose squared norm is the RSS. The pruning
    pass then works on R and Z alone (PRUNE-9). A is (n, M) with M ≤ n and Y is
    (n,) or (n, K). The columns must be linearly independent, as the kept
    forward terms are (FWD-11, PRUNE-2), so that R is nonsingular.
    Complexity: O(n·M·(M + K)) time, O(n·(M + K)) memory.
    """
    A = _matrix(A, "A")
    n, M = A.shape
    if n < M:
        raise ValueError(f"A has {M} columns and only {n} rows")
    Y2, _ = _responses(Y, n)
    sw = np.sqrt(_weights(w, n))[:, None]
    Q, R = np.linalg.qr(sw * A)
    E, Z = orthogonalize(Q, sw * Y2)
    return RFactor(R, Z, float(np.sum(np.square(E))))


def prefix_rss(Z: npt.ArrayLike, rss: float) -> FloatArray:
    """Return the RSS of each prefix of the working order (LA-1, PRUNE-3).

    Entry m - 1 is the RSS of the first m columns, m = 1, …, M: ``rss`` plus
    Σ_{j ≥ m} ‖Z_j‖², summed over the responses, with no cancellation since
    every term is at least 0. Z is (M,) or (M, K). Complexity: O(M·K).
    """
    Z = np.asarray(Z, dtype=np.float64)
    z2 = np.square(Z) if Z.ndim == 1 else np.sum(np.square(Z), axis=1)
    out = np.full(z2.shape[0], float(rss))
    out[:-1] += np.cumsum(z2[::-1])[::-1][1:]
    return out


def drop_costs(R: npt.ArrayLike, Z: npt.ArrayLike, pos: int) -> FloatArray:
    """Return the RSS increase when each of the first ``pos`` columns is removed.

    Entry i, for i = 0, …, pos - 1, is RSS(first pos columns without column i)
    minus RSS(first pos columns), summed over the responses; step 1 of a
    PRUNE-3 stage compares these after the intercept. With R_p the leading
    pos x pos block of R and β = R_p⁻¹·Z[:pos], the increase is
    Σ_k β_ik² / [(R_pᵀR_p)⁻¹]_ii, where [(R_pᵀR_p)⁻¹]_ii is the squared norm of
    row i of R_p⁻¹; R_p must be nonsingular. Complexity: O(pos³ + pos²·K).
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
    """Return new R and Z for the order with column i moved to position j.

    The columns between positions i and j shift one place toward i, as in step
    2 of a PRUNE-3 stage, which moves the removed term to position pos. Only
    rows min(i, j) to max(i, j) change: a Householder QR of that block restores
    the triangle, and its orthogonal factor turns the same rows of Z, so every
    prefix RSS is that of the new order (PRUNE-9, a downdate in place of a
    refit). Complexity: O(h²·(M + K)) with h = |i - j| + 1.
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
