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
EDGE-6 does, changes no bit of a result beyond that factor. EDGE-6 scales Y
only: the covariates and the columns of B must lie within about 1e-150 to 1e150
in absolute value, so that their squares stay in the normal range of float64;
``collinearity_ratio`` and ``drop_costs`` also scale by powers of 2 inside. N is
the ``math.fsum`` of the weights, not snapped to an integer (W-4); the
difference is at most 1e-8 relative, inside the bands of LA-5. Memory is
O(n·(M + K)) for n rows, M columns and K responses.

Public functions, for ``_scan``, ``_forward``, ``_pruning`` and ``_core``:

- ``orthogonalize``, ``gram_schmidt``: Gram-Schmidt applied twice (LA-1).
- ``collinearity_tolerance``, ``collinearity_ratio``, ``knot_rejected``: the
  collinearity test of a candidate knot (LA-3).
- ``weighted_variances``, ``pair_search``: the kind of a search (LA-7).
- ``independent_columns``, ``lm_fit``: least squares with the dependent columns
  of LA-4, the analogue of R's ``lm.fit`` (FWD-11, PRUNE-8).
- ``Conditioner``, ``large_covariates``, ``plain_dependent``,
  ``independent_terms``: columns that span what the terms span, free of the
  large mean of a covariate (LA-5), and the dependence test of LA-4 with the
  exact shift of such covariates (LA-1, LA-2, FWD-11).
- ``r_factor``, ``prefix_rss``, ``drop_costs``, ``move_column``: the R factor of
  the pruning pass and its downdates (PRUNE-3, PRUNE-9).

One PRUNE-3 stage at position pos (counted from 1, as in the spec) with
``R, Z, rss = r_factor(A, Y, w)`` for the columns in the working order: the RSS
without the column at 0-based index i, for i = 1, …, pos - 1, is
``prefix_rss(Z, rss)[pos - 1] + drop_costs(R, Z, pos)[i]``; after the tie rule,
``R, Z = move_column(R, Z, i, pos - 1)`` moves the removed term to position pos,
and ``prefix_rss(Z, rss)`` offers the new order. With several responses the
calls are the same, and pos shrinks by one after each removal.
"""

from __future__ import annotations

import hashlib
import math
import operator
from collections.abc import Sequence
from fractions import Fraction
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import scipy.linalg

from pymars import _terms

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int8]
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
    stops. A v in the span of Q leaves a v⊥ of rounding noise, of norm about
    1e-16·‖v‖, and then ``q`` is a unit vector in a random direction: never
    append it. Complexity: O(n·r) time, O(n) memory.
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
    h = h if w is None else np.where(wv > 0.0, h, 0.0)  # W-3: not a case
    hs = np.ldexp(h, -int(np.frexp(np.max(np.abs(cases)))[1]))
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
    pos = wv > 0.0  # W-3: a row with zero weight adds nothing to the RSS
    rss = float(np.sum(wv[pos, None] * np.square(residuals[pos])))
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
    row i of R_p⁻¹; R_p must be nonsingular. Each column of R_p is first
    multiplied by a power of 2, which changes no increase and keeps the squares
    in range when the columns of B differ widely in scale.
    Complexity: O(pos³ + pos²·K).
    """
    R = _matrix(R, "R")
    Z = np.asarray(Z, dtype=np.float64)
    pos = operator.index(pos)
    if not 1 <= pos <= R.shape[1]:
        raise ValueError(f"pos must be in 1..{R.shape[1]}, not {pos}")
    Rp = R[:pos, :pos]
    Rp = np.ldexp(Rp, -np.frexp(np.max(np.abs(Rp), axis=0))[1])
    Rinv = scipy.linalg.solve_triangular(Rp, np.eye(pos))
    beta = Rinv @ Z[:pos]
    num = np.square(beta) if beta.ndim == 1 else np.sum(np.square(beta), axis=1)
    return num / np.sum(np.square(Rinv), axis=1)


def move_column(
    R: npt.ArrayLike, Z: npt.ArrayLike, i: int, j: int
) -> tuple[FloatArray, FloatArray]:
    """Return new R and Z for the order with column i moved to position j.

    i and j are 0-based; the columns between them shift one place toward i. In
    step 2 of a PRUNE-3 stage, whose positions count from 1, the removed term
    goes to position pos, so j = pos - 1. Only rows min(i, j) to max(i, j)
    change: a Householder QR of that block restores the triangle, and its
    orthogonal factor turns the same rows of Z, so every prefix RSS is that of
    the new order (PRUNE-9, a downdate in place of a refit).
    Complexity: O(M² + M·K) for the copies of R and Z, and O(h²·(M + K)) for the
    QR, with h = |i - j| + 1.
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


_Atom = tuple  # (symbol, kind, offset): kind 0 is u, 1 is (x - t)+, 2 is (t - x)+
_Element = tuple  # a product of atoms, sorted


def large_covariates(X: npt.ArrayLike) -> tuple[BoolArray, FloatArray]:
    """LA-4: which covariates are shifted, and the shifts m_j. Covariate j is
    shifted when its smallest value m_j is larger in absolute value than its
    range (a constant covariate, with range 0, when it is not 0); then
    u_j = x_j - m_j, and m_j = 0 for the others. Complexity: O(n·p)."""
    X = np.asarray(X, dtype=np.float64)
    lo, hi = X.min(axis=0), X.max(axis=0)
    big = np.abs(lo) > hi - lo
    return big, np.where(big, lo, 0.0)


def plain_dependent(dist: float, norm: float) -> bool:
    """LA-4 for a column with no shifted covariate in it or in the earlier
    columns: it is dependent when the norm of its part orthogonal to them is
    below LM_TOL times its own norm, or when it is 0 at every case.
    Complexity: O(1)."""
    return norm == 0.0 or dist < LM_TOL * norm


def _order(m: _Element) -> tuple:
    """LA-4's total order of monomials: by degree, highest first, then
    lexicographically on the atoms (covariate, kind, knot)."""
    return (-len(m), m)


class Conditioner:
    """Exact changes of basis for covariates with a large mean (LA-4, LA-5).

    Covariate j is shifted when |m_j| exceeds its range, m_j its smallest value
    (``large_covariates``); u_j = x_j - m_j is exact in float64 where the
    covariate has no case near 0 (Sterbenz), and a linear factor x_j is
    u_j + m_j. Covariates whose columns u are bitwise equal are one symbol
    (the atoms of their terms are the same), so that a copy of a covariate at
    another shift cancels exactly.

    Columns for the projections (LA-5). A pair of hinges spans x_j with the
    constant, (t - x)₊ = (x - t)₊ - u_j + (t - m_j), so a term is a sum of
    elements, products of atoms u_j and (x_j - t)₊, with exact rational
    coefficients, and exact Gaussian elimination in that coefficient space
    removes the part of a term that earlier terms span: ``linear`` and
    ``hinge`` return unscaled columns with coefficients of size about 1 on
    elements of the size of the range, not of the size of the mean, that span
    with the terms registered by ``append`` what b·x_j and a new term span with
    them. With no term that expands (a linear factor, or a hinge (t - x)₊, of
    a shifted covariate) in the parent b, the caller uses b·(x_j - c_j), which
    spans the same with the terms, since they hold b. The pivots are the
    elements with the largest share of the reduced term, which keeps the
    reduced column small.

    The dependence test of LA-4 (``dependent``). The terms are polynomials in
    the atoms u_j, (x_j - t)₊ and (t - x_j)₊ (a hinge is its own atom), with
    exact coefficients; the registered terms are kept in their reduced echelon
    form for the order of LA-4 (``_order``: the pivot of a row is its first
    monomial), so the test does not depend on the order of the terms. A new
    term is dependent when its expansion is in the span, or when the part of
    its new part (its row in the echelon form of the registered terms, the
    extra rows and itself) that is orthogonal to the other rows is below
    LM_TOL times the norm of its pivot monomial (the new part's own norm when
    that is 0), or when the new part is 0 at every case. The orthogonal part is
    taken in float64 from a column that carries no digits of the mean, with a
    bound on its error, and exactly (rational arithmetic on the distinct rows
    of X) when that bound does not decide, and when the rows of the echelon
    form that hold the new pivot change, since their evaluation carries the
    mean.

    Complexity: a term with d factors has at most 2^d monomials in the test and
    3^d elements in a column; reducing it costs O(M·2^d) rational operations for
    M registered terms, forming a column O(n·d·3^d), and the exact test
    O(g·M²) rational operations for g distinct rows of X. The reduced
    coefficients of b·x_j are kept and reduced only by the terms that came
    after the last call. Memory O(M·p·3^d) coefficients, no columns, and O(g·K)
    rationals for K monomials in the exact test.
    """

    def __init__(self, X: npt.ArrayLike, w: npt.ArrayLike | None = None):
        X = np.asarray(X, dtype=np.float64)
        self._X = X
        self._w = None if w is None else np.asarray(w, dtype=np.float64)
        self._sw = None if w is None else np.sqrt(self._w)
        big, self._m = large_covariates(X)
        self.large = frozenset(int(j) for j in np.flatnonzero(big))
        self._ucols: dict = {}
        self._rep = list(range(X.shape[1]))  # a symbol: the first with the same u
        first: dict = {}
        for j in range(X.shape[1]):
            u = self._ucol(j)
            r = first.setdefault(
                hashlib.blake2b(u.tobytes(), digest_size=16).digest(), j
            )
            self._rep[j] = r if np.array_equal(u, self._ucol(r)) else j
        self._diff = self._near_copies()
        self._rows: list[tuple[IntArray, FloatArray]] = []
        self._pivots: list[tuple[_Element, dict]] = []
        self._done = 0  # the rows that the pivots cover
        self._norms: dict = {}
        self._lin: dict = {}
        self._hat: dict = {}  # a hinge atom -> (covariate, knot) that evaluates it
        self._acols: dict = {}
        self._erows: list[tuple[_Element, dict]] = []  # the echelon form of LA-4
        self._edone = 0
        self._mcols: dict = {}
        self._mnorms: dict = {}
        self._groups: tuple | None = None
        self._xcols: dict = {}
        self._shifted = False  # a registered term has a linear factor of a large one
        self.size = 0.0  # a bound on the magnitude of the last column formed

    def _ucol(self, j: int) -> FloatArray:
        """u_j = x_j - m_j (x_j itself for a covariate that is not shifted)."""
        if j not in self.large:
            return self._X[:, j]
        if j not in self._ucols:
            self._ucols[j] = self._X[:, j] - self._m[j]
        return self._ucols[j]

    def _near_copies(self) -> dict:
        """{j: (r, d)} for a shifted covariate j whose column u_j is a near-copy
        of that of an earlier symbol r: u_j = u_r + d exactly, d ≠ 0 (every
        difference is exact, by Sterbenz, and |d| is within 2^-10 of |u_j| in
        norm). The columns for the projections (``linear``, ``hinge``) then
        write u_j as u_r + d, so that the part of a product that the terms
        span cancels as a formula and the small d is not lost under the large
        mean (the one-ulp copy of PR #95's review, LA-5). The test of LA-4
        does not use it. Complexity: O(n·p_large·log p_large) for the sort and
        O(n) for each pair that the norms let through."""
        sym = sorted(
            (j for j in self.large if self._rep[j] == j),
            key=lambda j: float(np.linalg.norm(self._ucol(j))),
        )
        out: dict = {}
        for a, j in enumerate(sym):
            uj = self._ucol(j)
            for r in reversed(sym[max(0, a - 8) : a]):
                if r in out:
                    continue
                ur = self._ucol(r)
                d = uj - ur
                ok = np.all(
                    (d == 0) | ((ur != 0) & (uj != 0) & (uj <= 2 * ur) & (ur <= 2 * uj))
                )
                if ok and 0 < np.linalg.norm(d) <= 2.0**-10 * np.linalg.norm(uj):
                    out[j] = (r, d)
                    break
        return out

    def expands(self, row: npt.ArrayLike) -> bool:
        """Whether a term row has a factor that is expanded: a linear factor or
        a hinge (t - x)₊ of a shifted covariate. Complexity: O(p)."""
        row = np.asarray(row)
        return any(row[j] in (2, -1) for j in self.large)

    def shifted(self, *rows: npt.ArrayLike) -> bool:
        """Whether LA-4's test needs the change of basis for the registered
        terms and these rows: one of them has a linear factor of a shifted
        covariate. Otherwise every term is one monomial and the test is the
        plain one (``plain_dependent``). Complexity: O(p)."""
        return self._shifted or any(
            np.asarray(r)[j] == 2 for r in rows for j in self.large
        )

    def append(self, row: npt.ArrayLike, cut: npt.ArrayLike) -> None:
        """Register the next term. Complexity: O(p)."""
        row = np.asarray(row)
        self._rows.append((row, np.asarray(cut)))
        self._shifted = self._shifted or any(row[j] == 2 for j in self.large)

    # -- atoms and their columns

    def _atom(self, j: int, kind: int, t: float = 0.0) -> _Atom:
        r = self._rep[j]
        if kind == 0:
            return (r, 0, 0.0)
        key = (r, kind, float(t - self._m[j]))
        self._hat.setdefault(key, (j, t))
        return key

    def _atom_column(self, a: _Atom) -> FloatArray:
        if a not in self._acols:
            if a[1] == 0:
                col = self._ucol(a[0])
            elif a[1] == 3:
                col = self._diff[a[0]][1]
            else:
                j, t = self._hat[a]
                x = self._X[:, j]
                col = np.maximum(x - t, 0.0) if a[1] == 1 else np.maximum(t - x, 0.0)
            self._acols[a] = col
        return self._acols[a]

    def _element_column(self, e: _Element) -> FloatArray:
        if e not in self._mcols:
            out = np.ones(self._X.shape[0])
            for a in e:
                out = out * self._atom_column(a)
            self._mcols[e] = out
        return self._mcols[e]

    def _scaled_norm(self, v: FloatArray) -> float:
        return float(np.linalg.norm(v if self._sw is None else self._sw * v))

    def _norm(self, e: _Element) -> float:
        if e not in self._mnorms:
            self._mnorms[e] = self._scaled_norm(self._element_column(e))
        return self._mnorms[e]

    # -- columns that span what the terms span (LA-5)

    def _u_parts(self, j: int, sign: int) -> dict:
        """sign·u_j as {element: Fraction}: u_r + d_j for a near-copy."""
        if j not in self._diff:
            return {self._atom(j, 0): Fraction(sign)}
        return {
            self._atom(self._diff[j][0], 0): Fraction(sign),
            (j, 3, 0.0): Fraction(sign),
        }

    def _expansion(self, row: FloatArray, cut: FloatArray) -> dict:
        """The term as {element: Fraction}, a hinge (t - x)₊ rewritten.
        Complexity: O(3^d)."""
        out: dict = {(): Fraction(1)}
        for j in np.flatnonzero(row):
            code, t = int(row[j]), float(cut[j])
            m = Fraction(float(self._m[j]))
            if code == 1:
                parts: dict = {self._atom(j, 1, t): Fraction(1)}
            elif code == 2:
                parts = self._u_parts(j, 1)
                if j in self.large:
                    parts[None] = m
            else:  # (t - x)₊ = (x - t)₊ - u + (t - m), m = 0 if x is not shifted
                parts = self._u_parts(j, -1)
                parts[self._atom(j, 1, t)] = Fraction(1)
                parts[None] = Fraction(t) - m
            new: dict = {}
            for e, f in out.items():
                for q, g in parts.items():
                    key = e if q is None else tuple(sorted((*e, q)))
                    new[key] = new.get(key, 0) + f * g
            out = {e: f for e, f in new.items() if f != 0}
        return out

    def _reduce(self, v: dict, start: int) -> dict:
        """v minus its components along the pivots from ``start`` on."""
        for p, r in self._pivots[start:]:
            f = v.get(p, 0)
            if f != 0:
                for q, g in r.items():
                    v[q] = v.get(q, 0) - f * g
                v = {q: g for q, g in v.items() if g != 0}
        return v

    def _sync(self) -> None:
        """Make a pivot of every registered term that the pivots miss: the
        element with the largest share of the reduced term (by coefficient
        times column norm, the first in key order among equals), with the
        coefficients divided by its own. Complexity: O(M·3^d) per term."""
        while self._done < len(self._rows):
            v = self._reduce(self._expansion(*self._rows[self._done]), 0)
            self._done += 1
            if v:
                p = max(sorted(v), key=lambda q: abs(float(v[q])) * self._norm(q))
                self._pivots.append((p, {q: f / v[p] for q, f in v.items()}))

    def _column(self, v: dict) -> FloatArray:
        out, mag = np.zeros(self._X.shape[0]), np.zeros(self._X.shape[0])
        for e in sorted(v):
            col, c = self._element_column(e), float(v[e])
            out += c * col
            mag += abs(c) * np.abs(col)
        self.size = self._scaled_norm(mag)
        return out

    def linear(
        self, k: int, row: npt.ArrayLike, cut: npt.ArrayLike, j: int
    ) -> FloatArray:
        """A column that spans with the terms so far what b·x_j spans with them:
        the reduced expansion of b·x_j, for the term k with ``row`` and ``cut``
        as b and the covariate j that it lacks. The part of the work that the
        earlier calls for (k, j) did is kept. Complexity: see the class."""
        row, cut = np.array(row), np.array(cut, dtype=np.float64)
        row[j], cut[j] = 2, 0.0
        self._sync()
        ent = self._lin.get((k, j))
        if ent is None:
            ent = self._lin[k, j] = [0, self._expansion(row, cut)]
        ent[1] = self._reduce(ent[1], ent[0])
        ent[0] = len(self._pivots)
        return self._column(ent[1])

    def hinge(self, row: npt.ArrayLike, cut: npt.ArrayLike) -> FloatArray:
        """A column that spans with the terms so far what the term with
        ``row`` and ``cut`` spans with them: its reduced expansion.
        Complexity: see the class."""
        self._sync()
        v = self._expansion(np.asarray(row), np.asarray(cut, dtype=np.float64))
        return self._column(self._reduce(v, 0))

    # -- the dependence test of LA-4

    def _poly(self, row: npt.ArrayLike, cut: npt.ArrayLike) -> dict:
        """The term as a polynomial in the atoms of LA-4, {monomial: Fraction}:
        a linear factor of a shifted covariate is u_j + m_j, a hinge is its own
        atom. Complexity: O(2^d)."""
        row, cut = np.asarray(row), np.asarray(cut, dtype=np.float64)
        out: dict = {(): Fraction(1)}
        for j in np.flatnonzero(row):
            code, t = int(row[j]), float(cut[j])
            if code == 2:
                parts: dict = {(self._atom(j, 0),): Fraction(1)}
                if j in self.large:
                    parts[()] = Fraction(float(self._m[j]))
            else:
                parts = {(self._atom(j, 1 if code == 1 else 2, t),): Fraction(1)}
            new: dict = {}
            for e, f in out.items():
                for q, g in parts.items():
                    key = tuple(sorted(e + q))
                    new[key] = new.get(key, 0) + f * g
            out = {e: f for e, f in new.items() if f != 0}
        return out

    @staticmethod
    def _echelon_add(rows: list, poly: dict):
        """The reduced echelon form of ``rows`` with ``poly`` added, as (rows,
        pivot, new part), or None when the polynomial lies in their span. The
        rows are not changed. Complexity: O(M·|poly|)."""
        v = dict(poly)
        for p, R in rows:
            f = v.get(p, 0)
            if f != 0:
                for q, g in R.items():
                    v[q] = v.get(q, 0) - f * g
        v = {q: g for q, g in v.items() if g != 0}
        if not v:
            return None
        q = min(v, key=_order)
        N = {m: g / v[q] for m, g in v.items()}
        out = []
        for p, R in rows:
            f = R.get(q, 0)
            if f != 0:
                R = {m: R.get(m, 0) - f * N.get(m, 0) for m in R.keys() | N.keys()}
                R = {m: g for m, g in R.items() if g != 0}
            out.append((p, R))
        out.append((q, N))
        return out, q, N

    def _esync(self) -> None:
        while self._edone < len(self._rows):
            added = self._echelon_add(self._erows, self._poly(*self._rows[self._edone]))
            self._edone += 1
            if added is not None:
                self._erows = added[0]

    def dependent(
        self,
        row: npt.ArrayLike,
        cut: npt.ArrayLike,
        dist: float,
        size: float,
        extra: Sequence[tuple[npt.ArrayLike, npt.ArrayLike]] = (),
        kappa: float = 1.0,
    ) -> bool:
        """LA-4 for the term with ``row`` and ``cut`` as a new column after the
        registered terms and the ``extra`` terms (rows and cuts, not registered).

        ``dist`` is the √w-scaled norm of the part of the term's column that is
        orthogonal to the span of those columns, computed from a column that
        spans the same with them, and ``size`` a bound on the magnitude of that
        column's entries (``self.size`` after ``linear`` or ``hinge``, the
        column's norm when it is plain), and ``kappa`` the largest ratio of
        the size of a column of the span to the norm of its orthogonal part
        (``_scan.Rebuild.kappa``), which bounds the error of the basis, so that
        the float64 value is trusted only where its error cannot change the
        decision; otherwise, and when
        the rows of the echelon form change, the test is exact. Complexity: see
        the class."""
        self._esync()
        rows = self._erows
        for r, c in extra:
            added = self._echelon_add(rows, self._poly(r, c))
            rows = rows if added is None else added[0]
        added = self._echelon_add(rows, self._poly(row, cut))
        if added is None:
            return True
        after, q, N = added
        scale = self._norm(q)
        if scale == 0.0:  # the pivot monomial is 0 at every case
            scale = self._scaled_norm(self._evaluate(N))
        if scale == 0.0:
            return True
        adjusted = any(q in R for _, R in rows)
        if not adjusted:
            u = 2.0**-53
            err = 16.0 * u * (size * (1.0 + kappa) + scale) * math.sqrt(len(rows) + 2)
            if dist - err > LM_TOL * scale:
                return False
            if dist + err < LM_TOL * scale:
                return True
        return self._exact_dependent(q, N, after[:-1])

    def _evaluate(self, poly: dict) -> FloatArray:
        out = np.zeros(self._X.shape[0])
        for m in sorted(poly, key=_order):
            out += float(poly[m]) * self._element_column(m)
        return out

    # -- exact arithmetic on the distinct rows of X

    def _group_data(self) -> tuple:
        if self._groups is None:
            Xu, inv = np.unique(self._X, axis=0, return_inverse=True)
            inv = inv.reshape(-1)
            if self._w is None:
                W = [Fraction(int(c)) for c in np.bincount(inv, minlength=len(Xu))]
            else:
                W = [Fraction(0)] * len(Xu)
                for g, wi in zip(inv.tolist(), self._w.tolist(), strict=True):
                    W[g] += Fraction(wi)
            self._groups = (Xu, Xu - self._m, W)
        return self._groups

    def _exact_column(self, e: _Element) -> list:
        """The monomial on the distinct rows of X, as exact rationals."""
        if e not in self._xcols:
            Xu, Uu, _ = self._group_data()
            col = [Fraction(1)] * len(Xu)
            for a in e:
                if a[1] == 0:
                    vals = [Fraction(float(v)) for v in Uu[:, a[0]]]
                else:
                    j, t = self._hat[a]
                    sign = 1 if a[1] == 1 else -1
                    vals = [
                        max(sign * (Fraction(float(v)) - Fraction(t)), Fraction(0))
                        for v in Xu[:, j]
                    ]
                col = [c * v for c, v in zip(col, vals, strict=True)]
            self._xcols[e] = col
        return self._xcols[e]

    def _exact_vector(self, poly: dict) -> list:
        n = len(self._group_data()[2])
        out = [Fraction(0)] * n
        for m, c in poly.items():
            out = [o + c * x for o, x in zip(out, self._exact_column(m), strict=True)]
        return out

    def _exact_dependent(self, q: _Element, N: dict, rows: list) -> bool:
        """LA-4 in exact arithmetic: the weighted squared norm of N orthogonal to
        the rows is below LM_TOL² times that of the pivot monomial q (of N when
        q is 0 at every case). Complexity: O(g·M²) rational operations."""
        W = self._group_data()[2]

        def norm2(a: list, b: list) -> Fraction:
            return sum(
                (w * x * y for w, x, y in zip(W, a, b, strict=True)), Fraction(0)
            )

        basis: list = []  # orthogonal vectors with their squared norms
        for _, R in rows:
            v = self._exact_vector(R)
            for b, nb in basis:
                c = norm2(v, b) / nb
                if c != 0:
                    v = [x - c * y for x, y in zip(v, b, strict=True)]
            nv = norm2(v, v)
            if nv != 0:
                basis.append((v, nv))
        v = self._exact_vector(N)
        total = norm2(v, v)
        for b, nb in basis:
            c = norm2(v, b) / nb
            if c != 0:
                v = [x - c * y for x, y in zip(v, b, strict=True)]
        eq = self._exact_column(q)
        ref = norm2(eq, eq)
        if ref == 0:
            ref = total
        return ref == 0 or norm2(v, v) < Fraction(LM_TOL) ** 2 * ref


def independent_terms(
    X: npt.ArrayLike,
    dirs: npt.ArrayLike,
    cuts: npt.ArrayLike,
    w: npt.ArrayLike | None = None,
    B: npt.ArrayLike | None = None,
) -> BoolArray:
    """Return which terms LA-4 keeps, in their order, True for a kept term
    (FWD-11): the rule of ``independent_columns`` on the columns of the terms,
    and, when a term has a linear factor of a shifted covariate (``Conditioner``,
    ``large_covariates``), after the exact shift of such covariates, so that the
    result does not depend on the order of the terms. ``B`` is the basis matrix
    of the terms if the caller has it. Without a shifted covariate in a linear
    factor this is ``independent_columns`` of B, bit for bit. Complexity: that of
    ``independent_columns`` and, with a shifted covariate, of ``Conditioner``
    for each term.
    """
    X = np.asarray(X, dtype=np.float64)
    dirs, cuts = np.asarray(dirs), np.asarray(cuts, dtype=np.float64)
    cond = Conditioner(X, w)
    if not any(cond.shifted(r) for r in dirs):
        return independent_columns(
            _terms.basis_matrix(X, dirs, cuts) if B is None else B, w
        )
    sw = np.ones(X.shape[0]) if w is None else np.sqrt(np.asarray(w, dtype=np.float64))
    Q = np.empty((X.shape[0], 0))
    kept, kappa = np.zeros(len(dirs), dtype=bool), 1.0
    for k, (row, cut) in enumerate(zip(dirs, cuts, strict=True)):
        v = sw * cond.hinge(row, cut)
        gs = gram_schmidt(Q, v)
        if gs.q is None or cond.dependent(row, cut, gs.norm, cond.size, kappa=kappa):
            continue
        kappa = max(kappa, cond.size / gs.norm)
        cond.append(row, cut)
        Q = np.column_stack((Q, gs.q))
        kept[k] = True
    return kept
