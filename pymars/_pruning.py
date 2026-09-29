"""The pruning pass: the best subset of each size, the size with the lowest GCV,
and the least squares of the selected terms.

Spec: ``docs/algorithm.md``, "Pruning pass" (PRUNE-1 to PRUNE-9), and the rules
that it calls: LA-1 and LA-4 (least squares), GCV-1 to GCV-7 (the GCV and the
fit statistics), LIMIT-3 (``nprune``), W-2 to W-5 (weights), RESP-1 (several
responses) and CORE-3 (the pruning record). The plan's sections "Pruning pass"
and "Fast path" set the method.

Input. The pass works on the M_f terms that the forward pass kept (FWD-11), as
the n x M_f matrix B of their columns in forward order; column 0 is the
intercept, a column of ones (TERM-3). The columns must be linearly independent,
as FWD-11 makes them. Every index here is a pruning index, 0 to M_f - 1
(PRUNE-2); the core maps it to a forward index with ``ForwardRecord.kept``. Y
is (n,) or (n, K); w is (n,), or None, which means w_i = 1 exactly (W-5). A
row with zero weight is not a case, and both functions drop it first (W-3).
N = Σw and τ_N follow W-4 (``_gcv.total_weight``).

Method (PRUNE-9). One Householder QR of the √w-scaled B in the working order
gives R and Z = Qᵀ(√w·Y) (``_linalg.r_factor``). A stage reads the RSS of each
prefix of the order and the RSS of each drop of one term from R and Z
(``prefix_rss``, ``drop_costs``), then moves the removed term with a QR of the
rows that change (``move_column``); no subset is refit.

Numerics. float64 throughout; no function writes into its inputs; no absolute
epsilon; the tie rules are fixed: the term added last when two drops give the
same RSS (PRUNE-3), the earlier offer when two subsets of one size do, and the
smaller size when two sizes give the same GCV (PRUNE-5). Every subset holds
the intercept, so subtracting a constant from a response changes no RSS in
exact arithmetic. Each response is centered first, so that the rounding of
the projections is of the size of the spread of Y, not of its mean (LA-5);
a response that is constant over the cases (Conventions: all its values are
equal) becomes 0 exactly and adds 0 to every sum. For the same reason
``pruning_pass`` centers each column of B after the intercept in the same
way before the QR: a column far from 0 relative to its spread (x near 33 with
a spread of 0.25, say) would otherwise make the factor ill-conditioned, and
the RSS of a near-exact fit would lose about 1e-8 of its relative accuracy
(#84). Y enters only through
differences, products and sums, so multiplying Y by a power of 2 (EDGE-6)
multiplies every RSS and GCV by its square and changes no other bit. Memory
is O(n·(M_f + K) + M_f²).

Public functions, for ``_core``:

- ``pruning_pass``: the stages, the records and the selected terms (PRUNE-1 to
  PRUNE-7).
- ``final_fit``: the coefficients and statistics of the selected terms
  (PRUNE-8).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from pymars import _gcv, _linalg

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

#: PRUNE-1: the supported pruning methods.
PMETHODS = ("backward", "none")


class PruningPass(NamedTuple):
    """``pruning_pass(B, Y, w, ...)``, the fields of ``PruningRecord`` (CORE-3)
    and the selected terms.

    ``removed`` (M_f - 1,) int64: the term removed at each stage (PRUNE-4).
    ``rss_per_size``, ``gcv_per_size`` (M_f,): R[m] and GCV(R[m], m) at index
    m - 1. ``subsets`` (M_f, M_f) bool: row m - 1 marks the terms of T[m].
    ``selected_size``: m*. ``selected`` (m*,) int64: the selected terms,
    increasing (PRUNE-5, PRUNE-7).
    """

    removed: IntArray
    rss_per_size: FloatArray
    gcv_per_size: FloatArray
    subsets: BoolArray
    selected_size: int
    selected: IntArray


class FinalFit(NamedTuple):
    """``final_fit(B, Y, selected, w, ...)``: ``coef`` (m*, K), the weighted
    least-squares coefficients, and the ``rss``, ``gcv``, ``rsq`` and ``grsq``
    of the returned model (PRUNE-8)."""

    coef: FloatArray
    rss: float
    gcv: float
    rsq: float
    grsq: float


def _cases(
    B: npt.ArrayLike, Y: npt.ArrayLike, w: npt.ArrayLike | None
) -> tuple[FloatArray, FloatArray, FloatArray | None, float, float]:
    """Check the input, drop the rows with zero weight (W-3), and return B, Y as
    (n, K), w, N and τ_N (W-4). Complexity: O(n·(M + K))."""
    B = np.asarray(B, dtype=np.float64)
    if B.ndim != 2 or B.shape[1] < 1 or B.shape[0] < 1:
        raise ValueError(
            f"B must be a 2-D array with at least one column, not {B.shape}"
        )
    n = B.shape[0]
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    if Y.ndim != 2 or Y.shape[0] != n or Y.shape[1] < 1:
        raise ValueError(f"Y must have shape ({n},) or ({n}, K), not {Y.shape}")
    if not (np.isfinite(B).all() and np.isfinite(Y).all()):
        raise ValueError("B and Y must be finite")
    if w is not None:
        w = np.asarray(w, dtype=np.float64)
        if w.shape != (n,) or not (np.isfinite(w).all() and (w >= 0.0).all()):
            raise ValueError(f"w must be finite and nonnegative, with shape ({n},)")
        keep = w > 0.0
        if not keep.any():
            raise ValueError("every weight is zero")
        if not keep.all():
            B, Y, w = B[keep], Y[keep], w[keep]
    if np.any(B[:, 0] != 1.0):
        raise ValueError("column 0 of B must be the intercept, a column of ones")
    N, tau = _gcv.total_weight(B.shape[0], w)
    return B, Y, w, N, tau


def _centered(Y: FloatArray, w: FloatArray | None) -> tuple[FloatArray, FloatArray]:
    """Return Y minus one constant per response, and the constants.

    Each response is shifted by its data value nearest its weighted mean μ (the
    first such case in a tie), then by the weighted mean of the differences.
    The shift a is a data value, so the differences are exact for values
    within a factor of 2 of it, and N·(a - μ)² ≤ TSS, so the rounding of the
    others is of the size of the spread, whatever the weights and the order
    of the cases. A constant response becomes 0 exactly, and its constant is
    its value. Complexity: O(n·K).
    """
    near = np.abs(Y - np.average(Y, axis=0, weights=w)).argmin(axis=0)
    anchor = Y[near, np.arange(Y.shape[1])]
    D = Y - anchor
    mean = np.average(D, axis=0, weights=w)
    return D - mean, anchor + mean


def _stages(
    R: FloatArray, Z: FloatArray, rss: float, several: bool
) -> tuple[IntArray, FloatArray, BoolArray]:
    """Run the stages of PRUNE-3 on the factor of the forward order.

    Returns ``removed``, R[m] for m = 1, …, M_f and the rows of T[m]. With
    ``several`` false (K = 1), each new working order is offered: a prefix of
    size m replaces T[m] when its RSS is lower than R[m], so the earlier offer
    wins a tie. With ``several`` true (K ≥ 2), T[m] is the set left after
    M_f - m removals. Complexity: O(M_f³·(M_f + K)) in all, at most
    O(M_f²·(M_f + K)) per stage.
    """
    M = R.shape[1]
    order = np.arange(M)
    offered = _linalg.prefix_rss(Z, rss)
    best = offered.copy()
    subsets = np.tri(M, dtype=bool)  # the prefixes of the starting order
    removed = np.empty(M - 1, dtype=np.int64)
    for pos in range(M, 1, -1):
        # Step 1: the RSS without each term at positions 2, …, pos (0-based 1 to
        # pos - 1); ties go to the term with the largest index.
        drop = offered[pos - 1] + _linalg.drop_costs(R, Z, pos)[1:]
        tied = 1 + np.flatnonzero(drop == drop.min())
        i = int(tied[np.argmax(order[tied])])
        # Step 2: record the term and move it to position pos.
        removed[M - pos] = order[i]
        R, Z = _linalg.move_column(R, Z, i, pos - 1)
        order = np.insert(np.delete(order, i), pos - 1, order[i])
        # Step 3: offer the new order.
        offered = _linalg.prefix_rss(Z, rss)
        sizes = [pos - 2] if several else np.flatnonzero(offered < best)
        for m in sizes:
            best[m] = offered[m]
            subsets[m] = False
            subsets[m, order[: m + 1]] = True
    return removed, best, subsets


def pruning_pass(
    B: npt.ArrayLike,
    Y: npt.ArrayLike,
    w: npt.ArrayLike | None = None,
    *,
    penalty: float,
    pmethod: str = "backward",
    nprune: int | None = None,
) -> PruningPass:
    """Run the pruning pass on the kept forward terms (PRUNE-1 to PRUNE-7).

    PRUNE-3: with one response (Y 1-D or with one column), the stages offer the
    prefixes of a working order, leaps-style, and T[m] is the subset of size m
    with the lowest RSS among those offered, so the sets need not be nested;
    with K ≥ 2 columns, the pass is plain backward elimination on the summed
    RSS. K counts the columns of Y, constant ones included. At every stage the
    removed term is the one whose drop gives the lowest RSS, ties going to the
    term with the largest index, and the intercept (term 0) is never removed.
    PRUNE-4: the records; ``gcv_per_size`` follows GCV-2 with N = Σw (W-2),
    and it is +∞ for a degenerate fit, N ≤ 1 or every response constant
    (GCV-7). PRUNE-5: with ``pmethod="backward"`` the selected size is the
    smallest m ≤ min(M_f, ``nprune``) with the lowest GCV, and the selected
    terms are T[m*]. PRUNE-6: ``nprune`` changes only that range. PRUNE-7: with
    ``pmethod="none"`` the records are the same, m* = min(``nprune``, M_f), and
    the selected terms are 0, …, m* - 1.

    B (n, M_f), Y and w are as in the module docstring, with M_f ≤ n; the
    penalty d is resolved (GCV-1, GCV-4) and ``nprune`` is None or an integer
    ≥ 1 (LIMIT-3). Raises ValueError for input outside these ranges.
    Complexity: O(n·M_f·(M_f + K)) for the factor and O(M_f³·(M_f + K)) for the
    stages, at most O(n·M_f²) per stage for K ≤ M_f (PRUNE-9);
    memory O(n·(M_f + K) + M_f²).
    """
    if pmethod not in PMETHODS:
        raise ValueError(f"pmethod must be 'backward' or 'none', not {pmethod!r}")
    B, Y, w, N, tau = _cases(B, Y, w)
    M = B.shape[1]
    m_max = _gcv.nprune_limit(M, nprune)
    Bc = B.copy()
    if M > 1:
        Bc[:, 1:] = _centered(B[:, 1:], w)[0]  # the same spans (LA-5, #84)
    R, Z, rss = _linalg.r_factor(Bc, _centered(Y, w)[0], w)
    # fit_mars cannot reach this: FWD-11 drops a column constant over the cases.
    if np.any(np.diag(R) == 0.0):
        raise ValueError("the columns of B must be linearly independent (FWD-11)")
    removed, rss_per_size, subsets = _stages(R, Z, rss, several=Y.shape[1] >= 2)
    gcv_per_size = _gcv.gcv(rss_per_size, np.arange(1, M + 1), penalty, N, tau)
    if _gcv.is_degenerate(Y, N):
        gcv_per_size = np.full(M, np.inf)
    if pmethod == "backward":
        size = int(np.argmin(gcv_per_size[:m_max])) + 1
        selected = np.flatnonzero(subsets[size - 1])
    else:
        size = m_max
        selected = np.arange(size)
    return PruningPass(
        removed, rss_per_size, gcv_per_size, subsets, size, selected.astype(np.int64)
    )


def final_fit(
    B: npt.ArrayLike,
    Y: npt.ArrayLike,
    selected: npt.ArrayLike,
    w: npt.ArrayLike | None = None,
    *,
    penalty: float,
) -> FinalFit:
    """Return the coefficients and statistics of the selected terms (PRUNE-8).

    ``coef`` holds the weighted least-squares coefficients of Y on the columns
    ``selected`` of B, in increasing term order, one column per response, with
    LA-4 for dependent columns (``_linalg.lm_fit``). The fit is of the
    centered responses, and the intercept's coefficient takes back their
    constants, so a constant response gets its value there and 0 for the other
    terms, exactly. ``rss`` is the weighted RSS of the centered fit: that of
    the returned coefficients, up to the rounding of the intercept's
    coefficient. ``gcv``, ``rsq`` and ``grsq`` follow GCV-2 and GCV-5 to GCV-7
    with that RSS, M = m* and N = Σw. With ``pmethod="none"`` and m* < M_f the
    selected terms are not T[m*], so ``rss`` differs from
    ``rss_per_size[m* - 1]`` (PRUNE-7).

    B, Y and w are as for ``pruning_pass``; ``selected`` is an increasing
    integer array that starts with the intercept, 0. Raises ValueError
    otherwise. Complexity: O(n·m*·(m* + K)) time, O(n·(m* + K)) memory.
    """
    B, Y, w, N, tau = _cases(B, Y, w)
    sel = np.asarray(selected)
    if (
        sel.ndim != 1
        or sel.size == 0
        or not np.issubdtype(sel.dtype, np.integer)
        or sel[0] != 0
        or np.any(np.diff(sel) <= 0)
        or sel[-1] >= B.shape[1]
    ):
        raise ValueError(
            f"selected must be increasing term indices below {B.shape[1]}, from 0"
        )
    m = sel.size
    Yc, constants = _centered(Y, w)
    fit = _linalg.lm_fit(B[:, sel], Yc, w)
    coef = fit.coef
    coef[0] += constants  # column 0 is the intercept, a column of ones
    rss = fit.rss
    gcv = _gcv.gcv(rss, m, penalty, N, tau)
    if _gcv.is_degenerate(Y, N):
        gcv = np.inf
    tss = _gcv.tss(Y, w)
    rsq = _gcv.rsq(rss, tss, m)
    grsq = _gcv.grsq(rss, tss, m, penalty, N, tau)
    return FinalFit(coef, rss, float(gcv), rsq, grsq)
