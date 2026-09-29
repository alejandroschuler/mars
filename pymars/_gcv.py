"""The GCV and the fit statistics, the default penalty and the term limit.

Spec: ``docs/algorithm.md``, "GCV and fit statistics" (GCV-1 to GCV-8), "Term
limit" (LIMIT-1 to LIMIT-3) and the sums of weights of W-4. The rules that
call these functions are the stopping rules (STOP-3 to STOP-6), the pruning
pass (PRUNE-4, PRUNE-5, PRUNE-7, PRUNE-8), the spans (SPAN-1, SPAN-5) and the
knot scan (KNOT-6).

Weights. N = Σw takes the place of the number of cases everywhere (W-2), and
w=None means w_i = 1 exactly, so that N = n (W-5). N is ``math.fsum`` of the
weights, and τ_N = min(1e-8·N, 0.1) is the tolerance for sums of weights. A
sum within τ_N of an integer is replaced by that integer (W-4). As in the
reference (T06), τ_N comes from the ``math.fsum`` value before this
replacement; the functions that take N compute τ_N from the N they are given
unless ``tau`` is passed, and the two differ by less than 1e-9 in absolute
terms.

Numerics. float64 throughout; no function writes into its inputs; no
absolute epsilon. A response is constant when all its values are equal, and
its sum of squares is then 0.0 exactly ("Conventions for all rules").

Public functions, for ``_knots``, ``_forward``, ``_pruning``, ``_core`` and
the estimators:

- ``weight_tolerance``, ``snap_weight_sum``, ``total_weight``, ``weight_sum``:
  N, N_b and τ_N (W-4, W-5).
- ``effective_parameters``, ``gcv``: C(M) and the GCV (GCV-1 to GCV-3).
- ``tss``, ``rsq``, ``grsq``, ``is_degenerate``: the fit statistics (GCV-5 to
  GCV-8).
- ``default_penalty``, ``default_max_terms``, ``max_forward_steps``,
  ``nprune_limit``: the defaults and limits (GCV-4, LIMIT-1 to LIMIT-3).
"""

from __future__ import annotations

import math
import numbers

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

#: W-4: τ_N = min(TAU_RELATIVE·N, TAU_CAP).
TAU_RELATIVE = 1e-8
#: W-4: the cap keeps τ_N below the unit step of the knot scan (KNOT-6).
TAU_CAP = 0.1


def _int(value: object, name: str, minimum: int) -> int:
    """Return value as an int ≥ minimum; a bool is not an int here (CORE-2)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer, not {value!r}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, not {value}")
    return value


def _total(total: float) -> float:
    total = float(total)
    if not (math.isfinite(total) and total > 0.0):
        raise ValueError(f"the weight sum N must be finite and positive, not {total}")
    return total


def weight_tolerance(total: float) -> float:
    """Return τ_N = min(1e-8·N, 0.1) for the weight sum N (W-4).

    Complexity: O(1).
    """
    return min(TAU_RELATIVE * _total(total), TAU_CAP)


def snap_weight_sum(value: float, tau: float) -> float:
    """Return the integer nearest to a sum of weights when it is within τ_N (W-4).

    Where N or N_b is within ``tau`` (τ_N) of an integer, that integer replaces
    it in every rule; otherwise the sum is returned as it is. With integer
    weights every sum is exact, and nothing changes. Complexity: O(1).
    """
    value = float(value)
    nearest = round(value)
    return float(nearest) if abs(value - nearest) <= tau else value


def _weights(w: npt.ArrayLike, n: int | None = None) -> FloatArray:
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 1 or (n is not None and w.shape[0] != n):
        raise ValueError(
            f"w must be a 1-D array with one weight per case, not {w.shape}"
        )
    if not (np.isfinite(w).all() and (w >= 0.0).all()):
        raise ValueError("weights must be finite and nonnegative")
    return w


def total_weight(n: int, w: npt.ArrayLike | None = None) -> tuple[float, float]:
    """Return N and τ_N for the n cases of a fit (W-4, W-5).

    With w=None, N = n exactly. Otherwise N is ``math.fsum`` of the n weights,
    which is exactly rounded and so does not depend on the order of the cases;
    τ_N = min(1e-8·N, 0.1) from that sum; and N is then replaced by the nearest
    integer when it is within τ_N of it. The weights must be finite and ≥ 0
    with a positive sum (the core drops zero weights first, W-3). Returns
    ``(N, tau)``. Complexity: O(n).
    """
    n = _int(n, "n", 1)
    if w is None:
        return float(n), weight_tolerance(n)
    raw = math.fsum(_weights(w, n))
    tau = weight_tolerance(raw)
    return snap_weight_sum(raw, tau), tau


def weight_sum(w: npt.ArrayLike, tau: float) -> float:
    """Return a sum of weights, such as N_b, by W-4: ``math.fsum``, then snapped.

    ``tau`` is τ_N of the whole fit, from ``total_weight``: W-4 replaces N_b by
    an integer within τ_N of it. The sum of no weights is 0.0.
    Complexity: O(len(w)).
    """
    return snap_weight_sum(math.fsum(_weights(w)), tau)


def _penalty(penalty: float) -> float:
    if isinstance(penalty, bool) or not isinstance(penalty, numbers.Real):
        raise ValueError(f"penalty must be a real number, not {penalty!r}")
    d = float(penalty)
    if d != -1.0 and not (math.isfinite(d) and d >= 0.0):
        raise ValueError(f"penalty must be -1, or finite and at least 0, not {d}")
    return d


def _terms(n_terms: npt.ArrayLike) -> npt.NDArray[np.int64]:
    M = np.asarray(n_terms)
    if M.dtype == np.bool_ or not np.issubdtype(M.dtype, np.integer):
        raise ValueError("the number of terms must be an integer")
    if np.any(M < 1):
        raise ValueError("the number of terms must be at least 1")
    return M.astype(np.int64)


def effective_parameters(n_terms: npt.ArrayLike, penalty: float) -> float | FloatArray:
    """Return C(M), the effective number of parameters of a model (GCV-1).

    C(M) = M + d·(M - 1)/2 when d ≥ 0, and C(M) = 0 when d = -1, where M counts
    the terms, the intercept included (not the rank of B, and a single hinge
    or a linear term counts as one term). The intercept-only model has C = 1
    for every d ≥ 0. A penalty below 0 other than -1 raises ValueError.
    ``n_terms`` is an int or an integer array; the result is a float or a new
    float64 array of its shape. Complexity: O(size of n_terms).
    """
    d = _penalty(penalty)
    M = _terms(n_terms)
    C = np.zeros(M.shape) if d == -1.0 else M + d * (M - 1) / 2.0
    return float(C) if C.ndim == 0 else C


def gcv(
    rss: npt.ArrayLike,
    n_terms: npt.ArrayLike,
    penalty: float,
    total: float,
    tau: float | None = None,
) -> float | FloatArray:
    """Return GCV(RSS, M) = RSS / (N·(1 - C(M)/N)²), or +∞ when C(M) ≥ N - τ_N.

    GCV-2, with C(M) of GCV-1 and N the weight sum of the fit (W-2, W-4); τ_N
    is ``weight_tolerance(total)`` unless ``tau`` is given. With several
    responses ``rss`` is the RSS summed over them, which gives the sum of the
    per-response GCVs, since the GCV is linear in the RSS (GCV-3). With
    d = -1, C = 0 and GCV = RSS/N. A degenerate fit (GCV-7) has GCV +∞ whatever
    this returns; the core applies that override. ``rss`` and ``n_terms``
    broadcast; the result is a float or a new float64 array.
    Complexity: O(size of the broadcast inputs).
    """
    N = _total(total)
    tau = weight_tolerance(N) if tau is None else float(tau)
    params = np.asarray(effective_parameters(n_terms, penalty), dtype=np.float64)
    R, params = np.broadcast_arrays(np.asarray(rss, dtype=np.float64), params)
    out = np.full(R.shape, np.inf)
    ok = params < N - tau
    out[ok] = R[ok] / (N * (1.0 - params[ok] / N) ** 2)
    return float(out) if out.ndim == 0 else out


def tss(Y: npt.ArrayLike, w: npt.ArrayLike | None = None) -> float:
    """Return the weighted total sum of squares, summed over the responses.

    TSS = Σ_k Σ_i w_i (Y_ik - Ȳ_k)², with Ȳ_k the weighted mean of column k
    (GCV-5); the sums are not normalized (GCV-8). A response whose values over
    the cases are all equal adds 0.0 exactly, not a computed value
    (Conventions, GCV-7), so the TSS of constant responses is 0.0. Each
    response is first shifted by its value nearest its weighted mean (the
    first such case in a tie): the differences are exact for values within a
    factor of 2 of it, and the rounding is of the size of the spread, not of
    the mean, whatever the weights and the order of the cases. Y is (n,) or
    (n, K); w is (n,), or None for w_i = 1 (W-5). The rows are the cases of the
    fit, after zero weights are dropped (W-3). Complexity: O(n·K).
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    if Y.ndim != 2 or Y.shape[0] < 1:
        raise ValueError(f"Y must have shape (n,) or (n, K) with n ≥ 1, not {Y.shape}")
    wv = np.ones(Y.shape[0]) if w is None else _weights(w, Y.shape[0])
    total = 0.0
    for k in range(Y.shape[1]):
        y = Y[:, k]
        if np.all(y == y[0]):
            continue
        d = y - y[np.argmin(np.abs(y - float(wv @ y) / float(wv.sum())))]
        r = d - float(wv @ d) / float(wv.sum())
        total += float(wv @ (r * r))
    return total


def rsq(rss: float, tss_value: float, n_terms: int) -> float:
    """Return RSq = 1 - RSS/TSS (GCV-5), and 0.0 for the intercept-only model.

    GCV-7: RSq = 0 for M = 1 by definition, not computed. A model with M > 1
    is not degenerate, so TSS > 0; a TSS that is not positive and finite
    raises ValueError. With several responses RSS and TSS are the pooled sums
    (STOP-6). Complexity: O(1).
    """
    M = _int(n_terms, "n_terms", 1)
    if M == 1:
        return 0.0
    if not (math.isfinite(tss_value) and tss_value > 0.0):
        raise ValueError(f"RSq needs a positive TSS for M > 1, not {tss_value} (GCV-7)")
    return 1.0 - float(rss) / float(tss_value)


def grsq(
    rss: float,
    tss_value: float,
    n_terms: int,
    penalty: float,
    total: float,
    tau: float | None = None,
) -> float:
    """Return GRSq = 1 - GCV(RSS, M)/GCV(TSS, 1) (GCV-6), 0.0 for M = 1.

    Both GCVs use the model's penalty d and weight sum N (GCV-2). GRSq = -∞
    when GCV(RSS, M) is +∞, that is when C(M) ≥ N - τ_N. GCV-7: GRSq = 0 for
    the intercept-only model by definition. A model with M > 1 is not
    degenerate, so GCV(TSS, 1) is positive and finite; otherwise ValueError.
    Complexity: O(1).
    """
    M = _int(n_terms, "n_terms", 1)
    if M == 1:
        return 0.0
    base = gcv(tss_value, 1, penalty, total, tau)
    if not (math.isfinite(base) and base > 0.0):
        raise ValueError(
            f"GRSq needs a positive, finite GCV(TSS, 1), not {base} (GCV-7)"
        )
    return 1.0 - gcv(rss, M, penalty, total, tau) / base


def is_degenerate(Y: npt.ArrayLike, total: float) -> bool:
    """Return True for a degenerate fit: N ≤ 1, or every response constant.

    GCV-7 and EDGE-1: a response is constant when all its values over the n
    cases are equal (Conventions). ``total`` is N after W-4, so a sum within
    τ_N of 1 counts as 1. Y is (n,) or (n, K) over the cases of the fit.
    Complexity: O(n·K).
    """
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y[:, None]
    if Y.ndim != 2 or Y.shape[0] < 1:
        raise ValueError(f"Y must have shape (n,) or (n, K) with n ≥ 1, not {Y.shape}")
    return _total(total) <= 1.0 or bool(np.all(Y[0] == Y))


def default_penalty(max_degree: int) -> float:
    """Return the default penalty d: 2 when ``max_degree`` is 1, else 3 (GCV-4).

    Complexity: O(1).
    """
    return 2.0 if _int(max_degree, "max_degree", 1) == 1 else 3.0


def default_max_terms(p: int) -> int:
    """Return the default term limit M_max = min(200, max(20, 2p)) + 1 (LIMIT-1).

    p is the number of columns of X; the limit does not depend on n.
    Complexity: O(1).
    """
    return min(200, max(20, 2 * _int(p, "p", 1))) + 1


def max_forward_steps(max_terms: int) -> int:
    """Return the largest number of forward steps, ⌊(M_max - 1)/2⌋ (LIMIT-2).

    Every step counts 2 toward M_max, whatever it adds, and a step is taken
    only when 1 + 2(s + 1) ≤ M_max; so with M_max ≤ 2 the pass takes no step,
    and when M_max is even its last slot is never used. Complexity: O(1).
    """
    return (_int(max_terms, "max_terms", 1) - 1) // 2


def nprune_limit(n_terms: int, nprune: int | None) -> int:
    """Return the largest size that pruning may select, min(M_f, nprune).

    LIMIT-3: ``nprune`` is None (no limit) or an integer ≥ 1, the largest size
    in terms, the intercept included. This is m_max of PRUNE-5 and m* of
    PRUNE-7. Complexity: O(1).
    """
    M = _int(n_terms, "n_terms", 1)
    return M if nprune is None else min(M, _int(nprune, "nprune", 1))
