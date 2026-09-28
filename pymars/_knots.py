"""Spans and candidate knots of one knot search.

Spec: ``docs/algorithm.md``, "Spans" (SPAN-1 to SPAN-6), "Candidate knots"
(KNOT-1 to KNOT-7) and the sums of weights of W-4. The forward pass calls
these for each parent b and covariate x (FWD-2, FWD-3); FWD-6 uses
``linear_option_knot``.

A knot search has a parent term b and a covariate x. A case is active when
b(x_i) > 0 (KNOT-1); N is the weight sum of all n cases of the fit and N_b
that of the active cases, both by W-4 (``_gcv.total_weight`` and
``_gcv.weight_sum``). ``search_spans`` gives the minspan L and the endspan E*
of the search, and ``candidate_knots`` the knots, from x sorted once per
column (KNOT-7).

Weights. The scan runs on cumulative weight, so that an integer weight w_i
gives the same knots as w_i copies of row i (KNOT-6, W-1); w=None means
w_i = 1 exactly, and then the scan is KNOT-3 (W-5). The weights must be
positive: rows with zero weight are dropped before anything else (W-3).

Numerics. float64 and exact integer arithmetic; no function writes into its
inputs; every comparison of a sum of weights with an integer or a grid point
uses τ_N as SPAN-5 and KNOT-6 state, and no other tolerance. The knots do not
depend on the order of the rows, since cases with equal x are ordered by
activity alone (KNOT-2).

Public names, for ``_scan`` and ``_forward``:

- ``ALPHA``: the probability alpha = 0.05 of the span formulas.
- ``auto_minspan``, ``auto_endspan``, ``adjusted_endspan``, ``capped_endspan``,
  ``search_spans``: the spans (SPAN-1 to SPAN-6).
- ``start_counter``, ``candidate_knots``, ``KnotCandidates``: the scan
  (KNOT-1 to KNOT-7).
- ``linear_option_knot``: the knot m of the linear option (FWD-6, KNOT-5).
"""

from __future__ import annotations

import math
import numbers
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from pymars import _gcv

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

#: SPAN-1, SPAN-2: the probability alpha of Friedman's span formulas (F91 eq. 43, 45).
ALPHA = 0.05


class KnotCandidates(NamedTuple):
    """``candidate_knots``: the knots of one search.

    ``knots`` (k,) float64: the distinct candidate knots, largest first, the
    order in which FWD-5 breaks ties. ``split`` (k,) int64: the number of cases
    with x ≤ knots[i], so the hinge (x - knots[i])₊ is positive exactly at the
    sorted positions split[i] to n - 1 (the suffix sums of the fast path).
    ``at_minimum``: True when the lowest knot equals the smallest x of all n
    cases, which KNOT-5 allows only when that value is repeated; then
    b·(x - t)₊ = b·x - t·b at every case.
    """

    knots: FloatArray
    split: IntArray
    at_minimum: bool


def _int(value: object, name: str, minimum: int) -> int:
    """Return value as an int ≥ minimum; a bool is not an int here (CORE-2)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer, not {value!r}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, not {value}")
    return value


def _nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number, not {value!r}")
    value = float(value)
    if not (math.isfinite(value) and value >= 0.0):
        raise ValueError(f"{name} must be finite and at least 0, not {value}")
    return value


def auto_minspan(p: int, active_weight: float) -> int:
    """Return the automatic minspan L of SPAN-1 (F91 eq. 43).

    L = max(1, trunc(-log₂(-ln(1 - alpha)/(p·N_b))/2.5)), truncated, not
    rounded, with alpha = 0.05, p the number of columns of X and N_b the
    weight of the active cases of the parent (the count with unit weights),
    after W-4. The float64 evaluation order is
    z = -ln(1 - alpha)/(p·N_b), then trunc(-log₂(z)/2.5). The outer max keeps
    L ≥ 1 for small weights; with no active case (N_b = 0) L is 1, the limit
    of the formula, and the scan finds no knot anyway. Complexity: O(1).
    """
    p = _int(p, "p", 1)
    size = p * _nonnegative(active_weight, "active_weight")
    if size == 0.0:
        return 1
    z = -math.log(1.0 - ALPHA) / size
    return max(1, math.trunc(-math.log2(z) / 2.5))


def auto_endspan(p: int) -> int:
    """Return the automatic endspan E = trunc(3 - log₂(alpha/p)) (SPAN-2, F91 eq. 45).

    It depends only on p, the number of columns of X. Complexity: O(1).
    """
    return math.trunc(3.0 - math.log2(ALPHA / _int(p, "p", 1)))


def adjusted_endspan(endspan: int, parent_degree: int, adjust_endspan: float) -> int:
    """Return E₁ = E + ⌊a·E + 0.5⌋ for a parent of degree ≥ 1, else E (SPAN-4).

    a = ``adjust_endspan`` ≥ 0 and E is the user or the automatic endspan.
    E₁ is computed in float64 in this form, as earth does; it is E·(1 + a)
    rounded half up, a quirk that pymars copies. For the intercept (degree 0),
    E₁ = E. Complexity: O(1).
    """
    E = _int(endspan, "endspan", 1)
    a = _nonnegative(adjust_endspan, "adjust_endspan")
    if _int(parent_degree, "parent_degree", 0) == 0:
        return E
    return E + math.floor(a * E + 0.5)


def capped_endspan(endspan: int, total: float, tau: float) -> int:
    """Return E* = max(1, min(E₁, ⌊(N + τ_N)/2⌋ - 1)) (SPAN-5).

    E₁ = ``endspan`` is the adjusted endspan (SPAN-4); N = ``total`` counts all
    n cases, not only the active ones; τ_N = ``tau``. The cap applies to user
    and automatic values alike. Complexity: O(1).
    """
    E1 = _int(endspan, "endspan", 1)
    cap = math.floor((float(total) + float(tau)) / 2.0) - 1
    return max(1, min(E1, cap))


def search_spans(
    p: int,
    parent_degree: int,
    total: float,
    active_weight: float,
    *,
    minspan: int | None = None,
    endspan: int | None = None,
    adjust_endspan: float = 2.0,
    tau: float | None = None,
) -> tuple[int, int]:
    """Return (L, E*), the minspan and the endspan of one knot search.

    SPAN-1 to SPAN-6. ``minspan`` and ``endspan`` are the user values, integers
    ≥ 1 used in place of L and E, or None for the automatic ones (SPAN-3).
    The minspan is not capped (SPAN-6). The endspan is adjusted for a parent
    of degree ≥ 1 (SPAN-4) and then capped by N (SPAN-5). N = ``total`` and
    N_b = ``active_weight`` are the weight sums after W-4, and τ_N = ``tau``,
    by default ``_gcv.weight_tolerance(total)``. Complexity: O(1).
    """
    tau = _gcv.weight_tolerance(total) if tau is None else float(tau)
    L = (
        auto_minspan(p, active_weight)
        if minspan is None
        else _int(minspan, "minspan", 1)
    )
    E = auto_endspan(p) if endspan is None else _int(endspan, "endspan", 1)
    return L, capped_endspan(
        adjusted_endspan(E, parent_degree, adjust_endspan), total, tau
    )


def start_counter(total: float, minspan: int, endspan: int) -> int:
    """Return c₀ = E* + ⌈g/2⌉, the start of the scan's counter (KNOT-3, KNOT-6).

    g = D - L·⌊D/L⌋ and D = N - 2E* - 1, so g is the remainder of D in
    [0, L): the slack of the scan is split between its two ends, with the
    larger half at the top. N = ``total`` after W-4, L = ``minspan`` and
    E* = ``endspan``. Complexity: O(1).
    """
    L = _int(minspan, "minspan", 1)
    E = _int(endspan, "endspan", 1)
    D = float(total) - 2 * E - 1
    g = D - L * math.floor(D / L)
    return E + math.ceil(g / 2.0)


def _empty() -> KnotCandidates:
    return KnotCandidates(np.empty(0), np.empty(0, dtype=np.int64), False)


def candidate_knots(
    x: npt.ArrayLike,
    active: npt.ArrayLike,
    minspan: int,
    endspan: int,
    w: npt.ArrayLike | None = None,
    *,
    total: float | None = None,
    tau: float | None = None,
) -> KnotCandidates:
    """Return the candidate knots of one parent and one covariate.

    x (n,) holds the covariate of all n cases in ascending order, with any
    order among equal values; ``active`` (n,) bool holds b(x_i) > 0 in the
    same order (KNOT-1), and w (n,) the positive weights, or None for w_i = 1.
    L = ``minspan`` and E* = ``endspan`` come from ``search_spans``;
    N = ``total`` and τ_N = ``tau`` default to ``_gcv.total_weight(n, w)``.

    The scan is KNOT-6, which is KNOT-3 with unit weights. Among cases with
    equal x the inactive ones come first (KNOT-2). The scan visits
    u = N, N - 1, … while u ≥ E* + 2 - τ_N. The case that holds u is the
    first q with u ≤ W_q + τ_N (W_q the running weight), or case n; let a(u)
    be its activity, and t = x(u - 1) the value of the case that holds u - 1.
    Steps with t ≥ v, the largest active x, are skipped. The others with
    a(u) = 1 move a counter that starts at ``start_counter(N, L, E*)``; when
    it reaches 0, t is listed and the counter restarts at L. The knots are
    the distinct listed values, so no knot is at or above v, none is below
    x_(E*+1), and a knot can be the value of an inactive case (KNOT-4). With
    no active case there is no knot.

    Cases with equal x and equal activity are merged, since the scan does
    not tell them apart; the merged cases hold whole intervals of u, and each
    interval is counted at once (KNOT-7). Raises ValueError for unsorted or
    nonfinite x, a non-bool ``active``, or weights that are not positive.
    Complexity: O(n) time and memory.
    """
    x = np.asarray(x, dtype=np.float64)
    active = np.asarray(active)
    if x.ndim != 1 or x.shape[0] < 1:
        raise ValueError(f"x must be a nonempty 1-D array, not shape {x.shape}")
    n = x.shape[0]
    if active.dtype != np.bool_ or active.shape != x.shape:
        raise ValueError("active must be a bool array of the shape of x")
    if not np.isfinite(x).all():
        raise ValueError("x must be finite")
    if np.any(x[1:] < x[:-1]):
        raise ValueError("x must be sorted in ascending order")
    L = _int(minspan, "minspan", 1)
    E = _int(endspan, "endspan", 1)
    if w is not None:
        w = np.asarray(w, dtype=np.float64)
        if w.shape != x.shape or not (np.isfinite(w).all() and (w > 0.0).all()):
            raise ValueError("w must be finite and positive, one weight per case")
    if total is None:
        total, default_tau = _gcv.total_weight(n, w)
    else:
        default_tau = _gcv.weight_tolerance(total)
    N = float(total)
    tau = default_tau if tau is None else float(tau)
    k_max = math.floor(N - E - 2 + tau)  # the scan visits u = N - k, k = 0..k_max
    if k_max < 0 or not active.any():
        return _empty()

    # KNOT-2: one segment per run of equal x and equal activity, inactive first.
    first = np.ones(n, dtype=bool)
    first[1:] = x[1:] != x[:-1]
    starts = np.flatnonzero(first)
    n_active = np.add.reduceat(active.astype(np.int64), starts)
    counts = np.column_stack((np.diff(np.append(starts, n)) - n_active, n_active))
    if w is None:
        weights = counts.astype(np.float64)
    else:
        weights = np.column_stack(
            (
                np.add.reduceat(np.where(active, 0.0, w), starts),
                np.add.reduceat(np.where(active, w, 0.0), starts),
            )
        )
    keep = counts.ravel() > 0
    seg_x = np.repeat(x[starts], 2)[keep]
    seg_end = np.repeat(np.append(starts[1:], n), 2)[keep]  # end of its run of x
    seg_active = np.tile(np.array([False, True]), starts.shape[0])[keep]
    cum = np.cumsum(weights.ravel()[keep])  # W at the end of each segment (W-4)
    G = seg_x.shape[0]
    v = seg_x[np.flatnonzero(seg_active)[-1]]

    # KNOT-6: u = N - k lies in segment g when k is in [lo[g], hi[g]), since
    # u ≤ W + τ_N reads k ≥ N - W - τ_N. The top segment also holds every u
    # above its end ("case n if there is none"). hi[0] is set past k_max and
    # past lo[0], so the lowest segment is never empty and has no boundary.
    lo = np.ceil(N - cum - tau).astype(np.int64)
    lo[-1] = min(lo[-1], 0)
    hi = np.empty(G, dtype=np.int64)
    hi[0] = max(lo[0], k_max) + 2
    hi[1:] = lo[:-1]
    # t = x(u - 1): the own value of g inside its range, and at its last index
    # hi[g] - 1 the value of the next segment below that holds some u.
    held = np.where(lo < hi, np.arange(G), 0)
    below = np.zeros(G, dtype=np.int64)
    below[1:] = np.maximum.accumulate(held)[:-1]
    k0 = np.maximum(lo, 0)
    own_len = np.maximum(np.minimum(hi - 1, k_max + 1) - k0, 0)
    edge_len = ((hi - 1 >= k0) & (hi - 1 <= k_max)).astype(np.int64)

    # The runs in scan order (decreasing u): for g from the top, own then edge.
    top_down = np.arange(G - 1, -1, -1)
    run_seg = np.column_stack((top_down, below[top_down])).ravel()
    run_len = np.column_stack((own_len[top_down], edge_len[top_down])).ravel()
    run_active = np.repeat(seg_active[top_down], 2)
    run_t = seg_x[run_seg]
    counted = np.where(run_active & (run_t < v), run_len, 0)
    before = np.cumsum(counted) - counted
    # The counter hits 0 at the counted steps c0, c0 + L, c0 + 2L, …
    c0 = start_counter(N, L, E)
    first_step = np.maximum(before + 1, c0)
    hit = c0 + L * ((first_step - c0 + L - 1) // L)
    listed = (counted > 0) & (hit <= before + counted)
    knot_seg = run_seg[listed]
    t = seg_x[knot_seg]  # non-increasing along the scan
    distinct = np.ones(t.shape[0], dtype=bool)
    distinct[1:] = t[1:] != t[:-1]
    knots, split = t[distinct], seg_end[knot_seg][distinct].astype(np.int64)
    return KnotCandidates(knots, split, bool(knots.size) and bool(knots[-1] == x[0]))


def linear_option_knot(x: npt.ArrayLike) -> float:
    """Return m, the smallest value of x over all n cases (FWD-6, KNOT-5).

    With ``auto_linpreds=False`` the linear candidate b·x adds the hinge
    b·(x - m)₊ with this knot, which equals b·x - m·b on the training data;
    the active cases of b play no part. The linear option is not a scan
    candidate. Complexity: O(n).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.shape[0] < 1 or not np.isfinite(x).all():
        raise ValueError("x must be a nonempty, finite 1-D array")
    return float(x.min())
