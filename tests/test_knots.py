"""Tests of pymars/_knots.py: the SPAN and KNOT rules of docs/algorithm.md.

The earth fixture is ``validation/fixtures/components/knot_candidates``: the
``trace = 9`` logs of 45 earth 5.3.4 fits (degree 1 and 2; minspan automatic,
1, 3 and 10; endspan automatic, 1 and 5; Adjust.endspan 1 and 2; 20, 200 and
2,000 cases). For every knot search in them the test compares the minspan,
the endspan, the start of the counter, the visited cases and the evaluated
knots, exactly. ``validation/harness/trace_parse.py`` reads the logs.
"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pymars import _gcv, _knots

HARNESS = Path(__file__).resolve().parents[1] / "validation" / "harness"


def _trace_parse():
    """Import the harness's trace parser, a plain script outside the package."""
    if "trace_parse" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "trace_parse", HARNESS / "trace_parse.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["trace_parse"] = module
        spec.loader.exec_module(module)
    return sys.modules["trace_parse"]


def knot3(x, active, L, E):
    """KNOT-2 and KNOT-3 transcribed, with unit weights: positions q = n..E*+2."""
    order = np.lexsort((active, x))  # by x; among equal x, inactive first
    xs, a = x[order], active[order]
    n = len(xs)
    if not a.any():
        return []
    v = xs[a].max()
    counter = E + math.ceil(((n - 2 * E - 1) % L) / 2)
    listed = []
    for q in range(n, E + 1, -1):  # 1-based positions
        t = xs[q - 2]
        if t >= v:
            continue
        if a[q - 1]:
            counter -= 1
            if counter == 0:
                listed.append(t)
                counter = L
    return sorted(set(listed), reverse=True)


def knot6(x, active, L, E, w, total=None):
    """KNOT-2 and KNOT-6 transcribed: u = N, N - 1, … on cumulative weight."""
    order = np.lexsort((active, x))
    xs, a, ws = x[order], active[order], w[order]
    N, tau = _gcv.total_weight(len(w), w)
    if total is not None:
        N, tau = total, _gcv.weight_tolerance(total)
    if not a.any():
        return []
    W = np.cumsum(ws)
    v = xs[a].max()

    def holder(u):  # the smallest q with u ≤ W_q + τ_N, else case n
        return min(int(np.searchsorted(W + tau, u, side="left")), len(W) - 1)

    counter = _knots.start_counter(N, L, E)
    listed = []
    k = 0
    while N - k >= E + 2 - tau:
        u = N - k
        t = xs[holder(u - 1)]
        if t < v and a[holder(u)]:
            counter -= 1
            if counter == 0:
                listed.append(t)
                counter = L
        k += 1
    return sorted(set(listed), reverse=True)


def test_auto_minspan():
    """SPAN-1: truncated, with the active weight N_b, and at least 1."""
    assert [_knots.auto_minspan(2, nb) for nb in (8, 11, 20)] == [3, 3, 3]
    assert _knots.auto_minspan(1, 20) == 3
    assert _knots.auto_minspan(1, 2000) == 6
    # p·N_b = 200 gives 4.77, so rounding would give 5.
    assert -math.log2(-math.log(0.95) / 200) / 2.5 == pytest.approx(4.77, abs=0.01)
    assert _knots.auto_minspan(1, 200) == 4
    assert _knots.auto_minspan(5, 40) == _knots.auto_minspan(1, 200)
    assert _knots.auto_minspan(1, 2.5) == 2
    assert _knots.auto_minspan(1, 1) == 1
    assert _knots.auto_minspan(1, 0.2) == 1  # the outer max binds
    assert _knots.auto_minspan(1, 0.01) == 1
    assert _knots.auto_minspan(3, 0) == 1
    spans = [_knots.auto_minspan(1, nb) for nb in range(1, 5000)]
    assert spans == sorted(spans)
    for p, nb in ((0, 10), (True, 10), (1, -1.0), (1, math.nan), (1, math.inf)):
        with pytest.raises(ValueError):
            _knots.auto_minspan(p, nb)


def test_auto_endspan():
    """SPAN-2: E = trunc(3 - log₂(alpha/p)), from p alone."""
    assert [_knots.auto_endspan(p) for p in (1, 2, 5, 10, 100)] == [7, 8, 9, 10, 13]
    with pytest.raises(ValueError):
        _knots.auto_endspan(0)


def test_adjusted_endspan():
    """SPAN-4: E + ⌊a·E + 0.5⌋ for a parent of degree ≥ 1, E for the intercept."""
    assert _knots.adjusted_endspan(8, 0, 2.0) == 8
    assert _knots.adjusted_endspan(8, 1, 2.0) == 24
    assert _knots.adjusted_endspan(8, 2, 1.0) == 16
    assert _knots.adjusted_endspan(8, 1, 0.0) == 8
    assert _knots.adjusted_endspan(5, 1, 0.5) == 8  # 2.5 rounds half up
    assert _knots.adjusted_endspan(1, 1, 2.0) == 3
    # In float64 the form matters: 0.58·25 rounds below 14.5, and 1.58·25 to
    # 39.5, so ⌊(1 + a)·E + 0.5⌋ would give 40 and 45 here (bb06.12).
    assert _knots.adjusted_endspan(25, 1, 0.58) == 39
    assert math.floor((1 + 0.58) * 25 + 0.5) == 40
    assert _knots.adjusted_endspan(25, 1, 0.82) == 46
    assert math.floor((1 + 0.82) * 25 + 0.5) == 45
    for args in (
        (0, 1, 2.0),
        (8, -1, 2.0),
        (8, 1, -0.5),
        (8, 1, math.nan),
        (8, 1, True),
    ):
        with pytest.raises(ValueError):
            _knots.adjusted_endspan(*args)


def test_capped_endspan():
    """SPAN-5: E* = max(1, min(E₁, ⌊(N + τ_N)/2⌋ - 1)), with N of all cases."""
    tau20 = _gcv.weight_tolerance(20)
    assert _knots.capped_endspan(24, 20, tau20) == 9
    assert _knots.capped_endspan(8, 20, tau20) == 8
    assert _knots.capped_endspan(24, 21, tau20) == 9
    assert _knots.capped_endspan(24, 22, tau20) == 10
    assert _knots.capped_endspan(5, 3, 1e-8) == 1
    assert _knots.capped_endspan(5, 1, 1e-8) == 1
    # τ_N decides a sum just below an even number.
    assert _knots.capped_endspan(100, 21.99999995, 2.2e-7) == 10
    assert _knots.capped_endspan(100, 21.99999995, 0.0) == 9
    with pytest.raises(ValueError):
        _knots.capped_endspan(0, 20, tau20)


def test_search_spans():
    """SPAN-1 to SPAN-6 together, with the values of an earth trace (n = 20,
    p = 2): the intercept, and a degree-1 parent with 8 active cases."""
    assert _knots.search_spans(2, 0, 20.0, 20.0) == (3, 8)
    assert _knots.search_spans(2, 1, 20.0, 8.0) == (3, 9)
    assert _knots.search_spans(2, 1, 20.0, 8.0, adjust_endspan=0.0) == (3, 8)
    assert _knots.search_spans(2, 1, 20.0, 8.0, endspan=1) == (3, 3)
    assert _knots.search_spans(2, 1, 20.0, 8.0, endspan=1, adjust_endspan=1.0) == (3, 2)
    assert _knots.search_spans(1, 0, 20.0, 20.0, minspan=50, endspan=40) == (50, 9)
    assert _knots.search_spans(1, 0, 20.0, 20.0, tau=0.0) == (3, 7)
    for kwargs in ({"minspan": 0}, {"endspan": 0}, {"minspan": 2.0}, {"endspan": True}):
        with pytest.raises(ValueError):
            _knots.search_spans(2, 0, 20.0, 20.0, **kwargs)


def test_start_counter():
    """KNOT-3, KNOT-6: c₀ = E* + ⌈g/2⌉, g = (N - 2E* - 1) mod L in [0, L)."""
    assert _knots.start_counter(20, 3, 8) == 8
    assert _knots.start_counter(20, 3, 9) == 10
    assert _knots.start_counter(12, 3, 2) == 3
    assert _knots.start_counter(30, 4, 7) == 9
    assert _knots.start_counter(20, 1, 5) == 5
    assert _knots.start_counter(20.5, 3, 8) == 9
    assert _knots.start_counter(2, 3, 1) == 2  # D = -1, g = 2
    with pytest.raises(ValueError):
        _knots.start_counter(20, 0, 1)


def run(x, active, L, E, w=None):
    return _knots.candidate_knots(
        np.asarray(x, float), np.asarray(active, bool), L, E, w
    )


def test_knots_by_hand():
    """KNOT-3 and KNOT-4 for the intercept with distinct values: every L-th
    value from the top of x_(E*+1)..x_(n-E*), the slack split between the
    ends with the larger half at the top."""
    x = np.arange(1.0, 13.0)
    result = run(x, np.ones(12), 3, 2)
    np.testing.assert_array_equal(result.knots, [9.0, 6.0, 3.0])
    np.testing.assert_array_equal(result.split, [9, 6, 3])
    assert result.knots.dtype == np.float64 and result.split.dtype == np.int64
    assert not result.at_minimum
    np.testing.assert_array_equal(run(x[:10], np.ones(10), 1, 1).knots, x[1:9][::-1])
    np.testing.assert_array_equal(run(x[:3], [1, 1, 1], 1, 1).knots, [2.0])
    assert run(x[:3], [1, 1, 0], 1, 1).knots.size == 0
    assert run(x[:2], [1, 1], 1, 1).knots.size == 0


def test_knots_ties_and_activity():
    """KNOT-2: among equal x the inactive cases come first, whatever the row
    order; KNOT-4: the largest active value is never a knot, and a knot can
    be the value of an inactive case."""
    x = [1.0, 2.0, 3.0, 3.0, 4.0, 5.0]
    # The active 3 counts at the 3's upper position, so 2.0 is not listed; with
    # the active case first among the 3s, 2.0 would be listed as well.
    np.testing.assert_array_equal(run(x, [1, 1, 0, 1, 1, 1], 1, 1).knots, [4.0, 3.0])
    np.testing.assert_array_equal(run(x, [1, 1, 1, 0, 1, 1], 1, 1).knots, [4.0, 3.0])
    np.testing.assert_array_equal(
        run([1, 2, 3, 4], [1, 0, 1, 1], 1, 1).knots, [3.0, 2.0]
    )
    np.testing.assert_array_equal(
        run([1, 2, 3, 4, 5], [1, 1, 1, 0, 0], 1, 1).knots, [2.0]
    )
    assert run([1, 2, 3, 4], [0, 0, 0, 0], 1, 1).knots.size == 0


def test_knot_at_minimum():
    """KNOT-5: the lowest knot x_(E*+1) equals x_(1) only for a repeated minimum."""
    result = run([0.0, 0.0, 1.0, 2.0, 3.0], np.ones(5), 1, 1)
    np.testing.assert_array_equal(result.knots, [2.0, 1.0, 0.0])
    np.testing.assert_array_equal(result.split, [4, 3, 2])
    assert result.at_minimum
    assert not run([0.0, 0.5, 1.0, 2.0, 3.0], np.ones(5), 1, 1).at_minimum


def test_zero_knot_sign():
    """Conventions: -0.0 and 0.0 are equal x values, and a knot of zero is
    +0.0 whichever comes first, so the knots do not depend on the row order."""
    for x in ([-0.0, 0.0, 1.0, 2.0, 3.0], [0.0, -0.0, 1.0, 2.0, 3.0]):
        knots = run(x, np.ones(5), 1, 1).knots
        assert knots.tolist() == [2.0, 1.0, 0.0]
        assert math.copysign(1.0, knots[-1]) == 1.0
        for m in (_knots.linear_option_knot(x), _knots.linear_option_knot(x[::-1])):
            assert m == 0.0 and math.copysign(1.0, m) == 1.0


def test_linear_option_knot():
    """FWD-6, KNOT-5: m is the smallest x over all n cases, not the active ones."""
    assert _knots.linear_option_knot([3.0, -1.0, 2.0]) == -1.0
    for bad in ([], [[1.0]], [1.0, math.nan]):
        with pytest.raises(ValueError):
            _knots.linear_option_knot(bad)


def test_fractional_weights():
    """W-4: 100 cases of weight 0.1, 10 at each of 10 values, give the spans
    and knots of 10 cases of weight 1."""
    x10 = np.arange(10.0)
    x100 = np.repeat(x10, 10)
    w = np.full(100, 0.1)
    N, tau = _gcv.total_weight(100, w)
    assert N == 10.0
    for active10 in (np.ones(10, bool), np.arange(10) % 3 != 1):
        active100 = np.repeat(active10, 10)
        nb = _gcv.weight_sum(w[active100], tau)
        assert nb == active10.sum()
        spans = _knots.search_spans(1, 1, N, nb, minspan=None, endspan=1)
        assert spans == _knots.search_spans(
            1, 1, 10.0, float(active10.sum()), endspan=1
        )
        for L in (1, 2, 3):
            np.testing.assert_array_equal(
                _knots.candidate_knots(x100, active100, L, 1, w).knots,
                run(x10, active10, L, 1).knots,
            )


def test_case_n_holds_the_top():
    """KNOT-6: a u above W_n + τ_N is held by case n. W-4 bounds the rounding
    of W below τ_N/10, so only an N above the weight sum reaches this rule;
    here N exceeds it by 0.5, and u = N counts for the active top case."""
    x = np.arange(1.0, 7.0)
    active = np.ones(6, bool)
    w = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.3])
    knots = _knots.candidate_knots(x, active, 1, 1, w, total=5.8).knots
    assert knots.tolist() == knot6(x, active, 1, 1, w, total=5.8) == [4.0, 3.0]


def test_candidate_knots_errors():
    x = np.array([1.0, 2.0, 3.0])
    ones = np.ones(3, bool)
    for args in (
        ([3.0, 2.0, 1.0], ones, 1, 1, None),
        (x, [1, 1, 1], 1, 1, None),
        (x, ones[:2], 1, 1, None),
        ([1.0, math.nan, 3.0], ones, 1, 1, None),
        (np.empty(0), np.empty(0, bool), 1, 1, None),
        (x, ones, 0, 1, None),
        (x, ones, 1, 0, None),
        (x, ones, 1, 1, [1.0, 0.0, 1.0]),
        (x, ones, 1, 1, [1.0, 1.0]),
        (x, ones, 1, 1, [1.0, math.inf, 1.0]),
    ):
        with pytest.raises(ValueError):
            _knots.candidate_knots(*args)


def test_candidate_knots_keeps_inputs():
    x = np.array([1.0, 2.0, 2.0, 5.0])
    active = np.array([True, False, True, True])
    w = np.array([1.0, 2.0, 0.5, 1.5])
    copies = x.copy(), active.copy(), w.copy()
    _knots.candidate_knots(x, active, 1, 1, w)
    for before, after in zip(copies, (x, active, w), strict=True):
        np.testing.assert_array_equal(before, after)


@st.composite
def scans(draw, weights="none"):
    """x sorted with many ties, any activity, small spans, and weights."""
    n = draw(st.integers(1, 30))
    levels = draw(st.integers(1, n))
    x = np.sort(
        np.array(
            draw(st.lists(st.integers(0, levels - 1), min_size=n, max_size=n)), float
        )
    )
    active = np.array(draw(st.lists(st.booleans(), min_size=n, max_size=n)))
    L, E = draw(st.integers(1, 6)), draw(st.integers(1, 6))
    if weights == "integer":
        w = np.array(draw(st.lists(st.integers(1, 4), min_size=n, max_size=n)), float)
    elif weights == "real":
        tenths = st.integers(1, 30).map(lambda k: k / 10)
        real = st.floats(0.01, 3.0, allow_subnormal=False)
        w = np.array(draw(st.lists(tenths | real, min_size=n, max_size=n)))
    else:
        w = None
    return x, active, L, E, w


def check_split(x, result):
    np.testing.assert_array_equal(
        result.split, np.searchsorted(x, result.knots, "right")
    )


@given(scans())
def test_matches_knot3(scan):
    """The O(n) scan equals KNOT-3 transcribed, with unit weights (W-5)."""
    x, active, L, E, _ = scan
    result = _knots.candidate_knots(x, active, L, E)
    assert result.knots.tolist() == knot3(x, active, L, E)
    check_split(x, result)
    ones = _knots.candidate_knots(x, active, L, E, np.ones(len(x)))
    np.testing.assert_array_equal(ones.knots, result.knots)


@given(st.one_of(scans("integer"), scans("real")))
def test_matches_knot6(scan):
    """The O(n) scan equals KNOT-6 transcribed, with integer and real weights."""
    x, active, L, E, w = scan
    result = _knots.candidate_knots(x, active, L, E, w)
    assert result.knots.tolist() == knot6(x, active, L, E, w)
    check_split(x, result)


@given(scans("integer"))
def test_integer_weights_repeat_rows(scan):
    """KNOT-6, W-1: an integer weight w_i gives the knots of w_i copies of row i."""
    x, active, L, E, w = scan
    reps = w.astype(int)
    np.testing.assert_array_equal(
        _knots.candidate_knots(x, active, L, E, w).knots,
        _knots.candidate_knots(np.repeat(x, reps), np.repeat(active, reps), L, E).knots,
    )


@given(scans("real"), st.randoms(use_true_random=False))
def test_row_order(scan, random):
    """KNOT-2, Conventions: the knots do not depend on the order of the rows."""
    x, active, L, E, w = scan
    perm = list(range(len(x)))
    random.shuffle(perm)
    order = np.argsort(x[perm], kind="stable")
    shuffled = np.array(perm)[order]
    np.testing.assert_array_equal(
        _knots.candidate_knots(x[shuffled], active[shuffled], L, E, w[shuffled]).knots,
        _knots.candidate_knots(x, active, L, E, w).knots,
    )


@given(scans(), st.integers(-30, 30), st.integers(-100, 100))
def test_properties(scan, exponent, shift):
    """KNOT-4 with unit weights: knots are data values, below the largest
    active value and not below x_(E*+1); a positive scale or a shift of x
    moves the knots with it (exact for a power of 2 and integer data)."""
    x, active, L, E, _ = scan
    knots = _knots.candidate_knots(x, active, L, E).knots
    if knots.size:
        assert set(knots) <= set(x)
        assert knots.max() < x[active].max()
        assert knots.min() >= x[E]
    c = 2.0**exponent
    np.testing.assert_array_equal(
        _knots.candidate_knots(x * c, active, L, E).knots, knots * c
    )
    np.testing.assert_array_equal(
        _knots.candidate_knots(x + shift, active, L, E).knots, knots + shift
    )


@pytest.fixture(scope="module")
def knot_cases(load_fixture):
    return load_fixture("components/knot_candidates")["cases"]


def test_fixture_grid(knot_cases):
    """The fixture covers the plan's grid, trimmed as the harness states."""
    assert len(knot_cases) == 45
    assert {c["degree"] for c in knot_cases} == {1, 2}
    assert {c["minspan"] for c in knot_cases} == {None, 1, 3, 10}
    assert {c["endspan"] for c in knot_cases} == {None, 1, 5}
    assert {c["adjust_endspan"] for c in knot_cases} == {1.0, 2.0}
    assert {c["n"] for c in knot_cases} == {20, 200, 2000}


@pytest.mark.parametrize("index", range(45))
def test_earth_knot_searches(knot_cases, index, tmp_path):
    """SPAN-1 to SPAN-5, KNOT-1 to KNOT-4 against every knot search that
    earth traced: L (nMinSpan), E* (nEndSpan), c₀ (nStartSpan), the visited
    positions n - 1 down to E* + 1, and the set of evaluated knots, exactly.

    earth prints ``Case c`` for the 0-based sorted position c; the knot of
    that case is the value at position c - 1 (KNOT-3). Parents are earth's
    slots: 1 is the intercept, 2 and 3 the pair (x - t)₊, (t - x)₊ of step 1
    (FWD-6, FWD-9), active where x - t > 0 and t - x > 0 (KNOT-1).
    """
    case = knot_cases[index]
    X = np.array(case["X"])
    n, p = X.shape
    dirs = np.array(case["dirs"], dtype=np.int8)
    (var,) = np.flatnonzero(dirs[1])
    assert dirs[1, var] == 1 and dirs[2].tolist() == (-dirs[1]).tolist()
    t = case["cuts"][1][var]
    assert case["cuts"][2][var] == t
    signs = (np.ones(n), X[:, var] - t, t - X[:, var])
    degrees = (0, 1, 1)
    N, tau = _gcv.total_weight(n)
    path = tmp_path / "trace.txt"
    path.write_text(case["trace_text"], encoding="utf-8")
    searches = [
        s
        for step in _trace_parse().parse_trace(path).steps
        for s in step.searches
        if s.min_span is not None
    ]
    assert searches
    for search in searches:
        row, j = search.parent - 1, search.pred - 1
        active = signs[row] > 0.0
        order = np.argsort(X[:, j], kind="stable")
        x = X[order, j]
        L, E = _knots.search_spans(
            p,
            degrees[row],
            N,
            float(active.sum()),
            minspan=case["minspan"],
            endspan=case["endspan"],
            adjust_endspan=case["adjust_endspan"],
            tau=tau,
        )
        assert (search.min_span, search.end_span) == (L, E)
        assert _knots.start_counter(N, L, E) == search.start_span
        assert [c.case for c in search.cases] == list(range(n - 1, E, -1))
        evaluated = [c for c in search.cases if c.evaluated]
        for c in evaluated:
            assert c.cut == pytest.approx(x[c.case - 1], rel=1e-4)
        expected = sorted({x[c.case - 1] for c in evaluated}, reverse=True)
        result = _knots.candidate_knots(x, active[order], L, E, total=N, tau=tau)
        assert result.knots.tolist() == expected
