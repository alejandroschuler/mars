"""Tests of pymars/_gcv.py: the GCV, LIMIT and W-4 rules of docs/algorithm.md.

The earth fixtures are ``validation/fixtures/components/gcv_grid``
(``earth:::get.gcv`` for 1 to 41 terms, penalties 0 to 6 and -1, and n from
10 to 100,000) and the S datasets (``rss.per.subset``, ``gcv.per.subset``,
``rss``, ``rsq`` and ``grsq`` of earth 5.3.4). Tolerances, from the plan's
tolerance table: relative 1e-8 for GCV and sums of squares, absolute 1e-8 for
RSq and GRSq.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from pymars import _gcv

FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "fixtures"


def _s_fits(*, same_tss: bool) -> list[str]:
    """The S fixtures with an earth fit whose sums of squares are all positive:
    earth reports values below about 1e-10 in the units of y as 0 (a
    departure, PRUNE-4 and PRUNE-8), and a factor response is several
    indicator responses, not y itself. With ``same_tss``, only those where
    earth's TSS follows GCV-8 too: no zero weight, and weights not all equal
    (``test_s_fixture_weight_departures`` covers the others)."""
    names = []
    for path in sorted(FIXTURES.glob("S*.json")):
        fixture = json.loads(path.read_text(encoding="utf-8"))
        result, w = fixture["result"], fixture["inputs"]["weights"]
        if result.get("error") or result.get("levels") is not None:
            continue
        if min(result["rss_per_subset"]) <= 0.0 or result["rss"] <= 0.0:
            continue
        if same_tss and w is not None and (0.0 in w or len(set(w)) == 1):
            continue
        names.append(path.stem)
    return names


S_FITS = _s_fits(same_tss=False)
S_SAME_TSS_FITS = _s_fits(same_tss=True)


def assert_rel(actual, expected, rtol):
    """Equal infinities, and finite values within the relative tolerance."""
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    np.testing.assert_array_equal(np.isinf(actual), np.isinf(expected))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=rtol, atol=0)


def test_weight_tolerance():
    """W-4: τ_N = min(1e-8·N, 0.1)."""
    assert _gcv.weight_tolerance(20) == 20 * 1e-8
    assert _gcv.weight_tolerance(1e7) == 0.1
    assert _gcv.weight_tolerance(3e9) == 0.1
    for bad in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="weight sum"):
            _gcv.weight_tolerance(bad)


def test_snap_weight_sum():
    """W-4: a sum within τ_N of an integer is that integer."""
    assert _gcv.snap_weight_sum(6.000000005, 6e-8) == 6.0
    assert _gcv.snap_weight_sum(5.99999999, 6e-8) == 6.0
    assert _gcv.snap_weight_sum(6.5, 6e-8) == 6.5
    assert _gcv.snap_weight_sum(6.0000001, 6e-8) == 6.0000001
    assert _gcv.snap_weight_sum(0.25, 0.25) == 0.0


def test_total_weight():
    """W-4 and W-5: N = fsum, τ_N from it, then N snapped; no weights, N = n."""
    assert _gcv.total_weight(7) == (7.0, _gcv.weight_tolerance(7))
    N, tau = _gcv.total_weight(4, [1.0, 2.0, 3.0 + 5e-9, 0.0])
    assert (N, tau) == (6.0, _gcv.weight_tolerance(6.000000005))
    assert _gcv.total_weight(100, np.full(100, 0.1)) == (
        10.0,
        _gcv.weight_tolerance(10),
    )
    assert _gcv.total_weight(3, [0.5, 0.25, 2.0])[0] == 2.75
    w = np.random.default_rng(0).uniform(0, 1, 500)
    assert _gcv.total_weight(500, w) == _gcv.total_weight(500, w[::-1])
    for bad in ([1.0, -1.0], [1.0, math.nan], [[1.0, 1.0]], [1.0]):
        with pytest.raises(ValueError):
            _gcv.total_weight(2, bad)
    with pytest.raises(ValueError, match="weight sum"):
        _gcv.total_weight(2, [0.0, 0.0])
    with pytest.raises(ValueError, match="integer"):
        _gcv.total_weight(True)


def test_weight_sum():
    """W-4: N_b is the fsum of the active weights, snapped with the fit's τ_N."""
    assert _gcv.weight_sum([2.0, 3.0 + 1e-9], 1e-7) == 5.0
    assert _gcv.weight_sum([0.1] * 10, 1e-7) == 1.0
    assert _gcv.weight_sum([], 1e-7) == 0.0
    assert _gcv.weight_sum([0.3, 0.4], 1e-7) == math.fsum([0.3, 0.4])


def test_effective_parameters():
    """GCV-1: C(M) = M + d·(M - 1)/2, C(1) = 1 for d ≥ 0, and C = 0 for d = -1."""
    for d in (0, 0.5, 2, 3, 1000):
        assert _gcv.effective_parameters(1, d) == 1.0
    assert _gcv.effective_parameters(5, 2) == 9.0
    assert _gcv.effective_parameters(5, 3) == 11.0
    assert _gcv.effective_parameters(21, -1) == 0.0
    np.testing.assert_array_equal(
        _gcv.effective_parameters(np.arange(1, 5), 3), [1, 3.5, 6, 8.5]
    )
    for bad in (-0.5, -2, -10, math.nan, math.inf, True, "2"):
        with pytest.raises(ValueError, match="penalty"):
            _gcv.effective_parameters(3, bad)
    for bad in (0, 1.5, True, [2, 0]):
        with pytest.raises(ValueError, match="number of terms"):
            _gcv.effective_parameters(bad, 2)


def test_gcv_formula():
    """GCV-2: RSS / (N·(1 - C/N)²); d = -1 gives RSS/N."""
    assert _gcv.gcv(10.0, 3, 2, 20) == pytest.approx(
        10 / (20 * (1 - 5 / 20) ** 2), rel=1e-15
    )
    assert _gcv.gcv(10.0, 5, -1, 20) == 0.5
    assert _gcv.gcv(3.0, 4, 1.0, 12.5) == pytest.approx(
        3 / (12.5 * (1 - 5.5 / 12.5) ** 2)
    )
    out = _gcv.gcv(np.array([4.0, 2.0]), np.array([1, 3]), 2, 10)
    np.testing.assert_allclose(out, [4 / (10 * 0.81), 2 / (10 * 0.25)], rtol=1e-15)
    assert isinstance(_gcv.gcv(1.0, 1, 2, 10), float)


def test_gcv_infinite():
    """GCV-2: +∞ when C(M) ≥ N - τ_N, also when C is within τ_N below N."""
    assert _gcv.gcv(1.0, 20, 0, 20) == math.inf
    assert _gcv.gcv(1.0, 30, 0, 20) == math.inf
    assert _gcv.gcv(0.0, 20, 0, 20) == math.inf
    # C = 2 + d/2 at M = 2; earth's test C < n keeps these finite.
    tau = _gcv.weight_tolerance(20)
    assert _gcv.gcv(1.0, 2, 2 * (18 - tau / 2), 20) == math.inf
    assert math.isfinite(_gcv.gcv(1.0, 2, 2 * (18 - 2 * tau), 20))
    assert _gcv.gcv(1.0, 2, 2 * (18 - 2 * tau), 20, tau=3 * tau) == math.inf
    with pytest.raises(ValueError, match="weight sum"):
        _gcv.gcv(1.0, 2, 2, 0)


@given(
    st.floats(0, 1e6, allow_subnormal=False),
    st.floats(0, 1e6, allow_subnormal=False),
    st.integers(1, 41),
    st.sampled_from([-1.0, 0.0, 1.0, 2.0, 3.0, 6.0]),
    st.integers(2, 10_000),
)
def test_gcv_linear_in_rss(a, b, M, d, N):
    """GCV-3: the GCV is linear in the RSS, so the GCV of summed RSS values is
    the sum of their GCVs."""
    total, parts = _gcv.gcv(a + b, M, d, N), _gcv.gcv(a, M, d, N) + _gcv.gcv(b, M, d, N)
    if math.isinf(total):
        assert math.isinf(parts)
    else:
        assert total == pytest.approx(parts, rel=1e-14, abs=0)


def test_s_fixture_lists():
    """The collection found the fixtures."""
    assert len(S_FITS) >= 119 and len(S_SAME_TSS_FITS) >= 109


def test_gcv_grid(load_fixture):
    """GCV-1, GCV-2 against earth:::get.gcv: 64 cells of 41 sizes each, with
    penalty -1 (GCV = RSS/n) and the infinite cells where C ≥ n."""
    fixture = load_fixture("components/gcv_grid")
    M, rss = np.array(fixture["nterms"]), np.array(fixture["rss_per_subset"])
    penalties = set()
    for cell in fixture["grid"]:
        d, n = cell["penalty"], cell["ncases"]
        penalties.add(d)
        assert_rel(_gcv.gcv(rss, M, d, n), cell["gcv"], 1e-8)
        params = np.zeros(M.shape) if d == -1 else M + d * (M - 1) / 2
        np.testing.assert_array_equal(np.isinf(cell["gcv"]), params >= n)
    assert penalties == {-1, 0, 1, 2, 3, 4, 5, 6}
    assert sum(np.isinf(c["gcv"]).sum() for c in fixture["grid"]) > 0


def _fit(load_fixture, name):
    fixture = load_fixture(name)
    degree = fixture["earth_args"]["degree"]
    penalty = fixture["earth_args"].get("penalty", _gcv.default_penalty(degree))
    y = np.array(fixture["inputs"]["y"], dtype=float)
    return fixture["result"], y, fixture["inputs"]["weights"], penalty


@pytest.mark.parametrize("name", S_FITS)
def test_s_fixture_gcv(load_fixture, name):
    """GCV-2 on earth's pruning sizes and final model, with earth's penalty or
    the default of GCV-4. earth's GCV uses the number of cases also with
    weights; pymars passes N = Σw instead (W-2), with the same formula."""
    result, y, _, penalty = _fit(load_fixture, name)
    n = len(y)
    rss = np.array(result["rss_per_subset"])
    sizes = np.arange(1, len(rss) + 1)
    assert_rel(_gcv.gcv(rss, sizes, penalty, n), result["gcv_per_subset"], 1e-8)
    M = len(result["selected_terms"])
    assert_rel(_gcv.gcv(result["rss"], M, penalty, n), result["gcv"], 1e-8)


@pytest.mark.parametrize("name", S_SAME_TSS_FITS)
def test_s_fixture_statistics(load_fixture, name):
    """GCV-5 to GCV-8 against earth: the TSS (earth's RSS of the intercept),
    RSq and GRSq. With weights earth's GCV uses n, and TSS is weighted when
    the weights differ (GCV-8)."""
    result, y, weights, penalty = _fit(load_fixture, name)
    n = len(y)
    tss = _gcv.tss(y, weights)
    assert_rel(tss, result["rss_per_subset"][0], 1e-8)
    M = len(result["selected_terms"])
    assert abs(_gcv.rsq(result["rss"], tss, M) - result["rsq"]) <= 1e-8
    assert abs(_gcv.grsq(result["rss"], tss, M, penalty, n) - result["grsq"]) <= 1e-8


def test_s_fixture_weight_departures(load_fixture):
    """GCV-8, W-3: earth ignores weights that are all equal, so pymars's TSS is
    theirs times the weight; earth keeps zero-weight rows as tiny weights,
    which moves its TSS by about 3e-9 relative, while pymars drops them."""
    result, y, weights, _ = _fit(load_fixture, "S13_equal2_defaults_d1")
    assert_rel(_gcv.tss(y, weights), 2 * result["rss_per_subset"][0], 1e-8)
    for name in ("S13_int_zeros_defaults_d1", "S16_weighted_defaults_d1"):
        result, y, weights, _ = _fit(load_fixture, name)
        w = np.array(weights)
        tss = _gcv.tss(y[w > 0], w[w > 0])
        assert 1e-10 < abs(tss / result["rss_per_subset"][0] - 1) < 1e-8


def test_tss():
    """GCV-5, GCV-8: weighted, about the weighted mean, not normalized, and
    summed over the responses; a constant response adds 0.0 exactly."""
    assert _gcv.tss([1.0, 2.0, 3.0]) == 2.0
    assert _gcv.tss([1.0, 2.0, 3.0], [1.0, 1.0, 2.0]) == 2.75
    Y = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 8.0]])
    assert _gcv.tss(Y) == pytest.approx(2.0 + 6.0, rel=1e-15)
    constant = np.full(5, 0.1)
    w = np.full(5, 0.3)
    assert float(w @ (constant - (w @ constant) / w.sum()) ** 2) != 0.0
    assert _gcv.tss(constant, w) == 0.0
    assert _gcv.tss(np.column_stack((constant, constant))) == 0.0
    Y0 = Y.copy()
    _gcv.tss(Y, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(Y, Y0)
    for bad_y, bad_w in (
        (np.zeros((2, 2, 2)), None),
        (np.zeros(0), None),
        ([1.0, 2.0], [1.0]),
    ):
        with pytest.raises(ValueError):
            _gcv.tss(bad_y, bad_w)


@given(
    hnp.arrays(np.float64, st.integers(1, 12), elements=st.floats(-1e3, 1e3)),
    st.data(),
)
def test_tss_frequency_weights(y, data):
    """W-1: integer weights give the TSS of the repeated rows."""
    w = np.array(
        data.draw(st.lists(st.integers(1, 4), min_size=len(y), max_size=len(y)))
    )
    repeated = _gcv.tss(np.repeat(y, w))
    scale = float(np.max(np.abs(y))) ** 2 * w.sum()
    assert _gcv.tss(y, w.astype(float)) == pytest.approx(
        repeated, rel=1e-9, abs=1e-12 * scale
    )


def test_rsq_and_grsq():
    """GCV-5, GCV-6 and GCV-7."""
    assert _gcv.rsq(2.0, 10.0, 3) == 0.8
    assert _gcv.rsq(10.5, 10.0, 1) == 0.0
    assert _gcv.grsq(10.5, 10.0, 1, 2, 20) == 0.0
    expected = 1 - (2 / (20 * (1 - 5 / 20) ** 2)) / (10 / (20 * (1 - 1 / 20) ** 2))
    assert _gcv.grsq(2.0, 10.0, 3, 2, 20) == pytest.approx(expected, rel=1e-15)
    assert _gcv.grsq(2.0, 10.0, 3, -1, 20) == pytest.approx(1 - 2 / 10, rel=1e-15)
    assert _gcv.grsq(2.0, 10.0, 10, 3, 20) == -math.inf
    with pytest.raises(ValueError, match="GCV-7"):
        _gcv.rsq(1.0, 0.0, 2)
    with pytest.raises(ValueError, match="GCV-7"):
        _gcv.grsq(1.0, 0.0, 2, 2, 20)
    with pytest.raises(ValueError, match="GCV-7"):
        _gcv.grsq(1.0, 5.0, 2, 2, 1)


def test_is_degenerate():
    """GCV-7, EDGE-1: N ≤ 1, or every response constant over the cases."""
    y = np.array([1.0, 2.0, 3.0])
    assert _gcv.is_degenerate(y, 1.0)
    assert _gcv.is_degenerate(y, 0.5)
    assert not _gcv.is_degenerate(y, 1.5)
    assert _gcv.is_degenerate(np.full((4, 2), 0.1), 4.0)
    assert not _gcv.is_degenerate(np.array([[0.1, 1.0], [0.1, 2.0]]), 2.0)
    assert _gcv.is_degenerate(np.array([[7.0]]), 1.0)
    with pytest.raises(ValueError):
        _gcv.is_degenerate(np.zeros((2, 2, 2)), 2.0)


def test_defaults_and_limits():
    """GCV-4, LIMIT-1, LIMIT-2 and LIMIT-3."""
    assert [_gcv.default_penalty(k) for k in (1, 2, 3, 10)] == [2.0, 3.0, 3.0, 3.0]
    assert [_gcv.default_max_terms(p) for p in (1, 10, 11, 100, 101, 5000)] == [
        21,
        21,
        23,
        201,
        201,
        201,
    ]
    assert [_gcv.max_forward_steps(m) for m in (1, 2, 3, 4, 5, 21, 22)] == [
        0,
        0,
        1,
        1,
        2,
        10,
        10,
    ]
    assert _gcv.nprune_limit(10, None) == 10
    assert _gcv.nprune_limit(10, 3) == 3
    assert _gcv.nprune_limit(3, 10) == 3
    assert _gcv.nprune_limit(4, np.int64(2)) == 2
    for call in (
        lambda: _gcv.default_penalty(0),
        lambda: _gcv.default_penalty(1.0),
        lambda: _gcv.default_max_terms(0),
        lambda: _gcv.max_forward_steps(0),
        lambda: _gcv.nprune_limit(3, 0),
        lambda: _gcv.nprune_limit(3, True),
    ):
        with pytest.raises(ValueError):
            call()
