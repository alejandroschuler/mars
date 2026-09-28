"""Tests of pymars/_scan.py against brute force: explicit hinge columns and
least squares.

Spec: LA-2 (the RSS reduction of a candidate), LA-3 (the ratio rho), LA-5 (the
accuracy: 1e-8 of the RSS, and 1e-7·max(rho, tau)) and the plan's "Fast path"
(eq. 52). Every test of ``knot_scan`` also checks that its error bounds hold.
"""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pymars import _linalg, _scan

seeds = st.integers(0, 2**32 - 1)
TAU = 1e-5  # the smaller tolerance of LA-3


def _problem(seed, n, r, K, ties=False, shift=0.0, parent=True):
    """Sorted x, a parent b with zeros and negative values, a basis B whose
    first column is the intercept, its orthonormal Q, Y, E and every split."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    x = np.sort(np.round(x, 1) if ties else x) + shift
    b = rng.normal(size=n) * (rng.random(n) < 0.7) if parent else np.ones(n)
    B = np.column_stack((np.ones(n), rng.normal(size=(n, r - 1))))
    Q, _ = np.linalg.qr(B)
    Q *= np.sign(Q[0])  # column 0 is +1/√n, the intercept direction
    Y = rng.normal(size=(n, K))
    E = Y - Q @ (Q.T @ Y)
    return x, b, B, Q, Y, E, np.flatnonzero(x[1:] > x[:-1]) + 1


def _rss(A, Y):
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    return float(np.sum((Y - A @ coef) ** 2))


def _within(scan, i, rho, gain, rss, la5=True):
    """The bounds hold, and with ``la5`` the values meet LA-5."""
    assert abs(scan.ratio[i] - rho) <= scan.ratio_err[i]
    assert abs(scan.gain[i] - gain) <= scan.gain_err[i]
    if la5:
        assert abs(scan.ratio[i] - rho) <= 1e-7 * max(rho, TAU)
        assert abs(scan.gain[i] - gain) <= 1e-8 * rss


@given(
    seeds,
    st.integers(2, 40),
    st.integers(1, 4),
    st.booleans(),
    st.sampled_from([0.0, 1e6, 1e8]),
)
def test_hinge_products_equal_the_explicit_columns(seed, n, c, ties, shift):
    """The sums use differences of x only, so a shift adds no cancellation."""
    x, b, _, _, _, _, splits = _problem(seed, n, 1, 1, ties, shift)
    V = np.random.default_rng(seed + 1).normal(size=(n, c))
    HV, F = _scan.hinge_products(x, b, V, splits)
    assert HV.shape == (len(splits), c) and F.shape == (len(splits),)
    for i, s in enumerate(splits):
        h = b * np.maximum(x - x[s - 1], 0.0)
        scale = np.linalg.norm(h) * np.linalg.norm(V, axis=0)
        assert np.all(np.abs(HV[i] - h @ V) <= 1e-13 * scale)
        assert F[i] == pytest.approx(h @ h, rel=1e-13, abs=0.0)


@given(seeds, st.integers(1, 3000))
def test_suffix_sums_stay_within_their_bound(seed, m):
    """The blocked suffix sums: an error of at most gamma_{2⌈√m⌉}·Σ|terms|."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=m) * 10.0 ** rng.integers(-8, 8, size=m)
    out = _scan._suffix(a[:, None])[:, 0]
    bound = 2 * (math.isqrt(max(m - 1, 0)) + 1) * 2.0**-53
    for j in rng.choice(m, size=min(m, 20), replace=False):
        assert abs(out[j] - math.fsum(a[j:])) <= bound * np.sum(np.abs(a[j:]))


@given(
    seeds,
    st.integers(8, 60),
    st.integers(1, 4),
    st.integers(1, 2),
    st.booleans(),
    st.booleans(),
    st.sampled_from([0.0, 1e6]),
)
def test_knot_scan_equals_least_squares(seed, n, r, K, parent, ties, shift):
    """A general parent (``intercept=False``), with ties and a shift of x."""
    x, b, B, Q, Y, E, splits = _problem(seed, n, r, K, ties, shift, parent)
    scan = _scan.knot_scan(x, b, Q, E, splits)
    base = _rss(B, Y)
    for i, s in enumerate(splits):
        h = b * np.maximum(x - x[s - 1], 0.0)
        centered = np.sum((h - h.mean()) ** 2)
        if not np.any(h):  # a constant hinge column: 0 at every case
            assert scan.ratio[i] == scan.gain[i] == scan.ratio_err[i] == 0.0
            continue
        rho = _rss(B, h) / centered
        assert abs(scan.ratio[i] - rho) <= max(scan.ratio_err[i], 1e-12)
        if rho > 1e-6:
            gain = base - _rss(np.column_stack((B, h)), Y)
            assert abs(scan.gain[i] - gain) <= max(scan.gain_err[i], 1e-10 * base)


def _pair_basis(x, extra):
    """The intercept, ``extra`` columns, then x: G of a pair search (LA-7)."""
    n = x.shape[0]
    Q = np.full((n, 1), n**-0.5)
    for v in [*extra, x]:
        Q = np.column_stack((Q, _linalg.gram_schmidt(Q, v).q))
    return Q


@given(seeds, st.integers(20, 300), st.booleans(), st.booleans())
def test_the_intercept_parent_in_a_pair_search(seed, n, ties, adjacent):
    """``intercept=True`` with G = 1, B and x, where the lowest knots have a
    small rho: every knot, or single hinges at adjacent knots, meets LA-5
    against the explicit values of ``exact_knot``."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(size=n))
    x = np.round(x, 2) if ties else x
    extra = [np.maximum(x - np.median(x), 0.0)]
    Q = _pair_basis(x, [] if adjacent else extra)
    if adjacent:  # a single-hinge search next to a hinge already in the model
        Q = np.column_stack((Q, _linalg.gram_schmidt(Q, extra[0]).q))
    yc = np.sin(6 * x) + 0.1 * rng.normal(size=n)
    E = _linalg.orthogonalize(Q, yc - yc.mean())[0]
    splits = np.flatnonzero(x[1:] > x[:-1]) + 1
    scan = _scan.knot_scan(x, np.ones(n), Q, E, splits, intercept=True)
    rss = float(E @ E)
    for i, s in enumerate(splits):
        rho, gain = _scan.exact_knot(Q, E, np.maximum(x - x[s - 1], 0.0))
        if rho > 1e-4:
            _within(scan, i, rho, gain, rss)


def _spec_case(n):
    """Finding 1 of the spec review: two low values below a tight bulk."""
    rng = np.random.default_rng(0)
    x = np.r_[0.0, 9.0, 1000 + 0.01 * np.sort(rng.uniform(size=n - 2))]
    y = rng.normal(size=n)
    y[0] = 50.0
    return x, _pair_basis(x, []), y - y.mean(), 2


def _cluster_case(n, beta=1.0, covariates=0):
    """Finding 1 of the adversarial review: 15 low values below a cluster, a
    pair at the median in the model, and the knot x[7]."""
    rng = np.random.default_rng(0)
    x = np.sort(np.r_[beta * rng.uniform(0, 1, 15), 50 + rng.uniform(size=n - 15)])
    cols = [np.maximum(x - x[n // 2], 0), np.maximum(x[n // 2] - x, 0)]
    for _ in range(covariates):
        z = rng.uniform(size=n)
        cols += [np.maximum(z - 0.5, 0), np.maximum(0.5 - z, 0)]
    Q = np.full((n, 1), n**-0.5)
    for v in cols:
        Q = np.column_stack((Q, _linalg.gram_schmidt(Q, v).q))
    y = np.maximum(x - x[7], 0) + 1e-3 * rng.normal(size=n)
    return x, Q, y - y.mean(), 8


def _normal_case(n):
    """The adversarial review's N(0, 1) covariate and the knot x[19]."""
    rng = np.random.default_rng(0)
    x = np.sort(rng.normal(size=n))
    Q = np.full((n, 1), n**-0.5)
    for v in (np.maximum(x - x[n // 2], 0), np.maximum(x[n // 2] - x, 0)):
        Q = np.column_stack((Q, _linalg.gram_schmidt(Q, v).q))
    y = 50 * np.maximum(x[19] - x, 0) + 1e-3 * rng.normal(size=n)
    return x, Q, y - y.mean(), 20


@pytest.mark.parametrize(
    "case",
    [
        lambda: _spec_case(1000),
        lambda: _spec_case(20_000),
        lambda: _cluster_case(10_000),
        lambda: _cluster_case(10_000, covariates=6),
        lambda: _normal_case(100_000),
    ],
    ids=["spec-1000", "spec-20000", "cluster", "cluster-6", "normal-1e5"],
)
def test_the_reviewers_cases(case):
    """Round 1 of #57: a hinge far from its own mean at large n. The intercept
    path meets LA-5 and its bounds hold; the general path (D = ‖h‖² - (q₀ᵀh)²)
    can miss LA-5 there, but its bounds hold too."""
    x, Q, yc, s = case()
    n = x.shape[0]
    E = _linalg.orthogonalize(Q, yc)[0]
    rss, h = float(E @ E), np.maximum(x - x[s - 1], 0.0)
    rho, gain = _scan.exact_knot(Q, E, h)
    assert TAU < rho < 1e-4  # a knot that LA-3 keeps from step 8 on
    fast = _scan.knot_scan(x, np.ones(n), Q, E, [s], intercept=True)
    _within(fast, 0, rho, gain, rss)
    slow = _scan.knot_scan(x, np.ones(n), Q, E, [s])
    _within(slow, 0, rho, gain, rss, la5=False)


def test_a_knot_at_tau_is_left_to_the_explicit_values():
    """The adversarial review's knot at tau + 5e-11: within its bound of tau, so
    the caller must decide LA-3 with ``exact_knot``, which keeps it."""
    x, Q, yc, s = _cluster_case(10_000, beta=0.5991315689034221)
    E = _linalg.orthogonalize(Q, yc)[0]
    rho, _ = _scan.exact_knot(Q, E, np.maximum(x - x[s - 1], 0.0))
    assert 0.0 < rho - TAU < 1e-10
    scan = _scan.knot_scan(x, np.ones(x.shape[0]), Q, E, [s], intercept=True)
    assert abs(scan.ratio[0] - TAU) <= scan.ratio_err[0]
    assert not _linalg.knot_rejected(rho, 8)


def test_a_constant_hinge_has_ratio_and_gain_zero():
    """LA-3: a constant h (here 0 at every case) is rejected, with bounds 0."""
    x, _, _, Q, _, E, splits = _problem(3, 20, 2, 1)
    b = np.where(np.arange(20) < 10, 1.0, 0.0)  # zero above case 9
    scan = _scan.knot_scan(x, b, Q, E, splits)
    above = splits >= 10
    for a in scan:
        assert np.all(a[above] == 0.0)
    assert np.all(scan.ratio[~above] > 0.0)


@pytest.mark.parametrize("intercept", [False, True])
def test_a_hinge_in_the_span_is_rejected_for_sure(intercept):
    """KNOT-5: with a repeated minimum, h = x - t·1 lies in the span of 1 and x,
    and rho plus its bound stays below tau."""
    x = np.array([0.0, 0.0, 1.0, 2.0, 4.0, 7.0])
    Q = _pair_basis(x, [])
    scan = _scan.knot_scan(x, np.ones(6), Q, np.zeros(6), [2, 3], intercept=intercept)
    assert abs(scan.ratio[0]) + scan.ratio_err[0] < TAU < scan.ratio[1]


def test_extreme_scales_change_nothing():
    """Powers of 2 inside keep the squares in range: x·2^510 and b·2^-500 give
    the values of x and b, bit for bit."""
    x, _, _, Q, _, E, splits = _problem(4, 30, 3, 1)
    b = np.ones(30)
    base = _scan.knot_scan(x, b, Q, E, splits, intercept=True)
    for xs, bs in ((np.ldexp(x, 510), b), (x, np.ldexp(b, -500))):
        big = _scan.knot_scan(xs, bs, Q, E, splits, intercept=True)
        for a, c in zip(big, base, strict=True):
            np.testing.assert_array_equal(a, c)


@given(seeds, st.integers(10, 60), st.booleans())
def test_exact_knot(seed, n, weighted):
    """rho equals collinearity_ratio, and the reduction least squares; integer
    weights give the values of repeated rows (W-1)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=n)
    w = rng.integers(1, 4, size=n).astype(float) if weighted else np.ones(n)
    sw, N = np.sqrt(w), w.sum()
    B = np.column_stack((np.ones(n), rng.normal(size=n)))
    Q, _ = np.linalg.qr(sw[:, None] * B)
    Q *= np.sign(Q[0])
    y = rng.normal(size=n)
    yc = sw * (y - w @ y / N)
    E = _linalg.orthogonalize(Q, yc)[0]
    h = np.maximum(x - np.median(x), 0.0)
    rho, gain = _scan.exact_knot(Q, E, h, w if weighted else None)
    assert rho == pytest.approx(_linalg.collinearity_ratio(Q, h, w), rel=1e-12)
    rep = np.repeat(np.arange(n), w.astype(int))
    A = np.column_stack((B[rep], h[rep]))
    want = _rss(B[rep], y[rep]) - _rss(A, y[rep])
    assert gain == pytest.approx(want, rel=1e-9, abs=1e-12)
    assert _scan.exact_knot(Q, E, np.zeros(n)) == (0.0, 0.0)


@given(seeds, st.integers(6, 30), st.integers(1, 3), st.integers(1, 2))
def test_rebuild_appends_orthonormal_columns(seed, n, r, K):
    _, _, B, Q, Y, _, _ = _problem(seed, n, r, K)
    new = np.random.default_rng(seed + 2).normal(size=(n, 2))
    rb = _scan.rebuild(Q, Y, new)
    assert rb.Q.shape == (n, r + 2)
    np.testing.assert_allclose(rb.Q.T @ rb.Q, np.eye(r + 2), atol=1e-12)
    np.testing.assert_allclose(rb.Q[:, :r], Q)
    A = np.column_stack((B, new))
    assert rb.rss == pytest.approx(_rss(A, Y), rel=1e-9, abs=1e-12)
    fit = A @ np.linalg.lstsq(A, Y, rcond=None)[0]
    np.testing.assert_allclose(rb.resid, Y - fit, atol=1e-10)
    one = _scan.rebuild(Q, Y[:, 0], new[:, :1])
    assert one.resid.shape == (n, 1)


def test_rebuild_orthogonalizes_twice():
    """A new column with a mean of 1e6: one pass of Gram-Schmidt would leave
    about 1e-10 of the intercept in it, two leave rounding."""
    rng = np.random.default_rng(5)
    n = 200
    Q = np.full((n, 1), n**-0.5)
    v = 1e6 + rng.uniform(size=n)
    yc = rng.normal(size=n)
    rb = _scan.rebuild(Q, yc - yc.mean(), v[:, None])
    assert abs(rb.Q[:, 0] @ rb.Q[:, 1]) < 1e-14
    assert np.max(np.abs(rb.Q.T @ rb.resid)) < 1e-13 * np.linalg.norm(rb.resid)


def test_rebuild_refuses_a_column_in_the_span():
    _, _, _, Q, Y, _, _ = _problem(1, 10, 2, 1)
    assert _scan.rebuild(Q, Y, np.zeros((10, 1))) is None


@pytest.mark.parametrize(
    ("x", "b", "V", "split", "match"),
    [
        ([1.0, 0.0], [1.0, 1.0], np.ones((2, 1)), [1], "sorted"),
        ([0.0, 2.0, 1.0], [1.0] * 3, np.ones((3, 1)), [1], "sorted"),
        ([0.0, np.nan], [1.0, 1.0], np.ones((2, 1)), [1], "finite"),
        ([0.0, 1.0], [np.inf, 1.0], np.ones((2, 1)), [1], "finite"),
        ([0.0, 1.0], [1.0, 1.0], np.full((2, 1), np.nan), [1], "finite"),
        ([0.0, 1.0], [1.0], np.ones((2, 1)), [1], "one row"),
        ([0.0, 1.0], [1.0, 1.0], np.ones(2), [1], "one row"),
        ([0.0, 1.0], [1.0, 1.0], np.ones((2, 1)), [2], r"split must be in 1\.\.1$"),
        ([0.0, 1.0], [1.0, 1.0], np.ones((2, 1)), [0], r"split must be in 1\.\.1$"),
        ([0.0, 1.0, 2.0], [1.0] * 3, np.ones((3, 1)), [1.9], "integer"),
    ],
)
def test_hinge_products_checks_its_input(x, b, V, split, match):
    with pytest.raises(ValueError, match=match):
        _scan.hinge_products(x, b, V, split)


@pytest.mark.parametrize(
    ("Q", "E"),
    [
        (np.ones(2), np.ones(2)),
        (np.ones((2, 0)), np.ones(2)),
        (np.ones((2, 1)), np.ones((2, 1, 1))),
    ],
)
def test_knot_scan_checks_its_input(Q, E):
    with pytest.raises(ValueError, match="intercept"):
        _scan.knot_scan([0.0, 1.0], [1.0, 1.0], Q, E, [1])
