"""Tests of pymars/_scan.py against brute force: explicit hinge columns and
least squares.

Spec: LA-2 (the RSS reduction of a candidate), LA-3 (the ratio rho), LA-5 (the
accuracy) and the plan's "Fast path" (eq. 52). The tolerances are tighter
than LA-5, which allows 1e-8 of the RSS and 1e-7·max(rho, tau).
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pymars import _scan

seeds = st.integers(0, 2**32 - 1)


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


@given(seeds, st.integers(8, 40), st.integers(1, 4), st.integers(1, 2), st.booleans())
def test_knot_scan_equals_least_squares(seed, n, r, K, parent):
    x, b, B, Q, Y, E, splits = _problem(seed, n, r, K, parent=parent)
    scan = _scan.knot_scan(x, b, Q, E, splits)
    base = _rss(B, Y)
    for i, s in enumerate(splits):
        h = b * np.maximum(x - x[s - 1], 0.0)
        centered = np.sum((h - h.mean()) ** 2)
        if centered == 0.0:  # a constant hinge column
            assert scan.ratio[i] == 0.0
            continue
        rho = _rss(B, h) / centered
        assert scan.ratio[i] == pytest.approx(rho, rel=1e-9, abs=1e-12)
        if rho > 1e-6:
            gain = base - _rss(np.column_stack((B, h)), Y)
            assert scan.gain[i] == pytest.approx(gain, abs=1e-10 * base)


def test_a_constant_hinge_has_ratio_and_gain_zero():
    """LA-3: a constant h (here 0 at every case) is rejected."""
    x, _, _, Q, _, E, splits = _problem(3, 20, 2, 1)
    b = np.where(np.arange(20) < 10, 1.0, 0.0)  # zero above case 9
    scan = _scan.knot_scan(x, b, Q, E, splits)
    above = splits >= 10
    assert np.all(scan.ratio[above] == 0.0) and np.all(scan.gain[above] == 0.0)
    assert np.all(scan.ratio[~above] > 0.0)


def test_a_hinge_in_the_span_has_a_ratio_near_zero():
    """KNOT-5: with a repeated minimum, h = x - t·1 lies in the span of 1 and x."""
    x = np.array([0.0, 0.0, 1.0, 2.0, 4.0, 7.0])
    B = np.column_stack((np.ones(6), x))
    Q, _ = np.linalg.qr(B)
    Q *= np.sign(Q[0])
    scan = _scan.knot_scan(x, np.ones(6), Q, np.zeros(6), [2, 3])
    assert abs(scan.ratio[0]) < 1e-14 < scan.ratio[1]


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
    np.testing.assert_allclose(rb.resid, Y - A @ np.linalg.lstsq(A, Y)[0], atol=1e-10)
    one = _scan.rebuild(Q, Y[:, 0], new[:, :1])
    assert one.resid.shape == (n, 1)


def test_rebuild_refuses_a_column_in_the_span():
    _, _, _, Q, Y, _, _ = _problem(1, 10, 2, 1)
    assert _scan.rebuild(Q, Y, np.zeros((10, 1))) is None


@pytest.mark.parametrize(
    ("x", "b", "V", "split", "match"),
    [
        ([1.0, 0.0], [1.0, 1.0], np.ones((2, 1)), [1], "sorted"),
        ([0.0, 2.0, 1.0], [1.0] * 3, np.ones((3, 1)), [1], "sorted"),
        ([0.0, np.nan], [1.0, 1.0], np.ones((2, 1)), [1], "finite"),
        ([0.0, 1.0], [1.0], np.ones((2, 1)), [1], "one row"),
        ([0.0, 1.0], [1.0, 1.0], np.ones(2), [1], "one row"),
        ([0.0, 1.0], [1.0, 1.0], np.ones((2, 1)), [2], r"split must be in 1\.\.1$"),
        ([0.0, 1.0], [1.0, 1.0], np.ones((2, 1)), [0], r"split must be in 1\.\.1$"),
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
