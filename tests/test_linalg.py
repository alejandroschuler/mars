"""Tests of pymars/_linalg.py (docs/algorithm.md: LA-1 to LA-7, FWD-11, PRUNE-3,
PRUNE-8, PRUNE-9, W-1, W-3 to W-5, EDGE-6).

Every input is passed read-only, so a function that writes into its input
fails. The explicit least-squares answers come from numpy's SVD solver.
"""

from fractions import Fraction

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from pymars import _linalg as la
from pymars import _terms

seeds = st.integers(0, 2**32 - 1)
# (seed, n, m, k, weighted): n rows, m columns and k responses.
cases = st.tuples(
    seeds, st.integers(8, 30), st.integers(1, 6), st.integers(1, 3), st.booleans()
)


def _frozen(*arrays):
    out = []
    for a in arrays:
        a = np.array(a, dtype=np.float64)
        a.flags.writeable = False
        out.append(a)
    return out


def _basis(rng, n, m):
    """An intercept, then linear and hinge columns, as in a MARS basis."""
    cols = [np.ones(n)]
    for j in range(1, m):
        x = rng.standard_normal(n)
        t = np.quantile(x, rng.uniform(0.2, 0.6))
        cols.append(x if j % 2 else np.maximum(x - t, 0.0))
    return np.column_stack(cols)


def _case(seed, n, m, k, weighted):
    rng = np.random.default_rng(seed)
    A = _basis(rng, n, m)
    Y = A @ rng.standard_normal((m, k)) + rng.standard_normal((n, k))
    w = rng.uniform(0.2, 3.0, n) if weighted else None
    return A, Y, w


def _rss(A, Y, w):
    """The weighted RSS of Y on the columns of A, by numpy's lstsq (LA-1)."""
    w = np.ones(A.shape[0]) if w is None else w
    Y = Y.reshape(A.shape[0], -1)
    if A.shape[1] == 0:
        return float(np.sum(w[:, None] * Y**2))
    sw = np.sqrt(w)[:, None]
    b = np.linalg.lstsq(sw * A, sw * Y, rcond=None)[0]
    return float(np.sum(w[:, None] * (Y - A @ b) ** 2))


def _qbasis(G, w=None):
    sw = 1.0 if w is None else np.sqrt(w)[:, None]
    return np.linalg.qr(sw * G)[0]


def _unit_orthogonal(rng, G):
    """A unit vector orthogonal to the columns of G."""
    Q = np.linalg.qr(G)[0]
    z = rng.standard_normal(G.shape[0])
    for _ in range(2):
        z = z - Q @ (Q.T @ z)
    return z / np.linalg.norm(z)


def _near_dependent(rng, G, eps, w=None):
    """A column whose part orthogonal to G is eps of its norm, all √w-scaled."""
    sw = np.ones(G.shape[0]) if w is None else np.sqrt(w)
    us = sw * (G @ np.array([0.5, 1.0, -2.0]))
    zs = _unit_orthogonal(rng, sw[:, None] * G)
    return (us + eps * np.linalg.norm(us) * zs) / sw


def _exact_rss(A, y, w, subsets):
    """The weighted RSS of y on each subset of the columns of A, exactly.

    The normal equations are solved in rational arithmetic, so the only error
    is the final rounding to float64.
    """
    F = [[Fraction(v) for v in row] for row in A]
    fy, fw = [Fraction(v) for v in y], [Fraction(v) for v in w]
    m = A.shape[1]
    G = [
        [sum(c * r[a] * r[b] for c, r in zip(fw, F, strict=True)) for b in range(m)]
        for a in range(m)
    ]
    g = [sum(c * r[a] * v for c, r, v in zip(fw, F, fy, strict=True)) for a in range(m)]
    yy = sum(c * v * v for c, v in zip(fw, fy, strict=True))
    out = []
    for cols in subsets:
        rows = [[G[a][b] for b in cols] + [g[a]] for a in cols]
        k = len(cols)
        for p in range(k):  # elimination; the pivots of X'WX are positive
            for r in range(p + 1, k):
                f = rows[r][p] / rows[p][p]
                rows[r] = [x - f * v for x, v in zip(rows[r], rows[p], strict=True)]
        beta = [Fraction(0)] * k
        for p in reversed(range(k)):
            tail = sum(rows[p][c] * beta[c] for c in range(p + 1, k))
            beta[p] = (rows[p][k] - tail) / rows[p][p]
        out.append(float(yy - sum(beta[p] * g[cols[p]] for p in range(k))))
    return out


# Gram-Schmidt (LA-1; plan: Fast path)


@given(seed=seeds, n=st.integers(4, 30), r=st.integers(0, 4), k=st.integers(1, 3))
def test_orthogonalize_splits_v_into_span_and_orthogonal_parts(seed, n, r, k):
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.standard_normal((n, min(r, n - 1))))[0]
    V = rng.standard_normal((n, k))
    Qf, Vf = _frozen(Q, V)
    perp, C = la.orthogonalize(Qf, Vf)
    tol = 1e-13 * np.linalg.norm(V)
    assert np.abs(Q.T @ perp).max(initial=0.0) <= tol
    np.testing.assert_allclose(Q @ C + perp, V, rtol=0, atol=tol)
    np.testing.assert_allclose(C, Q.T @ V, rtol=0, atol=tol)


def test_gram_schmidt_applied_twice_keeps_a_nearly_dependent_column_orthogonal():
    rng = np.random.default_rng(3)
    Q = np.linalg.qr(rng.standard_normal((200, 10)))[0]
    v = Q @ rng.uniform(1e3, 2e3, 10) + 1e-9 * _unit_orthogonal(rng, Q)
    g = la.gram_schmidt(*_frozen(Q, v))
    assert np.abs(Q.T @ g.q).max() <= 1e-12
    assert g.norm == pytest.approx(1e-9, rel=1e-4)
    # One pass leaves rounding of size 1e-16·‖v‖ in the span, 1e-3 of v⊥ here.
    once = v - Q @ (Q.T @ v)
    assert np.abs(Q.T @ once).max() / np.linalg.norm(once) > 1e-8


def test_gram_schmidt_without_q_and_for_a_zero_column():
    v = np.array([3.0, 0.0, 4.0])
    g = la.gram_schmidt(np.zeros((3, 0)), v)
    assert g.norm == 5.0
    assert g.coef.shape == (0,)
    np.testing.assert_array_equal(g.q, v / 5.0)
    g0 = la.gram_schmidt(_qbasis(np.ones((3, 1))), np.zeros(3))
    assert g0.q is None
    assert g0.norm == 0.0
    with pytest.raises(ValueError, match="1-D"):
        la.gram_schmidt(np.zeros((3, 0)), np.zeros((3, 1)))


# The collinearity test (LA-3)


@pytest.mark.parametrize(
    ("step", "tau"), [(1, 0.01), (7, 0.01), (np.int64(7), 0.01), (8, 1e-5), (40, 1e-5)]
)
def test_collinearity_tolerance_changes_after_step_seven(step, tau):
    assert la.collinearity_tolerance(step) == tau
    with pytest.raises(ValueError, match="from 1"):
        la.collinearity_tolerance(0)


def test_knot_rejected_is_a_strict_comparison_with_tau():
    assert la.knot_rejected(0.01, 1) is False
    assert la.knot_rejected(np.nextafter(0.01, 0.0), 7) is True
    assert la.knot_rejected(1e-5, 8) is False
    assert la.knot_rejected(0.005, 8) is False
    np.testing.assert_array_equal(
        la.knot_rejected(np.array([0.0, 0.009, 0.01, 0.5]), 3),
        [True, True, False, False],
    )


@pytest.mark.parametrize(
    ("rho", "step", "rejected"),
    [
        (0.0099, 1, True),
        (0.0099, 7, True),
        (0.0101, 1, False),
        (0.0099, 8, False),
        (0.99e-5, 8, True),
        (1.01e-5, 8, False),
    ],
)
def test_knots_near_the_tolerance(rho, step, rejected):
    # h = x + c·z with z orthogonal to G = (1, x) gives 1 - R² = c²/(S + c²),
    # S = ‖x - x̄‖²; like bb11.1, where 0.009964 was rejected and 0.010021 kept.
    rng = np.random.default_rng(11)
    x = rng.standard_normal(50)
    G = np.column_stack([np.ones(50), x])
    S = np.sum((x - x.mean()) ** 2)
    h = x + np.sqrt(rho * S / (1.0 - rho)) * _unit_orthogonal(rng, G)
    ratio = la.collinearity_ratio(*_frozen(_qbasis(G), h))
    assert ratio == pytest.approx(rho, rel=1e-9)
    assert la.knot_rejected(ratio, step) is rejected


@given(seed=seeds, n=st.integers(6, 30), m=st.integers(1, 4), weighted=st.booleans())
def test_collinearity_ratio_is_one_minus_r2_of_a_centered_regression(
    seed, n, m, weighted
):
    G, _, w = _case(seed, n, m, 1, weighted)
    x = np.random.default_rng(seed + 1).standard_normal(n)
    h = G[:, -1] * np.maximum(x - np.quantile(x, 0.3), 0.0)
    assume(not np.all(h == h[0]))
    wv = np.ones(n) if w is None else w
    hbar = np.sum(wv * h) / np.sum(wv)
    expected = _rss(G, h, w) / np.sum(wv * (h - hbar) ** 2)
    ratio = la.collinearity_ratio(*_frozen(_qbasis(G, w), h), w)
    assert ratio == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_collinearity_ratio_of_a_constant_column_is_zero():
    # The computed centered sums of these constant columns are positive by
    # rounding, so only the exact test of the values (Conventions) gives 0.
    Q = _qbasis(np.ones((6, 1)))
    assert la.collinearity_ratio(Q, np.full(6, 0.1)) == 0.0
    assert la.collinearity_ratio(Q, np.zeros(6)) == 0.0
    # A zero-weight row is not a case (W-3).
    h, w = np.append(5.0, np.full(6, 0.1)), np.append(0.0, np.ones(6))
    assert la.collinearity_ratio(_qbasis(np.ones((7, 1)), w), h, w) == 0.0
    assert la.knot_rejected(0.0, 40)
    with pytest.raises(ValueError, match="1-D"):
        la.collinearity_ratio(Q, np.zeros((7, 1)))


@given(seed=seeds, n=st.integers(5, 20))
def test_collinearity_ratio_with_integer_weights_equals_repeated_rows(seed, n):
    rng = np.random.default_rng(seed)
    G = _basis(rng, n, 3)
    h = np.maximum(G[:, 1] - np.median(G[:, 1]), 0.0) + rng.uniform(0, 0.1, n)
    w = rng.integers(1, 4, n).astype(float)
    rep = np.repeat(np.arange(n), w.astype(int))
    got = la.collinearity_ratio(_qbasis(G, w), h, w)
    want = la.collinearity_ratio(_qbasis(G[rep]), h[rep])
    assert got == pytest.approx(want, rel=1e-10)


def test_collinearity_ratio_keeps_the_sums_of_squares_in_range():
    rng = np.random.default_rng(12)
    G = _basis(rng, 30, 3)
    h = np.maximum(G[:, 1] - 0.1, 0.0) * G[:, 2]
    want = la.collinearity_ratio(_qbasis(G), h)
    for scale in (1e200, 1e-200):
        got = la.collinearity_ratio(_qbasis(G), scale * h)
        assert got == pytest.approx(want, rel=1e-12)
    # Weights so small that the centered sum underflows to 0.
    w = np.full(4, 5e-324)
    assert la.collinearity_ratio(np.zeros((4, 0)), np.arange(4.0), w) == 0.0


def test_collinearity_ratio_ignores_rows_with_zero_weight():
    # W-3: a row with zero weight is not a case, whatever its value of h.
    rng = np.random.default_rng(15)
    G = _basis(rng, 30, 3)
    h = np.maximum(G[:, 1] - 0.1, 0.0) * G[:, 2]
    w = rng.uniform(0.5, 2.0, 30)
    want = la.collinearity_ratio(_qbasis(G, w), h, w)
    G0, w0 = np.vstack([G[:1], G]), np.append(0.0, w)
    for value in (1e300, -1e-300):
        got = la.collinearity_ratio(_qbasis(G0, w0), np.append(value, h), w0)
        assert got == pytest.approx(want, rel=1e-12)


# The kind of a search (LA-7)


def test_weighted_variances_use_the_divisor_n():
    X = np.array([[0.0, 1.0], [2.0, 1.0]])
    np.testing.assert_array_equal(la.weighted_variances(*_frozen(X)), [1.0, 0.0])
    x = np.array([[0.0], [4.0]])
    np.testing.assert_array_equal(la.weighted_variances(x, np.array([1.0, 3.0])), [3.0])
    with pytest.raises(ValueError, match="positive weight"):
        la.weighted_variances(x, np.zeros(2))


@pytest.mark.parametrize(("n", "value"), [(3, 0.1), (6, 0.1), (7, 0.3), (3, 0.7)])
def test_weighted_variance_of_a_constant_column_is_exactly_zero(n, value):
    # LA-7 needs sigma² = 0 for a constant covariate. The two-pass formula
    # leaves a rounding residue for these columns in some summation orders,
    # and for three 0.1s in every order (1.9e-34), so only the exact test of
    # the values (Conventions) gives 0.
    x3 = np.full(3, 0.1)
    assert np.mean((x3 - np.mean(x3)) ** 2) > 0.0
    X = np.column_stack([np.full(n, value), np.arange(float(n))])
    assert la.weighted_variances(X)[0] == 0.0
    Xw, w = np.vstack([[5.0, 0.0], X]), np.append(0.0, np.ones(n))
    assert la.weighted_variances(Xw, w)[0] == 0.0


@given(seed=seeds, n=st.integers(2, 20))
def test_weighted_variances_with_integer_weights_equal_repeated_rows(seed, n):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    w = rng.integers(0, 4, n).astype(float)
    w[0] = 1.0
    rep = np.repeat(np.arange(n), w.astype(int))
    np.testing.assert_allclose(
        la.weighted_variances(X, w), la.weighted_variances(X[rep]), rtol=1e-12
    )


def test_pair_search_threshold_is_one_percent_of_the_variance_product():
    assert la.pair_search(0.01, [2.0, 0.5]) is True
    assert la.pair_search(np.nextafter(0.01, 0.0), [2.0, 0.5]) is False
    assert la.pair_search(0.0101, np.array([1.0])) is True
    assert la.pair_search(0.0099, [1.0]) is False
    assert la.pair_search(1e300, [4.0, 0.0]) is False  # a constant covariate
    with pytest.raises(ValueError, match="at least"):
        la.pair_search(1.0, [])


def test_pair_search_does_not_depend_on_the_units_of_x():
    # A parent hinge b and a covariate x: A_w of b·x on (1, b) and σ² of x
    # both scale by c², so the decision does not change with c, while
    # earth's A ≥ 0.01 does (bb14.3).
    rng = np.random.default_rng(2)
    x1, x2 = rng.standard_normal((2, 60))
    b = np.maximum(x1 - 0.3, 0.0)
    Q = _qbasis(np.column_stack([np.ones(60), b]))
    decisions, earth = set(), set()
    for c in (1e-3, 1.0, 1e3):
        a_w = la.gram_schmidt(Q, b * c * x2).norm ** 2
        var = la.weighted_variances(np.column_stack([x1, c * x2]))
        decisions.add(la.pair_search(a_w, var))
        earth.add(a_w >= 0.01)
    assert decisions == {True}
    assert earth == {False, True}


# Least squares with dependent columns (LA-4, FWD-11, PRUNE-8)


@given(case=cases)
def test_lm_fit_equals_least_squares_on_independent_columns(case):
    A, Y, w = _case(*case)
    n = A.shape[0]
    args = _frozen(A, Y) + ([] if w is None else _frozen(w))
    fit = la.lm_fit(*args)
    sw = np.ones((n, 1)) if w is None else np.sqrt(w)[:, None]
    coef = np.linalg.lstsq(sw * A, sw * Y, rcond=None)[0]
    assert fit.kept.all()
    np.testing.assert_allclose(fit.coef, coef, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(fit.residuals, Y - A @ coef, rtol=1e-9, atol=1e-10)
    assert fit.rss == pytest.approx(_rss(A, Y, w), rel=1e-10)
    np.testing.assert_array_equal(la.independent_columns(*args[:1], w), fit.kept)


def test_lm_fit_keeps_the_first_of_two_dependent_columns():
    rng = np.random.default_rng(4)
    x, y = rng.standard_normal((2, 20))
    for A, kept in [
        (np.column_stack([np.ones(20), x, 2 * x]), [True, True, False]),
        (np.column_stack([np.ones(20), 2 * x, x]), [True, True, False]),
        (np.column_stack([x, np.ones(20), x + 1.0]), [True, True, False]),
        (np.column_stack([np.ones(20), np.zeros(20), x]), [True, False, True]),
    ]:
        fit = la.lm_fit(A, y)
        np.testing.assert_array_equal(fit.kept, kept)
        assert np.all(fit.coef[~fit.kept] == 0.0)
        coef = np.linalg.lstsq(A[:, fit.kept], y, rcond=None)[0]
        np.testing.assert_allclose(fit.coef[fit.kept], coef, rtol=1e-10)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize(("eps", "kept"), [(1.01e-7, True), (0.99e-7, False)])
def test_la4_threshold_is_1e_7_within_one_percent(eps, kept, weighted):
    # As bb09.8, which puts R's threshold within 1e-6 relative of 1e-7.
    rng = np.random.default_rng(6)
    G = _basis(rng, 40, 3)
    w = rng.uniform(0.05, 20.0, 40) if weighted else None
    A = np.column_stack([G, _near_dependent(rng, G, eps, w)])
    assert la.independent_columns(A, w)[3] is np.bool_(kept)
    assert la.lm_fit(A, rng.standard_normal(40), w).kept[3] is np.bool_(kept)


def test_la4_tests_the_sqrt_w_scaled_columns():
    # LA-4 and FWD-11 test √w-scaled columns, as lm.wfit does (bb09.14).
    rng = np.random.default_rng(0)
    G = _basis(rng, 40, 3)
    w = rng.uniform(0.2, 5.0, 40)
    sw = np.sqrt(w)
    us = sw * (G @ np.array([0.5, 1.0, -2.0]))
    zs = _unit_orthogonal(rng, sw[:, None] * G)

    def ratio(a, s):  # the LA-4 ratio of the column a with rows scaled by s
        Q = np.linalg.qr(s[:, None] * G)[0]
        return np.linalg.norm(s * a - Q @ (Q.T @ (s * a))) / np.linalg.norm(s * a)

    for eps, kept in [(1.1e-7, True), (0.9e-7, False)]:
        a = (us + eps * np.linalg.norm(us) * zs) / sw
        A = np.column_stack([G, a])
        assert la.independent_columns(A, w)[3] is np.bool_(kept)
        assert la.lm_fit(A, rng.standard_normal(40), w).kept[3] is np.bool_(kept)
        # With w in place of √w, or without weights, one of the two flips.
        wrong = ratio(a, w) < 1e-7 if kept else ratio(a, np.ones(40)) >= 1e-7
        assert wrong
    # A column equal to x on the rows with positive weight is dependent (W-3).
    x = G[:, 1]
    A = np.column_stack([np.ones(40), x, x + 3.0 * np.eye(40)[0]])
    w0 = np.append(0.0, np.ones(39))
    assert la.independent_columns(A).all()
    np.testing.assert_array_equal(la.independent_columns(A, w0), [True, True, False])
    np.testing.assert_array_equal(la.lm_fit(A, x, w0).kept, [True, True, False])


def test_la4_compares_with_the_kept_earlier_columns_only():
    # The dropped near-duplicate adds nothing to the span, so its orthogonal
    # direction, which comes next, is kept, as in R's lm.fit.
    rng = np.random.default_rng(14)
    G = _basis(rng, 30, 2)
    x = G[:, 1]
    z = np.linalg.norm(x) * _unit_orthogonal(rng, G)
    A = np.column_stack([G, x + 5e-8 * z, z])
    np.testing.assert_array_equal(la.independent_columns(A), [True, True, False, True])


def test_lm_fit_tests_the_norm_without_centering():
    # 1000 + 1e-5·z: its part orthogonal to the intercept is 1e-8 of its norm,
    # but most of its centered norm (bb09.9 refutes the centered test).
    rng = np.random.default_rng(7)
    G = np.column_stack([np.ones(40), rng.standard_normal(40)])
    a = 1000.0 + 1e-5 * _unit_orthogonal(rng, G) * np.sqrt(40)
    assert not la.independent_columns(np.column_stack([G, a]))[2]


@given(seed=seeds, n=st.integers(6, 20), weighted=st.booleans())
def test_lm_fit_with_integer_weights_equals_repeated_rows(seed, n, weighted):
    A, Y, _ = _case(seed, n, 3, 2, False)
    rng = np.random.default_rng(seed)
    w = rng.integers(1, 4, n).astype(float) if weighted else np.ones(n)
    rep = np.repeat(np.arange(n), w.astype(int))
    fit, fit_rep = la.lm_fit(A, Y, w), la.lm_fit(A[rep], Y[rep])
    np.testing.assert_allclose(fit.coef, fit_rep.coef, rtol=1e-9, atol=1e-12)
    assert fit.rss == pytest.approx(fit_rep.rss, rel=1e-10)


def test_lm_fit_unit_weights_equal_no_weights_and_zero_weights_drop_rows():
    A, Y, _ = _case(8, 12, 3, 1, False)
    y = Y[:, 0]
    plain, unit = la.lm_fit(A, y), la.lm_fit(A, y, np.ones(12))
    np.testing.assert_array_equal(plain.coef, unit.coef)
    assert plain.rss == unit.rss
    A2 = np.vstack([A, [1.0, 1e160, -1e160]])  # W-3: not read into the RSS
    fit = la.lm_fit(A2, np.append(y, 1e9), np.append(np.ones(12), 0.0))
    np.testing.assert_allclose(fit.coef, plain.coef, rtol=1e-12)
    assert fit.rss == pytest.approx(plain.rss, rel=1e-12)
    assert fit.residuals[-1] == pytest.approx(1e9 - A2[-1] @ fit.coef)


def test_lm_fit_scales_exactly_with_a_power_of_two_in_y():
    # EDGE-6: the core multiplies Y by 2^j, which must change no bit.
    A, Y, w = _case(9, 15, 4, 2, True)
    fit, fit8 = la.lm_fit(A, Y, w), la.lm_fit(A, Y * 2.0**-40, w)
    np.testing.assert_array_equal(fit8.coef, fit.coef * 2.0**-40)
    np.testing.assert_array_equal(fit8.residuals, fit.residuals * 2.0**-40)
    assert fit8.rss == fit.rss * 2.0**-80


def test_lm_fit_shapes_and_errors():
    A, Y, _ = _case(1, 10, 3, 1, False)
    fit = la.lm_fit(A, Y[:, 0])
    assert fit.coef.shape == (3,)
    assert fit.residuals.shape == (10,)
    assert la.lm_fit(A, Y).coef.shape == (3, 1)
    wide = la.lm_fit(A[:2], Y[:2, 0])  # rank 2: the last column is left out
    np.testing.assert_array_equal(wide.kept, [True, True, False])
    assert np.abs(wide.residuals).max() <= 1e-12 * np.abs(Y[:2]).max()
    empty = la.lm_fit(np.zeros((10, 0)), Y[:, 0])
    np.testing.assert_array_equal(empty.residuals, Y[:, 0])
    assert empty.rss == pytest.approx(np.sum(Y**2))
    with pytest.raises(ValueError, match="2-D"):
        la.lm_fit(A[:, 0], Y)
    with pytest.raises(ValueError, match=r"\(10,\)"):
        la.lm_fit(A, Y[:5])
    with pytest.raises(ValueError, match="shape"):
        la.lm_fit(A, Y, np.ones(9))
    with pytest.raises(ValueError, match="nonnegative"):
        la.lm_fit(A, Y, np.full(10, -1.0))


def test_lm_fit_matches_r_lm_fit_on_the_fixture(load_fixture):
    # T05's component fixture: R's lm.fit and lm.wfit on fixed columns, with
    # the plan's tolerances for coefficients (kappa ≤ 1e5) and fitted values.
    fx = load_fixture("components/lm_fit_coefficients")
    y = np.array(fx["y"])
    for case in fx["cases"]:
        A = np.array(case["x"])
        w = None if case["weights"] is None else np.array(case["weights"])
        want = np.array([np.nan if c is None else c for c in case["coefficients"]])
        fit = la.lm_fit(A, y, w)
        np.testing.assert_array_equal(fit.kept, ~np.isnan(want), err_msg=case["label"])
        assert fit.kept.sum() == case["rank"]
        sw = np.ones(len(y)) if w is None else np.sqrt(w)
        kappa = np.linalg.cond(sw[:, None] * A[:, fit.kept])
        if kappa <= 1e5:
            b = want[fit.kept]
            assert np.linalg.norm(fit.coef[fit.kept] - b) <= 1e-6 * np.linalg.norm(b)
        tol = 1e-8 * np.std(y, ddof=1) * max(1.0, kappa / 1e6)
        assert np.abs(fit.residuals - np.array(case["residuals"])).max() <= tol


# The R factor of the pruning pass and its downdates (PRUNE-3, PRUNE-9)


@given(case=cases)
def test_prefix_rss_and_drop_costs_equal_explicit_refits(case):
    A, Y, w = _case(*case)
    m = A.shape[1]
    f = la.r_factor(*_frozen(A, Y), w)
    R, Z = _frozen(f.R, f.Z)
    assert f.rss == pytest.approx(_rss(A, Y, w), rel=1e-10)
    prefix = la.prefix_rss(Z, f.rss)
    want = [_rss(A[:, :j], Y, w) for j in range(1, m + 1)]
    np.testing.assert_allclose(prefix, want, rtol=1e-10)
    for pos in range(1, m + 1):
        costs = la.drop_costs(R, Z, pos)
        rest = [[c for c in range(pos) if c != i] for i in range(pos)]
        want = [_rss(A[:, cols], Y, w) - prefix[pos - 1] for cols in rest]
        np.testing.assert_allclose(costs, want, rtol=1e-8, atol=1e-10 * prefix[0])
        assert np.all(costs >= 0.0)


@given(
    seed=seeds,
    m=st.integers(2, 7),
    weighted=st.booleans(),
    moves=st.lists(st.tuples(st.integers(0, 6), st.integers(0, 6)), max_size=5),
)
def test_move_column_gives_the_factor_of_the_new_order(seed, m, weighted, moves):
    A, Y, w = _case(seed, 25, m, 2, weighted)
    f = la.r_factor(A, Y, w)
    R, Z, order = f.R, f.Z, list(range(m))
    sw = np.ones((25, 1)) if w is None else np.sqrt(w)[:, None]
    for i, j in moves:
        i, j = i % m, j % m
        R, Z = la.move_column(*_frozen(R, Z), i, j)
        order.insert(j, order.pop(i))
        assert not np.tril(R, -1).any()
        As = sw * A[:, order]
        np.testing.assert_allclose(R.T @ R, As.T @ As, rtol=0, atol=1e-11 * m)
        np.testing.assert_allclose(R.T @ Z, As.T @ (sw * Y), rtol=0, atol=1e-10 * m)
    want = [_rss(A[:, order[:j]], Y, w) for j in range(1, m + 1)]
    np.testing.assert_allclose(la.prefix_rss(Z, f.rss), want, rtol=1e-10)


def test_move_column_to_its_own_place_and_a_one_response_z():
    A, Y, _ = _case(2, 10, 4, 1, False)
    f = la.r_factor(A, Y[:, 0])
    R, z = la.move_column(f.R, f.Z[:, 0], 2, 2)
    np.testing.assert_array_equal(R, f.R)
    np.testing.assert_array_equal(z, f.Z[:, 0])
    R, z = la.move_column(f.R, f.Z[:, 0], 1, 3)
    assert z.shape == (4,)
    want = _rss(A[:, [0, 2, 3]], Y, None)
    np.testing.assert_allclose(la.prefix_rss(z, f.rss)[2], want)
    np.testing.assert_array_equal(
        la.drop_costs(R, z, 4), la.drop_costs(R, z[:, None], 4)
    )


def test_pruning_factor_scales_exactly_with_a_power_of_two_in_y():
    # EDGE-6: Y enters only through products and sums.
    A, Y, w = _case(10, 20, 5, 2, True)
    f, f8 = la.r_factor(A, Y, w), la.r_factor(A, Y * 2.0**-40, w)
    np.testing.assert_array_equal(f8.R, f.R)
    np.testing.assert_array_equal(f8.Z, f.Z * 2.0**-40)
    assert f8.rss == f.rss * 2.0**-80
    np.testing.assert_array_equal(
        la.drop_costs(f8.R, f8.Z, 4), la.drop_costs(f.R, f.Z, 4) * 2.0**-80
    )
    Z = la.move_column(f.R, f.Z, 1, 4)[1]
    Z8 = la.move_column(f8.R, f8.Z, 1, 4)[1]
    np.testing.assert_array_equal(Z8, Z * 2.0**-40)
    np.testing.assert_array_equal(
        la.prefix_rss(Z8, f8.rss), la.prefix_rss(Z, f.rss) * 2.0**-80
    )


def test_drop_costs_do_not_depend_on_the_scale_of_a_column():
    A, Y, w = _case(16, 30, 5, 2, True)
    f = la.r_factor(A, Y, w)
    want = la.drop_costs(f.R, f.Z, 5)
    for c in (1e160, 1e-160):
        f2 = la.r_factor(A * np.array([1.0, 1.0, c, 1.0, 1.0]), Y, w)
        np.testing.assert_allclose(la.drop_costs(f2.R, f2.Z, 5), want, rtol=1e-12)


def test_pruning_factor_near_the_la4_limit_matches_exact_refits():
    # kappa above 1e6: a hinge and its copy with noise of relative size 1e-6,
    # against refits in exact rational arithmetic, within a tenth of the 1e-8
    # that LA-5 allows (1e-10 was seen with OpenBLAS).
    rng = np.random.default_rng(17)
    x1, x2 = rng.standard_normal((2, 24))
    h = np.maximum(x1 - 0.2, 0.0)
    near = h + 1e-6 * rng.standard_normal(24)
    A = np.column_stack([np.ones(24), x1, h, near, x2, np.maximum(x2, 0.0)])
    y = A @ rng.standard_normal(6) + rng.standard_normal(24)
    w = rng.uniform(0.5, 2.0, 24)
    assert np.linalg.cond(np.sqrt(w)[:, None] * A) > 1e6
    f = la.r_factor(A, y, w)
    prefix = la.prefix_rss(f.Z, f.rss)
    want = _exact_rss(A, y, w, [list(range(m)) for m in range(1, 7)])
    np.testing.assert_allclose(prefix, want, rtol=1e-9)
    for pos in range(2, 7):
        got = prefix[pos - 1] + la.drop_costs(f.R, f.Z, pos)
        rest = [[c for c in range(pos) if c != i] for i in range(pos)]
        np.testing.assert_allclose(got, _exact_rss(A, y, w, rest), rtol=1e-9)


def test_r_factor_edge_cases():
    A, Y, _ = _case(3, 5, 5, 2, False)
    assert la.r_factor(A, Y).rss <= 1e-24 * np.sum(Y**2)
    with pytest.raises(ValueError, match="only 4 rows"):
        la.r_factor(A[:4], Y[:4])
    f = la.r_factor(A, Y)
    with pytest.raises(ValueError, match="pos"):
        la.drop_costs(f.R, f.Z, 0)
    with pytest.raises(ValueError, match="i and j"):
        la.move_column(f.R, f.Z, 0, 5)


def _term_rows(spec, p):
    """dirs and cuts of terms given as lists of (covariate, code, cut)."""
    dirs, cuts = np.zeros((len(spec), p), dtype=np.int8), np.zeros((len(spec), p))
    for k, term in enumerate(spec):
        for j, code, cut in term:
            dirs[k, j], cuts[k, j] = code, cut
    return dirs, cuts


@pytest.mark.parametrize(
    ("x", "shifted", "m"),
    [
        ([1.0, 1.9], True, 1.0),  # |m| = 1 > range 0.9
        ([1.0, 2.0], False, 0.0),  # |m| = range: not larger
        ([-1.9, -1.0], True, -1.9),  # the smallest value, here the most negative
        ([5.0, 5.0], True, 5.0),  # a constant covariate: u = 0
        ([0.0, 0.0], False, 0.0),
        ([0.0, 1e10], False, 0.0),
        ([1e10, 1e10 + 8], True, 1e10),
    ],
)
def test_large_covariates_follow_la4(x, shifted, m):
    """LA-4: m_j is the smallest value when it is larger in absolute value than
    the range, and 0 otherwise."""
    big, shift = la.large_covariates(np.array(x)[:, None])
    assert (bool(big[0]), float(shift[0])) == (shifted, m)


_X75 = np.column_stack(
    (
        1e10 + np.array([2, 0, 0, 2, 1, 2, 1, 2, 1, 0.0]),
        np.zeros(10),
        np.array([1, 0, 0, 2, 2, 1, 0, 0, 1, 2.0]),
    )
)


@pytest.mark.parametrize(
    "order",
    [
        ["1", "x0", "x0x2", "x2"],  # x2 counts: ratio 0.46, not 5.6e-11 (LA-4)
        ["1", "x0", "x2", "x0x2"],
        ["1", "x0x2", "x2"],
        ["1", "x2", "x0x2"],
    ],
)
def test_the_order_of_the_terms_does_not_change_la4(order):
    """LA-4's example on #75's data: x0 = 1e10 + (…) is kept after the intercept
    (v1 and earth drop it), and so is every product and every term of the
    others, in any order, since the test is on the reduced echelon form."""
    lin = {"x0": [(0, 2, 0.0)], "x2": [(2, 2, 0.0)], "x0x2": [(0, 2, 0.0), (2, 2, 0.0)]}
    dirs, cuts = _term_rows([[]] + [lin[t] for t in order[1:]], 3)
    assert la.independent_terms(_X75, dirs, cuts).tolist() == [True] * len(order)


@pytest.mark.parametrize("shift", [0.0, 0.5, 1e4, 1e10])
@pytest.mark.parametrize(("a", "b"), [(1.0, 0.0), (3.0, 1.0)])
def test_a_product_that_the_terms_span_at_the_data_is_dependent(shift, a, b):
    """LA-4: with x3 = a·x2 + b, the term x0·x3 is a combination of 1, x0, x3 and
    x0·x2 at the ten cases, not as a formula, at every shift of x0; the shifted
    basis finds it, also at 1e10 with x3 = 3·x2 + 1, where the part of
    x0·x2 that carries the mean has to cancel exactly."""
    x0 = shift + np.array([2, 0, 0, 2, 1, 2, 1, 2, 1, 0.0])
    X = np.column_stack((x0, _X75[:, 1], _X75[:, 2], a * _X75[:, 2] + b))
    lin = [(0, 2, 0.0)], [(3, 2, 0.0)], [(0, 2, 0.0), (2, 2, 0.0)]
    dirs, cuts = _term_rows([[], lin[0], lin[1], lin[2], [*lin[0], *lin[1]]], 4)
    assert la.independent_terms(X, dirs, cuts).tolist() == [True] * 4 + [False]


@pytest.mark.parametrize("seed", range(8))
def test_independent_terms_without_a_shift_is_independent_columns(seed):
    """Without a linear factor of a shifted covariate the test is the plain one
    of LA-4 on the columns of the terms, bit for bit (hinges, products and
    linear factors of covariates that are not shifted)."""
    rng = np.random.default_rng(seed)
    X = np.floor(rng.uniform(size=(30, 3)) * 4) / 4
    cuts = np.zeros((7, 3))
    dirs = rng.integers(-1, 3, size=(7, 3)).astype(np.int8)
    dirs[0] = 0
    cuts[dirs == 1], cuts[dirs == -1] = 0.25, 0.5
    cuts[dirs == 2] = 0.0
    w = rng.uniform(0.5, 2.0, 30) if seed % 2 else None
    B = _terms.basis_matrix(X, dirs, cuts)
    got = la.independent_terms(X, dirs, cuts, w)
    assert got.tolist() == la.independent_columns(B, w).tolist()


@pytest.mark.parametrize("seed", range(12))
def test_the_float_test_of_la4_decides_as_the_exact_one(seed, monkeypatch):
    """LA-4 at a large mean on few levels, with copies of a covariate at other
    shifts and with a copy off by one unit in the last place: the decision from
    the float64 distance, trusted only inside its error bound, is that of exact
    rational arithmetic (the distance passed as NaN forces the exact test)."""
    rng = np.random.default_rng(seed)
    a = np.floor(rng.uniform(size=(24, 3)) * 3) / 4
    shifts = rng.choice([1e3, 2.0**26, 1e9, -1e6], 4)
    X = np.column_stack((a[:, 0], a[:, 0], a[:, 1], a[:, 2])) + shifts
    X[3, 1] = np.nextafter(X[3, 1], np.inf) if seed % 2 else X[3, 1]
    L = [[(j, 2, 0.0)] for j in range(4)]
    spec = [[], L[0], L[1], L[2], L[0] + L[2], L[1] + L[3], L[0] + L[1], L[3]]
    spec += [L[0] + L[2] + L[3], L[1] + L[2] + L[3], L[0] + L[3]]
    dirs, cuts = _term_rows(spec, 4)
    got = la.independent_terms(X, dirs, cuts)
    real = la.Conditioner.dependent

    def exact(self, row, cut, dist, size, extra=(), kappa=1.0):
        return real(self, row, cut, float("nan"), size, extra, kappa)

    monkeypatch.setattr(la.Conditioner, "dependent", exact)
    assert la.independent_terms(X, dirs, cuts).tolist() == got.tolist()
