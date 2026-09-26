"""Sanity tests of the reference implementation (``mars_ref``), without earth.

Each test checks a case whose answer is known by hand or by an independent
computation (numpy's least squares, repeated rows, a brute-force loop).
Bracketed IDs cite ``docs/algorithm.md``.
"""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from reference import mars_ref as ref


def lstsq_rss(B, Y, w):
    """The weighted RSS from numpy's SVD least squares, for comparison."""
    sw = np.sqrt(w)
    Y2 = Y.reshape(len(Y), -1)
    coef = np.linalg.lstsq(B * sw[:, None], Y2 * sw[:, None], rcond=None)[0]
    return float(np.sum(w[:, None] * (Y2 - B @ coef) ** 2))


# ---------------------------------------------------------------------------
# Terms [TERM-1 to TERM-4]


class TestTerms:
    def test_basis_matrix_by_hand(self):
        X = np.array([[0.0, 2.0], [1.0, -1.0], [3.0, 0.5]])
        dirs = np.array([[0, 0], [1, 0], [-1, 2], [1, -1]], dtype=np.int8)
        cuts = np.array([[0, 0], [0.5, 0], [2.0, 0], [0.5, 1.0]])
        expected = np.array(
            [
                [1.0, 0.0, 2.0 * 2.0, 0.0],
                [1.0, 0.5, 1.0 * -1.0, 0.5 * 2.0],
                [1.0, 2.5, 0.0, 2.5 * 0.5],
            ]
        )
        np.testing.assert_array_equal(ref.basis_matrix(X, dirs, cuts), expected)

    def test_basis_matrix_is_not_clipped_outside_the_data(self):
        dirs = np.array([[0], [1], [-1], [2]], dtype=np.int8)
        cuts = np.array([[0.0], [1.0], [1.0], [0.0]])
        B = ref.basis_matrix(np.array([[-100.0], [100.0]]), dirs, cuts)
        np.testing.assert_array_equal(B, [[1, 0, 101, -100], [1, 99, 0, 100]])

    def test_term_degree_counts_a_linear_factor_as_one(self):
        assert ref.term_degree([0, 0, 0]) == 0
        assert ref.term_degree([1, 0, 2]) == 2
        assert ref.term_degree([-1, 2, 1]) == 3

    def test_inputs_are_not_changed(self):
        X = np.array([[0.0], [2.0]])
        dirs = np.array([[0], [1]], dtype=np.int8)
        cuts = np.array([[0.0], [1.0]])
        copies = X.copy(), dirs.copy(), cuts.copy()
        ref.basis_matrix(X, dirs, cuts)
        for original, copy in zip((X, dirs, cuts), copies, strict=True):
            np.testing.assert_array_equal(original, copy)


# ---------------------------------------------------------------------------
# GCV [GCV-1 to GCV-7] and the term limit [LIMIT-1]


class TestGcv:
    @pytest.mark.parametrize(
        ("M", "d", "C"),
        [(1, 0.0, 1.0), (1, 3.0, 1.0), (5, 2.0, 9.0), (4, 3.0, 8.5), (7, -1, 0.0)],
    )
    def test_effective_params(self, M, d, C):
        assert ref.effective_params(M, d) == C

    def test_gcv_formula(self):
        # C(3) = 3 + 2 * 2 / 2 = 5; GCV = 2 / (20 (1 - 5/20)^2) = 2 / 11.25
        assert ref.gcv(2.0, 3, 2.0, 20.0, 2e-7) == pytest.approx(2.0 / 11.25, rel=1e-15)

    def test_gcv_is_inf_once_c_reaches_n(self):
        # C(5) at d = 2 is 9: finite for N = 9.5, +inf for N = 9 and just below
        assert math.isfinite(ref.gcv(1.0, 5, 2.0, 9.5, ref.weight_tol(9.5)))
        assert ref.gcv(1.0, 5, 2.0, 9.0, ref.weight_tol(9.0)) == math.inf
        # within tau_N above C is also +inf [GCV-2]
        N = 9.0 + 0.5e-7
        assert ref.gcv(1.0, 5, 2.0, N, ref.weight_tol(N)) == math.inf

    def test_penalty_minus_one_gives_rss_over_n(self):
        assert ref.gcv(3.0, 40, -1, 10.0, 1e-7) == pytest.approx(0.3, rel=1e-15)

    def test_grsq_is_minus_inf_when_the_gcv_is_inf(self):
        assert ref.grsq(1.0, 5, 10.0, 2.0, 9.0, 9e-8) == -math.inf

    def test_default_penalty(self):
        assert ref.default_penalty(1) == 2.0
        assert ref.default_penalty(2) == 3.0
        assert ref.default_penalty(3) == 3.0

    @pytest.mark.parametrize(("p", "M"), [(1, 21), (10, 21), (11, 23), (100, 201)])
    def test_default_max_terms(self, p, M):
        assert ref.default_max_terms(p) == M


# ---------------------------------------------------------------------------
# Spans [SPAN-1 to SPAN-6]


class TestSpans:
    # -ln(0.95) = 0.0512933; L = trunc(-log2(0.0512933 / (p N_b)) / 2.5)
    @pytest.mark.parametrize(
        ("p", "Nb", "L"),
        [(1, 200, 4), (1, 100, 4), (1, 20, 3), (5, 1000, 6), (1, 0.01, 1)],
    )
    def test_auto_minspan(self, p, Nb, L):
        assert ref.auto_minspan(p, Nb) == L

    @pytest.mark.parametrize(
        ("p", "E"), [(1, 7), (2, 8), (3, 8), (4, 9), (6, 9), (8, 10), (20, 11)]
    )
    def test_auto_endspan(self, p, E):
        assert ref.auto_endspan(p) == E

    @pytest.mark.parametrize(
        ("E", "degree", "a", "E1"),
        [
            (7, 0, 2.0, 7),
            (7, 1, 2.0, 21),
            (7, 2, 1.0, 14),
            (7, 1, 0.0, 7),
            (7, 1, 0.5, 11),
        ],
    )
    def test_adjusted_endspan(self, E, degree, a, E1):
        assert ref.adjusted_endspan(E, degree, a) == E1

    @pytest.mark.parametrize(
        ("E1", "N", "Estar"),
        [(21, 20.0, 9), (21, 3.0, 1), (5, 100.0, 5), (21, 10.5, 4)],
    )
    def test_scan_endspan_cap(self, E1, N, Estar):
        assert ref.scan_endspan(E1, N, ref.weight_tol(N)) == Estar

    def test_user_spans_replace_the_automatic_ones_and_are_adjusted(self):
        tau = ref.weight_tol(100.0)
        assert ref.search_spans(3, 0, 100.0, 100.0, tau, minspan=2, endspan=3) == (2, 3)
        # a parent of degree 1: E1 = 3 + floor(2 * 3 + 0.5) = 9
        assert ref.search_spans(3, 1, 40.0, 100.0, tau, minspan=50, endspan=3) == (
            50,
            9,
        )


# ---------------------------------------------------------------------------
# Candidate knots [KNOT-1 to KNOT-6]


def expanded(x, active, w):
    """Row i repeated w_i times."""
    return np.repeat(x, w), np.repeat(active, w)


@st.composite
def scans(draw, weights=False):
    n = draw(st.integers(1, 30))
    x = np.array(draw(st.lists(st.integers(-4, 4), min_size=n, max_size=n)), float)
    active = np.array(draw(st.lists(st.booleans(), min_size=n, max_size=n)))
    L = draw(st.integers(1, 5))
    E = draw(st.integers(1, 5))
    w = np.array(draw(st.lists(st.integers(1, 3), min_size=n, max_size=n)))
    return (x, active, L, E, w) if weights else (x, active, L, E)


def knot_scan_literal(x, active, w, L, E, N, tau):
    """KNOT-6 one visit at a time, with the counter of KNOT-3, for comparison
    with the vectorized ref.knot_scan."""
    if not active.any():
        return []
    order = ref.case_order(x, active)
    xs, act, W = x[order], active[order], np.cumsum(w[order])

    def holder(u):  # the smallest q with u <= W_q + tau_N, else case n
        return next((q for q in range(len(x)) if u <= W[q] + tau), len(x) - 1)

    v = x[active].max()
    D = N - 2 * E - 1
    counter = E + math.ceil((D - L * math.floor(D / L)) / 2)
    knots, k = [], 0
    while N - k >= E + 2 - tau:
        u = N - k
        t = xs[holder(u - 1)]
        if t < v and act[holder(u)]:
            counter -= 1
            if counter == 0:
                knots.append(float(t))
                counter = L
        k += 1
    return knots


class TestKnots:
    def test_intercept_distinct_values_every_third(self):
        x = np.arange(20.0)[::-1]
        knots = ref.knot_scan_unit(x, np.ones(20, bool), 3, 2)
        assert knots == [17.0, 14.0, 11.0, 8.0, 5.0, 2.0]

    def test_slack_is_split_with_the_larger_half_at_the_top(self):
        # n - 2E - 1 = 15 = 3 * 4 + 3: 2 values of slack at the top, 1 below
        knots = ref.knot_scan_unit(np.arange(20.0), np.ones(20, bool), 4, 2)
        assert knots == [15.0, 11.0, 7.0, 3.0]

    def test_a_knot_can_be_the_value_of_an_inactive_case(self):
        x = np.arange(6.0)
        active = np.array([False, False, True, True, True, True])
        assert ref.knot_scan_unit(x, active, 1, 1) == [4.0, 3.0, 2.0, 1.0]

    def test_no_active_case_no_knot(self):
        x = np.arange(10.0)
        assert ref.knot_scan_unit(x, np.zeros(10, bool), 1, 1) == []
        assert ref.knot_scan(x, np.zeros(10, bool), np.ones(10), 1, 1, 10.0, 1e-7) == []

    def test_inactive_cases_come_first_among_equal_values(self):
        order = ref.case_order(np.array([1.0, 1.0, 0.0]), np.array([True, False, True]))
        np.testing.assert_array_equal(order, [2, 1, 0])

    @given(scans())
    def test_cumulative_scan_with_unit_weights_is_the_step_by_step_scan(self, case):
        x, active, L, E = case
        n = len(x)
        weighted = ref.knot_scan(x, active, np.ones(n), L, E, float(n), 1e-8 * n)
        assert weighted == ref.knot_scan_unit(x, active, L, E)

    @given(scans(weights=True))
    def test_integer_weights_give_the_knots_of_repeated_rows(self, case):
        x, active, L, E, w = case
        N = float(w.sum())
        weighted = ref.knot_scan(x, active, w.astype(float), L, E, N, ref.weight_tol(N))
        assert weighted == ref.knot_scan_unit(*expanded(x, active, w), L, E)

    @given(
        scans(),
        st.lists(st.floats(0.05, 3.0), min_size=30, max_size=30),
    )
    def test_the_cumulative_scan_follows_knot_6_one_visit_at_a_time(
        self, case, weights
    ):
        x, active, L, E = case
        w = np.array(weights[: len(x)])
        N0 = ref.weight_sum(w)
        tau = ref.weight_tol(N0)
        N = ref.snap(N0, tau)
        assert ref.knot_scan(x, active, w, L, E, N, tau) == knot_scan_literal(
            x, active, w, L, E, N, tau
        )

    @given(scans(), st.randoms(use_true_random=False))
    def test_the_knots_do_not_depend_on_the_row_order(self, case, rnd):
        x, active, L, E = case
        perm = np.array(rnd.sample(range(len(x)), len(x)))
        assert ref.knot_scan_unit(x[perm], active[perm], L, E) == ref.knot_scan_unit(
            x, active, L, E
        )

    @given(scans())
    def test_properties_of_knot_4(self, case):
        x, active, L, E = case
        knots = ref.knot_scan_unit(x, active, L, E)
        if knots:
            v = x[active].max()
            lowest = np.sort(x)[E]  # x_(E*+1)
            assert all(t < v for t in knots)
            assert all(t >= lowest for t in knots)
            assert set(knots) <= set(x.tolist())
        distinct = ref.distinct_knots(knots)
        assert distinct == sorted(set(knots), reverse=True)

    def test_tenth_weights_give_the_knots_of_unit_weights(self):
        # W-4: 100 cases of weight 0.1 (10 at each of 10 values) and 10 cases
        # of weight 1 give the same spans and knots.
        values = np.arange(10.0)
        x = np.repeat(values, 10)
        w = np.full(100, 0.1)
        N = ref.snap(ref.weight_sum(w), ref.weight_tol(ref.weight_sum(w)))
        assert N == 10.0
        tau = ref.weight_tol(N)
        L, E = ref.search_spans(1, 0, N, N, tau)
        unit_spans = ref.search_spans(1, 0, 10.0, 10.0, ref.weight_tol(10.0))
        assert unit_spans == (L, E)
        assert ref.knot_scan(x, np.ones(100, bool), w, L, E, N, tau) == (
            ref.knot_scan_unit(values, np.ones(10, bool), L, E)
        )


# ---------------------------------------------------------------------------
# Linear algebra [LA-1 to LA-4]


@pytest.fixture
def design():
    rng = np.random.default_rng(3)
    n = 40
    X = rng.uniform(-1, 1, size=(n, 3))
    B = np.column_stack(
        [np.ones(n), X[:, 0], np.maximum(X[:, 1] - 0.2, 0), X[:, 0] * X[:, 2]]
    )
    Y = np.column_stack([B @ [1.0, 2.0, -1.0, 0.5], X[:, 2]]) + rng.normal(size=(n, 2))
    w = rng.uniform(0.5, 2.0, size=n)
    return B, Y, w


class TestLinearAlgebra:
    def test_rss_matches_numpy_least_squares(self, design):
        B, Y, w = design
        assert ref.rss(B, Y, w) == pytest.approx(lstsq_rss(B, Y, w), rel=1e-12)
        assert ref.rss(B, Y[:, 0], w) == pytest.approx(
            lstsq_rss(B, Y[:, 0], w), rel=1e-12
        )

    def test_dependent_or_rescaled_columns_do_not_change_the_rss(self, design):
        B, Y, w = design
        wider = np.column_stack([B, 3.0 * B[:, 1] - B[:, 3], B[:, 2]])
        assert ref.rss(wider, Y, w) == pytest.approx(ref.rss(B, Y, w), rel=1e-12)
        assert ref.rss(B * [1.0, 1e6, 1e-6, 7.0], Y, w) == pytest.approx(
            ref.rss(B, Y, w), rel=1e-12
        )

    def test_a_large_shift_of_a_linear_column_keeps_the_rss(self, design):
        B, Y, w = design
        shifted = B.copy()
        shifted[:, 1] += 1e8
        assert ref.rss(shifted, Y, w) == pytest.approx(ref.rss(B, Y, w), rel=1e-9)

    def test_collinearity_ratio_is_one_minus_r_squared(self, design):
        B, _, w = design
        h = np.maximum(B[:, 1] - 0.1, 0) * B[:, 2] + 0.3 * B[:, 1]
        G = B[:, :3]
        sw = np.sqrt(w)
        hc = h - np.average(h, weights=w)
        expected = lstsq_rss(G, h, w) / float(np.sum(w * hc**2))
        assert ref.collinearity_ratio(h, G, w) == pytest.approx(expected, rel=1e-12)
        assert 0 < expected < 1 and sw.shape == w.shape

    def test_weighted_variance(self):
        v = np.array([1.0, 2.0, 4.0])
        w = np.array([1.0, 2.0, 1.0])
        # weighted mean 9/4; sum w (v - mean)^2 = 25/16 + 2/16 + 49/16 = 76/16
        assert ref.weighted_variance(v, w, 4.0) == pytest.approx(76 / 64, rel=1e-15)
        assert ref.weighted_variance(np.full(3, 1e300), w, 4.0) == 0.0

    def test_dependent_columns_by_the_uncentered_tolerance(self, design):
        B, _, w = design
        n = len(w)
        rng = np.random.default_rng(5)
        # u: w-orthogonal to B's columns, with the w-norm of B[:, 1]
        z = rng.normal(size=n)
        Pu = ref.Projector(B, w)
        u = Pu.residual(z) / np.sqrt(w)
        base = B[:, 1]
        size = float(np.sqrt(np.sum(w * base**2)))
        u = u * size / float(np.sqrt(np.sum(w * u**2)))
        for eps, dependent in [(0.5e-7, True), (2e-7, False), (0.0, True)]:
            cols = np.column_stack([B, base + eps * u, np.zeros(n)])
            mask = ref.dependent_columns(cols, w)
            np.testing.assert_array_equal(mask, [False] * 4 + [dependent, True])

    def test_lstsq_coef_matches_numpy_and_gives_dependent_columns_zero(self, design):
        B, Y, w = design
        sw = np.sqrt(w)
        expected = np.linalg.lstsq(B * sw[:, None], Y * sw[:, None], rcond=None)[0]
        np.testing.assert_allclose(ref.lstsq_coef(B, Y, w), expected, rtol=1e-10)
        wider = np.column_stack([B[:, :2], 2.0 * B[:, 1], B[:, 2:]])
        coef = ref.lstsq_coef(wider, Y, w)
        np.testing.assert_array_equal(coef[2], [0.0, 0.0])
        np.testing.assert_allclose(coef[[0, 1, 3, 4]], expected, rtol=1e-10)
        assert ref.lstsq_coef(B, Y[:, 0], w).shape == (4,)
