"""Sanity tests of the reference implementation (``mars_ref``), without earth.

Each test checks a case whose answer is known by hand or by an independent
computation (numpy's least squares, repeated rows, a brute-force loop).
Bracketed IDs cite ``docs/algorithm.md``.
"""

import importlib.util
import math
from pathlib import Path

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
# Sums of weights [W-4]


class TestWeights:
    def test_the_tolerance_and_its_cap(self):
        assert ref.weight_tol(20.0) == pytest.approx(2e-7, rel=1e-15)
        assert ref.weight_tol(1e8) == 0.1

    def test_the_weight_sum_is_exactly_rounded(self):
        assert ref.weight_sum([1e16, 1.0, 1.0]) == 1e16 + 2  # np.sum gives 1e16

    def test_a_sum_within_tau_of_an_integer_is_that_integer(self):
        N0 = ref.weight_sum(np.full(10, 1 - 1e-9))  # 9.99999999, below 10
        assert ref.snap(N0, ref.weight_tol(N0)) == 10.0
        assert ref.snap(19.99999999, ref.weight_tol(19.99999999)) == 20.0
        assert ref.snap(10.3, ref.weight_tol(10.3)) == 10.3

    def test_the_snap_decides_the_knots(self):
        x, active = np.arange(20.0), np.ones(20, bool)
        w = np.ones(20)
        w[0] = 1 - 1e-8
        N0 = ref.weight_sum(w)
        tau = ref.weight_tol(N0)
        N = ref.snap(N0, tau)
        assert N == 20.0
        assert ref.knot_scan(x, active, w, 3, 2, N, tau) == [17, 14, 11, 8, 5, 2]
        assert ref.knot_scan_unit(x, active, 3, 2) == [17, 14, 11, 8, 5, 2]
        assert ref.knot_scan(x, active, w, 3, 2, N0, tau) == [15, 12, 9, 6, 3]


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

    def test_rsq_and_a_finite_grsq(self):
        assert ref.rsq(2.0, 10.0) == pytest.approx(0.8, rel=1e-15)
        assert ref.rsq(1.0, 4.0) == 0.75
        # (2 / 11.25) / (10 / 18.05): C(3) = 5 and C(1) = 1 with d = 2, N = 20
        grsq = ref.grsq(2.0, 3, 10.0, 2.0, 20.0, 2e-7)
        assert grsq == pytest.approx(1 - 36.1 / 112.5, rel=1e-14)

    def test_grsq_is_minus_inf_when_the_gcv_is_inf(self):
        assert ref.grsq(1.0, 5, 10.0, 2.0, 9.0, 9e-8) == -math.inf

    def test_default_penalty(self):
        assert ref.default_penalty(1) == 2.0
        assert ref.default_penalty(2) == 3.0
        assert ref.default_penalty(3) == 3.0

    @pytest.mark.parametrize(
        ("p", "M"), [(1, 21), (10, 21), (11, 23), (100, 201), (101, 201), (150, 201)]
    )
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
            # E + floor(a E + 0.5), where floor((1 + a) E + 0.5) differs (bb06.12)
            (15, 1, 3.1, 62),
            (25, 1, 0.58, 39),
            (25, 1, 1.3, 58),
            (25, 1, 1.18, 55),
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

    def test_the_automatic_minspan_uses_the_weight_of_the_active_cases(self):
        # N_b = 20 of N = 200 gives L = 3, and N_b = N gives 4 [SPAN-1, bb06.1]
        tau = ref.weight_tol(200.0)
        assert ref.search_spans(1, 1, 20.0, 200.0, tau) == (3, 21)
        assert ref.search_spans(1, 1, 200.0, 200.0, tau) == (4, 21)

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

    def test_a_zero_knot_has_no_sign(self):
        x = np.array([-0.0, 0.0, 1.0, 2.0, 3.0, 4.0])
        for rows in (x, x[::-1]):
            active = np.ones(6, bool)
            for knots in (
                ref.knot_scan_unit(rows, active, 1, 1),
                ref.knot_scan(rows, active, np.ones(6), 1, 1, 6.0, 6e-8),
            ):
                assert knots[-1] == 0.0 and math.copysign(1.0, knots[-1]) == 1.0

    def test_large_weights_are_counted_per_case(self):
        # KNOT-7: the cost of the scan does not grow with N
        x = np.array([0.3, 0.1, 0.4, 0.2, 0.5])
        active = np.array([True, True, False, True, True])
        w = np.full(5, 2000)
        expected = ref.knot_scan_unit(*expanded(x, active, w), 7, 9)
        tau = ref.weight_tol(1e4)
        assert ref.knot_scan(x, active, w.astype(float), 7, 9, 1e4, tau) == expected
        huge = ref.knot_scan(x, active, np.full(5, 1e9), 7, 9, 5e9, 0.1)
        assert huge and set(huge) <= {0.1, 0.2, 0.3}

    def test_tenth_weights_give_the_knots_of_unit_weights(self):
        # W-4: 100 cases of weight 0.1 (10 at each of 10 values) and 10 cases
        # of weight 1 give the same spans and knots.
        values = np.arange(10.0)
        x = np.repeat(values, 10)
        w = np.full(100, 0.1)
        tau = ref.weight_tol(ref.weight_sum(w))  # from the fsum value [W-4]
        N = ref.snap(ref.weight_sum(w), tau)
        assert N == 10.0
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

    @pytest.mark.parametrize("distance", [1e-6, 1e-9])
    def test_a_column_close_to_the_span_still_counts(self, design, distance):
        # [LA-1] the direction of a column at relative distance 1e-6 or 1e-9
        # from the span of the others is kept, as numpy's lstsq keeps it
        B, _, w = design
        z = np.random.default_rng(4).normal(size=len(w))
        u = ref.Projector(B[:, :2], w).residual(z) / np.sqrt(w)
        size = float(np.sqrt(np.sum(w * B[:, 1] ** 2)))
        u = u * size / float(np.sqrt(np.sum(w * u**2)))
        cols = np.column_stack([B[:, :2], B[:, 1] + distance * u])
        noise = 0.01 * np.random.default_rng(5).normal(size=len(w))
        y = 1 + B[:, 1] + 3 * u / size + noise
        assert ref.rss(cols, y, w) == pytest.approx(lstsq_rss(cols, y, w), rel=1e-6)
        assert ref.rss(cols, y, w) < 0.01 * ref.rss(cols[:, :2], y, w)

    def test_a_large_shift_of_a_linear_column_keeps_the_rss(self, design):
        B, Y, w = design
        shifted = B.copy()
        shifted[:, 1] += 1e8
        assert ref.rss(shifted, Y, w) == pytest.approx(ref.rss(B, Y, w), rel=1e-9)

    def test_collinearity_ratio_is_one_minus_r_squared(self, design):
        B, _, w = design
        h = np.maximum(B[:, 1] - 0.1, 0) * B[:, 2] + 0.3 * B[:, 1]
        G = B[:, :3]
        hc = h - np.average(h, weights=w)
        expected = lstsq_rss(G, h, w) / float(np.sum(w * hc**2))
        assert ref.collinearity_ratio(h, G, w) == pytest.approx(expected, rel=1e-12)
        assert 0 < expected < 1

    def test_weighted_variance(self):
        v = np.array([1.0, 2.0, 4.0])
        w = np.array([1.0, 2.0, 1.0])
        # weighted mean 9/4; sum w (v - mean)^2 = 25/16 + 2/16 + 49/16 = 76/16
        assert ref.weighted_variance(v, w, 4.0) == pytest.approx(76 / 64, rel=1e-15)
        assert ref.weighted_variance(np.full(3, 1e300), w, 4.0) == 0.0
        # exactly 0 for a constant, where the computed sum is 1.9e-34
        light = np.array([0.7, 0.2, 0.1])
        assert ref.weighted_variance(np.full(3, 0.1), light, 1.0) == 0.0

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
        for eps, dependent in [(0.9e-7, True), (1.1e-7, False), (0.0, True)]:
            cols = np.column_stack([B, base + eps * u, np.zeros(n)])
            mask = ref.dependent_columns(cols, w)
            np.testing.assert_array_equal(mask, [False] * 4 + [dependent, True])

    def test_dependent_columns_use_norms_that_are_not_centered(self):
        # a column with a large mean: its part outside [1, x0] is 1e-9 of its
        # norm, though not small beside its centered norm [LA-4, bb09.9]
        rng = np.random.default_rng(7)
        x0, z = rng.uniform(size=80), rng.normal(size=80)
        cols = np.column_stack([np.ones(80), x0, 1000 + 1e-6 * z])
        np.testing.assert_array_equal(
            ref.dependent_columns(cols, np.ones(80)), [False, False, True]
        )

    def test_dependent_columns_use_w_norms(self):
        x0 = np.random.default_rng(8).uniform(size=80)
        w = np.ones(80)
        w[5] = 1e-6
        e5 = np.zeros(80)
        e5[5] = 1.0
        cols = np.column_stack([np.ones(80), x0, x0 + 2e-5 * e5])
        np.testing.assert_array_equal(
            ref.dependent_columns(cols, w), [False, False, True]
        )
        # a w-metric ratio of 1.5e-7 from a light row keeps the column, as
        # the repeated rows do
        x = np.arange(6.0)
        w = np.array([9, 9, 9, 1, 9, 9])
        e3 = np.zeros(6)
        e3[3] = 1.0
        base = np.column_stack([np.ones(6), x])
        rest = float(np.linalg.norm(ref.Projector(base, w).residual(e3)))
        delta = 1.5e-7 * float(np.sqrt(np.sum(w * x**2))) / rest
        cols = np.column_stack([base, x + delta * e3])
        np.testing.assert_array_equal(ref.dependent_columns(cols, w), [False] * 3)
        repeated = np.repeat(cols, w, axis=0)
        np.testing.assert_array_equal(
            ref.dependent_columns(repeated, np.ones(len(repeated))), [False] * 3
        )

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


# ---------------------------------------------------------------------------
# Self-checks against earth's component fixtures (validation/fixtures)

HARNESS = Path(__file__).resolve().parents[2] / "validation" / "harness"


@pytest.fixture(scope="module")
def trace_parse():
    """The harness's parser of earth trace logs, loaded from its file."""
    import sys

    if "trace_parse" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "trace_parse", HARNESS / "trace_parse.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["trace_parse"] = module
        spec.loader.exec_module(module)
    return sys.modules["trace_parse"]


def test_the_gcv_matches_earth_on_its_grid(load_fixture):
    # earth:::get.gcv for penalties -1 to 6, 10 to 100000 cases, 1 to 41 terms
    grid = load_fixture("components/gcv_grid")
    for cell in grid["grid"]:
        N = float(cell["ncases"])
        for M, value, expected in zip(
            grid["nterms"], grid["rss_per_subset"], cell["gcv"], strict=True
        ):
            mine = ref.gcv(value, M, cell["penalty"], N, ref.weight_tol(N))
            if expected is None or math.isinf(expected):
                assert mine == math.inf
            else:
                assert mine == pytest.approx(expected, rel=1e-14)


def test_the_coefficients_match_lm_fit(load_fixture):
    # R's lm.fit and lm.wfit: an NA coefficient is a dependent column [LA-4]
    fixture = load_fixture("components/lm_fit_coefficients")
    y = np.array(fixture["y"])
    for case in fixture["cases"]:
        x = np.array(case["x"], dtype=float)
        w = np.ones(len(y)) if case["weights"] is None else np.array(case["weights"])
        expected = np.array([np.nan if c is None else c for c in case["coefficients"]])
        missing = np.isnan(expected)
        np.testing.assert_array_equal(
            ref.dependent_columns(x, w), missing, err_msg=case["label"]
        )
        coef = ref.lstsq_coef(x, y, w)
        assert np.all(coef[missing] == 0) and case["rank"] == np.sum(~missing)
        error = np.linalg.norm(coef[~missing] - expected[~missing])
        assert error <= 1e-8 * np.linalg.norm(expected[~missing]), case["label"]


def test_the_spans_and_knot_lists_match_earth_traces(
    load_fixture, trace_parse, tmp_path
):
    # For each knot search in earth's trace = 9 logs, earth's nMinSpan,
    # nEndSpan and nStartSpan are L, E* and c0, and its evaluated cases hold
    # the knot list: the case at 0-based position q in the order of KNOT-2
    # gives the cut x_(q - 1) [SPAN-1 to SPAN-5, KNOT-2, KNOT-3].
    searches = 0
    for number, case in enumerate(load_fixture("components/knot_candidates")["cases"]):
        X = np.array(case["X"], dtype=float)
        n, p = X.shape
        dirs = np.array(case["dirs"], dtype=np.int8)
        cuts = np.array(case["cuts"], dtype=float)
        # earth's slots: 1 holds the intercept, 2 and 3 the terms of step 1
        # when it added a pair [FWD-9]; the fits have at most two steps
        slots = {1: 0, 2: 1}
        if len(dirs) >= 3 and np.array_equal(dirs[1], -dirs[2]):
            slots[3] = 2
        path = tmp_path / f"trace{number}.txt"
        path.write_text(case["trace_text"], encoding="utf-8")
        for step in trace_parse.parse_trace(path).steps:
            for search in step.searches:
                if search.min_span is None or search.parent not in slots:
                    continue
                k = slots[search.parent]
                b = ref.basis_matrix(X, dirs[k : k + 1], cuts[k : k + 1])[:, 0]
                active = b > 0
                x = X[:, search.pred - 1]
                N = float(n)
                L, E = ref.search_spans(
                    p,
                    ref.term_degree(dirs[k]),
                    float(active.sum()),
                    N,
                    ref.weight_tol(N),
                    case["minspan"],
                    case["endspan"],
                    case["adjust_endspan"],
                )
                spans = (search.min_span, search.end_span, search.start_span)
                assert (L, E, ref.scan_start(N, L, E)) == spans
                ordered = x[ref.case_order(x, active)]
                evaluated = [
                    float(ordered[r.case - 1]) for r in search.cases if r.evaluated
                ]
                assert ref.knot_scan_unit(x, active, L, E) == evaluated
                searches += 1
    assert searches >= 200


@pytest.mark.parametrize("key", ["one_response", "several_responses"])
def test_the_rss_of_earths_pruning_subsets(load_fixture, key):
    # earth's subsets of a fixed basis, refitted by LA-1, give its
    # rss.per.subset and gcv.per.subset
    fixture = load_fixture("components/pruning_fixed_basis")
    X = np.array(fixture["X"])
    part = fixture[key]
    B = ref.basis_matrix(
        X, np.array(part["dirs"]).astype(np.int8), np.array(part["cuts"])
    )
    Y = np.array(part["y"]).reshape(len(X), -1)
    N = float(len(X))
    for m, row in enumerate(np.array(part["prune_terms"], dtype=int)):
        value = ref.rss(B[:, sorted(row[row > 0] - 1)], Y, np.ones(len(X)))
        assert value == pytest.approx(part["rss_per_subset"][m], rel=1e-10)
        gcv = ref.gcv(value, m + 1, fixture["penalty"], N, ref.weight_tol(N))
        assert gcv == pytest.approx(part["gcv_per_subset"][m], rel=1e-10)
