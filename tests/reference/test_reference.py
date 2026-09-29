"""Sanity tests of the reference implementation (``mars_ref``), without earth.

Each test checks a case whose answer is known by hand or by an independent
computation (numpy's least squares, repeated rows, a brute-force loop).
Bracketed IDs cite ``docs/algorithm.md``.
"""

import dataclasses
import importlib.util
import math
import types
from fractions import Fraction
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

    @given(scans(weights=True), st.booleans())
    def test_the_distinct_scan_keeps_the_first_listing_of_each_value(
        self, case, tenths
    ):
        x, active, L, E, w = case
        w = w * 0.1 if tenths else w.astype(float)
        N0 = ref.weight_sum(w)
        tau = ref.weight_tol(N0)
        N = ref.snap(N0, tau)
        full = ref.knot_scan(x, active, w, L, E, N, tau)
        assert ref.knot_scan(x, active, w, L, E, N, tau, distinct=True) == (
            ref.distinct_knots(full)
        )

    def test_the_distinct_scan_does_not_grow_with_n(self):
        # N = 5e9 and L = 7 list the three knots about 4.3e8 times in all
        x = np.array([0.3, 0.1, 0.4, 0.2, 0.5])
        active = np.array([True, True, False, True, True])
        knots = ref.knot_scan(x, active, np.full(5, 1e9), 7, 9, 5e9, 0.1, distinct=True)
        assert knots == [0.3, 0.2, 0.1]

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
        # KNOT-7: the state of the scan does not grow with N; the list does,
        # so the large cases take a large minspan
        x = np.array([0.3, 0.1, 0.4, 0.2, 0.5])
        active = np.array([True, True, False, True, True])
        w = np.full(5, 2000)
        expected = ref.knot_scan_unit(*expanded(x, active, w), 7, 9)
        tau = ref.weight_tol(1e4)
        assert ref.knot_scan(x, active, w.astype(float), 7, 9, 1e4, tau) == expected
        # c0 = 5e7, and each active case below v gives about 1e9 moves
        huge = ref.knot_scan(x, active, np.full(5, 1e9), 10**8, 9, 5e9, 0.1)
        assert huge == [0.3] * 10 + [0.2] * 10 + [0.1] * 10
        # c0 = 5e8 and 1e9 - 1 moves: one knot
        two = ref.knot_scan(
            [0.0, 1.0], [True, True], np.full(2, 1e9), 10**9, 1, 2e9, 0.1
        )
        assert two == [0.0]
        # A knot on the last move of a stretch. With x = (0, 1), all active and
        # E* = 1, the moves are the visit u = W0 + 1 (a stretch of one), then
        # u = W0, ..., 3: W0 - 1 moves in all, each with t = 0.
        # W = (1e9 + 1, 1e9 + 2), L = 1e9: D = 2e9, g = 0, c0 = 1, the knot is
        # the one move of the first stretch; W = (1e9, 1e9 - 1), L = 3e9:
        # D = g = 2e9 - 4, c0 = 1e9 - 1 = W0 - 1, the last move of the second.
        for w0, w1, L in [(1e9 + 1, 1e9 + 2, 10**9), (1e9, 1e9 - 1, 3 * 10**9)]:
            w = np.array([w0, w1])
            knots = ref.knot_scan([0.0, 1.0], [True, True], w, L, 1, w0 + w1, 0.1)
            assert knots == [0.0]

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

    @pytest.mark.parametrize(("distance", "rel"), [(1e-6, 1e-7), (1e-9, 1e-4)])
    def test_a_column_close_to_the_span_still_counts(self, design, distance, rel):
        # [LA-1] the direction of a column at relative distance 1e-6 or 1e-9
        # from the span of the others is kept, as numpy's lstsq keeps it; the
        # two agree to about 10 eps / distance, the conditioning of the fit
        B, _, w = design
        z = np.random.default_rng(4).normal(size=len(w))
        u = ref.Projector(B[:, :2], w).residual(z) / np.sqrt(w)
        size = float(np.sqrt(np.sum(w * B[:, 1] ** 2)))
        u = u * size / float(np.sqrt(np.sum(w * u**2)))
        cols = np.column_stack([B[:, :2], B[:, 1] + distance * u])
        noise = 0.01 * np.random.default_rng(5).normal(size=len(w))
        y = 1 + B[:, 1] + 3 * u / size + noise
        assert ref.rss(cols, y, w) == pytest.approx(lstsq_rss(cols, y, w), rel=rel)
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
# The pruning pass [PRUNE-1 to PRUNE-8]


def backward_elimination(B, Y, w):
    """Plain backward elimination with ref.rss, independent of ref.prune."""
    current = list(range(B.shape[1]))
    path = {len(current): ref.rss(B[:, current], Y, w)}
    removed = []
    while len(current) > 1:
        best, best_rss = -1, math.inf
        for term in current[1:]:
            value = ref.rss(B[:, [t for t in current if t != term]], Y, w)
            if value < best_rss or (value == best_rss and term > best):
                best, best_rss = term, value
        removed.append(best)
        current.remove(best)
        path[len(current)] = best_rss
    return removed, path


def random_basis(seed, n=30, M=6, K=1):
    rng = np.random.default_rng(seed)
    B = np.column_stack([np.ones(n), rng.normal(size=(n, M - 1))])
    Y = B @ rng.normal(size=(M, K)) + rng.normal(size=(n, K))
    return B, Y, np.ones(n)


def prune(B, Y, w, **kwargs):
    N = ref.weight_sum(w)
    kwargs.setdefault("penalty", 2.0)
    return ref.prune(B, Y, w, N=N, tau_N=ref.weight_tol(N), **kwargs)


def offered_sets(B, Y, w):
    """PRUNE-3 for one response, written out again: the working order, its
    offers, and the move of each removed term to position pos. Returns
    {m: (R[m], T[m])}."""
    Mf = B.shape[1]
    order = list(range(Mf))
    best = {}

    def offer():
        for m in range(1, Mf + 1):
            value = ref.rss(B[:, sorted(order[:m])], Y, w)
            if m not in best or value < best[m][0]:
                best[m] = (value, sorted(order[:m]))

    offer()
    for pos in range(Mf, 1, -1):
        values = {
            t: ref.rss(B[:, sorted(set(order[:pos]) - {t})], Y, w) for t in order[1:pos]
        }
        low = min(values.values())
        term = max(t for t, value in values.items() if value == low)
        order.remove(term)
        order.insert(pos - 1, term)
        offer()
    return best


def expected_statistics(B, Y, w, selected, penalty):
    """rss, gcv, rsq, grsq and tss of the model on the selected columns, from
    numpy's least squares and the formulas of GCV-2, GCV-5 and GCV-6."""
    N = ref.weight_sum(w)
    tau = ref.weight_tol(N)
    Y2 = Y.reshape(len(Y), -1)
    rss = lstsq_rss(B[:, selected], Y2, w)
    tss = float(np.sum(w[:, None] * (Y2 - np.average(Y2, axis=0, weights=w)) ** 2))
    gcv = ref.gcv(rss, len(selected), penalty, N, tau)
    grsq = 1 - gcv / ref.gcv(tss, 1, penalty, N, tau)
    return {"rss": rss, "gcv": gcv, "rsq": 1 - rss / tss, "grsq": grsq, "tss": tss}


class TestPruning:
    def test_selects_the_true_terms(self):
        rng = np.random.default_rng(11)
        n = 100
        x = rng.uniform(size=(n, 4))
        B = np.column_stack([np.ones(n), np.maximum(x - 0.3, 0)])
        y = 1.0 + 3.0 * B[:, 2] + rng.normal(scale=0.05, size=n)
        result = prune(B, y, np.ones(n))
        np.testing.assert_array_equal(result["selected"], [0, 2])
        assert result["coef"].shape == (2, 1)
        expected = np.linalg.lstsq(B[:, [0, 2]], y, rcond=None)[0]
        np.testing.assert_allclose(result["coef"][:, 0], expected, rtol=1e-12)
        np.testing.assert_allclose(expected, [1.0, 3.0], atol=0.1)
        expected_rss = lstsq_rss(B[:, [0, 2]], y, np.ones(n))
        assert result["rss"] == pytest.approx(expected_rss, rel=1e-8)

    @pytest.mark.parametrize("seed", range(5))
    def test_one_response_removes_as_backward_elimination_and_does_no_worse(self, seed):
        B, Y, w = random_basis(seed)
        result = prune(B, Y, w)
        removed, path = backward_elimination(B, Y, w)
        np.testing.assert_array_equal(result["removed"], removed)
        for m, value in path.items():
            assert result["rss_per_size"][m - 1] <= value
        np.testing.assert_array_equal(result["subsets"][0], [True] + [False] * 5)

    def test_one_response_offers_the_forward_order_and_several_do_not(self):
        # y = x1 + x2 and term 1 is x3 = x1 + x2 + noise. Backward elimination
        # removes term 1 first (the other two fit y exactly), so its size-2
        # set holds x1 or x2 alone. With one response the first offer of the
        # forward order has already found {0, 1}, which is far better
        # [PRUNE-3]; with two responses the pass is backward elimination.
        rng = np.random.default_rng(1)
        n = 30
        x1, x2 = rng.normal(size=(2, n))
        x3 = x1 + x2 + 0.01 * rng.normal(size=n)
        B = np.column_stack([np.ones(n), x3, x1, x2])
        y = x1 + x2
        one = prune(B, y, np.ones(n))
        removed, path = backward_elimination(B, y, np.ones(n))
        assert removed[0] == 1
        np.testing.assert_array_equal(one["removed"], removed)
        np.testing.assert_array_equal(one["subsets"][1], [True, True, False, False])
        assert one["rss_per_size"][1] < 1e-2 * path[2]
        two = prune(B, np.column_stack([y, 2 * y]), np.ones(n))
        assert not two["subsets"][1][1]

    @pytest.mark.parametrize("seed", range(3))
    def test_several_responses_are_plain_backward_elimination(self, seed):
        B, Y, w = random_basis(seed, K=2)
        result = prune(B, Y, w)
        removed, path = backward_elimination(B, Y, w)
        np.testing.assert_array_equal(result["removed"], removed)
        for m, value in path.items():
            assert result["rss_per_size"][m - 1] == value
        subsets = result["subsets"]
        for m in range(1, len(subsets)):
            assert np.all(subsets[m] >= subsets[m - 1])  # nested

    def test_two_terms_give_the_same_record_for_both_rules(self):
        B, Y, w = random_basis(2, M=2, K=2)
        one = prune(B, Y[:, :1], w)
        two = prune(B, np.column_stack([Y[:, 0], 0 * Y[:, 0]]), w)
        np.testing.assert_array_equal(one["removed"], two["removed"])
        np.testing.assert_array_equal(one["subsets"], two["subsets"])

    def test_the_intercept_alone(self):
        y = np.array([1.0, 2.0, 4.0])
        result = prune(np.ones((3, 1)), y, np.ones(3))
        assert result["removed"].shape == (0,)
        np.testing.assert_allclose(result["rss_per_size"], [14 / 3], rtol=1e-14)
        np.testing.assert_array_equal(result["subsets"], [[True]])
        assert result["selected_size"] == 1
        assert result["rsq"] == 0.0 and result["grsq"] == 0.0
        np.testing.assert_allclose(result["coef"], [[7 / 3]], rtol=1e-14)

    def test_one_response_follows_prune_3_step_by_step(self):
        for seed in range(20):
            B, Y, w = random_basis(seed, n=10, M=5)
            result = prune(B, Y, w)
            for m, (value, terms) in offered_sets(B, Y, w).items():
                assert result["rss_per_size"][m - 1] == value
                np.testing.assert_array_equal(
                    np.flatnonzero(result["subsets"][m - 1]), terms
                )

    def test_one_response_follows_prune_3_on_a_dependent_basis(self):
        # term 3 = term 1 + term 2, so some sets tie with their subsets up to
        # rounding; T[m] still holds m terms [PRUNE-2, PRUNE-3]
        for seed in range(10):
            rng = np.random.default_rng(seed)
            x1, x2 = rng.normal(size=(2, 10))
            B = np.column_stack([np.ones(10), x1, x2, x1 + x2])
            y = x1 - x2 + rng.normal(size=10)
            result = prune(B, y, np.ones(10))
            np.testing.assert_array_equal(result["subsets"].sum(axis=1), [1, 2, 3, 4])
            for m, (value, terms) in offered_sets(B, y, np.ones(10)).items():
                assert result["rss_per_size"][m - 1] == value
                np.testing.assert_array_equal(
                    np.flatnonzero(result["subsets"][m - 1]), terms
                )

    def test_a_pure_linear_truth_keeps_m_terms_in_each_set(self):
        # y = 1 + 2 x1 exactly, so every set with terms 0 and 1 has an RSS at
        # the level of rounding, and a smaller set can compute below a larger
        # one. The order keeps all M_f terms, so T[m] still has m terms
        # [PRUNE-2, PRUNE-3 step 2].
        for seed in range(5):
            rng = np.random.default_rng(seed)
            B = np.column_stack([np.ones(10), rng.normal(size=(10, 3))])
            Y = (1 + 2 * B[:, 1])[:, None]
            result = prune(B, Y, np.ones(10))
            np.testing.assert_array_equal(result["subsets"].sum(axis=1), [1, 2, 3, 4])
            for m, (value, terms) in offered_sets(B, Y, np.ones(10)).items():
                assert result["rss_per_size"][m - 1] == value
                np.testing.assert_array_equal(
                    np.flatnonzero(result["subsets"][m - 1]), terms
                )

    def test_a_removed_term_moves_to_position_pos(self):
        # T[2] = {0, 1}. A move of the removed term to the end of the order
        # would offer {0, 2} (RSS 11.97), which PRUNE-3 does not offer here.
        rng = np.random.default_rng(9)
        B = np.column_stack([np.ones(10), rng.normal(size=(10, 4))])
        y = B @ rng.normal(size=5) + rng.normal(size=10)
        result = prune(B, y, np.ones(10))
        np.testing.assert_array_equal(result["subsets"][1], [1, 1, 0, 0, 0])
        expected = lstsq_rss(B[:, :2], y, np.ones(10))
        assert result["rss_per_size"][1] == pytest.approx(expected, rel=1e-8)

    def test_equal_rss_removes_the_term_with_the_largest_index(self):
        # terms 1 and 2 are the same column, so both removals leave the same
        # matrix and the same RSS, bit for bit [PRUNE-3]
        rng = np.random.default_rng(0)
        x = rng.normal(size=8)
        B = np.column_stack([np.ones(8), x, x])
        y = x + 0.1 * rng.normal(size=8)
        np.testing.assert_array_equal(prune(B, y, np.ones(8))["removed"], [2, 1])
        B2 = np.column_stack([B, rng.normal(size=8)])
        Y2 = np.column_stack([y, x + rng.normal(size=8)])
        np.testing.assert_array_equal(prune(B2, Y2, np.ones(8))["removed"], [2, 3, 1])

    def test_nprune_caps_the_selected_size_only(self):
        B, Y, w = random_basis(4)
        free = prune(B, Y, w, penalty=0.0)
        capped = prune(B, Y, w, penalty=0.0, nprune=2)
        assert capped["selected_size"] <= 2
        for key in ("removed", "rss_per_size", "gcv_per_size", "subsets"):
            np.testing.assert_array_equal(free[key], capped[key])

    @pytest.mark.parametrize("K", [1, 2])
    def test_nprune_three_selects_three_terms_where_four_are_best(self, K):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(30, 3))
        B = np.column_stack([np.ones(30), X])
        y = X.sum(axis=1) + 0.1 * rng.normal(size=30)
        Y = np.column_stack([y] * K)
        assert prune(B, Y, np.ones(30))["selected_size"] == 4
        assert prune(B, Y, np.ones(30), nprune=3)["selected_size"] == 3

    def test_penalty_minus_one_selects_exactly_nprune_terms(self):
        # GCV = RSS / N falls with the size, so the cap decides [PRUNE-6]
        B, Y, w = random_basis(4)
        for k in range(1, 7):
            assert prune(B, Y, w, penalty=-1, nprune=k)["selected_size"] == k

    def test_nprune_below_one_is_an_error(self):
        B, Y, w = random_basis(0, M=3)
        for value in (0, -1):
            with pytest.raises(ValueError, match="nprune"):
                prune(B, Y, w, nprune=value)

    def test_the_final_statistics_with_weights(self):
        B, Y, _ = random_basis(12, n=40)
        w = np.random.default_rng(12).uniform(0.5, 2.0, size=40)
        result = prune(B, Y, w, penalty=3.0)
        assert result["selected_size"] > 1
        expected = expected_statistics(B, Y, w, result["selected"], 3.0)
        for key in ("rss", "gcv", "tss"):
            assert result[key] == pytest.approx(expected[key], rel=1e-10), key
        for key in ("rsq", "grsq"):
            assert result[key] == pytest.approx(expected[key], abs=1e-12), key
        N = ref.weight_sum(w)
        for m in range(1, 7):  # [PRUNE-4]
            value = ref.gcv(result["rss_per_size"][m - 1], m, 3.0, N, ref.weight_tol(N))
            assert result["gcv_per_size"][m - 1] == value

    def test_pmethod_none_reports_the_statistics_of_its_terms(self):
        # T[3] = {0, 2, 5}, so the statistics of terms 0 to 2 differ from R[3]
        # and GCV(R[3], 3) [PRUNE-7, PRUNE-8]
        B, Y, w = random_basis(6)
        result = prune(B, Y, w, pmethod="none", nprune=3)
        np.testing.assert_array_equal(np.flatnonzero(result["subsets"][2]), [0, 2, 5])
        expected = expected_statistics(B, Y, w, [0, 1, 2], 2.0)
        for key in ("rss", "gcv", "tss"):
            assert result[key] == pytest.approx(expected[key], rel=1e-10), key
        for key in ("rsq", "grsq"):
            assert result[key] == pytest.approx(expected[key], abs=1e-12), key
        assert result["gcv"] > 1.05 * result["gcv_per_size"][2]

    def test_a_large_penalty_selects_the_intercept_with_rsq_and_grsq_zero(self):
        # GCV-7: 0 by definition. The final rss of the intercept model is its
        # TSS, from the same centered projection, so RSS / TSS is 1 exactly
        rng = np.random.default_rng(1)
        n = int(rng.integers(5, 40))
        B = np.column_stack([np.ones(n), rng.normal(size=(n, 3))])
        y = rng.normal(size=n) * 10.0 ** rng.integers(-3, 4) + rng.normal() * 100
        result = prune(B, y, np.ones(n), penalty=1000.0)
        assert result["selected_size"] == 1
        assert result["rsq"] == 0.0 and result["grsq"] == 0.0
        assert result["rss"] == result["tss"]

    @pytest.mark.parametrize("offset", [0.0, 0.5e-7])
    def test_two_equal_selected_columns(self, offset):
        # LA-4 gives the later of two equal columns, or of two whose LA-4
        # ratio is 0.5e-7, the coefficient 0, and the final rss describes the
        # returned model [PRUNE-8]; y has a component along the difference, so
        # the projection on all three columns would have a smaller RSS
        rng = np.random.default_rng(6)
        x = rng.normal(size=30)
        u = ref.Projector(np.column_stack([np.ones(30), x]), np.ones(30)).residual(
            rng.normal(size=30)
        )
        u = u / np.linalg.norm(u)
        column = x + offset * np.linalg.norm(x) * u
        B = np.column_stack([np.ones(30), x, column, rng.normal(size=30)])
        y = 1 + 2 * x + rng.normal(size=30) + 10 * u
        result = prune(B, y, np.ones(30), pmethod="none", nprune=3)
        coef = result["coef"][:, 0]
        assert coef[2] == 0.0
        np.testing.assert_allclose(
            coef[:2], np.linalg.lstsq(B[:, :2], y, rcond=None)[0], rtol=1e-10
        )
        assert result["rss"] == pytest.approx(
            lstsq_rss(B[:, :2], y, np.ones(30)), rel=1e-10
        )

    def test_pmethod_none_keeps_the_first_terms(self):
        B, Y, w = random_basis(6)
        result = prune(B, Y, w, pmethod="none", nprune=3)
        np.testing.assert_array_equal(result["selected"], [0, 1, 2])
        assert result["rss"] == pytest.approx(ref.rss(B[:, :3], Y, w), rel=1e-12)
        full = prune(B, Y, w, pmethod="none")
        assert full["selected_size"] == 6
        assert full["rss"] == pytest.approx(full["rss_per_size"][-1], rel=1e-12)

    def test_tied_gcv_minima_go_to_the_smaller_size(self):
        assert ref.select_size([3.0, 2.0, 2.0, 1.0], 3) == 2
        assert ref.select_size([3.0, 2.0, 2.0, 1.0], 4) == 4
        assert ref.select_size([1.0, math.inf], 2) == 1

    def test_integer_weights_give_the_pruning_of_repeated_rows(self):
        B, Y, _ = random_basis(8, n=25)
        w = np.random.default_rng(8).integers(1, 4, size=25)
        weighted = prune(B, Y, w.astype(float))
        repeated = prune(
            np.repeat(B, w, axis=0), np.repeat(Y, w, axis=0), np.ones(w.sum())
        )
        np.testing.assert_array_equal(weighted["removed"], repeated["removed"])
        np.testing.assert_array_equal(weighted["selected"], repeated["selected"])
        np.testing.assert_allclose(
            weighted["rss_per_size"], repeated["rss_per_size"], rtol=1e-10
        )
        np.testing.assert_allclose(weighted["coef"], repeated["coef"], rtol=1e-9)
        for key in ("rss", "gcv", "rsq", "grsq", "tss"):
            assert weighted[key] == pytest.approx(repeated[key], rel=1e-9), key

    def test_unknown_pmethod_is_an_error(self):
        B, Y, w = random_basis(0, M=2)
        with pytest.raises(ValueError, match="pmethod"):
            prune(B, Y, w, pmethod="exhaustive")


# ---------------------------------------------------------------------------
# Parameters [CORE-2]


class TestParams:
    def test_defaults_and_resolved_values(self):
        assert ref.Params().resolved_max_terms(3) == 21
        assert ref.Params().resolved_penalty() == 2.0
        assert ref.Params(max_degree=2).resolved_penalty() == 3.0
        params = ref.Params(max_degree=np.int64(2), max_terms=np.int32(9), penalty=-1)
        assert params.resolved_max_terms(3) == 9 and params.resolved_penalty() == -1.0

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("max_degree", 0),
            ("max_degree", True),
            ("max_degree", 2.0),
            ("max_terms", 0),
            ("penalty", -0.5),
            ("penalty", math.inf),
            ("thresh", -1e-3),
            ("thresh", math.nan),
            ("minspan", 0),
            ("endspan", 1.5),
            ("adjust_endspan", -1.0),
            ("auto_linpreds", 1),
            ("fast_k", -1),
            ("fast_beta", math.inf),
            ("pmethod", "cv"),
            ("nprune", 0),
        ],
    )
    def test_a_value_out_of_range_names_its_field(self, field, value):
        with pytest.raises(ValueError, match=field):
            ref.Params(**{field: value})

    def test_as_params_reads_a_mapping_or_attributes(self):
        assert ref.as_params(None) == ref.Params()
        assert ref.as_params({"max_degree": 2}).max_degree == 2
        fields = dataclasses.asdict(ref.Params(max_degree=3, fast_k=0))
        other = types.SimpleNamespace(**fields)
        assert ref.as_params(other) == ref.Params(max_degree=3, fast_k=0)
        with pytest.raises(ValueError, match="unknown"):
            ref.as_params({"degree": 2})
        # a field that the object lacks is an error, not a default (CORE-6)
        del fields["nprune"]
        with pytest.raises(ValueError, match="nprune"):
            ref.as_params(types.SimpleNamespace(**fields))


# ---------------------------------------------------------------------------
# The forward pass [FWD, STOP, FAST]


def run_forward(X, y, w=None, trace=None, **params):
    """The forward pass on (X, y, w), with N, tau_N and TSS as a fit sets them."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(y, dtype=float).reshape(len(X), -1)
    w = np.ones(len(X)) if w is None else np.asarray(w, dtype=float)
    N = ref.weight_sum(w)
    tau = ref.weight_tol(N)
    record, _ = ref.forward_pass(
        X,
        Y,
        w,
        params,
        N=ref.snap(N, tau),
        tau_N=tau,
        tss=ref.rss(np.ones((len(X), 1)), Y, w),
        record_candidates=True,
        trace=trace,
    )
    return record


class TestQueue:
    def test_table_after_two_pair_steps_at_degree_one(self):
        # FAST-6: the intercept (entry 1) has aged rank 4, the others 2 or 3
        entries = [(5.0, 4), (math.inf, 2), (math.inf, 2), (math.inf, 4), (math.inf, 4)]
        assert ref.queue_table(entries, 2, 1.0) == [2, 4, 3, 5, 1]
        assert ref.queue_table(entries, 2, 0.0) == [2, 3, 4, 5, 1]

    def test_equal_aged_ranks_go_in_increasing_rank(self):
        # ranks: entry 2 -> 0, entry 3 -> 1, entry 1 -> 2; aged 2, 1, 2
        entries = [(3.0, 4), (math.inf, 2), (math.inf, 4)]
        assert ref.queue_table(entries, 2, 1.0) == [3, 2, 1]

    def test_the_first_table_holds_the_intercept(self):
        assert ref.queue_table([(math.inf, 0)], 0, 1.0) == [1]

    def test_window(self):
        table = list(range(1, 8))
        assert ref.window(table, 0) == table
        assert ref.window(table, 1) == [1, 2, 3]
        assert ref.window(table, 5) == [1, 2, 3, 4, 5]
        assert ref.window(table[:3], 5) == [1, 2, 3]


class TestLegality:
    def test_max_legal(self):
        assert ref.max_legal([10.0]) == pytest.approx(10.1)
        assert ref.max_legal([10.0, 8.0]) == pytest.approx(8.08)
        assert ref.max_legal([10.0, 9.5]) == pytest.approx(5.0)

    def test_knots_are_capped_and_linear_candidates_are_not(self):
        assert not ref.is_legal(ref.PAIR, 0.0, 5.0)
        assert ref.is_legal(ref.SINGLE, 5.0, 5.0)
        assert not ref.is_legal(ref.PAIR, 5.000001, 5.0)
        assert ref.is_legal(ref.LINEAR, 100.0, 5.0)
        assert not ref.is_legal(ref.LINEAR, 0.0, 5.0)

    def test_the_queue_value_of_a_searched_entry(self):
        # FAST-5 with RSS_s = 10 and MaxLegal_s = 5
        linear = ref.Candidate(0, 0, ref.LINEAR, math.nan, 2.0)  # reduction 8
        too_big = ref.Candidate(0, 0, ref.PAIR, 0.5, 3.0)  # reduction 7 > 5
        knot = ref.Candidate(0, 1, ref.PAIR, 0.5, 6.0)  # reduction 4
        assert ref.queue_value([linear, too_big, knot], True, 10.0, 5.0) == 8.0
        assert ref.queue_value([too_big, knot], True, 10.0, 5.0) == 4.0
        assert ref.queue_value([too_big], True, 10.0, 5.0) == 0.0
        assert ref.queue_value([], True, 10.0, 5.0) == 0.0
        assert ref.queue_value([], False, 10.0, 5.0) == -1.0

    def test_the_candidate_key(self):
        # CORE-3: a linear candidate has no knot, whatever NaN it holds
        a = ref.Candidate(0, 1, ref.LINEAR, math.nan, 1.0)
        b = ref.Candidate(0, 1, ref.LINEAR, float("nan"), 2.0)
        assert ref.candidate_key(a) == ref.candidate_key(b) == (0, 1, ref.LINEAR, None)
        knot = ref.Candidate(0, 1, ref.PAIR, 0.25, 1.0)
        assert ref.candidate_key(knot) == (0, 1, ref.PAIR, 0.25)

    def test_the_covariate_variances_divide_by_n(self):
        X = np.random.default_rng(26).normal(size=(12, 3))
        np.testing.assert_allclose(
            ref.covariate_variances(X, np.ones(12), 12.0), np.var(X, axis=0), rtol=1e-13
        )

    def test_collinearity_tolerance_changes_after_seven_steps(self):
        assert ref.collinearity_tol(0) == ref.collinearity_tol(6) == 0.01
        assert ref.collinearity_tol(7) == 1e-5


def exact_rss(A, y, w):
    """The weighted RSS of y on the columns of A, in exact rational
    arithmetic: Gram-Schmidt with fractions, which leaves out a column in the
    span of the earlier ones."""
    ww = [Fraction(float(v)) for v in w]

    def dot(u, v):
        return sum(c * a * b for c, a, b in zip(ww, u, v, strict=True))

    basis = []

    def residual(v):
        for q, qq in basis:
            f = dot(v, q) / qq
            if f:
                v = [vi - f * qi for vi, qi in zip(v, q, strict=True)]
        return v

    for k in range(len(A[0])):
        column = [
            v if isinstance(v, Fraction) else Fraction(float(v))
            for v in (row[k] for row in A)
        ]
        u = residual(column)
        uu = dot(u, u)
        if uu:
            basis.append((u, uu))
    r = residual([Fraction(float(v)) for v in y])
    return float(dot(r, r))


def exact_basis(X, dirs, cuts):
    """The columns of the terms [TERM-3] as rows of fractions: every factor and
    product is exact."""
    B = [[Fraction(1)] * len(dirs) for _ in range(len(X))]
    for k, (row, cut) in enumerate(zip(dirs, cuts, strict=True)):
        for j in np.flatnonzero(row):
            c = Fraction(float(cut[j]))
            for i, v in enumerate(Fraction(float(v)) for v in X[:, j]):
                B[i][k] *= {1: max(v - c, 0), -1: max(c - v, 0), 2: v}[int(row[j])]
    return B


def check_best_and_second(log, index, cands, rss_s, limit):
    """The log entry of a step against the legal candidates of the step:
    the first with the largest reduction, then the first of the others that
    differ from it [FWD-4, FWD-5, FWD-8, CORE-3]."""
    legal = [c for c in cands if ref.is_legal(c.kind, rss_s - c.rss, limit)]
    best = max(legal, key=lambda c: rss_s - c.rss)
    best = next(c for c in legal if rss_s - c.rss == rss_s - best.rss)
    others = [c for c in legal if ref.candidate_key(c) != ref.candidate_key(best)]
    second = next(
        c for c in others if rss_s - c.rss == max(rss_s - o.rss for o in others)
    )
    # the same values in another memory layout: equal up to the last bits
    assert log["best_rss"][index] == pytest.approx(best.rss, rel=1e-12)
    assert log["second_rss"][index] == pytest.approx(second.rss, rel=1e-12)
    assert log["second_parent"][index] == second.parent
    assert log["second_variable"][index] == second.variable
    assert log["second_kind"][index] == second.kind
    if second.kind == ref.LINEAR:
        assert math.isnan(log["second_knot"][index])
    else:
        assert log["second_knot"][index] == second.knot


def noisy_data(seed, n=60, p=2):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, p))
    y = np.abs(X[:, 0] - 0.4) + X[:, 0] * np.maximum(X[:, -1] - 0.5, 0)
    return X, y + rng.normal(scale=0.05, size=n)


class TestForwardPass:
    def test_every_candidate_rss_is_a_fresh_least_squares_fit(self):
        # [LA-2] with numpy's SVD solve on [B, b x, h], [B, b x] or [B, h]
        X, y = noisy_data(10, n=50)
        w = np.random.default_rng(10).uniform(0.5, 2.0, size=50)
        t0 = float(np.median(X[:, 0]))
        B = np.column_stack(
            [np.ones(50), np.maximum(X[:, 0] - t0, 0), np.maximum(t0 - X[:, 0], 0)]
        )
        rows = np.array([[0, 0], [1, 0], [-1, 0]], dtype=np.int8)
        N = ref.weight_sum(w)
        tau = ref.weight_tol(N)
        P_B = ref.Projector(B, w)
        sigma2 = [ref.weighted_variance(X[:, j], w, N) for j in range(2)]
        kinds = set()
        for k in range(3):
            cands, _ = ref._parent_candidates(
                k,
                rows[k],
                X,
                y[:, None],
                w,
                B,
                P_B,
                P_B.residual(y[:, None]),
                sigma2,
                N,
                tau,
                ref.Params(),
                0.01,
            )
            for c in cands:
                b, x = B[:, k], X[:, c.variable]
                cols = [B]
                if c.kind in (ref.PAIR, ref.LINEAR):
                    cols.append(b * x)
                if c.kind != ref.LINEAR:
                    cols.append(b * np.maximum(x - c.knot, 0))
                expected = lstsq_rss(np.column_stack(cols), y, w)
                assert c.rss == pytest.approx(expected, rel=1e-10)
                kinds.add(c.kind)
        assert kinds == {ref.PAIR, ref.SINGLE, ref.LINEAR}

    def test_a_single_true_knot(self):
        x = np.arange(21.0) / 20
        rec = run_forward(x[:, None], 2 * np.maximum(x - 0.5, 0), minspan=1, endspan=1)
        np.testing.assert_array_equal(rec["dirs"], [[0], [1], [-1]])
        np.testing.assert_array_equal(rec["cuts"], [[0.0], [0.5], [0.5]])
        np.testing.assert_array_equal(rec["parent"], [-1, 0, 0])
        np.testing.assert_array_equal(rec["step"], [0, 1, 1])
        assert rec["termination"] == ref.RSQ_HIGH
        assert rec["rss"][1] < 1e-20 * rec["rss"][0]
        log = rec["candidates"]
        np.testing.assert_array_equal(log["best_rss"], rec["rss"][1:])
        assert log["second_rss"][0] > log["best_rss"][0]
        default = run_forward(x[:, None], 2 * np.maximum(x - 0.5, 0))
        np.testing.assert_array_equal(default["cuts"], rec["cuts"])

    def test_a_linear_truth_adds_the_linear_term(self):
        x = np.arange(21.0) / 20
        rec = run_forward(x[:, None], 1 + 2 * x)
        np.testing.assert_array_equal(rec["dirs"], [[0], [2]])
        np.testing.assert_array_equal(rec["cuts"], [[0.0], [0.0]])
        assert rec["termination"] == ref.RSQ_HIGH
        hinge = run_forward(x[:, None], 1 + 2 * x, auto_linpreds=False)
        np.testing.assert_array_equal(hinge["dirs"], [[0], [1]])
        np.testing.assert_array_equal(hinge["cuts"], [[0.0], [0.0]])  # min x

    def test_at_an_exact_fit_the_second_best_may_round_below_the_best(self):
        # y = 1 + 2x: the linear candidate and many knots reduce the RSS to
        # rounding noise, with equal reductions in float64; FWD-5 takes the
        # linear candidate, and the logged second best (a knot) may have an
        # RSS a few 1e-31 below it, far inside 1e-12 of the RSS before the step
        x = np.arange(32.0) / 32
        rec = run_forward(x[:, None], 1 + 2 * x, minspan=1, endspan=1)
        np.testing.assert_array_equal(rec["dirs"], [[0], [2]])
        log = rec["candidates"]
        assert log["second_kind"][0] == ref.PAIR
        assert log["second_rss"][0] >= log["best_rss"][0] - 1e-12 * rec["rss"][0]
        assert (
            rec["rss"][0] - log["second_rss"][0] == rec["rss"][0] - log["best_rss"][0]
        )

    def test_a_binary_covariate_offers_only_its_linear_term(self):
        # its one knot is the repeated minimum, which LA-3 rejects [KNOT-5]
        rng = np.random.default_rng(0)
        x = (np.arange(40) % 2).astype(float)
        rec = run_forward(
            x[:, None], 3 * x + rng.normal(scale=0.1, size=40), max_terms=3
        )
        np.testing.assert_array_equal(rec["dirs"], [[0], [2]])
        assert rec["termination"] == ref.TERM_LIMIT
        log = rec["candidates"]
        assert log["second_kind"][0] == ref.NO_KIND
        assert log["second_rss"][0] == math.inf  # CORE-3: no second candidate
        assert log["second_parent"][0] == log["second_variable"][0] == -1
        assert math.isnan(log["second_knot"][0])

    def test_constant_and_duplicate_covariates_never_enter_at_degree_one(self):
        X, y = noisy_data(1)
        X3 = np.column_stack([X[:, 0], np.full(60, 3.0), X[:, 0]])
        rec = run_forward(X3, y)
        assert len(rec["dirs"]) > 3
        assert not rec["dirs"][:, 1:].any()  # [EDGE-3, EDGE-4]

    def test_a_second_search_on_the_same_covariate_is_a_single_hinge(self):
        x = np.arange(40.0) / 40
        y = 5 * np.abs(x - 0.3) + np.maximum(x - 0.7, 0)
        rec = run_forward(x[:, None], y, minspan=1, endspan=1)
        np.testing.assert_array_equal(rec["step"], [0, 1, 1, 2])
        assert sorted(rec["dirs"][1:3, 0].tolist()) == [-1, 1]
        assert rec["dirs"][3, 0] == 1  # b x is in the span: LA-7
        assert {rec["cuts"][1, 0], rec["cuts"][3, 0]} == {0.3, 0.7}
        assert rec["termination"] == ref.RSQ_HIGH

    def test_a_term_in_a_slot_above_m_waits(self):
        # step 1 adds the linear term x0 (slot 2; slot 3 stays empty), step 2 a
        # pair on x1 (slots 4 and 5). At step 3 the entries 1 to 4 stand for
        # slots 1 to 4, so the term in slot 5 is not searched [FAST-4, FWD-9].
        rng = np.random.default_rng(2)
        x0 = (np.arange(80) % 2).astype(float)
        x1 = rng.uniform(size=80)
        y = 3 * x0 + 2 * np.abs(x1 - 0.5) + rng.normal(scale=0.1, size=80)
        trace = []
        rec = run_forward(
            np.column_stack([x0, x1]), y, trace=trace, max_degree=2, fast_k=0
        )
        np.testing.assert_array_equal(rec["dirs"][1], [2, 0])
        assert sorted(rec["dirs"][2:4, 1].tolist()) == [-1, 1]
        np.testing.assert_array_equal(rec["parent"][:4], [-1, 0, 0, 0])
        assert trace[2]["visited"] == trace[2]["table"] and len(trace[2]["table"]) == 4
        assert set(trace[2]["searched"]) == {0, 1, 2}

    def test_a_parent_without_a_free_covariate_gets_minus_one(self):
        rng = np.random.default_rng(3)
        x = rng.uniform(size=60)
        trace = []
        run_forward(
            x[:, None],
            np.abs(x - 0.5) + rng.normal(scale=0.05, size=60),
            trace=trace,
            max_degree=2,
            max_terms=5,
        )
        assert trace[1]["entries"][1] == (-1.0, 4)  # [FAST-5]
        assert trace[1]["entries"][2] == (-1.0, 4)

    @pytest.mark.parametrize(("fast_k", "size"), [(3, 5), (5, 7)])
    def test_at_degree_one_the_pass_ends_when_the_intercept_leaves_the_window(
        self, fast_k, size
    ):
        rng = np.random.default_rng(4)
        X = rng.uniform(size=(100, 6))
        y = np.abs(X - 0.5) @ [32.0, 16.0, 8.0, 4.0, 2.0, 1.0]
        rec = run_forward(X, y, fast_k=fast_k, thresh=0.0)
        assert rec["termination"] == ref.NO_GAIN  # [FAST-6, STOP-2]
        assert len(rec["dirs"]) == size
        np.testing.assert_array_equal(np.bincount(rec["step"])[1:], 2)
        rec = run_forward(X, y, fast_k=fast_k)
        assert rec["termination"] == ref.RSQ_CHANGE_SMALL and len(rec["dirs"]) == size

    def test_the_term_limit(self):
        X, y = noisy_data(5)
        for max_terms in (1, 2):
            rec = run_forward(X, y, max_terms=max_terms)
            assert rec["termination"] == ref.NO_ROOM and len(rec["dirs"]) == 1
            assert len(rec["rss"]) == 1 and len(rec["candidates"]["best_rss"]) == 0
        for max_terms in (3, 4):
            rec = run_forward(X, y, max_terms=max_terms)
            assert rec["termination"] == ref.TERM_LIMIT and len(rec["rss"]) == 2

    def test_small_samples_stop_on_grsq(self):
        x = np.arange(5.0)[:, None]
        rec = run_forward(x, [0.0, 1.0, 0.0, 1.0, 0.0])
        assert rec["termination"] == ref.GRSQ_NEG_INF and len(rec["dirs"]) == 1
        x = np.arange(8.0)[:, None]
        rec = run_forward(x, [1.0, -1.0] * 4, penalty=4.0)
        assert rec["termination"] == ref.GRSQ_LOW and len(rec["dirs"]) == 1

    def test_integer_weights_give_the_pass_of_repeated_rows(self):
        X, y = noisy_data(6, n=30)
        w = np.random.default_rng(6).integers(1, 4, size=30)
        weighted = run_forward(X, y, w=w, max_degree=2)
        repeated = run_forward(np.repeat(X, w, axis=0), np.repeat(y, w), max_degree=2)
        for key in ("dirs", "cuts", "parent", "step", "termination"):
            np.testing.assert_array_equal(weighted[key], repeated[key])
        np.testing.assert_allclose(weighted["rss"], repeated["rss"], rtol=1e-9)

    def test_the_pass_does_not_depend_on_the_row_order_or_the_scale_of_y(self):
        X, y = noisy_data(7)
        base = run_forward(X, y, max_degree=2)
        perm = np.random.default_rng(7).permutation(60)
        for other, scale in [
            (run_forward(X[perm], y[perm], max_degree=2), 1.0),
            (run_forward(X, 1000.0 * y, max_degree=2), 1e6),
        ]:
            for key in ("dirs", "cuts", "parent", "step", "termination"):
                np.testing.assert_array_equal(other[key], base[key])
            np.testing.assert_allclose(other["rss"], scale * base["rss"], rtol=1e-9)

    def test_the_records_are_consistent(self):
        X, y = noisy_data(8, p=3)
        rec = run_forward(X, y, max_degree=3)
        M, S = len(rec["dirs"]), len(rec["rss"]) - 1
        assert rec["dirs"].dtype == np.int8 and rec["cuts"].shape == (M, 3)
        np.testing.assert_array_equal(
            np.sort(np.concatenate([rec["kept"], rec["dropped"]])), np.arange(M)
        )
        assert np.all(np.diff(rec["rss"]) < 0)
        log = rec["candidates"]
        kinds = log["second_kind"]
        assert len(kinds) == S and set(kinds.tolist()) <= {0, 1, 2, 3}
        np.testing.assert_array_equal(
            np.isnan(log["second_knot"]), np.isin(kinds, [0, 3])
        )
        np.testing.assert_array_equal(log["second_parent"] == -1, kinds == 0)
        # up to rounding: at an exact fit a second best can compute lower
        tolerance = 1e-12 * rec["rss"][:-1]
        assert np.all(log["second_rss"] >= log["best_rss"] - tolerance)
        for k in range(1, M):  # column k is the parent's column times the new factor
            assert (
                ref.term_degree(rec["dirs"][k])
                == ref.term_degree(rec["dirs"][rec["parent"][k]]) + 1
            )

    @pytest.mark.parametrize(
        ("degree", "case", "shift"),
        [
            (2, "hinges", 2.0**36),
            (3, "hinges", 2.0**36),
            (2, "continuous", 2.0**26),
            (2, "continuous", -(2.0**36)),
            (3, "continuous", 2.0**26),
            (3, "continuous", -(2.0**36)),
            (2, "grid", (1e8, 4e6, 7e8)),
            (2, "binary", (2.0**36, 0.0)),
            (3, "near the range", (1.001, -1.001, 0.5)),
        ],
    )
    def test_an_exact_shift_of_a_covariate_keeps_the_pass(
        self, degree, case, shift, monkeypatch
    ):
        # "hinges": with auto_linpreds=False every term is a product of
        # hinges, so an exact shift of x1 by 2^36 cannot change the fit
        # (Conventions): the pass must give the same terms with the cuts
        # shifted. The other cases use auto_linpreds=True, so covariates with
        # a large mean enter products as linear factors, next to the same
        # term without them, and the new part of a column is small beside the
        # means. "continuous": every covariate shifted. "grid": values 0 to
        # 4 and means near 1e8, 4e6 and 7e8, where the pass used to add x0 x2
        # a second time, with a false RSS drop of 32 percent. "binary": a
        # shifted 0/1 covariate times x1, next to a hinge pair on x1, whose
        # sum is linear in x1. In all cases, the RSS of each step and of its
        # second best candidate must be within 1e-8 of the RSS before the
        # step [LA-5] of the exact rational RSS of the same terms, with every
        # column formed exactly. "near the range": smallest values just above
        # the range, where the float columns lose no digits, so the pass must
        # be the same with and without the exact change of basis.
        opts = {"max_degree": degree, "auto_linpreds": case != "hinges"}
        opts["max_terms"] = 11 if case == "hinges" else 13
        if case == "hinges":
            rng = np.random.default_rng(4)
            n = 40
            grid = [rng.integers(0, k, size=n) / k for k in (64, 1024, 64)]
            X = np.column_stack(grid)[:, :degree]
            shift = np.array([0.0, shift, 0.0][:degree])
        elif case == "grid":
            rng = np.random.default_rng(2)
            n = 12
            X = rng.integers(0, 5, size=(n, 3)).astype(float)
            y = X[:, 2] + X[:, 0] * X[:, 2] + 0.1 * rng.normal(size=n)
        elif case == "binary":
            rng = np.random.default_rng(7)
            n = 17
            X = np.column_stack([rng.integers(0, 2, size=n), rng.uniform(size=n)])
            coef = rng.uniform([1, 1, 1], [4, 4, 6])
            y = (
                coef[0] * X[:, 0]
                + coef[1] * np.abs(X[:, 1] - 0.5)
                + coef[2] * X[:, 0] * X[:, 1]
                + 0.1 * rng.normal(size=n)
            )
            opts["max_terms"] = 9
        elif case == "near the range":
            rng = np.random.default_rng(27)
            n = 14
            X = np.column_stack([rng.uniform(size=n) for _ in range(3)])
            X[:, 0] = (X[:, 0] - X[:, 0].min()) / np.ptp(X[:, 0])
            y = X[:, 0] * X[:, 1] + X[:, 1] * X[:, 2] + 0.1 * rng.normal(size=n)
        else:
            rng = np.random.default_rng(0)
            n = 40
            X = rng.uniform(size=(n, 3))
        if case in ("hinges", "continuous"):
            hinge = np.maximum(X[:, 0] - 0.3, 0)
            y = 2 * hinge + 3 * hinge * X[:, 1]
            if degree == 3:
                y = y + 2 * hinge * X[:, 1] * X[:, 2]
            y = y + (0.05 if case == "hinges" else 0.2) * rng.normal(size=n)
        shifted_X = X + shift
        shifted = run_forward(shifted_X, y, **opts)
        dirs = shifted["dirs"]
        if case == "continuous":
            # a linear factor is the new factor of a term of the given degree
            # whose parent is not the intercept
            assert any(
                dirs[shifted["parent"][k], j] == 0
                and ref.term_degree(dirs[k]) == degree
                for k, j in np.argwhere(dirs == 2)
            )
        if case == "near the range":
            assert ref._large_mean(shifted_X) == {0, 1}
            monkeypatch.setattr(ref, "_large_mean", lambda X: set())
            plain = run_forward(shifted_X, y, **opts)
            for key in ("dirs", "cuts", "parent", "termination"):
                np.testing.assert_array_equal(shifted[key], plain[key])
            np.testing.assert_allclose(shifted["rss"], plain["rss"], rtol=1e-12)
        if case == "hinges":
            np.testing.assert_array_equal(shifted_X - shift, X)
            base = run_forward(X, y, **opts)
            for key in ("dirs", "parent", "step", "termination"):
                np.testing.assert_array_equal(shifted[key], base[key])
            cuts = base["cuts"].copy()
            cuts[:, 1] += np.where(np.abs(base["dirs"][:, 1]) == 1, shift[1], 0.0)
            np.testing.assert_array_equal(shifted["cuts"], cuts)
            before = base["rss"][:-1]
            change = np.abs(shifted["rss"][1:] - base["rss"][1:])
            assert np.all(change <= 1e-8 * before)
        before, log = shifted["rss"][:-1], shifted["candidates"]
        for s in range(1, len(shifted["rss"])):
            terms = np.flatnonzero(shifted["step"] <= s)
            B = exact_basis(shifted_X, dirs[terms], shifted["cuts"][terms])
            exact = exact_rss(B, y, np.ones(n))
            assert abs(shifted["rss"][s] - exact) <= 1e-8 * before[s - 1]
            # the second best of the step: RSS(B + {b x}), RSS(B + {b x, h})
            # or RSS(B + {h}) [LA-2], with B the terms before the step
            kind = log["second_kind"][s - 1]
            if kind == ref.NO_KIND:
                continue
            k, j = log["second_parent"][s - 1], log["second_variable"][s - 1]
            rows = [dirs[t] for t in terms if shifted["step"][t] < s]
            cut_rows = [shifted["cuts"][t] for t in terms if shifted["step"][t] < s]
            codes = {ref.LINEAR: [2], ref.PAIR: [2, 1], ref.SINGLE: [1]}[kind]
            for code in codes:
                rows.append(dirs[k].copy())
                cut_rows.append(shifted["cuts"][k].copy())
                rows[-1][j] = code
                cut_rows[-1][j] = log["second_knot"][s - 1] if code == 1 else 0.0
            exact = exact_rss(exact_basis(shifted_X, rows, cut_rows), y, np.ones(n))
            assert abs(log["second_rss"][s - 1] - exact) <= 1e-8 * before[s - 1]

    def test_a_power_of_two_scale_of_a_covariate_keeps_the_pass(self):
        # [LA-7] the threshold is 0.01 times the product of sigma_v^2 over the
        # covariates of the parent and x; with a power of 2 every value scales
        # exactly, so the pass is the same bit for bit and the cuts scale
        X, y = noisy_data(23, n=80, p=3)
        base = run_forward(X, y, max_degree=3)
        scaled_X = X.copy()
        scaled_X[:, 0] *= 2.0**-10
        scaled = run_forward(scaled_X, y, max_degree=3)
        for key in ("dirs", "parent", "step", "rss", "termination"):
            np.testing.assert_array_equal(scaled[key], base[key])
        cuts = base["cuts"].copy()
        cuts[:, 0] *= 2.0**-10
        np.testing.assert_array_equal(scaled["cuts"], cuts)
        assert (base["dirs"][:, 0] != 0).sum() > 1

    @pytest.mark.parametrize(("ratio", "pair"), [(1.02, True), (0.98, False)])
    def test_the_kind_of_a_search_near_the_la_7_threshold(self, ratio, pair):
        # B = [1, x0] and x1 = x0 + eta z, with eta such that
        # A_w / (0.01 sigma^2(x1)) = ratio; a pair search has a linear candidate
        rng = np.random.default_rng(24)
        n, N, w = 20, 20.0, np.ones(20)
        x0, z = rng.uniform(size=n), rng.normal(size=n)
        B = np.column_stack([np.ones(n), x0])
        P_B = ref.Projector(B, w)
        rest = float(np.sum(P_B.residual(z) ** 2))
        s0, sz = np.var(x0), np.var(z)
        c = float(np.mean((x0 - x0.mean()) * (z - z.mean())))
        roots = np.roots(
            [rest - 0.01 * ratio * sz, -0.02 * ratio * c, -0.01 * ratio * s0]
        )
        eta = float(max(roots.real))
        X = np.column_stack([x0, x0 + eta * z])
        sigma2 = ref.covariate_variances(X, w, N)
        assert P_B.rss(X[:, 1]) / (0.01 * sigma2[1]) == pytest.approx(ratio, rel=1e-9)
        y = X[:, 1] + 0.1 * rng.normal(size=n)
        cands, _ = ref._parent_candidates(
            0,
            np.zeros(2, np.int8),
            X,
            y[:, None],
            w,
            B,
            P_B,
            P_B.residual(y[:, None]),
            sigma2,
            N,
            ref.weight_tol(N),
            ref.Params(),
            0.01,
        )
        linear = [c for c in cands if c.kind == ref.LINEAR and c.variable == 1]
        assert bool(linear) == pair

    def test_a_linear_parent_is_active_where_it_is_positive(self):
        # KNOT-1: the parent x0 (code 2) is active where x0 > 0; FWD-5: in the
        # search the linear candidate comes first, then the knots from the top
        rng = np.random.default_rng(25)
        n, N, w = 40, 40.0, np.ones(40)
        X = np.column_stack([rng.uniform(-1, 1, size=n), rng.uniform(size=n)])
        y = X[:, 0] * np.maximum(X[:, 1] - 0.4, 0) + 0.1 * rng.normal(size=n)
        B = np.column_stack([np.ones(n), X[:, 0]])
        P_B = ref.Projector(B, w)
        tau_N = ref.weight_tol(N)
        cands, _ = ref._parent_candidates(
            1,
            np.array([2, 0], np.int8),
            X,
            y[:, None],
            w,
            B,
            P_B,
            P_B.residual(y[:, None]),
            ref.covariate_variances(X, w, N),
            N,
            tau_N,
            ref.Params(minspan=1, endspan=1),
            0.01,
        )
        assert cands[0].kind == ref.LINEAR and cands[0].variable == 1
        active = X[:, 0] > 0
        L, E = ref.search_spans(2, 1, float(active.sum()), N, tau_N, 1, 1)
        scan = ref.distinct_knots(ref.knot_scan_unit(X[:, 1], active, L, E))
        G = np.column_stack([B, X[:, 0] * X[:, 1]])
        expected = [
            t
            for t in scan
            if ref.collinearity_ratio(X[:, 0] * np.maximum(X[:, 1] - t, 0), G, w)
            >= 0.01
        ]
        assert [c.knot for c in cands[1:]] == expected
        assert len(expected) > 3 and expected == sorted(expected, reverse=True)

    def test_the_linear_option_uses_the_smallest_x_of_all_cases(self):
        # FWD-6 with auto_linpreds=False: m is the smallest x over all n
        # cases, also where the parent is 0: here the smallest x1 (0.05) is at
        # an inactive case, and the smallest at an active case is 0.2
        X = np.column_stack([[0.1, 0.5, 0.9, 0.3, 0.7], [0.05, 0.3, 0.9, 0.5, 0.2]])
        parent = np.maximum(X[:, 0] - 0.4, 0)
        columns = [np.ones(5), parent]
        dirs = [np.zeros(2, np.int8), np.array([1, 0], np.int8)]
        cuts = [np.zeros(2), np.array([0.4, 0.0])]
        c = ref.Candidate(1, 1, ref.LINEAR, math.nan, 0.0)
        ((row, cut, column),) = ref._new_terms(c, X, columns, dirs, cuts, False)
        assert row.tolist() == [1, 1] and cut[1] == 0.05
        np.testing.assert_array_equal(column, parent * np.maximum(X[:, 1] - 0.05, 0))

    def test_an_entry_whose_covariates_have_no_candidate_gets_zero(self):
        # FAST-5: the step-1 hinges can take the constant covariate, which has
        # no candidate, so their lambda is 0 and not -1
        rng = np.random.default_rng(27)
        X = np.column_stack([rng.uniform(size=60), np.full(60, 3.0)])
        y = np.abs(X[:, 0] - 0.5) + rng.normal(scale=0.05, size=60)
        trace = []
        run_forward(X, y, trace=trace, max_degree=2, max_terms=5)
        assert trace[1]["entries"][1] == (0.0, 4)
        assert trace[1]["entries"][2] == (0.0, 4)

    @pytest.mark.parametrize(
        ("penalty", "thresh", "code", "steps"),
        [(2.899495, 0.001, 2, 1), (2.992898, 0.001, 3, 0), (4.0, 0.5, 3, 0)],
    )
    def test_the_grsq_stop_at_minus_ten(self, penalty, thresh, code, steps):
        # STOP-3 on the 8 alternating cases: GRSq' is -9.5 at step 1 with
        # penalty 2.899495 (the step is taken), and -10.5 with 2.992898 (no
        # step). It comes before STOP-4: with thresh 0.5 the code is 3, not 4.
        x = np.arange(8.0)[:, None]
        rec = run_forward(x, [1.0, -1.0] * 4, penalty=penalty, thresh=thresh)
        assert rec["termination"] == code and len(rec["rss"]) - 1 == steps

    @pytest.mark.parametrize(
        ("factor", "thresh", "code", "steps"),
        [(0.97, 0.0, 5, 1), (5.0, 0.0, 7, 10), (None, 0.05, 5, 1)],
    )
    def test_the_stop_at_an_exact_fit(self, factor, thresh, code, steps):
        # STOP-5 after the pair at 0.5: y = 2 (x - 0.5)+ + eps v with v
        # orthogonal to the step-1 model, so RSS_1 = eps^2. At 0.97 times the
        # floor 1e-10 TSS / (N - 1) the pass stops, at 5 times it goes on,
        # and with noise of sd 0.02 RSq_1 is above 1 - thresh = 0.95.
        x = np.arange(21.0) / 20
        base = 2 * np.maximum(x - 0.5, 0)
        M = np.column_stack(
            [np.ones(21), np.maximum(x - 0.5, 0), np.maximum(0.5 - x, 0)]
        )
        v = ref.Projector(M, np.ones(21)).residual(
            np.random.default_rng(0).normal(size=21)
        )
        v = v / np.linalg.norm(v)
        tss = ref.rss(np.ones((21, 1)), base[:, None], np.ones(21))
        eps = (
            0.02 * np.sqrt(21)
            if factor is None
            else math.sqrt(factor * 1e-10 * tss / 20)
        )
        rec = run_forward(
            x[:, None], base + eps * v, minspan=1, endspan=1, thresh=thresh
        )
        assert rec["termination"] == code and len(rec["rss"]) - 1 == steps

    def test_a_term_that_depends_on_the_earlier_ones_is_dropped(self):
        # FWD-11: x0 = 1e7 + k / 20 is added as a linear term; its part
        # outside the intercept is 3e-8 of its norm, so LA-4 drops it
        x0 = 1e7 + np.arange(21.0) / 20
        rec = run_forward(x0[:, None], 1 + 2 * (x0 - 1e7))
        np.testing.assert_array_equal(rec["dirs"], [[0], [2]])
        np.testing.assert_array_equal(rec["kept"], [0])
        np.testing.assert_array_equal(rec["dropped"], [1])
        # with weights 50 on the three rows where x0 moves, the w-norm ratio
        # is 1.7e-7 and the term is kept; without the weights it would be 6.9e-8
        u = np.zeros(21)
        u[:3] = [-2.0, 2.0, 1.5]
        w = np.ones(21)
        w[:3] = 50.0
        rec = run_forward((1e7 + u)[:, None], 1 + 2 * u, w=w)
        np.testing.assert_array_equal(rec["dirs"], [[0], [2]])
        assert rec["dropped"].size == 0

    def test_candidate_rss_on_a_badly_conditioned_basis(self):
        # LA-5: two knots 1e-6 apart and a linear term with offset 1e4 give a
        # condition number above 1e10; each candidate RSS stays within 1e-8 of
        # the RSS before the step, against exact rational least squares
        rng = np.random.default_rng(29)
        n = 20
        x = np.sort(rng.uniform(size=n))
        x[11] = x[10] + 1e-6
        X = np.column_stack([x, rng.uniform(size=n)])
        B = np.column_stack(
            [np.ones(n), np.maximum(x - x[10], 0), np.maximum(x - x[11], 0), x + 1e4]
        )
        w = rng.uniform(0.5, 2.0, size=n)
        y = np.abs(X[:, 1] - 0.5) + x + 0.1 * rng.normal(size=n)
        assert np.linalg.cond(B * np.sqrt(w)[:, None]) > 1e10
        N0 = ref.weight_sum(w)
        tau_N = ref.weight_tol(N0)
        N = ref.snap(N0, tau_N)
        P_B = ref.Projector(B, w)
        before = exact_rss(B, y, w)
        cands, _ = ref._parent_candidates(
            0,
            np.zeros(2, np.int8),
            X,
            y[:, None],
            w,
            B,
            P_B,
            P_B.residual(y[:, None]),
            ref.covariate_variances(X, w, N),
            N,
            tau_N,
            ref.Params(minspan=3),
            0.01,
        )
        assert len(cands) > 4
        for c in cands:
            cols = [B]
            if c.kind in (ref.PAIR, ref.LINEAR):
                cols.append(X[:, c.variable])
            if c.kind != ref.LINEAR:
                cols.append(np.maximum(X[:, c.variable] - c.knot, 0))
            exact = exact_rss(np.column_stack(cols), y, w)
            assert abs(c.rss - exact) <= 1e-8 * before

    def test_the_second_best_candidate_of_the_first_step(self):
        # CORE-3 against every legal candidate of the intercept at step 1
        X, y = noisy_data(28, n=50, p=3)
        rec = run_forward(X, y)
        n, N, w = 50, 50.0, np.ones(50)
        B = np.ones((n, 1))
        P_B = ref.Projector(B, w)
        tss = rec["rss"][0]
        cands, _ = ref._parent_candidates(
            0,
            np.zeros(3, np.int8),
            X,
            y[:, None],
            w,
            B,
            P_B,
            P_B.residual(y[:, None]),
            ref.covariate_variances(X, w, N),
            N,
            ref.weight_tol(N),
            ref.Params(),
            0.01,
        )
        check_best_and_second(rec["candidates"], 0, cands, tss, 1.01 * tss)

    def test_inputs_are_not_changed(self):
        X, y = noisy_data(9)
        w = np.ones(60)
        copies = X.copy(), y.copy(), w.copy()
        run_forward(X, y, w=w)
        for original, copy in zip((X, y, w), copies, strict=True):
            np.testing.assert_array_equal(original, copy)


# ---------------------------------------------------------------------------
# The fit [CORE-1, CORE-3, CORE-5, W-1 to W-5, EDGE-1 to EDGE-6, RESP-1 to RESP-3]

FIT_FIELDS = {
    "dirs": np.int8,
    "cuts": np.float64,
    "coef": np.float64,
    "selected": np.int64,
}
FORWARD_FIELDS = {
    "dirs": np.int8,
    "cuts": np.float64,
    "kept": np.int64,
    "dropped": np.int64,
    "parent": np.int64,
    "step": np.int64,
    "rss": np.float64,
}
LOG_FIELDS = {
    "best_rss": np.float64,
    "second_rss": np.float64,
    "second_parent": np.int64,
    "second_variable": np.int64,
    "second_knot": np.float64,
    "second_kind": np.int8,
}
PRUNING_FIELDS = {
    "removed": np.int64,
    "rss_per_size": np.float64,
    "gcv_per_size": np.float64,
    "subsets": np.bool_,
}


FIT_KEYS = {
    "dirs",
    "cuts",
    "coef",
    "selected",
    "rss",
    "gcv",
    "rsq",
    "grsq",
    "n_eff",
    "max_terms",
    "penalty",
    "forward",
    "pruning",
}
FORWARD_KEYS = {
    "dirs",
    "cuts",
    "kept",
    "dropped",
    "parent",
    "step",
    "rss",
    "termination",
    "candidates",
}
PRUNING_KEYS = {"removed", "rss_per_size", "gcv_per_size", "subsets", "selected_size"}


def check_fields(fit, p, K):
    """The keys (exactly), dtypes and shapes of CORE-3 and CORE-5."""
    M = len(fit["selected"])
    fwd, pr = fit["forward"], fit["pruning"]
    assert set(fit) == FIT_KEYS and set(fwd) == FORWARD_KEYS
    assert set(pr) == PRUNING_KEYS
    if fwd["candidates"] is not None:
        assert set(fwd["candidates"]) == set(LOG_FIELDS)
    Ma, S, Mf = len(fwd["dirs"]), len(fwd["rss"]) - 1, len(fwd["kept"])
    shapes = {
        "dirs": (M, p),
        "cuts": (M, p),
        "coef": (M, K),
        "selected": (M,),
        "forward.dirs": (Ma, p),
        "forward.cuts": (Ma, p),
        "forward.kept": (Mf,),
        "forward.dropped": (Ma - Mf,),
        "forward.parent": (Ma,),
        "forward.step": (Ma,),
        "pruning.removed": (Mf - 1,),
        "pruning.rss_per_size": (Mf,),
        "pruning.gcv_per_size": (Mf,),
        "pruning.subsets": (Mf, Mf),
    }
    for group, fields in [("", FIT_FIELDS), ("forward.", FORWARD_FIELDS)]:
        source = fit if not group else fwd
        for key, dtype in fields.items():
            assert source[key].dtype == dtype, group + key
            if group + key in shapes:
                assert source[key].shape == shapes[group + key], group + key
    for key, dtype in PRUNING_FIELDS.items():
        assert pr[key].dtype == dtype and pr[key].shape == shapes["pruning." + key]
    if fwd["candidates"] is not None:
        for key, dtype in LOG_FIELDS.items():
            log = fwd["candidates"][key]
            assert log.dtype == dtype and log.shape == (S,), key
    for key in ("rss", "gcv", "rsq", "grsq", "n_eff", "penalty"):
        assert type(fit[key]) is float, key
    assert type(fit["max_terms"]) is int and type(pr["selected_size"]) is int
    assert type(fwd["termination"]) is int and pr["selected_size"] == M


def fit(X, y, w=None, **params):
    return ref.fit_mars(X, y, w, params, record_candidates=True)


class TestFit:
    @pytest.mark.parametrize("K", [1, 2])
    def test_the_fields_of_marsfit(self, K):
        X, y = noisy_data(11, p=3)
        Y = np.column_stack([y, X[:, 1]])[:, :K]
        result = fit(X, Y, max_degree=2)
        check_fields(result, 3, K)
        assert ref.fit_mars(X, Y, None, {})["forward"]["candidates"] is None
        np.testing.assert_array_equal(result["selected"][0], 0)
        np.testing.assert_array_equal(
            result["dirs"], result["forward"]["dirs"][result["selected"]]
        )

    def test_a_single_true_knot_is_recovered(self):
        rng = np.random.default_rng(12)
        x = np.sort(rng.uniform(size=100))
        y = 2 * np.maximum(x - 0.5, 0) + rng.normal(scale=0.01, size=100)
        result = fit(x[:, None], y)
        knots = result["cuts"][np.abs(result["dirs"][:, 0]) == 1, 0]
        assert knots.size and np.min(np.abs(knots - 0.5)) < 0.05
        assert result["rsq"] > 0.99

    def test_a_pure_linear_truth(self):
        x = np.arange(21.0) / 20
        result = fit(x[:, None], 1 + 2 * x)
        np.testing.assert_array_equal(result["dirs"], [[0], [2]])
        np.testing.assert_allclose(result["coef"], [[1.0], [2.0]], rtol=1e-12)
        assert result["forward"]["termination"] == ref.RSQ_HIGH
        assert result["rsq"] == pytest.approx(1.0, abs=1e-12)

    def test_a_constant_response_is_degenerate(self):
        X, _ = noisy_data(13)
        result = fit(X, np.full(60, 3.7))
        check_fields(result, 2, 1)
        assert result["forward"]["termination"] == ref.DEGENERATE
        assert result["gcv"] == math.inf and result["rsq"] == result["grsq"] == 0.0
        assert result["rss"] == 0.0 and result["forward"]["rss"].tolist() == [0.0]
        np.testing.assert_array_equal(result["coef"], [[3.7]])
        np.testing.assert_array_equal(result["pruning"]["gcv_per_size"], [math.inf])

    def test_a_weight_sum_of_at_most_one_is_degenerate(self):
        one = fit(np.array([[0.5]]), np.array([2.0]))
        assert one["forward"]["termination"] == ref.DEGENERATE and one["rss"] == 0.0
        # N = 0.9: rss is the RSS of the intercept model, 0.3 (2/3)^2 + 0.6 (1/3)^2
        two = fit(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]), np.array([0.3, 0.6]))
        assert two["forward"]["termination"] == ref.DEGENERATE
        assert two["rss"] == pytest.approx(0.2, rel=1e-14)
        assert two["forward"]["rss"][0] == pytest.approx(0.2, rel=1e-14)
        np.testing.assert_allclose(two["coef"], [[2 / 3]], rtol=1e-14)
        assert two["gcv"] == math.inf and two["n_eff"] == pytest.approx(0.9)

    def test_small_samples_give_the_intercept_by_grsq(self):
        # [EDGE-2]: no special case, the chosen candidate has GRSq = -inf
        result = fit(np.arange(5.0)[:, None], [0.0, 1.0, 0.0, 1.0, 0.0])
        assert result["forward"]["termination"] == ref.GRSQ_NEG_INF
        assert len(result["selected"]) == 1 and math.isfinite(result["gcv"])

    def test_zero_weights_drop_their_rows(self):
        X, y = noisy_data(14)
        w = np.random.default_rng(14).integers(0, 3, size=60).astype(float)
        dropped = fit(X, y, w)
        kept = fit(X[w > 0], y[w > 0], w[w > 0])
        for key in ("dirs", "cuts", "coef", "selected"):
            np.testing.assert_array_equal(dropped[key], kept[key])
        assert dropped["rss"] == kept["rss"] and dropped["n_eff"] == w.sum()

    def test_unit_weights_are_no_weights(self):
        X, y = noisy_data(15)
        a, b = fit(X, y), fit(X, y, np.ones(60))
        for key in ("dirs", "cuts", "coef", "selected"):
            np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(
            a["pruning"]["rss_per_size"], b["pruning"]["rss_per_size"]
        )

    def test_integer_weights_give_the_fit_of_repeated_rows(self):
        X, y = noisy_data(16, n=40)
        w = np.random.default_rng(16).integers(1, 4, size=40)
        weighted = fit(X, y, w.astype(float), max_degree=2)
        repeated = fit(np.repeat(X, w, axis=0), np.repeat(y, w), max_degree=2)
        for key in ("dirs", "cuts", "selected"):
            np.testing.assert_array_equal(weighted[key], repeated[key])
        for key in ("coef", "rss", "gcv", "rsq", "grsq", "n_eff"):
            np.testing.assert_allclose(weighted[key], repeated[key], rtol=1e-9)
        np.testing.assert_array_equal(
            weighted["pruning"]["subsets"], repeated["pruning"]["subsets"]
        )

    @pytest.mark.parametrize("scale", [1000.0, 1e-9, 1e150])
    def test_the_terms_do_not_change_with_a_positive_scale_of_y(self, scale):
        X, y = noisy_data(17, p=3)
        base, scaled = fit(X, y, max_degree=2), fit(X, scale * y, max_degree=2)
        for key in ("dirs", "cuts", "selected"):
            np.testing.assert_array_equal(scaled[key], base[key])
        np.testing.assert_allclose(scaled["coef"], scale * base["coef"], rtol=1e-9)
        assert scaled["rss"] == pytest.approx(scale**2 * base["rss"], rel=1e-9)
        assert scaled["grsq"] == pytest.approx(base["grsq"], rel=1e-9)

    def test_a_dropped_term_before_kept_ones(self):
        # FWD-11 in the fit: x0 = 1e7 + (k mod 2) makes the linear term of x0
        # dependent by LA-4 (5e-8 of its norm outside the intercept), so the
        # pruning pass gets the terms of kept, and pruning index m is forward
        # index kept[m] [CORE-3]
        rng = np.random.default_rng(5)
        n = 60
        X = np.column_stack([1e7 + (np.arange(n) % 2), rng.uniform(size=n)])
        noise = rng.normal(scale=0.05, size=n)
        y = 2 * (X[:, 0] - 1e7) + 3 * np.maximum(X[:, 1] - 0.5, 0) + noise
        result = fit(X, y)
        fwd, pr = result["forward"], result["pruning"]
        assert fwd["dropped"].size and fwd["dropped"].min() < fwd["kept"].max()
        check_fields(result, 2, 1)
        size = pr["selected_size"]
        np.testing.assert_array_equal(
            result["selected"], fwd["kept"][np.flatnonzero(pr["subsets"][size - 1])]
        )
        B = ref.basis_matrix(X, result["dirs"], result["cuts"])
        np.testing.assert_allclose(
            result["coef"], ref.lstsq_coef(B, y[:, None], np.ones(n)), rtol=1e-10
        )

    @pytest.mark.parametrize(
        ("K", "weights"),
        [(1, "none"), (1, "real"), (2, "none"), (2, "real"), (1, "sum 0.9")],
    )
    def test_the_final_rss_at_a_large_mean(self, K, weights):
        # PRUNE-8 and LA-5: with a mean of 1e13 and a spread of about 1, the
        # final rss is the RSS of the returned model to 1e-8 of the exact
        # rational value, and so are RSq and the degenerate fit's rss, its TSS
        # (weights that sum to 0.9, EDGE-1)
        rng = np.random.default_rng(31)
        X = rng.uniform(size=(30, 2))
        y = [np.abs(X[:, 0] - 0.5) + rng.normal(scale=0.1, size=30) for _ in range(K)]
        Y = 1e13 + np.column_stack(y)
        w = {
            "none": np.ones(30),
            "real": rng.uniform(0.5, 2.0, 30),
            "sum 0.9": np.full(30, 0.03),
        }
        result = fit(X, Y, w[weights])
        B = ref.basis_matrix(X, result["dirs"], result["cuts"])
        exact = sum(exact_rss(B, Y[:, k], w[weights]) for k in range(K))
        tss = sum(exact_rss(np.ones((30, 1)), Y[:, k], w[weights]) for k in range(K))
        assert result["rss"] == pytest.approx(exact, rel=1e-8)
        assert result["rsq"] == pytest.approx(
            0.0 if len(B[0]) == 1 else 1 - exact / tss, abs=1e-8
        )
        assert (weights == "sum 0.9") == (
            result["forward"]["termination"] == ref.DEGENERATE
        )

    def test_the_bound_of_edge_1_and_the_degenerate_record(self):
        # N = 1 exactly is degenerate [EDGE-1]; D = 10 gives j = -3, so every
        # value on the scale of y is scaled back [EDGE-6, GCV-7]
        X, y = np.array([[0.0], [1.0]]), np.array([0.0, 10.0])
        one = fit(X, y, np.array([0.5, 0.5]))
        check_fields(one, 1, 1)
        assert one["forward"]["termination"] == ref.DEGENERATE
        for value in (
            one["rss"],
            one["forward"]["rss"][0],
            one["pruning"]["rss_per_size"][0],
        ):
            assert value == pytest.approx(25.0, rel=1e-14)
        np.testing.assert_array_equal(one["coef"], [[5.0]])
        assert one["forward"]["parent"].tolist() == [-1]
        assert one["forward"]["step"].tolist() == [0]
        assert one["pruning"]["subsets"].tolist() == [[True]]
        # N = 1 + 2^-52 is within tau_N of 1, so it is 1 [W-4]
        near = fit(X, y, np.array([0.5, 0.5 + 2.0**-52]))
        assert near["forward"]["termination"] == ref.DEGENERATE and near["n_eff"] == 1.0
        # N = 2 has no special case [EDGE-2]
        assert fit(X, y)["forward"]["termination"] != ref.DEGENERATE
        # N = 0.4 with y in the thousands: weighted mean 2250, RSS 475000
        tiny = fit(
            np.array([[0.0], [1.0], [2.0]]),
            np.array([1000.0, 2000.0, 4000.0]),
            np.array([0.1, 0.2, 0.1]),
        )
        assert tiny["forward"]["termination"] == ref.DEGENERATE
        assert tiny["rss"] == pytest.approx(475000.0, rel=1e-14)
        assert tiny["forward"]["rss"][0] == pytest.approx(475000.0, rel=1e-14)
        np.testing.assert_allclose(tiny["coef"], [[2250.0]], rtol=1e-14)

    def test_the_passes_get_the_resolved_penalty_and_the_snapped_n(self):
        # weights 1 + 1e-10: the sum 60 + 6e-9 snaps to 60 [W-4]; max_degree 2
        # gives d = 3 [GCV-4]; the pruning pass uses both [PRUNE-4]
        X, y = noisy_data(11, p=3)
        result = fit(X, y, np.full(60, 1.0 + 1e-10), max_degree=2)
        assert result["penalty"] == 3.0 and result["n_eff"] == 60.0
        pr = result["pruning"]
        tau = ref.weight_tol(60.0)
        for m in range(1, len(pr["rss_per_size"]) + 1):
            expected = ref.gcv(pr["rss_per_size"][m - 1], m, 3.0, 60.0, tau)
            assert pr["gcv_per_size"][m - 1] == expected
        # weights 1 - 1e-9 sum to 19.99999998, which snaps to 20
        X, y = noisy_data(12, n=20)
        assert fit(X, y, np.full(20, 1 - 1e-9))["n_eff"] == 20.0

    def test_a_power_of_two_changes_no_bit(self):
        # [EDGE-6]: Y is scaled to D in [1, 2) before any sum
        X, y = noisy_data(18)
        base, scaled = fit(X, y), fit(X, 2.0**-40 * y)
        np.testing.assert_array_equal(scaled["coef"], 2.0**-40 * base["coef"])
        np.testing.assert_array_equal(
            scaled["forward"]["rss"], 2.0**-80 * base["forward"]["rss"]
        )
        assert scaled["rsq"] == base["rsq"] and scaled["grsq"] == base["grsq"]
        # the candidate log is on the scale of Y too, and best_rss is rss[s]
        # [CORE-3]; here max |y| is outside [1, 2), so j is not 0
        for result in (base, scaled):
            log = result["forward"]["candidates"]
            np.testing.assert_array_equal(log["best_rss"], result["forward"]["rss"][1:])
        assert ref.y_scale_power(2.0**-40 * y) != 0
        np.testing.assert_array_equal(
            scaled["forward"]["candidates"]["second_rss"],
            2.0**-80 * base["forward"]["candidates"]["second_rss"],
        )

    def test_a_tiny_response_has_a_positive_tss(self):
        # seven 0s and one 1e-170: the fit is not degenerate, although its TSS
        # (about 1e-340) underflows to 0 when it is reported [EDGE-6]
        y = np.zeros(8)
        y[5] = 1e-170
        result = fit(np.arange(8.0)[:, None], y)
        assert result["forward"]["termination"] != ref.DEGENERATE
        assert result["forward"]["rss"][0] == 0.0
        big = fit(np.arange(8.0)[:, None], y * 2.0**600)
        assert result["forward"]["termination"] == big["forward"]["termination"]
        np.testing.assert_array_equal(result["dirs"], big["dirs"])

    def test_an_out_of_range_weight_scale_is_an_error(self):
        X = np.arange(4.0)[:, None]
        with pytest.raises(ValueError, match="scale of y or of the weights"):
            fit(X, [0.0, 1.0, 0.0, 1.0], np.full(4, 1e-310))
        # weights 8e307 keep N finite, but the scaled TSS overflows; under the
        # repository's filterwarnings = error a warning would fail this test
        with pytest.raises(ValueError, match="scale of y or of the weights"):
            fit(np.array([[0.0], [1.0]]), np.array([-1.9, 1.9]), np.full(2, 8e307))

    def test_several_responses_share_one_basis(self):
        X, y = noisy_data(19, p=3)
        Y = np.column_stack([y, np.maximum(X[:, 2] - 0.3, 0)])
        both = fit(X, Y, max_degree=2)
        assert both["coef"].shape == (len(both["selected"]), 2)
        subsets = both["pruning"]["subsets"]
        for m in range(1, len(subsets)):
            assert np.all(subsets[m] >= subsets[m - 1])  # K >= 2: nested [PRUNE-3]
        B = ref.basis_matrix(X, both["dirs"], both["cuts"])
        for k in range(2):  # one column of coefficients per response [PRUNE-8]
            np.testing.assert_allclose(
                both["coef"][:, k],
                np.linalg.lstsq(B, Y[:, k], rcond=None)[0],
                rtol=1e-9,
            )
        one_column = fit(X, y[:, None])
        flat = fit(X, y)
        for key in ("dirs", "cuts", "coef", "selected"):
            np.testing.assert_array_equal(one_column[key], flat[key])  # [RESP-2]

    def test_pmethod_none_and_nprune(self):
        X, y = noisy_data(20)
        full = fit(X, y, pmethod="none")
        np.testing.assert_array_equal(full["selected"], full["forward"]["kept"])
        first = fit(X, y, pmethod="none", nprune=3)
        np.testing.assert_array_equal(first["selected"], full["forward"]["kept"][:3])
        capped = fit(X, y, nprune=2)
        assert len(capped["selected"]) <= 2

    def test_the_rows_order_does_not_matter(self):
        X, y = noisy_data(21)
        perm = np.random.default_rng(21).permutation(60)
        a, b = fit(X, y, max_degree=2), fit(X[perm], y[perm], max_degree=2)
        for key in ("dirs", "cuts", "selected"):
            np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_allclose(a["coef"], b["coef"], rtol=1e-9)

    def test_inputs_are_not_changed(self):
        X, y = noisy_data(22)
        w = np.linspace(0.0, 2.0, 60)
        copies = X.copy(), y.copy(), w.copy()
        fit(X, y, w)
        for original, copy in zip((X, y, w), copies, strict=True):
            np.testing.assert_array_equal(original, copy)


# ---------------------------------------------------------------------------
# The whole fit against earth (EARTH_NAMES and params_from_earth are below,
# with the forward-pass self-checks)


@pytest.mark.parametrize("name", ["S01_matched_d1", "S01_defaults_d1"])
def test_the_fit_matches_earth_on_s01(name, load_fixture):
    # The harness divides the fixture's x by its standard deviation, so
    # earth's rule for the kind of a search and the pymars rule agree
    # [LA-7]. Tolerances from the plan's table.
    fixture = load_fixture(name)
    earth = fixture["result"]
    X = np.array(fixture["inputs"]["X"])
    result = ref.fit_mars(
        X,
        np.array(fixture["inputs"]["y"]),
        None,
        params_from_earth(fixture["earth_args"]),
    )
    forward = result["forward"]
    np.testing.assert_array_equal(forward["dirs"], earth["dirs"])
    np.testing.assert_array_equal(forward["cuts"], earth["cuts"])  # no linear term
    assert forward["termination"] == earth["termcond"]
    # earth's fwd_rss[m - 1] is the RSS of the first m forward terms
    sizes = [
        1 + int(np.sum(forward["step"][1:] <= s)) for s in range(len(forward["rss"]))
    ]
    np.testing.assert_allclose(
        forward["rss"], np.array(earth["fwd_rss"])[np.array(sizes) - 1], rtol=1e-8
    )
    np.testing.assert_array_equal(
        result["selected"], np.array(earth["selected_terms"]) - 1
    )
    pruning = result["pruning"]
    np.testing.assert_allclose(
        pruning["rss_per_size"], earth["rss_per_subset"], rtol=1e-8
    )
    np.testing.assert_allclose(
        pruning["gcv_per_size"], earth["gcv_per_subset"], rtol=1e-8
    )
    for m, row in enumerate(np.array(earth["prune_terms"], dtype=int)):
        assert set(np.flatnonzero(pruning["subsets"][m])) == set(row[row > 0] - 1)
    coef = np.array(earth["coef"])
    assert np.linalg.norm(result["coef"] - coef) <= 1e-6 * np.linalg.norm(coef)
    assert result["rss"] == pytest.approx(earth["rss"], rel=1e-8)
    assert result["gcv"] == pytest.approx(earth["gcv"], rel=1e-8)
    assert result["rsq"] == pytest.approx(earth["rsq"], abs=1e-8)
    assert result["grsq"] == pytest.approx(earth["grsq"], abs=1e-8)


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
        assert error <= 1e-6 * np.linalg.norm(expected[~missing]), case["label"]


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
    assert searches == 214


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
        assert value == pytest.approx(part["rss_per_subset"][m], rel=1e-8)
        gcv = ref.gcv(value, m + 1, fixture["penalty"], N, ref.weight_tol(N))
        assert gcv == pytest.approx(part["gcv_per_subset"][m], rel=1e-8)


def check_pruning(result, earth):
    """T[m], the selected terms and the RSS and GCV per size against earth's
    prune.terms, selected.terms, rss.per.subset and gcv.per.subset, with the
    plan's tolerances."""
    for m, row in enumerate(np.array(earth["prune_terms"], dtype=int)):
        np.testing.assert_array_equal(
            np.flatnonzero(result["subsets"][m]), np.sort(row[row > 0] - 1)
        )
    np.testing.assert_array_equal(
        result["selected"], np.array(earth["selected_terms"]) - 1
    )
    np.testing.assert_allclose(
        result["rss_per_size"], earth["rss_per_subset"], rtol=1e-8
    )
    np.testing.assert_allclose(
        result["gcv_per_size"], earth["gcv_per_subset"], rtol=1e-8
    )


@pytest.mark.parametrize("key", ["one_response", "several_responses"])
def test_the_pruning_pass_matches_earth_on_a_fixed_basis(load_fixture, key):
    fixture = load_fixture("components/pruning_fixed_basis")
    X = np.array(fixture["X"])
    part = fixture[key]
    B = ref.basis_matrix(
        X, np.array(part["dirs"]).astype(np.int8), np.array(part["cuts"])
    )
    Y = np.array(part["y"]).reshape(len(X), -1)
    check_pruning(prune(B, Y, np.ones(len(X)), penalty=fixture["penalty"]), part)


@pytest.mark.parametrize(
    "name",
    [
        "S01_matched_d1",
        "S02_n050_matched_d1",
        "S04_p05_n0200_matched_d2",
        "S06_matched_d3",
        "S17_matched_d1",
    ],
)
def test_the_pruning_pass_matches_earth_on_its_forward_basis(load_fixture, name):
    # earth's forward terms, pruned by the reference, give earth's pruning
    # records and final model; S17 has three responses
    fixture = load_fixture(name)
    earth, args = fixture["result"], fixture["earth_args"]
    X = np.array(fixture["inputs"]["X"])
    B = ref.basis_matrix(
        X, np.array(earth["dirs"]).astype(np.int8), np.array(earth["cuts"])
    )
    Y = np.array(fixture["inputs"]["y"]).reshape(len(X), -1)
    penalty = args.get("penalty", ref.default_penalty(args["degree"]))
    result = prune(B, Y, np.ones(len(X)), penalty=penalty)
    check_pruning(result, earth)
    assert result["rss"] == pytest.approx(earth["rss"], rel=1e-8)
    assert result["gcv"] == pytest.approx(earth["gcv"], rel=1e-8)
    assert result["rsq"] == pytest.approx(earth["rsq"], abs=1e-8)
    assert result["grsq"] == pytest.approx(earth["grsq"], abs=1e-8)


def slot_of(record, k):
    """The slot of forward term k [FWD-9]: 1 for the intercept, and 2j or
    2j + 1 for the first or second term of step j."""
    if k == 0:
        return 1
    j = int(record["step"][k])
    return 2 * j + int(k != np.flatnonzero(record["step"] == j)[0])


def same_pass(a, b):
    """Two forward records with the same terms, parents, steps and code."""
    for key in ("dirs", "cuts", "parent", "step", "termination"):
        np.testing.assert_array_equal(a[key], b[key])


class TestForwardPassRules:
    """Rules of the pass that the adversarial review of #48 found open, with
    its cases."""

    @pytest.mark.parametrize("beta", [0.0, 0.5, 2.0])
    def test_the_table_of_each_step_uses_fast_beta(self, beta):
        # FAST-2: the table of step s + 1 is queue_table of the entries after
        # step s, with the given fast_beta
        rng = np.random.default_rng(21)
        X = rng.uniform(size=(80, 3))
        y = (
            np.abs(X[:, 0] - 0.4)
            + X[:, 0] * np.maximum(X[:, 1] - 0.5, 0)
            + X[:, 2]
            + rng.normal(scale=0.05, size=80)
        )
        trace = []
        rec = run_forward(
            X, y, trace=trace, max_degree=2, fast_k=3, fast_beta=beta, thresh=0.0
        )
        for s in range(1, len(trace)):
            entries = [list(e) for e in trace[s - 1]["entries"]]
            entries += [[math.inf, 2 * s]] * int(np.sum(rec["step"] == s))
            assert trace[s]["table"] == ref.queue_table(entries, s, beta)
        # FAST-5: the lambda that each searched parent stores is the largest
        # legal reduction among its own candidates, or 0 when there is none
        w, N, Y = np.ones(80), 80.0, y[:, None]
        params = ref.Params(max_degree=2, fast_k=3, fast_beta=beta, thresh=0.0)
        sigma2 = ref.covariate_variances(X, w, N)
        for s, step in enumerate(trace):
            terms = np.flatnonzero(rec["step"] <= s)
            B = ref.basis_matrix(X, rec["dirs"][terms], rec["cuts"][terms])
            P_B = ref.Projector(B, w)
            rss_s = rec["rss"][s]
            limit = ref.max_legal(list(rec["rss"][: s + 1]))
            for k in step["searched"]:
                cands, searchable = ref._parent_candidates(
                    k,
                    rec["dirs"][k],
                    X,
                    Y,
                    w,
                    B,
                    P_B,
                    P_B.residual(Y),
                    sigma2,
                    N,
                    ref.weight_tol(N),
                    params,
                    ref.collinearity_tol(s),
                )
                gains = [
                    rss_s - c.rss
                    for c in cands
                    if ref.is_legal(c.kind, rss_s - c.rss, limit)
                ]
                expected = max(gains, default=0.0) if searchable else -1.0
                lam, kappa = step["entries"][slot_of(rec, k) - 1]
                assert kappa == 2 * (s + 1)
                assert lam == pytest.approx(expected, rel=1e-10, abs=1e-300)

    def test_the_second_best_of_step_one_is_a_brute_force_second(self):
        # CORE-3: a second best with parent 0 and variable 1, so that the two
        # fields cannot be swapped unseen
        x = np.arange(41.0) / 40
        X = np.column_stack([np.cos(7 * x), x])
        y = 2 * np.maximum(x - 0.5, 0) + 0.01 * np.sin(13 * x)
        rec = run_forward(X, y, minspan=1, endspan=1)
        log = rec["candidates"]
        w = np.ones(41)
        B = np.ones((41, 1))
        P = ref.Projector(B, w)
        cands, _ = ref._parent_candidates(
            0,
            np.zeros(2, np.int8),
            X,
            y[:, None],
            w,
            B,
            P,
            P.residual(y[:, None]),
            ref.covariate_variances(X, w, 41.0),
            41.0,
            ref.weight_tol(41.0),
            ref.Params(minspan=1, endspan=1),
            0.01,
        )
        tss = rec["rss"][0]
        legal = sorted((c for c in cands if tss - c.rss > 0), key=lambda c: c.rss)
        best, second = legal[0], legal[1]
        assert (best.variable, best.knot) == (1, 0.5)
        assert log["second_parent"][0] == second.parent == 0
        assert log["second_variable"][0] == second.variable == 1
        assert log["second_knot"][0] == second.knot
        assert log["second_kind"][0] == second.kind

    def test_every_candidate_rss_sums_over_several_responses(self):
        # RESP-1: the RSS of a candidate is summed over the responses
        rng = np.random.default_rng(22)
        X = rng.uniform(size=(40, 2))
        Y = np.column_stack([np.abs(X[:, 0] - 0.3), 5 * np.maximum(X[:, 1] - 0.6, 0)])
        Y = Y + rng.normal(scale=0.05, size=Y.shape)
        w = np.ones(40)
        B = np.ones((40, 1))
        P = ref.Projector(B, w)
        cands, _ = ref._parent_candidates(
            0,
            np.zeros(2, np.int8),
            X,
            Y,
            w,
            B,
            P,
            P.residual(Y),
            ref.covariate_variances(X, w, 40.0),
            40.0,
            ref.weight_tol(40.0),
            ref.Params(),
            0.01,
        )
        kinds = set()
        for c in cands:
            x = X[:, c.variable]
            cols = [B, x]
            if c.kind != ref.LINEAR:
                cols.append(np.maximum(x - c.knot, 0))
            expected = lstsq_rss(np.column_stack(cols), Y, w)
            assert c.rss == pytest.approx(expected, rel=1e-10)
            kinds.add(c.kind)
        assert kinds == {ref.PAIR, ref.LINEAR}

    def test_the_larger_response_decides_the_first_knot(self):
        # RESP-1, RESP-3: the responses are not scaled to a common variance
        x = np.arange(41.0) / 40
        Y = np.column_stack([np.maximum(x - 0.3, 0), 10 * np.maximum(x - 0.7, 0)])
        rec = run_forward(x[:, None], Y, minspan=1, endspan=1, max_terms=3)
        np.testing.assert_array_equal(rec["cuts"][1:, 0], [0.7, 0.7])
        one = run_forward(x[:, None], Y[:, 1], minspan=1, endspan=1, max_terms=3)
        np.testing.assert_array_equal(one["cuts"], rec["cuts"])

    def test_the_floor_uses_the_weight_sum(self):
        # W-2 in STOP-5: with weight 2 on 21 rows, RSS_1 is 1.5 times the
        # floor 1e-10 TSS / (N - 1) with N = 42, so the pass goes on, as it
        # does on the 42 repeated rows
        x = np.arange(21.0) / 20
        B1 = np.column_stack(
            [np.ones(21), np.maximum(x - 0.5, 0), np.maximum(0.5 - x, 0)]
        )
        v = np.sin(9 * x)
        v = v - B1 @ np.linalg.lstsq(B1, v, rcond=None)[0]
        y0 = 2 * np.maximum(x - 0.5, 0)
        tss0 = float(np.sum((y0 - y0.mean()) ** 2))
        eps = math.sqrt(1.5 * 1e-10 * tss0 / 41 / float(v @ v))
        y = y0 + eps * v
        opts = {"minspan": 1, "endspan": 1, "thresh": 0.0}
        weighted = run_forward(x[:, None], y, w=np.full(21, 2.0), **opts)
        repeated = run_forward(np.repeat(x, 2)[:, None], np.repeat(y, 2), **opts)
        assert len(repeated["rss"]) > 2
        same_pass(weighted, repeated)

    def test_the_automatic_minspan_uses_the_weight_of_the_cases(self):
        # SPAN-1: weight 2 on 30 cases gives N_b = 60 and L = 4; the count 30
        # would give L = 3, whose list holds the kink at 22/29
        x = np.arange(30.0) / 29
        y = 3 * np.abs(x - 22 / 29)
        weighted = run_forward(x[:, None], y, w=np.full(30, 2.0), max_terms=3)
        repeated = run_forward(np.repeat(x, 2)[:, None], np.repeat(y, 2), max_terms=3)
        same_pass(weighted, repeated)
        assert weighted["cuts"][1, 0] != 22 / 29

    def test_grsq_after_a_linear_step_counts_the_real_terms(self):
        # STOP-3: M' counts the real terms, not the slots, after a one-term step
        n = 20
        x0 = (np.arange(n) % 2).astype(float)
        x1 = (np.arange(n) * 7 % n) / (n - 1)
        noise = 0.01 * np.random.default_rng(0).normal(size=n)
        y = 3 * x0 + 2 * np.maximum(x1 - 0.5, 0) + noise
        rec = run_forward(
            np.column_stack([x0, x1]),
            y,
            penalty=9.0,
            minspan=1,
            endspan=1,
            max_terms=11,
        )
        np.testing.assert_array_equal(rec["dirs"], [[0, 0], [2, 0], [0, 1], [0, -1]])
        assert rec["termination"] == ref.RSQ_HIGH

    def test_the_linear_option_knot_is_the_smallest_x(self):
        # FWD-6 with auto_linpreds=False
        x = 2 + np.arange(21.0) / 20
        rec = run_forward(x[:, None], 1 + 2 * x, auto_linpreds=False)
        np.testing.assert_array_equal(rec["dirs"], [[0], [1]])
        np.testing.assert_array_equal(rec["cuts"], [[0.0], [2.0]])

    def test_max_legal_uses_the_last_reduction(self):
        # FWD-4: 10 Delta_s with Delta_s = RSS_(s-1) - RSS_s of the step just done
        assert ref.max_legal([10.0, 8.0, 7.9]) == pytest.approx(1.0)

    def test_the_endspan_cap_uses_all_cases(self):
        # SPAN-5: E* is capped with N, the weight of all cases (E* = 19 and no
        # knot), not with N_b (15, which would give 6)
        rng = np.random.default_rng(24)
        X = rng.uniform(size=(40, 2))
        B = np.column_stack(
            [np.ones(40), np.maximum(X[:, 0] - 0.7, 0), np.maximum(0.7 - X[:, 0], 0)]
        )
        w = np.ones(40)
        Y = (X[:, 0] * X[:, 1])[:, None]
        P = ref.Projector(B, w)
        tau_N = ref.weight_tol(40.0)
        cands, _ = ref._parent_candidates(
            1,
            np.array([1, 0], np.int8),
            X,
            Y,
            w,
            B,
            P,
            P.residual(Y),
            ref.covariate_variances(X, w, 40.0),
            40.0,
            tau_N,
            ref.Params(),
            0.01,
        )
        active = B[:, 1] > 0
        L, E = ref.search_spans(2, 1, float(active.sum()), 40.0, tau_N)
        expected = ref.knot_scan(X[:, 1], active, w, L, E, 40.0, tau_N)
        found = [c.knot for c in cands if c.variable == 1 and c.kind != ref.LINEAR]
        assert expected == [] and found == []

    def test_the_pair_rule_uses_weighted_variances(self):
        # LA-7: A_w is 0.95 of the weighted threshold, so the search on the
        # parent x0 and x1 is a single hinge, as on the repeated rows
        eta = 0.0520134012737919
        x0 = np.r_[np.zeros(10), np.ones(10)]
        x1 = np.r_[np.arange(10.0), 4.5 + eta * np.linspace(-1, 1, 10)]
        X = np.column_stack([x0, x1])
        w = np.r_[np.full(10, 3.0), np.ones(10)]
        y = 2 * x0 + 3 * x0 * x1
        opts = {
            "max_terms": 5,
            "max_degree": 2,
            "thresh": 0.0,
            "minspan": 1,
            "endspan": 1,
        }
        weighted = run_forward(X, y, w=w, **opts)
        reps = w.astype(int)
        repeated = run_forward(np.repeat(X, reps, axis=0), np.repeat(y, reps), **opts)
        np.testing.assert_array_equal(weighted["dirs"], [[0, 0], [2, 0], [2, 1]])
        same_pass(weighted, repeated)

    @pytest.mark.parametrize(
        ("eta", "code"), [(0.01622815768436774, 2), (0.015905943651091013, 1)]
    )
    def test_the_kind_of_a_search_at_the_la_7_threshold_in_the_pass(self, eta, code):
        # LA-7 through the pass, which divides sigma_v^2 by N: with x0 binary,
        # step 1 adds x0, and at step 2 A_w / (0.01 sigma^2(x1)) is 1.02 (a
        # pair search, the linear x1) or 0.98 (a hinge)
        n = 20
        x0 = (np.arange(n) % 2).astype(float)
        X = np.column_stack([x0, x0 + eta * np.sin(np.arange(n) * 1.7)])
        y = 3 * x0 + 2 * X[:, 1]
        rec = run_forward(X, y, minspan=1, endspan=1, thresh=0.0, max_terms=5)
        np.testing.assert_array_equal(rec["dirs"][:3], [[0, 0], [2, 0], [0, code]])

    def test_tenth_weights_give_the_pass_of_unit_weights(self):
        # W-4 in the pass: 100 cases of weight 0.1 (10 at each of 10 values)
        # give the pass of the 10 unit cases; without tau_N the running sums
        # of 0.1 fall below the integers and the scan loses the knot 4
        x = np.arange(10.0)
        y = np.abs(x - 4.0)
        opts = {"minspan": 1, "endspan": 1, "max_terms": 3}
        unit = run_forward(x[:, None], y, **opts)
        tenth = run_forward(
            np.repeat(x, 10)[:, None], np.repeat(y, 10), w=np.full(100, 0.1), **opts
        )
        np.testing.assert_array_equal(unit["cuts"][1:, 0], [4.0, 4.0])
        same_pass(unit, tenth)


EARTH_NAMES = {  # [API-7]
    "degree": "max_degree",
    "nk": "max_terms",
    "penalty": "penalty",
    "thresh": "thresh",
    "minspan": "minspan",
    "endspan": "endspan",
    "Adjust.endspan": "adjust_endspan",
    "Auto.linpreds": "auto_linpreds",
    "fast.k": "fast_k",
    "fast.beta": "fast_beta",
    "pmethod": "pmethod",
    "nprune": "nprune",
}


def params_from_earth(args):
    """earth's arguments as reference parameters; a span of 0 is automatic."""
    return {
        EARTH_NAMES[key]: None
        if key in ("minspan", "endspan") and value == 0
        else value
        for key, value in args.items()
    }


def forward_on_fixture(fixture, trace=None):
    """The reference forward pass on an unweighted fixture, and its basis."""
    X = np.array(fixture["inputs"]["X"], dtype=float)
    Y = np.array(fixture["inputs"]["y"], dtype=float).reshape(len(X), -1)
    n = len(X)
    w = np.ones(n)
    tss = ref.rss(np.ones((n, 1)), Y, w)
    return ref.forward_pass(
        X,
        Y,
        w,
        params_from_earth(fixture["earth_args"]),
        N=float(n),
        tau_N=ref.weight_tol(float(n)),
        tss=tss,
        record_candidates=True,
        trace=trace,
    )


def check_forward_steps(record, earth):
    """The forward record against earth's, step by step: the rows of each step
    (codes exact, knots bit for bit), the RSS after it within 1e-8 of the RSS
    before it, and the termination code. The comparison stops at the first
    near-tie: best and second-best RSS closer than 1e-7 of the RSS before the
    step (plan: Ties). Returns the number of steps compared."""
    dirs = np.array(earth["dirs"]).astype(int)
    cuts = np.array(earth["cuts"], dtype=float)
    fwd_rss = np.array(earth["fwd_rss"], dtype=float)
    log = record["candidates"]
    for s in range(1, len(record["rss"])):
        before = record["rss"][s - 1]
        if log["second_rss"][s - 1] - log["best_rss"][s - 1] < 1e-7 * before:
            return s - 1
        rows = np.flatnonzero(record["step"] == s)
        assert rows.max() < len(dirs), f"step {s}"
        np.testing.assert_array_equal(
            record["dirs"][rows], dirs[rows], err_msg=f"step {s}"
        )
        hinge = np.abs(dirs[rows]) == 1
        np.testing.assert_array_equal(
            np.where(hinge, record["cuts"][rows], 0.0), np.where(hinge, cuts[rows], 0.0)
        )
        assert abs(record["rss"][s] - fwd_rss[rows.max()]) <= 1e-8 * before, f"step {s}"
    assert len(record["dirs"]) == len(dirs)
    assert record["termination"] == earth["termcond"]
    return len(record["rss"]) - 1


FORWARD_FIXTURES = [
    "S01_defaults_d1",
    "S01_matched_d1",
    "S02_n020_defaults_d1",
    "S02_n020_matched_d1",
    "S02_n020_span_m1_e1",
    "S02_n050_defaults_d1",
    "S02_n050_matched_d1",
    "S03_matched_d1_linear",
    "S05_matched_d2",
    "S07_matched_d1",
    "S08_matched_d1",
    "S11_n03_matched_d1",
    "S11_n05_matched_d1",
    "S11_n08_defaults_d1",
    "S11_n08_matched_d1",
    "S12_x_1em8_matched_d1",
    "S17_matched_d1",
    "S20_matched_d1",
]


@pytest.mark.parametrize("name", FORWARD_FIXTURES)
def test_the_forward_pass_matches_earth_step_by_step(load_fixture, name):
    fixture = load_fixture(name)
    record, _ = forward_on_fixture(fixture)
    check_forward_steps(record, fixture["result"])


@pytest.mark.slow
def test_the_forward_pass_matches_earth_on_every_small_fixture(load_fixture):
    # every unweighted matched and defaults fixture with n <= 400, raw x left
    # out (the LA-7 departure); factor responses have no least-squares pass
    compared = 0
    for path in sorted((HARNESS.parent / "fixtures").glob("S*.json")):
        name = path.stem
        if ("matched" not in name and "defaults" not in name) or "raw" in name:
            continue
        fixture = load_fixture(name)
        inputs = fixture["inputs"]
        if inputs.get("weights") is not None or len(inputs["X"]) > 400:
            continue
        if fixture["result"].get("dirs") is None:  # earth stopped with an error
            continue
        y = np.ravel(np.array(inputs["y"], dtype=object))
        if any(isinstance(v, str) for v in y) or all(v == y[0] for v in y):
            continue  # a factor response, or a constant one (EDGE-1)
        record, _ = forward_on_fixture(fixture)
        check_forward_steps(record, fixture["result"])
        compared += 1
    assert compared >= 60


def test_the_second_best_candidate_where_the_limit_binds(load_fixture):
    # CORE-3 at step 8 of S02_n020_matched_d1, where MaxLegal = 10 Delta_s
    # makes some knots illegal: every legal candidate of the searched parents
    fixture = load_fixture("S02_n020_matched_d1")
    trace = []
    record, B = forward_on_fixture(fixture, trace)
    X = np.array(fixture["inputs"]["X"], dtype=float)
    y = np.array(fixture["inputs"]["y"], dtype=float)[:, None]
    N, w = float(len(X)), np.ones(len(X))
    earlier = np.flatnonzero(record["step"] <= 7)
    B7 = B[:, earlier]
    P_B = ref.Projector(B7, w)
    rss_s = record["rss"][7]
    limit = ref.max_legal(list(record["rss"][:8]))
    assert limit < 1.01 * rss_s  # the limit 10 Delta_s binds
    params = ref.as_params(params_from_earth(fixture["earth_args"]))
    cands = []
    for k in trace[7]["searched"]:
        found, _ = ref._parent_candidates(
            k,
            record["dirs"][k],
            X,
            y,
            w,
            B7,
            P_B,
            P_B.residual(y),
            ref.covariate_variances(X, w, N),
            N,
            ref.weight_tol(N),
            params,
            ref.collinearity_tol(7),
        )
        cands += found
    assert any(not ref.is_legal(c.kind, rss_s - c.rss, limit) for c in cands)
    check_best_and_second(record["candidates"], 7, cands, rss_s, limit)
    # FAST-5: the intercept, the one parent searched, stores its largest legal
    # reduction (0.00288), not the larger reduction of an illegal knot (0.0213)
    assert trace[7]["searched"] == [0]
    gains = [rss_s - c.rss for c in cands if ref.is_legal(c.kind, rss_s - c.rss, limit)]
    lam, kappa = trace[7]["entries"][0]
    assert kappa == 16
    assert lam == pytest.approx(max(gains), rel=1e-10)
    assert max(rss_s - c.rss for c in cands) > 5 * lam
