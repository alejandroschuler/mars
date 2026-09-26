"""Tests of validation/sims/pilot_check.py, on hand cases."""

from __future__ import annotations

import math

import pytest

from validation.sims import pilot_check as pc


def test_from_log_ratios_hand_case():
    g_bar, s_p, n_pilot = pc.from_log_ratios([1.0, 2.0, 3.0])
    assert g_bar == pytest.approx(2.0)
    assert s_p == pytest.approx(1.0)  # sample sd (ddof=1) of [1, 2, 3]
    assert n_pilot == 3


def test_from_log_ratios_needs_at_least_two_values():
    with pytest.raises(ValueError, match="at least 2"):
        pc.from_log_ratios([1.0])


def test_from_log_ratios_rejects_nan_or_inf():
    with pytest.raises(ValueError, match="1 of 3"):
        pc.from_log_ratios([1.0, float("nan"), 2.0])
    with pytest.raises(ValueError, match="1 of 3"):
        pc.from_log_ratios([1.0, float("inf"), 2.0])


def test_pilot_z_hand_case():
    # se = 2.0 / sqrt(4) = 1.0
    assert pc.pilot_z(1.0, 2.0, 4) == pytest.approx(1.0)


def test_pilot_z_zero_gap_is_zero():
    assert pc.pilot_z(0.0, 2.0, 4) == 0.0


def test_difference_n_sim_hand_case():
    # n_sim = (5 * 2 / 1)^2 = 100
    sizing = pc.difference_n_sim(g_bar_pilot=1.0, s_pilot=2.0, n_pilot=100, k=5)
    assert sizing.n_sim == pytest.approx(100.0)
    assert sizing.z == pytest.approx(1.0 / (2.0 / 10.0))  # 5.0
    # lower_bound = max(0, 1 - 1.96*2/10) = 1 - 0.392 = 0.608
    # n_sim_safe = (5*2/0.608)^2
    expected_safe = (5 * 2.0 / 0.608) ** 2
    assert sizing.n_sim_safe == pytest.approx(expected_safe, rel=1e-9)
    assert not sizing.pilot_too_small


def test_difference_n_sim_default_k_is_5():
    with_default = pc.difference_n_sim(g_bar_pilot=1.0, s_pilot=2.0, n_pilot=100)
    with_5 = pc.difference_n_sim(g_bar_pilot=1.0, s_pilot=2.0, n_pilot=100, k=5)
    with_3 = pc.difference_n_sim(g_bar_pilot=1.0, s_pilot=2.0, n_pilot=100, k=3)
    assert with_default.n_sim == pytest.approx(with_5.n_sim)
    assert with_default.k == 5
    assert with_default.n_sim != pytest.approx(with_3.n_sim)


def test_difference_n_sim_zero_gap_is_infinite():
    sizing = pc.difference_n_sim(g_bar_pilot=0.0, s_pilot=1.0, n_pilot=50)
    assert math.isinf(sizing.n_sim)


def test_difference_n_sim_pilot_too_small_when_z_below_threshold():
    # se = 1.0 / sqrt(100) = 0.1; g_bar_pilot = 0.05 -> z = 0.5, well below 2.
    sizing = pc.difference_n_sim(g_bar_pilot=0.05, s_pilot=1.0, n_pilot=100)
    assert abs(sizing.z) < 2.0
    assert math.isinf(sizing.n_sim_safe)
    assert sizing.pilot_too_small


def test_pilot_too_small_flags_z_between_1_96_and_2():
    """A review's repro: n_sim_safe's own lower-bound guard uses the exact
    constant 1.96, so at |z| = 1.99 it is barely finite (a large but real
    number), which used to make ``pilot_too_small`` false even though the
    plan's own guidance ("below about 2") says this pilot should be flagged.
    """
    n_pilot = 100
    s_pilot = 1.0
    z_target = 1.99
    g_bar_pilot = z_target * s_pilot / math.sqrt(n_pilot)
    sizing = pc.difference_n_sim(g_bar_pilot, s_pilot, n_pilot)
    assert sizing.z == pytest.approx(z_target)
    assert math.isfinite(sizing.n_sim_safe)
    assert sizing.n_sim_safe > 1e6  # "barely finite": a huge, impractical number
    assert sizing.pilot_too_small


def test_difference_n_sim_rejects_negative_s_pilot():
    with pytest.raises(ValueError, match="non-negative"):
        pc.difference_n_sim(g_bar_pilot=1.0, s_pilot=-1.0, n_pilot=10)


def test_difference_n_sim_degenerate_spread_gives_nan_not_zero():
    """A review's repro: 100 identical g_i give s_p at or near machine
    epsilon, which used to make n_sim_safe collapse to about 0 (looking like
    "no more repetitions needed") instead of being reported as unsizeable.
    """
    sizing = pc.difference_n_sim(g_bar_pilot=0.1, s_pilot=0.0, n_pilot=100)
    assert sizing.degenerate
    assert math.isnan(sizing.n_sim)
    assert math.isnan(sizing.n_sim_safe)
    assert math.isnan(sizing.z)
    assert sizing.pilot_too_small


def test_equivalence_n_sim_hand_case():
    margin = pc.EQUIVALENCE_MARGIN
    sizing = pc.equivalence_n_sim(g_bar_pilot=0.0, s_pilot=0.15)
    expected_n_sim = (2 * 3 * 0.15 / margin) ** 2
    assert sizing.n_sim == pytest.approx(expected_n_sim, rel=1e-9)
    assert sizing.feasible
    # With g_bar_pilot = 0, n_sim_at_pilot_gap = (3 * 0.15 / margin)^2
    assert sizing.n_sim_at_pilot_gap == pytest.approx(
        (3 * 0.15 / margin) ** 2, rel=1e-9
    )


def test_equivalence_n_sim_matches_the_plans_worked_examples():
    # VALIDATION_PLAN.md: s_p = 0.15 needs about 340 repetitions, s_p = 0.3
    # needs about 1,360 (quadrupling when s_p doubles).
    sizing_015 = pc.equivalence_n_sim(g_bar_pilot=0.0, s_pilot=0.15)
    sizing_030 = pc.equivalence_n_sim(g_bar_pilot=0.0, s_pilot=0.30)
    assert sizing_015.n_sim == pytest.approx(340, abs=5)
    assert sizing_030.n_sim == pytest.approx(1360, abs=20)
    assert sizing_030.n_sim == pytest.approx(4 * sizing_015.n_sim, rel=1e-9)


def test_equivalence_n_sim_infeasible_once_pilot_gap_reaches_margin():
    margin = pc.EQUIVALENCE_MARGIN
    sizing = pc.equivalence_n_sim(g_bar_pilot=margin, s_pilot=0.1)
    assert not sizing.feasible
    assert math.isinf(sizing.n_sim_at_pilot_gap)
    # The headline planning number is still finite; only the pilot-gap-aware
    # figure and the feasibility flag report the problem.
    assert math.isfinite(sizing.n_sim)


def test_equivalence_n_sim_feasible_just_below_margin():
    margin = pc.EQUIVALENCE_MARGIN
    sizing = pc.equivalence_n_sim(g_bar_pilot=margin * 0.99, s_pilot=0.1)
    assert sizing.feasible
    assert math.isfinite(sizing.n_sim_at_pilot_gap)


def test_equivalence_n_sim_rejects_negative_s_pilot():
    with pytest.raises(ValueError, match="non-negative"):
        pc.equivalence_n_sim(g_bar_pilot=0.0, s_pilot=-0.1)


def test_equivalence_n_sim_degenerate_spread_gives_nan():
    sizing = pc.equivalence_n_sim(g_bar_pilot=0.01, s_pilot=1e-15)
    assert sizing.degenerate
    assert math.isnan(sizing.n_sim)
    assert math.isnan(sizing.n_sim_at_pilot_gap)


def test_equivalence_n_sim_flags_a_gap_above_half_the_margin():
    """A review's repro: at s_p = 0.15, the headline n_sim (340, assuming a
    gap of at most margin/2) understates what n_sim_at_pilot_gap says is
    needed once the observed gap passes margin/2 (1,361 at 0.75*margin,
    8,507 at 0.9*margin).
    """
    margin = pc.EQUIVALENCE_MARGIN
    below_half = pc.equivalence_n_sim(g_bar_pilot=margin * 0.4, s_pilot=0.15)
    above_half = pc.equivalence_n_sim(g_bar_pilot=margin * 0.75, s_pilot=0.15)
    assert not below_half.gap_exceeds_half_margin
    assert above_half.gap_exceeds_half_margin
    assert above_half.feasible
    assert above_half.n_sim_at_pilot_gap > above_half.n_sim
