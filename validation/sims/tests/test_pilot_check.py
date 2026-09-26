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


def test_difference_n_sim_zero_gap_is_infinite():
    sizing = pc.difference_n_sim(g_bar_pilot=0.0, s_pilot=1.0, n_pilot=50)
    assert math.isinf(sizing.n_sim)


def test_difference_n_sim_pilot_too_small_when_z_below_threshold():
    # g_bar_pilot barely above 0, well within 1.96 standard errors of 0:
    # se = 1.0 / sqrt(100) = 0.1; g_bar_pilot = 0.05 -> z = 0.5, |z| <= 1.96.
    sizing = pc.difference_n_sim(g_bar_pilot=0.05, s_pilot=1.0, n_pilot=100)
    assert abs(sizing.z) < 1.96
    assert math.isinf(sizing.n_sim_safe)
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
