"""Tests of validation/sims/metrics.py.

Brief: "the metrics are 0 for the true function and correct on a small hand
case."
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import expit

from validation.sims import metrics


def test_excess_risk_is_zero_for_the_true_function():
    rng = np.random.default_rng(0)
    f_true = rng.uniform(-5, 5, size=1000)
    assert metrics.excess_risk(f_true, f_true) == 0.0


def test_excess_risk_hand_case():
    f_hat = np.array([1.0, 2.0, 3.0])
    f_true = np.array([0.0, 2.0, 5.0])
    # squared errors: 1, 0, 4 -> mean 5/3
    assert metrics.excess_risk(f_hat, f_true) == pytest.approx(5.0 / 3.0)


def test_log_ratio_hand_case_and_antisymmetry():
    assert metrics.log_ratio(2.0, 1.0) == pytest.approx(math.log(2.0))
    assert metrics.log_ratio(1.0, 2.0) == pytest.approx(-metrics.log_ratio(2.0, 1.0))
    assert metrics.log_ratio(1.0, 1.0) == 0.0


def test_clip_probabilities_clips_and_counts():
    mu_hat = np.array([-0.1, 0.0, 0.5, 1.0, 1.1])
    clipped, n_clipped = metrics.clip_probabilities(mu_hat, clip=1e-6)
    assert n_clipped == 4  # -0.1, 0.0, 1.0 and 1.1 all fall outside (clip, 1-clip)
    assert clipped[0] == pytest.approx(1e-6)
    assert clipped[-1] == pytest.approx(1 - 1e-6)
    assert clipped[2] == 0.5


def test_excess_log_loss_is_zero_when_perfectly_calibrated():
    rng = np.random.default_rng(0)
    mu = expit(rng.uniform(-3, 3, size=500))
    loss, n_clipped = metrics.excess_log_loss(mu, mu)
    assert loss == pytest.approx(0.0, abs=1e-10)
    assert n_clipped == 0


def test_excess_log_loss_hand_case():
    mu_true = np.full(3, 0.8)
    mu_hat = np.full(3, 0.5)
    # KL(Bernoulli(0.8) || Bernoulli(0.5))
    #   = 0.8 log(0.8/0.5) + 0.2 log(0.2/0.5)
    expected = 0.8 * math.log(0.8 / 0.5) + 0.2 * math.log(0.2 / 0.5)
    loss, n_clipped = metrics.excess_log_loss(mu_hat, mu_true)
    assert loss == pytest.approx(expected, abs=1e-10)
    assert n_clipped == 0


def test_excess_brier_score_hand_case():
    mu_true = np.full(3, 0.8)
    mu_hat = np.full(3, 0.5)
    assert metrics.excess_brier_score(mu_hat, mu_true) == pytest.approx(0.09)


def test_excess_brier_score_is_zero_for_the_true_probability():
    rng = np.random.default_rng(0)
    mu = expit(rng.uniform(-3, 3, size=500))
    assert metrics.excess_brier_score(mu, mu) == 0.0


def test_calibration_slope_is_about_one_when_well_calibrated():
    rng = np.random.default_rng(0)
    n = 20_000
    x = rng.uniform(-3, 3, size=n)
    mu = expit(x)
    y = rng.binomial(1, mu)
    slope, n_clipped = metrics.calibration_slope(mu, y)
    assert n_clipped == 0
    assert slope == pytest.approx(1.0, abs=0.15)


def test_calibration_slope_departs_from_one_when_overconfident():
    # mu_hat exaggerates the true logit by 2x, so regressing y on
    # logit(mu_hat) recovers a slope of about 1/2.
    rng = np.random.default_rng(0)
    n = 20_000
    x = rng.uniform(-3, 3, size=n)
    mu_true = expit(x)
    y = rng.binomial(1, mu_true)
    mu_hat_overconfident = expit(2 * x)
    slope, _ = metrics.calibration_slope(mu_hat_overconfident, y)
    assert slope == pytest.approx(0.5, abs=0.1)


def test_calibration_slope_nan_when_one_class_only():
    mu = np.full(10, 0.5)
    y = np.zeros(10)
    slope, _ = metrics.calibration_slope(mu, y)
    assert math.isnan(slope)


@pytest.mark.parametrize(
    ("used", "relevant", "n_irrelevant", "any_irrelevant"),
    [
        ((0, 1, 2), (0, 1, 2), 0, False),
        ((0, 1, 5), (0, 1, 2), 1, True),
        ((5, 6), (0, 1, 2), 2, True),
        ((), (0, 1, 2), 0, False),
    ],
)
def test_irrelevant_covariates_used(used, relevant, n_irrelevant, any_irrelevant):
    assert metrics.irrelevant_covariates_used(used, relevant) == n_irrelevant
    assert metrics.uses_any_irrelevant_covariate(used, relevant) == any_irrelevant
