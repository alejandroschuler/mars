"""Tests of validation/sims/diagnostics.py."""

from __future__ import annotations

import numpy as np
import pytest

from validation.sims import dgps, diagnostics


def test_linear_r2_share_is_one_for_an_exactly_linear_function():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(5000, 3))
    f_values = X[:, 0] + 2 * X[:, 1] - X[:, 2]
    assert diagnostics._linear_r2_share(X, f_values) == pytest.approx(1.0, abs=1e-9)


def test_linear_r2_share_is_near_zero_when_f_is_independent_of_x():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(20_000, 2))
    f_values = rng.choice([-1.0, 1.0], size=20_000)  # a coin flip, independent of X
    assert diagnostics._linear_r2_share(X, f_values) == pytest.approx(0.0, abs=0.01)


@pytest.mark.parametrize("name", ["D1", "D3", "D4", "D5"])
@pytest.mark.parametrize("level", ["lo", "hi"])
def test_sigma_formula_gives_the_target_population_r2_on_a_large_draw(name, level):
    """VALIDATION_PLAN.md / the brief: "the sigma formula gives the target
    population R^2 on a large draw." This calls diagnostics.compute_diagnostics
    and dgps.generate themselves, not a copy of the sigma formula: a review
    found that a test which recomputed sigma from sd(f) and R^2 independently
    still passed after sd_f*sqrt(r2/(1-r2)) was substituted into
    diagnostics.py, or after dgps.generate was changed to always read
    sigma_lo regardless of the requested level.
    """
    diag = diagnostics.compute_diagnostics(n_draw=200_000)
    dgp = dgps.REGISTRY[name]
    rng = np.random.default_rng(2024)
    _x, y, truth = dgps.generate(dgp, rng, 300_000, level, diag)
    var_f = truth.var(ddof=0)
    empirical_r2 = var_f / y.var(ddof=0)
    target_r2 = dgps.R_SQUARED[level]
    # n=300,000 keeps the Monte Carlo SE of this R^2 well under 0.01; 0.03 is
    # a loose multiple of that, not a re-derivation of the target itself.
    assert empirical_r2 == pytest.approx(target_r2, abs=0.03)


def test_d6_sigma_is_exactly_one():
    diag = diagnostics.compute_diagnostics(n_draw=1000)
    assert diag["D6"]["sigma"] == 1.0


def test_compute_diagnostics_small_draw_has_every_registry_key_and_is_finite():
    result = diagnostics.compute_diagnostics(n_draw=20_000)
    for name in dgps.REGISTRY:
        assert name in result
    for name, entry in result.items():
        if name in ("n_draw", "logit_0.95"):
            continue
        for key, value in entry.items():
            if isinstance(value, float):
                assert np.isfinite(value), (name, key, value)


def test_binary_diagnostics_mu_quantiles_bracket_the_target_tail_probability():
    """lambda is set from whichever of |q01| and q99 is larger, so only that
    side's percentile reaches exactly 0.95 (or 0.05); the other side is less
    extreme (VALIDATION_PLAN.md's own D3/D4 asymmetry example). The more
    extreme of the two sides should still land at about 0.95.
    """
    result = diagnostics.compute_diagnostics(n_draw=200_000)
    for name in ("D3-bin", "D4-bin"):
        entry = result[name]
        tail_extremity = max(
            entry["mu_quantiles"]["99"], 1 - entry["mu_quantiles"]["1"]
        )
        assert tail_extremity == pytest.approx(0.95, abs=0.01)
        assert entry["lambda"] > 0


def test_committed_diagnostics_matches_the_plans_stated_values():
    """VALIDATION_PLAN.md states lambda ~= 1.60 for D3, ~= 0.28 for D4, and a
    D8 covariate correlation ~= 0.58 (from the plan's own 6/pi * asin(0.3)
    calculation). The brief asks to check the committed values against these
    and report any difference: all three matched closely on this run (D3-bin
    lambda 1.6005, D4-bin lambda 0.2772, D8 correlation 0.5823); see the PR
    body's Evidence section.
    """
    committed = dgps.load_diagnostics()
    assert committed["D3-bin"]["lambda"] == pytest.approx(1.60, abs=0.05)
    assert committed["D4-bin"]["lambda"] == pytest.approx(0.28, abs=0.05)
    assert committed["D8"]["empirical_corr_mean"] == pytest.approx(0.58, abs=0.02)


@pytest.mark.parametrize(
    "name", ["D1", "D3", "D4", "D5", "D7", "D8", "D3-bin", "D4-bin"]
)
def test_compute_diagnostics_agrees_with_the_committed_diagnostics_json(name):
    """diagnostics.py must be what produced diagnostics.json, not merely
    consistent with it: FAKE_DIAGNOSTICS-style tests in test_dgps.py never
    call compute_diagnostics, and test_committed_diagnostics_matches_the_plans
    _stated_values above reads only the committed file, so a review found
    neither one would catch diagnostics.py drifting from diagnostics.json (for
    example, dropping the centering by mean_f before computing lambda).
    ``diagnostic_rng`` is a fixed seed, so a smaller n_draw here is a prefix
    of the committed 10**6-case draw, not an independent resample; the
    tolerances below are generous multiples of the resulting Monte Carlo
    noise, not exact equality.
    """
    committed = dgps.load_diagnostics()[name]
    recomputed = diagnostics.compute_diagnostics(n_draw=200_000)[name]
    for key in ("var_f", "sd_f"):
        if key in committed:
            assert recomputed[key] == pytest.approx(committed[key], rel=0.03), key
    for key in committed:
        if key.startswith("sigma_"):
            assert recomputed[key] == pytest.approx(committed[key], rel=0.03), key
    if "lambda" in committed:
        assert recomputed["lambda"] == pytest.approx(committed["lambda"], abs=0.05)
        assert recomputed["mean_f"] == pytest.approx(committed["mean_f"], rel=0.05)
