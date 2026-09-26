"""Tests of the R-backed arms (E-def, E-pym), which need ``Rscript`` and the
earth package on ``PATH``. Run with the main venv's pytest, same as the pure
Python tests (the R fit happens in a subprocess ``run_block`` shells out to,
not in the test process itself): ``uv run --frozen pytest
validation/sims/tests/test_learners_r.py -q``. These do not skip: if R or
earth is missing, they fail rather than pass silently. earth is used only as
a black box: this exercises the arms through ``run_block``, which calls
``Rscript`` and reads its JSON output.
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.sims import learners

# Every test here needs Rscript and earth: gate A, gate B and CI skip
# "external" (pyproject.toml); gate C and a direct pytest invocation run it.
pytestmark = pytest.mark.external


def _friedman1_data(n, seed, binary=False):
    rng = np.random.default_rng(seed)
    p = 5
    x = rng.uniform(size=(n, p))
    f = (
        10 * np.sin(np.pi * x[:, 0] * x[:, 1])
        + 20 * (x[:, 2] - 0.5) ** 2
        + 10 * x[:, 3]
        + 5 * x[:, 4]
    )
    if not binary:
        return x, f + 0.5 * rng.standard_normal(n)
    from scipy.special import expit

    mu = expit(0.3 * (f - f.mean()))
    return x, rng.binomial(1, mu).astype(float)


@pytest.mark.parametrize("arm_name", ["E-def", "E-pym"])
def test_r_arm_regression_fits_and_predicts(arm_name):
    x_train, y_train = _friedman1_data(120, seed=0)
    x_test, y_test = _friedman1_data(30, seed=1)
    job = learners.BlockJob("job1", x_train, y_train, x_test, binary=False)
    outcomes = learners.ARMS[arm_name].run_block([job])
    outcome = outcomes["job1"]
    assert outcome.ok, outcome.error
    assert outcome.predictions.shape == (30,)
    assert np.isfinite(outcome.predictions).all()
    assert outcome.n_terms is not None and outcome.n_terms > 1
    assert outcome.covariates_used is not None
    # Friedman #1's relevant covariates are x0..x4 (all 5, here); at least
    # one should be picked up by a reasonable fit.
    assert set(outcome.covariates_used) <= {0, 1, 2, 3, 4}
    assert outcome.fit_seconds is not None and outcome.fit_seconds >= 0
    assert np.corrcoef(outcome.predictions, y_test)[0, 1] > 0.5


def test_e_def_binary_returns_probabilities():
    x_train, y_train = _friedman1_data(150, seed=0, binary=True)
    x_test, _y_test = _friedman1_data(30, seed=1, binary=True)
    job = learners.BlockJob("job1", x_train, y_train, x_test, binary=True)
    outcome = learners.ARMS["E-def"].run_block([job])["job1"]
    assert outcome.ok, outcome.error
    assert (outcome.predictions >= 0).all() and (outcome.predictions <= 1).all()


def test_r_block_runs_several_jobs_in_one_process():
    jobs = []
    for i in range(3):
        x_train, y_train = _friedman1_data(60, seed=i)
        x_test, _y_test = _friedman1_data(10, seed=100 + i)
        jobs.append(
            learners.BlockJob(f"job{i}", x_train, y_train, x_test, binary=False)
        )
    outcomes = learners.ARMS["E-def"].run_block(jobs)
    assert set(outcomes) == {"job0", "job1", "job2"}
    assert all(o.ok for o in outcomes.values())


def test_r_arm_covariates_used_survives_a_single_selected_covariate():
    """Regression test: jsonlite's auto_unbox serializes a length-1 R integer
    vector as a bare JSON scalar, not a one-element array, unless wrapped in
    I() (fit_earth_block.R). A block mixing several jobs, at least one of
    which selects exactly one covariate, used to raise "TypeError: 'int'
    object is not iterable" while reading back that one job's result.
    """
    rng = np.random.default_rng(0)
    # y depends only on covariate 3: however many terms the fit selects
    # (E-pym's low-penalty settings favor more, not fewer), every one of them
    # can only involve that single covariate, so covariates_used is (3,).
    x_train = rng.uniform(size=(200, 10))
    y_train = x_train[:, 3].copy()
    x_test = rng.uniform(size=(50, 10))
    jobs = [learners.BlockJob("job0", x_train, y_train, x_test, binary=False)]
    outcomes = learners.ARMS["E-pym"].run_block(jobs)
    outcome = outcomes["job0"]
    assert outcome.ok, outcome.error
    assert outcome.covariates_used == (3,)


def test_r_arm_records_an_error_instead_of_crashing_on_bad_data():
    # A single-row training set cannot fit anything sensible in earth; this
    # should come back as a per-job error, not raise out of run_block.
    x_train = np.zeros((1, 3))
    y_train = np.zeros(1)
    x_test = np.zeros((2, 3))
    job = learners.BlockJob("bad", x_train, y_train, x_test, binary=False)
    outcome = learners.ARMS["E-def"].run_block([job])["bad"]
    assert not outcome.ok
    assert outcome.error
