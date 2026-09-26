"""Tests of the legacy (mars-earth 1.0.4) arms: P-cur, P-ear, EarthClassifier
and GLMEarth. These need the venv ``.venv-legacy`` (``validation/legacy/
make_venv.sh``); run with the main venv's pytest, same as the pure Python
tests (the legacy fit happens in a ``.venv-legacy`` subprocess ``run_block``
shells out to, not in the test process itself): ``uv run --frozen pytest
validation/sims/tests/test_learners_legacy.py -q``. These do not skip: if
``.venv-legacy`` is missing, they fail rather than pass silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.sims import learners

# Every test here needs .venv-legacy: gate A, gate B and CI skip "external"
# (pyproject.toml); gate C and a direct pytest invocation run it.
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


@pytest.mark.parametrize("arm_name", ["P-cur", "P-ear"])
def test_legacy_regression_arm_fits_and_predicts(arm_name):
    x_train, y_train = _friedman1_data(80, seed=0)
    x_test, y_test = _friedman1_data(20, seed=1)
    job = learners.BlockJob("job1", x_train, y_train, x_test, binary=False)
    outcome = learners.ARMS[arm_name].run_block([job])["job1"]
    assert outcome.ok, outcome.error
    assert outcome.predictions.shape == (20,)
    assert np.isfinite(outcome.predictions).all()
    assert outcome.n_terms is not None and outcome.n_terms > 1
    assert outcome.covariates_used is not None
    assert np.corrcoef(outcome.predictions, y_test)[0, 1] > 0.3


@pytest.mark.parametrize("arm_name", ["EarthClassifier", "GLMEarth"])
def test_legacy_classifier_returns_probabilities_not_hard_labels(arm_name):
    x_train, y_train = _friedman1_data(100, seed=0, binary=True)
    x_test, _y_test = _friedman1_data(20, seed=1, binary=True)
    job = learners.BlockJob("job1", x_train, y_train, x_test, binary=True)
    outcome = learners.ARMS[arm_name].run_block([job])["job1"]
    assert outcome.ok, outcome.error
    predictions = outcome.predictions
    assert (predictions >= 0).all() and (predictions <= 1).all()
    # A continuous probability, not GLMEarth.predict's own 0/1 threshold
    # (VALIDATION_PLAN.md finding F9): some values must land strictly inside.
    assert np.any((predictions > 0.01) & (predictions < 0.99))


def test_legacy_block_runs_several_jobs_in_one_process():
    jobs = []
    for i in range(3):
        x_train, y_train = _friedman1_data(50, seed=i)
        x_test, _y_test = _friedman1_data(10, seed=100 + i)
        jobs.append(
            learners.BlockJob(f"job{i}", x_train, y_train, x_test, binary=False)
        )
    outcomes = learners.ARMS["P-cur"].run_block(jobs)
    assert set(outcomes) == {"job0", "job1", "job2"}
    assert all(o.ok for o in outcomes.values())


def test_legacy_arm_records_an_error_instead_of_crashing_on_bad_data():
    # A test set with fewer covariate columns than training: legacy_worker.py
    # selects the training columns (x1..xp) out of the test CSV too, which
    # raises a KeyError here instead of silently predicting on a subset.
    x_train, y_train = _friedman1_data(80, seed=0)
    x_test = np.zeros((2, x_train.shape[1] - 2))
    job = learners.BlockJob("bad", x_train, y_train, x_test, binary=False)
    outcome = learners.ARMS["P-cur"].run_block([job])["bad"]
    assert not outcome.ok
    assert outcome.error
