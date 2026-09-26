"""Tests of validation/sims/learners.py's inline arms and registry (pure
Python: no R and no .venv-legacy). The R- and legacy-venv-backed arms are
tested separately in test_learners_r.py and test_learners_legacy.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.sims import learners


def _linear_data(n=400, p=5, seed=0, binary=False):
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(n, p))
    raw = x[:, 0] + 2 * x[:, 1] - x[:, 2]
    if not binary:
        return x, raw + 0.1 * rng.standard_normal(n)
    from scipy.special import expit

    mu = expit(raw)
    return x, rng.binomial(1, mu).astype(float)


def test_arms_for_matches_the_plans_arm_lists():
    assert learners.arms_for(False) == [
        "E-def",
        "E-pym",
        "P-cur",
        "P-ear",
        "P-fix",
        "OLS",
        "HGB",
    ]
    assert learners.arms_for(True) == [
        "E-def",
        "EarthClassifier",
        "GLMEarth",
        "P-fix",
        "LogReg",
        "HGB",
    ]


def test_every_arm_has_a_nonempty_source_text():
    for name, arm in learners.ARMS.items():
        assert len(arm.source_text) > 0, name


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (3, (3,)),  # jsonlite's auto_unbox: a length-1 vector as a bare scalar
        ([], ()),
        ([1, 2, 3], (1, 2, 3)),
    ],
)
def test_to_covariates_tuple(value, expected):
    assert learners.to_covariates_tuple(value) == expected


def test_ols_fit_predict_recovers_a_linear_function():
    x_train, y_train = _linear_data(seed=0)
    x_test, y_test = _linear_data(seed=1)
    outcome = learners.ARMS["OLS"].fit_predict(x_train, y_train, x_test, False)
    assert outcome.ok
    assert outcome.n_terms is None
    assert outcome.covariates_used is None
    assert outcome.fit_seconds is not None and outcome.fit_seconds >= 0
    assert np.corrcoef(outcome.predictions, y_test)[0, 1] > 0.95


def test_hgb_regression_records_n_iter_and_predicts_reasonably():
    x_train, y_train = _linear_data(seed=0)
    x_test, y_test = _linear_data(seed=1)
    outcome = learners.ARMS["HGB"].fit_predict(x_train, y_train, x_test, False)
    assert outcome.ok
    assert 0 < outcome.extra["n_iter_"] <= 1000
    assert np.corrcoef(outcome.predictions, y_test)[0, 1] > 0.8


def test_hgb_classifier_predicts_probabilities():
    x_train, y_train = _linear_data(seed=0, binary=True)
    x_test, _y_test = _linear_data(seed=1, binary=True)
    outcome = learners.ARMS["HGB"].fit_predict(x_train, y_train, x_test, True)
    assert outcome.ok
    assert (outcome.predictions >= 0).all() and (outcome.predictions <= 1).all()


def test_logreg_is_unpenalized_and_predicts_probabilities():
    from sklearn.linear_model import LogisticRegression

    x_train, y_train = _linear_data(seed=0, binary=True)
    x_test, _y_test = _linear_data(seed=1, binary=True)
    outcome = learners.ARMS["LogReg"].fit_predict(x_train, y_train, x_test, True)
    assert outcome.ok
    assert (outcome.predictions >= 0).all() and (outcome.predictions <= 1).all()
    unpenalized = LogisticRegression(C=np.inf).fit(x_train, y_train)
    penalized = LogisticRegression().fit(x_train, y_train)  # default C=1
    # An unpenalized fit's coefficients are at least as large in magnitude as
    # a penalized fit's (F9: the legacy code's default penalty shrinks by
    # about 20%), which is what distinguishes the two in this arm.
    assert np.abs(unpenalized.coef_).sum() >= np.abs(penalized.coef_).sum()


def test_p_fix_raises_not_implemented_today():
    x_train, y_train = _linear_data(seed=0)
    x_test, _y_test = _linear_data(seed=1)
    with pytest.raises(NotImplementedError, match="EarthRegressor"):
        learners.ARMS["P-fix"].fit_predict(x_train, y_train, x_test, False)
    with pytest.raises(NotImplementedError, match="EarthClassifier"):
        learners.ARMS["P-fix"].fit_predict(x_train, y_train, x_test, True)


class _StandInEarthRegressor:
    """A stand-in for the pymars.EarthRegressor the arm will use once T13
    lands, exercising P-fix's own extraction logic (dirs_ -> n_terms /
    covariates_used) against a known, fixed basis.
    """

    def __init__(self, max_degree: int) -> None:
        self.max_degree = max_degree

    def fit(self, x, y):
        self.dirs_ = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0]], dtype=np.int8)
        self._coef = np.array([y.mean(), 0.0, 0.0])
        return self

    def predict(self, x):
        return np.full(len(x), self._coef[0])


def test_p_fix_uses_dirs_for_n_terms_and_covariates_once_available(monkeypatch):
    import pymars

    monkeypatch.setattr(pymars, "EarthRegressor", _StandInEarthRegressor, raising=False)
    x_train, y_train = _linear_data(seed=0)
    x_test, _y_test = _linear_data(seed=1)
    outcome = learners.ARMS["P-fix"].fit_predict(x_train, y_train, x_test, False)
    assert outcome.ok
    assert outcome.n_terms == 3
    assert outcome.covariates_used == (0, 1)  # dirs_ columns 0 and 1 are nonzero
