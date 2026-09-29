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


def test_mars_degree_is_2_for_every_simulation_arm():
    """VALIDATION_PLAN.md, "Behavior target": "Every simulation arm sets 2."
    A review tried degree 1 for every MARS arm and found no test failed.
    """
    assert learners.MARS_DEGREE == 2
    for name in ("P-cur", "P-ear", "EarthClassifier", "GLMEarth"):
        assert learners.ARMS[name].config["kwargs"]["max_degree"] == 2, name
    for name in ("E-def", "E-pym"):
        assert learners.ARMS[name].config["args"]["degree"] == 2, name


@pytest.mark.parametrize(
    ("name", "expected_config"),
    [
        ("E-def", {"args": {"degree": 2}}),
        (
            "E-pym",
            {
                "args": {
                    "degree": 2,
                    "penalty": 6,
                    "thresh": 0,
                    "minspan": 1,
                    "endspan": 1,
                    "fast.k": 0,
                    "Adjust.endspan": 1,
                }
            },
        ),
        ("P-cur", {"class_name": "Earth", "kwargs": {"max_degree": 2}}),
        (
            "P-ear",
            {
                "class_name": "Earth",
                "kwargs": {
                    "max_degree": 2,
                    "penalty": 1.5,
                    "minspan_alpha": 0.05,
                    "endspan_alpha": 0.05,
                },
            },
        ),
        (
            "EarthClassifier",
            {"class_name": "EarthClassifier", "kwargs": {"max_degree": 2}},
        ),
        ("GLMEarth", {"class_name": "GLMEarth", "kwargs": {"max_degree": 2}}),
    ],
)
def test_arm_config_matches_the_plans_learners_and_settings_table(
    name, expected_config
):
    """VALIDATION_PLAN.md, "Learners and settings". A review changed one
    setting at a time (E-pym's penalty, thresh, "fast.k", "Adjust.endspan";
    P-ear's penalty and minspan_alpha) in a scratch copy and found every test
    still passed; pinning the whole config dict catches any of them.
    """
    assert learners.ARMS[name].config == expected_config


@pytest.mark.parametrize(
    ("name", "substrings"),
    [
        ("OLS", ["LinearRegression"]),
    ],
)
def test_inline_arm_source_contains_its_plan_settings(name, substrings):
    """The settings for this arm are literal keyword arguments in its
    fit_predict source, not a separate config dict. (HGB and LogReg get a
    behavioral check instead, below and in test_logreg_is_unpenalized_and_
    predicts_probabilities: a review found that a source-text substring check
    for HGB's early_stopping=True still passed after the arm was changed to
    early_stopping="auto", because that string was still present, in a
    comment.)
    """
    source = learners.ARMS[name].source_text
    for substring in substrings:
        assert substring in source, (name, substring)


def test_hgb_fit_predict_uses_the_plans_exact_settings(monkeypatch):
    """Behavioral, not a source-text substring: records the keyword arguments
    the arm's own code actually passes to HistGradientBoosting{Regressor,
    Classifier}, by subclassing the real class (so .fit/.predict/.predict_proba
    keep their real behavior) and patching it in where _hgb_fit_predict's own
    local ``from sklearn.ensemble import ...`` will find it.
    """
    import sklearn.ensemble

    captured: dict = {}

    def _recording_subclass(real_cls):
        class _Recorder(real_cls):
            def __init__(self, **kwargs):
                captured.update(kwargs)
                super().__init__(**kwargs)

        return _Recorder

    monkeypatch.setattr(
        sklearn.ensemble,
        "HistGradientBoostingRegressor",
        _recording_subclass(sklearn.ensemble.HistGradientBoostingRegressor),
    )
    monkeypatch.setattr(
        sklearn.ensemble,
        "HistGradientBoostingClassifier",
        _recording_subclass(sklearn.ensemble.HistGradientBoostingClassifier),
    )
    x_train, y_train = _linear_data(seed=0)
    x_test, _y_test = _linear_data(seed=1)
    outcome = learners.ARMS["HGB"].fit_predict(x_train, y_train, x_test, False)
    assert outcome.ok
    assert captured == {
        "learning_rate": 0.1,
        "max_iter": 1000,
        "early_stopping": True,
        "random_state": 0,
    }


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
    """Compares the arm's own output (not a model the test fits on the side)
    against a directly-fit unpenalized reference, and separately shows a
    penalized fit would have given different predictions. A review's repro:
    the previous version of this test fit two reference models itself and
    never checked outcome.predictions against either, so replacing C=np.inf
    with scikit-learn's default (C=1, penalized) in the arm passed anyway.
    """
    from sklearn.linear_model import LogisticRegression

    x_train, y_train = _linear_data(seed=0, binary=True)
    x_test, _y_test = _linear_data(seed=1, binary=True)
    outcome = learners.ARMS["LogReg"].fit_predict(x_train, y_train, x_test, True)
    assert outcome.ok
    assert (outcome.predictions >= 0).all() and (outcome.predictions <= 1).all()

    unpenalized = LogisticRegression(C=np.inf).fit(x_train, y_train)
    reference_predictions = unpenalized.predict_proba(x_test)[:, 1]
    assert np.allclose(outcome.predictions, reference_predictions, atol=1e-8)

    penalized = LogisticRegression().fit(
        x_train, y_train
    )  # scikit-learn's default, C=1
    penalized_predictions = penalized.predict_proba(x_test)[:, 1]
    assert not np.allclose(outcome.predictions, penalized_predictions, atol=1e-6)


def test_p_fix_raises_not_implemented_today():
    # pymars.EarthClassifier exists since T14, but the fast core lacks Fast
    # MARS (fast_k=20, the default) until T11 stage 3 (#13).
    x_train, y_train = _linear_data(seed=0, binary=True)
    x_test, _y_test = _linear_data(seed=1, binary=True)
    with pytest.raises(NotImplementedError, match="T11 stage"):
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
