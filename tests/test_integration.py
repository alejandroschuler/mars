"""Integration tests: the estimators inside scikit-learn's meta-tools
(VALIDATION_PLAN.md, "scikit-learn compatibility"; docs/algorithm.md: API-2 to
API-4, ERR-2, GLM-6, W-6).

Every test runs `EarthRegressor` and `EarthClassifier`, at the defaults and at
``max_degree=2``, where the tool takes both. The fits use the fast core, since
GridSearchCV with ``n_jobs=2`` runs in other processes. The unit tests of the
estimators (test_estimators.py, test_glm.py) and scikit-learn's own checks
(test_sklearn_checks.py) pin the estimators alone; these tests pin what
changes when a scikit-learn tool wraps, clones, splits or routes to them.
"""

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn import config_context
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import TransformedTargetRegressor, make_column_transformer
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import pymars
from pymars import EarthClassifier, EarthRegressor

N = 90
_rng = np.random.default_rng(17)
X = _rng.uniform(-1.0, 1.0, size=(N, 3))
_signal = 2.0 * np.maximum(X[:, 0] - 0.1, 0.0) + X[:, 1] * X[:, 2] + X[:, 2]
y_reg = _signal + 0.2 * _rng.normal(size=N)
y_bin = (_signal + 1.0 * _rng.logistic(size=N) > 0.4).astype(int)
y_tri = np.digitize(_signal + 0.6 * _rng.normal(size=N), [0.0, 0.8])
ALPHA = 0.05  # keeps the logistic refit finite on 60 training rows
w_int = _rng.integers(1, 4, size=N)

CONFIGS = [
    pytest.param(EarthRegressor, y_reg, {}, id="reg-default"),
    pytest.param(EarthRegressor, y_reg, {"max_degree": 2}, id="reg-degree2"),
    pytest.param(EarthClassifier, y_bin, {"glm_alpha": ALPHA}, id="clf-default"),
    pytest.param(
        EarthClassifier, y_bin, {"max_degree": 2, "glm_alpha": ALPHA}, id="clf-degree2"
    ),
]


def _predictions(est, X):
    """Every prediction method that the estimator has, as arrays."""
    names = ("predict", "predict_proba", "decision_function")
    return {m: getattr(est, m)(X) for m in names if hasattr(est, m)}


def _assert_same_predictions(a, b, X):
    pa, pb = _predictions(a, X), _predictions(b, X)
    assert pa.keys() == pb.keys()
    for name in pa:
        assert_array_equal(pa[name], pb[name], err_msg=name)


@pytest.mark.parametrize(("cls", "y", "params"), CONFIGS)
def test_pipeline_with_one_hot_encoder_on_a_data_frame(cls, y, params):
    """ERR-2: the recipe of the error message, a ColumnTransformer with
    OneHotEncoder(drop="first") and a passthrough remainder, fits on a frame
    with numeric, string and categorical columns and predicts as the estimator
    does on the encoded matrix. Without the encoder, the same frame raises
    the ValueError that names OneHotEncoder."""
    frame = pd.DataFrame(
        {
            "a": X[:, 0],
            "b": X[:, 1],
            "color": np.where(X[:, 2] > 0, "red", "blue"),
            "size": pd.Categorical(np.digitize(X[:, 2], [-0.3, 0.3]), ordered=False),
        }
    )
    encoder = make_column_transformer(
        (OneHotEncoder(drop="first", sparse_output=False), ["color", "size"]),
        remainder="passthrough",
    )
    pipe = Pipeline([("encode", encoder), ("earth", cls(**params))]).fit(frame, y)
    encoded = encoder.transform(frame)
    assert encoded.shape == (N, 5)  # 1 dummy for color, 2 for size, a and b
    direct = cls(**params).fit(encoded, y)
    assert_array_equal(pipe.predict(frame), direct.predict(encoded))
    assert pipe.score(frame, y) == direct.score(encoded, y)
    with pytest.raises(ValueError, match="OneHotEncoder"):
        cls(**params).fit(frame[["a", "color"]], y)
    with pytest.raises(ValueError, match="OneHotEncoder"):
        cls(**params).fit(frame[["a", "size"]], y)


@pytest.mark.parametrize("cls", [EarthRegressor, EarthClassifier])
def test_grid_search_with_two_jobs(cls):
    """GridSearchCV over max_degree and penalty runs the fits in two other
    processes: the scores equal those of one job, and the refit best estimator
    equals a direct fit with the best parameters."""
    y = y_reg if cls is EarthRegressor else y_bin
    grid = {"max_degree": [1, 2], "penalty": [2.0, 3.0]}
    kw = {} if cls is EarthRegressor else {"glm_alpha": ALPHA}
    common = {"cv": KFold(3, shuffle=True, random_state=0)}
    two = GridSearchCV(cls(**kw), grid, n_jobs=2, **common).fit(X, y)
    one = GridSearchCV(cls(**kw), grid, n_jobs=1, **common).fit(X, y)
    assert len(two.cv_results_["params"]) == 4
    assert_array_equal(
        two.cv_results_["mean_test_score"], one.cv_results_["mean_test_score"]
    )
    assert two.best_params_ == one.best_params_
    direct = cls(**kw, **two.best_params_).fit(X, y)
    _assert_same_predictions(two.best_estimator_, direct, X)


@pytest.mark.parametrize("params", [{}, {"max_degree": 2}])
@pytest.mark.parametrize("y", [y_bin, y_tri], ids=["two-classes", "three-classes"])
def test_cross_val_predict_proba_of_the_classifier(y, params):
    """GLM-6 through cross_val_predict: the rows are probabilities that sum to
    1, one column per class, and each fold's rows are those of a fit on the
    other folds."""
    est = EarthClassifier(glm_alpha=ALPHA, **params)
    cv = StratifiedKFold(3, shuffle=True, random_state=1)
    got = cross_val_predict(est, X, y, cv=cv, method="predict_proba")
    assert got.shape == (N, len(np.unique(y)))
    assert_allclose(got.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    assert got.min() >= 0.0
    expected = np.empty_like(got)
    for train, test in cv.split(X, y):
        expected[test] = clone(est).fit(X[train], y[train]).predict_proba(X[test])
    assert_array_equal(got, expected)


@pytest.mark.parametrize("params", [{}, {"max_degree": 2}])
def test_stacking_regressor(params):
    """StackingRegressor with EarthRegressor as a base learner: predictions
    keep the shape of y and follow the signal, and the cloned base learner is
    fitted with its own parameters."""
    stack = StackingRegressor(
        [("earth", EarthRegressor(**params)), ("ridge", Ridge())],
        final_estimator=LinearRegression(),
        cv=3,
    ).fit(X, y_reg)
    pred = stack.predict(X)
    assert pred.shape == (N,)
    assert stack.score(X, y_reg) > 0.8
    fitted = stack.named_estimators_["earth"]
    assert isinstance(fitted, EarthRegressor) and fitted.max_degree == params.get(
        "max_degree", 1
    )
    assert_array_equal(
        fitted.predict(X), EarthRegressor(**params).fit(X, y_reg).predict(X)
    )


@pytest.mark.parametrize("y", [y_bin, y_tri], ids=["two-classes", "three-classes"])
@pytest.mark.parametrize("params", [{}, {"max_degree": 2}])
def test_stacking_classifier(params, y):
    """StackingClassifier with EarthClassifier as a base learner: it takes the
    classifier's predict_proba (GLM-6), the final probabilities sum to 1, and
    the accuracy beats the majority class."""
    stack = StackingClassifier(
        [
            ("earth", EarthClassifier(glm_alpha=ALPHA, **params)),
            ("lr", LogisticRegression()),
        ],
        final_estimator=LogisticRegression(),
        cv=3,
    ).fit(X, y)
    proba = stack.predict_proba(X)
    assert proba.shape == (N, len(np.unique(y)))
    assert_allclose(proba.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    assert_array_equal(stack.classes_, np.unique(y))
    assert stack.score(X, y) > np.bincount(y).max() / N


@pytest.mark.parametrize("params", [{}, {"max_degree": 2}])
def test_calibrated_classifier(params):
    """CalibratedClassifierCV around EarthClassifier (which has
    decision_function and predict_proba) gives probabilities that sum to 1."""
    est = EarthClassifier(glm_alpha=ALPHA, **params)
    for method in ("sigmoid", "isotonic"):
        cal = CalibratedClassifierCV(est, method=method, cv=3).fit(X, y_bin)
        proba = cal.predict_proba(X)
        assert proba.shape == (N, 2)
        assert_allclose(proba.sum(axis=1), 1.0, rtol=0, atol=1e-12)
        assert_array_equal(cal.predict(X), cal.classes_[np.argmax(proba, axis=1)])
        assert cal.score(X, y_bin) > np.bincount(y_bin).max() / N


@pytest.mark.parametrize("params", [{}, {"max_degree": 2}])
def test_transformed_target_regressor(params):
    """TransformedTargetRegressor fits on the transformed y and maps the
    predictions back: it equals a direct fit on log1p(y), then expm1."""
    y = np.exp(y_reg / 2.0)
    ttr = TransformedTargetRegressor(
        EarthRegressor(**params), func=np.log1p, inverse_func=np.expm1
    ).fit(X, y)
    direct = EarthRegressor(**params).fit(X, np.log1p(y))
    assert_array_equal(ttr.predict(X), np.expm1(direct.predict(X)))


@pytest.mark.parametrize(("cls", "y", "params"), CONFIGS)
def test_pandas_feature_names(cls, y, params):
    """API-3 and scikit-learn's rules: feature_names_in_ holds the column
    names of the frame; predict on other names or another order raises
    ValueError, and predict on an array warns; a fit on an array has no
    feature_names_in_ and predict on a frame warns."""
    cols = ["u", "v", "w"]
    frame = pd.DataFrame(X, columns=cols)
    est = cls(**params).fit(frame, y)
    assert list(est.feature_names_in_) == cols
    assert_array_equal(est.predict(frame), est.predict(frame.copy()))
    with pytest.raises(ValueError, match="feature names"):
        est.predict(frame.rename(columns={"u": "z"}))
    with pytest.raises(ValueError, match="feature names"):
        est.predict(frame[["v", "u", "w"]])
    with pytest.warns(UserWarning, match="does not have valid feature names"):
        from_array = est.predict(X)
    assert_array_equal(from_array, est.predict(frame))
    plain = cls(**params).fit(X, y)
    assert not hasattr(plain, "feature_names_in_")
    with pytest.warns(UserWarning, match="has feature names"):
        plain.predict(frame)


def _routed(cls, **params):
    """An estimator that asks for the weights in fit and not in score."""
    return (
        cls(**params)
        .set_fit_request(sample_weight=True)
        .set_score_request(sample_weight=False)
    )


@pytest.mark.parametrize(("cls", "y", "params"), CONFIGS)
def test_sample_weight_through_routing_in_a_pipeline(cls, y, params):
    """API-4: with metadata routing, Pipeline.fit(sample_weight=w) reaches the
    estimator, past a scaler that does not take them, and the fit equals
    fit(..., sample_weight=w) on the scaled X. The weights change the fit."""
    with config_context(enable_metadata_routing=True):
        scaler = StandardScaler().set_fit_request(sample_weight=False)
        pipe = make_pipeline(scaler, _routed(cls, **params))
        pipe.fit(X, y, sample_weight=w_int)
        unweighted = make_pipeline(
            StandardScaler().set_fit_request(sample_weight=False),
            _routed(cls, **params),
        ).fit(X, y)
    Xs = StandardScaler().fit_transform(X)
    direct = cls(**params).fit(Xs, y, sample_weight=w_int)
    assert_array_equal(pipe.predict(X), direct.predict(Xs))
    assert not np.array_equal(pipe.predict(X), unweighted.predict(X))


@pytest.mark.parametrize(("cls", "y", "params"), CONFIGS)
def test_sample_weight_through_routing_in_cross_validate(cls, y, params):
    """API-4: cross_validate(params={"sample_weight": w}) gives every fold the
    weights of its training rows, so each fold's estimator equals a direct
    weighted fit on the same rows."""
    cv = KFold(3, shuffle=True, random_state=2)
    with config_context(enable_metadata_routing=True):
        res = cross_validate(
            _routed(cls, **params),
            X,
            y,
            cv=cv,
            params={"sample_weight": w_int},
            return_estimator=True,
        )
    for fitted, (train, test) in zip(res["estimator"], cv.split(X), strict=True):
        direct = cls(**params).fit(X[train], y[train], sample_weight=w_int[train])
        _assert_same_predictions(fitted, direct, X)
        assert res["test_score"][list(res["estimator"]).index(fitted)] == direct.score(
            X[test], y[test]
        )


@pytest.mark.parametrize(("cls", "y", "params"), CONFIGS)
def test_pickle_and_clone_keep_predictions(cls, y, params):
    """A pickle round trip keeps every prediction bit for bit; clone gives an
    unfitted estimator with the same parameters, whose refit is identical."""
    est = cls(**params).fit(X, y)
    _assert_same_predictions(pickle.loads(pickle.dumps(est)), est, X)
    twin = clone(est)
    assert twin.get_params() == est.get_params()
    assert not hasattr(twin, "mars_")
    _assert_same_predictions(twin.fit(X, y), est, X)


def test_import_as_earth():
    """API-2: ``import pymars as earth; earth.Earth().fit(X, y)`` works, and
    Earth is the regressor."""
    assert pymars.Earth is EarthRegressor
    fitted = pymars.Earth().fit(X, y_reg)
    assert fitted.predict(X).shape == (N,)
    assert fitted.score(X, y_reg) > 0.5


@pytest.mark.parametrize(("cls", "y", "params"), CONFIGS)
def test_two_fits_in_one_process_agree(cls, y, params):
    """Determinism: two fits of the same data give the same terms and the same
    predictions bit for bit, warm or cold."""
    a = cls(**params).fit(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        b = cls(**params).fit(X, y)
    assert_array_equal(a.dirs_, b.dirs_)
    assert_array_equal(a.cuts_, b.cuts_)
    assert a.gcv_ == b.gcv_
    _assert_same_predictions(a, b, X)
