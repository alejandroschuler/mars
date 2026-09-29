"""Tests of pymars/_estimators.py and pymars/__init__.py (docs/algorithm.md:
API-1 to API-6, ERR-1 to ERR-4, W-1, W-3, W-6, W-7, CORE-6, RESP-2, TERM-3
and TERM-5).

Every fit here uses the reference implementation in place of the fast core
(CORE-6), since the fast core does not yet cover weights, several responses,
degree 2 or the default ``fast_k`` (T11 stage 1, #13). scikit-learn's checks
in test_sklearn_checks.py already pin much of the API: NotFittedError and the
number of columns in ``predict``, NaN and infinite values in X and y, sparse
and complex X, the TypeError of an object array that holds a dict, weights of
the wrong shape, pickling, ``clone``, and integer weights against repeated
rows on one small data set. test_core.py pins the pickling of ``MarsFit``.
These tests pin the rules that the checks leave open.
"""

import dataclasses
import re

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from reference import mars_ref
from sklearn import config_context
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils import get_tags

import pymars
from pymars import EarthRegressor, _core, _terms
from pymars._core import MarsFit, MarsParams

RECIPE = (
    'make_column_transformer((OneHotEncoder(drop="first"), cols), '
    'remainder="passthrough")'
)
ISSUE = "https://github.com/alejandroschuler/mars/issues/27"

_rng = np.random.default_rng(0)
X = _rng.uniform(-1.0, 1.0, size=(60, 2))
y = 3.0 * np.maximum(X[:, 0] - 0.2, 0.0) + X[:, 1] + 0.1 * _rng.normal(size=60)
y2 = np.maximum(0.3 - X[:, 1], 0.0) + 0.1 * _rng.normal(size=60)
n = len(y)


def reference_fit_mars(X, Y, w, params, *, record_candidates=False):
    """The reference's fit as a MarsFit, in place of ``_core.fit_mars`` (CORE-6)."""
    fit = mars_ref.fit_mars(X, Y, w, params, record_candidates=record_candidates)
    return MarsFit.from_dict(fit)


@pytest.fixture(autouse=True)
def reference_core(monkeypatch):
    monkeypatch.setattr(_core, "fit_mars", reference_fit_mars)


def _close(a, b, rtol=1e-9):
    """Normwise relative closeness, so that values near 0 need no own scale."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    assert np.linalg.norm(a - b) <= rtol * np.linalg.norm(b)


def test_the_parameters_reach_the_core_as_mars_params(monkeypatch):
    """API-1: the parameters and their defaults are the fields of MarsParams,
    and allow_missing. CORE-6: fit calls ``_core.fit_mars`` through the
    module, with float64 X and Y, Y of shape (n, K), and w None when no
    weights are given (W-5)."""
    defaults = {f.name: f.default for f in dataclasses.fields(MarsParams)}
    assert EarthRegressor().get_params() == {**defaults, "allow_missing": False}
    calls = []

    def spy(X, Y, w, params, **kw):
        calls.append((X, Y, w, params))
        return reference_fit_mars(X, Y, w, params, **kw)

    monkeypatch.setattr(_core, "fit_mars", spy)
    given = {
        "max_degree": 2,
        "max_terms": 9,
        "penalty": 2.5,
        "thresh": 0.01,
        "minspan": 2,
        "endspan": 3,
        "adjust_endspan": 1.0,
        "auto_linpreds": False,
        "fast_k": 4,
        "fast_beta": 0.5,
        "pmethod": "none",
        "nprune": 5,
    }
    EarthRegressor(**given).fit(X.astype(np.float32), y.tolist())
    [(Xc, Yc, w, params)] = calls
    assert params == MarsParams(**given)
    assert (Xc.dtype, Yc.dtype, Yc.shape) == (np.float64, np.float64, (n, 1))
    assert w is None


@pytest.mark.parametrize(
    ("params", "error", "match"),
    [
        ({"max_degree": 0}, ValueError, "max_degree"),
        ({"penalty": -0.5}, ValueError, "penalty"),
        ({"allow_missing": True}, NotImplementedError, re.escape(ISSUE)),
        ({"allow_missing": "no"}, ValueError, "allow_missing"),
    ],
)
def test_fit_checks_the_parameters(params, error, match):
    """ERR-4: ``__init__`` only stores the parameters, and ``fit`` raises the
    ValueError of MarsParams, which names the parameter. ERR-3:
    ``allow_missing=True`` raises NotImplementedError with the link to #27."""
    estimator = EarthRegressor(**params)
    assert estimator.get_params() == {**EarthRegressor().get_params(), **params}
    with pytest.raises(error, match=match):
        estimator.fit(X, y)


def _with_text(kind):
    """X with a text or categorical column, and the columns that hold them."""
    if kind == "numpy str":
        return X.astype(str), "[0, 1]"
    if kind == "numpy bytes":
        return X.astype("S"), "[0, 1]"
    if kind == "object array":
        A = X.astype(object)
        A[5, 1] = "a"
        return A, "[1]"
    if kind == "list":
        rows = X.tolist()
        rows[5][1] = b"a"
        return rows, "[1]"
    groups = np.where(X[:, 1] > 0.0, "a", "b")
    column = {
        "pandas object": groups.astype(object),
        "pandas string": pd.array(groups, dtype="string"),
        "pandas category": pd.Categorical(np.round(X[:, 1])),
    }[kind]
    return pd.DataFrame({"x": X[:, 0], "g": column}), "['g']"


@pytest.mark.parametrize(
    "kind",
    [
        "numpy str",
        "numpy bytes",
        "object array",
        "list",
        "pandas object",
        "pandas string",
        "pandas category",
    ],
)
def test_text_columns_raise_an_error_that_names_one_hot_encoder(kind):
    """ERR-2: a column of strings, bytes or pandas categories (numeric
    categories too) raises ValueError that names the columns, OneHotEncoder
    and the recipe, in ``fit`` and in the methods that take X."""
    Xt, columns = _with_text(kind)
    fitted = EarthRegressor().fit(X, y)
    calls = (
        lambda: EarthRegressor().fit(Xt, y),
        lambda: fitted.predict(Xt),
        lambda: fitted.basis_matrix(Xt),
    )
    for call in calls:
        with pytest.raises(ValueError, match="OneHotEncoder") as info:
            call()
        assert RECIPE in str(info.value)
        assert f"the columns {columns} of X" in str(info.value)


@pytest.mark.parametrize(
    ("weights", "match"),
    [
        (2.0, "one weight per row"),
        (np.r_[-1.0, np.ones(n - 1)], "at least 0"),
        (np.r_[np.nan, np.ones(n - 1)], "sample_weight contains NaN"),
        (np.r_[np.inf, np.ones(n - 1)], "sample_weight contains infinity"),
        (np.zeros(n), r"weight.*zero"),
    ],
)
def test_weights_outside_the_rules_raise_value_error(weights, match):
    """W-6 and ERR-1: weights must be 1-D with one weight per row, finite and
    at least 0; weights that are all 0 raise ValueError that matches
    ``weight.*zero``."""
    with pytest.raises(ValueError, match=match):
        EarthRegressor().fit(X, y, sample_weight=weights)


HALVES = np.tile([0.5, 1.5], n // 2)  # not integers, mean 1


@pytest.mark.parametrize(
    ("weights", "warns"),
    [
        (np.tile([0, 2, 3], n // 3), False),  # integers with zeros, mean 5/3
        (HALVES, False),
        (np.r_[np.ones(n - 1), 3.5], True),  # mixed integers and a half
        (np.tile([0.0, 0.5, 1.5, 2.0], n // 4), False),  # the mean counts the zeros
        (HALVES * (1 + 5e-7), False),
        (HALVES * (1 + 2e-6), True),
        (np.full(n, 0.25), True),
    ],
)
def test_weights_that_are_not_integers_warn_unless_their_mean_is_1(weights, warns):
    """W-7: a UserWarning, which says that the weights act as case counts and
    how to rescale importance weights, when some weight is not an integer and
    the mean of the weights differs from 1 by more than 1e-6. pytest turns
    any other warning into an error."""
    if not warns:
        EarthRegressor().fit(X, y, sample_weight=weights)
        return
    with pytest.warns(UserWarning, match="case counts") as record:
        EarthRegressor().fit(X, y, sample_weight=weights)
    assert "w * n / sum(w)" in str(record[0].message)


def test_integer_weights_act_as_repeated_rows():
    """W-1 and W-3 through the estimator: integer weights, zeros included,
    give the fit of each row repeated that many times, whatever the row
    order: the same terms and knots, and the same coefficients, statistics
    and predictions up to rounding."""
    w = np.random.default_rng(1).integers(0, 4, size=n)
    order = np.random.default_rng(2).permutation(n)
    weighted = EarthRegressor(max_degree=2).fit(
        X[order], y[order], sample_weight=w[order]
    )
    repeated = EarthRegressor(max_degree=2).fit(
        np.repeat(X, w, axis=0), np.repeat(y, w)
    )
    assert len(repeated.dirs_) >= 4 and (w == 0).any()
    assert_array_equal(weighted.dirs_, repeated.dirs_)
    assert_array_equal(weighted.cuts_, repeated.cuts_)
    for name in ("term_coef_", "rss_", "gcv_", "rsq_", "grsq_"):
        _close(getattr(weighted, name), getattr(repeated, name))
    _close(weighted.predict(X), repeated.predict(X))


def test_the_fitted_attributes_describe_the_returned_model():
    """API-3, API-4 and RESP-2: the attributes are those of ``mars_``, with the
    resolved ``max_terms_`` and ``penalty_``; a 2-D y with one column gives
    the fit of the 1-D y, and ``term_coef_`` and ``predict`` follow the shape
    of y; the predictions have the RSS ``rss_`` (PRUNE-8); ``basis_matrix``
    gives the terms of TERM-3, also outside the range of the data."""
    Y2 = np.column_stack([y, y2])
    one, col, two = (
        EarthRegressor(max_degree=2).fit(X, t) for t in (y, y[:, None], Y2)
    )
    M, M2 = len(one.dirs_), len(two.dirs_)
    for est, Y, coef_shape in (
        (one, y, (M,)),
        (col, y[:, None], (M, 1)),
        (two, Y2, (M2, 2)),
    ):
        fit = est.mars_
        assert isinstance(fit, MarsFit)
        for name in (
            "dirs",
            "cuts",
            "rss",
            "gcv",
            "rsq",
            "grsq",
            "max_terms",
            "penalty",
        ):
            assert_array_equal(getattr(est, name + "_"), getattr(fit, name))
        assert_array_equal(est.term_coef_, fit.coef.reshape(coef_shape))
        assert (est.max_terms_, est.penalty_) == (21, 3.0)  # LIMIT-1 at p = 2, GCV-4
        pred = est.predict(X)
        assert pred.shape == Y.shape
        assert_allclose(np.sum((Y - pred) ** 2), est.rss_, rtol=1e-10)
    assert_array_equal(one.dirs_, col.dirs_)
    assert_array_equal(one.cuts_, col.cuts_)
    assert_array_equal(one.term_coef_, col.term_coef_[:, 0])
    X_new = np.random.default_rng(4).uniform(-3.0, 3.0, size=(20, 2))
    B = two.basis_matrix(X_new)
    assert B.shape == (20, M2)
    _close(B, mars_ref.basis_matrix(X_new, two.dirs_, two.cuts_), 1e-14)


def test_summary_shows_each_term_with_its_label_and_coefficients():
    """API-4 and TERM-5: one line per selected term with its label, which
    uses the column names of a DataFrame (``feature_names_in_``), else x0,
    x1, ..., and its coefficients, one column per response; then ``rss_``,
    ``gcv_``, ``rsq_``, ``grsq_`` and the termination."""
    named = EarthRegressor().fit(pd.DataFrame(X, columns=["age", "dose"]), y)
    assert list(named.feature_names_in_) == ["age", "dose"]
    assert len(named.dirs_) > 1  # so that some label holds a name
    plain = EarthRegressor().fit(X, np.column_stack([y, y2]))
    assert not hasattr(plain, "feature_names_in_")
    for est, names, heads in (
        (named, ["age", "dose"], ["coef"]),
        (plain, None, ["y0", "y1"]),
    ):
        lines = est.summary().splitlines()
        labels = _terms.term_labels(est.dirs_, est.cuts_, names)
        coef = est.term_coef_.reshape(len(labels), -1)
        assert lines[0].split() == ["term", *heads]
        assert [s.split() for s in lines[1:-2]] == [
            [label, *(f"{c:.6g}" for c in row)]
            for label, row in zip(labels, coef, strict=True)
        ]
        stats = [
            (k, f"{getattr(est, k + '_'):.6g}") for k in ("rss", "gcv", "rsq", "grsq")
        ]
        assert lines[-2].split() == [s for pair in stats for s in pair]
        code = est.mars_.forward.termination
        assert lines[-1] == f"termination {int(code)} ({code.name})"


def test_basis_matrix_and_summary_need_a_fit():
    """ERR-4: ``basis_matrix`` and ``summary`` raise NotFittedError before a
    fit, also after a ``fit`` that failed, and ``basis_matrix`` checks the
    number of columns."""
    est = EarthRegressor()
    with pytest.raises(ValueError, match=r"weight.*zero"):
        est.fit(X, y, sample_weight=np.zeros(n))  # fails after validate_data
    with pytest.raises(NotFittedError):
        est.basis_matrix(X)
    with pytest.raises(NotFittedError):
        est.summary()
    est.fit(X, y)
    with pytest.raises(ValueError, match="features"):
        est.basis_matrix(X[:, :1])


def test_the_public_names_the_tags_and_what_is_left_out():
    """API-2: ``Earth`` is ``EarthRegressor``, and the package exports only the
    estimators and ``__version__``. API-5: the tags. API-6: no ``coef_``,
    ``transform``, ``max_iter`` or ``feature_importances_``."""
    assert pymars.Earth is pymars.EarthRegressor
    assert sorted(pymars.__all__) == [
        "Earth",
        "EarthClassifier",
        "EarthRegressor",
        "__version__",
    ]
    tags = get_tags(EarthRegressor())
    assert tags.target_tags.multi_output is True
    assert tags.input_tags.allow_nan is False
    est = EarthRegressor().fit(X, y)
    for name in ("coef_", "transform", "max_iter", "feature_importances_"):
        assert not hasattr(est, name)


def test_sample_weight_reaches_fit_through_metadata_routing():
    """API-4: with metadata routing on, a Pipeline passes ``sample_weight`` to
    ``fit`` once the estimator requests it."""
    w = np.random.default_rng(3).integers(1, 4, size=n)
    with config_context(enable_metadata_routing=True):
        step = EarthRegressor().set_fit_request(sample_weight=True)
        routed = Pipeline([("earth", step)]).fit(X, y, sample_weight=w)
    direct = EarthRegressor().fit(X, y, sample_weight=w)
    assert_array_equal(routed.predict(X), direct.predict(X))
    assert not np.array_equal(direct.predict(X), EarthRegressor().fit(X, y).predict(X))


def test_a_failed_refit_leaves_no_fitted_attribute():
    """API-3: a fit that raises after the new data were validated removes the
    old model, so the names and the terms never come from different fits."""
    est = EarthRegressor().fit(X, y)
    with pytest.raises(ValueError, match="at least 0"):
        est.fit(X, y, sample_weight=-np.ones(n))
    assert not hasattr(est, "mars_") and not hasattr(est, "term_coef_")
    with pytest.raises(NotFittedError):
        est.predict(X)
