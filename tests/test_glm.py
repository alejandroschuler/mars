"""Tests of the logistic refit (pymars/_glm.py) and EarthClassifier
(docs/algorithm.md: GLM-1 to GLM-7, W-3, LA-4, ERR-4, API-3 and API-4).

The refit is compared with R on fixed columns: R's ``glm`` and
``nnet::multinom`` on the columns of the component fixture, and earth's own
``glm`` on earth's selected basis for S14 and S20, and ``nnet::multinom`` on
earth's selected basis for S18 (the extra fixture ``s18_multinom``). The
tolerances are those of the plan's table: relative 1e-5 and absolute 1e-7.
scikit-learn's checks in test_sklearn_checks.py pin the rest of the
classifier contract: NotFittedError, shapes, ``predict`` against
``predict_proba`` and ``decision_function``, and the one-class errors.
"""

import json
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.exceptions import ConvergenceWarning
from test_conformance import FIXTURES, load_case

from pymars import EarthClassifier, _core, _glm, _terms

RTOL, ATOL = 1e-5, 1e-7  # the plan's tolerance for GLM coefficients and probabilities


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_basis(X, dirs, cuts, selected):
    rows = np.asarray(selected) - 1  # earth numbers terms from 1
    d = np.asarray(dirs)[rows].astype(np.int8)
    return _terms.basis_matrix(X, d, np.asarray(cuts, dtype=float)[rows])


def _r_case(name):
    """(B, codes, Q, R's coefficients (M, Q - 1), R's probabilities (n, Q),
    earth's result or None) for a comparison on fixed columns."""
    if name in ("binomial", "multinomial"):
        f = _json(FIXTURES / "components" / "classifier_refit.json")[name]
        B = np.asarray(f["x"], dtype=float)
        if name == "binomial":
            p = np.asarray(f["fitted_values"])
            coef = np.asarray(f["coefficients"])[:, None]
            return B, np.asarray(f["y"]).astype(int), 2, coef, np.c_[1 - p, p], None
        codes = np.searchsorted(f["levels"], f["y"])
        coef = np.asarray(f["coefficients"]).T
        return B, codes, 3, coef, np.asarray(f["fitted"]), None
    if name == "s18_multinom":
        f = _json(FIXTURES / "s18_multinom.json")
        X = np.asarray(f["x"], dtype=float)
        B = _selected_basis(X, f["dirs"], f["cuts"], f["selected_terms"])
        case = load_case("S18_matched_d1")
        assert_array_equal(case.X, X)
        assert f["multinom_convergence"] == 0
        codes = np.argmax(case.Y, axis=1)  # the indicators of earth's levels
        assert list(case.earth["levels"]) == list(f["multinom_levels"])
        coef = np.asarray(f["multinom_coefficients"]).T
        return B, codes, 3, coef, np.asarray(f["multinom_fitted"]), None
    case = load_case(name)
    e = case.earth
    B = _selected_basis(case.X, e["dirs"], e["cuts"], e["selected_terms"])
    p = np.asarray(e["pred_train"], dtype=float)[:, 0]
    coef = np.asarray(e["glm_coef"], dtype=float)
    return B, case.Y[:, 0].astype(int), 2, coef, np.c_[1 - p, p], e


@pytest.mark.parametrize(
    "name",
    [
        "binomial",
        "multinomial",
        "S14_defaults_d2",
        "S14_matched_d2",
        "s18_multinom",
        "S20_defaults_d1",
        "S20_matched_d1",
    ],
)
def test_the_refit_matches_r_on_fixed_columns(name):
    """GLM-2 and GLM-3: where R converged, the coefficients and the fitted
    probabilities agree with R's ``glm`` (binomial) and ``nnet::multinom``
    (three classes, reference-class form) to the plan's tolerance, and so do
    earth's predictions at new points (S14). GLM-4: on separated classes (S20)
    both warn, and the probabilities are within 1e-3 of the end of earth's
    where earth's are; a positive ``glm_alpha`` has a finite minimum and does
    not warn. (S14_matched: the fixture has earth's warning that fitted
    probabilities are numerically 0 or 1, and ``glm_converged`` true; its
    smallest ``pred_train`` is 2.2e-16, where the refit gives 2.1e-14, with
    |eta| = 31.5. The v2 threshold |eta| > 30 makes the refit warn there, as
    earth does, although it converges.)"""
    B, codes, Q, r_coef, r_prob, earth = _r_case(name)
    separated = earth is not None and not all(earth["glm_converged"])
    if not separated:
        if name == "S14_matched_d2":  # GLM-4 (v2): converged, but |eta| > 30
            with pytest.warns(ConvergenceWarning, match="numerically 0 or 1"):
                fit = _glm.fit_glm(B, codes, Q)
            assert fit.extreme
        else:
            fit = _glm.fit_glm(B, codes, Q)
        coef = fit.coef[:, None] if Q == 2 else fit.coef[:, 1:]
        assert fit.converged
        if Q > 2:
            assert_array_equal(fit.coef[:, 0], 0.0)
        assert_allclose(coef, r_coef, rtol=RTOL, atol=ATOL)
        prob = _glm.probabilities(_glm.linear_predictors(B, fit.coef))
        assert_allclose(prob, r_prob, rtol=RTOL, atol=ATOL)
        if earth is not None:
            case = load_case(name)
            B_test = _selected_basis(
                case.X_test, earth["dirs"], earth["cuts"], earth["selected_terms"]
            )
            eta = _glm.linear_predictors(B_test, fit.coef)
            got = _glm.probabilities(eta)[:, 1]
            assert_allclose(got, np.ravel(earth["pred_test"]), rtol=RTOL, atol=ATOL)
        return
    assert earth["warnings"]
    with pytest.warns(ConvergenceWarning, match="glm_alpha"):
        fit = _glm.fit_glm(B, codes, Q)
    assert fit.extreme
    p = _glm.probabilities(_glm.linear_predictors(B, fit.coef))[:, 1]
    theirs = r_prob[:, 1]
    assert np.all(p[theirs < 1e-3] < 1e-3)
    assert np.all(p[theirs > 1 - 1e-3] > 1 - 1e-3)
    penalized = _glm.fit_glm(B, codes, Q, alpha=1.0)
    assert penalized.converged and not penalized.extreme


def _gradient(B, codes, Q, w, alpha, coef):
    """The gradient of GLM-2's objective in the original columns, written from
    the spec: Σ w·(p - t)·b plus alpha·s_j²·beta_j for non-intercept column j,
    with s_j² the weighted variance, divisor N, of that column."""
    eta = _glm.linear_predictors(B, coef)
    P = np.exp(eta - np.log(np.exp(eta).sum(axis=1, keepdims=True)))
    T = np.eye(Q)[codes]
    N = math.fsum(w)
    var = w @ (B - w @ B / N) ** 2 / N
    beta = coef[:, None] if Q == 2 else coef[:, 1:]
    G = B.T @ (w[:, None] * (P - T)[:, 1:]) + alpha * var[:, None] * beta
    scale = np.abs(B).T @ w  # the size of the terms of the sum
    return G, scale


@pytest.mark.parametrize("alpha", [0.0, 2.5])
@pytest.mark.parametrize("name", ["binomial", "multinomial"])
def test_the_refit_minimizes_the_weighted_penalized_objective(name, alpha):
    """GLM-2: with weights (zeros among them, W-3) and with or without the
    penalty, the gradient of the objective, with the penalty on the
    standardized columns and none on the intercept, is 0 at the fit. LA-4 and
    GLM-2: a dependent column and a constant column get coefficient 0, and
    leave the fit unchanged. The penalty is invariant to the scale of a
    column: multiplying it by 1000 divides its coefficient by 1000 and
    leaves the probabilities unchanged."""
    B, codes, Q, _, _, _ = _r_case(name)
    w = np.random.default_rng(1).integers(0, 4, size=len(codes)).astype(float)
    fit = _glm.fit_glm(B, codes, Q, w, alpha)
    rows = w > 0
    G, scale = _gradient(B[rows], codes[rows], Q, w[rows], alpha, fit.coef)
    assert np.all(np.abs(G) <= 1e-10 * scale[:, None])

    dependent = 2.0 * B[:, 1] - B[:, 2]
    wide = np.column_stack([B, dependent, np.full(len(codes), 5.0)])
    fit_wide = _glm.fit_glm(wide, codes, Q, w, alpha)
    assert_array_equal(fit_wide.coef[-2:], 0.0)
    assert_allclose(fit_wide.coef[:-2], fit.coef, rtol=1e-9)

    scaled = B.copy()
    scaled[:, 1] *= 1000.0
    fit_scaled = _glm.fit_glm(scaled, codes, Q, w, alpha)
    expect = fit.coef.copy()
    expect[1] /= 1000.0
    assert_allclose(fit_scaled.coef, expect, rtol=1e-9)
    prob = _glm.probabilities(_glm.linear_predictors(B, fit.coef))
    prob_scaled = _glm.probabilities(_glm.linear_predictors(scaled, fit_scaled.coef))
    assert_allclose(prob_scaled, prob, rtol=1e-9)
    if alpha > 0:  # the penalty acts: the fit differs from the unpenalized one
        free = _glm.fit_glm(B, codes, Q, w)
        assert np.max(np.abs(free.coef[1:] - fit.coef[1:])) > 1e-3


_rng = np.random.default_rng(0)
X = _rng.uniform(-1.0, 1.0, size=(150, 2))
_eta = np.c_[np.zeros(150), 2.0 * np.maximum(X[:, 0], 0.0), 1.5 * X[:, 1] - X[:, 0]]
_P = np.exp(_eta) / np.exp(_eta).sum(axis=1, keepdims=True)
CODES3 = (_rng.uniform(size=(150, 1)) > np.cumsum(_P, axis=1)).sum(axis=1)
CODES2 = (CODES3 == 2).astype(int)


@pytest.mark.parametrize(
    ("codes", "labels"),
    [
        (CODES2, np.array(["no", "yes"])),
        (CODES2, np.array([False, True])),
        (CODES2, np.array([-1, 1])),
        (CODES3, np.array(["hi", "lo", "mid"])),
    ],
)
def test_labels_of_any_type_give_the_fit_of_the_codes(codes, labels):
    """GLM-1: string, boolean and {-1, 1} labels give the fit of the 0/1 (or
    0, 1, 2) codes, with ``classes_`` the sorted labels, and the passes run
    on one 0/1 response for two classes and Q indicators otherwise, so that
    ``term_coef_`` is the regressor's fit of those responses. API-3: the
    shapes of ``glm_``; GLM-2: ``glm_`` is the refit of the selected basis.
    GLM-6: ``decision_function`` has a first column of 0 for Q ≥ 3, and
    ``predict`` returns labels. API-4: ``summary()`` shows ``glm_``."""
    Q = len(labels)
    est = EarthClassifier().fit(X, labels[codes])
    base = EarthClassifier().fit(X, codes)
    assert_array_equal(est.classes_, labels)
    assert_array_equal(est.glm_, base.glm_)
    assert_array_equal(est.term_coef_, base.term_coef_)
    assert est.glm_.shape == ((len(est.dirs_),) if Q == 2 else (len(est.dirs_), Q))
    Y = np.eye(Q)[codes][:, 1:] if Q == 2 else np.eye(Q)[codes]
    ls = _core.fit_mars(X, Y, None, est._mars_params())
    assert_array_equal(est.dirs_, ls.dirs)
    assert_allclose(est.term_coef_, ls.coef[:, 0] if Q == 2 else ls.coef)
    B = est.basis_matrix(X)
    assert_array_equal(est.glm_, _glm.fit_glm(B, codes, Q).coef)
    decision = est.decision_function(X)
    if Q > 2:
        assert_array_equal(decision[:, 0], 0.0)
    assert_array_equal(est.predict(X), labels[np.argmax(est.predict_proba(X), 1)])
    assert f"{est.glm_.ravel()[1]:.6g}" in est.summary()
    assert str(labels[-1]) in est.summary().splitlines()[0]


def test_zero_weights_equal_dropping_the_rows():
    """W-3 and GLM-1: rows with weight 0 are dropped first, so their labels,
    here a fourth label that only they have, are not classes, and the fit is
    the fit without them. One of them lies far out on the trend, where its
    probability would be numerically 0 or 1 and warn (GLM-4) if it counted.
    GLM-7: with positive weight on one class only, fit raises the one-class
    error, and API-3: the failed refit leaves no fitted attribute."""
    labels = np.array(["hi", "lo", "mid"])[CODES3]
    w = np.random.default_rng(2).integers(1, 3, size=150).astype(float)
    X0 = np.vstack([X, X[:20] + 0.05, [[60.0, -60.0]]])
    y0 = np.concatenate([labels, np.full(10, "zz"), labels[:10], ["hi"]])
    w0 = np.r_[w, np.zeros(21)]
    dropped = EarthClassifier().fit(X, labels, sample_weight=w)
    est = EarthClassifier().fit(X0, y0, sample_weight=w0)
    assert_array_equal(est.classes_, ["hi", "lo", "mid"])
    assert_array_equal(est.dirs_, dropped.dirs_)
    assert_allclose(est.glm_, dropped.glm_, rtol=1e-12, atol=1e-14)
    assert_allclose(est.predict_proba(X), dropped.predict_proba(X), atol=1e-14)
    one = np.where(labels == "lo", w, 0.0)
    with pytest.raises(ValueError, match=r"one class"):
        est.fit(X, labels, sample_weight=one)
    assert not any(hasattr(est, a) for a in ("classes_", "glm_", "mars_"))


@pytest.mark.parametrize(
    ("codes", "weights", "expect", "first"),
    [
        (np.r_[0, 0, 1, 1], [1, 1, 1, 1], [0.5, 0.5], 0),
        (np.r_[0, 1, 2, 2], [2, 3, 1, 2], [0.25, 0.375, 0.375], 1),
    ],
)
def test_an_intercept_alone_gives_the_weighted_frequencies(
    codes, weights, expect, first
):
    """GLM-5: with the intercept alone (``nprune=1``) the probabilities are the
    weighted class frequencies. GLM-6: on a tie of the largest probabilities,
    ``predict`` returns the first such class."""
    rows = np.repeat(codes, 10)
    Xr = X[: len(rows)]
    w = np.repeat(np.asarray(weights, dtype=float), 10)
    est = EarthClassifier(nprune=1).fit(Xr, rows, sample_weight=w)
    assert len(est.dirs_) == 1
    assert_allclose(est.predict_proba(Xr[:3]), np.tile(expect, (3, 1)), rtol=1e-15)
    assert_array_equal(est.predict(Xr[:3]), first)


@pytest.mark.parametrize("alpha", [-1.0, math.nan, math.inf, "0.1", True, None, 3])
def test_glm_alpha_reaches_the_refit_and_must_be_a_finite_float(alpha):
    """ERR-4: ``fit`` checks ``glm_alpha``, and ``__init__`` only stores it.
    GLM-2: a valid ``glm_alpha`` is the penalty of the refit."""
    est = EarthClassifier(glm_alpha=alpha)
    assert est.glm_alpha is alpha
    if alpha != 3:
        with pytest.raises(ValueError, match="glm_alpha"):
            est.fit(X, CODES2)
        return
    est.fit(X, CODES2)
    penalized = _glm.fit_glm(est.basis_matrix(X), CODES2, 2, alpha=3.0)
    assert_array_equal(est.glm_, penalized.coef)
    assert not np.allclose(est.glm_, _glm.fit_glm(est.basis_matrix(X), CODES2, 2).coef)


@pytest.mark.parametrize(
    ("eta", "weight", "warns"),
    [(-32.0, 1.0, True), (31.0, 1.0, True), (-27.0, 1.0, False), (-60.0, 0.0, False)],
)
def test_the_warning_follows_eta_beyond_30_over_positive_weights(eta, weight, warns):
    """GLM-4 (v2): a converged refit warns when |eta| > 30 for some case of
    positive weight, and not for a case of weight 0, which W-3 drops, nor for
    |eta| below 30. A case far out on the trend of the others barely moves the
    fit, which converges."""
    B, codes, _, _, _, _ = _r_case("binomial")
    base = _glm.fit_glm(B, codes, 2).coef
    far = B[0].copy()
    far[1] = (eta - base[0] - base[2] * far[2]) / base[1]
    B_far, codes_far = np.vstack([B, far]), np.r_[codes, int(eta > 0)]
    w = np.r_[np.ones(len(B)), weight]
    if not warns:
        fit = _glm.fit_glm(B_far, codes_far, 2, w)
        assert fit.converged and not fit.extreme
        return
    with pytest.warns(ConvergenceWarning, match="numerically 0 or 1"):
        fit = _glm.fit_glm(B_far, codes_far, 2, w)
    assert fit.converged and fit.extreme


@pytest.mark.parametrize("k", [600, -1000])
def test_the_classifier_refit_runs_on_the_scaled_x(k):
    """EDGE-7: the refit sees the basis of X with column 0 times 2^k, scaled by
    the same powers of 2 as the core's fit, so the probabilities equal the
    unscaled fit's up to rounding. (Unscaled, the standard deviation of the
    column overflowed at 2^600 and the slope became 0 without a sign.)"""
    rng = np.random.default_rng(5)
    X = rng.uniform(size=(150, 2))
    y = (X[:, 0] + 0.3 * rng.normal(size=150) > 0.5).astype(int)
    Xk = X.copy()
    Xk[:, 0] = np.ldexp(X[:, 0], k)
    base = EarthClassifier().fit(X, y)
    scaled = EarthClassifier().fit(Xk, y)
    assert_allclose(scaled.predict_proba(Xk), base.predict_proba(X), atol=1e-8)


def test_a_zero_weight_row_of_huge_x_gives_no_warning():
    """EDGE-7 and W-3: the scaled basis of a dropped row may overflow, silently.
    (pytest turns a RuntimeWarning into an error.)"""
    rng = np.random.default_rng(6)
    X = rng.uniform(size=(80, 2))
    y = (X[:, 0] + 0.3 * rng.normal(size=80) > 0.5).astype(int)
    X[:, 1] *= 1e-200  # j = 664, so 1e300·2^664 overflows in the dropped row
    X[0, 1], w = 1e300, np.ones(80)
    w[0] = 0.0
    EarthClassifier().fit(X, y, sample_weight=w)
