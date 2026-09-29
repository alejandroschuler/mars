"""scikit-learn's estimator checks on EarthRegressor and EarthClassifier
(VALIDATION_PLAN.md, "scikit-learn"; docs/algorithm.md, API-1 to API-6,
GLM-1, GLM-6 and GLM-7).

The checks run twice on ``EarthRegressor()``, ``EarthRegressor(max_degree=2)``
and ``EarthClassifier()``: with the fast core, which covers weights, several
responses (the classifier's indicators for three classes), degree 2 and Fast
MARS since T11 stage 3 (#13), and with the reference
implementation in place of the fast core, through CORE-6. ``Earth`` is the
same class as ``EarthRegressor`` (API-2), so it needs no run of its own.

Every check must pass, on both cores. A check that fails on the fast core
for a known reason is listed in ``fast_core_failures`` with its issue; none
is listed now (the classifier's weight check, #81, passes since #83).

Several checks fit classes that a hyperplane of the basis separates, so the
unpenalized refit (``glm_alpha=0``, GLM-2) warns as GLM-4 requires; the
checks do not treat that warning as a failure, and this module does not
either, for that one message.
"""

import pytest
from reference import mars_ref
from sklearn.utils.estimator_checks import parametrize_with_checks

from pymars import EarthClassifier, EarthRegressor, _core
from pymars._core import MarsFit

ESTIMATORS = [EarthRegressor(), EarthRegressor(max_degree=2), EarthClassifier()]
#: GLM-4's warning of the refit on separated classes.
SEPARATION = pytest.mark.filterwarnings(
    "ignore:The logistic refit:sklearn.exceptions.ConvergenceWarning"
)


def fast_core_failures(estimator) -> dict[str, str]:
    """The checks that fail on the fast core, each with its issue."""
    return {}


def reference_fit_mars(X, Y, w, params, *, record_candidates=False):
    """The reference's fit as a MarsFit, in place of ``_core.fit_mars`` (CORE-6)."""
    fit = mars_ref.fit_mars(X, Y, w, params, record_candidates=record_candidates)
    return MarsFit.from_dict(fit)


@SEPARATION
@parametrize_with_checks(ESTIMATORS)
def test_checks_with_the_reference(estimator, check, monkeypatch):
    monkeypatch.setattr(_core, "fit_mars", reference_fit_mars)
    check(estimator)


@SEPARATION
@parametrize_with_checks(ESTIMATORS)
def test_checks_with_the_fast_core(estimator, check):
    issue = fast_core_failures(estimator).get(getattr(check, "func", check).__name__)
    try:
        check(estimator)
    except AssertionError:
        if issue is None:
            raise
        pytest.xfail(issue)
