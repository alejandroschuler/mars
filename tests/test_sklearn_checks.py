"""scikit-learn's estimator checks on EarthRegressor and EarthClassifier
(VALIDATION_PLAN.md, "scikit-learn"; docs/algorithm.md, API-1 to API-6,
GLM-1, GLM-6 and GLM-7).

The checks run twice on ``EarthRegressor()``, ``EarthRegressor(max_degree=2)``
and ``EarthClassifier()``: with the fast core, which covers weights, several
responses (the classifier's indicators for three classes), degree 2 and Fast
MARS since T11 stage 3 (#13), and, marked slow, with the reference
implementation in place of the fast core, through CORE-6. ``Earth`` is the
same class as ``EarthRegressor`` (API-2), so it needs no run of its own.

Every check must pass, with one exception on the fast core, listed in
``fast_core_failures`` with its issue. On the data of the classifier's
weight check, the last forward step is a near-tie at an exact fit, which the
fast core decides by a negative rounded RSS, so the weighted rows and the
repeated rows get different terms (#81). Rounding decides it: the check fails
with scikit-learn 1.9 and numpy 2.5 and passes with scikit-learn 1.6 and
numpy 2.0, so a failure there is an expected failure and a pass is a pass.
The reference passes it.

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
    if isinstance(estimator, EarthClassifier):
        return {
            "check_sample_weight_equivalence_on_dense_data": (
                "#81: a near-tie at an exact fit in the fast forward pass"
            )
        }
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
