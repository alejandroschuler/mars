"""scikit-learn's estimator checks on EarthRegressor and EarthClassifier, with
no expected failures (VALIDATION_PLAN.md, "scikit-learn"; docs/algorithm.md,
API-1 to API-6, GLM-1, GLM-6 and GLM-7).

Until the fast core covers the whole spec (T11 stage 3, issue #13), the
checks run twice:

- with the reference implementation in place of the fast core, through
  CORE-6, on ``EarthRegressor()``, ``EarthRegressor(max_degree=2)`` and
  ``EarthClassifier()``: every check, since the reference covers weights,
  several responses (the classifier's indicators for three classes), degree 2
  and Fast MARS; these runs are slow;
- with the fast core, on ``EarthRegressor(fast_k=0)`` and
  ``EarthClassifier(fast_k=0)``, the setting of T11 stage 1: every check
  whose fits the fast core supports. A check that needs weights or several
  responses meets the fast core's NotImplementedError, whose message names
  the T11 stage, and is skipped with a note that names #13, so the run grows
  by itself as the stages land. A check may wrap that error in its own
  AssertionError, whose cause it then is. Any other error fails.

Several checks fit classes that a hyperplane of the basis separates, so the
unpenalized refit (``glm_alpha=0``, GLM-2) warns as GLM-4 requires; the
checks do not treat that warning as a failure, and this module does not
either, for that one message.

When T11 is done, the fast-core run takes the estimators of the reference
run, and the reference run can go. ``Earth`` is the same class as
``EarthRegressor`` (API-2), so it needs no run of its own.
"""

import pytest
from reference import mars_ref
from sklearn.utils.estimator_checks import parametrize_with_checks

from pymars import EarthClassifier, EarthRegressor, _core
from pymars._core import MarsFit

#: The start of the fast core's message for a setting it lacks until #13.
FAST_CORE_STAGE = "T11 stage"
#: GLM-4's warning of the refit on separated classes.
SEPARATION = pytest.mark.filterwarnings(
    "ignore:The logistic refit:sklearn.exceptions.ConvergenceWarning"
)


def _stage_error(error: BaseException) -> BaseException | None:
    """The fast core's NotImplementedError for a setting it lacks, when it is
    ``error`` or the cause of ``error``, else None."""
    while error is not None:
        if isinstance(error, NotImplementedError) and str(error).startswith(
            FAST_CORE_STAGE
        ):
            return error
        error = error.__cause__
    return None


def reference_fit_mars(X, Y, w, params, *, record_candidates=False):
    """The reference's fit as a MarsFit, in place of ``_core.fit_mars`` (CORE-6)."""
    fit = mars_ref.fit_mars(X, Y, w, params, record_candidates=record_candidates)
    return MarsFit.from_dict(fit)


@SEPARATION
@parametrize_with_checks(
    [EarthRegressor(), EarthRegressor(max_degree=2), EarthClassifier()]
)
def test_checks_with_the_reference(estimator, check, monkeypatch):
    monkeypatch.setattr(_core, "fit_mars", reference_fit_mars)
    check(estimator)


@SEPARATION
@parametrize_with_checks([EarthRegressor(fast_k=0), EarthClassifier(fast_k=0)])
def test_checks_with_the_fast_core(estimator, check):
    try:
        check(estimator)
    except (NotImplementedError, AssertionError) as error:
        stage = _stage_error(error)
        if stage is None:
            raise
        pytest.skip(f"the fast core lacks a setting of this check until #13: {stage}")
