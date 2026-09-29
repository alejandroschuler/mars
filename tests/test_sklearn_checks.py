"""scikit-learn's estimator checks on EarthRegressor, with no expected failures
(VALIDATION_PLAN.md, "scikit-learn"; docs/algorithm.md, API-1 to API-6).

Until the fast core covers the whole spec (T11 stages 2 and 3, issue #13), the
checks run twice:

- with the reference implementation in place of the fast core, through
  CORE-6, on ``EarthRegressor()`` and ``EarthRegressor(max_degree=2)``: every
  check, since the reference covers weights, several responses, degree 2 and
  Fast MARS; these runs are slow;
- with the fast core, on ``EarthRegressor(fast_k=0)``, the setting of T11
  stage 1: every check whose fits the fast core supports. A check that needs
  weights or several responses meets the fast core's NotImplementedError and
  is skipped with a note that names #13, so the run grows by itself as the
  stages land.

When T11 is done, the fast-core run takes the two estimators of the reference
run, and the reference run can go. ``Earth`` is the same class as
``EarthRegressor`` (API-2), so it needs no run of its own.
"""

import pytest
from reference import mars_ref
from sklearn.utils.estimator_checks import parametrize_with_checks

from pymars import EarthRegressor, _core
from pymars._core import MarsFit


def reference_fit_mars(X, Y, w, params, *, record_candidates=False):
    """The reference's fit as a MarsFit, in place of ``_core.fit_mars`` (CORE-6)."""
    fit = mars_ref.fit_mars(X, Y, w, params, record_candidates=record_candidates)
    return MarsFit.from_dict(fit)


@parametrize_with_checks([EarthRegressor(), EarthRegressor(max_degree=2)])
def test_checks_with_the_reference(estimator, check, monkeypatch):
    monkeypatch.setattr(_core, "fit_mars", reference_fit_mars)
    check(estimator)


@parametrize_with_checks([EarthRegressor(fast_k=0)])
def test_checks_with_the_fast_core(estimator, check):
    try:
        check(estimator)
    except NotImplementedError as error:
        pytest.skip(f"the fast core lacks a setting of this check until #13: {error}")
