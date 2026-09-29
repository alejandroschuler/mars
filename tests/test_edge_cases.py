"""Edge cases of the plan's table (VALIDATION_PLAN.md, "Edge cases";
docs/algorithm.md, EDGE-1 to EDGE-5, GCV-1, GCV-2, STOP-3, LIMIT-1, LIMIT-2
and KNOT-4), through the estimator.

Other tests cover the rest of the table: a constant or duplicated column and
the scale and shift of S12 in test_invariance.py; a constant y in
test_core.py (``test_degenerate_fits``); NaN and infinite values in X and y
in scikit-learn's checks (test_sklearn_checks.py) and in the weights in
test_estimators.py, which also has ``allow_missing=True`` (ERR-3). The S11
fits with 3, 5, 8 and 12 cases are compared with earth in
test_conformance.py.
"""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pymars import EarthRegressor, _gcv, _terms


@given(
    st.integers(0, 2**32 - 1),
    st.sampled_from([2, 3, 5, 8, 12]),
    st.integers(1, 14),
    st.integers(1, 2),
)
def test_few_cases(seed, n, p, degree):
    """EDGE-2 and EDGE-5: no special case for small n or for n < p, and never
    more terms than the cases support. STOP-3 refuses a step whose model has
    C(M) ≥ N, where GCV-2 gives +∞, so every forward model and the selected
    one have C(M) < n, and the fit's GCV is finite. With 2 or 3 cases every
    model beyond the intercept has C ≥ n. With 5 cases no hinge enters: a pair
    gives M = 3 and C = 3 + d ≥ 5, and a single hinge needs an earlier step,
    after which it gives M ≥ 3 too; only a linear term (M = 2) can enter."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, p))
    y = 3.0 * X[:, 0] + rng.normal(size=n)
    est = EarthRegressor(max_degree=degree).fit(X, y)
    fwd, d = est.mars_.forward, est.penalty_
    sizes = [int(np.sum(fwd.step <= s)) for s in range(1, fwd.step[-1] + 1)]
    assert all(_gcv.effective_parameters(m, d) < n for m in sizes)
    assert _gcv.effective_parameters(len(est.dirs_), d) < n
    assert math.isfinite(est.gcv_) and np.all(np.isfinite(est.predict(X)))
    if n <= 3:
        assert len(fwd.dirs) == 1
    if n == 5:
        assert not np.isin(fwd.dirs, (_terms.PLUS, _terms.MINUS)).any()


@pytest.mark.parametrize(("p", "limit"), [(1, 21), (200, 201)])
def test_the_term_limit_at_the_ends_of_p(p, limit):
    """LIMIT-1: max(20, 2p) + 1 terms for p = 1, and the cap of 201 for p =
    200, where 2p + 1 would be 401 (the plan's edge-case row; earth caps nk
    the same way). With thresh = 0 the fit of p = 1 runs until the limit or
    an exact fit; LIMIT-2 lets it take at most (limit - 1)/2 steps."""
    rng = np.random.default_rng(p)
    X = rng.uniform(size=(40, p))
    y = np.sin(6.0 * X[:, 0]) + 0.1 * rng.normal(size=40)
    est = EarthRegressor(thresh=0.0).fit(X, y)
    assert est.max_terms_ == limit
    fwd = est.mars_.forward
    assert 1 <= fwd.step[-1] <= (limit - 1) // 2
    assert est.rsq_ > 0.5


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_one_outlier_in_x(sign):
    """One value of x0 at ±1e6, next to values in [0, 1], and a truth with a
    hinge at 0.5 on x0 that is 0 at the outlier: the fit completes with
    finite values, and the outlier is never a knot (KNOT-4: the largest
    value is never a knot, and the lowest E* values are not either).

    The two ends differ by LA-3. With the outlier at -1e6, the hinges
    (x0 - t)₊ are 0 there, and the fit finds the knot near 0.5. With the
    outlier at +1e6, every hinge (x0 - t)₊ is about 1e6 at the outlier, so
    its centered norm is about 1e6 while its part outside the span of the
    intercept and x0, which is (t - x0)₊, stays below 1; the ratio of LA-3 is
    about 1e-12 < τ, every knot of x0 is rejected, and x0 enters at most as a
    linear term."""
    rng = np.random.default_rng(11)
    X = rng.uniform(size=(100, 2))
    X[0, 0] = sign * 1e6
    y = np.maximum(sign * (0.5 - X[:, 0]), 0.0) + X[:, 1]  # 0 at the outlier
    y = y + 0.01 * rng.normal(size=100)
    est = EarthRegressor().fit(X, y)
    fwd = est.mars_.forward
    assert not np.any(np.abs(fwd.cuts) >= 1e6)
    assert math.isfinite(est.gcv_) and np.all(np.isfinite(est.term_coef_))
    hinges = np.abs(fwd.dirs[:, 0]) == 1
    if sign > 0:
        assert not hinges.any() and est.rsq_ > 0.5
    else:
        assert np.any(np.abs(fwd.cuts[hinges, 0] - 0.5) < 0.05)
        assert est.rsq_ > 0.99
