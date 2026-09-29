"""Tests of the sample weights through the estimator (VALIDATION_PLAN.md,
"Sample weights"; docs/algorithm.md, W-1 to W-8, KNOT-6, SPAN-1, SPAN-5,
GCV-2, RESP-1 and the Fast MARS queue).

The weights are case counts, so the whole fit must agree with the fit of
repeated rows: the terms and knots exactly, the forward and pruning records,
the GCV and the coefficients. Two fits of the same model on different rows
add in a different order, so the sums of squares agree within relative 1e-8
of the RSS before their step (LA-5), the coefficients within normwise
relative 1e-6 where κ(B) ≤ 1e5 and the fitted values within 1e-8·sd(y) (the
plan's tolerance table). A near-tie (the plan's "Ties") is the only allowed
exception: when the forward terms differ, the candidate log of ``fit_mars``
must show a near-tie at or before the first step that differs, and
hypothesis counts the draw with ``event``.

The warning of W-7 has its own test in test_estimators.py, and the
zero-weight and unit-weight tests of test_core.py and test_forward.py use a
fixed basis or the forward pass alone; the tests here run the whole fit on
random draws, with several responses, degree 2 and Fast MARS. Against earth,
S16 goes through repeated rows in test_conformance.py.
"""

import numpy as np
import pytest
from hypothesis import event, given
from hypothesis import strategies as st

from pymars import EarthRegressor, _core, _gcv
from pymars._core import MarsParams

seeds = st.integers(0, 2**32 - 1)
FAST_K = [0, 3, 20]  # every parent, the smallest window (FAST-3), the default


def _draw(seed, n, p, k):
    """Covariates without repeated values and k responses."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, p))
    x0, x1 = X[:, 0], X[:, 1 % p]
    f = 2.0 * np.maximum(x0 - 0.3, 0.0) + np.abs(x1) + x0 * np.maximum(x1, 0.0)
    Y = f[:, None] * rng.uniform(0.5, 2.0, size=k) + 0.3 * rng.normal(size=(n, k))
    return X, Y[:, 0] if k == 1 else Y


def _first_tie(X, Y, w, kw):
    """The first forward step with a near-tie in the candidate log, or inf."""
    fit = _core.fit_mars(X, Y, w, MarsParams(**kw), record_candidates=True)
    log, rss = fit.forward.candidates, fit.forward.rss
    tie = log.second_rss - log.best_rss < 1e-7 * rss[:-1]
    return int(np.argmax(tie)) + 1 if tie.any() else np.inf


def _same_fit(a, b, data_a, data_b, kw):
    """Check that estimators a and b hold the same model (module docstring).
    ``data_a`` and ``data_b`` are their (X, Y, w), which the check of a
    near-tie refits. Returns True when the whole fit was compared."""
    fa, fb = a.mars_.forward, b.mars_.forward
    path = ("dirs", "cuts", "parent")  # a parent can differ at an exact tie
    if not all(np.array_equal(getattr(fa, f), getattr(fb, f)) for f in path):
        S = min(len(fa.rss), len(fb.rss)) - 1
        same = [
            all(
                np.array_equal(
                    getattr(fa, f)[fa.step == s], getattr(fb, f)[fb.step == s]
                )
                for f in path
            )
            for s in range(1, S + 1)
        ]
        first = same.index(False) + 1 if False in same else S + 1
        tie = min(_first_tie(*data_a, kw), _first_tie(*data_b, kw))
        assert tie <= first, f"the forward terms differ at step {first}, no near-tie"
        event("near-tie")
        return False
    for name in ("step", "kept", "dropped"):
        np.testing.assert_array_equal(getattr(fb, name), getattr(fa, name))
    assert fb.termination == fa.termination
    before = np.r_[fa.rss[0], fa.rss[:-1]]
    assert np.all(np.abs(fb.rss - fa.rss) <= 1e-8 * before)
    pa, pb = a.mars_.pruning, b.mars_.pruning
    np.testing.assert_array_equal(pb.removed, pa.removed)
    np.testing.assert_array_equal(pb.subsets, pa.subsets)
    assert pb.selected_size == pa.selected_size
    tol = 1e-8 * fa.rss[0]
    assert np.all(np.abs(pb.rss_per_size - pa.rss_per_size) <= tol)
    # GCV-2: the GCV of size m is the RSS times a factor of N, m and d, so
    # it has the RSS's tolerance times that factor (+inf where C(m) >= N).
    sizes = np.arange(1, len(pa.gcv_per_size) + 1)
    factor = _gcv.gcv(np.ones(len(sizes)), sizes, a.penalty_, a.mars_.n_eff)
    np.testing.assert_array_equal(np.isinf(pb.gcv_per_size), np.isinf(factor))
    ok = np.isfinite(factor)
    diff = np.abs(pb.gcv_per_size[ok] - pa.gcv_per_size[ok])
    assert np.all(diff <= tol * factor[ok])
    np.testing.assert_array_equal(b.dirs_, a.dirs_)
    np.testing.assert_array_equal(b.cuts_, a.cuts_)
    assert b.mars_.n_eff == a.mars_.n_eff and b.max_terms_ == a.max_terms_
    assert abs(b.rss_ - a.rss_) <= tol
    X = data_a[0]
    if np.linalg.cond(a.basis_matrix(X)) <= 1e5:
        ca, cb = a.term_coef_, b.term_coef_
        assert np.linalg.norm(cb - ca) <= 1e-6 * np.linalg.norm(ca)
    sd = np.std(data_a[1], axis=0)
    assert np.all(np.abs(b.predict(X) - a.predict(X)) <= 1e-8 * sd)
    return True


def _bitwise(a, b):
    """Equal dicts of CORE-5, bit for bit."""
    assert type(a) is type(b)
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _bitwise(a[key], b[key])
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert a.tobytes() == b.tobytes()
    else:
        assert a == b or (a != a and b != b)  # a NaN is equal to itself here


@given(
    seeds,
    st.integers(15, 60),
    st.integers(1, 3),
    st.integers(1, 3),
    st.integers(1, 2),
    st.sampled_from(FAST_K),
    st.sampled_from(["random", "equal"]),
    st.integers(1, 3),
)
def test_integer_weights(seed, n, p, k, degree, fast_k, kind, c):
    """W-1, W-3, W-5 and W-8 on the whole fit, with one or several responses
    (RESP-1), at degree 1 or 2 and with or without the window of Fast MARS:

    - integer weights c·w, with w from 0 to 4 or all 1, give the fit of each
      row repeated c·w times; so equal weights c act as c copies of the data
      and are not ignored, and N = Σw is the number of cases in the GCV, the
      spans and the knots (W-2);
    - a zero weight gives the fit of the removed row, bit for bit, since the
      row is dropped before anything else (W-3);
    - weights that are all 1 give the fit of no weights, bit for bit (W-5).
    """
    X, Y = _draw(seed, n, p, k)
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 5, size=n) if kind == "random" else np.ones(n, dtype=int)
    w = c * w
    if not w.any():
        w[0] = c
    kw = {"max_degree": degree, "fast_k": fast_k}
    weighted = EarthRegressor(**kw).fit(X, Y, sample_weight=w.astype(float))
    Xr, Yr = np.repeat(X, w, axis=0), np.repeat(Y, w, axis=0)
    repeated = EarthRegressor(**kw).fit(Xr, Yr)
    _same_fit(weighted, repeated, (X, Y, w.astype(float)), (Xr, Yr, None), kw)
    keep = w > 0
    removed = EarthRegressor(**kw).fit(X[keep], Y[keep], sample_weight=w[keep])
    _bitwise(removed.mars_.to_dict(), weighted.mars_.to_dict())
    unit = EarthRegressor(**kw).fit(X, Y, sample_weight=np.ones(n))
    _bitwise(unit.mars_.to_dict(), EarthRegressor(**kw).fit(X, Y).mars_.to_dict())


@given(
    seeds,
    st.integers(10, 40),
    st.integers(1, 3),
    st.integers(1, 2),
    st.sampled_from([2, 4, 10]),
)
def test_a_row_split_into_equal_parts(seed, n, p, degree, parts):
    """W-4 and KNOT-6: a row of integer weight w split into w·parts rows of
    weight 1/parts gives the fit of the row with weight w, although the sums
    of 1/10 are not exact, since N, N_b and the cumulative weights of the
    knot scan are compared with integers within τ_N (W-4's example of 100
    cases of weight 0.1). These weights are not integers and their mean is
    1/parts, so fit warns (W-7)."""
    X, Y = _draw(seed, n, p, 1)
    w = np.random.default_rng(seed).integers(1, 4, size=n)
    kw = {"max_degree": degree}
    whole = EarthRegressor(**kw).fit(X, Y, sample_weight=w.astype(float))
    reps = w * parts
    Xs, Ys = np.repeat(X, reps, axis=0), np.repeat(Y, reps)
    ws = np.full(len(Ys), 1.0 / parts)
    with pytest.warns(UserWarning, match="case counts"):
        split = EarthRegressor(**kw).fit(Xs, Ys, sample_weight=ws)
    assert split.mars_.n_eff == whole.mars_.n_eff == w.sum()
    _same_fit(whole, split, (X, Y, w.astype(float)), (Xs, Ys, ws), kw)
