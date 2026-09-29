"""Invariance tests of the whole fit (VALIDATION_PLAN.md, "Invariance tests"
and "Determinism"; docs/algorithm.md, Conventions, CORE-1, LA-7, KNOT-2,
EDGE-4 and LIMIT-1).

Each test fits random draws with ``fit_mars``, the core API, since only the
core returns the candidate log that shows a near-tie; the estimators call the
same function (CORE-6). No reference is needed: a fit is compared with the fit
of transformed data.

- Terms and knots must be equal exactly. A knot is a value of the data, so the
  knot of the transformed fit is the transformed knot, computed the same way.
- Every RSS of the forward record and of the pruning record must agree within
  relative 1e-8 of the RSS before its step (LA-5), and the fitted values within
  1e-8·sd(y) (the plan's tolerance table).
- A draw with a near-tie is an allowed exception (the plan's "Ties", STOP-7,
  LA-5): some step of either fit has a second-best candidate whose RSS is
  within 1e-7 of the RSS before the step. The two fits must then agree up to
  that step, and hypothesis counts the draw with ``event``, so that
  ``--hypothesis-show-statistics`` gives the rate. The tolerances stay as
  above.
"""

import numpy as np
from hypothesis import event, given
from hypothesis import strategies as st

from pymars import _terms
from pymars._core import MarsParams, fit_mars

#: The plan's "Ties": a step is a near-tie when the best and the second-best
#: candidate differ by less than this, relative to the RSS before the step.
NEAR_TIE = 1e-7
#: LA-5: every RSS within this, relative to the RSS before its step.
RSS_TOL = 1e-8

seeds = st.integers(0, 2**32 - 1)


def _draw(seed, n, p, k=1):
    """Covariates without repeated values, and k responses with hinges, an
    interaction and noise, so that fits have several steps."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, p))
    x0, x1 = X[:, 0], X[:, 1 % p]
    f = 2.0 * np.maximum(x0 - 0.3, 0.0) + np.abs(x1) + x0 * np.maximum(x1, 0.0)
    Y = f[:, None] * rng.uniform(0.5, 2.0, size=k) + 0.3 * rng.normal(size=(n, k))
    return X, Y


def _fit(X, Y, **kw):
    return fit_mars(X, Y, None, MarsParams(**kw), record_candidates=True)


def _tie_step(*fits):
    """The first forward step, counted from 1, at which some of the fits has a
    near-tie, or None."""
    steps = []
    for fit in fits:
        log, rss = fit.forward.candidates, fit.forward.rss
        tie = log.second_rss - log.best_rss < NEAR_TIE * rss[:-1]
        if tie.any():
            steps.append(int(np.argmax(tie)) + 1)
    return min(steps) if steps else None


def _rss_close(a, b, before):
    """Each RSS within RSS_TOL of the RSS before its step (LA-5)."""
    assert a.shape == b.shape
    assert np.all(np.abs(a - b) <= RSS_TOL * before)


def _agree(a, b, *, cols=None, knot=None, y_scale=1.0):
    """Check that fit b, of transformed data, is fit a transformed.

    ``cols[j]`` is the column of b that holds column j of a (None: the same);
    ``knot(j, t)`` gives b's knot for a's knot t on column j (None: t); every
    sum of squares of b is y_scale² times a's. Up to a near-tie, the forward
    terms and their RSS must agree; without one, the whole fit must: the
    forward and pruning records, the selected terms, the GCV and n_eff. The
    coefficients and fitted values are the caller's to check, since they
    depend on the transform. Returns True when the whole fit was compared.
    """
    p = a.forward.dirs.shape[1]
    cols = np.arange(p) if cols is None else np.asarray(cols)

    def mapped(dirs, cuts):
        out_d = np.zeros((len(dirs), b.forward.dirs.shape[1]), dtype=np.int8)
        out_c = np.zeros_like(out_d, dtype=np.float64)
        out_d[:, cols] = dirs
        for j in range(p):
            hinge = np.abs(dirs[:, j]) == 1
            out_c[hinge, cols[j]] = [knot(j, t) if knot else t for t in cuts[hinge, j]]
        return out_d, out_c

    fa, fb = a.forward, b.forward
    tie = _tie_step(a, b)
    s2 = y_scale**2
    if tie is not None:
        event("near-tie")
        rows_a, rows_b = fa.step < tie, fb.step < tie
        d, c = mapped(fa.dirs[rows_a], fa.cuts[rows_a])
        np.testing.assert_array_equal(fb.dirs[rows_b], d)
        np.testing.assert_array_equal(fb.cuts[rows_b], c)
        _rss_close(fb.rss[:tie], s2 * fa.rss[:tie], s2 * np.r_[fa.rss[0], fa.rss][:tie])
        return False
    d, c = mapped(fa.dirs, fa.cuts)
    np.testing.assert_array_equal(fb.dirs, d)
    np.testing.assert_array_equal(fb.cuts, c)
    for name in ("parent", "step", "kept", "dropped"):
        np.testing.assert_array_equal(getattr(fb, name), getattr(fa, name))
    assert fb.termination == fa.termination
    _rss_close(fb.rss, s2 * fa.rss, s2 * np.r_[fa.rss[0], fa.rss[:-1]])
    pa, pb = a.pruning, b.pruning
    np.testing.assert_array_equal(pb.removed, pa.removed)
    np.testing.assert_array_equal(pb.subsets, pa.subsets)
    assert pb.selected_size == pa.selected_size
    np.testing.assert_array_equal(b.selected, a.selected)
    _rss_close(pb.rss_per_size, s2 * pa.rss_per_size, s2 * fa.rss[0])
    _rss_close(pb.gcv_per_size, s2 * pa.gcv_per_size, s2 * pa.gcv_per_size)
    assert abs(b.rss - s2 * a.rss) <= RSS_TOL * s2 * fa.rss[0]
    assert b.n_eff == a.n_eff and b.max_terms == a.max_terms
    return True


def _fitted(fit, X):
    return _terms.basis_matrix(X, fit.dirs, fit.cuts) @ fit.coef


def _same_model(a, b, Xa, Xb, Yb, *, cols=None, knot=None, x_factor=None, y=(0, 1)):
    """``_agree``, and when the whole fit was compared, the model: b's fitted
    values at Xb within 1e-8·sd(Yb) of shift + scale·(a's at Xa), for
    ``y = (shift, scale)``; and b's coefficients, divided by ``x_factor`` (one
    per term, None: 1), within normwise relative 1e-6 of a's times the scale,
    with the shift added to the intercept, where κ(B) ≤ 1e5 (tolerance
    table). Returns True when the whole fit was compared."""
    shift, scale = y
    if not _agree(a, b, cols=cols, knot=knot, y_scale=scale):
        return False
    want = shift + scale * _fitted(a, Xa)
    assert np.all(np.abs(_fitted(b, Xb) - want) <= 1e-8 * np.std(Yb, axis=0))
    B = _terms.basis_matrix(Xa, a.dirs, a.cuts)
    if np.linalg.cond(B) <= 1e5:
        coef = scale * a.coef
        coef[0] += shift
        got = b.coef if x_factor is None else b.coef / np.asarray(x_factor)[:, None]
        assert np.linalg.norm(got - coef) <= 1e-6 * np.linalg.norm(coef)
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


# Row order, column order and determinism


@given(seeds, st.integers(20, 90), st.integers(1, 4), st.integers(1, 2))
def test_row_and_column_order(seed, n, p, degree):
    """CORE-1 and KNOT-2: a permutation of the rows gives the same model, and
    a permutation of the columns the same model with its columns permuted
    (the draws have no ties). The same fit twice is the same, bit for bit
    (plan: Determinism, within one process)."""
    X, Y = _draw(seed, n, p)
    base = _fit(X, Y, max_degree=degree)
    _bitwise(_fit(X, Y, max_degree=degree).to_dict(), base.to_dict())
    rng = np.random.default_rng(seed)
    rows, cols = rng.permutation(n), rng.permutation(p)
    by_rows = _fit(X[rows], Y[rows], max_degree=degree)
    _same_model(base, by_rows, X, X, Y)
    by_cols = _fit(X[:, cols], Y, max_degree=degree)
    _same_model(base, by_cols, X, X[:, cols], Y, cols=np.argsort(cols))
