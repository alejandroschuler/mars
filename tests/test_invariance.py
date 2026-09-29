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

from pymars import _gcv, _terms
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


def _agree(a, b, *, cols=None, knot=None, y_scale=1.0, ties_of=None):
    """Check that fit b, of transformed data, is fit a transformed.

    ``cols[j]`` is the column of b that holds column j of a (None: the same);
    ``knot(j, t)`` gives b's knot for a's knot t on column j (None: t); every
    sum of squares of b is y_scale² times a's; ``ties_of`` are the fits whose
    logs show the near-ties (None: a and b). Up to a near-tie, the forward
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
    tie = _tie_step(*(ties_of or (a, b)))
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
    # GCV-2: the GCV of size m is the RSS times a factor of N, m and d, so it
    # has the RSS's tolerance times that factor (+inf where C(m) >= N).
    sizes = np.arange(1, len(pa.gcv_per_size) + 1)
    factor = _gcv.gcv(np.ones(len(sizes)), sizes, a.penalty, a.n_eff)
    np.testing.assert_array_equal(np.isinf(pb.gcv_per_size), np.isinf(factor))
    ok = np.isfinite(factor)
    diff = np.abs(pb.gcv_per_size[ok] - s2 * pa.gcv_per_size[ok])
    assert np.all(diff <= RSS_TOL * s2 * fa.rss[0] * factor[ok])
    assert abs(b.rss - s2 * a.rss) <= RSS_TOL * s2 * fa.rss[0]
    assert b.n_eff == a.n_eff and b.max_terms == a.max_terms
    return True


def _fitted(fit, X):
    return _terms.basis_matrix(X, fit.dirs, fit.cuts) @ fit.coef


def _same_model(a, b, Xa, Xb, Yb, *, x_factor=None, y=(0, 1), **agree):
    """``_agree``, and when the whole fit was compared, the model: b's fitted
    values at Xb within 1e-8·sd(Yb) of shift + scale·(a's at Xa), for
    ``y = (shift, scale)``; and b's coefficients, divided by ``x_factor`` (one
    per term, None: 1), within normwise relative 1e-6 of a's times the scale,
    with the shift added to the intercept, where κ(B) ≤ 1e5 (tolerance
    table). Returns True when the whole fit was compared."""
    shift, scale = y
    if not _agree(a, b, y_scale=scale, **agree):
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


# Scale and shift of a covariate (Conventions, LA-7)

#: The S12 fixture's transforms of x: (shift, scale).
X_MAPS = {"x·1e-8": (0.0, 1e-8), "x·1e8": (0.0, 1e8), "x + 1e6": (1e6, 1.0)}


@given(
    seeds,
    st.integers(20, 90),
    st.integers(1, 4),
    st.integers(1, 2),
    st.booleans(),
    st.sampled_from(sorted(X_MAPS)),
    st.data(),
)
def test_scale_and_shift_of_a_covariate(seed, n, p, degree, linpreds, name, data):
    """x_j to a + s·x_j with s > 0, at the S12 fixture's scales: the same terms,
    with knots a + s·t, the same RSS and GCV values, the same fitted values,
    and the coefficients of the terms that hold x_j divided by s
    (Conventions; LA-7 makes the choice between a pair and a single-hinge
    search free of the units). A shift changes the fit when x_j is a linear
    factor in a product, and in earth too (Conventions), so the shift runs
    with ``auto_linpreds=False`` or at degree 1."""
    shift, scale = X_MAPS[name]
    if shift != 0.0 and degree > 1:
        linpreds = False
    X, Y = _draw(seed, n, p)
    j = data.draw(st.integers(0, p - 1), label="j")
    Xs = X.copy()
    Xs[:, j] = shift + scale * X[:, j]
    kw = {"max_degree": degree, "auto_linpreds": linpreds}
    a, b = _fit(X, Y, **kw), _fit(Xs, Y, **kw)
    factor = np.where(a.dirs[:, j] != 0, 1.0 / scale, 1.0)
    _same_model(
        a,
        b,
        X,
        Xs,
        Y,
        knot=lambda v, t: shift + scale * t if v == j else t,
        x_factor=factor,
    )


def _rows(dirs, cuts):
    """The terms as a sorted list, for sets of terms in any order."""
    return sorted(
        zip(map(tuple, dirs.tolist()), map(tuple, cuts.tolist()), strict=True)
    )


def _ratio(h, x):
    """rho of LA-3 at the first step, where G holds the intercept and x."""
    G = np.column_stack((np.ones_like(x), x))
    r = h - G @ np.linalg.lstsq(G, h, rcond=None)[0]
    return float(r @ r) / float(np.sum((h - h.mean()) ** 2))


@given(
    seeds,
    st.integers(20, 90),
    st.integers(1, 4),
    st.sampled_from([-1.0, -1e-8, -1e8]),
    st.data(),
)
def test_a_negative_scale_mirrors_the_first_pair(seed, n, p, scale, data):
    """x_j to s·x_j with s < 0 mirrors the hinges, since (s·x - s·t)₊ =
    |s|·(t - x)₊. With minspan 1 the knot list of KNOT-3 is symmetric, and a
    pair search spans the same columns in both directions, so the first step
    takes the same covariate, with the knot s·t and the two hinges swapped,
    and the same RSS; the one-step model has the same fitted values. Later
    steps differ often, by the rules that treat the two ends differently: a
    single-hinge search adds only (x - t)₊, and LA-3 centers only that hinge.
    At the first step the exception is LA-3: the chosen knot's hinge, seen
    from the other fit, has rho < 0.01 there, so it is not a candidate. The
    test checks that every other difference is a near-tie."""
    X, Y = _draw(seed, n, p)
    j = data.draw(st.integers(0, p - 1), label="j")
    Xs = X.copy()
    Xs[:, j] = scale * X[:, j]
    kw = {"minspan": 1, "max_terms": 3}
    a, b = _fit(X, Y, **kw), _fit(Xs, Y, **kw)
    fa, fb = a.forward, b.forward
    if _tie_step(a, b) == 1:
        event("near-tie")
        return
    for f, x in ((fa, X[:, j]), (fb, Xs[:, j])):
        v = int(np.flatnonzero(f.dirs[1])[0]) if len(f.dirs) > 1 else -1
        if v == j and f.dirs[1, v] != _terms.LINEAR:
            t = f.cuts[1, v]
            if _ratio(np.maximum(t - x, 0.0), x) < 0.01:  # the mirror's (x' - t')₊
                event("LA-3 end rule")
                return
    mirror = fa.dirs.copy()
    mirror[:, j] = np.where(np.abs(mirror[:, j]) == 1, -mirror[:, j], mirror[:, j])
    cuts = fa.cuts.copy()
    cuts[:, j] = np.where(np.abs(fa.dirs[:, j]) == 1, scale * fa.cuts[:, j], 0.0)
    assert _rows(fb.dirs, fb.cuts) == _rows(mirror, cuts)
    _rss_close(fb.rss, fa.rss, np.r_[fa.rss[0], fa.rss[:-1]])
    assert np.all(np.abs(_fitted(b, Xs) - _fitted(a, X)) <= 1e-8 * np.std(Y))


# Scale and shift of y


#: The S12 fixture's scales of y, a negative scale and a shift: (shift, scale).
Y_MAPS = {
    "y·1e-9": (0.0, 1e-9),
    "y·1e9": (0.0, 1e9),
    "-3y": (0.0, -3.0),
    "y + 1e4": (1e4, 1.0),
    "7 - 0.2y": (7.0, -0.2),
}


@given(
    seeds,
    st.integers(20, 90),
    st.integers(1, 3),
    st.integers(1, 2),
    st.integers(1, 3),
    st.sampled_from(sorted(Y_MAPS)),
)
def test_scale_and_shift_of_y(seed, n, p, degree, k, name):
    """y to a + s·y with s != 0, one or several responses (RESP-1): the same
    terms and records, every RSS and GCV times s², the coefficients times s
    with a added to the intercept, and the fitted values transformed the same
    way. Every RSS scales by s², so the rankings and the relative stopping
    rules do not change (plan: Invariance tests; STOP-5 is relative for every
    K)."""
    shift, scale = Y_MAPS[name]
    X, Y = _draw(seed, n, p, k)
    Ys = shift + scale * Y
    kw = {"max_degree": degree}
    a, b = _fit(X, Y, **kw), _fit(X, Ys, **kw)
    _same_model(a, b, X, X, Ys, y=(shift, scale))


# Added columns (EDGE-3, EDGE-4, LIMIT-1, SPAN-1, SPAN-2)


@given(
    seeds,
    st.integers(20, 90),
    st.integers(1, 3),
    st.integers(1, 2),
    st.sampled_from([0.0, -2.5, 1e6]),
    st.data(),
)
def test_a_constant_or_duplicated_column(seed, n, p, degree, value, data):
    """A constant column, at any place and degree, and a duplicate of column
    0 after the columns leave the fit unchanged, with ``max_terms`` and both
    spans fixed, since p enters LIMIT-1, SPAN-1 and SPAN-2: a constant
    column never enters a term (EDGE-3), and at degree 1 a duplicate with the
    higher index is never used (EDGE-4, FWD-5). The duplicate ties with its
    original at each of its searches, so near-ties come from the fit without
    it."""
    X, Y = _draw(seed, n, p)
    kw = {"max_degree": degree, "max_terms": 15, "minspan": 2, "endspan": 3}
    base = _fit(X, Y, **kw)
    at = data.draw(st.integers(0, p), label="place")
    Xc = np.insert(X, at, value, axis=1)
    cols = np.delete(np.arange(p + 1), at)
    _same_model(base, _fit(Xc, Y, **kw), X, Xc, Y, cols=cols)
    if degree == 1:
        Xd = np.column_stack((X, X[:, 0]))
        dup = _fit(Xd, Y, **kw)
        _same_model(base, dup, X, Xd, Y, cols=np.arange(p), ties_of=(base,))
