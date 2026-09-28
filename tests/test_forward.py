"""Tests of pymars/_forward.py, stage 1 of T11: degree 1, one response, no
weights and fast_k = 0.

The earth fixtures (T05) are compared in the matched and the earth-compatible
(defaults) modes at degree 1 by the plan's rules ("What is compared, and the
tolerances", "Ties"): the steps exactly up to the first near-tie, the RSS
after each step within 1e-8 of the RSS before it (LA-5), and the termination
code when the whole path matched. The defaults mode runs earth's fast.k = 20:
at degree 1 with M_max ≤ fast_k + 2 the window holds every row at every step
(FAST-3, FAST-6), so fast_k = 0 gives the same fit.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pymars import _forward, _gcv, _knots, _terms
from pymars._forward import Termination

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "validation" / "fixtures"
KW = {"fast_k": 0, "record_candidates": True}


def _params(a):
    """earth's arguments (API-7) as forward_pass keywords."""
    return {
        "max_terms": a.get("nk"),
        "penalty": a.get("penalty"),
        "thresh": a.get("thresh", 0.001),
        "minspan": a.get("minspan") or None,
        "endspan": a.get("endspan") or None,
        "auto_linpreds": a.get("Auto.linpreds", True),
    }


def _stage1(name):
    """Degree 1, earth's scaled X, no weights, one numeric response that is not
    constant, and a queue that cannot matter (module docstring)."""
    d = json.loads((FIXTURES_DIR / f"{name}.json").read_text())
    a, y = d["earth_args"], d["inputs"]["y"]
    if a.get("degree") != 1 or d.get("scale") is None or "error" in d["result"]:
        return False
    if d["inputs"]["weights"] is not None or not isinstance(y[0], float):
        return False
    p = len(d["inputs"]["X"][0])
    max_terms = a.get("nk") or _gcv.default_max_terms(p)
    fast_k = a.get("fast.k", 20)
    return len(set(y)) > 1 and (fast_k == 0 or max_terms <= max(3, fast_k) + 2)


FIXTURES = sorted(f.stem for f in FIXTURES_DIR.glob("S*.json") if _stage1(f.stem))


def test_the_fixture_selection():
    assert len(FIXTURES) == 88


def _steps(dirs, cuts):
    """The terms of each step: a row with code -1 closes a pair (FWD-6)."""
    dirs, cuts, out, k = np.asarray(dirs), np.asarray(cuts), [], 1
    while k < len(dirs):
        m = 2 if k + 1 < len(dirs) and (dirs[k + 1] == _terms.MINUS).any() else 1
        d = np.asarray(dirs[k : k + m])
        out.append(
            (d.tolist(), np.where(np.abs(d) == 1, cuts[k : k + m], 0.0).tolist())
        )
        k += m
    return out


@pytest.mark.parametrize("name", FIXTURES)
def test_earth_fixture(load_fixture, name):
    d = load_fixture(name)
    X = np.asarray(d["inputs"]["X"], dtype=np.float64)
    y = np.asarray(d["inputs"]["y"], dtype=np.float64)
    fp = _forward.forward_pass(X, y, **_params(d["earth_args"]), **KW)
    ours = _steps(fp.dirs, fp.cuts)
    earth = _steps(d["result"]["dirs"], np.asarray(d["result"]["cuts"]))
    log = fp.candidates
    tie = log.second_rss - log.best_rss < 1e-7 * fp.rss[:-1]
    fwd_rss = d["result"]["fwd_rss"]
    matched = 0
    for s in range(len(ours)):
        if tie[s]:
            break
        assert s < len(earth) and ours[s] == earth[s], f"step {s + 1}"
        last = int(np.flatnonzero(fp.step == s + 1)[-1])
        assert abs(fp.rss[s + 1] - fwd_rss[last]) <= 1e-8 * fp.rss[s]
        matched += 1
    assert abs(fp.rss[0] - fwd_rss[0]) <= 1e-8 * fp.rss[0]
    if matched == len(ours):
        assert len(earth) == len(ours)
        assert fp.termination == d["result"]["termcond"]


def _fit(X, y, **kw):
    return _forward.forward_pass(np.asarray(X, float), np.asarray(y, float), **KW, **kw)


def _suppressed():
    """x1 is x0 plus noise, and y a difference of their hinges: one hinge alone
    explains little, and the two together much. Step 1 takes a small pair on
    x1, and at step 2 every pair on x0 exceeds 10·Δ_1."""
    rng = np.random.default_rng(0)
    x0 = rng.uniform(size=300)
    x1 = x0 + 0.02 * rng.normal(size=300)
    y = 5 * (np.maximum(x0 - 0.5, 0) - np.maximum(x1 - 0.5, 0))
    return np.column_stack((x0, x1)), y + 0.001 * rng.normal(size=300)


SUPPRESSED = {"max_terms": 5, "thresh": 0.0, "minspan": 1, "endspan": 1}


def test_the_limit_on_knots_leaves_the_linear_term_free():
    """FWD-4: no pair on x0 is legal at step 2; the linear term is legal at any
    size, and it wins, although the best pair would reduce the RSS more."""
    X, y = _suppressed()
    x0 = X[:, 0]
    fp = _fit(X, y, **SUPPRESSED)
    assert fp.dirs[3].tolist() == [_terms.LINEAR, 0]
    delta1, delta2 = fp.rss[0] - fp.rss[1], fp.rss[1] - fp.rss[2]
    assert delta2 > 10 * delta1
    B = _terms.basis_matrix(X, fp.dirs[:3], fp.cuts[:3])
    knots = _knots.candidate_knots(np.sort(x0), np.ones(300, bool), 1, 1).knots
    assert len(knots) == 298

    def rss(t):
        A = np.column_stack((B, x0, np.maximum(x0 - t, 0.0)))
        return float(np.sum((y - A @ np.linalg.lstsq(A, y)[0]) ** 2))

    assert fp.rss[1] - min(rss(t) for t in knots) > delta2


@pytest.mark.parametrize(("auto", "code"), [(True, _terms.LINEAR), (False, 1)])
def test_the_linear_option(auto, code):
    """FWD-6: the linear candidate adds b·x (code 2), or b·(x - m)₊ with m the
    smallest x of all cases (code +1); the RSS is the same."""
    X, y = _suppressed()
    fp = _fit(X, y, auto_linpreds=auto, **SUPPRESSED)
    assert fp.dirs[3].tolist() == [code, 0]
    assert fp.cuts[3, 0] == (0.0 if auto else X[:, 0].min())
    other = _fit(X, y, auto_linpreds=not auto, **SUPPRESSED)
    np.testing.assert_allclose(fp.rss, other.rss, rtol=1e-12)


def test_pairs_then_single_hinges_on_one_covariate():
    """LA-7: once x is in the span, its searches are single-hinge searches."""
    x = np.linspace(-1, 1, 101)
    y = np.abs(x) + np.maximum(x - 0.5, 0.0)
    fp = _fit(x[:, None], y, max_terms=7, thresh=0.0, minspan=1, endspan=1)
    assert fp.dirs[1:3, 0].tolist() == [1, -1] and fp.cuts[1, 0] == fp.cuts[2, 0]
    assert fp.dirs[3:, 0].tolist() == [1] * (len(fp.dirs) - 3)
    assert set(fp.candidates.second_kind[1:]) <= {_forward.KIND_HINGE}


def test_a_duplicated_column_is_never_used():
    """FWD-5 and EDGE-4: equal reductions go to the lower covariate index."""
    rng = np.random.default_rng(2)
    x = rng.uniform(size=100)
    X = np.column_stack((x, x, rng.uniform(size=100)))
    fp = _fit(X, np.sin(6 * x) + 0.1 * X[:, 2], thresh=0.0)
    assert not fp.dirs[:, 1].any()
    log = fp.candidates  # the second is the duplicate, with the scan's RSS
    assert log.second_variable[0] == 1
    assert log.second_rss[0] == pytest.approx(log.best_rss[0], rel=1e-10)


@pytest.mark.parametrize(
    ("n", "kw", "code"),
    [
        (30, {"max_terms": 1}, Termination.NO_ROOM),
        (30, {"max_terms": 2}, Termination.NO_ROOM),
        (30, {"max_terms": 4}, Termination.TERM_LIMIT),
        (20, {"penalty": 16.0}, Termination.GRSQ_LOW),
        (3, {}, Termination.GRSQ_NEG_INF),
        (30, {"thresh": 0.5}, Termination.RSQ_CHANGE_SMALL),
    ],
)
def test_the_stopping_rules(n, kw, code):
    """STOP-1, STOP-3 (with C(M') ≥ N at n = 3) and STOP-4."""
    rng = np.random.default_rng(n)
    fp = _fit(rng.uniform(size=(n, 1)), rng.normal(size=n), **kw)
    assert fp.termination == code
    assert len(fp.rss) == (2 if code == Termination.TERM_LIMIT else 1)


@pytest.mark.parametrize(
    ("n", "thresh", "code"),
    [(30, 0.001, Termination.RSQ_CHANGE_SMALL), (30, 0.0, Termination.NO_GAIN)],
)
def test_a_step_without_a_legal_candidate(n, thresh, code):
    """STOP-4 before STOP-2: a constant covariate gives no candidate (EDGE-3)."""
    y = np.random.default_rng(3).normal(size=n)
    fp = _fit(np.ones((n, 1)), y, thresh=thresh)
    assert fp.termination == code and fp.dirs.shape == (1, 1)


def test_an_exact_fit_stops_before_the_term_limit():
    """STOP-5 comes before STOP-1: the RSS floor at an exact fit."""
    x = np.linspace(0, 1, 41)
    y = 2 * np.maximum(x - x[12], 0.0)
    fp = _fit(x[:, None], y, max_terms=3, minspan=1, endspan=1)
    assert fp.termination == Termination.RSQ_HIGH
    assert fp.rss[1] < 1e-10 * fp.rss[0] / 40


def test_the_rsq_rule_of_stop5():
    x = np.linspace(0, 1, 41)
    y = np.maximum(x - x[12], 0.0) + 1e-3 * np.random.default_rng(4).normal(size=41)
    fp = _fit(x[:, None], y, thresh=0.01, minspan=1, endspan=1)
    assert fp.termination == Termination.RSQ_HIGH and len(fp.rss) == 2


@pytest.mark.parametrize("y", [[2.0] * 5, [1.5]])
def test_a_degenerate_fit(y):
    """EDGE-1, GCV-7: a constant response or a single case."""
    fp = _fit(np.arange(len(y), dtype=float)[:, None], y)
    assert fp.termination == Termination.DEGENERATE
    assert fp.rss.tolist() == [0.0] and fp.kept.tolist() == [0]
    assert fp.candidates.best_rss.shape == (0,)


def test_scaling_y():
    """EDGE-6: a power of 2 changes no bit; other factors, the rounding only."""
    rng = np.random.default_rng(5)
    X, y = rng.uniform(size=(60, 3)), rng.normal(size=60)
    base = _fit(X, y)
    exact = _fit(X, y * 2.0**-40)
    np.testing.assert_array_equal(exact.cuts, base.cuts)
    np.testing.assert_array_equal(exact.rss, base.rss * 2.0**-80)
    for c in (1e-9, 1e9):
        other = _fit(X, y * c)
        np.testing.assert_array_equal(other.cuts, base.cuts)
        np.testing.assert_allclose(other.rss, base.rss * c * c, rtol=1e-12)
    x = np.arange(8.0)[:, None]
    tiny = _fit(x, [0.0] * 7 + [1e-170])  # its TSS underflows on this scale
    big = _fit(x, [0.0] * 7 + [1e-170 * 2.0**600])
    assert tiny.termination == big.termination != Termination.DEGENERATE
    np.testing.assert_array_equal(tiny.cuts, big.cuts)
    np.testing.assert_array_equal(tiny.rss, np.ldexp(big.rss, -1200))


def test_the_record():
    """CORE-3, TERM-6, FWD-8 and FWD-11 on a fit of 10 steps."""
    rng = np.random.default_rng(6)
    X = rng.uniform(size=(150, 4))
    y = np.sin(4 * X[:, 0]) + X[:, 1] + 0.1 * rng.normal(size=150)
    X0 = X.copy()
    fp = _fit(X, y, thresh=0.0, auto_linpreds=False)
    np.testing.assert_array_equal(X, X0)
    M, S = fp.dirs.shape[0], len(fp.rss) - 1
    assert fp.dirs.dtype == np.int8 and fp.cuts.shape == (M, 4)
    _terms.check_terms(fp.dirs, fp.cuts)
    assert fp.parent.tolist() == [-1] + [0] * (M - 1)
    assert fp.step[0] == 0 and np.all(np.diff(fp.step) >= 0) and fp.step[-1] == S
    assert fp.kept.tolist() == list(range(M)) and fp.dropped.shape == (0,)
    assert np.all(np.diff(fp.rss) < 0) and fp.rss[0] == _gcv.tss(y)
    log = fp.candidates
    np.testing.assert_array_equal(log.best_rss, fp.rss[1:])
    assert np.all(log.second_rss >= log.best_rss - 1e-7 * fp.rss[:-1])
    linear = (log.second_kind == _forward.KIND_LINEAR) | (log.second_kind == 0)
    assert np.all(np.isnan(log.second_knot) == linear)
    for s in range(S):
        rows = np.flatnonzero(fp.step == s + 1)
        v = int(np.flatnonzero(fp.dirs[rows[0]])[0])
        assert (log.second_variable[s], log.second_knot[s]) != (v, fp.cuts[rows[0], v])
    bare = _forward.forward_pass(X, y, thresh=0.0, auto_linpreds=False, fast_k=0)
    assert bare.candidates is None
    np.testing.assert_array_equal(bare.rss, fp.rss)


@given(st.integers(0, 2**32 - 1), st.integers(5, 60), st.integers(1, 4))
def test_row_order_and_invariants(seed, n, p):
    """CORE-1: the same terms after a permutation of the rows, apart from
    near-ties; every step lowers the RSS, and knot steps obey FWD-4."""
    rng = np.random.default_rng(seed)
    X = np.round(rng.uniform(size=(n, p)), 2)
    y = X @ rng.normal(size=p) + np.maximum(X[:, 0] - 0.5, 0) + rng.normal(size=n)
    fp = _fit(X, y, thresh=0.0)
    perm = rng.permutation(n)
    other = _fit(X[perm], y[perm], thresh=0.0)
    log = fp.candidates
    gaps = (log.second_rss - log.best_rss) / fp.rss[:-1]
    if not np.any(gaps < 1e-7):
        np.testing.assert_array_equal(other.dirs, fp.dirs)
        np.testing.assert_array_equal(other.cuts, fp.cuts)
    red = -np.diff(fp.rss)
    assert np.all(red > 0)
    for s in range(1, len(red)):
        rows = np.flatnonzero(fp.step == s + 1)
        if not (len(rows) == 1 and (fp.dirs[rows[0]] == _terms.LINEAR).any()):
            assert red[s] <= 10 * red[s - 1] * (1 + 1e-9)


def _state(X, y, **kw):
    """A _Pass for white-box tests of the explicit check."""
    params = {"minspan": None, "endspan": None, "adjust_endspan": 2.0}
    params |= {"auto_linpreds": True, "thresh": 0.0, "penalty": 2.0} | kw
    Yc = (y - y.mean())[:, None]
    return _forward._Pass(X, Yc, float(np.sum(Yc**2)), params)


def _cand(kind, knot, variable=0):
    return _forward._Candidate(1.0, (0, variable, 1), 0, variable, kind, knot)


def test_the_explicit_check():
    """The rebuild refuses a zero column, a collinear hinge (KNOT-5: the knot
    at a repeated minimum in a pair), a knot above MaxLegal and a reduction
    that is not positive; the linear term has no upper limit (FWD-4)."""
    x = np.array([0.0, 0.0, 1.0, 2.0, 3.5, 5.0, 6.0, 8.0])
    y = np.array([1.0, 0.0, 2.0, 1.0, 4.0, 3.0, 7.0, 9.0])
    st_ = _state(x[:, None], y)
    assert st_.check(_cand(_forward.KIND_HINGE, 3.5)) is not None
    assert st_.check(_cand(_forward.KIND_HINGE, 8.0)) is None
    assert st_.check(_cand(_forward.KIND_PAIR, 0.0)) is None
    st_.rss = [st_.rss[0], 0.999 * st_.rss[0]]  # 10·Δ_1 is 0.01·TSS
    assert st_.check(_cand(_forward.KIND_PAIR, 3.5)) is None
    assert st_.check(_cand(_forward.KIND_LINEAR, math.nan)) is not None
    st_.rss = [st_.rss[0], 0.0]
    assert st_.check(_cand(_forward.KIND_LINEAR, math.nan)) is None


def test_a_candidate_that_fails_the_check_is_left_out(monkeypatch):
    """The scan's winner is built again; if the explicit values refuse it, the
    step is searched again without it (here a scan that accepts every knot and
    favors the collinear knot at the repeated minimum)."""
    x = np.array([0.0, 0.0, 1.0, 2.0, 3.5, 5.0, 6.0, 8.0, 9.0, 11.0])
    y = np.array([1.0, 0.0, 2.0, 1.0, 4.0, 3.0, 7.0, 9.0, 8.0, 12.0])
    kw = {"max_terms": 3, "minspan": 1, "endspan": 1}
    honest = _fit(x[:, None], y, **kw)
    real = _forward._scan.knot_scan
    calls = []

    def lying(x_, b, Q, E, split):
        out = real(x_, b, Q, E, split)
        calls.append(1)
        gain = np.where(np.asarray(split) == 2, float(np.sum(E**2)), out.gain)
        return out._replace(ratio=np.ones_like(out.ratio), gain=gain)

    monkeypatch.setattr(_forward._scan, "knot_scan", lying)
    fp = _fit(x[:, None], y, **kw)
    assert len(calls) == 2
    np.testing.assert_array_equal(fp.cuts, honest.cuts)


@pytest.mark.parametrize(
    "kw",
    [
        {"max_degree": 2},
        {"w": np.ones(4)},
        {"fast_k": 20},
    ],
)
def test_stage_1_limits(kw):
    X, y = np.arange(8.0).reshape(4, 2), np.arange(4.0)
    args = {"fast_k": 0} | kw
    with pytest.raises(NotImplementedError, match="stage 1"):
        _forward.forward_pass(X, y, **args)
    with pytest.raises(NotImplementedError, match="stage 1"):
        _forward.forward_pass(X, np.ones((4, 2)), fast_k=0)


@pytest.mark.parametrize(
    ("X", "y"),
    [
        (np.ones((3, 1)), np.ones(4)),
        (np.ones(3), np.ones(3)),
        (np.full((3, 1), np.inf), np.ones(3)),
        (np.ones((3, 1)), [0.0, np.nan, 1.0]),
    ],
)
def test_bad_input(X, y):
    with pytest.raises(ValueError, match="X"):
        _forward.forward_pass(X, y, fast_k=0)
