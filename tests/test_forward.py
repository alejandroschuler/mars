"""Tests of pymars/_forward.py, stages 1 and 2 of T11: every degree, one
response, no weights and fast_k = 0.

The earth fixtures (T05) are compared in the matched and the earth-compatible
(defaults) modes at degrees 1 to 3 by the plan's rules ("What is compared,
and the tolerances", "Ties"): the steps exactly up to the first near-tie, the
RSS after each step within 1e-8 of the RSS before it (LA-5), and the
termination code when the whole path matched. The defaults mode runs earth's
fast.k = 20: with M_max ≤ fast_k + 2 the window holds every row at every step
(FAST-3), so fast_k = 0 gives the same fit.
"""

import json
import math
import types
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from reference import mars_ref

from pymars import _forward, _gcv, _knots, _linalg, _scan, _terms
from pymars._forward import Termination

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "validation" / "fixtures"
KW = {"fast_k": 0, "record_candidates": True}


def _params(a):
    """earth's arguments (API-7) as forward_pass keywords."""
    return {
        "max_degree": a.get("degree", 1),
        "max_terms": a.get("nk"),
        "penalty": a.get("penalty"),
        "thresh": a.get("thresh", 0.001),
        "minspan": a.get("minspan") or None,
        "endspan": a.get("endspan") or None,
        "adjust_endspan": a.get("Adjust.endspan", 2.0),
        "auto_linpreds": a.get("Auto.linpreds", True),
    }


def _selected(name):
    """Earth's scaled X, no weights, one numeric response that is not constant,
    and a queue whose window holds every row (module docstring)."""
    d = json.loads((FIXTURES_DIR / f"{name}.json").read_text(encoding="utf-8"))
    a, y = d["earth_args"], d["inputs"]["y"]
    if d.get("scale") is None or "error" in d["result"]:
        return False
    if d["inputs"]["weights"] is not None or not isinstance(y[0], float):
        return False
    p = len(d["inputs"]["X"][0])
    max_terms = a.get("nk") or _gcv.default_max_terms(p)
    fast_k = a.get("fast.k", 20)
    return len(set(y)) > 1 and (fast_k == 0 or max_terms <= max(3, fast_k) + 2)


# The dataset fixtures are S01 to S20; the lowercase names are extras, which a
# case-insensitive glob (Windows) would also match.
FIXTURES = sorted(
    f.stem
    for f in FIXTURES_DIR.glob("S*.json")
    if f.name[0] == "S" and _selected(f.stem)
)


def test_the_fixture_selection():
    at = [sum(f"_d{d}" in name for name in FIXTURES) for d in (2, 3)]
    assert len(FIXTURES) == 105 and at == [15, 2]


def _steps(dirs, cuts):
    """The terms of each step (FWD-6): row k + 1 closes a pair when it differs
    from row k only in one covariate, where row k has +1 and row k + 1 has -1,
    with the same knots."""
    dirs, cuts, out, k = np.asarray(dirs), np.asarray(cuts), [], 1
    while k < len(dirs):
        m = 1
        if k + 1 < len(dirs):
            diff = np.flatnonzero(dirs[k] != dirs[k + 1])
            pair = len(diff) == 1 and np.array_equal(cuts[k], cuts[k + 1])
            m = 2 if pair and (dirs[k, diff[0]], dirs[k + 1, diff[0]]) == (1, -1) else 1
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


def _queue_design(name):
    """Data and settings for test_the_queue_against_the_reference, with no
    near-tie between candidates. "binary": a covariate with the values -1 and
    1 has no knot, so its linear term enters first and is a parent with
    negative cases (KNOT-1). "degree 3": two covariates, single hinges (the
    slots run ahead of M, FAST-4) and parents of degree 2 that hold both
    covariates (λ = -1, FAST-5). "ageing": the same data with other spans,
    fast_beta 2.5 (FAST-2) and adjust_endspan 0.5 (SPAN-4). "pair rule": x1
    is x0 plus noise inside an interaction, on covariates with variance 1/12,
    where the variances of the parent's covariates decide the kind of a
    search (LA-7)."""
    rng = np.random.default_rng({"binary": 14, "pair rule": 4}.get(name, 0))
    kw = {"max_degree": 2, "max_terms": 9, "thresh": 0.0}
    if name == "binary":
        X = np.column_stack((rng.choice([-1.0, 1.0], size=60), rng.uniform(size=60)))
        y = 2 * X[:, 0] + 3 * X[:, 0] * np.maximum(X[:, 1] - 0.4, 0)
        return X, y + 0.05 * rng.normal(size=60), kw
    if name == "pair rule":
        x0 = rng.uniform(size=100)
        X = np.column_stack(
            (x0, x0 + 0.003 * rng.normal(size=100), rng.uniform(size=100))
        )
        y = 5 * np.maximum(X[:, 2] - 0.5, 0) * (np.maximum(X[:, 1] - 0.5, 0) + x0)
        return X, y + 0.01 * rng.normal(size=100), kw | {"minspan": 1, "endspan": 1}
    X = rng.uniform(size=(80, 2))
    y = 10 * np.maximum(X[:, 0] - 0.4, 0) * np.maximum(X[:, 1] - 0.3, 0) + X[:, 0]
    kw |= {"max_degree": 3, "max_terms": 21}
    if name == "degree 3":
        kw |= {"minspan": 1, "endspan": 1}
    else:
        kw |= {"fast_beta": 2.5, "adjust_endspan": 0.5}
    return X, y + 0.05 * rng.normal(size=80), kw


@pytest.mark.parametrize("name", ["binary", "degree 3", "ageing", "pair rule"])
def test_the_queue_against_the_reference(monkeypatch, name):
    """FAST-1, FAST-2, FAST-4 and FAST-5 with fast_k = 0, and the searches of
    parents other than the intercept (FWD-2, KNOT-1 to KNOT-3, SPAN-4, LA-7),
    against the reference called as a black box with its trace: at every step
    the parents searched (entry e stands for slot e), and κ and λ of every
    entry after the search, λ within LA-5 of the TSS; the queue table and the
    order of the parents up to the first two computed λ within STOP-7's band
    of each other, which either program may order (two parents that reach
    the same product, as in "ageing"); then the whole record. λ is the best
    legal reduction of a parent, so it checks the search of every parent,
    also of those whose candidates do not win."""
    X, y, kw = _queue_design(name)
    fast, real = [], _forward._Pass.best

    def best(st_):
        table = st_.table()
        out = real(st_)
        fast.append((table, list(st_.rows), st_.lam.copy(), st_.kap.copy()))
        return out

    monkeypatch.setattr(_forward._Pass, "best", best)
    fp = _forward.forward_pass(X, y, **KW, **kw)
    Y = y[:, None]
    Ys, n = np.ldexp(Y, mars_ref.y_scale_power(Y)), len(y)
    tss = mars_ref.rss(np.ones((n, 1)), Ys, np.ones(n))
    trace = []
    ref, _ = mars_ref.forward_pass(
        X,
        Ys,
        np.ones(n),
        kw | {"fast_k": 0},
        N=float(n),
        tau_N=mars_ref.weight_tol(float(n)),
        tss=tss,
        record_candidates=True,
        trace=trace,
    )
    tie = False
    for (table, parents, lam, kap), r in zip(fast, trace, strict=True):
        if tie:
            assert sorted(parents) == sorted(r["searched"])
        else:
            assert (table, parents) == (list(r["table"]), list(r["searched"]))
        assert kap == [k for _, k in r["entries"]]
        want = [v for v, _ in r["entries"]]
        np.testing.assert_allclose(lam, want, rtol=0.0, atol=1e-8 * tss)
        values = np.sort([v for v in lam if 0.0 < v < math.inf])  # computed
        tie = tie or bool(np.any(np.diff(values) <= 4e-8 * tss))  # 2δ ≤ 4e-8·TSS
    for field in ("dirs", "cuts", "parent", "step", "kept"):
        np.testing.assert_array_equal(getattr(fp, field), ref[field])
    assert fp.termination == ref["termination"]
    log = fp.candidates
    assert np.all(log.second_rss - log.best_rss >= 1e-7 * fp.rss[:-1])


def _fit(X, y, **kw):
    return _forward.forward_pass(np.asarray(X, float), np.asarray(y, float), **KW, **kw)


def _rss(A, y):
    A = np.column_stack(A)
    return float(np.sum((y - A @ np.linalg.lstsq(A, y, rcond=None)[0]) ** 2))


def test_the_codes():
    """CORE-3's kinds and CORE-4's codes."""
    kinds = (_forward.KIND_NONE, _forward.KIND_PAIR, _forward.KIND_HINGE)
    assert (*kinds, _forward.KIND_LINEAR) == (0, 1, 2, 3)
    assert [(t.name, int(t)) for t in Termination] == [
        ("DEGENERATE", 0),
        ("NO_ROOM", 1),
        ("GRSQ_NEG_INF", 2),
        ("GRSQ_LOW", 3),
        ("RSQ_CHANGE_SMALL", 4),
        ("RSQ_HIGH", 5),
        ("NO_GAIN", 6),
        ("TERM_LIMIT", 7),
    ]


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
        return float(np.sum((y - A @ np.linalg.lstsq(A, y, rcond=None)[0]) ** 2))

    assert fp.rss[1] - min(rss(t) for t in knots) > delta2
    assert fp.candidates.second_rss[1] >= fp.candidates.best_rss[1]  # legal only


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
    [
        (30, 0.001, Termination.RSQ_CHANGE_SMALL),
        (30, 0.0, Termination.NO_GAIN),
        (3, 0.001, Termination.GRSQ_NEG_INF),
    ],
)
def test_a_step_without_a_legal_candidate(n, thresh, code):
    """STOP-4 before STOP-2: a constant covariate gives no candidate (EDGE-3);
    STOP-3 counts such a step as M + 1 = 2 terms, so C = 3 ≥ N at n = 3."""
    y = np.random.default_rng(3).normal(size=n)
    fp = _fit(np.ones((n, 1)), y, thresh=thresh)
    assert fp.termination == code and fp.dirs.shape == (1, 1)


def _eight():
    rng = np.random.default_rng(8)
    return rng.uniform(size=(8, 3)), rng.normal(size=8)


@pytest.mark.parametrize(
    ("data", "steps"), [(([[0.0], [0.0], [1.0]], [0.0, 1.0, 5.0]), 0), (_eight(), 1)]
)
def test_grsq_counts_the_real_terms(data, steps):
    """STOP-3 with M' = M + 1 for a linear term (the only candidate at n = 3:
    LA-3 rejects the knot at the repeated minimum) and M + 2 for a pair (the
    second pair at n = 8 has C = 5 + 2·2 = 9 ≥ N)."""
    fp = _fit(*data)
    assert fp.termination == Termination.GRSQ_NEG_INF and len(fp.rss) == steps + 1


def test_a_step_with_one_legal_candidate():
    """The only knot is at the repeated minimum (KNOT-5), so the linear term is
    the only candidate and the log has no second (CORE-3); then no candidate
    is left at all (NO_GAIN at thresh 0)."""
    fp = _fit([[0.0], [0.0], [1.0], [1.0]], [0.0, 1.0, 2.0, 3.0], thresh=0.0)
    assert fp.dirs.tolist() == [[0], [2]] and fp.termination == Termination.NO_GAIN
    log = fp.candidates
    second = (log.second_rss, log.second_parent, log.second_variable)
    assert [a.tolist() for a in second] == [[math.inf], [-1], [-1]]
    assert np.isnan(log.second_knot[0]) and log.second_kind[0] == _forward.KIND_NONE


@pytest.mark.parametrize(("sd", "pair"), [(4e-3, True), (1e-3, False)])
def test_the_pair_rule(sd, pair):
    """LA-7: x0 is x1 plus noise. After a pair on x1, A of x0 lies above
    0.01·sigma² for sd 4e-3 and below it for sd 1e-3, so its search at step 2
    is a pair search or a single-hinge search."""
    rng = np.random.default_rng(7)
    x0 = rng.uniform(size=200)
    X = np.column_stack((x0, x0 + sd * rng.normal(size=200)))
    fp = _fit(X, np.maximum(x0 - 0.5, 0) + 5 * X[:, 1], thresh=0.0, max_terms=5)
    ratio = _rss([_terms.basis_matrix(X, fp.dirs[:3], fp.cuts[:3])], x0) / (
        0.01 * _linalg.weighted_variances(X)[0]
    )
    assert (ratio > 2.0) if pair else (ratio < 0.5)
    assert fp.dirs[3:, 0].tolist() == ([1, -1] if pair else [1])


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
    assert fp.dirs.tolist() == [[0]] and fp.cuts.tolist() == [[0.0]]
    assert fp.rss.tolist() == [0.0] and fp.kept.tolist() == [0]
    assert fp.dropped.shape == (0,) and fp.dropped.dtype == np.int64
    assert fp.parent.tolist() == [-1] and fp.step.tolist() == [0]
    assert all(a.shape == (0,) for a in fp.candidates)
    assert fp.candidates.second_kind.dtype == np.int8


def _on_binary_grids(design):
    """The data of test_exact_shifts. Every value lies on a binary grid, so a
    shift by a power of 2 is exact. "pairs" and "one covariate" have a hinge
    in x0 and a sine in the last covariate; in "linear", x0 is x1 plus noise
    and y = 5·(x1 - x0); in "interaction", hinges on x1 multiply x0 and a
    hinge on x2."""
    n = 200
    if design == "interaction":
        rng = np.random.default_rng(3)
        X = np.round(rng.uniform(size=(n, 3)) * 2**12) / 2**12
        y = 5 * np.maximum(X[:, 1] - 0.4, 0) * (X[:, 0] + np.maximum(X[:, 2] - 0.5, 0))
        return X, np.round((y + 0.1 * rng.normal(size=n)) * 2**10) / 2**10
    if design == "linear":
        rng = np.random.default_rng(3)
        x1 = np.round(rng.uniform(size=n) * 2**12) / 2**12
        x0 = np.round((x1 + 0.02 * rng.normal(size=n)) * 2**12) / 2**12
        y = 5 * (x1 - x0) + 1e-3 * rng.normal(size=n)
        return np.column_stack((x0, x1)), np.round(y * 2**20) / 2**20
    grid, p = (12, 2) if design == "pairs" else (6, 1)
    rng = np.random.default_rng(0)
    u = np.round(rng.uniform(size=n) * 2**grid) / 2**grid
    X = np.column_stack((u, rng.uniform(size=n)))[:, :p]
    y = np.sin(5 * X[:, -1]) + np.maximum(u - 0.5, 0.0) + 0.05 * rng.normal(size=n)
    return X, np.round(y * 2**8) / 2**8


@pytest.mark.parametrize(
    ("design", "x_shift", "y_shift"),
    [
        ("one covariate", 2.0**46, 0),  # A of LA-7, in setup
        ("pairs", 2.0**36, 0),  # the first column of a pair
        ("linear", 2.0**24, 0),  # the column of the linear candidate
        ("pairs", 0, 2.0**44),  # the TSS
        ("interaction", 2.0**36, 0),  # b·(x - c) for a parent b other than 1
    ],
)
def test_exact_shifts(design, x_shift, y_shift):
    """Exact shifts of x0 or of y give the fit of the unshifted data: the same
    terms, knots moved by the shift, and every RSS, rss[0] = TSS and the
    log's second included, within 1e-8 of the RSS before its step (LA-5,
    CORE-3). x is centered before
    Gram-Schmidt (the plan's "Fast path"), and the TSS comes from the centered
    y (FWD-10); each case pins one of these uses. Near 1e14 an uncentered x
    makes A of LA-7 rounding noise above its threshold, and a single-hinge
    search becomes a pair search. In "linear", FWD-4 bars every knot of x0 at
    step 2, and the linear candidate of x0 wins. "interaction" runs at degree
    2 with auto_linpreds=False, where x0 enters products as b·(x0 - m)₊, which
    a shift does not change (FWD-6)."""
    X, y = _on_binary_grids(design)
    kw = {"thresh": 0.0, "minspan": 1, "endspan": 1, "max_terms": 11}
    if design == "interaction":
        kw |= {"max_degree": 2, "auto_linpreds": False}
    a = _fit(X, y, **kw)
    b = _fit(X + np.eye(X.shape[1])[0] * x_shift, y + y_shift, **kw)
    np.testing.assert_array_equal(b.dirs, a.dirs)
    moved = np.abs(a.dirs[:, 0]) == 1
    np.testing.assert_array_equal(b.cuts[moved, 0], a.cuts[moved, 0] + x_shift)
    assert np.all(np.abs(b.rss - a.rss) <= 1e-8 * np.r_[a.rss[0], a.rss[:-1]])
    second = a.candidates.second_rss, b.candidates.second_rss
    np.testing.assert_allclose(*second, rtol=0.0, atol=1e-8 * a.rss[0])


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
    assert np.all(np.diff(fp.rss) < 0)
    assert abs(fp.rss[0] - _gcv.tss(y)) <= 1e-8 * fp.rss[0]  # LA-5
    log = fp.candidates
    np.testing.assert_array_equal(log.best_rss, fp.rss[1:])
    assert np.all(log.second_rss >= log.best_rss - 1e-7 * fp.rss[:-1])
    linear = (log.second_kind == _forward.KIND_LINEAR) | (log.second_kind == 0)
    assert np.all(np.isnan(log.second_knot) == linear)
    assert np.all(log.second_kind > 0) and np.all(log.second_parent == 0)
    assert np.all(log.second_rss < fp.rss[:-1])
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
    """A _Pass on centered y, for the white-box tests."""
    params = {"minspan": None, "endspan": None, "adjust_endspan": 2.0}
    params |= {"auto_linpreds": True, "thresh": 0.0, "penalty": 2.0}
    params |= {"max_degree": 1, "fast_beta": 1.0} | kw
    Yc = (y - y.mean())[:, None]
    return _forward._Pass(X, Yc, float(np.sum(Yc**2)), params)


def _cand(kind, knot, variable=0):
    return _forward._Candidate(1.0, (0, variable, 1), 0, variable, kind, knot)


def _ns(rss=(12.0,), tss=1.0, N=11.0, penalty=-1.0, thresh=0.001):
    params = {"penalty": penalty, "thresh": thresh}
    tau = _gcv.weight_tolerance(N)
    return types.SimpleNamespace(rss=list(rss), tss=tss, N=N, tau=tau, params=params)


@pytest.mark.parametrize(
    ("ns", "chosen", "rss_new", "m_new", "code"),
    [
        (_ns(), True, 11.0, 3, None),  # GRSq' = -10 exactly: not below
        (_ns(), True, 11.5, 3, Termination.GRSQ_LOW),
        (_ns(thresh=0.0), True, 11.5, 3, None),  # STOP-3 needs thresh > 0
        (_ns(penalty=2.0), True, 1.0, 11, Termination.GRSQ_NEG_INF),
        (_ns((1.0,), thresh=0.25), True, 0.75, 3, None),  # a change of thresh
        (_ns((1.0,), thresh=0.25), True, 0.76, 3, Termination.RSQ_CHANGE_SMALL),
        (_ns((1.0,), thresh=0.0), None, 1.0, 2, Termination.NO_GAIN),
    ],
)
def test_the_stops_before_a_step(ns, chosen, rss_new, m_new, code):
    """STOP-3, STOP-4 and STOP-2 at their bounds (penalty -1: GRSq = RSq)."""
    assert _forward._stop(ns, chosen, rss_new, m_new) == code


@pytest.mark.parametrize(
    ("rss", "thresh", "high"),
    [
        (1.0, 0.25, True),  # RSS/TSS = thresh
        (1.0000001, 0.25, False),
        (3.8e-11, 0.0, True),  # the floor 1e-10·TSS/(N - 1) is 4e-11
        (1e-10 * 4.0 / 10.0, 0.0, False),
        (4.2e-11, 0.0, False),
    ],
)
def test_the_stop_after_a_step(rss, thresh, high):
    """STOP-5 at its bounds."""
    assert _forward._rsq_high(_ns(tss=4.0, thresh=thresh), rss) is high


@pytest.mark.parametrize("step", [1, 2])
def test_search_equals_least_squares(step):
    """FWD-3, FWD-4, LA-2, LA-3, LA-7: the best two legal candidates of one
    search against least squares on explicit columns. Step 1 is a pair search
    on x; step 2, after a pair on x, a single-hinge search."""
    rng = np.random.default_rng(10)
    x = np.round(rng.uniform(size=40), 2)
    y = np.sin(4 * x) + 0.3 * rng.normal(size=40)
    st_ = _state(x[:, None], y, minspan=1, endspan=1)
    if step == 2:
        st_.add(*st_.best()[:2])
    B = _terms.basis_matrix(x[:, None], st_.dirs, st_.cuts)
    G, base = ([B, x] if step == 1 else [B]), _rss([B], y)
    want = [(base - _rss([B, x], y), (0, 0, 0))] if step == 1 else []
    knots = _knots.candidate_knots(np.sort(x), np.ones(40, bool), 1, 1).knots
    for i, t in enumerate(knots):
        h = np.maximum(x - t, 0.0)
        red = base - _rss([*G, h], y)
        rho = _rss(G, h) / np.sum((h - h.mean()) ** 2)
        if rho >= _linalg.collinearity_tolerance(step) and red <= st_.max_legal():
            want.append((red, (0, 0, i + 1)))
    want = sorted(want, key=lambda c: (-c[0], c[1]))[:2]
    got = _forward._top_two(st_.refine(st_.search(0, 0, set())))
    assert [c.order for c in got] == [c[1] for c in want]
    for c, (red, _) in zip(got, want, strict=True):
        assert c.reduction == pytest.approx(red, rel=1e-9)


def _fixed_gain(monkeypatch, gain):
    """Make the scan accept every knot and give each the share ``gain(split)``
    of the RSS, as exact values (bounds of 0)."""
    real = _forward._scan.knot_scan

    def fake(x_, b, Q, E, split, **kw):
        out = real(x_, b, Q, E, split, **kw)
        g = gain(np.asarray(split)) * float(np.sum(E**2))
        zero = np.zeros_like(out.ratio)
        return _forward._scan.KnotScan(np.ones_like(out.ratio), g, zero, zero)

    monkeypatch.setattr(_forward._scan, "knot_scan", fake)


def _two_covariates():
    x = np.arange(12.0)
    return _state(np.column_stack((x, x[::-1] ** 2)), np.sin(x), minspan=1, endspan=1)


@pytest.mark.parametrize("gain", [0.0, 0.5])
def test_ties_within_a_search(monkeypatch, gain):
    """FWD-5: equal reductions go to the linear candidate before the knots,
    and to the larger knot first; an excluded place is left out, and only on
    its own covariate. Here the scan gives every knot the same gain."""
    _fixed_gain(monkeypatch, lambda split: np.full(split.shape, gain))
    st_ = _two_covariates()
    first = [(0, 0, 0), (0, 0, 1)] if gain == 0.0 else [(0, 0, 1), (0, 0, 2)]
    got = st_.search(0, 0, set())[:2]  # every knot ties, so every knot is kept
    assert [c.order for c in got] == first
    assert [(c.parent, c.variable) for c in got] == [(0, 0), (0, 0)]
    kinds = [_forward.KIND_LINEAR, _forward.KIND_PAIR]
    assert [c.kind for c in got] == (kinds if gain == 0.0 else kinds[1:] * 2)
    linear = [c for c in st_.search(0, 0, set()) if c.kind == _forward.KIND_LINEAR]
    assert [(c.err, c.sure) for c in linear] == [(0.0, True)]
    later = [(0, 0, 1), (0, 0, 2)] if gain == 0.0 else [(0, 0, 2), (0, 0, 3)]
    assert [c.order for c in st_.search(0, 0, {first[0]})[:2]] == later
    knots = [c.knot for c in st_.search(0, 0, {(0, 0, 0)})]
    assert knots == sorted(knots, reverse=True)
    other = [c.order for c in st_.search(0, 1, set())]
    assert [c.order for c in st_.search(0, 1, {(0, 0, 1), (0, 0, 2)})] == other


def test_leaving_out_the_linear_candidate_keeps_every_knot(monkeypatch):
    """The smallest knot, best here, stays when the linear place 0 is left out."""
    _fixed_gain(monkeypatch, lambda split: np.where(split == split.min(), 0.5, 0.1))
    st_ = _two_covariates()
    smallest = st_.search(0, 0, set())[0]
    assert st_.search(0, 0, {(0, 0, 0)})[0] == smallest
    assert smallest.knot == 1.0  # x_(E* + 1) with E* = 1 (KNOT-4)


def test_the_search_keeps_only_legal_knots(monkeypatch):
    """FWD-4 in the scan: with MaxLegal below every pair, only the linear
    candidate is legal; a single-hinge search whose knots gain nothing has no
    candidate, since a reduction must be positive."""
    st_ = _two_covariates()
    st_.rss = [st_.rss[0], (1.0 - 1e-4) * st_.rss[0]]  # MaxLegal = 1e-3·TSS
    assert [c.kind for c in st_.search(0, 0, set())] == [_forward.KIND_LINEAR]
    st_ = _two_covariates()
    st_.add(*st_.best()[:2])  # now x0 is in the span
    assert st_.search(0, 0, set()) != []
    _fixed_gain(monkeypatch, lambda split: np.zeros(split.shape))
    assert st_.search(0, 0, set()) == []


def test_max_legal_is_inclusive():
    """FWD-4: a knot whose explicit reduction equals MaxLegal is legal, in pass
    2 and in the explicit check; with no residual, no candidate is left, not
    even the linear one."""
    st_ = _two_covariates()
    best = _forward._top_two(st_.refine(st_.search(0, 0, set())))[0]
    assert best.kind == _forward.KIND_PAIR and best.err == 0.0 and best.sure
    st_.max_legal = lambda: best.reduction
    assert st_.refine([best._replace(err=1.0, sure=False)]) == [best]
    del st_.max_legal  # the check's reduction comes from the rebuilt RSS, which
    reduction = st_.rss[-1] - st_.check(best).rss  # can differ in the last bits
    st_.max_legal = lambda: reduction
    assert st_.check(best) is not None
    x = np.arange(12.0)
    assert _state(x[:, None], np.zeros(12)).search(0, 0, set()) == []


def test_the_explicit_check():
    """The check of the winner's rebuild refuses a zero column, a knot above
    MaxLegal and a reduction that is not positive; the linear term has no upper
    limit (FWD-4). LA-3 is decided in pass 2, not here."""
    x = np.array([0.0, 0.0, 1.0, 2.0, 3.5, 5.0, 6.0, 8.0])
    y = np.array([1.0, 0.0, 2.0, 1.0, 4.0, 3.0, 7.0, 9.0])
    st_ = _state(x[:, None], y)
    assert st_.check(_cand(_forward.KIND_HINGE, 3.5)) is not None
    assert st_.check(_cand(_forward.KIND_HINGE, 8.0)) is None
    st_.rss = [st_.rss[0], 0.999 * st_.rss[0]]  # 10·Δ_1 is 0.01·TSS
    assert st_.check(_cand(_forward.KIND_PAIR, 3.5)) is None
    assert st_.check(_cand(_forward.KIND_LINEAR, math.nan)) is not None
    st_.rss = [st_.rss[0], st_.check(_cand(_forward.KIND_LINEAR, math.nan)).rss]
    assert st_.check(_cand(_forward.KIND_LINEAR, math.nan)) is None  # 0 exactly


def test_a_scan_value_that_the_explicit_values_refuse_is_left_out(monkeypatch):
    """Pass 2 values the scan's best explicitly; here a scan that accepts every
    knot and favors the collinear knot at the repeated minimum, with bounds of
    0, loses that knot in pass 2, and the fit is the honest one."""
    x = np.array([0.0, 0.0, 1.0, 2.0, 3.5, 5.0, 6.0, 8.0, 9.0, 11.0])
    y = np.array([1.0, 0.0, 2.0, 1.0, 4.0, 3.0, 7.0, 9.0, 8.0, 12.0])
    kw = {"max_terms": 3, "minspan": 1, "endspan": 1}
    honest = _fit(x[:, None], y, **kw)
    real = _forward._scan.knot_scan
    calls = []

    def lying(x_, b, Q, E, split, **kw):
        out = real(x_, b, Q, E, split, **kw)
        calls.append(1)
        gain = np.where(np.asarray(split) == 2, float(np.sum(E**2)), out.gain)
        zero = np.zeros_like(out.ratio)
        return _forward._scan.KnotScan(np.ones_like(out.ratio), gain, zero, zero)

    monkeypatch.setattr(_forward._scan, "knot_scan", lying)
    fp = _fit(x[:, None], y, **kw)
    assert len(calls) == 1
    np.testing.assert_array_equal(fp.cuts, honest.cuts)


def test_a_candidate_that_fails_the_check_is_left_out(monkeypatch):
    """When the explicit rebuild of the chosen candidate fails its check, it is
    left out and its search is done again: the second best is chosen, and the
    third, which pass 1 had left out, is the new second (FWD-8)."""
    st_ = _two_covariates()
    _, _, second = st_.best()
    third = _explicit_best(st_, 3)[2]
    real = _forward._scan.rebuild
    calls = []

    def failing(Q, Y, columns):
        calls.append(1)
        return None if len(calls) == 1 else real(Q, Y, columns)

    monkeypatch.setattr(_forward._scan, "rebuild", failing)
    chosen, _, new_second = st_.best()
    assert chosen.order == second.order and len(calls) == 2
    assert new_second.order == third[1]
    assert new_second.reduction == pytest.approx(third[0], rel=1e-12)


def test_a_knot_within_its_bound_of_tau_is_decided_explicitly():
    """The adversarial review's case: 15 low values below a cluster at 50, a
    pair at the median in the model, step 8 (tau = 1e-5) and the knot x[7]
    with rho = tau + 5e-11, within the scan's bound of tau. Pass 2 decides
    LA-3 with the explicit rho, keeps the knot, and it is the best."""
    rng = np.random.default_rng(0)
    n, beta = 10_000, 0.5991315689034221
    x = np.sort(np.r_[beta * rng.uniform(0, 1, 15), 50 + rng.uniform(size=n - 15)])
    y = np.maximum(x - x[7], 0) + 1e-3 * rng.normal(size=n)
    st_ = _state(x[:, None], y, minspan=1, endspan=7)
    for v in (np.maximum(x - x[n // 2], 0), np.maximum(x[n // 2] - x, 0)):
        st_.Q = np.column_stack((st_.Q, _linalg.gram_schmidt(st_.Q, v).q))
    st_.E = _linalg.orthogonalize(st_.Q, st_.Yc)[0]
    st_.rss = [st_.rss[0]] * 7 + [float(np.sum(st_.E**2))]  # this is step 8
    kept = st_.search(0, 0, set())
    place = int(np.flatnonzero(st_.knots[0].knots == x[7])[0]) + 1
    low = [c for c in kept if c.order == (0, 0, place)]
    assert low and not low[0].sure  # the scan's bound straddles tau
    _, gain = _scan.exact_knot(st_.Q, st_.E, np.maximum(x - x[7], 0.0))
    assert abs(low[0].reduction - gain) <= 1e-8 * st_.rss[-1]  # the intercept path
    best = _forward._top_two(st_.refine(kept))[0]
    assert best.order == (0, 0, place) and best.sure and best.err == 0.0
    rho, gain = _scan.exact_knot(st_.Q, st_.E, np.maximum(x - x[7], 0.0))
    assert 0.0 < rho - 1e-5 < 1e-10
    chosen, _, second = st_.best()  # the whole step, with its check
    assert chosen == best and chosen.reduction == gain
    assert second.order == (0, 0, place - 1) and second.err == 0.0
    _, gain2 = _scan.exact_knot(st_.Q, st_.E, np.maximum(x - x[8], 0.0))
    assert second.reduction == gain2


def _fixed_passes(monkeypatch, scan, exact, top=None, steps=0):
    """A single-hinge search on x = 0..11 (x is in the model), whose knots are
    10, 9, ..., 1 at places 1 to 10. The scan gives, by place, the tuples
    (gain, err, ratio, ratio_err) of ``scan``, and ``exact_knot`` the pairs
    (rho, gain) of ``exact``; places that ``scan`` leaves out get a small gain.
    ``steps`` steps are done before (for the tau of LA-3). The check of the
    winner accepts. Returns the state and the list of places that pass 2
    valued."""
    x = np.arange(12.0)
    st_ = _state(x[:, None], 10 * np.sin(x), minspan=1, endspan=1)
    st_.Q = np.column_stack((st_.Q, _linalg.gram_schmidt(st_.Q, x).q))
    st_.E = _linalg.orthogonalize(st_.Q, st_.Yc)[0]
    st_.rss = st_.rss * (steps + 1)
    if top is not None:
        st_.max_legal = lambda: top
    st_.check = lambda c: object()  # the check has its own tests
    rows = [scan.get(i, (0.5, 0.01, 0.5, 0.0)) for i in range(1, 11)]
    gain, err, ratio, ratio_err = (np.array(c) for c in zip(*rows, strict=True))
    fake = _scan.KnotScan(ratio, gain, ratio_err, err)
    monkeypatch.setattr(_forward._scan, "knot_scan", lambda *a, **k: fake)
    valued = []

    def exact_knot(Q, E, h, w=None):
        place = 11 - int(x[np.flatnonzero(h)[0] - 1])
        valued.append(place)
        return exact.get(place, (0.5, 0.5))

    monkeypatch.setattr(_forward._scan, "exact_knot", exact_knot)
    return st_, valued


@pytest.mark.parametrize(
    ("scan", "exact", "best"),
    [
        (
            {1: (10.0, 0.5), 2: (9.8, 0.5), 3: (9.7, 0.05)},  # the floor is 9.5
            {1: 9.6, 2: 9.9, 3: 9.72},
            (2, 3),
        ),
        (
            {1: (10.0, 0.1), 2: (9.8, 0.1), 3: (9.5, 0.5)},  # 9.5 < floor 9.7 <= 10.0
            {1: 9.92, 2: 9.75, 3: 9.95},
            (3, 1),
        ),
    ],
)
def test_pass_2_ranks_by_the_explicit_values(monkeypatch, scan, exact, best):
    """Wide bounds: pass 1 keeps every knot whose upper bound reaches the
    second largest lower bound of the sure ones (also one whose scan value is
    below it), and only those are valued; the explicit values reorder them."""
    rows = {i: (g, e, 0.5, 0.0) for i, (g, e) in scan.items()}
    rows[4] = (3.0, 0.05, 0.5, 0.0)
    exact = {i: (0.5, g) for i, g in exact.items()}
    st_, valued = _fixed_passes(monkeypatch, rows, exact)
    assert [c.order[2] for c in st_.search(0, 0, set())] == [1, 2, 3]
    chosen, _, second = st_.best()
    assert (chosen.order[2], second.order[2]) == best and sorted(valued) == [1, 2, 3]
    assert (chosen.reduction, second.reduction) == tuple(exact[i][1] for i in best)


@pytest.mark.parametrize("steps", [0, 6])
def test_an_uncertain_rho_is_decided_in_pass_2(monkeypatch, steps):
    """The best scan value belongs to a knot whose rho is within its bound of
    tau; pass 2 finds rho below tau and drops it, and it does not count toward
    the lower bound that decides which knots pass 2 values. At steps 1 and 7
    tau is 0.01 (LA-3)."""
    tau = 0.01
    scan = {
        1: (20.0, 0.01, tau * (1 + 1e-9), tau * 1e-8),
        2: (9.9, 0.01, 0.5, 0.0),
        3: (9.0, 0.01, 0.5, 0.0),
    }
    exact = {1: (tau * (1 - 1e-9), 20.0), 2: (0.5, 9.9), 3: (0.5, 9.0)}
    st_, valued = _fixed_passes(monkeypatch, scan, exact, top=30.0, steps=steps)
    kept = st_.search(0, 0, set())
    assert [(c.order[2], c.sure) for c in kept] == [(1, False), (2, True), (3, True)]
    chosen, _, second = st_.best()
    assert (chosen.order[2], second.order[2]) == (2, 3) and sorted(valued) == [1, 2, 3]


@pytest.mark.parametrize(("value", "best"), [(10.0, (2, 3)), (9.93, (1, 2))])
def test_a_reduction_within_its_bound_of_max_legal(monkeypatch, value, best):
    """FWD-4 in pass 1: a knot with 9.8 ≤ MaxLegal = 9.95 < 10.0 (its bounds) is
    kept but not sure; pass 2 drops it when its explicit reduction is above
    MaxLegal, and chooses it when it is below."""
    scan = {1: (9.9, 0.1, 0.5, 0.0), 2: (9.5, 0.01, 0.5, 0.0), 3: (9.0, 0.01, 0.5, 0.0)}
    exact = {1: (0.5, value), 2: (0.5, 9.5), 3: (0.5, 9.0)}
    st_, _ = _fixed_passes(monkeypatch, scan, exact, top=9.95)
    kept = st_.search(0, 0, set())
    assert [(c.order[2], c.sure) for c in kept] == [(1, False), (2, True), (3, True)]
    chosen, _, second = st_.best()
    assert (chosen.order[2], second.order[2]) == best


@pytest.mark.parametrize(("value", "best"), [(0.05, (1, 2)), (0.0, (2, None))])
def test_a_reduction_within_its_bound_of_0(monkeypatch, value, best):
    """FWD-4 in pass 1: a knot with the scan's reduction 0 ± 0.1 is kept but not
    sure; pass 2 keeps it when its explicit reduction is positive, and drops it
    when that is 0. The other knots gain nothing and are dropped for sure."""
    zero = dict.fromkeys(range(3, 11), (0.0, 0.0, 0.5, 0.0))
    scan = zero | {1: (0.0, 0.1, 0.5, 0.0), 2: (0.02, 0.001, 0.5, 0.0)}
    st_, _ = _fixed_passes(monkeypatch, scan, {1: (0.5, value), 2: (0.5, 0.02)})
    assert [(c.order[2], c.sure) for c in st_.search(0, 0, set())] == [
        (2, True),
        (1, False),
    ]
    chosen, _, second = st_.best()
    assert (chosen.order[2], second and second.order[2]) == best


def _explicit_best(st_, top=2):
    """The best ``top`` of a step by the explicit values of every candidate:
    the spec's rules (FWD-3 to FWD-5, LA-3, LA-7) with no scan. The
    intercept is the only parent, in the table row of the step's search."""
    s, limit, found, row = len(st_.rss), st_.max_legal(), [], st_.rows[0]
    for j in range(st_.X.shape[1]):
        sr = st_.setup(0, j)
        if sr.pair and sr.lin > 0.0:
            found.append((sr.lin, (row, j, 0)))
        for i, t in enumerate(st_.knots[j].knots):
            rho, gain = _scan.exact_knot(sr.Q, sr.E, np.maximum(sr.x - t, 0.0))
            red = gain + sr.lin
            if not _linalg.knot_rejected(rho, s) and 0.0 < red <= limit:
                found.append((red, (row, j, i + 1)))
    return sorted(found, key=lambda c: (-c[0], c[1]))[:top]


def test_every_step_equals_the_explicit_choice():
    """A covariate with a few low values far below a tight cluster, next to an
    ordinary one: at each of 9 steps the chosen and the second candidate are
    those of the explicit values of every candidate, apart from near-ties."""
    rng = np.random.default_rng(11)
    n = 2000
    x0 = np.r_[rng.uniform(0, 1, 12), 40 + rng.uniform(size=n - 12)]
    X = np.column_stack((x0, rng.uniform(size=n)))
    y = np.maximum(x0 - 0.5, 0) + np.sin(6 * X[:, 1]) + 0.05 * rng.normal(size=n)
    st_ = _state(X, y)
    for _ in range(9):
        chosen, rb, second = st_.best()
        want = _explicit_best(st_)
        if want[0][0] - want[1][0] > 1e-7 * st_.rss[-1]:
            assert chosen.order == want[0][1]
            assert chosen.reduction == pytest.approx(want[0][0], rel=1e-12)
            assert second.reduction == pytest.approx(want[1][0], rel=1e-12)
        st_.add(chosen, rb)


@pytest.mark.parametrize("kw", [{"w": np.ones(4)}, {"fast_k": 20}])
def test_stage_2_limits(kw):
    X, y = np.arange(8.0).reshape(4, 2), np.arange(4.0)
    args = {"fast_k": 0} | kw
    with pytest.raises(NotImplementedError, match="stage 2"):
        _forward.forward_pass(X, y, **args)
    with pytest.raises(NotImplementedError, match="stage 2"):
        _forward.forward_pass(X, np.ones((4, 2)), fast_k=0)


@pytest.mark.parametrize(
    ("X", "y"),
    [
        (np.ones((3, 1)), np.ones(4)),
        (np.ones(3), np.ones(3)),
        (np.full((3, 1), np.inf), np.ones(3)),
        (np.ones((3, 1)), [0.0, np.nan, 1.0]),
        (np.ones((0, 1)), np.ones(0)),
    ],
)
def test_bad_input(X, y):
    with pytest.raises(ValueError, match="X"):
        _forward.forward_pass(X, y, fast_k=0)
