"""Tests of pymars/_forward.py (T11): every degree, Fast MARS, weights and
several responses.

The earth fixtures (T05) are compared in the matched and the earth-compatible
(defaults) modes at degrees 1 to 3 by the plan's rules ("What is compared,
and the tolerances", "Ties"): the steps exactly up to the first near-tie, the
RSS after each step within 1e-8 of the RSS before it (LA-5), and the
termination code when the whole path matched. earth counts cases where
pymars counts weight (W-2), so a weighted fit is compared with earth's fit of
the repeated rows, its "_repeated" fixture (W-1, W-3; the plan's "Sample
weights"). The queue with fast_k > 0, weights and several responses are
also compared with the reference.
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
from reference.test_reference import exact_basis, exact_rss

from pymars import _forward, _gcv, _linalg, _scan, _terms
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
        "fast_beta": a.get("fast.beta", 1.0),
    }


def _repeated(name):
    """The fixture of earth's fit of the repeated rows of a weighted fixture."""
    return name.replace("_matched", "_repeated_matched").replace(
        "_defaults", "_repeated_defaults"
    )


def _selected(name):
    """Earth's scaled X, numeric responses, not all constant, and no weights,
    or weights with a fixture of the repeated rows (module docstring)."""
    d = json.loads((FIXTURES_DIR / f"{name}.json").read_text(encoding="utf-8"))
    y, w = d["inputs"]["y"], d["inputs"]["weights"]
    if d.get("scale") is None or "error" in d["result"] or isinstance(y[0], str):
        return False
    if w is not None and not (FIXTURES_DIR / f"{_repeated(name)}.json").is_file():
        return False
    return len({json.dumps(v) for v in y}) > 1


# The dataset fixtures are S01 to S20; the lowercase names are extras, which a
# case-insensitive glob (Windows) would also match.
FIXTURES = sorted(
    f.stem
    for f in FIXTURES_DIR.glob("S*.json")
    if f.name[0] == "S" and _selected(f.stem)
)


def test_the_fixture_selection():
    at = [sum(f"_d{d}" in name for name in FIXTURES) for d in (2, 3)]
    assert len(FIXTURES) == 119 and at == [17, 2]
    assert {"S16_weighted_matched_d2", "S17_matched_d1"} <= set(FIXTURES)


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
    """FWD, STOP and FAST against earth; RESP-1 for S17 (two responses), and
    W-1 and W-3 for S13 and S16, with integer weights, zeros included,
    against earth's fit of the repeated rows."""
    d = load_fixture(name)
    X = np.asarray(d["inputs"]["X"], dtype=np.float64)
    y = np.asarray(d["inputs"]["y"], dtype=np.float64)
    w = d["inputs"]["weights"]
    kw = _params(d["earth_args"]) | {"fast_k": d["earth_args"].get("fast.k", 20)}
    fp = _forward.forward_pass(X, y, w, **kw, record_candidates=True)
    result = d["result"] if w is None else load_fixture(_repeated(name))["result"]
    ours = _steps(fp.dirs, fp.cuts)
    earth = _steps(result["dirs"], np.asarray(result["cuts"]))
    log = fp.candidates
    tie = log.second_rss - log.best_rss < 1e-7 * fp.rss[:-1]
    fwd_rss = result["fwd_rss"]
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
        assert fp.termination == result["termcond"]


def _col(y):
    return y[:, None]


def _queue_design(name):
    """Data and settings for test_the_queue_against_the_reference, with no
    near-tie between candidates. "binary": a covariate with the values -1 and
    1 has no knot, so its linear term enters first and is a parent with
    negative cases, which are inactive (KNOT-1) and do not count in N_b: 98
    of the 200 cases give the minspan 4, and all 200 would give 5 (SPAN-1).
    "linear option": the linear option of x1 on a hinge parent has its knot
    at the smallest x1 of all cases, not of the parent's active cases
    (FWD-6). "degree 3": two covariates, single hinges (the slots run ahead
    of M, FAST-4) and parents of degree 2 that hold both covariates (λ = -1,
    FAST-5). "ageing": the same data with other spans, fast_beta 2.5 (FAST-2)
    and adjust_endspan 0.5 (SPAN-4). "pair rule": x1 is x0 plus noise inside
    an interaction, on covariates with variance 1/12, where the variances of
    the parent's covariates decide the kind of a search (LA-7); in "pair
    rule, weighted", with weight 3 in the tails of x0 and 0.3 elsewhere, the
    weighted variances decide it (W-8). "window":
    fast_k = 1, which acts as 3 (FAST-3), with fast_beta 0.5, so the window
    leaves out rows of parents that could be searched. "weights": fast_k = 5
    at degree 3, two responses in different units (RESP-1, RESP-3) and
    weights that are not integers, so that N = Σw sets the spans and the
    knots (W-2, KNOT-6) and the weighted variances the kind of a search."""
    if name in ("window", "weights"):
        return _weighted_design(name)
    seed = {"binary": 14, "pair rule": 4, "pair rule, weighted": 4}.get(name, 0)
    rng = np.random.default_rng(seed)
    kw = {"max_degree": 2, "max_terms": 9, "thresh": 0.0}
    if name == "binary":
        x0, x1 = rng.choice([-1.0, 1.0], size=200), rng.uniform(size=200)
        y = 2 * x0 + 3 * x0 * np.maximum(x1 - 0.4, 0)
        return (
            np.column_stack((x0, x1)),
            _col(y + 0.05 * rng.normal(size=200)),
            None,
            kw,
        )
    if name == "linear option":
        X = rng.uniform(size=(100, 2))
        y = 3 * np.maximum(X[:, 0] - 0.5, 0) * X[:, 1] + 0.05 * rng.normal(size=100)
        return X, _col(y), None, kw | {"max_terms": 11, "auto_linpreds": False}
    if name.startswith("pair rule"):
        x0 = rng.uniform(size=100)
        X = np.column_stack(
            (x0, x0 + 0.003 * rng.normal(size=100), rng.uniform(size=100))
        )
        y = 5 * np.maximum(X[:, 2] - 0.5, 0) * (np.maximum(X[:, 1] - 0.5, 0) + x0)
        w = np.where(np.abs(x0 - 0.5) > 0.35, 3.0, 0.3)
        w = w if name.endswith("weighted") else None
        return (
            X,
            _col(y + 0.01 * rng.normal(size=100)),
            w,
            kw | {"minspan": 1, "endspan": 1},
        )
    X = rng.uniform(size=(80, 2))
    y = 10 * np.maximum(X[:, 0] - 0.4, 0) * np.maximum(X[:, 1] - 0.3, 0) + X[:, 0]
    kw |= {"max_degree": 3, "max_terms": 21}
    if name == "degree 3":
        kw |= {"minspan": 1, "endspan": 1}
    else:
        kw |= {"fast_beta": 2.5, "adjust_endspan": 0.5}
    return X, _col(y + 0.05 * rng.normal(size=80)), None, kw


def _weighted_design(name):
    rng = np.random.default_rng(7)
    n = 150
    X = rng.uniform(size=(n, 3))
    f = 4 * np.maximum(X[:, 0] - 0.3, 0) * np.maximum(X[:, 1] - 0.4, 0) + X[:, 2]
    Y = np.column_stack((f, 100 * np.sin(3 * X[:, 1]))) + 0.05 * rng.normal(size=(n, 2))
    kw = {"max_degree": 2, "max_terms": 21, "thresh": 0.0, "fast_k": 1}
    if name == "window":
        return X, Y[:, :1], None, kw | {"fast_beta": 0.5}
    kw |= {"max_degree": 3, "fast_k": 5, "max_terms": 25}
    return X, Y, rng.uniform(0.2, 3.0, size=n), kw


@pytest.mark.parametrize(
    "name",
    [
        "binary",
        "linear option",
        "degree 3",
        "ageing",
        "pair rule",
        "pair rule, weighted",
        "window",
        "weights",
    ],
)
def test_the_queue_against_the_reference(monkeypatch, name):
    """FAST-1 to FAST-5, and the searches of parents other than the intercept
    (FWD-2, FWD-6, KNOT-1 to KNOT-3, SPAN-1, SPAN-4, LA-7), with weights and
    two responses in "weights" (W-2, W-8, RESP-1),
    against the reference called as a black box with its trace: at every step
    the parents searched (entry e stands for slot e), and κ and λ of every
    entry after the search, λ within LA-5 of the TSS; the queue table and the
    order of the parents up to the first two computed λ within STOP-7's band
    of each other, which either program may order (two parents that reach
    the same product, as in "ageing"); then the whole record. λ is the best
    legal reduction of a parent, so it checks the search of every parent,
    also of those whose candidates do not win."""
    X, Y, w, kw = _queue_design(name)
    fast, real = [], _forward._Pass.best

    def best(st_):
        table = st_.table()
        out = real(st_)
        fast.append((table, list(st_.rows), st_.lam.copy(), st_.kap.copy()))
        return out

    monkeypatch.setattr(_forward._Pass, "best", best)
    fp = _forward.forward_pass(X, Y, w, **(KW | kw))
    Ys, n = np.ldexp(Y, mars_ref.y_scale_power(Y)), len(Y)
    wv = np.ones(n) if w is None else w
    tss = mars_ref.rss(np.ones((n, 1)), Ys, wv)
    N, tau = _gcv.total_weight(n, w)
    trace = []
    ref, _ = mars_ref.forward_pass(
        X,
        Ys,
        wv,
        {"fast_k": 0} | kw,
        N=N,
        tau_N=tau,
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


def _fit(X, y, w=None, **kw):
    X, y = np.asarray(X, float), np.asarray(y, float)
    return _forward.forward_pass(X, y, w, **(KW | kw))


def _twelve_hinges():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(300, 12))
    y = np.abs(X - 0.5) @ 0.8 ** np.arange(12) + 0.01 * rng.normal(size=300)
    return X, y


@pytest.mark.parametrize(
    ("fast_k", "terms"), [(1, 5), (2, 5), (3, 5), (5, 7), (10, 11), (0, None)]
)
@pytest.mark.parametrize(("thresh", "code"), [(0.0, 6), (0.001, 4)])
def test_the_window_ends_a_fit_of_pairs(fast_k, terms, thresh, code):
    """FAST-3 and FAST-6: at degree 1, with pairs only and fast_beta = 1, the
    pass ends at the first size M with M - 1 ≥ nu, since nu entries at
    max_degree then rank ahead of the intercept: 5, 7 and 11 terms for nu = 3,
    5 and 10, where fast_k = 1 and 2 act as 3. The step searches nothing, so
    STOP-4 ends the pass for thresh > 0 and STOP-2 for thresh = 0. Here each
    step takes a pair on a new covariate; fast_k = 0 goes on."""
    fp = _fit(*_twelve_hinges(), fast_k=fast_k, thresh=thresh, max_terms=41)
    if terms is None:
        assert fp.dirs.shape[0] > 11
    else:
        assert fp.dirs.shape[0] == terms and fp.termination == code
        assert np.all(fp.cuts[1::2] == fp.cuts[2::2])  # pairs


def test_unit_and_zero_weights():
    """W-5: weights all 1 give the fit of no weights, bit for bit. W-3: a row
    with zero weight is dropped before anything else, so that a far outlier
    in x and y with weight 0 changes no bit."""
    X, Y, _, kw = _weights_design("degree 2")
    base = _fit(X, Y, **kw)
    same = [_fit(X, Y, w=np.ones(120), **kw)]
    far = np.r_[X, [[1e6, -1e6, 0.5]]], np.r_[Y, [[1e9, -1e9]]]
    same.append(_fit(*far, w=np.r_[np.ones(120), 0.0], **kw))
    for fp in same:
        for field in ("dirs", "cuts", "parent", "step", "kept", "rss"):
            np.testing.assert_array_equal(getattr(fp, field), getattr(base, field))
        assert fp.termination == base.termination
        for a, b in zip(fp.candidates, base.candidates, strict=True):
            np.testing.assert_array_equal(a, b)


def _weights_design(name):
    """X, Y, integer weights and settings. "degree 2": several responses
    (RESP-1), degree 2 and the window of FAST-3. "exact fit": the data of
    scikit-learn's check_sample_weight_equivalence_on_dense_data for the
    classifier (15 cases, 30 covariates, 3 indicator responses, weights 0 to
    4), where step 5 reaches an exact fit and the linear candidates of x0
    and x23 both reduce the RSS to 0 up to rounding (issue #81). "exact fit,
    knot": 7 cases, 4 covariates and 3 indicator responses at degree 2,
    where step 1 fits exactly with a pair, and several knots do."""
    if name == "exact fit, knot":
        rng = np.random.RandomState(309)
        n, p = rng.randint(6, 16), rng.randint(1, 5)
        X, K = rng.rand(n, p), rng.randint(1, 4)
        Y = np.eye(K + 1)[rng.randint(0, K + 1, size=n)][:, :K]
        w, degree = rng.randint(0, 4, size=n), rng.randint(1, 3)
        return X, Y, w, {"max_degree": degree, "thresh": 0.0, "fast_k": 20}
    if name == "exact fit":
        rng = np.random.RandomState(42)
        X, y = rng.rand(15, 30), rng.randint(0, 3, size=15)
        return X, np.eye(3)[y], rng.randint(0, 5, size=15), {"thresh": 0.001}
    rng = np.random.default_rng(8)
    X = rng.uniform(size=(120, 3))
    Y = np.column_stack((np.sin(4 * X[:, 0]) * X[:, 1], X[:, 2] ** 2))
    Y = Y + 0.05 * rng.normal(size=Y.shape)
    w = np.random.default_rng(9).integers(1, 4, size=120)
    return X, Y, w, {"max_degree": 2, "fast_k": 4, "thresh": 0.0, "max_terms": 15}


@pytest.mark.parametrize("name", ["degree 2", "exact fit", "exact fit, knot"])
def test_integer_weights_are_repeated_rows(name):
    """W-1: integer weights (zeros included, W-3) give the record of the
    repeated rows, whose order is shuffled. "degree 2" has no near-tie, so the
    match is not luck. In the exact fits, candidates whose computed RSS is
    within the band EXACT_FIT·RSS_s of 0 count as exact fits: they tie
    exactly, and FWD-5 decides, the linear term of x0 in "exact fit" and one
    knot in "exact fit, knot", in both fits (issue #81)."""
    X, Y, w, kw = _weights_design(name)
    fp = _fit(X, Y, w=w.astype(float), **kw)
    perm = np.random.default_rng(0).permutation(int(w.sum()))
    rep = _fit(np.repeat(X, w, axis=0)[perm], np.repeat(Y, w, axis=0)[perm], **kw)
    for field in ("dirs", "cuts", "parent", "step", "kept"):
        np.testing.assert_array_equal(getattr(fp, field), getattr(rep, field))
    assert fp.termination == rep.termination
    assert np.all(np.abs(fp.rss - rep.rss) <= 1e-8 * np.r_[fp.rss[0], fp.rss[:-1]])
    for log in (fp.candidates, rep.candidates):
        assert np.all(log.second_rss >= 0.0)
    if name == "degree 2":
        gap = fp.candidates.second_rss - fp.candidates.best_rss
        assert np.all(gap >= 1e-7 * fp.rss[:-1])  # no near-tie


@pytest.mark.parametrize(("n", "a"), [(50, 1e-5), (2000, 1e-4)])
def test_a_small_kink_keeps_its_knot(n, a):
    """The band of exact fits is at the rounding level, so a real kink whose
    candidate RSS is about 1e-14·RSS_s is not tied with the others: y = x0 +
    a·(x0 - 0.5)₊ gets its pair at the largest x0 below 0.5, as the reference
    and the code without the band choose (FWD-4, FWD-5; #83)."""
    X = np.random.default_rng(1).uniform(size=(n, 2))
    y = X[:, 0] + a * np.maximum(X[:, 0] - 0.5, 0.0)
    fp = _fit(X, y, thresh=0.0)
    assert fp.dirs[1:3, 0].tolist() == [1, -1]
    assert fp.cuts[1, 0] == X[X[:, 0] <= 0.5, 0].max()


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


@pytest.mark.parametrize(
    ("y", "w", "tss"),
    [([2.0] * 5, None, 0.0), ([1.5], None, 0.0), ([0.0, 1.0, 2.0], [1, 1, 2], 0.6875)],
)
def test_a_degenerate_fit(y, w, tss):
    """EDGE-1, GCV-7: a constant response, a single case, or N = Σw ≤ 1 (W-2)
    with a response that is not constant, whose rss[0] is its weighted TSS."""
    w = None if w is None else np.array(w) / 4.0
    fp = _fit(np.arange(len(y), dtype=float)[:, None], y, w=w)
    assert fp.termination == Termination.DEGENERATE
    assert fp.dirs.tolist() == [[0]] and fp.cuts.tolist() == [[0.0]]
    assert fp.rss.tolist() == [tss] and fp.kept.tolist() == [0]
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


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize(
    ("design", "x_shift", "y_shift"),
    [
        ("one covariate", 2.0**46, 0),  # A of LA-7, in setup
        ("pairs", 2.0**36, 0),  # the first column of a pair
        ("linear", 2.0**24, 0),  # the column of the linear candidate
        ("pairs", 0, 2.0**44),  # the TSS
        ("interaction", 2.0**36, 0),  # b·(x - c) for a parent b other than 1
        ("interaction", 2.0**26, 2.0**34),
    ],
)
def test_exact_shifts(design, x_shift, y_shift, weighted):
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
    a shift does not change (FWD-6). With ``weighted`` the weights are
    multiples of 1/8 (W-8), and the shifted data get one more case: a copy of
    the first row with y = 2^52 and weight 2^-170, which changes no knot or
    term and no RSS by more than 1e-19, but which a shift of y by its first
    value would make the anchor of the centering (FWD-10)."""
    X, y = _on_binary_grids(design)
    kw = {"thresh": 0.0, "minspan": 1, "endspan": 1, "max_terms": 11}
    if design == "interaction":
        kw |= {"max_degree": 2, "auto_linpreds": False}
    w = None
    if weighted:
        w = np.random.default_rng(1).integers(1, 17, size=len(y)) / 8.0
    Xs, ys, ws = X + np.eye(X.shape[1])[0] * x_shift, y + y_shift, w
    if weighted:
        Xs, ys, ws = np.r_[Xs[:1], Xs], np.r_[2.0**52, ys], np.r_[2.0**-170, w]
    a = _fit(X, y, w=w, **kw)
    b = _fit(Xs, ys, w=ws, **kw)
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
        (_ns(penalty=2.0), True, 1.0, 11, Termination.GRSQ_NEG_INF),  # -inf
        # STOP-3 (v2): code 2 below -1000, not only at -inf
        (_ns(), True, 1000.0, 3, Termination.GRSQ_LOW),  # GRSq' = -999
        (_ns(), True, 1001.0, 3, Termination.GRSQ_LOW),  # -1000
        (_ns(), True, 1001.5, 3, Termination.GRSQ_NEG_INF),  # -1000.5
        (_ns(), None, 1.0e6, 3, Termination.GRSQ_NEG_INF),  # FAST-6: no candidate
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


def _merge_case(seed):
    """Discrete covariates, a product in the truth: x_i·x_j is often chosen from
    its two linear parents (FWD-12)."""
    rng = np.random.default_rng(seed)
    n = int(rng.choice([12, 20, 40, 100]))
    p = int(rng.choice([2, 3]))
    levels = int(rng.choice([2, 3, 4, 6, 10]))
    X = np.floor(rng.uniform(size=(n, p)) * levels) / levels
    if rng.uniform() < 0.5:
        X = X * float(rng.choice([1, 10, 1000]))
    y = sum(
        rng.normal() * np.maximum(X[:, i] - np.median(X[:, i]), 0) for i in range(p)
    )
    y = y + rng.normal() * np.prod(X[:, :2], axis=1) + 0.01 * rng.normal(size=n)
    return X, y, int(rng.choice([2, 3])), bool(rng.integers(2))


@pytest.mark.parametrize("seed", [6, 9, 74])
def test_the_second_best_does_not_merge_with_the_chosen(seed):
    """FWD-12, FWD-8: two linear candidates on the two linear parents of one
    product add the same row, so they are one candidate; the log's second best
    is another candidate. Before, the second best was the other occurrence at
    steps 4 and 6 of these fits (equal up to rounding)."""
    X, y, degree, auto = _merge_case(seed)
    fp = _fit(X, y, max_degree=degree, auto_linpreds=auto, thresh=0.0, max_terms=13)
    log = fp.candidates
    for s in range(len(log.best_rss)):
        if log.second_kind[s] != _forward.KIND_LINEAR:
            continue
        p, j = log.second_parent[s], log.second_variable[s]
        d, k = _terms.child_term(fp.dirs[p], fp.cuts[p], j, _terms.LINEAR)
        for t in np.flatnonzero(fp.step == s + 1):
            assert not (fp.dirs[t] == d).all() or not (fp.cuts[t] == k).all()


def test_two_searches_that_add_the_same_hinge_product_are_one_candidate(monkeypatch):
    """FWD-12: with h(x0 - t0) and h(x1 - t1) in the model, the single hinge on
    the first with covariate x1 and knot t1 and the one on the second with
    covariate x0 and knot t0 add the same row; the first in the order of FWD-5
    decides, and its reduction and legality hold for both. A pair search on one
    of the parents breaks the match, since the kinds differ."""
    rng = np.random.default_rng(2)
    X = rng.uniform(size=(30, 2))
    y = np.maximum(X[:, 0] - 0.4, 0) + np.maximum(X[:, 1] - 0.5, 0)
    st_ = _state(X, y + 0.1 * rng.normal(size=30), max_degree=2, minspan=1, endspan=1)
    t = [float(np.sort(X[:, j])[12]) for j in (0, 1)]
    for j in (0, 1):
        c = _forward._Candidate(1.0, (0, j, 1), 0, j, _forward.KIND_PAIR, t[j])
        st_.add(c, _scan.rebuild(st_.Q, st_.Yw, st_.columns(c)[0]))
    assert sorted(st_.parents()) == [0, 1, 2, 3, 4]
    k0, k1 = 1, 3  # the terms h(x0 - t0) and h(x1 - t1)
    place = int(np.flatnonzero(st_.knots_of(k0, 1).knots == t[1])[0]) + 1
    first = _forward._Candidate(
        1.0, (st_.rows[k0], 1, place), k0, 1, _forward.KIND_HINGE, t[1]
    )
    first = st_.keyed(first)
    assert len(st_.occurrences(first)) == 1  # the real rule: a pair search on k1
    monkeypatch.setattr(_forward._linalg, "pair_search", lambda *a, **k: False)
    occ = st_.occurrences(first)
    assert {(c.parent, c.variable, c.knot) for c in occ} == {
        (k0, 1, t[1]),
        (k1, 0, t[0]),
    }
    assert occ[0].order < occ[1].order and len({c.key for c in occ}) == 1
    got = [st_.decide(c) for c in occ]
    assert got[0] is not None
    assert got[1].reduction == pytest.approx(got[0].reduction, rel=1e-9)


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

    def failing(Q, Y, columns, sizes=None):
        calls.append(1)
        return None if len(calls) == 1 else real(Q, Y, columns, sizes)

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
    st_.E = _linalg.orthogonalize(st_.Q, st_.Yw)[0]
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
    st_.E = _linalg.orthogonalize(st_.Q, st_.Yw)[0]
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


@pytest.mark.parametrize(
    ("X", "y", "w", "match"),
    [
        (np.ones((3, 1)), np.ones(4), None, "X"),
        (np.ones(3), np.ones(3), None, "X"),
        (np.full((3, 1), np.inf), np.ones(3), None, "X"),
        (np.ones((3, 1)), [0.0, np.nan, 1.0], None, "X"),
        (np.ones((0, 1)), np.ones(0), None, "X"),
        (np.eye(3, 1), np.arange(3.0), [1.0, -1.0, 1.0], "w must"),
        (np.eye(3, 1), np.arange(3.0), [1.0, 1.0], "w must"),
        (np.eye(3, 1), np.arange(3.0), [0.0, 0.0, 0.0], "every weight is zero"),
        # EDGE-6: the scaled TSS underflows, possible only with extreme weights
        (np.eye(3, 1), [0.0, 1.0, 1.0], [1e-310, 1e-310, 2.0], "scale of y"),
        # the check comes before the degenerate test (N = 0.5 <= 1 here)
        (np.eye(2, 1), [0.0, 1.0], [1e-310, 0.5], "scale of y"),
    ],
)
def test_bad_input(X, y, w, match):
    with pytest.raises(ValueError, match=match):
        _forward.forward_pass(X, y, w, fast_k=0)


def _copy_of(kind, x0):
    """A covariate that copies x0 = 2^36 + u: the same, plus 2^-16 (the u columns
    are bitwise equal), or one unit in the last place off in one row."""
    if kind == "copy":
        return x0.copy()
    if kind == "plus 2^-16":
        return x0 + 2.0**-16
    x1 = x0.copy()
    x1[3] = np.nextafter(x1[3], np.inf)
    return x1


@pytest.mark.parametrize("kind", ["copy", "plus 2^-16", "one ulp"])
def test_rss_meets_la5_with_a_copy_of_a_large_mean_covariate(kind):
    """LA-5 (#85, #99, #104): x1 copies x0 = 2^36 + u, bitwise, plus 2^-16 in
    every row, or off by one unit in the last place in one row; degree 3 forms
    x0·x1·h(z) on the parent x0·h(z). The u columns of the first two are bitwise
    equal, so they are one symbol in the change of basis; the last is the copy
    u0 + d, with the difference d kept apart from the mean. Each RSS is within
    1e-8 of the RSS before its step of the exact rational RSS of the same terms
    (the last two were off by up to 2e-5 before)."""
    rng = np.random.default_rng(10)
    u, z = rng.uniform(size=40), rng.uniform(size=40)
    h = np.maximum(z - np.median(z), 0)
    x0 = 2.0**36 + u
    X = np.column_stack((x0, _copy_of(kind, x0), z))
    y = 10 * h + 6 * h * u + 4 * h * u**2
    fp = _forward.forward_pass(X, y, max_degree=3, thresh=0.0, fast_k=0, max_terms=9)
    B = exact_basis(X, fp.dirs, fp.cuts)
    for s in range(1, len(fp.rss)):
        cols = [k for k in range(len(fp.dirs)) if fp.step[k] <= s]
        exact = exact_rss([[row[k] for k in cols] for row in B], y, np.ones(40))
        assert abs(fp.rss[s] - exact) <= 1e-8 * fp.rss[s - 1]


@pytest.mark.parametrize("shifted", [False, True])
def test_a_linear_candidate_that_la4_finds_dependent_counts_as_0(monkeypatch, shifted):
    """LA-2: x2 is a combination of the intercept and x1 at the n cases (x2 =
    3·x1 + 1, or a copy of x1 at another shift, whose u columns are bitwise
    equal), so the column x2 is dependent, though not as a formula. In a pair
    search (forced here) it has the reduction 0 and no linear candidate, it is
    left out of G and of the chosen candidate's columns, and its rounding noise
    never decides. Before, the direction of its noise entered G."""
    rng = np.random.default_rng(4)
    a, z = np.floor(rng.uniform(size=24) * 4) / 4, rng.uniform(size=24)
    x1 = a + 2.0**30 if shifted else a
    X = np.column_stack((x1, z, x1 + 2.0**40 if shifted else 3 * a + 1))
    st_ = _state(X, np.sin(5 * z) + a, max_degree=2, minspan=1, endspan=1)
    lin = _forward._Candidate(1.0, (0, 0, 0), 0, 0, _forward.KIND_LINEAR, math.nan)
    st_.add(lin, _scan.rebuild(st_.Q, st_.Yw, st_.columns(lin)[0]))
    monkeypatch.setattr(_forward._linalg, "pair_search", lambda *a, **k: True)
    sr = st_.setup(0, 2)
    assert (sr.pair, sr.dep, sr.lin) == (True, True, 0.0) and sr.Q is st_.Q
    found = st_.search(0, 2, set())
    assert {c.kind for c in found} == {_forward.KIND_PAIR}  # no linear candidate
    pair = found[0]
    assert st_.columns(pair)[0].shape == (24, 1)
    assert st_.setup(0, 1).dep is False  # an independent covariate is a candidate
