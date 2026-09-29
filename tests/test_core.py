"""Tests of pymars/_core.py (docs/algorithm.md: CORE-1 to CORE-5, with W-3 to
W-8, LIMIT-1, GCV-4, GCV-7, EDGE-1, EDGE-6, FWD-11 and PRUNE-5 to PRUNE-8).

earth: ``fit_mars`` with a stub forward pass that returns earth's forward
terms, for ``components/pruning_fixed_basis`` and every S fit; and with the
real forward pass (T11 stage 1) for S01 and S04 at degree 1 in the matched
mode. Tolerances, from the plan's tolerance table: the exact terms and
subsets, relative 1e-8 for RSS and GCV, absolute 1e-8 for RSq and GRSq, and
normwise relative 1e-6 for the coefficients where κ(B) ≤ 1e5. earth reports
sums of squares below about 1e-10 as 0 (PRUNE-4), ignores weights that are
all equal (GCV-8) and counts cases in its GCV (W-2), so those values are not
compared. The reference's ``fit_mars`` (T06), called as a black box through
``MarsFit.from_dict`` (CORE-5, CORE-6), must give the same fits with the same
tolerances. Inputs are read-only where a test can make them so, so a write
into them fails.
"""

import dataclasses
import json
import math
import pickle
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from reference import mars_ref  # the oracle (tests/reference), as a black box

from pymars import _forward, _gcv, _terms
from pymars._core import (
    ForwardRecord,
    MarsFit,
    MarsParams,
    Termination,
    fit_mars,
)

FIXTURES = Path(__file__).resolve().parents[1] / "validation" / "fixtures"
S_FITS = sorted(
    p.stem
    for p in FIXTURES.glob("S*.json")
    if p.name.startswith("S")  # glob ignores case on Windows (s15_draws.json)
    and "error" not in json.loads(p.read_text(encoding="utf-8"))["result"]
)
MATCHED = ["S01_matched_d1"] + [
    f"S04_p{p}_n{n}_matched_d1" for p in ("05", "10") for n in ("0200", "1000")
]
FWD = ("max_degree", "thresh", "minspan", "endspan", "adjust_endspan")
FWD += ("auto_linpreds", "fast_k", "fast_beta", "max_terms", "penalty")


class Stub:
    """A forward pass that returns a fixed record and keeps its arguments."""

    def __init__(self, record):
        self.record, self.calls = record, []

    def __call__(self, X, Y, w=None, **kw):
        self.calls.append((X, Y, w, kw))
        return self.record


def _no_forward(*args, **kw):
    raise AssertionError("the forward pass ran")


def _record(dirs, cuts, rss=None, kept=None, code=Termination.TERM_LIMIT):
    """A forward record of the terms dirs and cuts, one term per step."""
    dirs = np.asarray(dirs, dtype=np.int8)
    cuts = np.where(np.abs(dirs) == 1, np.asarray(cuts, dtype=float), 0.0)
    M = len(dirs)
    kept = np.arange(M) if kept is None else np.asarray(kept)
    rss = np.linspace(2.0, 1.0, M) if rss is None else np.asarray(rss, dtype=float)
    parent = np.r_[-1, np.zeros(M - 1, dtype=np.int64)]
    dropped = np.setdiff1d(np.arange(M), kept)
    return _forward.ForwardPass(
        dirs, cuts, kept, dropped, parent, np.arange(M), rss, code, None
    )


# Pairs on x0 and x1, then one single hinge on each: 7 independent terms.
BASIS = _record(
    [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 0], [0, 1]],
    [[0, 0], [0.3, 0], [0.3, 0], [0, 0.5], [0, 0.5], [0.7, 0], [0, 0.2]],
)


def _data(seed, n=60, k=1):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, 2))
    B = _terms.basis_matrix(X, BASIS.dirs, BASIS.cuts)
    return X, B @ rng.normal(size=(7, k)) + 0.3 * rng.normal(size=(n, k))


def _frozen(*arrays):
    out = [None if a is None else np.array(a, dtype=float) for a in arrays]
    for a in out:
        if a is not None:
            a.flags.writeable = False
    return out


def _params(a):
    """earth's arguments (API-7) as MarsParams."""
    return MarsParams(
        max_degree=a["degree"],
        max_terms=a.get("nk"),
        penalty=a.get("penalty"),
        thresh=a.get("thresh", 0.001),
        minspan=a.get("minspan") or None,
        endspan=a.get("endspan") or None,
        adjust_endspan=a.get("Adjust.endspan", 2.0),
        auto_linpreds=a.get("Auto.linpreds", True),
        fast_k=a.get("fast.k", 20),
        pmethod=a.get("pmethod", "backward"),
    )


def _rel(actual, expected, rtol):
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    np.testing.assert_array_equal(np.isinf(actual), np.isinf(expected))
    ok = np.isfinite(expected)
    np.testing.assert_allclose(actual[ok], expected[ok], rtol=rtol, atol=0)


def _same(a, b):
    """Two fits, or their dicts, hold the same keys, dtypes and values."""
    a = a.to_dict() if isinstance(a, MarsFit) else a
    b = b.to_dict() if isinstance(b, MarsFit) else b
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], dict):
            _same(a[key], b[key])
        elif isinstance(a[key], np.ndarray):
            assert a[key].dtype == b[key].dtype, key
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        else:
            assert type(a[key]) is type(b[key]) and a[key] == b[key], key


def _arrays(d):
    """The arrays of a fit's dict, nested dicts included."""
    for value in d.values():
        if isinstance(value, dict):
            yield from _arrays(value)
        elif isinstance(value, np.ndarray):
            yield value


def _close(fit, other, rtol):
    """The same terms and records, and values within rtol (up to rounding)."""
    np.testing.assert_array_equal(fit.selected, other.selected)
    np.testing.assert_array_equal(fit.pruning.removed, other.pruning.removed)
    np.testing.assert_array_equal(fit.pruning.subsets, other.pruning.subsets)
    assert fit.n_eff == other.n_eff
    for key in ("rss", "gcv", "rsq", "grsq"):
        _rel(getattr(fit, key), getattr(other, key), rtol)
    _rel(fit.pruning.rss_per_size, other.pruning.rss_per_size, rtol)
    _rel(fit.pruning.gcv_per_size, other.pruning.gcv_per_size, rtol)
    np.testing.assert_allclose(fit.coef, other.coef, rtol=rtol, atol=rtol)


def _check_earth(fit, r, X, w):
    """The pruning record, the selected terms and the final fit against earth."""
    sets = [frozenset(np.flatnonzero(row)) for row in fit.pruning.subsets]
    assert sets == [frozenset(t - 1 for t in row if t) for row in r["prune_terms"]]
    rss = np.array(r["rss_per_subset"])
    shown = rss > 0.0
    scale = w[0] if w is not None and len(set(w)) == 1 else 1.0
    _rel(fit.pruning.rss_per_size[shown] / scale, rss[shown], 1e-8)
    if w is None:
        _rel(
            fit.pruning.gcv_per_size[shown], np.array(r["gcv_per_subset"])[shown], 1e-8
        )
        assert (fit.selected + 1).tolist() == r["selected_terms"]
    if (fit.selected + 1).tolist() != r["selected_terms"] or r["rss"] == 0.0:
        return
    coef = np.array(r["coef"]).reshape(fit.coef.shape)
    if np.linalg.cond(_terms.basis_matrix(X, fit.dirs, fit.cuts)) <= 1e5:
        assert np.linalg.norm(fit.coef - coef) <= 1e-6 * np.linalg.norm(coef)
    _rel(fit.rss / scale, r["rss"], 1e-8)
    if w is None:
        _rel(fit.gcv, r["gcv"], 1e-8)
        assert abs(fit.rsq - r["rsq"]) <= 1e-8 and abs(fit.grsq - r["grsq"]) <= 1e-8


def _check_reference(X, Y, w, params):
    """CORE-5, CORE-6: the reference's fit_mars through MarsFit.from_dict gives
    the core's forward terms up to the first near-tie of either log (the plan's
    Ties, where either choice passes); when the whole path matches, which it
    must without a near-tie, also the forward record, the pruning record, the
    selection and the final fit, with the plan's tolerances."""
    Y2 = np.asarray(Y, dtype=float).reshape(len(X), -1)
    ref = MarsFit.from_dict(mars_ref.fit_mars(X, Y2, w, params, record_candidates=True))
    fit = fit_mars(X, Y, w, params, record_candidates=True)
    f, g = fit.forward, ref.forward
    S = min(len(f.rss), len(g.rss)) - 1
    tie = np.zeros(S, dtype=bool)
    for rec in (f, g):
        tie |= rec.candidates.second_rss[:S] - rec.rss[1 : S + 1] < 1e-7 * rec.rss[:S]
    rows = f.step <= (np.argmax(tie) if tie.any() else S)
    np.testing.assert_array_equal(g.dirs[: rows.sum()], f.dirs[rows])
    np.testing.assert_array_equal(g.cuts[: rows.sum()], f.cuts[rows])
    same = f.dirs.shape == g.dirs.shape and np.array_equal(f.dirs, g.dirs)
    if not (same and np.array_equal(f.cuts, g.cuts)):
        assert tie.any()
        return
    for key in ("kept", "dropped", "parent", "step", "termination"):
        assert np.array_equal(getattr(f, key), getattr(g, key)), key
    before = np.r_[f.rss[0], f.rss[:-1]]
    assert np.all(np.abs(f.rss - g.rss) <= 1e-8 * before)
    a, b = f.candidates.second_rss, g.candidates.second_rss  # each within 1e-8
    np.testing.assert_array_equal(np.isinf(a), np.isinf(b))
    ok = np.isfinite(b)
    assert np.all(np.abs(a[ok] - b[ok]) <= 2e-8 * f.rss[:-1][ok])
    for key in ("removed", "subsets", "selected_size"):
        assert np.array_equal(getattr(fit.pruning, key), getattr(ref.pruning, key)), key
    _rel(fit.pruning.rss_per_size, ref.pruning.rss_per_size, 1e-8)
    _rel(fit.pruning.gcv_per_size, ref.pruning.gcv_per_size, 1e-8)
    np.testing.assert_array_equal(fit.selected, ref.selected)
    if np.linalg.cond(_terms.basis_matrix(X, fit.dirs, fit.cuts)) <= 1e5:
        assert np.linalg.norm(fit.coef - ref.coef) <= 1e-6 * np.linalg.norm(ref.coef)
    _rel([fit.rss, fit.gcv], [ref.rss, ref.gcv], 1e-8)
    assert np.allclose([fit.rsq, fit.grsq], [ref.rsq, ref.grsq], rtol=0, atol=1e-8)
    assert (fit.n_eff, fit.max_terms, fit.penalty) == (
        ref.n_eff,
        ref.max_terms,
        ref.penalty,
    )


# MarsParams (CORE-2)

BOUNDS = {"max_degree": 1, "max_terms": 1, "penalty": -1, "thresh": 0}
BOUNDS |= {"minspan": 1, "endspan": 1, "adjust_endspan": 0, "fast_k": 0}
BOUNDS |= {"fast_beta": 0, "nprune": 1, "pmethod": "none", "auto_linpreds": False}
BAD = [("max_degree", 0), ("max_terms", 0), ("minspan", 0), ("endspan", 0)]
BAD += [("fast_k", -1), ("nprune", 0), ("penalty", -0.5), ("penalty", -2)]
BAD += [("penalty", math.inf), ("penalty", math.nan), ("thresh", -1e-300)]
BAD += [("thresh", math.inf), ("adjust_endspan", -0.1), ("adjust_endspan", math.nan)]
BAD += [("fast_beta", -1e-9), ("fast_beta", -math.inf), ("thresh", 10**400)]
BAD += [("pmethod", "forward"), ("pmethod", "Backward")]
WRONG_TYPE = [("max_degree", True), ("max_terms", False), ("minspan", True)]
WRONG_TYPE += [("endspan", 2.0), ("fast_k", "20"), ("nprune", 1.0)]
WRONG_TYPE += [("max_degree", None), ("fast_k", None), ("penalty", "2")]
WRONG_TYPE += [("thresh", None), ("adjust_endspan", Decimal(2)), ("fast_beta", 1j)]
WRONG_TYPE += [("auto_linpreds", 1), ("auto_linpreds", None), ("pmethod", None)]
WRONG_TYPE += [("pmethod", 0), ("thresh", np.True_), ("adjust_endspan", None)]
WRONG_TYPE += [("fast_beta", None)]


def test_params_defaults_and_bounds():
    """CORE-2: the defaults, and every field at its bound."""
    assert dataclasses.asdict(MarsParams()) == {
        "max_degree": 1,
        "max_terms": None,
        "penalty": None,
        "thresh": 0.001,
        "minspan": None,
        "endspan": None,
        "adjust_endspan": 2.0,
        "auto_linpreds": True,
        "fast_k": 20,
        "fast_beta": 1.0,
        "pmethod": "backward",
        "nprune": None,
    }
    assert dataclasses.asdict(MarsParams(**BOUNDS)) == BOUNDS
    assert MarsParams(penalty=0).penalty == 0.0


@pytest.mark.parametrize(("name", "value"), BAD + WRONG_TYPE)
def test_params_errors_name_the_field(name, value):
    with pytest.raises(ValueError, match=f"MarsParams.{name} "):
        MarsParams(**{name: value})


def test_params_types():
    """An int field takes numpy integers, a float field any real number (a bool
    too, as the reference reads CORE-2, #44), and the fields hold plain Python
    values; the dataclass is frozen and compares by value; an error says what
    the field allows."""
    with pytest.raises(ValueError, match=r"MarsParams\.penalty must be -1, or finite"):
        MarsParams(penalty=-2)
    with pytest.raises(ValueError, match=r"MarsParams\.thresh must be finite and at"):
        MarsParams(thresh=-1)
    p = MarsParams(max_degree=np.int8(2), max_terms=np.uint16(9), nprune=np.int64(3))
    q = MarsParams(max_degree=2, max_terms=9, nprune=3)
    assert p == q and hash(p) == hash(q) and type(p.max_terms) is int
    f = MarsParams(penalty=3, thresh=Fraction(1, 4), fast_beta=np.float32(0.5))
    assert (f.penalty, f.thresh, f.fast_beta) == (3.0, 0.25, 0.5)
    assert type(f.penalty) is float and type(f.fast_beta) is float
    assert MarsParams(thresh=True).thresh == 1.0
    assert MarsParams(auto_linpreds=np.False_).auto_linpreds is False
    assert MarsParams(max_terms=10**20).max_terms == 10**20
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.max_degree = 1


# The records and their dict form (CORE-3, CORE-5)


def _fits():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(80, 3))
    y = np.sin(4 * X[:, 0]) + X[:, 1] + 0.1 * rng.normal(size=80)
    p = MarsParams(fast_k=0, thresh=0.0)
    return {
        "log": fit_mars(X, y, None, p, record_candidates=True),
        "bare": fit_mars(X, y, None, p),
        "intercept": fit_mars(X, np.full(80, 2.0), None, p),
        "intercept, log": fit_mars(X[:1], y[:1], None, p, record_candidates=True),
    }


FITS = _fits()
SPEC = [  # CORE-3: key, dtype and shape (from M, p, K, M_a, M_f, S)
    ("dirs", np.int8, "M p"),
    ("cuts", np.float64, "M p"),
    ("coef", np.float64, "M K"),
    ("selected", np.int64, "M"),
    ("forward.dirs", np.int8, "Ma p"),
    ("forward.cuts", np.float64, "Ma p"),
    ("forward.kept", np.int64, "Mf"),
    ("forward.dropped", np.int64, "Md"),
    ("forward.parent", np.int64, "Ma"),
    ("forward.step", np.int64, "Ma"),
    ("forward.rss", np.float64, "S1"),
    ("pruning.removed", np.int64, "Mf1"),
    ("pruning.rss_per_size", np.float64, "Mf"),
    ("pruning.gcv_per_size", np.float64, "Mf"),
    ("pruning.subsets", np.bool_, "Mf Mf"),
]
LOG = [("best_rss", np.float64), ("second_rss", np.float64)]
LOG += [("second_parent", np.int64), ("second_variable", np.int64)]
LOG += [("second_knot", np.float64), ("second_kind", np.int8)]


@pytest.mark.parametrize("case", FITS)
def test_to_dict_and_from_dict(case):
    """CORE-5: the keys, the dtypes and shapes of CORE-3 and the integer code;
    the round trip, also through JSON and pickle; new, read-only arrays."""
    fit = FITS[case]
    d = fit.to_dict()
    assert set(d) == {f.name for f in dataclasses.fields(MarsFit)}
    fwd, prn = d["forward"], d["pruning"]
    M, p = d["dirs"].shape
    n = {"M": M, "p": p, "K": d["coef"].shape[1], "Ma": len(fwd["dirs"])}
    n |= {"Mf": len(fwd["kept"]), "S1": len(fwd["rss"])}
    n |= {"Md": n["Ma"] - n["Mf"], "Mf1": n["Mf"] - 1}
    for key, dtype, shape in SPEC:
        a = d if "." not in key else d[key.split(".")[0]]
        a = a[key.split(".")[-1]]
        assert a.dtype == dtype and a.shape == tuple(n[s] for s in shape.split()), key
    for key in ("rss", "gcv", "rsq", "grsq", "n_eff", "penalty"):
        assert type(d[key]) is float
    assert type(d["max_terms"]) is int and type(prn["selected_size"]) is int
    assert type(fwd["termination"]) is int
    if fwd["candidates"] is not None:
        for key, dtype in LOG:
            a = fwd["candidates"][key]
            assert a.dtype == dtype and a.shape == (n["S1"] - 1,), key
    assert ("log" in case) == (fwd["candidates"] is not None)
    _same(MarsFit.from_dict(d), fit)
    assert all(a.flags.writeable for a in _arrays(d))  # new arrays, not the fit's
    if n["S1"] > 1 and fwd["candidates"] is not None:  # a NaN knot (none) reads back
        e = fit.to_dict()
        e["forward"]["candidates"]["second_knot"][0] = np.nan
        _same(MarsFit.from_dict(e), e)
    d["coef"][0, 0] += 1.0
    assert d["coef"][0, 0] != fit.coef[0, 0] and not fit.coef.flags.writeable
    assert not (fit.forward.cuts.flags.writeable or fit.pruning.subsets.flags.writeable)
    j = json.loads(json.dumps(fit.to_dict(), default=lambda a: a.tolist()))
    if j["forward"]["candidates"] is None:  # a missing key reads as None
        del j["forward"]["candidates"]
    _same(MarsFit.from_dict(j), fit)
    _same(pickle.loads(pickle.dumps(fit)), fit)


def _set(path, value):
    def change(d):
        *head, last = path.split(".")
        for key in head:
            d = d[key]
        if value is KeyError:
            del d[last]
        else:
            d[last] = value(d[last]) if callable(value) else value

    return change


@pytest.mark.parametrize(
    "change",
    [
        _set("rss", KeyError),
        _set("extra", 1.0),
        _set("coef", lambda a: a[:-1]),
        _set("dirs", lambda a: a + 0.5),
        _set("selected", lambda a: a[::-1]),
        _set("forward.termination", 9),
        _set("forward.candidates.best_rss", lambda a: a[1:]),
    ],
)
def test_from_dict_rejects_a_bad_dict(change):
    """CORE-3 to CORE-5: a missing or an extra key, a wrong shape, a value that
    does not convert exactly to its dtype, selected out of order, a code that
    CORE-4 does not define, and a log of the wrong length."""
    d = FITS["log"].to_dict()
    change(d)
    with pytest.raises(ValueError):
        MarsFit.from_dict(d)


def test_records_are_frozen():
    """CORE-3: MarsFit and its records are frozen dataclasses."""
    fit = FITS["log"]
    for rec in (fit, fit.forward, fit.pruning):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(rec, dataclasses.fields(rec)[0].name, None)


def test_bad_records():
    fp = _forward.forward_pass(np.eye(4, 1), np.arange(4.0), fast_k=0)
    with pytest.raises(ValueError, match="CandidateLog"):
        ForwardRecord(**fp._replace(candidates={"best_rss": []})._asdict())
    with pytest.raises(ValueError, match="covariate"):
        ForwardRecord(**fp._replace(dirs=fp.dirs[:, :0], cuts=fp.cuts[:, :0])._asdict())
    with pytest.raises(ValueError, match="rss"):
        ForwardRecord(**fp._replace(rss=fp.rss[:0])._asdict())
    with pytest.raises(ValueError, match="ForwardRecord"):
        dataclasses.replace(FITS["log"], forward=fp)
    with pytest.raises(TypeError, match="MarsParams"):
        fit_mars(*_data(0), None, {"max_degree": 1})
    with pytest.raises(TypeError):  # CORE-1: record_candidates is keyword-only
        fit_mars(*_data(0), None, MarsParams(fast_k=0), True)


# fit_mars against earth (CORE-1, PRUNE-5 to PRUNE-8)


@pytest.mark.parametrize("case", ["one_response", "several_responses"])
def test_component_fixture(monkeypatch, load_fixture, case):
    """The pruning fixture's degree-2 bases, pruned by earth:::pruning.pass."""
    fx = load_fixture("components/pruning_fixed_basis")
    c = fx[case]
    monkeypatch.setattr(_forward, "forward_pass", Stub(_record(c["dirs"], c["cuts"])))
    X, Y = _frozen(fx["X"], c["y"])
    fit = fit_mars(X, Y, None, MarsParams(max_degree=2, penalty=fx["penalty"]))
    sets = [frozenset(np.flatnonzero(row)) for row in fit.pruning.subsets]
    assert sets == [frozenset(int(t) - 1 for t in row if t) for row in c["prune_terms"]]
    _rel(fit.pruning.rss_per_size, c["rss_per_subset"], 1e-8)
    _rel(fit.pruning.gcv_per_size, c["gcv_per_subset"], 1e-8)
    assert (fit.selected + 1).tolist() == c["selected_terms"]


@pytest.mark.parametrize("name", S_FITS)
def test_s_fixture_through_the_core(monkeypatch, load_fixture, name):
    """earth's forward terms of every S fit, pruned and fitted by the core; the
    stub gets the rows with positive weight (W-3) and the resolved values."""
    d = load_fixture(name)
    r, inp = d["result"], d["inputs"]
    y = inp["y"]
    if isinstance(y[0], str):  # a factor: one indicator per level (RESP-1)
        y = np.column_stack([[v == level for v in y] for level in r["levels"]])
    X, Y, w = _frozen(inp["X"], y, inp["weights"])
    fwd = _record(r["dirs"], r["cuts"], r["fwd_rss"], code=Termination(r["termcond"]))
    stub = Stub(fwd)
    monkeypatch.setattr(_forward, "forward_pass", stub)
    params = _params(d["earth_args"])
    fit = fit_mars(X, Y, w, params)
    if not stub.calls:  # a constant response: EDGE-1, a departure (GCV-7)
        assert np.all(Y[0] == Y) and fit.forward.termination == Termination.DEGENERATE
        return
    Xs, Ys, ws, kw = stub.calls[0]
    keep = slice(None) if w is None else w > 0
    np.testing.assert_array_equal(Xs, X[keep])
    np.testing.assert_array_equal(Ys, Y[keep].reshape(len(Ys), -1))
    assert (ws is None) == (w is None) and kw["max_terms"] == fit.max_terms
    assert fit.n_eff == _gcv.total_weight(len(Xs), ws)[0]
    assert fit.forward.termination == r["termcond"]
    _check_earth(fit, r, X, w)


@pytest.mark.parametrize("name", MATCHED)
def test_matched_fits_against_earth(load_fixture, name):
    """End to end with the stage-1 forward pass: the forward steps up to the
    first near-tie (the plan's Ties), where either choice passes; when the
    whole path matches, which it must without a near-tie, also the termination,
    the RSS of each step, the pruning record, the selected terms and the
    coefficients. The reference gives the same fit (CORE-6)."""
    d = load_fixture(name)
    r = d["result"]
    X, y = _frozen(d["inputs"]["X"], d["inputs"]["y"])
    _check_reference(X, y, None, _params(d["earth_args"]))
    fit = fit_mars(X, y, None, _params(d["earth_args"]), record_candidates=True)
    f, log = fit.forward, fit.forward.candidates
    tie = log.second_rss - log.best_rss < 1e-7 * f.rss[:-1]
    rows = f.step <= (np.argmax(tie) if tie.any() else len(tie))
    dirs, cuts = np.array(r["dirs"]), np.where(np.abs(r["dirs"]) == 1, r["cuts"], 0.0)
    np.testing.assert_array_equal(f.dirs[rows], dirs[: rows.sum()])
    np.testing.assert_array_equal(f.cuts[rows], cuts[: rows.sum()])
    same = f.dirs.shape == dirs.shape and np.array_equal(f.dirs, dirs)
    same = same and np.array_equal(f.cuts, cuts)
    assert same or tie.any()
    if not same:
        return
    assert f.termination == r["termcond"] and f.kept.size == len(r["dirs"])
    last = [np.flatnonzero(f.step == s)[-1] for s in range(len(f.rss))]
    fwd_rss = np.array(r["fwd_rss"])[last]
    assert np.all(np.abs(f.rss - fwd_rss) <= 1e-8 * np.r_[f.rss[0], f.rss[:-1]])
    _check_earth(fit, r, X, None)


# Weights through the core (W-3 to W-8)


def test_zero_weights_are_rows_removed(monkeypatch):
    X, Y = _data(1)
    w = np.random.default_rng(1).integers(0, 3, 60).astype(float)
    X, Y, w = _frozen(X, Y, w)
    stub = Stub(BASIS)
    monkeypatch.setattr(_forward, "forward_pass", stub)
    fit = fit_mars(X, Y, w, MarsParams())
    keep = w > 0
    _same(fit, fit_mars(X[keep], Y[keep], w[keep], MarsParams()))
    np.testing.assert_array_equal(stub.calls[0][0], X[keep])
    np.testing.assert_array_equal(stub.calls[0][2], w[keep])
    assert fit.n_eff == w.sum() and 0 < keep.sum() < 60
    # W-5: weights that are all 1 give the fit of no weights, bit for bit.
    _same(fit_mars(X, Y, np.ones(60), MarsParams()), fit_mars(X, Y, None, MarsParams()))


@pytest.mark.parametrize("k", [1, 3])
def test_integer_weights_are_repeated_rows(monkeypatch, k):
    X, Y = _data(2, k=k)
    w = np.random.default_rng(2).integers(1, 4, 60)
    monkeypatch.setattr(_forward, "forward_pass", Stub(BASIS))
    fit = fit_mars(X, Y, w.astype(float), MarsParams())
    rep = fit_mars(np.repeat(X, w, axis=0), np.repeat(Y, w, axis=0), None, MarsParams())
    _close(fit, rep, 1e-9)


@pytest.mark.parametrize("c", [3.0, 0.37])
def test_rescaled_weights(monkeypatch, c):
    """W-8: c·w acts as c copies of the data. The subsets do not change, the
    sums of squares and N scale by c, and the GCV follows GCV-2 with c·N."""
    X, Y = _data(3)
    w = np.random.default_rng(3).uniform(0.5, 2.0, 60)
    monkeypatch.setattr(_forward, "forward_pass", Stub(BASIS))
    fit, base = fit_mars(X, Y, c * w, MarsParams()), fit_mars(X, Y, w, MarsParams())
    np.testing.assert_array_equal(fit.pruning.subsets, base.pruning.subsets)
    _rel(fit.pruning.rss_per_size, c * base.pruning.rss_per_size, 1e-12)
    assert fit.n_eff == math.fsum(c * w)
    gcv = _gcv.gcv(fit.pruning.rss_per_size, np.arange(1, 8), 2.0, fit.n_eff)
    _rel(fit.pruning.gcv_per_size, gcv, 1e-15)
    if c == 3.0:
        tile = [np.tile(a, (3, 1)[: a.ndim]) for a in (X, Y, w)]
        _close(fit, fit_mars(*tile, MarsParams()), 1e-10)


# Degenerate fits (EDGE-1, GCV-7), the term limit (LIMIT-2) and EDGE-6

DEGENERATE = {
    "one case": ([[1.0, 2.0]], [3.0], None, 0.0),
    "constant": (np.eye(8, 2), np.full(8, 2.5), None, 0.0),
    "two constant, weighted": (np.eye(8, 2), [[2.5, -1.0]] * 8, np.arange(1, 9), 0.0),
    "N = 1": ([[0.0], [1.0], [2.0]], [0.0, 1.0, 2.0], [0.25, 0.25, 0.5], 0.6875),
    "N < 1 after the drop": (
        [[0.0], [1.0], [2.0]],
        [0.0, 1.0, 2.0],
        [0.3, 0, 0.4],
        None,
    ),
    # EDGE-6 at its bound: the TSS is the smallest positive normal float64.
    "N < 1, TSS 2^-1022": ([[0.0], [1.0]], [0.0, 1.0], [2.0**-1021] * 2, 2.0**-1022),
}


@pytest.mark.parametrize("case", DEGENERATE)
@pytest.mark.parametrize("record", [False, True])
def test_degenerate_fits(monkeypatch, case, record):
    """The intercept alone, with the weighted mean, gcv +∞, rsq and grsq 0, code
    DEGENERATE and the trivial records; the forward pass does not run. The
    reference gives the same fit (CORE-6)."""
    X, Y, w, tss = DEGENERATE[case]
    X, Y, w = _frozen(X, Y, w)
    monkeypatch.setattr(_forward, "forward_pass", _no_forward)
    fit = fit_mars(X, Y, w, MarsParams(), record_candidates=record)
    Y2, wv = Y.reshape(len(Y), -1), np.ones(len(Y)) if w is None else w
    tss = _gcv.tss(Y2[wv > 0], wv[wv > 0]) if tss is None else tss
    f, prn = fit.forward, fit.pruning
    assert f.termination == Termination.DEGENERATE and f.dirs.shape == (1, X.shape[1])
    assert (f.kept.tolist(), f.dropped.tolist()) == ([0], [])
    assert (f.parent.tolist(), f.step.tolist()) == ([-1], [0])
    assert (f.candidates is not None) == record
    assert record is False or f.candidates.best_rss.shape == (0,)
    assert (prn.removed.shape, prn.subsets.tolist(), prn.selected_size) == (
        (0,),
        [[True]],
        1,
    )
    assert fit.selected.tolist() == [0] and fit.dirs.shape == (1, X.shape[1])
    np.testing.assert_allclose(
        fit.coef[0], np.average(Y2, axis=0, weights=wv), rtol=1e-15
    )
    for value in (f.rss[0], prn.rss_per_size[0], fit.rss):
        assert value == pytest.approx(tss, rel=1e-14, abs=0)
    assert (fit.gcv, prn.gcv_per_size[0], fit.rsq, fit.grsq) == (
        math.inf,
        math.inf,
        0,
        0,
    )
    assert fit.n_eff == math.fsum(wv)
    if record:
        _check_reference(X, Y, w, MarsParams())


@pytest.mark.parametrize("max_terms", [1, 2])
def test_no_room(max_terms):
    """max_terms ≤ 2: no forward step, code NO_ROOM, and the intercept model
    with its finite GCV, not a degenerate fit; the reference agrees."""
    rng = np.random.default_rng(4)
    X, y = _frozen(rng.uniform(size=(30, 2)), rng.normal(size=30))
    fit = fit_mars(X, y, None, MarsParams(max_terms=max_terms, fast_k=0))
    assert fit.forward.termination == Termination.NO_ROOM and fit.max_terms == max_terms
    assert fit.selected.tolist() == [0] and fit.pruning.gcv_per_size.shape == (1,)
    assert fit.rss == pytest.approx(_gcv.tss(y), rel=1e-14)
    assert fit.gcv == _gcv.gcv(fit.rss, 1, 2.0, 30.0) and fit.rsq == fit.grsq == 0.0
    assert fit.coef[0, 0] == pytest.approx(np.mean(y), rel=1e-14)
    _check_reference(X, y, None, MarsParams(max_terms=max_terms, fast_k=0))


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_a_power_of_two_on_y_changes_no_bit(sign):
    """EDGE-6: y of size 1e-170 has a TSS that underflows unscaled; its fit is
    that of y·2^600 with every value on the scale of y multiplied back, and
    the reference's fit. D is the largest |Y|, so the sign does not matter."""
    x = _frozen(np.arange(8.0)[:, None])[0]
    p = MarsParams(fast_k=0, minspan=1, endspan=1, thresh=0.0)
    y = [0.0] * 7 + [sign * 1e-170]
    tiny = fit_mars(x, y, None, p, record_candidates=True)
    big = fit_mars(x, np.array(y) * 2.0**600, None, p, record_candidates=True)
    assert tiny.forward.termination == big.forward.termination != Termination.DEGENERATE
    assert tiny.forward.dirs.shape[0] > 1
    back = big.to_dict()
    back["coef"] = np.ldexp(back["coef"], -600)
    for key in ("rss", "gcv"):
        back[key] = float(np.ldexp(back[key], -1200))
    for key in ("rss_per_size", "gcv_per_size"):
        back["pruning"][key] = np.ldexp(back["pruning"][key], -1200)
    back["forward"]["rss"] = np.ldexp(back["forward"]["rss"], -1200)
    log = back["forward"]["candidates"]
    for key in ("best_rss", "second_rss"):
        log[key] = np.ldexp(log[key], -1200)
    _same(tiny, back)
    _check_reference(x, y, None, p)


@pytest.mark.parametrize(
    ("Y", "w", "j"),
    [
        (np.column_stack([np.full(6, 8.0), [0, 1e-300, 0, 0, 0, 0]]), None, -3),
        (np.array([0.0, 1.0, 1.0]), np.array([1e-310, 1e-310, 2.0]), 0),
    ],
)
def test_a_tss_out_of_range_raises(monkeypatch, Y, w, j):
    """EDGE-6: a scaled TSS that is not a positive normal number, for a tiny
    response beside a constant one, and for extreme weights; the power of 2 is
    the j with D·2^j in [1, 2), D = max |Y|. The reference raises too."""
    monkeypatch.setattr(_forward, "forward_pass", _no_forward)
    X = np.arange(len(Y), dtype=float)[:, None]
    with pytest.raises(ValueError, match=rf"scale of y or of the weights.*y·2\^{j} is"):
        fit_mars(X, Y, w, MarsParams())
    with pytest.raises(ValueError, match="scale of y or of the weights"):
        mars_ref.fit_mars(X, Y.reshape(len(X), -1), w, MarsParams())


# The kept terms, pmethod and nprune, and the resolved values


def test_pruning_indices_are_kept_forward_indices(monkeypatch):
    """FWD-11, PRUNE-2: term 3 repeats term 1 and is dropped; the pruning pass
    sees the 5 kept terms, and pruning index m is forward index kept[m]."""
    dirs = [[0, 0], [1, 0], [-1, 0], [1, 0], [0, 1], [0, -1]]
    cuts = [[0, 0], [0.3, 0], [0.3, 0], [0.3, 0], [0, 0.5], [0, 0.5]]
    rec = _record(dirs, cuts, kept=[0, 1, 2, 4, 5])
    monkeypatch.setattr(_forward, "forward_pass", Stub(rec))
    X = np.random.default_rng(5).uniform(size=(50, 2))
    B = _terms.basis_matrix(X, rec.dirs, rec.cuts)
    y = B @ [1.0, 4.0, -3.0, 0.0, 5.0, 2.0] + 0.01 * np.sin(np.arange(50))
    fit = fit_mars(X, y, None, MarsParams())
    selected = [0, 1, 2, 4, 5]
    assert fit.pruning.rss_per_size.shape == (5,) and fit.selected.tolist() == selected
    np.testing.assert_array_equal(fit.dirs, rec.dirs[selected])
    coef = np.linalg.lstsq(B[:, selected], y, rcond=None)[0]
    np.testing.assert_allclose(fit.coef[:, 0], coef, rtol=1e-9)
    first = fit_mars(X, y, None, MarsParams(pmethod="none", nprune=4))
    assert first.selected.tolist() == [0, 1, 2, 4]


@pytest.mark.parametrize(
    ("pmethod", "nprune"),
    [("backward", None), ("backward", 2), ("none", None), ("none", 4), ("none", 50)],
)
def test_pmethod_and_nprune(monkeypatch, pmethod, nprune):
    """PRUNE-5 to PRUNE-8: nprune and pmethod change only the selection, and the
    statistics describe the selected terms."""
    X, Y = _data(6)
    monkeypatch.setattr(_forward, "forward_pass", Stub(BASIS))
    fit = fit_mars(X, Y, None, MarsParams(pmethod=pmethod, nprune=nprune))
    free = fit_mars(X, Y, None, MarsParams()).to_dict()["pruning"]
    prn = fit.to_dict()["pruning"]
    m, limit = prn.pop("selected_size"), min(7, nprune or 7)
    free.pop("selected_size")
    _same(prn, free)
    if pmethod == "backward":
        assert m == 1 + np.argmin(fit.pruning.gcv_per_size[:limit])
        assert (
            fit.selected.tolist() == np.flatnonzero(fit.pruning.subsets[m - 1]).tolist()
        )
    else:
        assert m == limit and fit.selected.tolist() == list(range(limit))
    B = _terms.basis_matrix(X, fit.dirs, fit.cuts)
    coef = np.linalg.lstsq(B, Y, rcond=None)[0]
    np.testing.assert_allclose(fit.coef, coef, rtol=1e-9, atol=1e-12)
    rss = float(np.sum((Y - B @ coef) ** 2))
    _rel(fit.rss, rss, 1e-10)
    assert fit.gcv == _gcv.gcv(fit.rss, m, 2.0, 60.0)
    assert fit.rsq == pytest.approx(1 - rss / _gcv.tss(Y), abs=1e-12)


@pytest.mark.parametrize(
    ("p", "degree", "max_terms", "penalty"),
    [(2, 1, 21, 2.0), (15, 2, 31, 3.0), (2, 3, 21, 3.0)],
)
def test_resolved_values(monkeypatch, p, degree, max_terms, penalty):
    """LIMIT-1 and GCV-4 for None; the forward pass gets the resolved values and
    every other field as it is, through the module attribute (CORE-1)."""
    X = np.random.default_rng(7).uniform(size=(40, p))
    y = np.sin(3 * X[:, 0])
    stub = Stub(_record([[0] * p], [[0.0] * p], [1.0], code=Termination.NO_GAIN))
    monkeypatch.setattr(_forward, "forward_pass", stub)
    fit = fit_mars(X, y, None, MarsParams(max_degree=degree))
    assert (fit.max_terms, fit.penalty) == (max_terms, penalty)
    kw = stub.calls[0][3]
    assert (kw["max_terms"], kw["penalty"], kw["max_degree"]) == (
        max_terms,
        penalty,
        degree,
    )
    given = MarsParams(max_degree=2, max_terms=9, penalty=0.5, thresh=0.01, minspan=3)
    given = dataclasses.replace(
        given, endspan=4, adjust_endspan=1.5, auto_linpreds=False
    )
    given = dataclasses.replace(given, fast_k=5, fast_beta=0.5)
    fit = fit_mars(X, y, None, given, record_candidates=True)
    kw = stub.calls[1][3]
    assert kw == {f: getattr(given, f) for f in FWD} | {"record_candidates": True}
    assert (fit.max_terms, fit.penalty) == (9, 0.5)


def test_pure_and_row_order():
    """CORE-1: read-only inputs, the same fit after a permutation of the rows
    (no near-tie here), and a 1-D y as one response; the reference's fit, also
    with pmethod="none" and nprune (PRUNE-7)."""
    rng = np.random.default_rng(8)
    X = np.round(rng.uniform(size=(90, 3)), 3)
    y = np.sin(4 * X[:, 0]) + np.maximum(X[:, 1] - 0.4, 0) + 0.1 * rng.normal(size=90)
    Xr, yr = _frozen(X, y)
    p = MarsParams(fast_k=0)
    fit = fit_mars(Xr, yr, None, p, record_candidates=True)
    log = fit.forward.candidates
    assert np.all(log.second_rss - log.best_rss > 1e-7 * fit.forward.rss[:-1])
    perm = rng.permutation(90)
    other = fit_mars(X[perm], y[perm, None], None, p)
    np.testing.assert_array_equal(other.forward.dirs, fit.forward.dirs)
    np.testing.assert_array_equal(other.forward.cuts, fit.forward.cuts)
    _close(fit, other, 1e-10)
    assert fit.coef.shape == (fit.selected.size, 1)
    for q in (p, dataclasses.replace(p, pmethod="none", nprune=4)):
        _check_reference(Xr, yr, None, q)


def test_weights_that_are_all_zero():
    """CORE-1: w needs a positive sum; the message matches W-6."""
    with pytest.raises(ValueError, match=r"weight.*zero"):
        fit_mars(np.ones((3, 1)), np.ones(3), np.zeros(3), MarsParams())
