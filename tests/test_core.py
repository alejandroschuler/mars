"""Tests of pymars/_core.py, part 1 of T12 (docs/algorithm.md: CORE-2 to
CORE-5): ``MarsParams``, the records and their dict form.

The fits come from the forward pass (T11 stage 1) and the pruning pass, put
together as CORE-1 says; part 2 of T12 adds ``fit_mars`` and its tests.
"""

import dataclasses
import json
import math
import pickle
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from pymars import _forward, _pruning, _terms
from pymars._core import ForwardRecord, MarsFit, MarsParams, PruningRecord


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


def _fit(X, y, record=False):
    """A fit from the forward pass and the pruning pass, as CORE-1 puts it
    together (pruning index m is forward index kept[m])."""
    fp = _forward.forward_pass(X, y, fast_k=0, thresh=0.0, record_candidates=record)
    B = _terms.basis_matrix(X, fp.dirs[fp.kept], fp.cuts[fp.kept])
    pp = _pruning.pruning_pass(B, y, penalty=2.0)
    ff = _pruning.final_fit(B, y, pp.selected, penalty=2.0)
    sel = fp.kept[pp.selected]
    stats = (ff.rss, ff.gcv, ff.rsq, ff.grsq, float(len(y)), 21, 2.0)
    fwd, prn = ForwardRecord(**fp._asdict()), PruningRecord(*pp[:5])
    return MarsFit(fp.dirs[sel], fp.cuts[sel], ff.coef, sel, *stats, fwd, prn)


def _fits():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(80, 3))
    y = np.sin(4 * X[:, 0]) + X[:, 1] + 0.1 * rng.normal(size=80)
    return {
        "log": _fit(X, y, record=True),
        "bare": _fit(X, y),
        "intercept": _fit(X, np.full(80, 2.0)),
        "intercept, log": _fit(X[:1], y[:1], record=True),
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


def _drop_unselected(d):
    """Move a kept term that is not selected to dropped: M_f no longer matches
    the pruning record."""
    fwd = d["forward"]
    t = int(np.setdiff1d(fwd["kept"], d["selected"])[-1])
    fwd["kept"], fwd["dropped"] = fwd["kept"][fwd["kept"] != t], np.array([t])


@pytest.mark.parametrize(
    "change",
    [
        _drop_unselected,
        _set("cuts", lambda a: a + 1.0),
        _set("rss", KeyError),
        _set("extra", 1.0),
        _set("forward.kept", KeyError),
        _set("forward", []),
        _set("rss", "1.0"),
        _set("max_terms", True),
        _set("coef", lambda a: a[:-1]),
        _set("coef", lambda a: a[:, :0]),
        _set("dirs", lambda a: a + 0.5),
        _set("cuts", lambda a: a.astype(str)),
        _set("pruning.subsets", lambda a: a.astype(int)),
        _set("selected", lambda a: a[::-1]),
        _set("selected", lambda a: a + 1),
        _set("dirs", lambda a: -a),
        _set("pruning.selected_size", 1),
        _set("pruning.selected_size", 0),
        _set("pruning.removed", lambda a: a[1:]),
        _set("forward.termination", 9),
        _set("forward.kept", lambda a: a[1:]),
        _set("forward.dropped", [0]),
        _set("forward.parent", lambda a: a[1:]),
        _set("forward.rss", []),
        _set("forward.candidates.best_rss", lambda a: a[1:]),
        _set("forward.candidates.second_kind", lambda a: a.astype(int) + 300),
    ],
)
def test_from_dict_rejects_a_bad_dict(change):
    d = FITS["log"].to_dict()
    change(d)
    with pytest.raises(ValueError):
        MarsFit.from_dict(d)


def _select(d, sel):
    """Select the forward terms sel, with dirs, cuts, coef and m* to match."""
    fwd = d["forward"]
    d |= {"selected": np.array(sel), "dirs": fwd["dirs"][sel], "cuts": fwd["cuts"][sel]}
    d["coef"] = np.zeros((len(sel), d["coef"].shape[1]))
    d["pruning"]["selected_size"] = len(sel)
    return d


@pytest.mark.parametrize("sel", [[0, 0], [1, 2], [0, 2, 1]])
def test_selected_is_increasing_kept_terms_from_0(sel):
    MarsFit.from_dict(_select(FITS["log"].to_dict(), [0, 1]))  # consistent
    with pytest.raises(ValueError, match="selected"):
        MarsFit.from_dict(_select(FITS["log"].to_dict(), sel))


def test_records_are_frozen_and_compare_by_identity():
    """CORE-3: frozen dataclasses; eq=False, so a comparison of two fits never
    compares arrays, and a fit can go in a set."""
    fit = FITS["log"]
    copy = pickle.loads(pickle.dumps(fit))
    for a, b in ((fit, copy), (fit.forward, copy.forward), (fit.pruning, copy.pruning)):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(a, dataclasses.fields(a)[0].name, None)
        assert a == a and a != b and len({a, b}) == 2


def test_a_forward_record_with_dropped_terms():
    """FWD-11: the record lists the terms that the pass dropped."""
    fwd = FITS["log"].to_dict()["forward"] | {"candidates": None}
    kept, last = fwd["kept"][:-1], fwd["kept"][-1:]
    rec = ForwardRecord(**(fwd | {"kept": kept, "dropped": last}))
    assert rec.kept.tolist() == kept.tolist() and rec.dropped.tolist() == last.tolist()


def test_bad_records():
    fp = _forward.forward_pass(np.eye(4, 1), np.arange(4.0), fast_k=0)
    with pytest.raises(ValueError, match="CandidateLog"):
        ForwardRecord(**fp._replace(candidates={"best_rss": []})._asdict())
    with pytest.raises(ValueError, match="covariate"):
        ForwardRecord(**fp._replace(dirs=fp.dirs[:, :0], cuts=fp.cuts[:, :0])._asdict())
    with pytest.raises(ValueError, match="rss"):
        ForwardRecord(**fp._replace(rss=fp.rss[:0])._asdict())
    with pytest.raises(ValueError, match="selected_size"):
        PruningRecord(**(FITS["log"].to_dict()["pruning"] | {"selected_size": 0}))
    with pytest.raises(ValueError, match="ForwardRecord"):
        dataclasses.replace(FITS["log"], forward=fp)
