"""Fit the legacy code once and record its forward searches (T18).

Run by ``conformance_legacy.py`` as ``.venv-legacy/bin/python -I
legacy_fit.py``, with a JSON config on stdin and a JSON result on stdout.
``code_path`` (a checkout of the tag ``legacy-1.0.4-head``) goes first on
``sys.path``, so that HEAD's Python code runs in the wheel's venv; without it
the installed 1.0.4 wheel runs.

Two wrappers observe the forward search without changing it: one sees the
basis matrix that each candidate builds (so, the candidate's new terms), the
other the GCV that it gets. Each search records its candidates' count, the
chosen one, the second best by the legacy criterion, and the legacy code's
own evaluation of earth's choice at that step (``earth_steps``), or why that
choice was not a candidate. Terms are recorded as signatures, as in
``legacy_triage.py``.
"""

import json
import sys
import time
import warnings

cfg = json.loads(sys.stdin.read())
if cfg.get("code_path"):
    sys.path.insert(0, cfg["code_path"])
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

import pymars  # noqa: E402
from pymars import _forward  # noqa: E402
from pymars._basis import (  # noqa: E402
    CategoricalBasisFunction,
    ConstantBasisFunction,
    HingeBasisFunction,
    LinearBasisFunction,
)


def factors(bf):
    out, node = [], bf
    while node is not None and not isinstance(node, ConstantBasisFunction):
        j = int(node.variable_idx)
        if isinstance(node, HingeBasisFunction):
            out.append((j, 1 if node.is_right_hinge else -1, float(node.knot_val)))
        elif isinstance(node, LinearBasisFunction):
            out.append((j, 2, None))
        elif isinstance(node, CategoricalBasisFunction):
            c = node.category
            out.append((j, 3, c if isinstance(c, str) else float(c)))
        else:
            out.append((j, 9, None))
        node = node.parent1
    return tuple(sorted(out, key=lambda f: (f[0], f[1])))


def kind(bfs):
    if len(bfs) == 2:
        return "pair"
    kinds = {HingeBasisFunction: "hinge", LinearBasisFunction: "linear"}
    return kinds.get(type(bfs[0]), "other")


def desc(entry):
    if entry is None:
        return None
    gcv, rss, _, new = entry
    return {"gcv": gcv, "rss": rss, "kind": kind(new), "new": [factors(b) for b in new]}


def knot_status(fp, earth_terms):
    """For each way earth's new terms split into a parent and a new factor:
    whether the legacy code has that parent, and whether the knot is in the
    legacy knot set for it (_get_allowable_knot_values)."""
    out, current = [], {factors(b): b for b in fp.current_basis_functions}
    for sig in earth_terms:
        for f in sig:
            parent = current.get(tuple(g for g in sig if g != f))
            item = {"knot": f[2], "parent_found": parent is not None}
            if parent is not None and f[1] in (1, -1):
                col = fp.X_fit_original[:, f[0]].astype(float)
                knots = np.asarray(fp._get_allowable_knot_values(col, parent, f[0]))
                item["n_knots"] = int(knots.size)
                item["knot_in_set"] = bool(np.any(knots == f[2]))
                if knots.size:
                    item["lo"], item["hi"] = float(knots.min()), float(knots.max())
            out.append(item)
    return out


EARTH = [
    frozenset(tuple(tuple(f) for f in sig) for sig in step)
    for step in cfg.get("earth_steps") or []
]
SEARCHES = []
FP = _forward.ForwardPasser
_find = FP._find_best_candidate_addition
_build, _gcv = FP._build_basis_matrix, _forward.calculate_gcv


def find(self):
    log, cur, n_cur = [], {}, len(self.current_basis_functions)

    def build(X, bfs):
        cur["new"] = bfs[n_cur:]
        return _build(self, X, bfs)

    def gcv(rss, n, eff):
        value = _gcv(rss, n, eff)
        log.append((float(value), float(rss), len(log), cur["new"]))
        return value

    self._build_basis_matrix, _forward.calculate_gcv = build, gcv
    try:
        _find(self)
    finally:
        del self._build_basis_matrix
        _forward.calculate_gcv = _gcv
    chosen = self._best_candidate_addition
    cid = next((e[2] for e in log if chosen and e[3][0] is chosen[0]), None)
    by_gcv = sorted(log, key=lambda e: (e[0], e[2]))
    s = {
        "n_cands": len(log),
        "n_terms_before": n_cur,
        "rss_before": float(self.current_rss),
        "chosen": desc(log[cid]) if cid is not None else None,
    }
    if cid is not None:
        new = frozenset(factors(b) for b in log[cid][3])
        others = (e for e in by_gcv if frozenset(factors(b) for b in e[3]) != new)
        s["second"] = desc(next(others, None))
    if len(SEARCHES) < len(EARTH):
        target = EARTH[len(SEARCHES)]
        match = [e for e in by_gcv if target <= frozenset(factors(b) for b in e[3])]
        s["earth_match"] = desc(match[0]) if match else None
        if not match:
            s["earth_knot"] = knot_status(self, target)
    SEARCHES.append(s)


FP._find_best_candidate_addition = find
X = np.asarray(cfg["X"], dtype=object if cfg.get("object_X") else float)
y = np.asarray(cfg["y"])
X_test = None if cfg.get("X_test") is None else np.asarray(cfg["X_test"], float)
out = {"pymars_file": pymars.__file__, "version": getattr(pymars, "__version__", "")}
t0 = time.perf_counter()
try:
    if cfg["estimator"] == "EarthClassifier":
        est = pymars.EarthClassifier(**cfg["kwargs"]).fit(X, y)
        model = est.earth_
    else:
        est = model = pymars.Earth(**cfg["kwargs"])
        extra = {} if cfg.get("weights") is None else {"sample_weight": cfg["weights"]}
        est.fit(X, y.astype(float), **extra)
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(out))
    sys.exit(0)
out["fit_time"] = time.perf_counter() - t0
rec = model.record_
fwd = list(rec.fwd_basis_[-1])
index = {id(b): i for i, b in enumerate(fwd)}
step_of = {}
for k, snap in enumerate(rec.fwd_basis_):
    for b in snap:
        step_of.setdefault(id(b), k)
out.update(
    terms=[{"sig": factors(b), "parent": index.get(id(b.parent1))} for b in fwd],
    steps=[
        [i for i, b in enumerate(fwd) if step_of[id(b)] == k]
        for k in range(1, len(rec.fwd_basis_))
    ],
    searches=SEARCHES,
    fwd_rss=[float(v) for v in rec.fwd_rss_],
    prune_subsets=[
        [index.get(id(b)) for b in sub] for sub in rec.pruning_trace_basis_functions_
    ],
    prune_rss=[float(v) for v in rec.pruning_trace_rss_],
    prune_gcv=[float(v) for v in rec.pruning_trace_gcv_],
    selected=[index.get(id(b)) for b in model.basis_],
    coef=np.asarray(model.coef_, float).ravel().tolist(),
    rss=float(model.rss_),
    gcv=float(model.gcv_),
    pred_train=np.asarray(model.predict(X), float).tolist(),
    pred_test=None if X_test is None else model.predict(X_test).tolist(),
)
if cfg["estimator"] == "EarthClassifier":
    out["classes"] = [str(c) for c in est.classes_]
    out["proba_train"] = est.predict_proba(X).tolist()
    if X_test is not None:
        out["proba_test"] = est.predict_proba(X_test).tolist()
print(json.dumps(out))
