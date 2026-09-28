"""The legacy code against earth on S01 to S20 (T18, issue #20).

VALIDATION_PLAN.md, "Correctness against earth", defines the modes, the
tolerances, the near-tie rule and the labels; ``legacy_triage.py`` holds the
comparison and labeling logic, and ``legacy_fit.py`` the instrumented legacy
fit. From the repository root, after ``. dev/env.sh`` and
``validation/legacy/make_venv.sh``:

    S=validation/legacy/conformance_legacy.py
    uv run --frozen --group validation python $S run --out validation/runs/t18
    uv run --frozen --group validation python $S probe --out validation/runs/t18
    uv run --frozen --group validation python $S report --out validation/runs/t18 \\
        --json validation/legacy/conformance_legacy.json

``run`` makes the earth fits that the fixtures lack (the legacy matched mode
at degrees 2 and 3, and the S15 draws), then fits the legacy code on each case
in ``.venv-legacy``: one JSON per case, written atomically and skipped when
present, so a restart resumes. ``run --code head --head-path <checkout of
legacy-1.0.4-head>`` runs HEAD's Python code in the same venv. ``probe`` runs
earth with ``trace = 9`` up to each first choice divergence. ``report``
compares, labels and writes the summary JSON. earth is used only as a black
box: its results and its trace text.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (REPO, REPO / "validation" / "harness", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import compare  # noqa: E402
import driver  # noqa: E402
import legacy_triage as lt  # noqa: E402
import trace_parse  # noqa: E402

FIXTURES = REPO / "validation" / "fixtures"
LEGACY_PYTHON = REPO / ".venv-legacy" / "bin" / "python"
GLM = {"S14", "S20"}
EPS = float(np.finfo(float).eps)


@dataclass
class Case:
    """One comparison: a dataset, a mode and a degree, the fixture with the
    legacy code's inputs, the legacy keyword arguments, and earth's side:
    the result of the fixture ``reference``, or a new run with
    ``earth_args`` on the reference inputs."""

    id: str
    dataset: str
    mode: str
    degree: int
    legacy: dict[str, Any]
    fixture: str | None = None
    reference: str | None = None
    earth_args: dict[str, Any] | None = None
    estimator: str = "Earth"
    weighted: bool = False
    draw: dict[str, Any] | None = field(default=None, repr=False)


def legacy_matched(args: dict[str, Any]) -> dict[str, Any]:
    """The legacy matched mode (VALIDATION_PLAN.md, "Comparison modes"): half
    the earth penalty, and legacy endspan 0 against earth endspan 1."""
    return {
        "max_degree": args["degree"],
        "penalty": args["penalty"] / 2,
        "max_terms": args["nk"],
        "minspan": args.get("minspan", 1),
        "endspan": 0,
        "allow_linear": bool(args.get("Auto.linpreds", False)),
    }


def legacy_spans(args: dict[str, Any]) -> dict[str, Any]:
    """S01/S02's span grid: the legacy defaults with the spans set. earth's
    automatic span maps to the legacy formula with alpha = 0.05, and earth
    endspan e to legacy endspan e - 1, as in the matched mode."""
    out: dict[str, Any] = {"max_degree": 1}
    if "minspan" in args:
        out["minspan"] = args["minspan"]
    else:
        out["minspan_alpha"] = 0.05
    if "endspan" in args:
        out["endspan"] = args["endspan"] - 1
    else:
        out["endspan_alpha"] = 0.05
    return out


def fixture(stem: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{stem}.json").read_text())


def build_cases() -> list[Case]:
    """Every (dataset, mode) fixture of S01 to S20, the legacy categorical
    path on S19's factor, and the 200 S15 draws."""
    cases = []
    pat = re.compile(r"^(S\d\d\w*?)_(matched_d\d\w*|defaults_d\d|span_\w+|raw_d1)$")
    for path in sorted(FIXTURES.glob("S*.json")):
        m = pat.match(path.stem)
        if m is None or path.stem == "S05_matched_d2_adjust1":  # a new run instead
            continue
        dataset, mode = m.groups()
        fx = fixture(path.stem)
        args, weighted = fx["earth_args"], fx["inputs"]["weights"] is not None
        c = Case(f"{dataset}/{mode}", dataset, mode.split("_")[0], args["degree"], {})
        c.fixture, c.reference, c.weighted = path.stem, path.stem, weighted
        if c.mode == "matched":
            c.legacy = legacy_matched(args)
            if args["degree"] > 1:  # Adjust.endspan = 0: no larger endspan (SPAN-4)
                c.earth_args = {**args, "Adjust.endspan": 0}
        elif c.mode == "span":
            c.legacy = legacy_spans(args)
        else:
            c.legacy = {"max_degree": args["degree"]}
        if weighted and "nonint" not in dataset:  # integer weights: repeated rows
            c.reference = path.stem.replace(dataset, f"{dataset}_repeated")
        if dataset in GLM | {"S18"}:
            # EarthClassifier hides minspan and endspan (F11); its defaults
            # (-1 with alpha 0) give the knots of minspan 1 and endspan 0.
            c.estimator = "EarthClassifier"
            spans = (c.legacy.pop("minspan", 1), c.legacy.pop("endspan", 0))
            if spans != (1, 0):
                raise ValueError(f"{c.id}: EarthClassifier cannot take spans {spans}")
        cases.append(c)
    for coding in ("strings", "codes"):
        legacy = {"max_degree": 1, "max_terms": 11, "categorical_features": [0]}
        cases.append(Case(f"S19_factor/{coding}", "S19_factor", "factor", 1, legacy))
        cases[-1].draw = {"coding": coding}
    matched = fixture("S01_matched_d1")["earth_args"]
    for d in fixture("s15_draws")["draws"]:
        if d["mode_family"] == "matched":
            args = {**matched, "degree": d["degree"], "Adjust.endspan": 0}
            legacy, mode = legacy_matched(args), "matched"
        else:
            args, legacy, mode = (
                {"degree": d["degree"]},
                {"max_degree": d["degree"]},
                "defaults",
            )
        c = Case(
            f"S15/draw{d['rep']:03d}", "S15", mode, d["degree"], legacy, earth_args=args
        )
        c.draw = {k: d[k] for k in ("rep", "dgp", "noise")}
        cases.append(c)
    return cases


def case_inputs(case: Case, stem: str | None = None) -> dict[str, Any]:
    """X, y, X_test, weights and the truth at X_test (S15 only) of the
    fixture ``stem`` (default: the case's own)."""
    if case.dataset == "S15":
        from gen_fixtures import scaled_matrix

        from validation.sims import dgps, seeds

        dgp, noise, diag = (
            dgps.REGISTRY[case.draw["dgp"]],
            case.draw["noise"],
            dgps.load_diagnostics(),
        )
        rng, test_rng = seeds.train_test_rngs(
            f"S15_{dgp.name}_{noise}", case.draw["rep"]
        )
        X, y, _ = dgps.generate(dgp, rng, 200, noise, diag)
        X_test, _, truth = dgps.generate(dgp, test_rng, 1000, noise, diag)
        Xs, scale = scaled_matrix(X)
        return {
            "X": Xs,
            "y": y,
            "X_test": X_test / scale,
            "weights": None,
            "truth": truth,
        }
    if case.dataset == "S19_factor":
        fx = fixture("s19_factor")
        X = np.array(fx["labels"], dtype=object).reshape(-1, 1)
        if case.draw["coding"] == "codes":
            X = np.array([["abcd".index(v)] for v in fx["labels"]], dtype=float)
        return {
            "X": X,
            "y": np.array(fx["y"]),
            "X_test": None,
            "weights": None,
            "truth": None,
        }
    inp = fixture(stem or case.fixture)["inputs"]
    arr = {
        k: None if inp[k] is None else np.array(inp[k])
        for k in ("X", "y", "X_test", "weights")
    }
    if arr["y"].dtype.kind in "iuf":
        arr["y"] = arr["y"].astype(float)
    return {**arr, "truth": None}


def reference_inputs(case: Case) -> dict[str, Any]:
    """The inputs of earth's side: repeated rows for integer weights."""
    return case_inputs(case, case.reference)


def earth_result(case: Case, out: Path) -> dict[str, Any]:
    if case.dataset == "S19_factor":
        return {"fitted": fixture("s19_factor")["factor"]["fitted"]}
    if case.earth_args is None:
        return fixture(case.reference)["result"]
    return json.loads((out / "earth" / f"{slug(case.id)}.json").read_text())


def slug(case_id: str) -> str:
    return case_id.replace("/", "__")


def write_json(path: Path, obj: Any) -> None:
    """Write atomically: a temporary file in the same folder, then a rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(
            obj, fh, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)
        )
    os.replace(tmp, path)


def earth_step_terms(res: dict[str, Any]) -> list[frozenset[lt.Sig]]:
    sigs = lt.earth_signatures(res["dirs"], res["cuts"])
    return [frozenset(sigs[r] for r in rows) for rows in lt.earth_steps(res["dirs"])]


def legacy_step_terms(leg: dict[str, Any]) -> list[frozenset[lt.Sig]]:
    sigs = [lt.sig_from_json(t["sig"]) for t in leg["terms"]]
    return [frozenset(sigs[i] for i in idx) for idx in leg["steps"]]


def _earth_job(case: Case, args: dict[str, Any], **kw: Any) -> driver.EarthJob:
    inp, glm = reference_inputs(case), "binomial" if case.dataset in GLM else None
    return driver.EarthJob(
        slug(case.id),
        inp["X"],
        inp["y"],
        args,
        weights=inp["weights"],
        glm_family=glm,
        **kw,
    )


def cmd_run(ns: argparse.Namespace) -> None:
    out = Path(ns.out)
    cases = select(ns.only)
    todo = [
        c
        for c in cases
        if c.earth_args and not (out / "earth" / f"{slug(c.id)}.json").exists()
    ]
    if todo:  # the earth fits that the fixtures lack, in one R process
        jobs = [
            _earth_job(c, c.earth_args, X_test=reference_inputs(c)["X_test"])
            for c in todo
        ]
        with tempfile.TemporaryDirectory(prefix="t18-earth-") as tmp:
            results = driver.run_earth(jobs, workdir=Path(tmp))
        for c in todo:
            write_json(out / "earth" / f"{slug(c.id)}.json", results[slug(c.id)])
    code_path = str(Path(ns.head_path).resolve()) if ns.code == "head" else None
    for i, c in enumerate(cases):
        path = out / ns.code / f"{slug(c.id)}.json"
        if path.exists():
            continue
        t0, earth, inp = time.time(), earth_result(c, out), case_inputs(c)
        cfg = {
            k: None if v is None else v.tolist() for k, v in inp.items() if k != "truth"
        }
        cfg.update(
            object_X=inp["X"].dtype == object, kwargs=c.legacy, estimator=c.estimator
        )
        cfg["code_path"] = code_path
        steps = earth_step_terms(earth) if "dirs" in earth else []
        cfg["earth_steps"] = [[list(s) for s in step] for step in steps]
        proc = subprocess.run(
            [str(LEGACY_PYTHON), "-I", str(HERE / "legacy_fit.py")],
            input=json.dumps(cfg),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"legacy fit of {c.id} failed:\n{proc.stderr[-3000:]}")
        write_json(path, {**json.loads(proc.stdout), "case": asdict(c)})
        print(
            f"[{i + 1}/{len(cases)}] {c.id} {ns.code} {time.time() - t0:.1f}s",
            flush=True,
        )


def select(only: str | None) -> list[Case]:
    return [c for c in build_cases() if only is None or re.search(only, c.id)]


def divergence_step(leg: dict[str, Any], earth: dict[str, Any]) -> int | None:
    """The 1-based step of the first choice divergence, or of the step that
    only the legacy code took (earth stopped first); None otherwise."""
    fc = lt.compare_forward(legacy_step_terms(leg), earth_step_terms(earth))
    if fc["first_choice"] is not None:
        return fc["first_choice"] + 1
    return fc["n_earth"] + 1 if fc["n_legacy"] > fc["n_earth"] else None


def cmd_probe(ns: argparse.Namespace) -> None:
    """earth with trace = 9 and nk = 2 * step + 1, so that the step of the
    divergence is the last one it searches (LIMIT-2), one R process."""
    out, jobs, metas = Path(ns.out), [], []
    for c in select(ns.only):
        leg_path, probe_path = (
            out / ns.code / f"{slug(c.id)}.json",
            out / "probes" / ns.code / f"{slug(c.id)}.json",
        )
        if (
            probe_path.exists()
            or not leg_path.exists()
            or c.dataset in ("S18", "S19_factor")
        ):
            continue
        leg, earth = json.loads(leg_path.read_text()), earth_result(c, out)
        step = (
            None
            if "error" in leg or "dirs" not in earth
            else divergence_step(leg, earth)
        )
        if step is None:
            continue
        args = {
            **(c.earth_args or fixture(c.reference)["earth_args"]),
            "nk": 2 * step + 1,
        }
        jobs.append(
            _earth_job(
                c, {**args, "pmethod": "none"}, trace=9, include_forward_path=False
            )
        )
        metas.append((c, step, leg))
    with tempfile.TemporaryDirectory(prefix="t18-probe-") as tmp:
        results = driver.run_earth(jobs, workdir=Path(tmp)) if jobs else {}
        for c, step, leg in metas:
            trace = Path(tmp) / f"{slug(c.id)}_trace.txt"
            probe = probe_choice(
                trace, results[slug(c.id)], leg, step, reference_inputs(c)
            )
            write_json(out / "probes" / ns.code / f"{slug(c.id)}.json", probe)
            print(f"probe {c.id} step {step}: {probe['status']}", flush=True)


def probe_choice(
    trace: Path,
    res: dict[str, Any],
    leg: dict[str, Any],
    step: int,
    inp: dict[str, Any],
) -> dict[str, Any]:
    """earth's view of the legacy code's choice at ``step``: whether earth
    searched its parent and variable, the kind of search, and the status of
    its knot (visited, evaluated, the four flags). A trace case number c
    stands for the knot sorted(x)[c - 1] (read off the traces)."""
    X, y = inp["X"], inp["y"]
    log = trace_parse.parse_trace(trace, sample_var_y=float(np.var(y, ddof=1)))
    tstep = next((s for s in log.steps if s.term == 2 * step), None)
    sigs = lt.earth_signatures(res["dirs"], res["cuts"])
    slot_of = {
        r: s
        for s, r in compare._slot_to_row_map(
            lt.earth_steps(res["dirs"]), log.steps
        ).items()
    }
    terms = [lt.sig_from_json(t["sig"]) for t in leg["terms"]]
    first = leg["steps"][step - 1][0]
    parent = terms[leg["terms"][first]["parent"]]
    var, code, knot = next(f for f in terms[first] if f not in parent)
    info: dict[str, Any] = {
        "step": step,
        "parent": lt.term_label(parent),
        "var": var,
        "knot": knot,
    }
    if tstep is None:
        return {**info, "status": "no_search_logged"}
    info["max_legal"] = tstep.max_legal_rss_delta
    legal = [c.rss for s in tstep.searches for c in s.cases if c.accepted]
    legal += [s.linear.rss for s in tstep.searches if s.linear is not None]
    info["earth_best_two"] = sorted(legal)[:2]
    if parent not in sigs:
        return {**info, "status": "parent_not_in_earth"}
    slot = slot_of.get(sigs.index(parent))
    search = next(
        (s for s in tstep.searches if s.parent == slot and s.pred == var + 1), None
    )
    if search is None or search.skipped_reason is not None:
        return {**info, "status": "parent_not_searched"}
    info["search"] = "single" if search.no_new_form else "pair"
    if code == 2:
        return {
            **info,
            "status": "linear",
            "earth_rss": search.linear and search.linear.rss,
        }
    xs = np.sort(X[:, var])
    hits = [
        c for c in search.cases if 1 <= c.case <= len(xs) and xs[c.case - 1] == knot
    ]
    if not any(c.evaluated for c in hits):
        visited = [xs[c.case - 1] for c in search.cases if 1 <= c.case <= len(xs)]
        active = X[lt.term_column(parent, X) > 0, var]
        where = "span" if hits else "grid"
        if active.size and knot >= active.max():
            where = "top"
        elif visited and knot < min(visited):
            below = lt.term_column(parent, X[X[:, var] <= min(visited)])
            where = "bottom_inactive" if np.any(below <= 0) else "bottom"
        return {**info, "status": "knot_not_evaluated", "where": where}
    case = next(c for c in hits if c.evaluated)
    info["earth_rss"] = case.rss
    info["flags"] = [case.bx1g, case.cov_col_g, case.tol_g, case.max_g]
    if case.tol_g is False:
        return {**info, "status": "tol"}
    if case.max_g is False:
        return {**info, "status": "maxlegal"}
    if search.no_new_form and len(leg["steps"][step - 1]) == 2:
        return {**info, "status": "single_search"}
    return {**info, "status": "legal" if case.accepted else "rejected"}


# --- analysis --------------------------------------------------------------------


@dataclass
class Fit:
    """One fit's data for the analysis: both sides' steps, and earth's
    reference inputs (repeated rows for integer weights)."""

    case: Case
    earth: dict[str, Any]
    leg: dict[str, Any]
    probe: dict[str, Any] | None

    def __post_init__(self) -> None:
        inp = reference_inputs(self.case)
        self.X, self.y, self.w = inp["X"], inp["y"], inp["weights"]
        self.e_sigs = lt.earth_signatures(self.earth["dirs"], self.earth["cuts"])
        self.e_groups = lt.earth_steps(self.earth["dirs"])
        self.e_steps = earth_step_terms(self.earth)
        self.l_steps = legacy_step_terms(self.leg)
        self.fc = lt.compare_forward(self.l_steps, self.e_steps)

    def rss_with(self, k: int, new: frozenset[lt.Sig]) -> float:
        """The RSS on earth's basis before 0-based step k, plus ``new``."""
        rows = [0] + [r for g in self.e_groups[:k] for r in g]
        sigs = [self.e_sigs[r] for r in rows] + sorted(new, key=lt.sig_key)
        return lt.rss(lt.basis_matrix(sigs, self.X), self.y, self.w)


def analyze(
    case: Case, earth: dict[str, Any], leg: dict[str, Any], probe: dict[str, Any] | None
) -> dict[str, Any]:
    """Compare one fit and label its differences ("Triage of differences").
    Each entry of ``diffs`` has a kind (error, design, structure, rss,
    choice, stop, pruning, final, glm), the 1-based forward step where it
    applies, a label, a finding ID and one sentence of evidence."""
    row: dict[str, Any] = {
        "id": case.id,
        "mode": case.mode,
        "degree": case.degree,
        "diffs": [],
    }

    def add(kind: str, step: int | None, v: lt.Verdict) -> None:
        row["diffs"].append(
            {"kind": kind, "step": step, "label": v[0], "finding": v[1], "text": v[2]}
        )

    if "error" in leg:
        err = leg["error"]
        key = (
            "weights"
            if "sample_weight" in err
            else "responses"
            if case.dataset == "S17"
            else "error"
        )
        add("error", None, lt.verdict(key, err=err))
        return row
    if case.dataset == "S19_factor":
        diff = float(np.max(np.abs(np.subtract(leg["pred_train"], earth["fitted"]))))
        row["final"] = {"legacy_terms": len(leg["selected"]), "max_fitted_diff": diff}
        if diff > 1e-8 * float(np.std(case_inputs(case)["y"])):
            add(
                "final",
                None,
                lt.verdict("categorical", terms=len(leg["selected"]), diff=diff),
            )
        return row
    f = Fit(case, earth, leg, probe)
    fc = f.fc
    row["forward"] = {
        "legacy": fc["n_legacy"],
        "earth": fc["n_earth"],
        "agree": len(fc["relations"]),
    }
    if case.dataset == "S18":
        first = (
            fc["first_choice"] if fc["first_choice"] is not None else fc["first_subset"]
        )
        add("design", None if first is None else first + 1, lt.verdict("classes"))
        return final(row, f)
    if fc["first_subset"] is not None:
        k = fc["first_subset"]
        legacy, earth_t = lt.terms_label(f.l_steps[k]), lt.terms_label(f.e_steps[k])
        add(
            "structure",
            k + 1,
            lt.verdict("structure", step=k + 1, earth=earth_t, legacy=legacy),
        )
    if f.y.dtype.kind == "f" and f.y.ndim == 1 and earth.get("fwd_rss"):
        for k in range(len(fc["relations"])):
            last = f.e_groups[k][-1]
            rss_e, rss_l = float(earth["fwd_rss"][last]), float(leg["fwd_rss"][k + 1])
            kappa = float(np.linalg.cond(lt.basis_matrix(f.e_sigs[: last + 1], f.X)))
            tol, _ = compare._kappa_scale(
                compare.FWD_RSS_REL, kappa, compare.KAPPA_RSS_LIMIT
            )
            rel = abs(rss_l - rss_e) / abs(rss_e) if rss_e else abs(rss_l)
            if rel > tol:
                span = (
                    fc["first_subset"] is not None
                    and fc["first_subset"] <= k
                    and rss_l < rss_e
                )
                key = "rss_span" if span else "rss_numeric"
                add(
                    "rss",
                    k + 1,
                    lt.verdict(
                        key, step=k + 1, rss_l=rss_l, rss_e=rss_e, rel=rel, kappa=kappa
                    ),
                )
                break
    if fc["first_choice"] is not None:
        add(
            "choice",
            fc["first_choice"] + 1,
            lt.classify_choice(evidence(f, fc["first_choice"])),
        )
    elif fc["n_legacy"] < fc["n_earth"]:
        ev = {"first": "legacy", "n_legacy": fc["n_legacy"], "n_earth": fc["n_earth"]}
        add(
            "stop",
            fc["n_legacy"] + 1,
            lt.classify_stop({**ev, "legacy_stop": legacy_stop(f)}),
        )
    elif fc["n_legacy"] > fc["n_earth"]:
        ev = {"first": "earth", "n_legacy": fc["n_legacy"], "n_earth": fc["n_earth"]}
        k, code = fc["n_earth"], earth.get("termcond")
        v = (
            lt.classify_choice(evidence(f, k))
            if code == 6 and probe
            else lt.classify_stop({**ev, "termcond": code})
        )
        add("stop", k + 1, v)
    elif all(r == "same" for r in fc["relations"]):
        pruning(add, row, f)
    return final(row, f)


def evidence(f: Fit, k: int) -> dict[str, Any]:
    """The evidence of lt.classify_choice at the 0-based step k."""
    search = f.leg["searches"][k]
    a, e = search["chosen"] or {}, search.get("earth_match") or {}
    e_step = f.e_steps[k] if k < len(f.e_steps) else frozenset()
    knot = next(
        (i for i in search.get("earth_knot", []) if i.get("knot_in_set") is False), None
    )
    why = (
        None
        if knot is None
        else f"knot {knot['knot']:.6g} is not among the {knot['n_knots']} legacy knots"
    )
    return {
        "step": k + 1,
        "legacy": lt.terms_label(f.l_steps[k]),
        "earth": lt.terms_label(e_step) if e_step else "no term (earth stops)",
        "leg_rss_a": a.get("rss"),
        "leg_gcv_a": a.get("gcv"),
        "leg_rss_e": e.get("rss"),
        "leg_gcv_e": e.get("gcv"),
        "leg_before": search["rss_before"],
        "same_cost": bool(a and e and a["kind"] == e["kind"]),
        "e_before": f.rss_with(k, frozenset()),
        "e_rss_a": f.rss_with(k, f.l_steps[k]),
        "e_rss_e": f.rss_with(k, e_step) if e_step else None,
        "leg_knot": why,
        "earth_stopped": not e_step,
        "probe": f.probe,
        "fast_k": (f.case.earth_args or fixture(f.case.reference)["earth_args"]).get(
            "fast.k"
        ),
    }


def legacy_stop(f: Fit) -> str:
    """Why the legacy forward pass stopped (its last, unsuccessful search)."""
    last, kw = f.leg["searches"][-1], f.case.legacy
    n, p = f.X.shape
    if last["n_cands"] == 0:
        limit = kw.get("max_terms") or min(n - 1, max(21, 2 * p + 1))
        return "term_limit" if last["n_terms_before"] + 2 > limit else "no_candidate"
    if last["chosen"] is None:
        return "gcv_inf"
    return "eps" if last["chosen"]["rss"] >= last["rss_before"] - EPS else "other"


def pruning(add: Any, row: dict[str, Any], f: Fit) -> None:
    """When both forward passes added the same terms: the pruning path, the
    selected terms and the final model, with compare_fit in earth's term
    numbering (the legacy path gives one subset per size)."""
    to_earth = [f.e_sigs.index(lt.sig_from_json(t["sig"])) + 1 for t in f.leg["terms"]]
    m_f, leg, earth = len(to_earth), f.leg, f.earth
    by_size = {
        len(s): (s, r, g)
        for s, r, g in zip(
            leg["prune_subsets"], leg["prune_rss"], leg["prune_gcv"], strict=True
        )
    }
    no_int = next(
        (m for m in sorted(by_size, reverse=True) if 0 not in by_size[m][0]), None
    )
    if no_int is not None:
        add("pruning", None, lt.verdict("intercept", size=no_int, total=m_f))
    pt = np.zeros((m_f, m_f), dtype=int)
    for m in range(1, m_f + 1):
        pt[m - 1, :m] = sorted(to_earth[i] for i in by_size[m][0])
    order = np.argsort([to_earth[i] for i in leg["selected"]])
    ours = {
        "selected_terms": sorted(to_earth[i] for i in leg["selected"]),
        "prune_terms": pt.tolist(),
        "rss_per_subset": [by_size[m][1] for m in range(1, m_f + 1)],
        "gcv_per_subset": [by_size[m][2] for m in range(1, m_f + 1)],
        "coef": np.asarray(leg["coef"])[order].reshape(-1, 1).tolist(),
        "gcv": leg["gcv"],
        "fitted": np.reshape(leg["pred_train"], (-1, 1)).tolist(),
        "pred_test": None
        if leg["pred_test"] is None
        else np.reshape(leg["pred_test"], (-1, 1)).tolist(),
    }
    B = lt.basis_matrix([f.e_sigs[r - 1] for r in earth["selected_terms"]], f.X)
    diffs = compare.compare_fit(
        ours,
        {k: earth.get(k) for k in ours},
        kappa=float(np.linalg.cond(B)),
        sd_y=float(np.std(f.y)),
    )
    row["pruning"] = fields = [d.field for d in diffs]
    if no_int is not None or not diffs:
        return
    d = diffs[0]
    if "pruning_removed" in fields:
        d = diffs[fields.index("pruning_removed")]
        m = d.step + 1
        rss_l, rss_e = by_size[m][1], earth["rss_per_subset"][m - 1]
        key = "prune_subset" if rss_l > rss_e * (1 + lt.NEAR_TIE_REL) else "prune_tie"
        add("pruning", None, lt.verdict(key, size=m, rss_l=rss_l, rss_e=rss_e))
    elif "selected_terms" in fields:
        key = "gcv_convention" if "gcv_per_subset" in fields else "size_tie"
        add(
            "pruning",
            None,
            lt.verdict(
                key, legacy=len(leg["selected"]), earth=len(earth["selected_terms"])
            ),
        )
    else:
        label = d.label or "unexplained"
        metric = "" if d.metric is None else f"{d.metric:.3g}"
        add(
            "final",
            None,
            lt.verdict(
                "final",
                label=label,
                field=d.field,
                detail=d.detail,
                metric=metric,
                tol=d.tolerance,
            ),
        )


def final(row: dict[str, Any], f: Fit) -> dict[str, Any]:
    """Summary numbers for every fit (the defaults-mode row of the tolerance
    table): terms, R2, GCV, the test error when the truth is known, and the
    largest prediction difference in units of sd(y) inside and outside the
    training box; and for the binary fits, the GLM refit (F9)."""
    leg, earth = f.leg, f.earth
    fin: dict[str, Any] = {
        "legacy_terms": len(leg["selected"]),
        "earth_terms": len(earth["selected_terms"]),
    }
    fin.update(
        legacy_gcv=leg["gcv"], earth_gcv=earth.get("gcv"), earth_rsq=earth.get("rsq")
    )
    y = f.y
    if y.dtype.kind == "f" and y.ndim == 1 and y.std() > 0:
        fin["legacy_rsq"] = (
            1 - leg["rss"] / float(np.sum((y - y.mean()) ** 2)) if f.w is None else None
        )
        if leg.get("pred_test") and earth.get("pred_test"):
            diff = (
                np.abs(np.ravel(leg["pred_test"]) - np.ravel(earth["pred_test"]))
                / y.std()
            )
            X, X_test = f.X, reference_inputs(f.case)["X_test"]
            box = (X_test >= X.min(axis=0)) & (X_test <= X.max(axis=0))
            inside = diff[np.all(box, axis=1)]
            fin["pred_diff_sd"] = [float(inside.max(initial=0)), float(diff.max())]
    truth = case_inputs(f.case)["truth"]
    if truth is not None:
        fin["test_mse"] = [
            float(np.mean((np.ravel(d["pred_test"]) - truth) ** 2))
            for d in (leg, earth)
        ]
    if f.case.dataset in GLM:
        row["diffs"].append(glm_refit(f))
    row["final"] = fin
    return row


def glm_refit(f: Fit) -> dict[str, Any]:
    """F9 on a binary fit: the legacy logistic refit (L2 penalty, C = 1)
    against an unpenalized refit on the legacy code's own selected basis,
    and against earth's glm (whose basis may differ)."""
    import warnings

    from sklearn.linear_model import LogisticRegression

    leg, X = f.leg, case_inputs(f.case)["X"]
    sel = [lt.sig_from_json(leg["terms"][i]["sig"]) for i in leg["selected"]]
    p_leg = np.asarray(leg["proba_train"])[:, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unpen = LogisticRegression(C=np.inf, fit_intercept=False, max_iter=100000)
        B = lt.basis_matrix(sel, X)
        p_unpen = unpen.fit(B, case_inputs(f.case)["y"]).predict_proba(B)[:, 1]
    same = sorted(sel) == sorted(f.e_sigs[r - 1] for r in f.earth["selected_terms"])
    v = lt.verdict(
        "glm",
        diff_own=float(np.max(np.abs(p_leg - p_unpen))),
        diff=float(np.max(np.abs(p_leg - np.ravel(f.earth["pred_train"])))),
        same="the same" if same else "other",
    )
    return {"kind": "glm", "step": None, "label": v[0], "finding": v[1], "text": v[2]}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts by label and finding (the first difference of each fit, and
    all differences) and the S15 rates."""
    first = [r["diffs"][0] for r in rows if r["diffs"]]
    diffs = [d for r in rows for d in r["diffs"]]
    s15 = {}
    for mode in ("matched", "defaults"):
        sel = [r for r in rows if r["id"].startswith("S15/") and r["mode"] == mode]
        div = [d for r in sel for d in r["diffs"] if d["kind"] in ("choice", "stop")]
        causes = lt.count_by(
            ({"key": f"{d['label']} {d['finding'] or ''}".strip()} for d in div), "key"
        )
        s15[mode] = {
            "fits": len(sel),
            "no_choice_divergence": len(sel) - len(div),
            "causes": causes,
            "first_divergence_step": lt.count_by(
                ({"key": str(d["step"])} for d in div), "key"
            ),
        }
    return {
        "fits": len(rows),
        "fits_with_differences": len(first),
        "first_by_label": lt.count_by(first, "label"),
        "first_by_finding": lt.count_by(first, "finding"),
        "all_by_label": lt.count_by(diffs, "label"),
        "all_by_finding": lt.count_by(diffs, "finding"),
        "s15": s15,
    }


def head_vs_wheel(out: Path, cases: list[Case]) -> dict[str, Any]:
    """Whether HEAD and the wheel give the same fit, on the unweighted cases
    run with both (the wheel takes no weights)."""
    same, differ = [], []
    for c in cases:
        paths = [out / code / f"{slug(c.id)}.json" for code in ("wheel", "head")]
        if not c.weighted and all(p.exists() for p in paths):
            a, b = (json.loads(p.read_text()) for p in paths)
            keys = (
                "terms",
                "steps",
                "selected",
                "coef",
                "pred_train",
                "prune_rss",
                "error",
            )
            (same if all(a.get(k) == b.get(k) for k in keys) else differ).append(c.id)
    return {"identical": len(same), "different": differ}


def cmd_report(ns: argparse.Namespace) -> None:
    out, cases, rows = Path(ns.out), select(ns.only), []
    for c in cases:
        for code in ("wheel", "head"):
            path = out / code / f"{slug(c.id)}.json"
            if path.exists() and (code == "wheel" or c.weighted):
                probe_path = out / "probes" / code / f"{slug(c.id)}.json"
                probe = (
                    json.loads(probe_path.read_text()) if probe_path.exists() else None
                )
                rows.append(
                    {
                        "code": code,
                        **analyze(
                            c, earth_result(c, out), json.loads(path.read_text()), probe
                        ),
                    }
                )
    result = {
        "command": "conformance_legacy.py " + " ".join(sys.argv[1:]),
        "summary": summarize(rows),
        "head_vs_wheel": head_vs_wheel(out, cases),
        "cases": rows,
    }
    if ns.json:
        write_json(Path(ns.json), result)
    print(json.dumps(result["summary"], indent=1))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("run", "probe", "report"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True, help="the folder of per-case results")
        p.add_argument("--only", help="a regular expression on the case ids")
        p.add_argument("--code", choices=("wheel", "head"), default="wheel")
        p.add_argument("--head-path", help="a checkout of legacy-1.0.4-head (run)")
        p.add_argument("--json", help="where to write the summary JSON (report)")
    ns = parser.parse_args(argv)
    {"run": cmd_run, "probe": cmd_probe, "report": cmd_report}[ns.cmd](ns)


if __name__ == "__main__":
    main()
