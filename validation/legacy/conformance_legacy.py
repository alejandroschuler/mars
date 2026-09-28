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
            check_probe_prefix(results[slug(c.id)], earth_result(c, out), c.id)
            probe = probe_choice(
                trace, results[slug(c.id)], leg, step, reference_inputs(c)
            )
            write_json(out / "probes" / ns.code / f"{slug(c.id)}.json", probe)
            print(f"probe {c.id} step {step}: {probe['status']}", flush=True)


def check_probe_prefix(
    probe: dict[str, Any], full: dict[str, Any], case_id: str
) -> None:
    """The probe run (``nk = 2s + 1``) must repeat the full run's first terms:
    its ``dirs`` and ``cuts`` are the first rows of the full run's, exactly."""
    rows = len(probe["dirs"])
    same = probe["dirs"] == full["dirs"][:rows] and probe["cuts"] == full["cuts"][:rows]
    if not same:
        raise RuntimeError(
            f"{case_id}: the probe's first {rows} terms differ from the full run's"
        )


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
        if hits and all(c.bx1 == 0 for c in hits):  # the case above is inactive
            where = "inactive"
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
    """One fit's data for the analysis: both sides' steps, earth's inputs
    ``ref`` (repeated rows for integer weights) and the legacy code's own
    inputs ``own``; both default to the case's fixtures."""

    case: Case
    earth: dict[str, Any]
    leg: dict[str, Any]
    probe: dict[str, Any] | None
    ref: dict[str, Any] | None = None
    own: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.ref = self.ref if self.ref is not None else reference_inputs(self.case)
        self.own = self.own if self.own is not None else case_inputs(self.case)
        self.X, self.y, self.w = self.ref["X"], self.ref["y"], self.ref["weights"]
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
    case: Case,
    earth: dict[str, Any],
    leg: dict[str, Any],
    probe: dict[str, Any] | None,
    ref: dict[str, Any] | None = None,
    own: dict[str, Any] | None = None,
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
        y = (own or case_inputs(case))["y"]
        if diff > 1e-8 * float(np.std(y)):
            add(
                "final",
                None,
                lt.verdict("categorical", terms=len(leg["selected"]), diff=diff),
            )
        return row
    f = Fit(case, earth, leg, probe, ref, own)
    fc = f.fc
    row["forward"] = {
        "legacy": fc["n_legacy"],
        "earth": fc["n_earth"],
        "agree": len(fc["relations"]),
    }
    if (
        f.y.dtype.kind == "f"
        and np.all(f.y == f.y.flat[0])
        and fc["n_legacy"] != fc["n_earth"]
    ):
        add(
            "stop",
            1,
            lt.verdict("constant_y", n_earth=fc["n_earth"], n_legacy=fc["n_legacy"]),
        )
        return final(row, f)
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
            steps = row["forward"].setdefault("rss_steps", [0, 0])  # compared, agree
            steps[0] += 1
            if rel <= tol:
                steps[1] += 1
                row["forward"]["rss_rel_max"] = max(
                    rel, row["forward"].get("rss_rel_max", 0.0)
                )
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
            lt.classify_stop({**ev, **stop_evidence(f)}),
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
        "e_knot_inactive": bool(e_step) and inactive_knot(f, k, e_step),
        "earth_stopped": not e_step,
        "probe": f.probe,
        "fast_k": earth_args(f.case).get("fast.k"),
    }


def earth_args(case: Case) -> dict[str, Any]:
    """earth's arguments for the case: a new run's, or the reference fixture's."""
    return case.earth_args or fixture(case.reference)["earth_args"]


def earth_penalty(case: Case) -> float:
    """earth's GCV penalty d: its argument, or its default (GCV-4)."""
    args = earth_args(case)
    return float(args.get("penalty", 2 if args["degree"] == 1 else 3))


def inactive_knot(f: Fit, k: int, e_step: frozenset[lt.Sig]) -> bool:
    """Whether earth's knot at the 0-based step k is the value of no case
    where its parent (a term of earth's basis before the step) is positive."""
    rows = [0] + [r for g in f.e_groups[:k] for r in g]
    before = {f.e_sigs[r] for r in rows}
    for sig in e_step:
        for var, code, knot in sig:
            parent = tuple(g for g in sig if g[0] != var)
            if code in (1, -1) and parent in before:
                active = f.X[lt.term_column(parent, f.X) > 0, var]
                return bool(parent) and not np.any(active == knot)
    return False


def stop_evidence(f: Fit) -> dict[str, Any]:
    """Why the legacy forward pass stopped, and its best candidate then."""
    last = f.leg["searches"][-1]
    chosen = last["chosen"] or {}
    new = frozenset(lt.sig_from_json(s) for s in chosen.get("new", []))
    gain = last["rss_before"] - chosen["rss"] if chosen else None
    second = last.get("second") or {}
    gain2 = last["rss_before"] - second["rss"] if second else None
    return {
        "legacy_stop": legacy_stop(f),
        "chosen": lt.terms_label(new) or "none",
        "gain": gain,
        "gain2": gain2,
        "rss_before": last["rss_before"],
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


def newest_codes(leg: dict[str, Any]) -> list[int | None]:
    """The code of each legacy term's newest factor (the term minus its
    parent): the legacy C counts the terms whose newest factor is a hinge."""
    sigs = [lt.sig_from_json(t["sig"]) for t in leg["terms"]]
    out: list[int | None] = []
    for sig, t in zip(sigs, leg["terms"], strict=True):
        parent = sigs[t["parent"]] if t["parent"] is not None else ()
        new = [factor[1] for factor in sig if factor not in parent]
        out.append(new[0] if new else None)
    return out


def pruning(add: Any, row: dict[str, Any], f: Fit) -> None:
    """When both forward passes added the same terms: the pruning path size
    by size in earth's term numbering (the legacy path gives one subset per
    size), with its GCVs, and the selected terms. The difference that decides
    the selected model comes first: the intercept missing from it (F3); else,
    when the selections differ, other GCVs at a size where both paths keep
    the same subset (F4), other subsets at a selected size, or a tie between
    sizes. Then the path's other differences: the intercept dropped below the
    selected size (F3), other subsets at other sizes. When the selected terms
    agree, the final model goes through compare_fit."""
    leg, earth = f.leg, f.earth
    to_earth = [f.e_sigs.index(lt.sig_from_json(t["sig"])) + 1 for t in leg["terms"]]
    codes = newest_codes(leg)
    path = zip(leg["prune_subsets"], leg["prune_rss"], leg["prune_gcv"], strict=True)
    by_size = {len(sub): (sub, r, g) for sub, r, g in path}
    mine = {m: frozenset(to_earth[i] for i in v[0]) for m, v in by_size.items()}
    theirs = {
        m + 1: frozenset(t for t in ts if t)
        for m, ts in enumerate(earth["prune_terms"])
    }
    sel_l = frozenset(to_earth[i] for i in leg["selected"])
    sel_e = frozenset(earth["selected_terms"])
    counts = {"legacy": len(sel_l), "earth": len(sel_e)}
    no_int = sorted(m for m in mine if 1 not in mine[m])
    other = sorted(m for m in mine if m not in no_int and mine[m] != theirs.get(m))
    apart = {}
    for m in mine:
        if m in no_int or m in other or not np.isfinite(by_size[m][2]):
            continue
        g_e = earth["gcv_per_subset"][m - 1]
        if abs(by_size[m][2] - g_e) > compare.GCV_REL * (abs(g_e) or 1.0):
            apart[m] = g_e

    def subset_verdict(m: int) -> lt.Verdict:
        rss_l, rss_e = by_size[m][1], earth["rss_per_subset"][m - 1]
        key = "prune_subset" if rss_l > rss_e * (1 + lt.NEAR_TIE_REL) else "prune_tie"
        return lt.verdict(key, size=m, rss_l=rss_l, rss_e=rss_e)

    def gcv_verdict(m: int) -> lt.Verdict:
        sub = by_size[m][0]
        h = sum(codes[i] in (1, -1) for i in sub)
        lin = sum(codes[i] == 2 for i in sub)
        d_l, d_e = float(f.case.legacy.get("penalty", 3.0)), earth_penalty(f.case)
        c_l, c_e = m + d_l * h, m + d_e * (m - 1) / 2
        return lt.verdict(
            "gcv_convention",
            size=m,
            h=h,
            lin=lin,
            d_l=d_l,
            d_e=d_e,
            c_l=c_l,
            c_e=c_e,
            gcv_l=by_size[m][2],
            gcv_e=apart[m],
            **counts,
        )

    done = set()
    if 1 not in sel_l:
        add(
            "pruning",
            None,
            lt.verdict("intercept_final", terms=len(sel_l), size=max(no_int)),
        )
    elif sel_l != sel_e:
        chosen = [m for m in (len(sel_e), len(sel_l)) if m in apart]
        at = [m for m in (len(sel_e), len(sel_l)) if m in other]
        if chosen or apart:
            m = (chosen or sorted(apart, reverse=True))[0]
            add("pruning", None, gcv_verdict(m))
            done.add("gcv")
        elif at:
            add("pruning", None, subset_verdict(at[0]))
            done.add(at[0])
        else:
            add("pruning", None, lt.verdict("size_tie", **counts))
    if no_int and 1 in sel_l:
        add(
            "pruning",
            None,
            lt.verdict("intercept_path", size=max(no_int), selected=len(sel_l)),
        )
    rest = [m for m in other if m not in done]
    if rest:
        add("pruning", None, subset_verdict(max(rest)))
    if apart and "gcv" not in done:
        add("pruning", None, gcv_verdict(max(apart)))
    if sel_l != sel_e:
        return
    order = np.argsort([to_earth[i] for i in leg["selected"]])
    ours = {"coef": np.asarray(leg["coef"])[order].reshape(-1, 1).tolist()}
    if not apart:  # other GCVs are F4, reported above
        ours["gcv"] = leg["gcv"]
    if f.case.dataset not in GLM:  # there, earth's fitted values are glm probabilities
        if f.case.reference == f.case.fixture:  # else earth fits repeated rows
            ours["fitted"] = np.reshape(leg["pred_train"], (-1, 1)).tolist()
        if leg["pred_test"] is not None:
            ours["pred_test"] = np.reshape(leg["pred_test"], (-1, 1)).tolist()
    B = lt.basis_matrix([f.e_sigs[r - 1] for r in earth["selected_terms"]], f.X)
    sd = float(np.std(f.y)) or float(np.max(np.abs(f.y))) or 1.0  # a constant y
    theirs_fit = {k: earth.get(k) for k in ours}
    diffs = compare.compare_fit(
        ours, theirs_fit, kappa=float(np.linalg.cond(B)), sd_y=sd
    )
    row["pruning"] = [d.field for d in diffs]
    for d in diffs[:1]:
        metric = "" if d.metric is None else f"{d.metric:.3g}"
        label = d.label or "unexplained"
        v = lt.verdict(
            "final",
            label=label,
            field=d.field,
            detail=d.detail,
            metric=metric,
            tol=d.tolerance,
        )
        add("final", None, v)


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
        glm = f.case.dataset in GLM  # earth's predictions are glm probabilities
        if leg.get("pred_test") and earth.get("pred_test") and not glm:
            diff = (
                np.abs(np.ravel(leg["pred_test"]) - np.ravel(earth["pred_test"]))
                / y.std()
            )
            X, X_test = f.X, f.ref["X_test"]
            box = (X_test >= X.min(axis=0)) & (X_test <= X.max(axis=0))
            inside = diff[np.all(box, axis=1)]
            fin["pred_diff_sd"] = [float(inside.max(initial=0)), float(diff.max())]
    truth = f.own.get("truth")
    if truth is not None:
        fin["test_mse"] = [
            float(np.mean((np.ravel(d["pred_test"]) - truth) ** 2))
            for d in (leg, earth)
        ]
    no_int = [len(sub) for sub in leg["prune_subsets"] if 0 not in sub]
    fin["intercept_dropped_at"] = max(no_int, default=None)
    fin["intercept"] = 0 in leg["selected"]
    fin["fit_time"] = leg.get("fit_time")
    if 0 not in leg["selected"] and not any(d["finding"] == "F3" for d in row["diffs"]):
        v = lt.verdict("intercept_final", terms=len(leg["selected"]), size=max(no_int))
        row["diffs"].append(
            {
                "kind": "pruning",
                "step": None,
                "label": v[0],
                "finding": v[1],
                "text": v[2],
            }
        )
    if f.case.dataset in GLM:
        row["diffs"].append(glm_refit(f))
    row["final"] = fin
    return row


def glm_numbers(f: Fit) -> tuple[float, float, bool]:
    """F9 on a binary fit: the largest difference of the legacy fitted
    probabilities from an unpenalized logistic refit on the legacy code's own
    selected basis, the largest from earth's glm, and whether the two
    programs selected the same terms."""
    import warnings

    from sklearn.linear_model import LogisticRegression

    leg, X = f.leg, f.own["X"]
    sel = [lt.sig_from_json(leg["terms"][i]["sig"]) for i in leg["selected"]]
    p_leg = np.asarray(leg["proba_train"])[:, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unpen = LogisticRegression(C=np.inf, fit_intercept=False, max_iter=100000)
        B = lt.basis_matrix(sel, X)
        p_unpen = unpen.fit(B, f.own["y"]).predict_proba(B)[:, 1]
    same = sorted(sel) == sorted(f.e_sigs[r - 1] for r in f.earth["selected_terms"])
    own = float(np.max(np.abs(p_leg - p_unpen)))
    return own, float(np.max(np.abs(p_leg - np.ravel(f.earth["pred_train"])))), same


def glm_refit(f: Fit) -> dict[str, Any]:
    """The F9 difference of a binary fit (glm_numbers), as a diff entry."""
    own, diff, same = glm_numbers(f)
    v = lt.verdict("glm", diff_own=own, diff=diff, same="the same" if same else "other")
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
    no_int = [r for r in rows if r.get("final", {}).get("intercept") is False]
    agree = [
        r["forward"]["rss_rel_max"]
        for r in rows
        if "rss_rel_max" in r.get("forward", {})
    ]
    rss_steps = [
        r["forward"]["rss_steps"] for r in rows if "rss_steps" in r.get("forward", {})
    ]
    for mode in s15:
        mse = [
            r["final"]["test_mse"]
            for r in rows
            if r["id"].startswith("S15/")
            and r["mode"] == mode
            and "test_mse" in r.get("final", {})
        ]
        ratio = np.array([a / b for a, b in mse])
        s15[mode]["test_mse_ratio"] = (
            {
                "median": float(np.median(ratio)),
                "legacy_lower": int(np.sum(ratio < 1)),
                "above_10": int(np.sum(ratio > 10)),
                "max": float(ratio.max()),
            }
            if len(ratio)
            else {}
        )
    times = sorted(
        (
            (r["final"]["fit_time"], r["id"])
            for r in rows
            if r.get("final", {}).get("fit_time")
        ),
        reverse=True,
    )
    return {
        "fits": len(rows),
        "fits_with_differences": len(first),
        "final_without_intercept": lt.count_by(
            ({"mode": r["mode"]} for r in no_int), "mode"
        ),
        "fits_by_mode": lt.count_by(
            ({"mode": r["mode"]} for r in rows if r.get("final")), "mode"
        ),
        "rss_steps_compared": sum(c for c, _ in rss_steps),
        "rss_steps_agree": sum(a for _, a in rss_steps),
        "rss_rel_max_agreeing_steps": max(agree, default=None),
        "slowest_legacy_fits": times[:4],
        "first_by_label": lt.count_by(first, "label"),
        "first_by_finding": lt.count_by(first, "finding"),
        "all_by_label": lt.count_by(diffs, "label"),
        "all_by_finding": lt.count_by(diffs, "finding"),
        "s15": s15,
    }


S12_BACK = {  # a knot of each S12 variant, in the base variant's units
    "S12_base": lambda k: k,
    "S12_x_1e8": lambda k: k / 1e8,
    "S12_x_1em8": lambda k: k / 1e-8,
    "S12_x_plus_1e6": lambda k: k - 1e6,
    "S12_y_1e9": lambda k: k,
    "S12_y_1em9": lambda k: k,
}


def s12_invariance(out: Path) -> dict[str, Any]:
    """S12's purpose: per mode, whether each variant selects the same terms
    as the base variant, with the knots in the base units (relative 1e-8),
    for the legacy code and for earth."""

    def knots(sigs: list[lt.Sig], stem: str, back: Any) -> list[float]:
        scale = np.asarray(fixture(stem)["scale"] or [1.0])
        hinge = [(f[0], f[2]) for s in sigs for f in s if f[1] in (1, -1)]
        return sorted(back(k * scale[v]) for v, k in hinge)

    table: dict[str, Any] = {}
    for mode in ("raw_d1", "defaults_d1", "matched_d1"):
        base: dict[str, Any] = {}
        for ds, back in S12_BACK.items():
            stem = f"{ds}_{mode}"
            path = out / "wheel" / f"{slug(f'{ds}/{mode}')}.json"
            if not path.exists():
                continue
            leg, res = json.loads(path.read_text()), fixture(stem)["result"]
            e_sigs = lt.earth_signatures(res["dirs"], res["cuts"])
            sides = {
                "legacy": [
                    lt.sig_from_json(leg["terms"][i]["sig"]) for i in leg["selected"]
                ],
                "earth": [e_sigs[r - 1] for r in res["selected_terms"]],
            }
            for side, sigs in sides.items():
                k = knots(sigs, stem, back)
                ref = base.setdefault(side, (len(sigs), k))
                same = ref[0] == len(sigs) and len(k) == len(ref[1])
                same = same and bool(np.allclose(k, ref[1], rtol=1e-8, atol=0))
                table.setdefault(mode, {}).setdefault(ds, {})[side] = {
                    "terms": len(sigs),
                    "same": same,
                }
    return table


def weights_vs_repeated(out: Path, cases: list[Case]) -> list[dict[str, Any]]:
    """F1 inside the legacy code: HEAD with integer weights against the
    legacy code (the wheel) without weights on the repeated rows."""
    rows = []
    for c in cases:
        a = out / "head" / f"{slug(c.id)}.json"
        b = (
            out
            / "wheel"
            / f"{slug(c.id.replace(c.dataset, c.dataset + '_repeated'))}.json"
        )
        if c.weighted and c.reference != c.fixture and a.exists() and b.exists():
            wa, wb = json.loads(a.read_text()), json.loads(b.read_text())
            sa, sb = legacy_step_terms(wa), legacy_step_terms(wb)
            sel = [
                sorted(lt.sig_from_json(d["terms"][i]["sig"]) for i in d["selected"])
                for d in (wa, wb)
            ]
            diff = np.max(np.abs(np.subtract(wa["pred_test"], wb["pred_test"])))
            rows.append(
                {
                    "id": c.id,
                    "steps": [len(sa), len(sb)],
                    "same_forward": sa == sb,
                    "same_selected": sel[0] == sel[1],
                    "max_pred_diff": float(diff),
                }
            )
    return rows


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
        "s12_invariance": s12_invariance(out),
        "weights_vs_repeated": weights_vs_repeated(out, cases),
        "cases": rows,
    }
    if ns.json:
        write_report(Path(ns.json), _rounded(result))
    if ns.markdown:
        Path(ns.markdown).write_text(markdown(result))
    print(json.dumps(result["summary"], indent=1))


def write_report(path: Path, result: dict[str, Any]) -> None:
    """The summary JSON with one case per line, so that diffs stay readable;
    written atomically, as write_json."""
    parts = []
    for key, value in result.items():
        if key == "cases":
            body = ",\n".join(json.dumps(r) for r in value)
            parts.append(f'"cases": [\n{body}\n]')
        else:
            parts.append(f"{json.dumps(key)}: {json.dumps(value)}")
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        fh.write("{\n" + ",\n".join(parts) + "\n}\n")
    os.replace(tmp, path)


def _rounded(obj: Any) -> Any:
    """Floats to 6 significant digits, to keep the summary JSON small."""
    if isinstance(obj, float):
        return float(f"{obj:.6g}")
    if isinstance(obj, dict):
        return {k: _rounded(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rounded(v) for v in obj]
    return obj


def markdown(result: dict[str, Any]) -> str:
    """The generated tables of DIFFERENCES_legacy.md: one row per fit of S01
    to S20 (S15 summarized), the counts by label and finding, the S15 rates,
    and HEAD against the wheel."""

    def cell(d: dict[str, Any] | None) -> str:
        if d is None:
            return "none | | "
        where = d["kind"] if d["step"] is None else f"{d['kind']} at step {d['step']}"
        return f"{where} | {d['label']} | {d['finding'] or ''}"

    def table(head: list[str], rows: list[list[Any]]) -> list[str]:
        out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        return out + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]

    fits = []
    for r in result["cases"]:
        if not r["id"].startswith("S15/"):
            d, fin = r["diffs"], r.get("final") or {}
            terms = ", ".join(
                str(fin[k]) for k in ("legacy_terms", "earth_terms") if k in fin
            )
            nxt = d[1] if len(d) > 1 else None
            fits.append(
                [r["id"], r["code"], cell(d[0] if d else None), cell(nxt), terms]
            )
    head = ["Fit", "Code", "First difference", "Label", "ID"]
    lines = table([*head, "Next difference", "Label", "ID", "Terms"], fits)
    s = result["summary"]
    by = [(k, s[f"first_by_{k}"], s[f"all_by_{k}"]) for k in ("label", "finding")]
    for name, first, every in by:
        keys = sorted(set(first) | set(every), key=_finding_order)
        rows = [[k, first.get(k, 0), every.get(k, 0)] for k in keys]
        lines += ["", *table([name.title(), "First difference of a fit", "All"], rows)]
    rows = []
    for mode, v in s["s15"].items():
        steps = sorted(v["first_divergence_step"].items(), key=lambda kv: int(kv[0]))
        causes = ", ".join(f"{k}: {n}" for k, n in v["causes"].items())
        steps_txt = ", ".join(f"{k}: {n}" for k, n in steps)
        rows.append([mode, v["fits"], v["no_choice_divergence"], steps_txt, causes])
    head = [
        "S15 mode",
        "Fits",
        "No choice divergence",
        "First divergence: step",
        "Cause",
    ]
    lines += ["", *table(head, rows)]
    rows = []
    for mode, variants in result.get("s12_invariance", {}).items():
        for ds, sides in variants.items():
            cells = [
                f"{v['terms']}, {'same' if v['same'] else 'differs'}"
                for v in sides.values()
            ]
            rows.append([mode, ds, *cells])
    lines += [
        "",
        *table(
            ["S12 mode", "Variant", "Legacy: terms, knots", "earth: terms, knots"], rows
        ),
    ]
    rows = [
        [
            w["id"],
            ", ".join(map(str, w["steps"])),
            w["same_forward"],
            w["same_selected"],
            f"{w['max_pred_diff']:.3g}",
        ]
        for w in result.get("weights_vs_repeated", [])
    ]
    head = [
        "HEAD with weights",
        "Steps (weights, repeated)",
        "Same forward terms",
        "Same selected terms",
        "Largest test difference",
    ]
    lines += ["", *table(head, rows)]
    hw = result["head_vs_wheel"]
    differ = ", ".join(hw["different"]) or "none"
    lines += [
        "",
        f"HEAD and the wheel: {hw['identical']} identical fits; different: {differ}.",
    ]
    return "\n".join(lines) + "\n"


def _finding_order(key: str) -> tuple:
    return (key[0] != "F", int(key[1:]) if key[1:].isdigit() else 0, key)


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
        p.add_argument("--markdown", help="where to write the tables (report)")
    ns = parser.parse_args(argv)
    {"run": cmd_run, "probe": cmd_probe, "report": cmd_report}[ns.cmd](ns)


if __name__ == "__main__":
    main()
