"""The fixture generator framework (VALIDATION_PLAN.md, "Correctness
against earth"): a registry of datasets and a registry of earth argument
sets ("modes"), which together make `validation/fixtures/<dataset>_<mode>
.json`, one file per (dataset, mode) pair, each with that pair's inputs,
earth arguments, `driver.run_earth` result and versions.

This registers one dataset, S01 ("one covariate, two true knots, 200
cases"), under two modes, to prove the pipeline; T05 adds S02 through S20.

Usage (also validation/README.md):
    python validation/harness/gen_fixtures.py           # write every fixture
    python validation/harness/gen_fixtures.py --check   # regenerate into a
                                                         # temp folder and
                                                         # report any diff
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blackbox
import driver
import trace_parse

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
COMPONENTS_DIR = FIXTURES_DIR / "components"


@dataclass
class CheckResult:
    """``check()``'s outcome: ``problems`` (a difference in what a fixture
    claims about earth: its inputs, earth arguments or result, or a
    fixture missing outright) fail the check; ``notes`` (a ``versions``
    difference: this machine's R, earth, numpy, scikit-learn or BLAS is not
    the one that made the committed fixture) do not, since nothing about
    earth conformance follows from running the generator on different
    software versions.
    """

    problems: list[str]
    notes: list[str]


@dataclass
class Dataset:
    """One registered dataset: everything `driver.EarthJob` needs besides
    the earth arguments (which a mode supplies)."""

    id: str
    X: np.ndarray
    y: np.ndarray
    X_test: np.ndarray | None = None
    factor_response: bool = False


DatasetFn = Callable[[], Dataset]
REGISTRY: dict[str, DatasetFn] = {}

# earth argument sets ("modes"), VALIDATION_PLAN.md's "Comparison modes":
# defaults_d1 is earth at its own defaults; matched_d1 is the legacy-code
# matched mode from the seed prototype (validation/legacy/compare_earth.py),
# hinge-only and every case a candidate knot, so the two implementations
# should make the same forward choices except at near-ties.
MODES: dict[str, dict[str, Any]] = {
    "defaults_d1": {"degree": 1},
    "matched_d1": {
        "degree": 1,
        "penalty": 2,
        "nk": 21,
        "thresh": 0,
        "minspan": 1,
        "endspan": 1,
        "fast.k": 0,
        "Auto.linpreds": False,
        "pmethod": "backward",
    },
}


def register(fn: DatasetFn) -> DatasetFn:
    """Decorator: register a dataset generator under its own return value's
    ``id`` (calling it once, at import time, to read that id)."""
    REGISTRY[fn().id] = fn
    return fn


@register
def s01() -> Dataset:
    """S01: one covariate, two true knots, 200 cases. Knot recovery."""
    rng = np.random.default_rng(1)
    n, n_test = 200, 50
    x = rng.uniform(0, 1, size=(n, 1))
    x_test = rng.uniform(0, 1, size=(n_test, 1))

    def f(x0: np.ndarray) -> np.ndarray:
        return 2 * np.maximum(0, x0 - 0.3) - 3 * np.maximum(0, x0 - 0.7)

    y = f(x[:, 0]) + rng.normal(scale=0.1, size=n)
    return Dataset(id="S01", X=x, y=y, X_test=x_test)


# Component fixtures (VALIDATION_PLAN.md, "Component tests"; T05 brief,
# deliverable 2): each one calls a blackbox.py function directly, rather
# than building an EarthJob for driver.py, since a component test targets
# one earth-adjacent internal (earth:::get.gcv, earth:::pruning.pass,
# lm.fit, predict.earth, glm.fit, nnet::multinom) or the trace-based knot
# scan, not a whole earth() fit compared end to end. Each generator returns
# a JSON-able payload (no "versions" key yet; _component_payload adds it,
# the same way _fixture_payload does for a dataset).
ComponentFn = Callable[[], dict[str, Any]]
COMPONENT_REGISTRY: dict[str, ComponentFn] = {}


def register_component(fn: ComponentFn) -> ComponentFn:
    """Decorator: register a component-fixture generator under its function
    name. Unlike ``register`` (datasets), this does not call ``fn`` at
    import time: a component payload comes from a real earth call (through
    ``blackbox.py``), so building it eagerly on every ``import
    gen_fixtures`` would run R on every test collection, not just when a
    fixture is actually (re)generated."""
    COMPONENT_REGISTRY[fn.__name__] = fn
    return fn


@register_component
def gcv_grid() -> dict[str, Any]:
    """VALIDATION_PLAN.md, "Component tests" > "GCV function": earth:::
    get.gcv over a grid of (rss.per.subset, 1 to 41 terms) at penalties 0 to
    6 and -1, and n from 10 to 100,000. get.gcv's formula (GCV-2) only needs
    one RSS value per term count, not a real fit, so this uses one fixed,
    deterministic, strictly decreasing sequence for every cell."""
    nterms = list(range(1, 42))
    rss_per_subset = [100.0 / m for m in nterms]
    penalties = [0, 1, 2, 3, 4, 5, 6, -1]
    ncases = [10, 15, 20, 41, 50, 100, 1_000, 100_000]
    grid = [
        {
            "penalty": penalty,
            "ncases": n,
            "gcv": blackbox.get_gcv(rss_per_subset, nterms, penalty, n).tolist(),
        }
        for penalty in penalties
        for n in ncases
    ]
    return {"nterms": nterms, "rss_per_subset": rss_per_subset, "grid": grid}


def _pruning_case(
    X: np.ndarray, y: np.ndarray, earth_args: dict[str, Any], penalty: float
) -> dict[str, Any]:
    basis = blackbox.fit_bx_dirs(X, y, earth_args)
    pruned = blackbox.pruning_pass(
        X, y, basis["bx"], basis["dirs"], penalty, pmethod="backward"
    )
    return {
        "y": y.tolist(),
        "dirs": basis["dirs"].tolist(),
        "cuts": basis["cuts"].tolist(),
        "rss_per_subset": pruned["rss_per_subset"].tolist(),
        "gcv_per_subset": pruned["gcv_per_subset"].tolist(),
        "prune_terms": pruned["prune_terms"].tolist(),
        "selected_terms": pruned["selected_terms"].tolist(),
    }


@register_component
def pruning_fixed_basis() -> dict[str, Any]:
    """VALIDATION_PLAN.md, "Component tests" > "Pruning of a fixed basis":
    earth's own earth:::pruning.pass on a real forward basis (built with
    pmethod = "none", so there is no earth-side pruning to undo first), for
    one response and for several. T08 builds the same terms in pymars
    (from ``dirs``/``cuts``, TERM-3) and runs ``_pruning.py`` on them, so
    ``bx`` itself does not need to be in the fixture."""
    rng = np.random.default_rng(11)
    n = 120
    X = np.column_stack([rng.uniform(0, 1, size=n), rng.uniform(0, 1, size=n)])
    y1 = (
        2 * np.maximum(0, X[:, 0] - 0.3)
        - 1.5 * np.maximum(0, 0.3 - X[:, 0])
        + 1.2 * np.maximum(0, X[:, 0] - 0.5) * np.maximum(0, X[:, 1] - 0.4)
        + rng.normal(scale=0.05, size=n)
    )
    y2 = 0.5 * y1 + rng.normal(scale=0.05, size=n)
    y3 = -0.3 * y1 + rng.normal(scale=0.05, size=n)
    y_multi = np.column_stack([y1, y2, y3])
    earth_args = {
        "degree": 2,
        "pmethod": "none",
        "nk": 15,
        "thresh": 0,
        "minspan": 1,
        "endspan": 1,
        "fast.k": 0,
        "Auto.linpreds": False,
    }
    penalty = 2.0
    return {
        "X": X.tolist(),
        "earth_args": earth_args,
        "penalty": penalty,
        "one_response": _pruning_case(X, y1, earth_args, penalty),
        "several_responses": _pruning_case(X, y_multi, earth_args, penalty),
    }


@register_component
def lm_fit_coefficients() -> dict[str, Any]:
    """VALIDATION_PLAN.md, "Component tests" > "Coefficients of fixed
    terms": pymars' weighted least-squares coefficients (LA-4) are compared
    with R's lm.fit on the same columns; a duplicated column exercises
    LA-4's aliased-coefficient (rank-deficient) rule."""
    rng = np.random.default_rng(21)
    n = 80
    x0 = rng.uniform(0, 1, size=n)
    x1 = rng.uniform(0, 1, size=n)
    y = 1.5 + 2.0 * x0 - 0.7 * x1 + rng.normal(scale=0.1, size=n)

    def case(X: np.ndarray, label: str) -> dict[str, Any]:
        result = blackbox.lm_fit(X, y)
        coefficients = result["coefficients"].ravel().tolist()
        return {
            "label": label,
            "x": X.tolist(),
            "coefficients": [None if c is None else float(c) for c in coefficients],
            "residuals": result["residuals"].ravel().tolist(),
            "rank": int(result["rank"]),
        }

    full_rank = np.column_stack([np.ones(n), x0, x1])
    duplicated = np.column_stack([np.ones(n), x0, x0])
    return {
        "y": y.tolist(),
        "cases": [case(full_rank, "full_rank"), case(duplicated, "duplicated_column")],
    }


@register_component
def predict_new_points() -> dict[str, Any]:
    """VALIDATION_PLAN.md, "Component tests" > "Prediction at new points":
    the basis evaluated at points outside the training range (TERM-3:
    "nothing is clipped"), compared with predict.earth; degree 1 (one
    covariate) and degree 2 (an interaction)."""
    rng = np.random.default_rng(31)
    n = 100
    x = rng.uniform(0, 1, size=(n, 1))
    y = 2 * np.maximum(0, x[:, 0] - 0.4) + rng.normal(scale=0.05, size=n)
    earth_args_d1 = {"degree": 1, "nk": 11, "thresh": 0}
    newx_d1 = np.array([[-2.0], [-0.5], [0.0], [0.4], [0.9], [1.0], [1.5], [3.0]])
    pred_d1 = blackbox.predict_earth(x, y, newx_d1, earth_args_d1)

    rng2 = np.random.default_rng(32)
    n2 = 150
    x2 = rng2.uniform(0, 1, size=(n2, 2))
    y2 = 2 * np.maximum(0, x2[:, 0] - 0.3) * np.maximum(
        0, x2[:, 1] - 0.5
    ) + rng2.normal(scale=0.05, size=n2)
    earth_args_d2 = {"degree": 2, "nk": 15, "thresh": 0}
    newx_d2 = np.array([[-1.0, -1.0], [0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [2.0, 2.0]])
    pred_d2 = blackbox.predict_earth(x2, y2, newx_d2, earth_args_d2)

    return {
        "degree1": {
            "x": x.tolist(),
            "y": y.tolist(),
            "earth_args": earth_args_d1,
            "newx": newx_d1.tolist(),
            "pred": pred_d1.tolist(),
        },
        "degree2": {
            "x": x2.tolist(),
            "y": y2.tolist(),
            "earth_args": earth_args_d2,
            "newx": newx_d2.tolist(),
            "pred": pred_d2.tolist(),
        },
    }


@register_component
def classifier_refit() -> dict[str, Any]:
    """VALIDATION_PLAN.md, "Binary outcomes": R's glm (binomial) and
    nnet::multinom on fixed columns, the reference for EarthClassifier's
    GLM-refit coefficients and fitted probabilities (binary and, for three
    or more classes, the multinom comparison the plan calls for)."""
    rng = np.random.default_rng(41)
    n = 300
    x0 = rng.uniform(-2, 2, size=n)
    x1 = rng.uniform(-2, 2, size=n)
    X = np.column_stack([np.ones(n), x0, x1])
    eta = 0.8 * x0 - 0.5 * x1
    p = 1.0 / (1.0 + np.exp(-eta))
    y_binary = (rng.uniform(size=n) < p).astype(float)
    glm = blackbox.glm_fit(X, y_binary, family="binomial")

    rng2 = np.random.default_rng(42)
    n2 = 300
    z0 = rng2.uniform(-3, 3, size=n2)
    z1 = rng2.uniform(-3, 3, size=n2)
    Xm = np.column_stack([np.ones(n2), z0, z1])
    score = z0 - 0.5 * z1
    y3 = np.where(score < -1, "lo", np.where(score > 1, "hi", "mid"))
    multinom = blackbox.multinom_fit(Xm, y3)

    return {
        "binomial": {
            "x": X.tolist(),
            "y": y_binary.tolist(),
            "coefficients": [
                None if c is None else float(c)
                for c in glm["coefficients"].ravel().tolist()
            ],
            "fitted_values": glm["fitted_values"].tolist(),
        },
        "multinomial": {
            "x": Xm.tolist(),
            "y": y3.tolist(),
            "coefficients": multinom["coefficients"].tolist(),
            "levels": multinom["levels"],
            "fitted": multinom["fitted"].tolist(),
        },
    }


# Candidate knot sets (VALIDATION_PLAN.md, "Component tests" > "Candidate
# knot sets"): the grid is minspan automatic/1/3/10, endspan automatic/1/5,
# degree 1 and 2, Adjust.endspan 1 and 2 (degree 2 only: SPAN-4 says it does
# not affect an intercept parent), and 20, 200 and 2,000 cases. A negative
# minspan is a `later` issue (T05 brief), so it is left out. Trimmed for
# fixture size (validation/README.md, "keep validation/fixtures/ small"):
# every visited case prints one trace = 9 line whether evaluated or
# span-skipped, so trace size scales with n regardless of minspan, and the
# full (minspan, endspan, Adjust.endspan) cross is the same combinatorial
# scan at any n; so it runs once, at the cheap n = 20, and n = 200 and
# 2,000 (which mainly test the automatic formulas' own dependence on n, the
# one thing n = 20 cannot) run at automatic minspan and endspan alone, still
# crossed with Adjust.endspan at degree 2. nk = 5 (at most 2 forward steps)
# keeps every combo to the searches the knot-set tests need.
_KNOT_MINSPANS: tuple[int | None, ...] = (None, 1, 3, 10)
_KNOT_ENDSPANS: tuple[int | None, ...] = (None, 1, 5)
_KNOT_ADJUST_ENDSPANS: tuple[float, ...] = (1.0, 2.0)
_KNOT_N_FULL = 20
_KNOT_N_REDUCED: tuple[int, ...] = (200, 2000)


def _knot_grid_xy(n: int, p: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, p))
    y = 2 * np.maximum(0, X[:, 0] - 0.4)
    if p > 1:
        y = y + 1.5 * np.maximum(0, X[:, 1] - 0.6)
    y = y + rng.normal(scale=0.02, size=n)
    return X, y


def _knot_jobs() -> list[tuple[dict[str, Any], driver.EarthJob]]:
    """Every (metadata, EarthJob) pair of the candidate-knot-set grid."""
    jobs: list[tuple[dict[str, Any], driver.EarthJob]] = []
    counter = 0

    def add(
        degree: int,
        minspan: int | None,
        endspan: int | None,
        adjust_endspan: float,
        n: int,
    ) -> None:
        nonlocal counter
        p = 1 if degree == 1 else 2
        X, y = _knot_grid_xy(n, p, seed=9_000 + counter)
        earth_args = {
            "degree": degree,
            "nk": 5,
            "thresh": 0,
            "pmethod": "none",
            "minspan": minspan if minspan is not None else 0,
            "endspan": endspan if endspan is not None else 0,
            "Adjust.endspan": adjust_endspan,
            "fast.k": 0,
        }
        job_id = f"knot_{counter:03d}"
        meta = {
            "degree": degree,
            "minspan": minspan,
            "endspan": endspan,
            "adjust_endspan": adjust_endspan,
            "n": n,
            "p": p,
        }
        job = driver.EarthJob(
            id=job_id,
            X=X,
            y=y,
            earth_args=earth_args,
            trace=9,
            include_forward_path=False,
        )
        jobs.append((meta, job))
        counter += 1

    for degree in (1, 2):
        adjust_values = _KNOT_ADJUST_ENDSPANS if degree == 2 else (2.0,)
        for minspan in _KNOT_MINSPANS:
            for endspan in _KNOT_ENDSPANS:
                for adjust_endspan in adjust_values:
                    add(degree, minspan, endspan, adjust_endspan, _KNOT_N_FULL)
        for adjust_endspan in adjust_values:
            for n in _KNOT_N_REDUCED:
                add(degree, None, None, adjust_endspan, n)
    return jobs


@register_component
def knot_candidates() -> dict[str, Any]:
    """VALIDATION_PLAN.md, "Component tests" > "Candidate knot sets": earth's
    trace = 9 case-by-case knot scan, across the grid above, for T08's
    `_knots.py` candidate-value tests to compare against. This only records
    what earth printed (a black-box trace, COMMON.md's clean room); the
    comparison, and any interpretation of which cases earth accepted or
    rejected, is T08's (T05 brief: "you do not interpret earth's internals
    beyond what the spec states")."""
    metas_jobs = _knot_jobs()
    metas = [m for m, _ in metas_jobs]
    jobs = [j for _, j in metas_jobs]
    cases = []
    with tempfile.TemporaryDirectory(prefix="pymars-knot-grid-") as tmp:
        workdir = Path(tmp)
        results = driver.run_earth(jobs, workdir=workdir)
        for meta, job in zip(metas, jobs, strict=True):
            result = results[job.id]
            trace_text = (workdir / f"{job.id}_trace.txt").read_text(encoding="utf-8")
            trace_parse.parse_trace(  # fail loudly here, not later in T08
                workdir / f"{job.id}_trace.txt"
            )
            cases.append(
                {
                    **meta,
                    "X": job.X.tolist(),
                    "y": job.y.tolist(),
                    "earth_args": dict(job.earth_args),
                    "dirs": result["dirs"],
                    "cuts": result["cuts"],
                    "selected_terms": result["selected_terms"],
                    "trace_text": trace_text,
                }
            )
    return {"cases": cases}


def _sanitize_versions(versions: dict[str, Any]) -> dict[str, Any]:
    """Drop the parts of ``driver.versions()`` that vary by machine or by
    commit without saying anything about earth conformance: an absolute
    path (threadpoolctl's and R's BLAS ``filepath``/``la_library`` include
    the machine's home folder, which also does not belong in a public
    repository) becomes just its basename, and ``pymars_commit`` is left
    out of the fixture entirely (``check()`` reports it separately; a
    fixture regenerated on a later commit always differs there, by
    design).
    """
    out = dict(versions)
    out.pop("pymars_commit", None)
    if out.get("blas"):
        out["blas"] = [
            {
                **lib,
                "filepath": Path(lib["filepath"]).name if lib.get("filepath") else None,
            }
            for lib in out["blas"]
        ]
    if out.get("r_blas"):
        out["r_blas"] = {
            key: (Path(value).name if value else None)
            for key, value in out["r_blas"].items()
        }
    return out


def _fixture_payload(dataset_id: str, mode: str) -> dict[str, Any]:
    ds = REGISTRY[dataset_id]()
    earth_args = MODES[mode]
    job = driver.EarthJob(
        id=f"{dataset_id}_{mode}",
        X=ds.X,
        y=ds.y,
        X_test=ds.X_test,
        earth_args=earth_args,
        factor_response=ds.factor_response,
    )
    with tempfile.TemporaryDirectory(prefix="pymars-fixture-") as tmp:
        result = driver.run_earth([job], workdir=Path(tmp))[job.id]
    return {
        "dataset": dataset_id,
        "mode": mode,
        "inputs": {
            "X": ds.X.tolist(),
            "y": ds.y.tolist(),
            "X_test": ds.X_test.tolist() if ds.X_test is not None else None,
        },
        "earth_args": earth_args,
        "result": result,
        "versions": _sanitize_versions(driver.versions(result)),
    }


def _component_payload(name: str) -> dict[str, Any]:
    payload = COMPONENT_REGISTRY[name]()
    versions = driver.versions()
    versions.update(blackbox.versions())
    return {"component": name, **payload, "versions": _sanitize_versions(versions)}


def _write_component(name: str, components_dir: Path) -> Path:
    payload = _component_payload(name)
    components_dir.mkdir(parents=True, exist_ok=True)
    path = components_dir / f"{name}.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def make_all_components(*, fixtures_dir: Path | None = None) -> list[Path]:
    """Write every registered component fixture (``validation/fixtures/
    components/<name>.json``) and return their paths, in registration
    order. ``fixtures_dir`` is the datasets' own root (its ``components``
    subfolder is where these go), read the same dynamic way ``make_all``
    reads its own default, for the same monkeypatching reason."""
    fixtures_dir = fixtures_dir if fixtures_dir is not None else FIXTURES_DIR
    components_dir = fixtures_dir / "components"
    return [_write_component(name, components_dir) for name in COMPONENT_REGISTRY]


def _write_fixture(dataset_id: str, mode: str, fixtures_dir: Path) -> Path:
    payload = _fixture_payload(dataset_id, mode)
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    path = fixtures_dir / f"{dataset_id}_{mode}.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def make_all(*, fixtures_dir: Path | None = None) -> list[Path]:
    """Write every registered (dataset, mode) fixture and return their
    paths, in registration order.

    ``fixtures_dir`` defaults to the module-level ``FIXTURES_DIR``, read
    inside this function (not bound as a default parameter value), so
    ``monkeypatch.setattr(gen_fixtures, "FIXTURES_DIR", ...)`` in a test
    changes what a plain ``make_all()`` call does.
    """
    fixtures_dir = fixtures_dir if fixtures_dir is not None else FIXTURES_DIR
    return [
        _write_fixture(dataset_id, mode, fixtures_dir)
        for dataset_id in REGISTRY
        for mode in MODES
    ]


def check(*, fixtures_dir: Path | None = None) -> CheckResult:
    """Regenerate every dataset and component fixture into a temporary
    folder and compare each one against ``fixtures_dir`` (default: the
    module-level ``FIXTURES_DIR``, read dynamically the same way
    ``make_all`` does).

    Every key but ``versions`` (what a fixture claims about earth: for a
    dataset, ``dataset``, ``mode``, ``inputs``, ``earth_args`` and
    ``result``; for a component, ``component`` and its own payload) or a
    fixture missing outright can add to ``.problems``; a ``versions``
    difference (this machine's R, earth, numpy, scikit-learn or BLAS is not
    the one that made the committed fixture) instead adds to ``.notes``,
    since ``_sanitize_versions`` already drops the one field
    (``pymars_commit``) that would differ on every run from a different
    commit by construction, and nothing about earth conformance follows
    from the rest of ``versions`` differing.
    """
    fixtures_dir = fixtures_dir if fixtures_dir is not None else FIXTURES_DIR
    problems: list[str] = []
    notes: list[str] = []
    with tempfile.TemporaryDirectory(prefix="pymars-fixture-check-") as tmp:
        tmp = Path(tmp)
        fresh_paths = make_all(fixtures_dir=tmp) + make_all_components(fixtures_dir=tmp)
        for fresh in fresh_paths:
            relative = fresh.relative_to(tmp)
            committed = fixtures_dir / relative
            if not committed.is_file():
                problems.append(f"{relative}: missing from {fixtures_dir}")
                continue
            fresh_payload = json.loads(fresh.read_text(encoding="utf-8"))
            committed_payload = json.loads(committed.read_text(encoding="utf-8"))
            fresh_checked = {k: v for k, v in fresh_payload.items() if k != "versions"}
            committed_checked = {
                k: v for k, v in committed_payload.items() if k != "versions"
            }
            if fresh_checked != committed_checked:
                problems.append(f"{relative}: differs from the committed fixture")
            elif fresh_payload.get("versions") != committed_payload.get("versions"):
                notes.append(f"{relative}: versions differ")
    return CheckResult(problems=problems, notes=notes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="remake every fixture in a temp folder and report any diff",
    )
    args = parser.parse_args(argv)

    if args.check:
        result = check()
        for message in result.notes:
            print(message, file=sys.stderr)
        for message in result.problems:
            print(message, file=sys.stderr)
        if result.problems:
            print(
                f"{len(result.problems)} fixture(s) did not reproduce exactly",
                file=sys.stderr,
            )
            return 1
        print("every fixture reproduced exactly")
        return 0

    paths = make_all() + make_all_components()
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
