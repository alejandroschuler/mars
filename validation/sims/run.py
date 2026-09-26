"""Run the statistical performance study (VALIDATION_PLAN.md, "Build").

Usage (from the repository root, after ``. dev/env.sh``)::

    uv run --frozen python -m validation.sims.run --out validation/runs/smoke \\
        --arms E-def,OLS,HGB --sizes 200 --n-jobs 2 --reps 0:2 --resume

Results land under ``--out``, one JSON file per (cell, arm, repetition):
``<out>/<cell-name>/<arm>/rep<NNNN>.json``, written atomically (a temp file,
then ``os.replace``). ``--resume`` skips any such file that already exists
and parses as JSON; a missing or truncated file (an interrupted run) is
redone. ``<out>/manifest.json`` records this invocation's settings, package
versions, the pymars commit, the earth version and each requested arm's
source hash. Long runs are meant to start under
``nohup caffeinate -i nice -n 15 ...``; this script writes its own PID to
``<out>/run.pid`` at startup.

Predictions are also cached, keyed by a hash of the training and test data,
the arm's name, its fixed settings, its source code and the package
versions, in the git-ignored ``validation/runs/.cache/`` (shared across every
``--out``, so a rerun with different filters still reuses earlier fits). The
per-repetition result files under ``--out`` are the record; the cache only
saves time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from . import dgps, learners, metrics, seeds

TEST_N = 10_000
DEFAULT_BLOCK_SIZE = 20
CACHE_DIR = learners.REPO_ROOT / "validation" / "runs" / ".cache"


# ---------------------------------------------------------------------------
# Versions (computed once per invocation; fed into every cache key so a
# dependency upgrade cannot silently reuse a stale prediction).
# ---------------------------------------------------------------------------


def _pymars_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(learners.REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _r_earth_version() -> str | None:
    try:
        out = subprocess.run(
            ["Rscript", "-e", "cat(as.character(packageVersion('earth')))"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _legacy_versions() -> dict:
    try:
        out = subprocess.run(
            [
                str(learners.LEGACY_PYTHON),
                "-c",
                "import importlib.metadata as m;"
                "print(m.version('mars-earth'), m.version('scikit-learn'))",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        mars_v, sk_v = out.stdout.split()
        return {"legacy_mars_earth": mars_v, "legacy_sklearn": sk_v}
    except (OSError, subprocess.CalledProcessError, ValueError):
        return {"legacy_mars_earth": None, "legacy_sklearn": None}


def gather_versions(arm_names: list[str]) -> dict:
    import scipy
    import sklearn

    import pymars

    versions = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "pymars": pymars.__version__,
        "pymars_commit": _pymars_commit(),
    }
    kinds = {learners.ARMS[a].kind for a in arm_names}
    if "r" in kinds:
        versions["r_earth"] = _r_earth_version()
    if "legacy" in kinds:
        versions.update(_legacy_versions())
    return versions


# ---------------------------------------------------------------------------
# The cache.
# ---------------------------------------------------------------------------


def cache_key(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    arm: learners.Arm,
    versions: dict,
) -> str:
    hasher = hashlib.sha256()
    for arr in (x_train, y_train, x_test):
        hasher.update(np.ascontiguousarray(arr).tobytes())
    hasher.update(arm.name.encode())
    hasher.update(json.dumps(arm.config, sort_keys=True).encode())
    hasher.update(arm.source_text.encode())
    hasher.update(json.dumps(versions, sort_keys=True).encode())
    return hasher.hexdigest()


def _outcome_to_dict(outcome: learners.FitOutcome) -> dict:
    return {
        "predictions": (
            outcome.predictions.tolist() if outcome.predictions is not None else None
        ),
        "n_terms": outcome.n_terms,
        "covariates_used": (
            list(outcome.covariates_used)
            if outcome.covariates_used is not None
            else None
        ),
        "fit_seconds": outcome.fit_seconds,
        "extra": outcome.extra,
        "error": outcome.error,
    }


def _outcome_from_dict(data: dict) -> learners.FitOutcome:
    predictions = data.get("predictions")
    return learners.FitOutcome(
        predictions=np.asarray(predictions, dtype=np.float64)
        if predictions is not None
        else None,
        n_terms=data.get("n_terms"),
        covariates_used=learners.to_covariates_tuple(data.get("covariates_used")),
        fit_seconds=data.get("fit_seconds"),
        extra=data.get("extra") or {},
        error=data.get("error"),
    )


def cache_get(key: str) -> learners.FitOutcome | None:
    path = CACHE_DIR / f"{key}.json"
    if not path.is_file():
        return None
    try:
        return _outcome_from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError, KeyError):
        return None


def cache_put(key: str, outcome: learners.FitOutcome) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(_outcome_to_dict(outcome)), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Atomic result files.
# ---------------------------------------------------------------------------


def result_path(out_dir: Path, cell: dgps.Cell, arm_name: str, rep: int) -> Path:
    return out_dir / cell.name / arm_name / f"rep{rep:04d}.json"


def is_done(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return True


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def compute_measures(
    outcome: learners.FitOutcome,
    dgp: dgps.Dgp,
    truth_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """The per-repetition measures for one fit (VALIDATION_PLAN.md,
    "Performance measures"): the failure count is implicit in ``error``.
    """
    measures: dict = {
        "error": outcome.error,
        "fit_seconds": outcome.fit_seconds,
        "n_terms": outcome.n_terms,
        "covariates_used": (
            list(outcome.covariates_used)
            if outcome.covariates_used is not None
            else None
        ),
        "extra": outcome.extra,
    }
    if outcome.covariates_used is not None:
        measures["n_irrelevant_covariates"] = metrics.irrelevant_covariates_used(
            outcome.covariates_used, dgp.relevant
        )
        measures["uses_irrelevant_covariate"] = metrics.uses_any_irrelevant_covariate(
            outcome.covariates_used, dgp.relevant
        )
    if not outcome.ok:
        return measures
    if dgp.binary:
        log_loss, n_clip_ll = metrics.excess_log_loss(outcome.predictions, truth_test)
        slope, n_clip_cal = metrics.calibration_slope(outcome.predictions, y_test)
        measures.update(
            {
                "excess_log_loss": log_loss,
                "excess_brier_score": metrics.excess_brier_score(
                    outcome.predictions, truth_test
                ),
                "calibration_slope": slope,
                "n_clipped_log_loss": n_clip_ll,
                "n_clipped_calibration": n_clip_cal,
            }
        )
    else:
        measures["excess_risk"] = metrics.excess_risk(outcome.predictions, truth_test)
    return measures


# ---------------------------------------------------------------------------
# Scheduling.
# ---------------------------------------------------------------------------


@dataclass
class Unit:
    cell: dgps.Cell
    arm_name: str
    rep: int
    dgp: dgps.Dgp
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    truth_test: np.ndarray
    y_test: np.ndarray
    cache_key: str


def generate_dataset(cell: dgps.Cell, rep: int, diagnostics: dict):
    dgp = dgps.REGISTRY[cell.dgp]
    train_rng, test_rng = seeds.train_test_rngs(cell.name, rep)
    x_train, y_train, _truth_train = dgps.generate(
        dgp, train_rng, cell.n, cell.noise, diagnostics
    )
    x_test, y_test, truth_test = dgps.generate(
        dgp, test_rng, TEST_N, cell.noise, diagnostics
    )
    return x_train, y_train, x_test, y_test, truth_test


def build_units(
    cells: list[dgps.Cell],
    arm_names: list[str],
    reps: range,
    out_dir: Path,
    diagnostics: dict,
    versions: dict,
    resume: bool,
) -> list[Unit]:
    units: list[Unit] = []
    for cell in cells:
        dgp = dgps.REGISTRY[cell.dgp]
        applicable = [a for a in arm_names if learners.ARMS[a].supports(dgp.binary)]
        if not applicable:
            continue
        for rep in reps:
            pending = [
                a
                for a in applicable
                if not (resume and is_done(result_path(out_dir, cell, a, rep)))
            ]
            if not pending:
                continue
            x_train, y_train, x_test, y_test, truth_test = generate_dataset(
                cell, rep, diagnostics
            )
            for arm_name in pending:
                arm = learners.ARMS[arm_name]
                units.append(
                    Unit(
                        cell=cell,
                        arm_name=arm_name,
                        rep=rep,
                        dgp=dgp,
                        x_train=x_train,
                        y_train=y_train,
                        x_test=x_test,
                        truth_test=truth_test,
                        y_test=y_test,
                        cache_key=cache_key(x_train, y_train, x_test, arm, versions),
                    )
                )
    return units


def _run_inline_unit(unit: Unit) -> tuple[Unit, learners.FitOutcome]:
    arm = learners.ARMS[unit.arm_name]
    try:
        outcome = arm.fit_predict(
            unit.x_train, unit.y_train, unit.x_test, unit.dgp.binary
        )
    except Exception as e:  # an arm's failure must not stop the run
        outcome = learners.FitOutcome(
            predictions=None, error=f"{type(e).__name__}: {e}"
        )
    return unit, outcome


def _run_block_chunk(
    arm_name: str, chunk: list[Unit]
) -> list[tuple[Unit, learners.FitOutcome]]:
    arm = learners.ARMS[arm_name]
    jobs = [
        learners.BlockJob(f"u{i}", u.x_train, u.y_train, u.x_test, u.dgp.binary)
        for i, u in enumerate(chunk)
    ]
    try:
        outcomes = arm.run_block(jobs)
    except Exception as e:  # a block-level crash must not stop the run
        message = f"{type(e).__name__}: {e}"
        outcomes = {
            job.job_id: learners.FitOutcome(predictions=None, error=message)
            for job in jobs
        }
    return [(u, outcomes[f"u{i}"]) for i, u in enumerate(chunk)]


def run_units(
    units: list[Unit], out_dir: Path, n_jobs: int, block_size: int, use_cache: bool
) -> None:
    cached_units, remaining = [], []
    for unit in units:
        cached = cache_get(unit.cache_key) if use_cache else None
        (cached_units if cached is not None else remaining).append((unit, cached))

    tasks = []
    for unit, outcome in cached_units:
        _finish_unit(unit, outcome, out_dir, use_cache=False)  # already cached

    inline_units = [
        u for u, _ in remaining if learners.ARMS[u.arm_name].kind == "inline"
    ]
    block_units_by_arm: dict[str, list[Unit]] = {}
    for u, _ in remaining:
        arm = learners.ARMS[u.arm_name]
        if arm.kind != "inline":
            block_units_by_arm.setdefault(u.arm_name, []).append(u)

    for unit in inline_units:
        tasks.append(joblib.delayed(_run_inline_unit)(unit))
    for arm_name, arm_units in block_units_by_arm.items():
        for i in range(0, len(arm_units), block_size):
            tasks.append(
                joblib.delayed(_run_block_chunk)(
                    arm_name, arm_units[i : i + block_size]
                )
            )

    if not tasks:
        return
    results = joblib.Parallel(n_jobs=n_jobs, backend="loky")(tasks)
    for r in results:
        pairs = r if isinstance(r, list) else [r]
        for unit, outcome in pairs:
            _finish_unit(unit, outcome, out_dir, use_cache=use_cache)


def _finish_unit(
    unit: Unit, outcome: learners.FitOutcome, out_dir: Path, use_cache: bool
) -> None:
    if use_cache and outcome.ok:
        cache_put(unit.cache_key, outcome)
    measures = compute_measures(outcome, unit.dgp, unit.truth_test, unit.y_test)
    atomic_write_json(
        result_path(out_dir, unit.cell, unit.arm_name, unit.rep), measures
    )


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def parse_cells(
    args: argparse.Namespace, all_cells: list[dgps.Cell]
) -> list[dgps.Cell]:
    if args.cells:
        wanted = set(args.cells.split(","))
        selected = [c for c in all_cells if c.name in wanted]
        missing = wanted - {c.name for c in selected}
        if missing:
            raise SystemExit(f"unknown cell name(s): {sorted(missing)}")
        return selected
    selected = all_cells
    if args.dgps:
        wanted_dgps = set(args.dgps.split(","))
        unknown = wanted_dgps - set(dgps.REGISTRY)
        if unknown:
            raise SystemExit(f"unknown DGP name(s): {sorted(unknown)}")
        selected = [c for c in selected if c.dgp in wanted_dgps]
    if args.sizes:
        wanted_sizes = {int(s) for s in args.sizes.split(",")}
        selected = [c for c in selected if c.n in wanted_sizes]
    if args.noise:
        wanted_noise = {
            None if s in ("na", "none") else s for s in args.noise.split(",")
        }
        selected = [c for c in selected if c.noise in wanted_noise]
    return selected


def parse_reps(text: str) -> range:
    start, end = text.split(":")
    return range(int(start), int(end))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--out", required=True, type=Path, help="output folder, under validation/runs/"
    )
    p.add_argument(
        "--arms", required=True, help="comma-separated arm names, e.g. E-def,OLS,HGB"
    )
    p.add_argument(
        "--cells",
        help="comma-separated exact cell names (overrides --dgps/--sizes/--noise)",
    )
    p.add_argument("--dgps", help="comma-separated DGP names, e.g. D1,D2,D3")
    p.add_argument("--sizes", help="comma-separated sample sizes, e.g. 200,1000")
    p.add_argument(
        "--noise",
        help="comma-separated noise levels: lo, hi, and/or na (D6, binary DGPs)",
    )
    p.add_argument(
        "--reps", required=True, help="repetition range start:end (half-open), e.g. 0:2"
    )
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="ignore and do not populate the prediction cache",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.pid").write_text(str(os.getpid()), encoding="utf-8")

    arm_names = args.arms.split(",")
    unknown_arms = set(arm_names) - set(learners.ARMS)
    if unknown_arms:
        raise SystemExit(f"unknown arm name(s): {sorted(unknown_arms)}")

    diagnostics = dgps.load_diagnostics()
    cells = parse_cells(args, dgps.all_cells())
    reps = parse_reps(args.reps)
    versions = gather_versions(arm_names)

    manifest = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arms": {name: learners.ARMS[name].config for name in arm_names},
        "arm_source_hash": {
            name: hashlib.sha256(learners.ARMS[name].source_text.encode()).hexdigest()[
                :16
            ]
            for name in arm_names
        },
        "cells": [c.name for c in cells],
        "reps": [reps.start, reps.stop],
        "versions": versions,
        "test_n": TEST_N,
    }
    atomic_write_json(out_dir / "manifest.json", manifest)

    units = build_units(
        cells, arm_names, reps, out_dir, diagnostics, versions, args.resume
    )
    print(
        f"{len(units)} (cell, arm, repetition) units to run, out of "
        f"{len(cells) * len(arm_names) * len(reps)} requested",
        file=sys.stderr,
    )
    run_units(units, out_dir, args.n_jobs, args.block_size, use_cache=not args.no_cache)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
