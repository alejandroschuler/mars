"""Run the statistical performance study (VALIDATION_PLAN.md, "Build").

Usage (from the repository root, after ``. dev/env.sh``)::

    uv run --frozen python -m validation.sims.run --out validation/runs/smoke \\
        --arms E-def,OLS,HGB --sizes 200 --n-jobs 2 --reps 0:2 --resume

Results land under ``--out``, one JSON file per (cell, arm, repetition):
``<out>/<cell-name>/<arm>/rep<NNNN>.json``. Each unit (one cell, one arm, one
repetition, or one block of several repetitions for an R arm; a legacy arm's
own block is always exactly one repetition, since its interpreter's own
start-up, 0.24 to 0.38 s, is negligible next to one fit, 15 to 130 s, and a
larger block would risk more already-finished work on a kill) generates its
own data, fits, computes its measures and writes its own result file *inside
the worker task that runs it*, atomically (a temp file, then ``os.replace``),
the moment that unit finishes, not after the whole invocation completes. A
run killed partway therefore keeps every unit that had already finished;
``--resume`` skips a result file that already exists
and parses as JSON, and redoes one that does not (an interrupted write) or,
with ``--retry-failed``, one that recorded an error (for example P-fix before
``pymars.EarthRegressor`` existed). ``<out>/manifest.json`` is a JSON list,
one entry per invocation (never overwritten), each with that invocation's
settings and package versions. Long runs are meant to start under
``nohup caffeinate -i nice -n 15 ...``; this script writes its own PID to
``<out>/run.pid`` at startup and removes it on a clean exit, and turns
SIGTERM/SIGINT into a normal exit so joblib shuts its worker processes down
instead of leaving them orphaned.

Predictions are also cached, keyed by a hash of the training and test data,
the arm's name, its fixed settings, its source code (including this module
and, for an R or legacy arm, the block script) and only the package versions
that arm's own kind depends on, in the git-ignored ``validation/runs/.cache/``
(shared across every ``--out``, so a rerun with different filters still
reuses earlier fits). The per-repetition result files under ``--out`` are the
record; the cache only saves time.

No dataset is ever held for more than one unit (or one block of units, for an
R/legacy arm) at a time: each is generated from its name and repetition
inside the task that fits it, not built up front for the whole invocation.
"""

from __future__ import annotations

import os

# One BLAS thread per process (dev/env.sh does the same); set before numpy
# loads, so this holds even if a caller forgets to source it.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from . import dgps, learners, metrics, seeds  # noqa: E402

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


def versions_for_arm(all_versions: dict, arm: learners.Arm) -> dict:
    """Only the versions ``arm``'s own kind depends on. Without this, every
    arm's cache key carried whichever of the R and legacy versions the
    invocation happened to gather for *other* requested arms: OLS got a
    different key depending on whether E-def was also requested, even though
    OLS does not touch R at all (a review found this).
    """
    result = {
        k: all_versions[k]
        for k in ("numpy", "scipy", "sklearn", "pymars", "pymars_commit")
    }
    if arm.kind == "r":
        result["r_earth"] = all_versions.get("r_earth")
    elif arm.kind == "legacy":
        result["legacy_mars_earth"] = all_versions.get("legacy_mars_earth")
        result["legacy_sklearn"] = all_versions.get("legacy_sklearn")
    return result


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


def atomic_write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def is_done(path: Path, retry_failed: bool) -> bool:
    """True when ``path`` holds a result that ``--resume`` should leave
    alone. A recorded failure counts as done unless ``retry_failed``: without
    that flag, a run into the same folder after P-fix's estimators land, or
    after installing a previously-missing R or .venv-legacy, would never
    retry any of the fits that failed for that reason.
    """
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return not (retry_failed and data.get("error") is not None)


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


def _write_measures(
    unit: Unit,
    outcome: learners.FitOutcome,
    dgp: dgps.Dgp,
    truth_test: np.ndarray,
    y_test: np.ndarray,
    out_dir: Path,
) -> None:
    try:
        measures = compute_measures(outcome, dgp, truth_test, y_test)
    except Exception as e:  # e.g. metrics.py rejects a non-finite prediction;
        # never let one bad fit stop the run.
        measures = {
            "error": f"{type(e).__name__}: {e}",
            "fit_seconds": outcome.fit_seconds,
        }
    atomic_write_json(
        result_path(out_dir, unit.cell, unit.arm_name, unit.rep), measures
    )


# ---------------------------------------------------------------------------
# Scheduling. A Unit is only a name: no data is generated, and no cache
# lookup happens, until the task that fits it actually runs (in the worker
# process, one unit or one block of units at a time), which is what keeps
# memory bounded regardless of how many repetitions the invocation covers.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Unit:
    cell: dgps.Cell
    arm_name: str
    rep: int


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
    resume: bool,
    retry_failed: bool,
) -> list[Unit]:
    units: list[Unit] = []
    for cell in cells:
        dgp = dgps.REGISTRY[cell.dgp]
        applicable = [a for a in arm_names if learners.ARMS[a].supports(dgp.binary)]
        for rep in reps:
            for arm_name in applicable:
                path = result_path(out_dir, cell, arm_name, rep)
                if resume and is_done(path, retry_failed):
                    continue
                units.append(Unit(cell, arm_name, rep))
    return units


def _process_inline_unit(
    unit: Unit, diagnostics: dict, out_dir: Path, versions: dict, use_cache: bool
) -> None:
    dgp = dgps.REGISTRY[unit.cell.dgp]
    x_train, y_train, x_test, y_test, truth_test = generate_dataset(
        unit.cell, unit.rep, diagnostics
    )
    arm = learners.ARMS[unit.arm_name]
    key = cache_key(x_train, y_train, x_test, arm, versions_for_arm(versions, arm))
    outcome = cache_get(key) if use_cache else None
    if outcome is None:
        try:
            outcome = arm.fit_predict(x_train, y_train, x_test, dgp.binary)
        except Exception as e:  # an arm's failure must not stop the run
            outcome = learners.FitOutcome(
                predictions=None, error=f"{type(e).__name__}: {e}"
            )
        if use_cache and outcome.ok:
            cache_put(key, outcome)
    _write_measures(unit, outcome, dgp, truth_test, y_test, out_dir)


def _process_block_chunk(
    chunk: list[Unit],
    arm_name: str,
    diagnostics: dict,
    out_dir: Path,
    versions: dict,
    use_cache: bool,
) -> None:
    arm = learners.ARMS[arm_name]
    contexts: dict[Unit, tuple] = {}
    keys: dict[Unit, str] = {}
    cached_outcomes: dict[Unit, learners.FitOutcome] = {}
    jobs: list[tuple[Unit, learners.BlockJob]] = []
    for i, unit in enumerate(chunk):
        dgp = dgps.REGISTRY[unit.cell.dgp]
        x_train, y_train, x_test, y_test, truth_test = generate_dataset(
            unit.cell, unit.rep, diagnostics
        )
        contexts[unit] = (dgp, truth_test, y_test)
        key = cache_key(x_train, y_train, x_test, arm, versions_for_arm(versions, arm))
        keys[unit] = key
        cached = cache_get(key) if use_cache else None
        if cached is not None:
            cached_outcomes[unit] = cached
        else:
            job = learners.BlockJob(f"u{i}", x_train, y_train, x_test, dgp.binary)
            jobs.append((unit, job))

    outcomes_by_job_id: dict[str, learners.FitOutcome] = {}
    if jobs:
        try:
            outcomes_by_job_id = arm.run_block([job for _, job in jobs])
        except Exception as e:  # a block-level crash must not stop the run
            message = f"{type(e).__name__}: {e}"
            outcomes_by_job_id = {
                job.job_id: learners.FitOutcome(predictions=None, error=message)
                for _, job in jobs
            }
    job_id_by_unit = {unit: job.job_id for unit, job in jobs}

    for unit in chunk:
        if unit in cached_outcomes:
            outcome = cached_outcomes[unit]
        else:
            outcome = outcomes_by_job_id[job_id_by_unit[unit]]
            if use_cache and outcome.ok:
                cache_put(keys[unit], outcome)
        dgp, truth_test, y_test = contexts[unit]
        _write_measures(unit, outcome, dgp, truth_test, y_test, out_dir)


def block_size_for(arm: learners.Arm, r_block_size: int) -> int:
    """How many repetitions one block (one subprocess call) covers for
    ``arm``. Always 1 for a legacy arm, regardless of ``r_block_size``: its
    interpreter starts in 0.24 to 0.38 s, against fits of 15 to 130 s (tens
    of minutes for D7), so batching saves almost nothing, while a kill loses
    every already-finished fit in an unfinished block (the block worker
    writes its outputs to a temporary folder that a kill removes unread; a
    review measured 2 of 4 legacy fits lost this way). An earth (R) fit
    takes milliseconds, so batching ``r_block_size`` of them still amortizes
    ``Rscript``'s own start-up meaningfully, and a lost block is cheap to
    redo.
    """
    return 1 if arm.kind == "legacy" else r_block_size


def run_units(
    units: list[Unit],
    diagnostics: dict,
    out_dir: Path,
    n_jobs: int,
    block_size: int,
    versions: dict,
    use_cache: bool,
) -> None:
    inline_units = [u for u in units if learners.ARMS[u.arm_name].kind == "inline"]
    block_units_by_arm: dict[str, list[Unit]] = {}
    for u in units:
        if learners.ARMS[u.arm_name].kind != "inline":
            block_units_by_arm.setdefault(u.arm_name, []).append(u)

    tasks = [
        joblib.delayed(_process_inline_unit)(
            u, diagnostics, out_dir, versions, use_cache
        )
        for u in inline_units
    ]
    for arm_name, arm_units in block_units_by_arm.items():
        size = block_size_for(learners.ARMS[arm_name], block_size)
        for i in range(0, len(arm_units), size):
            chunk = arm_units[i : i + size]
            tasks.append(
                joblib.delayed(_process_block_chunk)(
                    chunk, arm_name, diagnostics, out_dir, versions, use_cache
                )
            )

    if not tasks:
        return
    # inner_max_num_threads=1: each worker's own BLAS stays single-threaded
    # even if the caller did not source dev/env.sh (a review measured 5
    # OpenMP threads per worker at --n-jobs 2 without it).
    with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
        joblib.Parallel(n_jobs=n_jobs)(tasks)


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
    p.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help=(
            "repetitions per Rscript block for E-def/E-pym (default "
            f"{DEFAULT_BLOCK_SIZE}); a legacy arm (P-cur, P-ear, "
            "EarthClassifier, GLMEarth) always uses 1, regardless of this "
            "flag, since its own subprocess start-up is negligible next to "
            "one fit and a larger block risks more finished work on a kill"
        ),
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--retry-failed",
        action="store_true",
        help="with --resume, also redo a unit whose result recorded an error",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="ignore and do not populate the prediction cache",
    )
    return p


def _install_signal_handlers() -> tuple:
    """Installs the handlers and returns the previous ones, so ``main`` can
    put them back. Without this, ``main`` (called directly, not only as
    ``python -m validation.sims.run``) would leave its handler installed in
    the calling process for good; a review found a pytest session kept it
    after test_run.py's own tests finished.
    """
    previous = (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT))

    def _handle(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)
    return previous


def acquire_lock(out_dir: Path) -> Path:
    """``<out_dir>/run.lock``, made with ``mkdir`` (atomic, like the
    executor's own lock in VALIDATION_PLAN.md): stops a second invocation
    from writing into the same folder at once. Raises SystemExit, naming the
    lock, if one is already held; a run.py killed by SIGKILL cannot remove
    its own lock (nothing can run in the process at that point), so the
    message says to check run.pid before removing it by hand.
    """
    lock_path = out_dir / "run.lock"
    try:
        lock_path.mkdir()
    except FileExistsError:
        raise SystemExit(
            f"{lock_path} already exists: another run.py may be using {out_dir}. "
            f"Check {out_dir / 'run.pid'} (or `ps`) for a still-running process; "
            f"if none is running (a SIGKILL cannot clean up its own lock), "
            f"remove {lock_path} and retry."
        ) from None
    return lock_path


def main(argv: list[str] | None = None) -> None:
    previous_handlers = _install_signal_handlers()
    args = build_arg_parser().parse_args(argv)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = acquire_lock(out_dir)
    pid_path = out_dir / "run.pid"
    pid_path.write_text(str(os.getpid()), encoding="utf-8")

    try:
        arm_names = args.arms.split(",")
        unknown_arms = set(arm_names) - set(learners.ARMS)
        if unknown_arms:
            raise SystemExit(f"unknown arm name(s): {sorted(unknown_arms)}")

        diagnostics = dgps.load_diagnostics()
        cells = parse_cells(args, dgps.all_cells())
        reps = parse_reps(args.reps)
        versions = gather_versions(arm_names)

        append_manifest_entry(
            out_dir,
            {
                "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "arms": {name: learners.ARMS[name].config for name in arm_names},
                "arm_source_hash": {
                    name: hashlib.sha256(
                        learners.ARMS[name].source_text.encode()
                    ).hexdigest()[:16]
                    for name in arm_names
                },
                "cells": [c.name for c in cells],
                "reps": [reps.start, reps.stop],
                "versions": versions,
                "test_n": TEST_N,
                "resume": args.resume,
                "retry_failed": args.retry_failed,
            },
        )

        units = build_units(
            cells, arm_names, reps, out_dir, args.resume, args.retry_failed
        )
        print(
            f"{len(units)} (cell, arm, repetition) units to run, out of "
            f"{len(cells) * len(arm_names) * len(reps)} requested",
            file=sys.stderr,
        )
        run_units(
            units,
            diagnostics,
            out_dir,
            args.n_jobs,
            args.block_size,
            versions,
            use_cache=not args.no_cache,
        )
        print("done", file=sys.stderr)
    finally:
        with contextlib.suppress(OSError):
            pid_path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            lock_path.rmdir()
        signal.signal(signal.SIGTERM, previous_handlers[0])
        signal.signal(signal.SIGINT, previous_handlers[1])


def append_manifest_entry(out_dir: Path, entry: dict) -> None:
    """Appends to ``manifest.json`` (a JSON list, one entry per invocation)
    instead of overwriting it, so a folder that saw more than one invocation
    (for example the D7-excepted legacy arms, run separately from the rest)
    keeps every invocation's settings and versions, not just the last one.
    """
    path = out_dir / "manifest.json"
    history = []
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            history = loaded if isinstance(loaded, list) else [loaded]
        except (json.JSONDecodeError, OSError):
            history = []
    history.append(entry)
    atomic_write_json(path, history)


if __name__ == "__main__":
    main()
