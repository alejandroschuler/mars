"""Skewed covariates: PR #185's pair rule on lognormal data (issue #173).

Started from a runner worktree (see ``results/skew/``), which is made with
``git worktree add`` at the commit that the arm needs, since each of its fits
must come from one fixed version of pymars (``--expect-sha`` checks it)::

    . dev/env.sh
    .venv/bin/python -m validation.sims.skew_sims --out validation/runs/skew \\
        --arms P-fix --dgps D1,D2 --sizes 200 --noise lo --reps 0:1000 \\
        --n-jobs 4 --expect-sha <40-character sha>

The covariates the learners see are Z_j = exp(Phi^{-1}(U_j)), lognormal
margins, with U the grid's own covariates (``dgps.generate``, the same random
draws in the same order, so D8 keeps its copula). The truth stays f(U), and
the noise level is the grid's: sigma from ``diagnostics.json``. The cell name
is the grid's name with ``skew`` after the DGP (``D4skew_n00200_lo``), so the
seeds (``seeds.py``: a hash of the cell name and the repetition) differ from
the study's, and are the same in every runner worktree: two runs of one
(cell, repetition) see identical data, which each result file proves with
``data_sha256``.

The arms are the harness's own (``learners.ARMS``): P-fix (pymars, degree 2)
and E-def (earth at its defaults, degree 2). Nothing is cached. Results are
one atomic JSON file per (cell, arm, repetition) under
``<out>/<cell>/<arm>/rep<NNNN>.json``, written inside the worker that fits it;
``--resume`` skips a result that already parses, so a restart loses at most
the fits that were in flight.
"""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from scipy import special  # noqa: E402

import pymars  # noqa: E402
from validation.sims import dgps, learners, run, seeds  # noqa: E402

SKEW = "skew"
TEST_N = run.TEST_N
SCRIPT = Path(__file__).resolve()


def cell_for(base: str, n: int, noise: str) -> dgps.Cell:
    """The new cell: the grid's DGP name plus ``skew``, same n and noise."""
    return dgps.Cell(base + SKEW, n, noise)


def lognormal_covariates(u: np.ndarray) -> np.ndarray:
    """Z = exp(Phi^{-1}(U)): lognormal margins, monotone in U."""
    z = np.exp(special.ndtri(u))
    if not np.isfinite(z).all() or (z <= 0).any():
        raise ValueError("a covariate is 0 or 1 on the uniform scale")
    return z


def generate_dataset(base: str, n: int, noise: str, rep: int, diagnostics: dict):
    """(x_train, y_train, x_test, y_test, truth_test, sha) for one repetition.
    ``dgps.generate`` draws U, y and the truth f(U) exactly as the grid does;
    only the covariates handed to the learners change.
    """
    dgp = dgps.REGISTRY[base]
    cell = cell_for(base, n, noise)
    train_rng, test_rng = seeds.train_test_rngs(cell.name, rep)
    u_train, y_train, _ = dgps.generate(dgp, train_rng, n, noise, diagnostics)
    u_test, y_test, truth_test = dgps.generate(
        dgp, test_rng, TEST_N, noise, diagnostics
    )
    x_train = lognormal_covariates(u_train)
    x_test = lognormal_covariates(u_test)
    hasher = hashlib.sha256()
    for arr in (x_train, y_train, x_test, truth_test):
        hasher.update(np.ascontiguousarray(arr).tobytes())
    return x_train, y_train, x_test, y_test, truth_test, hasher.hexdigest()


def _write(
    base, n, noise, arm_name, rep, outcome, truth_test, y_test, sha, out_dir
) -> None:
    dgp = dgps.REGISTRY[base]
    cell = cell_for(base, n, noise)
    try:
        measures = run.compute_measures(outcome, dgp, truth_test, y_test)
    except Exception as e:  # one bad fit never stops the run
        measures = {
            "error": f"{type(e).__name__}: {e}",
            "fit_seconds": outcome.fit_seconds,
        }
    measures["data_sha256"] = sha
    measures["cell"] = cell.name
    run.atomic_write_json(run.result_path(out_dir, cell, arm_name, rep), measures)


def process_inline_chunk(base, n, noise, arm_name, reps, diagnostics, out_dir) -> None:
    arm = learners.ARMS[arm_name]
    for rep in reps:
        x_train, y_train, x_test, y_test, truth_test, sha = generate_dataset(
            base, n, noise, rep, diagnostics
        )
        try:
            outcome = arm.fit_predict(x_train, y_train, x_test, False)
        except Exception as e:
            outcome = learners.FitOutcome(
                predictions=None, error=f"{type(e).__name__}: {e}"
            )
        _write(base, n, noise, arm_name, rep, outcome, truth_test, y_test, sha, out_dir)


def process_block_chunk(base, n, noise, arm_name, reps, diagnostics, out_dir) -> None:
    arm = learners.ARMS[arm_name]
    contexts, jobs = {}, []
    for rep in reps:
        x_train, y_train, x_test, y_test, truth_test, sha = generate_dataset(
            base, n, noise, rep, diagnostics
        )
        contexts[rep] = (truth_test, y_test, sha)
        jobs.append(learners.BlockJob(f"r{rep}", x_train, y_train, x_test, False))
    try:
        outcomes = arm.run_block(jobs)
    except Exception as e:
        message = f"{type(e).__name__}: {e}"
        outcomes = {
            j.job_id: learners.FitOutcome(predictions=None, error=message) for j in jobs
        }
    for rep in reps:
        truth_test, y_test, sha = contexts[rep]
        _write(
            base, n, noise, arm_name, rep, outcomes[f"r{rep}"],
            truth_test, y_test, sha, out_dir,
        )  # fmt: skip


def build_tasks(
    cells, arm_names, reps, out_dir, resume, chunk, block_size, diagnostics
) -> list:
    tasks = []
    for base, n, noise in cells:
        cell = cell_for(base, n, noise)
        for arm_name in arm_names:
            arm = learners.ARMS[arm_name]
            todo = [
                r
                for r in reps
                if not (
                    resume
                    and run.is_done(run.result_path(out_dir, cell, arm_name, r), False)
                )
            ]
            size = block_size if arm.kind == "r" else chunk
            fn = process_block_chunk if arm.kind == "r" else process_inline_chunk
            for i in range(0, len(todo), size):
                tasks.append(
                    joblib.delayed(fn)(
                        base,
                        n,
                        noise,
                        arm_name,
                        todo[i : i + size],
                        diagnostics,
                        out_dir,
                    )
                )
    return tasks


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--arms", required=True, help="P-fix and/or E-def")
    p.add_argument("--dgps", required=True, help="base DGP names, e.g. D1,D2")
    p.add_argument("--sizes", required=True)
    p.add_argument("--noise", required=True, help="lo and/or hi")
    p.add_argument("--reps", required=True, help="half-open, e.g. 0:1000")
    p.add_argument("--n-jobs", type=int, required=True)
    p.add_argument("--block-size", type=int, default=20, help="E-def reps per Rscript")
    p.add_argument("--chunk", type=int, default=20, help="P-fix reps per task")
    p.add_argument("--expect-sha", required=True, help="the commit this tree must be")
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args(argv)

    root = learners.REPO_ROOT
    head = run._pymars_commit()
    if head != args.expect_sha:
        raise SystemExit(f"{root} is at {head}, expected {args.expect_sha}")
    if root not in Path(pymars.__file__).resolve().parents:
        raise SystemExit(f"pymars imported from {pymars.__file__}, not from {root}")
    arm_names = args.arms.split(",")
    if not set(arm_names) <= {"P-fix", "E-def"}:
        raise SystemExit(f"arms must be P-fix and/or E-def, got {arm_names}")
    bases = args.dgps.split(",")
    if "D6" in bases or not set(bases) <= set(dgps.REGRESSION_DGPS):
        raise SystemExit(f"regression DGPs with signal only, got {bases}")
    sizes = [int(s) for s in args.sizes.split(",")]
    noises = args.noise.split(",")
    cells = [(b, n, z) for b in bases for n in sizes for z in noises]
    start, stop = (int(s) for s in args.reps.split(":"))
    reps = list(range(start, stop))

    args.out.mkdir(parents=True, exist_ok=True)
    previous = run._install_signal_handlers()
    lock = run.acquire_lock(args.out)
    pid_path = args.out / "run.pid"
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    try:
        diagnostics = dgps.load_diagnostics()
        versions = run.gather_versions(arm_names)
        versions["pymars_file"] = pymars.__file__
        versions["python"] = sys.version.split()[0]
        versions["script_sha256"] = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
        run.append_manifest_entry(
            args.out,
            {
                "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "arms": {a: learners.ARMS[a].config for a in arm_names},
                "cells": [cell_for(*c).name for c in cells],
                "reps": [start, stop],
                "n_jobs": args.n_jobs,
                "test_n": TEST_N,
                "resume": not args.no_resume,
                "cache": "none",
                "versions": versions,
            },
        )
        tasks = build_tasks(
            cells, arm_names, reps, args.out, not args.no_resume,
            args.chunk, args.block_size, diagnostics,
        )  # fmt: skip
        print(
            f"{len(tasks)} tasks for {len(cells)} cells, reps {start}:{stop}",
            file=sys.stderr,
        )
        if tasks:
            with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
                joblib.Parallel(n_jobs=args.n_jobs)(tasks)
        print("done", file=sys.stderr)
    finally:
        with contextlib.suppress(OSError):
            pid_path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            lock.rmdir()
        import signal

        signal.signal(signal.SIGTERM, previous[0])
        signal.signal(signal.SIGINT, previous[1])


if __name__ == "__main__":
    main()
