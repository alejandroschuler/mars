"""The benchmark driver (VALIDATION_PLAN.md, "Speed and scaling").

Full grid (a detached job; see README.md in this folder):

    . dev/env.sh
    nohup caffeinate -i nice -n 15 uv run --frozen --group validation python \\
        validation/bench/run.py --out validation/runs/bench --resume --jobs 4 \\
        --pid-file validation/runs/bench/run.pid > validation/runs/bench/run.log 2>&1 &

Smoke test (3 fits, under a minute), for gate C:

    uv run --frozen python validation/bench/run.py --smoke

Each cell is one system at one setting of (n, p, degree, term limit,
weights). A cell runs ``--reps`` repetitions, each in a fresh process (one
thread), and keeps the median wall time and the median peak resident memory
of the process (``wait4``'s ru_maxrss, the number that ``/usr/bin/time -l``
prints). For the Python systems one extra run under ``tracemalloc`` gives the
peak Python allocation. A repetition that exceeds the timeout ends the cell
with status "timeout", and the larger settings of the same series are skipped.
Results go to ``<out>/results/<key>.json``, written atomically, and
``<out>/manifest.json`` lists every cell with its status.
"""

# ruff: noqa: E402
from __future__ import annotations

import os
import sys

# One thread everywhere, as dev/env.sh sets; set here too so that a call
# without the sourced file still measures one thread.
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import platform
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "validation" / "harness"))

import core
from names_map import to_earth_args

TRACEMALLOC_MAX_S = 60.0  # no extra tracemalloc run above this median time
SMOKE_BASELINE = HERE / "baseline.json"
SMOKE_TOLERANCE = 1.2


def load_1min() -> float | None:
    """The 1-minute load average, or None where the platform has none."""
    try:
        return os.getloadavg()[0]
    except (AttributeError, OSError):
        return None


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def code_id(system: str) -> str:
    """What the cache key says about the code under test."""
    helper = HERE / ("earth_worker.R" if system.startswith("earth") else "worker.py")
    h = hashlib.sha256(helper.read_bytes())
    if system.startswith("pymars"):
        for f in sorted((ROOT / "pymars").glob("*.py")):
            h.update(f.read_bytes())
        return f"{git('rev-parse', 'HEAD')}:{h.hexdigest()[:12]}"
    if system == "legacy":
        return f"legacy-1.0.4:{h.hexdigest()[:12]}"
    return f"earth:{h.hexdigest()[:12]}"


@dataclasses.dataclass
class Measured:
    returncode: int | None  # None: timeout
    stdout: str
    stderr: str
    seconds: float
    maxrss_bytes: int


def run_measured(cmd: list[str], timeout: float) -> Measured:
    """Run cmd to completion or to the timeout; return its output, wall time
    and peak resident set size (a child's ru_maxrss: bytes on macOS, KiB on
    Linux)."""
    with tempfile.TemporaryFile("w+") as so, tempfile.TemporaryFile("w+") as se:
        t0 = time.monotonic()
        proc = subprocess.Popen(cmd, stdout=so, stderr=se)
        while True:
            pid, status, ru = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                break
            if time.monotonic() - t0 > timeout:
                proc.kill()
                _, _, ru = os.wait4(proc.pid, 0)
                status = None
                break
            time.sleep(0.1)
        elapsed = time.monotonic() - t0
        so.seek(0)
        se.seek(0)
        scale = 1 if sys.platform == "darwin" else 1024
        return Measured(
            None if status is None else os.waitstatus_to_exitcode(status),
            so.read(),
            se.read()[-2000:],
            elapsed,
            int(ru.ru_maxrss * scale),
        )


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.out = Path(args.out)
        self.results = self.out / "results"
        self.args = args
        self.lock = threading.Lock()
        self.data_lock = threading.Lock()
        self.codes = {s: code_id(s) for s in core.SYSTEMS}
        self.status: dict[str, str] = {}
        self.cells: dict[str, core.Cell] = {}

    # data ---------------------------------------------------------------
    def data_files(self, n: int, p: int) -> tuple[Path, Path]:
        d = self.out / "data"
        npz, csv = d / f"n{n}_p{p}.npz", d / f"n{n}_p{p}.csv"
        with self.data_lock:
            if not (npz.is_file() and csv.is_file()):
                d.mkdir(parents=True, exist_ok=True)
                X, y, w = core.make_data(n, p)
                tmp_npz = d / f"n{n}_p{p}.tmp.npz"
                np.savez(tmp_npz, X=X, y=y, w=w)
                tmp_csv = d / f"n{n}_p{p}.tmp.csv"
                header = ",".join([f"x{j}" for j in range(p)] + ["y", "w"])
                np.savetxt(
                    tmp_csv,
                    np.column_stack([X, y, w]),
                    delimiter=",",
                    header=header,
                    comments="",
                    fmt="%.17g",
                )
                os.replace(tmp_csv, csv)
                os.replace(tmp_npz, npz)
        return npz, csv

    # commands -----------------------------------------------------------
    def command(
        self, cell: core.Cell, mode: str, prof: Path | None = None
    ) -> list[str]:
        npz, csv = self.data_files(cell.n, cell.p)
        if cell.system.startswith("earth"):
            earth = {"max_degree": cell.degree, "max_terms": cell.max_terms}
            if cell.system == "earth_fast0":
                earth["fast_k"] = 0
            cfg = to_earth_args(**earth)
            if cell.weights:
                cfg["use_weights"] = True
            return [
                "Rscript",
                "--vanilla",
                str(HERE / "earth_worker.R"),
                str(csv),
                json.dumps(cfg),
            ]
        py = [sys.executable]
        if cell.system == "legacy":
            py = [str(self.args.legacy_python), "-I"]
        cmd = [
            *py,
            str(HERE / "worker.py"),
            "--system", cell.system,
            "--data", str(npz),
            "--degree", str(cell.degree),
            "--max-terms", str(cell.max_terms),
            "--mode", mode,
        ]  # fmt: skip
        if cell.weights:
            cmd.append("--weights")
        if prof is not None:
            cmd += ["--profile-out", str(prof)]
        return cmd

    # one cell -----------------------------------------------------------
    def run_cell(
        self, cell: core.Cell, skip_reason: str | None = None
    ) -> dict[str, Any]:
        a = self.args
        key = core.cell_key(cell, self.codes[cell.system], a.reps, a.timeout)
        old = core.read_result(self.results, key)
        if a.resume and core.is_finished(old):
            return self.note(cell, key, old)
        if skip_reason:
            return self.save(
                cell, key, core.make_record(cell, key, "skipped", reason=skip_reason)
            )
        started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        load = load_1min()
        times: list[float] = []
        rss: list[float] = []
        info: dict[str, Any] = {}
        for _ in range(a.reps):
            m = run_measured(self.command(cell, "time"), a.timeout)
            if m.returncode is None:
                rec = core.make_record(
                    cell,
                    key,
                    "timeout",
                    timeout_s=a.timeout,
                    times=times,
                    started=started,
                )
                return self.save(cell, key, rec)
            try:
                if m.returncode != 0:
                    raise RuntimeError(f"exit {m.returncode}")
                info = json.loads(m.stdout.strip().splitlines()[-1])
            except (RuntimeError, ValueError, IndexError) as e:
                rec = core.make_record(
                    cell, key, "error", error=f"{e}: {m.stderr}", started=started
                )
                return self.save(cell, key, rec)
            times.append(float(info["seconds"]))
            rss.append(m.maxrss_bytes)
        rec = core.make_record(
            cell,
            key,
            "ok",
            terms=info["terms"],
            forward_terms=info["forward_terms"],
            code=self.codes[cell.system],
            started=started,
            load_1min=load,
            **core.summarize_reps(times, rss),
        )
        if not cell.system.startswith("earth"):
            if rec["time_median"] <= TRACEMALLOC_MAX_S:
                m = run_measured(self.command(cell, "tracemalloc"), a.timeout)
                if m.returncode == 0:
                    rec["py_peak_bytes"] = json.loads(
                        m.stdout.strip().splitlines()[-1]
                    )["py_peak_bytes"]
            if a.profile:
                prof = self.out / "profiles" / f"{key}.prof"
                prof.parent.mkdir(parents=True, exist_ok=True)
                m = run_measured(self.command(cell, "profile", prof), a.timeout)
                if m.returncode == 0:
                    rec["profile"] = str(prof.relative_to(self.out))
        return self.save(cell, key, rec)

    def note(self, cell: core.Cell, key: str, rec: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.status[key] = rec["status"]
            self.cells[key] = cell
            self.write_manifest()
        return rec

    def save(self, cell: core.Cell, key: str, rec: dict[str, Any]) -> dict[str, Any]:
        core.write_json_atomic(self.results / f"{key}.json", rec)
        took = f" median {rec['time_median']:.3f} s" if rec["status"] == "ok" else ""
        stamp = time.strftime("%H:%M:%S")
        print(f"{stamp} {rec['status']:8s} {cell.label()}{took}", flush=True)
        return self.note(cell, key, rec)

    def write_manifest(self) -> None:
        core.write_json_atomic(
            self.out / "manifest.json",
            {
                "planned": self.planned,
                "counts": {
                    s: sum(v == s for v in self.status.values())
                    for s in sorted(set(self.status.values()))
                },
                "cells": {
                    k: {"label": c.label(), "status": self.status[k]}
                    for k, c in self.cells.items()
                },
            },
        )

    # one series ---------------------------------------------------------
    def run_series(self, cells: list[core.Cell]) -> None:
        """Cells in increasing cost: after a timeout, the rest are skipped."""
        timed_out: core.Cell | None = None
        for cell in cells:
            reason = f"{timed_out.label()} timed out" if timed_out else None
            rec = self.run_cell(cell, reason)
            if rec["status"] == "timeout":
                timed_out = cell
            elif rec["status"] == "skipped":
                timed_out = timed_out or cell

    def run_all(self) -> None:
        a = self.args
        systems = tuple(a.systems)
        factors = set(a.factors)
        series = {
            k: v for k, v in core.build_series(systems).items() if k[1] in factors
        }
        cells = core.all_cells(series)
        self.planned = len(cells)
        self.out.mkdir(parents=True, exist_ok=True)
        core.write_json_atomic(self.out / "meta.json", self.meta())
        # Phase 0: the baseline cells, which every series shares.
        base = [
            c for c in cells if all(getattr(c, k) == v for k, v in core.BASE.items())
        ]
        with concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:
            list(pool.map(self.run_cell, base))
            futs = [pool.submit(self.run_series, cs) for cs in series.values()]
            for f in futs:
                f.result()

    def meta(self) -> dict[str, Any]:
        def r(code: str) -> str:
            p = subprocess.run(
                ["Rscript", "--vanilla", "-e", code], capture_output=True, text=True
            )
            return p.stdout.strip()

        brand = ""
        if sys.platform == "darwin":
            brand = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            ).stdout.strip()
        return {
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "commit": git("rev-parse", "HEAD"),
            "machine": platform.platform(),
            "cpu": brand,
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "R": r("cat(R.version.string)"),
            "earth": r("cat(as.character(packageVersion('earth')))"),
            "legacy": "mars-earth 1.0.4 wheel (validation/legacy/make_venv.sh)",
            "jobs": self.args.jobs,
            "reps": self.args.reps,
            "timeout_s": self.args.timeout,
            "load_1min_at_start": load_1min(),
            "threads": {
                v: os.environ.get(v)
                for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
        }


# smoke --------------------------------------------------------------------
SMOKE_FITS = [  # (n, p, degree, max_terms, weights)
    (2000, 10, 1, 21, False),
    (1000, 10, 2, 21, False),
    (1000, 10, 2, 21, True),
]


def calibration() -> float:
    """Seconds of a fixed numpy workload, to scale the machine out of the
    comparison with the stored baseline (best of 5)."""
    rng = np.random.default_rng(0)
    A, b = rng.standard_normal((3000, 60)), rng.standard_normal(3000)
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(20):
            np.linalg.lstsq(A, b, rcond=None)
        best = min(best, time.perf_counter() - t0)
    return best


def smoke(update: bool, tolerance: float) -> int:
    import warnings

    warnings.filterwarnings("ignore")
    from pymars._core import MarsParams, fit_mars

    calib = calibration()
    total = 0.0
    parts = []
    for n, p, degree, nk, weights in SMOKE_FITS:
        X, y, w = core.make_data(n, p)
        params = MarsParams(max_degree=degree, max_terms=nk)
        w_arg = w if weights else None
        fit_mars(X, y, w_arg, params)  # warm-up: imports and caches
        best = float("inf")
        for _ in range(5):
            t0 = time.perf_counter()
            fit_mars(X, y, w_arg, params)
            best = min(best, time.perf_counter() - t0)
        parts.append(
            {
                "n": n,
                "p": p,
                "degree": degree,
                "max_terms": nk,
                "weights": weights,
                "seconds": best,
            }
        )
        total += best
    ratio = total / calib
    print(
        f"smoke: {total:.3f} s for 3 fits, calibration {calib:.4f} s, ratio {ratio:.3f}"
    )
    for part in parts:
        print("  ", part)
    if update:
        core.write_json_atomic(
            SMOKE_BASELINE,
            {
                "commit": git("rev-parse", "HEAD"),
                "ratio": ratio,
                "total_s": total,
                "calibration_s": calib,
                "machine": platform.platform(),
                "fits": parts,
            },
        )
        print(f"baseline written to {SMOKE_BASELINE}")
        return 0
    if not SMOKE_BASELINE.is_file():
        print(
            "smoke: no baseline.json; run with --update-baseline on main",
            file=sys.stderr,
        )
        return 1
    base = json.loads(SMOKE_BASELINE.read_text())
    rel = ratio / base["ratio"]
    verdict = "PASS" if rel <= tolerance else "FAIL"
    print(
        f"smoke {verdict}: {rel:.3f} times the baseline (limit {tolerance}; "
        f"baseline commit {base['commit'][:10]})"
    )
    return 0 if rel <= tolerance else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default=str(ROOT / "validation" / "runs" / "bench"))
    ap.add_argument(
        "--resume", action="store_true", help="skip cells that are finished"
    )
    ap.add_argument(
        "--jobs", type=int, default=4, help="parallel processes (R counts as one)"
    )
    ap.add_argument(
        "--systems", nargs="+", default=list(core.SYSTEMS), choices=core.SYSTEMS
    )
    ap.add_argument(
        "--factors", nargs="+", default=list(core.FACTORS), choices=list(core.FACTORS)
    )
    ap.add_argument("--reps", type=int, default=core.REPS)
    ap.add_argument(
        "--timeout", type=float, default=core.TIMEOUT_S, help="seconds per fit"
    )
    ap.add_argument(
        "--profile", action="store_true", help="write cProfile stats per cell"
    )
    ap.add_argument(
        "--legacy-python",
        default=os.environ.get(
            "PYMARS_LEGACY_PYTHON", str(ROOT / ".venv-legacy" / "bin" / "python")
        ),
    )
    ap.add_argument("--pid-file")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--update-baseline",
        action="store_true",
        help="with --smoke: store this run as the baseline",
    )
    ap.add_argument("--tolerance", type=float, default=SMOKE_TOLERANCE)
    args = ap.parse_args()

    if args.smoke:
        code = smoke(args.update_baseline, args.tolerance)
        if code != 0 and not args.update_baseline:
            print("smoke: failed once, running it again")
            code = smoke(False, args.tolerance)
        return code
    if args.jobs > 4:
        ap.error("at most 4 jobs (the plan's limit for one agent)")
    if "legacy" in args.systems and not Path(args.legacy_python).is_file():
        ap.error(
            f"{args.legacy_python} does not exist; run "
            "validation/legacy/make_venv.sh or pass --legacy-python"
        )
    if args.pid_file:
        Path(args.pid_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.pid_file).write_text(f"{os.getpid()}\n")
    runner = Runner(args)
    runner.run_all()
    print(
        "done",
        json.dumps(
            runner.status
            and {
                s: list(runner.status.values()).count(s)
                for s in set(runner.status.values())
            }
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
