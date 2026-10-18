"""Fit time of pymars in a warm process, for validation/REPORT.md (speed claim).

``validation/bench/run.py`` times one fit per fresh process, so its times
include the one-time costs of the first fit in a process: loading numba's
cached machine code from disk and the first calls into numpy and scipy. A
user who fits many models in one session (cross-validation, a simulation, a
super learner) pays those costs once. This script times the first fit and
then the median of ``--reps`` further fits of the same data in the same
process, on the benchmark's data (Friedman #1, ``core.make_data``), at 10
covariates, degree 2 and 21 terms.

    . dev/env.sh
    NUMBA_NUM_THREADS=1 nice -n 15 uv run --frozen --group validation python \
        validation/sims/report/warm_times.py --out validation/sims/report/warm_times.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "validation" / "bench"))

import core  # noqa: E402


def one(n: int, reps: int) -> dict:
    import warnings

    import pymars

    warnings.filterwarnings("ignore")
    X, y, _ = core.make_data(n, 10)

    def fit():
        t0 = time.perf_counter()
        pymars.EarthRegressor(max_degree=2, max_terms=21).fit(X, y)
        return time.perf_counter() - t0

    first = fit()
    later = [fit() for _ in range(reps)]
    return {"n": n, "first_s": first, "warm_median_s": statistics.median(later)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[250, 1000, 10000, 100000])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--child", type=int, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child is not None:
        print(json.dumps(one(a.child, a.reps)))
        return
    commit = subprocess.run(
        ["git", "rev-parse", "--short=10", "HEAD"], cwd=ROOT, capture_output=True,
        text=True, check=True,
    ).stdout.strip()  # fmt: skip
    rows = []
    for n in a.sizes:  # each size in its own fresh process
        out = subprocess.run(
            [sys.executable, __file__, "--child", str(n), "--reps", str(a.reps),
             "--out", str(a.out)],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()[-1]  # fmt: skip
        rows.append({**json.loads(out), "commit": commit})
        print(rows[-1], flush=True)
    with a.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
