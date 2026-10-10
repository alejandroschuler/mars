"""The logic of the benchmark that needs no subprocess: the grid, the cell
keys, atomic result files, the resume rule and the log-log slopes
(VALIDATION_PLAN.md, "Benchmark design" and "Long computations")."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

SYSTEMS = ("pymars_fit", "pymars_est", "earth_default", "earth_fast0", "legacy")
LEGACY_MAX_N = 2000
TIMEOUT_S = 1800.0
REPS = 3
# The baseline of the plan: 1,000 cases, 10 covariates, degree 2, 21 terms.
BASE = {"n": 1000, "p": 10, "degree": 2, "max_terms": 21, "weights": False}
# One factor at a time; each list is in increasing cost, so that a timeout at
# one value skips the larger ones of the same series.
FACTORS: dict[str, list[Any]] = {
    "n": [250, 500, 1000, 2000, 5000, 10000, 100000],
    "p": [5, 10, 20, 50, 100],
    "degree": [1, 2, 3],
    "max_terms": [11, 21, 41, 81],
    "weights": [False, True],
}
DATA_SEED = 19


@dataclasses.dataclass(frozen=True)
class Cell:
    system: str
    n: int = BASE["n"]
    p: int = BASE["p"]
    degree: int = BASE["degree"]
    max_terms: int = BASE["max_terms"]
    weights: bool = BASE["weights"]

    def label(self) -> str:
        w = ",w" if self.weights else ""
        return (
            f"{self.system} n={self.n} p={self.p} deg={self.degree} "
            f"nk={self.max_terms}{w}"
        )


def build_series(
    systems: tuple[str, ...] = SYSTEMS,
) -> dict[tuple[str, str], list[Cell]]:
    """The cells of each (system, factor) series, in increasing cost. The
    baseline cell belongs to every series of its system. The legacy code runs
    up to 2,000 cases and takes no weights."""
    series: dict[tuple[str, str], list[Cell]] = {}
    for system in systems:
        for factor, values in FACTORS.items():
            cells = []
            for v in values:
                cell = Cell(system, **{**BASE, factor: v})
                if system == "legacy" and (cell.n > LEGACY_MAX_N or cell.weights):
                    continue
                cells.append(cell)
            if len(cells) > 1:
                series[(system, factor)] = cells
    return series


def all_cells(series: dict[tuple[str, str], list[Cell]]) -> list[Cell]:
    """Each distinct cell once, in the order of first appearance."""
    seen: dict[Cell, None] = {}
    for cells in series.values():
        for c in cells:
            seen.setdefault(c)
    return list(seen)


def data_seed(n: int, p: int) -> list[int]:
    """The seed sequence entropy of the data of n cases and p covariates: the
    same data for every system and every factor value that shares (n, p)."""
    return [DATA_SEED, n, p]


def make_data(n: int, p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Friedman #1: uniform covariates (the first 5 matter, the rest are
    noise), standard normal error; w holds integer case counts 1 to 3."""
    ss = np.random.SeedSequence(data_seed(n, p))
    rng_x, rng_w = (np.random.default_rng(s) for s in ss.spawn(2))
    X = rng_x.uniform(0.0, 1.0, (n, p))
    f = (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + 5 * X[:, 4]
    )
    y = f + rng_x.standard_normal(n)
    w = rng_w.integers(1, 4, n).astype(float)
    return X, y, w


def cell_key(
    cell: Cell, code_id: str, reps: int = REPS, timeout: float = TIMEOUT_S
) -> str:
    """The cache key: the cell, the data seed, the repetition rule and the
    code (the pymars commit and source hash, or the legacy and earth tags)."""
    blob = json.dumps(
        [dataclasses.asdict(cell), data_seed(cell.n, cell.p), reps, timeout, code_id],
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def write_json_atomic(path: Path, obj: Any) -> None:
    """Write a temporary file in the same folder, then rename it over path. A
    crash before the rename leaves a *.tmp file and the old content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_result(results_dir: Path, key: str) -> dict[str, Any] | None:
    path = results_dir / f"{key}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


# A finished cell is not run again by --resume. An error or a skip is decided
# again, because a fix or a finished predecessor can change it.
FINAL_STATUS = ("ok", "timeout")


def is_finished(record: dict[str, Any] | None) -> bool:
    return record is not None and record.get("status") in FINAL_STATUS


def make_record(cell: Cell, key: str, status: str, **extra: Any) -> dict[str, Any]:
    return {"key": key, "cell": dataclasses.asdict(cell), "status": status, **extra}


def summarize_reps(times: list[float], rss: list[float]) -> dict[str, Any]:
    """The median of the repetitions (the plan reports the median of 3)."""
    return {
        "times": times,
        "time_median": statistics.median(times),
        "rss_median_bytes": statistics.median(rss),
    }


def loglog_slope(xs: list[float], ys: list[float]) -> tuple[float, int] | None:
    """The least-squares slope of log y on log x over the points with positive
    x and y, and the number of points; None with fewer than 2 points or with
    all x equal."""
    pts = [
        (math.log(x), math.log(y))
        for x, y in zip(xs, ys, strict=True)
        if x > 0 and y > 0
    ]
    if len(pts) < 2:
        return None
    lx, ly = (np.array(v) for v in zip(*pts, strict=True))
    if np.ptp(lx) == 0:
        return None
    return float(np.polyfit(lx, ly, 1)[0]), len(pts)
