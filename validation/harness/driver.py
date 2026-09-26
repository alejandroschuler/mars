"""Python driver for the earth harness: writes datasets as CSV, runs R earth
through ``Rscript fit_earth.R``, and reads the results back as plain dicts.

This module never reads earth's C or R source. It writes inputs, calls
``fit_earth.R`` (which calls ``earth()``), and reads what comes back, which is
the black-box use ``VALIDATION_PLAN.md`` ("Instruction files and clean room")
allows.

Result schema
-------------
``run_earth`` returns one dict per job, parsed straight from ``fit_earth.R``'s
JSON (see that script's docstring for the exact fields: ``dirs``, ``cuts``,
``term_names``, ``selected_terms``, ``prune_terms``, ``rss_per_subset``,
``gcv_per_subset``, ``coef``, ``glm_coef``, ``rss``, ``rsq``, ``gcv``,
``grsq``, ``termcond``, ``levels``, ``fitted``, ``pred_train``, ``pred_test``,
``fwd_rss``, ``r_version``, ``earth_version``). ``dirs``, ``cuts``, ``coef``,
``glm_coef``, ``fitted``, ``pred_train``, ``pred_test`` and ``prune_terms``
are always nested (2-D) lists, never collapsed to a bare number or a flat
list when one dimension is 1; ``compare.py``, ``legacy_adapter.py`` and
``new_adapter.py`` share this schema (a later task adds the last two).
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
FIT_EARTH_R = HERE / "fit_earth.R"

# fit_earth.R writes Inf/-Inf/NaN/NA as these exact strings (its write_result
# comments explain why); undo that after json.loads() so a caller sees plain
# floats, matching what a finite value already looks like.
_R_JSON_SENTINELS = {
    "Inf": float("inf"),
    "-Inf": float("-inf"),
    "NaN": float("nan"),
    "NA": float("nan"),
}


def _desanitize(value: Any) -> Any:
    """Recursively convert fit_earth.R's Inf/-Inf/NaN/NA sentinel strings
    back to floats; every other string (an id, a term name, a level) passes
    through unchanged."""
    if isinstance(value, str):
        return _R_JSON_SENTINELS.get(value, value)
    if isinstance(value, list):
        return [_desanitize(v) for v in value]
    if isinstance(value, dict):
        return {k: _desanitize(v) for k, v in value.items()}
    return value


@dataclass
class EarthJob:
    """One dataset and earth call for ``run_earth``.

    ``earth_args`` uses earth's own argument names (for example ``"nk"``,
    ``"Auto.linpreds"``), so this module needs no knowledge of the pymars
    names ``names_map.py`` maps them from.
    """

    id: str
    X: np.ndarray
    y: np.ndarray
    earth_args: Mapping[str, Any] = field(default_factory=dict)
    weights: np.ndarray | None = None
    X_test: np.ndarray | None = None
    factor_response: bool = False
    glm_family: str | None = None
    trace: int = 0
    include_forward_path: bool = True


def write_csv(path: Path, columns: Mapping[str, np.ndarray]) -> None:
    """Write named equal-length columns as CSV.

    Numeric columns are written at 17 significant digits (``%.17g``), so
    ``read.csv`` on the R side loses no precision from the original double;
    a factor-response column of string labels is written as plain text.
    """
    names = list(columns)
    arrays = [np.asarray(columns[name]) for name in names]
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("all columns must have the same length")
    lines = [",".join(names)]
    for i in range(n):
        cells = [
            f"{a[i]:.17g}" if np.issubdtype(a.dtype, np.number) else str(a[i])
            for a in arrays
        ]
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _y_columns(y: np.ndarray, factor_response: bool) -> dict[str, np.ndarray]:
    """Split ``y`` into the named columns ``fit_earth.R`` expects.

    A factor response is one column named ``y``. Otherwise ``y`` is 1-D (one
    column ``y``) or 2-D (columns ``y1`` .. ``yK``, several responses that
    share one basis).
    """
    if factor_response:
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("a factor response must be a 1-D array of labels")
        return {"y": y}
    y = np.asarray(y)
    if y.ndim == 1:
        return {"y": y}
    if y.ndim == 2:
        if y.shape[1] == 1:
            return {"y": y[:, 0]}
        return {f"y{k + 1}": y[:, k] for k in range(y.shape[1])}
    raise ValueError(f"y must be 1-D or 2-D, got shape {y.shape}")


def _x_columns(X: np.ndarray) -> dict[str, np.ndarray]:
    X = np.atleast_2d(np.asarray(X))
    return {f"x{j}": X[:, j] for j in range(X.shape[1])}


def _write_job_data(job: EarthJob, workdir: Path) -> dict[str, Any]:
    """Write one job's CSVs and return its R config entry."""
    train_csv = workdir / f"{job.id}_train.csv"
    columns = _x_columns(job.X)
    y_cols_map = _y_columns(job.y, job.factor_response)
    y_cols = list(y_cols_map)  # insertion order: "y", or "y1", "y2", ...
    columns.update(y_cols_map)
    if job.weights is not None:
        columns["w"] = np.asarray(job.weights)
    write_csv(train_csv, columns)

    test_csv = None
    if job.X_test is not None:
        test_csv = workdir / f"{job.id}_test.csv"
        write_csv(test_csv, _x_columns(job.X_test))

    trace_file = workdir / f"{job.id}_trace.txt" if job.trace > 0 else None
    return {
        "id": job.id,
        "train_csv": str(train_csv),
        "test_csv": str(test_csv) if test_csv is not None else None,
        "y_cols": y_cols,
        "weight_col": "w" if job.weights is not None else None,
        "factor_response": job.factor_response,
        "earth_args": dict(job.earth_args),
        "glm_family": job.glm_family,
        "trace": job.trace,
        "trace_file": str(trace_file) if trace_file is not None else None,
        "include_forward_path": job.include_forward_path,
        "out": str(workdir / f"{job.id}_result.json"),
    }


def run_earth(
    jobs: Sequence[EarthJob], *, workdir: Path, rscript: str = "Rscript"
) -> dict[str, dict[str, Any]]:
    """Fit every job in ``jobs`` with one R process (a block), and return
    ``{job.id: result}`` in the schema this module's docstring gives.

    One process fits every job in the block, so the caller pays R's
    start-up time once rather than once per dataset.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    r_jobs = [_write_job_data(job, workdir) for job in jobs]
    config_path = workdir / "config.json"
    config_path.write_text(json.dumps({"jobs": r_jobs}), encoding="utf-8")

    proc = subprocess.run(
        [rscript, str(FIT_EARTH_R), str(config_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Rscript fit_earth.R failed (exit {proc.returncode}):\n"
            f"{proc.stderr}\n{proc.stdout}"
        )

    results: dict[str, dict[str, Any]] = {}
    for job, r_job in zip(jobs, r_jobs, strict=True):
        result = _desanitize(json.loads(Path(r_job["out"]).read_text(encoding="utf-8")))
        if "error" in result:
            raise RuntimeError(f"earth failed for job {job.id!r}: {result['error']}")
        results[job.id] = result
    return results


def _git_commit(cwd: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip()


def versions(earth_result: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Record the versions a fixture or a comparison run depends on.

    Python-side versions come straight from the imported packages;
    ``threadpoolctl`` names the BLAS libraries actually loaded. When
    ``earth_result`` is given (any dict ``run_earth`` returned), its
    ``r_version`` and ``earth_version`` are copied in too.
    """
    import sklearn
    import threadpoolctl

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "blas": [
            {
                "internal_api": lib["internal_api"],
                "filepath": lib["filepath"],
                "version": lib.get("version"),
            }
            for lib in threadpoolctl.threadpool_info()
        ],
        "pymars_commit": _git_commit(HERE),
    }
    if earth_result is not None:
        info["r_version"] = earth_result.get("r_version")
        info["earth_version"] = earth_result.get("earth_version")
    return info
