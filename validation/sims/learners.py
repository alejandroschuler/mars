"""The simulation arms (VALIDATION_PLAN.md, "Learners and settings").

Ten arms, registered by name in ``ARMS``. Each knows whether it runs on a
regression cell, a binary cell, or both (``supports_regression`` /
``supports_binary``), and how it fits:

- ``kind="inline"``: fits directly in this process (OLS, HGB, LogReg, P-fix).
- ``kind="r"``: fits through ``Rscript`` on a block of jobs at once
  (``fit_earth_block.R``): E-def, E-pym.
- ``kind="legacy"``: fits through ``.venv-legacy`` (the mars-earth 1.0.4
  wheel) on a block of jobs at once (``legacy_worker.py``): P-cur, P-ear,
  EarthClassifier, GLMEarth.

Every arm sets its MARS degree to 2 (VALIDATION_PLAN.md, "Behavior target":
"Every simulation arm sets 2"). P-fix needs ``pymars.EarthRegressor`` /
``pymars.EarthClassifier``. ``EarthRegressor`` exists (T13); until
``EarthClassifier`` does (T14), P-fix on a classification cell raises
``NotImplementedError``, and ``run.py`` records that as an ordinary per-fit
failure, like any other exception from an arm.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import pymars

REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_SCRIPT = Path(__file__)
R_SCRIPT = Path(__file__).with_name("fit_earth_block.R")
LEGACY_WORKER_SCRIPT = Path(__file__).with_name("legacy_worker.py")
LEGACY_PYTHON = REPO_ROOT / ".venv-legacy" / "bin" / "python"

MARS_DEGREE = 2  # every simulation arm's degree / max_degree (Behavior target)
_MISSING_OUTPUT = "missing output file (the block process may have failed)"
# Generous: a D7 (p=50) legacy fit alone can take minutes (Compute ledger), and
# a block batches several. This only guards against a truly hung subprocess.
SUBPROCESS_TIMEOUT_S = 4 * 3600


@dataclass(frozen=True)
class FitOutcome:
    """What one arm produced for one (cell, repetition): test-set predictions
    (probabilities, for a binary arm) plus the secondary measures the
    sparsity claim and the edge rule need, or ``error`` when the fit failed.
    """

    predictions: np.ndarray | None
    n_terms: int | None = None
    covariates_used: tuple[int, ...] | None = None
    fit_seconds: float | None = None
    extra: dict = field(default_factory=dict)  # e.g. HGB's n_iter_
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class BlockJob:
    """One (cell, repetition)'s data, queued for a block (R or legacy) arm."""

    job_id: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    binary: bool


@dataclass(frozen=True)
class Arm:
    name: str
    kind: str  # "inline", "r" or "legacy"
    supports_regression: bool
    supports_binary: bool
    source_text: str
    # The arm's own fixed settings, as a JSON-able dict: feeds run.py's cache
    # key (two arms can share source_text, e.g. E-def and E-pym both run
    # fit_earth_block.R, but must not share a cache entry).
    config: dict = field(default_factory=dict)
    fit_predict: (
        Callable[[np.ndarray, np.ndarray, np.ndarray, bool], FitOutcome] | None
    ) = None
    run_block: Callable[[Sequence[BlockJob]], dict[str, FitOutcome]] | None = None

    def supports(self, binary: bool) -> bool:
        return self.supports_binary if binary else self.supports_regression


def _inline_source(fn: Callable) -> str:
    return inspect.getsource(fn)


def _file_source(*paths: Path) -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in paths)


# ---------------------------------------------------------------------------
# Inline arms: OLS, HGB, LogReg, P-fix.
# ---------------------------------------------------------------------------


def _ols_fit_predict(x_train, y_train, x_test, binary) -> FitOutcome:
    import time

    from sklearn.linear_model import LinearRegression

    t0 = time.perf_counter()
    model = LinearRegression().fit(x_train, y_train)
    fit_seconds = time.perf_counter() - t0
    return FitOutcome(predictions=model.predict(x_test), fit_seconds=fit_seconds)


def _hgb_fit_predict(x_train, y_train, x_test, binary: bool) -> FitOutcome:
    import time

    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )

    # early_stopping=True (not "auto", which only triggers above 10,000 rows)
    # so the edge rule has an n_iter_ to check at every sample size here.
    cls = HistGradientBoostingClassifier if binary else HistGradientBoostingRegressor
    model = cls(learning_rate=0.1, max_iter=1000, early_stopping=True, random_state=0)
    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - t0
    predictions = model.predict_proba(x_test)[:, 1] if binary else model.predict(x_test)
    return FitOutcome(
        predictions=predictions,
        fit_seconds=fit_seconds,
        extra={"n_iter_": int(model.n_iter_)},
    )


def _logreg_fit_predict(x_train, y_train, x_test, binary):
    import time

    from sklearn.linear_model import LogisticRegression

    # C=np.inf: scikit-learn's unpenalized fit (see metrics.calibration_slope).
    t0 = time.perf_counter()
    model = LogisticRegression(C=np.inf).fit(x_train, y_train)
    fit_seconds = time.perf_counter() - t0
    return FitOutcome(
        predictions=model.predict_proba(x_test)[:, 1], fit_seconds=fit_seconds
    )


def _p_fix_fit_predict(x_train, y_train, x_test, binary: bool) -> FitOutcome:
    import time

    class_name = "EarthClassifier" if binary else "EarthRegressor"
    cls = getattr(pymars, class_name, None)
    if cls is None:
        raise NotImplementedError(
            f"P-fix needs pymars.{class_name} (pymars 2.0), which does not exist yet."
        )
    model = cls(max_degree=MARS_DEGREE)
    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - t0
    predictions = model.predict_proba(x_test)[:, 1] if binary else model.predict(x_test)
    # dirs_ (VALIDATION_PLAN.md, "Public API"): shape (M, p), the final
    # selected basis, code 0 where a term does not involve that covariate.
    dirs = model.dirs_
    covariates_used = tuple(int(j) for j in np.flatnonzero(np.any(dirs != 0, axis=0)))
    return FitOutcome(
        predictions=predictions,
        n_terms=dirs.shape[0],
        covariates_used=covariates_used,
        fit_seconds=fit_seconds,
    )


# ---------------------------------------------------------------------------
# Block execution shared by the R and legacy arms.
# ---------------------------------------------------------------------------


def _write_csv(path: Path, x: np.ndarray, y: np.ndarray | None) -> None:
    """CSV at full round-trip precision, for the legacy path: the reader is
    ``pandas.read_csv(..., float_precision="round_trip")`` in
    ``legacy_worker.py``, which recovers the exact float64 bit pattern (its
    default C parser can be off by 1-2 ULP; a review measured this on real
    data).
    """
    p = x.shape[1]
    header = ",".join(f"x{j + 1}" for j in range(p))
    cols = [x]
    if y is not None:
        header += ",y"
        cols.append(y.reshape(-1, 1))
    data = np.hstack(cols)
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.17g")


def _write_binary_matrix(path: Path, x: np.ndarray) -> None:
    """Raw float64, column-major (R's native matrix layout), so
    ``matrix(readBin(path, "double", n), nrow, ncol)`` (the default
    ``byrow = FALSE``) reconstructs it exactly. Unlike a decimal round trip,
    there is no parser to get almost-right: R and Python read the identical
    bit pattern this writes. ``ndarray.tofile`` always flattens in C
    (row-major) order regardless of the array's own memory layout (this is
    documented numpy behavior, not a bug to route around some other way), so
    this transposes first: a C-order flatten of ``x.T`` is a column-major
    flatten of ``x``.
    """
    np.ascontiguousarray(x.T, dtype=np.float64).tofile(path)


def _write_binary_vector(path: Path, y: np.ndarray) -> None:
    np.ascontiguousarray(y, dtype=np.float64).tofile(path)


def to_covariates_tuple(value) -> tuple[int, ...] | None:
    """A ``covariates_used`` JSON value as a tuple. jsonlite's ``auto_unbox``
    serializes a length-1 (or length-0) R integer vector as a bare JSON
    scalar rather than a one-element array unless the R side wraps it in
    ``I()`` (as ``fit_earth_block.R`` does); this tolerates that shape too,
    defensively, wherever a ``covariates_used`` JSON value is read back.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return (value,)
    return tuple(value)


def _read_result(path: Path) -> FitOutcome:
    if not path.is_file():
        return FitOutcome(predictions=None, error=_MISSING_OUTPUT)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "error" in data:
        return FitOutcome(predictions=None, error=data["error"])
    return FitOutcome(
        predictions=np.asarray(data["predictions"], dtype=np.float64),
        n_terms=data.get("n_terms"),
        covariates_used=to_covariates_tuple(data.get("covariates_used")),
        fit_seconds=data.get("fit_seconds"),
    )


def _run_subprocess_block(
    jobs: Sequence[BlockJob],
    command: list[str],
    build_job_entry: Callable[[BlockJob, Path], dict],
) -> dict[str, FitOutcome]:
    with tempfile.TemporaryDirectory(prefix="pymars-sim-block-") as tmp:
        tmp_path = Path(tmp)
        manifest_jobs = []
        out_paths = {}
        for job in jobs:
            out_json = tmp_path / f"{job.job_id}_out.json"
            out_paths[job.job_id] = out_json
            manifest_jobs.append(build_job_entry(job, tmp_path))
            manifest_jobs[-1]["out_json"] = str(out_json)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({"jobs": manifest_jobs}), encoding="utf-8")

        try:
            proc = subprocess.run(
                [*command, str(manifest_path)],
                capture_output=True,
                text=True,
                check=False,
                timeout=SUBPROCESS_TIMEOUT_S,
            )
            returncode, stderr = proc.returncode, proc.stderr
        except subprocess.TimeoutExpired:
            returncode, stderr = None, f"timed out after {SUBPROCESS_TIMEOUT_S} s"
        outcomes = {job_id: _read_result(p) for job_id, p in out_paths.items()}
        if returncode != 0:
            stderr_tail = stderr[-2000:]
            for job_id, outcome in outcomes.items():
                if outcome.error == _MISSING_OUTPUT:
                    outcomes[job_id] = FitOutcome(
                        predictions=None,
                        error=f"process exited {returncode}: {stderr_tail}",
                    )
        return outcomes


def _make_r_run_block(
    r_args: dict,
) -> Callable[[Sequence[BlockJob]], dict[str, FitOutcome]]:
    def build(job: BlockJob, tmp_path: Path) -> dict:
        train_x = tmp_path / f"{job.job_id}_train_x.bin"
        train_y = tmp_path / f"{job.job_id}_train_y.bin"
        test_x = tmp_path / f"{job.job_id}_test_x.bin"
        _write_binary_matrix(train_x, job.x_train)
        _write_binary_vector(train_y, job.y_train)
        _write_binary_matrix(test_x, job.x_test)
        return {
            "id": job.job_id,
            "train_x": str(train_x),
            "train_y": str(train_y),
            "test_x": str(test_x),
            "n_train": job.x_train.shape[0],
            "n_test": job.x_test.shape[0],
            "p": job.x_train.shape[1],
            "args": r_args,
            "glm_family": "binomial" if job.binary else None,
        }

    def run_block(jobs: Sequence[BlockJob]) -> dict[str, FitOutcome]:
        return _run_subprocess_block(jobs, ["Rscript", str(R_SCRIPT)], build)

    return run_block


def _make_legacy_run_block(
    class_name: str, kwargs: dict
) -> Callable[[Sequence[BlockJob]], dict[str, FitOutcome]]:
    def build(job: BlockJob, tmp_path: Path) -> dict:
        train_csv = tmp_path / f"{job.job_id}_train.csv"
        test_csv = tmp_path / f"{job.job_id}_test.csv"
        _write_csv(train_csv, job.x_train, job.y_train)
        _write_csv(test_csv, job.x_test, None)
        return {
            "id": job.job_id,
            "train_csv": str(train_csv),
            "test_csv": str(test_csv),
            "class_name": class_name,
            "kwargs": kwargs,
        }

    def run_block(jobs: Sequence[BlockJob]) -> dict[str, FitOutcome]:
        return _run_subprocess_block(
            jobs, [str(LEGACY_PYTHON), str(LEGACY_WORKER_SCRIPT)], build
        )

    return run_block


# ---------------------------------------------------------------------------
# The registry. Each R/legacy arm's settings are named once and used both to
# build its subprocess closure and as the ``config`` that feeds the cache key.
# ---------------------------------------------------------------------------

_E_DEF_ARGS = {"degree": MARS_DEGREE}
_E_PYM_ARGS = {
    "degree": MARS_DEGREE,
    "penalty": 6,
    "thresh": 0,
    "minspan": 1,
    "endspan": 1,
    "fast.k": 0,
    "Adjust.endspan": 1,
}
_P_CUR_KWARGS = {"max_degree": MARS_DEGREE}
_P_EAR_KWARGS = {
    "max_degree": MARS_DEGREE,
    "penalty": 1.5,
    "minspan_alpha": 0.05,
    "endspan_alpha": 0.05,
}
_EARTH_CLASSIFIER_KWARGS = {"max_degree": MARS_DEGREE}
_GLM_EARTH_KWARGS = {"max_degree": MARS_DEGREE}

ARMS: dict[str, Arm] = {
    "E-def": Arm(
        name="E-def",
        kind="r",
        supports_regression=True,
        supports_binary=True,
        source_text=_file_source(THIS_SCRIPT, R_SCRIPT),
        config={"args": _E_DEF_ARGS},
        run_block=_make_r_run_block(_E_DEF_ARGS),
    ),
    "E-pym": Arm(
        name="E-pym",
        kind="r",
        supports_regression=True,
        supports_binary=False,
        source_text=_file_source(THIS_SCRIPT, R_SCRIPT),
        config={"args": _E_PYM_ARGS},
        run_block=_make_r_run_block(_E_PYM_ARGS),
    ),
    "P-cur": Arm(
        name="P-cur",
        kind="legacy",
        supports_regression=True,
        supports_binary=False,
        source_text=_file_source(THIS_SCRIPT, LEGACY_WORKER_SCRIPT),
        config={"class_name": "Earth", "kwargs": _P_CUR_KWARGS},
        run_block=_make_legacy_run_block("Earth", _P_CUR_KWARGS),
    ),
    "P-ear": Arm(
        name="P-ear",
        kind="legacy",
        supports_regression=True,
        supports_binary=False,
        source_text=_file_source(THIS_SCRIPT, LEGACY_WORKER_SCRIPT),
        config={"class_name": "Earth", "kwargs": _P_EAR_KWARGS},
        run_block=_make_legacy_run_block("Earth", _P_EAR_KWARGS),
    ),
    "EarthClassifier": Arm(
        name="EarthClassifier",
        kind="legacy",
        supports_regression=False,
        supports_binary=True,
        source_text=_file_source(THIS_SCRIPT, LEGACY_WORKER_SCRIPT),
        config={"class_name": "EarthClassifier", "kwargs": _EARTH_CLASSIFIER_KWARGS},
        run_block=_make_legacy_run_block("EarthClassifier", _EARTH_CLASSIFIER_KWARGS),
    ),
    "GLMEarth": Arm(
        name="GLMEarth",
        kind="legacy",
        supports_regression=False,
        supports_binary=True,
        source_text=_file_source(THIS_SCRIPT, LEGACY_WORKER_SCRIPT),
        config={"class_name": "GLMEarth", "kwargs": _GLM_EARTH_KWARGS},
        run_block=_make_legacy_run_block("GLMEarth", _GLM_EARTH_KWARGS),
    ),
    "P-fix": Arm(
        name="P-fix",
        kind="inline",
        supports_regression=True,
        supports_binary=True,
        source_text=_inline_source(_p_fix_fit_predict),
        fit_predict=_p_fix_fit_predict,
    ),
    "OLS": Arm(
        name="OLS",
        kind="inline",
        supports_regression=True,
        supports_binary=False,
        source_text=_inline_source(_ols_fit_predict),
        fit_predict=_ols_fit_predict,
    ),
    "LogReg": Arm(
        name="LogReg",
        kind="inline",
        supports_regression=False,
        supports_binary=True,
        source_text=_inline_source(_logreg_fit_predict),
        fit_predict=_logreg_fit_predict,
    ),
    "HGB": Arm(
        name="HGB",
        kind="inline",
        supports_regression=True,
        supports_binary=True,
        source_text=_inline_source(_hgb_fit_predict),
        fit_predict=_hgb_fit_predict,
    ),
}


def arms_for(binary: bool) -> list[str]:
    """Arm names applicable to a regression cell (``binary=False``) or a
    binary cell (``binary=True``), in the registry's own order.
    """
    return [name for name, arm in ARMS.items() if arm.supports(binary)]
