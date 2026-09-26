"""Fit the legacy (mars-earth 1.0.4) estimators on a block of datasets, run
under ``.venv-legacy`` (VALIDATION_PLAN.md, "Build": "the legacy arms run in
their own venv... called through a subprocess that fits several repetitions
per process"). One process handles the whole block, so Python's own import
and interpreter start-up cost is paid once, not per repetition.

Usage: ``.venv-legacy/bin/python legacy_worker.py <manifest.json>``, from the
main venv's ``learners.py`` (never imported there: this module needs the
1.0.4 wheel installed).

The manifest is ``{"jobs": [{"id", "train_csv", "test_csv", "out_json",
"class_name", "kwargs"}, ...]}``. "class_name" is one of "Earth",
"EarthClassifier" or "GLMEarth" (``pymars`` 1.0.4's own classes); "kwargs" are
its constructor keywords. Each job's train CSV has columns x1..xp, y (y is
0/1 for a classifier); its test CSV has x1..xp only. Each job writes its own
out_json atomically (a temp file, then ``os.replace``); one job's error never
stops the rest of the block.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

import pymars as earth  # the 1.0.4 wheel (mars-earth), from .venv-legacy

CLASSES = {
    "Earth": earth.Earth,
    "EarthClassifier": earth.EarthClassifier,
    "GLMEarth": earth.GLMEarth,
}


def _x_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("x")]
    return sorted(cols, key=lambda c: int(c[1:]))


def _basis_covariates_used(basis) -> list[int]:
    """The 0-indexed covariates that appear in a non-constant basis term,
    from each term's own ``get_involved_variables()`` (matches the R side's
    0-indexed ``covariates_used``, from ``fit$dirs``).
    """
    used: set[int] = set()
    for term in basis:
        if term.is_constant():
            continue
        used.update(term.get_involved_variables())
    return sorted(used)


def _glm_earth_probabilities(model: earth.GLMEarth, x: np.ndarray) -> np.ndarray:
    """P(Y=1) from a fitted GLMEarth. ``GLMEarth.predict`` returns a 0.5
    threshold, not a probability (VALIDATION_PLAN.md finding F9), so this
    replicates its own ``predict`` up to ``glm_.predict_proba`` instead
    (``pymars/glm.py`` in the 1.0.4 wheel: ``earth_._scrub_input_data`` then
    ``earth_._build_basis_matrix`` build the same basis matrix ``glm_`` was
    fit on).
    """
    x_proc, mask, _ = model.earth_._scrub_input_data(x, np.zeros(len(x)))
    basis_matrix = model.earth_._build_basis_matrix(x_proc, model.basis_, mask)
    return model.glm_.predict_proba(basis_matrix)[:, 1]


def run_job(job: dict) -> dict:
    # float_precision="round_trip": pandas' default C parser can be off by
    # 1-2 ULP from the float64 numpy wrote (a review measured this on real
    # data); round_trip recovers the exact bit pattern, as every arm must see
    # the same data.
    train = pd.read_csv(job["train_csv"], float_precision="round_trip")
    test = pd.read_csv(job["test_csv"], float_precision="round_trip")
    x_cols = _x_columns(train)
    x_train = train[x_cols].to_numpy(dtype=np.float64)
    y_train = train["y"].to_numpy(dtype=np.float64)
    x_test = test[x_cols].to_numpy(dtype=np.float64)

    cls = CLASSES[job["class_name"]]
    model = cls(**job["kwargs"])

    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - t0

    if job["class_name"] == "EarthClassifier":
        predictions = model.predict_proba(x_test)[:, 1]
    elif job["class_name"] == "GLMEarth":
        predictions = _glm_earth_probabilities(model, x_test)
    else:
        predictions = model.predict(x_test)

    return {
        "predictions": np.asarray(predictions, dtype=np.float64).tolist(),
        "n_terms": len(model.basis_),
        "covariates_used": _basis_covariates_used(model.basis_),
        "fit_seconds": fit_seconds,
    }


def write_result(out_json: str, result: dict) -> None:
    tmp = f"{out_json}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f)
    os.replace(tmp, out_json)


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as f:
        manifest = json.load(f)
    for job in manifest["jobs"]:
        try:
            result = run_job(job)
        except Exception as e:  # one job's error must not stop the rest of the block
            result = {"error": f"{type(e).__name__}: {e}"}
        write_result(job["out_json"], result)


if __name__ == "__main__":
    main()
