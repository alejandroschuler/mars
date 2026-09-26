"""Python wrappers for the harness's black-box calls into earth internals.

Each function here writes a small JSON request, runs ``Rscript blackbox.R``
(next to this file), and reads back the JSON result: the black-box use
``VALIDATION_PLAN.md`` ("Instruction files and clean room") allows. The
argument names of the internal functions (``earth:::get.gcv`` and
``earth:::pruning.pass``) were read with R's ``formals()``, which shows no
code, not by reading earth's source.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
BLACKBOX_R = HERE / "blackbox.R"

# blackbox.R writes Inf/-Inf/NaN/NA as these exact strings (its write_result
# comment explains why); undo that after json.loads(), matching driver.py's
# own _desanitize().
_R_JSON_SENTINELS = {
    "Inf": float("inf"),
    "-Inf": float("-inf"),
    "NaN": float("nan"),
    "NA": float("nan"),
}


def _desanitize(value: Any) -> Any:
    if isinstance(value, str):
        return _R_JSON_SENTINELS.get(value, value)
    if isinstance(value, list):
        return [_desanitize(v) for v in value]
    if isinstance(value, dict):
        return {k: _desanitize(v) for k, v in value.items()}
    return value


def _rows(a: np.ndarray | None) -> list[list[float]] | None:
    """Encode an array as blackbox.R's row-major nested list of ``n`` rows.

    A 1-D array of length ``n`` is one column (``n`` rows), matching a plain
    y vector or a single x column; ``np.atleast_2d`` would instead make it
    one row, which is wrong here.
    """
    if a is None:
        return None
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return [row.tolist() for row in a]


def _run(request: dict[str, Any], *, rscript: str = "Rscript") -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="pymars-blackbox-") as tmp:
        req_path = Path(tmp) / "request.json"
        out_path = Path(tmp) / "result.json"
        req_path.write_text(json.dumps(request), encoding="utf-8")
        proc = subprocess.run(
            [rscript, str(BLACKBOX_R), str(req_path), str(out_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Rscript blackbox.R failed for call {request['call']!r} "
                f"(exit {proc.returncode}):\n{proc.stderr}\n{proc.stdout}"
            )
        return _desanitize(json.loads(out_path.read_text(encoding="utf-8")))


def get_gcv(
    rss_per_subset: np.ndarray, nterms: np.ndarray, penalty: float, ncases: float
) -> np.ndarray:
    """``earth:::get.gcv(rss.per.subset, ntermsVec, penalty, ncases)``.

    ``rss_per_subset`` and ``nterms`` are equal-length vectors, one entry per
    subset size; ``penalty`` and ``ncases`` are scalars (``penalty = -1``
    asks for GCV = RSS / ncases; a case count at or below the effective
    number of parameters gives an infinite GCV, both without special-casing
    on the Python side).
    """
    result = _run(
        {
            "call": "get_gcv",
            "rss_per_subset": list(np.asarray(rss_per_subset, dtype=float)),
            "nterms": list(np.asarray(nterms, dtype=float)),
            "penalty": float(penalty),
            "ncases": float(ncases),
        }
    )
    return np.asarray(result["gcv"], dtype=float)


def pruning_pass(
    x: np.ndarray,
    y: np.ndarray,
    bx: np.ndarray,
    dirs: np.ndarray,
    penalty: float,
    *,
    pmethod: str = "backward",
    nprune: int | None = None,
    force_xtx_prune: bool = False,
    exhaustive_tol: float = 1e-10,
) -> dict[str, Any]:
    """``earth:::pruning.pass`` on a fixed forward basis.

    ``bx`` and ``dirs`` normally come from a real earth fit run with
    ``pmethod = "none"`` (the forward pass only), so the backward pass alone
    can be compared against ``_pruning.py``. Returns a dict with
    ``rss_per_subset``, ``gcv_per_subset``, ``prune_terms`` (a subset-size by
    term matrix) and ``selected_terms`` (1-based, as earth returns them).
    """
    result = _run(
        {
            "call": "pruning_pass",
            "x": _rows(x),
            "y": _rows(y),
            "bx": _rows(bx),
            "dirs": _rows(dirs),
            "penalty": float(penalty),
            "pmethod": pmethod,
            "nprune": nprune,
            "force_xtx_prune": bool(force_xtx_prune),
            "exhaustive_tol": float(exhaustive_tol),
        }
    )
    return {
        "rss_per_subset": np.asarray(result["rss_per_subset"], dtype=float),
        "gcv_per_subset": np.asarray(result["gcv_per_subset"], dtype=float),
        "prune_terms": np.asarray(result["prune_terms"], dtype=float),
        "selected_terms": np.asarray(result["selected_terms"], dtype=int),
    }


def lm_fit(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Base R's ``lm.fit(x, y)`` on fixed columns (not an earth internal)."""
    result = _run({"call": "lm_fit", "x": _rows(x), "y": _rows(y)})
    return {
        "coefficients": np.asarray(result["coefficients"], dtype=float),
        "residuals": np.asarray(result["residuals"], dtype=float),
        "rank": result["rank"],
    }


def predict_earth(
    x: np.ndarray,
    y: np.ndarray,
    newx: np.ndarray,
    earth_args: dict[str, Any],
    *,
    type: str = "link",
) -> np.ndarray:
    """Fit earth on (x, y) and predict at ``newx`` (which may lie outside the
    training range), for the prediction-at-new-points component test."""
    result = _run(
        {
            "call": "predict_earth",
            "x": _rows(x),
            "y": _rows(y),
            "newx": _rows(newx),
            "earth_args": earth_args,
            "type": type,
        }
    )
    return np.asarray(result["pred"], dtype=float)


def glm_fit(
    x: np.ndarray, y: np.ndarray, *, family: str = "binomial"
) -> dict[str, Any]:
    """R's ``glm.fit(x, y, family = <family>)``, unpenalized."""
    result = _run(
        {
            "call": "glm_fit",
            "x": _rows(x),
            "y": list(np.asarray(y, dtype=float)),
            "family": family,
        }
    )
    return {
        "coefficients": np.asarray(result["coefficients"], dtype=float),
        "fitted_values": np.asarray(result["fitted_values"], dtype=float),
    }


def multinom_fit(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """``nnet::multinom`` on fixed columns (earth's selected basis, including
    its intercept column).

    ``y`` holds class labels. The reference for the probabilities in
    ``VALIDATION_PLAN.md``, "Binary outcomes", for three or more classes.
    Returns 0-based ``coefficients`` (one row per non-baseline level, in
    ``levels`` order after the first) and ``fitted`` probabilities (one
    column per level, in ``levels`` order).
    """
    y = np.asarray(y)
    result = _run(
        {"call": "multinom_fit", "x": _rows(x), "y": [str(v) for v in y.tolist()]}
    )
    return {
        "coefficients": np.asarray(result["coefficients"], dtype=float),
        "levels": list(result["levels"]),
        "fitted": np.asarray(result["fitted"], dtype=float),
    }
