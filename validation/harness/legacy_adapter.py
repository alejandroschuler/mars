"""Fit the legacy code (mars-earth 1.0.4, `validation/legacy/make_venv.sh`'s
`.venv-legacy`) in a subprocess, and convert its result to the common result
schema `driver.py`'s module docstring documents.

A subprocess is necessary, not a convenience: the legacy package and the new
one share the name ``pymars``, so both cannot be imported in one process.
The subprocess runs with ``-I`` (isolated mode): without it, a bare
``python -c`` run from a checkout's own root would import that checkout's
``pymars/`` (the new code) ahead of ``.venv-legacy``'s installed one, because
``-c`` puts the current directory first on ``sys.path``.

Common schema notes for this adapter specifically:

- ``dirs``/``cuts``/``coef`` describe the final, already-pruned model
  (``self.basis_``/``self.coef_``): unlike earth, the legacy code keeps no
  larger superset on the fitted object, so ``selected_terms`` is ``None``
  (not applicable, rather than a trivial range that would misleadingly
  suggest one exists in the schema's earth-derived sense).
- ``fwd_dirs``/``fwd_cuts`` describe the last forward-pass snapshot
  (``record_.fwd_basis_[-1]``, before pruning), and ``fwd_rss`` is
  ``record_.fwd_rss_`` directly. The legacy code logs the two at different
  points in the forward pass, so ``len(fwd_rss)`` need not equal
  ``fwd_dirs``'s row count; ``fwd_rss`` is still non-increasing (adding a
  column to a least-squares fit cannot raise its RSS), which is what a
  forward-RSS-path comparison needs.
- ``rss_per_subset``/``gcv_per_subset`` are ``record_.pruning_trace_rss_``/
  ``record_.pruning_trace_gcv_`` directly. Their length need not equal the
  number of terms; this adapter reports what the legacy code recorded.
- ``prune_terms`` is always ``None``: the legacy trace gives a full basis
  snapshot per step, not a fixed-shape (subset size x term) index matrix
  like earth's, and the plan does not ask this adapter to reshape it into
  one.
- No sample weights: the legacy ``Earth.fit(X, y)`` takes none.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
LEGACY_PYTHON = REPO_ROOT / ".venv-legacy" / "bin" / "python"

# Runs inside .venv-legacy, so pymars here is mars-earth 1.0.4. It converts a
# list of legacy BasisFunction objects to earth's (dirs, cuts) convention by
# walking each term's parent1 chain (VALIDATION_PLAN.md's harness section;
# the seed prototype validation/legacy/compare_earth.py's describe_py does
# the same walk, for the same reason: only parent1 is ever set in practice,
# parent2 stays None), reading only public attributes, never earth.
_LEGACY_FIT_SCRIPT = r"""
import json
import sys

import numpy as np
import pymars
from pymars._basis import ConstantBasisFunction, HingeBasisFunction, LinearBasisFunction


def to_dirs_cuts(basis_list, p):
    if not basis_list:
        return None, None
    dirs = np.zeros((len(basis_list), p), dtype=int)
    cuts = np.zeros((len(basis_list), p), dtype=float)
    for i, bf in enumerate(basis_list):
        node = bf
        while node is not None and not isinstance(node, ConstantBasisFunction):
            j = node.variable_idx
            if isinstance(node, HingeBasisFunction):
                dirs[i, j] = 1 if node.is_right_hinge else -1
                cuts[i, j] = node.knot_val
            elif isinstance(node, LinearBasisFunction):
                dirs[i, j] = 2
            node = node.parent1
    return dirs.tolist(), cuts.tolist()


cfg = json.loads(sys.stdin.read())
X = np.asarray(cfg["X"], dtype=float)
y = np.asarray(cfg["y"], dtype=float)
p = X.shape[1]

model = pymars.Earth(**cfg["earth_kwargs"])
model.fit(X, y)

dirs, cuts = to_dirs_cuts(model.basis_, p)
record = model.record_
fwd_dirs, fwd_cuts = (None, None)
if record is not None and record.fwd_basis_:
    fwd_dirs, fwd_cuts = to_dirs_cuts(record.fwd_basis_[-1], p)

coef = (
    np.asarray(model.coef_, dtype=float).reshape(-1, 1).tolist()
    if model.coef_ is not None else None
)
fwd_rss = [float(v) for v in record.fwd_rss_] if record is not None else None
rss_per_subset = (
    [float(v) for v in record.pruning_trace_rss_] if record is not None else None
)
gcv_per_subset = (
    [float(v) for v in record.pruning_trace_gcv_] if record is not None else None
)

out = {
    "dirs": dirs,
    "cuts": cuts,
    "selected_terms": None,
    "prune_terms": None,
    "coef": coef,
    "rss": float(model.rss_) if model.rss_ is not None else None,
    "gcv": float(model.gcv_) if model.gcv_ is not None else None,
    "fwd_dirs": fwd_dirs,
    "fwd_cuts": fwd_cuts,
    "fwd_rss": fwd_rss,
    "rss_per_subset": rss_per_subset,
    "gcv_per_subset": gcv_per_subset,
    "n_terms": len(model.basis_) if model.basis_ is not None else None,
}
json.dump(out, sys.stdout)
"""


def fit_legacy(X: np.ndarray, y: np.ndarray, **earth_kwargs: Any) -> dict[str, Any]:
    """Fit ``pymars.Earth(**earth_kwargs)`` (mars-earth 1.0.4) on ``(X, y)``
    in ``.venv-legacy``, and return the result in the common schema.

    Raises ``FileNotFoundError`` if ``.venv-legacy`` has not been made yet
    (``validation/legacy/make_venv.sh``), and ``RuntimeError`` if the legacy
    fit itself raises.
    """
    if not LEGACY_PYTHON.is_file():
        raise FileNotFoundError(
            f"{LEGACY_PYTHON} does not exist; run validation/legacy/make_venv.sh first"
        )
    payload = {
        "X": np.asarray(X, dtype=float).tolist(),
        "y": np.asarray(y, dtype=float).tolist(),
        "earth_kwargs": earth_kwargs,
    }
    proc = subprocess.run(
        [str(LEGACY_PYTHON), "-I", "-c", _LEGACY_FIT_SCRIPT],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"the legacy fit failed:\n{proc.stderr}\n{proc.stdout}")
    return json.loads(proc.stdout)
