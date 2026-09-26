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
from driver import _desanitize

HERE = Path(__file__).resolve().parent
BLACKBOX_R = HERE / "blackbox.R"

# blackbox.R writes Inf/-Inf/NaN/NA as these exact strings, with na = "string"
# (blackbox.R's write_result comment explains why), the same convention
# fit_earth.R uses; driver.py's _desanitize() undoes it the same way here, so
# that the two scripts share one decoder instead of drifting apart. NA
# becomes None, distinct from a NaN: lm_fit's coefficients has an R NA (not
# nan) for a column lm.fit cannot estimate (rank-deficient x, for example two
# identical columns), and multinom_fit's levels can itself contain the
# string "NA" as a real factor level, which _STRING_ONLY_KEYS protects.


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


def _has_none(value: Any) -> bool:
    if isinstance(value, list):
        return any(_has_none(v) for v in value)
    return value is None


def _float_array(values: Any) -> np.ndarray:
    """A numpy array of ``values`` (a possibly nested list, already through
    ``_desanitize``), ``dtype=float`` unless an R NA (``None``) is present.

    ``np.asarray(values, dtype=float)`` silently turns ``None`` into ``nan``
    (numpy's usual float cast), which would erase the NA/NaN distinction
    ``_desanitize`` exists to keep; a rank-deficient ``lm.fit``/``glm.fit``
    (for example two identical columns) gives R's real NA, not a NaN, for
    the aliased coefficient. ``dtype=object`` keeps ``None`` as ``None``
    instead.
    """
    dtype = object if _has_none(values) else float
    return np.asarray(values, dtype=dtype)


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


def versions(*, rscript: str = "Rscript") -> dict[str, str]:
    """The R and earth versions this process's ``Rscript`` would fit with.

    Every ``blackbox.R`` call's result already carries ``r_version`` and
    ``earth_version`` (so a component fixture has them without a
    ``driver.run_earth`` result to copy them from, the way ``fit_earth.R``'s
    callers do); this call needs no other request field.
    """
    result = _run({"call": "versions"}, rscript=rscript)
    return {"r_version": result["r_version"], "earth_version": result["earth_version"]}


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


def lm_fit(
    x: np.ndarray, y: np.ndarray, *, w: np.ndarray | None = None
) -> dict[str, Any]:
    """Base R's ``lm.fit(x, y)`` on fixed columns (not an earth internal),
    or ``lm.wfit(x, y, w)`` when ``w`` is given (LA-4/PRUNE-8's weighted
    coefficients; review round 1, #42 finding 5).

    ``coefficients`` is ``dtype=object`` (not ``float``) when ``x`` is rank
    deficient (for example two identical columns): R gives the aliased
    coefficient as NA, which comes back here as ``None``, not ``nan``.
    """
    result = _run(
        {
            "call": "lm_fit",
            "x": _rows(x),
            "y": _rows(y),
            "w": None if w is None else list(np.asarray(w, dtype=float)),
        }
    )
    return {
        "coefficients": _float_array(result["coefficients"]),
        "residuals": np.asarray(result["residuals"], dtype=float),
        "rank": result["rank"],
    }


def earth_factor_fit(
    labels: np.ndarray | None,
    y: np.ndarray,
    earth_args: dict[str, Any],
    *,
    other_x: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit earth with ``labels`` as one genuine R factor column (plus
    ``other_x``'s numeric columns, if any), and return a summary: fitted
    values, ``gcv``, ``rsq`` and the term count.

    The "factor" side of S19's comparison (VALIDATION_PLAN.md, "Categorical
    inputs"); ``labels=None`` (with ``other_x`` given, for example the
    OneHotEncoder-style dummy columns) is the "dummies" side, which could
    also go through ``driver.run_earth``, but sharing this call keeps both
    sides' fitted-value scale (a plain numeric fit, no CSV round trip)
    identical for the comparison. Only a summary comes back because a
    factor fit's ``dirs``/``cuts``/``bx`` are not comparable in shape with
    the dummy-encoded fit's.
    """
    label_list = (
        None if labels is None else [str(v) for v in np.asarray(labels).tolist()]
    )
    result = _run(
        {
            "call": "earth_factor_fit",
            "labels": label_list,
            "y": list(np.asarray(y, dtype=float)),
            "other_x": _rows(other_x),
            "earth_args": earth_args,
        }
    )
    return {
        "fitted": np.asarray(result["fitted"], dtype=float),
        "gcv": float(result["gcv"]),
        "rsq": float(result["rsq"]),
        "nterms": int(result["nterms"]),
    }


def predict_earth(
    x: np.ndarray,
    y: np.ndarray,
    newx: np.ndarray,
    earth_args: dict[str, Any],
    *,
    type: str = "link",
) -> dict[str, Any]:
    """Fit earth on (x, y), predict at ``newx`` (which may lie outside the
    training range), and return the model alongside the prediction: ``pred``,
    ``dirs``, ``cuts``, ``selected_terms`` (1-based) and ``coefficients``.

    For the prediction-at-new-points component test (review round 1, #42
    finding 2): without the model, matching ``pred`` needs a whole forward-
    pass refit that also depends on the forward pass and LA-7, not TERM-3
    (the basis evaluated at new points) in isolation.
    """
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
    return {
        "pred": np.asarray(result["pred"], dtype=float),
        "dirs": np.asarray(result["dirs"], dtype=float),
        "cuts": np.asarray(result["cuts"], dtype=float),
        "selected_terms": np.asarray(result["selected_terms"], dtype=int),
        "coefficients": np.asarray(result["coefficients"], dtype=float),
    }


def glm_fit(
    x: np.ndarray, y: np.ndarray, *, family: str = "binomial"
) -> dict[str, Any]:
    """R's ``glm.fit(x, y, family = <family>)``, unpenalized.

    ``coefficients`` is ``dtype=object`` (not ``float``) when ``x`` is rank
    deficient: see ``lm_fit``, the same aliasing. ``converged`` and
    ``warnings`` (review round 1, #42 finding 6) let a caller tell a valid
    GLM-3 reference from one earth itself would warn on (a quasi-separated
    fit, for example), since GLM-3 only promises agreement "where glm
    converges without a warning".
    """
    result = _run(
        {
            "call": "glm_fit",
            "x": _rows(x),
            "y": list(np.asarray(y, dtype=float)),
            "family": family,
        }
    )
    return {
        "coefficients": _float_array(result["coefficients"]),
        "fitted_values": np.asarray(result["fitted_values"], dtype=float),
        "converged": bool(result["converged"]),
        "warnings": list(result["warnings"] or []),
    }


def fit_bx_dirs(
    x: np.ndarray, y: np.ndarray, earth_args: dict[str, Any]
) -> dict[str, Any]:
    """Fit earth on (x, y) and return its basis: ``bx`` (the design matrix),
    ``dirs``, ``cuts`` and ``selected_terms`` (1-based, as earth returns
    them).

    Review round 2 (#42 spec, non-blocking): ``fit$dirs``/``fit$cuts`` hold
    every *forward* term earth ever added, selected or not (for example,
    ``s18_multinom``'s basis has 13 ``dirs`` rows but only 3
    ``selected_terms``); ``fit$bx`` holds only the *selected* columns,
    except with ``pmethod = "none"`` (no ``nprune``), which
    ``pruning_fixed_basis`` passes so that its own backward pass has
    something left to prune, and which leaves every forward term selected.
    A caller that wants the terms ``dirs``/``cuts`` describe restricted to
    the ones actually in the model takes ``dirs[selected_terms - 1]`` (and
    ``cuts[selected_terms - 1]``), 0-indexing the 1-based ``selected_terms``
    this call returns.

    For the pruning-of-a-fixed-basis and the classifier-refit component
    tests, which need a real ``bx``/``dirs`` pair to hand to
    ``pruning_pass``, ``lm_fit`` or ``multinom_fit``. ``bx`` is not part of
    ``driver.run_earth``'s result schema (it would duplicate earth's own
    forward-pass output in every dataset fixture), so this reaches past it
    with its own black-box call, the same way ``test_blackbox.py``'s
    ``_fixed_basis`` helper already does.
    """
    result = _run(
        {
            "call": "fit_bx_dirs",
            "x": _rows(x),
            "y": _rows(y),
            "earth_args": earth_args,
        }
    )
    return {
        "bx": np.asarray(result["bx"], dtype=float),
        "dirs": np.asarray(result["dirs"], dtype=float),
        "cuts": np.asarray(result["cuts"], dtype=float),
        "selected_terms": np.asarray(result["selected_terms"], dtype=int),
    }


def multinom_fit(
    x: np.ndarray, y: np.ndarray, *, maxit: int = 10000, reltol: float = 1e-15
) -> dict[str, Any]:
    """``nnet::multinom`` on fixed columns (earth's selected basis, including
    its intercept column).

    ``y`` holds class labels. The reference for the probabilities in
    ``VALIDATION_PLAN.md``, "Binary outcomes", for three or more classes.
    Returns 0-based ``coefficients`` (one row per non-baseline level, in
    ``levels`` order after the first) and ``fitted`` probabilities (one
    column per level, in ``levels`` order).

    ``maxit`` raises nnet's default cap of 100 iterations when a caller's
    design needs more; ``convergence`` (nnet's own code; 0 is converged)
    and ``warnings`` let a caller require convergence rather than silently
    keeping an unconverged iterate (review round 1, #42/#43 finding
    1/3/5). Separable labels never converge whatever ``maxit`` is: the fix
    there is a non-separable design, not a larger cap.

    ``reltol`` (review round 2, #42/#43 blocking finding 1): convergence
    code 0 only means the objective changed by less than ``reltol``
    between iterations, not that the fit reached GLM-2's actual minimum.
    nnet's own default (1e-8) can stop far enough short of the minimum
    that GLM-3's tolerance (1e-5 relative, 1e-7 absolute) fails a solver
    that reaches it, so this defaults far tighter. A caller should still
    check stability (``_assert_multinom_is_stable`` in gen_fixtures.py)
    before writing a fixture, since a tight ``reltol`` alone is not proof
    of convergence to the minimum, only evidence for it.
    """
    y = np.asarray(y)
    result = _run(
        {
            "call": "multinom_fit",
            "x": _rows(x),
            "y": [str(v) for v in y.tolist()],
            "maxit": maxit,
            "reltol": reltol,
        }
    )
    return {
        "coefficients": np.asarray(result["coefficients"], dtype=float),
        "levels": list(result["levels"]),
        "fitted": np.asarray(result["fitted"], dtype=float),
        "convergence": int(result["convergence"]),
        "warnings": list(result["warnings"] or []),
    }
