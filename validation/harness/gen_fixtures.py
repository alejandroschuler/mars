"""The fixture generator framework (VALIDATION_PLAN.md, "Correctness
against earth"): a registry of datasets and a registry of earth argument
sets ("modes"), which together make `validation/fixtures/<dataset>_<mode>
.json`, one file per (dataset, mode) pair, each with that pair's inputs,
earth arguments, `driver.run_earth` result and versions.

This registers one dataset, S01 ("one covariate, two true knots, 200
cases"), under two modes, to prove the pipeline; T05 adds S02 through S20.

Usage (also validation/README.md):
    python validation/harness/gen_fixtures.py           # write every fixture
    python validation/harness/gen_fixtures.py --check   # regenerate into a
                                                         # temp folder and
                                                         # report any diff
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blackbox
import driver
import trace_parse

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
COMPONENTS_DIR = FIXTURES_DIR / "components"


@dataclass
class CheckResult:
    """``check()``'s outcome: ``problems`` (a difference in what a fixture
    claims about earth: its inputs, earth arguments or result, or a
    fixture missing outright) fail the check; ``notes`` (a ``versions``
    difference: this machine's R, earth, numpy, scikit-learn or BLAS is not
    the one that made the committed fixture) do not, since nothing about
    earth conformance follows from running the generator on different
    software versions.
    """

    problems: list[str]
    notes: list[str]


@dataclass
class Dataset:
    """One registered dataset: everything `driver.EarthJob` needs besides
    the earth arguments (which a mode supplies)."""

    id: str
    X: np.ndarray
    y: np.ndarray
    X_test: np.ndarray | None = None
    factor_response: bool = False


DatasetFn = Callable[[], Dataset]
REGISTRY: dict[str, DatasetFn] = {}

# earth argument sets ("modes"), VALIDATION_PLAN.md's "Comparison modes":
# defaults_d1 is earth at its own defaults; matched_d1 is the legacy-code
# matched mode from the seed prototype (validation/legacy/compare_earth.py),
# hinge-only and every case a candidate knot, so the two implementations
# should make the same forward choices except at near-ties.
MODES: dict[str, dict[str, Any]] = {
    "defaults_d1": {"degree": 1},
    "matched_d1": {
        "degree": 1,
        "penalty": 2,
        "nk": 21,
        "thresh": 0,
        "minspan": 1,
        "endspan": 1,
        "fast.k": 0,
        "Auto.linpreds": False,
        "pmethod": "backward",
    },
}


def register(fn: DatasetFn) -> DatasetFn:
    """Decorator: register a dataset generator under its own return value's
    ``id`` (calling it once, at import time, to read that id)."""
    REGISTRY[fn().id] = fn
    return fn


@register
def s01() -> Dataset:
    """S01: one covariate, two true knots, 200 cases. Knot recovery."""
    rng = np.random.default_rng(1)
    n, n_test = 200, 50
    x = rng.uniform(0, 1, size=(n, 1))
    x_test = rng.uniform(0, 1, size=(n_test, 1))

    def f(x0: np.ndarray) -> np.ndarray:
        return 2 * np.maximum(0, x0 - 0.3) - 3 * np.maximum(0, x0 - 0.7)

    y = f(x[:, 0]) + rng.normal(scale=0.1, size=n)
    return Dataset(id="S01", X=x, y=y, X_test=x_test)


# Component fixtures (VALIDATION_PLAN.md, "Component tests"; T05 brief,
# deliverable 2): each one calls a blackbox.py function directly, rather
# than building an EarthJob for driver.py, since a component test targets
# one earth-adjacent internal (earth:::get.gcv, earth:::pruning.pass,
# lm.fit, predict.earth, glm.fit, nnet::multinom) or the trace-based knot
# scan, not a whole earth() fit compared end to end. Each generator returns
# a JSON-able payload (no "versions" key yet; _component_payload adds it,
# the same way _fixture_payload does for a dataset).
ComponentFn = Callable[[], dict[str, Any]]
COMPONENT_REGISTRY: dict[str, ComponentFn] = {}


def register_component(fn: ComponentFn) -> ComponentFn:
    """Decorator: register a component-fixture generator under its function
    name. Unlike ``register`` (datasets), this does not call ``fn`` at
    import time: a component payload comes from a real earth call (through
    ``blackbox.py``), so building it eagerly on every ``import
    gen_fixtures`` would run R on every test collection, not just when a
    fixture is actually (re)generated."""
    COMPONENT_REGISTRY[fn.__name__] = fn
    return fn


@register_component
def gcv_grid() -> dict[str, Any]:
    """VALIDATION_PLAN.md, "Component tests" > "GCV function": earth:::
    get.gcv over a grid of (rss.per.subset, 1 to 41 terms) at penalties 0 to
    6 and -1, and n from 10 to 100,000. get.gcv's formula (GCV-2) only needs
    one RSS value per term count, not a real fit, so this uses one fixed,
    deterministic, strictly decreasing sequence for every cell."""
    nterms = list(range(1, 42))
    rss_per_subset = [100.0 / m for m in nterms]
    penalties = [0, 1, 2, 3, 4, 5, 6, -1]
    ncases = [10, 15, 20, 41, 50, 100, 1_000, 100_000]
    grid = [
        {
            "penalty": penalty,
            "ncases": n,
            "gcv": blackbox.get_gcv(rss_per_subset, nterms, penalty, n).tolist(),
        }
        for penalty in penalties
        for n in ncases
    ]
    return {"nterms": nterms, "rss_per_subset": rss_per_subset, "grid": grid}


def _pruning_case(
    X: np.ndarray, y: np.ndarray, earth_args: dict[str, Any], penalty: float
) -> dict[str, Any]:
    basis = blackbox.fit_bx_dirs(X, y, earth_args)
    pruned = blackbox.pruning_pass(
        X, y, basis["bx"], basis["dirs"], penalty, pmethod="backward"
    )
    return {
        "y": y.tolist(),
        "dirs": basis["dirs"].tolist(),
        "cuts": basis["cuts"].tolist(),
        "rss_per_subset": pruned["rss_per_subset"].tolist(),
        "gcv_per_subset": pruned["gcv_per_subset"].tolist(),
        "prune_terms": pruned["prune_terms"].tolist(),
        "selected_terms": pruned["selected_terms"].tolist(),
    }


def _rss_of_subset(B: np.ndarray, Y: np.ndarray, cols: list[int]) -> float:
    """The weighted (here, unweighted) RSS of Y on B[:, cols] (LA-1),
    summed over Y's columns (GCV-3): the one primitive both self-check
    rules below share."""
    sub = B[:, cols]
    coef, *_ = np.linalg.lstsq(sub, Y, rcond=None)
    resid = Y - sub @ coef
    return float(np.sum(resid**2))


def _prune_k1(B: np.ndarray, Y: np.ndarray) -> dict[int, frozenset[int]]:
    """PRUNE-3's K = 1 rule (the leaps-style ordered swap), applied here to
    a possibly multi-column Y by summing RSS over its columns: what an
    implementation would give if it (wrongly) ran the K = 1 algorithm for
    K >= 2 too. Returns T[m], the lowest-RSS set of terms found at each
    size over every order offered (PRUNE-2), not just the final swap
    order, since T[m] need not be nested and a later, worse-looking swap
    can still offer a better T[m] at a smaller size than the current
    order's own prefix. This is a from-spec self-check inside the fixture
    generator (verifying that pruning_fixed_basis's several-responses case
    actually separates PRUNE-3's two rules), not part of the fixture or of
    any T08 test."""
    m_f = B.shape[1]
    order = list(range(m_f))
    best_rss: dict[int, float] = {}
    best_set: dict[int, frozenset[int]] = {}

    def offer(ord_list: list[int]) -> None:
        for m in range(1, m_f + 1):
            u = frozenset(ord_list[:m])
            rss = _rss_of_subset(B, Y, sorted(u))
            if m not in best_rss or rss < best_rss[m]:
                best_rss[m] = rss
                best_set[m] = u

    offer(order)
    for pos in range(m_f, 1, -1):
        best_idx, best_pos_rss = None, None
        for idx in range(1, pos):
            cols = [order[j] for j in range(pos) if j != idx]
            rss = _rss_of_subset(B, Y, cols)
            if (
                best_pos_rss is None
                or rss < best_pos_rss
                or (rss == best_pos_rss and order[idx] > order[best_idx])
            ):
                best_idx, best_pos_rss = idx, rss
        removed_term = order[best_idx]
        order = (
            order[:best_idx] + order[best_idx + 1 : pos] + [removed_term] + order[pos:]
        )
        offer(order)
    return best_set


def _prune_k2(B: np.ndarray, Y: np.ndarray) -> dict[int, frozenset[int]]:
    """PRUNE-3's K >= 2 rule (plain backward elimination): at each stage,
    remove the non-intercept term whose removal gives the lowest RSS, ties
    to the largest index. Self-check only, as `_prune_k1` above."""
    m_f = B.shape[1]
    kept = list(range(m_f))
    t: dict[int, frozenset[int]] = {m_f: frozenset(kept)}
    while len(kept) > 1:
        best_j, best_rss = None, None
        for j in kept:
            if j == 0:
                continue
            cols = [c for c in kept if c != j]
            rss = _rss_of_subset(B, Y, cols)
            if best_rss is None or rss < best_rss or (rss == best_rss and j > best_j):
                best_j, best_rss = j, rss
        kept.remove(best_j)
        t[len(kept)] = frozenset(kept)
    return t


@register_component
def pruning_fixed_basis() -> dict[str, Any]:
    """VALIDATION_PLAN.md, "Component tests" > "Pruning of a fixed basis":
    earth's own earth:::pruning.pass on a real forward basis (built with
    pmethod = "none", so there is no earth-side pruning to undo first), for
    one response and for several. T08 builds the same terms in pymars
    (from ``dirs``/``cuts``, TERM-3) and runs ``_pruning.py`` on them, so
    ``bx`` itself does not need to be in the fixture.

    Review round 1 (#42, adversarial finding 3): the original design (rng
    seed 11, y2/y3 noise sd 0.05) gave the same T[m] under PRUNE-3's K = 1
    and K >= 2 rules, so it could not catch an implementation that used the
    K = 1 rule for every K. Noise sd 0.3 on y2/y3 was the fix, but seed 102
    (round 1's choice) separates only the several-responses case; the
    one-response case's T[m] turned out nested there too (round 2,
    #42/#43 blocking finding 2, a regression from round 1's own fix for
    the other case). Seed 138 separates both: one response at sizes 5 to
    7 (RSS gap up to 1.9%) and three responses at sizes 5 to 7 (up to
    1.6%). The assertions below re-derive both rules from the spec
    (`_prune_k1`/`_prune_k2`) for *each* case and fail loudly if a future
    change stops separating either one.
    """
    rng = np.random.default_rng(138)
    n = 120
    X = np.column_stack([rng.uniform(0, 1, size=n), rng.uniform(0, 1, size=n)])
    y1 = (
        2 * np.maximum(0, X[:, 0] - 0.3)
        - 1.5 * np.maximum(0, 0.3 - X[:, 0])
        + 1.2 * np.maximum(0, X[:, 0] - 0.5) * np.maximum(0, X[:, 1] - 0.4)
        + rng.normal(scale=0.05, size=n)
    )
    y2 = 0.5 * y1 + rng.normal(scale=0.3, size=n)
    y3 = -0.3 * y1 + rng.normal(scale=0.3, size=n)
    y_multi = np.column_stack([y1, y2, y3])
    earth_args = {
        "degree": 2,
        "pmethod": "none",
        "nk": 15,
        "thresh": 0,
        "minspan": 1,
        "endspan": 1,
        "fast.k": 0,
        "Auto.linpreds": False,
    }
    penalty = 2.0

    def _diverge_at_sizes(
        bx: np.ndarray, Y: np.ndarray, *, case_label: str
    ) -> list[int]:
        t_k1 = _prune_k1(bx, Y)
        t_k2 = _prune_k2(bx, Y)
        diverges_at = sorted(m for m in t_k1 if t_k1[m] != t_k2[m])
        if not diverges_at:
            raise AssertionError(
                f"pruning_fixed_basis's {case_label} case no longer "
                "separates PRUNE-3's K=1 rule from its K>=2 rule (T[m] "
                "agree at every size); pick a different seed or noise level"
            )
        return diverges_at

    basis_one = blackbox.fit_bx_dirs(X, y1, earth_args)
    one_diverges_at = _diverge_at_sizes(
        basis_one["bx"], y1.reshape(-1, 1), case_label="one-response"
    )

    basis_multi = blackbox.fit_bx_dirs(X, y_multi, earth_args)
    multi_diverges_at = _diverge_at_sizes(
        basis_multi["bx"], y_multi, case_label="several-responses"
    )

    return {
        "X": X.tolist(),
        "earth_args": earth_args,
        "penalty": penalty,
        "one_response": {
            **_pruning_case(X, y1, earth_args, penalty),
            "k1_vs_k2_diverge_at_sizes": one_diverges_at,
        },
        "several_responses": {
            **_pruning_case(X, y_multi, earth_args, penalty),
            "k1_vs_k2_diverge_at_sizes": multi_diverges_at,
        },
    }


@register_component
def lm_fit_coefficients() -> dict[str, Any]:
    """VALIDATION_PLAN.md, "Component tests" > "Coefficients of fixed
    terms": pymars' weighted least-squares coefficients (LA-4, PRUNE-8) are
    compared with R's lm.fit/lm.wfit on the same columns.

    Review round 1 (#42 finding 5): besides the exact duplicate (any
    dependence tolerance from about 1e-15 to 0.9 passes that one case), two
    near-duplicate columns straddle LA-4's 1e-7 threshold on each side (an
    added-noise scale of 1e-6, ratio about 1.5e-6, clears it; 1e-8, ratio
    about 1.9e-8, does not); a fourth column (1000 + 1e-6 z, next to the
    intercept) is dependent under LA-4's uncentered norm (ratio about 1e-9)
    but not under a centered one (ratio about 0.99), the case bb09.9 notes;
    and a weighted case exercises PRUNE-8's ``lm.wfit`` path.
    """
    rng = np.random.default_rng(21)
    n = 80
    x0 = rng.uniform(0, 1, size=n)
    x1 = rng.uniform(0, 1, size=n)
    y = 1.5 + 2.0 * x0 - 0.7 * x1 + rng.normal(scale=0.1, size=n)

    def case(
        X: np.ndarray, label: str, *, w: np.ndarray | None = None
    ) -> dict[str, Any]:
        result = blackbox.lm_fit(X, y, w=w)
        coefficients = result["coefficients"].ravel().tolist()
        return {
            "label": label,
            "x": X.tolist(),
            "weights": None if w is None else w.tolist(),
            "coefficients": [None if c is None else float(c) for c in coefficients],
            "residuals": result["residuals"].ravel().tolist(),
            "rank": int(result["rank"]),
        }

    full_rank = np.column_stack([np.ones(n), x0, x1])
    duplicated = np.column_stack([np.ones(n), x0, x0])
    above_threshold = np.column_stack(
        [np.ones(n), x0, x1, x0 + 1e-6 * rng.normal(size=n)]
    )
    below_threshold = np.column_stack(
        [np.ones(n), x0, x1, x0 + 1e-8 * rng.normal(size=n)]
    )
    centered_vs_uncentered = np.column_stack(
        [np.ones(n), x0, x1, 1000.0 + 1e-6 * rng.normal(size=n)]
    )
    weights = rng.uniform(0.5, 3.0, size=n)
    return {
        "y": y.tolist(),
        "cases": [
            case(full_rank, "full_rank"),
            case(duplicated, "duplicated_column"),
            case(above_threshold, "near_duplicate_above_1e-7"),
            case(below_threshold, "near_duplicate_below_1e-7"),
            case(centered_vs_uncentered, "centered_vs_uncentered_norm"),
            case(full_rank, "weighted_full_rank", w=weights),
        ],
    }


def _predict_case(
    x: np.ndarray, y: np.ndarray, newx: np.ndarray, earth_args: dict[str, Any]
) -> dict[str, Any]:
    result = blackbox.predict_earth(x, y, newx, earth_args)
    return {
        "x": x.tolist(),
        "y": y.tolist(),
        "earth_args": earth_args,
        "newx": newx.tolist(),
        "pred": result["pred"].tolist(),
        "dirs": result["dirs"].tolist(),
        "cuts": result["cuts"].tolist(),
        "selected_terms": result["selected_terms"].tolist(),
        "coefficients": result["coefficients"].tolist(),
    }


@register_component
def predict_new_points() -> dict[str, Any]:
    """VALIDATION_PLAN.md, "Component tests" > "Prediction at new points":
    the basis evaluated at points outside the training range (TERM-3:
    "nothing is clipped"), compared with predict.earth; degree 1 (one
    covariate) and degree 2 (an interaction).

    Review round 1 (#42 finding 2): each case stores earth's own dirs,
    cuts, selected_terms and coefficients alongside pred, so a failure
    points at TERM-3 alone, not at a whole forward-pass refit. The
    ``linear_auto``/``linear_hinge`` pair is FWD-6: with an exactly linear
    truth (so a knot's RSS reduction ties the linear candidate's, and
    FWD-5 breaks that tie for the linear candidate), ``Auto.linpreds =
    TRUE`` selects the true linear term (code 2, extrapolating as a
    straight line) and ``Auto.linpreds = FALSE`` selects a hinge at the
    training minimum (code +1, extrapolating as a constant below it); the
    two agree inside the training range and diverge only below the
    minimum, which ``newx`` includes.
    """
    rng = np.random.default_rng(31)
    n = 100
    x = rng.uniform(0, 1, size=(n, 1))
    y = 2 * np.maximum(0, x[:, 0] - 0.4) + rng.normal(scale=0.05, size=n)
    earth_args_d1 = {"degree": 1, "nk": 11, "thresh": 0}
    newx_d1 = np.array([[-2.0], [-0.5], [0.0], [0.4], [0.9], [1.0], [1.5], [3.0]])

    rng2 = np.random.default_rng(32)
    n2 = 150
    x2 = rng2.uniform(0, 1, size=(n2, 2))
    y2 = 2 * np.maximum(0, x2[:, 0] - 0.3) * np.maximum(
        0, x2[:, 1] - 0.5
    ) + rng2.normal(scale=0.05, size=n2)
    earth_args_d2 = {"degree": 2, "nk": 15, "thresh": 0}
    newx_d2 = np.array([[-1.0, -1.0], [0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [2.0, 2.0]])

    rng3 = np.random.default_rng(33)
    n3 = 100
    x3 = rng3.uniform(0, 1, size=(n3, 1))
    y3 = 3.0 * x3[:, 0] - 1.0  # exactly linear: no noise, so every knot ties the
    # linear candidate's RSS reduction exactly (any mirrored hinge pair, or a
    # single hinge plus the intercept, reproduces a line identically), and
    # FWD-5 then favors the linear candidate on that tie.
    newx_linear = np.array(
        [[-2.0], [-0.5], [0.0], [0.5], [1.0], [1.5], [3.0]]
    )  # below, at and above the training minimum (about 0.0154)
    earth_args_linear_auto = {"degree": 1, "nk": 3, "thresh": 0, "Auto.linpreds": True}
    earth_args_linear_hinge = {
        "degree": 1,
        "nk": 3,
        "thresh": 0,
        "Auto.linpreds": False,
    }

    return {
        "degree1": _predict_case(x, y, newx_d1, earth_args_d1),
        "degree2": _predict_case(x2, y2, newx_d2, earth_args_d2),
        "linear_auto": _predict_case(x3, y3, newx_linear, earth_args_linear_auto),
        "linear_hinge": _predict_case(x3, y3, newx_linear, earth_args_linear_hinge),
    }


def _assert_multinom_is_stable(
    X: np.ndarray, y: np.ndarray, fit: dict[str, Any], *, label: str
) -> None:
    """Review round 2 (#42/#43 blocking finding 1): nnet's convergence code
    is 0 whenever the objective stopped changing by more than ``reltol``
    between iterations, which is not the same claim as "reached GLM-2's
    minimum". At nnet's own default reltol (1e-8) the stored coefficients
    were off by up to 8.7e-4 relative, which GLM-3's tolerance (``compare.
    GLM_COEF_REL`` = 1e-5 relative, ``compare.GLM_PROB_ABS`` = 1e-7
    absolute) cannot absorb. ``multinom_fit``'s new default (reltol =
    1e-15) is tight, but tight is not proof by itself: refit once more at
    a distinctly different reltol and require the two fits to already
    agree well inside GLM-3's tolerance (checked at a tenth of it, 1e-6
    relative and 1e-7 absolute: classifier_refit's simpler design agrees
    to 0 at this level, and #43's harder s18_multinom design to 1.6e-8
    relative and 7e-9 absolute, both comfortably inside). If they don't,
    the first fit was not close enough to the minimum to be a valid
    reference.
    """
    check = blackbox.multinom_fit(X, y, maxit=20000, reltol=1e-12)
    if check["convergence"] != 0:
        raise AssertionError(f"{label}: the stability refit did not converge")
    coef_rel = np.max(
        np.abs(fit["coefficients"] - check["coefficients"])
        / (np.abs(check["coefficients"]) + 1e-300)
    )
    prob_abs = np.max(np.abs(fit["fitted"] - check["fitted"]))
    if coef_rel > 1e-6 or prob_abs > 1e-7:
        raise AssertionError(
            f"{label}: multinom fit is not stable enough to be a GLM-3 "
            f"reference (coefficients differ by {coef_rel:.2e} relative, "
            f"probabilities by {prob_abs:.2e})"
        )


@register_component
def classifier_refit() -> dict[str, Any]:
    """VALIDATION_PLAN.md, "Binary outcomes": R's glm (binomial) and
    nnet::multinom on fixed columns, the reference for EarthClassifier's
    GLM-refit coefficients and fitted probabilities (binary and, for three
    or more classes, the multinom comparison the plan calls for).

    Review round 1 (#42 finding 1): the multinomial case's labels are now
    drawn from the softmax probabilities of a linear score (not a
    threshold rule), so the classes overlap and a finite MLE exists;
    nnet's convergence code is checked (0 required) and stored, since GLM-
    3/GLM-4 only promise agreement where the reference itself converged.
    """
    rng = np.random.default_rng(41)
    n = 300
    x0 = rng.uniform(-2, 2, size=n)
    x1 = rng.uniform(-2, 2, size=n)
    X = np.column_stack([np.ones(n), x0, x1])
    eta = 0.8 * x0 - 0.5 * x1
    p = 1.0 / (1.0 + np.exp(-eta))
    y_binary = (rng.uniform(size=n) < p).astype(float)
    glm = blackbox.glm_fit(X, y_binary, family="binomial")
    if not glm["converged"] or glm["warnings"]:
        raise AssertionError(
            f"classifier_refit's binomial case did not give a clean GLM-3 "
            f"reference: converged={glm['converged']}, "
            f"warnings={glm['warnings']}"
        )

    rng2 = np.random.default_rng(42)
    n2 = 300
    z0 = rng2.uniform(-3, 3, size=n2)
    z1 = rng2.uniform(-3, 3, size=n2)
    Xm = np.column_stack([np.ones(n2), z0, z1])
    score = z0 - 0.5 * z1
    scores = np.column_stack([np.zeros(n2), score, -score])  # mid, hi, lo
    probs = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    levels = np.array(["mid", "hi", "lo"])
    y3 = np.array([levels[rng2.choice(3, p=probs[i])] for i in range(n2)])
    multinom = blackbox.multinom_fit(Xm, y3)
    if multinom["convergence"] != 0 or multinom["warnings"]:
        raise AssertionError(
            f"classifier_refit's multinomial case did not converge: "
            f"convergence={multinom['convergence']}, "
            f"warnings={multinom['warnings']}"
        )
    _assert_multinom_is_stable(Xm, y3, multinom, label="classifier_refit")

    return {
        "binomial": {
            "x": X.tolist(),
            "y": y_binary.tolist(),
            "coefficients": [
                None if c is None else float(c)
                for c in glm["coefficients"].ravel().tolist()
            ],
            "fitted_values": glm["fitted_values"].tolist(),
            "converged": glm["converged"],
        },
        "multinomial": {
            "x": Xm.tolist(),
            "y": y3.tolist(),
            "coefficients": multinom["coefficients"].tolist(),
            "levels": multinom["levels"],
            "fitted": multinom["fitted"].tolist(),
            "convergence": multinom["convergence"],
        },
    }


# Candidate knot sets (VALIDATION_PLAN.md, "Component tests" > "Candidate
# knot sets"): the grid is minspan automatic/1/3/10, endspan automatic/1/5,
# degree 1 and 2, Adjust.endspan 1 and 2 (degree 2 only: SPAN-4 says it does
# not affect an intercept parent), and 20, 200 and 2,000 cases. A negative
# minspan is a `later` issue (T05 brief), so it is left out. Trimmed for
# fixture size (validation/README.md, "keep validation/fixtures/ small"):
# every visited case prints one trace = 9 line whether evaluated or
# span-skipped, so trace size scales with n regardless of minspan, and the
# full (minspan, endspan, Adjust.endspan) cross is the same combinatorial
# scan at any n; so it runs once, at the cheap n = 20, and n = 200 and
# 2,000 (which mainly test the automatic formulas' own dependence on n, the
# one thing n = 20 cannot) run at automatic minspan and endspan alone, still
# crossed with Adjust.endspan at degree 2. nk = 5 (at most 2 forward steps)
# keeps every combo to the searches the knot-set tests need.
_KNOT_MINSPANS: tuple[int | None, ...] = (None, 1, 3, 10)
_KNOT_ENDSPANS: tuple[int | None, ...] = (None, 1, 5)
_KNOT_ADJUST_ENDSPANS: tuple[float, ...] = (1.0, 2.0)
_KNOT_N_FULL = 20
_KNOT_N_REDUCED: tuple[int, ...] = (200, 2000)


def _knot_grid_xy(
    n: int, p: int, seed: int, *, low: float = 0.0, high: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(low, high, size=(n, p))
    mid = (low + high) / 2
    y = 2 * np.maximum(0, X[:, 0] - mid)
    if p > 1:
        y = y + 1.5 * np.maximum(0, X[:, 1] - (mid + 0.2 * (high - low)))
    y = y + rng.normal(scale=0.02 * (high - low), size=n)
    return X, y


def _knot_jobs() -> list[tuple[dict[str, Any], driver.EarthJob]]:
    """Every (metadata, EarthJob) pair of the candidate-knot-set grid."""
    jobs: list[tuple[dict[str, Any], driver.EarthJob]] = []
    counter = 0

    def add(
        degree: int,
        minspan: int | None,
        endspan: int | None,
        adjust_endspan: float,
        n: int,
        *,
        low: float = 0.0,
        high: float = 1.0,
        auto_linpreds: bool = False,
    ) -> None:
        nonlocal counter
        p = 1 if degree == 1 else 2
        X, y = _knot_grid_xy(n, p, seed=9_000 + counter, low=low, high=high)
        earth_args = {
            "degree": degree,
            "nk": 5,
            "thresh": 0,
            "pmethod": "none",
            "minspan": minspan if minspan is not None else 0,
            "endspan": endspan if endspan is not None else 0,
            "Adjust.endspan": adjust_endspan,
            "fast.k": 0,
            "Auto.linpreds": auto_linpreds,
        }
        job_id = f"knot_{counter:03d}"
        meta = {
            "degree": degree,
            "minspan": minspan,
            "endspan": endspan,
            "adjust_endspan": adjust_endspan,
            "n": n,
            "p": p,
        }
        job = driver.EarthJob(
            id=job_id,
            X=X,
            y=y,
            earth_args=earth_args,
            trace=9,
            include_forward_path=False,
        )
        jobs.append((meta, job))
        counter += 1

    for degree in (1, 2):
        adjust_values = _KNOT_ADJUST_ENDSPANS if degree == 2 else (2.0,)
        for minspan in _KNOT_MINSPANS:
            for endspan in _KNOT_ENDSPANS:
                for adjust_endspan in adjust_values:
                    add(degree, minspan, endspan, adjust_endspan, _KNOT_N_FULL)
        for adjust_endspan in adjust_values:
            for n in _KNOT_N_REDUCED:
                add(degree, None, None, adjust_endspan, n)

    # Review round 1 (#42, spec finding 4): at n = 20 the SPAN-5 cap (9)
    # hides Adjust.endspan's effect (SPAN-4) whenever the adjusted endspan
    # would exceed it, which endspan = 5 and the automatic endspan both do
    # there; one endspan = 5 case per Adjust.endspan value at the larger
    # n = 200 (automatic minspan) shows it uncapped.
    for adjust_endspan in _KNOT_ADJUST_ENDSPANS:
        add(2, None, 5, adjust_endspan, 200)

    # Also review round 1 (#42, spec finding 4): no case has negative x or
    # a linear term. x on [-1, 1] with Auto.linpreds = True exercises
    # KNOT-1's rule for a linear parent's negative values (a case is
    # inactive there) whenever earth's forward search happens to pick a
    # linear term as the first-step parent; several designs meant to force
    # that choice (a dominant, near-noiseless linear component) still had
    # earth's pair search prefer a mirrored hinge pair over the tied linear
    # candidate, unlike the single-covariate case, so this is left to
    # record whatever earth actually does rather than to force one
    # structure (T05 brief: no interpreting earth's internals beyond the
    # spec; this asymmetry is flagged in the pull request for the spec
    # writer, not resolved here by trial and error).
    add(2, None, None, 2.0, 200, low=-1.0, high=1.0, auto_linpreds=True)
    return jobs


@register_component
def knot_candidates() -> dict[str, Any]:
    """VALIDATION_PLAN.md, "Component tests" > "Candidate knot sets": earth's
    trace = 9 case-by-case knot scan, across the grid above, for T08's
    `_knots.py` candidate-value tests to compare against. This only records
    what earth printed (a black-box trace, COMMON.md's clean room); the
    comparison, and any interpretation of which cases earth accepted or
    rejected, is T08's (T05 brief: "you do not interpret earth's internals
    beyond what the spec states")."""
    metas_jobs = _knot_jobs()
    metas = [m for m, _ in metas_jobs]
    jobs = [j for _, j in metas_jobs]
    cases = []
    with tempfile.TemporaryDirectory(prefix="pymars-knot-grid-") as tmp:
        workdir = Path(tmp)
        results = driver.run_earth(jobs, workdir=workdir)
        for meta, job in zip(metas, jobs, strict=True):
            result = results[job.id]
            trace_text = (workdir / f"{job.id}_trace.txt").read_text(encoding="utf-8")
            trace_parse.parse_trace(  # fail loudly here, not later in T08
                workdir / f"{job.id}_trace.txt"
            )
            cases.append(
                {
                    **meta,
                    "X": job.X.tolist(),
                    "y": job.y.tolist(),
                    "earth_args": dict(job.earth_args),
                    "dirs": result["dirs"],
                    "cuts": result["cuts"],
                    "selected_terms": result["selected_terms"],
                    "trace_text": trace_text,
                }
            )
    return {"cases": cases}


def _sanitize_versions(versions: dict[str, Any]) -> dict[str, Any]:
    """Drop the parts of ``driver.versions()`` that vary by machine or by
    commit without saying anything about earth conformance: an absolute
    path (threadpoolctl's and R's BLAS ``filepath``/``la_library`` include
    the machine's home folder, which also does not belong in a public
    repository) becomes just its basename, and ``pymars_commit`` is left
    out of the fixture entirely (``check()`` reports it separately; a
    fixture regenerated on a later commit always differs there, by
    design).
    """
    out = dict(versions)
    out.pop("pymars_commit", None)
    if out.get("blas"):
        out["blas"] = [
            {
                **lib,
                "filepath": Path(lib["filepath"]).name if lib.get("filepath") else None,
            }
            for lib in out["blas"]
        ]
    if out.get("r_blas"):
        out["r_blas"] = {
            key: (Path(value).name if value else None)
            for key, value in out["r_blas"].items()
        }
    return out


def _fixture_payload(dataset_id: str, mode: str) -> dict[str, Any]:
    ds = REGISTRY[dataset_id]()
    earth_args = MODES[mode]
    job = driver.EarthJob(
        id=f"{dataset_id}_{mode}",
        X=ds.X,
        y=ds.y,
        X_test=ds.X_test,
        earth_args=earth_args,
        factor_response=ds.factor_response,
    )
    with tempfile.TemporaryDirectory(prefix="pymars-fixture-") as tmp:
        result = driver.run_earth([job], workdir=Path(tmp))[job.id]
    return {
        "dataset": dataset_id,
        "mode": mode,
        "inputs": {
            "X": ds.X.tolist(),
            "y": ds.y.tolist(),
            "X_test": ds.X_test.tolist() if ds.X_test is not None else None,
        },
        "earth_args": earth_args,
        "result": result,
        "versions": _sanitize_versions(driver.versions(result)),
    }


def _component_payload(name: str) -> dict[str, Any]:
    payload = COMPONENT_REGISTRY[name]()
    versions = driver.versions()
    versions.update(blackbox.versions())
    # Review round 1 (#42 finding, adversarial, non-blocking): a component
    # fixture's r_version/earth_version come from blackbox.versions() above,
    # so the driver.versions() call above has no earth_result, and
    # driver.versions() only fills in r_blas when given one. Reuse driver's
    # own R-BLAS probe directly rather than adding an earth_result-free
    # code path to driver.versions() itself (a harness file): this reads
    # one existing, side-effect-free helper, not a rewrite.
    versions["r_blas"] = driver._r_blas()
    return {"component": name, **payload, "versions": _sanitize_versions(versions)}


def _write_component(name: str, components_dir: Path) -> Path:
    payload = _component_payload(name)
    components_dir.mkdir(parents=True, exist_ok=True)
    path = components_dir / f"{name}.json"
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path


def make_all_components(*, fixtures_dir: Path | None = None) -> list[Path]:
    """Write every registered component fixture (``validation/fixtures/
    components/<name>.json``) and return their paths, in registration
    order. ``fixtures_dir`` is the datasets' own root (its ``components``
    subfolder is where these go), read the same dynamic way ``make_all``
    reads its own default, for the same monkeypatching reason."""
    fixtures_dir = fixtures_dir if fixtures_dir is not None else FIXTURES_DIR
    components_dir = fixtures_dir / "components"
    return [_write_component(name, components_dir) for name in COMPONENT_REGISTRY]


def _write_fixture(dataset_id: str, mode: str, fixtures_dir: Path) -> Path:
    payload = _fixture_payload(dataset_id, mode)
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    path = fixtures_dir / f"{dataset_id}_{mode}.json"
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path


def make_all(*, fixtures_dir: Path | None = None) -> list[Path]:
    """Write every registered (dataset, mode) fixture and return their
    paths, in registration order.

    ``fixtures_dir`` defaults to the module-level ``FIXTURES_DIR``, read
    inside this function (not bound as a default parameter value), so
    ``monkeypatch.setattr(gen_fixtures, "FIXTURES_DIR", ...)`` in a test
    changes what a plain ``make_all()`` call does.
    """
    fixtures_dir = fixtures_dir if fixtures_dir is not None else FIXTURES_DIR
    return [
        _write_fixture(dataset_id, mode, fixtures_dir)
        for dataset_id in REGISTRY
        for mode in MODES
    ]


def check(*, fixtures_dir: Path | None = None) -> CheckResult:
    """Regenerate every dataset and component fixture into a temporary
    folder and compare each one against ``fixtures_dir`` (default: the
    module-level ``FIXTURES_DIR``, read dynamically the same way
    ``make_all`` does).

    Every key but ``versions`` (what a fixture claims about earth: for a
    dataset, ``dataset``, ``mode``, ``inputs``, ``earth_args`` and
    ``result``; for a component, ``component`` and its own payload) or a
    fixture missing outright can add to ``.problems``; a ``versions``
    difference (this machine's R, earth, numpy, scikit-learn or BLAS is not
    the one that made the committed fixture) instead adds to ``.notes``,
    since ``_sanitize_versions`` already drops the one field
    (``pymars_commit``) that would differ on every run from a different
    commit by construction, and nothing about earth conformance follows
    from the rest of ``versions`` differing.

    Also reports (as a problem) a committed ``*.json`` under
    ``fixtures_dir`` or ``fixtures_dir/components`` that no generator
    wrote (review round 1, #42 adversarial finding 6): a stale file, for
    example left behind by a renamed dataset or component, would
    otherwise never be caught, since the loop above only ever compares
    fixtures a generator actually produced.
    """
    fixtures_dir = fixtures_dir if fixtures_dir is not None else FIXTURES_DIR
    problems: list[str] = []
    notes: list[str] = []
    fresh_relative: set[Path] = set()
    with tempfile.TemporaryDirectory(prefix="pymars-fixture-check-") as tmp:
        tmp = Path(tmp)
        fresh_paths = make_all(fixtures_dir=tmp) + make_all_components(fixtures_dir=tmp)
        for fresh in fresh_paths:
            relative = fresh.relative_to(tmp)
            fresh_relative.add(relative)
            committed = fixtures_dir / relative
            if not committed.is_file():
                problems.append(f"{relative}: missing from {fixtures_dir}")
                continue
            fresh_payload = json.loads(fresh.read_text(encoding="utf-8"))
            committed_payload = json.loads(committed.read_text(encoding="utf-8"))
            fresh_checked = {k: v for k, v in fresh_payload.items() if k != "versions"}
            committed_checked = {
                k: v for k, v in committed_payload.items() if k != "versions"
            }
            if fresh_checked != committed_checked:
                problems.append(f"{relative}: differs from the committed fixture")
            elif fresh_payload.get("versions") != committed_payload.get("versions"):
                notes.append(f"{relative}: versions differ")

    committed_relative: set[Path] = set()
    if fixtures_dir.is_dir():
        committed_relative |= {
            p.relative_to(fixtures_dir) for p in fixtures_dir.glob("*.json")
        }
        components_dir = fixtures_dir / "components"
        if components_dir.is_dir():
            committed_relative |= {
                p.relative_to(fixtures_dir) for p in components_dir.glob("*.json")
            }
    for stale in sorted(committed_relative - fresh_relative):
        problems.append(f"{stale}: committed but no generator writes it")
    return CheckResult(problems=problems, notes=notes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="remake every fixture in a temp folder and report any diff",
    )
    args = parser.parse_args(argv)

    if args.check:
        result = check()
        for message in result.notes:
            print(message, file=sys.stderr)
        for message in result.problems:
            print(message, file=sys.stderr)
        if result.problems:
            print(
                f"{len(result.problems)} fixture(s) did not reproduce exactly",
                file=sys.stderr,
            )
            return 1
        print("every fixture reproduced exactly")
        return 0

    paths = make_all() + make_all_components()
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
