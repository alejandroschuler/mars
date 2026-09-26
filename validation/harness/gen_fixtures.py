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
    weights: np.ndarray | None = None
    glm_family: str | None = None
    scale_override: np.ndarray | None = None
    """The LA-7 scale to use in place of ``scaled_matrix(X, weights)``'s
    own (review round 1, #43 finding 5): an integer-weight dataset and its
    repeated-row companion must divide by the *same* scale, computed once
    from the repeated rows, or the weighted formula and the unweighted-
    on-repeated-rows formula can differ in the last bit and move some of
    earth's knots off the weighted fixture's own X values. Set by
    `_shared_scale_from_repeated_rows` on both halves of such a pair."""


DatasetFn = Callable[[], Dataset]
REGISTRY: dict[str, DatasetFn] = {}

# earth argument sets ("modes"), VALIDATION_PLAN.md's "Comparison modes":
# defaults_d1 is earth at its own defaults; matched_d1 is the legacy-code
# matched mode from the seed prototype (validation/legacy/compare_earth.py),
# hinge-only and every case a candidate knot, so the two implementations
# should make the same forward choices except at near-ties. The other
# entries are degree, span-grid, linear-candidate and weighted variants of
# these two that individual S02-S20 datasets opt into (DATASET_MODES).
MODES: dict[str, dict[str, Any]] = {
    "defaults_d1": {"degree": 1},
    "defaults_d2": {"degree": 2},
    "defaults_d3": {"degree": 3},
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
MODES["matched_d2"] = {**MODES["matched_d1"], "degree": 2}
MODES["matched_d3"] = {**MODES["matched_d1"], "degree": 3}
# S03/S07 ("Linear terms"): matched but Auto.linpreds = TRUE, so a true
# linear covariate can be picked directly instead of approximated by a
# mirrored hinge pair.
MODES["matched_d1_linear"] = {**MODES["matched_d1"], "Auto.linpreds": True}
# S08 ("Ties"): matched but minspan = 5, so the candidate knots differ from
# matched_d1's (minspan = 1) by design; S08 measures that difference.
MODES["matched_d1_minspan5"] = {**MODES["matched_d1"], "minspan": 5}
# S01/S02 ("Knot recovery"/"Span formulas ... at small n"): the minspan x
# endspan grid (1, 5, automatic) x (1, 10, automatic), degree 1 throughout.
# Automatic is the plain absence of the key (earth's own default, 0).
for _minspan_label, _minspan in (("1", 1), ("5", 5), ("auto", None)):
    for _endspan_label, _endspan in (("1", 1), ("10", 10), ("auto", None)):
        _args: dict[str, Any] = {"degree": 1}
        if _minspan is not None:
            _args["minspan"] = _minspan
        if _endspan is not None:
            _args["endspan"] = _endspan
        MODES[f"span_m{_minspan_label}_e{_endspan_label}"] = _args
SPAN_GRID_MODES: tuple[str, ...] = tuple(k for k in MODES if k.startswith("span_"))
# S12 ("Invariance to scale and shift"): earth's own defaults, but the
# fixture generator must not apply the LA-7 harness rescaling to it (below)
# -- the whole point is earth's raw, unscaled behavior. RAW_MODES names
# every mode that must bypass that step.
MODES["raw_d1"] = {"degree": 1}
RAW_MODES: frozenset[str] = frozenset({"raw_d1"})

DEFAULT_DATASET_MODES: tuple[str, ...] = ("defaults_d1", "matched_d1")
# Which modes each dataset is generated under; a dataset not listed here
# uses DEFAULT_DATASET_MODES. Filled in as S02-S20 are registered below.
DATASET_MODES: dict[str, tuple[str, ...]] = {}


def register(fn: DatasetFn) -> DatasetFn:
    """Decorator: register a dataset generator under its own return value's
    ``id`` (calling it once, at import time, to read that id)."""
    REGISTRY[fn().id] = fn
    return fn


def scaled_matrix(
    X: np.ndarray, w: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """The LA-7 harness step (docs/algorithm.md, "Linear algebra contract";
    VALIDATION_PLAN.md, "Comparison modes"): each non-constant covariate
    divided by its weighted standard deviation (divisor N = sum(w), not
    centered); a constant column (every value equal over the CASES OF THE
    FIT, W-3: the rows with positive weight, the Conventions section's
    exact-equality test) stays as it is. Returns ``(X_scaled, scale)``,
    where ``scale[j]`` is column j's divisor (1.0 for a constant column),
    so a caller can apply the same divisors to a test matrix (``X_test /
    scale``) instead of rescaling it independently.

    Review round 1 (#43 adversarial finding 8): the constancy check now
    looks only at the rows with positive weight; a column constant there
    but not on a dropped zero-weight row no longer gets scale 0 (and the
    column, inf).
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    active = w > 0
    scale = np.ones(p)
    X_scaled = X.copy()
    for j in range(p):
        col = X[:, j]
        col_active = col[active]
        if col_active.size == 0 or np.all(col_active == col_active[0]):
            continue  # constant column (over the cases of the fit): stays as it is
        mean = np.average(col, weights=w)
        variance = np.average((col - mean) ** 2, weights=w)  # divisor N
        scale[j] = np.sqrt(variance)
        X_scaled[:, j] = col / scale[j]
    return X_scaled, scale


def _shared_scale_from_repeated_rows(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The LA-7 scale for an integer-weight dataset and its repeated-row
    companion, computed once, from the repeated rows, and meant to be
    used by both (review round 1, #43 finding 5: LA-7 says the scale for
    such a pair comes "from the repeated rows", and computing it twice --
    once by the weighted formula, once by np.repeat plus the unweighted
    formula -- gives the same value in exact arithmetic but not always in
    float64, which moves some of earth's knots off the weighted fixture's
    own X values)."""
    w_int = np.asarray(w).astype(int)
    idx = np.repeat(np.arange(len(w_int)), w_int)
    _, scale = scaled_matrix(X[idx])
    return scale


def _short_test_set(
    X: np.ndarray, rng: np.random.Generator, n_test: int = 12
) -> np.ndarray:
    """A short, deterministic test set for a dataset's covariates (review
    round 1, #43 finding 7: the harness collects "the predictions on a
    test set", and the tolerance table compares predictions on new data,
    for every dataset, not S01 alone): mostly fresh draws from the same
    box as X (its own per-column min/max), plus a few points outside that
    box on each side, so predictions can be compared there too (LA-7
    scales a test matrix by the training scale, not its own)."""
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    span = hi - lo
    n_in = max(n_test - 4, 1)
    x_in = lo + rng.uniform(size=(n_in, X.shape[1])) * span
    x_out = np.stack(
        [
            lo - 0.2 * span,
            hi + 0.2 * span,
            lo - 0.5 * span,
            hi + 0.5 * span,
        ]
    )
    return np.vstack([x_in, x_out])


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


DATASET_MODES["S01"] = (*DEFAULT_DATASET_MODES, *SPAN_GRID_MODES)


def _s01_f(x0: np.ndarray) -> np.ndarray:
    return 2 * np.maximum(0, x0 - 0.3) - 3 * np.maximum(0, x0 - 0.7)


def _s02(n: int, seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=(n, 1))
    y = _s01_f(x[:, 0]) + rng.normal(scale=0.1, size=n)
    x_test = _short_test_set(x, rng)
    return Dataset(id=f"S02_n{n:03d}", X=x, y=y, X_test=x_test)


@register
def s02_n020() -> Dataset:
    """S02: S01 with 20 cases. Span formulas and stopping rules at small n."""
    return _s02(20, seed=201)


@register
def s02_n050() -> Dataset:
    """S02: S01 with 50 cases. Span formulas and stopping rules at small n."""
    return _s02(50, seed=202)


DATASET_MODES["S02_n020"] = (*DEFAULT_DATASET_MODES, *SPAN_GRID_MODES)
DATASET_MODES["S02_n050"] = (*DEFAULT_DATASET_MODES, *SPAN_GRID_MODES)


@register
def s03() -> Dataset:
    """S03: three covariates, (x1)+, |x2| (two mirrored hinges) and a linear
    term in x3. Pairs against single hinges; linear terms."""
    rng = np.random.default_rng(3)
    n = 300
    X = rng.uniform(-1, 1, size=(n, 3))
    y = (
        2.0 * np.maximum(0, X[:, 0])
        + 1.5 * np.abs(X[:, 1])
        + 3.0 * X[:, 2]
        + rng.normal(scale=0.1, size=n)
    )
    X_test = _short_test_set(X, rng)
    return Dataset(id="S03", X=X, y=y, X_test=X_test)


DATASET_MODES["S03"] = (*DEFAULT_DATASET_MODES, "matched_d1_linear")


def _friedman1(X: np.ndarray) -> np.ndarray:
    """Friedman's #1 function [F91 §8, "simulated data"], p >= 5: the first
    5 covariates matter, the rest are noise by construction."""
    return (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + 5 * X[:, 4]
    )


def _s04(p: int, n: int, seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, p))
    y = _friedman1(X) + rng.normal(scale=1.0, size=n)
    X_test = _short_test_set(X, rng)
    return Dataset(id=f"S04_p{p:02d}_n{n:04d}", X=X, y=y, X_test=X_test)


@register
def s04_p05_n0200() -> Dataset:
    """S04: Friedman #1, 5 covariates, 200 cases. Interactions and
    irrelevant covariates, degree 1 and 2."""
    return _s04(5, 200, seed=4051)


@register
def s04_p05_n1000() -> Dataset:
    """S04: Friedman #1, 5 covariates, 1,000 cases."""
    return _s04(5, 1000, seed=4052)


@register
def s04_p10_n0200() -> Dataset:
    """S04: Friedman #1, 10 covariates (5 irrelevant), 200 cases."""
    return _s04(10, 200, seed=4101)


@register
def s04_p10_n1000() -> Dataset:
    """S04: Friedman #1, 10 covariates (5 irrelevant), 1,000 cases."""
    return _s04(10, 1000, seed=4102)


_S04_MODES = ("defaults_d1", "defaults_d2", "matched_d1", "matched_d2")
for _s04_id in ("S04_p05_n0200", "S04_p05_n1000", "S04_p10_n0200", "S04_p10_n1000"):
    DATASET_MODES[_s04_id] = _S04_MODES


@register
def s05() -> Dataset:
    """S05: a pure degree-2 hinge product, (x0 - 0.3)+ * (x1 - 0.6)+.
    Interaction search; Adjust.endspan."""
    rng = np.random.default_rng(5)
    n = 300
    X = rng.uniform(0, 1, size=(n, 2))
    y = 3.0 * np.maximum(0, X[:, 0] - 0.3) * np.maximum(0, X[:, 1] - 0.6) + rng.normal(
        scale=0.05, size=n
    )
    X_test = _short_test_set(X, rng)
    return Dataset(id="S05", X=X, y=y, X_test=X_test)


# Review round 1 (#43, adversarial finding 10, non-blocking): both of
# S05's modes used the default Adjust.endspan (2); matched_d2_adjust1
# tests SPAN-4 on a whole fit with a different value.
MODES["matched_d2_adjust1"] = {**MODES["matched_d2"], "Adjust.endspan": 1}
DATASET_MODES["S05"] = ("defaults_d2", "matched_d2", "matched_d2_adjust1")


@register
def s06() -> Dataset:
    """S06: a degree-3 product, (x0-0.3)+ * (x1-0.5)+ * (x2-0.7)+.
    Degree 3."""
    rng = np.random.default_rng(6)
    n = 400
    X = rng.uniform(0, 1, size=(n, 3))
    y = 4.0 * np.maximum(0, X[:, 0] - 0.3) * np.maximum(0, X[:, 1] - 0.5) * np.maximum(
        0, X[:, 2] - 0.7
    ) + rng.normal(scale=0.05, size=n)
    X_test = _short_test_set(X, rng)
    return Dataset(id="S06", X=X, y=y, X_test=X_test)


DATASET_MODES["S06"] = ("defaults_d3", "matched_d3")


@register
def s07() -> Dataset:
    """S07: a linear truth, y = 2*x0 - 1.5*x1 + noise, no true hinges.
    Auto.linpreds against pymars' linear candidates."""
    rng = np.random.default_rng(7)
    n = 250
    X = rng.uniform(-1, 1, size=(n, 2))
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + rng.normal(scale=0.1, size=n)
    X_test = _short_test_set(X, rng)
    return Dataset(id="S07", X=X, y=y, X_test=X_test)


DATASET_MODES["S07"] = ("defaults_d1", "matched_d1", "matched_d1_linear")


@register
def s08() -> Dataset:
    """S08: an integer covariate with 10 levels (0 to 9), 150 cases, so
    most values repeat many times. Repeated x values: distinct values
    against cases."""
    rng = np.random.default_rng(8)
    n = 150
    x0 = rng.integers(0, 10, size=n).astype(float)
    y = 2.0 * np.maximum(0, x0 - 4.0) + rng.normal(scale=0.2, size=n)
    X = x0.reshape(-1, 1)
    X_test = _short_test_set(X, rng)
    return Dataset(id="S08", X=X, y=y, X_test=X_test)


# "Ties": with minspan = 1 the candidate knots at a distinct value are the
# same in both programs; matched_d1_minspan5 (minspan = 5) is where they
# differ by design, which is what S08 measures.
DATASET_MODES["S08"] = ("defaults_d1", "matched_d1", "matched_d1_minspan5")


@register
def s09() -> Dataset:
    """S09: a 0/1 covariate and a 4-level categorical covariate, given to
    pymars 2.0 as one-hot dummies (OneHotEncoder(drop="first")): 3 dummy
    columns, "a" the baseline level. Categorical coding through
    OneHotEncoder."""
    rng = np.random.default_rng(9)
    n = 300
    binary = rng.integers(0, 2, size=n).astype(float)
    levels = np.array(["a", "b", "c", "d"])
    category = rng.choice(levels, size=n)
    dummy_b = (category == "b").astype(float)
    dummy_c = (category == "c").astype(float)
    dummy_d = (category == "d").astype(float)
    level_effect = {"a": 0.0, "b": 1.0, "c": -1.5, "d": 2.5}
    y = (
        1.0 * binary
        + np.array([level_effect[lvl] for lvl in category])
        + rng.normal(scale=0.2, size=n)
    )
    X = np.column_stack([binary, dummy_b, dummy_c, dummy_d])
    X_test = _short_test_set(X, rng)
    return Dataset(id="S09", X=X, y=y, X_test=X_test)


DATASET_MODES["S09"] = DEFAULT_DATASET_MODES


@register
def s10() -> Dataset:
    """S10: a duplicated column, a near-duplicate (x0 plus noise with
    standard deviation 1e-9), and a constant column. Tie-breaks across
    predictors; collinearity."""
    rng = np.random.default_rng(10)
    n = 200
    x0 = rng.uniform(0, 1, size=n)
    duplicate = x0.copy()
    near_duplicate = x0 + rng.normal(scale=1e-9, size=n)
    constant = np.full(n, 5.0)
    y = 2.0 * np.maximum(0, x0 - 0.4) + rng.normal(scale=0.05, size=n)
    X = np.column_stack([x0, duplicate, near_duplicate, constant])
    X_test = _short_test_set(X, rng)
    return Dataset(id="S10", X=X, y=y, X_test=X_test)


DATASET_MODES["S10"] = ("matched_d1", "defaults_d1")


def _s11(n: int, seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, 1, size=n)
    y = 2.0 * np.maximum(0, x0 - 0.5) + rng.normal(scale=0.05, size=n)
    X = x0.reshape(-1, 1)
    X_test = _short_test_set(X, rng)  # review round 1, #43 finding 7
    return Dataset(id=f"S11_n{n:02d}", X=X, y=y, X_test=X_test)


@register
def s11_n03() -> Dataset:
    """S11: 3 cases. Degenerate sizes."""
    return _s11(3, seed=1103)


@register
def s11_n05() -> Dataset:
    """S11: 5 cases."""
    return _s11(5, seed=1105)


@register
def s11_n08() -> Dataset:
    """S11: 8 cases."""
    return _s11(8, seed=1108)


@register
def s11_n12() -> Dataset:
    """S11: 12 cases."""
    return _s11(12, seed=1112)


for _s11_id in ("S11_n03", "S11_n05", "S11_n08", "S11_n12"):
    DATASET_MODES[_s11_id] = DEFAULT_DATASET_MODES


def _s12_base() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12)
    n = 150
    x = rng.uniform(0, 1, size=n)
    y = 2.0 * np.maximum(0, x - 0.4) + rng.normal(scale=0.05, size=n)
    X = x.reshape(-1, 1)
    # review round 1, #43 finding 7: X_test gets each variant's own
    # transform below (times 1e-8, plus 1e6, ...), the same as X, so a
    # variant's test points stay in that variant's own units.
    X_test = _short_test_set(X, rng)
    return X, y, X_test


@register
def s12_base() -> Dataset:
    """S12: the un-scaled, un-shifted base case. Invariance to scale and
    shift (earth is not scale invariant, bb14.4: this reference and its
    variants below all fit raw_d1, earth's own defaults with no LA-7
    rescaling, so their differences are earth's, not the harness's)."""
    X, y, X_test = _s12_base()
    return Dataset(id="S12_base", X=X, y=y, X_test=X_test)


@register
def s12_x_1e_minus8() -> Dataset:
    """S12: x times 1e-8."""
    X, y, X_test = _s12_base()
    return Dataset(id="S12_x_1em8", X=X * 1e-8, y=y, X_test=X_test * 1e-8)


@register
def s12_x_1e8() -> Dataset:
    """S12: x times 1e8."""
    X, y, X_test = _s12_base()
    return Dataset(id="S12_x_1e8", X=X * 1e8, y=y, X_test=X_test * 1e8)


@register
def s12_x_plus_1e6() -> Dataset:
    """S12: x plus 1e6."""
    X, y, X_test = _s12_base()
    return Dataset(id="S12_x_plus_1e6", X=X + 1e6, y=y, X_test=X_test + 1e6)


@register
def s12_y_1e_minus9() -> Dataset:
    """S12: y times 1e-9."""
    X, y, X_test = _s12_base()
    return Dataset(id="S12_y_1em9", X=X, y=y * 1e-9, X_test=X_test)


@register
def s12_y_1e9() -> Dataset:
    """S12: y times 1e9."""
    X, y, X_test = _s12_base()
    return Dataset(id="S12_y_1e9", X=X, y=y * 1e9, X_test=X_test)


for _s12_id in (
    "S12_base",
    "S12_x_1em8",
    "S12_x_1e8",
    "S12_x_plus_1e6",
    "S12_y_1em9",
    "S12_y_1e9",
):
    # raw_d1 (earth's own defaults, no LA-7 rescaling) is the point of
    # S12: earth's raw fits differ across these variants (bb14.4).
    # defaults_d1/matched_d1 (review round 1, #43 finding 3/blocking) add
    # the LA-7-scaled fits, without which no S12 fixture gives pymars and
    # earth the same matrix, so T07 cannot compare a variant against the
    # base at all.
    DATASET_MODES[_s12_id] = ("raw_d1", "defaults_d1", "matched_d1")


def _expand_by_weights(
    X: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Repeat each row of (X, y) by its integer weight, dropping zero-
    weight rows (W-1's frequency-weight definition, exactly): an
    unweighted earth fit on the result is the reference for a pymars fit
    on the original (X, y, w)."""
    w_int = np.asarray(w).astype(int)
    idx = np.repeat(np.arange(len(w_int)), w_int)
    return X[idx], y[idx]


def _s13_xy(seed: int, n: int = 120) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, 1, size=n)
    y = 2.0 * np.maximum(0, x0 - 0.4) + rng.normal(scale=0.1, size=n)
    X = x0.reshape(-1, 1)
    # review round 1, #43 finding 7. Drawn once per seed, so a weighted
    # fixture and its repeated-row sibling (same seed) get the identical
    # raw X_test, and so -- since they also share scale_override -- the
    # identical scaled X_test.
    X_test = _short_test_set(X, rng)
    return X, y, X_test


@register
def s13_int_zeros() -> Dataset:
    """S13: integer weights with zeros. Weights: removal."""
    X, y, X_test = _s13_xy(1301)
    rng = np.random.default_rng(9001)
    w = rng.integers(0, 4, size=len(y)).astype(float)  # 0 to 3, some zero
    scale = _shared_scale_from_repeated_rows(X, w)
    return Dataset(
        id="S13_int_zeros", X=X, y=y, weights=w, scale_override=scale, X_test=X_test
    )


@register
def s13_int_zeros_repeated() -> Dataset:
    """S13: the unweighted reference for s13_int_zeros (repeated rows,
    zero-weight rows dropped)."""
    X, y, X_test = _s13_xy(1301)
    rng = np.random.default_rng(9001)
    w = rng.integers(0, 4, size=len(y)).astype(float)
    scale = _shared_scale_from_repeated_rows(X, w)
    X_rep, y_rep = _expand_by_weights(X, y, w)
    return Dataset(
        id="S13_int_zeros_repeated",
        X=X_rep,
        y=y_rep,
        scale_override=scale,
        X_test=X_test,
    )


@register
def s13_int_random() -> Dataset:
    """S13: random positive integer weights. Weights: repetition."""
    X, y, X_test = _s13_xy(1302)
    rng = np.random.default_rng(9002)
    w = rng.integers(1, 5, size=len(y)).astype(float)  # 1 to 4, never zero
    scale = _shared_scale_from_repeated_rows(X, w)
    return Dataset(
        id="S13_int_random", X=X, y=y, weights=w, scale_override=scale, X_test=X_test
    )


@register
def s13_int_random_repeated() -> Dataset:
    """S13: the unweighted reference for s13_int_random (repeated rows)."""
    X, y, X_test = _s13_xy(1302)
    rng = np.random.default_rng(9002)
    w = rng.integers(1, 5, size=len(y)).astype(float)
    scale = _shared_scale_from_repeated_rows(X, w)
    X_rep, y_rep = _expand_by_weights(X, y, w)
    return Dataset(
        id="S13_int_random_repeated",
        X=X_rep,
        y=y_rep,
        scale_override=scale,
        X_test=X_test,
    )


@register
def s13_unit() -> Dataset:
    """S13: equal (unit) weights. Weights: unit weights are the same model
    as no weights."""
    X, y, X_test = _s13_xy(1303)
    w = np.ones(len(y))
    scale = _shared_scale_from_repeated_rows(X, w)
    return Dataset(
        id="S13_unit", X=X, y=y, weights=w, scale_override=scale, X_test=X_test
    )


@register
def s13_unit_repeated() -> Dataset:
    """S13: the unweighted reference for s13_unit (repeating every row
    once changes nothing)."""
    X, y, X_test = _s13_xy(1303)
    scale = _shared_scale_from_repeated_rows(X, np.ones(len(y)))
    return Dataset(
        id="S13_unit_repeated", X=X, y=y, scale_override=scale, X_test=X_test
    )


@register
def s13_equal2() -> Dataset:
    """S13: equal weights, all 2 (not 1). Weights: earth ignores weights
    that are all equal (bb02.15, GCV-8); pymars must not (W-8), so this is
    a different case from s13_unit, which repeating-by-1 cannot tell apart
    from an unweighted fit at all (review round 1, #43 adversarial
    finding 12)."""
    X, y, X_test = _s13_xy(1303)
    w = np.full(len(y), 2.0)
    scale = _shared_scale_from_repeated_rows(X, w)
    return Dataset(
        id="S13_equal2", X=X, y=y, weights=w, scale_override=scale, X_test=X_test
    )


@register
def s13_equal2_repeated() -> Dataset:
    """S13: the unweighted reference for s13_equal2 (every row twice)."""
    X, y, X_test = _s13_xy(1303)
    w = np.full(len(y), 2.0)
    scale = _shared_scale_from_repeated_rows(X, w)
    X_rep, y_rep = _expand_by_weights(X, y, w)
    return Dataset(
        id="S13_equal2_repeated",
        X=X_rep,
        y=y_rep,
        scale_override=scale,
        X_test=X_test,
    )


@register
def s13_nonint() -> Dataset:
    """S13: non-integer weights, rescaled to sum to n (VALIDATION_PLAN.md,
    "Sample weights"): compared only through the fixed-basis pruning path
    and the coefficients, against weighted earth, since repeated rows do
    not apply to a non-integer weight."""
    X, y, X_test = _s13_xy(1304)
    rng = np.random.default_rng(9004)
    w = rng.uniform(0.2, 3.0, size=len(y))
    w = w * (len(y) / w.sum())  # rescaled so that sum(w) == n
    return Dataset(id="S13_nonint", X=X, y=y, weights=w, X_test=X_test)


@register
def s13_constant_y_weighted() -> Dataset:
    """S13: a constant response with integer weights including zeros
    (docs/algorithm.md bb10.9): earth may error on this input; the fixture
    records whichever it is instead of dropping the case."""
    rng = np.random.default_rng(9005)
    n = 40
    X = rng.uniform(0, 1, size=(n, 1))
    y = np.full(n, 3.0)
    w = rng.integers(0, 4, size=n).astype(float)
    X_test = _short_test_set(X, rng)  # review round 1, #43 finding 7
    scale = _shared_scale_from_repeated_rows(X, w)
    return Dataset(
        id="S13_constant_y_weighted",
        X=X,
        y=y,
        weights=w,
        scale_override=scale,
        X_test=X_test,
    )


@register
def s13_constant_y_weighted_repeated() -> Dataset:
    """S13: the unweighted reference for s13_constant_y_weighted."""
    rng = np.random.default_rng(9005)
    n = 40
    X = rng.uniform(0, 1, size=(n, 1))
    y = np.full(n, 3.0)
    w = rng.integers(0, 4, size=n).astype(float)
    X_test = _short_test_set(X, rng)
    scale = _shared_scale_from_repeated_rows(X, w)
    X_rep, y_rep = _expand_by_weights(X, y, w)
    return Dataset(
        id="S13_constant_y_weighted_repeated",
        X=X_rep,
        y=y_rep,
        scale_override=scale,
        X_test=X_test,
    )


for _s13_id in (
    "S13_int_zeros",
    "S13_int_zeros_repeated",
    "S13_int_random",
    "S13_int_random_repeated",
    "S13_unit",
    "S13_unit_repeated",
    "S13_equal2",
    "S13_equal2_repeated",
    "S13_nonint",
    "S13_constant_y_weighted",
    "S13_constant_y_weighted_repeated",
):
    # Review round 1 (#43 finding 2/blocking): matched_d1 alone (minspan =
    # endspan = 1) makes almost every case a candidate knot regardless of
    # weight, so it cannot catch an implementation that ignored the
    # weights in the spans and knots (SPAN-1's N_b, SPAN-5's N, KNOT-6's
    # cumulative-weight scan); defaults_d1 (automatic spans) exercises them.
    DATASET_MODES[_s13_id] = ("matched_d1", "defaults_d1")


@register
def s14() -> Dataset:
    """S14: a binary response from a logistic truth, 5 covariates, 500
    cases. GLM refit."""
    rng = np.random.default_rng(14)
    n = 500
    X = rng.uniform(-1, 1, size=(n, 5))
    eta = (
        1.5 * np.maximum(0, X[:, 0] - 0.2)
        - 2.0 * np.maximum(0, -X[:, 1] - 0.1)
        + 1.0 * X[:, 2]
    )
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(size=n) < p).astype(float)
    X_test = _short_test_set(X, rng)
    return Dataset(id="S14", X=X, y=y, glm_family="binomial", X_test=X_test)


DATASET_MODES["S14"] = ("defaults_d2", "matched_d2")


@register
def s16_weighted() -> Dataset:
    """S16: S04 (5 covariates, 200 cases) with integer weights from 0 to 4.
    Frequency weights against repeated rows, compared with unweighted
    earth."""
    ds = s04_p05_n0200()
    rng = np.random.default_rng(9016)
    w = rng.integers(0, 5, size=len(ds.y)).astype(float)
    scale = _shared_scale_from_repeated_rows(ds.X, w)
    # X_test (review round 1, #43 finding 7) is S04's own: unaffected by
    # weights, so reusing it keeps this pair's test points identical too.
    return Dataset(
        id="S16_weighted",
        X=ds.X,
        y=ds.y,
        weights=w,
        scale_override=scale,
        X_test=ds.X_test,
    )


@register
def s16_weighted_repeated() -> Dataset:
    """S16: the unweighted reference for s16_weighted (repeated rows, zero-
    weight rows dropped)."""
    ds = s04_p05_n0200()
    rng = np.random.default_rng(9016)
    w = rng.integers(0, 5, size=len(ds.y)).astype(float)
    scale = _shared_scale_from_repeated_rows(ds.X, w)
    X_rep, y_rep = _expand_by_weights(ds.X, ds.y, w)
    return Dataset(
        id="S16_weighted_repeated",
        X=X_rep,
        y=y_rep,
        scale_override=scale,
        X_test=ds.X_test,
    )


# Review round 1 (#43 finding 2/blocking): defaults_d1 (automatic spans)
# exercises the weighted spans and knots that matched_d1 (minspan =
# endspan = 1) cannot; defaults_d2/matched_d2 exercise a hinge parent's
# own active weight N_b (SPAN-1), which only a degree-2 fit can reach.
DATASET_MODES["S16_weighted"] = (
    "matched_d1",
    "defaults_d1",
    "defaults_d2",
    "matched_d2",
)
DATASET_MODES["S16_weighted_repeated"] = DATASET_MODES["S16_weighted"]


@register
def s17() -> Dataset:
    """S17: three responses that share the covariates. Several responses
    with a shared basis."""
    rng = np.random.default_rng(17)
    n = 300
    X = rng.uniform(0, 1, size=(n, 3))
    base = 2.0 * np.maximum(0, X[:, 0] - 0.3) + 1.5 * np.maximum(0, X[:, 1] - 0.5)
    y1 = base + rng.normal(scale=0.1, size=n)
    y2 = 0.5 * base - 1.0 * np.maximum(0, X[:, 2] - 0.4) + rng.normal(scale=0.1, size=n)
    y3 = -0.8 * base + rng.normal(scale=0.1, size=n)
    Y = np.column_stack([y1, y2, y3])
    X_test = _short_test_set(X, rng)
    return Dataset(id="S17", X=X, y=Y, X_test=X_test)


DATASET_MODES["S17"] = ("defaults_d1", "matched_d1")


def _s18_score(X: np.ndarray) -> np.ndarray:
    return (
        1.2 * np.maximum(0, X[:, 0] - 0.2)
        - 1.2 * np.maximum(0, -X[:, 0] - 0.2)
        + 0.5 * X[:, 1]
    )


def _s18_labels(score: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Labels drawn from the softmax of ``score`` (review round 1, #43
    finding 5: a threshold rule instead let earth's basis separate the
    classes, so nnet::multinom never converged; a probabilistic draw
    keeps them overlapping)."""
    n = len(score)
    scores = np.column_stack([np.zeros(n), score, -score])  # mid, hi, lo
    probs = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    levels = np.array(["mid", "hi", "lo"])
    return np.array([levels[rng.choice(3, p=probs[i])] for i in range(n)])


@register
def s18() -> Dataset:
    """S18: a three-class response. Multiclass terms, and probabilities
    against nnet::multinom."""
    rng = np.random.default_rng(18)
    n = 400
    X = rng.uniform(-1, 1, size=(n, 2))
    labels = _s18_labels(_s18_score(X), rng)
    X_test = _short_test_set(X, rng)
    return Dataset(id="S18", X=X, y=labels, factor_response=True, X_test=X_test)


DATASET_MODES["S18"] = ("defaults_d1", "matched_d1")


def _s19_category_y() -> tuple[np.ndarray, np.ndarray]:
    """S19's shared draw: category labels and y, coded two ways below (as
    dummies, in ``s19_dummies``, and as a genuine R factor, in the extra
    fixture ``s19_factor``) so "the same data, coded two ways" is literal,
    not just the same distribution."""
    rng = np.random.default_rng(19)
    n = 250
    levels = np.array(["a", "b", "c", "d"])
    category = rng.choice(levels, size=n)
    level_effect = {"a": 0.0, "b": 1.0, "c": -1.5, "d": 2.5}
    y = np.array([level_effect[lvl] for lvl in category]) + rng.normal(
        scale=0.1, size=n
    )
    return category, y


@register
def s19_dummies() -> Dataset:
    """S19: a 4-level factor given to pymars 2.0 as OneHotEncoder(drop=
    "first") dummies. The OneHotEncoder recipe (the "factor" side, which
    earth must equal, is the extra fixture s19_factor below, since earth
    needs a genuine R factor column that the CSV-based harness cannot
    carry)."""
    category, y = _s19_category_y()
    dummy_b = (category == "b").astype(float)
    dummy_c = (category == "c").astype(float)
    dummy_d = (category == "d").astype(float)
    X = np.column_stack([dummy_b, dummy_c, dummy_d])
    X_test = _short_test_set(X, np.random.default_rng(1900))
    return Dataset(id="S19_dummies", X=X, y=y, X_test=X_test)


DATASET_MODES["S19_dummies"] = ("defaults_d1", "matched_d1")


@register
def s20() -> Dataset:
    """S20: a separable binary response. Separation warnings and fitted
    probabilities near 0 or 1."""
    rng = np.random.default_rng(20)
    n = 200
    X = rng.uniform(-1, 1, size=(n, 2))
    y = (X[:, 0] - 0.3 * X[:, 1] > 0).astype(float)  # a clean linear split
    X_test = _short_test_set(X, rng)
    return Dataset(id="S20", X=X, y=y, glm_family="binomial", X_test=X_test)


DATASET_MODES["S20"] = ("defaults_d1", "matched_d1")


# "Extra" fixtures: bespoke, one-off earth calls that do not fit the
# (dataset, mode) registry above, written straight to
# validation/fixtures/<name>.json. Like register_component, this does not
# call fn() at decoration time (a real earth call, run only when a fixture
# is actually (re)generated).
ExtraFn = Callable[[], dict[str, Any]]
EXTRA_REGISTRY: dict[str, ExtraFn] = {}


def register_extra(fn: ExtraFn) -> ExtraFn:
    EXTRA_REGISTRY[fn.__name__] = fn
    return fn


def _s15_earth_args(degree: int, mode_family: str) -> dict[str, Any]:
    """One of the two modes T07 compares (review round 1, #43 finding 4):
    "matched" (auto_linpreds/fast.k/thresh forced as matched_d1's are, but
    with automatic spans, since the new code's matched mode does not need
    them forced to 1, VALIDATION_PLAN.md "Comparison modes") or "defaults"
    (earth's own defaults, nothing forced). Both use pmethod = "none": with
    no nprune this still computes the backward-pass statistics as
    pmethod = "backward" would (PRUNE-7), so gcv_per_subset lets a caller
    derive the GCV-optimal selected size itself (PRUNE-5) without a second
    R call, while dirs/cuts/selected_terms stay the full forward set (not
    the pruned one), which is what the per-step comparison needs."""
    if mode_family == "matched":
        args = {
            k: v
            for k, v in MODES["matched_d1"].items()
            if k not in ("minspan", "endspan", "degree")
        }
    else:
        args = {}
    args["degree"] = degree
    args["pmethod"] = "none"
    return args


@register_extra
def s15_draws() -> dict[str, Any]:
    """S15: 200 small draws from validation/sims/dgps.py (VALIDATION_PLAN.md,
    "Data-generating processes"; T05 brief). Purpose: "Rates: the share of
    fits that agree, the step of the first divergence, and its cause".

    Review round 1 (#43, both reviewers, finding 4/blocking): n = 20,
    p = 10 and nk = 5 (2 forward steps) made every knot search evaluate
    exactly one case (SPAN-5's cap), so there was almost nothing to
    diverge on. n = 200 (the simulation's own smallest cell), earth's
    default term limit (nk = 21 at p = 10) and both degree 1 and 2, in
    both the "matched" and the "defaults" mode, exercise the search for
    real. To stay in the size budget without the full trace text (which
    would be large at this n and term limit), each draw is parsed
    immediately (compare.steps_from_trace, from trace = 8, which carries
    rss_before for near-tie detection unlike trace = 7, LA-5) into its
    per-step summary (parent, pred, direction, knot, best/second-best RSS,
    rss_before, flags) plus a rank-fix flag (FWD-11: whether the trace
    printed earth's own "Fixed rank deficient" line), and the raw text is
    then dropped. X is not stored either; (dgp, noise, rep) and this
    module's own seeds.train_test_rngs/dgps.generate reconstruct it
    exactly. Restricted to the p = 10 DGPs (D1-D6, D8; D7's p = 50 would
    cost far more compute for no benefit here).
    """
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import compare

    from validation.sims import dgps, seeds

    diagnostics = dgps.load_diagnostics()
    dgp_names = [name for name in dgps.REGRESSION_DGPS if name != "D7"]
    n = 200

    jobs: list[driver.EarthJob] = []
    metas: list[dict[str, Any]] = []
    for rep in range(200):
        dgp_name = dgp_names[rep % len(dgp_names)]
        dgp = dgps.REGISTRY[dgp_name]
        noise = dgp.levels[rep % len(dgp.levels)]
        degree = 1 if rep % 2 == 0 else 2
        mode_family = "matched" if (rep // 2) % 2 == 0 else "defaults"
        rng, _test_rng = seeds.train_test_rngs(f"S15_{dgp_name}_{noise}", rep)
        X, y, _truth = dgps.generate(dgp, rng, n, noise, diagnostics)
        X_scaled, _scale = scaled_matrix(X)
        job_id = f"draw{rep:03d}"
        jobs.append(
            driver.EarthJob(
                id=job_id,
                X=X_scaled,
                y=y,
                earth_args=_s15_earth_args(degree, mode_family),
                trace=8,
                include_forward_path=False,
            )
        )
        metas.append(
            {
                "rep": rep,
                "dgp": dgp_name,
                "noise": noise,
                "degree": degree,
                "mode_family": mode_family,
                "n": n,
            }
        )

    draws = []
    with tempfile.TemporaryDirectory(prefix="pymars-s15-") as tmp:
        workdir = Path(tmp)
        results = driver.run_earth(jobs, workdir=workdir)
        for meta, job in zip(metas, jobs, strict=True):
            result = results[job.id]
            trace_path = workdir / f"{job.id}_trace.txt"
            trace_text = trace_path.read_text(encoding="utf-8")
            trace_log = trace_parse.parse_trace(trace_path)
            dirs = result["dirs"]
            cuts = result["cuts"]
            # Not a "fail loudly" spot: steps_from_trace (compare.py, T02)
            # raises ValueError on some of these draws (about 1 in 15,
            # empirically: "trace step N: parent slot P maps to dirs row
            # R, which does not equal this step's own row ... with
            # predictor column ... removed"). FAST-4 (docs/algorithm.md,
            # part 2) says a step that adds a single term (a single hinge
            # or a linear term) leaves the next slot empty, so later slot
            # numbers run ahead of the dirs row count; steps_from_trace's
            # slot-to-row map does not appear to account for this. This is
            # a gap in the harness (T02), not something T05 fixes here
            # (COMMON.md: build on validation/harness/, do not rewrite
            # it), so those draws keep dirs/cuts/rss_per_subset/
            # gcv_per_subset (which do not need trace parsing) with
            # steps = None and steps_error set, rather than losing the
            # whole draw or silently patching compare.py. Reported in the
            # pull request for T02/the spec writer.
            steps: list[dict[str, Any]] | None
            steps_error: str | None
            try:
                steps = compare.steps_from_trace(trace_log, dirs, cuts)
                for step in steps:
                    step["direction"] = sorted(step["direction"])
                steps_error = None
            except ValueError as exc:
                steps = None
                steps_error = str(exc)
            draws.append(
                {
                    **meta,
                    "dirs": dirs,
                    "cuts": cuts,
                    "rss_per_subset": result["rss_per_subset"],
                    "gcv_per_subset": result["gcv_per_subset"],
                    "steps": steps,
                    "steps_error": steps_error,
                    "rank_fix": "Fixed rank deficient" in trace_text,
                }
            )
    return {"draws": draws}


@register_extra
def s18_multinom() -> dict[str, Any]:
    """S18's multiclass probability reference: nnet::multinom fitted on
    earth's own selected basis (VALIDATION_PLAN.md, "Binary outcomes": for
    three or more classes, pymars refits one multinomial model where earth
    fits one binomial model per class, so multinom on earth's selected
    basis is the probability reference, not earth's own glm.coefficients).

    Review round 1 (#43 finding 5): the basis now comes from S18's own
    LA-7-scaled matrix (the same one ``S18_matched_d1`` fits on, stored
    here too), not a fresh fit on the raw X, whose selected terms could
    agree with the matched fixture's only by chance; and nnet's
    convergence code is checked (0 required) and stored.
    """
    ds = s18()
    X_scaled, scale = scaled_matrix(ds.X)
    labels = np.asarray(ds.y)
    levels = sorted(set(labels.tolist()))
    Y_indicator = np.column_stack([(labels == lvl).astype(float) for lvl in levels])
    earth_args = {**MODES["matched_d1"]}
    basis = blackbox.fit_bx_dirs(X_scaled, Y_indicator, earth_args)
    multinom = blackbox.multinom_fit(basis["bx"], labels)
    if multinom["convergence"] != 0 or multinom["warnings"]:
        raise AssertionError(
            f"s18_multinom did not converge: "
            f"convergence={multinom['convergence']}, "
            f"warnings={multinom['warnings']}"
        )
    return {
        "x": X_scaled.tolist(),
        "scale": scale.tolist(),
        "levels": levels,
        "earth_args": earth_args,
        "dirs": basis["dirs"].tolist(),
        "cuts": basis["cuts"].tolist(),
        "selected_terms": basis["selected_terms"].tolist(),
        "multinom_coefficients": multinom["coefficients"].tolist(),
        "multinom_levels": multinom["levels"],
        "multinom_fitted": multinom["fitted"].tolist(),
        "multinom_convergence": multinom["convergence"],
    }


@register_extra
def s19_factor() -> dict[str, Any]:
    """S19's "factor" side: earth fit with the category as a genuine R
    factor column, for comparison with s19_dummies's earth-on-dummies fit
    (VALIDATION_PLAN.md, "Categorical inputs": "earth on the factor must
    equal earth on the dummies"). Both sides share one (labels, y) draw, so
    "the same data, coded two ways" is literal, not just same distribution.
    """
    category, y = _s19_category_y()
    dummy_b = (category == "b").astype(float)
    dummy_c = (category == "c").astype(float)
    dummy_d = (category == "d").astype(float)
    earth_args = {"degree": 1, "nk": 11}
    dummies = np.column_stack([dummy_b, dummy_c, dummy_d])
    factor_result = blackbox.earth_factor_fit(category, y, earth_args)
    dummy_result = blackbox.earth_factor_fit(None, y, earth_args, other_x=dummies)
    return {
        "labels": category.tolist(),
        "y": y.tolist(),
        "earth_args": earth_args,
        "factor": {
            "fitted": factor_result["fitted"].tolist(),
            "gcv": factor_result["gcv"],
            "rsq": factor_result["rsq"],
            "nterms": factor_result["nterms"],
        },
        "dummies": {
            "fitted": dummy_result["fitted"].tolist(),
            "gcv": dummy_result["gcv"],
            "rsq": dummy_result["rsq"],
            "nterms": dummy_result["nterms"],
        },
    }


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
    X, X_test = ds.X, ds.X_test
    scale = None
    if mode not in RAW_MODES:
        # LA-7's harness step: the same scaled matrix goes to both programs
        # (S12's raw_d1 fits are the one deliberate exception, above).
        # scale_override (review round 1, #43 finding 5) takes precedence
        # over computing it fresh, for a dataset that must share its exact
        # scale with a sibling (an integer-weight/repeated-row pair).
        if ds.scale_override is not None:
            scale = ds.scale_override
            X = ds.X / scale
        else:
            X, scale = scaled_matrix(ds.X, ds.weights)
        if X_test is not None:
            X_test = X_test / scale
    job = driver.EarthJob(
        id=f"{dataset_id}_{mode}",
        X=X,
        y=ds.y,
        X_test=X_test,
        earth_args=earth_args,
        factor_response=ds.factor_response,
        weights=ds.weights,
        glm_family=ds.glm_family,
    )
    with tempfile.TemporaryDirectory(prefix="pymars-fixture-") as tmp:
        # A fixture records what earth does with an input, including an
        # input earth itself rejects (VALIDATION_PLAN.md, "Test datasets":
        # "Where earth errors on an input ... record the error in the
        # fixture instead of dropping the case"), so this never raises.
        result = driver.run_earth([job], workdir=Path(tmp), raise_on_error=False)[
            job.id
        ]
    return {
        "dataset": dataset_id,
        "mode": mode,
        "inputs": {
            "X": X.tolist(),
            "y": ds.y.tolist(),
            "X_test": X_test.tolist() if X_test is not None else None,
            "weights": ds.weights.tolist() if ds.weights is not None else None,
        },
        "scale": scale.tolist() if scale is not None else None,
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


def _extra_payload(name: str) -> dict[str, Any]:
    payload = EXTRA_REGISTRY[name]()
    versions = driver.versions()
    versions.update(blackbox.versions())
    return {"extra": name, **payload, "versions": _sanitize_versions(versions)}


def _write_extra(name: str, fixtures_dir: Path) -> Path:
    payload = _extra_payload(name)
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    path = fixtures_dir / f"{name}.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def make_all_extras(*, fixtures_dir: Path | None = None) -> list[Path]:
    """Write every registered "extra" fixture (``validation/fixtures/
    <name>.json``, alongside the dataset fixtures) and return their paths,
    in registration order."""
    fixtures_dir = fixtures_dir if fixtures_dir is not None else FIXTURES_DIR
    return [_write_extra(name, fixtures_dir) for name in EXTRA_REGISTRY]


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
    """Write every registered dataset's fixtures, under the modes it opts
    into (``DATASET_MODES``, or ``DEFAULT_DATASET_MODES`` when it names
    none), and return their paths, in registration order.

    ``fixtures_dir`` defaults to the module-level ``FIXTURES_DIR``, read
    inside this function (not bound as a default parameter value), so
    ``monkeypatch.setattr(gen_fixtures, "FIXTURES_DIR", ...)`` in a test
    changes what a plain ``make_all()`` call does.
    """
    fixtures_dir = fixtures_dir if fixtures_dir is not None else FIXTURES_DIR
    return [
        _write_fixture(dataset_id, mode, fixtures_dir)
        for dataset_id in REGISTRY
        for mode in DATASET_MODES.get(dataset_id, DEFAULT_DATASET_MODES)
    ]


def check(*, fixtures_dir: Path | None = None) -> CheckResult:
    """Regenerate every dataset, component and extra fixture into a
    temporary folder and compare each one against ``fixtures_dir``
    (default: the module-level ``FIXTURES_DIR``, read dynamically the same
    way ``make_all`` does).

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
        fresh_paths = (
            make_all(fixtures_dir=tmp)
            + make_all_components(fixtures_dir=tmp)
            + make_all_extras(fixtures_dir=tmp)
        )
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

    paths = make_all() + make_all_components() + make_all_extras()
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
