"""Compare two harness results by VALIDATION_PLAN.md's tolerance table
("What is compared, and the tolerances") and forward-step rule ("Ties").

``compare_fit`` compares two results in the common schema (``driver.py``'s
module docstring): selected terms, ``rss_per_subset``/``gcv_per_subset``,
coefficients, GCV/R2/GRSq, fitted values and predictions, and GLM
coefficients and probabilities. Two quantities the schema does not carry are
needed for some of these and are passed in explicitly rather than guessed
at: kappa(B) (from ``condition_number`` on a basis matrix) gates the
coefficient and GCV-family comparisons, and sd(y) scales the fitted-value
tolerance.

``compare_forward_steps`` compares two per-step candidate logs (a list of
dicts, one per forward step, with ``parent``, ``pred``, ``knot``, ``kind``
("linear" or "hinge") and, when available, ``best_rss``/``second_best_rss``
and ``rss_before``): exact match up to the first near-tie, per ("Ties"): a
near-tie is when the best and second-best candidate RSS (from either side's
own log) differ by less than 1e-7 times the RSS before the step, and the
structural comparison stops there, because the two paths after two
different choices cannot be compared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

NEAR_TIE_REL = 1e-7
KAPPA_RSS_LIMIT = 1e6
KAPPA_COEF_LIMIT = 1e5

RSS_PER_SUBSET_REL = 1e-8
GCV_PER_SUBSET_REL = 1e-8
COEF_NORMWISE_REL = 1e-6
GCV_REL = 1e-8
RSQ_ABS = 1e-8
GRSQ_ABS = 1e-8
FITTED_ABS_SD_MULT = 1e-8
GLM_COEF_REL = 1e-5
GLM_PROB_ABS = 1e-7


def condition_number(basis: Any) -> float:
    """kappa(B), the 2-norm condition number of a basis matrix."""
    return float(np.linalg.cond(np.asarray(basis, dtype=float)))


@dataclass
class Difference:
    """One entry for DIFFERENCES.md ("Triage of differences"): a dataset and
    step (either may be None when not applicable), both values or choices,
    the discrepancy and the tolerance it failed, and a place for the label
    (rule/bug/quirk/tie/numeric) a human or a later step assigns."""

    field: str
    a: Any
    b: Any
    metric: float | None = None
    tolerance: float | None = None
    step: int | None = None
    dataset: str | None = None
    detail: str = ""
    label: str | None = None


def _relative_max(a: Any, b: Any) -> float:
    """max_i |a_i - b_i| / |b_i| (0 where b_i is 0 too, else the raw diff)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0:
        return 0.0
    denom = np.where(b == 0, 1.0, np.abs(b))
    return float(np.max(np.abs(a - b) / denom))


def _normwise_relative(a: Any, b: Any) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(b)
    numer = np.linalg.norm(a - b)
    return float(numer / denom) if denom != 0 else float(numer)


def _abs_max(a: Any, b: Any) -> float:
    return float(
        np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))
    )


def compare_fit(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    kappa: float | None = None,
    sd_y: float | None = None,
    glm: bool = False,
    dataset: str | None = None,
) -> list[Difference]:
    """Compare two common-schema results; a field absent from either side is
    skipped, not an error (an adapter may not populate every field, for
    example the legacy adapter's ``selected_terms``).

    ``kappa`` gates the coefficient (<= 1e5) and GCV/R2/GRSq/fitted-value
    (<= 1e6) comparisons, per the tolerance table; without it, every
    comparison runs (the caller is responsible for the kappa(B) check when
    it matters). ``sd_y`` is required for the fitted-value/prediction
    comparison, which has no other way to know the response's scale.
    """
    diffs: list[Difference] = []

    a_sel, b_sel = a.get("selected_terms"), b.get("selected_terms")
    if a_sel is not None and b_sel is not None and set(a_sel) != set(b_sel):
        diffs.append(
            Difference(
                field="selected_terms",
                a=sorted(set(a_sel)),
                b=sorted(set(b_sel)),
                dataset=dataset,
            )
        )

    for name, tol in (
        ("rss_per_subset", RSS_PER_SUBSET_REL),
        ("gcv_per_subset", GCV_PER_SUBSET_REL),
    ):
        av, bv = a.get(name), b.get(name)
        if av is not None and bv is not None and len(av) == len(bv) and len(av) > 0:
            metric = _relative_max(av, bv)
            if metric > tol:
                diffs.append(
                    Difference(
                        field=name,
                        a=av,
                        b=bv,
                        metric=metric,
                        tolerance=tol,
                        dataset=dataset,
                        detail="relative",
                    )
                )

    kappa_ok_coef = kappa is None or kappa <= KAPPA_COEF_LIMIT
    if kappa_ok_coef and a.get("coef") is not None and b.get("coef") is not None:
        metric = _normwise_relative(a["coef"], b["coef"])
        if metric > COEF_NORMWISE_REL:
            diffs.append(
                Difference(
                    field="coef",
                    a=a["coef"],
                    b=b["coef"],
                    metric=metric,
                    tolerance=COEF_NORMWISE_REL,
                    dataset=dataset,
                    detail="normwise relative",
                )
            )

    kappa_ok_rss = kappa is None or kappa <= KAPPA_RSS_LIMIT
    if kappa_ok_rss:
        for name, tol, kind in (
            ("gcv", GCV_REL, "relative"),
            ("rsq", RSQ_ABS, "absolute"),
            ("grsq", GRSQ_ABS, "absolute"),
        ):
            av, bv = a.get(name), b.get(name)
            if av is None or bv is None:
                continue
            metric = abs(av - bv) / (abs(bv) if kind == "relative" and bv != 0 else 1.0)
            if kind == "absolute":
                metric = abs(av - bv)
            if metric > tol:
                diffs.append(
                    Difference(
                        field=name,
                        a=av,
                        b=bv,
                        metric=metric,
                        tolerance=tol,
                        dataset=dataset,
                        detail=kind,
                    )
                )

        if sd_y is not None:
            for name in ("fitted", "pred_test"):
                av, bv = a.get(name), b.get(name)
                if av is not None and bv is not None:
                    tol = FITTED_ABS_SD_MULT * sd_y
                    metric = _abs_max(av, bv)
                    if metric > tol:
                        diffs.append(
                            Difference(
                                field=name,
                                a=None,
                                b=None,
                                metric=metric,
                                tolerance=tol,
                                dataset=dataset,
                                detail="absolute, scaled by sd(y)",
                            )
                        )

    if glm:
        av, bv = a.get("glm_coef"), b.get("glm_coef")
        if av is not None and bv is not None:
            metric = _relative_max(av, bv)
            if metric > GLM_COEF_REL:
                diffs.append(
                    Difference(
                        field="glm_coef",
                        a=av,
                        b=bv,
                        metric=metric,
                        tolerance=GLM_COEF_REL,
                        dataset=dataset,
                        detail="relative",
                    )
                )
        for name in ("pred_train", "pred_test"):
            av, bv = a.get(name), b.get(name)
            if av is not None and bv is not None:
                metric = _abs_max(av, bv)
                if metric > GLM_PROB_ABS:
                    diffs.append(
                        Difference(
                            field=f"{name}_prob",
                            a=None,
                            b=None,
                            metric=metric,
                            tolerance=GLM_PROB_ABS,
                            dataset=dataset,
                            detail="absolute",
                        )
                    )

    return diffs


def is_near_tie(
    best: float | None,
    second_best: float | None,
    rss_before: float | None,
    *,
    rel: float = NEAR_TIE_REL,
) -> bool:
    """Whether the best and second-best candidate RSS are within ``rel``
    times the RSS before the step ("Ties"); False when either RSS is
    missing (no second candidate was logged, or no baseline is known)."""
    if best is None or second_best is None or not rss_before:
        return False
    return abs(second_best - best) < rel * abs(rss_before)


_STEP_KEYS = ("parent", "pred", "knot", "kind")


@dataclass
class ForwardCompareResult:
    """One dataset's forward-step comparison: the step indices that matched
    exactly, the first mismatch (if the comparison did not stop at a
    near-tie first), the first near-tie (if any; the structural comparison
    stops here), and a Difference per mismatch encountered before stopping.
    """

    matched: list[int] = field(default_factory=list)
    first_mismatch: int | None = None
    first_near_tie: int | None = None
    differences: list[Difference] = field(default_factory=list)


def compare_forward_steps(
    a_steps: list[dict[str, Any]],
    b_steps: list[dict[str, Any]],
    *,
    dataset: str | None = None,
) -> ForwardCompareResult:
    """Compare two forward-step logs step by step, stopping at the first
    near-tie or the first exact mismatch, whichever comes first.

    Each step dict needs ``parent``, ``pred``, ``knot`` and ``kind``
    ("linear" for a single new column, "hinge" for a hinge pair); ``knot``
    is compared exactly, as an observed data value written and read at 17
    significant digits, not with a tolerance. ``best_rss``/``second_best_rss``
    /``rss_before`` are optional and only used for near-tie detection.
    """
    result = ForwardCompareResult()
    for i in range(min(len(a_steps), len(b_steps))):
        step_a, step_b = a_steps[i], b_steps[i]
        tie = is_near_tie(
            step_a.get("best_rss"),
            step_a.get("second_best_rss"),
            step_a.get("rss_before"),
        ) or is_near_tie(
            step_b.get("best_rss"),
            step_b.get("second_best_rss"),
            step_b.get("rss_before"),
        )
        same = all(step_a.get(k) == step_b.get(k) for k in _STEP_KEYS)
        if same:
            result.matched.append(i)
            if tie:
                result.first_near_tie = i
                break
            continue
        if tie:
            result.first_near_tie = i
            break
        result.first_mismatch = i
        result.differences.append(
            Difference(
                field="forward_step",
                step=i,
                dataset=dataset,
                a={k: step_a.get(k) for k in _STEP_KEYS},
                b={k: step_b.get(k) for k in _STEP_KEYS},
                detail="exact",
            )
        )
        break
    return result
