"""Compare two harness results by VALIDATION_PLAN.md's tolerance table
("What is compared, and the tolerances") and forward-step rule ("Ties").

``compare_fit`` compares two results in the common schema (``driver.py``'s
module docstring): selected terms, ``rss_per_subset``/``gcv_per_subset``,
the term removed at each pruning step, the forward RSS path, coefficients,
GCV/R2/GRSq, fitted values and predictions, and GLM coefficients and
probabilities. Two quantities the schema does not carry are needed for some
of these and are passed in explicitly rather than guessed at: kappa(B)
(from ``condition_number`` on a basis matrix) scales the coefficient and
GCV-family tolerances above the plan's limits (and labels the difference
``numeric`` there, rather than skipping the comparison), and sd(y) scales
the fitted-value tolerance. No comparison here is ever silently skipped
without saying why in the returned list of ``Difference``s: a missing
field, a shape or length mismatch, and a NaN or an infinity on only one
side are themselves differences, not passes.

``compare_forward_steps`` compares two per-step candidate logs (a list of
dicts, one per forward step, with ``parent``, ``pred``, ``knot``,
``direction`` (a ``frozenset`` of the ``dirs`` codes the step's winning
term added: ``{1, -1}`` for a hinge pair, ``{1}`` or ``{-1}`` for a single
hinge, ``{2}`` for a linear term) and, when available, ``best_rss``/
``second_best_rss``, ``rss_before`` and ``flags``): exact match up to the
first near-tie, per ("Ties"): a near-tie is when the best and second-best
candidate RSS (from either side's own log) differ by less than 1e-7 times
the RSS before the step, and the structural comparison stops there, because
the two paths after two different choices cannot be compared.
``steps_from_trace`` builds this step-dict list from a
``trace_parse.TraceLog``, for the earth side of that comparison.

``compare_defaults`` is the plan's "defaults mode" row: structures differ
between the two sides, so it only reports R2, GCV, the number of terms and
(when a test response is given) the test error, with no tolerance.
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
FWD_RSS_REL = 1e-8
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
    step (either may be ``None`` when not applicable), both values or
    choices, the discrepancy and the tolerance it failed, both candidate RSS
    values and earth's trace flags for a forward-step difference, and a
    place for the label (rule/bug/quirk/tie/numeric) a human or a later step
    assigns.
    """

    field: str
    a: Any
    b: Any
    metric: float | None = None
    tolerance: float | None = None
    step: int | None = None
    dataset: str | None = None
    detail: str = ""
    label: str | None = None
    candidate_rss: tuple[float | None, float | None] | None = None
    flags: dict[str, bool | None] | None = None


def _kappa_scale(
    base_tol: float, kappa: float | None, limit: float
) -> tuple[float, bool]:
    """The tolerance to use, and whether to label a difference "numeric":
    the base tolerance unchanged while kappa is at or below its limit (or
    unknown), otherwise the base tolerance scaled up in proportion to how
    far kappa is past the limit ("Above these kappa(B) limits, the harness
    compares fitted values with a tolerance scaled by kappa(B), and labels
    any difference numeric").
    """
    if kappa is None or kappa <= limit:
        return base_tol, False
    return base_tol * (kappa / limit), True


def _finite_mismatch(a: np.ndarray, b: np.ndarray) -> str | None:
    """``None`` if every NaN and every signed infinity in ``a`` lines up
    with one in ``b`` at the same position (so a plain numeric comparison
    over the remaining finite entries is meaningful); otherwise a detail
    string. A NaN or an infinity on only one side is always a difference,
    never a pass and never silently ignored.
    """
    a_nan, b_nan = np.isnan(a), np.isnan(b)
    if not np.array_equal(a_nan, b_nan):
        return "NaN on only one side"
    a_inf, b_inf = np.isinf(a), np.isinf(b)
    if not np.array_equal(a_inf, b_inf):
        return "an infinite value on only one side"
    if np.any(a_inf) and not np.array_equal(a[a_inf], b[b_inf]):
        return "opposite-signed infinities"
    return None


def _elementwise(a: Any, b: Any, *, relative: bool) -> tuple[float | None, str | None]:
    """Compare two array-likes position by position. Returns ``(metric,
    detail)``: when ``detail`` is not ``None`` (a shape mismatch or a
    non-finite value on only one side), that alone is an automatic
    difference regardless of any tolerance, and ``metric`` carries no
    meaningful magnitude; otherwise ``detail`` is ``None`` and ``metric``
    (the worst elementwise relative or absolute difference, over the
    finite entries) is what the caller compares against its tolerance.
    Never a silent pass on a shape or a non-finite mismatch.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return None, f"shape mismatch {a.shape} vs {b.shape}"
    reason = _finite_mismatch(a, b)
    if reason is not None:
        return None, reason
    finite = np.isfinite(a)  # same positions in a and b, by the check above
    af, bf = a[finite], b[finite]
    if af.size == 0:
        return 0.0, None
    if relative:
        denom = np.where(bf == 0, 1.0, np.abs(bf))
        metric = float(np.max(np.abs(af - bf) / denom))
    else:
        metric = float(np.max(np.abs(af - bf)))
    return metric, None


def _normwise(a: Any, b: Any) -> tuple[float | None, str | None]:
    """The normwise relative metric ``||a - b|| / ||b||`` over the finite
    entries, after the same shape and non-finite-position checks
    ``_elementwise`` uses (coefficients are compared this way, not
    position by position, but must not silently ignore a shape or a NaN
    mismatch either). Returns ``(metric, detail)``, the same contract as
    ``_elementwise``.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return None, f"shape mismatch {a.shape} vs {b.shape}"
    reason = _finite_mismatch(a, b)
    if reason is not None:
        return None, reason
    finite = np.isfinite(a)
    af, bf = a[finite], b[finite]
    denom = np.linalg.norm(bf)
    numer = np.linalg.norm(af - bf)
    metric = float(numer / denom) if denom != 0 else float(numer)
    return metric, None


def removed_sequence(result: dict[str, Any]) -> list[int] | None:
    """The term index removed at each pruning step, in order.

    From ``pruning_removed`` directly (``new_adapter.py``'s shape) if
    present, else derived from ``prune_terms`` (earth's/``driver.py``'s
    shape: one row per subset size, ascending, each row the term indices
    present at that size, 0-padded): going from the largest subset down to
    the smallest, the term missing from the next (smaller) row is the one
    removed at that step. ``None`` if neither field is usable.
    """
    if result.get("pruning_removed") is not None:
        return list(result["pruning_removed"])
    prune_terms = result.get("prune_terms")
    if prune_terms is None:
        return None
    pt = np.asarray(prune_terms)
    if pt.ndim != 2 or pt.shape[0] < 2:
        return None
    removed = []
    for i in range(pt.shape[0] - 1, 0, -1):
        larger = {t for t in pt[i].tolist() if t != 0}
        smaller = {t for t in pt[i - 1].tolist() if t != 0}
        gone = larger - smaller
        if len(gone) != 1:
            return None
        removed.append(gone.pop())
    return removed


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
    skipped (documented as such is not the same as a silent pass on a field
    both sides do carry, which never happens here).

    ``kappa`` scales the coefficient (base limit 1e5) and GCV/R2/GRSq/
    fitted-value/forward-RSS-path (base limit 1e6) tolerances above the
    plan's limits, labeling the difference ``numeric`` there, rather than
    skipping the comparison; without ``kappa``, every comparison runs at
    its base tolerance. ``sd_y`` is required for the fitted-value/
    prediction comparison, which has no other way to know the response's
    scale.
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
        if av is None or bv is None:
            continue
        metric, detail = _elementwise(av, bv, relative=True)
        if detail is not None:
            diffs.append(
                Difference(field=name, a=av, b=bv, dataset=dataset, detail=detail)
            )
        elif metric > tol:
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

    a_removed, b_removed = removed_sequence(a), removed_sequence(b)
    if a_removed is not None and b_removed is not None:
        if len(a_removed) != len(b_removed):
            diffs.append(
                Difference(
                    field="pruning_removed",
                    a=a_removed,
                    b=b_removed,
                    dataset=dataset,
                    detail=f"length mismatch {len(a_removed)} vs {len(b_removed)}",
                )
            )
        else:
            mismatch = next(
                (
                    i
                    for i, (x, y) in enumerate(zip(a_removed, b_removed, strict=True))
                    if x != y
                ),
                None,
            )
            if mismatch is not None:
                diffs.append(
                    Difference(
                        field="pruning_removed",
                        step=mismatch,
                        a=a_removed[mismatch],
                        b=b_removed[mismatch],
                        dataset=dataset,
                        detail="exact",
                    )
                )

    coef_tol, coef_numeric = _kappa_scale(COEF_NORMWISE_REL, kappa, KAPPA_COEF_LIMIT)
    if a.get("coef") is not None and b.get("coef") is not None:
        metric, detail = _normwise(a["coef"], b["coef"])
        if detail is not None:
            diffs.append(
                Difference(
                    field="coef",
                    a=a["coef"],
                    b=b["coef"],
                    dataset=dataset,
                    detail=detail,
                )
            )
        elif metric > coef_tol:
            diffs.append(
                Difference(
                    field="coef",
                    a=a["coef"],
                    b=b["coef"],
                    metric=metric,
                    tolerance=coef_tol,
                    dataset=dataset,
                    detail="normwise relative",
                    label="numeric" if coef_numeric else None,
                )
            )

    for name, base_tol, relative in (
        ("gcv", GCV_REL, True),
        ("rsq", RSQ_ABS, False),
        ("grsq", GRSQ_ABS, False),
    ):
        av, bv = a.get(name), b.get(name)
        if av is None or bv is None:
            continue
        tol, numeric = _kappa_scale(base_tol, kappa, KAPPA_RSS_LIMIT)
        metric, detail = _elementwise([av], [bv], relative=relative)
        if detail is not None:
            diffs.append(
                Difference(field=name, a=av, b=bv, dataset=dataset, detail=detail)
            )
        elif metric > tol:
            diffs.append(
                Difference(
                    field=name,
                    a=av,
                    b=bv,
                    metric=metric,
                    tolerance=tol,
                    dataset=dataset,
                    detail="relative" if relative else "absolute",
                    label="numeric" if numeric else None,
                )
            )

    if sd_y is not None:
        fitted_tol, fitted_numeric = _kappa_scale(
            FITTED_ABS_SD_MULT * sd_y, kappa, KAPPA_RSS_LIMIT
        )
        for name in ("fitted", "pred_test"):
            av, bv = a.get(name), b.get(name)
            if av is None or bv is None:
                continue
            metric, detail = _elementwise(av, bv, relative=False)
            if detail is not None:
                diffs.append(
                    Difference(
                        field=name, a=None, b=None, dataset=dataset, detail=detail
                    )
                )
            elif metric > fitted_tol:
                diffs.append(
                    Difference(
                        field=name,
                        a=None,
                        b=None,
                        metric=metric,
                        tolerance=fitted_tol,
                        dataset=dataset,
                        detail="absolute, scaled by sd(y)",
                        label="numeric" if fitted_numeric else None,
                    )
                )

    fwd_a, fwd_b = a.get("fwd_rss"), b.get("fwd_rss")
    if fwd_a is not None and fwd_b is not None:
        tol, numeric = _kappa_scale(FWD_RSS_REL, kappa, KAPPA_RSS_LIMIT)
        metric, detail = _elementwise(fwd_a, fwd_b, relative=True)
        if detail is not None:
            diffs.append(
                Difference(
                    field="fwd_rss", a=fwd_a, b=fwd_b, dataset=dataset, detail=detail
                )
            )
        elif metric > tol:
            diffs.append(
                Difference(
                    field="fwd_rss",
                    a=fwd_a,
                    b=fwd_b,
                    metric=metric,
                    tolerance=tol,
                    dataset=dataset,
                    detail="relative",
                    label="numeric" if numeric else None,
                )
            )

    if glm:
        av, bv = a.get("glm_coef"), b.get("glm_coef")
        if av is not None and bv is not None:
            metric, detail = _elementwise(av, bv, relative=True)
            if detail is not None:
                diffs.append(
                    Difference(
                        field="glm_coef", a=av, b=bv, dataset=dataset, detail=detail
                    )
                )
            elif metric > GLM_COEF_REL:
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
            if av is None or bv is None:
                continue
            metric, detail = _elementwise(av, bv, relative=False)
            if detail is not None:
                diffs.append(
                    Difference(
                        field=f"{name}_prob",
                        a=None,
                        b=None,
                        dataset=dataset,
                        detail=detail,
                    )
                )
            elif metric > GLM_PROB_ABS:
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


def compare_defaults(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    y_test: Any | None = None,
    dataset: str | None = None,
) -> dict[str, Any]:
    """The plan's "defaults mode" row: the two structures are expected to
    differ, so this only reports R2, GCV, the number of terms and (when
    ``y_test`` and a ``pred_test`` are available) the test MSE, for each
    side; there is no tolerance, and nothing here is a ``Difference``.
    """

    def n_terms(d: dict[str, Any]) -> int | None:
        if d.get("selected_terms") is not None:
            return len(d["selected_terms"])
        if d.get("dirs") is not None:
            return len(d["dirs"])
        return None

    def test_error(d: dict[str, Any]) -> float | None:
        if y_test is None or d.get("pred_test") is None:
            return None
        y = np.asarray(y_test, dtype=float)
        pred = np.asarray(d["pred_test"], dtype=float).reshape(y.shape)
        return float(np.mean((y - pred) ** 2))

    def side(d: dict[str, Any]) -> dict[str, Any]:
        return {
            "rsq": d.get("rsq"),
            "gcv": d.get("gcv"),
            "n_terms": n_terms(d),
            "test_error": test_error(d),
        }

    return {"dataset": dataset, "a": side(a), "b": side(b)}


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


_STEP_KEYS = ("parent", "pred", "direction", "knot")


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

    Each step dict needs ``parent``, ``pred``, ``knot`` and ``direction``
    (a ``frozenset`` of the ``dirs`` codes added: ``{1, -1}`` for a hinge
    pair, ``{1}``/``{-1}`` for a single hinge, ``{2}`` for a linear term);
    ``knot`` is compared exactly, as an observed data value written and
    read at 17 significant digits, not with a tolerance.
    ``best_rss``/``second_best_rss``/``rss_before`` are optional and used
    only for near-tie detection; ``flags`` (optional) and both sides'
    candidate RSS values are carried onto a mismatch's ``Difference`` for
    ``DIFFERENCES.md``. A log-length mismatch itself is reported, not
    silently truncated to the shorter one.
    """
    result = ForwardCompareResult()
    n = min(len(a_steps), len(b_steps))
    for i in range(n):
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
                diffs_here = result.differences
                diffs_here.append(
                    Difference(
                        field="forward_step",
                        step=i,
                        dataset=dataset,
                        label="tie",
                        a={k: step_a.get(k) for k in _STEP_KEYS},
                        b={k: step_b.get(k) for k in _STEP_KEYS},
                        candidate_rss=(
                            step_a.get("best_rss"),
                            step_a.get("second_best_rss"),
                        ),
                        flags=step_a.get("flags") or step_b.get("flags"),
                        detail="near-tie (matched anyway)",
                    )
                )
                break
            continue
        if tie:
            result.first_near_tie = i
            result.differences.append(
                Difference(
                    field="forward_step",
                    step=i,
                    dataset=dataset,
                    label="tie",
                    a={k: step_a.get(k) for k in _STEP_KEYS},
                    b={k: step_b.get(k) for k in _STEP_KEYS},
                    candidate_rss=(
                        step_a.get("best_rss"),
                        step_a.get("second_best_rss"),
                    ),
                    flags=step_a.get("flags") or step_b.get("flags"),
                    detail="near-tie",
                )
            )
            break
        result.first_mismatch = i
        result.differences.append(
            Difference(
                field="forward_step",
                step=i,
                dataset=dataset,
                a={k: step_a.get(k) for k in _STEP_KEYS},
                b={k: step_b.get(k) for k in _STEP_KEYS},
                candidate_rss=(step_a.get("best_rss"), step_b.get("best_rss")),
                flags=step_a.get("flags"),
                detail="exact",
            )
        )
        break
    if (
        result.first_mismatch is None
        and result.first_near_tie is None
        and len(a_steps) != len(b_steps)
    ):
        result.differences.append(
            Difference(
                field="forward_step_count",
                step=n,
                dataset=dataset,
                a=len(a_steps),
                b=len(b_steps),
                detail=f"one log has {len(a_steps)} steps, the other {len(b_steps)}",
            )
        )
    return result


def steps_from_trace(trace_log: Any) -> list[dict[str, Any]]:
    """Build ``compare_forward_steps``'s per-step dicts from a
    ``trace_parse.TraceLog``: for each ``ForwardStep``, the winning
    (parent, pred, direction, knot) (the last candidate tagged ``best`` in
    file order, across every search in the step, linear and hinge alike;
    ``trace_parse``'s own docstring explains why only the last tag is
    authoritative), the best and second-best candidate RSS pooled across
    every search (by resulting RSS, ascending), ``rss_before``, and the
    winning candidate's flags when it came from an evaluated hinge case
    with a matching cut.
    """
    steps = []
    for step in trace_log.steps:
        candidates: list[
            tuple[float, int, int, frozenset[int], float, bool, dict | None]
        ] = []
        for search in step.searches:
            if search.skipped_reason is not None or search.pred is None:
                continue
            if search.linear is not None:
                candidates.append(
                    (
                        search.linear.rss,
                        search.parent,
                        search.pred,
                        frozenset({2}),
                        search.linear.cut,
                        search.linear.best,
                        None,
                    )
                )
            if search.hinge is not None:
                flags = None
                for case in search.cases:
                    if case.evaluated and case.cut == search.hinge.cut:
                        flags = {
                            "bx1g": case.bx1g,
                            "cov_col_g": case.cov_col_g,
                            "tol_g": case.tol_g,
                            "max_g": case.max_g,
                        }
                        break
                candidates.append(
                    (
                        search.hinge.rss,
                        search.parent,
                        search.pred,
                        frozenset({1, -1}),
                        search.hinge.cut,
                        search.hinge.best,
                        flags,
                    )
                )
        if not candidates:
            steps.append(
                {
                    "parent": None,
                    "pred": None,
                    "direction": frozenset(),
                    "knot": None,
                    "best_rss": None,
                    "second_best_rss": None,
                    "rss_before": None,
                    "flags": None,
                }
            )
            continue
        tagged = [c for c in candidates if c[5]]
        winner = tagged[-1] if tagged else candidates[-1]
        by_rss = sorted(candidates, key=lambda c: c[0])
        rss_before = next(
            (sr.rss_before for sr in step.searches if sr.rss_before is not None), None
        )
        steps.append(
            {
                "parent": winner[1],
                "pred": winner[2],
                "direction": winner[3],
                "knot": winner[4],
                "best_rss": by_rss[0][0],
                "second_best_rss": by_rss[1][0] if len(by_rss) > 1 else None,
                "rss_before": rss_before,
                "flags": winner[6],
            }
        )
    return steps
