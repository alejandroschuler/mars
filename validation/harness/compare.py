"""Compare two harness results by VALIDATION_PLAN.md's tolerance table
("What is compared, and the tolerances") and forward-step rule ("Ties").

``compare_fit`` compares two results in the common schema (``driver.py``'s
module docstring): selected terms, ``rss_per_subset``/``gcv_per_subset``,
the term removed at each pruning step, the forward RSS path, coefficients,
GCV/R2/GRSq, fitted values and predictions, and GLM coefficients and
probabilities. Two quantities the schema does not carry are needed for some
of these and are passed in explicitly rather than guessed at: kappa(B)
(from ``condition_number`` on a basis matrix) scales the coefficient and
GCV-family tolerances above the plan's limits, capped at its value for
kappa(B) = ``KAPPA_CAP`` so a singular basis (kappa(B) = infinity, for
example a left hinge whose knot sits at the smallest x) cannot scale a
tolerance to infinity and swallow every difference; the difference is
labeled ``numeric`` there, rather than the comparison being skipped. sd(y)
scales the fitted-value tolerance. No comparison here is ever silently
skipped without a trace of why: a shape or length mismatch, and an NA
(R's, decoded to ``None``), a NaN or an infinity on only one side, are
themselves differences, not passes; a comparison that genuinely cannot
run (a field absent on one side, or one that needs ``sd_y`` when it was
not given) is left out of the returned ``Difference`` list, since it is
not itself a disagreement, but is still recorded, with its reason, in the
caller's ``skipped`` list when it passes one. The pruning-path comparison
compares the ``prune_terms`` subset at each size directly when both sides
give it, which works even when earth's own rows are not nested (one
response, ``VALIDATION_PLAN.md``, PRUNE-3); only when a side instead
gives ``pruning_removed`` (``new_adapter.py``'s shape, no fixed-shape
subset matrix) does it fall back to a removed-term sequence, which does
assume nesting.

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
``trace_parse.TraceLog`` and the same fit's ``dirs``/``cuts``, for the
earth side of that comparison.

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
# The scaled tolerance's ceiling: its value at kappa = 1e8, about 1/sqrt(u)
# for the unit roundoff u ~= 1.1e-16 (VALIDATION_PLAN.md, "What is compared,
# and the tolerances"). Without a ceiling, a singular basis (kappa(B) = inf,
# for example a left hinge whose knot sits at the smallest x, a zero column)
# scales every tolerance to infinity, so any difference, however large,
# reports as a pass; capped at kappa = 1e8, a difference such as a gcv of 1
# against 100 still exceeds even the most generous tolerance this table
# allows, and is reported (labeled "numeric", per VALIDATION_PLAN.md's
# "numeric: a numerical difference, such as a rank-deficient solve").
KAPPA_CAP = 1e8

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


@dataclass
class Skip:
    """One comparison ``compare_fit`` did not run, and why: a field present
    on only one side, or one that needs ``sd_y``/``kappa`` (or the like)
    that was not given. Not a ``Difference`` (a skip is not itself a
    disagreement between pymars and earth, just information the harness
    lacked), but also never silent: every skip a caller might want to see
    is recorded here when it passes a ``skipped`` list to collect them.
    """

    field: str
    dataset: str | None
    reason: str


def _kappa_scale(
    base_tol: float, kappa: float | None, limit: float
) -> tuple[float, bool]:
    """The tolerance to use, and whether to label a difference "numeric":
    the base tolerance unchanged while kappa is at or below its limit (or
    unknown), otherwise the base tolerance scaled up in proportion to how
    far kappa is past the limit, but never past its value at ``KAPPA_CAP``
    ("Above these kappa(B) limits, the harness compares fitted values with
    a tolerance scaled by kappa(B), and labels any difference numeric").

    A non-finite kappa (inf, or nan from a degenerate basis) is treated as
    ``KAPPA_CAP`` itself, the worst case this table still assigns a finite
    tolerance to, not as license to accept anything: a singular basis is
    still reported "numeric" whenever the difference exceeds that capped
    tolerance, never silently passed because the scale factor ran away to
    infinity.
    """
    if kappa is None or (np.isfinite(kappa) and kappa <= limit):
        return base_tol, False
    effective_kappa = kappa if np.isfinite(kappa) else KAPPA_CAP
    return base_tol * (min(effective_kappa, KAPPA_CAP) / limit), True


def _none_mask(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    """A boolean array of ``shape``, True where ``value``'s raw entry (read
    before any ``dtype=float`` cast) is ``None`` -- R's NA, as
    ``driver._desanitize``/``blackbox._desanitize`` decode it. Plain
    ``np.asarray(value, dtype=float)`` turns ``None`` into ``nan`` (the
    same bit pattern a genuine NaN has), which would silently conflate the
    two, so this is computed on an object array first.
    """
    obj = np.asarray(value, dtype=object)
    if obj.shape != shape:  # a shape mismatch is reported by the caller
        return np.zeros(shape, dtype=bool)
    return np.array([v is None for v in obj.ravel()], dtype=bool).reshape(shape)


def _finite_mismatch(
    a: np.ndarray, b: np.ndarray, a_none: np.ndarray, b_none: np.ndarray
) -> str | None:
    """``None`` if every NA, every NaN and every signed infinity in ``a``
    lines up with one in ``b`` at the same position (so a plain numeric
    comparison over the remaining finite entries is meaningful); otherwise
    a detail string. An NA (R's NA, decoded to ``None``), a NaN or an
    infinity on only one side is always a difference, never a pass and
    never silently ignored; an NA on one side against a NaN (not an NA) on
    the other at the same position is also always a difference, checked
    before the NaN comparison below (which ``a``/``b`` alone cannot make,
    since casting to ``dtype=float`` turns an NA into the same ``nan`` bit
    pattern a real NaN has).
    """
    if not np.array_equal(a_none, b_none):
        return "NA on only one side"
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
    a_none = _none_mask(a, np.shape(a))
    b_none = _none_mask(b, np.shape(b))
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return None, f"shape mismatch {a.shape} vs {b.shape}"
    reason = _finite_mismatch(a, b, a_none, b_none)
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
    a_none = _none_mask(a, np.shape(a))
    b_none = _none_mask(b, np.shape(b))
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return None, f"shape mismatch {a.shape} vs {b.shape}"
    reason = _finite_mismatch(a, b, a_none, b_none)
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


def _record_skip(
    skipped: list[Skip] | None, field: str, dataset: str | None, reason: str
) -> None:
    """Append a ``Skip`` when the caller asked to collect them (``skipped``
    is not ``None``); a no-op otherwise, so a caller that does not want
    them pays nothing and existing callers see no change of behavior."""
    if skipped is not None:
        skipped.append(Skip(field=field, dataset=dataset, reason=reason))


def _prune_terms_size_sets(prune_terms: Any) -> list[frozenset[int]] | None:
    """The set of term indices present at each pruned subset size (ascending
    row order), directly from a ``prune_terms`` matrix (one row per size,
    0-padded). Unlike ``removed_sequence``, this never assumes consecutive
    rows are nested: earth's own ``prune.terms`` rows need not be, for one
    response (``VALIDATION_PLAN.md``, PRUNE-3, leaps-style subsets), so
    comparing the subset at each size directly is the comparison that
    always applies when both sides give ``prune_terms``. ``None`` if
    ``prune_terms`` is ``None`` or not a 2-D matrix.
    """
    if prune_terms is None:
        return None
    pt = np.asarray(prune_terms)
    if pt.ndim != 2:
        return None
    return [frozenset(int(t) for t in row.tolist() if t != 0) for row in pt]


def compare_fit(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    kappa: float | None = None,
    sd_y: float | None = None,
    glm: bool = False,
    dataset: str | None = None,
    skipped: list[Skip] | None = None,
) -> list[Difference]:
    """Compare two common-schema results; a field absent from one side (or a
    comparison that needs ``sd_y`` when it was not given) is skipped, not
    compared at its base tolerance and not treated as a difference, but
    never silently: when the caller passes a list as ``skipped``, a
    ``Skip`` naming the field and the reason is appended to it for every
    such case (``skipped`` defaults to ``None``, so a caller that does not
    ask for them is unaffected).

    ``kappa`` scales the coefficient (base limit 1e5) and GCV/R2/GRSq/
    fitted-value/forward-RSS-path (base limit 1e6) tolerances above the
    plan's limits, capped at ``KAPPA_CAP`` (``_kappa_scale``), labeling the
    difference ``numeric`` there, rather than skipping the comparison;
    without ``kappa``, every comparison runs at its base tolerance. ``sd_y``
    is required for the fitted-value/prediction comparison, which has no
    other way to know the response's scale.
    """
    diffs: list[Difference] = []

    a_sel, b_sel = a.get("selected_terms"), b.get("selected_terms")
    if a_sel is None or b_sel is None:
        if a_sel is not None or b_sel is not None:
            _record_skip(skipped, "selected_terms", dataset, "absent on one side")
    elif set(a_sel) != set(b_sel):
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
            if av is not None or bv is not None:
                _record_skip(skipped, name, dataset, "absent on one side")
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

    a_pt, b_pt = a.get("prune_terms"), b.get("prune_terms")
    if a_pt is not None and b_pt is not None:
        # Both sides give prune_terms directly (for example
        # gen_fixtures.py's earth-vs-earth reproducibility check): compare
        # the subset at each size, which works whether or not the rows
        # happen to be nested.
        a_sets, b_sets = _prune_terms_size_sets(a_pt), _prune_terms_size_sets(b_pt)
        if a_sets is None or b_sets is None:
            _record_skip(
                skipped,
                "pruning_removed",
                dataset,
                "prune_terms is not a 2-D matrix on at least one side",
            )
        elif len(a_sets) != len(b_sets):
            diffs.append(
                Difference(
                    field="pruning_removed",
                    a=len(a_sets),
                    b=len(b_sets),
                    dataset=dataset,
                    detail=f"length mismatch {len(a_sets)} vs {len(b_sets)}",
                )
            )
        else:
            mismatch = next(
                (
                    i
                    for i, (sa, sb) in enumerate(zip(a_sets, b_sets, strict=True))
                    if sa != sb
                ),
                None,
            )
            if mismatch is not None:
                diffs.append(
                    Difference(
                        field="pruning_removed",
                        step=mismatch,
                        a=sorted(a_sets[mismatch]),
                        b=sorted(b_sets[mismatch]),
                        dataset=dataset,
                        detail="the subset at this size differs",
                    )
                )
    else:
        # At least one side has no prune_terms (for example pymars's own
        # pruning_removed, new_adapter.py's shape, which has no fixed-shape
        # subset matrix): fall back to the removed-term-sequence, which
        # needs prune_terms's rows (when that is a side's source) to nest.
        a_removed, b_removed = removed_sequence(a), removed_sequence(b)
        if a_removed is None or b_removed is None:
            has_something = any(
                x is not None
                for x in (
                    a.get("pruning_removed"),
                    a_pt,
                    b.get("pruning_removed"),
                    b_pt,
                )
            )
            if has_something:
                _record_skip(
                    skipped,
                    "pruning_removed",
                    dataset,
                    "not derivable on at least one side (prune_terms rows "
                    "may not be nested, or both fields are absent there)",
                )
        elif len(a_removed) != len(b_removed):
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
    a_coef, b_coef = a.get("coef"), b.get("coef")
    if a_coef is None or b_coef is None:
        if a_coef is not None or b_coef is not None:
            _record_skip(skipped, "coef", dataset, "absent on one side")
    else:
        metric, detail = _normwise(a_coef, b_coef)
        if detail is not None:
            diffs.append(
                Difference(
                    field="coef", a=a_coef, b=b_coef, dataset=dataset, detail=detail
                )
            )
        elif metric > coef_tol:
            diffs.append(
                Difference(
                    field="coef",
                    a=a_coef,
                    b=b_coef,
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
            if av is not None or bv is not None:
                _record_skip(skipped, name, dataset, "absent on one side")
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

    fitted_tol = fitted_numeric = None
    if sd_y is not None:
        fitted_tol, fitted_numeric = _kappa_scale(
            FITTED_ABS_SD_MULT * sd_y, kappa, KAPPA_RSS_LIMIT
        )
    for name in ("fitted", "pred_test"):
        av, bv = a.get(name), b.get(name)
        if av is None or bv is None:
            if av is not None or bv is not None:
                _record_skip(skipped, name, dataset, "absent on one side")
            continue
        if sd_y is None:
            _record_skip(skipped, name, dataset, "sd_y not given")
            continue
        metric, detail = _elementwise(av, bv, relative=False)
        if detail is not None:
            diffs.append(
                Difference(field=name, a=None, b=None, dataset=dataset, detail=detail)
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
    if fwd_a is None or fwd_b is None:
        if fwd_a is not None or fwd_b is not None:
            _record_skip(skipped, "fwd_rss", dataset, "absent on one side")
    else:
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
        if av is None or bv is None:
            if av is not None or bv is not None:
                _record_skip(skipped, "glm_coef", dataset, "absent on one side")
        else:
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
                if av is not None or bv is not None:
                    _record_skip(skipped, f"{name}_prob", dataset, "absent on one side")
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


def _is_mirror_pair(dirs: np.ndarray, cuts: np.ndarray, i: int, j: int) -> bool:
    """Whether ``dirs``/``cuts`` rows ``i`` and ``j`` are a mirrored hinge
    pair earth's forward pass adds in one step: identical at every
    predictor column except one, where the codes are ``+1`` and ``-1``
    (either order) and the cut is the same."""
    di, dj = dirs[i], dirs[j]
    diff = np.flatnonzero(di != dj)
    if diff.size != 1:
        return False
    (k,) = diff
    return {int(di[k]), int(dj[k])} == {1, -1} and cuts[i, k] == cuts[j, k]


def _dirs_row_groups(dirs: np.ndarray, cuts: np.ndarray) -> list[list[int]]:
    """Group ``dirs``/``cuts``'s rows after row 0 (the intercept) into the
    term or term pair each forward step actually added, in forward
    (row) order: a row joins the previous group when the two form a
    mirrored hinge pair (``_is_mirror_pair``); otherwise it starts a new,
    so far singleton, group (a linear term, or a hinge earth did not
    mirror, for example a boundary knot). A group never grows past 2 rows,
    since a step adds at most a pair."""
    groups: list[list[int]] = []
    for row in range(1, dirs.shape[0]):
        if (
            groups
            and len(groups[-1]) == 1
            and _is_mirror_pair(dirs, cuts, groups[-1][0], row)
        ):
            groups[-1].append(row)
        else:
            groups.append([row])
    return groups


def _step_candidates(
    step: Any,
) -> list[tuple[float, int, int, float, bool, dict | None]]:
    """Every live (not skipped) candidate a ``ForwardStep`` considered, as
    ``(rss, parent, pred, cut, best, flags)`` tuples: ``parent``/``pred``
    are the trace's own 1-based slot/column numbers, ``cut`` is the
    trace's rounded value (a sanity check only; the exact knot comes from
    ``cuts``, not from here), and ``flags`` is the matching evaluated
    hinge case's four flags, when there is one."""
    candidates: list[tuple[float, int, int, float, bool, dict | None]] = []
    for search in step.searches:
        if search.skipped_reason is not None or search.pred is None:
            continue
        if search.linear is not None:
            candidates.append(
                (
                    search.linear.rss,
                    search.parent,
                    search.pred,
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
                    search.hinge.cut,
                    search.hinge.best,
                    flags,
                )
            )
    return candidates


def _slot_to_row_map(groups: list[list[int]], steps: list[Any]) -> dict[int, int]:
    """Earth's own internal term-slot numbers (1-based; slot 1 is always the
    intercept, ``dirs`` row 0) mapped to the ``dirs`` row each became,
    built by walking the forward steps in order, ``groups`` alongside
    them: the step for "new term N" (``ForwardStep.term``) uses slot N for
    its first (or only) row, and slot N + 1 only when it is a pair, for
    the second row. A single hinge or a linear term leaves slot N + 1
    forever unused (round-3 review, PR #38 finding 1: every step reserves
    two slots whether or not it adds two terms), so ``parent_slot - 1`` is
    *not* a dirs row index in general, only up to a fit's first single-
    term step; a later step's parent slot can be offset by however many
    single-term steps came before it, and this map is the only way to
    resolve it correctly."""
    slot_to_row = {1: 0}
    for group, step in zip(groups, steps, strict=False):
        slot_to_row[step.term] = group[0]
        if len(group) == 2:
            slot_to_row[step.term + 1] = group[1]
    return slot_to_row


def steps_from_trace(trace_log: Any, dirs: Any, cuts: Any) -> list[dict[str, Any]]:
    """Build ``compare_forward_steps``'s per-step dicts from a
    ``trace_parse.TraceLog`` together with the same fit's ``dirs``/``cuts``
    (a ``pmethod = "none"`` forward-order fit, ``driver.py``'s schema): the
    trace alone cannot tell a single hinge from a mirrored pair (it prints
    one line either way) and only carries the knot to 5-6 significant
    digits, so ``direction`` and ``knot`` come from ``dirs``/``cuts``
    instead, matched to the trace's ``ForwardStep``\\ s by forward order
    (``_dirs_row_groups``). ``parent``/``pred`` are converted from the
    trace's 1-based earth slot/column numbers (``trace_parse``'s own
    docstring: ``parent`` counts internal slots, not ``dirs`` rows) to the
    0-based ``dirs`` row/column indices ``new_adapter.py``'s pymars side
    also uses. The slot-to-row conversion goes through ``_slot_to_row_map``
    (built from the same step history), not a bare ``parent_slot - 1``:
    every forward step reserves two slots, so a single hinge or a linear
    term (using only the first) leaves a gap that throws a fixed offset
    off after the first one. The result is checked against ``dirs``
    itself two ways: the parent row must be earlier than the step's own
    row, and the parent row's pattern must equal the new row's with the
    predictor column removed; either failing raises, rather than silently
    comparing against the wrong term.

    A trace step that added no term (earth's forward pass always stops on
    one, for example a final "reject" line) has no matching row group and
    is dropped, not padded with a placeholder; ``steps_from_trace``'s
    output can accordingly be shorter than ``trace_log.steps``, which
    ``compare_forward_steps`` (comparing step by step, index by index
    against pymars's own log) treats the same as any other length
    mismatch it is not otherwise explained by a tie.

    ``best_rss``/``second_best_rss`` are pooled from every live candidate
    the step considered (by resulting RSS, ascending); ``rss_before`` and
    the winning candidate's flags (only for a hinge winner with a matching
    evaluated case) come from the trace as before.
    """
    dirs = np.asarray(dirs)
    cuts = np.asarray(cuts, dtype=float)
    groups = _dirs_row_groups(dirs, cuts)
    if len(groups) > len(trace_log.steps):
        raise ValueError(
            f"{len(groups)} dirs row-groups (after the intercept) but only "
            f"{len(trace_log.steps)} trace steps; the trace does not match "
            "this fit"
        )
    slot_to_row = _slot_to_row_map(groups, trace_log.steps)
    steps = []
    for group, step in zip(groups, trace_log.steps, strict=False):
        candidates = _step_candidates(step)
        if not candidates:
            raise ValueError(
                f"trace step {step.term}: dirs has a row group {group} for "
                "it, but every one of its searches was skipped or had no "
                "candidate"
            )
        tagged = [c for c in candidates if c[4]]
        winner = tagged[-1] if tagged else candidates[-1]
        _, parent_slot, pred_slot, winner_cut, _, flags = winner
        pred_col = pred_slot - 1
        parent_row = slot_to_row.get(parent_slot)
        if parent_row is None:
            raise ValueError(
                f"trace step {step.term}: parent slot {parent_slot} does "
                "not map to any dirs row (it may be a slot a single-hinge "
                "or linear step earlier in the fit left unused, or one "
                "from a step that has not happened yet)"
            )
        if not (0 <= parent_row < group[0]):
            raise ValueError(
                f"trace step {step.term}: parent slot {parent_slot} maps "
                f"to row {parent_row}, not earlier than this step's own "
                f"{group}"
            )
        expected_parent_pattern = dirs[group[0]].copy()
        expected_parent_pattern[pred_col] = 0
        if not np.array_equal(dirs[parent_row], expected_parent_pattern):
            raise ValueError(
                f"trace step {step.term}: parent slot {parent_slot} maps "
                f"to dirs row {parent_row} ({dirs[parent_row].tolist()}), "
                f"which does not equal this step's own row {group[0]} "
                f"({dirs[group[0]].tolist()}) with predictor column "
                f"{pred_col} removed"
            )
        knots = {float(cuts[r, pred_col]) for r in group}
        if len(knots) != 1:
            raise ValueError(
                f"trace step {step.term}: rows {group} do not share one "
                f"cut at predictor column {pred_col}: {sorted(knots)}"
            )
        (knot,) = knots
        if not np.isnan(winner_cut) and abs(knot - winner_cut) > 1e-3 * max(
            1.0, abs(knot)
        ):
            raise ValueError(
                f"trace step {step.term}: the trace's rounded cut "
                f"{winner_cut!r} does not match cuts's {knot!r} at row "
                f"group {group}, predictor column {pred_col}"
            )
        direction = frozenset(int(dirs[r, pred_col]) for r in group)
        by_rss = sorted(candidates, key=lambda c: c[0])
        rss_before = next(
            (sr.rss_before for sr in step.searches if sr.rss_before is not None), None
        )
        steps.append(
            {
                "parent": parent_row,
                "pred": pred_col,
                "direction": direction,
                "knot": knot,
                "best_rss": by_rss[0][0],
                "second_best_rss": by_rss[1][0] if len(by_rss) > 1 else None,
                "rss_before": rss_before,
                "flags": flags,
            }
        )
    return steps
