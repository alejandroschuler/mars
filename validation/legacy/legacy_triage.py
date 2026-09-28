"""Comparison and triage logic for the description of the legacy code (T18,
issue #20): the pure functions behind ``conformance_legacy.py``, kept apart so
that they can be tested without R or the legacy venv.

A term is a *signature*: a tuple of factors ``(var, code, knot)`` sorted by
``var``, with ``code`` +1 for (x - knot)+, -1 for (knot - x)+, 2 for a linear
factor (``knot`` None) and 3 for a legacy categorical indicator (``knot`` the
category). The intercept is ``()``. Both programs' terms are compared as
signatures, so the comparison needs no row or slot numbers.

The labels and finding IDs follow VALIDATION_PLAN.md, "Triage of differences"
and "Preliminary findings" (F1 to F16); F17 to F19 are this task's
(validation/DIFFERENCES_legacy.md). A verdict whose ID is None names an earth
rule that no fit of S01 to S20 showed; it would need a new ID. earth's rules
are cited by their docs/algorithm.md IDs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

NEAR_TIE_REL = 1e-7  # VALIDATION_PLAN.md, "Ties"
# A gain within this fraction of the RSS is rounding: the candidate is in the
# model's span already (several orders of magnitude above u = 1.1e-16).
ZERO_GAIN_REL = 1e-10

Factor = tuple[int, int, Any]
Sig = tuple[Factor, ...]
Verdict = tuple[str, str | None, str]

# (label, finding, sentence) for each kind of difference; the sentence is a
# format string over the evidence.
VERDICTS: dict[str, Verdict] = {
    "tie": ("tie", None, "{head}; {rss}: a near-tie (below 1e-7 of the RSS)"),
    "mirror_parent": (
        "rule",
        "F5",
        "{head}; the legacy parent is the redundant second hinge of an "
        "earlier pair, which earth never added",
    ),
    "slot": ("quirk", None, "{head}; earth did not search that parent (FAST-4)"),
    "queue": (
        "rule",
        "F10",
        "{head}; earth's Fast MARS queue did not reach that parent",
    ),
    "top": (
        "rule",
        "F7",
        "{head}; earth places no knot at or above the parent's largest active "
        "value (KNOT-4), where the legacy right hinge is zero",
    ),
    "bottom_inactive": (
        "quirk",
        "F7",
        "{head}; earth's lower endspan counts cases where the parent is zero "
        "(KNOT-4), which the legacy knot set does not",
    ),
    "inactive": (
        "quirk",
        "F7",
        "{head}; earth evaluates a knot only just below a case where the parent "
        "is positive (KNOT-3), and the case above the legacy knot has the parent "
        "at zero",
    ),
    "inactive_earth": (
        "quirk",
        "F7",
        "{head}; earth's knot is the value of a case where the parent is zero "
        "(KNOT-4), which the legacy knot set leaves out",
    ),
    "grid": (
        "rule",
        "F7",
        "{head}; the legacy knot is not in earth's knot grid ({where})",
    ),
    "tol": (
        "rule",
        "F5",
        "{head}; earth rejects the legacy knot by its collinearity tolerance "
        "(TolG 0); {rss}",
    ),
    "maxlegal": (
        "quirk",
        None,
        "{head}; the legacy knot lowers the RSS by more than earth's limit "
        "MaxLegal {max_legal} (MaxG 0); {rss}",
    ),
    "single": (
        "rule",
        "F5",
        "{head}; earth scores that parent and variable in a single-hinge "
        "search (LA-7), without the linear part that the legacy pair adds; {rss}",
    ),
    "not_legacy": (
        "rule",
        "F7",
        "{head}; earth's choice is not a legacy candidate ({why})",
    ),
    "gcv_rank": (
        "rule",
        "F2",
        "{head}; the legacy code ranks by GCV: its choice has GCV {gcv_a} and "
        "RSS {rss_a}, earth's choice GCV {gcv_e} and the lower RSS {rss_e}",
    ),
    "unexplained_choice": ("unexplained", None, "{head}; {rss}; probe {status}"),
    "stop_gcv_inf": (
        "rule",
        "F2",
        "{head}; every legacy candidate has C = M + d*H >= n, so an infinite "
        "GCV, and the legacy code stops (earth's forward pass ignores the GCV)",
    ),
    "stop_no_candidate": (
        "rule",
        "F2",
        "{head}; every legacy candidate would make n or more columns, and the "
        "legacy code skips such candidates (its GCV would be infinite)",
    ),
    "stop_zero_gain": (
        "rule",
        "F2",
        "{head}; the best legacy candidate by GCV, {chosen}, is already in the "
        "model's span (its computed gain is {gain} of {rss_before}), and the "
        "legacy code stops on it: its stop test (F6) looks at the best candidate "
        "by GCV, and the next one lowers the RSS by {gain2}",
    ),
    "stop_eps": (
        "rule",
        "F6",
        "{head}; the best legacy candidate by GCV, {chosen}, lowers the RSS by "
        "{gain} of {rss_before}, not more than machine epsilon, an absolute test "
        "(the next best lowers it by {gain2})",
    ),
    "stop_earth_rules": (
        "rule",
        "F6",
        "{head}; earth stops by its relative rules (termination code {code}), "
        "which the legacy code lacks",
    ),
    "stop_other": ("unexplained", None, "{head}; legacy stop {why}, earth code {code}"),
    "structure": (
        "rule",
        "F5",
        "step {step}: earth adds the one term {earth}, the legacy code the pair "
        "{legacy}",
    ),
    "rss_span": (
        "rule",
        "F5",
        "step {step}: RSS {rss_l} (legacy) against {rss_e}: the legacy pair "
        "spans more than earth's single hinge",
    ),
    "rss_numeric": (
        "numeric",
        None,
        "step {step}: RSS {rss_l} against {rss_e}, relative {rel}, kappa {kappa}",
    ),
    "intercept_final": (
        "bug",
        "F3",
        "the legacy final model has {terms} terms and no intercept: its pruning "
        "pass removed the intercept at size {size}",
    ),
    "intercept_path": (
        "bug",
        "F3",
        "the legacy pruning path drops the intercept at size {size} and below; "
        "the selected model, with {selected} terms, keeps it",
    ),
    "prune_subset": (
        "rule",
        None,
        "size {size}: earth's subset has RSS {rss_e}, the legacy path's {rss_l}",
    ),
    "prune_tie": ("tie", None, "size {size}: the subsets differ with RSS within 1e-7"),
    "size_tie": (
        "tie",
        None,
        "the same subsets and GCVs, but {legacy} selected terms against {earth}: "
        "tied GCVs go to the larger model in the legacy code (strict <), to the "
        "smaller one in earth (which.min)",
    ),
    "gcv_convention": (
        "rule",
        "F4",
        "at size {size} both pruning paths keep the same subset, with {h} hinge "
        "and {lin} linear terms, but the legacy C = M + d*H = {c_l} (d = {d_l} "
        "for each hinge term, none for a linear term) and earth's C = M + "
        "d*(M - 1)/2 = {c_e} (d = {d_e}), GCV {gcv_l} against {gcv_e}; the legacy "
        "code selects {legacy} terms and earth {earth}",
    ),
    "constant_y": (
        "quirk",
        "F19",
        "a constant response: at thresh = 0 earth's forward pass adds terms on "
        "rounding noise (forward steps here: earth {n_earth}, the legacy code "
        "{n_legacy}), and its pruning pass removes them all",
    ),
    "final": ("{label}", None, "{field}: {detail} {metric} above {tol}"),
    "weights": (
        "rule",
        "F1",
        "the 1.0.4 wheel's Earth.fit takes no sample_weight (HEAD takes it)",
    ),
    "responses": ("rule", "F18", "three responses: the legacy fit raises {err}"),
    "error": ("unexplained", None, "{err}"),
    "classes": (
        "bug",
        "F9",
        "the legacy classifier regresses the integer class codes where earth "
        "fits one indicator response per class",
    ),
    "glm": (
        "bug",
        "F9",
        "GLM refit: the legacy logistic fit is L2-penalized (C = 1); its fitted "
        "probabilities differ by up to {diff_own} from an unpenalized refit on "
        "the same basis, and by up to {diff} from earth's glm on {same} terms",
    ),
    "categorical": (
        "bug",
        "F17",
        "legacy categorical_features on string labels: {terms} term(s), fitted "
        "values off earth's factor fit by up to {diff}",
    ),
}


def verdict(key: str, **ev: Any) -> Verdict:
    """The (label, finding, sentence) of VERDICTS[key], filled from ``ev``;
    floats are printed with 8 significant digits."""
    label, finding, text = VERDICTS[key]
    fmt = {k: (f"{v:.8g}" if isinstance(v, float) else v) for k, v in ev.items()}
    return label.format(**fmt), finding, text.format(**fmt)


def sig_from_json(sig: Iterable[Sequence[Any]]) -> Sig:
    """A signature from its JSON form (a list of [var, code, knot])."""
    return tuple(sorted(((int(f[0]), int(f[1]), f[2]) for f in sig), key=_fkey))


def _fkey(f: Factor) -> tuple[int, int]:
    return (f[0], f[1])


def sig_key(sig: Sig) -> tuple:
    return tuple(
        (v, c, -np.inf if k is None or isinstance(k, str) else k) for v, c, k in sig
    )


def term_label(sig: Sig) -> str:
    """earth-style label: h(x0-0.5), h(0.5-x0), x0, factors joined by *."""
    if not sig:
        return "(Intercept)"
    names = {1: "h(x{v}-{k:.6g})", -1: "h({k:.6g}-x{v})", 2: "x{v}", 3: "x{v}=={k}"}
    return "*".join(names[c].format(v=v, k=k) for v, c, k in sig)


def terms_label(sigs: Iterable[Sig]) -> str:
    return " + ".join(term_label(s) for s in sorted(sigs, key=sig_key))


def term_column(sig: Sig, X: np.ndarray) -> np.ndarray:
    """Term ``sig`` evaluated at the rows of X (TERM-3; code 3 an indicator)."""
    col = np.ones(X.shape[0])
    for var, code, knot in sig:
        x = X[:, var]
        if code == 1:
            col = col * np.maximum(0.0, x - knot)
        elif code == -1:
            col = col * np.maximum(0.0, knot - x)
        else:
            col = col * (x if code == 2 else (x == knot))
    return col


def basis_matrix(sigs: Iterable[Sig], X: np.ndarray) -> np.ndarray:
    return np.column_stack([term_column(s, X) for s in sigs])


def rss(B: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    """The (weighted) least-squares RSS of y on the columns of B (LA-1)."""
    sw = np.ones_like(y) if w is None else np.sqrt(w)
    coef = np.linalg.lstsq(B * sw[:, None], y * sw, rcond=None)[0]
    return float(np.sum((sw * (y - B @ coef)) ** 2))


def earth_signatures(dirs: Any, cuts: Any) -> list[Sig]:
    """One signature per row of earth's ``dirs``/``cuts`` (TERM-1)."""
    dirs, cuts = np.asarray(dirs), np.asarray(cuts, dtype=float)
    return [
        tuple(
            (int(v), int(dirs[r, v]), None if dirs[r, v] == 2 else float(cuts[r, v]))
            for v in np.flatnonzero(dirs[r])
        )
        for r in range(dirs.shape[0])
    ]


def earth_steps(dirs: Any) -> list[list[int]]:
    """earth's forward steps as groups of ``dirs`` rows after the intercept: a
    +1 row and the next row form a pair when the next row is its mirror (the
    same but for one code +1 -> -1, FWD-6); any other row is a one-term step
    (a single hinge, a linear term or the linear option)."""
    dirs = np.asarray(dirs)
    steps, row = [], 1
    while row < dirs.shape[0]:
        if row + 1 < dirs.shape[0]:
            diff = np.flatnonzero(dirs[row] != dirs[row + 1])
            if diff.size == 1 and (dirs[row, diff[0]], dirs[row + 1, diff[0]]) == (
                1,
                -1,
            ):
                steps.append([row, row + 1])
                row += 2
                continue
        steps.append([row])
        row += 1
    return steps


def compare_forward(
    legacy: Sequence[frozenset[Sig]], earth: Sequence[frozenset[Sig]]
) -> dict[str, Any]:
    """Compare two forward passes step by step, as sets of new terms. A step
    is ``same`` when both add the same terms, ``subset`` when earth adds one
    term of the legacy code's pair (F5), and a choice divergence otherwise,
    where the comparison stops (VALIDATION_PLAN.md, "Ties")."""
    relations: list[str] = []
    first_choice = None
    for k, (a, e) in enumerate(zip(legacy, earth, strict=False)):
        if a != e and not e < a:
            first_choice = k
            break
        relations.append("same" if a == e else "subset")
    first_subset = next((k for k, r in enumerate(relations) if r == "subset"), None)
    return {
        "n_legacy": len(legacy),
        "n_earth": len(earth),
        "relations": relations,
        "first_subset": first_subset,
        "first_choice": first_choice,
    }


def is_near_tie(a: float | None, b: float | None, before: float | None) -> bool:
    if a is None or b is None or not before:
        return False
    return abs(a - b) < NEAR_TIE_REL * abs(before)


def classify_choice(ev: dict[str, Any]) -> Verdict:
    """Label a choice divergence at the 1-based forward step ``ev["step"]``.

    ``ev`` holds both choices (``legacy``, ``earth``: labels); the legacy
    code's own RSS and GCV for both (``leg_rss_a``, ``leg_gcv_a``,
    ``leg_rss_e``, ``leg_gcv_e``, ``leg_before``; ``leg_rss_e`` is None when
    earth's choice was not a legacy candidate, and ``leg_knot`` then says
    why); the RSS of both on earth's basis (``e_rss_a``, ``e_rss_e``,
    ``e_before``); ``same_cost``, whether the legacy GCV charges both the
    same, so that it ranks them by RSS; ``fast_k``; and ``probe``, the earth
    trace status of the legacy's choice. The near-tie rule comes first:
    either choice passes when the two RSS are within 1e-7 of the RSS before
    the step, on either program's basis.
    """
    ev = {
        **ev,
        "head": f"step {ev['step']}: legacy {ev['legacy']}, earth {ev['earth']}",
    }
    ev["rss"] = (
        f"RSS {_g(ev.get('e_rss_a'))} (legacy choice) against "
        f"{_g(ev.get('e_rss_e'))} (earth choice) on earth's basis"
    )
    legacy_tie = ev.get("same_cost") and is_near_tie(
        ev.get("leg_rss_a"), ev.get("leg_rss_e"), ev.get("leg_before")
    )
    if legacy_tie:  # show the values that make the tie, at 10 digits
        a, e = ev.get("leg_rss_a"), ev.get("leg_rss_e")
        rss = f"the legacy code's RSS {a:.10g} (its choice) against {e:.10g} (earth's)"
        return verdict("tie", **{**ev, "rss": rss})
    if is_near_tie(ev.get("e_rss_a"), ev.get("e_rss_e"), ev.get("e_before")):
        a, e = ev["e_rss_a"], ev["e_rss_e"]
        rss = (
            f"RSS {a:.10g} (legacy choice) against {e:.10g} (earth's) on earth's basis"
        )
        return verdict("tie", **{**ev, "rss": rss})
    probe = ev.get("probe") or {}
    status = probe.get("status")
    e_a, e_e = ev.get("e_rss_a"), ev.get("e_rss_e")
    if (
        not ev.get("earth_stopped")
        and e_a is not None
        and e_e is not None
        and e_e < e_a
    ):
        # earth's choice has the lower RSS, so the legacy code passed over it
        if ev.get("leg_rss_e") is None:
            key = "inactive_earth" if ev.get("e_knot_inactive") else "not_legacy"
            return verdict(key, why=ev.get("leg_knot") or "no matching candidate", **ev)
        if not ev.get("same_cost") and ev["leg_rss_e"] < (
            ev.get("leg_rss_a") or np.inf
        ):
            return verdict(
                "gcv_rank",
                gcv_a=_g(ev.get("leg_gcv_a")),
                rss_a=_g(ev.get("leg_rss_a")),
                gcv_e=_g(ev.get("leg_gcv_e")),
                rss_e=_g(ev["leg_rss_e"]),
                **ev,
            )
        return verdict("unexplained_choice", status=f"{status}, earth's lower", **ev)
    # the legacy choice has the lower RSS (or earth stopped): why earth passed
    # over it is in earth's trace
    if status == "parent_not_in_earth":
        return verdict("mirror_parent", **ev)
    if status == "parent_not_searched":
        return verdict("slot" if ev.get("fast_k") == 0 else "queue", **ev)
    if status == "knot_not_evaluated":
        where = probe.get("where")
        key = where if where in ("top", "bottom_inactive", "inactive") else "grid"
        return verdict(key, where=where, **ev)
    if status == "tol":
        return verdict("tol", **ev)
    if status == "maxlegal":
        return verdict("maxlegal", max_legal=_g(probe.get("max_legal")), **ev)
    # A single-hinge search explains the divergence only when the legacy pair
    # beats earth's traced single-hinge RSS for that knot (5 digits in the
    # trace): otherwise the second hinge is redundant and the RSS the same.
    single_rss = probe.get("earth_rss") or 0.0
    if status == "single_search" and (e_a or np.inf) < single_rss * (1 - 1e-4):
        return verdict("single", **ev)
    if ev.get("earth_stopped"):
        return verdict("unexplained_choice", status=f"{status}, earth stopped", **ev)
    return verdict("unexplained_choice", status=status, **ev)


def classify_stop(ev: dict[str, Any]) -> Verdict:
    """Label a difference in the number of forward steps when every common
    step agreed; ``ev["first"]`` is the side that stopped first."""
    head = f"forward steps: legacy {ev['n_legacy']}, earth {ev['n_earth']}"
    why, code = ev.get("legacy_stop"), ev.get("termcond")
    if ev["first"] == "legacy" and why in ("gcv_inf", "eps", "no_candidate"):
        gain, before = ev.get("gain"), ev.get("rss_before")
        if why == "eps" and gain is not None and abs(gain) <= ZERO_GAIN_REL * before:
            why = "zero_gain"  # a term already in the span; not an epsilon matter
        fmt = {k: _g(ev.get(k)) for k in ("gain", "gain2", "rss_before")}
        return verdict(f"stop_{why}", head=head, chosen=ev.get("chosen"), **fmt)
    if ev["first"] == "earth" and code in (2, 3, 4, 5):
        return verdict("stop_earth_rules", head=head, code=code)
    return verdict("stop_other", head=head, why=why, code=code)


def _g(x: float | None) -> str:
    return "none" if x is None else f"{x:.8g}"


def count_by(items: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    """Counts of ``item[key]`` (None counted under its label in brackets)."""
    counts: dict[str, int] = {}
    for it in items:
        k = it.get(key) or f"({it.get('label')})"
        counts[k] = counts.get(k, 0) + 1
    return dict(sorted(counts.items()))
