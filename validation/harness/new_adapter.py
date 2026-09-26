"""Convert a `MarsFit` (VALIDATION_PLAN.md, "Core API") to the common result
schema `driver.py`'s module docstring documents.

`pymars._core` does not exist yet (a later task, T12, builds `fit_mars` and
`MarsFit` and connects this adapter to it), so this module works from the
plan's field list alone: `dirs`, `cuts`, `coef` (shape (M, K)), `rss`, `gcv`,
`rsq`, `grsq`, `n_eff`, a forward record (the terms, the RSS after each
step, the termination code, an optional candidate log) and a pruning record
(the term removed at each step, the RSS and GCV for each size, the selected
terms). The plan does not fix attribute names for the two sub-records, so
this module assumes a shape (`fit.forward.steps`, `fit.pruning.removed`, and
so on, all documented on `mars_fit_to_common` below) and reads every field
through `_get`, which accepts either an object or a dict ("the reference
returns the same fields as a dict"). T07 checks this assumed shape against
`docs/algorithm.md` and against `pymars._core` once T12 lands it.
"""

from __future__ import annotations

from typing import Any


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field whether `obj` is an object (`getattr`) or a dict
    (`.get`), so this module needs no assumption about which `MarsFit` is."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _to_nested_list(value: Any) -> list | None:
    """A (M, p) or (M, K) array-like to a plain nested list, or None."""
    if value is None:
        return None
    return [list(row) for row in value]


def mars_fit_to_common(fit: Any) -> dict[str, Any]:
    """Convert a `MarsFit` (an object or a dict with the same field names)
    to the common schema.

    Assumed shape of the two sub-records, beyond the plan's own field list:

    - `fit.forward` (or `fit["forward"]`): `.steps`, a list of one entry per
      forward step, each with `.parent`, `.pred`, `.knot` and `.direction`
      (the `dirs` codes this step added, as a collection: `{1, -1}` for a
      hinge pair, `{1}` or `{-1}` for a single hinge, `{2}` for a linear
      term -- compare.py's `compare_forward_steps` compares this as a
      `frozenset`, so a single bare code also works and is coerced to one);
      `.rss`, the RSS after each step; `.termcond`, the termination code;
      and, when `record_candidates` was requested, `.candidates`, a list of
      `(best_rss, second_best_rss)` pairs, one per step (merged onto each
      step's own dict below, so the result needs no separate zipping).
    - `fit.pruning` (or `fit["pruning"]`): `.removed`, the term index taken
      out at each pruning step, in order; `.rss`/`.gcv`, one value per
      subset size; `.selected`, the selected term indices.

    Any of these that is absent is `None` in the result, not an error, so
    this adapter keeps working if `pymars._core`'s actual shape adds fields
    this module does not yet know about.
    """
    forward = _get(fit, "forward")
    pruning = _get(fit, "pruning")
    forward_steps = _get(forward, "steps")
    candidates = _get(forward, "candidates")

    return {
        "dirs": _to_nested_list(_get(fit, "dirs")),
        "cuts": _to_nested_list(_get(fit, "cuts")),
        "selected_terms": list(_get(pruning, "selected"))
        if _get(pruning, "selected") is not None
        else None,
        "prune_terms": None,  # no fixed-shape subset-membership matrix in the plan
        "coef": _to_nested_list(_get(fit, "coef")),
        "rss": _get(fit, "rss"),
        "gcv": _get(fit, "gcv"),
        "rsq": _get(fit, "rsq"),
        "grsq": _get(fit, "grsq"),
        "n_eff": _get(fit, "n_eff"),
        "termcond": _get(forward, "termcond"),
        "rss_per_subset": _list_or_none(_get(pruning, "rss")),
        "gcv_per_subset": _list_or_none(_get(pruning, "gcv")),
        "pruning_removed": _list_or_none(_get(pruning, "removed")),
        "fwd_rss": _list_or_none(_get(forward, "rss")),
        "forward_steps": _build_forward_steps(forward_steps, candidates),
        "forward_candidates": _list_or_none(candidates),
    }


def _build_forward_steps(
    forward_steps: Any, candidates: Any
) -> list[dict[str, Any]] | None:
    if forward_steps is None:
        return None
    candidates = list(candidates) if candidates is not None else []
    steps = []
    for i, step in enumerate(forward_steps):
        direction = _get(step, "direction")
        if direction is not None and not isinstance(direction, frozenset | set):
            direction = (
                frozenset(direction)
                if _is_iterable(direction)
                else frozenset({direction})
            )
        best_rss, second_best_rss = (
            candidates[i] if i < len(candidates) else (None, None)
        )
        steps.append(
            {
                "parent": _get(step, "parent"),
                "pred": _get(step, "pred"),
                "direction": direction,
                "knot": _get(step, "knot"),
                "best_rss": best_rss,
                "second_best_rss": second_best_rss,
                "rss_before": _get(step, "rss_before"),
                "flags": _get(step, "flags"),
            }
        )
    return steps


def _is_iterable(value: Any) -> bool:
    try:
        iter(value)
    except TypeError:
        return False
    return True


def _list_or_none(value: Any) -> list | None:
    return None if value is None else list(value)
