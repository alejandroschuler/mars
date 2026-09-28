"""Convert a pymars 2.0 fit to the common result schema of ``driver.py``.

The input is a ``MarsFit`` (``pymars._core``, docs/algorithm.md CORE-3) or
the nested dict of CORE-5: the dict that the reference implementation
(``tests/reference/mars_ref.fit_mars``) returns and that
``MarsFit.to_dict()`` gives back. A ``MarsFit`` goes through ``to_dict()``,
so the conversion reads one shape only.

The common schema follows earth's conventions, so that ``compare.py`` can
compare the two sides directly:

- ``dirs``, ``cuts``: the M_f terms that the forward pass kept (FWD-11), in
  forward order, as earth's ``dirs`` and ``cuts`` hold the forward terms
  after its rank fix. pymars stores the cut 0.0 for a linear factor, where
  earth stores the smallest x; compare cuts only where ``dirs`` is 1 or -1
  (TERM-2).
- ``selected_terms``, ``prune_terms`` and ``pruning_removed`` number these
  terms from 1, as earth does: ``selected_terms`` is the selected pruning
  indices plus 1 (PRUNE-5, PRUNE-7); row m - 1 of the (M_f, M_f) matrix
  ``prune_terms`` holds the terms of T[m] in increasing order, padded with
  0 (PRUNE-4); ``pruning_removed`` is the removed term of each stage.
- ``coef`` (M, K), ``rss``, ``gcv``, ``rsq``, ``grsq``, ``n_eff``,
  ``max_terms``, ``penalty``, ``rss_per_subset``, ``gcv_per_subset`` and
  ``termcond`` (the termination code of CORE-4, earth's ``termcond``).
- ``forward_rss``: the RSS before the first step and after each step, S + 1
  values (``ForwardRecord.rss``). earth's ``fwd_rss`` in ``driver.py``'s
  schema has one value per term instead, so the two keys differ.
- ``forward_steps``: one dict per forward step, in the shape of
  ``compare.compare_forward_steps``: ``rows`` (the positions in ``dirs`` of
  the kept terms of the step), ``parent`` (the position of the parent in
  ``dirs``, or None if the parent was dropped), ``pred``, ``direction`` (the
  frozenset of the codes that the step added at ``pred``: {1, -1} for a
  pair, {1} for a single hinge or a linear option with
  ``auto_linpreds=False``, {2} for a linear term), ``knot`` (None for a
  linear term), ``best_rss`` and ``second_best_rss`` (from the candidate
  log, or None without one), ``rss_before`` and ``flags`` (None: earth's
  flags come from its trace only).

For a linear term, ``compare.steps_from_trace`` gives earth's stored cut
(the smallest x) as the knot, not None, so ``compare_forward_steps`` on the
two logs reports every linear-term step as a mismatch. A caller compares the
knot only where the codes are 1 or -1 (TERM-2), as tests/test_conformance.py
does.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def mars_fit_to_common(fit: Any) -> dict[str, Any]:
    """The common-schema dict of a ``MarsFit`` or of its CORE-5 dict.

    Complexity: O(M_f² + S·M_a·p), the size of the records.
    """
    d = fit.to_dict() if hasattr(fit, "to_dict") else fit
    forward, pruning = d["forward"], d["pruning"]
    kept = np.asarray(forward["kept"], dtype=np.int64)
    position = {int(k): m for m, k in enumerate(kept)}
    subsets = np.asarray(pruning["subsets"], dtype=bool)
    Mf = len(kept)
    prune_terms = np.zeros((Mf, Mf), dtype=np.int64)
    for m in range(Mf):
        terms = np.flatnonzero(subsets[m]) + 1
        prune_terms[m, : len(terms)] = terms
    return {
        "dirs": np.asarray(forward["dirs"])[kept].astype(int).tolist(),
        "cuts": np.asarray(forward["cuts"], dtype=float)[kept].tolist(),
        "selected_terms": [position[int(k)] + 1 for k in d["selected"]],
        "prune_terms": prune_terms.tolist(),
        "pruning_removed": [int(t) + 1 for t in pruning["removed"]],
        "rss_per_subset": _floats(pruning["rss_per_size"]),
        "gcv_per_subset": _floats(pruning["gcv_per_size"]),
        "coef": np.asarray(d["coef"], dtype=float).tolist(),
        "rss": float(d["rss"]),
        "gcv": float(d["gcv"]),
        "rsq": float(d["rsq"]),
        "grsq": float(d["grsq"]),
        "n_eff": float(d["n_eff"]),
        "max_terms": int(d["max_terms"]),
        "penalty": float(d["penalty"]),
        "termcond": int(forward["termination"]),
        "forward_rss": _floats(forward["rss"]),
        "forward_steps": _forward_steps(forward, position),
    }


def _floats(values: Any) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=float)]


def _forward_steps(forward: dict, position: dict[int, int]) -> list[dict[str, Any]]:
    """The steps of the forward record, with positions among the kept terms
    (TERM-6: ``step`` and ``parent`` of each term; CORE-3: the candidate
    log)."""
    dirs = np.asarray(forward["dirs"])
    cuts = np.asarray(forward["cuts"], dtype=float)
    step = np.asarray(forward["step"])
    parent = np.asarray(forward["parent"])
    rss = np.asarray(forward["rss"], dtype=float)
    log = forward.get("candidates")
    steps = []
    for s in range(1, len(rss)):
        terms = np.flatnonzero(step == s)
        first = int(terms[0])
        par = int(parent[first])
        pred = int(np.flatnonzero(dirs[first] != dirs[par])[0])
        codes = frozenset(int(dirs[t, pred]) for t in terms)
        steps.append(
            {
                "rows": [position[int(t)] for t in terms if int(t) in position],
                "parent": position.get(par),
                "pred": pred,
                "direction": codes,
                "knot": None if codes == {2} else float(cuts[first, pred]),
                "best_rss": None if log is None else float(log["best_rss"][s - 1]),
                "second_best_rss": None
                if log is None
                else float(log["second_rss"][s - 1]),
                "rss_before": float(rss[s - 1]),
                "flags": None,
            }
        )
    return steps
