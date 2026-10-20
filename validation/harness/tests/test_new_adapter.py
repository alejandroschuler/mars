"""Tests for new_adapter.py. tests/test_conformance.py runs the adapter on every
earth fixture, where the forward pass keeps every term; this file covers the
numbering when FWD-11 drops a term, which no fixture has. No R needed."""

from types import SimpleNamespace

import numpy as np
from new_adapter import mars_fit_to_common


def _fit_with_a_dropped_term():
    """A CORE-5 dict: step 1 adds the pair h(x0-0.5), h(0.5-x0) (forward
    terms 1 and 2), step 2 the single hinge h(x1-0.2) (term 3), which FWD-11
    drops, and step 3 the pair h(x0-0.5)*h(x1-0.7), h(x0-0.5)*h(0.7-x1)
    (terms 4 and 5) on parent 1. The pruning pass sees the kept terms 0, 1, 2,
    4 and 5 as pruning indices 0 to 4, and selects 0, 1 and 4 (forward 0, 1
    and 5)."""
    dirs = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [1, 1], [1, -1]], dtype=np.int8)
    cuts = np.array([[0, 0], [0.5, 0], [0.5, 0], [0, 0.2], [0.5, 0.7], [0.5, 0.7]])
    subsets = np.tril(np.ones((5, 5), dtype=bool))
    subsets[2] = [True, True, False, False, True]  # T[3] is not a prefix
    return {
        "dirs": dirs[[0, 1, 5]],
        "cuts": cuts[[0, 1, 5]],
        "coef": np.array([[1.0], [2.0], [-3.0]]),
        "selected": np.array([0, 1, 5]),
        "rss": 4.0,
        "gcv": 0.05,
        "rsq": 0.9,
        "grsq": 0.8,
        "n_eff": 50.0,
        "max_terms": 21,
        "penalty": 2.0,
        "forward": {
            "dirs": dirs,
            "cuts": cuts,
            "kept": np.array([0, 1, 2, 4, 5]),
            "dropped": np.array([3]),
            "parent": np.array([-1, 0, 0, 0, 1, 1]),
            "step": np.array([0, 1, 1, 2, 3, 3]),
            "rss": np.array([40.0, 10.0, 9.0, 4.0]),
            "termination": 7,
            "candidates": {
                "best_rss": np.array([10.0, 9.0, 4.0]),
                "second_rss": np.array([11.0, np.inf, 4.5]),
            },
        },
        "pruning": {
            "removed": np.array([3, 2, 4, 1]),
            "rss_per_size": np.array([40.0, 12.0, 4.0, 3.9, 3.8]),
            "gcv_per_size": np.array([0.9, 0.3, 0.05, 0.06, 0.07]),
            "subsets": subsets,
            "selected_size": 3,
        },
    }


def test_the_numbering_follows_earth_when_a_term_is_dropped():
    # CORE-3: pruning index m is forward index kept[m]; earth numbers its
    # forward terms after its rank fix from 1 (driver.py's schema), so the
    # common numbering is the pruning index plus 1 (FWD-11, PRUNE-2, PRUNE-4).
    common = mars_fit_to_common(_fit_with_a_dropped_term())
    assert common["dirs"] == [[0, 0], [1, 0], [-1, 0], [1, 1], [1, -1]]
    assert common["cuts"][3] == [0.5, 0.7]
    assert common["selected_terms"] == [1, 2, 5]
    assert common["prune_terms"][2] == [1, 2, 5, 0, 0]
    assert common["prune_terms"][4] == [1, 2, 3, 4, 5]
    assert common["pruning_removed"] == [4, 3, 5, 2]
    assert common["termcond"] == 7
    assert common["forward_rss"] == [40.0, 10.0, 9.0, 4.0]
    first, dropped, last = common["forward_steps"]
    assert first["rows"] == [1, 2] and first["direction"] == frozenset({1, -1})
    assert (first["parent"], first["pred"], first["knot"]) == (0, 0, 0.5)
    assert dropped["rows"] == [] and dropped["direction"] == frozenset({1})
    assert last["rows"] == [3, 4] and (last["parent"], last["pred"]) == (1, 1)
    assert (last["knot"], last["rss_before"]) == (0.7, 9.0)
    assert (dropped["best_rss"], dropped["second_best_rss"]) == (9.0, np.inf)


def test_a_mars_fit_goes_through_its_dict():
    # CORE-5: MarsFit.to_dict() gives the dict that the reference returns.
    d = _fit_with_a_dropped_term()
    fit = SimpleNamespace(to_dict=lambda: d)
    assert mars_fit_to_common(fit) == mars_fit_to_common(d)
