"""Tests for the description of the legacy code (T18): the triage logic of
``validation/legacy/legacy_triage.py`` and the analysis in
``conformance_legacy.py``, on small hand-made records and a short synthetic
``trace = 9`` text; one ``external`` test runs the script end to end (it
needs R with earth and ``.venv-legacy``)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

LEGACY = Path(__file__).resolve().parents[2] / "legacy"
sys.path.insert(0, str(LEGACY))

import conformance_legacy as cl  # noqa: E402
import legacy_triage as lt  # noqa: E402

H = (0, 1, 0.5)  # h(x0-0.5)
M = (0, -1, 0.5)  # h(0.5-x0)
H3, M3 = (0, 1, 0.3), (0, -1, 0.3)
PAIR = frozenset({(H,), (M,)})
X1 = np.arange(1, 9, dtype=float).reshape(-1, 1) / 10  # 0.1 to 0.8
Y1 = np.array([0.0, 0.1, 0.0, 0.2, 0.5, 0.9, 1.2, 1.6])


def test_signature_and_label():
    sig = lt.sig_from_json([[1, -1, 0.25], [0, 2, None]])
    assert sig == ((0, 2, None), (1, -1, 0.25))
    assert lt.term_label(sig) == "x0*h(0.25-x1)"
    assert lt.term_label(()) == "(Intercept)"
    assert lt.terms_label(PAIR) == "h(0.5-x0) + h(x0-0.5)"


def test_term_column_evaluates_every_factor_kind():
    X = np.array([[0.2, 3.0], [0.9, 1.0]])
    assert lt.term_column((H,), X).tolist() == [0.0, pytest.approx(0.4)]
    assert lt.term_column((M, (1, 2, None)), X).tolist() == [pytest.approx(0.9), 0.0]
    assert lt.term_column(((1, 3, 1.0),), X).tolist() == [0.0, 1.0]
    assert lt.basis_matrix([(), (H,)], X).shape == (2, 2)


def test_rss_is_weighted_least_squares():
    rng = np.random.default_rng(0)
    B, y = np.column_stack([np.ones(20), rng.uniform(size=20)]), rng.normal(size=20)
    w = rng.integers(1, 4, size=20).astype(float)
    rep = np.repeat(np.arange(20), w.astype(int))
    assert lt.rss(B, y, w) == pytest.approx(lt.rss(B[rep], y[rep]), rel=1e-12)


def test_earth_steps_groups_pairs_and_single_terms():
    dirs = [[0, 0], [1, 0], [-1, 0], [1, 0], [0, 2], [1, 1], [1, -1]]
    cuts = [[0, 0], [0.5, 0], [0.5, 0], [0.7, 0], [0, 0], [0.5, 0.3], [0.5, 0.3]]
    assert lt.earth_steps(dirs) == [[1, 2], [3], [4], [5, 6]]
    sigs = lt.earth_signatures(dirs, cuts)
    assert sigs[0] == () and sigs[4] == ((1, 2, None),)
    assert sigs[6] == ((0, 1, 0.5), (1, -1, 0.3))


def test_compare_forward_relations():
    single = frozenset({(H,)})
    other = frozenset({((0, 1, 0.7),), ((0, -1, 0.7),)})
    fc = lt.compare_forward([PAIR, PAIR, other], [PAIR, single, PAIR, PAIR])
    assert fc["relations"] == ["same", "subset"]
    assert (fc["first_subset"], fc["first_choice"]) == (1, 2)
    fc = lt.compare_forward([PAIR], [PAIR, PAIR])
    assert (fc["first_choice"], fc["n_legacy"], fc["n_earth"]) == (None, 1, 2)


def test_near_tie_threshold_is_relative_to_the_rss_before():
    assert lt.is_near_tie(10.0, 10.0 + 9e-7, 10.0)
    assert not lt.is_near_tie(10.0, 10.0 + 2e-6, 10.0)
    assert not lt.is_near_tie(None, 1.0, 1.0)


def _ev(**kw):
    """Evidence where the legacy choice has the lower RSS on earth's basis."""
    base = {"step": 3, "legacy": "A", "earth": "E", "e_rss_a": 1.0, "e_rss_e": 2.0}
    return {**base, "e_before": 5.0, "leg_rss_a": 1.0, "leg_before": 5.0, **kw}


EARTH_LOWER = {"e_rss_e": 0.5}  # earth's choice has the lower RSS


@pytest.mark.parametrize(
    ("ev", "expected", "phrase"),
    [
        (_ev(e_rss_e=1.0 + 1e-7), ("tie", None), "near-tie"),
        (_ev(leg_rss_e=1.0 + 1e-7, same_cost=True), ("tie", None), "near-tie"),
        (
            _ev(probe={"status": "parent_not_in_earth"}),
            ("rule", "F5"),
            "redundant second hinge",
        ),
        (
            _ev(probe={"status": "parent_not_searched"}, fast_k=0),
            ("quirk", None),
            "(FAST-4)",
        ),
        (
            _ev(probe={"status": "parent_not_searched"}, fast_k=20),
            ("rule", "F10"),
            "Fast MARS queue",
        ),
        (
            _ev(probe={"status": "knot_not_evaluated", "where": "top"}),
            ("rule", "F7"),
            "largest active value",
        ),
        (
            _ev(probe={"status": "knot_not_evaluated", "where": "bottom_inactive"}),
            ("quirk", "F7"),
            "lower endspan counts cases",
        ),
        (
            _ev(probe={"status": "knot_not_evaluated", "where": "span"}),
            ("rule", "F7"),
            "knot grid (span)",
        ),
        (
            _ev(probe={"status": "knot_not_evaluated", "where": "inactive"}),
            ("quirk", "F7"),
            "the case above the legacy knot has the parent at zero",
        ),
        (_ev(probe={"status": "tol"}), ("rule", "F5"), "tolerance (TolG 0)"),
        (
            _ev(probe={"status": "maxlegal", "max_legal": 3.0}),
            ("quirk", None),
            "MaxLegal 3",
        ),
        (
            _ev(probe={"status": "single_search", "earth_rss": 1.1}),
            ("rule", "F5"),
            "single-hinge search",
        ),
        (
            _ev(probe={"status": "single_search", "earth_rss": 1.00005}),
            ("unexplained", None),
            "probe single_search",
        ),
        (
            _ev(probe={"status": "legal"}, leg_rss_e=1.5, same_cost=True),
            ("unexplained", None),
            "probe legal",
        ),
        (
            _ev(probe={"status": "tol"}, earth_stopped=True),
            ("rule", "F5"),
            "tolerance (TolG 0)",
        ),
        (
            _ev(probe={"status": "legal"}, earth_stopped=True),
            ("unexplained", None),
            "earth stopped",
        ),
        (
            _ev(**EARTH_LOWER, leg_rss_e=None, leg_knot="knot 0.7 is out"),
            ("rule", "F7"),
            "not a legacy candidate (knot 0.7 is out)",
        ),
        (
            _ev(**EARTH_LOWER, leg_rss_e=None, e_knot_inactive=True),
            ("quirk", "F7"),
            "the value of a case where the parent is zero",
        ),
        (
            _ev(**EARTH_LOWER, probe={"status": "tol"}, leg_rss_e=None),
            ("rule", "F7"),
            "not a legacy candidate",
        ),
        (
            _ev(**EARTH_LOWER, leg_rss_e=0.5, same_cost=False, leg_gcv_a=1.0),
            ("rule", "F2"),
            "ranks by GCV: its choice has GCV 1",
        ),
        (
            _ev(**EARTH_LOWER, leg_rss_e=1.2, same_cost=True),
            ("unexplained", None),
            "earth's lower",
        ),
    ],
)
def test_classify_choice(ev, expected, phrase):
    label, finding, text = lt.classify_choice(ev)
    assert (label, finding) == expected
    assert text.startswith("step 3: legacy A, earth E")
    assert phrase in text


def test_a_tie_shows_the_values_that_make_it():
    text = lt.classify_choice(_ev(leg_rss_e=1.0 + 1e-8, same_cost=True))[2]
    assert "the legacy code's RSS 1 (its choice) against 1.00000001" in text
    text = lt.classify_choice(_ev(e_rss_e=1.0 + 1e-7))[2]
    assert "RSS 1 (legacy choice) against 1.0000001 (earth's) on earth's basis" in text


def test_classify_stop():
    ev = {"n_legacy": 2, "n_earth": 4, "first": "legacy"}
    assert lt.classify_stop({**ev, "legacy_stop": "gcv_inf"})[:2] == ("rule", "F2")
    assert lt.classify_stop({**ev, "legacy_stop": "no_candidate"})[:2] == ("rule", "F2")
    assert lt.classify_stop({**ev, "legacy_stop": "term_limit"})[0] == "unexplained"
    eps = {**ev, "legacy_stop": "eps", "chosen": "x0", "rss_before": 1e-17}
    label, finding, text = lt.classify_stop({**eps, "gain": 2e-18, "gain2": 3e-18})
    assert (label, finding) == ("rule", "F6")
    assert "lowers the RSS by 2e-18 of 1e-17, not more than machine epsilon" in text
    zero = {**eps, "rss_before": 0.5, "gain": -5e-17, "gain2": 0.03}
    label, finding, text = lt.classify_stop(zero)
    assert (label, finding) == ("rule", "F2")
    assert "x0, is already in the model's span" in text and "by 0.03" in text
    earth = {"n_legacy": 4, "n_earth": 2, "first": "earth"}
    for code in (2, 3, 4, 5):
        label, finding, text = lt.classify_stop({**earth, "termcond": code})
        assert (label, finding) == ("rule", "F6")
        assert f"termination code {code}" in text
    assert lt.classify_stop({**earth, "termcond": 6})[0] == "unexplained"


def test_verdict_fills_every_template():
    fields = {"head", "rss", "where", "max_legal", "why", "gcv_a", "rss_a", "gcv_e"}
    fields |= {"rss_e", "status", "code", "step", "earth", "legacy", "rss_l", "rel"}
    fields |= {"kappa", "size", "total", "field", "detail", "metric", "tol"}
    fields |= {"err", "diff", "diff_own", "same", "terms", "n_earth", "n_legacy"}
    fields |= {"chosen", "gain", "gain2", "rss_before", "selected", "h", "lin"}
    fields |= {"c_l", "c_e", "d_l", "d_e", "gcv_l"}
    for key in lt.VERDICTS:
        label, _, text = lt.verdict(key, **dict.fromkeys(fields, 1.5), label="rule")
        assert label in {"rule", "bug", "quirk", "tie", "numeric", "unexplained"}
        assert "{" not in text


def test_count_by():
    items = [{"label": "rule", "finding": "F5"}, {"label": "tie", "finding": None}]
    assert lt.count_by(items, "finding") == {"(tie)": 1, "F5": 1}


def test_legacy_modes_convert_the_earth_arguments():
    args = {"degree": 2, "penalty": 3, "nk": 21, "minspan": 1, "endspan": 1}
    kw = cl.legacy_matched({**args, "Auto.linpreds": False})
    assert kw == {
        "max_degree": 2,
        "penalty": 1.5,
        "max_terms": 21,
        "minspan": 1,
        "endspan": 0,
        "allow_linear": False,
    }
    assert cl.legacy_spans({"degree": 1, "minspan": 5, "endspan": 10}) == {
        "max_degree": 1,
        "minspan": 5,
        "endspan": 9,
    }
    auto = cl.legacy_spans({"degree": 1})
    assert auto["minspan_alpha"] == auto["endspan_alpha"] == 0.05


def test_build_cases_covers_s01_to_s20():
    cases = {c.id: c for c in cl.build_cases()}
    datasets = {c.dataset[:3] for c in cases.values()}
    assert datasets == {f"S{i:02d}" for i in range(1, 21)}
    assert sum(c.dataset == "S15" for c in cases.values()) == 200
    d2 = cases["S04_p10_n1000/matched_d2"]
    assert d2.earth_args["Adjust.endspan"] == 0 and d2.legacy["penalty"] == 1.0
    assert "S05/matched_d2_adjust1" not in cases
    # the legacy matched mode: no larger endspan for interactions (SPAN-4)
    matched = [c for c in cases.values() if c.mode == "matched" and c.degree > 1]
    assert len([c for c in matched if c.dataset == "S15"]) == 50
    assert all(c.earth_args["Adjust.endspan"] == 0 for c in matched)
    weighted = cases["S13_int_zeros/matched_d1"]
    assert (
        weighted.weighted and weighted.reference == "S13_int_zeros_repeated_matched_d1"
    )
    assert cases["S13_nonint/matched_d1"].reference == "S13_nonint_matched_d1"
    assert "minspan" not in cases["S14/matched_d2"].legacy


# --- hand-made fits for the analysis -------------------------------------------


def dirs_cuts(rows, p):
    """earth's dirs and cuts for rows of signatures, intercept first."""
    dirs = [[0] * p for _ in rows]
    cuts = [[0.0] * p for _ in rows]
    for r, sig in enumerate(rows):
        for var, code, knot in sig:
            dirs[r][var] = code
            cuts[r][var] = 0.0 if knot is None else knot
    return dirs, cuts


def earth_rec(rows, p=1, **kw):
    dirs, cuts = dirs_cuts(rows, p)
    base = {"selected_terms": [1], "gcv": 1.0, "rsq": 0.5, "pred_test": None}
    return {"dirs": dirs, "cuts": cuts, **base, **kw}


def leg_rec(terms, parents, steps, **kw):
    rec = {
        "terms": [
            {"sig": [list(f) for f in s], "parent": p}
            for s, p in zip(terms, parents, strict=True)
        ],
        "steps": steps,
        "selected": [0],
        "gcv": 1.0,
        "rss": 1.0,
        "pred_test": None,
        "prune_subsets": [[0]],
    }
    return {**rec, **kw}


def case(dataset="T", degree=1, legacy=None, **earth_args):
    c = cl.Case(f"{dataset}/test", dataset, "matched", degree, legacy or {})
    c.earth_args = {"degree": degree, **earth_args}
    return c


def inputs(X, y):
    return {"X": X, "y": y, "X_test": None, "weights": None, "truth": None}


def kinds(row):
    return [(d["kind"], d["step"], d["label"], d["finding"]) for d in row["diffs"]]


def test_analyze_labels_a_constant_response_f19():
    y = np.full(8, 3.0)
    earth = earth_rec([(), (H,), (M,)], fwd_rss=[0.0, 0.0, 0.0])
    search = {"n_cands": 7, "n_terms_before": 1, "rss_before": 0.0, "chosen": None}
    leg = leg_rec([()], [None], [], searches=[search], fwd_rss=[0.0])
    row = cl.analyze(case(), earth, leg, None, inputs(X1, y), inputs(X1, y))
    assert kinds(row) == [("stop", 1, "quirk", "F19")]
    assert "earth 1, the legacy code 0" in row["diffs"][0]["text"]


def test_analyze_labels_an_epsilon_stop_f6_and_not_f19():
    # Y1 is not constant, although Y1[0] appears again (Y1[2])
    earth = earth_rec(
        [(), (H,), (M,), (H3,), (M3,)], fwd_rss=[1e-16, 0, 1e-17, 0, 5e-18]
    )
    chosen = {"rss": 8e-18, "gcv": 1.0, "kind": "pair", "new": [[list(M3)], [list(H3)]]}
    last = {"n_cands": 5, "n_terms_before": 3, "rss_before": 1e-17, "chosen": chosen}
    last["second"] = {**chosen, "rss": 9e-18}
    leg = leg_rec(
        [(), (M,), (H,)],
        [None, 0, 0],
        [[1, 2]],
        fwd_rss=[1e-16, 1e-17],
        searches=[{}, last],
    )
    row = cl.analyze(case(), earth, leg, None, inputs(X1, Y1), inputs(X1, Y1))
    assert kinds(row) == [("stop", 2, "rule", "F6")]
    assert (
        "h(0.3-x0) + h(x0-0.3), lowers the RSS by 2e-18 of 1e-17"
        in row["diffs"][0]["text"]
    )


def test_analyze_labels_a_single_hinge_and_the_span_it_leaves_out():
    earth = earth_rec([(), (H,), (M,), (H3,)], fwd_rss=[2.0, 1.5, 1.0, 0.8])
    leg = leg_rec(
        [(), (M,), (H,), (M3,), (H3,)],
        [None, 0, 0, 0, 0],
        [[1, 2], [3, 4]],
        fwd_rss=[2.0, 1.0, 0.7],
    )
    row = cl.analyze(case(), earth, leg, None, inputs(X1, Y1), inputs(X1, Y1))
    assert kinds(row) == [("structure", 2, "rule", "F5"), ("rss", 2, "rule", "F5")]
    assert (
        "earth adds the one term h(x0-0.3), the legacy code the pair"
        in row["diffs"][0]["text"]
    )
    assert (
        "RSS 0.7 (legacy) against 0.8: the legacy pair spans more"
        in row["diffs"][1]["text"]
    )
    assert row["forward"]["rss_steps"] == [2, 1]


ROWS = [(), (H,), (M,), (H3,), (M3,)]  # earth: pairs at 0.5 and 0.3
EARTH_PRUNE = {
    "prune_terms": [
        [1, 0, 0, 0, 0],
        [1, 2, 0, 0, 0],
        [1, 2, 4, 0, 0],
        [1, 2, 3, 4, 0],
        [1, 2, 3, 4, 5],
    ],
    "rss_per_subset": [10.0, 6.0, 4.0, 3.0, 2.9],
    "gcv_per_subset": [1.0, 0.8, 0.7, 0.75, 0.9],
    "selected_terms": [1, 2, 4],
    "coef": [[0.5], [1.0], [2.0]],
    "gcv": 0.7,
}


def pruned(subsets, rss, gcvs, selected, **kw):
    """A legacy record with ROWS as its forward terms (the same order) and a
    pruning path from the full model down."""
    return leg_rec(
        ROWS,
        [None, 0, 0, 0, 0],
        [[1, 2], [3, 4]],
        prune_subsets=subsets,
        prune_rss=rss,
        prune_gcv=gcvs,
        selected=selected,
        **kw,
    )


def run_pruning(leg, earth=None, legacy=None):
    earth = earth or earth_rec(ROWS, **EARTH_PRUNE)
    f = cl.Fit(case(legacy=legacy), earth, leg, None, inputs(X1, Y1), inputs(X1, Y1))
    row = {"diffs": []}

    def add(kind, step, v):
        row["diffs"].append(
            {"kind": kind, "step": step, "label": v[0], "finding": v[1], "text": v[2]}
        )

    cl.pruning(add, row, f)
    return row


FULL = [[0, 1, 2, 3, 4], [0, 1, 2, 3], [0, 1, 3]]


def test_pruning_intercept_missing_from_the_final_model_comes_first():
    leg = pruned(
        [*FULL, [1, 3], [1]], [2.9, 3, 4, 5, 8], [0.9, 0.75, 0.7, 0.6, 1.2], [1, 3]
    )
    row = run_pruning(leg)
    assert kinds(row) == [("pruning", None, "bug", "F3")]
    text = row["diffs"][0]["text"]
    assert text == (
        "the legacy final model has 2 terms and no intercept: its pruning pass "
        "removed the intercept at size 2"
    )


def test_pruning_gcv_convention_comes_before_the_intercept_on_the_path():
    gcv = [1.2, 0.9, 0.72, 0.65, 1.1]  # sizes 5 to 1; earth's differ
    row = run_pruning(pruned([*FULL, [0, 1], [1]], [2.9, 3, 4, 6, 8], gcv, [0, 1]))
    assert kinds(row) == [
        ("pruning", None, "rule", "F4"),
        ("pruning", None, "bug", "F3"),
    ]
    assert row["diffs"][0]["text"] == (
        "at size 3 both pruning paths keep the same subset, with 2 hinge and 0 "
        "linear terms, but the legacy C = M + d*H = 9 (d = 3 for each hinge term, "
        "none for a linear term) and earth's C = M + d*(M - 1)/2 = 5 (d = 2), GCV "
        "0.72 against 0.7; the legacy code selects 2 terms and earth 3"
    )
    assert (
        "drops the intercept at size 1 and below; the selected model, with 2"
        in (row["diffs"][1]["text"])
    )


def test_pruning_other_subset_at_the_selected_size():
    subsets = [[0, 1, 2, 3, 4], [0, 1, 2, 3], [0, 1, 2], [0, 1], [0]]
    gcv = [0.9, 0.75, 0.65, 0.8, 1.0]  # the same as earth's where subsets agree
    row = run_pruning(pruned(subsets, [2.9, 3, 4.5, 6, 10], gcv, [0, 1, 2]))
    assert kinds(row) == [("pruning", None, "rule", None)]
    assert row["diffs"][0]["text"] == (
        "size 3: earth's subset has RSS 4, the legacy path's 4.5"
    )


def test_pruning_same_path_and_model_has_no_difference():
    subsets = [[0, 1, 2, 3, 4], [0, 1, 2, 3], [0, 1, 3], [0, 1], [0]]
    fitted = np.linspace(0, 1, 8)
    earth = earth_rec(ROWS, **EARTH_PRUNE, fitted=fitted.reshape(-1, 1).tolist())
    leg = pruned(
        subsets,
        [2.9, 3, 4, 6, 10],
        [0.9, 0.75, 0.7, 0.8, 1.0],
        [0, 1, 3],
        coef=[0.5, 1.0, 2.0],
        gcv=0.7,
        pred_train=fitted.tolist(),
    )
    row = run_pruning(leg, earth)
    assert row["diffs"] == [] and row["pruning"] == []


def test_final_labels_a_final_model_without_intercept():
    earth = earth_rec(ROWS, selected_terms=[1, 2])
    leg = leg_rec(ROWS, [None, 0, 0, 0, 0], [[1, 2], [3, 4]], selected=[1, 2])
    leg["prune_subsets"] = [[0, 1, 2, 3, 4], [1, 2, 3], [1, 2]]
    f = cl.Fit(case(), earth, leg, None, inputs(X1, Y1), inputs(X1, Y1))
    row = cl.final({"diffs": []}, f)
    assert kinds(row) == [("pruning", None, "bug", "F3")]
    assert row["diffs"][0]["text"].endswith("removed the intercept at size 3")
    assert row["final"]["intercept"] is False


# --- two covariates: inactive knots, evidence and the trace probe ---------------

X2 = np.column_stack([X1[:, 0], [0.35, 0.15, 0.55, 0.25, 0.85, 0.45, 0.75, 0.65]])
Y2 = np.array([0.1, 0.0, 0.2, 0.1, 1.0, 0.4, 0.8, 0.6])
P = (0, 1, 0.4)  # the parent h(x0-0.4): positive at x0 = 0.5 to 0.8
PARENT_ROWS = [(), (P,), ((0, -1, 0.4),)]


def two_step_fit(knot, kind="pair"):
    """earth's step 2 is the pair P*h(x1-knot), P*h(knot-x1); the legacy
    code's is the pair at x1 = 0.75."""
    earth_rows = [*PARENT_ROWS, (P, (1, 1, knot)), (P, (1, -1, knot))]
    earth = earth_rec(earth_rows, p=2)
    legacy_rows = [*PARENT_ROWS, (P, (1, -1, 0.75)), (P, (1, 1, 0.75))]
    chosen = {"rss": 1.0, "gcv": 2.0, "kind": "pair"}
    search = {
        "rss_before": 3.0,
        "chosen": chosen,
        "earth_match": {**chosen, "kind": kind},
    }
    leg = leg_rec(
        legacy_rows, [None, 0, 0, 1, 1], [[1, 2], [3, 4]], searches=[{}, search]
    )
    return cl.Fit(case(), earth, leg, None, inputs(X2, Y2), inputs(X2, Y2))


def test_inactive_knot_is_the_value_of_a_case_where_the_parent_is_zero():
    for knot, expected in ((0.35, True), (0.65, False), (0.15, True), (0.85, False)):
        f = two_step_fit(knot)
        assert cl.inactive_knot(f, 1, f.e_steps[1]) is expected


def test_evidence_of_a_choice():
    ev = cl.evidence(two_step_fit(0.35), 1)
    assert ev["same_cost"] is True and ev["e_knot_inactive"] is True
    assert ev["legacy"] == "h(x0-0.4)*h(0.75-x1) + h(x0-0.4)*h(x1-0.75)"
    assert ev["earth"] == "h(x0-0.4)*h(0.35-x1) + h(x0-0.4)*h(x1-0.35)"
    assert ev["e_before"] > ev["e_rss_a"] and ev["fast_k"] is None
    assert cl.evidence(two_step_fit(0.35, kind="linear"), 1)["same_cost"] is False


BEGIN = (  # the FindKnotBegin line of a search, with its fixed spans
    "--FindKnotBegin-- iPred {} iNewCol {} RssBeforeAddingHinge {} "
    "nMinSpan 1 nEndSpan 1 nStartSpan 1"
)
CASE = (  # an evaluated case: case, RSS, RSS delta, cut, TolG
    "--FindKnot--Case {} RssWithKnot {} RssDelta {} Cut {} "
    "bx1G 1 CovColG 1 TolG {} MaxG 1"
)
TRACE = f"""\
|FindTerm: Searching for new term 2 RssDelta 0 MaxLegalRssDelta 7.1
|Parent 1 Pred 1 Case -1 Cut 0.1< Rss 5.0 RssDeltaLin 2.0
{BEGIN.format(1, 3, 7)}
{CASE.format(4, 3.0, 4.0, 0.4, 1)} best
--FindKnotEnd--
|Parent 1 Pred 1 Case 4 Cut 0.4 Rss 3.0 RssDelta 4.0 best for term
|FindTerm: Searching for new term 4 RssDelta 4 MaxLegalRssDelta 3.03
|Parent 2 Pred 2 Case -1 Cut 0.15< Rss 2.8 RssDeltaLin 0.2
{BEGIN.format(2, 5, 3)}
{CASE.format(7, 2.5, 0.5, 0.75, 1)} best
--FindKnot--Case 6 iSpan 1 bx1 0
{CASE.format(5, 2.9, 0.1, 0.55, 0)}
--FindKnot--Case 4 iSpan 1 bx1 0.3
--FindKnotEnd--
|Parent 2 Pred 2 Case 7 Cut 0.75 Rss 2.5 RssDelta 0.5 best for term
"""


@pytest.mark.parametrize(
    ("knot", "status", "where"),
    [
        (0.75, "legal", None),  # Case 7 is the knot sorted(x1)[6]
        (0.55, "tol", None),  # Case 5 has TolG 0
        (0.65, "knot_not_evaluated", "inactive"),  # visited, bx1 0
        (0.45, "knot_not_evaluated", "span"),  # visited, bx1 0.3
        (0.85, "knot_not_evaluated", "top"),  # the largest active x1
        (0.25, "knot_not_evaluated", "bottom_inactive"),  # below every visited case
    ],
)
def test_probe_choice_reads_the_trace(tmp_path, knot, status, where):
    trace = tmp_path / "trace.txt"
    trace.write_text(TRACE)
    earth_rows = [*PARENT_ROWS, (P, (1, 1, 0.75)), (P, (1, -1, 0.75))]
    res = dict(zip(("dirs", "cuts"), dirs_cuts(earth_rows, 2), strict=True))
    legacy_rows = [*PARENT_ROWS, (P, (1, -1, knot)), (P, (1, 1, knot))]
    leg = leg_rec(legacy_rows, [None, 0, 0, 1, 1], [[1, 2], [3, 4]])
    probe = cl.probe_choice(trace, res, leg, 2, inputs(X2, Y2))
    assert probe["status"] == status and probe.get("where") == where
    assert (
        probe["parent"] == "h(x0-0.4)" and probe["var"] == 1 and probe["knot"] == knot
    )
    assert probe["search"] == "pair"
    assert probe["max_legal"] == pytest.approx(3.03 * np.var(Y2, ddof=1))


def test_probe_prefix_check():
    full = {"dirs": [[0], [1], [-1], [1]], "cuts": [[0], [0.5], [0.5], [0.3]]}
    cl.check_probe_prefix(
        {"dirs": full["dirs"][:3], "cuts": full["cuts"][:3]}, full, "x"
    )
    with pytest.raises(RuntimeError, match="first 3 terms differ"):
        cl.check_probe_prefix(
            {"dirs": full["dirs"][:3], "cuts": [[0], [0.5], [0.4]]}, full, "x"
        )


def test_glm_numbers_measure_the_penalty_on_the_legacy_basis():
    from sklearn.linear_model import LogisticRegression

    x = np.linspace(-1, 1, 40)
    X, y = x.reshape(-1, 1), (x > 0.1).astype(float)
    rows = [(), ((0, 1, 0.1),), ((0, -1, 0.1),)]
    B = lt.basis_matrix(rows, X)
    proba = (
        LogisticRegression(solver="lbfgs", random_state=0).fit(B, y).predict_proba(B)
    )
    earth = earth_rec(rows, selected_terms=[1, 2, 3], pred_train=proba[:, 1:].tolist())
    leg = leg_rec(
        rows, [None, 0, 0], [[1, 2]], selected=[0, 1, 2], proba_train=proba.tolist()
    )
    f = cl.Fit(case("S20"), earth, leg, None, inputs(X, y), inputs(X, y))
    own, diff, same = cl.glm_numbers(f)
    assert own > 0.3  # the penalized refit keeps the probabilities away from 0 and 1
    assert diff == 0.0 and same is True


def test_legacy_stop():
    c = cl.Case("x", "S01", "matched", 1, {"max_terms": 21})
    fit = type("F", (), {"case": c, "X": np.zeros((50, 1))})()
    search = {"n_cands": 10, "n_terms_before": 5, "rss_before": 1e-3, "chosen": None}
    fit.leg = {"searches": [search]}
    assert cl.legacy_stop(fit) == "gcv_inf"
    for gain, expected in (
        (2.2e-19, "eps"),
        (0.0, "eps"),
        (-1e-19, "eps"),
        (1e-15, "other"),
    ):
        fit.leg = {"searches": [{**search, "chosen": {"rss": 1e-3 - gain}}]}
        assert cl.legacy_stop(fit) == expected, gain
    fit.leg = {"searches": [{**search, "n_cands": 0, "n_terms_before": 21}]}
    assert cl.legacy_stop(fit) == "term_limit"
    fit.leg = {"searches": [{**search, "n_cands": 0, "n_terms_before": 3}]}
    assert cl.legacy_stop(fit) == "no_candidate"


def test_divergence_step():
    leg = leg_rec([(), (H,), (M,)], [None, 0, 0], [[1, 2]])
    earth = {"dirs": [[0], [1], [-1], [1]], "cuts": [[0], [0.5], [0.5], [0.7]]}
    assert cl.divergence_step(leg, earth) is None  # earth took one more step
    assert cl.divergence_step(leg, {"dirs": [[0], [1]], "cuts": [[0], [0.3]]}) == 1
    earth = {"dirs": [[0], [1], [-1]], "cuts": [[0], [0.5], [0.5]]}
    two = leg_rec([(), (H,), (M,), (H3,), (M3,)], [None, 0, 0, 0, 0], [[1, 2], [3, 4]])
    assert cl.divergence_step(two, earth) == 2  # the legacy code took one more step


def test_summarize_counts_first_and_all_differences():
    diff = {"kind": "choice", "step": 1, "label": "rule", "finding": "F5", "text": ""}
    stop = {**diff, "kind": "stop", "step": 3, "finding": "F6"}
    rows = [
        {"id": "S15/draw000", "mode": "matched", "diffs": [diff]},
        {"id": "S15/draw001", "mode": "matched", "diffs": []},
        {"id": "S15/draw002", "mode": "matched", "diffs": [stop]},
        {"id": "S01/x", "mode": "defaults", "diffs": [{**diff, "label": "tie"}, diff]},
    ]
    s = cl.summarize(rows)
    assert s["first_by_label"] == {"rule": 2, "tie": 1}
    assert s["all_by_label"] == {"rule": 3, "tie": 1}
    assert s["s15"]["matched"]["no_choice_divergence"] == 1
    assert s["s15"]["matched"]["causes"] == {"rule F5": 1, "rule F6": 1}
    assert s["s15"]["matched"]["first_divergence_step"] == {"1": 1, "3": 1}


def test_report_has_one_case_per_line(tmp_path):
    result = {
        "command": "c",
        "summary": {"fits": 2},
        "cases": [{"id": "a"}, {"id": "b"}],
    }
    path = tmp_path / "r.json"
    cl.write_report(path, result)
    lines = path.read_text().splitlines()
    assert json.loads(path.read_text()) == result
    assert '{"id": "a"},' in lines and '{"id": "b"}' in lines and len(lines) == 8


def test_markdown_tables():
    diff = {"kind": "choice", "step": 1, "label": "rule", "finding": "F5", "text": ""}
    rows = [
        {"id": "S01/matched_d1", "code": "wheel", "mode": "matched", "diffs": [diff]},
        {"id": "S15/draw000", "code": "wheel", "mode": "matched", "diffs": [diff]},
    ]
    rows[0]["final"] = {"legacy_terms": 8, "earth_terms": 6}
    result = {"summary": cl.summarize(rows), "cases": rows}
    result["head_vs_wheel"] = {"identical": 3, "different": []}
    text = cl.markdown(result)
    assert (
        "| S01/matched_d1 | wheel | choice at step 1 | rule | F5 | none | |  | 8, 6 |"
        in text
    )
    assert "S15/draw000" not in text
    assert "| matched | 1 | 0 | 1: 1 | rule F5: 1 |" in text
    assert text.endswith("HEAD and the wheel: 3 identical fits; different: none.\n")
    assert sorted(["F10", "(tie)", "F2"], key=cl._finding_order) == [
        "F2",
        "F10",
        "(tie)",
    ]


@pytest.mark.external
def test_the_script_runs_end_to_end_on_s01(tmp_path):
    only = "^S01/matched_d1$"
    script = [sys.executable, str(LEGACY / "conformance_legacy.py")]
    for cmd in ("run", "probe"):
        subprocess.run(
            [*script, cmd, "--out", str(tmp_path), "--only", only], check=True
        )
    out = tmp_path / "report.json"
    subprocess.run(
        [*script, "report", "--out", str(tmp_path), "--only", only, "--json", str(out)],
        check=True,
    )
    (row,) = json.loads(out.read_text())["cases"]
    kinds = [(d["kind"], d["step"], d["label"], d["finding"]) for d in row["diffs"]]
    assert kinds == [("structure", 2, "rule", "F5"), ("choice", 3, "rule", "F5")]
