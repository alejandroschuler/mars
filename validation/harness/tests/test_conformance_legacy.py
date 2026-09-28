"""Tests for the description of the legacy code (T18): the triage logic of
``validation/legacy/legacy_triage.py`` and the pure parts of
``conformance_legacy.py``; one ``external`` test runs the script end to end
(it needs R with earth and ``.venv-legacy``)."""

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
PAIR = frozenset({(H,), (M,)})


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
    base = {"step": 3, "legacy": "A", "earth": "E", "e_rss_a": 1.0, "e_rss_e": 2.0}
    return {**base, "e_before": 5.0, "leg_rss_a": 1.0, "leg_before": 5.0, **kw}


@pytest.mark.parametrize(
    ("ev", "expected"),
    [
        (_ev(e_rss_e=1.0 + 1e-7), ("tie", None)),
        (_ev(leg_rss_e=1.0 + 1e-7, same_cost=True), ("tie", None)),
        (_ev(probe={"status": "parent_not_in_earth"}), ("rule", "F5")),
        (_ev(probe={"status": "parent_not_searched"}, fast_k=0), ("quirk", "F21")),
        (_ev(probe={"status": "parent_not_searched"}, fast_k=20), ("rule", "F10")),
        (_ev(probe={"status": "knot_not_evaluated", "where": "top"}), ("rule", "F7")),
        (
            _ev(probe={"status": "knot_not_evaluated", "where": "bottom_inactive"}),
            ("quirk", "F7"),
        ),
        (_ev(probe={"status": "knot_not_evaluated", "where": "span"}), ("rule", "F7")),
        (_ev(probe={"status": "tol"}), ("rule", "F5")),
        (_ev(probe={"status": "maxlegal", "max_legal": 3.0}), ("quirk", "F19")),
        (_ev(probe={"status": "single_search"}), ("rule", "F5")),
        (_ev(probe={"status": "legal"}, leg_rss_e=None), ("rule", "F7")),
        (
            _ev(probe={"status": "linear"}, leg_rss_e=0.5, same_cost=False),
            ("rule", "F2"),
        ),
        (
            _ev(probe={"status": "legal"}, leg_rss_e=1.5, same_cost=True),
            ("unexplained", None),
        ),
        (_ev(probe={"status": "legal"}, earth_stopped=True), ("unexplained", None)),
    ],
)
def test_classify_choice(ev, expected):
    label, finding, text = lt.classify_choice(ev)
    assert (label, finding) == expected
    assert text.startswith("step 3: legacy A, earth E")


def test_classify_stop():
    ev = {"n_legacy": 2, "n_earth": 4, "first": "legacy"}
    assert lt.classify_stop({**ev, "legacy_stop": "gcv_inf"})[:2] == ("rule", "F2")
    assert lt.classify_stop({**ev, "legacy_stop": "eps"})[:2] == ("rule", "F6")
    earth = {"n_legacy": 4, "n_earth": 2, "first": "earth"}
    assert lt.classify_stop({**earth, "termcond": 4})[:2] == ("rule", "F6")
    assert lt.classify_stop({**earth, "termcond": 6})[0] == "unexplained"


def test_verdict_fills_every_template():
    fields = {"head", "rss", "where", "max_legal", "why", "gcv_a", "rss_a", "gcv_e"}
    fields |= {"rss_e", "status", "code", "step", "earth", "legacy", "rss_l", "rel"}
    fields |= {"kappa", "size", "total", "field", "detail", "metric", "tol"}
    fields |= {"err", "diff", "diff_own", "same", "terms"}
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
    assert cases["S05/matched_d2"].earth_args["Adjust.endspan"] == 0
    assert "S05/matched_d2_adjust1" not in cases
    weighted = cases["S13_int_zeros/matched_d1"]
    assert (
        weighted.weighted and weighted.reference == "S13_int_zeros_repeated_matched_d1"
    )
    assert cases["S13_nonint/matched_d1"].reference == "S13_nonint_matched_d1"
    assert "minspan" not in cases["S14/matched_d2"].legacy


def test_divergence_step_and_legacy_stop():
    sig = [[0, 1, 0.5]]
    leg = {
        "terms": [{"sig": []}, {"sig": sig}, {"sig": [[0, -1, 0.5]]}],
        "steps": [[1, 2]],
    }
    earth = {"dirs": [[0], [1], [-1], [1]], "cuts": [[0], [0.5], [0.5], [0.7]]}
    assert cl.divergence_step(leg, earth) is None  # earth took one more step
    earth = {"dirs": [[0], [1]], "cuts": [[0], [0.3]]}
    assert cl.divergence_step(leg, earth) == 1
    case = cl.Case("x", "S01", "matched", 1, {"max_terms": 21})
    fit = type("F", (), {"case": case, "X": np.zeros((50, 1))})()
    search = {"n_cands": 10, "n_terms_before": 5, "rss_before": 2.0, "chosen": None}
    fit.leg = {"searches": [search]}
    assert cl.legacy_stop(fit) == "gcv_inf"
    fit.leg = {"searches": [{**search, "chosen": {"rss": 2.0 - 1e-17}}]}
    assert cl.legacy_stop(fit) == "eps"
    fit.leg = {"searches": [{**search, "n_cands": 0, "n_terms_before": 21}]}
    assert cl.legacy_stop(fit) == "term_limit"


def test_summarize_counts_first_and_all_differences():
    diff = {"kind": "choice", "step": 1, "label": "rule", "finding": "F5", "text": ""}
    rows = [
        {"id": "S15/draw000", "mode": "matched", "diffs": [diff]},
        {"id": "S15/draw001", "mode": "matched", "diffs": []},
        {"id": "S01/x", "mode": "defaults", "diffs": [{**diff, "label": "tie"}, diff]},
    ]
    s = cl.summarize(rows)
    assert s["first_by_label"] == {"rule": 1, "tie": 1}
    assert s["all_by_label"] == {"rule": 2, "tie": 1}
    assert s["s15"]["matched"]["no_choice_divergence"] == 1
    assert s["s15"]["matched"]["causes"] == {"rule F5": 1}


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
