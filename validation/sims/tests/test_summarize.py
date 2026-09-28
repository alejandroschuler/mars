"""Tests of validation/sims/summarize.py: cell-name parsing and the
regression tests for one real bug found while running the smoke run
(``df.noise == None`` is false for every row in pandas, which silently hid
D6 and the binary DGPs, which have no noise level, from every display).

Also hand-computed cases for the ratio, its interval, the equivalence band,
the selection table's median/IQR, a missing display, and the exclusion of a
failed repetition from the aggregate: a review found that none of these were
checked against a known number, so an interval of +/-2 SE, a mean of ratios
in place of the geometric-mean ratio, a band of log 1.10, a median taken as a
mean, a missing cell shown as 1.000, and a failed repetition kept in the
ratio all passed every test that existed before this file changed.

Also the two figures, drawn with the installed matplotlib. Before these
tests no test drew a figure, so a matplotlib floor that was too old for
``box_plots`` passed every gate and CI job.
"""

from __future__ import annotations

import json
import math

import pytest

from validation.sims import summarize as summ


def _write_result(base, cell_name, arm, rep, data):
    path = base / cell_name / arm / f"rep{rep:04d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_parse_cell_name_with_and_without_noise():
    assert summ._parse_cell_name("D4_n00200_lo") == ("D4", 200, "lo")
    assert summ._parse_cell_name("D6_n00200") == ("D6", 200, None)
    assert summ._parse_cell_name("D3-bin_n01000") == ("D3-bin", 1000, None)


def test_parse_cell_name_skips_non_cell_directories():
    assert summ._parse_cell_name("report") is None
    assert summ._parse_cell_name(".cache") is None


def test_noise_mask_handles_none_correctly(tmp_path):
    _write_result(
        tmp_path, "D6_n00200", "E-def", 0, {"error": None, "excess_risk": 1.0}
    )
    _write_result(
        tmp_path, "D1_n00200_lo", "E-def", 0, {"error": None, "excess_risk": 2.0}
    )
    df = summ.load_results(tmp_path)
    assert summ._noise_mask(df, None).sum() == 1
    assert summ._noise_mask(df, "lo").sum() == 1


def test_ratio_table_and_selection_table_find_d6_data(tmp_path):
    for rep in range(2):
        _write_result(
            tmp_path,
            "D6_n00200",
            "E-def",
            rep,
            {
                "error": None,
                "excess_risk": 1.0,
                "n_terms": 1,
                "uses_irrelevant_covariate": False,
                "n_irrelevant_covariates": 0,
            },
        )
        _write_result(
            tmp_path,
            "D6_n00200",
            "P-cur",
            rep,
            {
                "error": None,
                "excess_risk": 2.0,
                "n_terms": 2,
                "uses_irrelevant_covariate": True,
                "n_irrelevant_covariates": 1,
            },
        )
    df = summ.load_results(tmp_path)
    ratios = summ.ratio_table(df)
    d6_row = ratios[ratios["DGP"] == "D6 pure noise"].iloc[0]
    assert d6_row["P-cur / E-def"] != summ.MISSING
    assert "2.000" in d6_row["P-cur / E-def"]

    selection = summ.selection_table(df)
    d6_e_def = selection[
        (selection["DGP"] == "D6 pure noise") & (selection["arm"] == "E-def")
    ].iloc[0]
    assert d6_e_def["median_terms"] == 1.0


def test_load_results_ignores_a_report_subdirectory(tmp_path):
    _write_result(
        tmp_path, "D1_n00200_lo", "E-def", 0, {"error": None, "excess_risk": 1.0}
    )
    (tmp_path / "report").mkdir()
    df = summ.load_results(tmp_path)
    assert len(df) == 1


def test_equivalence_margin_is_log_1_05():
    assert pytest.approx(math.log(1.05)) == summ.EQUIVALENCE_MARGIN
    assert pytest.approx(math.log(1.10)) != summ.EQUIVALENCE_MARGIN


def test_paired_log_ratio_hand_case(tmp_path):
    """E-def excess_risk = [1.0, 1.0]; P-cur excess_risk = [2.0, 8.0]:
    g_1 = log 2, g_2 = log 8, g_bar = log 4, s_g = log(4)/sqrt(2),
    se = s_g/sqrt(2) = log(4)/2 = log 2. ratio = exp(g_bar) = 4.
    interval = exp(g_bar +/- 3*se) = exp(log4 -/+ 3 log2) = [0.5, 32].
    A +/-2 SE interval (a review's mutation) would give [1, 16] instead.
    """
    for rep, (e_def_risk, p_cur_risk) in enumerate([(1.0, 2.0), (1.0, 8.0)]):
        _write_result(
            tmp_path,
            "D1_n00200_lo",
            "E-def",
            rep,
            {"error": None, "excess_risk": e_def_risk},
        )
        _write_result(
            tmp_path,
            "D1_n00200_lo",
            "P-cur",
            rep,
            {"error": None, "excess_risk": p_cur_risk},
        )
    df = summ.load_results(tmp_path)
    stats = summ.paired_log_ratio(df, "D1", 200, "lo", "P-cur", "E-def", "excess_risk")
    assert stats["n_sim"] == 2
    assert stats["se"] == pytest.approx(math.log(2))
    assert stats["ratio"] == pytest.approx(4.0)
    assert stats["ratio_lo"] == pytest.approx(0.5)
    assert stats["ratio_hi"] == pytest.approx(32.0)

    formatted = summ._format_ratio_cell(stats)
    assert "4.000" in formatted
    assert "0.500" in formatted
    assert "32.000" in formatted


def test_paired_log_ratio_excludes_a_failed_repetition(tmp_path):
    """P-cur has 3 repetitions on disk, but repetition 2 recorded a failure
    (no excess_risk at all); only repetitions 0 and 1 are paired. Keeping the
    failed repetition (for example by filling a missing excess_risk with 0
    or 1) would change g_bar and n_sim, both checked here.
    """
    for rep in (0, 1):
        _write_result(
            tmp_path, "D1_n00200_lo", "E-def", rep, {"error": None, "excess_risk": 1.0}
        )
        _write_result(
            tmp_path, "D1_n00200_lo", "P-cur", rep, {"error": None, "excess_risk": 2.0}
        )
    _write_result(
        tmp_path, "D1_n00200_lo", "E-def", 2, {"error": None, "excess_risk": 1.0}
    )
    _write_result(tmp_path, "D1_n00200_lo", "P-cur", 2, {"error": "RuntimeError: boom"})
    df = summ.load_results(tmp_path)
    stats = summ.paired_log_ratio(df, "D1", 200, "lo", "P-cur", "E-def", "excess_risk")
    assert stats["n_sim"] == 2  # not 3
    assert stats["ratio"] == pytest.approx(2.0)  # not pulled toward a filled-in value
    assert stats["n_fail_a"] == 1  # P-cur's one recorded failure, counted


def test_ratio_table_missing_when_no_data_at_all():
    """Never invents a number (VALIDATION_PLAN.md/the brief): a review tried
    showing a missing cell as "1.000" (a ratio of 1, "no gap") and found no
    test caught it.
    """
    import pandas as pd

    empty = pd.DataFrame(
        columns=["dgp", "n", "noise", "arm", "rep", "error", "excess_risk"]
    )
    table = summ.ratio_table(empty)
    d1_row = table[table["DGP"] == "D1 linear"].iloc[0]
    assert d1_row["P-cur / E-def"] == summ.MISSING
    assert d1_row["P-cur / E-def"] != "1.000"
    d7_row = table[table["DGP"] == "D7 many covariates"].iloc[0]
    assert d7_row["P-cur / E-def"].startswith("n/a")


def test_selection_table_median_and_iqr_hand_case(tmp_path):
    """n_terms = [1, 1, 1, 1, 10] (an asymmetric sample, pandas' default
    linear interpolation): median 1, mean 2.8, Q1 1, Q3 1, IQR 0. A mean
    computed in place of the median would give 2.8, not 1: this is the
    asymmetric case a review asked for, since [1, 2, 3, 4, 5]'s median and
    mean coincide (3.0 either way) and so cannot tell the two apart.
    uses_irrelevant_covariate = [True, False, False, False, False] and
    n_irrelevant_covariates = [2, 0, 0, 0, 0]: a mean gives 0.2 and 0.4; a max
    (also tried) would give 1 and 2 instead.
    """
    n_terms_values = [1, 1, 1, 1, 10]
    uses_irrelevant = [True, False, False, False, False]
    n_irrelevant_values = [2, 0, 0, 0, 0]
    for rep in range(5):
        _write_result(
            tmp_path,
            "D4_n00200_lo",
            "E-def",
            rep,
            {
                "error": None,
                "n_terms": n_terms_values[rep],
                "uses_irrelevant_covariate": uses_irrelevant[rep],
                "n_irrelevant_covariates": n_irrelevant_values[rep],
            },
        )
    df = summ.load_results(tmp_path)
    table = summ.selection_table(df)
    row = table[
        (table["DGP"] == "D4 Friedman #1")
        & (table["arm"] == "E-def")
        & (table["noise"] == "lo")
    ].iloc[0]
    assert row["median_terms"] == pytest.approx(1.0)
    assert row["median_terms"] != pytest.approx(2.8)  # rules out a mean
    assert row["iqr_terms"] == pytest.approx(0.0)
    assert row["share_irrelevant"] == pytest.approx(0.2)
    assert row["share_irrelevant"] != pytest.approx(1.0)  # rules out a max
    assert row["mean_n_irrelevant"] == pytest.approx(0.4)
    assert row["mean_n_irrelevant"] != pytest.approx(2.0)  # rules out a max


def test_binary_outcome_table_uses_excess_log_loss_not_excess_risk(tmp_path):
    """A review's repro: a mutation that computed the binary table's ratio
    from excess_risk instead of excess_log_loss passed every existing test.
    Gives each measure a distinct, hand-chosen value so the two cannot agree
    by coincidence.
    """
    for rep in range(2):
        _write_result(
            tmp_path,
            "D3-bin_n00200",
            "E-def",
            rep,
            {"error": None, "excess_risk": 100.0, "excess_log_loss": 1.0},
        )
        _write_result(
            tmp_path,
            "D3-bin_n00200",
            "EarthClassifier",
            rep,
            {"error": None, "excess_risk": 300.0, "excess_log_loss": 4.0},
        )
    df = summ.load_results(tmp_path)
    table = summ.binary_outcome_table(df)
    row = table[
        (table["DGP"] == "D3-bin")
        & (table["arm"] == "EarthClassifier")
        & (table["n"] == 200)
    ].iloc[0]
    # excess_log_loss ratio: 4/1 = 4. excess_risk ratio would be 300/100 = 3.
    assert "4.000" in row["excess_log_loss_ratio"]
    assert "3.000" not in row["excess_log_loss_ratio"]


def test_paired_log_ratio_is_sensitive_to_pairing_order(tmp_path):
    """The mean of paired log ratios is invariant to which repetition of one
    arm gets paired with which of the other (it is still the same values
    summed, just reordered), so a reversed-pairing bug cannot be caught by
    checking the ratio alone; the previous hand case also held E-def
    constant, which independently makes any pairing give the same result. Both
    arms vary here, and asymmetrically, so the *spread* (se, and so the
    interval) differs under a reversed pairing even though the ratio would not.
    """
    e_def = [1.0, 2.0]
    p_cur = [3.0, 9.0]
    for rep in range(2):
        _write_result(
            tmp_path,
            "D1_n00200_lo",
            "E-def",
            rep,
            {"error": None, "excess_risk": e_def[rep]},
        )
        _write_result(
            tmp_path,
            "D1_n00200_lo",
            "P-cur",
            rep,
            {"error": None, "excess_risk": p_cur[rep]},
        )
    df = summ.load_results(tmp_path)
    stats = summ.paired_log_ratio(df, "D1", 200, "lo", "P-cur", "E-def", "excess_risk")
    # Correct (same-repetition) pairing: g = [log3 - log1, log9 - log2].
    assert stats["se"] == pytest.approx(0.202733, abs=1e-5)
    # A reversed pairing (rep 0 of P-cur with rep 1 of E-def and vice versa)
    # would give se = 0.895880 instead, from the same two g_bar-preserving
    # but differently spread values (g_bar itself, 1.301345, is identical
    # either way: a sum of the same two values in a different order).
    assert stats["se"] != pytest.approx(0.895880, abs=1e-5)
    assert stats["g_bar"] == pytest.approx(1.301345, abs=1e-5)


def test_main_draws_both_figures_and_labels_each_box(tmp_path, monkeypatch):
    """Draws both figures through ``main``. ``box_plots`` passes
    ``tick_labels``, a keyword that matplotlib added in 3.9, but the
    validation group first allowed matplotlib 3.8. With 3.8 this test fails
    with a TypeError. The x tick labels are read back, so a box plot drawn
    without them (ticks numbered 1 and 2) also fails.
    """
    results = tmp_path / "results"
    risks = {"E-def": [1.0, 1.0], "P-cur": [2.0, 8.0], "P-fix": [1.0, 2.0]}
    for arm, values in risks.items():
        for rep, risk in enumerate(values):
            _write_result(
                results, "D1_n00200_lo", arm, rep, {"error": None, "excess_risk": risk}
            )
    figures = []
    subplots = summ.plt.subplots

    def recording_subplots(*args, **kwargs):
        fig, axes = subplots(*args, **kwargs)
        figures.append(fig)
        return fig, axes

    monkeypatch.setattr(summ.plt, "subplots", recording_subplots)
    summ.main(["--results", str(results), "--out", str(tmp_path / "report")])

    for name in ("equivalence_figure.png", "box_plots.png"):
        assert (tmp_path / "report" / name).read_bytes().startswith(b"\x89PNG")
    (box_ax,) = [
        ax
        for fig in figures
        for ax in fig.axes
        if ax.get_title().startswith("Gap and parity claims")
    ]
    labels = [text.get_text() for text in box_ax.get_xticklabels()]
    assert labels == ["D1/P-cur/lo", "D1/P-fix/lo"]


def test_both_figures_draw_with_no_results(tmp_path):
    """Before any cell has run, the box plot shows a note in place of boxes
    and the equivalence figure has empty panels; both files are written."""
    empty = summ.load_results(tmp_path)
    summ.box_plots(empty, tmp_path / "report" / "box_plots.png")
    summ.equivalence_figure(empty, tmp_path / "report" / "equivalence_figure.png")
    for name in ("box_plots.png", "equivalence_figure.png"):
        assert (tmp_path / "report" / name).read_bytes().startswith(b"\x89PNG")
