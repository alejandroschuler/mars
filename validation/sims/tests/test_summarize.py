"""Tests of validation/sims/summarize.py: cell-name parsing and the
regression tests for one real bug found while running the smoke run
(``df.noise == None`` is false for every row in pandas, which silently hid
D6 and the binary DGPs, which have no noise level, from every display).
"""

from __future__ import annotations

import json

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
