"""The committed results of the rerun after PR #185 and of the skewed-covariate
study give the numbers that ``validation/REPORT.md`` reports.

``make_report.py`` builds the report's tables from ``results/t173-grid`` (the
P-fix arm over the whole grid at 286d994, paired with the stored E-def) and
``results/skew`` (the old and the new pair rule on lognormal covariates). These
tests pin the headline numbers of REPORT.md, sections 3 and "A known limit":
they fail if a results file is truncated, collected from other repetitions, or
paired with the wrong arm, and the tables would then silently show other
numbers. They write no file in the repository (the tables go to a temporary
folder) and fit no model.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MAKE_REPORT = Path(__file__).resolve().parents[1] / "report" / "make_report.py"


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("make_report", MAKE_REPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tables = tmp_path_factory.mktemp("tables")
    mod.TABLES = tables
    df = mod.load()
    p = mod.parity(df)
    return mod, df, p, mod.parity_rerun(df, p), mod.skew(mod.load_skew())


def test_rerun_after_185_is_equivalent_in_every_cell(report):
    """REPORT.md, section 3: 51 of 51 cells equivalent after #185 (48 at the
    freeze); D6's ratio falls at every n, to 0.997 [0.993, 1.001] at 200 cases
    (it was 1.076 [1.051, 1.101]); no cell with signal rises above the upper
    bound of the study's cell."""
    _, _, p, q, _ = report
    assert (p.decision == "equivalent").sum() == 48
    assert len(q) == 51 and (q.new_decision == "equivalent").all()
    assert (q.n_sim.to_numpy() == p.n_sim.to_numpy()).all()
    d6 = q[q.dgp == "D6"].set_index("n")
    assert [round(d6.new_ratio[n], 3) for n in (200, 1000, 5000)] == [0.997, 1.001, 1.0]
    assert (round(d6.new_lo[200], 3), round(d6.new_hi[200], 3)) == (0.993, 1.001)
    assert round(d6.old_ratio[200], 3) == 1.076
    assert (d6.new_ratio < d6.old_ratio).all()
    signal = q[q.dgp != "D6"]
    assert signal.ratio_within_stored_bound.all() and signal.verdict_kept.all()


def test_sparsity_on_noise_after_185_is_earths(report):
    """REPORT.md, section 4: on pure noise at 200 cases pymars selects 4.87
    terms after #185, as earth does (4.98 at the freeze)."""
    _, _, _, q, _ = report
    row = q[(q.dgp == "D6") & (q.n == 200)].iloc[0]
    assert round(row.terms_earth, 2) == round(row.terms_new, 2) == 4.87
    assert round(row.terms_freeze, 2) == 4.98


def test_skewed_covariates_loss_is_small(report):
    """REPORT.md, "A known limit": on lognormal covariates the new rule is
    1.019 [1.010, 1.027] times the old one on D4 at 200 cases and 1.007
    [1.003, 1.010] at 1,000 cases, D8 (23,000 repetitions) is not resolved at
    1.009 [1.000, 1.018], the other cells lie between 0.970 and 0.996, the old
    rule equals earth, and no upper bound reaches the margin 1.05."""
    *_, s = report
    s = s.set_index(["dgp", "n"])

    def interval(key):
        r = s.loc[key]
        return tuple(round(r[f"new_old_{k}"], 3) for k in ("ratio", "lo", "hi"))

    assert interval(("D4", 200)) == (1.019, 1.010, 1.027)
    assert interval(("D4", 1000)) == (1.007, 1.003, 1.010)
    assert interval(("D8", 200)) == (1.009, 1.000, 1.018)
    assert s.loc[("D8", 200)].new_old_n_sim == 23_000
    assert s.loc[("D8", 200)].new_old_lo <= 1.0
    others = s.drop([("D4", 200), ("D4", 1000), ("D8", 200)])
    assert round(others.new_old_ratio.min(), 3) == 0.970
    assert round(others.new_old_ratio.max(), 3) == 0.996
    assert (others.new_old_lo <= 1.0).all()
    assert ((s.old_earth_lo <= 1.0) & (s.old_earth_hi >= 1.0)).all()
    assert (s.new_old_hi < 1.05).all()
