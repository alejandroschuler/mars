# ruff: noqa: C408
"""Collect the T21 full run (issue #23) and build the summary tables.

Usage, from a worktree with the validation group installed:
    uv run --frozen python validation/sims/results/full/analyze.py \
        <runs-folder> <legacy results.csv>

Writes results.csv (P-fix and E-def, every unit), failure_counts.csv,
parity.csv (P-fix / E-def, every cell, with the equivalence rule),
fit_times.csv and summaries/ (the ratio, selection, binary and appendix
tables of validation.sims.summarize, on the full run pooled with the legacy
results of T04 for the arms the full run did not rerun). No figures.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

from validation.sims import pilot_check, summarize

HERE = Path(__file__).parent
runs, legacy_csv = (Path(a) for a in sys.argv[1:3])

df = summarize.load_results(runs).drop(columns=["extra"])
df.to_csv(HERE / "results.csv", index=False)

fail = (
    df.assign(failed=df.error.notna())
    .groupby(["dgp", "n", "noise", "arm"], dropna=False)
    .agg(attempted=("rep", "size"), failed=("failed", "sum"))
    .reset_index()
)
fail.to_csv(HERE / "failure_counts.csv", index=False)

times = (
    df.groupby(["dgp", "n", "arm"])
    .fit_seconds.agg(["mean", "max", "sum", "size"])
    .reset_index()
)
times.to_csv(HERE / "fit_times.csv", index=False)

# Parity: P-fix / E-def in every cell, equivalence rule of the plan.
delta = pilot_check.EQUIVALENCE_MARGIN
rows = []
for (dgp, n, noise), _ in df.groupby(["dgp", "n", "noise"], dropna=False):
    noise = None if pd.isna(noise) else noise
    meas = "excess_log_loss" if dgp.endswith("-bin") else "excess_risk"
    st = summarize.paired_log_ratio(df, dgp, n, noise, "P-fix", "E-def", meas)
    g_bar, s, k = st["g_bar"], st["s_g"], st["n_sim"]
    half = 3 * s / math.sqrt(k)
    rows.append(
        dict(
            dgp=dgp,
            n=n,
            noise=noise or "na",
            measure=meas,
            n_sim=k,
            ratio=st["ratio"],
            ratio_lo=math.exp(g_bar - half),
            ratio_hi=math.exp(g_bar + half),
            g_bar=g_bar,
            s_g=s,
            equivalent=bool(abs(g_bar) + half <= delta),
            fails=st["n_fail_a"] + st["n_fail_b"],
        )
    )
parity = pd.DataFrame(rows)
parity.to_csv(HERE / "parity.csv", index=False)

# Pooled with the legacy results for the arms not rerun here.
legacy = pd.read_csv(legacy_csv)
# T04 wrote the one-level noise as the string "na"; run.py's cell names give NaN.
legacy["noise"] = legacy.noise.where(legacy.noise != "na", float("nan"))
legacy = legacy[~legacy.arm.isin(["E-def"]) | (legacy.dgp == "__none__")]
legacy = legacy[legacy.arm != "P-fix"]
pool = pd.concat(
    [df, legacy[df.columns.intersection(legacy.columns)]], ignore_index=True
)
S = HERE / "summaries"
summarize.ratio_table(pool).to_csv(S / "ratio_table.csv", index=False)
summarize.ratio_table(pool, n=1000).to_csv(S / "ratio_table_n1000.csv", index=False)
summarize.selection_table(pool).to_csv(S / "selection_table.csv", index=False)
summarize.binary_outcome_table(pool).to_csv(S / "binary_outcome_table.csv", index=False)
summarize.appendix_table(pool).to_csv(S / "appendix_table.csv", index=False)

pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 100)
print("failed units:", int(fail.failed.sum()), "of", int(fail.attempted.sum()))
print("fit core-hours:", df.fit_seconds.sum() / 3600)
print(parity.round(4).to_string())
