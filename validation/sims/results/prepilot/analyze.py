"""Collect and size the pre-freeze pilot (T21 part a, issue #23).

Usage, from a worktree with the validation group installed:
    uv run --frozen python validation/sims/results/prepilot/analyze.py \
        <runs-folder> <legacy results.csv>

Writes results.csv, failure_counts.csv, sizing.csv and fit_times.csv next to
this file. summary.md is written by hand from them.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

from validation.sims import pilot_check, summarize

HERE = Path(__file__).parent
runs = Path(sys.argv[1])
legacy_csv = Path(sys.argv[2])

df = summarize.load_results(runs)
df = df.drop(columns=["extra"])
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
    .fit_seconds.agg(["mean", "max", "size"])
    .reset_index()
)
times.to_csv(HERE / "fit_times.csv", index=False)

# E-pym on D7 at 200 cases comes from the T04 legacy run (same seeds).
legacy = pd.read_csv(legacy_csv)
leg = legacy[(legacy.dgp == "D7") & (legacy.n == 200) & (legacy.arm == "E-pym")]
dup = (df.dgp == "D7") & (df.n == 200) & (df.arm == "E-pym")
pool = pd.concat(
    [df[~dup], leg[df.columns.intersection(leg.columns)]], ignore_index=True
)

rows = []


def contrast(kind, dgp, n, noise, a, b, reps):
    sub = pool[pool.rep < reps]
    stats = summarize.paired_log_ratio(sub, dgp, n, noise, a, b, "excess_risk")
    if stats is None or stats["n_sim"] < 2:
        return
    cell = sub[(sub.dgp == dgp) & (sub.n == n) & (sub.noise == noise)]
    sec = (
        cell[cell.arm.isin([a, b]) & (cell.rep < reps)]
        .groupby("arm")
        .fit_seconds.mean()
    )
    sec_pair = float(sec.sum())
    g_bar, s_p, n_pilot = stats["g_bar"], stats["s_g"], stats["n_sim"]
    row = {
        "kind": kind,
        "cell": f"{dgp}_n{n:05d}_{noise}",
        "contrast": f"{a}/{b}",
        "n_pilot": n_pilot,
        "ratio": stats["ratio"],
        "g_bar": g_bar,
        "s_p": s_p,
        "z": pilot_check.pilot_z(g_bar, s_p, n_pilot),
        "sec_per_rep": sec_pair,
        "fails": stats["n_fail_a"] + stats["n_fail_b"],
    }
    if kind == "equivalence":
        s = pilot_check.equivalence_n_sim(g_bar, s_p)
        row.update(
            n_sim=s.n_sim, n_sim_at_pilot_gap=s.n_sim_at_pilot_gap, feasible=s.feasible
        )
    else:
        s = pilot_check.difference_n_sim(g_bar, s_p, n_pilot)
        row.update(n_sim=s.n_sim, n_sim_safe=s.n_sim_safe, feasible=True)
    n_for_cost = row["n_sim"]
    row["core_hours_at_n_sim"] = (
        n_for_cost * sec_pair / 3600 if math.isfinite(n_for_cost) else math.nan
    )
    rows.append(row)


for dgp in ("D3", "D4", "D5", "D7"):
    for noise in ("lo", "hi"):
        contrast("equivalence", dgp, 1000, noise, "P-fix", "E-def", 100)
for noise in ("lo", "hi"):
    contrast("difference", "D7", 200, noise, "E-pym", "E-def", 100)
    contrast("difference-300reps", "D7", 200, noise, "E-pym", "E-def", 300)

sizing = pd.DataFrame(rows)
sizing.to_csv(HERE / "sizing.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30)
print(sizing.round(4).to_string())
print(fail[fail.failed > 0])
print(times.round(3).to_string())
