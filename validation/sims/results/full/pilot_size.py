# ruff: noqa: C408
"""Pilot sizing: python pilot_size.py <runs folder> <out csv> (P-fix / E-def)."""

import sys
from pathlib import Path

import pandas as pd

from validation.sims import pilot_check, summarize

df = summarize.load_results(Path(sys.argv[1]))
print("fails total", int(df.error.notna().sum()))
rows = []
for (dgp, n, noise), _ in df.groupby(["dgp", "n", "noise"], dropna=False):
    noise = None if pd.isna(noise) else noise
    meas = "excess_log_loss" if dgp.endswith("-bin") else "excess_risk"
    st = summarize.paired_log_ratio(df, dgp, n, noise, "P-fix", "E-def", meas)
    if not st or st["n_sim"] < 2:
        continue
    cell = df[
        (df.dgp == dgp)
        & (df.n == n)
        & ((df.noise == noise) if noise else df.noise.isna())
    ]
    sec = float(cell.groupby("arm").fit_seconds.mean().sum())
    s = pilot_check.equivalence_n_sim(st["g_bar"], st["s_g"])
    rows.append(
        dict(
            cell=f"{dgp}_n{n}_{noise}",
            ratio=st["ratio"],
            g_bar=st["g_bar"],
            s_p=st["s_g"],
            n_pilot=st["n_sim"],
            n_sim=s.n_sim,
            n_at_gap=s.n_sim_at_pilot_gap,
            feasible=s.feasible,
            sec_pair=sec,
            fails=st["n_fail_a"] + st["n_fail_b"],
        )
    )
r = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(r.round(4).to_string())
r.to_csv(sys.argv[2], index=False)
