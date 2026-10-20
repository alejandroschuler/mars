"""Build the tables and figures of validation/REPORT.md (T22b, issue #24).

Usage, from a worktree with the validation group installed:

    uv run --frozen --group validation python validation/sims/report/make_report.py \
        [--bench-run <run folder of validation/bench/run.py>]

Reads the per-repetition results of the T21 full run
(``validation/sims/results/full/results.csv``: P-fix and E-def) and of the
T04 legacy run (``validation/sims/results/legacy/results.csv``: P-cur, P-ear,
E-pym, EarthClassifier, GLMEarth, OLS, HGB, and E-def on the same seeds).
The two E-def columns are equal where both exist, so every contrast is paired
on the same datasets. It also reads the rerun of the P-fix arm over the whole
grid after PR #185 (``results/t173-grid/results.csv``, arm ``P-fix-185``, at
the same seeds and repetitions, paired with the stored E-def) and the
skewed-covariate study (``results/skew/results.csv``: arms P-old, P-new and
E-def). With ``--bench-run`` it also copies the benchmark's medians to
``bench_times.csv``; without it the stored copy is used.

Writes, in this folder, ``tables/*.md`` (pasted into REPORT.md),
``tables/*.csv`` and ``figures/*.png``. Nothing here fits a model.

The measures are those of VALIDATION_PLAN.md, "Performance measures": for two
arms A and B in repetition i, g_i = log R_A,i - log R_B,i; the ratio is
exp(mean g) and the interval exp(mean g +- 3 sd(g)/sqrt(n_sim)).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SIMS = HERE.parent
ROOT = SIMS.parents[1]
TABLES, FIGURES = HERE / "tables", HERE / "figures"
DELTA = math.log(1.05)
K = 3.0
RERUN = "P-fix-185"  # the P-fix arm at 286d994, after the pair rule of PR #185
DGPS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]
NAMES = {
    "D1": "D1 linear",
    "D2": "D2 additive hinges",
    "D3": "D3 additive smooth",
    "D4": "D4 Friedman #1",
    "D5": "D5 hinge interaction",
    "D6": "D6 pure noise",
    "D7": "D7 many covariates",
    "D8": "D8 correlated",
    "D3-bin": "D3-bin",
    "D4-bin": "D4-bin",
}
# The dataviz skill's reference palette, light mode, first three slots, and
# its text inks.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#eceae4"


def load() -> pd.DataFrame:
    full = pd.read_csv(SIMS / "results" / "full" / "results.csv")
    leg = pd.read_csv(SIMS / "results" / "legacy" / "results.csv", dtype=str)
    leg = leg[leg.arm != "arm"]  # a repeated header row in the collected file
    num = [
        "n",
        "rep",
        "excess_risk",
        "excess_log_loss",
        "excess_brier_score",
        "calibration_slope",
        "n_terms",
        "n_irrelevant_covariates",
        "fit_seconds",
    ]
    for c in num:
        leg[c] = pd.to_numeric(leg[c])
    leg["uses_irrelevant_covariate"] = leg.uses_irrelevant_covariate == "True"
    for d in (full, leg):
        d["noise"] = d.noise.fillna("na")
    # E-def is in both files with the same seeds and values; keep the full run's.
    leg = leg[leg.arm != "E-def"]
    rerun = pd.read_csv(SIMS / "results" / "t173-grid" / "results.csv")
    rerun["noise"] = rerun.noise.fillna("na")
    rerun["arm"] = RERUN
    df = pd.concat([full, leg, rerun], ignore_index=True)
    df["n"] = df.n.astype(int)
    df["rep"] = df.rep.astype(int)
    return df


def measure(dgp: str) -> str:
    return "excess_log_loss" if dgp.endswith("-bin") else "excess_risk"


def paired(df, dgp, n, noise, a, b="E-def", max_rep=None):
    """The log ratios g_i of arm a over arm b, paired by repetition, or None.
    With max_rep, only repetitions below it (the legacy arms' range)."""
    m = measure(dgp)
    cell = df[(df.dgp == dgp) & (df.n == n) & (df.noise == noise)]
    if max_rep is not None:
        cell = cell[cell.rep < max_rep]
    x = cell[cell.arm == a].set_index("rep")[m]
    y = cell[cell.arm == b].set_index("rep")[m]
    j = x.index.intersection(y.index)
    if len(j) == 0:
        return None
    return np.log(x[j].to_numpy()) - np.log(y[j].to_numpy())


def paired_series(df, dgp, n, noise, a, b="E-def"):
    """The log ratios g_i of arm a over arm b as a Series indexed by the
    repetition i (sorted), so that two such Series can be subtracted."""
    m = measure(dgp)
    cell = df[(df.dgp == dgp) & (df.n == n) & (df.noise == noise)]
    x = cell[cell.arm == a].set_index("rep")[m]
    y = cell[cell.arm == b].set_index("rep")[m]
    j = x.index.intersection(y.index).sort_values()
    return pd.Series(np.log(x[j].to_numpy()) - np.log(y[j].to_numpy()), index=j)


def summary(g) -> dict:
    k = len(g)
    gb, s = float(np.mean(g)), float(np.std(g, ddof=1))
    se = s / math.sqrt(k)
    return {
        "n_sim": k,
        "g_bar": gb,
        "s_g": s,
        "se": se,
        "ratio": math.exp(gb),
        "lo": math.exp(gb - K * se),
        "hi": math.exp(gb + K * se),
        "z": gb / se if se > 0 else 0.0,
        "share_equal": float(np.mean(np.abs(g) < 1e-9)),
    }


def fmt_ratio(st: dict | None, mark: bool = True) -> str:
    if st is None:
        return "not run"
    flag = "*" if mark and abs(st["z"]) >= K else ""
    return f"{st['ratio']:.3f}{flag} [{st['lo']:.3f}, {st['hi']:.3f}]"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def write(name: str, text: str) -> None:
    (TABLES / name).write_text(text)


# The gap claim -------------------------------------------------------------
def gap_tables(df) -> None:
    arms = ["P-cur", "P-ear", "E-pym", "P-fix"]
    recs = []
    for n, noises in ((200, ["lo", "hi"]), (1000, ["lo"])):
        for noise in noises:
            rows = []
            for d in DGPS:
                nz = "na" if d == "D6" else noise
                if d == "D6" and noise == "hi":
                    continue
                row = [NAMES[d]]
                for a in arms:
                    # P-fix on the legacy arms' repetitions, so that every
                    # column of a row uses the same datasets.
                    g = paired(df, d, n, nz, a, max_rep=300 if n == 200 else 200)
                    st = None if g is None else summary(g)
                    if st:
                        recs.append({"dgp": d, "n": n, "noise": nz, "arm": a, **st})
                    if st is None and d == "D7" and a in ("P-cur", "P-ear"):
                        row.append("not run (too slow)")
                    else:
                        row.append(fmt_ratio(st))
                rows.append(row)
            hdr = ["DGP"] + [f"{a} / E-def" for a in arms]
            write(f"gap_n{n}_{noise}.md", md_table(hdr, rows))
    pd.DataFrame(recs).to_csv(TABLES / "gap.csv", index=False)
    # Does moving the legacy code to earth's settings shrink its gap?
    rows = []
    for n, noise in ((200, "lo"), (200, "hi"), (1000, "lo")):
        for d in DGPS:
            nz = "na" if d == "D6" else noise
            if (d == "D6" and noise == "hi") or d == "D7":
                continue
            g = paired(df, d, n, nz, "P-ear", "P-cur")
            if g is not None:
                rows.append([NAMES[d], nz, f"{n:,}", fmt_ratio(summary(g))])
    write("gap_settings.md", md_table(["DGP", "noise", "n", "P-ear / P-cur"], rows))


# The parity claim ----------------------------------------------------------
def decision(st) -> str:
    half = K * st["se"]
    if abs(st["g_bar"]) + half <= DELTA:
        return "equivalent"
    if abs(st["g_bar"]) - half >= DELTA:
        return "not equivalent"
    return "not decided"


def parity(df) -> pd.DataFrame:
    recs = []
    for (d, n, nz), _ in df[df.arm == "P-fix"].groupby(["dgp", "n", "noise"]):
        st = summary(paired(df, d, n, nz, "P-fix"))
        recs.append({"dgp": d, "n": n, "noise": nz, **st, "decision": decision(st)})
    p = pd.DataFrame(recs)
    p.to_csv(TABLES / "parity.csv", index=False)
    rows = []
    for _, r in p.sort_values(["dgp", "noise", "n"]).iterrows():
        rows.append(
            [
                NAMES[r.dgp],
                r.noise,
                f"{r.n:,}",
                f"{r.n_sim:,}",
                f"{r.ratio:.3f} [{r.lo:.3f}, {r.hi:.3f}]",
                f"{100 * r.share_equal:.0f}%",
                r.decision,
            ]
        )
    hdr = ["DGP", "noise", "n", "reps", "P-fix / E-def", "equal fits", "decision"]
    write("parity_all.md", md_table(hdr, rows))
    off = p[p.decision != "equivalent"]
    write(
        "parity_off.md",
        md_table(
            hdr,
            [r for r in rows if r[6] != "equivalent"],
        ),
    )
    counts = p.decision.value_counts().to_dict()
    (TABLES / "parity_counts.json").write_text(
        json.dumps({"cells": len(p), **counts, "off": off.dgp.tolist()}, indent=1)
    )
    return p


def style(ax) -> None:
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def parity_figure(
    p: pd.DataFrame, name: str = "parity.png", arm: str = "P-fix"
) -> None:
    order = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D3-bin", "D4-bin"]
    marks = {
        "lo": ("o", "low noise"),
        "hi": ("s", "high noise"),
        "na": ("D", "one level"),
    }
    colors = {"equivalent": BLUE, "not decided": ORANGE, "not equivalent": "#e34948"}
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.6), sharey=True)
    lo_x, hi_x = 0.93, 1.12
    for ax, n in zip(axes, (200, 1000, 5000), strict=True):
        ax.axvspan(1 / 1.05, 1.05, color=BAND, zorder=0)
        ax.axvline(1.0, color=INK2, linewidth=0.8, zorder=1)
        for i, d in enumerate(order):
            for nz, dy in (("lo", -0.18), ("hi", 0.18), ("na", 0.0)):
                r = p[(p.dgp == d) & (p.n == n) & (p.noise == nz)]
                if r.empty:
                    continue
                r = r.iloc[0]
                c = colors[r.decision]
                y = i + dy
                ax.plot([r.lo, r.hi], [y, y], color=c, linewidth=2, zorder=2)
                ax.plot(
                    r.ratio, y, marks[nz][0], color=c, markersize=6,
                    markeredgecolor="white", markeredgewidth=1, zorder=3,
                )  # fmt: skip
        ax.set_xscale("log")
        ax.set_xlim(lo_x, hi_x)
        ticks = [0.95, 1.0, 1.05, 1.1]
        ax.set_xticks(ticks, [f"{t:.2f}" for t in ticks])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(f"n = {n:,}", fontsize=10, color=INK, loc="left")
        style(ax)
    axes[0].set_yticks(range(len(order)), [NAMES[d] for d in order])
    axes[0].invert_yaxis()
    axes[1].set_xlabel(f"ratio of excess risk, {arm} / E-def (log scale)", color=INK2)
    handles = [
        plt.Line2D([], [], color=colors[k], linewidth=2, label=k) for k in colors
    ] + [
        plt.Line2D([], [], color=INK2, marker=m, linestyle="", label=lab)
        for m, lab in marks.values()
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=8,
        bbox_to_anchor=(0.5, -0.01),
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGURES / name, dpi=150, facecolor="white")
    plt.close(fig)


def parity_rerun(df: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """The parity grid again after PR #185 (issue #173): P-fix at 286d994 over
    E-def, on the same cells, seeds and repetitions as the study at the freeze
    (``p``, from parity()), with the E-def fits of the study. The columns with
    the prefix old are the study's, those with new the rerun's."""
    recs = []
    for r in p.itertuples():
        gn = paired_series(df, r.dgp, r.n, r.noise, RERUN)
        go = paired_series(df, r.dgp, r.n, r.noise, "P-fix")
        if not go.index.equals(gn.index):
            raise SystemExit(f"{r.dgp} n={r.n} {r.noise}: other repetitions")
        so, sn = summary(go.to_numpy()), summary(gn.to_numpy())
        ch = summary((gn - go).to_numpy())  # the paired change of the log ratio
        m = measure(r.dgp)
        cell = df[(df.dgp == r.dgp) & (df.n == r.n) & (df.noise == r.noise)]
        cell = cell[cell.rep.isin(go.index)]
        by = {a: cell[cell.arm == a].set_index("rep").loc[go.index] for a in
              ("E-def", "P-fix", RERUN)}  # fmt: skip
        same = np.abs(np.log(by[RERUN][m]) - np.log(by["P-fix"][m])) < 1e-9
        rec = {
            "dgp": r.dgp, "n": r.n, "noise": r.noise, "n_sim": sn["n_sim"],
            "same_fit_as_freeze": float(same.mean()),
            "chg_ratio": ch["ratio"], "chg_lo": ch["lo"], "chg_hi": ch["hi"],
        }  # fmt: skip
        for tag, st in (("old", so), ("new", sn)):
            rec.update({f"{tag}_{k}": st[k] for k in ("ratio", "lo", "hi", "g_bar")})
            rec[f"{tag}_equal"] = st["share_equal"]
            rec[f"{tag}_decision"] = decision(st)
        for tag, a in (("freeze", "P-fix"), ("new", RERUN), ("earth", "E-def")):
            rec[f"terms_{tag}"] = float(by[a].n_terms.mean())
            rec[f"irrelevant_{tag}"] = float(by[a].n_irrelevant_covariates.mean())
        recs.append(rec)
    q = pd.DataFrame(recs)
    # The acceptance test of the issue, fixed before the rerun: in no cell with
    # signal does the ratio rise above the stored cell's upper bound (to within
    # 1e-9) or the verdict go from equivalent to another one; D6's ratio falls.
    q["signal"] = q.dgp != "D6"
    q["ratio_within_stored_bound"] = q.new_ratio <= q.old_hi + 1e-9
    q["verdict_kept"] = ~(
        (q.old_decision == "equivalent") & (q.new_decision != "equivalent")
    )
    q.to_csv(TABLES / "parity_rerun.csv", index=False)

    def cellname(r) -> list[str]:
        return [NAMES[r.dgp], r.noise, f"{r.n:,}", f"{r.n_sim:,}"]

    def interval(r, tag) -> str:
        return f"{r[f'{tag}_ratio']:.3f} [{r[f'{tag}_lo']:.3f}, {r[f'{tag}_hi']:.3f}]"

    def row(r) -> list[str]:
        return [
            *cellname(r),
            interval(r, "old"),
            interval(r, "new"),
            f"{r.old_decision} to {r.new_decision}"
            if r.old_decision != r.new_decision
            else r.new_decision,
            f"{100 * r.old_equal:.0f}% to {100 * r.new_equal:.0f}%",
        ]

    hdr = ["DGP", "noise", "n", "reps", "freeze: P-fix / E-def",
           "after #185: P-fix / E-def", "decision",
           "equal fits to E-def"]  # fmt: skip
    ordered = [r for _, r in q.sort_values(["dgp", "noise", "n"]).iterrows()]
    write("parity_rerun_all.md", md_table(hdr, [row(r) for r in ordered]))
    changed = [
        r for r in ordered
        if r.dgp == "D6" or r.old_decision != r.new_decision
    ]  # fmt: skip
    hdr2 = [*hdr, "paired change"]
    rows2 = [
        [*row(r), f"{r.chg_ratio:.3f} [{r.chg_lo:.3f}, {r.chg_hi:.3f}]"]
        for r in changed
    ]
    write("parity_rerun_changed.md", md_table(hdr2, rows2))

    sig = q[q.signal]
    d6 = q[~q.signal].sort_values("n")
    counts = {
        "cells": len(q),
        "equivalent_freeze": int((q.old_decision == "equivalent").sum()),
        "equivalent_after": int((q.new_decision == "equivalent").sum()),
        "signal_cells": len(sig),
        "signal_ratio_above_stored_bound": sig[~sig.ratio_within_stored_bound][
            ["dgp", "n", "noise"]
        ].to_dict("records"),
        "signal_verdict_lost": sig[~sig.verdict_kept][["dgp", "n", "noise"]].to_dict(
            "records"
        ),
        "signal_cells_above_stored_point": int(
            (sig.new_ratio > sig.old_ratio + 1e-9).sum()
        ),
        "signal_ratio_range_after": [
            float(sig.new_ratio.min()),
            float(sig.new_ratio.max()),
        ],
        "signal_highest_upper_bound_after": float(sig.new_hi.max()),
        "d6_ratio_falls_at_every_n": bool((d6.new_ratio < d6.old_ratio).all()),
        "d6_paired_change": {
            int(r.n): [r.chg_ratio, r.chg_lo, r.chg_hi] for r in d6.itertuples()
        },
        "verdict_changes": q[q.old_decision != q.new_decision][
            ["dgp", "n", "noise", "old_decision", "new_decision"]
        ].to_dict("records"),
    }
    (TABLES / "parity_rerun_counts.json").write_text(json.dumps(counts, indent=1))
    return q


def box_figure(df) -> None:
    arms = ["P-cur", "P-ear", "E-pym", "P-fix"]
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.2), sharex=True)
    for ax, d in zip(axes.ravel(), DGPS, strict=True):
        data, pos, cols = [], [], []
        for j, a in enumerate(arms):
            for nz, dy, c in (("lo", -0.18, BLUE), ("hi", 0.18, ORANGE)):
                nzz = "na" if d == "D6" else nz
                if d == "D6" and nz == "hi":
                    continue
                g = paired(df, d, 200, nzz, a, max_rep=300)
                if g is None:
                    continue
                data.append(g)
                pos.append(j + (0 if d == "D6" else dy))
                cols.append(c)
        if data:
            bp = ax.boxplot(
                data, positions=pos, orientation="horizontal", widths=0.3, whis=(5, 95),
                showfliers=False, patch_artist=True,
            )  # fmt: skip
            for b, c in zip(bp["boxes"], cols, strict=True):
                b.set(facecolor=c, edgecolor=c, alpha=0.85)
            for k, c in enumerate(cols):
                for part in ("whiskers", "caps"):
                    for line in bp[part][2 * k : 2 * k + 2]:
                        line.set(color=c)
                bp["medians"][k].set(color="white", linewidth=1.5)
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_yticks(range(len(arms)), arms)
        ax.set_ylim(len(arms) - 0.5, -0.5)
        ax.set_xlim(-4, 3)
        ax.set_title(NAMES[d], fontsize=9, color=INK, loc="left")
        style(ax)
    for ax in axes[1]:
        ax.set_xlabel("log ratio to E-def", color=INK2, fontsize=8)
    handles = [
        matplotlib.patches.Patch(color=BLUE, label="low noise (D6: one level)"),
        matplotlib.patches.Patch(color=ORANGE, label="high noise"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGURES / "per_rep_boxes.png", dpi=150, facecolor="white")
    plt.close(fig)


# The sparsity claim --------------------------------------------------------
SPARSITY_ARMS = [
    ("E-def", "E-def"),
    ("P-cur", "P-cur"),
    ("P-fix", "P-fix (freeze)"),
    (RERUN, "P-fix (after #185)"),
]


def sparsity_rows(df, d, n, nz, arms, recs, digits=2):
    """The rows of Table 4.1 for one cell: the size and selection of every arm,
    and its mean paired difference from E-def with ±3 Monte Carlo SEs."""
    rows = []
    cell = df[(df.dgp == d) & (df.n == n) & (df.noise == nz)]
    e = cell[cell.arm == "E-def"].set_index("rep")
    for a, label in arms:
        x = cell[cell.arm == a].set_index("rep")
        if x.empty:
            rows.append(
                [NAMES[d], nz, f"{n:,}", label, "not run (too slow)", "", "", "", ""]
            )
            continue
        q = x.n_terms.quantile([0.25, 0.5, 0.75])
        diff = ""
        if a != "E-def":
            j = x.index.intersection(e.index)
            parts = []
            for col in ("n_terms", "n_irrelevant_covariates"):
                dd = (x.loc[j, col] - e.loc[j, col]).astype(float)
                m, se = dd.mean(), dd.std(ddof=1) / math.sqrt(len(dd))
                lo, hi = m - K * se, m + K * se
                parts.append(f"{m:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]")
                recs.append(
                    {"dgp": d, "n": n, "noise": nz, "arm": a, "measure": col,
                     "mean_diff": m, "se": se, "reps": len(dd)}
                )  # fmt: skip
            diff = " / ".join(parts)
        rows.append(
            [
                NAMES[d],
                nz,
                f"{n:,}",
                label,
                f"{q[0.5]:.0f} ({q[0.25]:.0f} to {q[0.75]:.0f})",
                f"{x.n_terms.mean():.2f}",
                f"{100 * x.uses_irrelevant_covariate.mean():.0f}%",
                f"{x.n_irrelevant_covariates.mean():.2f}",
                diff,
            ]
        )
    return rows


SPARSITY_HEADER = [
    "DGP",
    "noise",
    "n",
    "arm",
    "terms, median (IQR)",
    "terms, mean",
    "uses an irrelevant covariate",
    "irrelevant covariates, mean",
    "minus E-def: terms / irrelevant covariates",
]


def sparsity(df) -> None:
    rows, recs = [], []
    for d in ["D4", "D6", "D7", "D8"]:
        for nz in ["na"] if d == "D6" else ["lo", "hi"]:
            rows += sparsity_rows(df, d, 200, nz, SPARSITY_ARMS, recs)
    write("sparsity.md", md_table(SPARSITY_HEADER, rows))
    # Pure noise at every sample size, for the freeze and after #185.
    d6, arms = [], [a for a in SPARSITY_ARMS if a[0] != "P-cur"]
    for n in (200, 1000, 5000):
        d6 += sparsity_rows(df, "D6", n, "na", arms, recs, digits=3)
    write("sparsity_d6.md", md_table(SPARSITY_HEADER, d6))
    pd.DataFrame(recs).drop_duplicates().to_csv(TABLES / "sparsity.csv", index=False)


# Binary outcomes ------------------------------------------------------------
def binary(df) -> None:
    rows = []
    for d in ["D3-bin", "D4-bin"]:
        for n in (200, 1000, 5000):
            for a in ["E-def", "EarthClassifier", "GLMEarth", "P-fix"]:
                x = df[(df.dgp == d) & (df.n == n) & (df.arm == a)]
                if x.empty:
                    continue
                st = None if a == "E-def" else summary(paired(df, d, n, "na", a))
                slope = x.calibration_slope.median()
                rows.append(
                    [
                        d,
                        f"{n:,}",
                        a,
                        f"{len(x):,}",
                        "1 (reference)" if st is None else fmt_ratio(st),
                        f"{slope:.2f}",
                    ]
                )
    hdr = [
        "DGP",
        "n",
        "arm",
        "reps",
        "excess log loss / E-def",
        "calibration slope, median",
    ]
    write("binary.md", md_table(hdr, rows))


def binary_appendix(df, run_dirs: list[Path]) -> None:
    """Excess Brier score (ratio to E-def) and clipped probabilities per fit.
    The clipped counts are only in the per-repetition files of the runs, not
    in results.csv; with --run-dirs they are collected into clipped.csv."""
    out = TABLES / "clipped.csv"
    if run_dirs:
        recs = []
        for root in run_dirs:
            for f in sorted(root.glob("D[34]-bin_n*/*/rep*.json")):
                r = json.loads(f.read_text())
                if r.get("error"):
                    continue
                cell, arm = f.parent.parent.name, f.parent.name
                recs.append({"cell": cell, "arm": arm, "rep": int(f.stem[3:]),
                             "n_clipped": r.get("n_clipped_log_loss")})  # fmt: skip
        c = pd.DataFrame(recs).drop_duplicates(["cell", "arm", "rep"], keep="last")
        c.to_csv(out, index=False)
    c = pd.read_csv(out) if out.is_file() else None
    rows = []
    for d in ["D3-bin", "D4-bin"]:
        for n in (200, 1000, 5000):
            for a in ["E-def", "EarthClassifier", "GLMEarth", "P-fix"]:
                x = df[(df.dgp == d) & (df.n == n) & (df.arm == a)]
                if x.empty:
                    continue
                if a == "E-def":
                    br = "1 (reference)"
                else:
                    m = "excess_brier_score"
                    e = df[(df.dgp == d) & (df.n == n) & (df.arm == "E-def")]
                    xi, ei = x.set_index("rep")[m], e.set_index("rep")[m]
                    j = xi.index.intersection(ei.index)
                    br = fmt_ratio(
                        summary(np.log(xi[j].to_numpy()) - np.log(ei[j].to_numpy()))
                    )
                clip = "not collected"
                if c is not None:
                    cc = c[(c.cell == f"{d}_n{n:05d}") & (c.arm == a)].n_clipped
                    if len(cc):
                        clip = f"{cc.mean():.1f} ({100 * (cc > 0).mean():.0f}%)"
                rows.append([d, f"{n:,}", a, br, clip])
    hdr = ["DGP", "n", "arm", "excess Brier score / E-def",
           "clipped probabilities per fit, mean (share of fits with any)"]  # fmt: skip
    write("binary_appendix.md", md_table(hdr, rows))


# Skewed covariates ------------------------------------------------------------
SKEW_ORDER = ["D1", "D2", "D3", "D4", "D5", "D7", "D8"]
SKEW_CONTRASTS = {
    "new_old": ("P-new", "P-old"),
    "new_earth": ("P-new", "E-def"),
    "old_earth": ("P-old", "E-def"),
}


def load_skew() -> pd.DataFrame:
    sk = pd.read_csv(SIMS / "results" / "skew" / "results.csv")
    sk["noise"] = sk.noise.fillna("na")
    return sk


def skew(sk) -> pd.DataFrame:
    """Lognormal covariates, truth f(U): the pair rule of PR #185 (arm P-new)
    against the rule before it (P-old) and against earth, paired by
    repetition. The decision rule was fixed in advance: no material loss when
    every upper bound of new / old is below 1.05."""
    recs = []
    cells = sk[["dgp", "n", "noise"]].drop_duplicates().itertuples(index=False)
    for c in sorted(cells, key=lambda c: (SKEW_ORDER.index(c.dgp), c.n)):
        cell = sk[(sk.dgp == c.dgp) & (sk.n == c.n) & (sk.noise == c.noise)]
        if cell.error.notna().any():
            raise SystemExit(f"{c}: failed fits in the skewed study")
        hashes = cell.groupby("rep").data_hash.nunique()
        if (hashes != 1).any():
            raise SystemExit(f"{c}: the arms did not see the same data")
        rec = {"dgp": c.dgp, "n": c.n, "noise": c.noise}
        for name, (a, b) in SKEW_CONTRASTS.items():
            st = summary(paired(sk, c.dgp, c.n, c.noise, a, b))
            rec.update({f"{name}_{k}": v for k, v in st.items()})
        for arm in ("P-old", "P-new"):
            rec[f"terms_{arm}"] = float(cell[cell.arm == arm].n_terms.mean())
        recs.append(rec)
    q = pd.DataFrame(recs)
    q.to_csv(TABLES / "skew.csv", index=False)

    def ratio(r, name) -> str:
        st = {k: r[f"{name}_{k}"] for k in ("ratio", "lo", "hi", "z")}
        return fmt_ratio(st)

    rows = [
        [
            NAMES[r.dgp],
            f"{r.n:,}",
            f"{int(r.new_old_n_sim):,}",
            ratio(r, "new_old"),
            ratio(r, "new_earth"),
            ratio(r, "old_earth"),
            f"{100 * r.new_old_share_equal:.0f}%",
            f"{100 * r.old_earth_share_equal:.0f}% / "
            f"{100 * r.new_earth_share_equal:.0f}%",
        ]
        for _, r in q.iterrows()
    ]
    hdr = ["DGP", "n", "reps", "new / old", "new / E-def", "old / E-def",
           "equal fits, new vs old", "equal fits to E-def, old / new"]  # fmt: skip
    write("skew.md", md_table(hdr, rows))
    resolved = q[q.new_old_lo > 1.0]
    counts = {
        "cells": len(q),
        "largest_upper_bound": float(q.new_old_hi.max()),
        "no_material_loss": bool((q.new_old_hi < 1.05).all()),
        "resolved_losses": {
            f"{r.dgp} n={r.n}": [r.new_old_ratio, r.new_old_lo, r.new_old_hi]
            for r in resolved.itertuples()
        },
        "not_resolved_range": [
            float(q[q.new_old_lo <= 1.0].new_old_ratio.min()),
            float(q[q.new_old_lo <= 1.0].new_old_ratio.max()),
        ],
    }
    (TABLES / "skew_counts.json").write_text(json.dumps(counts, indent=1))
    return q


# Failure counts --------------------------------------------------------------
def failures(df, sk) -> None:
    f = (
        df.assign(failed=df.error.notna() & (df.error.astype(str) != ""))
        .groupby("arm")
        .agg(attempted=("rep", "size"), failed=("failed", "sum"))
        .reset_index()
    )
    rows = [[r.arm, f"{r.attempted:,}", f"{r.failed}"] for r in f.itertuples()]
    g = (
        sk.assign(failed=sk.error.notna())
        .groupby("arm")
        .agg(attempted=("rep", "size"), failed=("failed", "sum"))
        .reset_index()
    )
    rows += [
        [f"{r.arm} (skewed covariates)", f"{r.attempted:,}", f"{r.failed}"]
        for r in g.itertuples()
    ]
    write("failures.md", md_table(["arm", "fits attempted", "failed"], rows))


# Speed ----------------------------------------------------------------------
def bench_times(run: Path | None) -> pd.DataFrame:
    out = HERE / "bench_times.csv"
    if run is not None:
        recs = []
        for f in sorted((run / "results").glob("*.json")):
            r = json.loads(f.read_text())
            if r.get("status") != "ok":
                continue
            recs.append(
                {
                    **r["cell"],
                    "seconds": r["time_median"],
                    "terms": r["terms"],
                    "started": r.get("started", ""),
                }
            )
        b = pd.DataFrame(recs).sort_values("started")
        key = ["system", "n", "p", "degree", "max_terms", "weights"]
        b = b.drop_duplicates(key, keep="last").drop(columns="started")
        b.to_csv(out, index=False)
    return pd.read_csv(out)


def speed_figure(b: pd.DataFrame) -> None:
    base = (b.p == 10) & (b.degree == 2) & (b.max_terms == 21) & (~b.weights)
    series = [
        ("pymars_est", "pymars 2.0, fresh process", BLUE),
        ("earth_default", "earth, defaults", ORANGE),
        ("legacy", "mars-earth 1.0.4", AQUA),
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    for s, lab, c in series:
        x = b[base & (b.system == s)].sort_values("n")
        if x.empty:
            continue
        ax.plot(
            x.n, x.seconds, marker="o", linestyle="solid", color=c, linewidth=2,
            markersize=6, markeredgecolor="white", label=lab,
        )  # fmt: skip
    warm = HERE / "warm_times.csv"
    if warm.is_file():
        w = pd.read_csv(warm).sort_values("n")
        ax.plot(
            w.n, w.warm_median_s, marker="o", linestyle="dashed", color=BLUE,
            linewidth=2, markersize=6, markerfacecolor="white",
            label="pymars 2.0, warm process",
        )  # fmt: skip
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cases n (10 covariates, degree 2, 21 terms)", color=INK2)
    ax.set_ylabel("fit time, seconds (median of 3)", color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right", handlelength=4)
    fig.tight_layout()
    fig.savefig(FIGURES / "speed.png", dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-run", type=Path)
    ap.add_argument(
        "--run-dirs", type=Path, nargs="*", default=[],
        help="per-repetition run folders, for the clipped-probability counts",
    )  # fmt: skip
    a = ap.parse_args()
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    df = load()
    sk = load_skew()
    gap_tables(df)
    p = parity(df)
    parity_figure(p)
    q = parity_rerun(df, p)
    parity_figure(
        q.assign(ratio=q.new_ratio, lo=q.new_lo, hi=q.new_hi, decision=q.new_decision),
        "parity_rerun.png",
        "P-fix after #185",
    )
    box_figure(df)
    sparsity(df)
    binary(df)
    binary_appendix(df, a.run_dirs)
    skew(sk)
    failures(df, sk)
    if a.bench_run is not None or (HERE / "bench_times.csv").is_file():
        speed_figure(bench_times(a.bench_run))


if __name__ == "__main__":
    main()
