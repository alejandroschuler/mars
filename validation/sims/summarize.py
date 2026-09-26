"""Build every display in VALIDATION_PLAN.md's "Mockups" from the
per-repetition result files ``run.py`` writes: the ratio table, the
equivalence figure, the per-repetition box plots, the selection table and the
binary-outcome table, plus appendix tables with Monte Carlo standard errors
and failure counts. A cell or arm comparison that has not been run yet is
shown as missing; this module never invents a number.

Usage: ``uv run --frozen python -m validation.sims.summarize --results DIR
--out DIR`` (``--results`` is a ``run.py --out`` folder; ``--out`` defaults
to ``<results>/report``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MISSING = "missing"

DGP_LABELS = {
    "D1": "D1 linear",
    "D2": "D2 additive hinges",
    "D3": "D3 additive smooth",
    "D4": "D4 Friedman #1",
    "D5": "D5 hinge interaction",
    "D6": "D6 pure noise",
    "D7": "D7 many covariates",
    "D8": "D8 correlated",
}
RATIO_TABLE_DGPS = list(DGP_LABELS)
EQUIVALENCE_MARGIN = np.log(1.05)


# ---------------------------------------------------------------------------
# Loading.
# ---------------------------------------------------------------------------


def _parse_cell_name(cell_name: str) -> tuple[str, int, str | None] | None:
    """The (dgp, n, noise) a ``dgps.Cell.name`` encodes, or ``None`` when
    ``cell_name`` is not a cell directory at all (for example ``report``,
    written by this module's own previous run into the same results folder).
    """
    if "_n" not in cell_name:
        return None
    dgp, rest = cell_name.split("_n", 1)
    parts = rest.split("_", 1)
    if not parts[0].isdigit():
        return None
    n = int(parts[0])
    noise = parts[1] if len(parts) > 1 else None
    return dgp, n, noise


def load_results(results_dir: Path) -> pd.DataFrame:
    """One row per (cell, arm, repetition) that has a result file on disk."""
    rows = []
    for cell_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        parsed = _parse_cell_name(cell_dir.name)
        if parsed is None:
            continue
        dgp, n, noise = parsed
        for arm_dir in sorted(p for p in cell_dir.iterdir() if p.is_dir()):
            for result_file in sorted(arm_dir.glob("rep*.json")):
                try:
                    data = json.loads(result_file.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                row = {
                    "dgp": dgp,
                    "n": n,
                    "noise": noise,
                    "arm": arm_dir.name,
                    "rep": int(result_file.stem[len("rep") :]),
                }
                row.update(data)
                rows.append(row)
    columns = [
        "dgp",
        "n",
        "noise",
        "arm",
        "rep",
        "error",
        "excess_risk",
        "excess_log_loss",
        "calibration_slope",
        "n_terms",
        "uses_irrelevant_covariate",
        "n_irrelevant_covariates",
        "fit_seconds",
    ]
    return (
        pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)
    )


# ---------------------------------------------------------------------------
# The shared paired-log-ratio computation behind the ratio table, the
# equivalence figure, the box plots and the binary-outcome table.
# ---------------------------------------------------------------------------


def _noise_mask(df: pd.DataFrame, noise: str | None) -> pd.Series:
    """``df.noise == noise``, but correct when ``noise`` is ``None`` (D6, the
    binary DGPs): a plain ``==`` against ``None`` is false for every row in
    pandas, so D6 and the binary DGPs would silently show as never run.
    """
    return df.noise.isna() if noise is None else df.noise == noise


def _noise_for(dgp: str, noise: str) -> str | None:
    """D6 has one noise level, stored as ``None`` (dgps.Cell); every other
    regression DGP in the ratio table, the equivalence figure and the box
    plots has both "lo" and "hi".
    """
    return None if dgp == "D6" else noise


def paired_log_ratio(
    df: pd.DataFrame,
    dgp: str,
    n: int,
    noise: str | None,
    arm_a: str,
    arm_b: str,
    measure: str,
) -> dict | None:
    """Mean and Monte Carlo SE of g_i = log(measure_a) - log(measure_b),
    paired by repetition, plus the failure count for each arm. ``None`` when
    neither arm has any successful, paired repetition yet.
    """
    cell = df[(df.dgp == dgp) & (df.n == n) & _noise_mask(df, noise)]
    a = cell[cell.arm == arm_a]
    b = cell[cell.arm == arm_b]
    a_ok = a[a.error.isna()].set_index("rep")[measure]
    b_ok = b[b.error.isna()].set_index("rep")[measure]
    common = a_ok.index.intersection(b_ok.index)
    n_fail_a = int(a.error.notna().sum())
    n_fail_b = int(b.error.notna().sum())
    if len(common) == 0:
        return None
    g = np.log(a_ok.loc[common].to_numpy()) - np.log(b_ok.loc[common].to_numpy())
    n_sim = len(g)
    g_bar = float(g.mean())
    s_g = float(g.std(ddof=1)) if n_sim > 1 else float("nan")
    se = s_g / np.sqrt(n_sim) if n_sim > 1 else float("nan")
    return {
        "g_bar": g_bar,
        "s_g": s_g,
        "se": se,
        "n_sim": n_sim,
        "ratio": float(np.exp(g_bar)),
        "ratio_lo": float(np.exp(g_bar - 3 * se)) if n_sim > 1 else float("nan"),
        "ratio_hi": float(np.exp(g_bar + 3 * se)) if n_sim > 1 else float("nan"),
        "n_fail_a": n_fail_a,
        "n_fail_b": n_fail_b,
    }


def _format_ratio_cell(stats: dict | None) -> str:
    if stats is None:
        return MISSING
    if stats["n_sim"] < 2:
        return f"{stats['ratio']:.3f} (n_sim=1, no interval)"
    return f"{stats['ratio']:.3f} [{stats['ratio_lo']:.3f}, {stats['ratio_hi']:.3f}]"


# ---------------------------------------------------------------------------
# Ratio table.
# ---------------------------------------------------------------------------


def ratio_table(df: pd.DataFrame, n: int = 200, noise: str = "lo") -> pd.DataFrame:
    """Excess risk of P-cur, P-ear and E-pym relative to E-def, at `n` cases
    and the `noise` level (VALIDATION_PLAN.md, "Ratio table"). Tests the gap
    claim; the two ablation columns show whether settings or rules cause it.
    """
    rows = []
    for dgp in RATIO_TABLE_DGPS:
        row = {"DGP": DGP_LABELS[dgp]}
        for label, arm in (
            ("P-cur / E-def", "P-cur"),
            ("P-ear / E-def", "P-ear"),
            ("E-pym / E-def", "E-pym"),
        ):
            if dgp == "D7" and arm in ("P-cur", "P-ear"):
                row[label] = "n/a (legacy code too slow at p=50)"
                continue
            stats = paired_log_ratio(
                df, dgp, n, _noise_for(dgp, noise), arm, "E-def", "excess_risk"
            )
            row[label] = _format_ratio_cell(stats)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Equivalence figure: P-fix / E-def, every DGP x sample size x noise level.
# ---------------------------------------------------------------------------


def equivalence_figure(df: pd.DataFrame, out_path: Path) -> None:
    """Log ratio of excess risk, P-fix over E-def, +/- 3 Monte Carlo SEs, one
    row per DGP and one panel per sample size (VALIDATION_PLAN.md,
    "Equivalence figure"). Tests the parity claim. A DGP/size/noise
    combination with no paired repetitions yet is left blank, not invented.
    """
    sizes = [200, 1000, 5000]
    dgps_here = RATIO_TABLE_DGPS
    fig, axes = plt.subplots(
        1, len(sizes), figsize=(4 * len(sizes), 0.5 * len(dgps_here) + 1), sharey=True
    )
    for ax, n in zip(axes, sizes, strict=True):
        y_labels = []
        for i, dgp in enumerate(dgps_here):
            y_labels.append(DGP_LABELS[dgp])
            for noise, marker, offset in (("lo", "o", 0.12), ("hi", "s", -0.12)):
                if dgp == "D6" and noise == "hi":
                    continue  # D6 has one noise level; "lo" plots it, once
                stats = paired_log_ratio(
                    df, dgp, n, _noise_for(dgp, noise), "P-fix", "E-def", "excess_risk"
                )
                if stats is None:
                    continue
                y = i + offset
                ax.errorbar(
                    stats["g_bar"],
                    y,
                    xerr=3 * stats["se"] if stats["n_sim"] > 1 else 0,
                    fmt=marker,
                    color="tab:blue" if noise == "lo" else "tab:orange",
                    capsize=2,
                )
        ax.axvspan(-EQUIVALENCE_MARGIN, EQUIVALENCE_MARGIN, color="0.85", zorder=0)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_title(f"n = {n}")
        ax.set_xlabel("log ratio (P-fix / E-def)")
        ax.set_yticks(range(len(dgps_here)))
        ax.set_yticklabels(y_labels)
    fig.suptitle(
        "Excess risk log ratio, P-fix / E-def, +/-3 MC SE "
        "(shaded: equivalence margin +/-log 1.05)"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-repetition box plots at n=200.
# ---------------------------------------------------------------------------


def box_plots(df: pd.DataFrame, out_path: Path, n: int = 200) -> None:
    """Per-repetition log ratios against E-def, at `n` cases, for P-cur,
    P-ear, E-pym and P-fix (VALIDATION_PLAN.md, "Per-repetition box plots").
    Shows whether a few bad fits drive the ratio table's and the equivalence
    figure's means.
    """
    arms = ["P-cur", "P-ear", "E-pym", "P-fix"]
    data, labels = [], []
    for dgp in RATIO_TABLE_DGPS:
        for arm in arms:
            for noise in ("lo", "hi") if dgp != "D6" else (None,):
                if dgp == "D7" and arm in ("P-cur", "P-ear"):
                    continue
                cell = df[(df.dgp == dgp) & (df.n == n) & _noise_mask(df, noise)]
                a = cell[(cell.arm == arm) & cell.error.isna()].set_index("rep")[
                    "excess_risk"
                ]
                b = cell[(cell.arm == "E-def") & cell.error.isna()].set_index("rep")[
                    "excess_risk"
                ]
                common = a.index.intersection(b.index)
                if len(common) == 0:
                    continue
                g = np.log(a.loc[common].to_numpy()) - np.log(b.loc[common].to_numpy())
                data.append(g)
                labels.append(f"{dgp}/{arm}/{noise or 'na'}")
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(labels)), 5))
    if data:
        ax.boxplot(data, tick_labels=labels)
        plt.setp(ax.get_xticklabels(), rotation=90)
    else:
        ax.text(0.5, 0.5, "no paired repetitions yet", ha="center", va="center")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("log ratio (arm / E-def)")
    ax.set_title(f"Per-repetition log ratios at n = {n}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Selection table.
# ---------------------------------------------------------------------------


def selection_table(df: pd.DataFrame, n: int = 200) -> pd.DataFrame:
    """Model size and selection on D4, D6, D7 and D8 at `n` cases, for E-def,
    P-cur and P-fix (VALIDATION_PLAN.md, "Selection table"). Supports the
    sparsity claim.
    """
    rows = []
    for dgp in ("D4", "D6", "D7", "D8"):
        for arm in ("E-def", "P-cur", "P-fix"):
            if dgp == "D7" and arm == "P-cur":
                continue
            noises = (None,) if dgp == "D6" else ("lo", "hi")
            for noise in noises:
                sub = df[
                    (df.dgp == dgp)
                    & (df.n == n)
                    & _noise_mask(df, noise)
                    & (df.arm == arm)
                    & df.error.isna()
                ]
                row = {"DGP": DGP_LABELS[dgp], "noise": noise or "na", "arm": arm}
                if sub.empty:
                    row.update(
                        {
                            "median_terms": MISSING,
                            "iqr_terms": MISSING,
                            "share_irrelevant": MISSING,
                            "mean_n_irrelevant": MISSING,
                        }
                    )
                else:
                    terms = sub["n_terms"].dropna()
                    row["median_terms"] = (
                        float(terms.median()) if len(terms) else float("nan")
                    )
                    row["iqr_terms"] = (
                        float(terms.quantile(0.75) - terms.quantile(0.25))
                        if len(terms)
                        else float("nan")
                    )
                    row["share_irrelevant"] = float(
                        sub["uses_irrelevant_covariate"].mean()
                    )
                    row["mean_n_irrelevant"] = float(
                        sub["n_irrelevant_covariates"].mean()
                    )
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Binary-outcome table.
# ---------------------------------------------------------------------------


def binary_outcome_table(df: pd.DataFrame) -> pd.DataFrame:
    """Excess log loss ratio (arm / E-def) and calibration slope, for
    EarthClassifier and GLMEarth (200 cases) and P-fix (200, 1,000 and 5,000
    cases), on D3-bin and D4-bin (VALIDATION_PLAN.md, "Binary-outcome table").
    """
    rows = []
    for dgp in ("D3-bin", "D4-bin"):
        for arm, sizes in (
            ("EarthClassifier", (200,)),
            ("GLMEarth", (200,)),
            ("P-fix", (200, 1000, 5000)),
        ):
            for n in sizes:
                stats = paired_log_ratio(
                    df, dgp, n, None, arm, "E-def", "excess_log_loss"
                )
                cal = df[
                    (df.dgp == dgp) & (df.n == n) & (df.arm == arm) & df.error.isna()
                ]["calibration_slope"]
                rows.append(
                    {
                        "DGP": dgp,
                        "n": n,
                        "arm": arm,
                        "excess_log_loss_ratio": _format_ratio_cell(stats),
                        "calibration_slope": float(cal.mean()) if len(cal) else MISSING,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Appendix: every cell, its Monte Carlo SE and its failure count.
# ---------------------------------------------------------------------------


def appendix_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dgp, n, noise, arm), group in df.groupby(
        ["dgp", "n", "noise", "arm"], dropna=False
    ):
        ok = group[group.error.isna()]
        measure = "excess_log_loss" if "-bin" in dgp else "excess_risk"
        values = ok[measure].dropna() if measure in ok else pd.Series(dtype=float)
        rows.append(
            {
                "dgp": dgp,
                "n": n,
                "noise": noise or "na",
                "arm": arm,
                "n_ok": len(ok),
                "n_failed": int(group.error.notna().sum()),
                "mean_log_measure": float(np.log(values).mean())
                if len(values)
                else float("nan"),
                "mc_se_log_measure": (
                    float(np.log(values).std(ddof=1) / np.sqrt(len(values)))
                    if len(values) > 1
                    else float("nan")
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["dgp", "n", "noise", "arm"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out_dir = args.out or (args.results / "report")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.results)
    print(f"loaded {len(df)} (cell, arm, repetition) results from {args.results}")

    # CSV, not Markdown: pandas' to_markdown needs the optional "tabulate"
    # package, which is not one of this project's dependencies.
    ratio_table(df).to_csv(out_dir / "ratio_table.csv", index=False)
    selection_table(df).to_csv(out_dir / "selection_table.csv", index=False)
    binary_outcome_table(df).to_csv(out_dir / "binary_outcome_table.csv", index=False)
    appendix_table(df).to_csv(out_dir / "appendix_table.csv", index=False)
    equivalence_figure(df, out_dir / "equivalence_figure.png")
    box_plots(df, out_dir / "box_plots.png")
    print(f"wrote the mockups' displays to {out_dir}")


if __name__ == "__main__":
    main()
