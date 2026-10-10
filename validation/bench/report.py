"""Write REPORT.md from the results of ``run.py`` (VALIDATION_PLAN.md,
"Speed and scaling"): a table of times by factor, the log-log slopes with the
model sizes reached, and the ratio to earth at 10,000 cases, 10 covariates,
degree 2 and 21 terms (the definition of done, item 5).

    python validation/bench/report.py [--run DIR] [--out validation/bench/REPORT.md]
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import core  # noqa: E402

LABELS = {
    "pymars_fit": "pymars fit_mars",
    "pymars_est": "pymars EarthRegressor",
    "earth_default": "earth defaults",
    "earth_fast0": "earth fast.k=0",
    "legacy": "legacy 1.0.4",
}
FACTOR_NAMES = {
    "n": "cases n",
    "p": "covariates p",
    "degree": "degree",
    "max_terms": "term limit",
    "weights": "weights",
}


def load(run: Path) -> list[dict]:
    return [
        json.loads(f.read_text())
        for f in sorted((run / "results").glob("*.json"))
        if f.suffix == ".json"
    ]


def index(records: list[dict]) -> dict[core.Cell, dict]:
    """One record per cell: the newest by its start time (a results folder can
    hold files of older code versions)."""
    newest: dict[core.Cell, dict] = {}
    for r in sorted(records, key=lambda r: r.get("started", "")):
        newest[core.Cell(**r["cell"])] = r
    return newest


def seconds(s: float) -> str:
    if s >= 100:
        return f"{s:,.0f}"
    if s >= 1:
        return f"{s:.2f}"
    return f"{s:.4f}" if s < 0.01 else f"{s:.3f}"


def entry(rec: dict | None) -> str:
    if rec is None:
        return "not run"
    st = rec["status"]
    if st == "timeout":
        return f">{rec['timeout_s'] / 60:.0f} min"
    if st == "skipped":
        return "skipped"
    if st == "error":
        return "error"
    return f"{seconds(rec['time_median'])} ({rec['terms']}/{rec['forward_terms']})"


def mem(rec: dict | None) -> str:
    if rec is None or rec["status"] != "ok":
        return "-"
    return f"{rec['rss_median_bytes'] / 2**20:,.0f}"


def value_label(factor: str, v) -> str:
    return ("on" if v else "off") if factor == "weights" else f"{v:,}"


def build(run: Path) -> str:
    recs = index(load(run))
    meta = (
        json.loads((run / "meta.json").read_text())
        if (run / "meta.json").is_file()
        else {}
    )
    systems = [s for s in core.SYSTEMS if any(c.system == s for c in recs)]
    out = ["# Benchmark report", ""]
    out += [
        "Written by `validation/bench/report.py` from the results of `validation/bench/run.py`. "
        "Each point is the median of "
        f"{meta.get('reps', core.REPS)} runs, each in a fresh process with one thread; "
        f"the timeout is {meta.get('timeout_s', core.TIMEOUT_S) / 60:g} min per fit. "
        "Times are wall seconds of the fit alone. In the tables a cell reads "
        "`time (selected terms/forward terms)`. The data are Friedman #1 with fixed seeds, "
        "the same for every system; the baseline is 1,000 cases, 10 covariates, degree 2 "
        "and a limit of 21 terms. `not run` means the setting is not in the design for that "
        "system (the legacy code stops at 2,000 cases and takes no weights) or has not finished. "
        "Each run is a fresh process, so one-time start costs of pymars fall inside the "
        "timed fit; the numbers are the cost of a single fit, not of a repeated one.",
        "",
    ]
    if meta:
        out += [
            f"Run on {meta.get('cpu') or meta.get('machine')} ({meta.get('cpu_count')} cores), "
            f"Python {meta.get('python')}, numpy {meta.get('numpy')}, {meta.get('R')}, "
            f"earth {meta.get('earth')}, {meta.get('jobs')} cells at a time, "
            f"1-minute load average {meta.get('load_1min_at_start') or 0:.1f} at the start, "
            f"pymars commit `{str(meta.get('commit'))[:10]}`. "
            "Several cells ran at once on a shared machine, so times carry some noise.",
            "",
        ]

    # Definition of done, item 5
    out += ["## Ratio to earth at 10,000 cases", ""]
    row = {s: recs.get(core.Cell(s, n=10000)) for s in systems}
    if all(
        row.get(s) and row[s]["status"] == "ok" for s in ("pymars_fit", "earth_default")
    ):
        e = row["earth_default"]["time_median"]
        pm = row["pymars_fit"]["time_median"]
        out += [
            "10,000 cases, 10 covariates, degree 2, 21 terms "
            "(the target of the plan: within 10 times earth's time).",
            "",
            "| System | Time (s) | Ratio to earth defaults | Ratio to earth fast.k=0 |",
            "|---|---|---|---|",
        ]
        f0 = row.get("earth_fast0")
        for s in systems:
            r = row[s]
            if r is None or r["status"] != "ok":
                out.append(f"| {LABELS[s]} | {entry(r)} | | |")
                continue
            t = r["time_median"]
            r0 = f"{t / f0['time_median']:.1f}" if f0 and f0["status"] == "ok" else "-"
            out.append(f"| {LABELS[s]} | {seconds(t)} | {t / e:.1f} | {r0} |")
        out += [
            "",
            f"pymars `fit_mars` is {pm / e:.1f} times earth at its defaults.",
            "",
        ]
    else:
        out += ["The 10,000-case cells are not finished yet.", ""]

    # Tables by factor
    out += ["## Times by factor", ""]
    for factor, values in core.FACTORS.items():
        out += [f"### {FACTOR_NAMES[factor]}", ""]
        out += [
            "| "
            + FACTOR_NAMES[factor]
            + " | "
            + " | ".join(LABELS[s] for s in systems)
            + " |",
            "|---" * (len(systems) + 1) + "|",
        ]
        for v in values:
            cells = [
                recs.get(core.Cell(s, **{**core.BASE, factor: v})) for s in systems
            ]
            out.append(
                f"| {value_label(factor, v)} | "
                + " | ".join(entry(c) for c in cells)
                + " |"
            )
        out.append("")

    # Slopes
    out += ["## Log-log slopes", ""]
    out += [
        "The slope of log time on log of the factor, by least squares over the finished points "
        "(the number of points in brackets). Early stops change the amount of work, so read the "
        "slopes with the model sizes in the tables above. The slope for the degree has three "
        "points and the weights have two, so the tables give those.",
        "",
        "| System | "
        + " | ".join(FACTOR_NAMES[f] for f in ("n", "p", "max_terms"))
        + " |",
        "|---" * 4 + "|",
    ]
    for s in systems:
        cols = []
        for factor in ("n", "p", "max_terms"):
            xs, ys = [], []
            for v in core.FACTORS[factor]:
                r = recs.get(core.Cell(s, **{**core.BASE, factor: v}))
                if r and r["status"] == "ok":
                    xs.append(float(v))
                    ys.append(r["time_median"])
            sl = core.loglog_slope(xs, ys)
            cols.append("-" if sl is None else f"{sl[0]:.2f} ({sl[1]})")
        out.append(f"| {LABELS[s]} | " + " | ".join(cols) + " |")
    out += [""]

    # Memory
    out += [
        "## Peak memory",
        "",
        "Peak resident set size of the whole process in MiB (median of the runs; for R it "
        "includes R itself, about 100 MiB), and for the Python systems the peak Python "
        "allocation under `tracemalloc` in MiB, from one extra run for fits under a minute.",
        "",
        "| Setting | " + " | ".join(LABELS[s] for s in systems) + " |",
        "|---" * (len(systems) + 1) + "|",
    ]
    for factor, values in core.FACTORS.items():
        for v in values:
            cells = [
                recs.get(core.Cell(s, **{**core.BASE, factor: v})) for s in systems
            ]
            parts = []
            for c in cells:
                if c is not None and c["status"] == "ok" and "py_peak_bytes" in c:
                    parts.append(f"{mem(c)} / {c['py_peak_bytes'] / 2**20:,.0f}")
                else:
                    parts.append(mem(c))
            out.append(
                f"| {FACTOR_NAMES[factor]} {value_label(factor, v)} | "
                + " | ".join(parts)
                + " |"
            )
    out.append("")
    # Unfinished cells
    bad = [r for r in recs.values() if r["status"] not in ("ok",)]
    if bad:
        out += ["## Cells without a time", ""]
        for r in sorted(bad, key=lambda r: core.Cell(**r["cell"]).label()):
            why = r.get("reason") or r.get("error", "")[:120] or r["status"]
            out.append(f"- {core.Cell(**r['cell']).label()}: {r['status']} ({why})")
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run", default=str(HERE.parents[1] / "validation" / "runs" / "bench")
    )
    ap.add_argument("--out", default=str(HERE / "REPORT.md"))
    a = ap.parse_args()
    text = build(Path(a.run))
    Path(a.out).write_text(text)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
