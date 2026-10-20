"""Collect the skewed-covariate study of PR #185's pair rule (issue #173).

Usage, from a worktree with the validation group installed:
    uv run --frozen --group validation python \
        validation/sims/results/skew/analyze.py <old out> <new out>

The study: lognormal covariates Z = exp(Phi^{-1}(U)) with the truth f(U), low
noise, made by ``validation/sims/skew_sims.py``. ``<old out>`` holds P-fix at
``fdaca4a`` (main before PR #185: the old pair rule), from the runner worktree
``.worktrees/runner-t173-skew-old``. ``<new out>`` holds P-fix at ``ac67b38``
(PR #185's head when the run started; the same behavior as the merge
``ab923ca``) and E-def (earth 5.3.4, degree 2), from
``.worktrees/runner-t173-skew-new``. Between ``fdaca4a`` and ``ac67b38`` the
only change to ``pymars/`` is LA-7's threshold. The drivers
``driver_skew_stage1.sh`` and ``driver_skew_stage2.sh`` ran from
``validation/runs/`` of each runner, 4 workers each (see
``driver_log_summary_*.txt`` and ``manifest_*.json``).

The design, fixed before the runs: a screen of D1 to D5, D7 and D8 at n = 200
with 1,000 repetitions, and D4 at n = 1,000 as a pilot; then the confirmation
of D4 at n = 200 and n = 1,000 with 7,000 repetitions each. The extension of
D8 at n = 200 to 23,000 repetitions was added after the screen and is not part
of the fixed design. The seeds nest, so the first 1,000 repetitions of an
extended cell are its screen. A repetition enters only when all three fits
exist and the three result files carry the same ``data_sha256``; the script
stops otherwise.

Writes ``results.csv`` in the layout of ``../full/results.csv``, one row per
fit, with the arms ``P-old``, ``P-new`` and ``E-def``, the DGP name without
``skew`` (every cell has lognormal covariates), and ``data_hash``, the first
12 hex digits of ``data_sha256``. Writes ``failure_counts.csv`` as well.
``validation/sims/report/make_report.py`` builds every table from
``results.csv``. Nothing here fits a model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
ARMS = (("old", "P-fix", "P-old"), ("new", "P-fix", "P-new"), ("new", "E-def", "E-def"))
COLUMNS = [
    "dgp",
    "n",
    "noise",
    "arm",
    "rep",
    "error",
    "excess_risk",
    "n_terms",
    "data_hash",
]


def cell_arm(out: Path, cell: str, arm: str) -> dict[int, dict]:
    found = {}
    for path in sorted((out / cell / arm).glob("rep*.json")):
        try:
            found[int(path.stem[len("rep") :])] = json.loads(path.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            continue  # an interrupted write is not a result
    return found


def main() -> None:
    outs = {"old": Path(sys.argv[1]), "new": Path(sys.argv[2])}
    cells = sorted(
        {c.name for c in outs["old"].iterdir() if (c / "P-fix").is_dir()}
        & {c.name for c in outs["new"].iterdir() if (c / "P-fix").is_dir()}
    )
    rows = []
    for cell in cells:
        dgp, rest = cell.split("skew_n", 1)  # D4skew_n00200_lo
        n_text, noise = rest.split("_", 1)
        n = int(n_text)
        data = {name: cell_arm(outs[w], cell, arm) for w, arm, name in ARMS}
        reps = sorted(set.intersection(*(set(d) for d in data.values())))
        for rep in reps:
            shas = {data[name][rep].get("data_sha256") for _, _, name in ARMS}
            if len(shas) != 1 or None in shas:
                raise SystemExit(f"{cell} rep {rep}: data differ across arms: {shas}")
            for _, _, name in ARMS:
                rec = data[name][rep]
                rows.append(
                    {
                        "dgp": dgp,
                        "n": n,
                        "noise": noise,
                        "arm": name,
                        "rep": rep,
                        "error": rec.get("error"),
                        "excess_risk": rec.get("excess_risk"),
                        "n_terms": rec.get("n_terms"),
                        "data_hash": shas.copy().pop()[:12],
                    }
                )
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(HERE / "results.csv", index=False)
    fail = (
        df.assign(failed=df.error.notna())
        .groupby(["dgp", "n", "noise", "arm"], dropna=False)
        .agg(attempted=("rep", "size"), failed=("failed", "sum"))
        .reset_index()
    )
    fail.to_csv(HERE / "failure_counts.csv", index=False)
    print("failed units:", int(fail.failed.sum()), "of", int(fail.attempted.sum()))
    print(fail.groupby(["dgp", "n"]).attempted.first().to_string())


if __name__ == "__main__":
    main()
