"""Collect the P-fix rerun of the T21 grid at 286d994 (issue #173, PR #185).

Usage, from a worktree with the validation group installed:
    uv run --frozen --group validation python \
        validation/sims/results/t173-grid/analyze.py <runs folder>

The run: arm P-fix on all 51 cells in the runner worktree
``.worktrees/runner-t173`` at 286d9946cbff88b15cc3fbd53bf5a680ac340226 (the
head of PR #185 when the run started; the merge ab923ca changes a comment and
a docstring of ``pymars/_linalg.py`` only), with ``driver_t173.sh`` (the log
is ``driver_log_summary.txt``, the invocations are ``manifest.json``). Same
cells, seeds and repetitions as the study at the freeze commit 0274c5a: the
repetitions of a cell are ``0:n_sim`` with the n_sim of ``../full/parity.csv``.
286d994 also holds #172 (the fix for #164), #167 and other changes made after
the freeze, so the rerun does not isolate the pair rule; phase 1 of #173 does.

Writes ``results.csv`` (one row per fit, with the columns of
``../full/results.csv``) and ``failure_counts.csv``. Only the arm P-fix ran;
the E-def fits that its ratios use are the stored ones in ``../full``, and
``validation/sims/report/make_report.py`` pairs the two by cell and
repetition and builds every table from these files. Nothing here fits a model.
"""

from __future__ import annotations

import sys
from pathlib import Path

from validation.sims import summarize

HERE = Path(__file__).parent
runs = Path(sys.argv[1])

df = summarize.load_results(runs).drop(columns=["extra"])
assert set(df.arm) == {"P-fix"}, set(df.arm)
df.to_csv(HERE / "results.csv", index=False)

fail = (
    df.assign(failed=df.error.notna())
    .groupby(["dgp", "n", "noise", "arm"], dropna=False)
    .agg(attempted=("rep", "size"), failed=("failed", "sum"))
    .reset_index()
)
fail.to_csv(HERE / "failure_counts.csv", index=False)
print("failed units:", int(fail.failed.sum()), "of", int(fail.attempted.sum()))
print("fit core-hours:", df.fit_seconds.sum() / 3600)
