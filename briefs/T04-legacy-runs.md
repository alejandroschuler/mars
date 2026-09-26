# Brief: T04 legacy runs (pilot, then the full legacy run)

You run the simulations of the legacy code (mars-earth 1.0.4) and of earth, with the harness that T03 merged into `validation/sims/`. Read `briefs/COMMON.md` (same folder) first; its rules hold for you, in particular "Long jobs". Issue: T04, #6. Reviewer: 1 (`single`). Invoke the skill `stats-research:design-and-report-simulations` first.

## Plan sections to read

"Statistical performance study" ("Goals", "Mockups", "Learners and settings", "Pilot and number of repetitions", "Compute ledger"), "Long computations", "Concurrency and cores", and `validation/sims/README.md` on `origin/main`.

## Where the jobs run

- A runner worktree at a fixed commit, outside any session worktree: `git -C /Users/aschuler/Documents/research/projects/pymars worktree add --detach /Users/aschuler/Documents/research/projects/pymars/.worktrees/runner-legacy <main-sha>`, where `<main-sha>` is `origin/main` after T03 merged. Record the SHA. Make its venvs there (`uv sync --frozen --group dev --group validation`, and `validation/legacy/make_venv.sh` for `.venv-legacy`), after reading the scripts.
- Every job: `nohup caffeinate -i nice -n 15 <command> > <log> 2>&1 &`, with `echo $! > <pidfile>`, both in the run's output folder under `validation/runs/` of the runner worktree. Source `dev/env.sh` first. At most 6 workers in total across your jobs (the executor may lower this with a restart and `--resume` when the load is high).
- Your own editable worktree for the pull request is `<S>/.worktrees/t04-legacy-runs` (branch `t04-legacy-runs` from `origin/main`); it holds only the run configuration, the pilot report and, at the end, the results and summaries.

## Steps

1. Claim the issue. Make both worktrees.
2. Check run: 2 repetitions per cell for the arms E-def, E-pym, P-cur, P-ear, OLS and HGB at 200 cases on all regression cells except the D7 legacy arms, and the binary arms at 200 cases. Fix nothing in the harness yourself; if the harness has a bug, report it to the executor with a minimal reproduction and wait.
3. Order of work, changed by the executor to save Claude usage (the weekly limit is near): do not wait for a separate pilot. Start the full legacy run at the plan's caps right after the check run passes, as detached jobs on 6 workers in total. Run the pilot cells first (D4 and D5 at 200 cases, both noise levels, arms P-cur, P-ear, E-def and E-pym), so that their first 100 repetitions finish early; seeds come from names, so these are the pilot repetitions. The other blocks follow in this order:
   - 200 cases, regression: the other cells without D7, P-cur and P-ear, 300 repetitions;
   - 200 cases, binary: D3-bin and D4-bin with the 1.0.4 `EarthClassifier` only (the reviewer found that `GLMEarth` gives bitwise-identical probabilities in 1.0.4; fit `GLMEarth` on 20 repetitions per cell to confirm, and say so in the report), 300 repetitions;
   - 1,000 cases: D3, D4, D5 and D8 at the low-noise level, 200 repetitions, P-cur and P-ear;
   - the cheap arms: E-def with the same repetitions as the legacy arms on their cells, E-pym on the gap cells only, OLS and HGB on all 51 cells up to 300 repetitions per cell (the parity run in T21 adds E-def repetitions where P-fix needs them);
   - not now: the low-priority batch, and D7 for the legacy arms.
   Use `--resume` so that a restart loses nothing. Record the command lines, the PIDs, the logs and the expected core-hours per block.
4. Later (a later agent does this when the executor asks): `pilot_check.py` on the first 100 repetitions of the pilot cells for the contrasts P-cur/E-def, P-ear/E-def and E-pym/E-def, and `validation/sims/pilot_legacy.md` with the pilot means, standard deviations, z statistics, `n_sim` and `n_sim_safe` per contrast and cell, and whether the caps (300 at 200 cases, 200 at 1,000 cases) suffice.
5. Open a draft pull request (`Part of #6`) with the run configuration, as soon as the full run is started. Then reply to the executor and stop: the executor checks the jobs on each wake. Do not wait for the full run.
6. Later, when the executor restarts you (or a new T04 agent) after the full run: collect the per-repetition results into `validation/sims/results/legacy/` (compact CSV), run `summarize.py` for the displays that the legacy arms fill (the ratio table, the box plots for P-cur, P-ear and E-pym, the selection table for E-def and P-cur, the binary table for the 1.0.4 classifiers), report the failure counts and the core-hours, and mark the pull request ready (`Closes #6`).

## Report

After step 5, in 30 lines or fewer: the runner SHA, the check-run result, the jobs started (command, PID file, log, expected core-hours), and anything the executor must watch.
