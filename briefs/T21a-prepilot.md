# Brief: T21 part a, a sizing pilot for the parity claim before the freeze (#23)

Read `COMMON.md` (same folder) first; its rules hold for you. Issue: #23 (T21). Branch: `t21-prepilot`. Reviewer: 1 (`single`).

Read `VALIDATION_PLAN.md` on `origin/main`: "Statistical performance study" (all of it: claims, arms, DGPs, learners, measures, build, "Pilot and number of repetitions", "Compute ledger"), and "Long computations". Read `validation/sims/` (the harness of T03: run.py with --resume, summaries, pilot_check.py) and how T04 ran the legacy pilot and full run (`git log` for #41, `validation/sims/results/legacy/`).

## Why now

The freeze tag `sim-freeze-1` waits for the spec v2 follow-ups. The code changes they bring are small for prediction error, so a pilot on today's main estimates s_p well enough to size the full run and its compute. It is labeled a pre-freeze pilot everywhere; the plan's post-freeze pilot may be rerun or skipped later by the executor.

## Task

1. Pilot steps 1 and 2 on the new code (2 repetitions per cell; one large draw per DGP with the diagnostics).
2. Pilot step 3 for the parity claim: 100 repetitions at 1,000 cases on D3, D4, D5 and D7 at both noise levels, arms P-fix and E-def. Also E-def on D7 at the legacy cells' settings, which T04 never ran (the gap noted in T04's summary), so that E-pym/E-def on D7 can be filled.
3. Run them as detached jobs from a runner worktree `<main>/.worktrees/runner-t21a` at origin/main's current commit (record it), `nohup caffeinate -i nice -n 15`, at most 4 workers (R counts as one core; a benchmark job already uses 4 cores). Results per cell, atomic, with a manifest and --resume.
4. When the jobs finish, run `pilot_check.py` for the parity contrasts and write `validation/sims/results/prepilot/` (a short summary.md with n_sim per cell and the implied core-hours, plus the summary tables) in a PR `Part of #23`. If the jobs take longer than your session, report the PID file and log path and stop; the executor restarts you later.

Files: `validation/sims/results/prepilot/` (new), and harness changes in `validation/sims/` only if a bug blocks the run (say so). No `pymars/` changes. Add the core-hours to your report.
