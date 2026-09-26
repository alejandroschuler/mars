# Legacy simulation results (T04, issue #6)

This folder holds the finished legacy run of `VALIDATION_PLAN.md`'s
statistical performance study: mars-earth 1.0.4 (P-cur, P-ear,
EarthClassifier, GLMEarth) against R's earth (E-def, E-pym), plus the OLS
and HGB floors. `validation/sims/legacy_full_run.sh` produced it in the
runner worktree, at runner commit `8760ceefc5b2564318db47f547c3332535a1a5e4`
(`8760cee`, `origin/main` after T03's harness merge). This pull request was
opened as a draft while the run was starting (brief step 5); this is step
6, collecting and summarizing the finished run.

## What was run

- Arms: E-def, E-pym, P-cur, P-ear, EarthClassifier, GLMEarth, OLS, HGB.
- Cells: all 51 of the plan's grid, but not every arm on every cell.
  P-cur and P-ear cover 200 cases on D1-D6 and D8 (300 repetitions), the
  200-case binary cells (EarthClassifier 300 repetitions, GLMEarth 20),
  and 1,000 cases on D3, D4, D5 and D8 at the low-noise level (200
  repetitions). Neither runs on D7 or at 5,000 cases (Compute ledger: D7
  is about 150 times more expensive for the legacy code). E-def and E-pym
  match the legacy arms' own cells at 200 and 1,000 cases; E-pym also
  covers D7 at 200 cases (a gap-claim cell), but E-def never runs on D7 at
  any size, because `legacy_full_run.sh` gives E-def the legacy arms' own
  cell list and the legacy arms skip D7. OLS and HGB run on all 51 cells,
  up to 300 repetitions, including 5,000 cases.
- Runner commit: `8760ceefc5b2564318db47f547c3332535a1a5e4`.
- Workers and times: `driver.log`, 6 workers, 2026-09-26 01:46 PDT to about
  11:08 PDT (about 9.35 hours); `driver_n4.log`, 4 workers, about 11:06 PDT
  to 16:13 PDT after a restart for load (about 5.12 hours). `driver_n4.log`
  ends `ALL BLOCKS DONE` at 16:13:36 PDT. About 76 core-hours in total
  (9.35 x 6 + 5.12 x 4 is about 76.6).
- 49,440 (cell, arm, repetition) result files in all, collected below.

## Files here

- `results.csv`: one row per (cell, arm, repetition): the dgp, sample
  size, noise level, arm, repetition, error type, the measures (excess
  risk for the regression cells; excess log loss, excess Brier score and
  calibration slope for the binary cells), the fit time, the number of
  terms, whether an irrelevant covariate was used, how many, and the
  covariates used. No predictions and no cache, per the brief.
- `manifest.json`: a copy of the run's own manifest, one entry per
  `run.py` invocation (18 in all; several are `--resume` no-ops logged
  again after the restart). Each entry has that invocation's arms, cells,
  repetition range and package versions.
- `failure_counts.csv`: attempted and completed fits per (dgp, n, noise,
  arm), 172 rows.
- `summaries/`: the displays built by `summarize.py`; see "Displays" below.

## Failure counts

Zero. All 49,440 attempted fits completed with no error, across all 172
(dgp, n, noise, arm) combinations this run touched. `results.csv` has no
non-blank error field, and every row of `failure_counts.csv` reads
`n_failed = 0`.

## EarthClassifier and GLMEarth

The brief asked for a check that `GLMEarth` stays bitwise-identical to
`EarthClassifier` in mars-earth 1.0.4, a finding from the T03 review; that
is why `GLMEarth` ran on only 20 repetitions per binary cell instead of the
full 300. `EarthClassifier` and `GLMEarth` agree exactly (`==`, not a
tolerance) on excess log loss, excess Brier score and calibration slope,
for all 40 shared (dgp, repetition) pairs: D3-bin and D4-bin at 200 cases,
repetitions 0-19. Predictions themselves are not stored in the result
files, so this compares every measure the predictions determine; an exact
match on all three, for every pair, is what bitwise-identical predictions
would produce.

## Displays

Built with `uv run --frozen python -m validation.sims.summarize`, through
its own `ratio_table`, `selection_table`, `binary_outcome_table`,
`appendix_table` and `box_plots` functions, unchanged, against this run's
results. `equivalence_figure.png` is not built here: it is entirely P-fix
versus E-def, and P-fix does not exist on this branch. Captions are in
`summaries/captions.md`, copied from `VALIDATION_PLAN.md`'s "Mockups". One
line each, no more:

- `ratio_table.csv`: P-cur/E-def, P-ear/E-def and E-pym/E-def at 200
  cases, low noise, for D1-D8; D7 reads "n/a" for P-cur/P-ear and
  "missing" for E-pym/E-def, since E-def never ran on D7 (above).
- `ratio_table_n1000.csv`: the same ratios at 1,000 cases; D3, D4, D5 and
  D8 (low noise) are filled, D1, D2 and D6 read "missing" (not run at
  1,000 cases here), D7 as above.
- `selection_table.csv`: median terms, IQR and irrelevant-covariate use
  for E-def, P-cur and P-fix on D4, D6, D7 and D8 at 200 cases; P-fix
  reads "missing" throughout, D7's E-def rows read "missing" and its
  P-cur rows read "n/a".
- `binary_outcome_table.csv`: excess-log-loss ratio and calibration slope
  for EarthClassifier and GLMEarth (200 cases) and P-fix (200, 1,000 and
  5,000 cases, all "missing") on D3-bin and D4-bin.
- `appendix_table.csv`: every (dgp, n, noise, arm) cell this run touched,
  with its mean log measure, its Monte Carlo standard error, and its
  n_ok / n_failed counts.
- `box_plots.png`: per-repetition log ratios against E-def at 200 cases,
  for P-cur, P-ear and E-pym (P-fix has no data yet and draws no box).

See `validation/sims/pilot_legacy.md` for the pilot sizing (brief step 4),
computed from this same run's first 100 repetitions of the D4/D5 pilot
cells. T22's report interprets these displays alongside the rest of the
statistical performance study; this file only says what was run and what
each display holds.
