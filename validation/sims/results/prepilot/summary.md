# Pre-freeze pilot (T21 part a, issue #23)

This is a pre-freeze pilot. It ran on `origin/main` at `02f788c`, before the
tag `sim-freeze-1`, so P-fix here is today's `EarthRegressor`, not the frozen
one. It sizes the parity run and its compute. The plan's post-freeze pilot
may be rerun or skipped later.

## What ran

Runner worktree `.worktrees/runner-t21a`, commit
`02f788cfa006765e022dcfd04e0e5846260798b4`, one driver (`driver_t21a.sh`,
log `driver.log`), 4 workers under `nice -n 15`, 2026-10-10 02:10 to 02:19
UTC (about 8.6 minutes). Results per cell and repetition in
`validation/runs/prepilot` (ignored); `results.csv` here holds all of them.

| Block | Cells | Arms | n_sim per cell |
|---|---|---|---|
| Step 1 | all 51 | E-def, E-pym, OLS, HGB, P-fix | 2 |
| Step 2 | the 10^6-case diagnostics, recomputed | | |
| E-def on D7, the T04 gap | D7 at 200 cases, both noise levels | E-def | 300 |
| Step 3 | D3, D4, D5, D7 at 1,000 cases, both noise levels | P-fix, E-def | 100 |

- Step 1: 486 units ran, none failed (`failure_counts.csv` has no failures).
  `EarthClassifier` fits all six binary cells.
- Step 2: the recomputed `diagnostics.json` equals the committed one
  exactly (`diagnostics_recomputed.json` is not kept; the check printed
  `True` in `driver.log`).
- Step 3: 1,568 units ran (the other 32 were step 1's), none failed.
- Core-hours: the sum of fit times is 0.41 h. Wall time 8.6 minutes on 4
  workers bounds the cost at 0.57 core-hours, including R start-up and data
  generation.

## Sizing: parity contrast P-fix / E-def, 1,000 cases, 100 repetitions

`n_sim` is the plan's headline number, (2k s_p / Δ)² with k = 3 and
Δ = log 1.05, from `pilot_check.equivalence_n_sim`. `n_at_gap` is the
requirement at the observed gap. No cell has |ḡ_p| near Δ (the largest is
0.021, D5 low noise), so equivalence is feasible everywhere.

| Cell | ratio | ḡ_p | s_p | n_sim | n_at_gap | sec per rep (both arms) | core-hours at n_sim |
|---|---:|---:|---:|---:|---:|---:|---:|
| D3 lo | 1.0000 | 3e-15 | 2e-14 | degenerate | degenerate | 0.46 | |
| D3 hi | 1.0011 | 0.0011 | 0.0674 | 69 | 18 | 0.42 | 0.008 |
| D4 lo | 1.0037 | 0.0037 | 0.0389 | 23 | 7 | 0.59 | 0.004 |
| D4 hi | 1.0043 | 0.0043 | 0.0313 | 15 | 5 | 0.56 | 0.002 |
| D5 lo | 1.0211 | 0.0208 | 0.1932 | 564 | 430 | 0.39 | 0.061 |
| D5 hi | 1.0109 | 0.0108 | 0.1643 | 408 | 169 | 0.41 | 0.047 |
| D7 lo | 0.9963 | -0.0037 | 0.0188 | 6 | 2 | 5.58 | 0.008 |
| D7 hi | 1.0070 | 0.0070 | 0.0761 | 88 | 30 | 3.78 | 0.092 |

- On D3 at low noise, P-fix and E-def return the same fit to machine
  precision (the 100 log ratios are all within 1e-13 of 0; both select the
  same number of terms in every repetition). s_p is 0, so `n_sim` is not
  defined; any n_sim shows equivalence there.
- D5 (the hinge interaction) sets the repetitions: s_p of 0.16 to 0.19 needs
  about 400 to 560. Every other cell needs 90 or fewer.
- Summed over the eight cells at their own `n_sim`, the fits cost 0.22
  core-hours. At the largest, 564 repetitions in every cell, they cost 1.9
  core-hours. These are 1,000-case figures only. Fit times at 200 and 5,000
  cases are in `fit_times.csv` (2 repetitions per cell, so rough): P-fix
  takes about 0.2 s, 0.5 s and 0.7 to 2.7 s at 200, 1,000 and 5,000 cases on
  p = 10, and 2.6 s, 4.9 s and 13 s on D7. s_p at other sample sizes was not
  piloted.

## E-def on D7 at 200 cases (the T04 gap)

E-def now covers D7 at 200 cases for 300 repetitions. Paired with the E-pym
results of T04 (`../legacy/results.csv`, same datasets, since seeds come from
the cell name and repetition), the contrast E-pym / E-def on D7 is a
difference contrast:

| Cell | n_pilot | ratio | ḡ_p | s_p | z | n_sim (k = 5) | n_sim_safe |
|---|---:|---:|---:|---:|---:|---:|---:|
| D7 n200 lo | 100 | 0.628 | -0.466 | 0.908 | -5.1 | 95 | 249 |
| D7 n200 hi | 100 | 0.638 | -0.450 | 1.016 | -4.4 | 128 | 411 |
| D7 n200 lo, all 300 | 300 | 0.600 | -0.512 | 0.817 | -10.8 | 64 | 95 |
| D7 n200 hi, all 300 | 300 | 0.651 | -0.429 | 1.013 | -7.3 | 140 | 260 |

E-pym has a lower excess risk than E-def on D7 (ratio about 0.6 to 0.65),
and the 300-repetition run resolves it at both noise levels.

## Files

`results.csv` (every unit), `failure_counts.csv`, `fit_times.csv`,
`sizing.csv`, `manifest.json`, `driver_t21a.sh`, `driver.log`, and
`analyze.py` (builds the CSVs; run it with the runs folder and
`../legacy/results.csv`).
