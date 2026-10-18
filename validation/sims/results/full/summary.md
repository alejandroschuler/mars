# Full run (T21, issue #23)

Runner worktree `.worktrees/runner-t21` at the freeze commit
`0274c5a039d5a087ab55515e535e3d6370b68f56` (spec v3, numba knot scan; numba
0.68.0, earth 5.3.4). Arms P-fix and E-def on all 51 cells, 6 workers under
`nice -n 15`, one BLAS and one numba thread per worker, `--resume`. Wall time
14:05:26 to 14:26:06 UTC on 2026-10-18 (20.7 min). Per-repetition files are in
`validation/runs/full` (ignored); `results.csv` holds all of them.

## Numbers

- 58,400 units (51 cells, 2 arms, per-cell n_sim below), 0 failed
  (`failure_counts.csv` has no failure in any cell or arm).
- Core-hours: the sum of fit times is 2.27 h (`fit_times.csv`); the wall time
  on 6 workers bounds the cost at 2.07 core-hours plus start-up. The first
  100 repetitions of every cell came from the shared prediction cache that the
  pilot filled, so the fit times of those units are the cached ones.
- Repetitions per cell, from the post-freeze pilot (`pilot_sizing.csv`,
  `pilot_size.py`, `driver_pilot3*.sh`): 300 in 38 cells, 600 in 6, 1,300 in 4
  and 3,000 in 3 (D1 n=200 hi, D6 n=200, D6 n=1000). The seeds come from the
  cell name and repetition, so the tiers nest and the pilot repetitions are the
  first repetitions. Three cells used 3,000 repetitions, above the plan's 1,400
  cap: the pilot asked for 2,853, 2,172 and 1,705, and the cost was small.
- The pilot sized the cells with the plan's equivalence rule, n_sim =
  (2k s_p / Delta)^2, k = 3, Delta = log 1.05, from 100 repetitions per cell.
  Eight cells had s_p = 0 (identical fits) and got 300.

## Parity, P-fix / E-def (`parity.csv`)

Equivalence holds when the 3-SE interval for the mean log ratio lies inside
+-log 1.05. It holds in 44 of 51 cells (8 of them trivially, with identical
fits). It does not hold in 7: D1 n=1000 hi and lo, D1 n=5000 hi, D5 n=200 lo,
and D6 at n=200, 1000 and 5000. Their ratios are 1.023 to 1.068, and their 3-SE intervals reach above 1.05.

## Other tables (`summaries/`)

Built with `validation.sims.summarize` on the full run pooled with T04's
legacy results (P-cur, P-ear, E-pym, OLS, HGB, EarthClassifier, GLMEarth).
The E-def values are from this run (same seeds as T04's). Known gaps from
`summarize.py`, not changed here: the D6 and binary-DGP rows of the ratio
table and the legacy rows of the binary table show `missing` (a None noise
level is not matched against the legacy rows). No figures.
