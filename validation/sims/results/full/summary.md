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
+-log 1.05. After the extension below it holds in 48 of 51 cells (8 trivially,
with identical fits). The other 3:

- D6 n=200 (7,500 repetitions): ratio 1.076, interval [1.051, 1.101]. The
  whole interval is above 1.05, so this is a failure of the claim.
- D1 n=5000 hi (600 repetitions): ratio 1.038, interval [1.019, 1.056]. Not
  decidable at this n: the repetitions needed at the current s_g are about
  1,300, and s_g grew with every extension.
- D6 n=1000 (3,800 repetitions): ratio 1.030, interval [1.007, 1.053]. Not
  decidable at this n: about 5,000 repetitions at the current s_g.

## Extension (Part of #24)

The reviewer of PR #168 found that most of the 7 first-run failures were
under-sized. The 7 cells were extended on the same frozen runner (nested
seeds, `--resume`; `driver_extend.sh`, `driver_extend2.sh`, logs in
`driver_extend_log_summary.txt`) to: D1 n=1000 hi 900, lo 750, D1 n=5000 hi
400 then 600, D5 n=200 lo 900, D6 n=200 7,500, D6 n=1000 3,300 then 3,800, D6
n=5000 12,400 (the larger of the plan's rule (2k s/Delta)^2 and the count at the
observed gap, from the full-run s_g). Result: D1 n=1000 hi and lo, D5 n=200 lo
and D6 n=5000 now pass. Total 94,500 units, 0 failed, 3.74 fit core-hours
(2.27 before the extension). The disk filled during D6 n=5000 (the shared
prediction cache under `validation/runs/.cache` of the runner reached 16 GB);
the block was resumed with `--no-cache`, and the later blocks ran with it.

## Other tables (`summaries/`)

Built with `validation.sims.summarize` on the full run pooled with T04's
legacy results (P-cur, P-ear, E-pym, OLS, HGB, EarthClassifier, GLMEarth).
The E-def values are from this run (same seeds as T04's). Known gaps from
`analyze.py` now maps T04's noise string `na` to NaN, so the D6 ratio row, the
binary legacy rows and the D6 P-cur selection row are filled. No figures.
