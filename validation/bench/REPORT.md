# Benchmark report

Written by `validation/bench/report.py` from the results of `validation/bench/run.py`. Each point is the median of 3 runs, each in a fresh process with one thread; the timeout is 30 min per fit. Times are wall seconds of the fit alone. In the tables a cell reads `time (selected terms/forward terms)`. The data are Friedman #1 with fixed seeds, the same for every system; the baseline is 1,000 cases, 10 covariates, degree 2 and a limit of 21 terms. `not run` means the setting is not in the design for that system (the legacy code stops at 2,000 cases and takes no weights) or has not finished. Each run is a fresh process, so one-time start costs of pymars fall inside the timed fit; the numbers are the cost of a single fit, not of a repeated one.

Run on Apple M1 Pro (10 cores), Python 3.12.13, numpy 2.5.3, R version 4.4.3 (2025-02-28), earth 5.3.4, 4 cells at a time, 1-minute load average 3.0 at the start, pymars commit `faac561e41`. Several cells ran at once on a shared machine, so times carry some noise.

**Notes added by hand (T22b; `report.py` does not write them).** The legacy times are those of the T19 run at `ae9d199` (the same worker and data, so the same cell keys); the legacy code does not change, and this run did not repeat its 2 hours. The timer starts after the imports and the data loading (`worker.py`), so import time is in no number here. The fresh-process times of pymars do include the first-call costs of the fit: numba loads the compiled kernels from its disk cache on the first call, and the first numpy and scipy calls warm up. Together these add about 0.3 s per process. A program that fits many models in one process pays them once. `validation/sims/report/warm_times.py` times `EarthRegressor` after a first fit in the same process (median of 3, `validation/sims/report/warm_times.csv`, at `faac561`'s `pymars/`):

| Cases | pymars, fresh (s) | pymars, warm (s) | earth defaults (s) | Factor, fresh | Factor, warm |
|---|---|---|---|---|---|
| 250 | 0.379 | 0.066 | 0.019 | 20 | 3.5 |
| 1,000 | 0.400 | 0.097 | 0.049 | 8.2 | 2.0 |
| 10,000 | 0.801 | 0.441 | 0.462 | 1.7 | 0.95 |
| 100,000 | 5.91 | 5.25 | 7.51 | 0.79 | 0.70 |

The fixed cost also flattens the log-log slopes below: from 10,000 to 100,000 cases the time of `EarthRegressor` grows by a factor of 7.4 (slope 0.87).

## Ratio to earth at 10,000 cases

10,000 cases, 10 covariates, degree 2, 21 terms (the target of the plan: within 10 times earth's time).

| System | Time (s) | Ratio to earth defaults | Ratio to earth fast.k=0 |
|---|---|---|---|
| pymars fit_mars | 0.764 | 1.7 | 1.6 |
| pymars EarthRegressor | 0.801 | 1.7 | 1.7 |
| earth defaults | 0.462 | 1.0 | 1.0 |
| earth fast.k=0 | 0.472 | 1.0 | 1.0 |
| legacy 1.0.4 | not run | | |

pymars `fit_mars` is 1.7 times earth at its defaults.

## Times by factor

### cases n

| cases n | pymars fit_mars | pymars EarthRegressor | earth defaults | earth fast.k=0 | legacy 1.0.4 |
|---|---|---|---|---|---|
| 250 | 0.378 (16/19) | 0.379 (16/19) | 0.019 (15/18) | 0.019 (15/18) | 25.79 (17/21) |
| 500 | 0.383 (18/19) | 0.382 (18/19) | 0.027 (18/18) | 0.028 (18/18) | 65.12 (15/21) |
| 1,000 | 0.398 (18/18) | 0.400 (18/18) | 0.049 (18/18) | 0.048 (18/18) | 144 (16/21) |
| 2,000 | 0.441 (16/18) | 0.447 (16/18) | 0.091 (16/18) | 0.092 (16/18) | 637 (16/21) |
| 5,000 | 0.588 (17/19) | 0.592 (17/19) | 0.230 (18/18) | 0.239 (18/18) | not run |
| 10,000 | 0.764 (17/19) | 0.801 (17/19) | 0.462 (17/19) | 0.472 (17/19) | not run |
| 100,000 | 6.01 (16/19) | 5.91 (16/19) | 7.51 (16/19) | 7.12 (16/19) | not run |

### covariates p

| covariates p | pymars fit_mars | pymars EarthRegressor | earth defaults | earth fast.k=0 | legacy 1.0.4 |
|---|---|---|---|---|---|
| 5 | 0.360 (16/19) | 0.360 (16/19) | 0.028 (16/18) | 0.028 (16/18) | 91.26 (16/21) |
| 10 | 0.398 (18/18) | 0.400 (18/18) | 0.049 (18/18) | 0.048 (18/18) | 144 (16/21) |
| 20 | 0.525 (17/19) | 0.513 (17/19) | 0.090 (17/19) | 0.090 (17/19) | 377 (18/21) |
| 50 | 0.809 (18/19) | 0.851 (18/19) | 0.213 (18/19) | 0.215 (18/19) | 794 (16/21) |
| 100 | 1.30 (16/19) | 1.32 (16/19) | 0.420 (16/19) | 0.423 (16/19) | 1,594 (16/21) |

### degree

| degree | pymars fit_mars | pymars EarthRegressor | earth defaults | earth fast.k=0 | legacy 1.0.4 |
|---|---|---|---|---|---|
| 1 | 0.323 (16/17) | 0.325 (16/17) | 0.032 (16/17) | 0.032 (16/17) | 40.22 (11/21) |
| 2 | 0.398 (18/18) | 0.400 (18/18) | 0.049 (18/18) | 0.048 (18/18) | 144 (16/21) |
| 3 | 0.424 (18/18) | 0.419 (18/18) | 0.053 (18/18) | 0.053 (18/18) | 194 (16/21) |

### term limit

| term limit | pymars fit_mars | pymars EarthRegressor | earth defaults | earth fast.k=0 | legacy 1.0.4 |
|---|---|---|---|---|---|
| 11 | 0.343 (10/11) | 0.364 (10/11) | 0.033 (10/11) | 0.034 (10/11) | 43.10 (11/11) |
| 21 | 0.398 (18/18) | 0.400 (18/18) | 0.049 (18/18) | 0.048 (18/18) | 144 (16/21) |
| 41 | 0.468 (21/21) | 0.463 (21/21) | 0.063 (21/21) | 0.062 (21/21) | 1,433 (31/41) |
| 81 | 0.469 (21/21) | 0.485 (21/21) | 0.063 (21/21) | 0.062 (21/21) | >30 min |

### weights

| weights | pymars fit_mars | pymars EarthRegressor | earth defaults | earth fast.k=0 | legacy 1.0.4 |
|---|---|---|---|---|---|
| off | 0.398 (18/18) | 0.400 (18/18) | 0.049 (18/18) | 0.048 (18/18) | 144 (16/21) |
| on | 0.420 (18/18) | 0.436 (18/18) | 16.31 (15/17) | 16.21 (15/17) | not run |

## Log-log slopes

The slope of log time on log of the factor, by least squares over the finished points (the number of points in brackets). Early stops change the amount of work, so read the slopes with the model sizes in the tables above. The slope for the degree has three points and the weights have two, so the tables give those.

| System | cases n | covariates p | term limit |
|---|---|---|---|
| pymars fit_mars | 0.44 (7) | 0.43 (5) | 0.16 (4) |
| pymars EarthRegressor | 0.44 (7) | 0.44 (5) | 0.15 (4) |
| earth defaults | 1.01 (7) | 0.91 (5) | 0.33 (4) |
| earth fast.k=0 | 1.00 (7) | 0.91 (5) | 0.31 (4) |
| legacy 1.0.4 | 1.50 (4) | 0.98 (5) | 2.67 (3) |

## Peak memory

Peak resident set size of the whole process in MiB (median of the runs; for R it includes R itself, about 100 MiB), and for the Python systems the peak Python allocation under `tracemalloc` in MiB, from one extra run for fits under a minute.

| Setting | pymars fit_mars | pymars EarthRegressor | earth defaults | earth fast.k=0 | legacy 1.0.4 |
|---|---|---|---|---|---|
| cases n 250 | 274 / 22 | 271 / 22 | 80 | 80 | 196 / 15 |
| cases n 500 | 271 / 22 | 279 / 22 | 80 | 76 | 222 |
| cases n 1,000 | 266 / 23 | 258 / 23 | 82 | 82 | 275 |
| cases n 2,000 | 276 / 23 | 272 / 23 | 87 | 85 | 348 |
| cases n 5,000 | 285 / 26 | 280 / 26 | 106 | 106 | - |
| cases n 10,000 | 314 / 38 | 311 / 38 | 146 | 134 | - |
| cases n 100,000 | 761 / 249 | 731 / 249 | 411 | 409 | - |
| covariates p 5 | 275 / 22 | 275 / 22 | 83 | 82 | 225 |
| covariates p 10 | 266 / 23 | 258 / 23 | 82 | 82 | 275 |
| covariates p 20 | 274 / 23 | 275 / 23 | 82 | 84 | 343 |
| covariates p 50 | 269 / 24 | 276 / 24 | 98 | 92 | 637 |
| covariates p 100 | 285 / 27 | 279 / 27 | 112 | 110 | 1,112 |
| degree 1 | 265 / 23 | 280 / 23 | 87 | 80 | 194 / 11 |
| degree 2 | 266 / 23 | 258 / 23 | 82 | 82 | 275 |
| degree 3 | 274 / 23 | 269 / 23 | 83 | 78 | 293 |
| term limit 11 | 275 / 23 | 280 / 23 | 78 | 84 | 243 / 59 |
| term limit 21 | 266 / 23 | 258 / 23 | 82 | 82 | 275 |
| term limit 41 | 271 / 23 | 279 / 23 | 84 | 83 | 280 |
| term limit 81 | 266 / 23 | 272 / 23 | 85 | 85 | - |
| weights off | 266 / 23 | 258 / 23 | 82 | 82 | 275 |
| weights on | 268 / 23 | 281 / 23 | 86 | 90 | - |

## Cells without a time

- legacy n=1000 p=10 deg=2 nk=81: timeout (timeout)
