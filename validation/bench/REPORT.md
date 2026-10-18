# Benchmark report

Written by `validation/bench/report.py` from the results of `validation/bench/run.py`. Each point is the median of 3 runs, each in a fresh process with one thread; the timeout is 30 min per fit. Times are wall seconds of the fit alone. In the tables a cell reads `time (selected terms/forward terms)`. The data are Friedman #1 with fixed seeds, the same for every system; the baseline is 1,000 cases, 10 covariates, degree 2 and a limit of 21 terms. `not run` means the setting is not in the design for that system (the legacy code stops at 2,000 cases and takes no weights) or has not finished. Each run is a fresh process, so one-time start costs of pymars fall inside the timed fit; the numbers are the cost of a single fit, not of a repeated one.

Run on Apple M1 Pro (10 cores), Python 3.12.13, numpy 2.5.3, R version 4.4.3 (2025-02-28), earth 5.3.4, 4 cells at a time, 1-minute load average 1.5 at the start, pymars commit `95212fa421`. Several cells ran at once on a shared machine, so times carry some noise.

**Notes added by hand (T22b; `report.py` does not write them).** The legacy times are those of the T19 run at `ae9d199` (the same worker and data, so the same cell keys); the legacy code does not change, and this run did not repeat its 2 hours. The timer starts after the imports and the data loading (`worker.py`), so import time is in no number here. The fresh-process times of pymars do include the first-call costs of the fit: numba loads the compiled kernels from its disk cache on the first call, and the first numpy and scipy calls warm up. Together these add about 0.3 s per process. A program that fits many models in one process pays them once.

Every pymars time in the tables below is the numba path. The plan's target ("Fast path": within 10 times earth's time at 10,000 x 10 x degree 2, in pure numpy) refers to the numpy code, which Intel Macs run. `validation/sims/report/warm_times.py` times `EarthRegressor` in one process per size: the first fit ("fresh") and the median of 3 later fits ("warm"), for the numba path (`warm_times.csv`) and, with `pymars._kernels.ENABLED = False`, for the numpy path (`warm_times_numpy.csv`), both on this commit's `pymars/`. It runs one size at a time, while this benchmark ran 4 cells at a time, so its times are about 15 percent lower at 10,000 cases (0.69 s against 0.81 s for the first numba fit).

| Cases | earth defaults (s) | numba, fresh (s) | numba, warm (s) | numpy, fresh (s) | numpy, warm (s) | Factor: numba fresh / warm | Factor: numpy fresh / warm |
|---|---|---|---|---|---|---|---|
| 250 | 0.018 | 0.373 | 0.063 | 0.150 | 0.148 | 21 / 3.5 | 8.3 / 8.2 |
| 1,000 | 0.049 | 0.388 | 0.094 | 0.244 | 0.238 | 7.9 / 1.9 | 5.0 / 4.9 |
| 10,000 | 0.464 | 0.809 | 0.425 | 1.22 | 1.26 | 1.7 / 0.92 | 2.6 / 2.7 |
| 100,000 | 7.57 | 5.77 | 5.09 | 15.3 | 15.3 | 0.76 / 0.67 | 2.0 / 2.0 |

Earth and numba fresh are from this benchmark; the other columns from `warm_times.py`. At 10,000 cases the numpy path is 2.6 times earth fresh and 2.7 times warm, inside the plan's target of 10. The fixed cost also flattens the log-log slopes below: from 10,000 to 100,000 cases the numba time of `EarthRegressor` grows by a factor of 7.1 (slope 0.85).

## Ratio to earth at 10,000 cases

10,000 cases, 10 covariates, degree 2, 21 terms (the target of the plan: within 10 times earth's time).

| System | Time (s) | Ratio to earth defaults | Ratio to earth fast.k=0 |
|---|---|---|---|
| pymars fit_mars | 0.767 | 1.7 | 1.6 |
| pymars EarthRegressor | 0.809 | 1.7 | 1.7 |
| earth defaults | 0.464 | 1.0 | 1.0 |
| earth fast.k=0 | 0.474 | 1.0 | 1.0 |
| legacy 1.0.4 | not run | | |

pymars `fit_mars` is 1.7 times earth at its defaults.

## Times by factor

### cases n

| cases n | pymars fit_mars | pymars EarthRegressor | earth defaults | earth fast.k=0 | legacy 1.0.4 |
|---|---|---|---|---|---|
| 250 | 0.384 (16/19) | 0.373 (16/19) | 0.018 (15/18) | 0.019 (15/18) | 25.79 (17/21) |
| 500 | 0.395 (18/19) | 0.381 (18/19) | 0.027 (18/18) | 0.027 (18/18) | 65.12 (15/21) |
| 1,000 | 0.389 (18/18) | 0.388 (18/18) | 0.049 (18/18) | 0.049 (18/18) | 144 (16/21) |
| 2,000 | 0.448 (16/18) | 0.446 (16/18) | 0.090 (16/18) | 0.090 (16/18) | 637 (16/21) |
| 5,000 | 0.586 (17/19) | 0.623 (17/19) | 0.230 (18/18) | 0.231 (18/18) | not run |
| 10,000 | 0.767 (17/19) | 0.809 (17/19) | 0.464 (17/19) | 0.474 (17/19) | not run |
| 100,000 | 5.80 (16/19) | 5.77 (16/19) | 7.57 (16/19) | 6.29 (16/19) | not run |

### covariates p

| covariates p | pymars fit_mars | pymars EarthRegressor | earth defaults | earth fast.k=0 | legacy 1.0.4 |
|---|---|---|---|---|---|
| 5 | 0.364 (16/19) | 0.358 (16/19) | 0.028 (16/18) | 0.028 (16/18) | 91.26 (16/21) |
| 10 | 0.389 (18/18) | 0.388 (18/18) | 0.049 (18/18) | 0.049 (18/18) | 144 (16/21) |
| 20 | 0.525 (17/19) | 0.512 (17/19) | 0.091 (17/19) | 0.089 (17/19) | 377 (18/21) |
| 50 | 0.812 (18/19) | 0.813 (18/19) | 0.213 (18/19) | 0.214 (18/19) | 794 (16/21) |
| 100 | 1.31 (16/19) | 1.37 (16/19) | 0.427 (16/19) | 0.423 (16/19) | 1,594 (16/21) |

### degree

| degree | pymars fit_mars | pymars EarthRegressor | earth defaults | earth fast.k=0 | legacy 1.0.4 |
|---|---|---|---|---|---|
| 1 | 0.325 (16/17) | 0.328 (16/17) | 0.031 (16/17) | 0.032 (16/17) | 40.22 (11/21) |
| 2 | 0.389 (18/18) | 0.388 (18/18) | 0.049 (18/18) | 0.049 (18/18) | 144 (16/21) |
| 3 | 0.426 (18/18) | 0.421 (18/18) | 0.052 (18/18) | 0.052 (18/18) | 194 (16/21) |

### term limit

| term limit | pymars fit_mars | pymars EarthRegressor | earth defaults | earth fast.k=0 | legacy 1.0.4 |
|---|---|---|---|---|---|
| 11 | 0.352 (10/11) | 0.349 (10/11) | 0.033 (10/11) | 0.034 (10/11) | 43.10 (11/11) |
| 21 | 0.389 (18/18) | 0.388 (18/18) | 0.049 (18/18) | 0.049 (18/18) | 144 (16/21) |
| 41 | 0.461 (21/21) | 0.464 (21/21) | 0.063 (21/21) | 0.062 (21/21) | 1,433 (31/41) |
| 81 | 0.470 (21/21) | 0.479 (21/21) | 0.063 (21/21) | 0.062 (21/21) | >30 min |

### weights

| weights | pymars fit_mars | pymars EarthRegressor | earth defaults | earth fast.k=0 | legacy 1.0.4 |
|---|---|---|---|---|---|
| off | 0.389 (18/18) | 0.388 (18/18) | 0.049 (18/18) | 0.049 (18/18) | 144 (16/21) |
| on | 0.421 (18/18) | 0.448 (18/18) | 16.21 (15/17) | 15.72 (15/17) | not run |

## Log-log slopes

The slope of log time on log of the factor, by least squares over the finished points (the number of points in brackets). Early stops change the amount of work, so read the slopes with the model sizes in the tables above. The slope for the degree has three points and the weights have two, so the tables give those.

| System | cases n | covariates p | term limit |
|---|---|---|---|
| pymars fit_mars | 0.44 (7) | 0.44 (5) | 0.16 (4) |
| pymars EarthRegressor | 0.44 (7) | 0.45 (5) | 0.17 (4) |
| earth defaults | 1.02 (7) | 0.91 (5) | 0.33 (4) |
| earth fast.k=0 | 0.98 (7) | 0.91 (5) | 0.30 (4) |
| legacy 1.0.4 | 1.50 (4) | 0.98 (5) | 2.67 (3) |

## Peak memory

Peak resident set size of the whole process in MiB (median of the runs; for R it includes R itself, about 100 MiB), and for the Python systems the peak Python allocation under `tracemalloc` in MiB, from one extra run for fits under a minute.

| Setting | pymars fit_mars | pymars EarthRegressor | earth defaults | earth fast.k=0 | legacy 1.0.4 |
|---|---|---|---|---|---|
| cases n 250 | 269 / 22 | 258 / 22 | 73 | 74 | 196 / 15 |
| cases n 500 | 260 / 22 | 257 / 22 | 77 | 78 | 222 |
| cases n 1,000 | 254 / 23 | 260 / 23 | 80 | 81 | 275 |
| cases n 2,000 | 263 / 23 | 259 / 23 | 86 | 88 | 348 |
| cases n 5,000 | 274 / 26 | 273 / 26 | 105 | 113 | - |
| cases n 10,000 | 294 / 38 | 300 / 38 | 134 | 136 | - |
| cases n 100,000 | 762 / 249 | 728 / 249 | 391 | 401 | - |
| covariates p 5 | 266 / 22 | 267 / 22 | 79 | 77 | 225 |
| covariates p 10 | 254 / 23 | 260 / 23 | 80 | 81 | 275 |
| covariates p 20 | 266 / 23 | 261 / 23 | 78 | 86 | 343 |
| covariates p 50 | 268 / 24 | 263 / 24 | 95 | 93 | 637 |
| covariates p 100 | 273 / 27 | 272 / 27 | 107 | 111 | 1,112 |
| degree 1 | 272 / 23 | 260 / 23 | 80 | 79 | 194 / 11 |
| degree 2 | 254 / 23 | 260 / 23 | 80 | 81 | 275 |
| degree 3 | 274 / 23 | 264 / 23 | 80 | 78 | 293 |
| term limit 11 | 269 / 23 | 270 / 23 | 83 | 80 | 243 / 59 |
| term limit 21 | 254 / 23 | 260 / 23 | 80 | 81 | 275 |
| term limit 41 | 270 / 23 | 267 / 23 | 80 | 82 | 280 |
| term limit 81 | 264 / 23 | 266 / 23 | 82 | 84 | - |
| weights off | 254 / 23 | 260 / 23 | 80 | 81 | 275 |
| weights on | 268 / 23 | 261 / 23 | 88 | 86 | - |

## Cells without a time

- legacy n=1000 p=10 deg=2 nk=81: timeout (timeout)
