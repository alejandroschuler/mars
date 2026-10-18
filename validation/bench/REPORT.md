# Benchmark report

Written by `validation/bench/report.py` from the results of `validation/bench/run.py`. Each point is the median of 3 runs, each in a fresh process with one thread; the timeout is 30 min per fit. Times are wall seconds of the fit alone. In the tables a cell reads `time (selected terms/forward terms)`. The data are Friedman #1 with fixed seeds, the same for every system; the baseline is 1,000 cases, 10 covariates, degree 2 and a limit of 21 terms. `not run` means the setting is not in the design for that system (the legacy code stops at 2,000 cases and takes no weights) or has not finished. Each run is a fresh process, so one-time start costs of pymars fall inside the timed fit; the numbers are the cost of a single fit, not of a repeated one.

Run on Apple M1 Pro (10 cores), Python 3.12.13, numpy 2.5.3, R version 4.4.3 (2025-02-28), earth 5.3.4, 4 cells at a time, 1-minute load average 2.9 at the start, pymars commit `3e1aefc7a7`. Several cells ran at once on a shared machine, so times carry some noise.

**Notes added by hand (T22b; `report.py` does not write them).** The legacy times are those of the T19 run at `ae9d199` (the same worker and data, so the same cell keys); the legacy code does not change, and this run did not repeat its 2 hours. The timer starts after the imports and the data loading (`worker.py`), so import time is not in any number here. The fresh-process times of pymars do include the first-call costs of the fit: numba loads the compiled knot scan from its disk cache on the first call, and the first numpy and scipy calls warm up. Together these add about 0.3 s per process. A program that fits many models in one process pays them once. `validation/sims/report/warm_times.py` times `EarthRegressor` after a first fit in the same process (median of 3, `validation/sims/report/warm_times.csv`):

| Cases | pymars, fresh (s) | pymars, warm (s) | earth defaults (s) | Factor, fresh | Factor, warm |
|---|---|---|---|---|---|
| 250 | 0.382 | 0.076 | 0.020 | 19 | 3.8 |
| 1,000 | 0.424 | 0.120 | 0.049 | 8.7 | 2.4 |
| 10,000 | 1.12 | 0.630 | 0.503 | 2.2 | 1.3 |
| 100,000 | 10.8 | 7.76 | 8.59 | 1.3 | 0.90 |

The fixed cost also flattens the log-log slopes below: from 10,000 to 100,000 cases the time of `EarthRegressor` grows by a factor of 9.6 (slope 0.98), close to linear in n.

## Ratio to earth at 10,000 cases

10,000 cases, 10 covariates, degree 2, 21 terms (the target of the plan: within 10 times earth's time).

| System | Time (s) | Ratio to earth defaults | Ratio to earth fast.k=0 |
|---|---|---|---|
| pymars fit_mars | 0.982 | 2.0 | 1.8 |
| pymars EarthRegressor | 1.12 | 2.2 | 2.1 |
| earth defaults | 0.503 | 1.0 | 0.9 |
| earth fast.k=0 | 0.539 | 1.1 | 1.0 |
| legacy 1.0.4 | not run | | |

pymars `fit_mars` is 2.0 times earth at its defaults.

## Times by factor

### cases n

| cases n | pymars fit_mars | pymars EarthRegressor | earth defaults | earth fast.k=0 | legacy 1.0.4 |
|---|---|---|---|---|---|
| 250 | 0.397 (16/19) | 0.382 (16/19) | 0.020 (15/18) | 0.019 (15/18) | 25.79 (17/21) |
| 500 | 0.408 (18/19) | 0.396 (18/19) | 0.029 (18/18) | 0.028 (18/18) | 65.12 (15/21) |
| 1,000 | 0.425 (18/18) | 0.424 (18/18) | 0.049 (18/18) | 0.049 (18/18) | 144 (16/21) |
| 2,000 | 0.490 (16/18) | 0.477 (16/18) | 0.095 (16/18) | 0.093 (16/18) | 637 (16/21) |
| 5,000 | 0.687 (17/19) | 0.722 (17/19) | 0.242 (18/18) | 0.237 (18/18) | not run |
| 10,000 | 0.982 (17/19) | 1.12 (17/19) | 0.503 (17/19) | 0.539 (17/19) | not run |
| 100,000 | 10.40 (16/19) | 10.76 (16/19) | 8.59 (16/19) | 7.36 (16/19) | not run |

### covariates p

| covariates p | pymars fit_mars | pymars EarthRegressor | earth defaults | earth fast.k=0 | legacy 1.0.4 |
|---|---|---|---|---|---|
| 5 | 0.376 (16/19) | 0.374 (16/19) | 0.030 (16/18) | 0.029 (16/18) | 91.26 (16/21) |
| 10 | 0.425 (18/18) | 0.424 (18/18) | 0.049 (18/18) | 0.049 (18/18) | 144 (16/21) |
| 20 | 0.566 (17/19) | 0.540 (17/19) | 0.094 (17/19) | 0.090 (17/19) | 377 (18/21) |
| 50 | 0.916 (18/19) | 0.922 (18/19) | 0.216 (18/19) | 0.211 (18/19) | 794 (16/21) |
| 100 | 1.46 (16/19) | 1.50 (16/19) | 0.433 (16/19) | 0.421 (16/19) | 1,594 (16/21) |

### degree

| degree | pymars fit_mars | pymars EarthRegressor | earth defaults | earth fast.k=0 | legacy 1.0.4 |
|---|---|---|---|---|---|
| 1 | 0.328 (16/17) | 0.318 (16/17) | 0.034 (16/17) | 0.032 (16/17) | 40.22 (11/21) |
| 2 | 0.425 (18/18) | 0.424 (18/18) | 0.049 (18/18) | 0.049 (18/18) | 144 (16/21) |
| 3 | 0.455 (18/18) | 0.484 (18/18) | 0.052 (18/18) | 0.052 (18/18) | 194 (16/21) |

### term limit

| term limit | pymars fit_mars | pymars EarthRegressor | earth defaults | earth fast.k=0 | legacy 1.0.4 |
|---|---|---|---|---|---|
| 11 | 0.353 (10/11) | 0.402 (10/11) | 0.034 (10/11) | 0.033 (10/11) | 43.10 (11/11) |
| 21 | 0.425 (18/18) | 0.424 (18/18) | 0.049 (18/18) | 0.049 (18/18) | 144 (16/21) |
| 41 | 0.512 (21/21) | 0.496 (21/21) | 0.062 (21/21) | 0.062 (21/21) | 1,433 (31/41) |
| 81 | 0.500 (21/21) | 0.589 (21/21) | 0.060 (21/21) | 0.064 (21/21) | >30 min |

### weights

| weights | pymars fit_mars | pymars EarthRegressor | earth defaults | earth fast.k=0 | legacy 1.0.4 |
|---|---|---|---|---|---|
| off | 0.425 (18/18) | 0.424 (18/18) | 0.049 (18/18) | 0.049 (18/18) | 144 (16/21) |
| on | 0.453 (18/18) | 0.451 (18/18) | 16.42 (15/17) | 16.17 (15/17) | not run |

## Log-log slopes

The slope of log time on log of the factor, by least squares over the finished points (the number of points in brackets). Early stops change the amount of work, so read the slopes with the model sizes in the tables above. The slope for the degree has three points and the weights have two, so the tables give those.

| System | cases n | covariates p | term limit |
|---|---|---|---|
| pymars fit_mars | 0.53 (7) | 0.46 (5) | 0.19 (4) |
| pymars EarthRegressor | 0.55 (7) | 0.47 (5) | 0.20 (4) |
| earth defaults | 1.02 (7) | 0.90 (5) | 0.29 (4) |
| earth fast.k=0 | 1.01 (7) | 0.90 (5) | 0.33 (4) |
| legacy 1.0.4 | 1.50 (4) | 0.98 (5) | 2.67 (3) |

## Peak memory

Peak resident set size of the whole process in MiB (median of the runs; for R it includes R itself, about 100 MiB), and for the Python systems the peak Python allocation under `tracemalloc` in MiB, from one extra run for fits under a minute.

| Setting | pymars fit_mars | pymars EarthRegressor | earth defaults | earth fast.k=0 | legacy 1.0.4 |
|---|---|---|---|---|---|
| cases n 250 | 273 / 22 | 270 / 22 | 81 | 77 | 196 / 15 |
| cases n 500 | 266 / 22 | 271 / 22 | 80 | 81 | 222 |
| cases n 1,000 | 262 / 23 | 257 / 23 | 81 | 84 | 275 |
| cases n 2,000 | 283 / 23 | 273 / 23 | 88 | 85 | 348 |
| cases n 5,000 | 286 / 24 | 290 / 24 | 106 | 101 | - |
| cases n 10,000 | 320 / 34 | 315 / 34 | 132 | 142 | - |
| cases n 100,000 | 894 / 209 | 860 / 209 | 416 | 398 | - |
| covariates p 5 | 273 / 22 | 276 / 22 | 79 | 78 | 225 |
| covariates p 10 | 262 / 23 | 257 / 23 | 81 | 84 | 275 |
| covariates p 20 | 275 / 23 | 274 / 23 | 86 | 86 | 343 |
| covariates p 50 | 284 / 24 | 277 / 24 | 94 | 93 | 637 |
| covariates p 100 | 278 / 25 | 280 / 25 | 113 | 112 | 1,112 |
| degree 1 | 273 / 23 | 272 / 23 | 83 | 80 | 194 / 11 |
| degree 2 | 262 / 23 | 257 / 23 | 81 | 84 | 275 |
| degree 3 | 263 / 23 | 265 / 23 | 81 | 79 | 293 |
| term limit 11 | 270 / 23 | 286 / 23 | 81 | 79 | 243 / 59 |
| term limit 21 | 262 / 23 | 257 / 23 | 81 | 84 | 275 |
| term limit 41 | 277 / 23 | 274 / 23 | 83 | 80 | 280 |
| term limit 81 | 276 / 23 | 271 / 23 | 88 | 83 | - |
| weights off | 262 / 23 | 257 / 23 | 81 | 84 | 275 |
| weights on | 263 / 23 | 273 / 23 | 86 | 85 | - |

## Cells without a time

- legacy n=1000 p=10 deg=2 nk=81: timeout (timeout)
