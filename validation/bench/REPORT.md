# Benchmark report

Written by `validation/bench/report.py` from the results of `validation/bench/run.py`. Each point is the median of 3 runs, each in a fresh process with one thread; the timeout is 30 min per fit. Times are wall seconds of the fit alone. In the tables a cell reads `time (selected terms/forward terms)`. The data are Friedman #1 with fixed seeds, the same for every system; the baseline is 1,000 cases, 10 covariates, degree 2 and a limit of 21 terms. `not run` means the setting is not in the design for that system (the legacy code stops at 2,000 cases and takes no weights) or has not finished.

Run on Apple M1 Pro (10 cores), Python 3.12.13, numpy 2.5.3, R version 4.4.3 (2025-02-28), earth 5.3.4, 4 cells at a time, 1-minute load average 3.8 at the start, pymars commit `ae9d199afd`. Several cells ran at once on a shared machine, so times carry some noise.

## Ratio to earth at 10,000 cases

10,000 cases, 10 covariates, degree 2, 21 terms (the target of the plan: within 10 times earth's time).

| System | Time (s) | Ratio to earth defaults | Ratio to earth fast.k=0 |
|---|---|---|---|
| pymars fit_mars | 3.89 | 7.0 | 6.1 |
| pymars EarthRegressor | 4.32 | 7.7 | 6.8 |
| earth defaults | 0.559 | 1.0 | 0.9 |
| earth fast.k=0 | 0.637 | 1.1 | 1.0 |
| legacy 1.0.4 | not run | | |

pymars `fit_mars` is 7.0 times earth at its defaults.

## Times by factor

### cases n

| cases n | pymars fit_mars | pymars EarthRegressor | earth defaults | earth fast.k=0 | legacy 1.0.4 |
|---|---|---|---|---|---|
| 250 | 0.265 (16/19) | 0.274 (16/19) | 0.020 (15/18) | 0.025 (15/18) | not run |
| 500 | 0.334 (18/19) | 0.349 (18/19) | 0.029 (18/18) | 0.037 (18/18) | not run |
| 1,000 | 0.470 (18/18) | 0.482 (18/18) | 0.049 (18/18) | 0.049 (18/18) | 144 (16/21) |
| 2,000 | 0.828 (16/18) | 0.804 (16/18) | 0.095 (16/18) | 0.104 (16/18) | not run |
| 5,000 | 1.82 (17/19) | 1.85 (17/19) | 0.247 (18/18) | 0.298 (18/18) | not run |
| 10,000 | 3.89 (17/19) | 4.32 (17/19) | 0.559 (17/19) | 0.637 (17/19) | not run |
| 100,000 | 54.44 (16/19) | 57.25 (16/19) | 10.97 (16/19) | 12.79 (16/19) | not run |

### covariates p

| covariates p | pymars fit_mars | pymars EarthRegressor | earth defaults | earth fast.k=0 | legacy 1.0.4 |
|---|---|---|---|---|---|
| 5 | 0.208 (16/19) | 0.241 (16/19) | 0.036 (16/18) | 0.030 (16/18) | not run |
| 10 | 0.470 (18/18) | 0.482 (18/18) | 0.049 (18/18) | 0.049 (18/18) | 144 (16/21) |
| 20 | 1.00 (17/19) | 1.17 (17/19) | 0.101 (17/19) | 0.092 (17/19) | not run |
| 50 | 2.80 (18/19) | 2.54 (18/19) | 0.259 (18/19) | 0.219 (18/19) | not run |
| 100 | 5.12 (16/19) | 5.64 (16/19) | 0.500 (16/19) | 0.503 (16/19) | not run |

### degree

| degree | pymars fit_mars | pymars EarthRegressor | earth defaults | earth fast.k=0 | legacy 1.0.4 |
|---|---|---|---|---|---|
| 1 | 0.055 (16/17) | 0.064 (16/17) | 0.040 (16/17) | 0.032 (16/17) | not run |
| 2 | 0.470 (18/18) | 0.482 (18/18) | 0.049 (18/18) | 0.049 (18/18) | 144 (16/21) |
| 3 | 0.546 (18/18) | 0.605 (18/18) | 0.065 (18/18) | 0.054 (18/18) | not run |

### term limit

| term limit | pymars fit_mars | pymars EarthRegressor | earth defaults | earth fast.k=0 | legacy 1.0.4 |
|---|---|---|---|---|---|
| 11 | 0.127 (10/11) | 0.154 (10/11) | 0.035 (10/11) | 0.039 (10/11) | not run |
| 21 | 0.470 (18/18) | 0.482 (18/18) | 0.049 (18/18) | 0.049 (18/18) | 144 (16/21) |
| 41 | 0.765 (21/21) | 1.11 (21/21) | 0.083 (21/21) | 0.064 (21/21) | not run |
| 81 | 0.873 (21/21) | 1.03 (21/21) | 0.076 (21/21) | 0.068 (21/21) | not run |

### weights

| weights | pymars fit_mars | pymars EarthRegressor | earth defaults | earth fast.k=0 | legacy 1.0.4 |
|---|---|---|---|---|---|
| off | 0.470 (18/18) | 0.482 (18/18) | 0.049 (18/18) | 0.049 (18/18) | 144 (16/21) |
| on | 0.552 (18/18) | 0.735 (18/18) | 21.14 (15/17) | 19.95 (15/17) | not run |

## Log-log slopes

The slope of log time on log of the factor, by least squares over the finished points (the number of points in brackets). Early stops change the amount of work, so read the slopes with the model sizes in the tables above. The slope for the degree has three points and the weights have two, so the tables give those.

| System | cases n | covariates p | term limit |
|---|---|---|---|
| pymars fit_mars | 0.90 (7) | 1.08 (5) | 0.94 (4) |
| pymars EarthRegressor | 0.91 (7) | 1.05 (5) | 0.98 (4) |
| earth defaults | 1.06 (7) | 0.91 (5) | 0.43 (4) |
| earth fast.k=0 | 1.06 (7) | 0.94 (5) | 0.29 (4) |
| legacy 1.0.4 | - | - | - |

## Peak memory

Peak resident set size of the whole process in MiB (median of the runs; for R it includes R itself, about 100 MiB), and for the Python systems the peak Python allocation under `tracemalloc` in MiB, from one extra run for fits under a minute.

| Setting | pymars fit_mars | pymars EarthRegressor | earth defaults | earth fast.k=0 | legacy 1.0.4 |
|---|---|---|---|---|---|
| cases n 250 | 183 / 1 | 195 / 1 | 78 | 75 | - |
| cases n 500 | 201 / 1 | 191 / 1 | 79 | 81 | - |
| cases n 1,000 | 193 / 2 | 192 / 2 | 81 | 81 | 275 |
| cases n 2,000 | 209 / 3 | 211 / 3 | 90 | 91 | - |
| cases n 5,000 | 234 / 8 | 226 / 8 | 103 | 109 | - |
| cases n 10,000 | 296 / 16 | 308 / 16 | 136 | 136 | - |
| cases n 100,000 | 1,317 / 151 | 1,195 / 151 | 418 | 427 | - |
| covariates p 5 | 185 / 2 | 199 / 2 | 80 | 82 | - |
| covariates p 10 | 193 / 2 | 192 / 2 | 81 | 81 | 275 |
| covariates p 20 | 194 / 2 | 188 / 2 | 86 | 87 | - |
| covariates p 50 | 207 / 3 | 196 / 3 | 94 | 93 | - |
| covariates p 100 | 204 / 4 | 204 / 4 | 111 | 112 | - |
| degree 1 | 195 / 1 | 207 / 1 | 84 | 82 | - |
| degree 2 | 193 / 2 | 192 / 2 | 81 | 81 | 275 |
| degree 3 | 189 / 2 | 199 / 2 | 85 | 83 | - |
| term limit 11 | 186 / 1 | 199 / 1 | 81 | 83 | - |
| term limit 21 | 193 / 2 | 192 / 2 | 81 | 81 | 275 |
| term limit 41 | 199 / 2 | 205 / 2 | 81 | 85 | - |
| term limit 81 | 196 / 2 | 200 / 2 | 85 | 85 | - |
| weights off | 193 / 2 | 192 / 2 | 81 | 81 | 275 |
| weights on | 199 / 2 | 200 / 2 | 87 | 90 | - |
