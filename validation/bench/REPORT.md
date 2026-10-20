# Benchmark report

Written by `validation/bench/report.py` from the results of `validation/bench/run.py`. Each point is the median of 3 runs, each in a fresh process with one thread; the timeout is 30 min per fit. Times are wall seconds of the fit alone. In the tables a cell reads `time (selected terms/forward terms)`. The data are Friedman #1 with fixed seeds, the same for every system; the baseline is 1,000 cases, 10 covariates, degree 2 and a limit of 21 terms. `not run` means the setting is not in the design for that system (the legacy code stops at 2,000 cases and takes no weights) or has not finished. Each run is a fresh process, so one-time start costs of pymars fall inside the timed fit; the numbers are the cost of a single fit, not of a repeated one.

Run on Apple M1 Pro (10 cores), Python 3.12.13, numpy 2.5.3, R version 4.4.3 (2025-02-28), earth 5.3.4, 4 cells at a time, 1-minute load average 3.6 at the start, pymars commit `37ffcb2516`. Several cells ran at once on a shared machine, so times carry some noise.

**Notes added by hand (W184, issue #184; `report.py` does not write them).** The timings come from `main` at `37ffcb2` (the merge of #186, which holds the pair rule of #185), run from a runner worktree at that commit with `run.py --jobs 4`. The grid took 4 minutes. The legacy times are those of the T19 run at `ae9d199`, copied into the run folder as in the previous version of this report (the legacy code and `worker.py` do not change, so the cell keys are the same); this run did not repeat its 2 hours. The timer starts after the imports and the data loading (`worker.py`), so import time is in no number here. The fresh-process times of pymars do include the first-call costs of the fit: numba loads the compiled kernels from its disk cache on the first call, and the first numpy and scipy calls warm up. Together these add about 0.3 s per process. A program that fits many models in one process pays them once.

Against the previous version of this report (the run at `95212fa`, before the pair rule of #185), the median ratio of new time to old time is 1.03 over the 17 cells of `EarthRegressor` (range 0.98 to 1.15) and 1.01 over the 16 unweighted cells of earth at its defaults (0.99 to 1.06). The pair rule did not change the speed: at 10,000 cases `EarthRegressor` took 0.81 s in a fresh process before and 0.82 s after. The model sizes did change. `EarthRegressor` now selects and keeps as many terms as earth in 16 of the 17 cells that both run; before, it differed from earth at 250, 500 and 5,000 cases. The weighted cell differs as before; the two programs treat weights differently (`docs/weights.md`).

Every pymars time in the tables below is the numba path. The plan's target ("Fast path": within 10 times earth's time at 10,000 x 10 x degree 2, in pure numpy) refers to the numpy code, which Intel Macs run. `validation/sims/report/warm_times.py` times `EarthRegressor` in one process per size: the first fit ("fresh") and the median of 3 later fits ("warm"), for the numba path (`warm_times.csv`) and, with `pymars._kernels.ENABLED = False`, for the numpy path (`warm_times_numpy.csv`), both on `37ffcb2`. It runs one size at a time, while this benchmark ran 4 cells at a time. Other jobs shared the machine: the 1-minute load average was 3.6 at the start of the benchmark and about 5.6 while `warm_times.py` ran. The two runs therefore do not have the same conditions. At 10,000 cases the first numba fit of the script took 0.78 s against 0.82 s in the benchmark.

Run again on the same day, `warm_times.py` gave the same times within 2 percent (numba warm 0.48 s and numpy 1.5 s at 10,000 cases). The same fits at `d75a4c5` (the commit of the previous warm and numpy columns), `95212fa` and `37ffcb2` also took the same time on that day. The previous columns (0.425 s and 1.22 s at 10,000 cases) were not reproduced at their own commit, so the difference from them comes from the state of the machine on the two days, not from the code. Read the warm and numpy columns with an error between days of about 20 percent.

| Cases | earth defaults (s) | numba, fresh (s) | numba, warm (s) | numpy, fresh (s) | numpy, warm (s) | Factor: numba fresh / warm | Factor: numpy fresh / warm |
|---|---|---|---|---|---|---|---|
| 250 | 0.019 | 0.372 | 0.068 | 0.167 | 0.160 | 20 / 3.6 | 8.8 / 8.4 |
| 1,000 | 0.051 | 0.448 | 0.104 | 0.287 | 0.280 | 8.8 / 2.0 | 5.6 / 5.5 |
| 10,000 | 0.468 | 0.819 | 0.481 | 1.52 | 1.57 | 1.8 / 1.0 | 3.3 / 3.4 |
| 100,000 | 7.98 | 6.04 | 6.22 | 18.6 | 19.3 | 0.76 / 0.78 | 2.3 / 2.4 |

Earth and numba fresh are from this benchmark; the other columns from `warm_times.py`. At 10,000 cases the numpy path is 3.3 times earth fresh and 3.4 times warm, inside the plan's target of 10. At 250 cases earth's time is 0.019 s, read to 1 ms, so the factors there carry an error of about 5 percent. At 100,000 cases the warm time is a little above the fresh one. The first-fit cost is 0.3 s of 6 s, and the two columns come from runs under different loads. The fixed cost also flattens the log-log slopes below: from 10,000 to 100,000 cases the numba time of `EarthRegressor` grows by a factor of 7.4 (slope 0.87).

## Ratio to earth at 10,000 cases

10,000 cases, 10 covariates, degree 2, 21 terms (the target of the plan: within 10 times earth's time).

| System | Time (s) | Ratio to earth defaults | Ratio to earth fast.k=0 |
|---|---|---|---|
| pymars fit_mars | 0.773 | 1.7 | 1.6 |
| pymars EarthRegressor | 0.819 | 1.8 | 1.7 |
| earth defaults | 0.468 | 1.0 | 1.0 |
| earth fast.k=0 | 0.472 | 1.0 | 1.0 |
| legacy 1.0.4 | not run | | |

pymars `fit_mars` is 1.7 times earth at its defaults.

## Times by factor

### cases n

| cases n | pymars fit_mars | pymars EarthRegressor | earth defaults | earth fast.k=0 | legacy 1.0.4 |
|---|---|---|---|---|---|
| 250 | 0.399 (15/18) | 0.372 (15/18) | 0.019 (15/18) | 0.022 (15/18) | 25.79 (17/21) |
| 500 | 0.407 (18/18) | 0.389 (18/18) | 0.027 (18/18) | 0.028 (18/18) | 65.12 (15/21) |
| 1,000 | 0.435 (18/18) | 0.448 (18/18) | 0.051 (18/18) | 0.051 (18/18) | 144 (16/21) |
| 2,000 | 0.451 (16/18) | 0.458 (16/18) | 0.093 (16/18) | 0.092 (16/18) | 637 (16/21) |
| 5,000 | 0.585 (18/18) | 0.670 (18/18) | 0.229 (18/18) | 0.235 (18/18) | not run |
| 10,000 | 0.773 (17/19) | 0.819 (17/19) | 0.468 (17/19) | 0.472 (17/19) | not run |
| 100,000 | 6.38 (16/19) | 6.04 (16/19) | 7.98 (16/19) | 7.39 (16/19) | not run |

### covariates p

| covariates p | pymars fit_mars | pymars EarthRegressor | earth defaults | earth fast.k=0 | legacy 1.0.4 |
|---|---|---|---|---|---|
| 5 | 0.378 (16/18) | 0.362 (16/18) | 0.028 (16/18) | 0.028 (16/18) | 91.26 (16/21) |
| 10 | 0.435 (18/18) | 0.448 (18/18) | 0.051 (18/18) | 0.051 (18/18) | 144 (16/21) |
| 20 | 0.569 (17/19) | 0.521 (17/19) | 0.092 (17/19) | 0.090 (17/19) | 377 (18/21) |
| 50 | 0.835 (18/19) | 0.855 (18/19) | 0.214 (18/19) | 0.215 (18/19) | 794 (16/21) |
| 100 | 1.31 (16/19) | 1.36 (16/19) | 0.424 (16/19) | 0.424 (16/19) | 1,594 (16/21) |

### degree

| degree | pymars fit_mars | pymars EarthRegressor | earth defaults | earth fast.k=0 | legacy 1.0.4 |
|---|---|---|---|---|---|
| 1 | 0.344 (16/17) | 0.335 (16/17) | 0.032 (16/17) | 0.032 (16/17) | 40.22 (11/21) |
| 2 | 0.435 (18/18) | 0.448 (18/18) | 0.051 (18/18) | 0.051 (18/18) | 144 (16/21) |
| 3 | 0.451 (18/18) | 0.460 (18/18) | 0.053 (18/18) | 0.053 (18/18) | 194 (16/21) |

### term limit

| term limit | pymars fit_mars | pymars EarthRegressor | earth defaults | earth fast.k=0 | legacy 1.0.4 |
|---|---|---|---|---|---|
| 11 | 0.364 (10/11) | 0.378 (10/11) | 0.035 (10/11) | 0.034 (10/11) | 43.10 (11/11) |
| 21 | 0.435 (18/18) | 0.448 (18/18) | 0.051 (18/18) | 0.051 (18/18) | 144 (16/21) |
| 41 | 0.503 (21/21) | 0.475 (21/21) | 0.063 (21/21) | 0.061 (21/21) | 1,433 (31/41) |
| 81 | 0.478 (21/21) | 0.499 (21/21) | 0.063 (21/21) | 0.062 (21/21) | >30 min |

### weights

| weights | pymars fit_mars | pymars EarthRegressor | earth defaults | earth fast.k=0 | legacy 1.0.4 |
|---|---|---|---|---|---|
| off | 0.435 (18/18) | 0.448 (18/18) | 0.051 (18/18) | 0.051 (18/18) | 144 (16/21) |
| on | 0.427 (18/18) | 0.440 (18/18) | 16.27 (15/17) | 16.41 (15/17) | not run |

## Log-log slopes

The slope of log time on log of the factor, by least squares over the finished points (the number of points in brackets). Early stops change the amount of work, so read the slopes with the model sizes in the tables above. The slope for the degree has three points and the weights have two, so the tables give those.

| System | cases n | covariates p | term limit |
|---|---|---|---|
| pymars fit_mars | 0.44 (7) | 0.41 (5) | 0.14 (4) |
| pymars EarthRegressor | 0.45 (7) | 0.44 (5) | 0.13 (4) |
| earth defaults | 1.02 (7) | 0.90 (5) | 0.30 (4) |
| earth fast.k=0 | 0.99 (7) | 0.91 (5) | 0.30 (4) |
| legacy 1.0.4 | 1.50 (4) | 0.98 (5) | 2.67 (3) |

## Peak memory

Peak resident set size of the whole process in MiB (median of the runs; for R it includes R itself, about 100 MiB), and for the Python systems the peak Python allocation under `tracemalloc` in MiB, from one extra run for fits under a minute.

| Setting | pymars fit_mars | pymars EarthRegressor | earth defaults | earth fast.k=0 | legacy 1.0.4 |
|---|---|---|---|---|---|
| cases n 250 | 267 / 22 | 276 / 22 | 78 | 76 | 196 / 15 |
| cases n 500 | 266 / 22 | 276 / 22 | 79 | 80 | 222 |
| cases n 1,000 | 268 / 22 | 268 / 22 | 80 | 79 | 275 |
| cases n 2,000 | 270 / 23 | 268 / 23 | 87 | 86 | 348 |
| cases n 5,000 | 286 / 26 | 285 / 26 | 107 | 106 | - |
| cases n 10,000 | 310 / 38 | 313 / 38 | 143 | 142 | - |
| cases n 100,000 | 722 / 249 | 688 / 249 | 414 | 404 | - |
| covariates p 5 | 268 / 22 | 274 / 22 | 81 | 81 | 225 |
| covariates p 10 | 268 / 22 | 268 / 22 | 80 | 79 | 275 |
| covariates p 20 | 272 / 23 | 276 / 23 | 82 | 87 | 343 |
| covariates p 50 | 271 / 24 | 274 / 24 | 95 | 93 | 637 |
| covariates p 100 | 280 / 27 | 282 / 27 | 112 | 110 | 1,112 |
| degree 1 | 272 / 22 | 268 / 22 | 82 | 80 | 194 / 11 |
| degree 2 | 268 / 22 | 268 / 22 | 80 | 79 | 275 |
| degree 3 | 271 / 22 | 271 / 22 | 79 | 80 | 293 |
| term limit 11 | 265 / 22 | 275 / 22 | 82 | 83 | 243 / 59 |
| term limit 21 | 268 / 22 | 268 / 22 | 80 | 79 | 275 |
| term limit 41 | 270 / 22 | 269 / 22 | 79 | 80 | 280 |
| term limit 81 | 270 / 22 | 265 / 22 | 83 | 83 | - |
| weights off | 268 / 22 | 268 / 22 | 80 | 79 | 275 |
| weights on | 273 / 22 | 263 / 22 | 84 | 84 | - |

## Cells without a time

- legacy n=1000 p=10 deg=2 nk=81: timeout (timeout)
