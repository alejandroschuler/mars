# Benchmark harness (T19)

Times pymars (`fit_mars` and `EarthRegressor`), R earth (defaults and `fast.k = 0`, called as a black box through `Rscript`) and the legacy code (the mars-earth 1.0.4 wheel in `.venv-legacy`, up to 2,000 cases) on the plan's grid ("Speed and scaling" in `VALIDATION_PLAN.md`): one factor at a time around 1,000 cases, 10 covariates, degree 2 and 21 terms.

| File | Purpose |
|---|---|
| `run.py` | The driver: the grid, `--resume`, `--jobs`, `--profile`, and `--smoke` for gate C |
| `worker.py` | One fit of a Python system in a fresh process (also run by the legacy venv) |
| `earth_worker.R` | One earth fit; the earth arguments come from `validation/harness/names_map.py` |
| `core.py` | The grid, the data, the cell keys, atomic writes and the log-log slope |
| `report.py` | Writes `REPORT.md` from a run folder |
| `baseline.json` | The stored smoke baseline from `main` |

## Full grid

```sh
. dev/env.sh
nohup caffeinate -i nice -n 15 uv run --frozen --group validation python \
    validation/bench/run.py --out validation/runs/bench --resume --jobs 4 \
    --pid-file validation/runs/bench/run.pid > validation/runs/bench/run.log 2>&1 &
```

Run it from a runner worktree at a fixed commit (`.worktrees/runner-t19`). A cell is one system at one setting. It holds the median of 3 runs (each in a fresh process, one thread): wall time of the fit alone, peak resident memory of the process, the peak Python allocation under `tracemalloc` (one extra run, for fits under a minute) and the model sizes (selected and forward terms). A run over 30 minutes ends the cell with status `timeout` and skips the larger settings of the same series. Results are `results/<key>.json`, written by a temporary file and a rename. The key holds the cell, the data seed, the repetition rule, and the pymars commit and source hash (or the earth and legacy tags). `manifest.json` lists the cells and their status. `--resume` skips cells with status `ok` or `timeout`; it runs errors and skips again. At most 4 cells run at once, and an R process counts as one.

`--profile` adds one cProfile run per Python cell (`profiles/<key>.prof`, outside the timed runs), for T20.

Data: Friedman #1, seeded by `(19, n, p)`, the same for every system. Weights are integer counts 1 to 3; the legacy code takes none. The legacy venv is `.venv-legacy` at the repository root, or `--legacy-python`.

## Smoke test

`run.py --smoke` fits three models (1,000 to 2,000 cases, degree 1 and 2, one weighted), takes the best of 3 for each, and divides the total by the time of a fixed numpy workload, so that the machine drops out. It fails when this ratio is above 1.2 times the ratio in `baseline.json`. After a deliberate speed change, store a new baseline on `main` with `--smoke --update-baseline`. Gate C calls it.

## Report

```sh
uv run --frozen --group validation python validation/bench/report.py --run validation/runs/bench
```
