# The statistical performance study

This folder is the simulation harness for `VALIDATION_PLAN.md`'s "Statistical
performance study": the DGPs, the learners, the run harness and the report
displays that test the gap, parity and sparsity claims. Read that section
first; this file only says how to run things.

## Layout

| File | What it does |
|---|---|
| `dgps.py` | D1-D8, D3-bin, D4-bin; the 51-cell grid (`all_cells()`) |
| `diagnostics.py` | one 10**6-case draw per DGP; writes the committed `diagnostics.json` |
| `seeds.py` | a stable hash from a cell's name and repetition, to `numpy.random.default_rng` |
| `metrics.py` | excess risk, log ratios, the binary-outcome measures |
| `learners.py` | the ten arms: inline (OLS, HGB, LogReg, P-fix), R-backed (E-def, E-pym) and legacy-venv-backed (P-cur, P-ear, EarthClassifier, GLMEarth) |
| `fit_earth_block.R` | fits R's earth on a block of datasets in one `Rscript` process |
| `legacy_worker.py` | fits the mars-earth 1.0.4 wheel on a block of datasets in one `.venv-legacy` process |
| `run.py` | the CLI: generates data, dispatches arms, writes results and the cache |
| `pilot_check.py` | `n_sim` / `n_sim_safe` for a difference contrast, `n_sim` for an equivalence contrast |
| `summarize.py` | builds every mockup display from the result files on disk |
| `tests/` | see "Tests" below |

## One-time setup

```sh
. dev/env.sh
uv sync --frozen --group dev --group validation
bash validation/legacy/make_venv.sh   # builds .venv-legacy (mars-earth 1.0.4)
```

`Rscript` and the R package `earth` must already be on `PATH` for the R-backed
arms (E-def, E-pym); nothing here installs R.

## Diagnostics (run once; the output is committed)

```sh
uv run --frozen python -m validation.sims.diagnostics
```

Writes `validation/sims/diagnostics.json`. `dgps.generate` reads it for sigma
(the regression DGPs), and lambda and mean_f (D3-bin, D4-bin); nothing else
in this folder draws a fresh 10**6-case sample.

## Running the harness

```sh
uv run --frozen python -m validation.sims.run \
  --out validation/runs/<name> \
  --arms E-def,E-pym,P-cur,P-ear,OLS,HGB \
  --dgps D1,D2,D3,D4,D5,D6,D8 --sizes 200 \
  --reps 0:2 --n-jobs 3 --resume
```

- `--out` is a folder under the ignored `validation/runs/`; results land at
  `<out>/<cell-name>/<arm>/rep<NNNN>.json`, one file per (cell, arm,
  repetition), written atomically.
- `--cells` names exact cells (`D4_n01000_lo`); or filter the full 51-cell
  grid with `--dgps`, `--sizes` and `--noise` (`lo`, `hi`, and/or `na` for D6
  and the binary DGPs, which have one level).
- `--reps start:end` is a half-open repetition range.
- `--resume` skips a (cell, arm, repetition) whose result file already
  parses as JSON, and redoes one that does not (an interrupted write).
- Predictions are also cached under `validation/runs/.cache/` (shared across
  every `--out`), keyed by the training and test data, the arm's name,
  settings and source code, and the package versions. Deleting it changes
  nothing but runtime.
- D7 (50 covariates) is far more expensive for the legacy arms (P-cur,
  P-ear): run it as its own invocation without them, for example
  `--arms E-def,E-pym,OLS,HGB --dgps D7`.
- A run this size (minutes or more) is meant to start under
  `nohup caffeinate -i nice -n 15 ...`; `run.py` writes its own PID to
  `<out>/run.pid` at startup.

### Smoke run

Two repetitions per cell, at 200 cases, checks the harness runs end to end
before committing to a pilot or a full run:

```sh
uv run --frozen python -m validation.sims.run \
  --out validation/runs/smoke \
  --arms E-def,E-pym,P-cur,P-ear,OLS,HGB \
  --dgps D1,D2,D3,D4,D5,D6,D8 --sizes 200 --reps 0:2 --n-jobs 3 --resume
uv run --frozen python -m validation.sims.run \
  --out validation/runs/smoke \
  --arms E-def,E-pym,OLS,HGB --dgps D7 --sizes 200 --reps 0:2 --n-jobs 3 --resume
```

(P-fix is left out: it needs `pymars.EarthRegressor` / `EarthClassifier`,
which do not exist yet, and raises `NotImplementedError` until they do.)

### Pilot and full run

Not part of this task (T04): the pilot sizes the full run with
`pilot_check.py` (100 repetitions on the gap-claim and parity-claim cells the
plan names), and the full run uses the resulting `n_sim`. Both are the same
`run.py` command above, at a larger `--reps` range and (after the freeze tag)
including `P-fix`.

## Reports

```sh
uv run --frozen python -m validation.sims.summarize --results validation/runs/<name>
```

Writes `<results>/report/`: `ratio_table.csv`, `selection_table.csv`,
`binary_outcome_table.csv`, `appendix_table.csv` (every cell, its Monte Carlo
standard error and its failure count), `equivalence_figure.png` and
`box_plots.png`. A cell or comparison with no successful, paired repetitions
yet shows as `missing`; nothing is invented.

## Tests

Pure Python, no R and no `.venv-legacy` (`test_dgps.py`, `test_seeds.py`,
`test_diagnostics.py`, `test_metrics.py`, `test_pilot_check.py`,
`test_learners.py`, `test_run.py`, `test_summarize.py`):

```sh
uv run --frozen pytest validation/sims/tests -q \
  --ignore=validation/sims/tests/test_learners_r.py \
  --ignore=validation/sims/tests/test_learners_legacy.py
```

R-backed (`test_learners_r.py`, needs `Rscript` and earth) and
legacy-venv-backed (`test_learners_legacy.py`, needs `.venv-legacy`): these do
not skip when the dependency is missing; they fail.

```sh
uv run --frozen pytest validation/sims/tests/test_learners_r.py -q
uv run --frozen pytest validation/sims/tests/test_learners_legacy.py -q
```

`validation/sims/tests/` is not on the repository's root `testpaths`, so gate
A and gate B's bare `pytest` do not collect these automatically; run them
directly with the commands above.
