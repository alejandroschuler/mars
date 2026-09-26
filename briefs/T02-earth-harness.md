# Brief: T02 earth harness author

You build the harness that runs R earth as a black box and compares its fits with pymars. Read `briefs/COMMON.md` (same folder) first; its rules hold for you. Issue: T02 (the executor gives you the number). Reviewer: 1 (`single`).

## Plan sections to read

"Correctness against earth" (all of it: "Harness", "Comparison modes", "Test datasets", "What is compared, and the tolerances", "Ties", "Triage of differences", "Component tests"), "Behavior target: scikit-learn first", "Core API", "Tests and validation folders", and the first appendix. The prototypes in `validation/legacy/` (`fit_earth.R`, `compare_earth.py`, `out/trace9.txt`) seed the harness; read them.

## Clean room

earth is a black box: its outputs, its `trace = 7` to `9` logs, and calls to internal functions such as `earth:::get.gcv(...)` and `earth:::pruning.pass(...)`. You may look at the formal arguments of an internal function with `formals()` or `args()`, which show no code. Never print a function body, never read earth's C code, never quote the help pages or the notes.

## Deliverables (all under `validation/harness/`, plus its tests)

- `fit_earth.R`: reads a JSON config (data file paths, earth arguments, trace level, what to collect), fits earth, writes the results as JSON with full double precision (17 significant digits). It collects `dirs`, `cuts`, `selected.terms`, `prune.terms`, `rss.per.subset`, `gcv.per.subset`, `coefficients`, `glm.coefficients` (with `glm = list(family = binomial)`), `rsq`, `grsq`, `gcv`, `rss`, `termcond`, the fitted values, the predictions on a test set, and the forward RSS path (a fit with `pmethod = "none"`). It supports weights, several responses and a factor response. It records the R and earth versions. With `trace = 7` to `9` it captures the printed trace to a text file.
- `driver.py`: `run_earth(...)` in Python. It writes each dataset as CSV with `%.17g`, calls `Rscript fit_earth.R <config>`, reads the JSON back, and records the versions of R, earth, Python, numpy, scikit-learn, the BLAS library (threadpoolctl) and the pymars commit. One R process may fit several datasets (a block) to pay the R start-up time once.
- `blackbox.py`: calls for the component tests: `earth:::get.gcv` over a grid of RSS values, term counts, penalties (0 to 6 and −1) and case counts; `earth:::pruning.pass` (or `pmethod` outputs) on a fixed basis; `lm.fit` on fixed columns; `predict.earth` at new points; R's `glm` and `nnet::multinom` on fixed columns.
- `trace_parse.py`: parses a `trace = 7` to `9` log into records per forward step and per knot search (parent, predictor): each visited case, whether it was evaluated, the cut, the RSS with that knot multiplied by the sample variance of y (the traced RSS is on the standardized scale; check this on a known fit), and the flags `bx1G`, `CovColG`, `TolG` and `MaxG`. Test it on `validation/legacy/out/trace9.txt` and on one new trace at degree 2.
- `names_map.py`: the table from pymars 2.0 parameter names to earth arguments, as in the plan's "Public API" (`max_degree` to `degree`, `max_terms` to `nk`, `penalty`, `thresh`, `minspan`, `endspan`, `adjust_endspan` to `Adjust.endspan`, `auto_linpreds` to `Auto.linpreds`, `fast_k` to `fast.k`, `fast_beta` to `fast.beta`, `pmethod`, `nprune`), with the automatic values. The spec (`docs/algorithm.md`, written in parallel by T01) has the final table; leave a comment that T07 checks it against the spec.
- `legacy_adapter.py`: fits the legacy code (mars-earth 1.0.4 in the venv `.venv-legacy`, made by `validation/legacy/make_venv.sh`) in a subprocess and returns `basis_`, `coef_`, `gcv_`, `rss_`, `record_.fwd_basis_`, `record_.fwd_rss_` and `record_.pruning_trace_*` in the common result schema, converting the legacy basis to `dirs` and `cuts` where it can.
- `new_adapter.py`: converts a `MarsFit` (plan "Core API") to the common schema. `pymars._core` does not exist yet, so write it against the plan's field list with a small fake record in the tests; T12 connects it.
- `compare.py`: compares two results by the plan's tolerance table: the forward steps (parent, variable, direction, knot) up to the first near-tie, the RSS path (relative 1e-8), the pruning steps and `rss.per.subset` and `gcv.per.subset`, the selected terms, the coefficients (normwise relative 1e-6, only when κ(B) ≤ 1e5), GCV (relative 1e-8), R² and GRSq (absolute 1e-8), fitted values and predictions (1e-8 × sd(y)), the GLM coefficients and probabilities (relative 1e-5, absolute 1e-7), and the defaults mode (report only). It computes κ(B), finds near-ties (best and second-best candidate RSS within 1e-7 of the RSS before the step), stops the structural comparison at the first near-tie, and returns a structured report of every difference with the fields that `DIFFERENCES.md` needs (dataset, step, both choices, both candidate RSS values, earth's flags from the trace).
- `gen_fixtures.py`: the fixture generator framework: a registry of datasets, each a function that returns the inputs; for each dataset and mode it runs earth and writes `validation/fixtures/<case>.json` with the inputs, the earth arguments, the outputs and the versions. One command makes all registered fixtures, and a `--check` flag makes them again in a temporary folder and reports any diff. Register one example (S01: one covariate, two true knots, 200 cases) to prove the pipeline. T05 adds S02 to S20.
- Tests in `validation/harness/tests/` (not in `tests/`, because CI has no R). Tests that need R run only when you call them; do not add skips. Pure-Python parts (the parser, `compare.py`, the adapters with fakes) get tests that need no R.
- A section in `validation/README.md` (make the file if it does not exist) that says how to run the harness and the fixture generator.

## Size

If the change grows past about 800 lines that are not generated, split it into two pull requests on branches `t02-harness-r` and `t02-harness-compare`: first `fit_earth.R`, the driver, the parser, the black-box calls and the names table; then the adapters, `compare.py` and `gen_fixtures.py`. The first says `Part of #<issue>`, the second `Closes #<issue>`.

## Checks

- Gate A and gate B pass (they cover `pymars/` and `tests/`; ruff also checks `validation/`).
- `uv run --frozen --group validation pytest validation/harness/tests` passes, with its R-based tests run once; say in the pull request which tests needed R and how long they took.
- The S01 fixture is made twice with no diff.
- New Python dependencies go only in the `validation` dependency group (update `uv.lock`).

## Report

30 lines or fewer: the pull request number and head SHA, the gate results, the harness test results, the S01 fixture check, and anything the spec writer or T05 should know.
