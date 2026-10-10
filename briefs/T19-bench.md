# Brief: T19, benchmark harness (#21)

Read `COMMON.md` (same folder) first; its rules hold for you. Issue: #21 (T19). Branch: `t19-bench`. Reviewer: 1 (`single`).

Read `VALIDATION_PLAN.md` on `origin/main`: "Speed and scaling" (all of it: "Cost model", "Benchmark design", "Profiling", "Fast MARS status", "Fast path"), "Preliminary findings" ("Fit time against earth"), and "Long computations".

## Task

Build `validation/bench/`: a harness that times pymars (`pymars.fit_mars` and `EarthRegressor`), R earth (black box, through `Rscript`; earth at its defaults and with `fast.k = 0`), and the legacy code (the tag `legacy-1.0.4-head`, up to 2,000 cases; see how `validation/harness/` or `validation/legacy/` runs it) on the plan's design: one factor at a time around 1,000 cases, 10 covariates, degree 2 and 21 terms; n, p, degree and the term limit varied; median of 3 runs with one thread (`dev/env.sh`); wall time and peak memory; a timeout of 30 minutes; log-log slopes, with the model sizes reached. Fix the data seeds. Match earth's settings to pymars with the parameter map the harness already uses.

- Results per cell, written atomically, with a manifest and `--resume`, as the plan's "Long computations" says. The full grid runs as a detached job under `nice -n 15` on at most 4 cores (R counts as one core) from a runner worktree `<main>/.worktrees/runner-t19` at a fixed commit. Start it once the harness passes a smoke run, and put its PID file and log path in the PR body.
- A `--smoke` mode (3 fits, under a minute) that gate C can call: pymars no slower than 1.2 times a stored baseline from `main`. Read `dev/gate_c.sh` (if it exists on `origin/main`) and wire the smoke test in, or say in the PR why not.
- A report script that writes `validation/bench/REPORT.md` from the results: a table of times by factor, the slopes, and the ratio to earth at 10,000 cases, 10 covariates and degree 2 (the definition of done item 5). Commit a first report from the finished cells; a later PR can refresh it.
- Profiling hooks (`--profile` with cProfile output per cell) for T20; no optimization of `pymars/` in this PR.

Files: `validation/bench/` (new), `dev/gate_c.sh` (smoke wiring only), `pyproject.toml` only if a dev dependency is needed (from PyPI, into the uv lock). Tests: a few fast tests of the harness logic (result records, resume, slope fit); no test per script. PR body: `Closes #21`.
