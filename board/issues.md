# Board issues

Each block below becomes one issue in `alejandroschuler/mars`. `board/create_issues.py` reads this file. A block starts with a line `=== <title>` and a line `labels: <comma-separated labels>`; the rest of the block is the body. The script skips a block whose title already exists as an issue (open or closed), so it is safe to run again.

=== T00: bootstrap (safety setup, clean slate, skeleton, board, journal, watchdog)
labels: task,P0,claimed
Task T00 of `VALIDATION_PLAN.md` (sections "How the work is executed", "Surviving usage limits and crashes" and the bootstrap appendix).

**Needs.** Nothing. **Reviewers.** 1. **Target hours.** 0 to 2.

- [x] Safety setup: gh default repo, upstream push URL disabled, `.worktrees/` excluded, `validation-plan` and `legacy-1.0.4-head` pushed, caffeinate, versions in the journal.
- [x] Journal branch `executor`.
- [x] Watchdog task `pymars-executor-watchdog`.
- [ ] Bootstrap pull request merged.
- [ ] Issues, labels, task issues and the missing-values issue.
- [ ] CI runs on `main`, or a `needs-user` issue for the Actions button.
- [ ] Checkouts synced to `main`.
- [ ] Recovery drill: a stopped helper restarts on the next wake; the watchdog run starts in `<main>`, in Auto mode, and stops without acting.

=== T01: spec v1 (docs/algorithm.md), then spec v2 after the triage
labels: task,P1,todo
Task T01 of `VALIDATION_PLAN.md` ("Tasks", "Behavior target: scikit-learn first", "Code reading against the references", "Speed and scaling", "Removal and target architecture").

**Content.** The written rules of the fitting algorithm, each with its source (Friedman 1991 and 1993 by equation, Milborrow's notes by section, paraphrased and never quoted, or a black-box experiment in `validation/blackbox/`). It fixes the core API so the reference, the fast code and the estimators can be written in parallel. Spec v2 follows the triage in T07.
**Needs.** T00. **Reviewers.** 2 (spec against the plan and the references; adversarial). **Target hours.** 2 to 10; v2 by 16.
**Owns.** `docs/algorithm.md`, `validation/blackbox/`.
**Clean room.** Only the spec writer and its helpers run trace experiments. Black-box earth only: outputs, `trace = 7` to `9` logs, calls to internal functions without reading their code. No R function bodies, no earth C code, nothing quoted from the notes or the help pages.

- [ ] Part 1 pull request (unblocks T08 to T10): terms (`dirs`, `cuts`, degree, labels), the GCV and GRSq (C = M + d(M − 1)/2, `penalty = -1`, infinite when C ≥ N, default penalty), the default term limit, the spans (eq. 43 and 45 with α = 0.05, truncation, `adjust_endspan`), the candidate knots on the weighted distribution of the active cases, the linear algebra contract (weighted Gram-Schmidt, collinearity test, pivoted QR, downdates), the pruning pass (backward by RSS, the intercept kept, GCV per size, ties to the smaller model, `nprune`, `pmethod="none"`), the core API (`fit_mars`, `MarsParams`, `MarsFit`, the records, the reference dict, termination codes).
- [ ] Part 2 pull request: the forward pass (parents, variables, selection by the largest RSS reduction, pairs, single hinges, the linear option `auto_linpreds`, the collinearity tolerance and its counter, the knot at the minimum, the stopping rules, y scaling, ties), Fast MARS (`fast_k`, `fast_beta`), weights (frequency weights, N = Σw, spans and knots in weight, zero weights dropped, the warning rule), several responses, the GLM refit (binomial, multinomial, `glm_alpha`, separation), degenerate inputs, errors (NaN, strings, `allow_missing=True`), the public API and the table that maps pymars names to earth arguments.
- [ ] Each rule cites its source; each black-box claim has a script and its output in `validation/blackbox/`.
- [ ] Spec v2 after T07: the triage results, with follow-up tasks for both implementations.

=== T02: earth harness (fit_earth.R, driver, trace parser, fixture generator)
labels: task,P1,todo
Task T02 of `VALIDATION_PLAN.md` ("Correctness against earth": "Harness", "Comparison modes", "What is compared, and the tolerances").

**Content.** A Python driver that writes each dataset as CSV with 17 significant digits, runs earth through `Rscript`, and reads the results back as JSON, with the versions of R, earth, Python, numpy, scikit-learn and BLAS and the pymars commit. It collects `dirs`, `cuts`, `selected.terms`, `prune.terms`, `rss.per.subset`, `gcv.per.subset`, `coefficients`, `glm.coefficients`, `rsq`, `grsq`, `termcond`, fitted values and test predictions; the forward RSS path with `pmethod = "none"`; the `trace = 7` to `9` logs and their parser (per knot search: evaluated cases, cuts, RSS rescaled by the sample variance of y, the flags `bx1G`, `CovColG`, `TolG`, `MaxG`); black-box calls such as `earth:::get.gcv` and `earth:::pruning.pass`; adapters for the legacy code (the venv from `validation/legacy/make_venv.sh`) and for the new `MarsFit` record; the table that maps pymars names to earth arguments; the fixture generator framework that T05 fills.
**Needs.** T00. **Reviewers.** 1. **Target hours.** 2 to 6.
**Owns.** `validation/harness/`, the harness tests.

- [ ] `fit_earth.R` and the Python driver, seeded from `validation/legacy/`.
- [ ] Trace capture and `trace_parse.py`, tested on a stored trace.
- [ ] Version records.
- [ ] Legacy adapter and a stub for the new-code adapter.
- [ ] `gen_fixtures.py` framework with one example dataset.
- [ ] Tests that need R are marked and documented; they run in gate C and in the optional workflow.

=== T03: simulation harness (DGPs, learners, run.py with resume, summaries, pilot_check.py)
labels: task,P1,todo
Task T03 of `VALIDATION_PLAN.md` ("Statistical performance study", "Long computations", "Compute ledger").

**Content.** `validation/sims/dgps.py`, `learners.py`, `run.py` (per-cell atomic results, a manifest, `--resume`, a cache keyed by the data, the learner settings, the learner source hash and the versions), `summarize.py` (every mockup), `pilot_check.py` (`n_sim` and `n_sim_safe` for difference contrasts, the equivalence sizing), the DGP diagnostics from one draw of 10⁶ cases, the arms E-def, E-pym, P-cur, P-ear, P-fix, OLS and HGB and the binary arms, earth through `Rscript` in blocks, the legacy arms through a subprocess in `.venv-legacy`.
**Needs.** T00. **Reviewers.** 1. **Target hours.** 2 to 6.
**Owns.** `validation/sims/`.

- [ ] DGPs D1 to D8, D3-bin, D4-bin, with σ from the population R², the copula for D8, and λ for the binary DGPs.
- [ ] Seeds from names; the test set from a second stream.
- [ ] Learners with fixed settings; P-fix as a registered arm that works once `pymars.EarthRegressor` exists.
- [ ] `run.py --resume`, atomic writes, manifest, failure counts, joblib with one BLAS thread, `nice`.
- [ ] Measures: excess risk on 10,000 test points, log ratios, excess log loss, Brier score, calibration slope with clipping, term counts, irrelevant covariates, fit time.
- [ ] `summarize.py` fills the mockups; `pilot_check.py`.
- [ ] Diagnostics from one draw of 10⁶ cases per DGP.

=== T04: legacy pilot, then the legacy full run (detached jobs)
labels: task,P1,todo
Task T04 of `VALIDATION_PLAN.md` ("Pilot and number of repetitions", "Compute ledger", "Long computations").

**Content.** The legacy pilot (4 cells × 100 repetitions × 3 arms: D4 and D5 at 200 cases, both noise levels, P-cur, P-ear and E-def), `pilot_check.py` on it, then the legacy full run as detached jobs on 4 workers under `nice`, capped at 300 repetitions at 200 cases and 200 at 1,000 cases, and the low-priority batch when cores are free.
**Needs.** T03. **Reviewers.** 1. **Target hours.** 5 to 28.
**Owns.** the run configuration in `validation/sims/`, the results under `validation/runs/` and their summaries.

- [ ] Pilot run and its report.
- [ ] Full legacy run started, with PID files and logs; ledger entries.
- [ ] Results collected and summarized.

=== T05: fixtures S01 to S20
labels: task,P1,todo
Task T05 of `VALIDATION_PLAN.md` ("Test datasets", "What is compared, and the tolerances", "Ties").

**Content.** The deterministic datasets S01 to S20 and the earth outputs for each, as JSON with the R and earth versions, made by one command; the optional workflow `earth-conformance.yml` (`workflow_dispatch`) that installs R and earth, makes the fixtures again and compares them.
**Needs.** T02. **Reviewers.** 2. **Target hours.** 6 to 12.
**Owns.** `validation/fixtures/`, the dataset definitions in `validation/harness/`, `.github/workflows/earth-conformance.yml`.

- [ ] S01 to S20 inputs and earth outputs, in the matched, earth-compatible and defaults modes where they apply.
- [ ] One command makes them again with no diff.
- [ ] Fixture sizes kept small.

=== T06: reference implementation (tests/reference/mars_ref.py), from the spec only
labels: task,P2,todo
Task T06 of `VALIDATION_PLAN.md` ("Tests and validation folders", "Fast path").

**Content.** The oracle: plain numpy, no pymars imports, written from `docs/algorithm.md` alone by an agent that has not read the fast code. For every candidate it builds the explicit hinge columns and solves by pivoted QR, it applies the collinearity rule by an explicit centered regression, and its pruning refits every subset. It covers the whole spec (weights, several responses, Fast MARS, linear terms, stopping rules) and is practical up to n ≤ 300, p ≤ 6 and degree ≤ 3. `tests/reference/test_reference.py` holds its sanity tests.
**Needs.** T01, T05. **Reviewers.** 2. **Target hours.** 8 to 18.
**Owns.** `tests/reference/`.

- [ ] Forward pass, pruning pass and GCV per the spec.
- [ ] Weights, several responses, Fast MARS, linear terms, stopping rules.
- [ ] Sanity tests.

=== T07: conformance tests and triage (DIFFERENCES.md)
labels: task,P2,todo
Task T07 of `VALIDATION_PLAN.md` ("Comparison modes", "What is compared, and the tolerances", "Ties", "Triage of differences").

**Content.** `tests/test_conformance.py`: both implementations against earth in the matched and the defaults mode, with the tolerance table. Every difference goes into `validation/DIFFERENCES.md` with the dataset, the step, both choices, both candidate RSS values, earth's flags and one label (`rule`, `bug`, `quirk`, `tie`, `numeric`).
**Needs.** T05, T06. **Reviewers.** 1. **Target hours.** 8 to 18.
**Owns.** `tests/test_conformance.py`, `validation/DIFFERENCES.md`.

- [ ] Conformance tests for the reference.
- [ ] Triage of every difference; spec follow-ups for T01 v2.
- [ ] The near-tie rate on S15.

=== T08: _terms.py, _gcv.py and _knots.py
labels: task,P2,todo
Task T08 of `VALIDATION_PLAN.md` ("Target modules", "Term structure", "Component tests").

**Content.** `pymars/_terms.py`, `pymars/_gcv.py`, `pymars/_knots.py` per the spec, with `tests/test_terms.py`, `tests/test_gcv.py` (against an `earth:::get.gcv` fixture, `penalty = -1` included) and `tests/test_knots.py` (against earth trace fixtures).
**Needs.** T01 part 1. **Reviewers.** 2. **Target hours.** 10 to 18.

- [ ] Modules and unit tests; coverage at least 90 percent.

=== T09: _linalg.py
labels: task,P2,todo
Task T09 of `VALIDATION_PLAN.md` ("Target modules", "Fast path").

**Content.** `pymars/_linalg.py`: weighted Gram-Schmidt applied twice to append a column to Q; the collinearity test (1 − R² of the centered new column on the existing ones); pivoted-QR least squares; subset downdates for pruning. `tests/test_linalg.py`.
**Needs.** T01 part 1. **Reviewers.** 2. **Target hours.** 10 to 18.

- [ ] Module and unit tests, including rank-deficient and ill-conditioned cases.

=== T10: _pruning.py
labels: task,P2,todo
Task T10 of `VALIDATION_PLAN.md` ("Pruning pass", "Pruning of a fixed basis").

**Content.** `pymars/_pruning.py`: backward elimination by RSS with the intercept kept; the RSS and GCV for each size; the size with the lowest GCV, ties to the smaller model; `nprune`; `pmethod="none"`. `tests/test_pruning.py` (a fixed basis against earth).
**Needs.** T01 part 1. **Reviewers.** 2. **Target hours.** 10 to 18.

- [ ] Module and unit tests against earth fixtures.

=== T11: _scan.py and _forward.py (staged: degree 1; interactions, linear option, collinearity; Fast MARS, weights and several responses)
labels: task,P3,todo
Task T11 of `VALIDATION_PLAN.md` ("Fast path", "Cost model", "Forward pass").

**Content.** The fast forward pass. One pull request per stage, each gated on the oracle tests: stage 1 degree 1; stage 2 interactions, the linear option and the collinearity rule; stage 3 Fast MARS, and weights with several responses.
**Needs.** T08, T09. **Reviewers.** 2. **Target hours.** 12 to 30.

- [ ] Stage 1.
- [ ] Stage 2.
- [ ] Stage 3.

=== T12: _core.py
labels: task,P3,todo
Task T12 of `VALIDATION_PLAN.md` ("Core API").

**Content.** `pymars/_core.py`: `MarsParams`, `MarsFit` and the records; `fit_mars()` runs the y scaling, the forward pass, the pruning pass and the final weighted least squares on the original scale.
**Needs.** T08 to T10. **Reviewers.** 2. **Target hours.** 14 to 30.

- [ ] Module and tests.

=== T13: estimators (EarthRegressor, EarthClassifier, Earth)
labels: task,P3,todo
Task T13 of `VALIDATION_PLAN.md` ("Public API", "Behavior target: scikit-learn first", "scikit-learn").

**Content.** `pymars/_estimators.py` and `pymars/__init__.py`: validation in `fit`, `validate_data`, weight checks, tags, fitted attributes, `summary()`, `basis_matrix(X)`, the `allow_missing=True` error with a link to the missing-values issue. Built against the reference until stage 1 of T11 lands. `tests/test_sklearn_checks.py` with `parametrize_with_checks` and no expected failures.
**Needs.** T12. **Reviewers.** 2. **Target hours.** 14 to 30.

- [ ] Regressor, `Earth` alias, checks.

=== T14: _glm.py and the classifier
labels: task,P3,todo
Task T14 of `VALIDATION_PLAN.md` ("Binary outcomes", "Behavior target").

**Content.** `pymars/_glm.py`: binomial for 2 classes, multinomial for 3 or more, unpenalized by default, `glm_alpha`, separation warnings; `EarthClassifier` with `predict_proba` and `decision_function`; `tests/test_glm.py` against R's `glm` and `nnet::multinom`.
**Needs.** T13. **Reviewers.** 2. **Target hours.** 14 to 30.

- [ ] Solver spike (LogisticRegression with `C=np.inf` or IRLS) to 1e-5 of R's `glm`.
- [ ] Module, classifier and tests.

=== T15: oracle tests and gate C
labels: task,P3,todo
Task T15 of `VALIDATION_PLAN.md` ("Fast path", "Gates").

**Content.** `tests/test_oracle.py`: the fast code against the reference, on hypothesis data and on every fixture dataset, up to the first near-tie; `dev/gate_c.sh`.
**Needs.** T06. **Reviewers.** 2. **Target hours.** 14 to 30.

- [ ] Oracle tests; gate C script.

=== T16: invariance and weight tests
labels: task,P3,todo
Task T16 of `VALIDATION_PLAN.md` ("Invariance tests", "Sample weights", "Edge cases").

**Content.** `tests/test_invariance.py`, `tests/test_weights.py`, `tests/test_edge_cases.py`. Reuse the idea of the legacy weight-shift test (`git show legacy-1.0.4-head:tests/test_sklearn_compat.py`, lines 97 to 111) and the hypothesis strategy pattern (`git show legacy-1.0.4-head:tests/test_property.py`, lines 14 to 40).
**Needs.** T13. **Reviewers.** 1. **Target hours.** 14 to 30.

- [ ] Tests.

=== T17: integration tests
labels: task,P3,todo
Task T17 of `VALIDATION_PLAN.md` ("scikit-learn", "Definition of done").

**Content.** `tests/test_sklearn_integration.py`: clone and pickle; Pipeline with a scaler, and with ColumnTransformer and OneHotEncoder; `cross_val_score` and GridSearchCV with `n_jobs=2`; `cross_val_predict` with `method="predict_proba"`; StackingRegressor and StackingClassifier; TransformedTargetRegressor and CalibratedClassifierCV; pandas feature names; `sample_weight` through metadata routing.
**Needs.** T13. **Reviewers.** 1. **Target hours.** 14 to 30.

- [ ] Tests.

=== T18: description of the legacy code (DIFFERENCES_legacy.md)
labels: task,P1,todo
Task T18 of `VALIDATION_PLAN.md` ("Triage of differences", "Preliminary findings").

**Content.** The legacy code against earth on S01 to S20 with the harness; every difference labeled in `validation/DIFFERENCES_legacy.md`; the upstream reproductions for the findings.
**Needs.** T02 (and the fixtures of T05 where needed). **Reviewers.** 1. **Target hours.** 10 to 24.

- [ ] Legacy comparisons and labels.

=== T19: benchmark harness
labels: task,P4,todo
Task T19 of `VALIDATION_PLAN.md` ("Benchmark design", "Profiling").

**Content.** `validation/bench/`: one factor at a time around 1,000 cases, 10 covariates, degree 2 and 21 terms; earth at its defaults and with `fast.k = 0`; median of 3 runs with one thread; wall time and peak memory; a timeout of 30 minutes; the legacy code up to 2,000 cases; log-log slopes with the model sizes reached.
**Needs.** T15. **Reviewers.** 1. **Target hours.** 20 to 40.

- [ ] Harness and a first report.

=== T20: performance passes
labels: task,P4,todo
Task T20 of `VALIDATION_PLAN.md` ("Fast path", "Profiling").

**Content.** Profile (`line_profiler`, `py-spy`, counters) and speed up the fast code, each pass gated on the oracle tests. Target: within 10 times earth's time at 10,000 cases, 10 covariates and degree 2.
**Needs.** T19. **Reviewers.** 2. **Target hours.** 20 to 40.

- [ ] Passes and the benchmark report.

=== T21: gate D, the freeze, the new-code pilot and full run
labels: task,P5,todo
Task T21 of `VALIDATION_PLAN.md` ("Gates", "Pilot and number of repetitions", "Compute ledger").

**Content.** Gate D at hour 32, the freeze `sim-freeze-1` by hour 40, then the new pilot (1,000 cases on D3, D4, D5 and D7 at both noise levels, P-fix and E-def) and the full run on 8 workers.
**Needs.** T11 to T17. **Reviewers.** 1. **Target hours.** 32 to 48.

- [ ] Gate D; freeze.
- [ ] Pilot and sizing.
- [ ] Full run.

=== T22: summaries, figures, docs, README, CHANGELOG and REPORT.md
labels: task,P6,todo
Task T22 of `VALIDATION_PLAN.md` ("Reporting, and what to offer upstream").

**Content.** `validation/REPORT.md` organized by claim (conformance, gap, parity, sparsity, speed), each with its ADEMP description, evidence and statement; every mockup filled; the docs (usage, weights, differences from earth), the README and the CHANGELOG.
**Needs.** T04, T21. **Reviewers.** 2. **Target hours.** 40 to 52.

- [ ] Report, docs, README, CHANGELOG.

=== T23: final architecture review, upstream drafts, ctml-skills note
labels: task,P6,todo
Task T23 of `VALIDATION_PLAN.md` ("Upstream", "For the ctml-skills supervised-learning skill", "Definition of done").

**Content.** Two fresh agents review the finished architecture; every finding is fixed or filed as `later`. The upstream drafts in `validation/upstream/` (not posted). `validation/ctml-skills-note.md`.
**Needs.** T22. **Reviewers.** 2. **Target hours.** 48 to 56.

- [ ] Review and fixes.
- [ ] Upstream drafts.
- [ ] ctml-skills note.

=== T24: final audit and wrap-up
labels: task,P6,todo
Task T24 of `VALIDATION_PLAN.md` ("Definition of done").

**Content.** Map every criterion of the definition of done to evidence in `LOG.md`; delete the watchdog task; stop caffeinate; the final summary to the user.
**Needs.** all. **Target hours.** 56 to 60.

- [ ] Audit; wrap-up.

=== later: support missing values (allow_missing=True)
labels: later
pymars 2.0 rejects missing values: `validate_data` raises a ValueError for NaN, and `allow_missing=True` raises NotImplementedError with a link to this issue (`VALIDATION_PLAN.md`, "Missing values", decision Q4).

Design for the later work:

- py-earth's published design, in which a hinge on a covariate is multiplied by an indicator that the covariate is present, and missingness indicators can enter as terms.
- A comparison with earth on an augmented design: missing values set to 0, presence indicators added as columns, and the degree raised by one so that earth can multiply a hinge by an indicator. That design can represent the same functions, although earth's search over it differs, so the comparison checks predictions and model quality rather than exact terms.
- Checks: `allow_missing=True` gives the same model as `False` when no value is missing; predictions are finite for rows with missing values; a covariate with values missing completely at random is selected no more often than when it is fully observed.

The legacy code gave NaN predictions for every row with a missing value and scored candidates on different row subsets (finding F8).

=== later: negative minspan (at most k evenly spaced knots per variable)
labels: later
earth's negative `minspan` asks for at most that many evenly spaced knots per variable. pymars 2.0 does not support it at first (`VALIDATION_PLAN.md`, "Behavior target", row "Names and automatic values"). Add it with a spec section, reference support, and a knot-set test against an earth trace.

=== later: newvar.penalty
labels: later
earth's `newvar.penalty` penalizes a new variable in the forward pass. Not in pymars 2.0 at first (`VALIDATION_PLAN.md`, "Behavior target"). Add it with a spec section, reference support and conformance tests.

=== later: linpreds (covariates forced to enter linearly)
labels: later
earth's `linpreds` makes chosen covariates enter only linearly. Not in pymars 2.0 at first (`VALIDATION_PLAN.md`, "Behavior target"). Add it with a spec section, reference support and conformance tests.

=== later: allowed (a function that restricts interactions)
labels: later
earth's `allowed` argument restricts which terms may enter. Not in pymars 2.0 at first (`VALIDATION_PLAN.md`, "Behavior target"). Design a Python analogue (for example a callable or a list of allowed variable pairs) with a spec section and tests.

=== later: other pmethod values and nfold
labels: later
pymars 2.0 supports `pmethod="backward"` and `"none"`. earth also has `"exhaustive"`, `"forward"`, `"seqrep"` and `"cv"`, and the `nfold` cross-validation of the pruning pass. Add them later with spec sections and conformance tests (`VALIDATION_PLAN.md`, "Behavior target").

=== later: feature_importances_ (earth's evimp)
labels: later
earth's `evimp` gives variable importance from the pruning sequence (counts, and GCV or RSS changes). pymars 2.0 has no `feature_importances_` at first (`VALIDATION_PLAN.md`, "Behavior target", row "Coefficients and transform"). Add it with a spec section and a test against `evimp`.
