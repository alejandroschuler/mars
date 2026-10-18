# Brief: week 4 follow-ups (#120, #119) and T20

Read `COMMON.md` (same folder) first; its rules hold. The executor is session `local_a3dbbbaa-86cb-4746-9ad7-7dae14ff92f9` (since 2026-10-17 20:17 PDT) and runs in `<main>` = `/Users/aschuler/Documents/research/projects/pymars`. Worktrees go in `<main>/.worktrees/<branch>` (`bash <main>/dev/tools/new_worktree.sh <branch>`; read it first). main is `d477f6d` or later. Work lean: the fewest tests that pin the rules, a mutation sample of at most about 30. Commit and push after each step. Open the draft pull request early. If you stop, push a work-in-progress commit.

The R115 scratch scripts that #120 names are gone. Rebuild what you need from #110's review comments (the generator) and from #120's numbers. Each seed or case in #120 states the exact value; check your fix against exact rational arithmetic (fractions.Fraction) or the reference (`tests/reference/mars_ref.py`), never against the fast code.

## F120P: fast pruning (branch `t120-fast-pruning`; Part of #120)

Files: `pymars/_pruning.py`, its test file under `tests/`, and `_compare_final` in `tests/test_oracle.py`.

1. #120 items 1 and 2: the fast pruning misses LA-5 at seeds 90 (sizes 4 and 5, 9.3 and 7.5 times the bound) and 213 (size 7, 1.027 times). Find the cause and fix it within the spec (docs/algorithm.md, LA-5, PRUNE-*). Add the two seeds as oracle or pruning cases (one parametrized test).
2. #120 item 5: seeds 37, 100, 154, 218 and 273 fail only the bound on the RSS of the fast coefficients in `_compare_final`; both RSS values equal the exact value, and even exactly rounded coefficients exceed that bound (PRUNE-8's stated limit). Change the bound to what PRUNE-8 states, with a comment that cites it. Show that a planted wrong coefficient still fails.
3. #119 item 2: the affine-copy symbols in `_pruning.py` may serve LA-5, but they must not change an LA-4 decision in `final_fit`. Check whether they can; if they can, fix it and pin it with one case.
Reviewers: 2 (`spec` and `adversarial`). Run gate B and gate C (`bash dev/gate_c.sh`) on the head.

## F120F: fast forward pass (branch `t120-fast-forward`; Closes #120 when F120P has merged, else Part of #120)

Files: `pymars/_linalg.py`, `pymars/_forward.py`, `pymars/_scan.py` if the cause is there, their test files, and the `ties` draws in `tests/test_oracle.py`. Do not change `pymars/_pruning.py` (F120P owns it); if the fix needs it, report to the executor.

1. #120 item 3: fast LA-4 at seed 15857 of the draw (n = 7): the term x0·h(x2 - 0x1.f4p+0) is exactly in the span of the kept terms, and the fast code keeps it.
2. #120 item 4: the forward pass on copy kinds 2 and 4 (seeds 1258, 1150, 1281 of #110's generator with the copy kind forced): candidate RSS values off the exact value, one below the exact RSS of its span.
3. #119 item 1: the `Conditioner` must map -0 to +0 before it hashes columns (LA-4's copy rule).
4. Then add the copy kinds to the oracle's `ties` draws, as #120 asks. If the fast code and the reference still disagree on some, report the seeds.
Reviewers: 2 (`spec` and `adversarial`). Run gate B and gate C on the head.

## S119: spec and reference nits (branch `t119-spec-ref-nits`; Closes #119 when F120P and F120F cover items 1 and 2, else Part of #119)

Files: `docs/algorithm.md`, `tests/reference/mars_ref.py`, `tests/reference/test_reference.py`. Never read `pymars/` (running tests is fine): the reference is written from the spec only.

1. #119 item 3 (non-blocking items 4, 8 and 10 of the #117 review linked there) and item 4 (the recheck's nits).
2. #119's comment: items 3, 5 and 7 of the #118 review (W-3 check for copy symbols in the constant-columns test; the unused `x_powers` of `Atoms`; nothing to change for the PR body claim, but note it in this PR's body). Reference speed on near-copy fits (4 to 6 times slower) is a note only, unless a simple fix keeps it exact.
Reviewer: 1 (`single`). Run gate B on the head; gate C too if the reference changed.

## T20: performance (branch `t20-perf`; issue #22)

Read the plan's sections "Fast path" and "Profiling", and `validation/bench/REPORT.md` (7.0 times earth at 10,000 x 10 x degree 2, pymars at `ae9d199`; the target is within 10 times, already met there, so the task is to confirm on current main and take the cheap wins).

Stage 1, now: on current main, profile the 10,000 x 10 x degree 2 baseline and the cases in REPORT.md where pymars is slowest relative to earth (small n: 13 times at n = 250; pruning cost at large covariate means, 150 degree-3 shifted terms 2.3 s). Use `py-spy` or `cProfile` (install into the uv venv only). Report the top costs and a plan of passes to the executor, with expected gains. Do not change code in stage 1.
Stage 2, after the executor's go: each pass a commit, gated on the oracle tests (`bash dev/gate_c.sh` before the pull request is ready) and the benchmark smoke. Files: the fast modules your plan names; F120P and F120F own `_pruning.py`, `_linalg.py`, `_forward.py` and `_scan.py` until they merge, so start with other files or wait. Then rerun the benchmark grid on main as a detached job (validation/bench/run.py; read it first) and refresh REPORT.md.
Reviewers: 2 (`spec` and `adversarial`).

## T22a: user docs (branch `t22-docs`; Part of #24)

Files: `README.md`, `CHANGELOG.md`, and new pages `docs/usage.md`, `docs/weights.md`, `docs/differences.md`. Do not change `pymars/`, `docs/algorithm.md` or `validation/`.

Read the plan's T22 row, the definition of done (items 1, 3, 5 and 7), `dev/DECISIONS.md`, `validation/DIFFERENCES.md`, `validation/bench/REPORT.md`, and the public API in `pymars/__init__.py` and the estimators' docstrings. Then write:

1. README: what pymars is, install, a short regression and classification example that runs, the scikit-learn integration in a few lines, the status (validated against earth; link the reports), speed in one sentence with the measured factor, and the legacy section. Every code block must run; add a test only if the repo already has a docs test pattern, else run them by hand and say so in the PR body.
2. `docs/usage.md`: the parameters of each estimator with defaults and their earth names, the fitted attributes, `summary` or its equivalent, GLM and classification, pandas feature names, pipelines and grid search.
3. `docs/weights.md`: what `sample_weight` means, zero weights (W-3), how weights enter GCV and pruning, with the spec sections cited.
4. `docs/differences.md`: the user-facing differences from earth, from `validation/DIFFERENCES.md` and DECISIONS.md, in plain words with the label of each (rule, quirk, tie, numeric).
5. CHANGELOG: the 2.0.0.dev0 entry gains "Added" for the estimators and the validation.
Leave simulation results out: T22b adds them after T21. Plain prose (COMMON.md). Reviewer: 1 (`single`) now; the plan asks for two agent reviews of the docs, and the second comes in T23.

## C07: the fast code against earth (branch `t07-fast-conformance`; definition of done item 4)

Files: `tests/test_conformance.py`, `validation/DIFFERENCES.md`, and the conformance helpers it imports under `validation/` if needed. Do not change `pymars/` or `tests/reference/`.

`tests/test_conformance.py` still has `# T12 adds the fast code: "fast": Fast(),`, so the fast code is never compared with earth. The definition of done asks that the fast code conform to earth as well as the reference does.

1. Turn on the `Fast` implementation and run the conformance suite on current main (gate C runs the `external` tests; read the test file for how to run it with R).
2. Regenerate `validation/DIFFERENCES.md` for both implementations on current main (its counts are from the reference at 2a00063). Every fast-code difference from earth that the reference does not share is a finding: triage it (rule, quirk, tie, numeric, bug) with exact arithmetic or the reference, and report each bug with its fixture to the executor. Do not fix `pymars/`; F120F and F120P own the fast modules now.
3. The test must fail when a fast-code result is planted wrong (as the reference's planted test does), with the fewest lines.
Reviewer: 1 (`single`). Run gate B and gate C on the head. The fast code changes when #122, #123 and T20 merge; the executor reruns this suite before the freeze.

## S127: the forward basis after an LA-4 drop (branch `t127-spec-la1-fwd11`; issue #127)

Files: `docs/algorithm.md`, and `validation/blackbox/` for a new black-box experiment if you run one. You are the spec writer: you may run black-box earth experiments (trace logs, outputs) under the clean-room rules; never read earth's C code or print R function bodies.

Read #127 and the review comments it cites. Decide which basis the forward pass projects onto after a step whose new term LA-4 drops, and what RSS_s is then, so that LA-1, LA-4 and FWD-11 read the same. Prefer the rule that earth follows, if a black-box run on a case of this kind shows it (earth's tolerances differ, so build a case where the drop is clear). Prefer a rule that keeps RSS_s equal to the RSS of the basis that the next step projects onto. State the modules that must follow it (the reference and the fast forward pass) in the PR body; the executor opens the implementation tasks. Reviewers: 2 (`spec` and `adversarial`).

## R127: the reference follows FWD-13 (branch `t127-ref-fwd13`; Part of #127)

Files: `tests/reference/mars_ref.py`, `tests/reference/test_reference.py`, and the draws in `tests/test_oracle.py`. Write from the spec only and never read `pymars/` (running tests is fine). The reference is the oracle.

FWD-13 (merged with #129) defines the forward basis F_s: F_{s+1} is F_s plus the chosen candidate's columns that LA-2 counted, in LA-2's order; RSS_s = RSS(F_s); FWD-11 runs once after the pass and changes no forward value. Today `forward_pass` rebuilds B from the LA-4-kept terms at each step. Change it to build B and LA-2's echelon from F_s, and keep FWD-11's term test for kept and dropped only. The fast code already follows FWD-13 (checked on 6bce1d1). Then:
1. Add #127's seed 1258 (#110's generator with copy kind 2 forced; the generator draws minspan, then endspan), FWD-13's case (b), and one draw of #134's family (seed 150: m1 = 2^26, x2 = 1e15 + k/8) as oracle cases, so the fast-against-reference comparison pins FWD-13.
2. Remove any test that assumes rss_per_size[M_f-1] equals forward.rss[S].
3. Run the thorough oracle (gate C) and report any new difference between the two programs with its seed.
Reviewers: 2 (`spec` and `adversarial`). Gate B and gate C on the head.

## H127: the harness's earth forward path after earth's rank fix (branch `t127-harness-fwd-path`; Part of #127)

Files: `validation/harness/fit_earth.R`, `validation/harness/tests/test_driver.py`, and the tests that compare pymars's forward RSS with that path: `tests/test_conformance.py` (around lines 355 to 400), `tests/test_forward.py` (around 118), `tests/test_core.py` (around 551). If the fixtures must be made again, `validation/harness/gen_fixtures.py` makes them with R; the fixture diff must then be explained in the PR body.

`forward_rss_path` refits earth's `bx` after earth's rank fix, so for a fit where the fix removes a term it is not earth's forward RSS (bb32 shows this). Use earth's trace RSq for the forward path, or skip the forward RSS after "Fixed rank deficient bx", whichever keeps the most comparisons true; black-box earth runs only (clean room). Reviewer: 1 (`single`). Gate B and gate C (it regenerates fixtures with R if the harness changed).

## T21: the freeze, the pilot and the full run (branch `t21-full-run`; Closes #23)

Read `T21a-prepilot.md` (same folder) and its result `validation/sims/results/prepilot/summary.md` on main, the plan's "Statistical performance study" (all of it), "Pilot and number of repetitions", "Compute ledger" and "Long computations". The executor sets the tag `sim-freeze-1` on main and gives you its commit; P-fix is the `EarthRegressor` at that commit.

1. Runner worktree `<main>/.worktrees/runner-t21` at the tag's commit (record it). Post-freeze pilot step 3 (100 repetitions, D3, D4, D5, D7 at 1,000 cases, both noise levels, P-fix and E-def), then `pilot_check.py`; compare with the pre-freeze pilot and size every contrast.
2. The full run: all cells and arms of the plan's design at the sized n_sim (the legacy arms are done, T04; reuse their results), as detached jobs (`nohup caffeinate -i nice -n 15`, at most 6 workers, R counts as one core), results per cell, atomic, manifest, `--resume`. Report the PID file, log path and an ETA as soon as the job starts, then stop; the executor restarts you when it ends.
3. When it ends: failure counts per cell and arm, the core-hours, and `validation/sims/results/full/` (results.csv, failure_counts.csv, summary tables) in a PR. No figures or REPORT.md text (T22b does them).
Files: `validation/sims/results/full/` (new) and harness fixes in `validation/sims/` only if a bug blocks the run (say so). No `pymars/` changes after the tag. Reviewer: 1 (`single`).

## Restart round (2026-10-17 20:20 PDT)

The executor of 2026-10-16 stopped at about 23:00 PDT (the app's sign-in token expired), and its agents stopped with it. Their review comments and pushed commits stand. These tasks restart them.

### F124: T20 review fixes (branch `t20-perf`, PR #124)

Files: the files #124 already changes. Read the `REVIEW spec 93f1182...` comment on #124. Fix blocking item 1: do not keep `g` and `b` in the scan stores (recompute them in O(n) per call), and count `F`, `D` and `zero` in the budget next to K·m, so the stated 4·n·(p + M) holds. Measure the peak RSS that the reviewer measured (n = 20000, p = 30, degree 2, max_terms 61) on main and on the new head, and state both in the PR body. Non-blocking item 2: one small test that forces the budget low and compares `to_dict()` with the default fit. Keep bit-equality: `validation/bench/hash_fits.py` on main and the head must give the same hash (533 fits). Rebase on `origin/main`, then gate B and gate C on the head. Report the head SHA, the hash, the memory and the 10k timing.

### F136: platform-independent FWD-13 oracle cases (branch `t127-ref-fwd13`, PR #136)

Files: `tests/test_oracle.py` (the new test `test_the_forward_basis_keeps_a_direction_that_fwd_11_drops`), and `tests/reference/` only if the cause is there. CI fails on every Ubuntu job (not on macOS): `check_forward(case).near_tie` is `'gap'` instead of `None` for seed 1258 (copy kind 2) and for #134's seed 150 (run 39003503017). Find which forward step is the near tie on Linux (a different BLAS or libm gives a different last bit), and make the test pin FWD-13 without depending on that bit: for example, compare the forward bases and RSS values only up to the near-tie step, or choose draws whose steps have no near tie on either platform, while the test still fails if the reference rebuilds B from the LA-4-kept terms (check that with a planted change in a scratch copy). Do not read `pymars/`. Rebase on `origin/main`; gate B and gate C; wait for CI to be green on all 12 jobs; report.

### F139: a kept term in the span gets an arbitrary fast coefficient (branch `t139-final-fit-span`; issue #139)

Files: `pymars/_pruning.py` (the final fit) and its test file; `tests/test_oracle.py` only to add cases to an existing comparison. PR #138 (branch `t137-final-fit-la4`) changes the same function and is under review: start your branch from `origin/t137-final-fit-la4`, and rebase on `origin/main` after #138 merges (the executor tells you). Read #139 and docs/algorithm.md (PRUNE-8, LA-4, LA-5, and what the final fit gives a kept term whose column the data span). If the spec does not say, report to the executor before you code: a spec change goes to a spec writer. Make the fast coefficients follow the spec (the reference gives 0), and make seed 57's final RSS meet LA-5. Use the generator of #132's spec review (`rev-132-spec`; rebuild it from the review comment on #132 if the script is gone) for a sweep of 1500 seeds against the reference: report the counts of coefficient and RSS differences before and after. Pin the rule with the fewest cases (seeds 37, 49, 57 as one parametrized oracle case). Reviewers: 2. Gate B and gate C on the head.

### S139: the coefficient of a kept term that the data span (branch `t139-spec-span-coef`; Part of #139)

Files: `docs/algorithm.md`, and `validation/blackbox/` for a new black-box experiment if you run one. You are the spec writer (black-box earth runs allowed under the clean-room rules; never read earth's C code or print R function bodies). Do not read `pymars/`.

F139's author found a gap: on #139's seeds LA-4 keeps a term (its new part counts as a formula), but its column is in the span of the other kept columns at the cases, so the least-squares solution on the kept columns is not unique. LA-4 and PRUNE-8 give 0 only to columns that LA-4 drops. The reference (`Conditioned.coef`, `lstsq_coef` in tests/reference/mars_ref.py) gives 0 through a second, unstated test (orthogonal part below 1e-7 of the uncentered w-norm against the earlier kept columns, run on the shifted basis where a covariate is shifted, then mapped back exactly). Decide the rule and write it into LA-4 and PRUNE-8: which solution (the author proposes: a kept column that the earlier kept columns span at the cases gets 0, as R's lm.fit does; minimum norm breaks scale invariance), on which columns the test runs (term columns or the shifted basis; they can give different zero sets), its tolerance and source, and that the fit statistics stay those of LA-1. If earth shows its behavior on such a case black-box (coefficients of a term in the span), prefer it. State in the PR body the modules that must follow (the reference, if its test differs from the rule, and the fast final fit; F139 does the fast part). Seed 57's LA-5 miss is already covered by PRUNE-8; say so. Reviewers: 2 (`spec` and `adversarial`).

### F140: the fast final RSS far below the exact RSS (branch `t140-final-rss`; issue #140)

Files: `pymars/_pruning.py` (the final fit and what it calls in `_pruning.py`), `pymars/_linalg.py` only if the cause is there (then say so; #124 also changes `_linalg.py`, so keep that change small), their test files, and `tests/test_oracle.py` only to add cases to an existing comparison. Start from `origin/t137-final-fit-la4` (PR #138, approved in spirit, rechecks pending); rebase on `origin/main` when the executor says #138 merged. Read #140, the #138 adversarial review it cites, docs/algorithm.md (PRUNE-8, LA-1, LA-4, LA-5), and PR #141 (a spec change under review: the coefficient of a kept term that the data span; do not implement #141 here, but do not contradict it).

1. Reproduce seeds 201 (reverse order), 529, 327 and 743 (weights) with the reviewer's generator (`rev-138-adversarial/search.py` and `dig.py` in the scratchpad `/private/tmp/claude-503/-Users-aschuler-Documents-research-projects-pymars/d6d985a2-02ca-48c6-88b1-338ade8641f5/scratchpad/`). Compute the exact RSS of the kept terms with Fractions or the reference.
2. Find why the fast final RSS is below the exact value (an RSS below the exact least-squares RSS of those columns means the RSS does not belong to the returned columns, or the columns differ). Fix it within PRUNE-8 and LA-5.
3. A sweep of the 1500 weighted seeds and the reverse-order seeds: counts of LA-5 misses before and after. Pin with the fewest cases (one parametrized test).
Reviewers: 2 (`spec` and `adversarial`). Gate B and gate C on the head.
