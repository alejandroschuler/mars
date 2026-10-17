# Brief: week 4 follow-ups (#120, #119) and T20

Read `COMMON.md` (same folder) first; its rules hold. The executor is session `local_f8e3ee8e-db3a-47de-b16c-dfc9c109b194` and runs in `<main>` = `/Users/aschuler/Documents/research/projects/pymars`. Worktrees go in `<main>/.worktrees/<branch>` (`bash <main>/dev/tools/new_worktree.sh <branch>`; read it first). main is `d477f6d` or later. Work lean: the fewest tests that pin the rules, a mutation sample of at most about 30. Commit and push after each step. Open the draft pull request early. If you stop, push a work-in-progress commit.

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
