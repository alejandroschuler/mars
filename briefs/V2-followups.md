# Brief: spec v2 follow-ups (#92 merged as 0c164a8)

Read `COMMON.md` (same folder) first; its rules hold for you. Read `docs/algorithm.md` on `origin/main` (spec v2) and the body of PR #92, section "Changed rules and the modules they affect" (`gh pr view 92 --repo alejandroschuler/mars`). Your section below names your issue, branch, files and rules. Each PR body lists the rules it implements, with evidence. Reviewers: 2 (`spec`, `adversarial`) for code in `pymars/` or the reference; 1 (`single`) for V2-TESTS.

Other authors work at the same time on the other sections; change only your files. If a rule is unclear, stop and report the exact question to the executor; do not change `docs/`.

## V2-REF: the reference (branch `v2-ref`)

Files: `tests/reference/mars_ref.py`, `tests/reference/test_reference.py`. The reference is the oracle: write from the spec only, never read `pymars/`.
Rules: FWD-5 (exact-fit band), FWD-12 / FAST-5 / CORE-3 / FWD-8 (one candidate per row set, same-kind merge), STOP-3 / CORE-4 / FAST-6 (code 2 below -1000), EDGE-7 (power-of-2 scaling of X, ldexp, ValueError on coefficient overflow or underflow, knots on the original scale; in the reference or its core adapter), W-6 (scalar weights), LA-2 and LA-4 / LA-1 / FWD-11 / PRUNE-8 (order-free dependence after the exact shift: reduced echelon form of the exact expansions in shifted atoms, computed exactly), the final fit, and a check that PRUNE-8's rss is the LA-1 value. Also issue #99 and its comment (exact copies and near-copies of a large-mean covariate: one symbol for bitwise-equal columns and for bitwise-equal x - m columns).

## V2-CORE: core, estimators and GLM (branch `v2-core`)

Files: `pymars/_core.py`, `pymars/_estimators.py`, `pymars/_glm.py`, their tests (`tests/test_core.py`, `tests/test_estimators.py`, `tests/test_glm.py`, `tests/test_weights.py`, `tests/test_edge_cases.py`).
Rules: EDGE-7 (power-of-2 scaling of X by ldexp in the core; ValueError when a coefficient overflows or underflows to 0 or a subnormal; knots on the original scale in `cuts`, the forward record and `second_knot`), EDGE-6's ldexp for s in the core, W-6 (int and float scalars broadcast; bool and 0-d arrays raise ValueError; weight-sum overflow and N >= 2^52 raise ValueError before EDGE-6), GLM-4 (warning when |eta| > 30 over positive-weight cases), CORE-4 (the termination code enum, if it changes). Do not change `pymars/_forward.py`, `_scan.py`, `_linalg.py` or `_pruning.py` (other authors); if EDGE-7 needs a hook there, report it.

## V2-TESTS: tolerances and the T07 harness (branch `v2-tests`)

Files: `tests/test_oracle.py` (tolerances and near-tie labels only; another author changes the cap lines later), `tests/test_core.py` only for LA-5 tolerances if no other file can carry them (coordinate: V2-CORE owns test_core.py; prefer a small separate commit and report it), `tests/test_pruning.py` (tolerances only), `validation/harness/` and `tests/test_conformance.py` (T07).
Rules: LA-5's epsilon (1e-8 R + c sqrt(TSS V*) + (c/2)^2 TSS, c = 1e-14) wherever the tests compare RSS; PRUNE-10 (pruning near-tie at 1e-7 of the lower RSS); STOP-7 (the bands, incl. 2 eps); W-3 (zero-weight fits at 1e-7 for sums of squares); FWD-12 (compare rows, not `parent`, at steps that add the same rows from different parents); STOP-3 code 2 and FWD-11's `quirk` label in the conformance triage. If the harness changes, gate C remakes the fixtures with R; run `validation/harness/gen_fixtures.py --check` yourself first (read it).

## Later sections (after #95 merges)

V2-FWD (`pymars/_forward.py`, `_scan.py`, `_linalg.py`) and V2-PRUNE (`pymars/_pruning.py` plus the #96 rework); the executor writes them when #95 is in.
