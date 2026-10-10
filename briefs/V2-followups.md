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

## V2-FWD: the fast forward pass (branch `v2-fwd`, issue #104)

#95 (merged as 1c6c83f) added `Conditioner` in `pymars/_linalg.py`: an exact change of basis for large-mean covariates. Build on it.
Files: `pymars/_forward.py`, `pymars/_scan.py`, `pymars/_linalg.py`, `tests/test_forward.py`, `tests/test_scan.py`, `tests/test_linalg.py`, and in `tests/test_oracle.py` only the `ties` duplicated-covariate SHIFT_CAP line and its comment (lift it) plus designed cases.
Rules: FWD-12 / FAST-5 / CORE-3 / FWD-8 (one candidate per row set; only same-kind occurrences merge), STOP-3 / FAST-6 (code 2 below -1000), EDGE-6 (the TSS check before the degenerate test, in the forward pass), LA-2 and LA-4 / LA-1 / FWD-11 (order-free dependence after the exact shift, computed exactly or in the shifted basis; m_j and the shift criterion as the spec says, which may differ from #95's LARGE_MEAN = 64 and middle value: follow the spec), and the near-copy cases of #99's comment and #104's comment (seed 36: the fast second-best linear candidate off by 1.5 %; one symbol for bitwise-equal x - m columns). Keep ordinary data bit for bit where the spec does not change it, and state the complexity.

## V2-PRUNE: the fast pruning pass (branch `t84-fast-pruning`, draft PR #96, issue #105 and #84)

Continue the draft #96 (read its body and the executor's brief T84b-fast-pruning.md). It reduced the error at shift/spread 2^20 but still misses LA-5 at 2^36 for a parent with a linear factor and for a parent separated by other kept terms (about 1e-5), because R is rounded. Rework it on #95's `Conditioner` (exact change of basis per covariate, as the reference does in prune via span_columns; do not read the reference's code, work from the spec) so that rss_per_size meets spec v2's LA-5 for shifts up to 2^36.
Files: `pymars/_pruning.py`, `pymars/_core.py` (only the arguments passed to the pruning pass; V2-CORE also changes _core.py, keep your change small and rebase), `pymars/_linalg.py` only if the Conditioner needs a small extension (V2-FWD also changes it: coordinate by keeping changes additive), `tests/test_pruning.py`, and in `tests/test_oracle.py` the `pruning_path` SHIFT_CAP (lift it).
Rules: LA-5, LA-4 / LA-1 / PRUNE-8 (dependence after the exact shift in the pruning pass and the final fit), PRUNE-10.
