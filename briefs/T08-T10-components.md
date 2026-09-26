# Brief: T08, T09 and T10, the component modules

Each of these three tasks writes one or more modules of the fast package `pymars/` from the spec, with unit tests. Read `briefs/COMMON.md` (same folder) first. Each task has its own issue, branch and pull request, and 2 reviewers (`spec`, `adversarial`). Model: the strongest.

- T08 (#10): `pymars/_terms.py`, `pymars/_gcv.py`, `pymars/_knots.py`; tests `tests/test_terms.py`, `tests/test_gcv.py`, `tests/test_knots.py`.
- T09 (#11): `pymars/_linalg.py`; tests `tests/test_linalg.py`.
- T10 (#12): `pymars/_pruning.py`; tests `tests/test_pruning.py`. It uses `_gcv` and `_linalg`, so it starts when T08 and T09 have merged, or it builds on their branches with the executor's approval.

## Rules for all three

- Implement from `docs/algorithm.md` on `origin/main` (spec v1). Every function's docstring cites the spec rules it implements (for example "GCV-2, GCV-7"). Where the spec is unclear, raise it on #44 (spec v2) and follow the spec's stated default.
- Do not read `tests/reference/` (T06 writes the oracle in parallel; the two must fail independently). Do not read earth's code; use the fixtures.
- The plan's "Target modules", "Term structure" and "Fast path" sections set the responsibilities and the numerics: float64; never write into inputs; no absolute epsilons (use the spec's relative tolerances); fixed tie rules; memory O(n·(p + nk)); state each function's complexity in its docstring.
- Keep each module's public surface small and documented, because T11 (`_scan`, `_forward`), T12 (`_core`) and the reference compare against it: plain functions on numpy arrays, no classes unless the spec needs state.
- Tests: unit tests of every rule the module implements, hypothesis tests where a rule has a clean property, and the earth fixtures from T05 (`validation/fixtures/components/`): the `gcv_grid` for `_gcv` (penalty −1 included, and C ≥ N infinite), `knot_candidates` for `_knots`, `pruning_fixed_basis` for `_pruning`, `lm_fit_coefficients` for `_linalg`'s least squares. Tolerances from the plan's tolerance table. A one-token mutation of a rule should fail a test (reviewers check this).
- Coverage of at least 90 percent on the new modules (gate B checks all of `pymars/`).

## What each module holds (from the plan and the spec)

- `_terms.py`: the `dirs` (int8: 0, +1, −1, 2) and `cuts` (float64) arrays, row 0 the intercept; `basis_matrix(X, dirs, cuts)`; degree, variables and labels of a term (TERM rules).
- `_gcv.py`: the effective number of parameters C, the GCV (infinite when C ≥ N, with the τ_N rule), RSq and GRSq, the default penalty and the default term limit (GCV, LIMIT rules), with weights (W-4).
- `_knots.py`: minspan and endspan (SPAN rules, α = 0.05, truncation, `adjust_endspan` for interaction terms), and the candidate knots for one parent and one variable on the weighted empirical distribution of the active cases, with the knot-at-minimum flag for the linear option (KNOT rules, W-4).
- `_linalg.py`: weighted Gram-Schmidt applied twice to append a column; the collinearity test (1 − R² of the centered new column on the existing ones, spec LA rules and thresholds); pivoted-QR least squares (LA-4, the analogue of `lm.fit`); subset downdates for pruning (for example `scipy.linalg.qr_delete`).
- `_pruning.py`: the pruning pass of the spec (PRUNE rules): for one response the leaps-style best subsets among prefixes of a working order, for several responses nested backward elimination on the summed RSS; the intercept never removed; RSS and GCV for each size; the size with the lowest GCV, ties to the smaller model; `nprune`; `pmethod="none"` (the first k forward terms, with the statistics of that model, a departure); the final coefficients (PRUNE-8).

## Report

30 lines or fewer when the pull request is ready: the number and head, the gate results, the fixture comparisons with their largest errors, the mutation checks you ran, and the spec questions you raised.
