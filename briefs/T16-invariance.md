# Brief: T16 the invariance, weight and edge-case tests

Read `briefs/COMMON.md` (same folder) first, including the scratch-file and mutation-check rules. Issue: T16, #18. Reviewers: 1 (`single`), who covers the spec and the adversarial view. Model: the strongest. Start after T11 stage 3 (#75) has merged; it has.

## Read first

- `VALIDATION_PLAN.md`: "Sample weights", "Edge cases", "Invariance tests", "Determinism", the tolerance table, and the S12 and S16 rows of the fixture table.
- `docs/algorithm.md` on `origin/main`: W (W-1 to W-7), EDGE (above all EDGE-1 and EDGE-6), STOP-7 and LA-5 (the near-tie bands, which are the only allowed exceptions), KNOT and SPAN (the ends of the range and the spacing grid, which break the mirror symmetry), CORE-2 (the default term limit depends on p).
- The public estimators in `pymars/_estimators.py` and `pymars/_core.py`'s `fit_mars`, by their docstrings. The tests use the public API wherever it can pin the rule.

## Scope

Three test files, and nothing else unless a test finds a bug: `tests/test_invariance.py`, `tests/test_weights.py` and `tests/test_edge_cases.py`. If a test finds a bug in `pymars/`, stop, keep the failing test in a work-in-progress commit on your branch, and report it with a minimal reproduction; the executor decides who fixes it. No `xfail` without an issue.

## What to test

- Invariance (hypothesis, no reference needed): row order; column order (labels only, apart from ties); x to a + s·x for s > 0 (same terms, transformed knots, predictions within relative 1e-8), and s < 0 apart from ties and the end rules; y to a + s·y for s ≠ 0 (same terms, coefficients scale by s, the intercept absorbs a); adding a constant or a duplicated column with `max_terms` and both spans fixed. Use the S12 fixture's scales (x times 1e-8 and 1e8, x plus 1e6, y times 1e-9 and 1e9). A draw that stops at a near-tie (STOP-7 or LA-5) counts as an allowed exception: detect it from the forward record (the best and second-best RSS), do not hide it with a looser tolerance, and report the rate.
- Weights: integer weights against repeated rows (the whole fit: terms, knots, pruning path, GCV, coefficients); zero weight against a removed row; unit weights against no weights, bit for bit; several responses and Fast MARS settings included; the UserWarning of W-7 for non-integer weights whose mean is not 1; multiplying all weights by c equals c copies. Against earth, S16 already goes through repeated rows in the conformance tests; do not repeat it.
- Edge cases from the plan's table: a constant column never used; a duplicated column uses the lower index; constant y gives the intercept only; 3, 5, 8 and 12 cases (never more terms than the cases support; with 5 cases no hinge enters); n < p, p = 1, p = 200 (the term limit cap); one outlier in x at 1e6; Inf or NaN gives a ValueError from `validate_data`; `allow_missing=True` raises NotImplementedError with the link to the missing-values issue, if the API has that parameter (check the spec; if it does not, skip this item and say so).
- Determinism within one process: the same fit twice, bit for bit. Cross-process and cross-BLAS runs are for CI and the report, not this task.

## Test rules

- Meaningful tests only (COMMON.md "Tests"). One property test that pins several rules is better than many hand cases. Keep the hypothesis `ci` profile fast (gate B has a 10-minute budget); heavier settings go under the `thorough` profile, which gate C runs.
- Do not duplicate tests that `tests/test_forward.py`, `tests/test_core.py` or `tests/test_estimators.py` already have (for example the bit-for-bit y·2^600 test and the zero-weight tests of T11 stage 3); read them first, and extend them only if the new case belongs there.
- Mutation sample: about 20 mutants in the weight and scale handling of `pymars/` (not in your tests), to show the tests catch real faults.

## Report

When the pull request is ready: 30 lines or fewer, with the number and head, the gate results, the near-tie exception rates, the mutation results, any bug found, and any spec question (raise it on #44).
