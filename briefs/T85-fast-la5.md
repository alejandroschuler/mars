# Brief: #85, LA-5 in the fast forward pass at large covariate means

Read `COMMON.md` (same folder) first; its rules hold for you. Issue: #85 (read its body and all comments: four reproductions). Branch: `t85-fast-la5`. Reviewers: 2 (`spec`, `adversarial`).

## Problem

The fast forward pass (`pymars/_forward.py`, `pymars/_scan.py`, maybe `pymars/_linalg.py`) misses the spec's LA-5 accuracy contract (`docs/algorithm.md`) when a candidate or a parent has a linear factor of a covariate with a large mean: products of linear factors at degree 2 and 3, exact duplicate covariates at a large mean, and the "shifted" kind at degree 3. The reference (`tests/reference/mars_ref.py`, after #79) meets LA-5 there with an exact change of basis: it works with x - m, m a middle data value, as the hinge columns already do. You read the spec and the fast code; you may read the reference to see what results it gives, but implement in the fast code from the spec and the issue.

## Task

1. Reproduce each case in #85 as a failing check (the fast RSS against the reference or an exact rational RSS).
2. Fix the fast code so that it meets LA-5 in those cases: build or reduce the columns of terms with a linear factor from x - m before the projection, or an equal method. Keep the cost model of the fast path (no new O(n·M^2) work per candidate) and say the complexity in the PR.
3. In `tests/test_oracle.py`, lift the parts of SHIFT_CAP that #85 names: the duplicate covariate (`ties` kind), the `scaled` kind at degree 2 and 3, and the `shifted` kind at degree 3. Keep the `pruning_path` cap (that is #84, another author). Update the comment above SHIFT_CAP.
4. Add the issue's reproductions as few oracle cases (one parametrized test), not one test per case.

Files: `pymars/_forward.py`, `pymars/_scan.py`, `pymars/_linalg.py` (only if needed), `tests/test_oracle.py`, `tests/test_forward.py` (only if needed). Run the oracle file at the `thorough` hypothesis profile before you mark the PR ready (gate C runs it; time it). PR body: `Closes #85`.

If the fix needs a spec change, stop and report to the executor with the exact question; do not change `docs/algorithm.md`.
