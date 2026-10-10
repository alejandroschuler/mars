# Brief: #84 part 2, LA-5 in the fast pruning pass at large covariate means

Read `COMMON.md` (same folder) first; its rules hold for you. Issue: #84 (read its body and comments). Branch: `t84-fast-pruning`. Reviewers: 2 (`spec`, `adversarial`).

## Problem

PR #94 fixes the reference's pruning pass (tests/reference/mars_ref.py). The fast pruning pass (`pymars/_pruning.py`) still misses LA-5 (`docs/algorithm.md`): its rss_per_size, against exact rational arithmetic on its own kept terms, is off by 1.1e-8 relative at a covariate mean of 2^23 with spread 8 (also 2^26 with spread 64, the same ratio 2^20), and by 2.2e-9 at 2^21 with spread 4. #87 centers the columns after the intercept, which is not enough when a linear factor has mean/spread near 2^20. A reproduction script: `/private/tmp/claude-503/-Users-aschuler-Documents-research-projects-pymars/21fc279e-e671-49aa-b4a9-aab139f89607/scratchpad/t84-ref-pruning/repro5.py` (run from a worktree with `. dev/env.sh && uv run python <script> 8388608 8 30`; expect `seed 11 1.108e-08`). Read it before you run it.

## Task

1. Turn the reproduction into a failing test: one case in an existing parametrized test in `tests/test_pruning.py` (exact RSS per size with `fractions.Fraction`, or the oracle against the reference once #94 merges).
2. Fix `pymars/_pruning.py` so that rss_per_size meets LA-5 for covariate shift to spread ratios up to 2^36: reduce the columns of terms with linear factors from x - m (m a middle data value) before the projection, or an equal method, from the spec. Do not read `tests/reference/mars_ref.py`'s fix; work from the spec and this brief. Keep the cost model (state the complexity) and check that 121 fixtures and the oracle file still pass.
3. If #94 has merged when you are done, rebase and check the oracle's whole-fit comparison at the `thorough` profile.

Files: `pymars/_pruning.py`, `pymars/_linalg.py` only if needed (another author may change it for #85; keep any change small), `tests/test_pruning.py`. PR body: `Closes #84` (with #94 merged), else `Part of #84`.
