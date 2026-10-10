# Brief: #84, pruning accuracy at intermediate covariate means

Read `COMMON.md` (same folder) first; its rules hold for you. Issue: #84 (read its body and comments). Branch: `t84-ref-pruning`. Reviewers: 2 (`spec`, `adversarial`).

## Problem

After #79 (reference forward pass) and #87 (fast pruning centers its columns), the reference's pruning pass in `tests/reference/mars_ref.py` still misses LA-5 at intermediate covariate means: with a covariate shift near 2^23 times the spread, rss_per_size has a relative error of 3.6e-8 against exact arithmetic (4 of 40 fits). So `tests/test_oracle.py` keeps SHIFT_CAP for its `pruning_path` (whole-fit) draws at degree 2 and 3.

The reference is the test oracle and must stay independent of the fast code: write its fix from the spec (`docs/algorithm.md`, LA-5 and the PRUNE rules) and the issue, and do not read `pymars/_pruning.py`.

## Task

1. Reproduce the miss (exact rational RSS per size, for example with `fractions.Fraction`, on a small case at 2^23).
2. Fix the reference's pruning pass so that rss_per_size meets LA-5 for shifts up to at least 2^36 (center or reduce the columns before the projection, as the reference's forward pass does since #79). Check the whole-fit comparison too.
3. Check the fast pruning pass at the same shifts through the oracle comparison. If it misses LA-5, report it to the executor with a reproduction; do not change `pymars/` in this PR.
4. In `tests/test_oracle.py`, lift the `pruning_path` part of SHIFT_CAP (`capped = params["max_degree"] >= 2 and pruning_path`). Another author lifts the other parts for #85 at the same time; keep your change to that line and the comment, and rebase if theirs merges first.
5. Add the reproduction as one case in an existing parametrized test.

Files: `tests/reference/mars_ref.py`, `tests/reference/test_reference.py`, `tests/test_oracle.py`. Run the oracle file at the `thorough` profile before you mark the PR ready. PR body: `Closes #84` if both passes meet LA-5; otherwise `Part of #84`.
