# Brief: T11 the fast forward pass (pymars/_scan.py and pymars/_forward.py)

Read `briefs/COMMON.md` (same folder) first, including its mutation-check rule. Issue: T11, #13. Reviewers: 2 per pull request (`spec`, `adversarial`). Model: the strongest. This is the largest and most important module of the fast code.

## Read first

- `docs/algorithm.md` on `origin/main` (spec v1): the FWD, STOP and FAST rules above all, and KNOT, SPAN, LA (LA-3 collinearity, LA-5 accuracy and near-tie bands, LA-7 the pair-search rule with pymars's scale-invariant departure), W, RESP, EDGE-6, CORE-3 (the forward record and the candidate log with the best and the second-best RSS) and CORE-4 (termination codes).
- `VALIDATION_PLAN.md`: "Speed and scaling" ("Cost model", "Fast MARS status", "Fast path" with the derivation of the suffix sums, eq. 52), "Target modules", "Term structure".
- The merged modules you build on, by their docstrings: `pymars/_terms.py`, `pymars/_gcv.py`, `pymars/_knots.py`, `pymars/_linalg.py` (and `pymars/_pruning.py` if PR #54 has merged; the forward pass does not need it).

## Independence

Do not read `tests/reference/` (the oracle). You may import and call the reference as a black box in local checks and in tests (`tests/reference/mars_ref.py` on main has part 1 now; the full `fit_mars` comes with PR #49), but do not read its source. Never read earth's code; use the fixtures in `validation/fixtures/`.

## What to build

- `_scan.py`: for one parent and one variable, the RSS reduction of the pair, of the single hinge and of the linear term at every candidate knot, from suffix sums over the sorted variable, with weights and several responses (the plan's "Fast path" derivation). Cost O(n·r) per parent and variable. The winner is then built again explicitly, checked, tested by the collinearity rule (LA-3) and orthogonalized twice (`_linalg`).
- `_forward.py`: the forward driver: eligible parents and variables (FWD rules), the 10x limit on knot candidates only (FWD-4), pairs, single hinges, the linear option (`auto_linpreds`), the collinearity tolerance with its counter, the pymars pair rule (LA-7), the fixed tie order (FWD-5), the slots and the term-limit bookkeeping, the stopping rules in the spec's order (STOP-1, 3, 4, 2, 5, with the STOP-7 near-tie bands), Fast MARS (FAST: the queue, `fast_k` below 3 acting as 3, `fast_beta` ageing, parents by slot), the y prescaling (EDGE-6), weights and several responses, and the forward record with the optional candidate log.
- Memory O(n·(p + nk)); never write into X, Y or w; no absolute epsilons beyond those the spec states; state the complexity of each function.

## Stages (one pull request each, each tested before the next starts)

1. Degree 1, one response, unit weights, `fast_k = 0`: the scan and the driver for the intercept parent, with the pair, single-hinge and linear-option rules, the collinearity rule, the stopping rules and the forward record. Adjust the split if a rule cannot be tested alone, and say so.
2. Interactions (degree 2 and 3, parents other than the intercept, the Adjust.endspan span, the slot rules).
3. Fast MARS, weights, several responses.

## Tests

- Against the earth fixtures on main in the matched and earth-compatible modes where the spec says they apply (forward steps: parent, variable, direction, knot, exact up to the first near-tie; the RSS path within relative 1e-8; the termination code), and against the reference's `fit_mars` as a black box once PR #49 has merged (before that, against the reference branch's head as a black box, or earth fixtures only).
- Unit tests of the scan against brute force (explicit hinge columns and a least-squares solve) on random data, with weights and several responses in stage 3.
- Mutation checks with python -B and a fresh copy per mutant.

## Report

After each stage's pull request is ready: 30 lines or fewer, with the pull request number and head, the gate results, the fixture and reference comparisons with the largest errors and the first-divergence counts, the mutation results, and any spec question (raise it on #44).
