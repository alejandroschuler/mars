# Brief: T06 reference implementation (the test oracle)

You write `tests/reference/mars_ref.py`, the slow and literal implementation of the spec that the fast code is tested against, and its sanity tests. Read `briefs/COMMON.md` (same folder) first. Issue: T06, #8. Reviewers: 2 (`spec`, `adversarial`). Model: the strongest.

## Independence (this is the point of the task)

- Write the reference from `docs/algorithm.md` alone (on `origin/main`: spec v1, PR #34 and PR #37). Two implementations only catch each other's errors if they fail independently.
- Do not read `pymars/` (the fast modules that T08 to T12 write in parallel), and do not read the black-box scripts' R code beyond what a spec rule cites as its evidence. Do not import anything from `pymars`.
- Where the spec is unclear, do not guess: write the question in the pull request and in a comment on #44 (spec v2), and use the spec's stated default if it has one.

## Read first

- `docs/algorithm.md`, all of it, including the Departures, Quirks and Open questions.
- `VALIDATION_PLAN.md`: "Tests and validation folders" (the reference's scope), "Correctness against earth", "Fast path" (only to know what the oracle tests will compare).
- `validation/fixtures/` and `validation/harness/` on `origin/main` once T05 merges (#42, #43), to run a few self-checks against earth fixtures. The conformance tests themselves are T07's job.

## What the reference does

- Plain numpy (scipy only for a pivoted QR if you need it). Practical up to n ≤ 300, p ≤ 6, degree ≤ 3; speed does not matter.
- For every candidate, build the explicit hinge columns and solve the least-squares problem by pivoted QR; apply the collinearity rule by an explicit centered regression; in the pruning pass, refit every subset the spec's rules consider.
- Cover the whole spec: frequency weights (N = Σw with math.fsum, τ_N, zero-weight rows dropped), several responses, the y prescaling of EDGE-6, the forward pass (FWD: pairs, single hinges, the linear option and `auto_linpreds`, the 10x limit on knot candidates, the fixed tie order, the slot rules), Fast MARS (FAST), the stopping rules in the spec's order (STOP), the pruning pass (PRUNE, with its K = 1 and K ≥ 2 rules, `nprune`, `pmethod`), the GCV (GCV), degenerate inputs (EDGE) and errors (ERR, where they apply below the estimator level).
- Return the fields of `MarsFit` (spec CORE) as a dict, with the forward record (terms, parent and step, RSS after each step, termination code, and the candidate log with the best and second-best RSS at each step when asked) and the pruning record. The oracle tests compare these fields.
- `tests/reference/test_reference.py`: sanity tests that do not need earth (hand-built cases where the answer is known: a single true knot, a pure linear truth, a constant y, repeated rows against integer weights, invariance of the terms to a positive rescaling of y), and a few self-checks against earth fixtures marked `external` if they need R.

## Size

The reference will be larger than 800 lines. Split it into pull requests that can be reviewed on their own, for example: (1) terms, GCV, spans, knots and the pruning pass on a fixed basis; (2) the forward pass without Fast MARS; (3) Fast MARS, weights and several responses, and the full `fit` entry point.

## Report

After each pull request is ready, and when you stop: 30 lines or fewer, with the pull request numbers and heads, the gate results, the spec questions you raised, and the self-check results against the fixtures.
