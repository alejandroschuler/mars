# Brief: finish #113 and #115 with the rest of week 3's usage

Read `COMMON.md` (same folder) first; its rules hold, except the paths. This executor is session `local_a59469db-c3f1-480c-81c4-a4c7beae39f4`, which runs in `<S>` = `/Users/aschuler/Documents/research/projects/pymars/.claude/worktrees/competent-poincare-5a4d49`. The desktop app lets the Write and Edit tools change files only inside `<S>`, so make your worktree with `PYMARS_WORKTREES=<S>/.worktrees bash /Users/aschuler/Documents/research/projects/pymars/dev/tools/new_worktree.sh <branch>` (read the script first). Gate logs stay in `<main>/.git/pymars-executor/gates/`.

The user asked to finish these two issues with what is left of this week's usage, and little is left. Work lean: no exploratory campaigns, a mutation sample of at most about 30, and the fewest tests that pin the rules. Commit and push after each step, so that a stop loses nothing. If the executor stops you, push a work-in-progress commit and report.

## S113: spec decisions (issue #113, branch `s113-copy-scope`)

Files: `docs/algorithm.md`; `tests/test_oracle.py` only if a draw that the spec now scopes out can fail the oracle.

Read #113 with its comments and the review comments that it links. Decide each item, with a short reason, in the style of spec v2's changes:

1. The scope of LA-5 for (a) a product that is linear on the cases, b·(x1 − t)+ with b = 0 wherever x1 < t, at a large covariate mean, and (b) a near-affine copy of a large-mean covariate, such as x1 = x0·(1 + 2^-40), which is not bitwise equal and not a power-of-2 multiple. Prefer to scope them out of LA-5 with a stated bound (a Conventions entry, or an open question next to OQ-9), unless one simple rule covers them in both implementations.
2. Which copies share one symbol in LA-4 and EDGE-7: bitwise-equal columns, bitwise-equal x − m columns, power-of-2 multiples (the fast code makes them one symbol, which keeps EDGE-7's bit-for-bit property), and affine copies such as x3 = 3·x2 + 1 (LA-4 sees only formal relations among atoms). State the rule so that the reference and the fast code can both follow it exactly, and record what stays order-dependent.
3. A power-of-2 symbol must not merge two columns through underflow to a subnormal: state that the relation must hold exactly (an exact ldexp check).

As soon as items 2 and 3 are settled, post them on #113 in one comment: the R115 author needs them. Do not change `pymars/` or `tests/reference/`. The PR body lists every changed rule and the modules that must follow it. Reviewer: 1 (`single`); the executor starts it.

## R115: reference fixes (issue #115, branch `r115-ref-copies`)

Files: `tests/reference/mars_ref.py`, `tests/reference/test_reference.py`; `tests/test_oracle.py` only to add copy kinds to the `ties` draws. Write from the spec only and never read `pymars/` (running the tests is fine). The reference is the oracle.

Read #115 with its comments and the review comments that it links (#110's review has the generator; #112's recheck has the 22 cases). Then:

1. The cases from #112's recheck: LA-4's new part must ignore weight-0 rows (W-3); a part that is exactly 0 counts as 0 even when its float evaluation is not; and `_exact_rest_sq` is wrong at seed 3002 (an independent exact computation gives 0).
2. The 2 of 300 wrong forward cases: a power-of-2 copy at another shift (6 rows, degree 2) and a one-ulp copy at −2^30. Follow S113's symbol rule (its comment on #113, or its PR). If it is not settled when you reach this item, do item 3 first.
3. Triage the 7 whole-fit pruning differences (seeds 37, 100, 104, 154, 213, 218, 273 of #110's generator) against exact rational arithmetic. Each is a reference bug (fix it), a fast-code bug (report the seed and the exact values to the executor; do not read or change `pymars/`), or PRUNE-8's stated limit (say so in the PR body).

Then add the copy kinds of item 2 to the oracle's `ties` draws if the fast code and the fixed reference agree on them; if they do not, report the seed. Reviewer: 1 (`adversarial`); the executor starts it. Run gate B and gate C (`bash dev/gate_c.sh`) on the head.
