# Brief: T01 v2, spec v2 (#44)

Read `COMMON.md` and `T01-spec-writer.md` (same folder) first; their rules hold for you, including the clean room and the form of the spec. Issue: #44 (read its body and every comment: the questions that implementation, conformance (T07) and the oracle tests collected). Branch: `t01-spec-v2`. Reviewers: 2 (`spec`, `adversarial`).

## Task

Revise `docs/algorithm.md` on `origin/main` to settle each question on #44. The current list in the executor's journal: the exact-fit band of #83 (write FWD-5's band, EXACT_FIT = 1e-14 of RSS_s, as a definition with its rounding bound), W-1 at near-ties, the GLM-4 threshold, the LA-5 conditioning clause (#85 and #84 are being fixed in code now, so LA-5 should stay a plain accuracy contract unless you find it cannot be met), an outlier at +1e6 and LA-3, the mirror and LA-3, weight sums, the scale of X, one term reachable from two parents (#71), PRUNE-8 RSS against the LA-1 RSS at large means, LA-5 for a hinge that fits one far case, and the four questions from T07 (#66). Check the comments for more.

For each question: decide from the sources (a black-box earth run where earth's behavior is the question, `validation/blackbox/bbNN_*`), or record a pymars choice with its reason and add it to "Departures from earth" if it departs. Keep rule IDs stable; add new IDs for new rules; mark changed rules. Close "Open questions" items you settle.

At the end of the PR body, list each changed rule and the modules it affects (`pymars/_*.py`, `tests/reference/mars_ref.py`), so the executor can open follow-up tasks for both implementations. Do not change code.

Files: `docs/algorithm.md`, `validation/blackbox/` (new scripts and outputs, README). If the diff passes about 800 lines, split it into two PRs. PR body: `Closes #44` on the last PR.
