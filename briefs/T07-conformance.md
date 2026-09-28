# Brief: T07 conformance tests and triage (tests/test_conformance.py, validation/DIFFERENCES.md)

Read `briefs/COMMON.md` (same folder) first, including the scratch-file and long-job rules. Issue: T07, #9. Reviewer: 1 (`single`). Model: the strongest. Start after PR #49 (the reference's `fit_mars`) has merged.

The conformance suite checks the new code against earth. It runs the reference implementation now, and the fast code (`pymars._core.fit_mars`) as soon as it exists end to end. Every later change to the fitting code must pass it.

## Read first

- `VALIDATION_PLAN.md` on `origin/main`: "Correctness against earth", all of it (above all "Comparison modes" for the new code, "Test datasets", "What is compared, and the tolerances", "Ties", "Triage of differences", "Sample weights", "Categorical inputs", "Edge cases"), and "Behavior target".
- `docs/algorithm.md`: all of it, above all CORE-3 and CORE-5 (the record fields), the Departures, the Quirks and the Open questions (OQ-2 asks you to fix the threshold for a pruning near-tie; OQ-6 and the FWD-11 row tell you how to label the hidden term).
- `validation/README.md`, `validation/harness/compare.py`, `validation/harness/new_adapter.py`, `validation/harness/names_map.py`, and the fixtures in `validation/fixtures/`.

## Rules for this task

- You test the reference and the fast code. You may read both, but you change neither: a `bug` in the reference goes to the executor (in your report, and as a comment on #8), and a `bug` in the fast code blocks that code's pull request.
- Clean room as in COMMON.md. The fixtures are earth's side; if you need a new earth run, use the harness as a black box.

## What to build

1. `validation/harness/new_adapter.py`: replace its assumed record shape with the fields of CORE-3 and CORE-5 (the reference's dict now, `MarsFit.to_dict()` later), and update its tests.
2. `tests/test_conformance.py`: for every fixture dataset and mode, the fit against earth by the plan's tolerance table: the forward steps up to the first near-tie, the RSS path, the pruning record on the same forward basis (up to the first pruning near-tie, OQ-2), the selected terms, the coefficients, GCV, RSq, GRSq, the fitted values and the predictions, with the κ(B) limits. Parametrize by implementation so that adding one is one line: `reference` now, and `fast` when T12's `fit_mars` has merged (add it in this pull request if T12 is on `main` first; otherwise T12 adds it). Mark as `slow` the cases that the reference cannot fit in a few seconds.
3. Encode the spec's departures as expected differences, each tied to its rule (for example the degenerate fits of EDGE-1 and GCV-7, the frequency weights through repeated rows, the earth zeroing of small RSS values), so that the tests fail on a new, unlabeled difference and pass on a labeled one. Keep the list in one machine-readable file (for example `validation/differences.json`) that the test and `DIFFERENCES.md` share.
4. `validation/DIFFERENCES.md`: every difference with the dataset, the step, both choices, both candidate RSS values, earth's flags where the trace has them, and one of the labels `rule`, `bug`, `quirk`, `tie`, `numeric`. On S15, give the rates: the share of fits that agree, the step of the first divergence and its cause, and the share that stop at a near-tie (the plan asks for a review above 5 percent).
5. Spec questions that the triage raises go on #44 (spec v2), one comment each, with the evidence.

## Report

30 lines or fewer when the pull request is ready: the number and head, the gate results, the counts by label and by dataset, any `bug` with its evidence, the S15 rates, and the questions raised on #44.
