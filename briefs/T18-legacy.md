# Brief: T18 the description of the legacy code (validation/DIFFERENCES_legacy.md)

Read `briefs/COMMON.md` (same folder) first, including the scratch-file rule and the long-job rule. Issue: T18, #20. Reviewer: 1 (`single`). Model: the strongest.

The legacy code is mars-earth 1.0.4 (the PyPI wheel) and HEAD `d68b54a` (the tag `legacy-1.0.4-head`). The plan describes it and does not fix it. Your task is the P1 exit item: every difference between the legacy code and earth on S01 to S20 has a label, in `validation/DIFFERENCES_legacy.md`.

## Read first

- `VALIDATION_PLAN.md` on `origin/main`: "Preliminary findings" (F1 to F16, with "Fit time against earth"), "Code reading against the references" (the pymars column describes the legacy code), and in "Correctness against earth": "Harness", "Comparison modes" (the legacy matched mode: hinges only, `fast.k = 0`, `thresh = 0`, `minspan = 1`, earth `endspan = 1` against legacy `endspan = 0`, `Adjust.endspan = 1`, the legacy penalty equal to half the earth penalty, and the list of expected exceptions), "Test datasets", "What is compared, and the tolerances", "Ties" and "Triage of differences".
- `validation/README.md`, `validation/harness/legacy_adapter.py` (`fit_legacy`), `validation/harness/compare.py`, `validation/harness/driver.py`, `validation/legacy/README.md`, and the fixture files in `validation/fixtures/`.

## Rules for this task

- Clean room as in COMMON.md. You may read the legacy code (it is the subject): the installed wheel in `.venv-legacy`, and the tag `legacy-1.0.4-head` through `git show`. You may run earth as a black box through the harness, with trace levels 7 to 9 as fixture data, to get earth's side for settings that the fixtures lack. Never read earth's C code or print R function bodies.
- Make the venv with `validation/legacy/make_venv.sh` in your worktree (read it first; it installs from PyPI into a uv venv).
- The legacy code is slow (F10: about 130 s for one degree-2 fit at 1,000 cases). A run of all datasets that takes more than about 10 minutes goes to a detached job with a PID file and a log, under `nice -n 15`, one core; results per case, written atomically.

## What to do

1. For every dataset in S01 to S20 and the settings that its purpose needs (degree 1, 2 and 3 where the case says), fit the legacy code in the legacy matched mode and at its defaults, and get earth's side from the fixtures, or from new black-box runs where the fixtures lack the legacy matched settings.
2. Compare by the plan's tolerance table and find the first divergence of each fit (parent, variable, direction and knot; the RSS path; the pruning; the selected terms; the coefficients, GCV, R², GRSq and predictions where the terms agree). Use the near-tie rule of "Ties" with both programs' best and second-best candidate RSS where the logs give them.
3. Label each difference `rule`, `bug`, `quirk`, `tie` or `numeric` ("Triage of differences"), with the finding ID (F1 to F16) where one applies, and one sentence of evidence (the step, both choices, both RSS values, earth's flags where the trace has them). A difference that fits no finding gets a new ID (F17 onward), stated with the same kind of evidence as the plan's findings table.
4. Run HEAD as well on S01, S04 and S15 (PYTHONPATH to a worktree of the tag, inside `.venv-legacy`, if its Python code runs without the Rust part), and say whether HEAD and the wheel differ.

## Deliverables

- `validation/DIFFERENCES_legacy.md`: a summary table (dataset, mode, degree, the first divergent step, the label, the finding ID), counts by label, a short section for each finding that the runs confirm or refute, the new findings, and the command that makes the document's numbers again.
- The script that does the runs and the comparison (for example `validation/legacy/conformance_legacy.py`), with a small JSON output next to it; keep the committed files small.
- Tests for any comparison logic that is more than a call to `compare.py`, and one `external` test that runs the script on one small dataset.

## Report

30 lines or fewer when the pull request is ready: the number and head, the gate results, the counts by label, the confirmed and refuted findings, any new finding, and whether HEAD and the wheel differ.
