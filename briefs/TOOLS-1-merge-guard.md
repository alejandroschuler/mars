# Brief: TOOLS-1, a stacked-branch guard in merge_pr.sh, and the executor's process decisions

Read `briefs/COMMON.md` (same folder) first. No task issue: this is executor tooling; the pull request body says "Refs #2" (T00, the bootstrap that made the script). Reviewer: 1 (`single`). Files: `dev/tools/merge_pr.sh`, `dev/tools/test_merge_pr.sh` and `dev/DECISIONS.md`, nothing else. Branch: `tools-merge-guard` from `origin/main`.

## Why

`merge_pr.sh` merges with `gh pr merge --squash --delete-branch`. When another open pull request uses the merged branch as its base, GitHub closes that pull request as soon as the branch is deleted (this happened to #39). The executor now retargets such pull requests to `main` by hand before a merge; the script should refuse the merge until that is done.

## What to change

1. `merge_pr.sh`: one more check. List the open pull requests whose base is the head branch (`gh pr list --repo "$REPO" --state open --base "$head_ref" --json number --jq ...`). None: a PASS line. Some: a FAIL line that names them and says to retarget them to `main` first, because `--delete-branch` would make GitHub close them. Keep the script's style: one `pass` or `fail` line per check, no early exit, and a failed `gh` call counts as a failure. Update the header comment that lists the checks.
2. `test_merge_pr.sh`: the fake `gh` must also answer `pr list` (for example from a second JSON file). Add tests: no stacked pull request passes; one stacked pull request fails and the output names its number; a failed `pr list` call fails the check. All existing tests must still pass.
3. `dev/DECISIONS.md`: a new section "Executor process" with three short entries, in the file's style:
   - a pure rebase gets a re-approval from a small checker agent, which verifies that `git range-diff` shows every commit as `=`, that gate B passed on the new head and that CI is green, and then posts for every required role a verdict line for the new head that cites the approvals of the old head;
   - a fix that only touches tests, only for CI, of about 20 lines or less, gets a narrow re-approval from a checker for every role in the same way;
   - before a stacked lower part merges, the executor retargets the next part to `main`; `merge_pr.sh` refuses until it has.

## Gates and report

Gate A on each commit, gate B before ready (it runs `test_merge_pr.sh`), CI green. Read `dev/gate_a.sh` and `dev/gate_b.sh` before you run them. Report in 15 lines or fewer: the pull request number and head, the gate results, and the new test names.
