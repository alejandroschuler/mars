# Executor state

Work done: no

Updated: 2026-09-26 01:24 PDT, by executor session `local_a59469db-c3f1-480c-81c4-a4c7beae39f4`.

This file and LOG.md are the executor journal (branch `executor`; only the executor pushes). A new executor rebuilds its state from this file, LOG.md, the issues and pull requests of `alejandroschuler/mars`, ListAgents and the job manifests.

## Paths

- `<main>` = `/Users/aschuler/Documents/research/projects/pymars`.
- `<S>` = this executor's session worktree, `<main>/.claude/worktrees/competent-poincare-5a4d49`. The desktop app lets the Write and Edit tools change files only inside the session worktree, so this executor keeps the journal worktree at `<S>/.worktrees/journal` and task worktrees at `<S>/.worktrees/<branch>`. An executor that runs in `<main>` itself (for example from the watchdog) uses `<main>/.worktrees/` as the plan says.
- Lock, heartbeat, caffeinate PID, gate logs: `<main>/.git/pymars-executor/` (shared by all worktrees).
- Briefs: `briefs/` in this journal.

## Phase

P0 done (recovery drill passed). P1 in progress since 18:45 PDT 2026-09-25.

Board: T00 #2 (closed), T01 #3, T02 #4, T03 #5, T04 #6, T05 #7, T06 #8, T07 #9, T08 #10, T09 #11, T10 #12, T11 #13, T12 #14, T13 #15, T14 #16, T15 #17, T16 #18, T17 #19, T18 #20, T19 #21, T20 #22, T21 #23, T22 #24, T23 #25, T24 #26; later: missing values #27, negative minspan #28, newvar.penalty #29, linpreds #30, allowed #31, pmethod and nfold #32, evimp #33. `board/numbers.json` has the map.

main: dbe5794 (PR #35, T03 part 1) on top of 31e6c13 (PR #1, bootstrap). CI runs on the fork.

Open pull requests (21:45):
- #34 T01 spec part 1, head d556e28; round 3 dual review running (workflow run wf_95bda7f4-7a6). Merge needs APPROVE from roles `spec` and `adversarial` on the head: `bash <main>/dev/tools/merge_pr.sh --session <id> 34 spec adversarial`. Before merging it, retarget #37 to main (`gh pr edit 37 --repo alejandroschuler/mars --base main`).
- #37 T01 spec part 2, head 0c4121d, stacked on #34; ready; its dual review starts after #34 merges (review `git diff d556e28..HEAD`). Rebase onto main after #34 merges.
- #36 T02 part 1, head fd71f83; #38 T02 part 2, head 3173b3e (base already main); re-review by rev-t02-harness (role `single`) running. Merge #36 first, then rebase #38 onto main, quick re-approval by range-diff, merge #38.
- #40 T03 part 2, head d5d4c54; re-review by rev-pr35-sims1 (role `single`) running. After it merges, start T04 (brief `briefs/T04-legacy-runs.md`).

## Done in P0

- Lock `<main>/.git/pymars-executor/lock` taken; heartbeat file in it.
- `gh repo set-default alejandroschuler/mars`; upstream push URL is `DISABLED`; `.worktrees/` in `.git/info/exclude`.
- `validation-plan` pushed to the fork; tag `legacy-1.0.4-head` (d68b54a) pushed.
- `caffeinate -dimsu` detached, PID in `<main>/.git/pymars-executor/caffeinate.pid`.
- Planning prototypes are on `validation-plan` in `validation/legacy/`; a copy is in `<main>/.git/pymars-executor/legacy-prototypes/exp/`.
- Text of Milborrow's notes saved for local reading at `<main>/.git/pymars-executor/ref/earth-notes.txt` (never commit or quote it).
- Watchdog task `pymars-executor-watchdog` created (cron `17 * * * *`, the app shows 25 past the hour; notifyOnCompletion false). Its folder and mode are not checked yet: `run_scheduled_task` was denied by the auto-mode classifier, so the user was asked to click Run now once.

## Next actions

1. Merge #40 when approved; start T04 (sonnet author, runner worktree in `<main>/.worktrees/runner-legacy`, full legacy run at the caps with 6 workers; pilot analysis later).
2. Merge #34 when both roles approve (retarget #37 to main first); then the dual review of #37.
3. Merge #36 and #38 when approved; then T05 (fixtures; its brief is not written yet) and T18 (legacy description; brief not written yet).
4. After spec part 1 merges: T08 (_terms, _gcv, _knots), T09 (_linalg), T10 (_pruning) with briefs that cite the spec rule IDs (TERM, GCV, LIMIT, SPAN, KNOT, LA, PRUNE, CORE, W). T06 (the reference) needs the whole spec and T05.
5. Pacing: weekly 69 % at 21:38 (reset 2026-09-27 17:00 PDT). At 85 %: authors checkpoint and stop. At 90 %: only finish open pull requests. The detached legacy jobs need no Claude usage.
6. Follow-ups: merge_pr.sh should refuse while another open PR uses the head branch as its base (GitHub closed #39 when #35's branch was deleted). T16 must exclude shift tests where a shifted covariate is a linear factor in a term of degree 2 or more. The plan's edge-case row that calls earth scale invariant is wrong (bb14.4).

## Running agents

- `t01-spec` (opus): done with spec v1 parts 1 and 2; waiting for review findings (SendMessage to it).
- `t02-harness` (sonnet): waiting for the re-review of #36 and #38.
- `t03-sims` (sonnet): waiting for the re-review of #40.
- `rev-t02-harness` (opus): re-reviewing #36 and #38.
- `rev-pr35-sims1` (opus): re-reviewing #40 (it also reviewed #35).
- Workflow run wf_95bda7f4-7a6: round 3 of #34 (spec + adversarial). The reusable script is `tools/dual_review.js` in this journal (args: pr, title, head, brief, roles, focus).

## Running jobs

None.

## Usage and resets

- 2026-09-25 16:19 PDT: 5-hour 19 % (resets 2026-09-25 19:59 PDT); weekly all models 45 % (resets 2026-09-27 16:59 PDT); weekly Fable 0 %.

## Guard notes

- The `dcg` hook blocks `rm -rf`, `git reset --hard`, `git clean`, `git checkout -- <file>`. It allows `git rm -r`, `git push --force-with-lease`, `gh pr merge --squash --delete-branch`, `git worktree remove`, `git rebase`, `git branch -d`, `mv`, `kill`. Check with `dcg test "<command>"`.
- The app's worktree guard blocks Write and Edit outside the session worktree, also after request_directory for `<main>`. change_directory is refused for a worktree session.
- The auto-mode classifier denied `run_scheduled_task`.
