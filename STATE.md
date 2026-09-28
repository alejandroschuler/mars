# Executor state

Work done: no

Updated: 2026-09-27 19:18 PDT, by executor session `local_a59469db-c3f1-480c-81c4-a4c7beae39f4`.

This file and LOG.md are the executor journal (branch `executor`; only the executor pushes). A new executor rebuilds its state from this file, LOG.md, the issues and pull requests of `alejandroschuler/mars`, ListAgents and the job manifests.

## Paths

- `<main>` = `/Users/aschuler/Documents/research/projects/pymars`.
- `<S>` = this executor's session worktree, `<main>/.claude/worktrees/competent-poincare-5a4d49`. The desktop app lets the Write and Edit tools change files only inside the session worktree, so this executor keeps the journal worktree at `<S>/.worktrees/journal` and task worktrees at `<S>/.worktrees/<branch>`. An executor that runs in `<main>` itself (for example from the watchdog) uses `<main>/.worktrees/` as the plan says.
- Lock, heartbeat, caffeinate PID, gate logs: `<main>/.git/pymars-executor/` (shared by all worktrees).
- Briefs: `briefs/` in this journal.

## Phase

RUNNING (week 2). Weekly 11 % at 19:05 Sunday 2026-09-27; reset 2026-10-04 17:00 PDT. Daily allowance: cumulative 14 % per day (14 % by Monday 17:00, 28 % by Tuesday 17:00, ...). The user has not set a cap for week 2; default is the plan's rule (projected use at the reset at most 95 %). At most 2 to 3 agents at a time; sonnet for mechanical work (integrator, checkers).

main: 955b44c (#46 T06 part 1, the reference's terms to linear algebra), eabac8c (#52 _terms), f9cfc5a (#45 _linalg), 85ac0e6 (#51 _gcv and _knots), 36ac4ca (#41 T04 results), d6cac73 (#43), 288d93c (#42), a8dd024 (#37), b0d642d (#34), d4f74f4 (#38), 8760cee (#40), dbe5794 (#35), 31e6c13 (#1).
Done: T00 #2, T01 v1 #3 (v2 is #44), T02 #4, T03 #5, T04 #6, T05 #7, T08 #10, T09 #11.

Open pull requests:
- #46 merged as 955b44c (T06 part 1). #47 retargeted to main.
- #47 part 2 pruning (4bf9e43), #48 part 3 forward pass (804ea69), #49 part 4 fit_mars (37df4c3, Closes #8): stacked; each needs a dual review after the part below merges and `t06-reference` rebases it.
Process for merges since main moves: the `integrator` agent (sonnet) does pure rebases; a checker (`rebase-check-45`, sonnet) verifies range-diff and posts both roles' re-approvals; merge_pr.sh then merges. CI-only test fixes of about 20 lines or less get a narrow re-approval the same way.

## Done in P0

- Lock `<main>/.git/pymars-executor/lock` taken; heartbeat file in it.
- `gh repo set-default alejandroschuler/mars`; upstream push URL is `DISABLED`; `.worktrees/` in `.git/info/exclude`.
- `validation-plan` pushed to the fork; tag `legacy-1.0.4-head` (d68b54a) pushed.
- `caffeinate -dimsu` detached, PID in `<main>/.git/pymars-executor/caffeinate.pid`.
- Planning prototypes are on `validation-plan` in `validation/legacy/`; a copy is in `<main>/.git/pymars-executor/legacy-prototypes/exp/`.
- Text of Milborrow's notes saved for local reading at `<main>/.git/pymars-executor/ref/earth-notes.txt` (never commit or quote it).
- Watchdog task `pymars-executor-watchdog` created (cron `17 * * * *`, the app shows 25 past the hour; notifyOnCompletion false). Its folder and mode are not checked yet: `run_scheduled_task` was denied by the auto-mode classifier, so the user was asked to click Run now once.

## Next actions

1. #47 (T06 part 2, pruning): `t06-reference` rebases it onto main (`git rebase --onto origin/main 75a8f5539309d2571f7ac39af86c1378919bc670 t06-reference-part2`), then its dual review (tools/dual_review.js) when the daily allowance allows (after Monday 17:00 if the weekly is near 14 %); then #48 and #49 the same way; before merging a part, retarget the next part to main.
2. T10 `_pruning.py`: `t10-pruning` started 19:08 Sunday; its dual review follows its report.
3. T11 `_scan.py`, `_forward.py` (opus, staged: degree 1; interactions, linear option, collinearity; Fast MARS, weights, several responses): brief to write; needs T08, T09 (merged); gated on the oracle tests, so it needs the reference (#46 to #49) merged for its oracle tests (T15), or it tests against the reference branch.
4. T07 conformance (brief to write) after #49 merges. T12 core after T10. T13 estimators after T12. T18 legacy description (brief to write; sonnet).
5. Follow-ups: T05 notes PR; tooling PR (merge_pr.sh: refuse while another open PR uses the head branch as its base; accept rebase-only re-approvals); T08 non-blocking notes (child_term validation; bool for penalty and adjust_endspan per CORE-2; the tie-order knife edge in _knots run sums; tss ZeroDivisionError; OverflowError at extreme values; SPAN-4 test at bb06.12's values); T09 notes (the scale range must include weights and products for LA-7; the LA-4 <= fixed-point test). All go on #44 (spec v2) or later PRs.
6. Housekeeping: the unused branch origin/t08-terms-gcv and worktree <S>/.worktrees/t08-terms-gcv-knots; about 100 MB of reviewer mutation copies in this session's scratchpad (rev46spec_mut, rev46spec_base) that the guard blocked from deletion (for the user to remove).

## Running agents

- `t06-reference` (opus): adding the #46 large-N boundary test.
- `t10-pruning` (opus): T10 `_pruning.py`, issue #12, started 19:08 Sunday.
- `rev46-r3-adv`, `rev46-r3-spec` (sonnet): #46 round-3 reviewers, resumable for the re-check.
- `integrator` (sonnet): pure rebases; `rebase-check-45` (sonnet): rebase-only re-approvals.
- Resumable authors: `t08-terms-gcv-knots`, `t09-linalg`, `t05-fixtures`, `t01-spec` (spec v2).

## Running jobs

- None. The legacy full run (T04) finished at 16:13 PDT Saturday 2026-09-26 (ALL BLOCKS DONE); results in `<main>/.worktrees/runner-legacy/validation/runs/legacy_full` (49,441 files), logs `driver.log` (6 workers) and `driver_n4.log` (4 workers). Keep the runner worktree until T04 step 6 has collected the results into `validation/sims/results/legacy/`.

## Usage and resets

- 2026-09-25 16:19 PDT: 5-hour 19 % (resets 2026-09-25 19:59 PDT); weekly all models 45 % (resets 2026-09-27 16:59 PDT); weekly Fable 0 %.

## Guard notes

- The `dcg` hook blocks `rm -rf`, `git reset --hard`, `git clean`, `git checkout -- <file>`. It allows `git rm -r`, `git push --force-with-lease`, `gh pr merge --squash --delete-branch`, `git worktree remove`, `git rebase`, `git branch -d`, `mv`, `kill`. Check with `dcg test "<command>"`.
- The app's worktree guard blocks Write and Edit outside the session worktree, also after request_directory for `<main>`. change_directory is refused for a worktree session.
- The auto-mode classifier denied `run_scheduled_task`.
