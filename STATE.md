# Executor state

Work done: no

Updated: 2026-09-27 23:32 PDT, by executor session `local_a59469db-c3f1-480c-81c4-a4c7beae39f4`.

This file and LOG.md are the executor journal (branch `executor`; only the executor pushes). A new executor rebuilds its state from this file, LOG.md, the issues and pull requests of `alejandroschuler/mars`, ListAgents and the job manifests.

## Paths

- `<main>` = `/Users/aschuler/Documents/research/projects/pymars`.
- `<S>` = this executor's session worktree, `<main>/.claude/worktrees/competent-poincare-5a4d49`. The desktop app lets the Write and Edit tools change files only inside the session worktree, so this executor keeps the journal worktree at `<S>/.worktrees/journal` and task worktrees at `<S>/.worktrees/<branch>`. An executor that runs in `<main>` itself (for example from the watchdog) uses `<main>/.worktrees/` as the plan says.
- Lock, heartbeat, caffeinate PID, gate logs: `<main>/.git/pymars-executor/` (shared by all worktrees).
- Briefs: `briefs/` in this journal.

## Phase

RUNNING (week 2). Weekly 17 % at 23:25 Sunday 2026-09-27 (5-hour 24 %); reset 2026-10-04 17:00 PDT. BUDGET RULE FROM THE USER (22:05 Sunday 2026-09-27): use up to 50 % of this week's weekly limit (reset 2026-10-04 17:00 PDT), with no daily pacing: burn it now if useful. When the work pauses at 50 %, CHECK BACK IN WITH THE USER in the session (tell them it paused and ask whether more budget may be used). Concurrency per the plan (at most 4 authors, 3 reviewers); sonnet for mechanical work (integrator, checkers).

main: eec81d2 (#54 T10 _pruning.py), 955b44c (#46 T06 part 1, the reference's terms to linear algebra), eabac8c (#52 _terms), f9cfc5a (#45 _linalg), 85ac0e6 (#51 _gcv and _knots), 36ac4ca (#41 T04 results), d6cac73 (#43), 288d93c (#42), a8dd024 (#37), b0d642d (#34), d4f74f4 (#38), 8760cee (#40), dbe5794 (#35), 31e6c13 (#1).
Done: T00 #2, T01 v1 #3 (v2 is #44), T02 #4, T03 #5, T04 #6, T05 #7, T08 #10, T09 #11, T10 #12.

Open pull requests:
- #47 part 2 pruning (d3b3423, base main): round 1 REQUEST_CHANGES from both roles (missing tests; the code is correct). `t06-reference` adds the tests, then a narrow round 2.
- #48 part 3 forward pass (3414260) and #49 part 4 fit_mars (f5f9a83, Closes #8): after #47, each rebased and dual-reviewed in order.
- #55 T11 stage 1 (t11-forward-stage1, 48adca8, draft): `t11-forward` finishes it; dual review when ready.
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

1. #47: when `t06-reference` reports, run the narrow round 2 (tools/dual_review.js, focus on the round-1 findings), then merge; then #48 and #49 in order (rebase, dual review, merge).
2. #55 (T11 stage 1): dual review when `t11-forward` reports; then stages 2 and 3.
3. T12 `_core.py` (#14): `t12-core` started 23:40 Sunday, brief `briefs/T12-core.md`; branch t12-core stacked on t11-forward-stage1; PR after #55 merges.
4. T18 `DIFFERENCES_legacy.md` (#20): `t18-legacy` started 23:40 Sunday, brief `briefs/T18-legacy.md`; 1 reviewer (`single`).
5. T07 conformance (brief to write) after #49 merges. T13 estimators after T12 (built against the reference through CORE-6 until the fast core is in). T15 oracle tests after T06 completes. T14, T16, T17 after T13.
6. Follow-ups: T05 notes PR; tooling PR (merge_pr.sh: refuse while another open PR uses the head branch as its base; accept rebase-only re-approvals); T08 non-blocking notes (child_term validation; bool for penalty and adjust_endspan per CORE-2; the tie-order knife edge in _knots run sums; tss ZeroDivisionError; OverflowError at extreme values; SPAN-4 test at bb06.12's values); T09 notes (the scale range must include weights and products for LA-7; the LA-4 <= fixed-point test); T10 non-blocking notes from #54's reviews. All go on #44 (spec v2) or later PRs.
7. Housekeeping: the unused branch origin/t08-terms-gcv and worktree <S>/.worktrees/t08-terms-gcv-knots; about 100 MB of reviewer mutation copies in this session's scratchpad (rev46spec_mut, rev46spec_base) that the guard blocked from deletion (for the user to remove). Note: two #54 reviewers shared one scratchpad mutate.py; #54 is merged, so this is a note only (COMMON.md now gives each agent its own subfolder).

## Running agents

- `t06-reference` (opus): #47 round-1 fixes (tests for the listed mutants), then rebases of #48 and #49.
- `t11-forward` (opus): T11 stage 1, PR #55 (draft).
- `t12-core` (opus): T12, issue #14, from 23:40 Sunday.
- `t18-legacy` (opus): T18, issue #20, from 23:40 Sunday.
- Resumable: `t10-pruning` (#54 follow-ups), `integrator` (sonnet, pure rebases), `rebase-check-45` (sonnet, rebase-only re-approvals), `rev46-r3-adv`, `rev46-r3-spec`, `t08-terms-gcv-knots`, `t09-linalg`, `t05-fixtures`, `t01-spec` (spec v2).

## Running jobs

- None. The legacy full run (T04) finished at 16:13 PDT Saturday 2026-09-26 (ALL BLOCKS DONE); results in `<main>/.worktrees/runner-legacy/validation/runs/legacy_full` (49,441 files), logs `driver.log` (6 workers) and `driver_n4.log` (4 workers). Keep the runner worktree until T04 step 6 has collected the results into `validation/sims/results/legacy/`.

## Usage and resets

- 2026-09-25 16:19 PDT: 5-hour 19 % (resets 2026-09-25 19:59 PDT); weekly all models 45 % (resets 2026-09-27 16:59 PDT); weekly Fable 0 %.

## Guard notes

- The `dcg` hook blocks `rm -rf`, `git reset --hard`, `git clean`, `git checkout -- <file>`. It allows `git rm -r`, `git push --force-with-lease`, `gh pr merge --squash --delete-branch`, `git worktree remove`, `git rebase`, `git branch -d`, `mv`, `kill`. Check with `dcg test "<command>"`.
- The app's worktree guard blocks Write and Edit outside the session worktree, also after request_directory for `<main>`. change_directory is refused for a worktree session.
- The auto-mode classifier denied `run_scheduled_task`.
