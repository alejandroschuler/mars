# Executor state

Work done: no

Updated: 2026-09-28 05:29 PDT, by executor session `local_a59469db-c3f1-480c-81c4-a4c7beae39f4`.

This file and LOG.md are the executor journal (branch `executor`; only the executor pushes). A new executor rebuilds its state from this file, LOG.md, the issues and pull requests of `alejandroschuler/mars`, ListAgents and the job manifests.

## Paths

- `<main>` = `/Users/aschuler/Documents/research/projects/pymars`.
- `<S>` = this executor's session worktree, `<main>/.claude/worktrees/competent-poincare-5a4d49`. The desktop app lets the Write and Edit tools change files only inside the session worktree, so this executor keeps the journal worktree at `<S>/.worktrees/journal` and task worktrees at `<S>/.worktrees/<branch>`. An executor that runs in `<main>` itself (for example from the watchdog) uses `<main>/.worktrees/` as the plan says.
- Lock, heartbeat, caffeinate PID, gate logs: `<main>/.git/pymars-executor/` (shared by all worktrees).
- Briefs: `briefs/` in this journal.

## Phase

RUNNING (week 2). Weekly 30 % at 03:22 Monday 2026-09-28 (5-hour 1 %, next reset 08:20 PDT); reset 2026-10-04 17:00 PDT. BUDGET RULE FROM THE USER (22:05 Sunday 2026-09-27): use up to 50 % of this week's weekly limit (reset 2026-10-04 17:00 PDT), with no daily pacing: burn it now if useful. When the work pauses at 50 %, CHECK BACK IN WITH THE USER in the session (tell them it paused and ask whether more budget may be used). Pacing to stop cleanly at 50 %: from 44 % weekly, start only small agents that finish open pull requests (checkers, narrow re-reviews, preferably sonnet), and no new author work; from 48 %, start nothing and ask running agents to push a checkpoint and stop; at 50 %, pause and tell the user. The burn is about 8 points per hour with 5 to 7 agents (34 % at 03:52 Monday). Concurrency per the plan (at most 4 authors, 3 reviewers); sonnet for mechanical work (integrator, checkers).

main: 0c2a896 (#48 T06 part 3, forward pass), 1918863 (#56 TOOLS-1 merge guard), 77b6692 (#47 T06 part 2, pruning), eec81d2 (#54 T10 _pruning.py), 955b44c (#46 T06 part 1, the reference's terms to linear algebra), eabac8c (#52 _terms), f9cfc5a (#45 _linalg), 85ac0e6 (#51 _gcv and _knots), 36ac4ca (#41 T04 results), d6cac73 (#43), 288d93c (#42), a8dd024 (#37), b0d642d (#34), d4f74f4 (#38), 8760cee (#40), dbe5794 (#35), 31e6c13 (#1).
Done: T00 #2, T01 v1 #3 (v2 is #44), T02 #4, T03 #5, T04 #6, T05 #7, T08 #10, T09 #11, T10 #12.

Open pull requests:
- #49 part 4 fit_mars (Closes #8): t06-reference rebases it onto 0c2a896 and removes the duplicates now in #48 (EARTH_NAMES, params_from_earth); then its dual review.
- #57 round 1 (wf_e0ddfd90-495): both REQUEST_CHANGES. Blocking: knot_scan misses LA-5 at large n (cancellation in D = ||h||^2 - (q0'h)^2 and n·u error in the cumsum of q0'h; errors up to 650x the bound; a knot near tau can be wrongly rejected, and the driver never rechecks it); CI red (lstsq rcond). All 227 mutants killed. Sent to t11-forward (fix design: cancellation-free sums plus a per-knot error bound with explicit recomputation). #57 T11 stage 1 part 1 `_scan.py` (10c6e8e, 321 lines, on 77b6692, base main) and #55 part 2 `_forward.py` (8f0e178, 1091 lines, contains #57's commits, base main): both ready, gate B PASS. Earth: 82 of 88 degree-1 fixtures match the whole forward path, 6 stop at a near-tie (match past it); the reference (#49) gives identical records on all 88; mutants 459 of 463. After the reset: dual review of #57, merge, t11-forward rebases #55, then dual review of #55 (review only the _forward.py commits), merge; then message t12-core.
- TOOLS-2 (later, small): the `pr view` branch of the fake gh in test_merge_pr.sh also ignores its arguments.
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

1. #49 (T06 part 4, Closes #8): after t06-reference's rebase, a dual review (spec, adversarial), fixes, merge. That completes T06 and unblocks T07 (brief `briefs/T07-conformance.md`) and T15.
2. #57 (T11 `_scan.py`): t11-forward fixes the scan accuracy; then a narrow round 2, merge; t11-forward rebases #55 onto main; then the dual review of #55 (only the `_forward.py` commits), merge; then message t12-core (rebase, open its PR, `Closes #14`), and T11 stage 2 (interactions).
3. #58 (T18): narrow round 2 (sonnet) running (wf_f0fee117-267); then merge; then remove the worktrees runner-t18, t18-legacy-head, t18-legacy-gate (keep t18-legacy until the user restores its files, issue #60).
4. Later: T13 estimators (brief `briefs/T13-estimators.md`) after T12; T15 oracle tests after T06; T14, T16, T17 after T13; T01 v2 (#44) after T07's triage.
5. Follow-ups: TOOLS-2 (the fake gh's `pr view` branch ignores its arguments); DECISIONS.md: the legacy matched mode needs Adjust.endspan = 0 (T18); T05 notes PR; T08 non-blocking notes (child_term validation; bool for penalty and adjust_endspan per CORE-2; the tie-order knife edge in _knots run sums; tss ZeroDivisionError; OverflowError at extreme values; SPAN-4 test at bb06.12's values); T09 notes (the scale range must include weights and products for LA-7; the LA-4 <= fixed-point test); T10 non-blocking notes from #54's reviews. They go on #44 or later PRs.
6. Housekeeping: issue #60 (needs-user: local cleanup the guard blocked); issue #59 (needs-user, no action needed: auto mode denied new_worktree.sh with a start point); the unused branch origin/t08-terms-gcv and its worktree <S>/.worktrees/t08-terms-gcv-knots.

## Running agents

- `t06-reference` (opus): rebasing #49 onto 0c2a896.
- `t11-forward` (opus): #57 round-1 fix (scan accuracy), then the #55 rebase.
- `t12-core` (opus): idle until #55 merges (then rebase and open the PR).
- `t18-legacy` (opus): idle; #58 round 2 in review.
- `tools-merge-guard` (sonnet): done (#56 merged); resumable for TOOLS-2.
- Resumable: `t10-pruning` (#54 follow-ups), `integrator` (sonnet, pure rebases), `rebase-check-45` (sonnet, rebase-only re-approvals), `rev46-r3-adv`, `rev46-r3-spec`, `t08-terms-gcv-knots`, `t09-linalg`, `t05-fixtures`, `t01-spec` (spec v2).

## Running jobs

- None. The legacy full run (T04) finished at 16:13 PDT Saturday 2026-09-26 (ALL BLOCKS DONE); results in `<main>/.worktrees/runner-legacy/validation/runs/legacy_full` (49,441 files), logs `driver.log` (6 workers) and `driver_n4.log` (4 workers). Keep the runner worktree until T04 step 6 has collected the results into `validation/sims/results/legacy/`.

## Usage and resets

- 2026-09-27 23:56 PDT: 5-hour 44 % (resets 2026-09-28 03:20 PDT); weekly 21 % (resets 2026-10-04 17:00 PDT).
- 2026-09-25 16:19 PDT: 5-hour 19 % (resets 2026-09-25 19:59 PDT); weekly all models 45 % (resets 2026-09-27 16:59 PDT); weekly Fable 0 %.

## Guard notes

- Heartbeat near a 5-hour limit (decision 2026-09-27 23:58): once the 5-hour use is 70 % or more, the executor writes max(now, reset + 15 min) into the heartbeat, so that a watchdog run that resumes with the reset does not take the lock from an executor that also resumes. A future heartbeat means that. Current 5-hour reset: 2026-09-28 08:19:59 PDT (epoch 1790608799). If the executor does not resume, the watchdog takes over about 75 minutes after that time.
- The `dcg` hook blocks `rm -rf`, `git reset --hard`, `git clean`, `git checkout -- <file>`. It allows `git rm -r`, `git push --force-with-lease`, `gh pr merge --squash --delete-branch`, `git worktree remove`, `git rebase`, `git branch -d`, `mv`, `kill`. Check with `dcg test "<command>"`.
- The app's worktree guard blocks Write and Edit outside the session worktree, also after request_directory for `<main>`. change_directory is refused for a worktree session.
- The auto-mode classifier denied `run_scheduled_task`.
