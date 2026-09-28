# Executor state

Work done: no

Updated: 2026-09-28 08:43 PDT, by executor session `local_a59469db-c3f1-480c-81c4-a4c7beae39f4`.

This file and LOG.md are the executor journal (branch `executor`; only the executor pushes). A new executor rebuilds its state from this file, LOG.md, the issues and pull requests of `alejandroschuler/mars`, ListAgents and the job manifests.

## Paths

- `<main>` = `/Users/aschuler/Documents/research/projects/pymars`.
- `<S>` = this executor's session worktree, `<main>/.claude/worktrees/competent-poincare-5a4d49`. The desktop app lets the Write and Edit tools change files only inside the session worktree, so this executor keeps the journal worktree at `<S>/.worktrees/journal` and task worktrees at `<S>/.worktrees/<branch>`. An executor that runs in `<main>` itself (for example from the watchdog) uses `<main>/.worktrees/` as the plan says.
- Lock, heartbeat, caffeinate PID, gate logs: `<main>/.git/pymars-executor/` (shared by all worktrees).
- Briefs: `briefs/` in this journal.

## Phase

PAUSED for the budget (week 2). Weekly 46 % at 08:48 Monday 2026-09-28 (the user's limit for this week is 50 %; the weekly reset is 2026-10-04 17:00 PDT). The executor asked the user in its session whether more budget may be used. No agents are running and no jobs. The heartbeat is set to 2026-10-04 17:00 PDT, so the watchdog does not resume the work during the pause. If the user allows more budget, the executor in this session resumes and refreshes the heartbeat normally. If a watchdog run takes over after the weekly reset, it first asks the user in its session about the week-3 budget, then follows the plan's pacing.

BUDGET RULE FROM THE USER (22:05 Sunday 2026-09-27): use up to 50 % of this week's weekly limit, with no daily pacing. When the work pauses at 50 %, check back in with the user in the session. Concurrency per the plan (at most 4 authors, 3 reviewers); sonnet for mechanical work (integrators, checkers, narrow re-reviews).

## Done in P0

- Lock `<main>/.git/pymars-executor/lock` taken; heartbeat file in it.
- `gh repo set-default alejandroschuler/mars`; upstream push URL is `DISABLED`; `.worktrees/` in `.git/info/exclude`.
- `validation-plan` pushed to the fork; tag `legacy-1.0.4-head` (d68b54a) pushed.
- `caffeinate -dimsu` detached, PID in `<main>/.git/pymars-executor/caffeinate.pid`.
- Planning prototypes are on `validation-plan` in `validation/legacy/`; a copy is in `<main>/.git/pymars-executor/legacy-prototypes/exp/`.
- Text of Milborrow's notes saved for local reading at `<main>/.git/pymars-executor/ref/earth-notes.txt` (never commit or quote it).
- Watchdog task `pymars-executor-watchdog` created (cron `17 * * * *`, the app shows 25 past the hour; notifyOnCompletion false). Its folder and mode are not checked yet: `run_scheduled_task` was denied by the auto-mode classifier, so the user was asked to click Run now once.

## Next actions (on resume)

1. #57 (T11 `_scan.py`, head 754c4bf) and #55 (`_forward.py`, 071d181, on #57): a round-2 review of #57 by both roles (opus: the error bounds of the fix need a strong check; the round-1 comments are https://github.com/alejandroschuler/mars/pull/57#issuecomment-5868460714 and #issuecomment-5868616482; the author's reply #issuecomment-5870015920), merge; t11-forward rebases #55 onto main; the dual review of #55 (only the `_forward.py` commits), merge; then message t12-core (rebase onto main, open its PR with `Closes #14`, ready) and start T11 stage 2 (interactions).
2. T07 conformance (brief `briefs/T07-conformance.md`; T06 is done, so it can start now; add the fast implementation when T12 merges).
3. T15 oracle tests (brief to write; needs T06, done, and the fast code as it lands).
4. T13 estimators (brief `briefs/T13-estimators.md`) after T12; then T14, T16, T17.
5. Follow-ups: TOOLS-2 (the fake gh's `pr view` branch ignores its arguments); DECISIONS.md: the legacy matched mode needs Adjust.endspan = 0 (T18); T05 notes PR; T08, T09 and T10 non-blocking notes; spec v2 (#44) collects the questions from #47 to #49 and #57.
6. Housekeeping: issues #59 and #60 (needs-user); the unused branch origin/t08-terms-gcv and worktree <S>/.worktrees/t08-terms-gcv-knots.

## Running agents

- None running. Resumable with SendMessage: `t11-forward` (#57, #55, stage 2), `t12-core` (T12 PR after #55), `t06-reference` (done), `t18-legacy` (done), `tools-merge-guard` (TOOLS-2), and older ones.

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
