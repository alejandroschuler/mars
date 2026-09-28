# Executor state

Work done: no

Updated: 2026-09-28 12:06 PDT, by executor session `local_a59469db-c3f1-480c-81c4-a4c7beae39f4`.

This file and LOG.md are the executor journal (branch `executor`; only the executor pushes). A new executor rebuilds its state from this file, LOG.md, the issues and pull requests of `alejandroschuler/mars`, ListAgents and the job manifests.

## Paths

- `<main>` = `/Users/aschuler/Documents/research/projects/pymars`.
- `<S>` = this executor's session worktree, `<main>/.claude/worktrees/competent-poincare-5a4d49`. The desktop app lets the Write and Edit tools change files only inside the session worktree, so this executor keeps the journal worktree at `<S>/.worktrees/journal` and task worktrees at `<S>/.worktrees/<branch>`. An executor that runs in `<main>` itself (for example from the watchdog) uses `<main>/.worktrees/` as the plan says.
- Lock, heartbeat, caffeinate PID, gate logs: `<main>/.git/pymars-executor/` (shared by all worktrees).
- Briefs: `briefs/` in this journal.

## Phase

PAUSED for the budget (week 2). Weekly 53 % at 11:55 Monday 2026-09-28; the user's limit this week is 55 % (10:05 Monday), and the pacing rule starts nothing new from 53 %. The executor told the user in its session and asked whether more budget may be used. No agents or jobs are running. The heartbeat is set to the weekly reset (2026-10-04 17:00 PDT), so the watchdog does not resume the work during the pause. If the user allows more, the executor in this session resumes and refreshes the heartbeat. If a watchdog run takes over after the weekly reset, it first asks the user in its session about the week-3 budget.

Efficiency under the meaningful-tests rule (from 10:10 Monday, 46 % to 53 %): #57 round 2 cost about 1 point (earlier full rounds about 3); #57's fix, a narrow check and the merge about 2; the full dual review of #55 (1,349 lines) about 3. The reviewers now sample about 30 mutants, flag performative tests for deletion, and spend most effort on wrong results; both rounds found real numerical bugs.

## Done in P0

- Lock `<main>/.git/pymars-executor/lock` taken; heartbeat file in it.
- `gh repo set-default alejandroschuler/mars`; upstream push URL is `DISABLED`; `.worktrees/` in `.git/info/exclude`.
- `validation-plan` pushed to the fork; tag `legacy-1.0.4-head` (d68b54a) pushed.
- `caffeinate -dimsu` detached, PID in `<main>/.git/pymars-executor/caffeinate.pid`.
- Planning prototypes are on `validation-plan` in `validation/legacy/`; a copy is in `<main>/.git/pymars-executor/legacy-prototypes/exp/`.
- Text of Milborrow's notes saved for local reading at `<main>/.git/pymars-executor/ref/earth-notes.txt` (never commit or quote it).
- Watchdog task `pymars-executor-watchdog` created (cron `17 * * * *`, the app shows 25 past the hour; notifyOnCompletion false). Its folder and mode are not checked yet: `run_scheduled_task` was denied by the auto-mode classifier, so the user was asked to click Run now once.

## Next actions (on resume)

1. #55 (T11 stage 1 part 2, `_forward.py`, head 5399a75): round 1 (wf_24150913-342) REQUEST_CHANGES from both roles; send to `t11-forward` on resume. Blocking (both reviewers checked their fixes): center x (for example x - mean, or a data value) before b·x enters Gram-Schmidt, in `setup` (line 207) and `columns` (277-279), as the plan's Fast path says (LA-5 fails on shifted covariates: 3.9e-7 at X + 1e9; 57 of 150 shifted fits differ from the reference; near 1e14 the pair rule fails); rss[0] = TSS from `_gcv.tss(Yc)` with Yc = Ys - Ys.mean(axis=0) (line 481; off by 8.8e-6 at y + 2^44); one extension of test_a_shift_of_y with exact shifts (2^36, 2^40, 2^44) pins both. Tests to delete (performative): test_second_largest, test_max_legal, the signature-defaults block of test_the_codes_and_the_defaults (lines 130-144), repeated cases in test_the_stop_after_a_step. Non-blocking: `_linalg.orthogonalize(Q_new, self.E)` in the pair search (line 212); np.errstate(over='ignore') for the ldexp overflow (lines 409, 484); the stage-1 LA-3 recheck is equivalent (keep only with a docstring note); document the CORE-7 worst case (pass 2 values every knot when y is nearly linear in one covariate); the two test extensions that kill the surviving sampled mutants; say in the PR that the committed comparison with the reference's fit_mars belongs to T15. Comments: https://github.com/alejandroschuler/mars/pull/55#issuecomment-5876389020 (spec), #issuecomment-5876432481 (adversarial). After the fix: rebase onto main with only #55's own commits (`git rebase --onto origin/main 2b5d35f`), gate B, CI; a checker's narrow re-approval of both roles; merge. Then message `t12-core` (rebase onto main, open its PR with `Closes #14`, ready), and T11 stage 2.
2. T07 conformance (brief `briefs/T07-conformance.md`; T06 is done, so it can start now; add the fast implementation when T12 merges).
3. T15 oracle tests (brief to write; needs T06, done, and the fast code as it lands).
4. T13 estimators (brief `briefs/T13-estimators.md`) after T12; then T14, T16, T17.
5. Test pruning (the user's rule of 2026-09-28: meaningful tests only): once T07's conformance tests cover the reference against earth, prune the reference's unit tests that only repeat that coverage (tests/reference/test_reference.py is 2,633 lines for 1,259 lines of code); the same pass for the fast modules before the freeze (T21) and in T23.
6. Follow-ups: TOOLS-2 (the fake gh's `pr view` branch ignores its arguments); DECISIONS.md: the legacy matched mode needs Adjust.endspan = 0 (T18); T05 notes PR; T08, T09 and T10 non-blocking notes; spec v2 (#44) collects the questions from #47 to #49 and #57.
7. Housekeeping: issues #59 and #60 (needs-user); the unused branch origin/t08-terms-gcv and worktree <S>/.worktrees/t08-terms-gcv-knots.

## Running agents

- `t11-forward` (opus): idle; #57 fixed, #55 rebased on it. Resumable with SendMessage: `t12-core` (T12 PR after #55), `t06-reference` (done), `t18-legacy` (done), `tools-merge-guard` (TOOLS-2), and older ones.

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
