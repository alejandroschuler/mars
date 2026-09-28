# Executor state

Work done: no

Updated: 2026-09-28 05:24 PDT, by executor session `local_a59469db-c3f1-480c-81c4-a4c7beae39f4`.

This file and LOG.md are the executor journal (branch `executor`; only the executor pushes). A new executor rebuilds its state from this file, LOG.md, the issues and pull requests of `alejandroschuler/mars`, ListAgents and the job manifests.

## Paths

- `<main>` = `/Users/aschuler/Documents/research/projects/pymars`.
- `<S>` = this executor's session worktree, `<main>/.claude/worktrees/competent-poincare-5a4d49`. The desktop app lets the Write and Edit tools change files only inside the session worktree, so this executor keeps the journal worktree at `<S>/.worktrees/journal` and task worktrees at `<S>/.worktrees/<branch>`. An executor that runs in `<main>` itself (for example from the watchdog) uses `<main>/.worktrees/` as the plan says.
- Lock, heartbeat, caffeinate PID, gate logs: `<main>/.git/pymars-executor/` (shared by all worktrees).
- Briefs: `briefs/` in this journal.

## Phase

RUNNING (week 2). Weekly 30 % at 03:22 Monday 2026-09-28 (5-hour 1 %, next reset 08:20 PDT); reset 2026-10-04 17:00 PDT. BUDGET RULE FROM THE USER (22:05 Sunday 2026-09-27): use up to 50 % of this week's weekly limit (reset 2026-10-04 17:00 PDT), with no daily pacing: burn it now if useful. When the work pauses at 50 %, CHECK BACK IN WITH THE USER in the session (tell them it paused and ask whether more budget may be used). Pacing to stop cleanly at 50 %: from 44 % weekly, start only small agents that finish open pull requests (checkers, narrow re-reviews, preferably sonnet), and no new author work; from 48 %, start nothing and ask running agents to push a checkpoint and stop; at 50 %, pause and tell the user. The burn is about 8 points per hour with 5 to 7 agents (34 % at 03:52 Monday). Concurrency per the plan (at most 4 authors, 3 reviewers); sonnet for mechanical work (integrator, checkers).

main: 1918863 (#56 TOOLS-1 merge guard), 77b6692 (#47 T06 part 2, pruning), eec81d2 (#54 T10 _pruning.py), 955b44c (#46 T06 part 1, the reference's terms to linear algebra), eabac8c (#52 _terms), f9cfc5a (#45 _linalg), 85ac0e6 (#51 _gcv and _knots), 36ac4ca (#41 T04 results), d6cac73 (#43), 288d93c (#42), a8dd024 (#37), b0d642d (#34), d4f74f4 (#38), 8760cee (#40), dbe5794 (#35), 31e6c13 (#1).
Done: T00 #2, T01 v1 #3 (v2 is #44), T02 #4, T03 #5, T04 #6, T05 #7, T08 #10, T09 #11, T10 #12.

Open pull requests:
- #48 part 3 forward pass: round 1 done. Spec (on 5756c66): 33 surviving mutants, fixed in 8a56ea3. Adversarial (on 8a56ea3): pass right against earth (2320 fresh fits; 2237 exact, the rest explained); 8 more surviving mutants, and the reviewer's 16 killer tests (scratchpad rev-48-adversarial-2/test_killers2.py) go in now; spec questions (code 2 for GRSq' < -1000 in earth; second_rss below best_rss by rounding) go to #44. Batch 2 done at c2fbfbc (rebased on 1918863; the 8 mutants fail; knot_scan(distinct=True); spec questions on #44; gate B PASS; CI 12/12). Round 2: adversarial APPROVE on c2fbfbc; spec REQUEST_CHANGES (the FAST-5 call of queue_value is unpinned; the reviewer gave the exact test). t06-reference adds that test and small ones (tests only); then a checker's narrow re-approval for both roles, merge, then #49.
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

1. #47: when `t06-reference` reports, run the narrow round 2 (tools/dual_review.js, focus on the round-1 findings), then merge; then #48 and #49 in order (rebase, dual review, merge).
2. T11: reviews of #57 then #55 after the 03:20 reset (no new agents at 81 %); stage 2 (interactions) after the stage-1 review, by `t11-forward` (resumable).
3. T12 `_core.py` (#14): branch origin/t12-core (e040a86 types, 0659fcd fit_mars; on T11 head f564cd5; gate B PASS on both). Decision: one PR with two commits (1354 lines, over the 800 guide). When #55 merges, message t12-core: rebase, gate B, open PR (Closes #14), ready. Then the dual review, commit by commit. Earth: 129 of 131 S fits match through a stub forward pass (2 constant-y fits are the GCV-7 departure); end to end at degree 1 S01 and S04 match. Mutants 291 of 314 killed.
4. T18 `DIFFERENCES_legacy.md` (#20): PR #58 ready (a514801, on 1918863, gate B PASS). 350 fits; first differences: rule 316, quirk 23, bug 8, none unexplained. Confirmed F1-F7, F9, F10; qualified F14 (2 of 100 matched S15 draws agree at every step), F15 (legacy test error lower in 63 of 100 S15 draws at defaults); new F17 (bug: categorical_features on string labels gives intercept only), F18 (several responses unsupported), F19 (earth fits rounding noise on a constant y). HEAD equals the wheel on 243 unweighted cases. Plan correction: the legacy matched mode needs Adjust.endspan = 0, not 1. Decision: one PR (about 2,060 code and test lines; validation only). Round 1 REQUEST_CHANGES (wf_490d20d1-c2a): the labeling logic lacks unit tests (16 of 22 mutants survive, some change the counts), and the F4 evidence is wrong for 4 of 5 fits; 34 sampled labels, F17 to F19 and the Adjust.endspan correction confirmed. Fixed at 870c482 (56 unit tests; round-1 mutants killed; F4 text fixed; bug 8 to 6; gate B PASS; CI 12/12); narrow round 2 (sonnet) running; fixes go to the worktree t18-legacy-gate, branch t18-legacy-rebase, pushed to t18-legacy.
5. T07 conformance (brief to write) after #49 merges. T13 estimators after T12 (built against the reference through CORE-6 until the fast core is in). T15 oracle tests after T06 completes. T14, T16, T17 after T13.
6. Follow-ups: the plan's legacy matched mode says Adjust.endspan = 1; T18 and its reviewer found it must be 0 (SPAN-4); record in dev/DECISIONS.md (the plan text stays). T05 notes PR; tooling PR (merge_pr.sh: refuse while another open PR uses the head branch as its base; accept rebase-only re-approvals); T08 non-blocking notes (child_term validation; bool for penalty and adjust_endspan per CORE-2; the tie-order knife edge in _knots run sums; tss ZeroDivisionError; OverflowError at extreme values; SPAN-4 test at bb06.12's values); T09 notes (the scale range must include weights and products for LA-7; the LA-4 <= fixed-point test); T10 non-blocking notes from #54's reviews. All go on #44 (spec v2) or later PRs.
7. Housekeeping: issue #60 (needs-user: local cleanup the guard blocked: restore 7 reformatted prototype files in <S>/.worktrees/t18-legacy, delete <S>/.venv, delete the scratchpad at the end). Issue #59 (needs-user, no action needed: auto mode denied new_worktree.sh with a start point). 124 MB of mutant copies in <scratchpad>/t12-core/ for the user to remove. the unused branch origin/t08-terms-gcv and worktree <S>/.worktrees/t08-terms-gcv-knots; about 100 MB of reviewer mutation copies in this session's scratchpad (rev46spec_mut, rev46spec_base) that the guard blocked from deletion (for the user to remove). Note: two #54 reviewers shared one scratchpad mutate.py; #54 is merged, so this is a note only (COMMON.md now gives each agent its own subfolder).

## Running agents

- `t06-reference` (opus): #48 round-2 test (FAST-5 lambda).
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
