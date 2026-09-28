# Executor state

Work done: no

Updated: 2026-09-28 14:22 PDT, by executor session `local_a59469db-c3f1-480c-81c4-a4c7beae39f4`.

This file and LOG.md are the executor journal (branch `executor`; only the executor pushes). A new executor rebuilds its state from this file, LOG.md, the issues and pull requests of `alejandroschuler/mars`, ListAgents and the job manifests.

## Paths

- `<main>` = `/Users/aschuler/Documents/research/projects/pymars`.
- `<S>` = this executor's session worktree, `<main>/.claude/worktrees/competent-poincare-5a4d49`. The desktop app lets the Write and Edit tools change files only inside the session worktree, so this executor keeps the journal worktree at `<S>/.worktrees/journal` and task worktrees at `<S>/.worktrees/<branch>`. An executor that runs in `<main>` itself (for example from the watchdog) uses `<main>/.worktrees/` as the plan says.
- Lock, heartbeat, caffeinate PID, gate logs: `<main>/.git/pymars-executor/` (shared by all worktrees).
- Briefs: `briefs/` in this journal.

## Phase

RUNNING. The weekly counter reset to 0 % at about 13:00 Monday 2026-09-28 (the reset the user mentioned; the next reset shown is still 2026-10-04 17:00 PDT). BUDGET RULE FROM THE USER (13:25 Monday 2026-09-28): go to 50 % of the new weekly counter (until the reset on 2026-10-04 17:00 PDT), then pause and tell the user. Pacing: from 44 %, only small agents that finish open pull requests; from 48 %, nothing new; at 50 %, pause, set the heartbeat to the weekly reset, and tell the user. The 5-hour limit still applies (no new agents from 80 % of it; the 70 % heartbeat rule; the 5-hour reset is 18:10 PDT). New rule (13:30 Monday): start a full review or a long author task only while the 5-hour window is below 60 %, because the limit killed reviews twice (the #48 adversarial, the #55 round 2) and their work was lost. Concurrency per the plan (at most 4 authors, 3 reviewers); sonnet for mechanical work. Test rules: COMMON.md "Tests" and REVIEWER.md.

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

1. T11 stage 1 done: #57 and #55 merged (20cdde0). Next: stage 2 (interactions: degree 2 and 3, parents other than the intercept, Adjust.endspan, the slot rules) by a fresh author `t11-stage2` (brief `briefs/T11-forward.md`, stage 2; a fresh agent because resuming the long-context t11-forward is costly); T12's PR #70 ready (328af43, on 20cdde0; commits 05b7f87 types, 282da81 fit_mars, 328af43 the reference comparison; 16 cases equal the reference's fit_mars to 8.8e-14; gate B PASS): its dual review (spec, adversarial) waits for the 5-hour window to be below 60 % again (60 % at 14:04; reset 18:10) and for reviewer slots; then t13-estimators rebases and opens T13's PR.
2. T07 conformance: PR #66 ready (80bd7f5, on 2a00063; gate B PASS; CI 12/12). The reference against earth: 124 of 133 dataset fixtures and 197 of 200 S15 draws agree in full; validation/differences.json has 7 labeled entries (4 rule: LA-7 in raw units, earth's zeroing of small sums of squares twice, a constant y; 3 quirk: FWD-11's hidden term), no bug; S15: 98.5 % agree, near-tie stops 0 %. OQ-2 decided (pruning near-tie at 1e-7 of the lower RSS). Four questions posted on #44. About 1,000 lines of test and harness code. Round 1 (wf_0064e414-0b1) REQUEST_CHANGES: the tie rule lets FWD-5 and FWD-6 errors pass; the degenerate branch always passes; E7's evidence is wrong; E1 should be a quirk. Sent to t07-conformance; then a narrow round 2. Follow-up for the pruning pass: four earth self-checks in test_reference.py now duplicate this suite. PRs from the user's own separate sessions, to review (single role, sonnet) and merge after the 5-hour reset at 13:20: #61 merged as 11202d3 (after update-branch and update-check-61). (For the user's branches, update-branch avoids the force-push that auto mode denied for #64.) #63 merged (after update-branch and update-check-63). Follow-ups from its review (validation only, for T22): load_results skips unparseable rep files silently; the appendix table shows a blank noise for None (nan is truthy), #64 (TOOLS-2; closes #62): APPROVE at 6e15ed4 (sonnet); rebase-64 rebased it to 99f1eee (range-diff '='; gate B PASS), but auto mode denied the force-push to the user's branch; needs-user issue opened with the exact command; after the user pushes: a rebase-check re-approval, then the merge. Keep the worktree <S>/.worktrees/rebase-64 until then. TOOLS-3 follow-up (from #64's review): the fake gh never fails a pr view call, so 5 exit-code mutants survive; a FAKE_PR_VIEW_FAIL case; open an issue for it.
3. T15 oracle tests: PR #67 ready (1ebabee, on 20cdde0; +1283 -481; gate B and gate C PASS; CI 12/12). Compares the fast forward pass and pruning with the reference on 122 fixture fits, 200 S15 draws, 13 designs and 10,000 hypothesis fits (0.7 % near-tie stops, none on fixtures); deleted 29 narrow tests whose designs are now oracle cases (the oracle kills every mutant they killed). Size accepted (oracle tests replacing narrow ones). Its dual review waits for the 5-hour reset (with #70's). Later: connect compare_fits when T12 merges; apply its shifted-response patch after #69. It found a real bug in merged pymars/_pruning.py (no centering of Y before the QR; LA-5 missed by up to 1.2e-6 with a large-mean y); Fix PR #69 ready (2a9edb3, on 20cdde0; center by a data value then the weighted mean in pruning_pass, final_fit and _gcv.tss; 1e-16 on T15's case; gate B PASS): spec APPROVE (sonnet); adversarial (opus, wf_36a83534-922) runs. The spec review found the same class of bug in the reference's final_rss (1.3e-4 relative at mean 1e13): t06-reference fixes it in a small PR (branch t06-final-rss); then T15 adds its shifted-response draws.
4. T13's PR after T12 merges (see Running agents); then T14 (`_glm.py` and the classifier; brief to write), T16, T17.
5. Test pruning (the user's rule of 2026-09-28: meaningful tests only): once T07's conformance tests cover the reference against earth, prune the reference's unit tests that only repeat that coverage (tests/reference/test_reference.py is 2,633 lines for 1,259 lines of code); the same pass for the fast modules before the freeze (T21) and in T23.
6. Follow-ups: `_gcv.tss` centers y about one computed mean, so T12's statistics keep a small rss[0] error for a y with a large mean (t11-forward's note; fix in `_gcv` as #55 did: center about a data value, or twice); TOOLS-2 (the fake gh's `pr view` branch ignores its arguments); DECISIONS.md: the legacy matched mode needs Adjust.endspan = 0 (T18); T05 notes PR; T08, T09 and T10 non-blocking notes; spec v2 (#44) collects the questions from #47 to #49 and #57.
7. Housekeeping: issues #59 and #60 (needs-user); the unused branch origin/t08-terms-gcv and worktree <S>/.worktrees/t08-terms-gcv-knots.

## Running agents

- `t11-forward` (opus): done (stage 1 merged); resumable for stage-1 questions.
- `t11-stage2` (opus): T11 stage 2 (interactions), a fresh author, from about 14:20 Monday.
- `t07-conformance` (opus): #66 round-1 fixes.
- `t15-oracle` (opus): idle; PR #67 ready.
- `t13-estimators` (opus): T13 done up to T12's merge: branch t13-estimators at 073811b (a scaffold merge of main 278fcaf; after T12 merges: `git rebase --onto origin/main 278fcaf`, then the PR with Closes #15). EarthRegressor, Earth, _EarthBase; scikit-learn checks pass on the reference (no expected failures) and 52 on the fast core (7 skips name #13 until T11 stages land); 30 of 30 sampled mutants killed. Decision: the gate B and CI smoke fits use fast_k=0 until T11 stage 3 (restore then; note on #13).
- `t10-pruning` (opus): idle; PR #69 ready.
- Idle, resumable: `t12-core` (open its PR after #55 merges), `t06-reference`, `t18-legacy`, `tools-merge-guard`.

## Running jobs

- None. The legacy full run (T04) finished at 16:13 PDT Saturday 2026-09-26 (ALL BLOCKS DONE); results in `<main>/.worktrees/runner-legacy/validation/runs/legacy_full` (49,441 files), logs `driver.log` (6 workers) and `driver_n4.log` (4 workers). Keep the runner worktree until T04 step 6 has collected the results into `validation/sims/results/legacy/`.

## Usage and resets

- 2026-09-27 23:56 PDT: 5-hour 44 % (resets 2026-09-28 03:20 PDT); weekly 21 % (resets 2026-10-04 17:00 PDT).
- 2026-09-25 16:19 PDT: 5-hour 19 % (resets 2026-09-25 19:59 PDT); weekly all models 45 % (resets 2026-09-27 16:59 PDT); weekly Fable 0 %.

## Guard notes

- Heartbeat near a 5-hour limit (decision 2026-09-27 23:58): once the 5-hour use is 70 % or more, the executor writes max(now, reset + 15 min) into the heartbeat, so that a watchdog run that resumes with the reset does not take the lock from an executor that also resumes. A future heartbeat means that. Current 5-hour reset: 2026-09-28 13:19:59 PDT. If the executor does not resume, the watchdog takes over about 75 minutes after that time.
- The `dcg` hook blocks `rm -rf`, `git reset --hard`, `git clean`, `git checkout -- <file>`. It allows `git rm -r`, `git push --force-with-lease`, `gh pr merge --squash --delete-branch`, `git worktree remove`, `git rebase`, `git branch -d`, `mv`, `kill`. Check with `dcg test "<command>"`.
- The app's worktree guard blocks Write and Edit outside the session worktree, also after request_directory for `<main>`. change_directory is refused for a worktree session.
- The auto-mode classifier denied `run_scheduled_task`.
