# Executor log

Merges, incidents, decisions and the core-hour ledger. Times are PDT.

## 2026-09-25

- 16:15 Executor session `local_a59469db-c3f1-480c-81c4-a4c7beae39f4` started in the desktop worktree `competent-poincare-5a4d49`, permission mode Auto, model Opus 5.5.
- 16:16 Took the lock. Safety setup: gh default repo is the fork; upstream push URL `DISABLED`; `.worktrees/` excluded.
- 16:17 Pushed `validation-plan` (eae5286) and tag `legacy-1.0.4-head` (d68b54a) to the fork.
- 16:18 Started `caffeinate -dimsu` (detached, PID file in `.git/pymars-executor/`).
- 16:19 Versions: R 4.4.3; earth 5.3.4; nnet 7.3.20; uv 0.11.11; Python 3.12.13 (uv), 3.13.5 and 3.14.7 (Homebrew); numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1 (planning venv); Apple M1 Pro, 10 cores, 32 GB, Darwin 24.6.0.
- 16:20 Incident: the `dcg` hook blocked `git rm -rf .` in a new journal worktree. No deletion was needed, so the journal branch was made with `git mktree` and `git commit-tree` instead.
- 16:21 Created the watchdog task `pymars-executor-watchdog` (cron `17 * * * *`; the app dispatches it at 25 past the hour).
- 16:22 Incident: the auto-mode classifier denied `run_scheduled_task` for the recovery drill. Not retried by any other route. The user was asked in the session to click Run now once on the routine.
- 16:30 Incident: the app's worktree guard blocks the Write and Edit tools outside the session worktree (also after request_directory granted `<main>`; change_directory is refused for a worktree session). Decision: this executor keeps the journal worktree and the task worktrees under its session worktree, `<S>/.worktrees/`. The lock, heartbeat and gate logs stay in `<main>/.git/pymars-executor/`, written by the plan's shell commands. Moved the journal worktree to `<S>/.worktrees/journal`.
- 16:38 Started agent `t00-author` (opus) with `briefs/T00-bootstrap-author.md`.
- 17:25 The first push of `t00-bootstrap` registered Actions workflows on the fork: `ci.yml` (from the branch) and the inherited `welcome.yml` (pull_request_target from `main`, which posted a welcome comment on PR #1). CI now runs on pull requests, so the merge rule "CI green" applies from PR #1 on. Draft PR #1 is open.
- 17:30 Wake: 5-hour 27 %, weekly 46 %. Told `t00-author` that CI must pass on the final head.
- 17:54 T00 author reported: PR #1 ready at a663d38; CoC contact fixed; gate B passed; CI 12/12. Started reviewer `rev-pr1-bootstrap`.
- 18:08 Recovery drill, watchdog part: the scheduled run at 17:25 (session local_899fec89-88af-4c0a-a262-d50ad9a73f83) started in <main> (not a worktree), model opus-5-5[1m], effort medium, read the fresh heartbeat and stopped after one Bash call. Its permission mode is not reported to this session; the user's default mode is auto. The user no longer needs to click Run now.
- 18:19 PR #1 review round 1 (rev-pr1-bootstrap): REQUEST_CHANGES at a663d38; blocking: merge_pr.sh counted REVIEW lines from any account and from any line of a comment. Sent the fix and 6 small non-blocking fixes to t00-author.
- 18:40 Merged PR #1 (bootstrap) as 31e6c13 with merge_pr.sh after review round 2 APPROVE. Issues on; 14 labels; 32 issues (#2 to #33). Synced <main> and <S> to main. Closed #2. Started t01-spec (opus), t02-harness (sonnet), t03-sims (sonnet). Drill: started and stopped drill-helper.
- 18:45 Recovery drill, helper part: passed. drill-helper was stopped with TaskStop before it wrote anything; on the next wake SendMessage resumed it with its context, and it wrote env/versions.json (R 4.4.3, earth 5.3.4, plotmo 3.6.4, nnet 7.3.20, macOS 15.7.4). Both parts of the P0 recovery drill are done.
- 19:11 19:10 wake: 5-hour 53 % (reset 20:00), weekly 51 %. PR #35 (T03 part 1, 57f4f87, gate B and CI passed) ready; started reviewer rev-pr35-sims1 (opus, single). PR #34 (T01 part 1) is a draft.
- 19:32 PR #35 review round 1 (rev-pr35-sims1): REQUEST_CHANGES at 57f4f87: weak formula tests (24 of 34 one-token mutations passed), validation tests not in any gate or CI, shape broadcasting in metrics. Decision: validation test folders go in testpaths; marker 'external' for tests that need R or the legacy venv; gate A runs -m 'not slow and not external', gate B and CI -m 'not external', gate C runs external. Sent to t03-sims and t02-harness.
- 19:34 T01 part 1 ready: PR #34 at 252082c (gate B passed). Started the dual-review workflow (spec + adversarial), run wf_cbba59e1-bd2; the script is saved as tools/dual_review.js (args: pr, title, head, brief, roles, focus). Spec findings: leaps-style pruning (PRUNE-3), interaction endspan E + round(a*E) capped, exact knot scan, row-order departure within ties, tolerance 1e-2 then 1e-5 from step 8.
- 19:41 19:41 wake: 5-hour 75 % (reset 20:00), weekly 54 %. PR #36 (T02 part 1, 52b01e0) is ready; its review waits until after the 20:00 reset (pacing rule near 80 %). Drafts: #37 (T01 part 2), #38 (T02 part 2).
- 19:54 T02 author done: PR #36 (part 1, 962f6a1) and PR #38 (part 2, stacked on t02-harness-r, 6d9bd30), gate B and CI passed, 128 harness tests (R tests marked external). earth-conformance.yml belongs to T05. new_adapter.py's record shape is a guess until T12. T03 part 1 fixed at 96a3d2f; re-review by rev-pr35-sims1 running. Reviewer for #36 and #38 starts after the 20:00 reset.

## Core-hour ledger

| Date | Block | Core-hours | Notes |
|---|---|---|---|
