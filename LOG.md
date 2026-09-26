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

## Core-hour ledger

| Date | Block | Core-hours | Notes |
|---|---|---|---|
