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

## Core-hour ledger

| Date | Block | Core-hours | Notes |
|---|---|---|---|
