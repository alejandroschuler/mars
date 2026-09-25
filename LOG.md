# Executor log

Merges, incidents, decisions and the core-hour ledger. Times are PDT.

## 2026-09-25

- 16:15 Executor session `local_a59469db-c3f1-480c-81c4-a4c7beae39f4` started in the desktop worktree `competent-poincare-5a4d49`, permission mode Auto, model Opus 5.5.
- 16:16 Took the lock. Safety setup: gh default repo is the fork; upstream push URL `DISABLED`; `.worktrees/` excluded.
- 16:17 Pushed `validation-plan` (eae5286) and tag `legacy-1.0.4-head` (d68b54a) to the fork.
- 16:18 Started `caffeinate -dimsu` (detached, PID file in `.git/pymars-executor/`).
- 16:19 Versions: R 4.4.3; earth 5.3.4; nnet 7.3.20; uv 0.11.11; Python 3.12.13 (uv), 3.13.5 and 3.14.7 (Homebrew); numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1 (planning venv); Apple M1 Pro, 10 cores, 32 GB, Darwin 24.6.0.
- 16:20 Incident: the `dcg` hook blocked `git rm -rf .` in a new journal worktree. No deletion was needed, so the journal branch was made with `git mktree` and `git commit-tree` instead.

## Core-hour ledger

| Date | Block | Core-hours | Notes |
|---|---|---|---|
