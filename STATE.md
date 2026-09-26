# Executor state

Work done: no

Updated: 2026-09-25 18:08 PDT, by executor session `local_a59469db-c3f1-480c-81c4-a4c7beae39f4`.

This file and LOG.md are the executor journal (branch `executor`; only the executor pushes). A new executor rebuilds its state from this file, LOG.md, the issues and pull requests of `alejandroschuler/mars`, ListAgents and the job manifests.

## Paths

- `<main>` = `/Users/aschuler/Documents/research/projects/pymars`.
- `<S>` = this executor's session worktree, `<main>/.claude/worktrees/competent-poincare-5a4d49`. The desktop app lets the Write and Edit tools change files only inside the session worktree, so this executor keeps the journal worktree at `<S>/.worktrees/journal` and task worktrees at `<S>/.worktrees/<branch>`. An executor that runs in `<main>` itself (for example from the watchdog) uses `<main>/.worktrees/` as the plan says.
- Lock, heartbeat, caffeinate PID, gate logs: `<main>/.git/pymars-executor/` (shared by all worktrees).
- Briefs: `briefs/` in this journal.

## Phase

P0 bootstrap (T00).

## Done in P0

- Lock `<main>/.git/pymars-executor/lock` taken; heartbeat file in it.
- `gh repo set-default alejandroschuler/mars`; upstream push URL is `DISABLED`; `.worktrees/` in `.git/info/exclude`.
- `validation-plan` pushed to the fork; tag `legacy-1.0.4-head` (d68b54a) pushed.
- `caffeinate -dimsu` detached, PID in `<main>/.git/pymars-executor/caffeinate.pid`.
- Planning prototypes are on `validation-plan` in `validation/legacy/`; a copy is in `<main>/.git/pymars-executor/legacy-prototypes/exp/`.
- Text of Milborrow's notes saved for local reading at `<main>/.git/pymars-executor/ref/earth-notes.txt` (never commit or quote it).
- Watchdog task `pymars-executor-watchdog` created (cron `17 * * * *`, the app shows 25 past the hour; notifyOnCompletion false). Its folder and mode are not checked yet: `run_scheduled_task` was denied by the auto-mode classifier, so the user was asked to click Run now once.

## Next actions

1. Wait for the T00 author's report; then one reviewer; then merge with `dev/tools/merge_pr.sh` (read it first).
2. After the merge: turn on issues, labels, task issues T01 to T24, the missing-values `later` issue, `needs-user` issues for the watchdog check; check that CI runs; sync checkouts; recovery drill (TaskStop one helper and check the restart; check the watchdog run with list_task_runs and get_session).
3. Start P1: T01 spec, T02 earth harness, T03 simulation harness.

## Running agents

- `t00-author` (opus): done; PR #1 ready at a663d38 (gate B and CI 12/12 passed).
- `rev-pr1-bootstrap` (opus): reviewing PR #1 as role `bootstrap`, started 18:05 PDT.

## Running jobs

None.

## Usage and resets

- 2026-09-25 16:19 PDT: 5-hour 19 % (resets 2026-09-25 19:59 PDT); weekly all models 45 % (resets 2026-09-27 16:59 PDT); weekly Fable 0 %.

## Guard notes

- The `dcg` hook blocks `rm -rf`, `git reset --hard`, `git clean`, `git checkout -- <file>`. It allows `git rm -r`, `git push --force-with-lease`, `gh pr merge --squash --delete-branch`, `git worktree remove`, `git rebase`, `git branch -d`, `mv`, `kill`. Check with `dcg test "<command>"`.
- The app's worktree guard blocks Write and Edit outside the session worktree, also after request_directory for `<main>`. change_directory is refused for a worktree session.
- The auto-mode classifier denied `run_scheduled_task`.
