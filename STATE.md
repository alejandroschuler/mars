# Executor state

Work done: no

Updated: 2026-09-25 16:25 PDT, by executor session `local_a59469db-c3f1-480c-81c4-a4c7beae39f4`.

This file and LOG.md are the executor journal (branch `executor`, worktree `<main>/.worktrees/journal`, only the executor pushes). A new executor rebuilds its state from this file, LOG.md, the issues and pull requests of `alejandroschuler/mars`, ListAgents and the job manifests in `<main>/validation/runs/` (per worktree).

## Phase

P0 bootstrap (T00).

## Done in P0

- Lock `<main>/.git/pymars-executor/lock` taken; heartbeat file in it.
- `gh repo set-default alejandroschuler/mars`; upstream push URL is `DISABLED`; `.worktrees/` in `.git/info/exclude`.
- `validation-plan` pushed to the fork; tag `legacy-1.0.4-head` (d68b54a) pushed.
- `caffeinate -dimsu` detached, PID in `<main>/.git/pymars-executor/caffeinate.pid`.
- Planning prototypes saved in `<main>/.git/pymars-executor/legacy-prototypes/exp/` (for `validation/legacy/`).
- Text of Milborrow's notes saved for local reading at `<main>/.git/pymars-executor/ref/earth-notes.txt` (never commit or quote it).

## Next actions

1. Create the watchdog task `pymars-executor-watchdog` and check its folder and mode.
2. Bootstrap author writes branch `t00-bootstrap` (from `validation-plan`) and opens the bootstrap pull request; one reviewer; merge.
3. After the merge: turn on issues, labels, task issues T01 to T24, the missing-values `later` issue; check that CI runs; sync checkouts; recovery drill.
4. Start P1: T01 spec, T02 earth harness, T03 simulation harness.

## Running agents

None yet.

## Running jobs

None.

## Usage and resets

- 2026-09-25 16:19 PDT: 5-hour 19 % (resets 2026-09-25 19:59 PDT); weekly all models 45 % (resets 2026-09-27 16:59 PDT); weekly Fable 0 %.

## Guard notes

The `dcg` hook blocks `rm -rf`, `git reset --hard`, `git clean`, `git checkout -- <file>`. It allows `git rm -r`, `git push --force-with-lease`, `gh pr merge --squash --delete-branch`, `git worktree remove`, `git rebase`, `git branch -d`, `mv`, `kill`. Check with `dcg test "<command>"`.
