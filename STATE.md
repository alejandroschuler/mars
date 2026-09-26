# Executor state

Work done: no

Updated: 2026-09-25 21:05 PDT, by executor session `local_a59469db-c3f1-480c-81c4-a4c7beae39f4`.

This file and LOG.md are the executor journal (branch `executor`; only the executor pushes). A new executor rebuilds its state from this file, LOG.md, the issues and pull requests of `alejandroschuler/mars`, ListAgents and the job manifests.

## Paths

- `<main>` = `/Users/aschuler/Documents/research/projects/pymars`.
- `<S>` = this executor's session worktree, `<main>/.claude/worktrees/competent-poincare-5a4d49`. The desktop app lets the Write and Edit tools change files only inside the session worktree, so this executor keeps the journal worktree at `<S>/.worktrees/journal` and task worktrees at `<S>/.worktrees/<branch>`. An executor that runs in `<main>` itself (for example from the watchdog) uses `<main>/.worktrees/` as the plan says.
- Lock, heartbeat, caffeinate PID, gate logs: `<main>/.git/pymars-executor/` (shared by all worktrees).
- Briefs: `briefs/` in this journal.

## Phase

P0 done, including both parts of the recovery drill. P1 started 18:45 PDT: T01 (#3), T02 (#4), T03 (#5).

Board: T00 #2 (closed), T01 #3, T02 #4, T03 #5, T04 #6, T05 #7, T06 #8, T07 #9, T08 #10, T09 #11, T10 #12, T11 #13, T12 #14, T13 #15, T14 #16, T15 #17, T16 #18, T17 #19, T18 #20, T19 #21, T20 #22, T21 #23, T22 #24, T23 #25, T24 #26; later: missing values #27, negative minspan #28, newvar.penalty #29, linpreds #30, allowed #31, pmethod and nfold #32, evimp #33. `board/numbers.json` has the map.

main: 31e6c13 (bootstrap, PR #1). `<main>` and `<S>` are at main. CI runs on the fork (registered by the first push); CI on main passed.

## Done in P0

- Lock `<main>/.git/pymars-executor/lock` taken; heartbeat file in it.
- `gh repo set-default alejandroschuler/mars`; upstream push URL is `DISABLED`; `.worktrees/` in `.git/info/exclude`.
- `validation-plan` pushed to the fork; tag `legacy-1.0.4-head` (d68b54a) pushed.
- `caffeinate -dimsu` detached, PID in `<main>/.git/pymars-executor/caffeinate.pid`.
- Planning prototypes are on `validation-plan` in `validation/legacy/`; a copy is in `<main>/.git/pymars-executor/legacy-prototypes/exp/`.
- Text of Milborrow's notes saved for local reading at `<main>/.git/pymars-executor/ref/earth-notes.txt` (never commit or quote it).
- Watchdog task `pymars-executor-watchdog` created (cron `17 * * * *`, the app shows 25 past the hour; notifyOnCompletion false). Its folder and mode are not checked yet: `run_scheduled_task` was denied by the auto-mode classifier, so the user was asked to click Run now once.

## Next actions

2. When T03 is ready: one reviewer (`single`, opus); merge; then T04 (legacy pilot, then the legacy full run as detached jobs from a runner worktree in `<main>/.worktrees/`), as early as possible: the detached jobs keep running during a usage pause.
3. When T01 part 1 is ready: two reviewers (`spec`, `adversarial`); merge; then T08, T09, T10.
4. When T02 is ready: one reviewer; merge; then T05 (fixtures) and T18 (legacy description).
5. Pacing: weekly 49 % at 18:38 with the reset on 2026-09-27 16:59 PDT; bootstrap cost about 4 weekly points. Keep at most 3 authors; use sonnet for mechanical tasks; opus for the spec, the reference, the core modules and high-risk reviews.

## Running agents

- `t01-spec` (opus): T01 spec writer, issue #3, branches `t01-spec-part1` then `t01-spec-part2`, worktrees in `<S>/.worktrees/`. Started 18:45.
- `t02-harness` (sonnet): T02 earth harness, issue #4, branch `t02-harness` (or `t02-harness-r` and `t02-harness-compare`). Started 18:45.
- `t03-sims` (sonnet): T03 simulation harness, issue #5, branch `t03-sims`. Started 18:45.
- `drill-helper` (haiku): drill passed; done.

## Running jobs

None.

## Usage and resets

- 2026-09-25 16:19 PDT: 5-hour 19 % (resets 2026-09-25 19:59 PDT); weekly all models 45 % (resets 2026-09-27 16:59 PDT); weekly Fable 0 %.

## Guard notes

- The `dcg` hook blocks `rm -rf`, `git reset --hard`, `git clean`, `git checkout -- <file>`. It allows `git rm -r`, `git push --force-with-lease`, `gh pr merge --squash --delete-branch`, `git worktree remove`, `git rebase`, `git branch -d`, `mv`, `kill`. Check with `dcg test "<command>"`.
- The app's worktree guard blocks Write and Edit outside the session worktree, also after request_directory for `<main>`. change_directory is refused for a worktree session.
- The auto-mode classifier denied `run_scheduled_task`.
