# Executor state

Work done: no

Updated: 2026-09-27 19:00 PDT, by executor session `local_a59469db-c3f1-480c-81c4-a4c7beae39f4`.

This file and LOG.md are the executor journal (branch `executor`; only the executor pushes). A new executor rebuilds its state from this file, LOG.md, the issues and pull requests of `alejandroschuler/mars`, ListAgents and the job manifests.

## Paths

- `<main>` = `/Users/aschuler/Documents/research/projects/pymars`.
- `<S>` = this executor's session worktree, `<main>/.claude/worktrees/competent-poincare-5a4d49`. The desktop app lets the Write and Edit tools change files only inside the session worktree, so this executor keeps the journal worktree at `<S>/.worktrees/journal` and task worktrees at `<S>/.worktrees/<branch>`. An executor that runs in `<main>` itself (for example from the watchdog) uses `<main>/.worktrees/` as the plan says.
- Lock, heartbeat, caffeinate PID, gate logs: `<main>/.git/pymars-executor/` (shared by all worktrees).
- Briefs: `briefs/` in this journal.

## Phase

RUNNING since the weekly reset at 17:00 PDT Sunday 2026-09-27 (weekly 0 %; next reset 2026-10-04 17:00 PDT). Pace at about 14 % of the weekly limit per day; sonnet for mechanical work.

main: d6cac73 (#43 T05 datasets), 288d93c (#42 T05 components), a8dd024 (#37 spec part 2), b0d642d (#34 spec part 1), d4f74f4 (#38 T02), 8760cee (#40), dbe5794 (#35), 31e6c13 (#1).
Done: T00 (#2), T01 v1 (#3; v2 is #44), T02 (#4), T03 (#5), T05 (#7).

Open pull requests, all waiting for reviews after the reset:
- #45 T09 `_linalg.py` (head 4958fd0; gate B passed; 100 % coverage; 35 of 35 mutants caught). First: `t09-linalg` adds the lm_fit_coefficients fixture test (its code is in PR comment 5848489320) and rebases on main; then the dual review (spec, adversarial).
- #46 to #49 T06 reference, stacked: #46 part 1 (4c6f6e2, base main), #47 part 2 (3868316), #48 part 3 (403ed6a), #49 part 4 (4080a44, Closes #8). `t06-reference` rebases after each merge and adds a committed test on the component fixtures (now on main). Dual review each; before merging a part, retarget the next part to main.
- #41 T04 draft: the run configuration and `pilot_legacy.md` (head 9855f80). Ready when the legacy run ends and the results are collected (T04 step 6).
Board: T04 #6, T06 #8, T07 #9, T08 #10, T09 #11, T10 #12, T11 #13, T12 #14, T13 #15, T14 #16, T15 #17, T16 #18, T17 #19, T18 #20, T19 #21, T20 #22, T21 #23, T22 #24, T23 #25, T24 #26, T01 v2 #44; later #27 to #33.

## Done in P0

- Lock `<main>/.git/pymars-executor/lock` taken; heartbeat file in it.
- `gh repo set-default alejandroschuler/mars`; upstream push URL is `DISABLED`; `.worktrees/` in `.git/info/exclude`.
- `validation-plan` pushed to the fork; tag `legacy-1.0.4-head` (d68b54a) pushed.
- `caffeinate -dimsu` detached, PID in `<main>/.git/pymars-executor/caffeinate.pid`.
- Planning prototypes are on `validation-plan` in `validation/legacy/`; a copy is in `<main>/.git/pymars-executor/legacy-prototypes/exp/`.
- Text of Milborrow's notes saved for local reading at `<main>/.git/pymars-executor/ref/earth-notes.txt` (never commit or quote it).
- Watchdog task `pymars-executor-watchdog` created (cron `17 * * * *`, the app shows 25 past the hour; notifyOnCompletion false). Its folder and mode are not checked yet: `run_scheduled_task` was denied by the auto-mode classifier, so the user was asked to click Run now once.

## Next actions

Until the reset: a wake every 50 minutes to refresh the heartbeat and check the legacy run; no agents.
After the reset (Sunday 17:00 PDT), at most 4 authors and 3 reviewers:
1. `t09-linalg`: add the fixture test to #45, rebase; then the dual review of #45; merge.
2. Dual review of #46 (T06 part 1); merge (retarget #47 to main first); then #47, #48, #49 in order, each after `t06-reference` rebases it.
3. T08 (opus, `briefs/T08-T10-components.md`, issue #10) at once; T10 when T08 and T09 are merged.
4. When the legacy run ends: T04 step 6 (sonnet): collect, summarize, mark #41 ready; one reviewer.
5. T07 conformance (brief not written; needs T06 and T05, both near): write the brief after #46 merges. T18 legacy description (brief not written; sonnet). T01 v2 (#44) after T07.
6. Small follow-up PRs: T05 notes (X_test outside the training range; the missing-file skip in the pair test; the multinom stability bar and docstring; a README sentence that T10 compares rss_per_subset, gcv_per_subset and prune_terms at every size); tooling (merge_pr.sh refuses while another open PR uses the head branch as its base; accepts a rebase-only re-approval by range-diff; the sims README pgrep pattern and stale run.lock steps; a DECISIONS.md line for rebase-only re-approvals).
Pacing: this week (bootstrap to now) cost about 51 weekly points for T00 to T03, T05, spec v1, T06 and T09 authoring. Keep sonnet for mechanical work.

## Running agents

- `t09-linalg` (opus): adding the lm_fit_coefficients fixture test to #45 and rebasing; then the dual review of #45.
- Workflow run wf_840068d4-87c: dual review of #46 (T06 part 1). Before merging #46, retarget #47 to main; then `t06-reference` rebases #47 and so on.
- `t08-terms-gcv-knots` (opus): T08, issue #10.
- `rev-pr41-legacy` (sonnet): single review of #41 (T04 results).
- Resumable: `t06-reference` (for review fixes and rebases of #47 to #49), `t05-fixtures` (T05 follow-up PR), `t01-spec` (spec v2, #44).

## Running jobs

- None. The legacy full run (T04) finished at 16:13 PDT Saturday 2026-09-26 (ALL BLOCKS DONE); results in `<main>/.worktrees/runner-legacy/validation/runs/legacy_full` (49,441 files), logs `driver.log` (6 workers) and `driver_n4.log` (4 workers). Keep the runner worktree until T04 step 6 has collected the results into `validation/sims/results/legacy/`.

## Usage and resets

- 2026-09-25 16:19 PDT: 5-hour 19 % (resets 2026-09-25 19:59 PDT); weekly all models 45 % (resets 2026-09-27 16:59 PDT); weekly Fable 0 %.

## Guard notes

- The `dcg` hook blocks `rm -rf`, `git reset --hard`, `git clean`, `git checkout -- <file>`. It allows `git rm -r`, `git push --force-with-lease`, `gh pr merge --squash --delete-branch`, `git worktree remove`, `git rebase`, `git branch -d`, `mv`, `kill`. Check with `dcg test "<command>"`.
- The app's worktree guard blocks Write and Edit outside the session worktree, also after request_directory for `<main>`. change_directory is refused for a worktree session.
- The auto-mode classifier denied `run_scheduled_task`.
