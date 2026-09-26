# Executor state

Work done: no

Updated: 2026-09-26 10:13 PDT, by executor session `local_a59469db-c3f1-480c-81c4-a4c7beae39f4`.

This file and LOG.md are the executor journal (branch `executor`; only the executor pushes). A new executor rebuilds its state from this file, LOG.md, the issues and pull requests of `alejandroschuler/mars`, ListAgents and the job manifests.

## Paths

- `<main>` = `/Users/aschuler/Documents/research/projects/pymars`.
- `<S>` = this executor's session worktree, `<main>/.claude/worktrees/competent-poincare-5a4d49`. The desktop app lets the Write and Edit tools change files only inside the session worktree, so this executor keeps the journal worktree at `<S>/.worktrees/journal` and task worktrees at `<S>/.worktrees/<branch>`. An executor that runs in `<main>` itself (for example from the watchdog) uses `<main>/.worktrees/` as the plan says.
- Lock, heartbeat, caffeinate PID, gate logs: `<main>/.git/pymars-executor/` (shared by all worktrees).
- Briefs: `briefs/` in this journal.

## Phase

RESUMED at 10:12 Saturday 2026-09-26: the user allowed weekly usage up to 99 % (start new work up to about 96 %; at about 98 % authors checkpoint and stop). Weekly reset 2026-09-27 17:00 PDT. P1 and P2 in progress.

main: a8dd024 (spec v1 complete: PR #34 and #37), d4f74f4 (T02 earth harness, #38), 8760cee (T03 part 2, #40), b0d642d, dbe5794 (T03 part 1, #35), 31e6c13 (bootstrap, #1). CI runs on the fork.

Done: T00 (#2), T01 v1 (#3; v2 is #44), T02 (#4), T03 (#5).
Open pull requests:
- #42 T05 component fixtures, head a8bf2e8, round-1 fixes done; needs round 2 of the dual review (roles `spec` and `adversarial`; reuse tools/dual_review.js with both PRs, as in run wf_a0b1ef34-9d5).
- #43 T05 datasets S02 to S20, head 359fe5c, base main (stacked on #42's branch); same review. Merge #42 first, then rebase #43 and a rebase-only re-approval by a small checker (as for #40), then merge #43.
- #41 T04 draft (run configuration); ready when the legacy run ends and results are collected.
Board: T00 #2 (closed), T01 #3 (closed), T01 v2 #44, T02 #4 (closed), T03 #5 (closed), T04 #6, T05 #7, T06 #8, T07 #9, T08 #10, T09 #11, T10 #12, T11 #13, T12 #14, T13 #15, T14 #16, T15 #17, T16 #18, T17 #19, T18 #20, T19 #21, T20 #22, T21 #23, T22 #24, T23 #25, T24 #26; later #27 to #33.

## Done in P0

- Lock `<main>/.git/pymars-executor/lock` taken; heartbeat file in it.
- `gh repo set-default alejandroschuler/mars`; upstream push URL is `DISABLED`; `.worktrees/` in `.git/info/exclude`.
- `validation-plan` pushed to the fork; tag `legacy-1.0.4-head` (d68b54a) pushed.
- `caffeinate -dimsu` detached, PID in `<main>/.git/pymars-executor/caffeinate.pid`.
- Planning prototypes are on `validation-plan` in `validation/legacy/`; a copy is in `<main>/.git/pymars-executor/legacy-prototypes/exp/`.
- Text of Milborrow's notes saved for local reading at `<main>/.git/pymars-executor/ref/earth-notes.txt` (never commit or quote it).
- Watchdog task `pymars-executor-watchdog` created (cron `17 * * * *`, the app shows 25 past the hour; notifyOnCompletion false). Its folder and mode are not checked yet: `run_scheduled_task` was denied by the auto-mode classifier, so the user was asked to click Run now once.

## Next actions

Until the reset: wake every 50 minutes; refresh the heartbeat; check the legacy run; start no agents.
After the reset (Sunday 17:00 PDT), in this order, at most 4 authors and 3 reviewers:
1. Round 2 of the dual review of #42 and #43; merge #42, then #43 (rebase-only re-approval), closing T05 (#7).
2. T08 and T09 (opus authors, brief `briefs/T08-T10-components.md`), T06 (opus, `briefs/T06-reference.md`; its first part can start before T05 merges), and T04 step 4 (the pilot analysis, sonnet, cheap).
3. When the legacy run ends: T04 step 6 (collect, summarize, mark #41 ready; sonnet); one reviewer.
4. T10 after T08 and T09; T18 (legacy description; brief not written; sonnet) after T05; T07 after T06's first parts; T01 v2 (#44) after T07.
5. One small tooling PR: merge_pr.sh refuses while another open PR uses the head branch as its base, and accepts a rebase-only re-approval by range-diff; the sims README pgrep pattern and stale run.lock steps; a DECISIONS.md line for rebase-only re-approvals.
Pacing next week: bootstrap-to-now cost about 45 weekly points for T00 to T03, T01 v1 and T05's first round. Use sonnet for mechanical work, opus for the reference, the core modules and the reviews that decide correctness.

## Running agents

- Workflow run wf_7ce0e408-2a3: round 2 of the dual review of #42 and #43 (T05). Started 10:13 Saturday.
- `t06-reference` (opus): T06 reference, issue #8, branches `t06-reference-part1` and later parts. Started 10:13.
- `t09-linalg` (opus): T09 `_linalg.py`, issue #11, branch `t09-linalg`. Started 10:13.
- `t04-pilot` (sonnet): T04 step 4, the pilot report `validation/sims/pilot_legacy.md` on branch `t04-legacy-runs` (draft PR #41). Started 10:14.
- Resumable later: `t05-fixtures` (for round-2 fixes), `t01-spec` (spec v2).

## Running jobs

- Legacy full run (T04, #6), started 2026-09-26 01:47 PDT by `t04-legacy`. Runner worktree `<main>/.worktrees/runner-legacy` at 8760cee (detached). Driver `validation/sims/legacy_full_run.sh` (12 `run.py --resume` blocks into one output folder), launched as `nohup caffeinate -i nice -n 15 bash validation/sims/legacy_full_run.sh > validation/runs/legacy_full/driver.log 2>&1 &` from the runner root. PID file `validation/runs/legacy_full/driver.pid` (65765). 6 workers. Expected about 98 core-hours, 16 to 17 hours wall. Order: pilot D4/D5 at 200 (reps 0:100, then 100:300), other 200-case regression cells without D7, 200-case binary, 1,000-case D3/D4/D5/D8 low noise, then the cheap arms.
- Check on each wake: `ps -p $(cat <runner>/validation/runs/legacy_full/driver.pid)`, `tail <runner>/validation/runs/legacy_full/driver.log`, the count of result files. Restart after a crash: end leftover loky workers (`pgrep -fl "<runner>/.venv/bin/python.*joblib.externals.loky"`), remove a stale `run.lock` folder with rmdir if no run.py uses the folder, and rerun the same launch command (every block resumes).
- Draft PR #41 (branch `t04-legacy-runs`, head 1c64432) holds the run configuration.

## Usage and resets

- 2026-09-25 16:19 PDT: 5-hour 19 % (resets 2026-09-25 19:59 PDT); weekly all models 45 % (resets 2026-09-27 16:59 PDT); weekly Fable 0 %.

## Guard notes

- The `dcg` hook blocks `rm -rf`, `git reset --hard`, `git clean`, `git checkout -- <file>`. It allows `git rm -r`, `git push --force-with-lease`, `gh pr merge --squash --delete-branch`, `git worktree remove`, `git rebase`, `git branch -d`, `mv`, `kill`. Check with `dcg test "<command>"`.
- The app's worktree guard blocks Write and Edit outside the session worktree, also after request_directory for `<main>`. change_directory is refused for a worktree session.
- The auto-mode classifier denied `run_scheduled_task`.
