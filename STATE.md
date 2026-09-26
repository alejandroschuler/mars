# Executor state

Work done: no

Updated: 2026-09-26 04:34 PDT, by executor session `local_a59469db-c3f1-480c-81c4-a4c7beae39f4`.

This file and LOG.md are the executor journal (branch `executor`; only the executor pushes). A new executor rebuilds its state from this file, LOG.md, the issues and pull requests of `alejandroschuler/mars`, ListAgents and the job manifests.

## Paths

- `<main>` = `/Users/aschuler/Documents/research/projects/pymars`.
- `<S>` = this executor's session worktree, `<main>/.claude/worktrees/competent-poincare-5a4d49`. The desktop app lets the Write and Edit tools change files only inside the session worktree, so this executor keeps the journal worktree at `<S>/.worktrees/journal` and task worktrees at `<S>/.worktrees/<branch>`. An executor that runs in `<main>` itself (for example from the watchdog) uses `<main>/.worktrees/` as the plan says.
- Lock, heartbeat, caffeinate PID, gate logs: `<main>/.git/pymars-executor/` (shared by all worktrees).
- Briefs: `briefs/` in this journal.

## Phase

P0 done (recovery drill passed). P1 in progress since 18:45 PDT 2026-09-25.

Board: T00 #2 (closed), T01 #3, T02 #4, T03 #5, T04 #6, T05 #7, T06 #8, T07 #9, T08 #10, T09 #11, T10 #12, T11 #13, T12 #14, T13 #15, T14 #16, T15 #17, T16 #18, T17 #19, T18 #20, T19 #21, T20 #22, T21 #23, T22 #24, T23 #25, T24 #26; later: missing values #27, negative minspan #28, newvar.penalty #29, linpreds #30, allowed #31, pmethod and nfold #32, evimp #33. `board/numbers.json` has the map.

main: dbe5794 (PR #35, T03 part 1) on top of 31e6c13 (PR #1, bootstrap). CI runs on the fork.

Open pull requests (21:45):
- #34 T01 spec part 1, head d556e28; round 3 dual review running (workflow run wf_95bda7f4-7a6). Merge needs APPROVE from roles `spec` and `adversarial` on the head: `bash <main>/dev/tools/merge_pr.sh --session <id> 34 spec adversarial`. Before merging it, retarget #37 to main (`gh pr edit 37 --repo alejandroschuler/mars --base main`).
- #37 T01 spec part 2, head 0c4121d, stacked on #34; ready; its dual review starts after #34 merges (review `git diff d556e28..HEAD`). Rebase onto main after #34 merges.
- #36 T02 part 1, head fd71f83; #38 T02 part 2, head 3173b3e (base already main); re-review by rev-t02-harness (role `single`) running. Merge #36 first, then rebase #38 onto main, quick re-approval by range-diff, merge #38.
- #40 T03 part 2, head d5d4c54; re-review by rev-pr35-sims1 (role `single`) running. After it merges, start T04 (brief `briefs/T04-legacy-runs.md`).

## Done in P0

- Lock `<main>/.git/pymars-executor/lock` taken; heartbeat file in it.
- `gh repo set-default alejandroschuler/mars`; upstream push URL is `DISABLED`; `.worktrees/` in `.git/info/exclude`.
- `validation-plan` pushed to the fork; tag `legacy-1.0.4-head` (d68b54a) pushed.
- `caffeinate -dimsu` detached, PID in `<main>/.git/pymars-executor/caffeinate.pid`.
- Planning prototypes are on `validation-plan` in `validation/legacy/`; a copy is in `<main>/.git/pymars-executor/legacy-prototypes/exp/`.
- Text of Milborrow's notes saved for local reading at `<main>/.git/pymars-executor/ref/earth-notes.txt` (never commit or quote it).
- Watchdog task `pymars-executor-watchdog` created (cron `17 * * * *`, the app shows 25 past the hour; notifyOnCompletion false). Its folder and mode are not checked yet: `run_scheduled_task` was denied by the auto-mode classifier, so the user was asked to click Run now once.

## Next actions

1. T05: dual review of #42 and #43 running (run wf_a0b1ef34-9d5). Fixes by `t05-fixtures` (SendMessage); merge #42 first (#43's base is already main), then rebase #43 and a rebase-only re-approval (small sonnet checker, as for #40), then merge #43.
2. Pacing: weekly 86 % at 04:22 Saturday. After T05, start no new work until the weekly reset on 2026-09-27 17:00 PDT; keep the heartbeat fresh with short wakes; keep checking the legacy run (detached; no usage).
3. After the reset, start in parallel (at most 4 authors): T06 reference (brief `briefs/T06-reference.md`, opus), T08 and T09 (brief `briefs/T08-T10-components.md`, opus), and T18 legacy description (brief not written; sonnet), plus the T04 pilot analysis (steps 4 of `briefs/T04-legacy-runs.md`, sonnet, cheap). T10 after T08 and T09. T07 after T06's first parts. T01 v2 (#44) after T07.
4. When the legacy run ends (about 16 to 17 hours after 01:47 Saturday): T04 step 6 (collect, summarize, mark #41 ready; sonnet).
5. Tooling follow-ups for one small PR: merge_pr.sh refuses while another open PR uses the head branch as its base; accept a rebase-only re-approval by range-diff; the sims README pgrep pattern and stale run.lock steps (T03 review); DECISIONS.md line for rebase-only re-approvals.

## Running agents

- `t01-spec` (opus): done with spec v1 parts 1 and 2; waiting for review findings (SendMessage to it).
- `t02-harness` (sonnet): waiting for the re-review of #36 and #38.
- `t03-sims` (sonnet): waiting for the re-review of #40.
- `rev-t02-harness` (opus): re-reviewing #36 and #38.
- `rev-pr35-sims1` (opus): re-reviewing #40 (it also reviewed #35).
- Workflow run wf_95bda7f4-7a6: round 3 of #34 (spec + adversarial). The reusable script is `tools/dual_review.js` in this journal (args: pr, title, head, brief, roles, focus).

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
