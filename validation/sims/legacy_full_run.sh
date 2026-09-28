#!/usr/bin/env bash
# T04 (issue #6): the legacy full run, in order (VALIDATION_PLAN.md,
# "Statistical performance study" > "Pilot and number of repetitions" and
# "Compute ledger"; briefs/T04-legacy-runs.md steps 2-3). Runs each block of
# the legacy full run in sequence against a fixed-SHA runner worktree, with
# `--resume` throughout, so restarting this script (after a kill, a crash, or
# the machine sleeping) loses nothing: every already-finished (cell, arm,
# repetition) unit is skipped near-instantly, and the script picks up wherever
# it stopped.
#
# Usage (from the runner worktree, after `. dev/env.sh`, with both venvs
# built): `bash validation/sims/legacy_full_run.sh`. Meant to be started once
# under `nohup caffeinate -i nice -n 15 bash validation/sims/legacy_full_run.sh
# > <out>/driver.log 2>&1 &`, with `echo $! > <out>/driver.pid`.
#
# All ten arms in this study are registered in learners.py; this script only
# chooses which (arms, cells, repetitions) each block asks run.py for. The
# order follows the brief exactly: the pilot cells (D4, D5 at 200 cases, both
# noise levels) first, split into reps 0:100 then 100:300 so that all four
# pilot cells' first 100 repetitions (the ones a later agent's pilot_check.py
# needs) finish before any of them reaches repetition 100 - not just before
# the rest of the script runs. Then the other 200-case regression cells
# (without D7, which is far too expensive for the legacy arms - see the
# Compute ledger), the 200-case binary cells (EarthClassifier to the full
# cap, GLMEarth on only 20 repetitions to confirm it stays bitwise-identical
# to EarthClassifier in 1.0.4, a finding from the T03 review), the four
# 1,000-case cells, and finally the cheap arms (E-def matching whatever
# legacy arms just ran on each cell, E-pym on the gap-claim cells, OLS and
# HGB on all 51 cells). The low-priority batch and D7 for the legacy arms are
# deliberately not here (brief: "not now").
set -euo pipefail
trap 'echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) ABORTED (see above for the failing command) ==="' ERR

RUNNER=/Users/aschuler/Documents/research/projects/pymars/.worktrees/runner-legacy
OUT="$RUNNER/validation/runs/legacy_full"
N_JOBS=6

cd "$RUNNER"
. dev/env.sh
mkdir -p "$OUT"

run_block() {
  local desc=$1
  shift
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) START $desc ==="
  echo "+ uv run --frozen python -m validation.sims.run --out $OUT $* --n-jobs $N_JOBS --resume"
  uv run --frozen python -m validation.sims.run --out "$OUT" "$@" --n-jobs "$N_JOBS" --resume
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) DONE  $desc ==="
}

# --- Pilot cells first: D4, D5 at 200 cases, both noise levels -------------
# P-cur, P-ear, E-def, E-pym; split 0:100 / 100:300 so every pilot cell's
# first 100 repetitions (seeded by name, not loop position) are on disk
# before any of the four reaches repetition 100.
run_block "pilot D4,D5 n200 P-cur,P-ear,E-def,E-pym reps 0:100" \
  --arms P-cur,P-ear,E-def,E-pym --dgps D4,D5 --sizes 200 --reps 0:100
run_block "pilot D4,D5 n200 P-cur,P-ear,E-def,E-pym reps 100:300" \
  --arms P-cur,P-ear,E-def,E-pym --dgps D4,D5 --sizes 200 --reps 100:300

# --- 200 cases, regression: the other cells without D7 ---------------------
run_block "200-case regression D1,D2,D3,D6,D8 P-cur,P-ear reps 0:300" \
  --arms P-cur,P-ear --dgps D1,D2,D3,D6,D8 --sizes 200 --reps 0:300

# --- 200 cases, binary ------------------------------------------------------
run_block "200-case binary D3-bin,D4-bin EarthClassifier reps 0:300" \
  --arms EarthClassifier --dgps D3-bin,D4-bin --sizes 200 --reps 0:300
run_block "200-case binary D3-bin,D4-bin GLMEarth reps 0:20 (confirm bitwise-identical)" \
  --arms GLMEarth --dgps D3-bin,D4-bin --sizes 200 --reps 0:20

# --- 1,000 cases: D3, D4, D5, D8 at the low-noise level ---------------------
run_block "1000-case D3,D4,D5,D8 lo P-cur,P-ear reps 0:200" \
  --arms P-cur,P-ear --dgps D3,D4,D5,D8 --sizes 1000 --noise lo --reps 0:200

# --- Cheap arms --------------------------------------------------------------
# E-def with the same repetitions as the legacy arms on their cells (D4, D5
# are already covered by the pilot block above; --resume skips them here).
run_block "cheap E-def on D1,D2,D3,D6,D8 n200 reps 0:300" \
  --arms E-def --dgps D1,D2,D3,D6,D8 --sizes 200 --reps 0:300
run_block "cheap E-def on D3-bin,D4-bin n200 reps 0:300" \
  --arms E-def --dgps D3-bin,D4-bin --sizes 200 --reps 0:300
run_block "cheap E-def on D3,D4,D5,D8 n1000 lo reps 0:200" \
  --arms E-def --dgps D3,D4,D5,D8 --sizes 1000 --noise lo --reps 0:200

# E-pym on the gap-claim cells only (D1-D8 at 200 cases both noise levels,
# and D3/D4/D5/D8 at 1,000 cases low noise); D4/D5 at 200 overlap the pilot
# block and are skipped again here by --resume.
run_block "cheap E-pym gap cells n200 reps 0:300" \
  --arms E-pym --dgps D1,D2,D3,D4,D5,D6,D7,D8 --sizes 200 --reps 0:300
run_block "cheap E-pym gap cells n1000 lo reps 0:200" \
  --arms E-pym --dgps D3,D4,D5,D8 --sizes 1000 --noise lo --reps 0:200

# OLS and HGB on all 51 cells, up to 300 repetitions per cell (no --dgps/
# --sizes/--noise filter: every cell dgps.all_cells() returns).
run_block "cheap OLS,HGB all 51 cells reps 0:300" \
  --arms OLS,HGB --reps 0:300

echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) ALL BLOCKS DONE ==="
