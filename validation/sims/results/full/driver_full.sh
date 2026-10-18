#!/usr/bin/env bash
# T21 full run at freeze 0274c5a (issue #23): P-fix and E-def on all 51 cells,
# n_sim by tier from the post-freeze pilot (300, 600, 1300 or 3000 per cell).
# Restart-safe: every block uses --resume; seeds come from cell name and rep,
# so the tiers nest. Start with
#   nohup caffeinate -i nice -n 15 bash validation/runs/driver_full.sh >> validation/runs/full/driver.log 2>&1 &
set -euo pipefail
trap 'echo "=== $(date -u +%FT%TZ) ABORTED ==="' ERR
RUNNER=/Users/aschuler/Documents/research/projects/pymars/.worktrees/runner-t21
OUT="$RUNNER/validation/runs/full"
cd "$RUNNER"; . dev/env.sh; export NUMBA_NUM_THREADS=1; mkdir -p "$OUT"
run_block() {
  local desc=$1; shift
  echo "=== $(date -u +%FT%TZ) START $desc ==="
  uv run --frozen python -m validation.sims.run --out "$OUT" --arms P-fix,E-def "$@" --n-jobs 6 --resume
  echo "=== $(date -u +%FT%TZ) DONE  $desc ==="
}
T600=D1_n00200_hi,D1_n00200_lo,D3-bin_n00200,D5_n00200_hi,D5_n00200_lo,D5_n01000_hi,D5_n01000_lo,D5_n05000_hi,D6_n00200,D6_n01000,D6_n05000,D8_n00200_hi,D8_n00200_lo
T1300=D1_n00200_hi,D1_n00200_lo,D3-bin_n00200,D5_n00200_hi,D6_n00200,D6_n01000,D6_n05000
T3000=D1_n00200_hi,D6_n00200,D6_n01000
run_block "all 51 cells reps 0:300" --reps 0:300
run_block "tier 600 reps 300:600" --cells "$T600" --reps 300:600
run_block "tier 1300 reps 600:1300" --cells "$T1300" --reps 600:1300
run_block "tier 3000 reps 1300:3000" --cells "$T3000" --reps 1300:3000
echo "=== $(date -u +%FT%TZ) ALL BLOCKS DONE ==="
