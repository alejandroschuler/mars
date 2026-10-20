#!/usr/bin/env bash
# I173 acceptance check (task t173-grid): the pymars arm (P-fix) of the whole T21
# parity grid at 286d9946cbff88b15cc3fbd53bf5a680ac340226 (PR #185, LA-7 threshold
# 0.01*prod(12*sigma_v^2)). Same cells, seeds and repetitions per cell as the stored
# study (validation/sims/results/full/parity.csv n_sim; seeds nest, so reps
# 0:n_sim). Arm P-fix only; E-def comes from the stored results. No prediction
# cache. Restart-safe (--resume). Start with
#   nohup caffeinate -i nice -n 15 bash validation/runs/driver_t173.sh >> validation/runs/t173-grid/driver.log 2>&1 &
set -euo pipefail
trap 'echo "=== $(date -u +%FT%TZ) ABORTED ==="' ERR
RUNNER=/Users/aschuler/Documents/research/projects/pymars/.worktrees/runner-t173
OUT="$RUNNER/validation/runs/t173-grid"
cd "$RUNNER"; . dev/env.sh; export NUMBA_NUM_THREADS=1; mkdir -p "$OUT"
echo $$ > "$RUNNER/validation/runs/t173-grid.pid"
echo "runner HEAD $(git rev-parse HEAD)"
run_block() {
  local desc=$1; shift
  echo "=== $(date -u +%FT%TZ) START $desc ==="
  "$RUNNER/.venv/bin/python" -m validation.sims.run --out "$OUT" --arms P-fix "$@" --n-jobs 8 --resume --no-cache
  echo "=== $(date -u +%FT%TZ) DONE  $desc ==="
}
run_block "all 51 cells reps 0:300" --reps 0:300
run_block "n_sim 600: reps 300:600" --cells D1_n05000_hi,D5_n01000_hi,D5_n01000_lo,D5_n05000_hi,D8_n00200_hi,D8_n00200_lo --reps 300:600
run_block "n_sim 750: reps 300:750" --cells D1_n01000_lo --reps 300:750
run_block "n_sim 900: reps 300:900" --cells D1_n01000_hi,D5_n00200_lo --reps 300:900
run_block "n_sim 1300: reps 300:1300" --cells D1_n00200_lo,D3-bin_n00200,D5_n00200_hi --reps 300:1300
run_block "n_sim 3000: reps 300:3000" --cells D1_n00200_hi --reps 300:3000
run_block "n_sim 3800: reps 300:3800" --cells D6_n01000 --reps 300:3800
run_block "n_sim 7500: reps 300:7500" --cells D6_n00200 --reps 300:7500
run_block "n_sim 12400: reps 300:12400" --cells D6_n05000 --reps 300:12400
echo "=== $(date -u +%FT%TZ) ALL BLOCKS DONE ==="
rm -f "$RUNNER/validation/runs/t173-grid.pid"
