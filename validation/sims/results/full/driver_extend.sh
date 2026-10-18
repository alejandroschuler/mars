#!/usr/bin/env bash
# T21 extension (Part of #24): the parity cells that failed in the full run get
# more repetitions (nested seeds, --resume), same frozen runner 0274c5a.
#   nohup caffeinate -i nice -n 15 bash validation/runs/driver_extend.sh >> validation/runs/full/driver_extend.log 2>&1 &
set -euo pipefail
trap 'echo "=== $(date -u +%FT%TZ) ABORTED ==="' ERR
RUNNER=/Users/aschuler/Documents/research/projects/pymars/.worktrees/runner-t21
OUT="$RUNNER/validation/runs/full"
cd "$RUNNER"; . dev/env.sh; export NUMBA_NUM_THREADS=1
run_block() {
  local desc=$1; shift
  echo "=== $(date -u +%FT%TZ) START $desc ==="
  uv run --frozen python -m validation.sims.run --out "$OUT" --arms P-fix,E-def "$@" --n-jobs 6 --resume --no-cache
  echo "=== $(date -u +%FT%TZ) DONE  $desc ==="
}
run_block "D1 n1000 hi 300:900"  --cells D1_n01000_hi --reps 300:900
run_block "D1 n1000 lo 300:750"  --cells D1_n01000_lo --reps 300:750
run_block "D1 n5000 hi 300:400"  --cells D1_n05000_hi --reps 300:400
run_block "D5 n200 lo 600:900"   --cells D5_n00200_lo --reps 600:900
run_block "D6 n200 3000:7500"    --cells D6_n00200 --reps 3000:7500
run_block "D6 n1000 3000:3300"   --cells D6_n01000 --reps 3000:3300
run_block "D6 n5000 1300:12400"  --cells D6_n05000 --reps 1300:12400
echo "=== $(date -u +%FT%TZ) ALL BLOCKS DONE ==="
