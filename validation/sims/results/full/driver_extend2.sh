#!/usr/bin/env bash
# T21 extension, round 2 (Part of #24): D1 n5000 hi and D6 n1000, sized again
# from the extended s_g. Same frozen runner, nested seeds, --resume.
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
run_block "D1 n5000 hi 400:600" --cells D1_n05000_hi --reps 400:600
run_block "D6 n1000 3300:3800" --cells D6_n01000 --reps 3300:3800
echo "=== $(date -u +%FT%TZ) ALL BLOCKS DONE ==="
