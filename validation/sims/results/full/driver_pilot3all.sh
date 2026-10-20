#!/usr/bin/env bash
# T21 post-freeze pilot step 3 at freeze 0274c5a (issue #23). Restart-safe (--resume).
set -euo pipefail
trap 'echo "=== $(date -u +%FT%TZ) ABORTED ==="' ERR
RUNNER=/Users/aschuler/Documents/research/projects/pymars/.worktrees/runner-t21
OUT="$RUNNER/validation/runs/pilot3all"
cd "$RUNNER"; . dev/env.sh; export NUMBA_NUM_THREADS=1; mkdir -p "$OUT"
echo "=== $(date -u +%FT%TZ) START pilot3all ==="
uv run --frozen python -m validation.sims.run --out "$OUT" --arms P-fix,E-def \
  --reps 0:100 --n-jobs 6 --resume
echo "=== $(date -u +%FT%TZ) DONE pilot3all ==="
