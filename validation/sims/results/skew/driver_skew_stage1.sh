#!/usr/bin/env bash
# Task t173-skew, stage 1 (screen and pilot). Does PR #185's pair rule raise
# pymars's excess risk against the old rule when the covariates are lognormal?
# Same script in both runner worktrees; the tree decides the arms.
#   runner-t173-skew-old  fdaca4a  P-fix            (old rule)
#   runner-t173-skew-new  ac67b38  P-fix and E-def  (new rule; earth 5.3.4 degree 2)
# Screen: D1 to D5, D7, D8 (all with signal), low noise, n = 200, reps 0:1000.
# Pilot for the confirm size: D4 low noise, n = 1000, reps 0:1000.
# No prediction cache, 4 workers per tree (8 in all). Restart-safe (--resume).
# Start with
#   nohup caffeinate -i nice -n 15 bash validation/runs/driver_skew_stage1.sh >> validation/runs/skew/driver.log 2>&1 &
set -euo pipefail
trap 'echo "=== $(date -u +%FT%TZ) ABORTED ==="' ERR
RUNNER="$(cd "$(dirname "$0")/../.." && pwd)"
case "$(basename "$RUNNER")" in
  runner-t173-skew-old) ARMS=P-fix;        SHA=fdaca4a1f5142bc601fffb44d4ef9862c6039591 ;;
  runner-t173-skew-new) ARMS=P-fix,E-def;  SHA=ac67b38abc35106a27235e61b07540da13a8fe94 ;;
  *) echo "unknown runner $RUNNER" >&2; exit 2 ;;
esac
OUT="$RUNNER/validation/runs/skew"
cd "$RUNNER"; . dev/env.sh; export NUMBA_NUM_THREADS=1; mkdir -p "$OUT"
echo $$ > "$RUNNER/validation/runs/skew-driver.pid"
echo "runner HEAD $(git rev-parse HEAD), arms $ARMS"
run_block() {
  local desc=$1; shift
  echo "=== $(date -u +%FT%TZ) START $desc ==="
  "$RUNNER/.venv/bin/python" -m validation.runs.skew_sims --out "$OUT" --arms "$ARMS" \
    --noise lo --n-jobs 4 --expect-sha "$SHA" "$@"
  echo "=== $(date -u +%FT%TZ) DONE  $desc ==="
}
run_block "screen n=200, reps 0:1000" --dgps D4,D1,D2,D3,D5,D7,D8 --sizes 200 --reps 0:1000
run_block "pilot D4 n=1000, reps 0:1000" --dgps D4 --sizes 1000 --reps 0:1000
echo "=== $(date -u +%FT%TZ) STAGE 1 DONE ==="
rm -f "$RUNNER/validation/runs/skew-driver.pid"
