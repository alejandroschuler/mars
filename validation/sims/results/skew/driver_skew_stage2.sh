#!/usr/bin/env bash
# Task t173-skew, stage 2. Same trees, arms, cells' seeds and settings as stage 1
# (see driver_skew_stage1.sh); the repetitions nest, so these blocks add to the
# stage-1 results.
#   Confirm (fixed in advance): D4 low noise, lognormal covariates, n = 200 and
#   n = 1000, reps 1000:7000 (7,000 in all; the screen's sd(g) = 0.224 at
#   n = 200 asks for 4,750 for a 3-SE half-width of 0.01, the pilot's 0.088 at
#   n = 1000 asks for 701; 7,000 is the brief's rough figure for both).
#   Extension (added after the screen, not in the fixed design): D8 low noise,
#   n = 200, reps 1000:23000. Its screen interval [0.985, 1.081] could not
#   decide the 1.05 margin; sd(g) = 0.487 asks for 22,755 for a half-width of 0.01.
# Each block runs under /usr/bin/time -l, which reports the CPU seconds of the
# main process and the workers it reaped.
# Start with
#   nohup caffeinate -i nice -n 15 bash validation/runs/driver_skew_stage2.sh >> validation/runs/skew/driver.log 2>&1 &
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
  /usr/bin/time -l "$RUNNER/.venv/bin/python" -m validation.runs.skew_sims --out "$OUT" --arms "$ARMS" \
    --noise lo --n-jobs 4 --expect-sha "$SHA" "$@"
  echo "=== $(date -u +%FT%TZ) DONE  $desc ==="
}
run_block "confirm D4 n=200 and n=1000, reps 1000:7000" --dgps D4 --sizes 200,1000 --reps 1000:7000
run_block "extension D8 n=200, reps 1000:23000" --dgps D8 --sizes 200 --reps 1000:23000
echo "=== $(date -u +%FT%TZ) STAGE 2 DONE ==="
rm -f "$RUNNER/validation/runs/skew-driver.pid"
