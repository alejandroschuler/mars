#!/usr/bin/env bash
# T21 part a (issue #23): the pre-freeze pilot on origin/main 02f788c.
# Restart-safe: every block uses --resume. Start with
#   nohup caffeinate -i nice -n 15 bash validation/runs/driver_t21a.sh >> validation/runs/prepilot/driver.log 2>&1 &
set -euo pipefail
trap 'echo "=== $(date -u +%FT%TZ) ABORTED ==="' ERR
RUNNER=/Users/aschuler/Documents/research/projects/pymars/.worktrees/runner-t21a
OUT="$RUNNER/validation/runs/prepilot"
N_JOBS=4
cd "$RUNNER"; . dev/env.sh; mkdir -p "$OUT"
run_block() {
  local desc=$1; shift
  echo "=== $(date -u +%FT%TZ) START $desc ==="
  uv run --frozen python -m validation.sims.run --out "$OUT" "$@" --n-jobs "$N_JOBS" --resume
  echo "=== $(date -u +%FT%TZ) DONE  $desc ==="
}
# Step 2: the diagnostics, recomputed and compared with the committed file.
if [ ! -f "$OUT/diagnostics_recomputed.json" ]; then
  echo "=== $(date -u +%FT%TZ) START diagnostics ==="
  uv run --frozen python -c "
import json
from validation.sims import diagnostics, dgps
d = diagnostics.compute_diagnostics()
json.dump(d, open('$OUT/diagnostics_recomputed.json', 'w'), indent=1, sort_keys=True)
old = json.loads(dgps.DIAGNOSTICS_PATH.read_text())
print('recomputed equals committed:', json.loads(json.dumps(d)) == old)
"
  echo "=== $(date -u +%FT%TZ) DONE  diagnostics ==="
fi
# Step 1: two repetitions in every one of the 51 cells, all new-code arms.
run_block "step1 all cells reps 0:2" --arms E-def,E-pym,OLS,HGB,P-fix --reps 0:2
# Gap in T04: E-def on D7 at the legacy cells' settings (200 cases, 300 reps).
run_block "E-def D7 n200 reps 0:300" --arms E-def --dgps D7 --sizes 200 --reps 0:300
# Step 3, parity pilot: 1,000 cases, D3 D4 D5 D7, both noise levels.
run_block "step3 parity D3,D4,D5,D7 n1000 P-fix,E-def reps 0:100" \
  --arms P-fix,E-def --dgps D3,D4,D5,D7 --sizes 1000 --reps 0:100
echo "=== $(date -u +%FT%TZ) ALL BLOCKS DONE ==="
