#!/usr/bin/env bash
# Gate C (VALIDATION_PLAN.md, "Gates"): 30 minutes or less, 2 cores or fewer.
# - the oracle tests (tests/test_oracle.py) and the invariance tests
#   (tests/test_invariance.py, which T16 adds) at the hypothesis thorough
#   profile, the slow ones included, and the share of the oracle's fits that
#   stop at a near-tie where the two programs choose differently, next to the
#   plan's count ("Ties": fits with any compared step whose best and second
#   differ by less than 1e-7 of the RSS before the step), from the counts that
#   the oracle writes (PYMARS_ORACLE_TALLY); a warning line when S15 is above
#   the plan's 5 percent;
# - when the branch changes validation/harness/ (the diff from the merge base
#   with origin/main), the fixtures made again with R and earth, which must
#   reproduce exactly (validation/harness/gen_fixtures.py --check);
# - the benchmark smoke test of the plan (validation/bench/run.py --smoke: 3
#   fits, no slower than 1.2 times the stored baseline of main, with a numpy
#   calibration that scales the machine out; about 15 s).
# The log is <git-common-dir>/pymars-executor/gates/<head-sha>.gateC.log, and
# its last line is "GATE C PASS <sha>" or "GATE C FAIL <sha>", as
# dev/tools/merge_pr.sh --require-gate-c reads. The gate refuses to run with
# uncommitted changes, so that the log belongs to a commit.
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
. dev/env.sh

if [ -n "$(git status --porcelain)" ]; then
  echo "gate C: commit or remove these changes first:" >&2
  git status --short >&2
  exit 1
fi
sha=$(git rev-parse HEAD)
gates="$(git rev-parse --path-format=absolute --git-common-dir)/pymars-executor/gates"
mkdir -p "$gates" || exit 1
log="$gates/$sha.gateC.log"
partial="$log.partial.$$"
tally=$(mktemp -d "${TMPDIR:-/tmp}/pymars-gate-c.XXXXXX") || exit 1
failed=0

step() { # step <name> <command...>: run, log the output, record a failure
  local name=$1 status start=$SECONDS
  shift
  echo "== $name: $*" | tee -a "$partial"
  "$@" 2>&1 | tee -a "$partial"
  status=${PIPESTATUS[0]}
  echo "== $name: exit $status after $((SECONDS - start)) s" | tee -a "$partial"
  [ "$status" -eq 0 ] || failed=1
}

near_ties() { # the oracle's counts, summed over the pytest workers
  uv run --frozen --python 3.12 python - "$1" <<'EOF'
import collections, json, pathlib, sys

files = sorted(pathlib.Path(sys.argv[1]).glob("tally-*.json"))
if not files:
    sys.exit("no counts from the oracle tests")
total = collections.Counter()
for f in files:
    total.update(json.loads(f.read_text()))
for key in sorted(k for k in total if k.endswith(": fits")):
    group = key[: -len(": fits")]
    fits, stops = total[key], total[f"{group}: near-tie stops"]
    plan = total[f"{group}: plan near-ties"]
    print(f"{group}: {fits} fits, {total[f'{group}: steps compared']} steps "
          f"compared, {stops} stop at a near-tie ({100 * stops / fits:.1f} %); "
          f"plan's count {plan} ({100 * plan / fits:.1f} %)")
    if "S15" in group and max(stops, plan) > 0.05 * fits:
        print(f"WARNING: {group} is above the plan's 5 % of near-ties (Ties)")
for key in sorted(k for k in total if "(" in k and "plan" not in k):
    print(f"  {key}: {total[key]}")
EOF
}

echo "gate C for $sha ($(git log -1 --format=%s)), $(date -u +%FT%TZ), $(uv --version)" |
  tee "$partial"

files=(tests/test_oracle.py)
if [ -f tests/test_invariance.py ]; then
  files+=(tests/test_invariance.py)
else
  echo "== tests/test_invariance.py is not in this checkout yet (T16)" | tee -a "$partial"
fi
step "oracle and invariance tests, thorough profile" env HYPOTHESIS_PROFILE=thorough \
  PYMARS_ORACLE_TALLY="$tally" uv run --frozen --python 3.12 --group validation \
  pytest -n 2 -q "${files[@]}"
step "near-tie share of the oracle's fits" near_ties "$tally"

base=$(git merge-base origin/main HEAD)
if git diff --quiet "$base" HEAD -- validation/harness/; then
  echo "== validation/harness/ has no change from origin/main (merge base $base)" |
    tee -a "$partial"
else
  step "fixtures made again with R" uv run --frozen --python 3.12 --group validation \
    python validation/harness/gen_fixtures.py --check
fi

step "benchmark smoke" uv run --frozen --python 3.12 python validation/bench/run.py --smoke

if [ -n "$(git status --porcelain)" ] || [ "$(git rev-parse HEAD)" != "$sha" ]; then
  echo "== the checkout changed while the gate ran" | tee -a "$partial"
  failed=1
fi
verdict=PASS
[ "$failed" -eq 0 ] || verdict=FAIL
echo "== total: $SECONDS s" | tee -a "$partial"
echo "GATE C $verdict $sha" | tee -a "$partial"
mv "$partial" "$log" && echo "log: $log"
[ "$failed" -eq 0 ]
