#!/usr/bin/env bash
# Gate B (VALIDATION_PLAN.md, "Gates"): 10 minutes or less on 2 cores or fewer.
# - ruff and the uv.lock check;
# - the full suite at the hypothesis ci profile on Python 3.12, with coverage of
#   at least 90 percent on pymars/ (fail_under in pyproject.toml);
# - the fast tests on Python 3.13 and 3.14 at the dev profile, each in its own
#   project venv;
# - the tests of dev/tools/merge_pr.sh (dev/tools/test_merge_pr.sh);
# - uv build, then the wheel installed into a fresh venv and a smoke check.
# testpaths also covers validation/sims/tests and validation/harness/tests;
# "external" tests (need R and earth, or .venv-legacy) are left to gate C and
# a direct pytest invocation, since CI and the fast Python-version jobs have
# neither.
# Every selection above holds tests, so pytest exit code 5 (no tests) fails.
# The log is <git-common-dir>/pymars-executor/gates/<head-sha>.gateB.log, and
# its last line is "GATE B PASS <sha>" or "GATE B FAIL <sha>". The gate refuses
# to run with uncommitted changes, so that the log belongs to a commit.
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
. dev/env.sh

if [ -n "$(git status --porcelain)" ]; then
  echo "gate B: commit or remove these changes first:" >&2
  git status --short >&2
  exit 1
fi
sha=$(git rev-parse HEAD)
gates="$(git rev-parse --path-format=absolute --git-common-dir)/pymars-executor/gates"
mkdir -p "$gates" || exit 1
log="$gates/$sha.gateB.log"
partial="$log.partial.$$"
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

echo "gate B for $sha ($(git log -1 --format=%s)), $(date -u +%FT%TZ), $(uv --version)" |
  tee "$partial"

step "ruff check" uv run --frozen --python 3.12 ruff check
step "ruff format" uv run --frozen --python 3.12 ruff format --check
step "uv.lock" uv lock --check
step "full suite, Python 3.12" env HYPOTHESIS_PROFILE=ci \
  uv run --frozen --python 3.12 --group validation \
  pytest -n 2 -m "not external" --cov --cov-report=term-missing -q
for v in 3.13 3.14; do
  step "fast tests, Python $v" env UV_PROJECT_ENVIRONMENT=".venv-$v" \
    uv run --frozen --python "$v" --group validation \
    pytest -n 2 -m "not slow and not external" -q
done
step "merge_pr.sh tests" bash dev/tools/test_merge_pr.sh

# The build folder is reused; the --clear options empty it, so no rm is needed.
out=build/gate_b
smoke=$(
  cat <<'EOF'
import importlib.metadata
import pymars
assert pymars.__version__ == importlib.metadata.version("mars-earth")
assert "site-packages" in pymars.__file__, pymars.__file__
if hasattr(pymars, "Earth"):
    import numpy as np
    X = np.random.default_rng(0).uniform(size=(100, 3))
    y = X[:, 0] + np.maximum(X[:, 1] - 0.5, 0.0)
    assert np.isfinite(pymars.Earth().fit(X, y).predict(X)).all()
print("smoke check passed:", pymars.__version__, pymars.__file__)
EOF
)
step "build" uv build --clear --out-dir "$out/dist"
wheel=$(ls "$out"/dist/mars_earth-*-py3-none-any.whl 2>/dev/null)
step "pure-Python wheel" test -n "$wheel"
step "fresh venv" uv venv --clear --python 3.12 "$out/venv"
step "install the wheel" uv pip install --python "$out/venv/bin/python" "$wheel"
step "smoke check" "$out/venv/bin/python" -I -c "$smoke"

if [ -n "$(git status --porcelain)" ] || [ "$(git rev-parse HEAD)" != "$sha" ]; then
  echo "== the checkout changed while the gate ran" | tee -a "$partial"
  failed=1
fi
verdict=PASS
[ "$failed" -eq 0 ] || verdict=FAIL
echo "== total: $SECONDS s" | tee -a "$partial"
echo "GATE B $verdict $sha" | tee -a "$partial"
mv "$partial" "$log" && echo "log: $log"
[ "$failed" -eq 0 ]
