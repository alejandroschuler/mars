#!/usr/bin/env bash
# Tests of dev/tools/merge_pr.sh. A fake gh answers from a JSON file, and the git
# checks run on a scratch repository in a new temporary folder, which the script
# leaves for the system to clean up. Every run is a dry run, nothing reaches
# GitHub, and nothing is pushed or deleted. Needs git and jq. Gate B runs it.
set -uo pipefail
root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel) || exit 1
dir=$(mktemp -d "${TMPDIR:-/tmp}/merge_pr_tests.XXXXXX") || exit 1
g() { git -C "$dir/work" -c user.name=test -c user.email=test@example.invalid "$@"; }
mkdir -p "$dir/bin" "$dir/work/dev/tools" || exit 1

# The scratch repository is its own origin: main holds one commit, and the
# branch t99-test adds a second one.
g init -q -b t99-test && g commit -q --allow-empty -m base && g branch main &&
  g commit -q --allow-empty -m head && g remote add origin "$dir/work" || exit 1
head=$(g rev-parse t99-test) old=$(g rev-parse main)
cp "$root/dev/tools/merge_pr.sh" "$dir/work/dev/tools/" || exit 1
state=$dir/work/.git/pymars-executor
mkdir -p "$state/lock" "$state/gates"
echo test-session >"$state/lock/owner"
echo "GATE B PASS $head" >"$state/gates/$head.gateB.log"

cat >"$dir/bin/gh" <<'EOF'
#!/usr/bin/env bash
# Fake gh: "pr view ... --jq <program>" runs the program on $FAKE_PR_JSON.
[ "$1 $2" = "pr view" ] || { echo "fake gh: unexpected call: $*" >&2; exit 1; }
while [ $# -gt 0 ]; do [ "$1" = --jq ] && prog=$2; shift; done
exec jq -r "$prog" "$FAKE_PR_JSON"
EOF
chmod +x "$dir/bin/gh"

entry() { # entry <time field> <login> <time> <body>: one comment or review
  printf '{"author": {"login": "%s"}, "%s": "%s", "body": %s}' "$2" "$1" "$3" "$(jq -Rs . <<<"$4")"
}
c() { entry createdAt "$@"; }
r() { entry submittedAt "$@"; }
pr() { # pr <comments> <reviews> [<checks>]: write the fake pull request
  local checks=${3:-'[{"name": "ci", "status": "COMPLETED", "conclusion": "SUCCESS"}]'}
  printf '{"state": "OPEN", "isDraft": false, "baseRefName": "main", "headRefName": "t99-test",
    "headRefOid": "%s", "title": "test: a fake pull request", "mergeCommit": null,
    "comments": [%s], "reviews": [%s], "statusCheckRollup": %s}' \
    "$head" "$1" "$2" "$checks" >"$dir/pr.json"
}
failed=0
expect() { # expect <label> <pattern the output must match> [<extra option>]
  local out
  out=$(cd "$dir/work" && PATH="$dir/bin:$PATH" FAKE_PR_JSON="$dir/pr.json" \
    bash dev/tools/merge_pr.sh --dry-run --session test-session ${3:+"$3"} 1 boot 2>&1)
  if grep -Eq "$2" <<<"$out"; then
    echo "ok   $1"
  else
    echo "FAIL $1" && echo "     ${out//$'\n'/$'\n'     }" && failed=1
  fi
}
me=alejandroschuler A="REVIEW boot $head: APPROVE" R="REVIEW boot $head: REQUEST_CHANGES"
passed='^dry run: every check passed'
newest='^FAIL the newest boot review'

pr "$(c $me 2026-01-01T01:00:00Z "$A")" ""
expect "an approval passes" "$passed"
pr "" "$(r $me 2026-01-01T01:00:00Z "$A")"
expect "an approval in a review body passes" "$passed"
pr "$(c $me 2026-01-01T01:00:00Z "$A"$'\r\nNo findings.')" ""
expect "a CRLF body passes" "$passed"
pr "$(c $me 2026-01-01T01:00:00Z "$R"), $(c someone-else 2026-01-01T01:05:00Z "$A")" ""
expect "an approval from another account does not count" "$newest.*REQUEST_CHANGES"
pr "$(c $me 2026-01-01T01:00:00Z "$R"$'\n\n> '"$A"$'\n'"$A")" ""
expect "an approval below a REQUEST_CHANGES first line does not count" "$newest.*REQUEST_CHANGES"
pr "$(c $me 2026-01-01T01:00:00Z "REVIEW boot ${head:0:7}: APPROVE")" ""
expect "a 7-character SHA does not count" "$newest is 'none'"
pr "$(c $me 2026-01-01T01:00:00Z "REVIEW boot $old: APPROVE")" ""
expect "an approval of an older head fails" "$newest.*$old APPROVE"
pr "$(c $me 2026-01-01T01:00:00Z "$A"), $(c $me 2026-01-01T02:00:00Z "$R")" ""
expect "a REQUEST_CHANGES newer than the approval fails" "$newest.*REQUEST_CHANGES"
pr "$(c $me 2026-01-01T01:00:00Z "$A")" "" "[]"
expect "no checks fail" "^FAIL #1 has no checks"
expect "no checks pass with --allow-no-checks" "$passed" --allow-no-checks
pr "$(c $me 2026-01-01T01:00:00Z "$A")" "" '[{"name": "ci", "status": "IN_PROGRESS", "conclusion": null}]'
expect "a pending check fails" "^FAIL checks that did not pass"

[ "$failed" -eq 0 ] && echo "all merge_pr.sh tests passed"
