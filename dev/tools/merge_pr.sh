#!/usr/bin/env bash
# Squash-merge a pull request of alejandroschuler/mars when every merge rule of
# VALIDATION_PLAN.md ("Pull requests and reviews") holds. Read it before you run it.
# Usage: dev/tools/merge_pr.sh [--dry-run] [--require-gate-c] [--allow-no-checks]
#                              [--session <id>] <pr-number> <role> [<role> ...]
# The checks:
# - the pull request is open, not a draft and based on main, and its title (the
#   squash subject) is a conventional commit subject;
# - no open pull request has the head branch as its base; --delete-branch would
#   close such a pull request once the branch is gone, so it must be retargeted
#   to main first;
# - for each role, the newest verdict is APPROVE for the head, and no
#   REQUEST_CHANGES for the head is newer. A verdict is the first line of a
#   comment or review by the fork's account (every agent posts as that account),
#   exactly "REVIEW <role> <sha>: APPROVE" or "... REQUEST_CHANGES", with the
#   full 40-character SHA;
# - the gate B log of the head passed (and the gate C log with --require-gate-c);
# - every check on the head passed; no checks at all fails, unless
#   --allow-no-checks is given;
# - the head contains origin/main;
# - fencing: <git-common-dir>/pymars-executor/lock/owner holds the session ID
#   from --session or $EXECUTOR_SESSION.
# --dry-run prints the checks and does not merge.
# pass() always succeeds, so "test && pass || fail" is safe (SC2015), and the
# jq programs are in single quotes on purpose (SC2016).
# shellcheck disable=SC2015,SC2016
set -uo pipefail
REPO=alejandroschuler/mars
TRAILER=${MERGE_TRAILER:-"Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"}
dry_run=0 gate_c=0 no_checks_ok=0 session=${EXECUTOR_SESSION:-} args=()
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run) dry_run=1 ;;
    --require-gate-c) gate_c=1 ;;
    --allow-no-checks) no_checks_ok=1 ;;
    --session) session=${2:-} && shift ;;
    -*) echo "unknown option: $1" >&2 && exit 2 ;;
    *) args+=("$1") ;;
  esac
  shift
done
if [ ${#args[@]} -lt 2 ]; then
  echo "usage: merge_pr.sh [--dry-run] [--require-gate-c] [--allow-no-checks] [--session <id>] <pr> <role>..." >&2
  exit 2
fi
pr=${args[0]} roles=("${args[@]:1}")
here=$(dirname "$0")
state=$(git -C "$here" rev-parse --path-format=absolute --git-common-dir)/pymars-executor
failed=0
pass() { echo "PASS $*"; }
fail() { echo "FAIL $*" && failed=1; }

fields=$(gh pr view "$pr" --repo "$REPO" --json state,isDraft,baseRefName,headRefName,headRefOid,title \
  --jq '[.state, .isDraft, .baseRefName, .headRefName, .headRefOid, .title] | @tsv') || exit 1
IFS=$'\t' read -r pr_state draft base head_ref head title <<<"$fields"
[ "$pr_state" = OPEN ] && pass "#$pr is open" || fail "#$pr is $pr_state"
[ "$draft" = false ] && pass "#$pr is not a draft" || fail "#$pr is a draft"
[ "$base" = main ] && pass "#$pr is based on main" || fail "#$pr is based on $base"
conventional='^(build|chore|ci|docs|feat|fix|perf|refactor|revert|style|test)(\([^)]+\))?!?: .+'
[[ $title =~ $conventional ]] && pass "conventional title: $title" || fail "not a conventional title: $title"

# --delete-branch removes head_ref on merge, which closes any open pull request
# based on it (this happened to #39). Refuse until such pull requests are
# retargeted to main.
if stacked=$(gh pr list --repo "$REPO" --state open --base "$head_ref" --json number --jq '.[] | "#\(.number)"'); then
  if [ -z "$stacked" ]; then
    pass "no open pull request is based on $head_ref"
  else
    fail "retarget these to main first, or --delete-branch would close them: $(tr '\n' ' ' <<<"$stacked")"
  fi
else
  fail "gh pr list --base $head_ref failed"
fi

# One line per verdict, oldest first: "<time> <role> <sha> <verdict>". Only the
# first line of a body by the fork's account (alejandroschuler) can be a verdict.
reviews=$(gh pr view "$pr" --repo "$REPO" --json comments,reviews --jq '
  [(.comments[] | select(.author.login == "alejandroschuler") | {at: .createdAt, body}),
   (.reviews[] | select(.submittedAt and .author.login == "alejandroschuler")
    | {at: .submittedAt, body})]
  | sort_by(.at) | .[] | .at as $at | .body | split("\n") | .[0] | sub("\r$"; "")
  | capture("^REVIEW (?<role>[A-Za-z0-9_.-]+) (?<sha>[0-9a-f]{40}): (?<verdict>APPROVE|REQUEST_CHANGES) *$")
  | "\($at) \(.role) \(.sha) \(.verdict)"') || exit 1
for role in "${roles[@]}"; do
  newest=$(awk -v r="$role" '$2 == r' <<<"$reviews" | tail -n 1)
  read -r at _ sha verdict <<<"$newest"
  if [ -n "$newest" ] && [ "$sha" = "$head" ] && [ "$verdict" = APPROVE ]; then
    pass "$role approved $head at $at"
    later=$(awk -v r="$role" -v t="$at" -v h="$head" \
      '$2 == r && $4 == "REQUEST_CHANGES" && $3 == h && $1 > t' <<<"$reviews")
    [ -z "$later" ] || fail "$role asked for changes after its approval: $later"
  else
    fail "the newest $role review is '${newest:-none}', not APPROVE for $head"
  fi
done

gate_ok() { [ -f "$1" ] && [ "$(tail -n 1 "$1")" = "$2" ]; }
log=$state/gates/$head.gateB.log
gate_ok "$log" "GATE B PASS $head" && pass "gate B passed ($log)" || fail "no passing gate B log at $log"
if [ "$gate_c" = 1 ]; then
  log=$state/gates/$head.gateC.log
  gate_ok "$log" "GATE C PASS $head" && pass "gate C passed ($log)" || fail "no passing gate C log at $log"
fi

checks=$(gh pr view "$pr" --repo "$REPO" --json statusCheckRollup --jq '.statusCheckRollup[]
  | "\(.status // "-")\t\(.conclusion // .state // "-")\t\(.name // .context)"') || exit 1
if [ -z "$checks" ] && [ "$no_checks_ok" = 1 ]; then
  echo "NOTE #$pr has no checks, which --allow-no-checks accepts"
elif [ -z "$checks" ]; then
  fail "#$pr has no checks, so CI did not run for the head (--allow-no-checks accepts that)"
else
  pending=$(awk -F'\t' '!(($1 == "COMPLETED" || $1 == "-") && $2 ~ /^(SUCCESS|NEUTRAL|SKIPPED)$/)' <<<"$checks")
  [ -z "$pending" ] && pass "all $(wc -l <<<"$checks" | tr -d ' ') checks passed" ||
    fail "checks that did not pass: $(tr '\t\n' ' ;' <<<"$pending")"
fi

git -C "$here" fetch --quiet origin main "$head_ref" || fail "git fetch of main and $head_ref failed"
git -C "$here" merge-base --is-ancestor origin/main "$head" 2>/dev/null &&
  pass "the head contains origin/main" || fail "the head does not contain origin/main; rebase first"

owner=$(cat "$state/lock/owner" 2>/dev/null)
[ -n "$session" ] && [ "$owner" = "$session" ] && pass "the executor lock is held by $session" ||
  fail "the executor lock owner '$owner' is not the session '${session:-unset}'"

if [ "$failed" -ne 0 ]; then
  echo "not merged: a check failed" && exit 1
elif [ "$dry_run" = 1 ]; then
  echo "dry run: every check passed; not merged" && exit 0
fi
body="Reviews: ${roles[*]}, all APPROVE on ${head:0:12}. Gate B passed on that head.

$TRAILER"
gh pr merge "$pr" --repo "$REPO" --squash --delete-branch --match-head-commit "$head" \
  --subject "$title (#$pr)" --body "$body"
result=$(gh pr view "$pr" --repo "$REPO" --json state,mergeCommit --jq '"\(.state) \(.mergeCommit.oid // "")"')
echo "after the merge: $result"
[ "${result%% *}" = MERGED ]
