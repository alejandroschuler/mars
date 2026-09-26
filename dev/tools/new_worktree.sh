#!/usr/bin/env bash
# Make a task worktree (VALIDATION_PLAN.md, "Appendix: bootstrap commands and
# prompts"): a worktree for <branch> from origin/main or <base>, then the dev venv.
# Usage: dev/tools/new_worktree.sh [--parent <folder>] <branch> [<base>]
# The parent folder is --parent, else $PYMARS_WORKTREES, else <main>/.worktrees,
# where <main> is the main clone. Safe to run again: it reuses the worktree, the
# local branch or origin/<branch> when one of them exists.
set -euo pipefail
parent=${PYMARS_WORKTREES:-}
if [ "${1:-}" = "--parent" ]; then
  parent=${2:?"--parent needs a folder"}
  shift 2
fi
branch=${1:?"usage: new_worktree.sh [--parent <folder>] <branch> [<base>]"}
base=${2:-origin/main}
common=$(git -C "$(dirname "$0")" rev-parse --path-format=absolute --git-common-dir)
main=$(dirname "$common")
path=${parent:-$main/.worktrees}/$branch

git -C "$main" fetch --quiet origin
if [ -e "$path/.git" ]; then
  current=$(git -C "$path" branch --show-current)
  if [ "$current" != "$branch" ]; then
    echo "$path holds the branch '$current', not '$branch'" >&2
    exit 1
  fi
  echo "reusing the worktree $path"
elif git -C "$main" show-ref --verify --quiet "refs/heads/$branch"; then
  git -C "$main" worktree add "$path" "$branch"
elif git -C "$main" show-ref --verify --quiet "refs/remotes/origin/$branch"; then
  git -C "$main" worktree add --track -b "$branch" "$path" "origin/$branch"
else
  git -C "$main" worktree add --no-track -b "$branch" "$path" "$base"
fi

cd "$path"
. dev/env.sh
uv sync --frozen --group dev --python 3.12
echo "ready: $path (branch $branch)"
