#!/usr/bin/env bash
# autopush — snapshot the working tree onto refs/autopush/<branch>, without
# touching HEAD, the real index or the tree.
#
# Why it exists: this repo had no rolling snapshot of uncommitted work. In the
# sibling project (Improve-ImgSLI) two branch switches lost edits that lived only
# in the worktree; the mechanism below is that fix, adapted. The differences are
# listed so nobody has to diff the two scripts to find them:
#
#   * no push. This checkout's only remote is the local source repo, so a
#     snapshot is local by design (see projects/rve/docs/backlog/DELEGATION.md).
#   * retention. The sibling kept an unbounded chain; a 15-minute timer would
#     grow it forever, so the chain is bounded here (see "Retention" below).
#   * explicit identity. A timer run must not depend on the invoking user's
#     git config, and a scratch repo with no user.email must still work.
#   * `git -C`. Callable from any cwd, which is what systemd hands us.
#
# Usage:
#   scripts/autopush.sh [REPO]        # REPO defaults to $AUTOPUSH_REPO, then cwd
#   scripts/autopush.sh --repo PATH [--keep N] [--dry-run]
#
# Recovery:
#   git for-each-ref refs/autopush                  # what exists, and when
#   git log refs/autopush/<name> --oneline -5       # snapshot timeline
#   git show refs/autopush/<name>:<path>            # content of one file
#   git checkout refs/autopush/<name> -- <path>     # restore one path (your call,
#                                                   # this script never does it)
#
# Environment:
#   AUTOPUSH_REPO     repository path when no argument is given
#   AUTOPUSH_KEEP     snapshots kept per branch, default 48 (= 12 h at 15 min)
#   AUTOPUSH_TMPDIR   where the throwaway index is created; defaults to the git
#                     dir, and is redirected there if the override is inside the
#                     work tree (a temp file inside the tree would be snapshotted)
#
# Deliberately absent: `git stash` (per-repo global state, one writer at a time
# per machine — a timer must never take it), `git reset --hard`, any `checkout`,
# and any commit onto a branch. The snapshot is a detached object chain reachable
# only from refs/autopush/*, which is why a pre-PR squash cannot drop it.
set -euo pipefail

# A timer inherits a clean environment; a human shell may not. Detach from any
# caller-provided repo plumbing so `git -C "$REPO"` is the only thing that decides
# which repository and which index we touch.
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_OBJECT_DIRECTORY GIT_COMMON_DIR

PROG=autopush

say() { printf '%s: %s\n' "$PROG" "$*"; }
warn() { printf '%s: %s\n' "$PROG" "$*" >&2; }

usage() {
    sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
}

REPO_ARG=""
KEEP="${AUTOPUSH_KEEP:-48}"
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --repo)    REPO_ARG="${2:-}"; shift ;;
        --repo=*)  REPO_ARG="${1#--repo=}" ;;
        --keep)    KEEP="${2:-}"; shift ;;
        --keep=*)  KEEP="${1#--keep=}" ;;
        --dry-run) DRY_RUN=1 ;;
        -*)        warn "unknown option: $1"; exit 2 ;;
        *)         REPO_ARG="$1" ;;
    esac
    shift
done

case "$KEEP" in
    ''|*[!0-9]*) warn "AUTOPUSH_KEEP/--keep must be a positive integer, got '$KEEP'"; exit 2 ;;
esac
[ "$KEEP" -ge 1 ] || { warn "--keep must be >= 1"; exit 2; }

REPO="${REPO_ARG:-${AUTOPUSH_REPO:-}}"
if [ -z "$REPO" ]; then
    REPO="$(git rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [ -z "$REPO" ] || [ ! -d "$REPO" ]; then
    warn "no repository: pass a path, set AUTOPUSH_REPO, or run inside a work tree"
    exit 2
fi
REPO="$(cd "$REPO" && pwd -P)"

# Every git call goes through -C, so cwd is irrelevant from here on.
g() { git -C "$REPO" "$@"; }

if ! g rev-parse --git-dir >/dev/null 2>&1; then
    warn "not a git repository: $REPO"
    exit 2
fi

# 1. Never snapshot an in-progress merge: the snapshot would capture conflict
#    markers as if they were work, and the parent chain would record a tree that
#    never existed as a resolved state. Resolve first, then the next run captures.
if g rev-parse -q --verify MERGE_HEAD >/dev/null 2>&1 \
   || [ -n "$(g diff --name-only --diff-filter=U 2>/dev/null || true)" ]; then
    say "merge in progress in $REPO, skipped"
    exit 0
fi

# 2. A detached HEAD has no branch to name the ref after.
BRANCH="$(g symbolic-ref -q --short HEAD || true)"
if [ -z "$BRANCH" ]; then
    say "detached HEAD in $REPO, skipped"
    exit 0
fi

# 3. A clean tree has nothing to save. This is the common case on a timer.
STATUS="$(g status --porcelain --untracked-files=normal)"
if [ -z "$STATUS" ]; then
    say "clean tree in $REPO ($BRANCH), skipped"
    exit 0
fi
CHANGED="$(printf '%s\n' "$STATUS" | wc -l | tr -d ' ')"

# Branch names contain '/', which would make refs/autopush/a/b collide with a
# sibling refs/autopush/a. Flatten to '--' and print the real branch instead.
REF="refs/autopush/${BRANCH//\//--}"

# The throwaway index carries the snapshot; the real index is never named.
# It lives in the git dir, not $TMPDIR, for one reason found by the acceptance
# run: a temp file inside the work tree gets picked up by `git add -A` and
# becomes part of the snapshot, so two runs on an unchanged tree produce
# different trees. The git dir is never snapshotted.
GIT_DIR_ABS="$(g rev-parse --absolute-git-dir)"
TMP_BASE="$GIT_DIR_ABS"
if [ -n "${AUTOPUSH_TMPDIR:-}" ]; then
    mkdir -p "$AUTOPUSH_TMPDIR"
    TMP_BASE="$(cd "$AUTOPUSH_TMPDIR" && pwd -P)"
    case "$TMP_BASE/" in
        "$GIT_DIR_ABS"/*) : ;;
        "$REPO"/*)
            warn "$TMP_BASE is inside the work tree; using $GIT_DIR_ABS for the throwaway index"
            TMP_BASE="$GIT_DIR_ABS"
            ;;
    esac
fi
TMP_INDEX="$(mktemp "$TMP_BASE/rve-autopush-index.XXXXXX")"
trap 'rm -f "$TMP_INDEX"' EXIT

if g rev-parse -q --verify HEAD >/dev/null 2>&1; then
    GIT_INDEX_FILE="$TMP_INDEX" g read-tree HEAD
else
    GIT_INDEX_FILE="$TMP_INDEX" g read-tree --empty
fi
# -A stages modifications, deletions and untracked files; .gitignore'd paths stay
# out, which is the point — they are derived artefacts, not work.
GIT_INDEX_FILE="$TMP_INDEX" g add -A
TREE="$(GIT_INDEX_FILE="$TMP_INDEX" g write-tree)"

# Retention: the ref holds one snapshot per run, parented onto the previous one,
# so `git log refs/autopush/<name>` is the timeline. When the chain reaches
# AUTOPUSH_KEEP the newest snapshot starts a new root, which drops the previous
# segment whole. Sliding the window one step instead would mean rewriting N
# commits on every run — cheaper to implement, but it would change the SHA of the
# snapshot we just printed, and a SHA you can write down outlives a flat window.
# What the bound guarantees is the part that matters for a timer: the ref always
# carries the newest snapshot, so the worst case is losing the tree as it was
# before the previous run, never the tree as it is now.
PREV=""
DEPTH=0
REASON=""
if g rev-parse -q --verify "$REF" >/dev/null 2>&1; then
    PREV_TREE="$(g rev-parse "${REF}^{tree}")"
    if [ "$PREV_TREE" = "$TREE" ]; then
        say "tree unchanged since last snapshot in $REPO ($BRANCH), skipped"
        exit 0
    fi
    DEPTH="$(g rev-list --count "$REF")"
    if [ "$DEPTH" -lt "$KEEP" ]; then
        PREV="$REF"
    else
        REASON=", segment reset at depth $KEEP"
        DEPTH=0
    fi
fi

MSG="autopush: ${BRANCH} @ $(date -u '+%Y-%m-%dT%H:%M:%SZ') (${CHANGED} changed path(s))"
IDENT=(-c user.name=rve-autopush -c user.email=autopush@localhost)

if [ "$DRY_RUN" = 1 ]; then
    say "dry run: would write $REF from $BRANCH ($CHANGED changed path(s), depth $((DEPTH + 1)))"
    exit 0
fi

if [ -n "$PREV" ]; then
    SHA="$(g "${IDENT[@]}" commit-tree "$TREE" -p "$REF" -m "$MSG")"
else
    SHA="$(g "${IDENT[@]}" commit-tree "$TREE" -m "$MSG")"
fi

# update-ref, not `git branch -f`: the ref is not a branch and must never look
# like one, so no branch listing, no checkout target and no push refspec sees it.
g update-ref -m "autopush: snapshot $BRANCH" "$REF" "$SHA"
NEW_DEPTH="$((DEPTH + 1))"
say "$REF $SHA files=$CHANGED depth=$NEW_DEPTH branch=$BRANCH${REASON}"
