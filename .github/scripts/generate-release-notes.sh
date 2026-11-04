#!/usr/bin/env bash
# Builds "What's Changed" release notes from the commits between the previous
# published release and TARGET_REF, then creates or refreshes a DRAFT release.
#
# Commits are read straight from the compare API rather than GitHub's
# auto-generated notes, because changes often land on main without a PR and the
# auto-generator only lists merged PRs.
#
# Environment:
#   REPO            owner/name (required)
#   TARGET_REF      commit SHA (or tag, for PRINT_ONLY replays) the release points at (required)
#   VERSION         version to release, without the "v" (required)
#   PREVIOUS_TAG    tag to diff against; defaults to the latest published, non-prerelease release
#   PRINT_ONLY=1    print the notes and exit without creating or editing a release
#   GITHUB_STEP_SUMMARY  set by Actions; notes and skipped commits are appended to it
set -euo pipefail

VERSION_PATTERN='^[0-9]+\.[0-9]+\.[0-9]+([a-z0-9.-]*)?$'
TAG_PATTERN='^[A-Za-z0-9._-]+$'
# A housekeeping commit is a `chore:` (optionally ticket-prefixed) whose subject
# mentions the version or changelog, e.g. "chore: Bump version to 1.2.4 ...".
HOUSEKEEPING_PATTERN='^(\[[^\]]+\]\s*)?chore(\([^)]*\))?:.*(version|changelog)'

fail() {
  echo "::error::$1" >&2
  exit 1
}

: "${REPO:?REPO is required}"
: "${TARGET_REF:?TARGET_REF is required}"
: "${VERSION:?VERSION is required}"
PRINT_ONLY="${PRINT_ONLY:-0}"

[[ "$VERSION" =~ $VERSION_PATTERN ]] || fail "Version '$VERSION' is not a valid version (expected e.g. 1.2.5)."
TAG="v${VERSION}"

if [[ -z "${PREVIOUS_TAG:-}" ]]; then
  PREVIOUS_TAG="$(gh release list -R "$REPO" --exclude-drafts --exclude-pre-releases --limit 1 \
    --json tagName --jq '.[0].tagName // ""')"
  [[ -n "$PREVIOUS_TAG" ]] || fail "No published release found to diff against; set previous_tag."
fi
[[ "$PREVIOUS_TAG" =~ $TAG_PATTERN ]] || fail "Previous tag '$PREVIOUS_TAG' contains unexpected characters."
gh api "repos/${REPO}/git/ref/tags/${PREVIOUS_TAG}" --silent \
  || fail "Previous tag '$PREVIOUS_TAG' does not exist in ${REPO}."
[[ "$PREVIOUS_TAG" != "$TAG" ]] || fail "Previous tag and release tag are both '$TAG'."

# The compare API returns at most 250 commits per page set; --paginate walks them.
COMMITS_JSON="$(gh api --paginate "repos/${REPO}/compare/${PREVIOUS_TAG}...${TARGET_REF}?per_page=100" \
  --jq '.commits[]' | jq -s '.')"

# One object per non-merge commit: its first message line and how to credit it.
ENTRIES_JSON="$(jq --arg housekeeping "$HOUSEKEEPING_PATTERN" '
  map(select((.parents | length) <= 1))
  | map(
      (.commit.message | split("\n")[0]) as $subject
      | {
          subject: $subject,
          sha: .sha[0:7],
          skipped: ($subject | test($housekeeping; "i")),
          line: (
            "* "
            + ($subject
                | sub("^\\[[^\\]]+\\]\\s*"; "")
                | sub("\\s*\\(#(?<pr>[0-9]+)\\)$"; " in #\(.pr)"))
            + " by "
            + (if .author.login then "@" + .author.login else .commit.author.name end)
          )
        }
    )' <<<"$COMMITS_JSON")"

BULLETS="$(jq -r 'map(select(.skipped | not)) | .[].line' <<<"$ENTRIES_JSON")"
SKIPPED="$(jq -r 'map(select(.skipped)) | .[] | "- `\(.sha)` \(.subject)"' <<<"$ENTRIES_JSON")"

[[ -n "$BULLETS" ]] || fail "Nothing to release since ${PREVIOUS_TAG}: no commits left after skipping housekeeping."

NOTES_FILE="$(mktemp)"
trap 'rm -f "$NOTES_FILE"' EXIT
cat >"$NOTES_FILE" <<EOF
## What's Changed

${BULLETS}

**Full Changelog**: https://github.com/${REPO}/compare/${PREVIOUS_TAG}...${TAG}
EOF

cat "$NOTES_FILE"
if [[ -n "$SKIPPED" ]]; then
  printf '\nSkipped as version/changelog housekeeping:\n%s\n' "$SKIPPED" >&2
fi

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  {
    echo "# Draft release notes for ${TAG} (since ${PREVIOUS_TAG})"
    echo
    cat "$NOTES_FILE"
    echo
    echo "### Skipped as version/changelog housekeeping"
    echo "${SKIPPED:-_None_}"
  } >>"$GITHUB_STEP_SUMMARY"
fi

[[ "$PRINT_ONLY" == "1" ]] && exit 0

# Draft releases are invisible to /releases/tags/{tag}, so scan the release list.
# Drafts and recent releases sort first, so the first 100 are enough.
EXISTING_JSON="$(gh api "repos/${REPO}/releases?per_page=100" \
  | jq --arg tag "$TAG" 'map(select(.tag_name == $tag)) | map({id, draft})')"

if jq -e 'any(.[]; .draft | not)' <<<"$EXISTING_JSON" >/dev/null; then
  fail "Release ${TAG} is already published; refusing to modify it."
fi

DRAFT_COUNT="$(jq 'length' <<<"$EXISTING_JSON")"
case "$DRAFT_COUNT" in
  0)
    gh release create "$TAG" -R "$REPO" --draft --target "$TARGET_REF" --title "$TAG" --notes-file "$NOTES_FILE"
    ;;
  1)
    DRAFT_ID="$(jq -r '.[0].id' <<<"$EXISTING_JSON")"
    gh api -X PATCH "repos/${REPO}/releases/${DRAFT_ID}" \
      -F "body=@${NOTES_FILE}" -f "target_commitish=${TARGET_REF}" --jq '.html_url'
    echo "Updated existing draft ${TAG}."
    ;;
  *)
    fail "Found ${DRAFT_COUNT} drafts for ${TAG}; delete the extras and re-run."
    ;;
esac
