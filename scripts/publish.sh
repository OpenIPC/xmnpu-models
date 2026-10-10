#!/bin/bash
# publish.sh <tag> <notes-file> <file>...: creates release <tag> at $GITHUB_SHA
# in $GITHUB_REPOSITORY and uploads the files, with $GH_TOKEN. For runners
# without the gh CLI; refuses when the tag already exists.
set -euo pipefail
tag=$1 notes=$2; shift 2
api=https://api.github.com/repos/$GITHUB_REPOSITORY
auth=(-H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json")
if curl -sf "${auth[@]}" "$api/releases/tags/$tag" >/dev/null; then
	echo "release $tag already exists"; exit 1
fi
body=$(python3 -c 'import json,sys; print(json.dumps({"tag_name": sys.argv[1],
	"target_commitish": sys.argv[2], "name": sys.argv[1], "body": open(sys.argv[3]).read()}))' \
	"$tag" "$GITHUB_SHA" "$notes")
id=$(curl -sf "${auth[@]}" -X POST "$api/releases" -d "$body" |
	python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
for f in "$@"; do
	curl -sf "${auth[@]}" -H "Content-Type: application/octet-stream" --data-binary @"$f" \
		"https://uploads.github.com/repos/$GITHUB_REPOSITORY/releases/$id/assets?name=$(basename "$f")" >/dev/null
	echo "uploaded $(basename "$f")"
done
