#!/bin/bash
# fetch.sh <list> <cache-dir>: download every "file sha256 url" line of <list>
# into <cache-dir> unless it is already there, and check every hash.
set -euo pipefail
mkdir -p "$2"
grep -v '^#' "$1" | while read -r file sha url; do
	[ -n "$file" ] || continue
	[ -f "$2/$file" ] || curl -sfL --retry 3 -o "$2/$file" "$url"
	echo "$sha  $2/$file" | sha256sum -c --quiet
done
