#!/bin/bash
# build.sh <wheels> [out-dir]
#
# The whole build, the same here and in .github/workflows/release.yml:
#   1. download the pinned weights and images (sources.txt), check their hashes
#   2. export each model in models.txt to head-less ONNX (export.py, this host's
#      Python with requirements-export.txt installed)
#   3. compile them with XMTVMC in an ubuntu:20.04 container (compile.sh)
#   4. check the int8 models against FP32 (eval.py) and write SHA256SUMS
#
# <wheels> is a directory with XMTVMC's vendor wheels: caffe-1.0, tvm_base-1.0
# and xmtvmc-1.0 (*-py3-none-any.whl), as wheels/ in this repository has them;
# their hashes are checked against wheels/SHA256SUMS.
set -euo pipefail
src=$(cd "$(dirname "$0")/.." && pwd)
wheels=$(readlink -f "$1") out=$(readlink -f "${2:-$src/out}")
cache=$src/.cache
(cd "$wheels" && sha256sum -c --quiet "$src/wheels/SHA256SUMS")
mkdir -p "$cache" "$out/onnx" "$out/images/cal" "$out/images/eval"

grep -v '^#' "$src/sources.txt" | while read -r file sha url; do
	[ -n "$file" ] || continue
	[ -f "$cache/$file" ] || curl -sfL -o "$cache/$file" "$url"
	echo "$sha  $cache/$file" | sha256sum -c --quiet
done
[ -d "$cache/coco128" ] || unzip -qo -d "$cache" "$cache/coco128.zip"
grep -v '^#' "$src/images.txt" | while read -r set id _; do
	[ -n "$set" ] && cp "$cache/coco128/images/train2017/$id.jpg" "$out/images/$set/"
done

grep -v '^#' "$src/models.txt" | while read -r name weights h w keep; do
	[ -n "$name" ] && python3 "$src/scripts/export.py" "$cache/$weights" "$h" "$w" \
		"$keep" "$out/onnx/$name.onnx"
done

# seccomp=unconfined: the default profile refuses the personality() call that
# turns address-space randomisation off for XMTVMC (see compile.sh).
docker run --rm --security-opt seccomp=unconfined -v "$src:/src:ro" -v "$wheels:/wheels:ro" -v "$out:/out" \
	ubuntu:20.04 bash -c "bash /src/scripts/compile.sh /wheels /out/onnx /out; \
	rc=\$?; chown -R $(id -u):$(id -g) /out; exit \$rc"

python3 "$src/scripts/eval.py" "$out"
cd "$out" && sha256sum ./*.xmm | sed 's# \./# #' > SHA256SUMS && cat SHA256SUMS
