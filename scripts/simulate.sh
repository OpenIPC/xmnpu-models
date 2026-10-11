#!/bin/bash
# simulate.sh <wheels> <name>.x86.so <nv21-dir> <out> <shard> <shards>
#
# Runs in ubuntu:20.04 (see xmtvmc-env.sh). The int8 model's detections, from
# its xmnpu_x86 build (compile.sh), on every <shards>-th frame of <nv21-dir>
# starting at <shard>, one process per CPU: sim_dets.py's format, in <out>.
set -euo pipefail
src=$(cd "$(dirname "$0")/.." && pwd)
lib=$(readlink -f "$2") frames=$(readlink -f "$3") out=$(readlink -f "$4")
shard=$5 shards=$6
source "$src/scripts/xmtvmc-env.sh" "$1"
# One thread a process and a process a CPU: the graph runtime's own threads
# spend more time in the kernel than in the convolutions at this size.
export TVM_NUM_THREADS=1
n=$(nproc) tmp=$(mktemp -d)
for i in $(seq 0 $((n - 1))); do
	"$venv/bin/python" "$src/scripts/sim_dets.py" "$lib" "$frames" "$tmp/$i.txt" \
		$((shard + i * shards)) $((n * shards)) &
done
wait
cat "$tmp"/*.txt > "$out"
