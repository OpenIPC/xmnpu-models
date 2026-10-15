#!/bin/bash
# train/run.sh train  <model> <epochs> <batch> <out-dir>
# train/run.sh import <model> <run-dir> <out-dir>
#
# Fine-tunes a COCO YOLOv8 (yolov8n, yolov8s) with LeakyReLU activations and
# the camera's eight classes, or trains pvdn (person and vehicle: from the
# checkpoint sources.txt pins, else from scratch, train/pvdn.yaml), or takes a
# run already made on this host (import);
# then evaluates it and writes what a release carries to <out-dir>: the
# checkpoint (<model>-lrelu-c8.pt, or pvdn.pt), results.csv, args.yaml,
# inputs.sha256.gz (every image trained and validated on), eval.md and
# SHA256SUMS.
#
# COCO_DIR is the COCO 2017 copy (see dataset.py); TRAIN_WORK keeps the
# environment, the dataset and the runs between jobs. pvdn also trains on MEVA
# security-camera frames (meva.py), fetched into TRAIN_WORK/cache, and its
# eval.md scores the MEVA cameras held out of training as well as COCO.
set -euo pipefail
src=$(cd "$(dirname "$0")/.." && pwd)
mode=$1 model=$2
work=${TRAIN_WORK:?} coco=${COCO_DIR:?} uv=${UV:-$HOME/.local/bin/uv}
# systemd-logind's RemoveIPC (on by default) deletes a user's /dev/shm files
# when their last session closes -- the dataloader's shared batches included,
# which stalls training minutes in. A lingering user is never logged out.
if command -v loginctl >/dev/null &&
	! loginctl show-user "$(id -un)" -p Linger 2>/dev/null | grep -q yes; then
	echo "enable lingering first: sudo loginctl enable-linger $(id -un)"; exit 1
fi
mkdir -p "$work/cache"

"$uv" venv -q --allow-existing --python 3.12 "$work/venv"
"$uv" pip install -q --python "$work/venv/bin/python" --index-strategy unsafe-best-match \
	-r "$src/requirements-train.txt"
py="$work/venv/bin/python"

# What each model starts from, trains on, and is published as.
case $model in
pvdn)
	set_=pv data=$work/pv/pv.yaml ckpt=pvdn.pt ref=
	# From the checkpoint sources.txt pins, when it pins one: a fine-tune of
	# the released model on what has been added since. From scratch
	# (train/pvdn.yaml) only when there is none.
	start=$(awk '$1 == "pvdn.pt" { print $2, $3 }' "$src/sources.txt")
	if [ -n "$start" ]; then
		read -r sha url <<< "$start"
		[ -f "$work/cache/pvdn-start.pt" ] && echo "$sha  $work/cache/pvdn-start.pt" | sha256sum -c --quiet ||
			curl -sfL -o "$work/cache/pvdn-start.pt" "$url"
		echo "$sha  $work/cache/pvdn-start.pt" | sha256sum -c --quiet
		init() { cp "$work/cache/pvdn-start.pt" "$1"; }
	else
		init() { "$py" "$src/train/init.py" "$src/train/pvdn.yaml" "$1" person,vehicle; }
	fi
	;;
*)
	set_=c8 data=$work/coco8/coco8.yaml ckpt=$model-lrelu-c8.pt ref=$work/cache/$model.pt
	grep -v '^#' "$src/sources.txt" | while read -r file sha url; do
		[ "$file" = "$model.pt" ] || continue
		[ -f "$work/cache/$file" ] || curl -sfL -o "$work/cache/$file" "$url"
		echo "$sha  $work/cache/$file" | sha256sum -c --quiet
	done
	[ -f "$ref" ] || { echo "no $model.pt in sources.txt"; exit 1; }
	init() { "$py" "$src/train/init.py" "$ref" "$1"; }
	;;
esac

case $mode in train) out=$(realpath -m "$5") ;; import) out=$(realpath -m "$4") ;; esac
cd "$work"
"$py" "$src/train/dataset.py" "$coco" "$(dirname "$data")" "$set_"
# The person/vehicle set also trains on security-camera frames (meva.py).
[ "$set_" = pv ] && "$py" "$src/train/meva.py" build "$work/cache" "$(dirname "$data")"
case $mode in
train)
	epochs=$3 batch=$4
	run=$work/runs/${ckpt%.pt}-$(date -u +%Y%m%d-%H%M%S)
	init "$work/init-$model.pt"
	mkdir -p "$run"
	# The trainer's progress bars are carriage returns without newlines, hours
	# of them; written into the runner's pipe they stalled a job four minutes
	# in and cut its traceback off at "Trace". They go to train.log instead,
	# and the job log gets one line per finished epoch from results.csv.
	PYTHONUNBUFFERED=1 "$py" "$src/train/fit.py" "$work/init-$model.pt" \
		"$data" "$epochs" "$batch" "$run" > "$run/train.log" 2>&1 &
	fit=$!
	trap 'kill $fit 2>/dev/null || true' EXIT  # a cancelled job must not leave it training
	seen=1
	while kill -0 $fit 2>/dev/null; do
		sleep 60
		if [ -f "$run/results.csv" ]; then
			n=$(wc -l < "$run/results.csv")
			[ "$n" -gt "$seen" ] && tail -n +$((seen + 1)) "$run/results.csv" |
				cut -d, -f1,2,8,9 | sed 's/^/epoch,time,mAP50,mAP50-95: /' && seen=$n
		fi
	done
	wait $fit || { tail -c 4000 "$run/train.log" | tr '\r' '\n' | tail -40; exit 1; }
	;;
import)
	run=$(readlink -f "$3")
	;;
*)
	echo "mode is train or import"; exit 2
	;;
esac
mkdir -p "$out"
cp "$run/weights/best.pt" "$out/$ckpt"
cp "$run/results.csv" "$run/args.yaml" "$out/"
gzip -9c "$(dirname "$data")/inputs.sha256" > "$out/inputs.sha256.gz"
[ -f "$run/train.log" ] && tr '\r' '\n' < "$run/train.log" | grep -av '^ *$' | gzip -9 > "$out/train.log.gz"
"$py" "$src/train/evaluate.py" "$out/$ckpt" "$data" "$out/eval.md" $ref
if [ "$set_" = pv ]; then
	"$py" "$src/train/evaluate.py" "$out/$ckpt" "$(dirname "$data")/meva.yaml" "$out/eval-meva.md"
	{ echo; cat "$out/eval-meva.md"; } >> "$out/eval.md" && rm "$out/eval-meva.md"
fi
(cd "$out" && sha256sum ./* | sed 's# \./# #' | grep -v SHA256SUMS > SHA256SUMS && cat SHA256SUMS)
