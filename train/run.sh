#!/bin/bash
# train/run.sh train  <model> <epochs> <batch> <out-dir>
# train/run.sh import <model> <run-dir> <out-dir>
#
# Fine-tunes a COCO YOLOv8 (yolov8n, yolov8s) with LeakyReLU activations and
# the camera's eight classes, or takes a fine-tune already run on this host
# (import), then evaluates it and writes what a release carries to <out-dir>:
# <model>-lrelu-c8.pt, results.csv, args.yaml, eval.md and SHA256SUMS.
#
# COCO_DIR is the COCO 2017 copy (see dataset.py); TRAIN_WORK keeps the
# environment, the dataset and the runs between jobs.
set -euo pipefail
src=$(cd "$(dirname "$0")/.." && pwd)
mode=$1 model=$2
work=${TRAIN_WORK:?} coco=${COCO_DIR:?} uv=${UV:-$HOME/.local/bin/uv}
mkdir -p "$work/cache"

"$uv" venv -q --allow-existing --python 3.12 "$work/venv"
"$uv" pip install -q --python "$work/venv/bin/python" --index-strategy unsafe-best-match \
	-r "$src/requirements-train.txt"
py="$work/venv/bin/python"

grep -v '^#' "$src/sources.txt" | while read -r file sha url; do
	[ "$file" = "$model.pt" ] || continue
	[ -f "$work/cache/$file" ] || curl -sfL -o "$work/cache/$file" "$url"
	echo "$sha  $work/cache/$file" | sha256sum -c --quiet
done
[ -f "$work/cache/$model.pt" ] || { echo "no $model.pt in sources.txt"; exit 1; }

case $mode in train) out=$(realpath -m "$5") ;; import) out=$(realpath -m "$4") ;; esac
cd "$work"
"$py" "$src/train/dataset.py" "$coco" "$work/coco8"
case $mode in
train)
	epochs=$3 batch=$4
	run=$work/runs/$model-lrelu-c8-$(date -u +%Y%m%d-%H%M%S)
	"$py" "$src/train/init.py" "$work/cache/$model.pt" "$work/init-$model.pt"
	"$py" "$src/train/fit.py" "$work/init-$model.pt" "$work/coco8/coco8.yaml" \
		"$epochs" "$batch" "$run"
	;;
import)
	run=$(readlink -f "$3")
	;;
*)
	echo "mode is train or import"; exit 2
	;;
esac
mkdir -p "$out"
cp "$run/weights/best.pt" "$out/$model-lrelu-c8.pt"
cp "$run/results.csv" "$run/args.yaml" "$out/"
"$py" "$src/train/evaluate.py" "$out/$model-lrelu-c8.pt" "$work/coco8/coco8.yaml" \
	"$out/eval.md" "$work/cache/$model.pt"
(cd "$out" && sha256sum ./* | sed 's# \./# #' | grep -v SHA256SUMS > SHA256SUMS && cat SHA256SUMS)
