#!/bin/bash
# compile.sh <wheels> <onnx-dir> <out-dir>
#
# Runs inside ubuntu:20.04 (see scripts/build.sh): XMTVMC is a Python 3.8
# toolchain that links each compiled model with the system g++, its vendor
# caffe module wants focal's libhdf5-103, and focal is old enough that
# onnxruntime 1.13's executable-stack request is honoured.
#
# <wheels> holds the SDK's vendor wheels (caffe, tvm_base, xmtvmc); everything
# else is installed from PyPI at the versions in requirements-xmtvmc.txt. For
# each line of models.txt this compiles <onnx-dir>/<name>.onnx into
# <out-dir>/<name>.xmm and runs the NPU's C-model -- bit-exact with the
# hardware -- over every eval image, for scripts/eval.py to check.
set -euo pipefail
src=$(cd "$(dirname "$0")/.." && pwd)
wheels=$(readlink -f "$1") onnx=$(readlink -f "$2") out=$(readlink -f "$3")
mkdir -p "$out"

source "$src/scripts/xmtvmc-env.sh" "$wheels"
# The class maps of a detector are almost all negative over any calibration
# set; see patches/_calibrate.patch and the README.
export XM_CALIB_OUTPUT_MINMAX=8
# XMTVMC writes some of its own host pointers into the .xmm's tensor table --
# meaningless on the camera, but they follow the build host's memory layout.
# The main build runs without address-space randomisation and a second one
# with it; sanitize.py zeroes every byte where the two differ.
xmtvmc() { setarch "$(uname -m)" -R "$venv/bin/xmtvmc" "$@"; }

grep -v '^#' "$src/models.txt" | while read -r name weights h w keep; do
	[ -n "$name" ] || continue
	work=$out/work/$name && rm -rf "$work" && mkdir -p "$work" && cd "$work"
	cat > cfg.yaml <<-Y
	input_shape:
	  images: [1, 3, $h, $w]
	input_normalize:
	  images:
	    - [0, 0, 0]
	    - [255, 255, 255]
	input_layout: NCHW
	input_format: NV21_BT709_TV
	input_dtype: uint8
	quantize:
	  calibrate_mode: xm_kl_divergence_input
	  weight_scale: xm_kl_divergence_weight
	  do_hBC: True
	  do_SQuant: True
	compile:
	  group_partition: True
	  memory_mode: 0
	Y
	echo "== $name: import, quantize, compile"
	xmtvmc import --framework onnx-float --model "$onnx/$name.onnx" \
		--config-file cfg.yaml > log.txt 2>&1
	xmtvmc quantize --model $name.json --weight $name.params \
		--calibrate-dataset "$out/images/cal" --config-file cfg.yaml >> log.txt 2>&1
	grep -a "output range" log.txt || { echo "output ranges were not widened"; exit 1; }
	xmtvmc compile --model $name.qnn.json --weight $name.qnn.params \
		--target xmnpu_npu --config-file cfg.yaml >> log.txt 2>&1
	for img in "$out"/images/eval/*.jpg; do
		id=$(basename "$img" .jpg)
		echo "== $name: C-model on $id"
		xmtvmc run --lib $name.qnn.xmnpu_npu.so --params $name.qnn.xmnpu_npu.so.params \
			--target xmnpu_npu --input-data "$img" --config-file cfg.yaml >> log.txt 2>&1
		mkdir -p eval/$id && mv "$id.jpg.xmnpu_npu.npy" eval/$id/outputs.npy
		unzip -qo -d eval/$id $name.qnn.xmnpu_npu.zip 'input_data*'
	done
	unzip -qo -p $name.qnn.xmnpu_npu.zip neuron_network.xmm > a.xmm
	# The same quantized graph built for x86: its outputs are bit-identical
	# to the C-model's at a fraction of the cost, which is what lets
	# simulate.sh run it over all of COCO val2017 for the presets.
	xmtvmc compile --model $name.qnn.json --weight $name.qnn.params \
		--target xmnpu_x86 --config-file cfg.yaml >> log.txt 2>&1
	cp $name.qnn.xmnpu_x86.so "$out/$name.x86.so"
	cp $name.qnn.xmnpu_x86.so.params "$out/$name.x86.so.params"
	# The same compile with randomisation on, so every byte that follows the
	# host's memory layout shows up as a difference for sanitize.py to zero.
	echo "== $name: second build, for the host-pointer fields"
	mkdir -p aslr && (cd aslr && cp ../cfg.yaml ../$name.qnn.json ../$name.qnn.params . &&
		"$venv/bin/xmtvmc" compile --model $name.qnn.json --weight $name.qnn.params \
			--target xmnpu_npu --config-file cfg.yaml > log.txt 2>&1 &&
		"$venv/bin/xmtvmc" run --lib $name.qnn.xmnpu_npu.so \
			--params $name.qnn.xmnpu_npu.so.params --target xmnpu_npu \
			--input-data "$(ls "$out"/images/eval/*.jpg | head -1)" \
			--config-file cfg.yaml >> log.txt 2>&1 &&
		unzip -qo -p $name.qnn.xmnpu_npu.zip neuron_network.xmm > ../b.xmm)
	python "$src/scripts/sanitize.py" a.xmm b.xmm "$out/$name.xmm"
done
