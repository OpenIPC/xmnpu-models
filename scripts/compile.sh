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

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends python3.8 python3.8-venv \
	libpython3.8 libhdf5-103 libtinfo5 libgl1 libglib2.0-0 libgomp1 \
	g++ patch unzip ca-certificates >/dev/null

venv=/opt/xmtvmc
python3.8 -m venv $venv
$venv/bin/pip install -q --upgrade pip==24.0
$venv/bin/pip install -q -r "$src/requirements-xmtvmc.txt"
$venv/bin/pip install -q --no-deps "$wheels"/caffe-1.0-py3-none-any.whl \
	"$wheels"/tvm_base-1.0-py3-none-any.whl "$wheels"/xmtvmc-1.0-py3-none-any.whl
sp=$($venv/bin/python -c 'import site; print(site.getsitepackages()[0])')
for p in "$src"/patches/*.patch; do patch -s -d "$sp" -p1 < "$p"; done

# Ubuntu names its serial HDF5 libhdf5_serial*; caffe was linked against the
# upstream names. xmnpu looks for its libraries in <venv>/lib/build, where the
# SDK's install.sh would have copied them.
extra=$venv/lib/extra && mkdir -p $extra
ln -sf /usr/lib/x86_64-linux-gnu/libhdf5_serial.so.103 $extra/libhdf5.so.103
ln -sf /usr/lib/x86_64-linux-gnu/libhdf5_serial_hl.so.100 $extra/libhdf5_hl.so.100
mkdir -p $venv/lib/build && ln -sf "$sp"/xmnpu/*.so $venv/lib/build/
export LD_LIBRARY_PATH=$sp/xmnpu:$sp/caffe/lib:$extra
export PATH=$venv/bin:$PATH
# The class maps of a detector are almost all negative over any calibration
# set; see patches/_calibrate.patch and the README.
export XM_CALIB_OUTPUT_MINMAX=8
# XMTVMC writes some of its own host pointers into the .xmm's tensor table --
# meaningless on the camera, but they make two builds of the same model differ
# by a few bytes. Without address-space randomisation they are the same every
# time, so a release can be rebuilt bit for bit.
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
	unzip -qo -p $name.qnn.xmnpu_npu.zip neuron_network.xmm > "$out/$name.xmm"
done
