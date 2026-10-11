# check_x86.py <name>.qnn.xmnpu_x86.so <eval-dir>
#
# The presets are measured on the quantized graph's xmnpu_x86 build, on the
# claim that it computes what the NPU does. This holds it to that: on every
# eval image compile.sh ran through the NPU C-model (bit-exact with the
# hardware), the x86 build is given the same input planes and its outputs must
# equal the C-model's exactly. Runs in the XMTVMC environment; exits non-zero
# on any difference, which fails the build before anything is measured on it.
import glob
import os
import sys

import numpy as np
import tvm
import tvm.contrib.graph_executor
import xmtvmc.io as io
from xmtvmc.hooks import POSTPROCESS_RUNNER, _running_hook

lib_path, eval_dir = sys.argv[1:3]
params = io.load_tensor_info(lib_path + ".params")
mod = tvm.contrib.graph_executor.GraphModule(io.load_module(lib_path)["default"](tvm.cpu(0)))
h, w = mod.get_input("Y").shape[-2:]
dirs = sorted(d for d in glob.glob(f"{eval_dir}/*") if os.path.isdir(d))
if not dirs:
    sys.exit(f"check_x86: no C-model results under {eval_dir}")
bad = 0
for d in dirs:
    y = np.fromfile(glob.glob(f"{d}/input_data0_*.bin")[0], np.uint8).reshape(h, w)
    vu = np.fromfile(glob.glob(f"{d}/input_data1_*.bin")[0], np.uint8).reshape(h // 2, w)
    mod.set_input("Y", y)
    mod.set_input("UV", vu)
    mod.run()
    outs = [mod.get_output(i).asnumpy() for i in range(mod.get_num_outputs())]
    outs, _ = _running_hook(POSTPROCESS_RUNNER, outs, params, output_dtype="float")
    ref = np.load(f"{d}/outputs.npy", allow_pickle=True).item()
    for i, o in enumerate(outs):
        r = np.asarray(ref[f"output_{i}"])
        if r.shape != o.shape or not np.array_equal(r, o):
            n = int(np.sum(r != o)) if r.shape == o.shape else -1
            print(f"check_x86: {os.path.basename(d)} output {i} differs ({n} values)")
            bad += 1
if bad:
    sys.exit(f"check_x86: {lib_path} does not reproduce the C-model")
print(f"check_x86: {os.path.basename(lib_path)} matches the C-model on {len(dirs)} images")
