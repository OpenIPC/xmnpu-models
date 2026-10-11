# sim_dets.py <model>.qnn.xmnpu_x86.so <nv21-dir> <out> [<shard> <shards>]
#
# The int8 model's detections on every <nv21-dir>/*.nv21 frame, as npu_eval
# prints them on the camera: "<image id> x1 y1 x2 y2 score class" in input
# pixels, every class at or above 0.001 in each cell, per-class NMS at IoU
# 0.7, at most 300 per image -- what operating_points.py scores.
#
# The model runs as XMTVMC's xmnpu_x86 build of the quantized graph, whose
# outputs are bit-identical to the NPU C-model's (and so the hardware's) and
# which costs a fraction of a second a frame where the C-model costs a
# minute. The frames are the picture stretched to the input, NV21 BT.709
# limited range, the way the camera's VPSS hands them over. Runs in the
# XMTVMC environment (compile.sh). <shard> of <shards> takes every
# <shards>-th frame, for running several at once.
import glob
import os
import sys

import numpy as np
import tvm
import tvm.contrib.graph_executor
import xmtvmc.io as io
from xmtvmc.hooks import POSTPROCESS_RUNNER, _running_hook

CONF, NMS_IOU, MAX_DET, BINS = 0.001, 0.7, 300, 16


def iou(a, b):
    w = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
    h = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - i)


def decode(outs, h):
    """npu_eval's decode: every (cell, class) at or above CONF, DFL boxes."""
    boxes, scores, classes = [], [], []
    logit_min = np.log(CONF / (1 - CONF))
    for box, cls in zip(outs[0::2], outs[1::2]):
        nc, gh, gw = cls.shape
        stride = h / gh
        c_idx, cell = np.nonzero(cls.reshape(nc, -1) >= logit_min)
        if not len(cell):
            continue
        d = box.reshape(4, BINS, -1)[:, :, cell]
        d = np.exp(d - d.max(1, keepdims=True))
        dist = (d / d.sum(1, keepdims=True) * np.arange(BINS)[None, :, None]).sum(1) * stride
        cx, cy = (cell % gw + 0.5) * stride, (cell // gw + 0.5) * stride
        boxes.append(np.stack([cx - dist[0], cy - dist[1], cx + dist[2], cy + dist[3]], 1))
        scores.append(1 / (1 + np.exp(-cls.reshape(nc, -1)[c_idx, cell])))
        classes.append(c_idx)
    if not boxes:
        return []
    b, s, c = np.concatenate(boxes), np.concatenate(scores), np.concatenate(classes)
    order = np.argsort(-s, kind="stable")
    b, s, c = b[order], s[order], c[order]
    keep = []
    for i in range(len(s)):  # in score order, across classes, as npu_eval does
        if len(keep) == MAX_DET:
            break
        k = np.array(keep, int)
        same = k[c[k] == c[i]] if len(k) else k
        if len(same) and (iou(b[i], b[same]) > NMS_IOU).any():
            continue
        keep.append(i)
    return [(*b[i], s[i], c[i]) for i in keep]


def main(lib_path, src, out, shard=0, shards=1):
    shard, shards = int(shard), int(shards)
    lib = io.load_module(lib_path)
    params = io.load_tensor_info(lib_path + ".params")
    mod = tvm.contrib.graph_executor.GraphModule(lib["default"](tvm.cpu(0)))
    h, w = mod.get_input("Y").shape[-2:]
    frames = sorted(glob.glob(f"{src}/*.nv21"))[shard::shards]
    with open(out, "w") as o:
        for n, f in enumerate(frames):
            raw = np.fromfile(f, np.uint8)
            if raw.size != w * h * 3 // 2:
                sys.exit(f"{f}: not a {w}x{h} NV21 frame")
            mod.set_input("Y", raw[:w * h].reshape(h, w))
            mod.set_input("UV", raw[w * h:].reshape(h // 2, w))
            mod.run()
            outs = [mod.get_output(i).asnumpy() for i in range(mod.get_num_outputs())]
            outs, _ = _running_hook(POSTPROCESS_RUNNER, outs, params, output_dtype="float")
            iid = os.path.basename(f)[:-5]
            for x1, y1, x2, y2, s, c in decode([x[0] for x in outs], h):
                o.write(f"{iid} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {s:.4f} {c}\n")
            if n % 250 == 0:
                print(f"{out}: {n}/{len(frames)}", flush=True)
        # How many frames this covered, which a frame with no detections at
        # all does not otherwise say: the publish job adds these up.
        o.write(f"# frames {len(frames)}\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
