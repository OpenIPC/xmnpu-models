# eval.py <out-dir>
#
# Checks every compiled model against its FP32 ONNX on the eval images. The
# int8 side is the NPU C-model's output (bit-exact with the hardware), from
# compile.sh; the FP32 side runs onnxruntime on the very pixels the C-model was
# given -- its NV21 planes, converted back the way the fused model converts
# them -- so the comparison measures quantization and nothing else.
#
# A model passes when every FP32 detection at or above PASS_SCORE has an int8
# detection of the same class overlapping it by at least PASS_IOU, and the
# other way round. Writes <out-dir>/eval.md and exits non-zero on a failure.
import glob
import os
import sys

import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

PASS_SCORE, PASS_IOU, REPORT_SCORE = 0.5, 0.6, 0.3


def yuv2rgb(y, vu):
    """NV21, BT.709 limited range, to RGB 0..255 -- NV21_BT709_TV's inverse."""
    y = (y.astype(np.float32) - 16) * 255 / 219
    up = lambda p: np.repeat(np.repeat(p, 2, 0), 2, 1).astype(np.float32) - 128
    v, u = up(vu[:, 0::2]) * 255 / 224, up(vu[:, 1::2]) * 255 / 224
    rgb = np.stack([y + 1.5748 * v, y - 0.1873 * u - 0.4681 * v, y + 1.8556 * u], -1)
    return np.clip(rgb, 0, 255)


def decode(outs, h, conf, iou=0.45):
    """YOLOv8 DFL decode and per-class NMS: [(x1, y1, x2, y2, score, cls)]."""
    det = []
    for box, cls in zip(outs[0::2], outs[1::2]):
        _, gh, gw = box.shape
        stride = h / gh
        p = 1 / (1 + np.exp(-cls.reshape(cls.shape[0], -1)))
        c, s = p.argmax(0), p.max(0)
        for i in np.nonzero(s > conf)[0]:
            d = box.reshape(4, 16, -1)[:, :, i]
            d = np.exp(d - d.max(1, keepdims=True))
            dist = (d / d.sum(1, keepdims=True) * np.arange(16)).sum(1) * stride
            cx, cy = (i % gw + 0.5) * stride, (i // gw + 0.5) * stride
            det.append((cx - dist[0], cy - dist[1], cx + dist[2], cy + dist[3], s[i], c[i]))
    det.sort(key=lambda d: -d[4])
    kept = []
    for d in det:
        if all(k[5] != d[5] or overlap(k, d) <= iou for k in kept):
            kept.append(d)
    return kept


def overlap(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def unmatched(ref, other):
    return [r for r in ref if r[4] >= PASS_SCORE and not any(
        o[5] == r[5] and overlap(o, r) >= PASS_IOU for o in other)]


out = sys.argv[1]
models = [l.split() for l in open(os.path.join(os.path.dirname(__file__), "..", "models.txt"))
          if l.strip() and not l.startswith("#")]
report, failed = ["| model | image | FP32 | int8 (NPU C-model) |", "|---|---|---|---|"], False
for name, weights, h, w, keep in models:
    labels = YOLO(f"{os.path.dirname(__file__)}/../.cache/{weights}").names
    h, w, names = int(h), int(w), [labels[int(c)] for c in keep.split(",")]
    sess = ort.InferenceSession(f"{out}/onnx/{name}.onnx")
    fmt = lambda ds: ", ".join(f"{names[int(d[5])]} {d[4]:.2f}" for d in ds) or "-"
    for d in sorted(glob.glob(f"{out}/work/{name}/eval/*")):
        y = np.fromfile(glob.glob(f"{d}/input_data0_*.bin")[0], np.uint8).reshape(h, w)
        vu = np.fromfile(glob.glob(f"{d}/input_data1_*.bin")[0], np.uint8).reshape(h // 2, w)
        x = (yuv2rgb(y, vu) / 255).transpose(2, 0, 1)[None].astype(np.float32)
        fp = decode([o[0] for o in sess.run(None, {"images": x})], h, REPORT_SCORE)
        q = np.load(f"{d}/outputs.npy", allow_pickle=True).item()
        i8 = decode([q[f"output_{i}"][0] for i in range(len(q))], h, REPORT_SCORE)
        bad = unmatched(fp, i8) + unmatched(i8, fp)
        failed |= bool(bad)
        report.append(f"| {name} | {os.path.basename(d)} | {fmt(fp)} | {fmt(i8)}"
                      f"{' **MISMATCH**' if bad else ''} |")
open(f"{out}/eval.md", "w").write("\n".join(report) + "\n")
print("\n".join(report))
sys.exit(1 if failed else 0)
