# init.py <silu.pt> <out.pt>
#
# The starting point for a fine-tune: a COCO YOLOv8 checkpoint with every SiLU
# replaced by LeakyReLU(0.1) -- which the NPU fuses into the convolution,
# where SiLU costs a lookup-table pass of its own -- and the class head cut
# down to the camera's eight classes, keeping their trained rows.
import sys

import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils import ROOT

KEEP = [0, 1, 2, 3, 5, 7, 15, 16]  # COCO-80 ids
NAMES = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "cat", "dog"]

src, out = sys.argv[1:3]
scale = YOLO(src).model.yaml.get("scale") or src.split("/")[-1][len("yolov8")]
cfg = yaml.safe_load(open(ROOT / "cfg/models/v8/yolov8.yaml"))
cfg.update(scale=scale, activation="nn.LeakyReLU(0.1)")
yaml.safe_dump(cfg, open(f"yolov8{scale}-lrelu.yaml", "w"), sort_keys=False)
y = YOLO(f"yolov8{scale}-lrelu.yaml").load(src)  # weights carry over: activations have none
m, det = y.model, y.model.model[-1]
for seq in det.cv3:
    conv = seq[-1]
    new = torch.nn.Conv2d(conv.in_channels, len(KEEP), 1, bias=True)
    new.weight.data = conv.weight.data[KEEP].clone()
    new.bias.data = conv.bias.data[KEEP].clone()
    seq[-1] = new
det.nc, det.no = len(KEEP), len(KEEP) + det.reg_max * 4
m.nc, m.yaml["nc"], m.names = len(KEEP), len(KEEP), dict(enumerate(NAMES))
y.save(out)
acts = {type(x).__name__ for x in YOLO(out).model.modules() if "LU" in type(x).__name__}
print(f"{out}: yolov8{scale}, {len(KEEP)} classes, activations {sorted(acts)}")
