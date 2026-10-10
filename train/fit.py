# fit.py <init.pt> <data.yaml> <epochs> <batch> <out-dir>: the fine-tune.
import os
import sys

from ultralytics import YOLO

init, data, epochs, batch, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
YOLO(init).train(data=data, epochs=epochs, batch=batch, imgsz=640, device=0, workers=8,
                 amp=True, cos_lr=True, close_mosaic=10, patience=0, plots=False,
                 project=os.path.dirname(os.path.abspath(out)), name=os.path.basename(out),
                 exist_ok=True)
