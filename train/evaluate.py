# evaluate.py <weights> <data.yaml> <out.md> [reference.pt]
#
# COCO val2017 mAP50-95 per class for a checkpoint, as a markdown table (or
# the MEVA cameras held out of training, for meva.yaml, see meva.py); with
# [reference.pt], an 80-class COCO model, its scores for the same classes go
# beside them (for the 8-class models, whose classes are COCO's own).
import sys

from ultralytics import YOLO

KEEP80 = [0, 1, 2, 3, 5, 7, 15, 16]
w, data, out = sys.argv[1:4]
r = YOLO(w).val(data=data, imgsz=640, batch=32, device=0, verbose=False, plots=False)
names, ours = r.names, [r.box.maps[i] for i in range(len(r.names))]
ref = None
if len(sys.argv) > 4:
    rr = YOLO(sys.argv[4]).val(data=data.replace("coco8.yaml", "coco80.yaml"), imgsz=640,
                                batch=32, device=0, verbose=False, plots=False)
    ref = [rr.box.maps[c] for c in KEEP80]
head = "| class | this model |" + (" reference |" if ref else "")
rows = [head, "|---|---|" + ("---|" if ref else "")]
for i, n in names.items():
    rows.append(f"| {n} | {ours[i] * 100:.1f} |" + (f" {ref[i] * 100:.1f} |" if ref else ""))
rows.append(f"| **mean** | **{sum(ours) / len(ours) * 100:.1f}** |" +
            (f" **{sum(ref) / len(ref) * 100:.1f}** |" if ref else ""))
# The MEVA cameras' labels are MEVA's boxes merged with a teacher model's
# (meva.py), not boxes checked by hand, so that table measures agreement with
# that reference -- it can say a model learned the camera view, not that it
# found someone the teacher missed. The heading says so.
what = ("MEVA held-out cameras (labels: MEVA's own boxes merged with a teacher "
        "model's, not checked by hand)" if data.endswith("meva.yaml") else "COCO val2017")
open(out, "w").write(f"{what} mAP50-95, per class\n\n" + "\n".join(rows) + "\n")
print("\n".join(rows))
