# evaluate.py <weights> <data.yaml> <out.md> [reference.pt]
#
# COCO val2017 mAP50-95 per class for an 8-class checkpoint, as a markdown
# table; with [reference.pt], an 80-class COCO model, its scores for the same
# classes go beside them.
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
open(out, "w").write("COCO val2017 mAP50-95, per class\n\n" + "\n".join(rows) + "\n")
print("\n".join(rows))
