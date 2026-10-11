# operating_points.py <instances_val2017.json> <dets> <out.json> [<out.md>]
#
# The thresholds behind majestic's three object-detection presets, measured
# for one model. <dets> is one detection per line, "<image id> x1 y1 x2 y2
# score class" in network-input pixels (640x384, the picture stretched into
# it, as the camera does), class being the model's channel index -- what
# npu_eval prints on the camera and fp32_dets.py prints here.
#
# A person is told about people, vehicles and animals, not COCO's
# categories, so a box counts as right when it overlaps a ground-truth box of
# the same group (bicycle, car, motorcycle, bus and truck are all "vehicle")
# by IoU 0.5. Matching is greedy by score, so the matches of the boxes above
# a threshold are the same whether or not lower ones exist, and one pass
# gives exact precision and recall at every threshold. Crowd regions are
# neither a hit nor a miss.
#
# Per group:
#   balanced          the F1-optimal threshold
#   fewerFalseAlarms  the lowest threshold with precision >= 0.90, else the
#                     F0.5 optimum
#   catchEverything   the highest threshold with recall >= 0.90, else the
#                     F2 optimum
#
# Writes {"presets": {preset: {group: threshold}}} for majestic to read
# beside the model, and a table of what each preset achieves.
import json
import sys
from collections import defaultdict

import numpy as np

W, H = 640, 384
COCO_IDS = [1, 2, 3, 4, 6, 8, 17, 18]  # the models' eight channels
GROUP = {1: "human", 2: "vehicle", 3: "vehicle", 4: "vehicle", 6: "vehicle",
         8: "vehicle", 17: "animal", 18: "animal"}
GROUPS = ["human", "vehicle", "animal"]
IOU, NMS_IOU = 0.5, 0.45  # NMS as majestic's NPU backend does it, per class
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)
PRESETS = ["fewerFalseAlarms", "balanced", "catchEverything"]


def iou(a, b):
    """a: (4,), b: (n, 4), x1 y1 x2 y2."""
    w = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
    h = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - i)


def nms(dets):
    """Per class, greedy, IoU NMS_IOU: what the camera publishes."""
    kept = []
    for c in {d[5] for d in dets}:
        mine = sorted((d for d in dets if d[5] == c), key=lambda d: -d[4])
        while mine:
            best, mine = mine[0], mine[1:]
            kept.append(best)
            if mine:
                o = iou(np.array(best[:4]), np.array([m[:4] for m in mine]))
                mine = [m for m, v in zip(mine, o) if v <= NMS_IOU]
    return kept


def main(ann_path, det_path, out_json, out_md=None):
    ann = json.load(open(ann_path))
    size = {im["id"]: (im["width"], im["height"]) for im in ann["images"]}
    gt = defaultdict(lambda: defaultdict(list))     # image -> group -> boxes
    crowd = defaultdict(lambda: defaultdict(list))
    for a in ann["annotations"]:
        g = GROUP.get(a["category_id"])
        if not g:
            continue
        x, y, w, h = a["bbox"]
        (crowd if a["iscrowd"] else gt)[a["image_id"]][g].append((x, y, x + w, y + h))

    dets = defaultdict(list)
    for line in open(det_path):
        p = line.split()
        if len(p) != 7 or not p[0].isdigit():
            continue  # libxmedia_cl prints a version banner to stdout
        iid = int(p[0])
        sx, sy = size[iid][0] / W, size[iid][1] / H
        x1, y1, x2, y2 = (float(v) for v in p[1:5])
        dets[iid].append((x1 * sx, y1 * sy, x2 * sx, y2 * sy, float(p[5]), int(p[6])))

    scores = {g: [] for g in GROUPS}   # (score, is_hit)
    npos = {g: 0 for g in GROUPS}
    # Read before the loop below indexes gt and crowd, which creates entries.
    empty_images = {i for i in size if not gt.get(i) and not crowd.get(i)}
    empty_scores = {g: [] for g in GROUPS}   # false alarms on images with nothing
    for iid in size:
        kept = nms(dets.get(iid, []))
        for g in GROUPS:
            mine = sorted((d for d in kept if GROUP[COCO_IDS[d[5]]] == g),
                          key=lambda d: -d[4])
            truth = np.array(gt[iid][g]).reshape(-1, 4)
            crowds = np.array(crowd[iid][g]).reshape(-1, 4)
            npos[g] += len(truth)
            used = np.zeros(len(truth), bool)
            for d in mine:
                box = np.array(d[:4])
                hit = False
                if len(truth):
                    o = np.where(used, -1, iou(box, truth))
                    j = int(o.argmax())
                    if o[j] >= IOU:
                        used[j] = hit = True
                if not hit and len(crowds):
                    # inside a crowd region: COCO's rule, area of the box
                    # covered, not IoU
                    w = np.clip(np.minimum(box[2], crowds[:, 2]) - np.maximum(box[0], crowds[:, 0]), 0, None)
                    h = np.clip(np.minimum(box[3], crowds[:, 3]) - np.maximum(box[1], crowds[:, 1]), 0, None)
                    if (w * h / max((box[2] - box[0]) * (box[3] - box[1]), 1e-9)).max() >= IOU:
                        continue
                scores[g].append((d[4], hit))
                if iid in empty_images:
                    empty_scores[g].append(d[4])

    presets = {p: {} for p in PRESETS}
    rows = []
    for g in GROUPS:
        s = np.array([x[0] for x in scores[g]])
        hit = np.array([x[1] for x in scores[g]], bool)
        tp = np.array([(hit & (s >= t)).sum() for t in THRESHOLDS], float)
        fp = np.array([(~hit & (s >= t)).sum() for t in THRESHOLDS], float)
        p = np.divide(tp, tp + fp, out=np.ones_like(tp), where=tp + fp > 0)
        r = tp / max(npos[g], 1)
        f = lambda b: (1 + b * b) * p * r / np.maximum(b * b * p + r, 1e-9)
        idx = {"balanced": int(f(1).argmax())}
        ok = np.nonzero(p >= 0.90)[0]
        idx["fewerFalseAlarms"] = int(ok[0]) if len(ok) else int(f(0.5).argmax())
        ok = np.nonzero(r >= 0.90)[0]
        idx["catchEverything"] = int(ok[-1]) if len(ok) else int(f(2).argmax())
        es = np.array(empty_scores[g])
        for name in PRESETS:
            i = idx[name]
            t = float(THRESHOLDS[i])
            presets[name][g] = t
            fpi = (es >= t).sum() / max(len(empty_images), 1)
            rows.append(f"| {g} | {name} | {t:.2f} | {p[i]:.3f} | {r[i]:.3f} | "
                        f"{2 * p[i] * r[i] / max(p[i] + r[i], 1e-9):.3f} | {fpi:.4f} |")

    json.dump({"presets": presets}, open(out_json, "w"), indent=1, sort_keys=True)
    table = "\n".join([
        f"COCO val2017, {len(size)} images ({len(empty_images)} with none of the "
        f"eight classes); ground truth: {npos['human']} people, "
        f"{npos['vehicle']} vehicles, {npos['animal']} animals.",
        "",
        "| class | preset | threshold | precision | recall | F1 | false alarms per empty image |",
        "|---|---|---|---|---|---|---|"] + rows) + "\n"
    if out_md:
        open(out_md, "w").write(table)
    print(table)


if __name__ == "__main__":
    main(*sys.argv[1:])
