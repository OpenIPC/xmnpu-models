# meva.py: surveillance-camera frames from MEVA for the person/vehicle set.
#
# MEVA (Multiview Extended Video with Activities, Kitware and IARPA, CC BY 4.0,
# https://mevadata.org) is five-minute clips from 29 fixed security cameras --
# indoor and outdoor, mounted high, looking down on people and cars -- which is
# the view a counting camera has and COCO mostly does not. Its annotations box
# only the actors of its 37 scripted activities: a car park of forty cars
# carries one box. So its boxes are merged with a teacher's, and a frame's
# labels are what both found. The teacher is torchvision's Faster R-CNN
# ResNet-50 FPN v2 (BSD-3-Clause, COCO weights), run once by `label`; its
# output is committed (meva/labels.tsv.gz) so a build needs neither it nor a
# GPU and gets the same labels every time.
#
#   meva.py select <meva-data-repo> <out keys.txt>
#       the clips: from each annotated camera, PER_CAMERA spread over the days
#   meva.py label <keys.txt> <cache> <out-dir> <meva-data-repo>
#       fetch them, hash them, label the sampled frames: <out-dir>/clips.txt,
#       <out-dir>/labels.tsv.gz and <out-dir>/frames.sha256.gz. A clip
#       already labelled in <out-dir> (same sha256) keeps its labels.
#   meva.py build <cache> <pv-dir>
#       what training runs, after dataset.py has built <pv-dir>: fetch the
#       pinned clips, check them, extract the frames, check THEM against
#       frames.sha256.gz -- the clips are pinned, but the pixels are whatever
#       this host's ffmpeg decodes them to, so a different decoder fails the
#       build here rather than training on other pictures -- and write a YOLO set,
#       <pv-dir>/meva/{images,labels}/{train,val}. The train frames join
#       pv.yaml's train list (MEVA_REPEAT times, so a few thousand frames are
#       not lost among COCO's sixty thousand) and inputs.sha256; the val split
#       -- whole cameras the model never sees in training (HELD_OUT) -- is
#       meva.yaml, for evaluate.py.
#
# Classes are the pv set's: 0 person, 1 vehicle.
import gzip
import hashlib
import os
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PINNED = f"{HERE}/meva"
BUCKET = "https://mevadata-public-01.s3.amazonaws.com"
PER_CAMERA = 12
FIRST, EVERY = 150, 300  # frames: one every ten seconds at 30 fps
HELD_OUT = {"G420", "G421", "G424"}  # a stairwell, a lobby, a car park
MEVA_REPEAT = 2
TEACHER_MIN = 0.5  # teacher boxes kept from this score
GT_NEW_IOU = 0.3  # a ground-truth box below this overlap with every kept box is added
COCO_TO_PV = {1: 0, 2: 1, 3: 1, 4: 1, 6: 1, 8: 1}  # torchvision COCO-91 ids
ANNOTATIONS = "annotation/DIVA-phase-2/MEVA/kitware-meva-training"


def camera(clip):
    return clip.split(".")[-1]


def fetch(key, path, sha=None):
    if not (os.path.exists(path) and os.path.getsize(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # A timeout per socket operation, as dataset.py's fetch has: a stalled
        # connection fails and is tried again rather than holding the build.
        for i in range(5):
            try:
                with urllib.request.urlopen(f"{BUCKET}/{key}", timeout=60) as r, \
                        open(path + ".part", "wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
                os.replace(path + ".part", path)
                break
            except OSError:
                time.sleep(2 * (i + 1))
        else:
            raise SystemExit(f"cannot fetch {key}")
    if sha:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        if h.hexdigest() != sha:
            raise SystemExit(f"{path}: sha256 {h.hexdigest()}, want {sha}")


def frames(path):
    """The sampled frame numbers of a clip, from its own frame count."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout
    return list(range(FIRST, int(out.strip().split(",")[0]), EVERY))


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def extract(path, ns, out_dir, stem, want_sha=None):
    """Frames ns of the clip as <out_dir>/<stem>_<n>.jpg, decoded in order.
    With want_sha ({file name: sha256}), frames already on disk are kept only
    if they match, and freshly decoded ones must."""
    os.makedirs(out_dir, exist_ok=True)
    want = [f"{out_dir}/{stem}_{n:05d}.jpg" for n in ns]
    def good(w):
        if not (os.path.exists(w) and os.path.getsize(w)):
            return False
        return not want_sha or file_sha256(w) == want_sha.get(os.path.basename(w))
    if all(good(w) for w in want):
        return want
    sel = "+".join(f"eq(n\\,{n})" for n in ns)
    tmp = f"{out_dir}/.{stem}_%05d.jpg"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", path, "-vf",
                    f"select={sel}", "-vsync", "0", "-q:v", "2", tmp], check=True)
    for i, w in enumerate(want):
        os.replace(tmp % (i + 1), w)
    if want_sha:
        bad = [w for w in want if not good(w)]
        if bad:
            raise SystemExit(f"{os.path.basename(bad[0])}: decodes to other pixels than "
                             f"frames.sha256.gz pins; this host's ffmpeg is not the one the "
                             f"labels were made with")
    return want


def bucket_keys(prefix):
    """Every key under prefix, from the bucket's public listing."""
    import urllib.parse
    import xml.etree.ElementTree as ET
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    keys, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        with urllib.request.urlopen(f"{BUCKET}/?{urllib.parse.urlencode(q)}") as r:
            root = ET.fromstring(r.read())
        keys += [c.find(f"{ns}Key").text for c in root.iter(f"{ns}Contents")]
        token = root.findtext(f"{ns}NextContinuationToken")
        if not token:
            return keys


def spread(items, k):
    """k of items, evenly spaced from the first to the last."""
    n = len(items)
    if n <= k:
        return list(items)
    return [items[round(i * (n - 1) / (k - 1))] for i in range(k)]


def select(repo, out):
    there = {os.path.basename(k)[:-len(".r13.avi")]: k for k in bucket_keys("drops-123-r13/")
             if k.endswith(".r13.avi")}
    clips = sorted(f[:-len(".geom.yml")] for _, _, fs in os.walk(f"{repo}/{ANNOTATIONS}")
                   for f in fs if f.endswith(".geom.yml") and f[:-len(".geom.yml")] in there)
    by_cam = {}
    for c in clips:
        by_cam.setdefault(camera(c), []).append(c)
    keys = []
    for cam, cs in sorted(by_cam.items()):
        keys += [there[c] for c in spread(cs, PER_CAMERA)]
    with open(out, "w") as f:
        f.write("\n".join(keys) + "\n")
    print(f"{len(keys)} clips from {len(by_cam)} cameras")


def gt_boxes(repo, clip):
    """MEVA's own boxes of a clip, {frame: [(cls, x1, y1, x2, y2)]}."""
    import ast
    base = None
    for root, _, fs in os.walk(f"{repo}/{ANNOTATIONS}"):
        if f"{clip}.geom.yml" in fs:
            base = f"{root}/{clip}"
    types, boxes = {}, {}
    if base is None:
        return boxes
    for line in open(f"{base}.types.yml"):
        d = ast.literal_eval(line.strip()[2:]) if line.startswith("- {") else {}
        if "types" in d:
            t = max(d["types"]["cset3"], key=d["types"]["cset3"].get)
            types[d["types"]["id1"]] = {"person": 0, "vehicle": 1, "bicycle": 1}.get(t)
    for line in open(f"{base}.geom.yml"):
        d = ast.literal_eval(line.strip()[2:]) if line.startswith("- {") else {}
        g = d.get("geom")
        if g and types.get(g["id1"]) is not None:
            boxes.setdefault(g["ts0"], []).append(
                (types[g["id1"]], *map(float, g["g0"].split())))
    return boxes


def label(keys_txt, cache, out, repo):
    import torch
    import torchvision
    from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights,
                                              fasterrcnn_resnet50_fpn_v2)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                                     box_score_thresh=TEACHER_MIN).eval().to(dev)
    keys = open(keys_txt).read().split()
    # What an earlier run in `out` labelled, kept for a clip whose bytes have
    # not changed: the teacher is the slow part, and a change of selection
    # should only pay for the clips it adds.
    old_sha, old_rows, old_frames = {}, {}, {}
    if os.path.exists(f"{out}/clips.txt"):
        for line in open(f"{out}/clips.txt"):
            if line.strip() and not line.startswith("#"):
                key, sha = line.split()
                old_sha[os.path.basename(key)[:-len(".r13.avi")]] = sha
        for line in gzip.open(f"{out}/labels.tsv.gz", "rt"):
            old_rows.setdefault(line.split("\t", 1)[0], []).append(line.rstrip("\n"))
        if os.path.exists(f"{out}/frames.sha256.gz"):
            for line in gzip.open(f"{out}/frames.sha256.gz", "rt"):
                sha, name = line.split()
                old_frames[name] = sha
    pinned, rows, hashes = [], [], []
    for k, key in enumerate(keys):
        clip = os.path.basename(key)[:-len(".r13.avi")]
        path = f"{cache}/meva/{os.path.basename(key)}"
        fetch(key, path)
        h = file_sha256(path)
        pinned.append(f"{key}\t{h}")
        ns = frames(path)
        imgs = extract(path, ns, f"{cache}/meva-frames", clip)
        hashes += [f"{file_sha256(i)}  {os.path.basename(i)}" for i in imgs]
        if old_sha.get(clip) == h and clip in old_rows:
            rows += old_rows[clip]
            print(f"{k + 1}/{len(keys)} {clip}: kept", flush=True)
            continue
        gt = gt_boxes(repo, clip)
        for n, img in zip(ns, imgs):
            x = torchvision.io.read_image(img).float().div(255).to(dev)
            H, W = x.shape[1:]
            with torch.no_grad():
                o = net([x])[0]
            kept = [(COCO_TO_PV[int(l)], *b.tolist()) for b, l in zip(o["boxes"], o["labels"])
                    if int(l) in COCO_TO_PV]
            for g in gt.get(n, []):
                same = [b for b in kept if b[0] == g[0]]
                if not same or torchvision.ops.box_iou(
                        torch.tensor([g[1:]]), torch.tensor([b[1:] for b in same])).max() < GT_NEW_IOU:
                    kept.append(g)
            boxes = []
            for c, x1, y1, x2, y2 in kept:
                x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
                if x2 - x1 > 1 and y2 - y1 > 1:
                    boxes.append(f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                                 f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
            rows.append(f"{clip}\t{n}\t" + ("|".join(boxes) or "-"))
        print(f"{k + 1}/{len(keys)} {clip}: {len(ns)} frames", flush=True)
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/clips.txt", "w") as f:
        f.write("# MEVA clips, CC BY 4.0, Kitware Inc. and IARPA: key  sha256\n")
        f.write("\n".join(pinned) + "\n")
    with gzip.GzipFile(f"{out}/labels.tsv.gz", "wb", mtime=0) as f:
        f.write(("\n".join(rows) + "\n").encode())
    with gzip.GzipFile(f"{out}/frames.sha256.gz", "wb", mtime=0) as f:
        f.write(("\n".join(hashes) + "\n").encode())


def build(cache, out):
    pinned = [line.split("\t") for line in open(f"{PINNED}/clips.txt")
              if line.strip() and not line.startswith("#")]
    frame_sha = {}
    for line in gzip.open(f"{PINNED}/frames.sha256.gz", "rt"):
        sha, name = line.split()
        frame_sha[name] = sha
    labels = {}
    for line in gzip.open(f"{PINNED}/labels.tsv.gz", "rt"):
        clip, n, boxes = line.rstrip("\n").split("\t")
        labels.setdefault(clip, {})[int(n)] = [] if boxes == "-" else boxes.split("|")
    lists = {"train": [], "val": []}
    for key, sha in pinned:
        clip = os.path.basename(key)[:-len(".r13.avi")]
        path = f"{cache}/meva/{os.path.basename(key)}"
        fetch(key, path, sha.strip())
        split = "val" if camera(clip) in HELD_OUT else "train"
        ns = sorted(labels[clip])
        imgs = extract(path, ns, f"{cache}/meva-frames", clip, frame_sha)
        img_dir, lab_dir = f"{out}/meva/images/{split}", f"{out}/meva/labels/{split}"
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lab_dir, exist_ok=True)
        for n, img in zip(ns, imgs):
            name = os.path.basename(img)
            if not os.path.lexists(f"{img_dir}/{name}"):
                os.symlink(img, f"{img_dir}/{name}")
            with open(f"{lab_dir}/{name[:-4]}.txt", "w") as f:
                f.write("".join(b + "\n" for b in labels[clip][n]))
            lists[split].append(f"./meva/images/{split}/{name}")
    with open(f"{out}/meva-train.txt", "w") as f:
        f.write("\n".join(lists["train"] * MEVA_REPEAT) + "\n")
    with open(f"{out}/meva-val.txt", "w") as f:
        f.write("\n".join(lists["val"]) + "\n")
    for split, items in lists.items():
        print(f"meva {split}: {len(items)} frames")
    # Training sees COCO and MEVA; what it validates on stays COCO's, so a
    # run's mAP stays comparable with one made before MEVA joined.
    with open(f"{out}/pv.yaml") as f:
        y = f.read()
    if "meva-train.txt" not in y:
        y = y.replace("train: train2017.txt", "train: [train2017.txt, meva-train.txt]")
        with open(f"{out}/pv.yaml", "w") as f:
            f.write(y)
    with open(f"{out}/meva.yaml", "w") as f:
        f.write(y.split("train:")[0] + "train: meva-val.txt\nval: meva-val.txt\n" +
                "names:" + y.split("names:")[1])
    with open(f"{out}/inputs.sha256", "a") as f:
        for split in ("train", "val"):
            for item in lists[split]:
                h = hashlib.sha256(open(f"{out}/{item[2:]}", "rb").read()).hexdigest()
                f.write(f"{h}  {item[2:]}\n")


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    if cmd == "select":
        select(*a)
    elif cmd == "label":
        keys_txt, cache, out, repo = a
        label(keys_txt, cache, out, repo)
    elif cmd == "build":
        build(*a)
    else:
        raise SystemExit("usage: meva.py select|label|build ... (see the header)")
