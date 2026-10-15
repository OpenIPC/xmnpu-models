# dataset.py <coco-dir> <out-dir> [c8|pv]
#
# Builds a training set from a COCO 2017 copy: <coco-dir>/train2017 (and
# train2017_extra) with instances_train2017.json, and val2017 with
# annotations/instances_val2017.json. Labels are remapped and every other class
# is dropped:
#   c8 (default)  the camera's eight classes, <out-dir>/coco8.yaml
#   pv            two: person, and vehicle (bicycle, car, motorcycle, bus and
#                 truck as one class), <out-dir>/pv.yaml
#
# The copy need not be complete. Train images that hold one of the eight
# classes but are not on disk are fetched into train2017_extra, with a fixed
# sample of images that hold none of them as background; a copy holding only
# COCO's person images (as the lab's does) otherwise leaves a model with 17%
# of the cats and no background at all. Images with a cat or a dog are listed
# PETS_REPEAT times where the set keeps them: they are the rarest of the eight.
#
# The person/vehicle set also lists every train image with a fire hydrant, a
# stop sign or a parking meter HARD_REPEAT times, whether or not it holds a
# person or a vehicle: a person/vehicle model took a red hydrant in a car park
# for a person in every frame of a held-out camera (pvdn-v1), and those three
# are the street furniture COCO names. Their people and vehicles stay labelled
# -- COCO labels every person -- and the object itself is background.
import collections
import concurrent.futures
import hashlib
import json
import os
import random
import shutil
import sys
import time
import urllib.request

# Per socket operation, not per file: a download that is moving, however
# slowly, is never cut off, and one whose server has gone quiet fails and is
# tried again. urlretrieve has no timeout at all -- sixteen fetch threads once
# sat for an hour on connections that had stopped mid-file, and the training
# job behind them would have waited out its three days.
#
# The waits between tries double up to a minute, so the tries span about a
# quarter of an hour: images.cocodataset.org goes quiet for several minutes at
# a time under a long fetch, and five tries a few seconds apart (five and a
# half minutes in all) were used up inside one such spell.
FETCH_TIMEOUT, FETCH_TRIES = 60, 10


def backoff(i):
    return min(60, 2 << i)

# COCO-91 id -> output class, and the output classes' names, per set.
SETS = {
    "c8": ({1: 0, 2: 1, 3: 2, 4: 3, 6: 4, 8: 5, 17: 6, 18: 7},
           ["person", "bicycle", "car", "motorcycle", "bus", "truck", "cat", "dog"]),
    "pv": ({1: 0, 2: 1, 3: 1, 4: 1, 6: 1, 8: 1}, ["person", "vehicle"]),
}
# The annotations every set is built from, pinned: a copy that differs would
# change what is trained on without anything else noticing.
ANNOTATIONS = {
    "instances_train2017.json": "610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d",
    "instances_val2017.json": "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f",
}
PETS = {17, 18}
PETS_REPEAT, BACKGROUND, SEED = 3, 6000, 0
HARD = {11, 13, 14}  # fire hydrant, stop sign, parking meter
HARD_SETS, HARD_REPEAT = {"pv"}, 2
URL = "http://images.cocodataset.org"

coco, out = (os.path.abspath(os.path.expanduser(p)) for p in sys.argv[1:3])
kind = sys.argv[3] if len(sys.argv) > 3 else "c8"
remap, NAMES = SETS[kind]


def fetch(url, path):
    if os.path.exists(path) and os.path.getsize(path):
        return 0
    for i in range(FETCH_TRIES):
        try:
            with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT) as r, \
                    open(path + ".part", "wb") as f:
                shutil.copyfileobj(r, f)
                # A connection closed early is an EOF, not an error: without
                # this the cut-off image would be cached as a whole one.
                want = r.headers.get("Content-Length")
                if want is not None and f.tell() != int(want):
                    raise OSError(f"short read: {f.tell()} of {want} bytes")
            os.replace(path + ".part", path)
            return 1
        except (OSError, ValueError):  # ValueError: a garbled Content-Length
            if i + 1 < FETCH_TRIES:  # not after the last: nothing follows it
                time.sleep(backoff(i))
    raise SystemExit(f"cannot fetch {url}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_annotations(path):
    want = ANNOTATIONS[os.path.basename(path)]
    if sha256(path) != want:
        sys.exit(f"{path} is not the COCO 2017 release's (sha256 should be {want})")


def ensure_val():
    if not os.path.isdir(f"{coco}/val2017"):
        sys.exit(f"{coco}/val2017 is missing (val2017.zip from {URL}/zips)")
    if not os.path.exists(f"{coco}/annotations/instances_val2017.json"):
        sys.exit(f"{coco}/annotations/instances_val2017.json is missing")


def build(split, ann, dirs, train):
    d = json.load(open(ann))
    imgs = {i["id"]: i for i in d["images"]}
    by_img = collections.defaultdict(list)
    for a in d["annotations"]:
        by_img[a["image_id"]].append(a)
    on_disk, hard = {}, set()
    for sd in dirs:
        if os.path.isdir(sd):
            for f in os.listdir(sd):
                if f.endswith(".jpg") and os.path.getsize(f"{sd}/{f}"):
                    on_disk.setdefault(f, f"{sd}/{f}")
    if train:
        has = {i: {a["category_id"] for a in by_img[i]} for i in imgs}
        need = sorted(imgs[i]["file_name"] for i in imgs
                      if has[i] & set(remap) and imgs[i]["file_name"] not in on_disk)
        # The background sample is the same for every class set: images
        # holding none of the eight classes, which hold nothing any set
        # keeps. So it does not depend on the set (or on what is on disk), the
        # eight-class models' sample is the one they were trained with, and a
        # host that built one set already holds the others' backgrounds.
        c8 = set(SETS["c8"][0])
        pool = sorted(imgs[i]["file_name"] for i in imgs if not has[i] & c8)
        bg = set(random.Random(SEED).sample(pool, BACKGROUND))
        need += sorted(f for f in bg if f not in on_disk)
        if kind in HARD_SETS:
            hard = {imgs[i]["file_name"] for i in imgs if has[i] & HARD}
            need += sorted(f for f in hard - bg if f not in on_disk and f not in need)
        extra = f"{coco}/{split}_extra"
        os.makedirs(extra, exist_ok=True)
        with concurrent.futures.ThreadPoolExecutor(16) as ex:
            got = sum(ex.map(lambda f: fetch(f"{URL}/{split}/{f}", f"{extra}/{f}"), need))
        print(f"{split}: fetched {got} of {len(need)} missing images")
        for f in need:
            on_disk.setdefault(f, f"{extra}/{f}")
    img_dir, lab_dir = f"{out}/images/{split}", f"{out}/labels/{split}"
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)
    listed, pets, n_bg, n_hard = [], 0, 0, 0
    for i, im in sorted(imgs.items()):
        f = im["file_name"]
        if f not in on_disk:
            continue
        rows = []
        for a in by_img[i]:
            x, y, w, h = a["bbox"]
            if a.get("iscrowd") or a["category_id"] not in remap or w <= 1 or h <= 1:
                continue
            W, H = im["width"], im["height"]
            rows.append(f"{remap[a['category_id']]} {(x + w / 2) / W:.6f} "
                        f"{(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
        is_hard = train and f in hard
        if train and not rows and f not in bg and not is_hard:
            continue  # only the fixed background sample stands in for "nothing"
        if not os.path.lexists(f"{img_dir}/{f}"):
            os.symlink(on_disk[f], f"{img_dir}/{f}")
        with open(f"{lab_dir}/{f[:-4]}.txt", "w") as lf:
            lf.write("\n".join(rows) + ("\n" if rows else ""))
        entry = f"./images/{split}/{f}"
        is_pet = any(a["category_id"] in PETS & set(remap)
                     for a in by_img[i] if not a.get("iscrowd"))
        listed += [entry] * (PETS_REPEAT if train and is_pet
                             else HARD_REPEAT if is_hard else 1)
        pets += is_pet
        n_hard += is_hard
        n_bg += not rows
    if train:
        random.Random(SEED).shuffle(listed)
    with open(f"{out}/{split}.txt", "w") as lf:
        lf.write("\n".join(listed) + "\n")
    print(f"{split}: {len(set(listed))} images ({pets} with a cat or dog, {n_hard} "
          f"with a hydrant, stop sign or parking meter, {n_bg} background), "
          f"{len(listed)} listed")


ensure_val()
check_annotations(f"{coco}/instances_train2017.json")
check_annotations(f"{coco}/annotations/instances_val2017.json")
build("train2017", f"{coco}/instances_train2017.json",
      [f"{coco}/train2017", f"{coco}/train2017_extra"], True)
build("val2017", f"{coco}/annotations/instances_val2017.json", [f"{coco}/val2017"], False)

# What was trained and validated on, image by image: published with the
# checkpoint (run.sh), so a release can be tied to its exact inputs even though
# the images themselves are fetched rather than pinned one by one.
listed = []
for split in ("train2017", "val2017"):
    listed += sorted({l.strip() for l in open(f"{out}/{split}.txt") if l.strip()})
with open(f"{out}/inputs.sha256", "w") as mf:
    with concurrent.futures.ThreadPoolExecutor(8) as ex:
        for rel, h in zip(listed, ex.map(lambda r: sha256(f"{out}/{r}"), listed)):
            mf.write(f"{h}  {rel[2:]}\n")
print(f"inputs.sha256: {len(listed)} images")

# val2017 with all 80 classes, so evaluate.py can score an 80-class reference
# (the stock SiLU model) on the very same images and boxes.
d = json.load(open(f"{coco}/annotations/instances_val2017.json"))
cid = {c: i for i, c in enumerate(sorted(c["id"] for c in d["categories"]))}
imgs = {i["id"]: i for i in d["images"]}
os.makedirs(f"{out}/coco80/labels/val2017", exist_ok=True)
if not os.path.lexists(f"{out}/coco80/images"):
    os.symlink(f"{out}/images", f"{out}/coco80/images")
rows = collections.defaultdict(list)
for a in d["annotations"]:
    x, y, w, h = a["bbox"]
    if a.get("iscrowd") or w <= 1 or h <= 1:
        continue
    im = imgs[a["image_id"]]
    rows[im["file_name"]].append(f"{cid[a['category_id']]} {(x + w / 2) / im['width']:.6f} "
                                 f"{(y + h / 2) / im['height']:.6f} {w / im['width']:.6f} "
                                 f"{h / im['height']:.6f}")
listed80 = []
for im in sorted(imgs.values(), key=lambda im: im["file_name"]):
    f = im["file_name"]
    with open(f"{out}/coco80/labels/val2017/{f[:-4]}.txt", "w") as lf:
        lf.write("\n".join(rows[f]) + ("\n" if rows[f] else ""))
    listed80.append(f"./images/val2017/{f}")
# its own list, so each image resolves through coco80/images to coco80/labels
with open(f"{out}/coco80/val2017.txt", "w") as lf:
    lf.write("\n".join(listed80) + "\n")
with open(f"{out}/coco80.yaml", "w") as y:
    names80 = [c["name"] for c in sorted(d["categories"], key=lambda c: c["id"])]
    y.write(f"path: {out}/coco80\ntrain: val2017.txt\nval: val2017.txt\nnames:\n")
    y.write("".join(f"  {i}: {n}\n" for i, n in enumerate(names80)))
with open(f"{out}/{'coco8' if kind == 'c8' else kind}.yaml", "w") as y:
    y.write(f"path: {out}\ntrain: train2017.txt\nval: val2017.txt\nnames:\n")
    y.write("".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
