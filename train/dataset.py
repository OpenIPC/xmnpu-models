# dataset.py <coco-dir> <out-dir>
#
# Builds the 8-class training set, <out-dir>/coco8.yaml, from a COCO 2017 copy:
# <coco-dir>/train2017 (and train2017_extra) with instances_train2017.json,
# and val2017 with annotations/instances_val2017.json. Labels are remapped to
# the camera's eight classes and every other class is dropped.
#
# The copy need not be complete. Train images that hold one of the eight
# classes but are not on disk are fetched into train2017_extra, with a fixed
# sample of images that hold none of them as background; a copy holding only
# COCO's person images (as the lab's does) otherwise leaves a model with 17%
# of the cats and no background at all. Images with a cat or a dog are listed
# PETS_REPEAT times: they are the rarest of the eight.
import collections
import concurrent.futures
import json
import os
import random
import sys
import urllib.request

KEEP = [1, 2, 3, 4, 6, 8, 17, 18]  # COCO-91 ids, in output order
NAMES = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "cat", "dog"]
PETS = {17, 18}
PETS_REPEAT, BACKGROUND, SEED = 3, 6000, 0
URL = "http://images.cocodataset.org"

coco, out = (os.path.abspath(os.path.expanduser(p)) for p in sys.argv[1:3])
remap = {c: i for i, c in enumerate(KEEP)}


def fetch(url, path):
    if os.path.exists(path) and os.path.getsize(path):
        return 0
    for _ in range(3):
        try:
            urllib.request.urlretrieve(url, path + ".part")
            os.replace(path + ".part", path)
            return 1
        except OSError:
            pass
    raise SystemExit(f"cannot fetch {url}")


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
    on_disk = {}
    for sd in dirs:
        if os.path.isdir(sd):
            for f in os.listdir(sd):
                if f.endswith(".jpg") and os.path.getsize(f"{sd}/{f}"):
                    on_disk.setdefault(f, f"{sd}/{f}")
    if train:
        has = {i: {a["category_id"] for a in by_img[i]} for i in imgs}
        need = sorted(imgs[i]["file_name"] for i in imgs
                      if has[i] & set(remap) and imgs[i]["file_name"] not in on_disk)
        pool = sorted(imgs[i]["file_name"] for i in imgs if not has[i] & set(remap))
        bg = set(random.Random(SEED).sample(pool, BACKGROUND))
        need += sorted(f for f in bg if f not in on_disk)
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
    listed, pets, n_bg = [], 0, 0
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
        if train and not rows and f not in bg:
            continue  # only the fixed background sample stands in for "nothing"
        if not os.path.lexists(f"{img_dir}/{f}"):
            os.symlink(on_disk[f], f"{img_dir}/{f}")
        with open(f"{lab_dir}/{f[:-4]}.txt", "w") as lf:
            lf.write("\n".join(rows) + ("\n" if rows else ""))
        entry = f"./images/{split}/{f}"
        is_pet = any(a["category_id"] in PETS for a in by_img[i] if not a.get("iscrowd"))
        listed += [entry] * (PETS_REPEAT if train and is_pet else 1)
        pets += is_pet
        n_bg += not rows
    if train:
        random.Random(SEED).shuffle(listed)
    with open(f"{out}/{split}.txt", "w") as lf:
        lf.write("\n".join(listed) + "\n")
    print(f"{split}: {len(set(listed))} images ({pets} with a cat or dog, {n_bg} "
          f"background), {len(listed)} listed")


ensure_val()
build("train2017", f"{coco}/instances_train2017.json",
      [f"{coco}/train2017", f"{coco}/train2017_extra"], True)
build("val2017", f"{coco}/annotations/instances_val2017.json", [f"{coco}/val2017"], False)

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
with open(f"{out}/coco8.yaml", "w") as y:
    y.write(f"path: {out}\ntrain: train2017.txt\nval: val2017.txt\nnames:\n")
    y.write("".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
