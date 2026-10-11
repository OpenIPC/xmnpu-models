# prep_nv21.py <val2017-dir> <out-dir> [<w> <h>]
#
# Every COCO image as the camera hands a frame to the network: the whole
# picture stretched to the input (640x384 by default), as the VPSS scales it,
# and converted to NV21, BT.709 limited range -- the input_format the models
# are compiled for.
import glob
import os
import sys

import cv2
import numpy as np


def nv21(rgb, w, h):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 16 + (0.2126 * r + 0.7152 * g + 0.0722 * b) * 219 / 255
    cb = 128 + (-0.1146 * r - 0.3854 * g + 0.5 * b) * 224 / 255
    cr = 128 + (0.5 * r - 0.4542 * g - 0.0458 * b) * 224 / 255
    sub = lambda c: c.reshape(h // 2, 2, w // 2, 2).mean((1, 3))
    vu = np.stack([sub(cr), sub(cb)], -1).reshape(h // 2, w)
    return (np.clip(y, 0, 255).round().astype(np.uint8),
            np.clip(vu, 0, 255).round().astype(np.uint8))


def main(src, dst, w=640, h=384):
    w, h = int(w), int(h)
    os.makedirs(dst, exist_ok=True)
    for f in sorted(glob.glob(f"{src}/*.jpg")):
        out = f"{dst}/{os.path.basename(f)[:-4]}.nv21"
        rgb = cv2.resize(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB), (w, h),
                         interpolation=cv2.INTER_AREA).astype(np.float32)
        y, vu = nv21(rgb, w, h)
        with open(out, "wb") as o:
            o.write(y.tobytes() + vu.tobytes())


if __name__ == "__main__":
    main(*sys.argv[1:])
