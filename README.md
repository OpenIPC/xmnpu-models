# xmnpu-models

Object detectors compiled for the NPU of the GK7205V500/V510/V530 (the XMedia
"xmnpu"), for majestic's `npuDetect` on OpenIPC firmware.

| model | input | classes | NPU time at 600 MHz |
|---|---|---|---|
| `yolov8n.xmm` | 640x384 NV21 | person, bicycle, car, motorcycle, bus, truck, cat, dog | 46-50 ms |
| `yolov8s.xmm` | 640x384 NV21 | same | 102-112 ms |

Each is YOLOv8 (Ultralytics, COCO) without its decode/NMS head: two uint8
inputs, the Y and VU planes of an NV21 frame (BT.709 limited range, converted
to RGB inside the model), and six uint8 outputs, a 64-channel DFL box map and a
class map per stride (8, 16, 32). The class channels are the COCO classes in
`models.txt`, in that order. Decoding is the consumer's job; see majestic's
`src/hisi/npudetect.c` and `docs/npu-detection.md`.

## Releases

Models are published only by the `release` workflow (Actions -> release -> Run
workflow, with a new tag), never uploaded by hand, so every asset traces to the
commit and run that built it. A release carries the `.xmm` files, `SHA256SUMS`,
and `eval.md`: every model's int8 output on held-out images -- from the NPU's
C-model, which is bit-exact with the hardware -- next to the FP32 model's on the
same pixels. The workflow fails, and publishes nothing, if an int8 detection
and an FP32 one at 0.5 or more do not match each other (same class, IoU 0.6).

## Building

```sh
pip install -r requirements-export.txt     # Python 3.11
bash scripts/build.sh wheels [out]         # needs Docker
```

`wheels/` holds the three wheels of the XMTVMC compiler that are not on PyPI,
as the XMedia SDK (`XMediaIPCLinuxV100R002C00SPC020`,
`tools/toolchains/ai_toolchains/wheels`) ships them: `caffe-1.0`,
`tvm_base-1.0` and `xmtvmc-1.0`, checked against `wheels/SHA256SUMS`. They are
the vendor's, under the vendor's terms, and are kept here only so the models
can be rebuilt. Everything else is pinned:

- `sources.txt`: the Ultralytics weights and the coco128 images, by URL and
  sha256.
- `models.txt`: what is built, at which input size, keeping which classes.
- `images.txt`: which coco128 images calibrate the quantizer and which,
  disjoint from them, evaluate the result.
- `requirements-*.txt`: the export and compiler Python environments.
- `patches/`: three fixes to XMTVMC, applied to the installed compiler.

The steps (`scripts/`): `export.py` writes head-less ONNX, rewriting C2f's
split as two 1x1 convolutions (TVM 0.7's frontend mis-sizes the ONNX Slice, and
the quantizer cannot rescale a Split) and cutting the class head down to the
kept classes; `compile.sh` runs XMTVMC's import, quantize, compile and C-model
in an `ubuntu:20.04` container; `eval.py` compares int8 with FP32.

### The patches

- `divergence.patch`: the calibrator divides by a tensor's range, which is zero
  for a tensor that is constant over the calibration set, and builds a
  histogram over a positive range that does not exist for a never-positive one.
- `pattern.patch`: the LUT builders for sigmoid/SiLU need the output dtype,
  which the shipped rewrite never passed.
- `_calibrate.patch`: with `XM_CALIB_OUTPUT_MINMAX=<floor>`, graph outputs keep
  their observed range, with the maximum at least `<floor>`. A detector's class
  maps are almost all negative over any calibration set, so the stock range
  tops out near a logit of 0 and no detection can score above about 0.5; the
  build floors it at 8 (0.9997).

## License

AGPL-3.0, as the Ultralytics YOLOv8 weights the models are compiled from. This
repository is the corresponding source of every released model. The exceptions
are `wheels/`, the XMedia SDK's own files under its terms, and the patches to
them in `patches/`, which apply to that vendor code.
