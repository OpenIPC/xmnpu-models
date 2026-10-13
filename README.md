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

Each model also comes with `<model>.json`, installed beside it, which is what
majestic's three object-detection presets mean for that model: per class
(human, vehicle, animal), the threshold for

- `balanced`: the F1-optimal one;
- `fewerFalseAlarms`: the F0.5 optimum (precision weighted);
- `catchEverything`: the highest keeping recall at 0.90, else the F2 optimum.

They are measured on all of COCO val2017 from the int8 model itself, on the
frames the camera gives it (the picture stretched to the input, NV21), with a
box counted right at IoU 0.5 against its class group; `points.md` in the
release shows what each preset achieves. The `points` jobs run the quantized
graph's `xmnpu_x86` build, whose outputs are bit-identical to the C-model's
(and so the hardware's) at a fraction of its cost, in shards.

## Training

The checkpoints the build compiles are trained by the `train` workflow, by hand
only, on the self-hosted GPU runner, and published as releases of their own
(`train/run.sh` does the work):

- `yolov8n`, `yolov8s`: COCO YOLOv8 fine-tuned with LeakyReLU(0.1) in place of
  SiLU -- the NPU fuses it into the convolution, where SiLU costs a pass of
  its own -- and cut to the eight classes above.
- `pvdn` (`train/pvdn.yaml`): a person/vehicle detector trained from scratch,
  640x352. Its body is the layout of the person/vehicle model camera vendors
  ship for customer counting on these SoCs -- a slimmed YOLOv5n: a stride-4
  stem, C3 stages of 32/64/120/232 channels, SPPF, a PAN neck -- with
  YOLOv8's head, so it is built, gated and decoded like the others. Two
  classes: person, and vehicle (COCO's bicycle, car, motorcycle, bus and truck
  as one).

  It trains on COCO and on frames of security-camera video from MEVA
  (`train/meva.py`): fixed cameras mounted high over school, bus-depot,
  hospital and office grounds, indoors and out, the view a counting camera
  has and COCO mostly lacks. MEVA's own boxes cover only the actors of its
  scripted activities, so each frame's labels are those merged with a
  teacher's (torchvision's Faster R-CNN v2); the merged labels are committed
  in `train/meva/`, so a build needs no teacher. Three cameras are held out of
  training, and `eval.md` scores them beside COCO val2017.

### Training data

| set | licence | used for |
|---|---|---|
| COCO 2017 | annotations CC BY 4.0; images under Flickr terms | every model |
| [MEVA](https://mevadata.org) | CC BY 4.0 (video and annotations) | pvdn |

"Multiview Extended Video with Activities" (MEVA) dataset by Kitware Inc. and
the Intelligence Advanced Research Projects Activity (IARPA), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The labels in
`train/meva/labels.tsv.gz` are derived from it (merged with a model's) and are
under the same licence.

Sets that were considered and are not used, because their terms rule out
commercial use or redistribution: CrowdHuman, HABBOF, CEPDOF, WEPDTOF, MOT17/20
and VIRAT.

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
  sha256; `coco.txt`: COCO val2017 and its annotations, for the presets.
- `models.txt`: what is built, at which input size, keeping which classes.
- `images.txt`: which coco128 images calibrate the quantizer and which,
  disjoint from them, evaluate the result.
- `requirements-*.txt`: the export and compiler Python environments.
- `patches/`: three fixes to XMTVMC, applied to the installed compiler.

The steps (`scripts/`): `export.py` writes head-less ONNX, rewriting C2f's
split as two 1x1 convolutions (TVM 0.7's frontend mis-sizes the ONNX Slice, and
the quantizer cannot rescale a Split) and cutting the class head down to the
kept classes; `compile.sh` runs XMTVMC's import, quantize, compile and C-model
in an `ubuntu:20.04` container; `sanitize.py` zeroes the host memory XMTVMC
leaves in each `.xmm` -- host pointers, and struct padding holding the upper
half of one, which keeps its fixed unrandomised value (they follow the build
machine's memory layout and mean nothing on the camera) -- so a rebuild of the
same commit is bit-identical;
`eval.py` compares int8 with FP32; `check_x86.py` holds the x86 build to the
C-model's exact outputs on the eval images. For the presets, `prep_nv21.py` makes the
camera's frames, `simulate.sh` runs `sim_dets.py` over them and
`operating_points.py` scores the result.

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
