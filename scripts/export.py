# export.py <weights.pt> <H> <W> <classes> <out.onnx>
#
# Exports a YOLOv8 model to ONNX without its decode/NMS head: the outputs are
# the raw per-scale convolution maps, box (64 DFL bins) then class, which is
# what an NPU should run; decoding stays on the CPU (majestic's npudetect.c).
#
# <classes> is a comma-separated list of COCO class ids to keep in the class
# head, in output-channel order; the last convolution of each class branch is
# cut down to those rows, so nothing else in the network changes.
import sys
import torch
import torch.nn.functional as F
import onnx
import onnxsim
from ultralytics import YOLO
from ultralytics.nn.modules import block


# C2f splits cv1's output in two. As ONNX Slice, TVM 0.7's frontend mis-sizes
# it; as Split, XMTVMC's quantizer cannot rescale the Concat it feeds. With BN
# folded into cv1 (model.fuse()), the two halves are just two 1x1 convolutions
# with half the weights each.
def _c2f_forward(self, x):
    w, b, c = self.cv1.conv.weight, self.cv1.conv.bias, self.c
    y = [self.cv1.act(F.conv2d(x, w[:c], b[:c])),
         self.cv1.act(F.conv2d(x, w[c:], b[c:]))]
    y.extend(m(y[-1]) for m in self.m)
    return self.cv2(torch.cat(y, 1))


block.C2f.forward = _c2f_forward

weights, h, w, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[5]
keep = [int(c) for c in sys.argv[4].split(",")]
m = YOLO(weights).model.fuse(verbose=False).eval()
det = m.model[-1]
for seq in det.cv3:  # last layer of each class branch: Conv2d(c, nc, 1)
    conv = seq[-1]
    new = torch.nn.Conv2d(conv.in_channels, len(keep), 1, bias=True)
    new.weight.data = conv.weight.data[keep].clone()
    new.bias.data = conv.bias.data[keep].clone()
    seq[-1] = new


class Body(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        y = []
        for layer in self.m.model[:-1]:
            if layer.f != -1:
                x = y[layer.f] if isinstance(layer.f, int) else \
                    [x if j == -1 else y[j] for j in layer.f]
            x = layer(x)
            y.append(x)
        xs = [y[j] for j in det.f]
        # Separate outputs rather than one Concat: XMTVMC cannot end a graph
        # on a Concat.
        return tuple(t for i in range(det.nl)
                     for t in (det.cv2[i](xs[i]), det.cv3[i](xs[i])))


torch.onnx.export(Body(m), torch.zeros(1, 3, h, w), out, opset_version=11,
                  input_names=["images"],
                  output_names=[f"p{i}{k}" for i in range(det.nl) for k in "bc"],
                  dynamo=False)
mo, ok = onnxsim.simplify(onnx.load(out))
assert ok, "onnxsim could not validate the simplified graph"
onnx.save(mo, out)
print(out, sorted({n.op_type for n in mo.graph.node}))
