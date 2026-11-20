"""Rewrite PReLU into forms TensorRT's fusion catalog recognises, keeping the function identical.

Why: the int8 path closes on the boundary - "a per-channel PReLU cannot join an int8 epilogue, so every hop
pays a full-precision pass" (harness/docs/wsp/verified/trt-int8-calibrated.txt). PReLU is positive-homogeneous, so
the dequantise commutes through it, and whether the *builder* keeps int8 across the boundary is a graph-form
question. This rewrite is the cheap way to ask: no new weight tensors, no kernel, no plugin.

Two variants, both exact for a per-channel slope `a` of shape `[C,1,1]`:

  mul_relu  (the one the verified page uses):  a*x + (1-a)*relu(x)
  relu_add  (kept as a second algebraic form):  x + (a-1)*relu(x)

`relu_add` is NOT PReLU - it returns `x` for x < 0 - and it exists here only because the first attempt used
it: it built fine, looked 25% faster, and the fp16 control read 100.00000% of pixels off. Any conclusion
taken from it is false; the control is what caught it.

Usage:
  python3 harness/scripts/prelu_form_rewrite.py <in.onnx> <out_dir> [--variants mul_relu]
Writes <out_dir>/<stem>_mul_relu.onnx (and _relu_add.onnx when asked) and prints the node census.
"""
from __future__ import annotations

import argparse
import collections
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _const(name: str, arr: np.ndarray) -> TensorProto:
    return numpy_helper.from_array(arr.astype(np.float32), name)


def rewrite(model: onnx.ModelProto, variant: str) -> onnx.ModelProto:
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    nodes: list = []
    new_inits: list = []

    for idx, node in enumerate(g.node):
        if node.op_type != "PRelu":
            nodes.append(node)
            continue
        x, slope_name = node.input
        out = node.output[0]
        a = numpy_helper.to_array(inits[slope_name]).astype(np.float32)
        s = str(idx)

        if variant == "mul_relu":
            n_a, n_b = f"prelu_a_{s}", f"prelu_b_{s}"
            m1, r1, m2, add = (f"prelu_mul_a_{s}", f"prelu_relu_{s}",
                               f"prelu_mul_b_{s}", f"prelu_add_{s}")
            new_inits += [_const(n_a, a), _const(n_b, 1.0 - a)]
            nodes += [
                helper.make_node("Mul", [x, n_a], [m1], name=m1),
                helper.make_node("Relu", [x], [r1], name=r1),
                helper.make_node("Mul", [r1, n_b], [m2], name=m2),
                helper.make_node("Add", [m1, m2], [out], name=add),
            ]
        elif variant == "relu_add":
            n_d = f"prelu_d_{s}"
            r1, m1, add = f"prelu_relu_{s}", f"prelu_mul_d_{s}", f"prelu_add_{s}"
            new_inits.append(_const(n_d, a - 1.0))
            nodes += [
                helper.make_node("Relu", [x], [r1], name=r1),
                helper.make_node("Mul", [r1, n_d], [m1], name=m1),
                helper.make_node("Add", [x, m1], [out], name=add),
            ]
        else:
            raise SystemExit(f"unknown variant {variant}")

    used = {i for n in nodes for i in n.input}
    keep = [i for i in g.initializer if i.name in used]
    del g.node[:]
    g.node.extend(nodes)
    del g.initializer[:]
    g.initializer.extend(keep + new_inits)
    onnx.checker.check_model(model)
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("out_dir")
    ap.add_argument("--variants", default="mul_relu",
                    help="comma-separated: mul_relu, relu_add (see the module docstring)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.onnx))[0]
    for variant in [v.strip() for v in a.variants.split(",") if v.strip()]:
        m = rewrite(onnx.load(a.onnx), variant)
        path = os.path.join(a.out_dir, f"{stem}_{variant}.onnx")
        onnx.save(m, path)
        ops = collections.Counter(n.op_type for n in m.graph.node)
        print(f"{path}: {len(m.graph.node)} nodes {dict(ops.most_common())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
