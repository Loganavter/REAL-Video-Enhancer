"""Calibrate and build an explicit Q/DQ model with torch on the GPU - no onnxruntime.

Why not onnxruntime's quantize_static: it runs the model once per calibration frame on the CPU,
which is 8 x 2.5 TFLOP of convolutions on 16 threads with an arena on top. On this host that made the
desktop stutter ("almost killed mutter", 2026-11-17). The same ranges come from forward hooks on the
GPU in about half a second per frame.

What it reproduces (ledger E9'): per-tensor activation scales calibrated on real frames, per-channel
int8 weights, the 16 body convolutions quantised, head and tail left in fp16.

Usage:
  python3 scripts/make_qdq.py <model.pth> <onnx-in> <onnx-out> --frames DIR --width W --height H
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

import torch
import torch.nn as nn
import torch.nn.functional as F


class Compact(nn.Module):
    def __init__(self, state: dict) -> None:
        super().__init__()
        self.body = nn.ModuleList()
        i = 0
        while (key := f"body.{i}.weight") in state:
            w = state[key]
            self.body.append(nn.Conv2d(w.shape[1], w.shape[0], 3, 1, 1) if w.dim() == 4
                             else nn.PReLU(num_parameters=w.shape[0]))
            i += 1
        self.scale = int(round((state[f"body.{i - 1}.weight"].shape[0] / 3) ** 0.5))
        self.upsampler = nn.PixelShuffle(self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clamp(0.0, 1.0)
        for layer in self.body:
            out = layer(out)
        return self.upsampler(out) + F.interpolate(x, scale_factor=self.scale, mode="nearest")


def calibrate(model: nn.Module, paths: list[str], w: int, h: int) -> list[tuple[float, float]]:
    """Per-conv input range over the frames, on the GPU, batch of one."""
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    ranges = [[float("inf"), float("-inf")] for _ in convs]
    hooks = []

    def make_hook(idx: int):
        def hook(_mod, inp, _out):
            x = inp[0]
            lo, hi = float(x.min()), float(x.max())
            ranges[idx][0] = min(ranges[idx][0], lo)
            ranges[idx][1] = max(ranges[idx][1], hi)
        return hook

    for i, c in enumerate(convs):
        hooks.append(c.register_forward_hook(make_hook(i)))
    model.eval().cuda()
    with torch.no_grad():
        for p in paths:
            raw = np.fromfile(p, dtype=np.uint8).reshape(h, w, 3).transpose(2, 0, 1)
            x = torch.from_numpy(raw).float().div(255.0)[None].cuda()
            model(x)
            del x
            torch.cuda.empty_cache()
    for hk in hooks:
        hk.remove()
    return [(lo, hi) for lo, hi in ranges]


def qdq_graph(onnx_in: str, onnx_out: str, ranges: list[tuple[float, float]],
              keep_fp16_head_tail: bool = True) -> dict:
    m = onnx.load(onnx_in)
    inits = {i.name: i for i in m.graph.initializer}
    conv_idx = 0
    conv_total = sum(1 for n in m.graph.node if n.op_type == "Conv")
    new_nodes: list = []
    added = {"q": 0, "dq": 0, "wq": 0}
    counters = {"t": 0}

    def uniq(prefix: str) -> str:
        counters["t"] += 1
        return f"{prefix}_{counters['t']}"

    for node in m.graph.node:
        if node.op_type != "Conv":
            new_nodes.append(node)
            continue
        first, last = conv_idx == 0, conv_idx == conv_total - 1
        wname = node.input[1]
        w = numpy_helper.to_array(inits[wname])
        # per-channel symmetric int8 weights
        amax = np.abs(w).reshape(w.shape[0], -1).max(axis=1)
        wscale = (amax / 127.0).astype(np.float32)
        wscale[wscale == 0] = 1e-8
        wq = np.clip(np.round(w / wscale.reshape(-1, *([1] * (w.ndim - 1)))), -127, 127).astype(np.int8)
        wq_name, wscale_name, wdq_name = uniq("wq"), uniq("ws"), uniq("wdq")
        # the int8 weights are an INITIALISER, not a runtime Q: emitting both defines the name twice
        new_nodes.append(helper.make_node("DequantizeLinear", [wq_name, wscale_name], [wdq_name], axis=0))
        node.input[1] = wdq_name
        pending = getattr(qdq_graph, "_pending", [])
        pending.append(numpy_helper.from_array(wq, wq_name))
        pending.append(numpy_helper.from_array(wscale, wscale_name))
        qdq_graph._pending = pending
        added["wq"] += 1

        lo, hi = ranges[conv_idx] if conv_idx < len(ranges) else (0.0, 1.0)
        if not (keep_fp16_head_tail and (first or last)):
            s = max(abs(lo), abs(hi)) / 127.0
            s = s if s > 0 else 1e-8
            s_name = uniq("as")
            q_name, dq_name = uniq("aq"), uniq("ad")
            new_nodes.append(helper.make_node("QuantizeLinear", [node.input[0], s_name], [q_name]))
            new_nodes.append(helper.make_node("DequantizeLinear", [q_name, s_name], [dq_name]))
            pending = qdq_graph._pending
            pending.append(numpy_helper.from_array(np.array(s, dtype=np.float32), s_name))
            node.input[0] = dq_name
            added["q"] += 1
            added["dq"] += 1

        new_nodes.append(node)           # the conv itself, quantised on both sides
        conv_idx += 1

    del m.graph.node[:]
    m.graph.node.extend(new_nodes)
    m.graph.initializer.extend(getattr(qdq_graph, "_pending", []))
    onnx.save(m, onnx_out)
    return {"convs": conv_total, "activation_qdq_pairs": added["q"], "quantised_weights": added["wq"],
            "ranges": [[round(lo, 4), round(hi, 4)] for lo, hi in ranges]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("onnx_in")
    ap.add_argument("onnx_out")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.frames, "*.raw")))
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    state = state.get("params", state)
    model = Compact(state)
    model.load_state_dict(state, strict=True)
    ranges = calibrate(model, paths, args.width, args.height)
    print("calibrated on", len(paths), "frames:", [(round(a, 3), round(b, 3)) for a, b in ranges])

    stats = qdq_graph(args.onnx_in, args.onnx_out, ranges)
    manifest = {"source": args.onnx_in, "qdq": args.onnx_out,
                "frames": [{"file": os.path.basename(p),
                            "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()} for p in paths],
                "calibration": "torch forward hooks on the GPU, per-tensor min/max, batch 1",
                **stats}
    with open(args.onnx_out + ".manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"qdq -> {args.onnx_out}: {stats['activation_qdq_pairs']} activation Q/DQ pairs, "
          f"{stats['quantised_weights']} per-channel int8 weights, head/tail fp16")
    return 0


if __name__ == "__main__":
    sys.exit(main())
