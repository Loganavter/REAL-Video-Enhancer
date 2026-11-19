"""SPAN scope: where does the time actually go? A measurement, not a kernel.

The row asks one question before any int8 work on the SPAN family: if only the 3x3 convolutions could be quantised,
would there be anything left to win? SPAN is not a plain conv stack - it has 1x1 expanders, a parameter-free
attention block (an elementwise sigmoid multiply), and a four-way concatenation fusion - so the answer is a share
of the frame, and it decides whether the family is worth a kernel at all.

Method: `torch.profiler` over a few forwards, aggregated by operator category, with the total cross-checked by
CUDA events. No hand instrumentation, so nothing is attributed by guesswork.

Usage: python scripts/span_scope_probe.py [--model ...] [--iters 5] [--out-file x.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import torch

APP = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(APP, "backend", "src", "pytorch"))
sys.path.insert(0, os.path.join(REPO, "backend"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from model_loader import read_shape  # noqa: E402
from src.pytorch.TorchUtils import TorchUtils  # noqa: E402
from tail_int8_emulation import CALIB, H, W, decode_yuv10  # noqa: E402

ROOT = "/home/jorj/Загрузки/Upscaling"
DEFAULT = "2x_ModernSpanimationV1.pth"

CATS = (
    ("conv3x3", re.compile(r"aten::(convolution|conv2d|_convolution)")),
    ("cat", re.compile(r"aten::cat")),
    ("pixel_shuffle", re.compile(r"aten::pixel_shuffle")),
    ("upsample", re.compile(r"aten::upsample")),
    ("act", re.compile(r"aten::(silu|leaky_relu|prelu|relu|elu|gelu)")),
    ("elementwise", re.compile(r"aten::(mul|add|sub|div|sigmoid|exp|pow|mean|rsqrt|sqrt|clamp|neg|tanh)")),
)


def find(name: str) -> str:
    for dirpath, _dirs, files in os.walk(ROOT):
        if name in files:
            return os.path.join(dirpath, name)
    raise SystemExit(f"not found: {name}")


def classify(name: str, shapes: str) -> str:
    for label, pat in CATS:
        if pat.search(name):
            if label == "conv3x3":
                # the weight shape is the second one in the key: [out, in, kh, kw]
                m = re.findall(r"\[([\d, ]+)\]", shapes or "")
                if m and len(m) >= 2:
                    dims = [int(x) for x in m[1].split(",") if x.strip().isdigit()]
                    if len(dims) >= 4:
                        return "conv3x3" if dims[2] == 3 and dims[3] == 3 else "conv1x1"
                return "conv3x3"
            return label
    return "other"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out-file", default="")
    args = ap.parse_args(argv)

    from spandrel import ModelLoader

    path = find(args.model)
    desc = ModelLoader().load_from_file(path)
    model = desc.model.eval().cuda().half()
    # read_shape is a Compact helper; SPAN's keys are not positional, so take the scale from the descriptor
    shape = {"scale": int(desc.scale), "arch": type(model).__name__}

    tu = TorchUtils(width=W, height=H, device_type="cuda", hdr_mode=True, gpu_id=0, high_bitdepth=True)
    st = torch.cuda.Stream()
    x = tu.yuv420p10_to_tensor(decode_yuv10(CALIB[0]), st, torch.device("cuda", 0), torch.float16)

    def forward():
        with torch.no_grad():
            model(x)

    for _ in range(3):
        forward()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(args.iters):
        forward()
    e1.record()
    torch.cuda.synchronize()
    total_ms = e0.elapsed_time(e1) / args.iters

    # THE PROFILER CANNOT BE USED HERE: this torch build reports `self_device_time_total = 0` even for a matmul
    # (measured: key_averages dev=0 with CUDA activity requested), i.e. CUPTI device timing is not collected. So
    # every Conv2d is bracketed with a CUDA event pair through pre/post hooks - sequential stream, so the gap
    # between the two events is that convolution's kernel - and the non-conv remainder is total minus convs.
    agg: dict[str, float] = {}
    counts: dict[str, int] = {}
    spans: dict[str, list] = {}

    def _cat(m) -> str:
        return "conv3x3" if tuple(m.kernel_size) == (3, 3) else "conv1x1"

    def _pre(m, _inp, cat=_cat):
        label = cat(m)
        counts[label] = counts.get(label, 0) + 1
        e = torch.cuda.Event(True)
        e.record()
        spans.setdefault(label, []).append([e, None])

    def _post(m, _inp, _out, cat=_cat):
        label = cat(m)
        e = torch.cuda.Event(True)
        e.record()
        spans[label][-1][1] = e

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.register_forward_pre_hook(_pre)
            m.register_forward_hook(_post)

    for _ in range(2):
        forward()
    torch.cuda.synchronize()
    spans.clear()
    counts.clear()
    for _ in range(args.iters):
        forward()
    torch.cuda.synchronize()
    for label, pairs in spans.items():
        agg[label] = sum(a.elapsed_time(b) for a, b in pairs if b is not None) / args.iters

    rows = sorted(agg.items(), key=lambda kv: -kv[1])
    conv3 = agg.get("conv3x3", 0.0)
    conv1 = agg.get("conv1x1", 0.0)
    rest = max(total_ms - conv3 - conv1, 0.0)
    rows.append(("non-conv (1x1-adjacent, attention, cat, shuffle)", rest))
    print(f"{args.model}: forward {total_ms:.2f} ms (events, no hooks), {shape}")
    print(f"{'category':46s} {'ms':>8s} {'share':>7s} {'count':>6s}")
    for label, ms in rows:
        print(f"{label:46s} {ms:8.2f} {100.0 * ms / max(total_ms, 1e-9):6.1f}% "
              f"{counts.get(label, ''):>6}")
    print(f"\nconvolutions are {100.0 * (conv3 + conv1) / max(total_ms, 1e-9):.1f}% of the frame, "
          f"of which 3x3 is {100.0 * conv3 / max(total_ms, 1e-9):.1f}% and 1x1 "
          f"{100.0 * conv1 / max(total_ms, 1e-9):.1f}%")
    print(f"ceiling for an int8 path over the 3x3 convs only: {100.0 * conv3 / max(total_ms, 1e-9):.1f}% of the "
          f"frame, and it needs a quantise/dequantise boundary per 3x3 group")

    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        json.dump({"model": args.model, "shape": shape, "total_ms": round(total_ms, 3),
                   "categories": {k: round(v, 3) for k, v in agg.items()},
                   "counts": counts,
                   "non_conv_ms": round(rest, 3),
                   "take": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                  open(args.out_file, "w"), indent=1)
        print(f"# -> {args.out_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
