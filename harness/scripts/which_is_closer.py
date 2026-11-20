"""Which render is closer to what the model actually computes?

The owner looked at frame 169 and said the fp16 render is worse than the int8 one. That is testable
without an opinion: run the model itself in fp16 (torch, the reference the whole lane uses) and measure
each render against it on the same frame. Two candidates, one ground truth:

  * `ref_fp16_y8.mkv`  - the app's fp16 TensorRT engine
  * `int8_pt.mkv`      - the app's int8 body (head/tail fp16, sixteen int8 hops)

If the int8 render is closer to the model than the fp16 one, then the engine - not the quantiser - is the
approximation in the shipped fp16 path, and every "int8 vs fp16" number in the lane is measuring the
difference between two approximations, not between the model and a quantised model.

Usage: python harness/scripts/which_is_closer.py --frame 169
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np
import torch

import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)
from cut_channels_naive import load  # noqa: E402

FF = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/bin/ffmpeg"
W, H = 3840, 2160
SIZE = W * H * 3


def render_frame(path: str, idx: int) -> np.ndarray:
    raw = subprocess.run([FF, "-loglevel", "error", "-i", path, "-vf", f"select=eq(n\\,{idx})",
                          "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    if len(raw) < SIZE:
        raise SystemExit(f"no frame {idx} in {path}")
    return np.frombuffer(raw[:SIZE], dtype=np.uint8).reshape(H, W, 3).astype(np.float32) / 255.0


def model_frame(model, src: str, idx: int) -> np.ndarray:
    raw = subprocess.run([FF, "-loglevel", "error", "-i", src, "-vf", f"select=eq(n\\,{idx})",
                          "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    x = np.frombuffer(raw[:1920 * 1080 * 3], dtype=np.uint8).reshape(1080, 1920, 3)
    t = torch.from_numpy(x.astype(np.float32) / 255.0).permute(2, 0, 1)[None].cuda().half()
    with torch.no_grad():
        out = model(t).float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return out


def compare(name: str, a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    d = np.abs(a - b).max(axis=2) * 255.0
    return float((d > 8).mean() * 100), float(d.max()), float(np.abs(a - b).mean() * 255.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=169)
    ap.add_argument("--source", default="/tmp/opencode/bd_violet20s.mkv")
    ap.add_argument("--model", default="/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/"
                                       "custom_models/2x-Futsuu-Anime(1).pth")
    ap.add_argument("--fp16", default="/var/tmp/rve-rem/ref_fp16_y8.mkv")
    ap.add_argument("--int8", dest="int8_path", default="/var/tmp/rve-rem/int8_pt.mkv")
    args = ap.parse_args(argv)

    model, _, _ = load(args.model)
    model = model.eval().cuda().half()
    truth = model_frame(model, args.source, args.frame)
    fp16 = render_frame(args.fp16, args.frame)
    int8 = render_frame(args.int8_path, args.frame)

    print(f"frame {args.frame}: share>8 / peak / mean|d| against the MODEL's own fp16 output")
    for name, img in (("TRT fp16 engine", fp16), ("int8 body", int8)):
        share, peak, mad = compare(name, img, truth)
        print(f"  {name:<18} {share:8.5f}%   peak {peak:5.1f}   mean |d| {mad:6.3f} levels")
    share, peak, mad = compare("fp16 vs int8", fp16, int8)
    print(f"  {'fp16 vs int8':<18} {share:8.5f}%   peak {peak:5.1f}   mean |d| {mad:6.3f} levels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
