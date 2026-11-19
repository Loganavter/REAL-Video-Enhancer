"""Compile the int8 body's Triton kernels once, so the first render does not pay for it.

Measured 2026-11-20 (`int8-startup-split2.txt`): the body's first call costs **0.275 s** when Triton's cache is
warm and **2.531 s** when it is cold - 19 kernels compiled from scratch. The cache lives in `~/.triton` and
persists, so the price is paid once per machine, on the user's first render. Running this after installation moves
that price to the install, where nobody is waiting for a video.

Shapes: the app's envelope allows up to 2.5 Mpx of LR input, and the two that matter are 1920x1080 and 1440x1080.
Each shape compiles its own kernels, so both are warmed.

Usage (the app's own interpreter, once per machine):
  python3 scripts/warm_triton.py --model <model>.pth --shapes 1920x1080,1440x1080
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "backend"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from cut_channels_naive import load  # noqa: E402
from src.pytorch.Int8Body import Int8Body  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shapes", default="1920x1080,1440x1080")
    ap.add_argument("--calib", default="")
    a = ap.parse_args(argv)

    model, scale, _ = load(a.model)
    model = model.eval().cuda().half()
    for spec in a.shapes.split(","):
        w, h = (int(v) for v in spec.lower().split("x"))
        x = torch.rand(1, 3, h, w, device="cuda", dtype=torch.float16)
        t0 = time.time()
        body = Int8Body(model)
        with torch.no_grad():
            # calibration on one random frame is enough for the COMPILER: the kernels' shapes depend on the
            # frame geometry, not on the grids' values, and this script's only product is the Triton cache.
            body.calibrate([x])
            body._build()
            body(x)
        torch.cuda.synchronize()
        print(f"{w}x{h}: compiled and ran in {time.time() - t0:.2f} s", flush=True)
        del body
        torch.cuda.empty_cache()
    print("Triton cache: " + os.environ.get("TRITON_CACHE_DIR", os.path.expanduser("~/.triton")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
