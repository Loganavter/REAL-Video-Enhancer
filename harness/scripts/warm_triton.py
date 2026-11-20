"""Compile the int8 body's Triton kernels once, so the first render does not pay for it.

Measured 2026-11-20 (`int8-startup-split2.txt`): the body's first call costs **0.275 s** when Triton's cache is
warm and **2.531 s** when it is cold - 19 kernels compiled from scratch. The cache lives in `~/.triton` and
persists, so the price is paid once per machine, on the user's first render. Running this after installation moves
that price to the install, where nobody is waiting for a video.

Shapes: ONE is enough for the app's two real cases, and that is measured, not assumed. The kernels specialise on
the channel count, the tile height and the warp count - not on the frame geometry - so a cache warmed at 1920x1080
serves 1440x1080 for free:

  1920x1080, empty cache   2.58 s      (compiles)
  1440x1080, same cache    0.48 s      (reuses everything)
  2560x1440, same cache    1.84 s      (some new kernels)

So the default is the single shape the lane renders, and a larger one pays only the difference on its first use.
Warming "for the shape the user actually runs" cannot be done at install time - the installer does not know it -
and it would not help anyway: the compile happens on first use in that process, so moving it earlier only moves the
wait.

Usage (the app's own interpreter, once per machine):
  python3 harness/scripts/warm_triton.py --model <model>.pth --shapes 1920x1080,1440x1080
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "backend"))
import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)

from cut_channels_naive import load  # noqa: E402
from src.pytorch.Int8Body import Int8Body  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shapes", default="1920x1080")
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
