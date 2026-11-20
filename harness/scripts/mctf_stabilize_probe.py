"""Does motion-compensated temporal filtering make a boiling source reusable?

The training-free AABB stabiliser lowered the gate's floor but the render still missed the bar by
8x, because clipping a 3x3 neighbourhood does not know what moved. Both learned answers are out of
reach on this host (Lai et al. need torch 0.4 + compiled FlowNet2 layers; All-In-One-Deflicker
needs a conda env and trains an atlas per clip), so this tests the *idea* with the motion
estimator that is already installed - OpenCV's Farneback flow - and the classical MCTF: warp the
neighbours onto the current frame, then take their per-pixel median.

What must hold for the work to be worth a fast implementation:

* the gate's floor under 1.5 (it engages at all) and most pairs under the threshold;
* the gated render of the STABILISED clip within 0.1% of pixels >8 levels of the ungated render
  of the same clip - the statistic that refused the AABB stabiliser.

Farneback on the CPU is ~0.2-0.4 s/frame at 1440x1080: this is a viability test, not a prefilter.

Usage: python3 harness/scripts/mctf_stabilize_probe.py CLIP [--frames 12] [--radius 1]
"""
from __future__ import annotations
import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _bos.path.join(_ROOT, "backend"), _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)

import argparse

import cv2
import numpy as np

from temporal_stabilize_probe import block_means, gate_statistics, read_frames


def flow_to(prev_gray, cur_gray):
    return cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None, 0.5, 4, 21, 3, 7, 1.5, 0)


def warp(img, flow):
    h, w = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def mctf(frames, radius=1):
    """Warp the neighbours onto each frame and take the per-pixel median."""
    out = []
    grays = [cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) for f in frames]
    for t, cur in enumerate(frames):
        stack = [cur]
        for d in range(1, radius + 1):
            for nb in (t - d, t + d):
                if 0 <= nb < len(frames):
                    flow = flow_to(grays[nb], grays[t])        # nb -> t
                    stack.append(warp(frames[nb], flow))
        out.append(np.median(np.stack(stack, axis=0), axis=0).astype(np.float32))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clip")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=1080)
    args = ap.parse_args()

    frames = read_frames(args.clip, args.width, args.height, args.frames, args.start)
    deltas, floor = gate_statistics(frames)
    print(f"source : floor {floor:.2f}, worst pair {max(deltas):.2f}")
    st = mctf(frames, args.radius)
    deltas2, floor2 = gate_statistics(st)
    print(f"mctf r={args.radius}: floor {floor2:.2f}, worst pair {max(deltas2):.2f}, "
          f"pairs {[round(x,1) for x in deltas2]}")
    drift, peak = 0.0, 0.0
    for i in range(len(frames)):
        d = np.abs(st[i] - frames[i]).max(axis=2) * 255.0
        drift, peak = max(drift, float((d > 8).mean()) * 100.0), max(peak, float(d.max()))
    print(f"          drift vs source {drift:.3f}% of pixels, peak {peak:.0f} levels")
    print("\ngate needs the floor under 1.5; the bar for the drift is the owner's judgement, not a "
          "number - this probe only says what changes and by how much")


if __name__ == "__main__":
    _bsys.exit(main())
