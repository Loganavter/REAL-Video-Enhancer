"""Can a cheap, training-free temporal stabiliser make a boiling source reusable?

The gate refuses the denoised Pink Panther because its edges are re-rendered every frame: 3.8-4.2%
of pixels move more than 8 levels between consecutive source frames, with no global shift. A whole
frame has nothing stable to freeze, so the gate is right to refuse - and residual compute was
measured to be slower than a dense hop, so the only remaining lever is the material.

This asks whether the cheapest known stabiliser is enough: TAA-style history with a 3x3 AABB
clip against the current frame (Salvi 2016's neighbourhood clipping, the form the project's prior
art already names), blended with a weight a. Two numbers decide it:

* stability - the gate's own statistic on consecutive STABILISED frames. The gate's floor must
  fall under 1.5 and the deltas under the threshold (2-4 levels) for reuse to engage at all;
* fidelity - how far the stabilised frame drifts from the original. A prefilter that moves the
  picture by more than 8 levels is a different picture, not a fix.

Usage: python3 harness/scripts/temporal_stabilize_probe.py CLIP [--frames 16] [--start 0]
"""
from __future__ import annotations
import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _bos.path.join(_ROOT, "backend"), _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)

import argparse
import subprocess

import numpy as np

FFMPEG = "bin/ffmpeg"


def read_frames(path, width, height, count, start=0):
    cmd = [FFMPEG, "-v", "error", "-ss", str(start), "-i", path, "-frames:v", str(count),
           "-pix_fmt", "rgb48be", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    arr = np.frombuffer(raw, dtype=">u2")
    n = arr.size // (width * height * 3)
    a = arr[: n * width * height * 3].reshape(n, height, width, 3).astype(np.float32) / 65535.0
    return a[:count]


def block_means(lum, block=32):
    h, w = lum.shape
    gh, gw = -(-h // block), -(-w // block)
    pad = np.zeros((gh * block, gw * block), np.float32)
    pad[:h, :w] = lum
    return pad.reshape(gh, block, gw, block).mean(axis=(1, 3)) * 255.0


def gate_statistics(frames):
    """What the reuse gate would read: per-pair mean_abs, and its floor (median of the
    calmest third), exactly as `TemporalReuse._noise_floor` computes it."""
    deltas = []
    for i in range(len(frames) - 1):
        d = block_means(np.abs(frames[i + 1] - frames[i]).max(axis=2))
        deltas.append(float(d.max()))        # mean_abs over luma would be close; worst channel is stricter
    ordered = sorted(deltas)
    k = max(1, len(ordered) // 3)
    return deltas, ordered[k // 2]


def aabb_clip(history, current, radius=1):
    """Clip the history to the neighbourhood colour box of the current frame (3x3 AABB)."""
    h, w, c = current.shape
    lo = current.copy()
    hi = current.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = np.roll(np.roll(current, dy, axis=0), dx, axis=1)
            np.minimum(lo, shifted, out=lo)
            np.maximum(hi, shifted, out=hi)
    return np.clip(history, lo, hi)


def stabilise(frames, alpha):
    out = [frames[0]]
    for i in range(1, len(frames)):
        clipped = aabb_clip(out[-1], frames[i])
        out.append(alpha * frames[i] + (1.0 - alpha) * clipped)
    return np.array(out, dtype=np.float32)


def damage_vs(a, b):
    d = np.abs(a - b).max(axis=2) * 255.0
    return float((d > 8).mean()) * 100.0, float(d.max())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clip")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=1080)
    args = ap.parse_args()

    frames = read_frames(args.clip, args.width, args.height, args.frames, args.start)
    deltas, floor = gate_statistics(frames)
    print(f"source: {len(frames)} frames; per-pair mean_abs max {max(deltas):.2f}, "
          f"floor {floor:.2f} (gate limit 1.5, threshold clamp 2-4)")
    print(f"{'alpha':>6} {'floor':>6} {'worst pair':>11} {'worst drift vs source':>22} {'peak':>5}")
    for alpha in (0.0, 0.1, 0.2, 0.35, 0.5):
        st = stabilise(frames, alpha) if alpha > 0 else frames
        d, fl = gate_statistics(st)
        worst_drift, worst_peak = 0.0, 0.0
        for i in range(len(frames)):
            share, peak = damage_vs(st[i], frames[i])
            worst_drift, worst_peak = max(worst_drift, share), max(worst_peak, peak)
        print(f"{alpha:>6.2f} {fl:>6.2f} {max(d):>10.2f}  {worst_drift:>20.5f}% {worst_peak:>5.0f}")
    print("\nreuse needs: floor < 1.5 AND worst pair under the chosen threshold (2-4 levels);")
    print("fidelity bar: the stabilised frame must stay within 0.1% of pixels >8 levels of the source")


if __name__ == "__main__":
    _bsys.exit(main())
