"""How much reuse can this material give at all - asked before any mechanism is tried.

The 2026-11-20 reuse investigation spent a day on mechanisms (residual compute, AABB and MCTF
stabilisers, a learned blind-consistency model) before anyone measured the ceiling: the share of
consecutive FRAME PAIRS whose delta is under the gate's threshold. It is ~10% on that material,
so no mechanism could ever have produced the 2x everyone was looking for. It also produced three
measurement errors worth guarding against here:

1. **Sampling at 1 frame per second.** A cel held for 2-3 frames at 24 fps is invisible at 1 fps
   and the material looked 100% static; at native rate most pairs are 10-15 levels of motion.
   This instrument reads every frame, in contiguous chunks.
2. **A run's calibration sample judged against the whole run.** The gate certified 0.00% on 16
   calm frames and the render then measured 0.482% over 540. Printing the calibration-vs-clip gap
   next to the ceiling makes that failure visible before a render is spent.
3. **A model or a probe fed an unaligned pair.** The blind-consistency check first ran with the
   wrong "original" and read as "the model does nothing"; the true pair correlates 0.9997 and
   differs by 1.08 levels. `--pair A B` prints exactly that before any model is trusted.

Usage:
  python3 harness/scripts/reuse_ceiling.py --clip CLIP [--start S] [--duration S] [--step 1]
  python3 harness/scripts/reuse_ceiling.py --pair ORIGINAL PROCESSED [--start S] [--frames N]
  python3 harness/scripts/reuse_ceiling.py --wobble CLIP [--start S] [--frames N]
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
BLOCK = 32


def frames(path, width, height, count, start=0, step=1):
    vf = f"select='not(mod(n\\,{step}))',scale={width}:{height}"
    cmd = [FFMPEG, "-v", "error", "-ss", str(start), "-i", path, "-frames:v", str(count * step),
           "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.uint8)
    n = a.size // (width * height * 3)
    return a[: n * width * height * 3].reshape(n, height, width, 3).astype(np.float32) / 255.0


def blocks(x):
    h, w = x.shape
    gh, gw = -(-h // BLOCK), -(-w // BLOCK)
    pad = np.zeros((gh * BLOCK, gw * BLOCK), np.float32)
    pad[:h, :w] = x
    return pad.reshape(gh, BLOCK, gw, BLOCK).mean(axis=(1, 3)) * 255.0


def delta(a, b):
    """The gate's statistic: max over 32x32 blocks of the mean |difference|, worst channel."""
    return float(blocks(np.abs(b - a).max(axis=2)).max())


def controls():
    """A zero that must hold and a difference that must not: an instrument check, not a framing."""
    still = np.random.default_rng(0).random((270, 360, 3)).astype(np.float32)
    zero = delta(still, still)
    one = delta(still, np.clip(still + 8.0 / 255.0, 0, 1))
    ok = abs(zero) < 1e-9
    print(f"controls: identical frames read {zero:.5f} (must be 0) -> {'ok' if ok else 'BROKEN'}; "
          f"+8 levels read {one:.2f} (must be >= 8) -> {'ok' if one >= 8 else 'BROKEN'}")
    return ok and one >= 8


def ceiling(clip, start, duration, step, width, height, fps):
    count = int(duration * fps / step)
    fr = frames(clip, width, height, count, start, step)
    deltas = np.array([delta(fr[i], fr[i + 1]) for i in range(len(fr) - 1)])
    cuts = float(np.mean(deltas > 32)) * 100.0
    print(f"{len(deltas)} frame pairs from {start}s for {duration}s at 1/{step} of {fps:g} fps")
    print(f"  median {np.median(deltas):6.2f}   >32 levels (cut or big motion) {cuts:5.1f}%")
    for thr in (2.0, 4.0, 8.0):
        share = float(np.mean(deltas <= thr)) * 100.0
        print(f"  <= {thr:.0f} levels: {share:5.1f}% of pairs -> reuse ceiling at that threshold")
    print("  the ceiling is the answer to 'can this material be reused at all'; a mechanism that")
    print("  cannot exceed it is not worth building. Calibration on 16 frames does not change it.")


def wobble(clip, start, count, width, height):
    """Is the pair-delta motion, or a sub-pixel drift of a held frame?

    A scan's gate weave moves every edge by a fraction of a pixel, and a block-mean statistic
    reads that as motion even when nothing in the picture moved. This reports the estimated
    shift per pair and the delta left after it is compensated: what a warp-compensated reuse
    could take, and what it could not.
    """
    import cv2
    fr = frames(clip, width, height, count, start)
    before, after, shifts = [], [], []
    for i in range(len(fr) - 1):
        a = cv2.cvtColor((fr[i] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        b = cv2.cvtColor((fr[i + 1] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        (dx, dy), _resp = cv2.phaseCorrelate(a, b)
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        aligned = cv2.warpAffine(fr[i + 1], M, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        before.append(delta(fr[i], fr[i + 1]))
        after.append(delta(fr[i], aligned))
        shifts.append((dx, dy))
    shifts = np.array(shifts)
    before, after = np.array(before), np.array(after)
    print(f"wobble over {len(before)} pairs from {start}s:")
    print(f"  shift per pair: median ({np.median(np.abs(shifts[:,0])):.2f}, "
          f"{np.median(np.abs(shifts[:,1])):.2f}) px, max {np.abs(shifts).max():.2f} px")
    print(f"  delta median {np.median(before):.2f} -> {np.median(after):.2f} after shift compensation")
    for thr in (2.0, 4.0):
        print(f"  <= {thr:.0f} levels: {100*np.mean(before<=thr):5.1f}% before, "
              f"{100*np.mean(after<=thr):5.1f}% after - the warp-compensated ceiling")


def pair(original, processed, start, count, width, height):
    a = frames(original, width, height, count, start)
    b = frames(processed, width, height, count, start)
    n = min(len(a), len(b))
    corr, mean_abs, over8 = [], [], []
    for i in range(n):
        ga, gb = a[i].mean(2), b[i].mean(2)
        corr.append(float(np.corrcoef(ga.ravel(), gb.ravel())[0, 1]) if ga.std() and gb.std() else 1.0)
        d = np.abs(a[i] - b[i]).max(axis=2) * 255.0
        mean_abs.append(float(d.mean()))
        over8.append(float((d > 8).mean()) * 100.0)
    print(f"pair over {n} frames from {start}s:")
    print(f"  correlation {np.median(corr):.5f}   mean |difference| {np.median(mean_abs):.2f} levels   "
          f">8 levels on {np.median(over8):.2f}% of pixels")
    if np.median(corr) < 0.95:
        print("  WARNING: these do not look like the same frames - a model fed this pair reads as "
              "'does nothing', which is a measurement error, not a result")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", help="single clip: report the reuse ceiling")
    ap.add_argument("--wobble", help="single clip: is the delta motion, or a sub-pixel drift?")
    ap.add_argument("--pair", nargs=2, metavar=("ORIGINAL", "PROCESSED"),
                    help="two versions of the same frames: is the pair aligned, and by how much?")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--step", type=int, default=1, help="1 = every frame (the default, on purpose)")
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=float, default=24.0)
    args = ap.parse_args()
    if not controls():
        print("controls failed: fix the instrument before reading any number below")
        return 2
    if args.pair:
        pair(args.pair[0], args.pair[1], args.start, args.frames, args.width, args.height)
    if args.clip:
        ceiling(args.clip, args.start, args.duration, args.step, args.width, args.height, args.fps)
    if args.wobble:
        wobble(args.wobble, args.start, min(args.frames, 16), args.width, args.height)
    if not args.pair and not args.clip and not args.wobble:
        ap.error("give --clip or --pair")
    return 0


if __name__ == "__main__":
    _bsys.exit(main())
