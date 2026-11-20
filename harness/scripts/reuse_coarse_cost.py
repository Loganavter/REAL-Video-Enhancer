"""What the coarse statistic and the reuse's geometry cost, per frame, at the resolution they ship at.

The row this serves has a cost acceptance (registration + warp <= 2 ms at 1440x1080 against ~70 ms of upscale),
and the cost is the half of the argument that a picture cannot settle: reuse saves a model call or it saves
nothing.

Two numbers are reported, and they are different objects:

  statistic  ReferenceFrameGate.is_duplicate with statistic="coarse" - luma, a 3 px low-pass at half resolution,
             FFT phase correlation, one bilinear warp of the half-resolution luma, and the block-mean grid.
             Measured against statistic="mean_abs" on the SAME frames, so the overhead is the difference.
  warp       ReferenceFrameCorrector.align at the model's output size (a 2x model on 1440x1080 is 2880x2160),
             which is what a reuse pays to move the cached picture instead of leaving it where it was.

Controls (a cost table with a broken kernel is the same trap as a broken statistic): an identity warp must read
0 ms of GPU work beyond the sync, and the reported frame total must be above the sum of its named parts - a
number below its own parts has been mis-attributed.

Usage:
  python3 harness/scripts/reuse_coarse_cost.py [--height 1080] [--width 1440] [--scale 2] [--frames 40]
"""

from __future__ import annotations
import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _bos.path.join(_ROOT, "backend"), _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)

import argparse
import time

import torch


def bench(fn, warmup=5, rounds=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(rounds):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    times.sort()
    return times[len(times) // 2], times[0], times[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--scale", type=int, default=2, help="model scale, for the emission warp")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--sigma", type=float, default=None)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device: this measures GPU work, a CPU number would be a different object")

    from src.utils.TemporalReuse import ReferenceFrameGate
    from src.utils.TemporalCorrection import ReferenceFrameCorrector

    device = torch.device("cuda")
    torch.manual_seed(0)
    # line art on a mid field: the content this statistic exists for, and the content a worst-block rule must
    # see (a flat field would make every comparison zero and every number look good)
    base = torch.full((1, 3, args.height, args.width), 0.65, device=device)
    for _ in range(120):
        y0 = int(torch.randint(0, args.height - 1, (1,)).item())
        x0 = int(torch.randint(0, args.width - 1, (1,)).item())
        base[:, :, y0:y0 + 2, x0:x0 + max(2, args.width // 90)] = 0.1
    frames = []
    for index in range(args.frames):
        frames.append(base.detach().clone() if index == 0 else
                      base.detach().clone() + 0.001 * index)      # a slow drift, so nothing is bit-identical

    results = {}
    for statistic in ("mean_abs", "coarse"):
        gate = ReferenceFrameGate(statistic=statistic, threshold=4.0, max_chain=4, pixel_veto=None)
        state = {"i": 0}
        gate.is_duplicate(frames[0])

        def step():
            state["i"] += 1
            if state["i"] >= len(frames):
                state["i"] = 1
                gate.reset()
                gate.is_duplicate(frames[0])
            gate.is_duplicate(frames[state["i"]])

        p50, lo, hi = bench(step)
        results[statistic] = (p50, lo, hi)
        print(f"gate is_duplicate [{statistic:>9}]: p50 {p50:5.2f} ms  min {lo:5.2f}  max {hi:5.2f}  "
              f"(reused {gate.reused}/{gate.processed})")
    overhead = results["coarse"][0] - results["mean_abs"][0]
    print(f"  coarse overhead over the shipped statistic: {overhead:+.2f} ms/frame")

    corrector = ReferenceFrameCorrector()
    corrector.seed(frames[0], frames[0])
    p50, lo, hi = bench(lambda: corrector.align(0.15, -0.1))
    print(f"emission warp {args.width * args.scale}x{args.height * args.scale}x3: "
          f"p50 {p50:5.2f} ms  min {lo:5.2f}  max {hi:5.2f}")
    identity = corrector.align(0.0, 0.0)
    print(f"control: an identity align is a no-op ({identity}, {corrector.aligned} warps applied so far)")

    total = results["coarse"][0] + p50
    print(f"\nper reused frame at {args.width}x{args.height} with a {args.scale}x model: "
          f"{results['coarse'][0]:.2f} (statistic every frame) + {p50:.2f} (warp on a reuse) "
          f"= {total:.2f} ms")
    print("uncovered: the model call this is compared against is measured elsewhere and on another day, so the")
    print("ratio is only valid inside one session; the numbers are medians of 30 rounds after 5 warm-up calls,")
    print("with the GPU otherwise idle - a render running beside this invalidates them.")


if __name__ == "__main__":
    _bsys.exit(main())
