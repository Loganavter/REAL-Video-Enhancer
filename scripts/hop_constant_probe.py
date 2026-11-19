"""Is the hop's 0.59 ms a per-LAUNCH constant or per-WORK ramp? Shrink the frame and watch.

Decides what the constant is, which decides who can remove it:
  per-launch   (grid setup, host dispatch, pipeline fill) -> CUDA graphs, or fewer launches (cascade)
  per-work     (it shrinks with the frame)                 -> it is arithmetic/latency that scales, and
                                                              only a different algorithm touches it

At 256x256 the hop has 0.0164x the pixels of 1080p, so a constant is unmistakable: 0.59 ms would remain exactly,
while any per-work term would fall to 1/61st.

Usage: python scripts/hop_constant_probe.py [--out-file x.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "backend"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from cut_channels_naive import load  # noqa: E402
from src.pytorch.Int8Body import SHIFT  # noqa: E402
from src.pytorch.Int8Kernels import body_prologue_kernel  # noqa: E402

APP = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2"
MODEL = f"{APP}/custom_models/2x-Futsuu-Anime(1).pth"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-file", default="")
    args = ap.parse_args(argv)

    dev = torch.device("cuda", 0)
    # the body's own launch parameters, recreated without calibrating a frame
    model, _, _ = load(MODEL)
    model = model.eval().half()
    from src.pytorch.Int8Body import Int8Body
    body = Int8Body(model)
    body._build=lambda: None
    # a synthetic launch set: the shapes are what matter, not the values
    C = 64
    taps = torch.zeros((9, C, C), device=dev, dtype=torch.int8)
    vec = torch.ones(C, device=dev, dtype=torch.float32)
    bias = torch.zeros(C, device=dev, dtype=torch.float32)
    slope = torch.ones(C, device=dev, dtype=torch.float32)
    osc = torch.ones(C, device=dev, dtype=torch.float32)
    mn = torch.full((C,), 1 << SHIFT, device=dev, dtype=torch.int32)

    sizes = [(256, 256), (512, 512), (1024, 1024), (1920, 1080), (3840, 2160)]
    rows = []
    print(f"{'frame':>12} {'px':>10} {'ms':>8} {'ms/Mpx':>9} {'TOPS':>7}")
    for H, W in sizes:
        n = H * W
        src = torch.zeros((1, H, W, C), device=dev, dtype=torch.int8)
        dst = torch.empty_like(src)
        bm = 128
        grid = ((n + bm - 1) // bm,)

        def run():
            body_prologue_kernel[grid](src, taps, mn, SHIFT, vec, bias, slope, osc, dst, H, W,
                                       BLOCK_M=bm, C=C, num_warps=4, FOLD=0)
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        draws = []
        for _ in range(3):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            for _ in range(20):
                run()
            e1.record()
            torch.cuda.synchronize()
            draws.append(e0.elapsed_time(e1) / 20)
        ms = statistics.median(draws)
        gflop = 2 * n * C * C * 9 / 1e9
        rows.append(dict(H=H, W=W, px=n, ms=round(ms, 4), ms_per_mpx=round(ms / (n / 1e6), 4),
                         tops=round(gflop / ms, 1)))
        print(f"{H}x{W:>5} {n/1e6:9.2f}M {ms:8.4f} {ms/(n/1e6):9.4f} {gflop/ms:7.1f}")

    a, b = rows[0], rows[-1]
    scale = b["px"] / a["px"]
    print(f"\n{scale:.0f}x more pixels: {a['ms']:.4f} -> {b['ms']:.4f} ms "
          f"(x{b['ms']/a['ms']:.1f} in time)")
    print(f"a per-launch constant would show time x1.0; pure per-work would show x{scale:.0f}")
    const = (b["ms"] - a["ms"] * scale) / (1 - scale)
    print(f"implied per-launch constant: {const:.3f} ms, per-work: {a['ms'] - const:.4f} ms at 256x256")

    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        json.dump({"take": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": rows,
                   "implied_constant_ms": round(const, 4)}, open(args.out_file, "w"), indent=1)
        print(f"# -> {args.out_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
