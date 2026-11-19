"""Occupancy sweep for the hop: ncu says 3.96 of 12 warps per scheduler and 56% of cycles with no eligible warp.

The profiler's own reading (2026-11-19): SM Busy 52.7%, Issue Slots Busy 43.7%, "Tensor is the highest-utilized
pipeline (52.7%) but should not be a bottleneck", LSU highest by instruction count (73.9%), "frequent low-latency
instructions", and Scheduler Statistics: **Active Warps Per Scheduler 3.96 of 12, No Eligible 56.21%, Issued
Warp Per Scheduler 0.44**.

That last trio is the whole story: the kernel is starved for warps in flight, not bound by arithmetic or by
DRAM. The likely cause is the accumulator itself - BLOCK_M x 64 int32 is BLOCK_M * 64 * 4 bytes of registers per
CTA, i.e. 32 KB at BLOCK_M 128 - which caps how many CTAs the SM can keep resident. So sweep the two knobs that
decide it: BLOCK_M (less state per CTA) and warps per CTA (more threads to hide the same latency).

Every arm is the SHIPPED kernel with different launch parameters, so the codes must be identical.

Usage: python scripts/hop_occupancy_probe.py [--rounds 3] [--iters 20] [--out-file x.json]
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
from src.pytorch.Int8Body import Int8Body, QB, SHIFT  # noqa: E402
from src.pytorch.Int8Kernels import body_prologue_kernel, quantise_nchw_to_nhwc_prelu  # noqa: E402
from src.pytorch.TorchUtils import TorchUtils  # noqa: E402
from tail_int8_emulation import CALIB, CLIP, H, W, decode_yuv10  # noqa: E402

APP = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2"
MODEL = f"{APP}/custom_models/2x-Futsuu-Anime(1).pth"
GFLOP = 152.9


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=169)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out-file", default="")
    args = ap.parse_args(argv)

    dev = torch.device("cuda", 0)
    tu = TorchUtils(width=W, height=H, device_type="cuda", hdr_mode=True, gpu_id=0, high_bitdepth=True)
    stream = torch.cuda.Stream(device=dev)

    def frame(idx: int) -> torch.Tensor:
        return tu.yuv420p10_to_tensor(decode_yuv10(idx), stream, dev, torch.float16)

    model, _, _ = load(MODEL)
    model = model.eval().cuda().half()
    body = Int8Body(model)
    with torch.no_grad():
        body.calibrate([frame(i) for i in CALIB])
        body._build()
        x = frame(args.frame)
        head = body.head_conv(x.clamp(0.0, 1.0))

    n_pix = H * W
    qgrid = ((n_pix + QB - 1) // QB,)
    src_buf = torch.empty((1, H, W, body.feat), device=dev, dtype=torch.int8)
    quantise_nchw_to_nhwc_prelu[qgrid](head, body._s_t1, body._slope_half, src_buf, n_pix, BLOCK=QB,
                                       C=body.feat, num_warps=4)
    dst = torch.empty_like(src_buf)
    torch.cuda.synchronize()
    L = body._launches[0]

    def arm(bm: int, warps: int):
        grid = ((n_pix + bm - 1) // bm,)

        def run():
            body_prologue_kernel[grid](src_buf, L["w"], L["mn"], SHIFT, L["vec"], L["bias"], L["slope"],
                                       L["os"], dst, H, W, BLOCK_M=bm, C=body.feat, num_warps=warps,
                                       FOLD=0)
            return dst
        return run

    combos = [(128, 4), (128, 8), (64, 2), (64, 4), (64, 8), (32, 2), (32, 4), (256, 4), (256, 8)]
    arms = {f"BM{bm}/w{wr}": arm(bm, wr) for bm, wr in combos}

    # codes must be identical across launch parameters - only the schedule changes
    with torch.no_grad():
        base = arm(128, 4)()
        torch.cuda.synchronize()
        base = base.clone()
        for k, (bm, wr) in zip(arms, combos):
            got = arm(bm, wr)().clone()
            torch.cuda.synchronize()
            d = (base.to(torch.int16) - got.to(torch.int16)).abs()
            if int((d > 0).sum()):
                print(f"!! {k}: {int((d > 0).sum())} codes differ, max {int(d.max())}")

    def timeit(fn) -> float:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(args.iters):
            fn()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / args.iters

    draws = {k: [] for k in arms}
    print(f"{'round':>5} " + "  ".join(f"{k:>10}" for k in arms))
    for r in range(1, args.rounds + 1):
        cells = []
        for k, fn in arms.items():
            ms = timeit(fn)
            draws[k].append(ms)
            cells.append(f"{ms:10.3f}")
        print(f"{r:>5} " + "  ".join(cells))

    tail = slice(1, None) if args.rounds > 2 else slice(None)
    med = {k: statistics.median(v[tail]) for k, v in draws.items()}
    print(f"\n{'arm':>12} {'ms':>8} {'TOPS':>7} {'vs BM128/w4':>12}")
    for k in arms:
        rel = 100.0 * (med["BM128/w4"] - med[k]) / med["BM128/w4"]
        print(f"{k:>12} {med[k]:8.3f} {GFLOP / med[k]:7.1f} {rel:+11.1f}%")
    best = min(med, key=med.get)
    print(f"\nbest: {best} at {med[best]:.3f} ms "
          f"({100.0 * (med['BM128/w4'] - med[best]) / med['BM128/w4']:+.1f}% against the shipped BM128/w4)")

    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        with open(args.out_file, "w") as fh:
            json.dump({"take": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "clip": os.path.basename(CLIP), "frame": args.frame,
                       "rounds": args.rounds, "iters": args.iters, "combos": combos,
                       "draws": draws, "median": med, "best": best}, fh, indent=1)
        print(f"# -> {args.out_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
