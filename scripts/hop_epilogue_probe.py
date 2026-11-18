"""Is the hop's epilogue ALU-limited? One hop and the full sixteen-hop chain, two epilogues.

The shipped epilogue computes `v = acc*w_scale + bias`, applies PReLU, then `q = floor(v/out_scale + 0.5)`
- a per-element fp32 DIVIDE over 132 M elements, in the one kernel all sixteen hops run. PReLU is
homogeneous of degree one, so a positive rescale commutes with it: with `k = w_scale/out_scale` and
`c = bias/out_scale` the epilogue becomes one FMA, the PReLU, one add and a floor.

Arms (row `rve-epilogue-divide-2026-11-19`, researcher's C5):
  fold=0   the shipped epilogue, divide per element
  fold=1   the folded epilogue, one divide per channel per CTA

Measured on real codes: the frame is decoded from the clip through the app's own yuv420p10_to_tensor, the
head runs, and the quantise pass produces the hop-1 input exactly as the app does. Weights, grids, scales
and the launch parameters are the body's own (`Int8Body._build`).

Also reported: how many int8 codes the two epilogues disagree on, and the maximum difference - the rewrite
is exact in real arithmetic but not bit-identical in fp32, so this is a different-arithmetic change and
the code diff is the first half of its picture evidence (the second half is a render pair).

Usage: python scripts/hop_epilogue_probe.py [--clip /tmp/opencode/bd_violet20s.mkv] [--rounds 3]
                                            [--iters 20] [--out-file x.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "backend"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from cut_channels_naive import load  # noqa: E402
from src.pytorch.Int8Body import Int8Body, SHIFT, QB  # noqa: E402
from src.pytorch.Int8Kernels import (body_prologue_kernel, quantise_nchw_to_nhwc_prelu,  # noqa: E402
                                     quantise_nhwc_prelu)
from src.pytorch.TorchUtils import TorchUtils  # noqa: E402

APP = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2"
FFMPEG = f"{APP}/bin/ffmpeg"
MODEL = f"{APP}/custom_models/2x-Futsuu-Anime(1).pth"
CALIB = (20, 60, 100, 140, 180, 220, 260, 300)


def decode_yuv10(clip: str, frame: int, width: int, height: int) -> bytes:
    """One frame of the clip as the app's input pipe carries it: packed yuv420p10le."""
    out = subprocess.run([FFMPEG, "-loglevel", "error", "-i", clip, "-vf", f"select=eq(n\\,{frame})",
                          "-frames:v", "1", "-pix_fmt", "yuv420p10le", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    want = width * height * 3
    if len(out) < want:
        raise SystemExit(f"frame {frame}: {len(out)} bytes, wanted {want}")
    return out[:want]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="/tmp/opencode/bd_violet20s.mkv")
    ap.add_argument("--frame", type=int, default=169)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out-file", default="")
    args = ap.parse_args(argv)

    dev = torch.device("cuda", 0)
    tu = TorchUtils(width=args.width, height=args.height, device_type="cuda", hdr_mode=True, gpu_id=0,
                    high_bitdepth=True)
    stream = torch.cuda.Stream(device=dev)

    def frame_tensor(idx: int) -> torch.Tensor:
        return tu.yuv420p10_to_tensor(decode_yuv10(args.clip, idx, args.width, args.height), stream,
                                      dev, torch.float16)

    model, scale, _ = load(MODEL)
    model = model.eval().cuda().half()
    body = Int8Body(model)
    with torch.no_grad():
        body.calibrate([frame_tensor(i) for i in CALIB])
        body._build()
    print(f"body: {len(body.inner)} hops, feat {body.feat}, block_m {body.block_m}, warps {body.warps}")

    h, w = args.height, args.width
    n_pix = h * w
    x = frame_tensor(args.frame)
    with torch.no_grad():
        head = body.head_conv(x.clamp(0.0, 1.0))
    src = torch.empty((1, h, w, body.feat), device=dev, dtype=torch.int8)
    qgrid = ((n_pix + QB - 1) // QB,)
    if head.is_contiguous(memory_format=torch.channels_last):
        quantise_nhwc_prelu[qgrid](head, body._s_t1, body._slope_half, src, n_pix, BLOCK=QB,
                                   C=body.feat, num_warps=4)
    else:
        quantise_nchw_to_nhwc_prelu[qgrid](head, body._s_t1, body._slope_half, src, n_pix, BLOCK=QB,
                                          C=body.feat, num_warps=4)
    torch.cuda.synchronize()

    grid = ((n_pix + body.block_m - 1) // body.block_m,)
    scratch = torch.empty_like(src)
    launch = body._launches[0]

    def one_hop(fold: int, dst):
        body_prologue_kernel[grid](src, launch["w"], launch["mn"], SHIFT, launch["vec"],
                                   launch["bias"], launch["slope"], launch["os"], dst, h, w,
                                   BLOCK_M=body.block_m, C=body.feat, num_warps=body.warps, FOLD=fold)

    def chain(fold: int):
        """The sixteen hops as the body runs them, never writing into the input buffer.

        Two scratch buffers and no copy back into `src`: an even hop count would otherwise land the
        result in `src` and the next arm would read its own output as input (that mistake made the first
        run of this probe report 41.8% differing codes and max 254).
        """
        bufs = (torch.empty_like(src), torch.empty_like(src))
        cur = src
        for i, L in enumerate(body._launches):
            dst = bufs[i % 2]
            body_prologue_kernel[grid](cur, L["w"], L["mn"], SHIFT, L["vec"], L["bias"], L["slope"],
                                       L["os"], dst, h, w, BLOCK_M=body.block_m, C=body.feat,
                                       num_warps=body.warps, FOLD=fold)
            cur = dst
        return cur

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

    arms = {
        "hop": {"0": lambda: one_hop(0, scratch), "1": lambda: one_hop(1, scratch)},
        "chain": {"0": lambda: chain(0), "1": lambda: chain(1)},
    }
    names = arms
    draws = {k: {"0": [], "1": []} for k in names}
    print(f"{'round':>5} {'hop fold0':>11} {'hop fold1':>11} {'chain fold0':>12} {'chain fold1':>12}"
          "   (ms)")
    for r in range(1, args.rounds + 1):
        row = []
        for kind, by_fold in arms.items():
            for fold in ("0", "1"):
                ms = timeit(by_fold[fold])
                draws[kind][fold].append(ms)
                row.append(ms)
        print(f"{r:>5} {row[0]:11.3f} {row[1]:11.3f} {row[2]:12.3f} {row[3]:12.3f}")

    tail = slice(1, None) if args.rounds > 2 else slice(None)
    med = {k: {f: statistics.median(draws[k][f][tail]) for f in ("0", "1")} for k in names}
    print("\nmedians of rounds 2+:")
    for kind in names:
        d = med[kind]["1"] - med[kind]["0"]
        rel = 100.0 * d / med[kind]["0"]
        print(f"  {kind:>5}: fold0 {med[kind]['0']:7.3f} ms   fold1 {med[kind]['1']:7.3f} ms   "
              f"delta {d:+7.3f} ({rel:+.1f}%)")
    print(f"  sixteen hops are what the frame pays: {med['chain']['1'] - med['chain']['0']:+.3f} ms "
          f"per frame")

    # ---- the codes, on the same input
    out0 = torch.empty_like(src)
    out1 = torch.empty_like(src)
    one_hop(0, out0)
    one_hop(1, out1)
    torch.cuda.synchronize()
    diff = (out0.to(torch.int16) - out1.to(torch.int16)).abs()
    n_diff = int((diff > 0).sum())
    print(f"\ncodes: {n_diff} of {diff.numel()} differ ({100.0 * n_diff / diff.numel():.5f}%), "
          f"max |diff| {int(diff.max())}")
    chain0, chain1 = chain(0).clone(), chain(1).clone()
    cdiff = (chain0.to(torch.int16) - chain1.to(torch.int16)).abs()
    print(f"after sixteen hops: {int((cdiff > 0).sum())} of {cdiff.numel()} codes differ "
          f"({100.0 * int((cdiff > 0).sum()) / cdiff.numel():.5f}%), max |diff| {int(cdiff.max())}")

    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        with open(args.out_file, "w") as fh:
            json.dump({"take": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "clip": os.path.basename(args.clip), "frame": args.frame,
                       "width": args.width, "height": args.height, "rounds": args.rounds,
                       "iters": args.iters, "draws": draws, "median": med,
                       "codes": {"hop_diff": n_diff, "hop_total": diff.numel(),
                                 "hop_max": int(diff.max()),
                                 "chain_diff": int((cdiff > 0).sum()),
                                 "chain_total": cdiff.numel(), "chain_max": int(cdiff.max())}},
                      fh, indent=1)
        print(f"# -> {args.out_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
