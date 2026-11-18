"""Is the hop schedule-bound? Three arms, one process: today, vectorised loads, weights hoisted.

The hop runs 1.68 ms, which is 152.9 GFLOP / 1.68 ms = **91 TOPS** AND 265 MB / 1.68 ms = **158 GB/s** - about
52-55% of both of this card's roofs at once. A kernel at half of two roofs is bound by neither; it is bound by
its schedule. The two negatives already on record (unrolling all nine taps: 2.919 against 2.198 ms; Triton
pipelining at 2-3 stages: 2.497/2.635) are context, not results to repeat - the open part is exactly the load
path and the weight traffic.

Arms (row `rve-hop-load-schedule-2026-11-19`, researcher's C3):
  today     the shipped kernel, BLOCK_M 128, four warps
  hints     the same kernel with `max_contiguous`/`multiple_of` on the channel lanes, so the 64-byte row is
            allowed to become four 128-bit transactions instead of a masked gather
  hoist2    BLOCK_M 64 with TWO row-tiles per CTA: the weights are loaded once per tap and used by both dots,
            halving the weight traffic at the shipped register budget (one 128-wide accumulator's worth)

Reported: ms, achieved TOPS and GB/s per arm, against the roofs the lane uses (176 dense int8 TOPS,
288 GB/s).

Usage: python scripts/hop_schedule_probe.py [--rounds 3] [--iters 20] [--out-file x.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch
import triton
import triton.language as tl

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
TOPS, GBPS = 176.0, 288.0
GFLOP_PER_HOP = 152.9          # 64x64x9x2 x 2.0736 Mpx, the record's number
MB_PER_HOP = 265.0


@triton.jit
def hop_hints(x_ptr, w_ptr, m_next_ptr, shift, w_scale_ptr, bias_ptr, slope_ptr, out_scale_ptr,
              y_ptr, H, W, BLOCK_M: tl.constexpr, C: tl.constexpr):
    """The shipped kernel with the channel lanes marked contiguous - the only difference."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    h = rows // W
    w = rows % W
    lanes = tl.max_contiguous(tl.multiple_of(tl.arange(0, C), C), C)
    acc = tl.zeros((BLOCK_M, C), dtype=tl.int32)
    for tap in tl.range(0, 9, num_stages=1):
        dy = tap // 3 - 1
        dx = tap % 3 - 1
        yy = h + dy
        xx = w + dx
        inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        src = (yy * W + xx)[:, None] * C + lanes[None, :]
        a = tl.load(x_ptr + src, mask=inside[:, None], other=0)
        b = tl.load(w_ptr + tap * C * C + lanes[:, None] * C + lanes[None, :])
        acc += tl.dot(a, b, out_dtype=tl.int32)
    v = acc.to(tl.float32) * tl.load(w_scale_ptr + lanes)[None, :] + tl.load(bias_ptr + lanes)[None, :]
    slope = tl.load(slope_ptr + lanes)[None, :]
    v = tl.where(v >= 0, v, v * slope)
    out_scale = tl.load(out_scale_ptr + lanes)[None, :]
    q = tl.floor(v / out_scale + 0.5)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int32)
    q = ((q * tl.load(m_next_ptr + lanes)[None, :]) + (1 << (shift - 1))) >> shift
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(y_ptr + rows[:, None] * C + lanes[None, :], q.to(tl.int8))


@triton.jit
def hop_hoist2(x_ptr, w_ptr, m_next_ptr, shift, w_scale_ptr, bias_ptr, slope_ptr, out_scale_ptr,
               y_ptr, H, W, BLOCK_M: tl.constexpr, C: tl.constexpr):
    """Two row-tiles per CTA: each tap's weights are loaded ONCE and used by both dots."""
    pid = tl.program_id(0)
    base = pid * (2 * BLOCK_M)
    lanes = tl.arange(0, C)
    acc0 = tl.zeros((BLOCK_M, C), dtype=tl.int32)
    acc1 = tl.zeros((BLOCK_M, C), dtype=tl.int32)
    rows0 = base + tl.arange(0, BLOCK_M)
    rows1 = base + BLOCK_M + tl.arange(0, BLOCK_M)
    h0 = rows0 // W
    w0 = rows0 % W
    h1 = rows1 // W
    w1 = rows1 % W
    for tap in tl.range(0, 9, num_stages=1):
        dy = tap // 3 - 1
        dx = tap % 3 - 1
        b = tl.load(w_ptr + tap * C * C + lanes[:, None] * C + lanes[None, :])
        yy0 = h0 + dy
        xx0 = w0 + dx
        ok0 = (yy0 >= 0) & (yy0 < H) & (xx0 >= 0) & (xx0 < W)
        a0 = tl.load(x_ptr + ((yy0 * W + xx0)[:, None] * C + lanes[None, :]), mask=ok0[:, None], other=0)
        acc0 += tl.dot(a0, b, out_dtype=tl.int32)
        yy1 = h1 + dy
        xx1 = w1 + dx
        ok1 = (yy1 >= 0) & (yy1 < H) & (xx1 >= 0) & (xx1 < W)
        a1 = tl.load(x_ptr + ((yy1 * W + xx1)[:, None] * C + lanes[None, :]), mask=ok1[:, None], other=0)
        acc1 += tl.dot(a1, b, out_dtype=tl.int32)
    slope = tl.load(slope_ptr + lanes)[None, :]
    out_scale = tl.load(out_scale_ptr + lanes)[None, :]
    mn = tl.load(m_next_ptr + lanes)[None, :]
    ws = tl.load(w_scale_ptr + lanes)[None, :]
    bs = tl.load(bias_ptr + lanes)[None, :]
    v0 = acc0.to(tl.float32) * ws + bs
    v0 = tl.where(v0 >= 0, v0, v0 * slope)
    q0 = tl.floor(v0 / out_scale + 0.5)
    q0 = tl.minimum(tl.maximum(q0, -127.0), 127.0).to(tl.int32)
    q0 = ((q0 * mn) + (1 << (shift - 1))) >> shift
    q0 = tl.minimum(tl.maximum(q0, -127.0), 127.0)
    tl.store(y_ptr + rows0[:, None] * C + lanes[None, :], q0.to(tl.int8))
    v1 = acc1.to(tl.float32) * ws + bs
    v1 = tl.where(v1 >= 0, v1, v1 * slope)
    q1 = tl.floor(v1 / out_scale + 0.5)
    q1 = tl.minimum(tl.maximum(q1, -127.0), 127.0).to(tl.int32)
    q1 = ((q1 * mn) + (1 << (shift - 1))) >> shift
    q1 = tl.minimum(tl.maximum(q1, -127.0), 127.0)
    tl.store(y_ptr + rows1[:, None] * C + lanes[None, :], q1.to(tl.int8))


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
    bm, warps = body.block_m, body.warps

    def today():
        body_prologue_kernel[((n_pix + bm - 1) // bm,)](src_buf, L["w"], L["mn"], SHIFT, L["vec"],
                                                        L["bias"], L["slope"], L["os"], dst, H, W,
                                                        BLOCK_M=bm, C=body.feat, num_warps=warps,
                                                        FOLD=0)
        return dst

    def hints():
        hop_hints[((n_pix + bm - 1) // bm,)](src_buf, L["w"], L["mn"], SHIFT, L["vec"], L["bias"],
                                             L["slope"], L["os"], dst, H, W, BLOCK_M=bm,
                                             C=body.feat, num_warps=warps)
        return dst

    def hoist2():
        hop_hoist2[((n_pix + 2 * 64 - 1) // (2 * 64),)](src_buf, L["w"], L["mn"], SHIFT, L["vec"],
                                                        L["bias"], L["slope"], L["os"], dst, H, W,
                                                        BLOCK_M=64, C=body.feat, num_warps=warps)
        return dst

    # controls: yesterday's codes must come out the same on all three arms
    with torch.no_grad():
        base = today().clone()
        for name, fn in (("hints", hints), ("hoist2", hoist2)):
            got = fn().clone()
            torch.cuda.synchronize()
            d = (base.to(torch.int16) - got.to(torch.int16)).abs()
            print(f"codes {name} vs today: {int((d > 0).sum())} of {d.numel()} differ, "
                  f"max {int(d.max())}")

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

    arms = {"today": today, "hints": hints, "hoist2": hoist2}
    draws = {k: [] for k in arms}
    print(f"\n{'round':>5} " + "  ".join(f"{k:>9}" for k in arms) + "   (ms)")
    for r in range(1, args.rounds + 1):
        cells = []
        for k, fn in arms.items():
            ms = timeit(fn)
            draws[k].append(ms)
            cells.append(f"{ms:9.3f}")
        print(f"{r:>5} " + "  ".join(cells))

    tail = slice(1, None) if args.rounds > 2 else slice(None)
    med = {k: statistics.median(v[tail]) for k, v in draws.items()}
    print(f"\nmedians of rounds 2+ (roofs: {TOPS:.0f} TOPS, {GBPS:.0f} GB/s):")
    for k in arms:
        tops = GFLOP_PER_HOP / med[k]
        gbps = MB_PER_HOP / med[k]
        print(f"  {k:>7}: {med[k]:6.3f} ms   {tops:5.1f} TOPS ({100 * tops / TOPS:4.1f}% of roof)   "
              f"{gbps:6.1f} GB/s ({100 * gbps / GBPS:4.1f}%)")
    best = min(med, key=med.get)
    print(f"\nbest arm: {best} at {med[best]:.3f} ms "
          f"({100.0 * (med['today'] - med[best]) / med['today']:+.1f}% against today)")
    moves = med[best] < med["today"] * 0.95
    print(f"verdict: {'the schedule moves - ' + best + ' is worth a row' if moves else 'no arm moves the hop by 5% - it is at its floor on this shape'}")

    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        with open(args.out_file, "w") as fh:
            json.dump({"take": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "clip": os.path.basename(CLIP), "frame": args.frame,
                       "rounds": args.rounds, "iters": args.iters,
                       "roofs": {"tops": TOPS, "gbps": GBPS},
                       "gflop_per_hop": GFLOP_PER_HOP, "mb_per_hop": MB_PER_HOP,
                       "draws": draws, "median": med,
                       "tops": {k: GFLOP_PER_HOP / v for k, v in med.items()},
                       "gbps": {k: MB_PER_HOP / v for k, v in med.items()},
                       "moves": bool(moves)}, fh, indent=1)
        print(f"# -> {args.out_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
