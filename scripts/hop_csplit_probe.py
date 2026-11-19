"""Split the OUTPUT channels across CTAs: halve the accumulator, double the warps, pay re-reads.

The measured bottleneck is occupancy, not traffic: 3.96 of 12 warps per scheduler, 56% of cycles with no eligible
warp, and the first tap costing 0.59 ms because there are no warps to hide its round trip
(docs/wsp/verified/int8-hop-occupancy.txt). The accumulator is BLOCK_M x 64 int32 - 32 KB per CTA at BLOCK_M 128 -
which is what caps the resident CTAs.

The BLOCK_M/warps sweep could not fix it: shrinking BLOCK_M keeps the 64-wide output and thins the MMA.
This does the other thing: keep BLOCK_M and split the 64 OUTPUT channels into two CTAs of 32 each, so
  - the accumulator per CTA halves (BLOCK_M x 32), letting ~twice as many CTAs stay resident,
  - the MMA is still 32 wide (a legal tensor-core tile),
  - and each CTA re-reads the same input, i.e. the input traffic doubles.

That trade is only sane because the kernel is NOT traffic-bound: L2 runs at 44-50% and DRAM at 54%
(int8-hop-ncu.txt, int8-hop-dram.txt), while the SM runs at 74% with no warps to issue. If occupancy is the
binding term, doubling the resident warps at double the L2 traffic should win; if it is not, this loses and the
staging argument for CUDA gets stronger.

Arms: the shipped 64-wide kernel, and the split into 2 x 32 and 4 x 16 output channels per CTA.

Usage: python scripts/hop_csplit_probe.py [--rounds 3] [--iters 20] [--out-file x.json]
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
from src.pytorch.Int8Kernels import quantise_nchw_to_nhwc_prelu  # noqa: E402
from src.pytorch.TorchUtils import TorchUtils  # noqa: E402
from tail_int8_emulation import CALIB, CLIP, H, W, decode_yuv10  # noqa: E402

APP = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2"
MODEL = f"{APP}/custom_models/2x-Futsuu-Anime(1).pth"
GFLOP = 152.9


@triton.jit
def hop_split(x_ptr, w_ptr, m_next_ptr, shift, w_scale_ptr, bias_ptr, slope_ptr, out_scale_ptr, y_ptr,
              H, W, n_tiles, BLOCK_M: tl.constexpr, C: tl.constexpr, COUT: tl.constexpr):
    """One CTA computes COUT of the C output channels for its row block; `n_tiles` blocks per channel group."""
    pid = tl.program_id(0)
    tile = pid % n_tiles
    grp = pid // n_tiles
    rows = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    h = rows // W
    w = rows % W
    li = tl.arange(0, C)
    lo = tl.arange(0, COUT)
    oc = grp * COUT + lo
    acc = tl.zeros((BLOCK_M, COUT), dtype=tl.int32)
    for tap in tl.range(0, 9, num_stages=1):
        dy = tap // 3 - 1
        dx = tap % 3 - 1
        yy = h + dy
        xx = w + dx
        inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        a = tl.load(x_ptr + ((yy * W + xx)[:, None] * C + li[None, :]), mask=inside[:, None], other=0)
        b = tl.load(w_ptr + tap * C * C + li[:, None] * C + oc[None, :])
        acc += tl.dot(a, b, out_dtype=tl.int32)
    v = acc.to(tl.float32) * tl.load(w_scale_ptr + oc)[None, :] + tl.load(bias_ptr + oc)[None, :]
    slope = tl.load(slope_ptr + oc)[None, :]
    v = tl.where(v >= 0, v, v * slope)
    osc = tl.load(out_scale_ptr + oc)[None, :]
    q = tl.floor(v / osc + 0.5)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int32)
    q = ((q * tl.load(m_next_ptr + oc)[None, :]) + (1 << (shift - 1))) >> shift
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(y_ptr + rows[:, None] * C + oc[None, :], q.to(tl.int8))


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

    def arm(cout: int, nw: int):
        n_tiles = (n_pix + bm - 1) // bm
        groups = 64 // cout

        def run():
            hop_split[(n_tiles * groups,)](src_buf, L["w"], L["mn"], SHIFT, L["vec"], L["bias"], L["slope"],
                                           L["os"], dst, H, W, n_tiles, BLOCK_M=bm, C=body.feat, COUT=cout,
                                           num_warps=nw)
            return dst
        return run

    arms = {"64 ch (shipped)": arm(64, 4), "2x32 ch, 4 warps": arm(32, 4),
            "4x16 ch, 4 warps": arm(16, 4), "2x32 ch, 2 warps": arm(32, 2)}

    with torch.no_grad():
        base = arms["64 ch (shipped)"]().clone()
        torch.cuda.synchronize()
        for k, fn in arms.items():
            dst.zero_()
            got = fn().clone()
            torch.cuda.synchronize()
            d = (base.to(torch.int16) - got.to(torch.int16)).abs()
            print(f"codes {k:>18} vs shipped: {int((d > 0).sum())} of {d.numel()} differ, max {int(d.max())}")

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
    print(f"\n{'round':>5} " + "  ".join(f"{k:>18}" for k in arms))
    for r in range(1, args.rounds + 1):
        cells = []
        for k, fn in arms.items():
            ms = timeit(fn)
            draws[k].append(ms)
            cells.append(f"{ms:18.3f}")
        print(f"{r:>5} " + "  ".join(cells))

    tail = slice(1, None) if args.rounds > 2 else slice(None)
    med = {k: statistics.median(v[tail]) for k, v in draws.items()}
    print(f"\n{'arm':>18} {'ms':>8} {'TOPS':>7} {'vs shipped':>12}")
    for k in arms:
        rel = 100.0 * (med["64 ch (shipped)"] - med[k]) / med["64 ch (shipped)"]
        print(f"{k:>18} {med[k]:8.3f} {GFLOP / med[k]:7.1f} {rel:+11.1f}%")
    best = min(med, key=med.get)
    print(f"\nbest: {best} ({100.0 * (med['64 ch (shipped)'] - med[best]) / med['64 ch (shipped)']:+.1f}%)")

    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        with open(args.out_file, "w") as fh:
            json.dump({"take": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "clip": os.path.basename(CLIP), "frame": args.frame,
                       "rounds": args.rounds, "iters": args.iters,
                       "draws": draws, "median": med, "best": best}, fh, indent=1)
        print(f"# -> {args.out_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
