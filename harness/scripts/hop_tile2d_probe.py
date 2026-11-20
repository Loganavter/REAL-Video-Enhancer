"""The tile SHAPE, not its size: 1-D flat runs read 3 rows for 128 pixels, a 2-D tile reads far less.

Measured the same day: the hop's L2 traffic is 1528 MB against 250.6 MB of DRAM - a 6.1x amplification - and the
nine gathers are the cause. The shipped CTA covers BLOCK_M = 128 CONSECUTIVE pixels, i.e. 128 pixels of ONE image
row (W = 1920), so its 3x3 window needs three rows of ~130 pixels to produce 128:

    amplification >= (128 + 2) * 3 / 128 = 3.05x     (one row per CTA)

Two rows per CTA does not help; a RECTANGULAR tile does:

    (Tw, Th) = (32, 4):  (32+2) * (4+2) / (32*4) =  1.59x
    (Tw, Th) = (16, 8):  (16+2) * (8+2) / (16*8) =  1.41x
    (Tw, Th) = (64, 2):  (64+2) * (2+2) / (64*2) =  2.06x

So the same 128 pixels per CTA, arranged as a rectangle, need 2-4x fewer input pixels fetched. That is the whole
Im2win argument (arXiv:2306.14316: "direct convolution loses to GEMM-based because of discontinuous memory
access") reduced to the part Triton can express: change which 128 pixels a CTA owns.

Arms: the shipped flat run (128x1) and rectangular tiles 32x4, 16x8, 64x2 - all BLOCK_M 128, all 4 warps, so the
MMA shape is unchanged and only the addressing moves. Codes must match exactly: the tiling changes no arithmetic.

Usage: python harness/scripts/hop_tile2d_probe.py [--rounds 3] [--iters 20] [--out-file x.json]
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

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "backend"))
import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)

from cut_channels_naive import load  # noqa: E402
from src.pytorch.Int8Body import Int8Body, QB, SHIFT  # noqa: E402
from src.pytorch.Int8Kernels import quantise_nchw_to_nhwc_prelu  # noqa: E402
from src.pytorch.TorchUtils import TorchUtils  # noqa: E402
from tail_int8_emulation import CALIB, CLIP, H, W, decode_yuv10  # noqa: E402

APP = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2"
MODEL = f"{APP}/custom_models/2x-Futsuu-Anime(1).pth"
GFLOP = 152.9


@triton.jit
def hop_tile(x_ptr, w_ptr, m_next_ptr, shift, w_scale_ptr, bias_ptr, slope_ptr, out_scale_ptr, y_ptr,
             H, W, BLOCK_M: tl.constexpr, C: tl.constexpr, TW: tl.constexpr, TH: tl.constexpr,
             TILES_X: tl.constexpr):
    """A CTA owns a TW x TH rectangle of pixels instead of a flat run of BLOCK_M along one row."""
    pid = tl.program_id(0)
    ty = pid // TILES_X
    tx = pid % TILES_X
    lane = tl.arange(0, BLOCK_M)
    i = lane // TW
    j = lane % TW
    h = ty * TH + i
    w = tx * TW + j
    lanes = tl.arange(0, C)
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
    osc = tl.load(out_scale_ptr + lanes)[None, :]
    q = tl.floor(v / osc + 0.5)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int32)
    q = ((q * tl.load(m_next_ptr + lanes)[None, :]) + (1 << (shift - 1))) >> shift
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(y_ptr + (h * W + w)[:, None] * C + lanes[None, :], q.to(tl.int8), mask=(h < H)[:, None])


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

    def tile_arm(tw: int, th: int):
        bm = tw * th
        tiles_x = (W + tw - 1) // tw
        tiles_y = (H + th - 1) // th

        def run():
            hop_tile[(tiles_x * tiles_y,)](src_buf, L["w"], L["mn"], SHIFT, L["vec"], L["bias"], L["slope"],
                                           L["os"], dst, H, W, BLOCK_M=bm, C=body.feat, TW=tw, TH=th,
                                           TILES_X=tiles_x, num_warps=4)
            return dst
        return run

    arms = {"128x1 (shipped)": tile_arm(128, 1), "32x4": tile_arm(32, 4),
            "16x8": tile_arm(16, 8), "64x2": tile_arm(64, 2), "128x2 (=256 rows/CTA)": None}
    arms.pop("128x2 (=256 rows/CTA)")
    arms["64x4"] = tile_arm(64, 4)

    # every arm must reproduce the shipped codes: a tile change moves no arithmetic
    with torch.no_grad():
        base = arms["128x1 (shipped)"]().clone()
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
    print(f"\n{'arm':>18} {'ms':>8} {'TOPS':>7} {'vs shipped':>12} {'window amp':>11}")
    for k, (bmt, _) in zip(arms, [(128, 1), (32, 4), (16, 8), (64, 2), (64, 4)]):
        tw, th = {"128x1 (shipped)": (128, 1), "32x4": (32, 4), "16x8": (16, 8),
                  "64x2": (64, 2), "64x4": (64, 4)}[k]
        amp = (tw + 2) * (th + 2) / (tw * th)
        rel = 100.0 * (med["128x1 (shipped)"] - med[k]) / med["128x1 (shipped)"]
        print(f"{k:>18} {med[k]:8.3f} {GFLOP / med[k]:7.1f} {rel:+11.1f}% {amp:10.2f}x")
    best = min(med, key=med.get)
    print(f"\nbest: {best} ({100.0 * (med['128x1 (shipped)'] - med[best]) / med['128x1 (shipped)']:+.1f}%)")

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
