"""Is the first tap's round trip hideable at all? Issue the nine loads together and only then the nine dots.

The theory the CUDA/smem plan rests on (int8-hop-occupancy.txt): one tap costs 0.59 ms, three cost the same,
nine cost 1.55 - so the FIRST gather's round trip is not hidden, because there are only 3.96 of 12 warps per
scheduler to hide it behind. Shared-memory staging is the classic fix: fetch the three-row window once, form the
nine taps from it, and the round trips stop being dependent.

Before writing that in CUDA, the same question can be asked inside Triton, because the *shape* of the fix is
"issue the loads early, consume them late":

  rolled     today: `tl.range(0, 9, num_stages=1)` - load tap t, dot tap t, load tap t+1 ...
  unrolled   `tl.static_range(0, 9)` - the loop is unrolled, so the compiler is free to hoist loads
  prefetch   nine explicit loads into nine variables, THEN nine dots (the smem-staging order, without smem)

If `prefetch` approaches the nine-taps-for-the-price-of-three shape (i.e. ~1.07 ms instead of 1.55), the round
trip IS hideable and the fix does not need CUDA. If it is worse, register pressure is what stops it - and that is
exactly the argument for smem, which stores those tiles off-register.

Codes must be identical: none of this changes arithmetic - only the order of the same nine loads and dots - so any
difference is a bug.

Usage: python harness/scripts/hop_prefetch_probe.py [--rounds 3] [--iters 20] [--out-file x.json]
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
def _epi(acc, w_scale_ptr, bias_ptr, slope_ptr, out_scale_ptr, m_next_ptr, shift, y_ptr, rows, lanes,
         C: tl.constexpr):
    v = acc.to(tl.float32) * tl.load(w_scale_ptr + lanes)[None, :] + tl.load(bias_ptr + lanes)[None, :]
    slope = tl.load(slope_ptr + lanes)[None, :]
    v = tl.where(v >= 0, v, v * slope)
    osc = tl.load(out_scale_ptr + lanes)[None, :]
    q = tl.floor(v / osc + 0.5)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int32)
    q = ((q * tl.load(m_next_ptr + lanes)[None, :]) + (1 << (shift - 1))) >> shift
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(y_ptr + rows[:, None] * C + lanes[None, :], q.to(tl.int8))


@triton.jit
def hop_rolled(x_ptr, w_ptr, m_next_ptr, shift, w_scale_ptr, bias_ptr, slope_ptr, out_scale_ptr, y_ptr,
               H, W, BLOCK_M: tl.constexpr, C: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    h = rows // W
    w = rows % W
    lanes = tl.arange(0, C)
    acc = tl.zeros((BLOCK_M, C), dtype=tl.int32)
    for tap in tl.range(0, 9, num_stages=1):
        dy = tap // 3 - 1
        dx = tap % 3 - 1
        yy = h + dy
        xx = w + dx
        inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        a = tl.load(x_ptr + ((yy * W + xx)[:, None] * C + lanes[None, :]), mask=inside[:, None], other=0)
        b = tl.load(w_ptr + tap * C * C + lanes[:, None] * C + lanes[None, :])
        acc += tl.dot(a, b, out_dtype=tl.int32)
    _epi(acc, w_scale_ptr, bias_ptr, slope_ptr, out_scale_ptr, m_next_ptr, shift, y_ptr, rows, lanes, C)


@triton.jit
def hop_unrolled(x_ptr, w_ptr, m_next_ptr, shift, w_scale_ptr, bias_ptr, slope_ptr, out_scale_ptr, y_ptr,
                 H, W, BLOCK_M: tl.constexpr, C: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    h = rows // W
    w = rows % W
    lanes = tl.arange(0, C)
    acc = tl.zeros((BLOCK_M, C), dtype=tl.int32)
    for tap in tl.static_range(0, 9):
        dy = tap // 3 - 1
        dx = tap % 3 - 1
        yy = h + dy
        xx = w + dx
        inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        a = tl.load(x_ptr + ((yy * W + xx)[:, None] * C + lanes[None, :]), mask=inside[:, None], other=0)
        b = tl.load(w_ptr + tap * C * C + lanes[:, None] * C + lanes[None, :])
        acc += tl.dot(a, b, out_dtype=tl.int32)
    _epi(acc, w_scale_ptr, bias_ptr, slope_ptr, out_scale_ptr, m_next_ptr, shift, y_ptr, rows, lanes, C)


@triton.jit
def hop_prefetch(x_ptr, w_ptr, m_next_ptr, shift, w_scale_ptr, bias_ptr, slope_ptr, out_scale_ptr, y_ptr,
                 H, W, BLOCK_M: tl.constexpr, C: tl.constexpr):
    """Nine loads FIRST, nine dots after - the smem-staging ORDER, without smem."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    h = rows // W
    w = rows % W
    lanes = tl.arange(0, C)
    # dy = -1 row, dy = 0, dy = +1: 3x3 taps issued as three groups of three loads
    a0 = tl.load(x_ptr + (((h - 1) * W + (w - 1))[:, None] * C + lanes[None, :]),
                 mask=(((h - 1) >= 0) & ((w - 1) >= 0))[:, None], other=0)
    a1 = tl.load(x_ptr + (((h - 1) * W + w)[:, None] * C + lanes[None, :]), mask=((h - 1) >= 0)[:, None], other=0)
    a2 = tl.load(x_ptr + (((h - 1) * W + (w + 1))[:, None] * C + lanes[None, :]),
                 mask=(((h - 1) >= 0) & ((w + 1) < W))[:, None], other=0)
    a3 = tl.load(x_ptr + ((h * W + (w - 1))[:, None] * C + lanes[None, :]), mask=((w - 1) >= 0)[:, None], other=0)
    a4 = tl.load(x_ptr + ((h * W + w)[:, None] * C + lanes[None, :]))
    a5 = tl.load(x_ptr + ((h * W + (w + 1))[:, None] * C + lanes[None, :]), mask=((w + 1) < W)[:, None], other=0)
    a6 = tl.load(x_ptr + (((h + 1) * W + (w - 1))[:, None] * C + lanes[None, :]),
                 mask=(((h + 1) < H) & ((w - 1) >= 0))[:, None], other=0)
    a7 = tl.load(x_ptr + (((h + 1) * W + w)[:, None] * C + lanes[None, :]), mask=((h + 1) < H)[:, None], other=0)
    a8 = tl.load(x_ptr + (((h + 1) * W + (w + 1))[:, None] * C + lanes[None, :]),
                 mask=(((h + 1) < H) & ((w + 1) < W))[:, None], other=0)
    acc = tl.zeros((BLOCK_M, C), dtype=tl.int32)
    for tap in tl.static_range(0, 9):
        b = tl.load(w_ptr + tap * C * C + lanes[:, None] * C + lanes[None, :])
        if tap == 0:
            acc += tl.dot(a0, b, out_dtype=tl.int32)
        elif tap == 1:
            acc += tl.dot(a1, b, out_dtype=tl.int32)
        elif tap == 2:
            acc += tl.dot(a2, b, out_dtype=tl.int32)
        elif tap == 3:
            acc += tl.dot(a3, b, out_dtype=tl.int32)
        elif tap == 4:
            acc += tl.dot(a4, b, out_dtype=tl.int32)
        elif tap == 5:
            acc += tl.dot(a5, b, out_dtype=tl.int32)
        elif tap == 6:
            acc += tl.dot(a6, b, out_dtype=tl.int32)
        elif tap == 7:
            acc += tl.dot(a7, b, out_dtype=tl.int32)
        else:
            acc += tl.dot(a8, b, out_dtype=tl.int32)
    _epi(acc, w_scale_ptr, bias_ptr, slope_ptr, out_scale_ptr, m_next_ptr, shift, y_ptr, rows, lanes, C)


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
    grid = ((n_pix + bm - 1) // bm,)

    def make(kernel):
        def run():
            kernel[grid](src_buf, L["w"], L["mn"], SHIFT, L["vec"], L["bias"], L["slope"], L["os"], dst,
                         H, W, BLOCK_M=bm, C=body.feat, num_warps=warps)
            return dst
        return run

    arms = {"rolled (today)": make(hop_rolled), "unrolled": make(hop_unrolled),
            "prefetch (loads first)": make(hop_prefetch)}

    with torch.no_grad():
        base = arms["rolled (today)"]().clone()
        torch.cuda.synchronize()
        for k, fn in arms.items():
            dst.zero_()
            got = fn().clone()
            torch.cuda.synchronize()
            d = (base.to(torch.int16) - got.to(torch.int16)).abs()
            print(f"codes {k:>22} vs rolled: {int((d > 0).sum())} of {d.numel()} differ, max {int(d.max())}")

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
    print(f"\n{'round':>5} " + "  ".join(f"{k:>22}" for k in arms))
    for r in range(1, args.rounds + 1):
        cells = []
        for k, fn in arms.items():
            ms = timeit(fn)
            draws[k].append(ms)
            cells.append(f"{ms:22.3f}")
        print(f"{r:>5} " + "  ".join(cells))

    tail = slice(1, None) if args.rounds > 2 else slice(None)
    med = {k: statistics.median(v[tail]) for k, v in draws.items()}
    print(f"\n{'arm':>22} {'ms':>8} {'TOPS':>7} {'vs rolled':>11}")
    for k in arms:
        rel = 100.0 * (med["rolled (today)"] - med[k]) / med["rolled (today)"]
        print(f"{k:>22} {med[k]:8.3f} {GFLOP / med[k]:7.1f} {rel:+10.1f}%")
    best = min(med, key=med.get)
    gain = 100.0 * (med["rolled (today)"] - med[best]) / med["rolled (today)"]
    print(f"\nbest: {best} ({gain:+.1f}%). The theory says the nine-taps-for-three price would be ~1.07 ms "
          f"(-31%); anything near that means the round trip IS hideable without CUDA.")

    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        with open(args.out_file, "w") as fh:
            json.dump({"take": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "clip": os.path.basename(CLIP), "frame": args.frame,
                       "rounds": args.rounds, "iters": args.iters,
                       "draws": draws, "median": med, "best": best, "gain_pct": gain}, fh, indent=1)
        print(f"# -> {args.out_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
