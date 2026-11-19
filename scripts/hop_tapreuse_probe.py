"""Tap reuse in Triton: load the three-row band ONCE, form the nine taps by shifting inside registers.

The measurement that motivates it (int8-hop-dram.txt, int8-hop-ncu.txt): the hop reads every pixel NINE times -
9 x 132.71 MB = 1194 MB of logical reads per hop - and L2 sees 1528.7 MB, i.e. 128% of that, so coalescing and the
NHWC layout are fine. LSU is the highest-utilised pipe at 73.9% by instruction count. Six families of schedule
knobs were tried around those nine gathers and all lost; none of them touched the reads themselves.

The obstruction used to be that Triton cannot allocate shared memory explicitly, so `int8-hop-prefetch.txt`
closed the family after a CUDA microbenchmark of cooperative staging came out slower. Triton 3.5 has
`tl.gather`, which makes the horizontal shift expressible **inside a tile**: load three rows as (BLOCK_M+2, C)
tensors, then for each of the nine taps build the (BLOCK_M, C) operand with a gather at index arange+dx.

  today     9 masked/unmasked loads of (BLOCK_M, C)      -> 9 reads per pixel
  reuse     3 loads of (BLOCK_M+2, C) + 9 register gathers -> 3.05 reads per pixel

Codes must be identical: this changes where an operand comes from, not what is computed.

Usage: python scripts/hop_tapreuse_probe.py [--rounds 3] [--iters 20] [--out-file x.json]
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
HP, WP = H + 2, W + 2


@triton.jit
def hop_reuse(x_ptr, w_ptr, m_next_ptr, shift, w_scale_ptr, bias_ptr, slope_ptr, out_scale_ptr, y_ptr,
              H, W, BLOCK_M: tl.constexpr, C: tl.constexpr):
    """One output row per CTA; the three input rows are loaded once and shifted in registers."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    h = rows // W
    w = rows % W
    lanes = tl.arange(0, C)
    # tl.arange needs a power of two, so the band is 2*BLOCK_M wide and only BLOCK_M+2 lanes are used:
    # 3 rows x 2*BLOCK_M loads per BLOCK_M outputs = 6 reads per pixel against today's 9, i.e. 1.5x.
    # A hand-written CUDA kernel is not bound by this and would get the full 3.05x (and 8x with vertical reuse).
    band = tl.arange(0, 2 * BLOCK_M)
    h0 = pid * BLOCK_M // W                      # this tile's image row (BLOCK_M divides W in our grid)
    w0 = (pid * BLOCK_M) % W
    # three rows of the zero-haloed buffer, BLOCK_M+2 wide, unmasked: the halo is already zero
    r0 = tl.load(x_ptr + ((h0 + 0) * (W + 2) + (w0 + band))[:, None] * C + lanes[None, :])
    r1 = tl.load(x_ptr + ((h0 + 1) * (W + 2) + (w0 + band))[:, None] * C + lanes[None, :])
    r2 = tl.load(x_ptr + ((h0 + 2) * (W + 2) + (w0 + band))[:, None] * C + lanes[None, :])
    acc = tl.zeros((BLOCK_M, C), dtype=tl.int32)
    off = tl.arange(0, BLOCK_M)[:, None] + tl.zeros((BLOCK_M, C), dtype=tl.int32)
    for tap in tl.range(0, 9, num_stages=1):
        dy = tap // 3
        dx = tap % 3
        idx = off + dx
        if dy == 0:
            a = tl.gather(r0, idx, 0)
        elif dy == 1:
            a = tl.gather(r1, idx, 0)
        else:
            a = tl.gather(r2, idx, 0)
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
    tl.store(y_ptr + rows[:, None] * C + lanes[None, :], q.to(tl.int8))


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
    seed = torch.empty((1, H, W, body.feat), device=dev, dtype=torch.int8)
    quantise_nchw_to_nhwc_prelu[qgrid](head, body._s_t1, body._slope_half, seed, n_pix, BLOCK=QB,
                                       C=body.feat, num_warps=4)
    halo = torch.zeros((HP, WP, body.feat), device=dev, dtype=torch.int8)
    halo[1:H + 1, 1:W + 1, :] = seed.view(H, W, body.feat)
    dst = torch.empty_like(seed)
    torch.cuda.synchronize()

    L = body._launches[0]
    bm, warps = body.block_m, body.warps
    grid = ((n_pix + bm - 1) // bm,)
    big = halo.view(1, HP, WP, body.feat)

    def today():
        from src.pytorch.Int8Kernels import hop_padded_kernel
        hop_padded_kernel[grid](big, L["w"], L["mn"], SHIFT, L["vec"], L["bias"], L["slope"], L["os"], dst,
                                H, W, BLOCK_M=bm, C=body.feat, READ_PADDED=1, WRITE_PADDED=0,
                                num_warps=warps)
        return dst

    def reuse(bm_used: int):
        g = ((n_pix + bm_used - 1) // bm_used,)

        def run():
            hop_reuse[g](big, L["w"], L["mn"], SHIFT, L["vec"], L["bias"], L["slope"], L["os"], dst,
                         H, W, BLOCK_M=bm_used, C=body.feat, num_warps=warps)
            return dst
        return run

    arms = {"today (9 gathers)": today, "reuse BM128": reuse(128), "reuse BM64": reuse(64)}

    with torch.no_grad():
        base = today().clone()
        torch.cuda.synchronize()
        for k, fn in arms.items():
            if k.startswith("today"):
                continue
            dst.zero_()
            got = fn().clone()
            torch.cuda.synchronize()
            d = (base.to(torch.int16) - got.to(torch.int16)).abs()
            print(f"codes {k:>18} vs today: {int((d > 0).sum())} of {d.numel()} differ, max {int(d.max())}")

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
    print(f"\n{'arm':>18} {'ms':>8} {'TOPS':>7} {'vs today':>10}")
    for k in arms:
        print(f"{k:>18} {med[k]:8.3f} {GFLOP / med[k]:7.1f} "
              f"{100.0 * (med['today (9 gathers)'] - med[k]) / med['today (9 gathers)']:+9.1f}%")
    best = min(med, key=med.get)
    print(f"\nbest: {best} ({100.0 * (med['today (9 gathers)'] - med[best]) / med['today (9 gathers)']:+.1f}%)")

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
