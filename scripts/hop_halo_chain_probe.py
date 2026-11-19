"""The inline halo, end to end: 16 hops through padded buffers, no build pass anywhere.

The single-hop result (int8-hop-tiles.txt): a zero-halo buffer with an unmasked kernel is 2.3% faster
(1.590 -> 1.552 ms, codes identical), but building the halo with a strided copy costs 1.591 ms - 25x the gain.
The repair that was written down and not measured: each hop WRITES into the next hop's padded buffer at a
(+1, +1) offset, so the halo costs nothing per hop, and only the border needs zeroing once per frame (~0.5 MB).

This measures exactly that, as a chain:

  today    three (H, W, C) buffers, masked kernel, 16 launches
  padded   three (H+2, W+2, C) buffers, unmasked kernel, the store offset into the interior, border zeroed
           once per frame (timed separately, since it happens once against sixteen launches)

Codes must match exactly in the interior: a halo changes no arithmetic.

Usage: python scripts/hop_halo_chain_probe.py [--rounds 3] [--iters 15] [--out-file x.json]
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
HP, WP = H + 2, W + 2


@triton.jit
def hop_padded(x_ptr, w_ptr, m_next_ptr, shift, w_scale_ptr, bias_ptr, slope_ptr, out_scale_ptr, y_ptr,
               H, W, BLOCK_M: tl.constexpr, C: tl.constexpr):
    """Read from a zero-haloed buffer, write into the next one's interior: both unmasked."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    h = rows // W
    w = rows % W
    lanes = tl.arange(0, C)
    acc = tl.zeros((BLOCK_M, C), dtype=tl.int32)
    for tap in tl.range(0, 9, num_stages=1):
        dy = tap // 3 - 1
        dx = tap % 3 - 1
        src = ((h + 1 + dy) * (W + 2) + (w + 1 + dx))[:, None] * C + lanes[None, :]
        a = tl.load(x_ptr + src)
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
    dst = ((h + 1) * (W + 2) + (w + 1))[:, None] * C + lanes[None, :]
    tl.store(y_ptr + dst, q.to(tl.int8))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=169)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=15)
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
    seeds = torch.empty((1, H, W, body.feat), device=dev, dtype=torch.int8)
    quantise_nchw_to_nhwc_prelu[qgrid](head, body._s_t1, body._slope_half, seeds, n_pix, BLOCK=QB,
                                       C=body.feat, num_warps=4)
    torch.cuda.synchronize()

    bm, warps = body.block_m, body.warps
    grid = ((n_pix + bm - 1) // bm,)
    L = body._launches

    # ---- today: three (H, W, C) buffers, masked kernel, 16 launches
    a_t, b_t, c_t = (torch.empty_like(seeds) for _ in range(3))
    a_t.copy_(seeds)

    def chain_today():
        src, nxt, other = a_t, b_t, c_t
        for i, launch in enumerate(L):
            body_prologue_kernel[grid](src, launch["w"], launch["mn"], SHIFT, launch["vec"],
                                       launch["bias"], launch["slope"], launch["os"], nxt, H, W,
                                       BLOCK_M=bm, C=body.feat, num_warps=warps, FOLD=0)
            src = nxt
            nxt = other if nxt is b_t else (b_t if nxt is c_t else b_t)
            if nxt is src:
                nxt = c_t if src is b_t else b_t
        return src

    # ---- padded: three (H+2, W+2, C) buffers, unmasked kernel, interior offsets
    padded = [torch.zeros((HP, WP, body.feat), device=dev, dtype=torch.int8) for _ in range(3)]
    padded[0].view(HP, WP, body.feat)[1:H + 1, 1:W + 1, :] = seeds.view(H, W, body.feat)

    def zero_borders():
        for p in padded:
            p[0, :, :] = 0
            p[-1, :, :] = 0
            p[:, 0, :] = 0
            p[:, -1, :] = 0

    def chain_padded():
        src, nxt, other = padded[0], padded[1], padded[2]
        for launch in L:
            hop_padded[grid](src, launch["w"], launch["mn"], SHIFT, launch["vec"], launch["bias"],
                             launch["slope"], launch["os"], nxt, H, W, BLOCK_M=bm, C=body.feat,
                             num_warps=warps)
            src, nxt, other = nxt, other, src
        return src

    with torch.no_grad():
        zero_borders()
        torch.cuda.synchronize()
        out_t = chain_today().clone()
        out_p = chain_padded().clone()
        torch.cuda.synchronize()
        interior = out_p[1:H + 1, 1:W + 1, :].reshape(1, H, W, body.feat)
        d = (out_t.to(torch.int16) - interior.to(torch.int16)).abs()
        print(f"codes today vs padded: {int((d > 0).sum())} of {d.numel()} differ, max {int(d.max())}")

    def timeit(fn, iters) -> float:
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(iters):
            fn()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / iters

    def zero_time() -> float:
        for _ in range(3):
            zero_borders()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(20):
            zero_borders()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / 20

    arms = {"chain today": chain_today, "chain padded": chain_padded}
    draws = {k: [] for k in arms}
    print(f"\n{'round':>5} " + "  ".join(f"{k:>16}" for k in arms) + "   (ms for 16 hops)")
    for r in range(1, args.rounds + 1):
        cells = []
        for k, fn in arms.items():
            ms = timeit(fn, args.iters)
            draws[k].append(ms)
            cells.append(f"{ms:16.3f}")
        print(f"{r:>5} " + "  ".join(cells))

    tail = slice(1, None) if args.rounds > 2 else slice(None)
    med = {k: statistics.median(v[tail]) for k, v in draws.items()}
    zt = zero_time()
    print(f"\n{'arm':>16} {'ms/16 hops':>12} {'per frame':>11}")
    for k in arms:
        print(f"{k:>16} {med[k]:12.3f} {med[k]:11.3f}")
    gain = med["chain today"] - med["chain padded"]
    print(f"\nthe halo is worth {gain:+.3f} ms of sixteen hops, and the border zeroing costs {zt:+.3f} ms "
          f"-> net {gain - zt:+.3f} ms per frame ({100.0 * (gain - zt) / med['chain today']:+.1f}%)")

    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        with open(args.out_file, "w") as fh:
            json.dump({"take": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "clip": os.path.basename(CLIP), "frame": args.frame,
                       "rounds": args.rounds, "iters": args.iters, "draws": draws, "median": med,
                       "zero_ms": zt, "net_gain_ms": gain - zt,
                       "codes_differ": int((d > 0).sum())}, fh, indent=1)
        print(f"# -> {args.out_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
