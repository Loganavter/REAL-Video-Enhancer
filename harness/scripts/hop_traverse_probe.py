"""Reverse the tile traversal on alternate hops: does the previous hop's tail stay in L2?

From an outside reader's list, the one item this corpus never tested and that is expressible in one line. The
hop writes 132 MB per launch and L2 is 32 MB, so when hop k+1 starts reading tile 0 it finds data that was
written 132 MB ago - long evicted. If hop k+1 walks its tiles in REVERSE, it first reads the tiles hop k wrote
LAST, i.e. what is still in L2 (up to ~32 MB of 132).

Ceiling, from the counters (int8-hop-dram.txt): 132.84 MB of DRAM read per hop; if a quarter of it becomes an L2
hit, that is ~33 MB = 0.11 ms per hop, ~1.8 ms over sixteen. Not nothing, and it costs one subtraction.

Arms, all with identical codes expected (this changes the ORDER of independent CTAs, not the arithmetic):
  forward         today: every hop walks tiles 0..N
  alternate       hop k+1 walks N..0 when hop k walked 0..N
  reverse-all     every hop walks N..0 (control: if this alone helps, it is not about adjacency)

Usage: python harness/scripts/hop_traverse_probe.py [--rounds 3] [--iters 15] [--out-file x.json]
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


@triton.jit
def hop_rev(x_ptr, w_ptr, m_next_ptr, shift, w_scale_ptr, bias_ptr, slope_ptr, out_scale_ptr, y_ptr,
            H, W, n_tiles, BLOCK_M: tl.constexpr, C: tl.constexpr, REVERSE: tl.constexpr):
    pid = tl.program_id(0)
    if REVERSE:
        pid = n_tiles - 1 - pid
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
    seed = torch.empty((1, H, W, body.feat), device=dev, dtype=torch.int8)
    quantise_nchw_to_nhwc_prelu[qgrid](head, body._s_t1, body._slope_half, seed, n_pix, BLOCK=QB,
                                       C=body.feat, num_warps=4)
    torch.cuda.synchronize()

    bm, warps = body.block_m, body.warps
    n_tiles = (n_pix + bm - 1) // bm
    grid = (n_tiles,)
    L = body._launches
    bufs = [torch.empty_like(seed) for _ in range(3)]

    def chain(mode: str):
        src, nxt, other = seed, bufs[0], bufs[1]
        for k, launch in enumerate(L):
            if mode == "forward":
                rev = 0
            elif mode == "alternate":
                rev = k % 2
            else:
                rev = 1
            hop_rev[grid](src, launch["w"], launch["mn"], SHIFT, launch["vec"], launch["bias"],
                          launch["slope"], launch["os"], nxt, H, W, n_tiles, BLOCK_M=bm, C=body.feat,
                          REVERSE=rev, num_warps=warps)
            src, nxt, other = nxt, other, src
        return src

    with torch.no_grad():
        base = chain("forward").clone()
        torch.cuda.synchronize()
        for k in ("alternate", "reverse-all"):
            got = chain(k).clone()
            torch.cuda.synchronize()
            d = (base.to(torch.int16) - got.to(torch.int16)).abs()
            print(f"codes {k:>12} vs forward: {int((d > 0).sum())} of {d.numel()} differ, max {int(d.max())}")

    def timeit(fn) -> float:
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(args.iters):
            fn()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / args.iters

    draws = {k: [] for k in ("forward", "alternate", "reverse-all")}
    print(f"\n{'round':>5} " + "  ".join(f"{k:>13}" for k in draws) + "   (ms for 16 hops)")
    for r in range(1, args.rounds + 1):
        cells = []
        for k in draws:
            ms = timeit(lambda: chain(k))
            draws[k].append(ms)
            cells.append(f"{ms:13.3f}")
        print(f"{r:>5} " + "  ".join(cells))

    tail = slice(1, None) if args.rounds > 2 else slice(None)
    med = {k: statistics.median(v[tail]) for k, v in draws.items()}
    print(f"\n{'arm':>13} {'ms/16':>8} {'vs forward':>12}")
    for k in draws:
        print(f"{k:>13} {med[k]:8.3f} {100.0 * (med['forward'] - med[k]) / med['forward']:+11.2f}%")
    best = min(med, key=med.get)
    print(f"\nbest: {best} ({100.0 * (med['forward'] - med[best]) / med['forward']:+.2f}%); "
          f"the L2-adjacency claim would show as `alternate` beating BOTH forward and reverse-all")

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
