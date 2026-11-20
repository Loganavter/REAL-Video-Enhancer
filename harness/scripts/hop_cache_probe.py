"""Can the hop's 6.1x L2 amplification be cut by keeping the three-row window closer?

ncu (2026-11-19) on one hop: DRAM 250.6 MB, **L2 1528 MB - 6.1x the DRAM** at a 91.25% hit rate, and **L1/TEX
hit rate only 22.3%**. The nine taps of a pixel read the same three image rows over and over, so the traffic is
amplified in L2 (which serves it) and barely served by L1.

A CTA's window is 128 pixels x 3 rows x 64 B = 25 KB, which fits an SM's 128 KB L1 several times over. So the
question is not capacity but POLICY: Triton exposes `eviction_policy` (evict_normal / evict_first / evict_last)
and `cache_modifier` (.ca / .cg / .cs) per load. This sweeps them on the real kernel, all with the shipped
BLOCK_M 128 / 4 warps, and checks the codes (a policy changes no arithmetic - any difference is a bug).

`evict_last` on the activation loads is the natural candidate: the row that is about to be re-read by the next
tap should survive, and the weights (read once per CTA) should not occupy the cache - so the pair
(activations=evict_last, weights=evict_first) is the arm this file exists to test.

Usage: python harness/scripts/hop_cache_probe.py [--rounds 3] [--iters 20] [--out-file x.json]
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
MB_PER_HOP = 250.6      # measured DRAM, int8-hop-dram.txt


@triton.jit
def hop(x_ptr, w_ptr, m_next_ptr, shift, w_scale_ptr, bias_ptr, slope_ptr, out_scale_ptr, y_ptr,
        H, W, BLOCK_M: tl.constexpr, C: tl.constexpr,
        EV_A: tl.constexpr, EV_W: tl.constexpr, CM_A: tl.constexpr):
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
        src = (yy * W + xx)[:, None] * C + lanes[None, :]
        a = tl.load(x_ptr + src, mask=inside[:, None], other=0, eviction_policy=EV_A,
                    cache_modifier=CM_A)
        b = tl.load(w_ptr + tap * C * C + lanes[:, None] * C + lanes[None, :], eviction_policy=EV_W)
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
    src_buf = torch.empty((1, H, W, body.feat), device=dev, dtype=torch.int8)
    quantise_nchw_to_nhwc_prelu[qgrid](head, body._s_t1, body._slope_half, src_buf, n_pix, BLOCK=QB,
                                       C=body.feat, num_warps=4)
    dst = torch.empty_like(src_buf)
    torch.cuda.synchronize()
    L = body._launches[0]
    bm = body.block_m
    grid = ((n_pix + bm - 1) // bm,)

    def arm(ev_a: str, ev_w: str, cm_a: str):
        def run():
            hop[grid](src_buf, L["w"], L["mn"], SHIFT, L["vec"], L["bias"], L["slope"], L["os"], dst,
                      H, W, BLOCK_M=bm, C=body.feat, EV_A=ev_a, EV_W=ev_w, CM_A=cm_a,
                      num_warps=body.warps)
            return dst
        return run

    arms = {
        "shipped (normal)": arm("", "", ""),
        "act evict_last": arm("evict_last", "", ""),
        "act evict_last + w evict_first": arm("evict_last", "evict_first", ""),
        "act evict_first": arm("evict_first", "", ""),
        "act .cg (skip L1)": arm("", "", ".cg"),

    }

    with torch.no_grad():
        base = arms["shipped (normal)"]().clone()
        torch.cuda.synchronize()
        for k, fn in arms.items():
            got = fn().clone()
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
    print(f"\n{'round':>5} " + "  ".join(f"{k[:14]:>16}" for k in arms))
    for r in range(1, args.rounds + 1):
        cells = []
        for k, fn in arms.items():
            ms = timeit(fn)
            draws[k].append(ms)
            cells.append(f"{ms:16.3f}")
        print(f"{r:>5} " + "  ".join(cells))

    tail = slice(1, None) if args.rounds > 2 else slice(None)
    med = {k: statistics.median(v[tail]) for k, v in draws.items()}
    print(f"\n{'arm':>34} {'ms':>8} {'GB/s':>7} {'vs shipped':>12}")
    for k in arms:
        rel = 100.0 * (med["shipped (normal)"] - med[k]) / med["shipped (normal)"]
        print(f"{k:>34} {med[k]:8.3f} {MB_PER_HOP / med[k]:7.1f} {rel:+11.1f}%")
    best = min(med, key=med.get)
    print(f"\nbest: {best} ({100.0 * (med['shipped (normal)'] - med[best]) / med['shipped (normal)']:+.1f}%)")

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
