"""Does a persistent CTA grid move the hop? The one schedule variant that does not add an accumulator.

The hop is closed at 1.756 ms per launch from both ends already measured (vectorised loads +0.9%, weight hoist
-54% on spill, 6-bit codes dead on the picture). What is left is the grid's SHAPE: today the launch is 16200
CTAs of one 128-row tile each, so every CTA pays its own index arithmetic, weight loads and teardown. A
persistent grid starts `k` CTAs (k = SMs, or 2k) and lets each loop over tiles with stride k: the launch and
teardown cost is paid k times instead of 16200, and the weight tiles stay hot in L2 across the loop.

What this canNOT claim: the weights themselves are not held in registers across the loop (9 x 64 x 64 int8 is
36 KB - it would spill, which is exactly what killed the hoist variant), so the win here is launch/teardown
and L2 locality, not weight reuse.

Usage: python harness/scripts/hop_persistent_probe.py [--rounds 3] [--iters 20] [--out-file x.json]
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
from src.pytorch.Int8Kernels import body_prologue_kernel, quantise_nchw_to_nhwc_prelu  # noqa: E402
from src.pytorch.TorchUtils import TorchUtils  # noqa: E402
from tail_int8_emulation import CALIB, CLIP, H, W, decode_yuv10  # noqa: E402

APP = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2"
MODEL = f"{APP}/custom_models/2x-Futsuu-Anime(1).pth"
TOPS, GBPS = 176.0, 288.0
GFLOP_PER_HOP = 152.9
MB_PER_HOP = 265.0


@triton.jit
def hop_persistent(x_ptr, w_ptr, m_next_ptr, shift, w_scale_ptr, bias_ptr, slope_ptr, out_scale_ptr,
                   y_ptr, H, W, n_tiles, BLOCK_M: tl.constexpr, C: tl.constexpr):
    pid = tl.program_id(0)
    npg = tl.num_programs(0)
    lanes = tl.arange(0, C)
    ws = tl.load(w_scale_ptr + lanes)[None, :]
    bs = tl.load(bias_ptr + lanes)[None, :]
    slope = tl.load(slope_ptr + lanes)[None, :]
    osc = tl.load(out_scale_ptr + lanes)[None, :]
    mn = tl.load(m_next_ptr + lanes)[None, :]
    for tile in tl.range(pid, n_tiles, npg):
        rows = tile * BLOCK_M + tl.arange(0, BLOCK_M)
        h = rows // W
        w = rows % W
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
        v = acc.to(tl.float32) * ws + bs
        v = tl.where(v >= 0, v, v * slope)
        q = tl.floor(v / osc + 0.5)
        q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int32)
        q = ((q * mn) + (1 << (shift - 1))) >> shift
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
    sms = torch.cuda.get_device_properties(0).multi_processor_count
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
    n_tiles = (n_pix + bm - 1) // bm
    print(f"{sms} SMs, {n_tiles} tiles of {bm} rows")

    def today():
        body_prologue_kernel[(n_tiles,)](src_buf, L["w"], L["mn"], SHIFT, L["vec"], L["bias"], L["slope"],
                                         L["os"], dst, H, W, BLOCK_M=bm, C=body.feat, num_warps=warps,
                                         FOLD=0)
        return dst

    def persist(k: int):
        def run():
            hop_persistent[(k,)](src_buf, L["w"], L["mn"], SHIFT, L["vec"], L["bias"], L["slope"], L["os"],
                                 dst, H, W, n_tiles, BLOCK_M=bm, C=body.feat, num_warps=warps)
            return dst
        return run

    arms = {"today": today, f"persist{sms}": persist(sms), f"persist{2 * sms}": persist(2 * sms)}

    with torch.no_grad():
        base = today().clone()
        for k, fn in arms.items():
            if k == "today":
                continue
            got = fn().clone()
            torch.cuda.synchronize()
            d = (base.to(torch.int16) - got.to(torch.int16)).abs()
            print(f"codes {k} vs today: {int((d > 0).sum())} of {d.numel()} differ, max {int(d.max())}")

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
    print(f"\n{'round':>5} " + "  ".join(f"{k:>11}" for k in arms) + "   (ms)")
    for r in range(1, args.rounds + 1):
        cells = []
        for k, fn in arms.items():
            ms = timeit(fn)
            draws[k].append(ms)
            cells.append(f"{ms:11.3f}")
        print(f"{r:>5} " + "  ".join(cells))

    tail = slice(1, None) if args.rounds > 2 else slice(None)
    med = {k: statistics.median(v[tail]) for k, v in draws.items()}
    print(f"\nmedians of rounds 2+ (roofs {TOPS:.0f} TOPS / {GBPS:.0f} GB/s):")
    for k in arms:
        tops, gbps = GFLOP_PER_HOP / med[k], MB_PER_HOP / med[k]
        print(f"  {k:>11}: {med[k]:6.3f} ms   {tops:5.1f} TOPS ({100 * tops / TOPS:4.1f}%)   "
              f"{gbps:6.1f} GB/s ({100 * gbps / GBPS:4.1f}%)")
    best = min(med, key=med.get)
    gain = 100.0 * (med["today"] - med[best]) / med["today"]
    print(f"\nbest: {best}, {gain:+.1f}% against today -> "
          f"{'MOVES the hop' if gain >= 8 else 'no arm reaches the 8% the row asked for - the hop stays at its floor'}")

    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        with open(args.out_file, "w") as fh:
            json.dump({"take": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "clip": os.path.basename(CLIP), "frame": args.frame, "sms": sms,
                       "n_tiles": n_tiles, "rounds": args.rounds, "iters": args.iters,
                       "roofs": {"tops": TOPS, "gbps": GBPS},
                       "draws": draws, "median": med,
                       "tops": {k: GFLOP_PER_HOP / v for k, v in med.items()},
                       "gbps": {k: MB_PER_HOP / v for k, v in med.items()},
                       "best": best, "gain_pct": gain}, fh, indent=1)
        print(f"# -> {args.out_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
