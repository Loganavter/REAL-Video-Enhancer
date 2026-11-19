"""What the hop is actually bound by: decompose it into taps, store and launch, without ncu.

The lane's own diagnosis says the hop runs at 91 TOPS and 158 GB/s - "52-55% of BOTH roofs, therefore
schedule-bound" - and the lane's own conclusion says "25.7 ms is the floor". Those are different claims, and
the cause the diagnosis names has never been measured on its own. `ncu` would do it in one run, but it is not
installed on this host, so this decomposes the kernel instead by removing one component at a time:

  full        today's kernel: 9 taps, int32 accumulate, epilogue, store        (the reference)
  taps=1      one tap only (the centre one): the marginal cost of a tap
  taps=3      three taps: linearity check
  no-store    the whole convolution and epilogue, the store removed
  store-only  the epilogue and store over a zero accumulator: the write alone

What each difference means:
  (full - no-store)      the cost of writing 132 MB of int8 codes
  (full - store-only)    the cost of the nine gathers + the dots
  (full - taps=1) / 8    the marginal cost of one tap  -> if ~the same as taps=1, loads+dot are linear in taps
  store-only vs 132 MB / 288 GB/s   whether the write is near its own traffic floor

The codes are not compared here (removing taps changes the arithmetic by construction): this is a timing
decomposition, not a correctness test.

Usage: python scripts/hop_roof_probe.py [--rounds 3] [--iters 20] [--out-file x.json]
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
WRITE_MB = 132.7          # 2.0736 Mpx x 64 B of int8 codes
READ_MB = 132.7           # compulsory reads (each input pixel is read once from DRAM, 9 times from L2)
BW = 288.0


@triton.jit
def hop_variant(x_ptr, w_ptr, m_next_ptr, shift, w_scale_ptr, bias_ptr, slope_ptr, out_scale_ptr,
                y_ptr, H, W, BLOCK_M: tl.constexpr, C: tl.constexpr, TAPS: tl.constexpr,
                STORE: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    h = rows // W
    w = rows % W
    lanes = tl.arange(0, C)
    acc = tl.zeros((BLOCK_M, C), dtype=tl.int32)
    if TAPS > 0:
        for tap in tl.range(0, TAPS, num_stages=1):
            # taps are taken from the CENTRE outwards when TAPS < 9, so a 1-tap arm is one centre load
            t = (tap + 4) % 9 if TAPS == 1 else tap
            dy = t // 3 - 1
            dx = t % 3 - 1
            yy = h + dy
            xx = w + dx
            inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            src = (yy * W + xx)[:, None] * C + lanes[None, :]
            a = tl.load(x_ptr + src, mask=inside[:, None], other=0)
            b = tl.load(w_ptr + t * C * C + lanes[:, None] * C + lanes[None, :])
            acc += tl.dot(a, b, out_dtype=tl.int32)
    v = acc.to(tl.float32) * tl.load(w_scale_ptr + lanes)[None, :] + tl.load(bias_ptr + lanes)[None, :]
    slope = tl.load(slope_ptr + lanes)[None, :]
    v = tl.where(v >= 0, v, v * slope)
    out_scale = tl.load(out_scale_ptr + lanes)[None, :]
    q = tl.floor(v / out_scale + 0.5)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int32)
    q = ((q * tl.load(m_next_ptr + lanes)[None, :]) + (1 << (shift - 1))) >> shift
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    if STORE:
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
    bm, warps = body.block_m, body.warps
    grid = ((n_pix + bm - 1) // bm,)

    def arm(taps: int, store: bool):
        def run():
            hop_variant[grid](src_buf, L["w"], L["mn"], SHIFT, L["vec"], L["bias"], L["slope"], L["os"],
                              dst, H, W, BLOCK_M=bm, C=body.feat, TAPS=taps, STORE=store,
                              num_warps=warps)
            return dst
        return run

    arms = {"full 9 taps": arm(9, True), "taps=1": arm(1, True), "taps=3": arm(3, True),
            "no-store": arm(9, False), "store-only": arm(0, True)}

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
    print(f"{'round':>5} " + "  ".join(f"{k:>11}" for k in arms) + "   (ms)")
    for r in range(1, args.rounds + 1):
        cells = []
        for k, fn in arms.items():
            ms = timeit(fn)
            draws[k].append(ms)
            cells.append(f"{ms:11.3f}")
        print(f"{r:>5} " + "  ".join(cells))

    tail = slice(1, None) if args.rounds > 2 else slice(None)
    med = {k: statistics.median(v[tail]) for k, v in draws.items()}
    full = med["full 9 taps"]
    print("\nmedians of rounds 2+ and what each difference means:")
    for k in arms:
        print(f"  {k:>11}: {med[k]:6.3f} ms")
    dw = full - med["no-store"]
    dc = full - med["store-only"]
    per_tap = (full - med["taps=1"]) / 8.0
    write_floor = WRITE_MB / BW
    print(f"\n  the write (full - no-store):        {dw:+.3f} ms   "
          f"(132.7 MB at 288 GB/s = {write_floor:.2f} ms)")
    print(f"  the nine taps + dots (full - store): {dc:+.3f} ms")
    print(f"  marginal cost of one tap:            {per_tap:.3f} ms   "
          f"(taps=1 alone reads {med['taps=1']:.3f} ms)")
    print(f"  one tap for all nine:                {9 * per_tap:.3f} ms "
          f"vs {dc:.3f} measured - "
          f"{'linear in taps' if abs(9 * per_tap - dc) < 0.25 * dc else 'NOT linear in taps'}")
    print(f"\n  achieved: {152.9 / full:5.1f} TOPS, "
          f"{(WRITE_MB + READ_MB) / full:5.1f} GB/s of compulsory traffic")

    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        with open(args.out_file, "w") as fh:
            json.dump({"take": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "clip": os.path.basename(CLIP), "frame": args.frame,
                       "rounds": args.rounds, "iters": args.iters, "draws": draws, "median": med,
                       "write_ms": dw, "compute_ms": dc, "per_tap_ms": per_tap,
                       "write_floor_ms": write_floor}, fh, indent=1)
        print(f"# -> {args.out_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
