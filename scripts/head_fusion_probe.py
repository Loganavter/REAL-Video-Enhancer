"""Is a fused head + PReLU + quantise kernel worth it? One Triton conv 3 -> 64 that writes int8 codes.

The shipped head is three steps: a cuDNN fp16 convolution (4.65 ms in the app), a separate PReLU, and a
quantise pass that reads the 265 MB fp16 activation and writes 133 MB of codes (1.60 ms). The convolution is
3 -> 64, 3x3, i.e. K = 27 - small enough that the whole 3x3x3 window of one pixel is ONE dot of K = 27 padded
to 32, and the epilogue can apply the bias, the PReLU and the requant before the store.

Arms (row `rve-head-quantise-fusion-2026-11-19`):
  today     head_conv + F.prelu + quantise_nchw_to_nhwc_prelu   (what the app runs)
  head      head_conv alone
  quantise  the quantise pass alone
  fused     one kernel: conv + bias + PReLU + requant, writes int8 NHWC

The saving has two parts and they are reported separately: the eliminated 265 MB fp16 write and the
eliminated pass. Codes are compared element-wise first - the epilogue order is the same as the shipped path's
(conv, then bias, then PReLU, then round-to-nearest on the per-channel grid), so this is a candidate for
agreement rather than for the picture bar.

Usage: python scripts/head_fusion_probe.py [--rounds 3] [--iters 20] [--block-m 128] [--warps 4]
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
import torch.nn.functional as TF

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "backend"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from cut_channels_naive import load  # noqa: E402
from src.pytorch.Int8Body import QB  # noqa: E402
from src.pytorch.Int8Kernels import quantise_nchw_to_nhwc_prelu  # noqa: E402
from src.pytorch.TorchUtils import TorchUtils  # noqa: E402
from tail_int8_emulation import CALIB, CLIP, H, W, decode_yuv10  # noqa: E402

APP = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2"
MODEL = f"{APP}/custom_models/2x-Futsuu-Anime(1).pth"
CIN, COUT, K = 3, 64, 32


@triton.jit
def head_int8_kernel(x_ptr, w_ptr, bias_ptr, slope_ptr, s_ptr, y_ptr, H, W, BLOCK_M: tl.constexpr,
                     C: tl.constexpr, K: tl.constexpr, CIN: tl.constexpr):
    """3 -> C fp16 convolution with the PReLU and the int8 requant in the epilogue.

    K = 27 real lanes (3 channels x 9 taps) padded to 32; the padding lanes are masked out of both the
    activation load and the weight load, so they contribute exactly zero. The input is the app's contiguous
    NCHW fp16 frame, so lane k indexes plane `k // 9` at the shifted coordinate.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    h = rows // W
    w = rows % W
    ki = tl.arange(0, K)
    c = ki // 9
    t = ki % 9
    dy = t // 3 - 1
    dx = t % 3 - 1
    yy = h[:, None] + dy[None, :]
    xx = w[:, None] + dx[None, :]
    inside = (c[None, :] < CIN) & (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    src = c[None, :] * (H * W) + yy * W + xx
    a = tl.load(x_ptr + src, mask=inside, other=0.0)                     # (BLOCK_M, K) fp16
    lo = tl.arange(0, C)
    b = tl.load(w_ptr + ki[:, None] * C + lo[None, :], mask=(c[:, None] < CIN), other=0.0)
    acc = tl.dot(a, b)                                                   # (BLOCK_M, C) fp32
    v = acc + tl.load(bias_ptr + lo)[None, :]
    slope = tl.load(slope_ptr + lo)[None, :]
    v = tl.where(v >= 0, v, v * slope)
    s = tl.load(s_ptr + lo)[None, :]
    q = tl.floor(v / s + 0.5)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(y_ptr + rows[:, None] * C + lo[None, :], q.to(tl.int8))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=169)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--block-m", type=int, default=128)
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--out-file", default="")
    args = ap.parse_args(argv)

    dev = torch.device("cuda", 0)
    tu = TorchUtils(width=W, height=H, device_type="cuda", hdr_mode=True, gpu_id=0, high_bitdepth=True)
    stream = torch.cuda.Stream(device=dev)

    def frame(idx: int) -> torch.Tensor:
        return tu.yuv420p10_to_tensor(decode_yuv10(idx), stream, dev, torch.float16)

    model, _, _ = load(MODEL)
    model = model.eval().cuda().half()
    from src.pytorch.Int8Body import Int8Body
    body = Int8Body(model)
    with torch.no_grad():
        body.calibrate([frame(i) for i in CALIB])
        body._build()
        x = frame(args.frame).clamp(0.0, 1.0)

    n_pix = H * W
    qgrid = ((n_pix + QB - 1) // QB,)
    w = body.head_conv.weight.detach().half()        # (64, 3, 3, 3)
    bias = body.head_conv.bias.detach().float()
    slope = body.head_prelu.weight.detach().float()
    s_act = body._s_t1.detach().float()

    # (K, C) fp16 weights: lane k = c*9 + t, exactly the kernel's own index
    w32 = torch.zeros((K, COUT), device=dev, dtype=torch.float16)
    for c in range(CIN):
        for i in range(3):
            for j in range(3):
                w32[c * 9 + i * 3 + j] = w[:, c, i, j]

    buf = torch.empty((1, H, W, COUT), device=dev, dtype=torch.int8)
    grid = ((n_pix + args.block_m - 1) // args.block_m,)

    def fused():
        head_int8_kernel[grid](x, w32, bias, slope, s_act, buf, H, W, BLOCK_M=args.block_m, C=COUT,
                               K=K, CIN=CIN, num_warps=args.warps)
        return buf

    def head_only():
        return body.head_conv(x)

    def quantise_only():
        hd = body.head_conv(x)
        quantise_nchw_to_nhwc_prelu[qgrid](hd, s_act, body._slope_half, buf, n_pix, BLOCK=QB, C=COUT,
                                           num_warps=4)
        return buf

    def today():
        return quantise_only()

    # ---- correctness: the same codes, or a named difference
    with torch.no_grad():
        ref = quantise_only().clone()
        got = fused().clone()
        torch.cuda.synchronize()
        d = (ref.to(torch.int16) - got.to(torch.int16)).abs()
        n_diff = int((d > 0).sum())
        print(f"codes: {n_diff} of {d.numel()} differ ({100.0 * n_diff / d.numel():.5f}%), "
              f"max |diff| {int(d.max())}")

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

    arms = {"today": today, "head": head_only, "quantise": quantise_only, "fused": fused}
    draws = {k: [] for k in arms}
    print(f"\n{'round':>5} " + "  ".join(f"{k:>10}" for k in arms) + "   (ms)")
    for r in range(1, args.rounds + 1):
        cells = []
        for k, fn in arms.items():
            ms = timeit(fn)
            draws[k].append(ms)
            cells.append(f"{ms:10.3f}")
        print(f"{r:>5} " + "  ".join(cells))

    tail = slice(1, None) if args.rounds > 2 else slice(None)
    med = {k: statistics.median(v[tail]) for k, v in draws.items()}
    print("\nmedians of rounds 2+:")
    for k in arms:
        print(f"  {k:>10}: {med[k]:7.3f} ms")
    save = med["today"] - med["fused"]
    print(f"\nsaving: {save:+.3f} ms per frame ({100.0 * save / med['today']:+.1f}%)")
    print(f"  of which the eliminated quantise pass: {med['quantise'] - med['head']:+.3f} ms")
    print(f"  and the eliminated 265 MB fp16 write inside it (0.92 ms of traffic at 288 GB/s)")
    print(f"  the fused kernel reads {med['fused']:.3f} ms for conv + PReLU + requant against "
          f"{med['head']:.3f} for the conv alone")
    verdict = "PASS" if save >= 1.5 else "FAIL - the shape is at its floor, close the row"
    print(f"verdict against the row's 1.5 ms bar: {verdict}")

    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        with open(args.out_file, "w") as fh:
            json.dump({"take": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "clip": os.path.basename(CLIP), "frame": args.frame,
                       "rounds": args.rounds, "iters": args.iters,
                       "block_m": args.block_m, "warps": args.warps,
                       "draws": draws, "median": med, "saving_ms": save,
                       "codes": {"n_diff": n_diff, "total": d.numel(), "max": int(d.max())},
                       "pass": save >= 1.5}, fh, indent=1)
        print(f"# -> {args.out_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
