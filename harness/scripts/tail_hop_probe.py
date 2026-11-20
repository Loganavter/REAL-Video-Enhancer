"""Stage (b) of `rve-tail-as-int8-hop`: one Triton conv 576 -> 16 at 1080p, timed against today's tail.

Stage (a) (`harness/scripts/tail_int8_emulation.py`) showed the fold is inside both bars on the four held-out frames:
worst share 0.00147% against 0.1%, peak 13 against 20, and the weight quantisation adds 0.0002% over the
algebra alone. What is left is whether the kernel is FAST - the candidate replaces two steps with one:

  today      dequantise_nhwc (1.58 ms) + cuDNN fp16 conv (2.33-2.40 ms)
  candidate  one int8 conv whose weights already carry the dequant scale, epilogue fused
             (sigma * acc + bias -> fp16 NHWC), outputs padded 12 -> 16 with zeros

The kernel is the body's hop kernel with N = 16 instead of 64: nine taps, each an int8 dot of
(BLOCK_M, 64) x (64, 16) into an int32 accumulator. No requant into a next grid, no PReLU - the tail's
activation is the model's own output layer.

Usage: python harness/scripts/tail_hop_probe.py [--rounds 3] [--iters 20] [--out-file x.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
import triton
import triton.language as tl
import torch.nn.functional as TF

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "backend"))
import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)

from cut_channels_naive import load  # noqa: E402
from src.pytorch.Int8Body import Int8Body, QB  # noqa: E402
from src.pytorch.Int8Kernels import dequantise_nhwc  # noqa: E402
from src.pytorch.TorchUtils import TorchUtils  # noqa: E402
from tail_int8_emulation import CALIB, CLIP, H, W, body_codes, decode_yuv10  # noqa: E402

APP = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2"
MODEL = f"{APP}/custom_models/2x-Futsuu-Anime(1).pth"
CIN, COUT, COUT_PAD = 64, 12, 16


@triton.jit
def tail_int8_kernel(x_ptr, w_ptr, sigma_ptr, bias_ptr, y_ptr, H, W, BLOCK_M: tl.constexpr,
                     CIN: tl.constexpr, COUT_PAD: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    h = rows // W
    w = rows % W
    li = tl.arange(0, CIN)
    lo = tl.arange(0, COUT_PAD)
    acc = tl.zeros((BLOCK_M, COUT_PAD), dtype=tl.int32)
    for tap in tl.range(0, 9, num_stages=1):
        dy = tap // 3 - 1
        dx = tap % 3 - 1
        yy = h + dy
        xx = w + dx
        inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        src = (yy * W + xx)[:, None] * CIN + li[None, :]
        a = tl.load(x_ptr + src, mask=inside[:, None], other=0)
        b = tl.load(w_ptr + tap * CIN * COUT_PAD + li[:, None] * COUT_PAD + lo[None, :])
        acc += tl.dot(a, b, out_dtype=tl.int32)
    # the dequant scale is already inside the weights, so the epilogue is one multiply and one add
    v = acc.to(tl.float32) * tl.load(sigma_ptr + lo)[None, :] + tl.load(bias_ptr + lo)[None, :]
    tl.store(y_ptr + rows[:, None] * COUT_PAD + lo[None, :], v.to(tl.float16))


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
    body = Int8Body(model)
    with torch.no_grad():
        body.calibrate([frame(i) for i in CALIB])
        body._build()
        x = frame(args.frame)
        codes, s_k = body_codes(body, x)

    n_pix = H * W
    qgrid = ((n_pix + QB - 1) // QB,)

    # ---- the candidate's weights: fold s_k into w, quantise per output channel, pad 12 -> 16
    w_fp = body.tail_conv.weight.detach().float()
    b_fp = body.tail_conv.bias.detach().float()
    w_fold = w_fp * s_k.view(1, -1, 1, 1)
    sigma = w_fold.abs().amax(dim=(1, 2, 3)) / 127.0
    w_q = torch.round(w_fold / sigma.view(-1, 1, 1, 1)).clamp(-127, 127).to(torch.int8)
    taps = torch.zeros((9, CIN, COUT_PAD), device=dev, dtype=torch.int8)
    for i in range(3):
        for j in range(3):
            taps[i * 3 + j, :, :COUT] = w_q[:, :, i, j].t()
    sigma16 = torch.zeros(COUT_PAD, device=dev, dtype=torch.float32)
    sigma16[:COUT] = sigma
    bias16 = torch.zeros(COUT_PAD, device=dev, dtype=torch.float32)
    bias16[:COUT] = b_fp

    src = codes.permute(0, 2, 3, 1).contiguous().to(torch.int8)     # (1,H,W,64) NHWC codes
    y_nhwc = torch.empty((1, H, W, COUT_PAD), device=dev, dtype=torch.float16)
    deq = torch.empty((1, H, W, CIN), device=dev, dtype=torch.float16)
    grid = ((n_pix + args.block_m - 1) // args.block_m,)

    def candidate():
        tail_int8_kernel[grid](src, taps, sigma16, bias16, y_nhwc, H, W,
                               BLOCK_M=args.block_m, CIN=CIN, COUT_PAD=COUT_PAD, num_warps=args.warps)
        return y_nhwc

    last_nhwc = torch.empty((1, H, W, CIN), device=dev, dtype=torch.float16)

    def today():
        dequantise_nhwc[qgrid](src, body._launches[-1]["os"].contiguous(), last_nhwc, n_pix,
                              BLOCK=QB, C=CIN, num_warps=4)
        return body.tail_conv(last_nhwc.permute(0, 3, 1, 2))

    def dequant_only():
        dequantise_nhwc[qgrid](src, body._launches[-1]["os"].contiguous(), deq, n_pix,
                               BLOCK=QB, C=CIN, num_warps=4)

    def conv_only():
        return body.tail_conv(last_nhwc.permute(0, 3, 1, 2))

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

    # the candidate's numbers must be the ones stage (a) said were admissible, not merely fast
    with torch.no_grad():
        today()                                    # fills last_nhwc, which the reference reads
        got = candidate()
        ref = body.tail_conv(last_nhwc.permute(0, 3, 1, 2)).float()
        torch.cuda.synchronize()
        d = (got[:, :, :, :COUT].permute(0, 3, 1, 2).float() - ref).abs()
        print(f"kernel vs cuDNN tail, on the same codes: max |d| {float(d.max()):.6f}, "
              f"mean {float(d.mean()):.6f} (output mean {float(ref.abs().mean()):.4f})")

    arms = {"tail today": today, "dequant": dequant_only, "conv": conv_only, "tail int8": candidate}
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
    print("\nmedians of rounds 2+:")
    for k in arms:
        print(f"  {k:>11}: {med[k]:7.3f} ms")
    save = med["tail today"] - med["tail int8"]
    print(f"\nsaving: {save:+.3f} ms per frame "
          f"({100.0 * save / med['tail today']:+.1f}%) - of which the pass {med['dequant']:.3f} ms "
          f"and the conv {med['conv'] - med['tail int8']:+.3f} ms")
    print(f"against the recorded in-app conv (2.35 ms): the candidate reads {med['tail int8']:.3f} ms for "
          f"conv + epilogue against {med['conv']:.3f} for the conv alone")

    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        with open(args.out_file, "w") as fh:
            json.dump({"take": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "clip": os.path.basename(CLIP), "frame": args.frame,
                       "rounds": args.rounds, "iters": args.iters,
                       "block_m": args.block_m, "warps": args.warps,
                       "draws": draws, "median": med, "saving_ms": save}, fh, indent=1)
        print(f"# -> {args.out_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
