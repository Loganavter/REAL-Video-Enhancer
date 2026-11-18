"""Can a Triton head conv (3 -> 64) beat torch's 3.2 ms at all? The falsifier before the fusion.

The fusion this serves (`rve-head-quantise-fusion-2026-11-19`) is worth 1.5-2.5 ms only if a fused kernel can
match or beat the torch head conv it replaces - 3.2-3.8 ms for 7.17 GFLOP at 1080p, which is 1.9-2.2 TFLOPS
and 3.3x its own traffic floor. If a hand-written 3-input convolution is slower than cudnn's, the fusion has
nothing to stand on and the row closes with that measurement.

This probe measures the CONV only (fp16 out), not the epilogue: it is the falsifier, not the feature.
Arithmetic: out[oc, y, x] = sum over 3 input channels x 9 taps, one program per block of pixels, C=64
output channels held in a tile. Random tensors - latency does not depend on the values.

Usage: python scripts/head_conv_triton.py [--height 1080] [--width 1920] [--block 64]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def head_conv_3x3(q_ptr, w_ptr, b_ptr, y_ptr, n_pix, W: tl.constexpr, H: tl.constexpr,
                  C: tl.constexpr, BLOCK: tl.constexpr):
    """One program per BLOCK pixels; NCHW input, C output channels, 3x3, zero pad."""
    pid = tl.program_id(0)
    pix = pid * BLOCK + tl.arange(0, BLOCK)
    mask = pix < n_pix
    y = pix // W
    x = pix - y * W
    acc = tl.zeros((BLOCK, C), dtype=tl.float32)
    for ic in tl.static_range(3):
        for ky in tl.static_range(3):
            for kx in tl.static_range(3):
                yy = y + ky - 1
                xx = x + kx - 1
                ok = mask & (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
                src = tl.load(q_ptr + ic * n_pix + yy * W + xx, mask=ok, other=0.0)
                # weights laid out [C, 3, 3, 3] -> tap slice (ic, ky, kx) over C
                w = tl.load(w_ptr + tl.arange(0, C) * 27 + ic * 9 + ky * 3 + kx)
                acc += src[:, None] * w[None, :]
    bias = tl.load(b_ptr + tl.arange(0, C))
    acc += bias[None, :]
    tl.store(y_ptr + pix[:, None] * C + tl.arange(0, C)[None, :], acc.to(tl.float16), mask=mask[:, None])


def timeit(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--c", type=int, default=64)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args(argv)

    px = args.height * args.width
    flop = px * args.c * 27 * 2 / 1e9
    x = torch.rand(3, args.height, args.width, device="cuda", dtype=torch.float16).contiguous()
    conv = torch.nn.Conv2d(3, args.c, 3, padding=1, bias=True).cuda().half()
    w = conv.weight.detach().permute(0, 2, 3, 1).contiguous()   # [C, 3, 3, 3] as the kernel indexes it
    b = conv.bias.detach().contiguous()
    out = torch.empty(px, args.c, device="cuda", dtype=torch.float16)
    grid = (triton.cdiv(px, args.block),)

    with torch.no_grad():
        torch_ms = timeit(lambda: conv(x[None]))
    triton_ms = timeit(lambda: head_conv_3x3[grid](x, w, b, out, px, args.width, args.height,
                                                  args.c, args.block))
    with torch.no_grad():
        ref = conv(x[None]).float()[0].permute(1, 2, 0).reshape(px, args.c)
    d = (out.float() - ref).abs()
    print(f"head 3->{args.c} 3x3 at {args.width}x{args.height}: {flop:.2f} GFLOP")
    print(f"  torch (cudnn)  {torch_ms:7.3f} ms  {flop/torch_ms:5.2f} TFLOPS")
    print(f"  triton BLOCK={args.block} {triton_ms:7.3f} ms  {flop/triton_ms:5.2f} TFLOPS")
    print(f"  agreement: max |d| {float(d.max()):.4f} levels, mean {float(d.mean()):.5f}, "
          f"share >8 {float((d > 8).float().mean())*100:.4f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
