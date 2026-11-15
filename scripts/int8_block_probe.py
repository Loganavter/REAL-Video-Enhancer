"""int8 block probe: one 3x3 conv 64->64 with a fused per-channel PReLU and requant.

The measured ceiling (40.24 ms) used a SCALAR activation fused into the int8 epilogue; the plugin this
project would need carries a per-channel slope and a per-channel output scale, accumulates in int32 and
requantises to int8. That configuration has never been measured here. This probe measures it as a
kernel rather than as a promise.

Acceptance, from the plan:
  block time <= 1.9-2.0 ms for 64->64 at 1920x1080 including the epilogue
  >= 76-80 TOPS while doing the epilogue, not before it
  picture after the requant against an fp16 reference, worst-channel share above 8 levels

Layout is NHWC int8 for activations and K-major int8 weights per tap, which is what the 147 TOPS
measurement says the hardware wants (39 TOPS in the other layout). Weight scales are per-tensor here so
the int32 accumulator stays valid; the output scale and the slope are per channel, which is the part
that could not fit a TensorRT epilogue.

Usage: int8_block_probe.py [--height 1080] [--width 1920] [--iters 20] [--check-only]
"""
import argparse
import sys

import torch
import triton
import triton.language as tl

BLOCK_M = 128


@triton.jit
def conv3x3_int8_prelu_kernel(
    x_ptr, w_ptr, slope_ptr, out_scale_ptr,
    y_ptr, H, W, in_scale, w_scale,
    BLOCK_M: tl.constexpr, C: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)          # linear pixel index h*W + w
    h = rows // W
    w = rows % W
    lanes = tl.arange(0, C)
    acc = tl.zeros((BLOCK_M, C), dtype=tl.int32)

    for tap in tl.static_range(9):
        dy = tap // 3 - 1
        dx = tap % 3 - 1
        yy = h + dy
        xx = w + dx
        inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        src = (yy * W + xx)[:, None] * C + lanes[None, :]
        a = tl.load(x_ptr + src, mask=inside[:, None], other=0)
        b = tl.load(w_ptr + tap * C * C + lanes[:, None] * C + lanes[None, :])
        acc += tl.dot(a, b, out_dtype=tl.int32)

    # epilogue: dequantise once, per-channel PReLU, per-channel requant, clamp, store as int8
    scale = in_scale * w_scale
    v = acc.to(tl.float32) * scale
    slope = tl.load(slope_ptr + lanes)[None, :]
    v = tl.where(v >= 0, v, v * slope)
    out_scale = tl.load(out_scale_ptr + lanes)[None, :]
    q = tl.floor(v / out_scale + 0.5)          # round-half-up; the reference does the same
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    dst = rows[:, None] * C + lanes[None, :]
    tl.store(y_ptr + dst, q.to(tl.int8))


@triton.jit
def conv3x3_int8_prelu_pertap_kernel(
    x_ptr, w_ptr, tap_scale_ptr, slope_ptr, out_scale_ptr,
    y_ptr, H, W, in_scale,
    BLOCK_M: tl.constexpr, C: tl.constexpr,
):
    """Same maths with the dequant INSIDE the tap loop: one weight scale per (channel, tap).

    Exact, not an approximation - each tap accumulates 64 products in int32 (max 64*127*127 = 1.03M), the
    taps are summed in fp32. TensorRT cannot do this: its dequant must follow the whole accumulation.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    h = rows // W
    w = rows % W
    lanes = tl.arange(0, C)
    acc = tl.zeros((BLOCK_M, C), dtype=tl.float32)

    for tap in tl.static_range(9):
        dy = tap // 3 - 1
        dx = tap % 3 - 1
        yy = h + dy
        xx = w + dx
        inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        src = (yy * W + xx)[:, None] * C + lanes[None, :]
        a = tl.load(x_ptr + src, mask=inside[:, None], other=0)
        b = tl.load(w_ptr + tap * C * C + lanes[:, None] * C + lanes[None, :])
        part = tl.dot(a, b, out_dtype=tl.int32).to(tl.float32)
        acc += part * (in_scale * tl.load(tap_scale_ptr + tap * C + lanes)[None, :])

    v = acc
    slope = tl.load(slope_ptr + lanes)[None, :]
    v = tl.where(v >= 0, v, v * slope)
    out_scale = tl.load(out_scale_ptr + lanes)[None, :]
    q = tl.floor(v / out_scale + 0.5)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    dst = rows[:, None] * C + lanes[None, :]
    tl.store(y_ptr + dst, q.to(tl.int8))


@triton.jit
def conv3x3_int8_prelu_grouped_kernel(
    x_ptr, w_ptr, group_scale_ptr, slope_ptr, out_scale_ptr,
    y_ptr, H, W, in_scale,
    BLOCK_M: tl.constexpr, C: tl.constexpr,
):
    """Three int32 accumulators instead of nine: one dequant per row of the 3x3 kernel."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    h = rows // W
    w = rows % W
    lanes = tl.arange(0, C)
    acc = tl.zeros((BLOCK_M, C), dtype=tl.float32)

    for group in tl.static_range(3):
        part = tl.zeros((BLOCK_M, C), dtype=tl.int32)
        for inner in tl.static_range(3):
            tap = group * 3 + inner
            dy = group - 1
            dx = inner - 1
            yy = h + dy
            xx = w + dx
            inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            src = (yy * W + xx)[:, None] * C + lanes[None, :]
            a = tl.load(x_ptr + src, mask=inside[:, None], other=0)
            b = tl.load(w_ptr + tap * C * C + lanes[:, None] * C + lanes[None, :])
            part += tl.dot(a, b, out_dtype=tl.int32)
        acc += part.to(tl.float32) * (in_scale * tl.load(group_scale_ptr + group * C + lanes)[None, :])

    slope = tl.load(slope_ptr + lanes)[None, :]
    v = tl.where(acc >= 0, acc, acc * slope)
    out_scale = tl.load(out_scale_ptr + lanes)[None, :]
    q = tl.floor(v / out_scale + 0.5)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    dst = rows[:, None] * C + lanes[None, :]
    tl.store(y_ptr + dst, q.to(tl.int8))


def reference(x, weight, slope, in_scale, w_scale, out_scale):
    """Shifts plus int8 matmul, deliberately not conv2d: the first version's conv2d weight layout was
    wrong and made a correct kernel look broken. This one has no layout opinions."""
    _, h, w, c = x.shape
    acc = torch.zeros(h * w, c, device=x.device, dtype=torch.int32)
    for tap in range(9):
        dy, dx = tap // 3 - 1, tap % 3 - 1
        out = torch.zeros_like(x)
        ys, yt = slice(max(0, dy), h + min(0, dy)), slice(max(0, -dy), h + min(0, -dy))
        xs, xt = slice(max(0, dx), w + min(0, dx)), slice(max(0, -dx), w + min(0, -dx))
        out[:, yt, xt, :] = x[:, ys, xs, :]
        acc += torch._int_mm(out.reshape(h * w, c).contiguous(), weight[tap].contiguous())
    v = acc.float() * (in_scale * w_scale)
    v = torch.where(v >= 0, v, v * slope[None, :])
    q = torch.floor(v / out_scale[None, :] + 0.5).clamp(-127, 127)
    return q.to(torch.int8).reshape(1, h, w, c)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--check-size", type=int, default=64)
    ap.add_argument("--block", type=int, default=BLOCK_M)
    ap.add_argument("--warps", type=int, default=2)
    ap.add_argument("--per-tap", action="store_true")
    ap.add_argument("--band", action="store_true", help="one weight scale per kernel row")
    ap.add_argument("--stages", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(0)
    C = 64
    dev = "cuda"

    def tensors(h, w):
        x = torch.randint(-127, 127, (1, h, w, C), device=dev, dtype=torch.int8)
        # [9, Cin, Cout], K-major so the dot reads contiguous output channels
        weight = torch.randint(-127, 127, (9, C, C), device=dev, dtype=torch.int8)
        slope = (torch.rand(C, device=dev) * 2.4 - 0.7)          # the measured -1.39..1.70 range
        out_scale = torch.rand(C, device=dev) * 0.05 + 0.004
        return x, weight, slope, out_scale

    in_scale, w_scale = 0.037, 0.011

    if args.check_only or args.check_size:
        h = w = args.check_size
        x, weight, slope, out_scale = tensors(h, w)
        y = torch.empty_like(x)
        grid = (triton.cdiv(h * w, BLOCK_M),)
        conv3x3_int8_prelu_kernel[grid](x, weight, slope, out_scale, y, h, w,
                                        in_scale, w_scale, BLOCK_M=BLOCK_M, C=C)
        ref = reference(x, weight, slope, in_scale, w_scale, out_scale)
        err = (y.int() - ref.int()).abs()
        print(f"check {h}x{w}: max |diff| {int(err.max())}, mean {float(err.float().mean()):.3f}, "
              f"share != 0 {float((err != 0).float().mean()) * 100:.2f}%")
        if args.check_only:
            return 0

    x, weight, slope, out_scale = tensors(args.height, args.width)
    y = torch.empty_like(x)
    grid = (triton.cdiv(args.height * args.width, args.block),)

    tap_scale = (torch.rand(9, C, device=dev) * 0.004 + 0.008)
    group_scale = (torch.rand(3, C, device=dev) * 0.004 + 0.008)

    def run():
        if args.band:
            conv3x3_int8_prelu_grouped_kernel[grid](x, weight, group_scale, slope, out_scale, y,
                                                    args.height, args.width, in_scale,
                                                    BLOCK_M=args.block, C=C, num_warps=args.warps, num_stages=args.stages)
        elif args.per_tap:
            conv3x3_int8_prelu_pertap_kernel[grid](x, weight, tap_scale, slope, out_scale, y,
                                                   args.height, args.width, in_scale,
                                                   BLOCK_M=args.block, C=C, num_warps=args.warps, num_stages=args.stages)
        else:
            conv3x3_int8_prelu_kernel[grid](x, weight, slope, out_scale, y, args.height, args.width,
                                            in_scale, w_scale, BLOCK_M=args.block, C=C,
                                            num_warps=args.warps, num_stages=args.stages)

    for _ in range(5):
        run()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(args.iters):
        run()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / args.iters
    flops = 2.0 * C * C * 9 * args.height * args.width
    print(f"block {args.height}x{args.width}: {ms:.3f} ms -> {flops / ms / 1e9:.1f} TOPS "
          f"(budget 1.9-2.0 ms, 76-80 TOPS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
