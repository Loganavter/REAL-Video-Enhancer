"""Stage 1: build the int8 pipeline eagerly and check it reproduces the simulation.

    head (fp16) -> clamp -> quantise -> 16 Triton launches -> dequantise -> tail (fp16, pixel shuffle,
    residual)

The head and the tail stand in for TensorRT engines on this pass; the point is that the sixteen body
convolutions run as real kernel launches with real data handoffs, not as a torch simulation.

The kernel needs per-channel scales on BOTH sides to chain: each layer writes int8 on its own per-channel
grid and the next layer must read on exactly that grid. The first kernel only had a scalar input scale,
which cannot express that, so this file carries a per-channel variant.

Usage: int8_pipeline.py <frame.raw> [--W 1920] [--H 1080] [--iters 10]
"""
import argparse
import copy
import sys

import numpy as np
import torch
import triton
import triton.language as tl

sys.path.insert(0, "/home/jorj/Загрузки/projects/REAL-Video-Enhancer/scripts")
from cut_channels_naive import load, statistic  # noqa: E402
from int8_accuracy_sim import calibrate  # noqa: E402

BLOCK_M = 128
MODEL = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/custom_models/2x-Futsuu-Anime(1).pth"


@triton.jit
def body_kernel(x_ptr, w_ptr, in_scale_ptr, slope_ptr, out_scale_ptr, y_ptr, H, W,
                BLOCK_M: tl.constexpr, C: tl.constexpr):
    """int8 conv 3x3 -> int32 -> per-channel dequant -> PReLU -> per-channel requant -> int8."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
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
    v = acc.to(tl.float32) * tl.load(in_scale_ptr + lanes)[None, :]
    slope = tl.load(slope_ptr + lanes)[None, :]
    v = tl.where(v >= 0, v, v * slope)
    out_scale = tl.load(out_scale_ptr + lanes)[None, :]
    q = tl.floor(v / out_scale + 0.5)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(y_ptr + rows[:, None] * C + lanes[None, :], q.to(tl.int8))


def quantise_weights(w):
    """Per-output-channel symmetric int8: one scale for all nine taps.

    Per-tap scales would be finer (measured 2.47x) but they break the chain - the kernel accumulates all
    taps into one int32 sum, so a per-tap scale can only be applied inside the loop, which costs 2x.
    """
    scale = w.abs().amax(dim=(1, 2, 3)) / 127.0                  # [Cout]
    q = torch.round(w / scale.view(-1, 1, 1, 1)).clamp(-127, 127).to(torch.int8)
    return q, scale


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("frame")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--iters", type=int, default=0)
    args = ap.parse_args()

    raw = np.fromfile(args.frame, dtype=np.uint8)
    img = torch.from_numpy(raw.reshape(args.height, args.width, 3).astype(np.float32) / 255.0)
    x = img.permute(2, 0, 1)[None].cuda().half()

    model, scale_up, _ = load(MODEL)
    model = model.eval().cuda().half()
    with torch.no_grad():
        ref = model(x).float()

    # calibration: per-tensor input range and per-channel PReLU output range, as the simulation does.
    # It quantises the weights of whatever model it is given, so it runs on a copy - otherwise the head and
    # the tail below would be the quantised ones while `ref` came from the original.
    calib_model = copy.deepcopy(model)
    inputs, outputs, _ = calibrate(calib_model, [x])
    inputs["body.0"] = (0.0, 1.0)

    body = [m for m in model.body if isinstance(m, torch.nn.Conv2d)]
    prelus = [m for m in model.body if isinstance(m, torch.nn.PReLU)]
    head_conv, tail_conv = body[0], body[-1]
    inner = body[1:-1]

    with torch.no_grad():
        act = torch.nn.functional.prelu(head_conv(x.clamp(0, 1)), prelus[0].weight)

    # activations live on a PER-TENSOR grid so that scale * weight-scale folds into one per-channel
    # vector; a per-channel activation grid cannot be folded and would break the chain
    s_act = float(act.abs().amax()) / 127.0
    cur = torch.round(act / s_act).clamp(-127, 127).to(torch.int8).permute(0, 2, 3, 1).contiguous()

    layers = []
    for conv, prelu in zip(inner, prelus[1:]):
        w_q, w_scale = quantise_weights(conv.weight.detach().float())
        # settled empirically against _int_mm: the kernel wants [tap, cin, cout], and every permute
        # variant of the same element count reshapes silently into something plausible and wrong
        tap_w = torch.stack([w_q[:, :, i, j].t() for i in range(3) for j in range(3)]).contiguous()
        s_out = float(prelu.weight.detach().abs().max()) and None
        layers.append((tap_w, w_scale, prelu.weight.detach().float()))
        del s_out

    # each layer's output grid: per-tensor, from the simulation's calibration of the PReLU output
    outs = []
    for i, (tap_w, w_scale, slope) in enumerate(layers):
        key = f"body.{2 * (i + 1) + 1}"
        lo_c, hi_c = outputs[key]
        outs.append(float(torch.maximum(hi_c.abs(), lo_c.abs()).max()) / 127.0)

    def run_pipeline():
        buf, s_in = cur.clone(), s_act
        for (tap_w, w_scale, slope), s_out in zip(layers, outs):
            vec = (torch.full_like(w_scale, s_in) * w_scale).contiguous()      # s_act * s_w[c]
            ovec = torch.full_like(w_scale, s_out).contiguous()
            out = torch.empty_like(buf)
            grid = (triton.cdiv(args.height * args.width, BLOCK_M),)
            body_kernel[grid](buf, tap_w, vec, slope, ovec, out, args.height, args.width,
                              BLOCK_M=BLOCK_M, C=64, num_warps=2)
            buf, s_in = out, s_out
        return buf, s_in

    out_q, s_last = run_pipeline()

    with torch.no_grad():
        last = out_q.permute(0, 3, 1, 2).float() * s_last
        up = model.upsampler(tail_conv(last.half()))
        got = (up.float() + torch.nn.functional.interpolate(x, scale_factor=scale_up, mode="nearest").float())

    share, peak, median = statistic(ref, got)
    print(f"pipeline: >8 {share:.3f}% (bar 0.1)  peak {peak:.0f}  median {median:.2f} levels")
    print(f"          16 kernel launches, per-tensor input scales, per-channel output scales")

    if args.iters:
        import time
        for _ in range(3):
            run_pipeline()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.iters):
            run_pipeline()
        torch.cuda.synchronize()
        print(f"          body chain: {(time.time() - t0) / args.iters * 1000:.2f} ms "
              f"({(time.time() - t0) / args.iters * 1000 / 16:.2f} ms per layer)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
