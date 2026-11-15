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
from int8_prologue_kernel import body_prologue_kernel  # noqa: E402
from int8_accuracy_sim import calibrate  # noqa: E402

BLOCK_M = 128
SHIFT = 20
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
    ap.add_argument("--sat", action="store_true", help="per-layer saturation share")
    ap.add_argument("--margin", type=float, default=1.0, help="safety factor on every grid")
    ap.add_argument("--pct", type=float, default=99.9,
                    help="percentile for the grids; 100 means the maximum")
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
    # grids from a percentile rather than the maximum: max|activation| over 2 Mpx is a tail event, and
    # the resulting step is what made the full frame fail while a 256x256 crop passed
    qq = {}
    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.PReLU):
            hooks.append(mod.register_forward_hook(
                lambda _m, _i, out, k=name: qq.__setitem__(
                    k, out.detach().float().abs().flatten(2)[:, :, ::16].quantile(args.pct / 100.0, dim=2).cuda().flatten())))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()

    # each layer's output grid: per-tensor, from the simulation's calibration of the PReLU output
    # per-channel activation grid: the whole point of the fold below
    s_act_vec = (qq["body.1"] / 127.0).clamp_min(1e-8)
    cur = torch.round(act / s_act_vec.view(1, -1, 1, 1)).clamp(-127, 127).to(torch.int8)
    cur = cur.permute(0, 2, 3, 1).contiguous()

    # the full float weights: the input scale is folded into them per build, see run_pipeline
    layers = [(conv.weight.detach().float(), prelu.weight.detach().float())
              for conv, prelu in zip(inner, prelus[1:])]

    outs = []
    for i in range(len(layers)):
        key = f"body.{2 * (i + 1) + 1}"
        lo_c, hi_c = outputs[key]
        # per CHANNEL, not per tensor: the PReLU slopes span -1.39..1.70, so the post-PReLU range differs
        # by factors across channels and a single scalar grid throws bits away on the narrow ones
        outs.append((torch.maximum(hi_c.abs(), lo_c.abs()).float() / 127.0).clamp_min(1e-8))

    def run_pipeline(outs_used=None, collect=False, sat=False):
        """Per-channel activation grids throughout, with each layer's input scale folded into its weights.

        The dot is integer, so a per-input-channel activation scale cannot be applied inside it without
        leaving int8 MMA. Folding `s_in[cin]` into the weights at build time keeps the dot integer AND the
        grid per-channel: acc * s_w[cout] = sum_cin x_q[cin] * (w*s_in)[cin,cout], which is the real value.
        """
        buf, s_in = cur.clone(), s_act_vec
        measured = []
        for idx, ((w_full, slope), s_out) in enumerate(zip(layers, outs_used or outs), 1):
            # prologue requant: the buffer is int8 on a per-channel grid, the dot wants one per-tensor
            # grid, so the ratio goes in as an integer multiplier and the accumulator stays int32
            s_t = float(s_in.max())
            m_int = torch.round(s_in / s_t * (1 << SHIFT)).to(torch.int32).contiguous()
            w_q, w_scale = quantise_weights(w_full)
            tap_w = torch.stack([w_q[:, :, i, j].t() for i in range(3) for j in range(3)]).contiguous()
            vec = (w_scale * s_t).contiguous()
            ovec = s_out.contiguous()
            out = torch.empty_like(buf)
            grid = (triton.cdiv(args.height * args.width, BLOCK_M),)
            body_prologue_kernel[grid](buf, tap_w, m_int, SHIFT, vec, slope, ovec, out,
                                       args.height, args.width, BLOCK_M=BLOCK_M, C=64, num_warps=4)
            if sat:
                print(f"    layer {idx:2d}: saturated "
                      f"{float((out.abs() >= 127).float().mean()) * 100:5.2f}%", flush=True)
            if collect:
                key = f"body.{2 * (idx) + 1}"
                chain_mag = float((out.permute(0, 3, 1, 2).float() * s_out.view(1, -1, 1, 1)).abs().mean())
                ref_mag = fp16_mag.get(key, float('nan'))
                print(f"    layer {idx:2d}: |chain| {chain_mag:7.3f}  |fp16| {ref_mag:7.3f}  "
                      f"ratio {chain_mag / ref_mag:7.3f}", flush=True)
                # the chain's OWN per-channel maximum, which is what the next grid has to cover
                measured.append((out.permute(0, 3, 1, 2).float() * s_out.view(1, -1, 1, 1))
                                .amax(dim=(0, 2, 3)).abs())
            buf, s_in = out, s_out
        return (buf, s_out, measured) if collect else (buf, s_out)

    fp16_mag = {}
    hh = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.PReLU):
            hh.append(mod.register_forward_hook(
                lambda _m, _i, out, k=name: fp16_mag.__setitem__(k, float(out.float().abs().mean()))))
    with torch.no_grad():
        model(x)
    for h in hh:
        h.remove()

    # iterate: grids from the fp16 calibration first, then from the chain's own maxima, until the
    # per-layer maximum stops moving (an fp16 grid is fitted to a distribution the chain never sees)
    _, _, measured = run_pipeline(collect=True)
    for step in range(3):
        outs = [(m.float() / 127.0).clamp_min(1e-8) * args.margin for m in measured]
        _, _, new_measured = run_pipeline(outs_used=outs, collect=True)
        drift = max(float((a - b).abs().max() / b.abs().max().clamp_min(1e-9))
                    for a, b in zip(new_measured, measured))
        print(f"  calibration pass {step + 1}: max grid drift {drift * 100:.2f}%")
        measured = new_measured
        if drift < 0.01:
            break
    outs = [(m.float() / 127.0).clamp_min(1e-8) for m in measured]
    out_q, s_last = run_pipeline(outs_used=outs, sat=args.sat)

    with torch.no_grad():
        last = out_q.permute(0, 3, 1, 2).float() * s_last.view(1, -1, 1, 1)
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
