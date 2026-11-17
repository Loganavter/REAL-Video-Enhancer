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
import glob
import os
import sys

import numpy as np
import torch
import triton
import triton.language as tl

sys.path.insert(0, "/home/jorj/Загрузки/projects/REAL-Video-Enhancer/scripts")
from cut_channels_naive import load, statistic  # noqa: E402
from int8_prologue_kernel import body_prologue_kernel  # noqa: E402
from int8_prologue_kernel import dequantise_nhwc_to_nchw, dequantise_nhwc_to_nhwc  # noqa: E402
from int8_prologue_kernel import quantise_nhwc_fp16_to_int8  # noqa: E402
from int8_accuracy_sim import calibrate  # noqa: E402

BLOCK_M = 128
SHIFT = 20
MODEL = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/custom_models/2x-Futsuu-Anime(1).pth"


@triton.jit
def body_kernel(x_ptr, w_ptr, bias_ptr, in_scale_ptr, slope_ptr, out_scale_ptr, y_ptr, H, W,
                BLOCK_M: tl.constexpr, C: tl.constexpr):
    """int8 conv 3x3 -> int32 -> dequant + bias -> PReLU -> per-channel requant -> int8.

    The epilogue form: the input grid is folded into the weights at build time, so the accumulator
    carries the true value and one per-output-channel multiply finishes it. That is what removes the
    prologue pass (3.42 ms/hop) and lands the hop at the block probe's 1.91 ms.
    """
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
    v += tl.load(bias_ptr + lanes, mask=lanes < C, other=0.0)[None, :]
    slope = tl.load(slope_ptr + lanes)[None, :]
    v = tl.where(v >= 0, v, v * slope)
    out_scale = tl.load(out_scale_ptr + lanes)[None, :]
    q = tl.floor(v / out_scale + 0.5)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(y_ptr + rows[:, None] * C + lanes[None, :], q.to(tl.int8))


@triton.jit
def body_kernel_g3(x_ptr, w_ptr, s0_ptr, s1_ptr, s2_ptr, bias_ptr, slope_ptr, out_scale_ptr,
                   y_ptr, H, W, BLOCK_M: tl.constexpr, C: tl.constexpr):
    """int8 conv 3x3 with THREE int32 accumulators, one per kernel row, scales applied after the loop.

    The ledger recorded this as "the one untried variant": per-tap weights give 0.008% of pixels over
    the bar instead of 0.114%, but the kernel that produced them accumulated in float, which takes the
    dot out of int8 mode and cost 2x. Three int32 accumulators keep the MMA integer; the price is
    registers (3 x [BM, C] int32), so BLOCK_M has to come down.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    h = rows // W
    w = rows % W
    lanes = tl.arange(0, C)
    acc0 = tl.zeros((BLOCK_M, C), dtype=tl.int32)
    acc1 = tl.zeros((BLOCK_M, C), dtype=tl.int32)
    acc2 = tl.zeros((BLOCK_M, C), dtype=tl.int32)
    for tap in tl.static_range(3):
        dy = tap // 3 - 1
        dx = tap % 3 - 1
        yy = h + dy
        xx = w + dx
        inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        src = (yy * W + xx)[:, None] * C + lanes[None, :]
        a = tl.load(x_ptr + src, mask=inside[:, None], other=0)
        b = tl.load(w_ptr + tap * C * C + lanes[:, None] * C + lanes[None, :])
        acc0 += tl.dot(a, b, out_dtype=tl.int32)
    for tap in tl.static_range(3, 6):
        dy = tap // 3 - 1
        dx = tap % 3 - 1
        yy = h + dy
        xx = w + dx
        inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        src = (yy * W + xx)[:, None] * C + lanes[None, :]
        a = tl.load(x_ptr + src, mask=inside[:, None], other=0)
        b = tl.load(w_ptr + tap * C * C + lanes[:, None] * C + lanes[None, :])
        acc1 += tl.dot(a, b, out_dtype=tl.int32)
    for tap in tl.static_range(6, 9):
        dy = tap // 3 - 1
        dx = tap % 3 - 1
        yy = h + dy
        xx = w + dx
        inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        src = (yy * W + xx)[:, None] * C + lanes[None, :]
        a = tl.load(x_ptr + src, mask=inside[:, None], other=0)
        b = tl.load(w_ptr + tap * C * C + lanes[:, None] * C + lanes[None, :])
        acc2 += tl.dot(a, b, out_dtype=tl.int32)
    v = (acc0.to(tl.float32) * tl.load(s0_ptr + lanes)[None, :]
         + acc1.to(tl.float32) * tl.load(s1_ptr + lanes)[None, :]
         + acc2.to(tl.float32) * tl.load(s2_ptr + lanes)[None, :])
    v += tl.load(bias_ptr + lanes)[None, :]
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
    ap.add_argument("--in-margin", type=float, default=1.0,
                    help="extra margin on the INPUT (head) grid, which the output margin does not touch")
    ap.add_argument("--margin", type=float, default=1.0, help="safety factor on every grid")
    ap.add_argument("--dump", default=None, help="write the output image as fp16 raw")
    ap.add_argument("--emit", default=None, help="write the measurement as harness records (JSONL)")
    ap.add_argument("--per-tensor", action="store_true",
                    help="one scalar grid per layer, as the simulation does, instead of per channel")
    ap.add_argument("--recal", action="store_true",
                    help="the chain self-calibration; OFF by default because it is worse on both populations")
    ap.add_argument("--sim-grids", action="store_true",
                    help="prologue target from the simulation's per-tensor calibration")
    ap.add_argument("--tail-cl", action="store_true", default=True,
                    help="channels_last tail: dequant is a copy instead of a transpose")
    ap.add_argument("--no-tail-cl", dest="tail_cl", action="store_false")
    ap.add_argument("--calib-frames", default="",
                    help="glob of frames whose ranges are merged into the calibration grids "
                         "(split calibration is a correctness condition, not a tuning knob)")
    ap.add_argument("--ref-cl", action="store_true",
                    help="also make the REFERENCE run in channels_last, so the comparison isolates "
                         "quantisation from the difference between two fp16 conv algorithms")
    ap.add_argument("--diag", action="store_true",
                    help="where the bad pixels are: 2-pixel border vs interior, and a coarse map")
    ap.add_argument("--eval-kernel", default="epilogue",
                    choices=("prologue", "epilogue", "g3"))
    ap.add_argument("--eval-frames", default="",
                    help="glob of raw 1920x1080 frames for the accuracy sample (worst-of, not one frame)")
    ap.add_argument("--trt-ref-out", action="store_true",
                    help="use the TRT fp16 plan's OUTPUT as the accuracy reference (the app's own "
                         "reference), instead of the torch path")
    ap.add_argument("--trt-ref", default="",
                    help="fp16 TensorRT plan timed in the same process, for a same-session ratio")
    ap.add_argument("--bm", type=int, default=256, help="BLOCK_M for the body kernels (swept)")
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--g3-bm", type=int, default=128,
                    help="BLOCK_M for the three-accumulator hops (their own swept optimum)")
    ap.add_argument("--g3-layers", default="",
                    help="comma list of hop indices (1..16) that use the three-accumulator kernel; "
                         "the rest use the single-scale epilogue one")
    ap.add_argument("--sweep-kernel", default="epilogue", choices=("prologue", "epilogue", "g3"))
    ap.add_argument("--sweep", action="store_true",
                    help="time the 16-hop chain across BLOCK_M and num_warps")
    ap.add_argument("--compare", action="store_true",
                    help="run BOTH body kernels (prologue and epilogue) and report window+fps+accuracy")
    ap.add_argument("--compare-iters", type=int, default=5)
    ap.add_argument("--pct", type=float, default=99.9,
                    help="percentile for the grids; 100 means the maximum")
    args = ap.parse_args()

    raw = np.fromfile(args.frame, dtype=np.uint8)
    img = torch.from_numpy(raw.reshape(args.height, args.width, 3).astype(np.float32) / 255.0)
    x = img.permute(2, 0, 1)[None].cuda().half()

    model, scale_up, _ = load(MODEL)
    model = model.eval().cuda().half()
    if args.ref_cl:
        # the same weights through the other fp16 kernel family: if THIS reference is what the chain is
        # measured against, the residual is quantisation and not a difference between two conv
        # implementations.
        model = model.to(memory_format=torch.channels_last)
        x = x.contiguous(memory_format=torch.channels_last)
    with torch.no_grad():
        ref = model(x).float()

    # calibration: per-tensor input range and per-channel PReLU output range, as the simulation does.
    # It quantises the weights of whatever model it is given, so it runs on a copy - otherwise the head and
    # the tail below would be the quantised ones while `ref` came from the original.
    calib_files = sorted(glob.glob(args.calib_frames)) if args.calib_frames else []
    calib_xs = []
    for path in calib_files:
        raw_c = np.fromfile(path, dtype=np.uint8)
        img_c = torch.from_numpy(raw_c.reshape(args.height, args.width, 3).astype(np.float32) / 255.0)
        calib_xs.append(img_c.permute(2, 0, 1)[None].cuda().half())
    if calib_xs:
        print(f"  calibration: {len(calib_xs)} frames merged ({os.path.basename(args.calib_frames)})")
    calib_model = copy.deepcopy(model)
    inputs, outputs, _ = calibrate(calib_model, calib_xs or [x])
    inputs["body.0"] = (0.0, 1.0)

    body = [m for m in model.body if isinstance(m, torch.nn.Conv2d)]
    prelus = [m for m in model.body if isinstance(m, torch.nn.PReLU)]
    head_conv, tail_conv = body[0], body[-1]
    inner = body[1:-1]

    with torch.no_grad():
        # channels_last output: the quantise then reads and writes the same layout, with no 64x2 M-element
        # strided gather (which measured 16.19 ms for what is a 400 MB pass)
        act = torch.nn.functional.prelu(
            head_conv(x.clamp(0, 1).to(memory_format=torch.channels_last)), prelus[0].weight)

    # activations live on a PER-TENSOR grid so that scale * weight-scale folds into one per-channel
    # vector; a per-channel activation grid cannot be folded and would break the chain
    # grids from a percentile rather than the maximum: max|activation| over 2 Mpx is a tail event, and
    # the resulting step is what made the full frame fail while a 256x256 crop passed
    def frame_grids(xf):
        qq_f = {}
        hooks = []
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.PReLU):
                hooks.append(mod.register_forward_hook(
                    lambda _m, _i, out, k=name: qq_f.__setitem__(
                        k, (out.detach().float().abs().flatten(2).amax(dim=2) if args.pct >= 100
                            else out.detach().float().abs().flatten(2)[:, :, ::8].quantile(
                                args.pct / 100.0, dim=2)
                            ).cuda().flatten())))
        with torch.no_grad():
            model(xf)
        for h in hooks:
            h.remove()
        return qq_f

    qq = {}
    for xf in (calib_xs or [x]):
        for k, v in frame_grids(xf).items():
            qq[k] = v if k not in qq else torch.maximum(qq[k], v)
    hooks = []

    # each layer's output grid: per-tensor, from the simulation's calibration of the PReLU output
    # per-channel activation grid: the whole point of the fold below
    s_act_vec = (qq["body.1"] / 127.0).clamp_min(1e-8) * args.in_margin
    if args.per_tensor:
        s_act_vec = s_act_vec.max().expand_as(s_act_vec).contiguous()
    # ONE pass, not four: the torch round/clamp/cast then permute+contiguous measured 10.0 ms
    QB = 1024
    cur = torch.empty((1, args.height, args.width, 64), device="cuda", dtype=torch.int8)
    if not act.contiguous().is_contiguous(memory_format=torch.channels_last):
        act = act.contiguous(memory_format=torch.channels_last)
    quantise_nhwc_fp16_to_int8[(triton.cdiv(args.height * args.width, QB),)](
        act, s_act_vec.contiguous(), cur, args.height * args.width,
        BLOCK=QB, C=64, num_warps=4)

    # the full float weights: the input scale is folded into them per build, see run_pipeline
    # the conv bias travels with the layer: dropping it changes the network, not the rounding
    layers = [(conv.weight.detach().float(), conv.bias.detach().float(), prelu.weight.detach().float())
              for conv, prelu in zip(inner, prelus[1:])]

    outs = []
    for i in range(len(layers)):
        key = f"body.{2 * (i + 1) + 1}"
        lo_c, hi_c = outputs[key]
        # per CHANNEL, not per tensor: the PReLU slopes span -1.39..1.70, so the post-PReLU range differs
        # by factors across channels and a single scalar grid throws bits away on the narrow ones
        _o = (torch.maximum(hi_c.abs(), lo_c.abs()).float() / 127.0).clamp_min(1e-8) * args.margin
        outs.append(_o.max().expand_as(_o).contiguous() if args.per_tensor else _o)

    def run_pipeline(outs_used=None, collect=False, sat=False, kernel="prologue"):
        """Per-channel activation grids throughout, with each layer's input scale folded into its weights.

        The dot is integer, so a per-input-channel activation scale cannot be applied inside it without
        leaving int8 MMA. Folding `s_in[cin]` into the weights at build time keeps the dot integer AND the
        grid per-channel: acc * s_w[cout] = sum_cin x_q[cin] * (w*s_in)[cin,cout], which is the real value.
        """
        buf, s_in = cur.clone(), s_act_vec
        measured = []
        for idx, ((w_full, bias, slope), s_out) in enumerate(zip(layers, outs_used or outs), 1):
            # prologue requant: the buffer is int8 on a per-channel grid, the dot wants one per-tensor
            # grid, so the ratio goes in as an integer multiplier and the accumulator stays int32
            # the simulation's own per-tensor input grid (calibrated on the fp16 activation) instead of
            # max(per-channel grid) taken from the chain - the one structural difference left untested
            out = torch.empty_like(buf)
            grid = (triton.cdiv(args.height * args.width, BLOCK_M),)
            layer_g3 = kernel == "g3" or idx in g3_layers
            if not layer_g3 and kernel == "prologue":
                s_t = sim_t[idx - 1] if args.sim_grids else float(s_in.max())
                m_int = torch.round(s_in / s_t * (1 << SHIFT)).to(torch.int32).contiguous()
                w_q, w_scale = quantise_weights(w_full)
                tap_w = torch.stack([w_q[:, :, i, j].t()
                                     for i in range(3) for j in range(3)]).contiguous()
                vec = (w_scale * s_t).contiguous()
                ovec = s_out.contiguous()
                body_prologue_kernel[grid](buf, tap_w, m_int, SHIFT, vec, bias, slope, ovec, out,
                                          args.height, args.width, BLOCK_M=BLOCK_M, C=64, num_warps=4)
            else:
                # epilogue: the per-channel input grid is folded into the float weights BEFORE the
                # quantisation, so the integer dot already carries the real value and the only thing
                # left after the accumulator is one per-output-channel multiply plus the bias.
                w_scaled = w_full * s_in.view(-1, 1, 1, 1)
                w_q, w_scale = quantise_weights(w_scaled)
                tap_w = torch.stack([w_q[:, :, i, j].t()
                                     for i in range(3) for j in range(3)]).contiguous()
                body_kernel[grid](buf, tap_w, bias.contiguous(), w_scale.contiguous(),
                                  slope.contiguous(), s_out.contiguous(), out,
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

    g3_layers = set()
    if args.g3_layers:
        g3_layers = {int(v) for v in args.g3_layers.split(",") if v.strip()}

    def prepare(kernel="prologue"):
        """Every per-layer tensor, computed ONCE.

        Quantisation is a build step, not a frame step: the grids come from calibration and do not
        depend on the frame, yet the first version recomputed 16 x (quantise_weights + stack +
        prologue multiplier) inside the frame loop. That is the difference between the block probe's
        1.91 ms/hop and the 3.42 ms/hop the chain measured.
        """
        prep, s_in = [], s_act_vec
        for idx, ((w_full, bias, slope), s_out) in enumerate(zip(layers, outs), 1):
            layer_g3 = kernel == "g3" or idx in g3_layers
            if not layer_g3 and kernel == "prologue":
                s_t = sim_t[idx - 1] if args.sim_grids else float(s_in.max())
                m_int = torch.round(s_in / s_t * (1 << SHIFT)).to(torch.int32).contiguous()
                w_q, w_scale = quantise_weights(w_full)
                extra = (m_int, (w_scale * s_t).contiguous())
            elif not layer_g3:
                # fold the per-channel INPUT grid into the weights: scale dim 1, not dim 0. Scaling
                # dim 0 by a [Cin] vector is silently legal while Cin == Cout == 64, and it produced a
                # 99.99672% picture - the trap is the shape, not the arithmetic.
                w_q, w_scale = quantise_weights(w_full * s_in.view(1, -1, 1, 1))
                extra = (w_scale.contiguous(),)
            else:
                # g3: one weight scale per (output channel, kernel row). int32 accumulation stays
                # integer; the three scales are applied after the loop.
                w_scaled = w_full * s_in.view(1, -1, 1, 1)
                scales, taps = [], []
                for g in range(3):
                    wg = w_scaled[:, :, g, :]
                    sg = wg.abs().amax(dim=(1, 2)).clamp_min(1e-8) / 127.0
                    qg = torch.round(wg / sg.view(-1, 1, 1)).clamp(-127, 127).to(torch.int8)
                    scales.append(sg.contiguous())
                    for c in range(3):
                        taps.append(qg[:, :, c].t())
                extra = tuple(scales)
            if not layer_g3:
                tap_w = torch.stack([w_q[:, :, i, j].t()
                                     for i in range(3) for j in range(3)]).contiguous()
            else:
                tap_w = torch.stack(taps).contiguous()
            tag = "g3" if layer_g3 else ("prologue" if kernel == "prologue" else "epilogue")
            prep.append((tag, tap_w, extra, bias.contiguous(),
                         slope.contiguous(), s_out.contiguous()))
            s_in = s_out
        return prep

    def run_prepared(prep, kernel="prologue", bm=None, warps=4, sat=False):
        # `body_prologue_kernel` is a different kernel with different register pressure: at BLOCK_M=256
        # it builds and runs and returns the WRONG picture (23.79% of pixels), so its block size stays
        # at the 128 it was validated with. 256 is the swept optimum of the epilogue kernel only.
        if bm is None:
            bm = 64 if kernel == "g3" else (128 if kernel == "prologue" else args.bm)
        buf = cur
        grid = (triton.cdiv(args.height * args.width, bm),)
        for idx, (layer_kernel, tap_w, extra, bias_t, slope_t, ovec) in enumerate(prep, 1):
            out = torch.empty_like(buf)
            # the three-accumulator kernel spills at BLOCK_M 256 (3 x [BM, 64] int32 in registers),
            # so it keeps its own block size; mixing them at one BM costs 79.76 ms instead of 44
            layer_bm = args.g3_bm if layer_kernel == "g3" else bm
            grid = (triton.cdiv(args.height * args.width, layer_bm),)
            if layer_kernel == "prologue":
                m_int, vec = extra
                body_prologue_kernel[grid](buf, tap_w, m_int, SHIFT, vec, bias_t, slope_t, ovec, out,
                                           args.height, args.width, BLOCK_M=BLOCK_M, C=64,
                                           num_warps=4)
            elif layer_kernel == "epilogue":
                (w_scale,) = extra
                body_kernel[grid](buf, tap_w, bias_t, w_scale, slope_t, ovec, out,
                                  args.height, args.width, BLOCK_M=layer_bm, C=64, num_warps=warps)
            else:
                s0, s1, s2 = extra
                body_kernel_g3[grid](buf, tap_w, s0, s1, s2, bias_t, slope_t, ovec, out,
                                     args.height, args.width, BLOCK_M=layer_bm, C=64,
                                     num_warps=warps)
            if sat:
                print(f"    layer {idx:2d}: saturated "
                      f"{float((out.abs() >= 127).float().mean()) * 100:6.3f}%", flush=True)
            buf = out
        return buf, prep[-1][-1]

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

    sim_t = []
    for k in range(1, 17):
        lo_i, hi_i = inputs[f"body.{2 * k}"]
        sim_t.append(max(abs(lo_i), abs(hi_i)) / 127.0)

    # NO recalibration on the chain's own output. Fitting the grid to the values the clamp already
    # truncated makes the clamp a fixed point: the measured maximum IS the clamp, so the grid can never
    # grow and the clipping reproduces itself. The grids come from the calibration on the fp16
    # activations, which is what the simulation uses.
    if args.recal:
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
    _prep_cache = {}

    def frame(kernel="epilogue", sat=False):
        if kernel not in _prep_cache:
            _prep_cache[kernel] = prepare(kernel)
        # the swept BLOCK_M belongs to the epilogue kernel; the prologue kernel is only correct at 128
        out_q, s_last = run_prepared(_prep_cache[kernel], kernel,
                                     bm=None if kernel == "prologue" else args.bm,
                                     warps=args.warps, sat=sat)
        with torch.no_grad():
            if args.tail_cl:
                last_cl = torch.empty((1, args.height, args.width, 64), device="cuda",
                                      dtype=torch.float16)
                dequantise_nhwc_to_nhwc[(triton.cdiv(args.height * args.width, 1024),)](
                    out_q, s_last.contiguous(), last_cl, args.height * args.width,
                    BLOCK=1024, C=64, num_warps=4)
                last = last_cl.permute(0, 3, 1, 2)          # a view, channels_last layout
            else:
                last = torch.empty((1, 64, args.height, args.width), device="cuda",
                                   dtype=torch.float16)
                dequantise_nhwc_to_nchw[(triton.cdiv(args.height * args.width, 1024),)](
                    out_q, s_last.contiguous(), last, args.height * args.width,
                    BLOCK=1024, C=64, num_warps=4)
            up = model.upsampler(tail_conv(last))
            return up + torch.nn.functional.interpolate(x, scale_factor=scale_up, mode="nearest")

    got = frame(sat=args.sat).float()  # the shipping form: epilogue, BLOCK_M from the sweep

    if args.dump:
        (got[0].permute(1, 2, 0).contiguous().cpu().numpy().astype(np.float16)
         .tofile(args.dump))
        print(f"dumped {got.shape[2]}x{got.shape[3]} to {args.dump}")
    share, peak, median = statistic(ref, got)
    print(f"pipeline: >8 {share:.5f}% (bar 0.1)  peak {peak:.0f}  median {median:.2f} levels")

    if args.emit:
        import hashlib as _h
        sys.path.insert(0, "scripts")
        from harness import Population, Record, Store
        fsha = _h.sha256(open(args.frame, "rb").read()).hexdigest()[:16]
        store = Store()
        pop = Population.of("pixels of one frame", [fsha],
                            label=f"{os.path.basename(args.frame)} {args.width}x{args.height}, "
                                  f"int8 chain vs fp16 model, {MODEL.split('/')[-1]}")
        common = dict(population=pop,
                      instrument="int8_pipeline.py: 16 Triton launches, statistic() worst-channel levels",
                      config=f"per-tensor input grids, per-channel output grids, {args.width}x{args.height}",
                      session=store.session, run="run-int8-pipeline",
                      extra={"artifact_sha256": fsha, "of": "share", "quantity": "accuracy"})
        store.emit(Record(value=round(float(share), 5), unit="%", statistic="single", **common))
        store.emit(Record(value=round(float(peak), 1), unit="levels", statistic="max",
                          **{**common, "extra": {**common["extra"], "of": "peak"}}))
        store.dump(args.emit)
        print(f"records -> {args.emit}")
    print(f"          16 kernel launches")

    if args.sweep:
        import time as _time
        prep = prepare(args.sweep_kernel)
        print(f"  kernel={args.sweep_kernel}  {'BM':>4s} {'warps':>5s} {'chain ms':>9s} {'ms/hop':>7s}")
        best = None
        for bm in ((32, 64, 128) if args.sweep_kernel == "g3" else (64, 128, 256, 512)):
            for warps in (4, 8):
                try:
                    for _ in range(2):
                        run_prepared(prep, args.sweep_kernel, bm=bm, warps=warps)
                    torch.cuda.synchronize()
                    t0 = _time.time()
                    for _ in range(4):
                        run_prepared(prep, args.sweep_kernel, bm=bm, warps=warps)
                    torch.cuda.synchronize()
                    ms = (_time.time() - t0) / 4 * 1000
                except Exception as exc:                       # noqa: BLE001
                    print(f"  {bm:4d} {warps:5d}  failed: {str(exc)[:60]}")
                    continue
                print(f"  {bm:4d} {warps:5d} {ms:9.2f} {ms / 16:7.2f}")
                if best is None or ms < best[0]:
                    best = (ms, bm, warps)
        if best:
            print(f"  BEST: BM {best[1]} warps {best[2]} -> chain {best[0]:.2f} ms "
                  f"({best[0] / 16:.2f} ms/hop)")
        return 0

    if args.compare:
        import time as _time
        import time as _t0
        # the SAME-SESSION fp16 reference: the whole model forward, which is the same population as
        # frame() (head + body + tail + upsample + residual) without the int8 chain
        def fp16_frame():
            with torch.no_grad():
                return model(x)
        for _ in range(2):
            fp16_frame()
        torch.cuda.synchronize()
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ev0.record()
        for _ in range(args.compare_iters):
            fp16_frame()
        ev1.record()
        torch.cuda.synchronize()
        fp16_ms = ev0.elapsed_time(ev1) / args.compare_iters
        print(f"  {'fp16':9s} window {fp16_ms:7.2f} ms  {1000.0 / fp16_ms:5.2f} fps  (reference)")

        # the certified baseline: the SAME process times the fp16 TensorRT plan with the same events,
        # so the ratio is same-session rather than same-day
        trt_ms = None
        if args.trt_ref:
            import tensorrt as trt
            eng = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(
                open(args.trt_ref, "rb").read())
            ctx = eng.create_execution_context()
            bufs = []
            for i in range(eng.num_io_tensors):
                n = eng.get_tensor_name(i)
                dt = eng.get_tensor_dtype(n)
                tdt = {trt.DataType.HALF: torch.float16, trt.DataType.FLOAT: torch.float32,
                       trt.DataType.INT8: torch.int8}.get(dt, torch.float32)
                t = torch.zeros(tuple(eng.get_tensor_shape(n)), dtype=tdt, device="cuda")
                ctx.set_tensor_address(n, t.data_ptr())
                bufs.append(t)
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                for _ in range(6):
                    ctx.execute_async_v3(stream.cuda_stream)
            torch.cuda.synchronize()
            ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
            with torch.cuda.stream(stream):
                ev0.record()
                for _ in range(args.compare_iters):
                    ctx.execute_async_v3(stream.cuda_stream)
                ev1.record()
            torch.cuda.synchronize()
            trt_ms = ev0.elapsed_time(ev1) / args.compare_iters
            print(f"  {'trt fp16':9s} window {trt_ms:7.2f} ms  {1000.0 / trt_ms:5.2f} fps  "
                  f"(same process, same events: {os.path.basename(args.trt_ref)})")

        rows = []
        for kernel in ("prologue", "epilogue"):
            g = frame(kernel).float()
            sh, pk, md = statistic(ref, g)
            for _ in range(2):
                frame(kernel)
            torch.cuda.synchronize()
            ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
            ev0.record()
            for _ in range(args.compare_iters):
                frame(kernel)
            ev1.record()
            torch.cuda.synchronize()
            ms = ev0.elapsed_time(ev1) / args.compare_iters
            rows.append((kernel, ms, sh, pk))
            print(f"  {kernel:9s} window {ms:7.2f} ms  {1000.0 / ms:5.2f} fps  "
                  f">8 {sh:.5f}%  peak {pk:.0f}")
        # where the window goes: head vs chain vs tail, so the next decision is aimed
        import time as _t
        def stage(fn, n=4):
            for _ in range(2):
                fn()
            torch.cuda.synchronize()
            t0 = _t.time()
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
            return (_t.time() - t0) / n * 1000
        prep = prepare("epilogue")
        probe_out = torch.empty_like(cur)
        head_ms = stage(lambda: quantise_nhwc_fp16_to_int8[(triton.cdiv(args.height * args.width, 1024),)](
            act, s_act_vec.contiguous(), probe_out, args.height * args.width,
            BLOCK=1024, C=64, num_warps=4))
        chain_ms = stage(lambda: run_prepared(prep, "epilogue", bm=args.bm, warps=args.warps))
        frame_ms = stage(lambda: frame("epilogue"), 4)
        q_probe = torch.empty((1, args.height, args.width, 64), device="cuda", dtype=torch.float16)
        deq_ms = stage(lambda: dequantise_nhwc_to_nhwc[(triton.cdiv(args.height * args.width, 1024),)](
            cur, s_act_vec.contiguous(), q_probe, args.height * args.width,
            BLOCK=1024, C=64, num_warps=4))
        def tailconv_only():
            with torch.no_grad():
                return model.upsampler(tail_conv(q_probe.permute(0, 3, 1, 2)))
        tc_ms = stage(tailconv_only, 4)
        print(f"  tail split: dequant {deq_ms:.2f}  tail conv+pixelshuffle {tc_ms:.2f}")
        print(f"  stages (epilogue): head-quantise {head_ms:.2f}  chain {chain_ms:.2f}  "
              f"tail+residual {frame_ms - chain_ms:.2f}  whole frame {frame_ms:.2f} ms")
        best = min(rows, key=lambda r: r[1])
        print(f"  best: {best[0]} {best[1]:.2f} ms = {1000.0 / best[1]:.2f} fps  "
              f"({fp16_ms / best[1]:.3f}x the same-session fp16; target 43.48 ms / 23 fps)")
        if args.emit and trt_ms is not None:
            from harness import Population, Record, Store
            st = Store()
            pop = Population.of("frames", [f"frame-{i + 1}" for i in range(args.compare_iters)],
                                label=f"whole model window at {args.width}x{args.height}, "
                                      f"no gate, no encoder")
            st.emit(Record(value=round(trt_ms, 2), unit="ms", statistic="mean", population=pop,
                           instrument="torch.cuda.Event around plan executions, same process",
                           config=f"TRT fp16 plan {os.path.basename(args.trt_ref)}, "
                                  f"{args.width}x{args.height}",
                           session=st.session, run="window-trt-fp16",
                           artifact=__import__("hashlib").sha256(
                               open(args.trt_ref, "rb").read()).hexdigest(),
                           extra={"of": "frame_time", "quantity": "speed",
                                  "artifact_sha256": __import__("hashlib").sha256(
                                      open(args.trt_ref, "rb").read()).hexdigest(),
                                  "fps": round(1000.0 / trt_ms, 2)}))
            st.dump(args.emit + ".trt")
            print(f"records -> {args.emit}.trt")
        if args.emit:
            from harness import Population, Record, Store
            st = Store()
            pop = Population.of("frames", [f"frame-{i + 1}" for i in range(args.compare_iters)],
                                label=f"whole model window at {args.width}x{args.height}, "
                                      f"no gate, no encoder")
            st.emit(Record(value=round(fp16_ms, 2), unit="ms", statistic="mean", population=pop,
                           instrument="torch.cuda.Event around whole-frame calls after 2 warmups "
                                      "(same instrument as bench_plan.py)",
                           config=f"fp16 model forward, {args.width}x{args.height}",
                           session=st.session, run="window-fp16",
                           extra={"of": "frame_time", "quantity": "speed",
                                  "fps": round(1000.0 / fp16_ms, 2)}))
            st.dump(args.emit + ".fp16")
            print(f"records -> {args.emit}.fp16")
        if args.emit:
            from harness import Population, Record, Store
            st = Store()
            pop = Population.of("frames", [f"frame-{i + 1}" for i in range(args.compare_iters)],
                                label=f"whole model window at {args.width}x{args.height}, "
                                      f"int8 body, no gate, no encoder")
            for kernel, ms, sh, pk in rows:
                st.emit(Record(value=round(ms, 2), unit="ms", statistic="mean", population=pop,
                               instrument="torch.cuda.Event around whole-frame calls after 2 warmups "
                                          "(same instrument as bench_plan.py)",
                               config=f"kernel={kernel}, per-tensor in / per-channel out, "
                                      f"{args.width}x{args.height}, torch head+tail fp16",
                               session=st.session, run=f"int8-window-{kernel}",
                               extra={"of": "frame_time", "quantity": "speed",
                                      "fps": round(1000.0 / ms, 2),
                                      "share_above_8": round(float(sh), 5),
                                      "peak_levels": round(float(pk), 1)}))
            st.dump(args.emit + ".window")
            print(f"records -> {args.emit}.window")

    trt_engine = None
    if args.trt_ref_out:
        if not args.trt_ref:
            raise SystemExit("--trt-ref-out needs --trt-ref <plan>")
        import tensorrt as _trt
        trt_engine = {
            "eng": _trt.Runtime(_trt.Logger(_trt.Logger.ERROR)).deserialize_cuda_engine(
                open(args.trt_ref, "rb").read())}
        trt_engine["ctx"] = trt_engine["eng"].create_execution_context()
        trt_engine["bufs"] = {}
        for i in range(trt_engine["eng"].num_io_tensors):
            n = trt_engine["eng"].get_tensor_name(i)
            dt = trt_engine["eng"].get_tensor_dtype(n)
            tdt = {_trt.DataType.HALF: torch.float16, _trt.DataType.FLOAT: torch.float32}.get(
                dt, torch.float32)
            t = torch.zeros(tuple(trt_engine["eng"].get_tensor_shape(n)), dtype=tdt, device="cuda")
            trt_engine["ctx"].set_tensor_address(n, t.data_ptr())
            trt_engine["bufs"][n] = t
        trt_engine["in_name"] = next(n for n in trt_engine["bufs"]
                                     if trt_engine["eng"].get_tensor_mode(n)
                                     == _trt.TensorIOMode.INPUT)
        trt_engine["out_name"] = next(n for n in trt_engine["bufs"]
                                      if trt_engine["eng"].get_tensor_mode(n)
                                      == _trt.TensorIOMode.OUTPUT)
        print(f"  accuracy reference: the fp16 TRT plan's output ({os.path.basename(args.trt_ref)})")

    if args.eval_frames:
        import glob as _glob
        files = sorted(_glob.glob(args.eval_frames))
        print(f"  accuracy sample over {len(files)} frame(s), kernel={args.eval_kernel}:")
        worst_share, worst_peak, rows_acc = 0.0, 0.0, []
        for path in files:
            raw_f = np.fromfile(path, dtype=np.uint8)
            img_f = torch.from_numpy(raw_f.reshape(args.height, args.width, 3).astype(np.float32)
                                     / 255.0)
            x = img_f.permute(2, 0, 1)[None].cuda().half()
            with torch.no_grad():
                act_f = torch.nn.functional.prelu(
                    head_conv(x.clamp(0, 1).to(memory_format=torch.channels_last)),
                    prelus[0].weight)
                if not act_f.is_contiguous(memory_format=torch.channels_last):
                    act_f = act_f.contiguous(memory_format=torch.channels_last)
                if trt_engine is not None:
                    trt_engine["bufs"][trt_engine["in_name"]].copy_(x)
                    trt_engine["ctx"].execute_async_v3(torch.cuda.current_stream().cuda_stream)
                    torch.cuda.synchronize()
                    ref_f = trt_engine["bufs"][trt_engine["out_name"]].float()
                else:
                    ref_f = model(x).float()
            quantise_nhwc_fp16_to_int8[(triton.cdiv(args.height * args.width, QB),)](
                act_f, s_act_vec.contiguous(), cur, args.height * args.width,
                BLOCK=QB, C=64, num_warps=4)
            g = frame(args.eval_kernel).float()
            sh_f, pk_f, md_f = statistic(ref_f, g)
            rows_acc.append((os.path.basename(path), float(sh_f), float(pk_f), float(md_f)))
            if args.diag:
                diff = (got - ref_f).abs().amax(dim=1)[0]
                bad = diff > 8.0
                H2, W2 = bad.shape
                border = torch.zeros_like(bad)
                border[:2, :] = border[-2:, :] = True
                border[:, :2] = border[:, -2:] = True
                nb, ni = int(bad.sum()), int((bad & ~border).sum())
                print(f"      bad px {nb}  border {nb - ni}  interior {ni}  "
                      f"| border share of all px {float(border.float().mean()) * 100:.2f}%", flush=True)
                # 8x8 coarse map of where the interior failures sit
                gh, gw = H2 // 8, W2 // 8
                gridm = bad[:gh * 8, :gw * 8].view(gh, 8, gw, 8).sum(dim=(1, 3)).float()
                tot = gridm.sum().clamp_min(1)
                top = torch.topk(gridm.flatten(), min(5, gridm.numel()))
                for v, idx in zip(top.values.tolist(), top.indices.tolist()):
                    if v <= 0:
                        continue
                    print(f"      cell row {idx // gw:3d} col {idx % gw:3d}: {v:7.0f} px "
                          f"({v / float(tot) * 100:4.0f}% of bad)", flush=True)
            worst_share = max(worst_share, float(sh_f))
            worst_peak = max(worst_peak, float(pk_f))
            print(f"    {os.path.basename(path):16s} >8 {float(sh_f):.5f}%  peak {float(pk_f):.0f}  "
                  f"median {float(md_f):.2f}")
        print(f"  WORST of {len(files)}: >8 {worst_share:.5f}% (bar 0.1)  peak {worst_peak:.0f} "
              f"(proposed bar 20)")
        if args.emit:
            from harness import Population, Record, Store
            st = Store()
            pop = Population.of("frames", [os.path.basename(p) for p in files],
                                label=f"{len(files)} frames, {args.width}x{args.height}, "
                                      f"int8 epilogue chain vs fp16, calibration on frame.raw")
            st.emit(Record(value=round(worst_share, 5), unit="%", statistic="worst_of",
                           population=pop,
                           instrument="int8_pipeline.py statistic(): worst channel >8 levels",
                           config=f"epilogue chain, BM {args.bm}, channels_last head/tail, "
                                  f"{args.width}x{args.height}",
                           session=st.session, run="int8-window-accuracy",
                           extra={"of": "share", "quantity": "accuracy",
                                  "peak_levels": round(worst_peak, 1),
                                  "per_frame": rows_acc}))
            st.emit(Record(value=round(worst_peak, 1), unit="levels", statistic="max",
                           population=pop,
                           instrument="int8_pipeline.py statistic(): peak level difference",
                           config=f"epilogue chain, {args.width}x{args.height}",
                           session=st.session, run="int8-window-accuracy",
                           extra={"of": "peak", "quantity": "accuracy"}))
            st.dump(args.emit + ".acc")
            print(f"records -> {args.emit}.acc")


    if args.iters:
        import time
        for _ in range(3):
            run_pipeline()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.iters):
            run_pipeline()
        torch.cuda.synchronize()
        chain_ms = (time.time() - t0) / args.iters * 1000
        print(f"          body chain: {chain_ms:.2f} ms "
              f"({chain_ms / 16:.2f} ms per layer)")
        if args.emit:
            import sys as _sys
            _sys.path.insert(0, "scripts")
            from harness import Population, Record, Store
            st = Store()
            pop2 = Population.of("chain launches", [f"launch-{i+1}" for i in range(16)],
                                 label=f"16 int8 hops at {args.width}x{args.height}")
            st.emit(Record(value=round(chain_ms, 2), unit="ms", statistic="mean",
                           population=pop2,
                           instrument="wall clock around 16 launches after 3 warmups",
                           config=f"per-tensor in, per-channel out, {args.width}x{args.height}",
                           session=st.session, run="run-int8-pipeline-time",
                           extra={"of": "engine_time", "quantity": "speed"}))
            st.dump(args.emit + ".time")
            print(f"records -> {args.emit}.time")
    return 0


if __name__ == "__main__":
    sys.exit(main())
