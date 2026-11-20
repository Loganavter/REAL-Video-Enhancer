"""Stage 2: segmented timing of the int8 body against the fp16 baseline, in ONE session.

The plan asks for six separate segments (head engine, quantise, 16 launches, dequantise, tail engine, full
cycle), an fp16 baseline measured in the same session, at least three interleaved rounds, and clocks in the
log. Everything a kernel needs - grids, requant multipliers, transposed weight taps - is built BEFORE the
clock starts, so the numbers are launches, not host overhead.

What this does NOT cover: NVENC alongside. That needs the app (only it has the encoder), and it is the next
step, not this one.

Usage: int8_timing.py [frame.raw] [--width W] [--height H] [--rounds N] [--model M]
"""
import argparse
import subprocess
import sys
import time

import numpy as np
import torch
import triton

import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)
from cut_channels_naive import load  # noqa: E402
from int8_accuracy_sim import calibrate  # noqa: E402
from int8_pipeline import BLOCK_M, SHIFT, quantise_weights  # noqa: E402
from int8_prologue_kernel import body_prologue_kernel  # noqa: E402


def clocks():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip()
    except Exception as exc:  # noqa: BLE001 - the log must not depend on nvidia-smi being there
        return f"unavailable ({exc})"


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


def main() -> int:
    torch.backends.cudnn.benchmark = True   # fixed shapes: let cudnn pick, not guess
    ap = argparse.ArgumentParser()
    ap.add_argument("frame", nargs="?", default="/var/tmp/opencode/frame.raw")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--model", default="/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2"
                                       "/custom_models/2x-Futsuu-Anime(1).pth")
    args = ap.parse_args()

    raw = np.fromfile(args.frame, dtype=np.uint8).reshape(args.height, args.width, 3)
    x = torch.from_numpy(raw.astype(np.float32) / 255.0).permute(2, 0, 1)[None].cuda().half()

    model, scale_up, _ = load(args.model)
    model = model.eval().cuda().half()
    import copy
    calibration = copy.deepcopy(model)
    inputs, outputs, _ = calibrate(calibration, [x])
    inputs["body.0"] = (0.0, 1.0)

    convs = [m for m in model.body if isinstance(m, torch.nn.Conv2d)]
    prels = [m for m in model.body if isinstance(m, torch.nn.PReLU)]
    head_conv, tail_conv = convs[0], convs[-1]
    inner = convs[1:-1]

    # grids from the calibration, per channel, exactly as the passing build uses them
    s_act = (torch.maximum(*[t.abs() for t in (outputs["body.1"][1], outputs["body.1"][0])])
             / 127.0).clamp_min(1e-8)
    outs = []
    for i in range(16):
        lo_c, hi_c = outputs[f"body.{2 * (i + 1) + 1}"]
        outs.append((torch.maximum(hi_c.abs(), lo_c.abs()).float() / 127.0).clamp_min(1e-8))

    # ---- everything a launch needs, built once, outside the clock
    launches = []
    s_in = s_act
    for idx in range(1, 17):
        s_out = outs[idx - 1]
        s_t = float(s_in.max())
        m_int = torch.round(s_in / s_t * (1 << SHIFT)).to(torch.int32).contiguous()
        w_q, w_sc = quantise_weights(inner[idx - 1].weight.detach().float())
        tap_w = torch.stack([w_q[:, :, i, j].t() for i in range(3) for j in range(3)]).contiguous()
        launches.append(dict(
            w=tap_w, m=m_int, vec=(w_sc * s_t).contiguous(),
            bias=inner[idx - 1].bias.detach().float().contiguous(),
            slope=prels[idx].weight.detach().float().contiguous(),
            os=s_out.contiguous()))
        s_in = s_out
    grid = (triton.cdiv(args.height * args.width, BLOCK_M),)

    with torch.no_grad():
        head = torch.nn.functional.prelu(head_conv(x), prels[0].weight)
        cur = torch.round(head / s_act.view(1, -1, 1, 1)).clamp(-127, 127).to(torch.int8)
        cur = cur.permute(0, 2, 3, 1).contiguous()
        buffers = [torch.empty_like(cur) for _ in range(2)]
        xq = torch.round(x.float() / (1.0 / 127.0)).clamp(-127, 127) * (1.0 / 127.0)

    def run_int8_body():
        src, dst = cur, buffers[0]
        for i, L in enumerate(launches):
            body_prologue_kernel[grid](src, L["w"], L["m"], SHIFT, L["vec"], L["bias"], L["slope"],
                                       L["os"], dst, args.height, args.width,
                                       BLOCK_M=BLOCK_M, C=64, num_warps=4)
            src, dst = dst, src
        return src

    def run_fp16_body():
        with torch.no_grad():
            a = head
            for c, p in zip(inner, prels[1:]):
                a = torch.nn.functional.prelu(c(a), p.weight)
            return a

    def run_head():
        with torch.no_grad():
            return torch.nn.functional.prelu(head_conv(x), prels[0].weight)

    def run_quantise():
        with torch.no_grad():
            t = torch.round(head / s_act.view(1, -1, 1, 1)).clamp(-127, 127).to(torch.int8)
            return t.permute(0, 2, 3, 1).contiguous()

    def run_dequantise():
        return cur.permute(0, 3, 1, 2).float() * outs[-1].view(1, -1, 1, 1)

    def run_tail():
        with torch.no_grad():
            last = cur.permute(0, 3, 1, 2).float() * outs[-1].view(1, -1, 1, 1)
            up = model.upsampler(tail_conv(last.half()))
            return up.float() + torch.nn.functional.interpolate(x, scale_factor=scale_up, mode="nearest")

    # the fair fp16 baseline: the same arithmetic in channels_last, which is the layout the app's engines
    # and the int8 NHWC buffers both use. NCHW fp16 falls back to a much slower cudnn path.
    cl_model = copy.deepcopy(model).to(memory_format=torch.channels_last)
    cl_convs = [m for m in cl_model.body if isinstance(m, torch.nn.Conv2d)]
    cl_prels = [m for m in cl_model.body if isinstance(m, torch.nn.PReLU)]
    x_cl = x.contiguous(memory_format=torch.channels_last)

    def run_fp16_body_cl():
        with torch.no_grad():
            a = torch.nn.functional.prelu(cl_convs[0](x_cl), cl_prels[0].weight)
            for c, p_ in zip(cl_convs[1:-1], cl_prels[1:]):
                a = torch.nn.functional.prelu(c(a), p_.weight)
            return a

    def run_fp16_cycle_cl():
        with torch.no_grad():
            a = torch.nn.functional.prelu(cl_convs[0](x_cl), cl_prels[0].weight)
            for c, p_ in zip(cl_convs[1:-1], cl_prels[1:]):
                a = torch.nn.functional.prelu(c(a), p_.weight)
            up = cl_model.upsampler(cl_convs[-1](a))
            return up.float() + torch.nn.functional.interpolate(x_cl, scale_factor=scale_up,
                                                                mode="nearest").float()

    def run_fp16_cycle():
        with torch.no_grad():
            a = head
            for c, p in zip(inner, prels[1:]):
                a = torch.nn.functional.prelu(c(a), p.weight)
            up = model.upsampler(tail_conv(a))
            return up.float() + torch.nn.functional.interpolate(x, scale_factor=scale_up, mode="nearest")

    def run_int8_cycle():
        run_int8_body()
        return run_tail()

    # nothing else on the card, and the clocks are in the log
    print(f"segment timing, {args.width}x{args.height}, {args.rounds} interleaved rounds of {args.iters} iters")
    print(f"clocks before: {clocks()}")
    rows = {"head": [], "quantise": [], "body_int8": [], "dequantise": [], "tail": [],
            "body_fp16": [], "cycle_int8": [], "cycle_fp16": [],
            "body_fp16_cl": [], "cycle_fp16_cl": []}
    for r in range(args.rounds):
        order = [("head", run_head), ("quantise", run_quantise), ("body_int8", run_int8_body),
                 ("dequantise", run_dequantise), ("tail", run_tail),
                 ("body_fp16", run_fp16_body), ("cycle_int8", run_int8_cycle),
                 ("cycle_fp16", run_fp16_cycle), ("body_fp16_cl", run_fp16_body_cl),
                 ("cycle_fp16_cl", run_fp16_cycle_cl)]
        if r % 2:
            order = order[:5] + list(reversed(order[5:]))   # alternate the order, not just the round
        for name, fn in order:
            rows[name].append(float(timeit(fn, iters=args.iters)))
        print(f"  round {r + 1}: fp16 nchw {rows['body_fp16'][-1]:6.2f}  "
              f"fp16 nhwc {rows['body_fp16_cl'][-1]:6.2f}  int8 {rows['body_int8'][-1]:6.2f} ms   "
              f"nchw {rows['body_fp16'][-1] / rows['body_int8'][-1]:5.3f}x  "
              f"nhwc {rows['body_fp16_cl'][-1] / rows['body_int8'][-1]:5.3f}x", flush=True)
    print(f"clocks after: {clocks()}")

    med = {k: float(np.median(v)) for k, v in rows.items()}
    print("\nmedian over rounds:")
    for k in ("head", "quantise", "body_int8", "dequantise", "tail", "cycle_int8"):
        print(f"  {k:<12} {med[k]:7.3f} ms")
    print(f"  {'body_fp16':<12} {med['body_fp16']:7.3f} ms")
    print(f"  {'cycle_fp16':<12} {med['cycle_fp16']:7.3f} ms")
    print(f"  {'body_fp16_cl':<12} {med['body_fp16_cl']:7.3f} ms")
    print(f"  {'cycle_fp16_cl':<12} {med['cycle_fp16_cl']:7.3f} ms")
    print(f"\nengine ratio vs NCHW fp16: {med['body_fp16'] / med['body_int8']:.3f}x")
    print(f"engine ratio vs NHWC fp16: {med['body_fp16_cl'] / med['body_int8']:.3f}x "
          f"(gate 1.40x, kill 1.15x)")
    print(f"cycle ratio  vs NHWC fp16: {med['cycle_fp16_cl'] / med['cycle_int8']:.3f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
