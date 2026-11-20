"""Stage 2 gate: the cached fp16 TensorRT engine against the int8 cycle, in one session, with NVENC alongside.

The plan's gate is engine >= 1.4x, kill <= 1.15x, and it insists on three conditions this script implements:
the fp16 baseline in the same session with both engines cached, the boundaries inside the pipeline rather than
as standalone kernels, and NVENC encoding concurrently (the fp16 engine pays 9% there; the int8 body had never
been measured under it).

Round order alternates so a warming clock cannot favour whichever side runs second; clocks are logged before,
between and after; every launch argument is built before the clock starts.

Usage: int8_app_gate.py [--rounds 3] [--iters 10] [--nvenc on|off|both] [--width 1920] [--height 1080]
"""
import argparse
import copy
import subprocess
import sys
import time

import numpy as np
import torch
import torch_tensorrt  # noqa: F401 - registers torch.classes.tensorrt.Engine for torch.jit.load
import torch_tensorrt.dynamo.runtime._TorchTensorRTModule  # noqa: F401,E402
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
from int8_prologue_kernel import (body_prologue_kernel, dequantise_nhwc_to_nchw,  # noqa: E402
                                  quantise_nchw_to_nhwc)

BUNDLE = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2"


def clocks():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({exc})"


def timeit(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def start_nvenc(width, height, frame_path):
    """Keep NVENC busy for the whole measurement: a real h264 encode loop, not a stub."""
    cmd = ["ffmpeg", "-v", "error", "-stream_loop", "-1", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{width}x{height}", "-r", "30", "-i", frame_path,
           "-c:v", "h264_nvenc", "-preset", "p1", "-b:v", "20M", "-f", "null", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    time.sleep(3.0)          # let the encoder spin up before any clock starts
    if proc.poll() is not None:
        raise RuntimeError(f"NVENC died at startup: {proc.stderr.read().decode()[:400]}")
    return proc


def stop_nvenc(proc):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> int:
    torch.backends.cudnn.benchmark = True
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--nvenc", choices=["on", "off", "both"], default="both")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--frame", default="/var/tmp/opencode/frame.raw")
    ap.add_argument("--bm", type=int, default=64, help="BLOCK_M; 64x2 warps measured best at 1920x1080")
    ap.add_argument("--warps", type=int, default=2)
    ap.add_argument("--engine", default=None)
    ap.add_argument("--model", default=f"{BUNDLE}/custom_models/2x-Futsuu-Anime(1).pth")
    args = ap.parse_args()

    engine_path = args.engine
    if engine_path is None:
        import glob
        import os
        stem = os.path.basename(args.model)
        hits = glob.glob(f"{BUNDLE}/custom_models/{stem}_fp16_*"
                         f"{args.width}x{args.height}*.engine")
        if not hits:
            print(f"no cached fp16 engine for {args.width}x{args.height}", file=sys.stderr)
            return 2
        engine_path = hits[0]

    raw = np.fromfile(args.frame, dtype=np.uint8).reshape(args.height, args.width, 3)
    x = torch.from_numpy(raw.astype(np.float32) / 255.0).permute(2, 0, 1)[None].cuda().half()

    engine = torch.jit.load(engine_path).eval().cuda()
    model, scale_up, _ = load(args.model)
    model = model.eval().cuda().half()
    calibration = copy.deepcopy(model)
    inputs, outputs, _ = calibrate(calibration, [x])
    inputs["body.0"] = (0.0, 1.0)

    convs = [m for m in model.body if isinstance(m, torch.nn.Conv2d)]
    prels = [m for m in model.body if isinstance(m, torch.nn.PReLU)]
    head_conv, tail_conv = convs[0], convs[-1]
    inner = convs[1:-1]

    lo_h, hi_h = outputs["body.1"]
    # .float() is not decoration: calibrate() records amin/amax from half tensors, so without it the
    # requant multiplier s_in/s_t * 2^SHIFT overflows half (max 65504) to inf and casts to INT32_MAX
    s_act = (torch.maximum(hi_h.abs(), lo_h.abs()).float() / 127.0).clamp_min(1e-8)
    outs = []
    for i in range(16):
        lo_c, hi_c = outputs[f"body.{2 * (i + 1) + 1}"]
        outs.append((torch.maximum(hi_c.abs(), lo_c.abs()).float() / 127.0).clamp_min(1e-8))

    launches, s_in = [], s_act
    for idx in range(1, 17):
        s_out = outs[idx - 1]
        s_t = float(s_in.max())
        m_int = torch.round(s_in / s_t * (1 << SHIFT)).to(torch.int32).contiguous()
        w_q, w_sc = quantise_weights(inner[idx - 1].weight.detach().float())
        tap_w = torch.stack([w_q[:, :, i, j].t() for i in range(3) for j in range(3)]).contiguous()
        launches.append(dict(w=tap_w, m=m_int, vec=(w_sc * s_t).contiguous(),
                             bias=inner[idx - 1].bias.detach().float().contiguous(),
                             slope=prels[idx].weight.detach().float().contiguous(), os=s_out.contiguous()))
        s_in = s_out
    grid = (triton.cdiv(args.height * args.width, args.bm),)

    QB = 1024
    qgrid = (triton.cdiv(args.height * args.width, QB),)

    def quantise_head(head):
        t = torch.empty((1, args.height, args.width, 64), device="cuda", dtype=torch.int8)
        quantise_nchw_to_nhwc[qgrid](head, s_act.contiguous(), t, args.height * args.width,
                                     BLOCK=QB, C=64, num_warps=4)
        return t

    with torch.no_grad():
        head = torch.nn.functional.prelu(head_conv(x), prels[0].weight)
        cur = quantise_head(head)
        ref_cur = (torch.round(head / s_act.view(1, -1, 1, 1)).clamp(-127, 127).to(torch.int8)
                   .permute(0, 2, 3, 1).contiguous())
        print(f"quantise kernel agrees with torch: max code diff "
              f"{int((cur.to(torch.int16) - ref_cur.to(torch.int16)).abs().max())}")
        ping = torch.empty_like(cur)

    scratch = torch.empty_like(cur)

    def int8_cycle():
        # cycle the destination between the two scratch buffers only - a plain swap would write into
        # `cur` on the second hop and destroy the input, so every call after the first would start from
        # the previous call's last hop (timing unaffected, picture silently wrong)
        src, dst = cur, ping
        for L in launches:
            body_prologue_kernel[grid](src, L["w"], L["m"], SHIFT, L["vec"], L["bias"], L["slope"],
                                       L["os"], dst, args.height, args.width,
                                       BLOCK_M=args.bm, C=64, num_warps=args.warps)
            src, dst = dst, (scratch if dst is ping else ping)
        last = torch.empty((1, 64, args.height, args.width), device="cuda", dtype=torch.float16)
        dequantise_nhwc_to_nchw[qgrid](src, outs[-1].contiguous(), last, args.height * args.width,
                                       BLOCK=QB, C=64, num_warps=4)
        with torch.no_grad():
            up = model.upsampler(tail_conv(last))
            return up.float() + torch.nn.functional.interpolate(x, scale_factor=scale_up, mode="nearest")

    def fp16_engine():
        with torch.no_grad():
            return engine(x)

    with torch.no_grad():                      # one hop against the fp16 one, before anything is timed
        src, dst = cur, ping
        L = launches[0]
        body_prologue_kernel[grid](src, L["w"], L["m"], SHIFT, L["vec"], L["bias"], L["slope"],
                                   L["os"], dst, args.height, args.width,
                                   BLOCK_M=args.bm, C=64, num_warps=args.warps)
        mine = dst.permute(0, 3, 1, 2).float() * L["os"].view(1, -1, 1, 1)
        fp = torch.nn.functional.prelu(inner[0](torch.nn.functional.prelu(head_conv(x), prels[0].weight)),
                                       prels[1].weight)
        print(f"hop 1: int8 |mean| {float(mine.abs().mean()):.4f}  fp16 |mean| {float(fp.abs().mean()):.4f}"
              f"  s_t {float(s_act.max()):.5f}  m_int[:3] {L['m'][:3].tolist()}  "
              f"is the head buffer the input? codes {int(cur.abs().max())}")
        cur = torch.round(torch.nn.functional.prelu(head_conv(x), prels[0].weight)
                          / s_act.view(1, -1, 1, 1)).clamp(-127, 127).to(torch.int8).permute(0, 2, 3, 1).contiguous()
    with torch.no_grad():                      # shapes must be the same picture, or this is not a comparison
        a, b = fp16_engine(), int8_cycle()
        print(f"shapes: fp16 engine {tuple(a.shape)}  int8 cycle {tuple(b.shape)}  "
              f"{'MATCH' if a.shape == b.shape else 'DIFFER'}")
        d = (a.float() - b.float()).abs() * 255.0
        per_px = d.amax(dim=1)
        print(f"int8 vs fp16 engine image: >8 {float((per_px > 8).float().mean()) * 100:.5f}%  "
              f"peak {float(per_px.max()):.0f}  median {float(per_px.median()):.2f} levels")
        ref = model(x).float()          # the same cycle in torch, to tell "engine" from "my cycle"
        d2 = (ref - b.float()).abs() * 255.0
        p2 = d2.amax(dim=1)
        print(f"int8 vs torch fp16 model:  >8 {float((p2 > 8).float().mean()) * 100:.5f}%  "
              f"peak {float(p2.max()):.0f}  median {float(p2.median()):.2f} levels")
        print(f"engine vs torch:           >8 "
              f"{float((((a.float() - ref).abs() * 255).amax(dim=1) > 8).float().mean()) * 100:.5f}%")

    conditions = {"off": [False], "on": [True], "both": [False, True]}[args.nvenc]
    for with_nvenc in conditions:
        proc = start_nvenc(args.width, args.height, args.frame) if with_nvenc else None
        tag = "NVENC alongside" if with_nvenc else "NVENC idle"
        try:
            print(f"\n--- {tag} ---")
            print(f"clocks before: {clocks()}")
            eng, i8b, i8c = [], [], []
            for r in range(args.rounds):
                order = [("engine", fp16_engine), ("int8", int8_cycle)]
                if r % 2:
                    order.reverse()
                got = {}
                for name, fn in order:
                    got[name] = float(timeit(fn, iters=args.iters))
                eng.append(got["engine"])
                i8c.append(got["int8"])
                print(f"  round {r + 1}: fp16 engine {got['engine']:7.2f} ms   int8 cycle {got['int8']:7.2f} ms"
                      f"   ratio {got['engine'] / got['int8']:5.3f}x", flush=True)
            print(f"clocks after:  {clocks()}")
            e, i = float(np.median(eng)), float(np.median(i8c))
            print(f"  median: engine {e:.2f} ms, int8 cycle {i:.2f} ms, engine/int8 = {e / i:.3f}x "
                  f"(gate 1.40x, kill 1.15x)")
        finally:
            if proc:
                stop_nvenc(proc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
