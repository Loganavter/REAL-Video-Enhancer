"""Stage 3: the CUDA graph over the 16 int8 launches - bit-exactness first, the stopwatch second.

The plan asks three things in order, and the order is the point:
  1. the graph must reproduce eager BIT-EXACTLY on the whole body, not merely "close";
  2. capture with scales A, overwrite the device buffers with B, replay, compare against eager on B --
     if it matches B the graph binds by POINTER and one capture serves every video's calibration; if it
     matches A the values are baked in and the app must capture per video (or move the scales into device
     memory anyway, which is what this test decides);
  3. only then does the replay get timed.

The first enqueue does lazy initialisation, so every launch is warmed before the capture begins.

Usage: int8_graph.py [--width W] [--height H] [--iters N]
"""
import argparse
import copy
import sys

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
from int8_pipeline import SHIFT, quantise_weights  # noqa: E402
from int8_prologue_kernel import body_prologue_kernel, dequantise_nhwc_to_nchw  # noqa: E402
from int8_prologue_kernel import quantise_nchw_to_nhwc  # noqa: E402

BUNDLE = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2"
BM, WARPS, QB = 64, 2, 1024


def timeit(fn, iters, warmup=5):
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--frame", default="/var/tmp/opencode/frame.raw")
    args = ap.parse_args()

    _full = np.fromfile(args.frame, dtype=np.uint8)
    _h, _w = _full.size // (3 * args.width), args.width
    raw = _full.reshape(_h, _w, 3)[:args.height, :args.width]
    x = torch.from_numpy(raw.astype(np.float32) / 255.0).permute(2, 0, 1)[None].cuda().half()
    model, _, _ = load(f"{BUNDLE}/custom_models/2x-Futsuu-Anime(1).pth")
    model = model.eval().cuda().half()
    calibration = copy.deepcopy(model)
    inputs, outputs, _ = calibrate(calibration, [x])
    convs = [m for m in model.body if isinstance(m, torch.nn.Conv2d)]
    prels = [m for m in model.body if isinstance(m, torch.nn.PReLU)]

    lo, hi = outputs["body.1"]
    s_act = (torch.maximum(hi.abs(), lo.abs()).float() / 127.0).clamp_min(1e-8)
    outs = []
    for i in range(16):
        lo_c, hi_c = outputs[f"body.{2 * (i + 1) + 1}"]
        outs.append((torch.maximum(hi_c.abs(), lo_c.abs()).float() / 127.0).clamp_min(1e-8))

    launches, s_in = [], s_act
    for idx in range(1, 17):
        s_out = outs[idx - 1]
        s_t = float(s_in.max())
        m_int = torch.round(s_in / s_t * (1 << SHIFT)).to(torch.int32).contiguous()
        w_q, w_sc = quantise_weights(convs[idx].weight.detach().float())
        tap = torch.stack([w_q[:, :, i, j].t() for i in range(3) for j in range(3)]).contiguous()
        launches.append(dict(
            w=tap, m=m_int, vec=(w_sc * s_t).contiguous(),
            bias=convs[idx].bias.detach().float().contiguous(),
            slope=prels[idx].weight.detach().float().contiguous(), os=s_out.contiguous()))
        s_in = s_out
    # a second calibration: every per-hop grid halved, so a pointer-bound graph must follow it
    alt = []
    for L in launches:
        alt.append(dict(w=L["w"], m=(L["m"] // 2).to(torch.int32).contiguous(), vec=L["vec"] * 0.5,
                        bias=L["bias"], slope=L["slope"], os=L["os"] * 0.5))

    with torch.no_grad():
        head = torch.nn.functional.prelu(convs[0](x), prels[0].weight)
    # THREE buffers: the ping-pong writes over its own input, so the source has to be separate or every
    # run after the first starts from the previous run's last hop (the trap that made the first graph
    # comparison read 35 M differing codes).
    inp = torch.empty((1, args.height, args.width, 64), device="cuda", dtype=torch.int8)
    quantise_nchw_to_nhwc[(triton.cdiv(args.height * args.width, QB),)](
        head, s_act.contiguous(), inp, args.height * args.width, BLOCK=QB, C=64, num_warps=4)
    buf_a = torch.empty_like(inp)
    buf_b = torch.empty_like(inp)
    last = torch.empty((1, 64, args.height, args.width), device="cuda", dtype=torch.float16)
    grid = (triton.cdiv(args.height * args.width, BM),)
    qgrid = (triton.cdiv(args.height * args.width, QB),)

    def run_body(params):
        # the destination must cycle between buf_a and buf_b ONLY: a plain swap would hand `inp` back as
        # the destination on the second hop and destroy the source, so every run after the first started
        # from a previous run's output and a captured graph re-read that garbage as its input.
        src, dst = inp, buf_a
        for L in params:
            body_prologue_kernel[grid](src, L["w"], L["m"], SHIFT, L["vec"], L["bias"], L["slope"],
                                       L["os"], dst, args.height, args.width,
                                       BLOCK_M=BM, C=64, num_warps=WARPS)
            src, dst = dst, (buf_b if dst is buf_a else buf_a)
        dequantise_nhwc_to_nchw[qgrid](src, params[-1]["os"].contiguous(), last,
                                       args.height * args.width, BLOCK=QB, C=64, num_warps=4)
        return src

    # ---- 1. the graph against eager, bit-exact
    for _ in range(5):                     # the first enqueue does lazy initialisation
        run_body(launches)
    torch.cuda.synchronize()
    eager_a = run_body(launches).clone()          # 16 hops, so the ping-pong ends on buf_b
    torch.cuda.synchronize()
    eager_last = last.clone()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run_body(launches)
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    for nm, b in (("buf_a", buf_a), ("buf_b", buf_b)):
        d = (b.to(torch.int16) - eager_a.to(torch.int16)).abs()
        print(f"   {nm}: max diff {int(d.max())}, differing {int((d > 0).sum())}")
    r1 = buf_b.clone(); g.replay(); torch.cuda.synchronize()
    print(f"   replay-to-replay determinism: max diff {int((buf_b.to(torch.int16) - r1.to(torch.int16)).abs().max())}")
    diff = (buf_b.to(torch.int16) - eager_a.to(torch.int16)).abs()
    print(f"1. graph vs eager, codes: max diff {int(diff.max())}, differing {int((diff > 0).sum())} "
          f"of {diff.numel()}")
    print(f"   dequantised fp16 tail input: max |diff| "
          f"{float((last.float() - eager_last.float()).abs().max()):.6f}")

    # ---- 2. pointer or value? overwrite the captured buffers with a different calibration, replay
    eager_b = run_body(alt).clone()
    torch.cuda.synchronize()
    for L_dst, L_src in zip(launches, alt):
        L_dst["m"].copy_(L_src["m"])
        L_dst["vec"].copy_(L_src["vec"])
        L_dst["os"].copy_(L_src["os"])
    g.replay()
    torch.cuda.synchronize()
    got_b = buf_b.clone()
    d_ptr = (got_b.to(torch.int16) - eager_b.to(torch.int16)).abs().max().item()
    d_val = (got_b.to(torch.int16) - eager_a.to(torch.int16)).abs().max().item()
    print(f"2. after overwriting the grids with a second calibration and replaying:")
    print(f"   graph vs eager-on-B: max diff {d_ptr}   graph vs eager-on-A: max diff {d_val}")
    verdict = ("POINTERS - one capture serves any calibration" if d_ptr == 0 and d_val > 0
               else "VALUES BAKED IN - capture per calibration" if d_val == 0 and d_ptr > 0
               else "INCONCLUSIVE - the two calibrations did not separate")
    print(f"   verdict: {verdict}")

    # ---- 3. the stopwatch, only now
    for L_dst, L_src in zip(launches, alt):     # restore the first calibration
        L_dst["m"].copy_(L_src["m"])
    ms_eager = timeit(lambda: run_body(launches), args.iters)
    ms_graph = timeit(g.replay, args.iters)
    print(f"3. body+dequantise: eager {ms_eager:.2f} ms, graph replay {ms_graph:.2f} ms, "
          f"{ms_eager / ms_graph:.3f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
