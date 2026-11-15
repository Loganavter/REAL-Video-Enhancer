"""E30 phase 0: what does a channel cut cost in TIME, before anyone asks about the picture?

The plan's arithmetic multiplied MACs by 0.56 and called it 1.78x, which E19 already refuted as a
general rule (C=3 -> 8 raised FLOPs 0.48% and moved time 3.36 ms because the kernel CLASS changed).
So measure the engine's own time at three widths with the same builder the app uses.

Three synthesized models of the same architecture (SRVGG compact, 16 body convs, x2) at num_feat
64 / 48 / 32, random weights - tactics do not depend on weight values, only on shapes - exported at
1920x1080 and built with `build_plan`'s settings. Per-layer times come from trt.IProfiler, so the body
convolutions can be separated from the non-GEMM remainder.

The tile question this answers: the shipped kernel is tilesize128x64x32, i.e. an N tile of 64. A
48-channel model fills 75% of that tile and K = 432 is 14 steps of 32 against 18; a 32-channel model
fills it exactly and K = 288 is 9 steps. If the measured ratio is 1.29x rather than 1.78x, the plan's
target was arithmetic on paper.

Usage: probe_channel_width.py [--widths 64,48,32] [--height 1080] [--width 1920]
"""
import argparse
import os
import sys

import torch

BACKEND = "/home/jorj/Загрузки/projects/REAL-Video-Enhancer/backend"
# spandrel comes from the bundled install (the repo has no venv), RawTRTEngine from THIS checkout
sys.path.insert(0, "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/backend/src/pytorch")
sys.path.insert(0, BACKEND)

from src.pytorch.RawTRTEngine import build_plan, export_onnx  # noqa: E402
from spandrel.architectures.Compact.__arch.SRVGG import SRVGGNetCompact  # noqa: E402

import tensorrt as trt  # noqa: E402

OUT = "/tmp/opencode/widths"
LOG = trt.Logger(trt.Logger.ERROR)


def profile(plan, h, w, iters=20, warm=5):
    import numpy as np
    engine = trt.Runtime(LOG).deserialize_cuda_engine(open(plan, "rb").read())
    ctx = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    i_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    o_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
    x = torch.rand(1, 3, h, w, device="cuda", dtype=torch.float32)
    y = torch.empty(tuple(engine.get_tensor_shape(o_name)), device="cuda", dtype=torch.float32)
    ctx.set_tensor_address(i_name, x.data_ptr())
    ctx.set_tensor_address(o_name, y.data_ptr())
    stream = torch.cuda.current_stream().cuda_stream
    for _ in range(warm):
        ctx.execute_async_v3(stream)
    torch.cuda.synchronize()

    class P(trt.IProfiler):
        def __init__(self):
            super().__init__()
            self.per = {}

        def report_layer_time(self, name, ms):
            if ms > 0:
                self.per[name] = self.per.get(name, 0.0) + ms

    prof = P()
    ctx.profiler = prof
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        ctx.execute_async_v3(stream)
    e.record()
    torch.cuda.synchronize()
    total = s.elapsed_time(e) / iters
    per = {k: v / iters for k, v in prof.per.items()}
    # the fused block is named __myl_CorrMulLtSele_* (conv+PReLU in one kernel); when fusion breaks
    # the same work shows up as /body.N/Conv plus a separate PWN(/body.N/PRelu), which is the
    # difference worth reporting - name-based grouping has to know both spellings or it reports
    # "body 3.95 ms" for a 48 ms body
    import re
    fused = [v for k, v in per.items() if "myl" in k.lower()]
    split_conv = [v for k, v in per.items() if re.search(r"/body\.\d+/Conv", k)]
    split_act = [v for k, v in per.items() if "PRelu" in k or "PWN(" in k]
    body = fused + split_conv
    shape = (f"{len(fused)} fused" if fused else
             f"{len(split_conv)} conv + {len(split_act)} separate PReLU")
    return total, sum(body), shape, sorted(per.items(), key=lambda kv: -kv[1])[:4]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="64,48,32")
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    torch.manual_seed(0)
    results = {}
    for feat in [int(v) for v in args.widths.split(",")]:
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=feat, num_conv=16, upscale=2)
        # fp32 for the export: TRT's FP16 flag is what makes the engine half, and a half model
        # plus a float dummy is the mismatch that killed the first run
        model = model.eval().float().cuda()
        dummy = torch.rand(1, 3, args.height, args.width, device="cuda", dtype=torch.float32)
        onnx_path = f"{OUT}/compact_f{feat}.onnx"
        plan_path = f"{OUT}/compact_f{feat}.engine"
        if not os.path.exists(onnx_path):
            export_onnx(model, (dummy,), onnx_path)
        if not os.path.exists(plan_path):
            build_plan(onnx_path, plan_path, fp16=True, tiling="full")
        total, body, shape, top = profile(plan_path, args.height, args.width, iters=args.iters)
        macs = (2 * 9 * 3 * feat + 16 * 2 * 9 * feat * feat + 2 * 9 * feat * 3 * 4) * args.height * args.width
        results[feat] = (total, body, macs, top)
        print(f"feat={feat:>3} total {total:6.2f} ms | body {body:6.2f} ms ({shape}) | "
              f"{macs / total / 1e9:5.1f} TFLOP/s | top: {[(n[:38], round(t, 2)) for n, t in top]}")

    if 64 in results:
        base_t, base_b, base_m, _ = results[64]
        for feat, (total, body, macs, _) in results.items():
            print(f"feat={feat:>3}: MAC ratio {base_m / macs:4.2f}x  "
                  f"total {base_t / total:4.2f}x  body {base_b / body:4.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
