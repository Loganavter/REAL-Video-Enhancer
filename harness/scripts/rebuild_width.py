"""B2: the same ONNX built several times - is the tactic choice stable, and does the fused one appear?

Two builds of the identical C=32 graph picked different families in one session: `scudnn_winograd`
(3.12 ms, PReLU split out) once and `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc`
(1.235 ms, fused) the next time. If that is real, the "C=32 broke the fusion" result is a property of
one build, not of the shape, and the prize (16 x ~1.4 ms + non-GEMM ~4 = ~26 ms, 2.15x) is one rebuild
away.

Builds with the app's settings plus a generous workspace: a tactic skipped for lack of a scratch buffer
is absent from the result for the wrong reason, and the ledger already documents a CUDA error 2
allocating a 4.2 GB buffer during autotuning.

Usage: rebuild_width.py [--channels 32] [--rounds 3] [--workspace-gb 8]
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, "/home/jorj/Загрузки/projects/REAL-Video-Enhancer/backend")
sys.path.insert(0, "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/backend/src/pytorch")

import tensorrt as trt  # noqa: E402
from spandrel.architectures.Compact.__arch.SRVGG import SRVGGNetCompact  # noqa: E402

OUT = "/tmp/opencode/widths"
LOG = trt.Logger(trt.Logger.ERROR)


def ensure_onnx(channels, h, w):
    path = f"{OUT}/compact_f{channels}.onnx"
    if not os.path.exists(path):
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=channels, num_conv=16,
                                upscale=2).eval().cuda()
        torch.onnx.export(model, (torch.rand(1, 3, h, w, device="cuda"),), path, opset_version=17,
                          input_names=["input"], output_names=["output"],
                          do_constant_folding=True, dynamo=False)
    return path


def build(onnx_path, plan_path, workspace_gb):
    builder = trt.Builder(LOG)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(net, LOG)
    with open(onnx_path, "rb") as fh:
        if not parser.parse(fh.read()):
            raise RuntimeError("parse failed")
    cfg = builder.create_builder_config()
    cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.builder_optimization_level = 5
    cfg.tiling_optimization_level = trt.TilingOptimizationLevel.FULL
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb) << 30)
    blob = builder.build_serialized_network(net, cfg)
    if blob is None:
        raise RuntimeError("build failed")
    with open(plan_path, "wb") as fh:
        fh.write(bytes(blob))
    return plan_path


def inspect(plan_path, h, w, iters=15):
    engine = trt.Runtime(LOG).deserialize_cuda_engine(open(plan_path, "rb").read())
    insp = engine.create_engine_inspector()
    ctx = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    i_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    o_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
    x = torch.rand(1, 3, h, w, device="cuda")
    y = torch.empty(tuple(engine.get_tensor_shape(o_name)), device="cuda")
    ctx.set_tensor_address(i_name, x.data_ptr())
    ctx.set_tensor_address(o_name, y.data_ptr())
    stream = torch.cuda.current_stream().cuda_stream
    for _ in range(5):
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

    rows = []
    for i in range(engine.num_layers):
        info = json.loads(insp.get_layer_information(i, trt.LayerInformationFormat.JSON))
        rows.append((info.get("Name", ""), info.get("TacticName") or "", prof.per.get(info.get("Name"), 0) / iters))
    fused = [r for r in rows if "CorrMulLtSele" in r[0] or "CorrMulLtSele" in r[1]]
    split = [r for r in rows if "PRelu" in r[0]]
    families = sorted({r[1].split("_f16")[0] for r in rows if r[1]})
    return total, len(fused), len(split), families, sorted(rows, key=lambda r: -r[2])[:3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--workspace-gb", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    onnx_path = ensure_onnx(args.channels, args.height, args.width)
    print(f"C={args.channels}, identical ONNX, {args.rounds} builds, workspace {args.workspace_gb} GB")
    totals = []
    for r in range(args.rounds):
        plan = f"{OUT}/rebuild_c{args.channels}_r{r}.engine"
        build(onnx_path, plan, args.workspace_gb)
        total, nfused, nsplit, families, top = inspect(plan, args.height, args.width)
        totals.append(total)
        print(f"  build {r}: {total:6.2f} ms | fused layers {nfused} | separate PReLU {nsplit} | "
              f"{families}")
        for name, tactic, ms in top:
            print(f"      {ms:6.2f}  {name[:34]:<34} {tactic[:52]}")
    print(f"spread across builds: {min(totals):.2f}..{max(totals):.2f} ms "
          f"({max(totals) / min(totals):.2f}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
