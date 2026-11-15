"""B2: is there ANY tactic for Conv(32->32, 3x3) + PReLU that myelin will fuse?

The selector cannot filter by kernel name - IAlgorithmVariant gives `implementation` (an int) and a
64-bit tactic hash - but it can restrict the build to ONE choice index at a time, and the resulting
engine names its layers (`__myl_CorrMulLtSele_*` when the PReLU went into the same kernel, a bare
`Conv` plus a `PWN(.../PRelu)` when it did not). So the fused variant is found by scanning indices and
reading the names, not by guessing names.

Single-layer graph on purpose: 293 choices per layer on the full model means a full rebuild per index,
while a one-conv graph builds in seconds. Shapes are what tactics depend on, so a 1080p one-conv graph
makes the same decision the body layer makes.

Usage: force_tactic.py [--channels 32] [--limit 40] [--index N]
  no --index: scan the fastest `--limit` choices and stop at the first fused one
  --index N : build with only choice N and report what came out
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/backend/src/pytorch")

import tensorrt as trt  # noqa: E402

OUT = "/tmp/opencode/force"
LOG = trt.Logger(trt.Logger.ERROR)


def make_onnx(channels, h, w):
    import torch.nn as nn
    path = f"{OUT}/block_c{channels}_{h}x{w}.onnx"

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)
            self.act = nn.PReLU(channels)

        def forward(self, x):
            return self.act(self.conv(x))

    if not os.path.exists(path):
        m = Block().eval()
        torch.onnx.export(m, (torch.rand(1, channels, h, w),), path, opset_version=17,
                          input_names=["input"], output_names=["output"],
                          do_constant_folding=True, dynamo=False)
    return path


class Pick(trt.IAlgorithmSelector):
    """Allow every choice except on the target layer, where only `index` is allowed."""

    def __init__(self, index):
        super().__init__()
        self.index = index
        self.seen = None

    def select_algorithms(self, context, choices):
        names = [c.algorithm_variant.tactic for c in choices]
        if self.seen is None:
            self.seen = names
        if self.index is None or self.index >= len(choices):
            return list(range(len(choices)))
        return [self.index]

    def report_algorithms(self, context, choices):
        pass


def build(onnx_path, index=None):
    builder = trt.Builder(LOG)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(net, LOG)
    with open(onnx_path, "rb") as fh:
        if not parser.parse(fh.read()):
            return None, [f"parse: {parser.get_error(i)}" for i in range(parser.num_errors)]
    cfg = builder.create_builder_config()
    cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.builder_optimization_level = 5
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    cfg.algorithm_selector = Pick(index)
    blob = builder.build_serialized_network(net, cfg)
    return (bytes(blob) if blob else None), []


def describe(engine_bytes, h, w, iters=15):
    engine = trt.Runtime(LOG).deserialize_cuda_engine(engine_bytes)
    insp = engine.create_engine_inspector()
    ctx = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    i_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    o_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
    x = torch.rand(1, engine.get_tensor_shape(i_name)[1], h, w, device="cuda")
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

    layers = []
    for i in range(engine.num_layers):
        info = json.loads(insp.get_layer_information(i, trt.LayerInformationFormat.JSON))
        layers.append((info.get("Name", f"l{i}"), info.get("TacticName") or "", prof.per.get(info.get("Name"), 0.0)))
    layers = [(n, t, ms / iters) for n, t, ms in layers]   # the profiler accumulates across iterations
    fused = any("CorrMulLtSele" in n or "CorrMulLtSele" in t for n, t, _ in layers)
    return total, layers, fused


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--index", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    onnx_path = make_onnx(args.channels, args.height, args.width)

    if args.index is not None:
        blob, errs = build(onnx_path, args.index)
        if blob is None:
            print(f"index {args.index}: build failed {errs[:1]}")
            return 1
        total, layers, fused = describe(blob, args.height, args.width)
        print(f"index {args.index}: {total:.3f} ms fused={fused} layers={[(n[:30], t[:40]) for n, t, _ in layers]}")
        return 0

    # default: what does TRT choose, and is it fused?
    blob, errs = build(onnx_path, None)
    total, layers, fused = describe(blob, args.height, args.width)
    print(f"default: {total:.3f} ms fused={fused}")
    for n, t, ms in layers:
        print(f"   {ms:6.3f}  {n[:36]:<36} {t[:60]}")

    seen_fused = set()
    for index in range(args.limit):
        blob, errs = build(onnx_path, index)
        if blob is None:
            continue
        try:
            total, layers, fused = describe(blob, args.height, args.width, iters=10)
        except Exception as exc:  # a tactic that cannot run is a legitimate answer
            print(f"index {index}: ran but failed: {exc}")
            continue
        tag = "FUSED" if fused else "split"
        names = {t[:46] for _, t, _ in layers if t}
        if fused and index not in seen_fused:
            print(f"index {index}: {total:6.3f} ms {tag}  {sorted(names)}")
            seen_fused.add(index)
            break
        print(f"index {index}: {total:6.3f} ms {tag}  {sorted(names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
