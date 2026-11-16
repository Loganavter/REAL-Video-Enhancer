"""Time a TensorRT plan without torch-tensorrt.

Why this exists: `backend/tests/tools/bench_engine.py` is the repo's instrument, but it loads the
engine through `torch_tensorrt`, which pins an exact torch version. When the only torch available is
a different patch release, the tool refuses and the plan cannot be timed at all. This is the same
measurement with the same protocol - CUDA events around N calls after warmup, one GPU job at a time,
every repeat printed instead of an average - using `tensorrt` + `torch` for buffers only.

What it does NOT do: build engines, decide the tiling (a plan already carries it), or compare against
another session. Ratios are only meaningful inside one session; the ledger says so and this tool
prints every repeat for that reason.

Usage:
  python3 scripts/bench_plan.py <plan> [--width 1920] [--height 1080] [--iters 25] [--repeat 3]
  python3 scripts/bench_plan.py <plan> --dump-layers     # layer/reformat counts, no timing
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
import tensorrt as trt


def io_tensors(engine):
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    ins = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    outs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
    return ins, outs


def allocate(engine, name: str, width: int, height: int) -> torch.Tensor:
    shape = tuple(engine.get_tensor_shape(name))
    if any(d < 0 for d in shape):                      # dynamic: assume NCHW with H, W last
        shape = tuple(height if d < 0 and i == 2 else width if d < 0 and i == 3 else d
                      for i, d in enumerate(shape))
    dtype = {trt.DataType.HALF: torch.float16, trt.DataType.FLOAT: torch.float32,
             trt.DataType.INT8: torch.int8}[engine.get_tensor_dtype(name)]
    return torch.zeros(shape, dtype=dtype, device="cuda")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--warm", type=int, default=6)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--dump-layers", action="store_true")
    args = ap.parse_args()

    logger = trt.Logger(trt.Logger.ERROR)
    with open(args.plan, "rb") as fh:
        plan = fh.read()
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
    if engine is None:
        print(f"{args.plan}: the runtime refused the plan (built by another TRT version?)")
        return 1

    ins, outs = io_tensors(engine)
    print(f"{args.plan}: {engine.num_layers} layers, in={ins} out={outs}, "
          f"trt {trt.__version__}, device {torch.cuda.get_device_name(0)}")

    if args.dump_layers:
        insp = engine.create_engine_inspector()
        info = insp.get_engine_information(trt.LayerInformationFormat.JSON)
        import json
        layers = json.loads(info) if isinstance(info, str) else info
        rows = layers.get("Layers", layers) if isinstance(layers, dict) else layers
        names = [str(r.get("Name", "")) for r in rows]
        print(f"total {len(names)}, __myl_ {sum(n.startswith('__myl_') for n in names)}, "
              f"Reformat {sum('Reformat' in n for n in names)}, "
              f"PointWise {sum('PointWise' in n or n.startswith('PWN') for n in names)}, "
              f"Conv/Gemm {sum(('Conv' in n or 'Gemm' in n or 'xmma' in n.lower()) for n in names)}")
        return 0

    ctx = engine.create_execution_context()
    for n in ins:
        ctx.set_tensor_address(n, allocate(engine, n, args.width, args.height).data_ptr())
    for n in outs:
        ctx.set_tensor_address(n, allocate(engine, n, args.width, args.height).data_ptr())
    stream = torch.cuda.Stream()

    def once() -> None:
        ctx.execute_async_v3(stream.cuda_stream)

    with torch.cuda.stream(stream):
        for _ in range(args.warm):
            once()
        torch.cuda.synchronize()
        for r in range(1, args.repeat + 1):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for _ in range(args.iters):
                once()
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end) / args.iters
            flops = 2.0 * 64 * 64 * 9 * 16 * args.height * args.width   # 16 body convs, as MODEL.md §1
            print(f"repeat {r}: {ms:.2f} ms/frame -> {flops / ms / 1e9:.1f} TFLOP/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
