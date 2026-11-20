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
  python3 harness/scripts/bench_plan.py <plan> [--width 1920] [--height 1080] [--iters 25] [--repeat 3]
  python3 harness/scripts/bench_plan.py <plan> --dump-layers     # layer/reformat counts, no timing
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
    ap.add_argument("--harness-note", default="one GPU job at a time", help="notes for the record's instrument field")
    ap.add_argument("--dump-layers", action="store_true")
    ap.add_argument("--emit", default="", help="write the repeats as harness records (JSONL)")
    ap.add_argument("--profile", action="store_true", help="per-layer times from trt.IProfiler")
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
        names = [str(r.get("Name", "")) if isinstance(r, dict) else str(r) for r in rows]
        print(f"total {len(names)}, __myl_ {sum(n.startswith('__myl_') for n in names)}, "
              f"Reformat {sum('Reformat' in n for n in names)}, "
              f"PointWise {sum('PointWise' in n or n.startswith('PWN') for n in names)}, "
              f"Conv/Gemm {sum(('Conv' in n or 'Gemm' in n or 'xmma' in n.lower()) for n in names)}")
        return 0

    ctx = engine.create_execution_context()

    class Layers(trt.IProfiler):
        def __init__(self) -> None:
            trt.IProfiler.__init__(self)
            self.ms: dict[str, float] = {}

        def report_layer_time(self, name, ms) -> None:      # noqa: N802 - TRT API
            self.ms[name] = self.ms.get(name, 0.0) + float(ms)

    profiler = Layers() if args.profile else None
    if profiler is not None:
        ctx.profiler = profiler
    for n in ins:
        ctx.set_tensor_address(n, allocate(engine, n, args.width, args.height).data_ptr())
    for n in outs:
        ctx.set_tensor_address(n, allocate(engine, n, args.width, args.height).data_ptr())
    stream = torch.cuda.Stream()

    def once() -> None:
        ctx.execute_async_v3(stream.cuda_stream)

    results: list[float] = []
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
            results.append(ms)
            flops = 2.0 * 64 * 64 * 9 * 16 * args.height * args.width   # 16 body convs, as MODEL.md §1
            print(f"repeat {r}: {ms:.2f} ms/frame -> {flops / ms / 1e9:.1f} TFLOP/s")

    if profiler is not None and profiler.ms:
        total = sum(profiler.ms.values())
        print(f"per-layer (one execution, total {total:.2f} ms):")
        for name, ms in sorted(profiler.ms.items(), key=lambda kv: -kv[1])[:12]:
            print(f"  {ms:6.2f} ms  {name[:90]}")

    if args.emit:
        import statistics
        import os as _bos, sys as _bsys
        _HARNESS_ROOT = _bos.path.dirname(_bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
        if _HARNESS_ROOT not in _bsys.path:
            _bsys.path.insert(0, _HARNESS_ROOT)
        from harness import Population, Record, Store
        store = Store()
        insp = engine.create_engine_inspector()
        import json as _json
        layers = _json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))
        rows = layers.get("Layers", layers) if isinstance(layers, dict) else layers
        names = [str(r.get("Name", "")) if isinstance(r, dict) else str(r) for r in rows]
        graph = {"total": len(names),
                 "fused_mylin": sum(n.startswith("__myl_") for n in names),
                 "reformat": sum("Reformat" in n for n in names),
                 "pointwise": sum("PointWise" in n or n.startswith("PWN") for n in names)}
        pop = Population.of("plan repeats", [f"repeat-{i+1}" for i in range(len(results))],
                            label=f"{args.plan}: {args.width}x{args.height}, fp16, one process, no encoder")
        common = dict(population=pop,
                      instrument=f"torch.cuda.Event around {args.iters} executions, {args.harness_note}",
                      config=f"trt {trt.__version__}, {torch.cuda.get_device_name(0)}, "
                             f"static {args.width}x{args.height}, opt level 5, iters {args.iters}",
                      session=store.session, run=f"run-bench-{args.width}x{args.height}",
                      extra={"of": "frame_time", "quantity": "speed", "graph": graph,
                             "artifact_sha256": __import__("hashlib").sha256(plan).hexdigest()})
        store.emit(Record(value=round(statistics.median(results), 2), unit="ms", statistic="median", **common))
        store.emit(Record(value=round(max(results), 2), unit="ms", statistic="max", **common))
        store.dump(args.emit)
        print(f"records -> {args.emit}  (graph {graph})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
