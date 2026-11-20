"""C: does this card overlap two TensorRT executions at all? The strips idea rests on that alone.

The proposal is to cut the frame into horizontal strips, give each strip its own context and stream,
and hide the boundary behind concurrency instead of removing it with a halo. The halo tax is
arithmetic (2 strips 1.067x, 3 strips 1.10x, 4 strips 1.133x more work), so the idea only pays if
two executions actually overlap. That is the refutes clause in the queue and it needs no strips:

  a 56 ms execution already runs at 100% SM utilisation and 162 W of a 165 W cap (MODEL.md §3),
  so the question is whether a second execution on another stream makes the pair finish in less
  than twice the time.

Timing protocol: the same plan bytes deserialized into two independent engines, each with its own
context, on two streams. Both streams enqueue N executions, then one synchronize; the reported number is
wall time divided by the executions that completed, so it is "ms per frame through this many streams".

Read the ratio as **work per unit time**, not as a fraction: `concurrent / serial` of 1.00 means the
second stream bought nothing (the pair takes twice as long as one), 0.50 would be perfect overlap (two
frames in the time of one). Interleaved rounds.

Usage (one GPU job at a time):
  flock -n /var/tmp/opencode/gpu.lock ~/.cache/opencode/venvs/trt310/bin/python \
      harness/scripts/probe_strips.py --plan /tmp/opencode/compact2x_1080p_fp16.plan
"""
from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
import threading
import time

import torch


def load_engine(plan_path: str):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.ERROR)
    with open(plan_path, "rb") as fh:
        return trt.Runtime(logger).deserialize_cuda_engine(fh.read())


def bind(engine, h: int, w: int):
    """Context plus its BUFFERS, returned together.

    The first version returned only the context: the tensors were local, Python freed them on return,
    and the addresses set on the context dangled. The engine then "executed" in 0.17 ms - a number that
    looks like perfect overlap and is actually nothing running. Hold the buffers for as long as the
    context lives.
    """
    import tensorrt as trt
    ctx = engine.create_execution_context()
    buffers = []
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        dt = engine.get_tensor_dtype(n)
        tdt = {trt.DataType.HALF: torch.float16, trt.DataType.INT8: torch.int8,
               trt.DataType.FLOAT: torch.float32}.get(dt, torch.float32)
        t = torch.zeros(tuple(engine.get_tensor_shape(n)), dtype=tdt, device="cuda")
        ctx.set_tensor_address(n, t.data_ptr())
        buffers.append(t)
    return ctx, buffers


class Sampler(threading.Thread):
    """nvidia-smi in a thread - host-side, does not touch the GPU's own counters."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.rows: list[tuple[float, float]] = []
        self.stop = threading.Event()

    def run(self) -> None:
        import subprocess
        while not self.stop.is_set():
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True).stdout.strip().split(",")
            if len(out) == 2:
                try:
                    self.rows.append((float(out[0]), float(out[1])))
                except ValueError:
                    pass
            time.sleep(0.2)


# Instrument note: an earlier version recorded CUDA events OUTSIDE the stream context, so the events
# landed on the default stream while the work ran on another one - and reported 0.1 ms for a 50 ms
# plan. Wall clock around a device synchronize cannot make that mistake, and at 50 ms the host-side
# enqueue cost is ~0.1% of what is being measured. The record says "wall clock" for that reason.
def time_serial(ctxs, streams, iters: int) -> float:
    with torch.cuda.stream(streams[0]):
        for ctx in ctxs:
            for _ in range(6):
                ctx.execute_async_v3(streams[0].cuda_stream)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        for ctx in ctxs:
            ctx.execute_async_v3(streams[0].cuda_stream)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / (iters * len(ctxs))


def time_concurrent(ctxs, streams, iters: int) -> float:
    for ctx, stream in zip(ctxs, streams):
        with torch.cuda.stream(stream):
            for _ in range(6):
                ctx.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        for ctx, stream in zip(ctxs, streams):
            with torch.cuda.stream(stream):
                ctx.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / (iters * len(ctxs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="/tmp/opencode/compact2x_1080p_fp16.plan")
    ap.add_argument("--contexts", type=int, default=2)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--emit", default="")
    args = ap.parse_args()

    plan_bytes = open(args.plan, "rb").read()
    engines = [load_engine(args.plan) for _ in range(args.contexts)]
    if any(e is None for e in engines):
        raise SystemExit(f"{args.plan}: the runtime refused the plan")
    bound = [bind(e, args.height, args.width) for e in engines]
    ctxs = [b[0] for b in bound]
    _keep = [b[1] for b in bound]      # buffers must outlive the contexts; see bind()
    streams = [torch.cuda.Stream() for _ in range(args.contexts)]

    print(f"{args.plan}: {engines[0].num_layers} layers, {args.contexts} independent engines/contexts, "
          f"{args.width}x{args.height}")

    # serial per-context time (one context on one stream, the others idle) is the reference
    serial, concurrent, one = [], [], []
    for r in range(1, args.rounds + 1):
        one_ms = time_serial(ctxs[:1], streams[:1], args.iters)
        ser = time_serial(ctxs, streams, args.iters)
        sampler = Sampler()
        sampler.start()
        con = time_concurrent(ctxs, streams, args.iters)
        sampler.stop.set()
        sampler.join(timeout=1)
        serial.append(ser)
        concurrent.append(con)
        one.append(one_ms)
        util = statistics.mean([u for u, _ in sampler.rows]) if sampler.rows else float("nan")
        watt = statistics.mean([p for _, p in sampler.rows]) if sampler.rows else float("nan")
        print(f"round {r}: one context {one_ms:6.2f} ms  {args.contexts} serial "
              f"{ser:6.2f} ms/ctx  {args.contexts} concurrent {con:6.2f} ms/ctx  "
              f"| SM {util:5.1f}%  {watt:5.1f} W")

    s_med = statistics.median(serial[1:]) if len(serial) > 1 else serial[0]
    c_med = statistics.median(concurrent[1:]) if len(concurrent) > 1 else concurrent[0]
    ratio = c_med / s_med
    print(f"\nconcurrent/serial time per frame = {ratio:.3f}x "
          f"(1.00 = NO overlap, 0.50 = perfect overlap)")
    print(f"throughput gain from the extra stream(s) = {1.0 / ratio:.2f}x  "
          f"(the strips proposal needs this to exceed the halo tax: 1.067x at 2 strips, 1.10x at 3)")
    print("\nNOT checked: the strip geometry itself (needs plans built at the strip height and the "
          "halo tax paid); the picture; whether the app's writer can consume two streams")

    if args.emit:
        import os as _bos, sys as _bsys
        _HERE = _bos.path.dirname(_bos.path.abspath(__file__))
        _ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
        for _p in (_ROOT, _HERE):
            if _p not in _bsys.path:
                _bsys.path.insert(0, _p)
        from harness import Population, Record, Store
        import tensorrt as trt
        store = Store()
        pop = Population.of("rounds", [f"round-{i + 1}" for i in range(len(concurrent))],
                            label=f"{args.contexts} independent contexts of one plan, "
                                  f"{args.width}x{args.height}, no encoder")
        store.emit(Record(value=round(ratio, 3), unit="x", statistic="median", population=pop,
                          instrument="wall clock around one device synchronize, N executions per "
                                     "stream, one GPU job at a time",
                          config=f"trt {trt.__version__}, {torch.cuda.get_device_name(0)}, "
                                 f"{args.contexts} engines from one plan, {args.width}x{args.height}, "
                                 f"iters {args.iters}",
                          session=store.session, run="C-strips-overlap",
                          artifact=hashlib.sha256(plan_bytes).hexdigest(),
                          extra={"of": "concurrency_ratio", "artifact_sha256":
                                 hashlib.sha256(plan_bytes).hexdigest(),
                                 "serial_ms": round(s_med, 2), "concurrent_ms": round(c_med, 2)}))
        store.dump(args.emit)
        print(f"records -> {args.emit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
