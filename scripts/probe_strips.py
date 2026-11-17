"""C: does this card overlap two TensorRT executions at all? The strips idea rests on that alone.

The proposal is to cut the frame into horizontal strips, give each strip its own context and stream,
and hide the boundary behind concurrency instead of removing it with a halo. The halo tax is
arithmetic (2 strips 1.067x, 3 strips 1.10x, 4 strips 1.133x more work), so the idea only pays if
two executions actually overlap. That is the refutes clause in the queue and it needs no strips:

  a 56 ms execution already runs at 100% SM utilisation and 162 W of a 165 W cap (MODEL.md §3),
  so the question is whether a second execution on another stream makes the pair finish in less
  than twice the time.

Timing protocol: the same plan bytes deserialized into two independent engines, each with its own
context, on two streams. Both streams enqueue N executions, then one synchronize. The ratio
`concurrent / serial` is 1.0 for perfect overlap and 2.0 for none. Interleaved rounds.

Usage (one GPU job at a time):
  flock -n /var/tmp/opencode/gpu.lock ~/.cache/opencode/venvs/trt310/bin/python \
      scripts/probe_strips.py --plan /tmp/opencode/compact2x_1080p_fp16.plan
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
    import tensorrt as trt
    ctx = engine.create_execution_context()
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        dt = engine.get_tensor_dtype(n)
        tdt = {trt.DataType.HALF: torch.float16, trt.DataType.INT8: torch.int8,
               trt.DataType.FLOAT: torch.float32}.get(dt, torch.float32)
        t = torch.zeros(tuple(engine.get_tensor_shape(n)), dtype=tdt, device="cuda")
        ctx.set_tensor_address(n, t.data_ptr())
    return ctx


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


def time_serial(ctxs, streams, iters: int) -> float:
    with torch.cuda.stream(streams[0]):
        for _ in range(6):
            ctxs[0].execute_async_v3(streams[0].cuda_stream)
        for ctx in ctxs[1:]:
            for _ in range(6):
                ctx.execute_async_v3(streams[0].cuda_stream)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for ctx in ctxs:
        for _ in range(iters):
            ctx.execute_async_v3(streams[0].cuda_stream)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / (iters * len(ctxs))


def time_concurrent(ctxs, streams, iters: int) -> float:
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for ctx, stream in zip(ctxs, streams):
        with torch.cuda.stream(stream):
            for _ in range(iters):
                ctx.execute_async_v3(stream.cuda_stream)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / (iters * len(ctxs))


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
    ctxs = [bind(e, args.height, args.width) for e in engines]
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
    print(f"\nconcurrent/serial per-context time = {ratio:.3f}x "
          f"(1.00 = perfect overlap, 2.00 = none)")
    print(f"throughput gain from the second stream = {1.0 / ratio:.2f}x  "
          f"(the strips proposal needs this to exceed the halo tax)")
    print("\nNOT checked: the strip geometry itself (needs plans built at the strip height and the "
          "halo tax paid); the picture; whether the app's writer can consume two streams")

    if args.emit:
        sys.path.insert(0, "scripts")
        from harness import Population, Record, Store
        import tensorrt as trt
        store = Store()
        pop = Population.of("rounds", [f"round-{i + 1}" for i in range(len(concurrent))],
                            label=f"{args.contexts} independent contexts of one plan, "
                                  f"{args.width}x{args.height}, no encoder")
        store.emit(Record(value=round(ratio, 3), unit="x", statistic="median", population=pop,
                          instrument="torch.cuda.Event around N executions per stream, one "
                                     "synchronize, one GPU job at a time",
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
