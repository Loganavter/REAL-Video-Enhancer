"""E33: where does the int8 body's 55-60% of peak go - tiling, tactic family, or the shape?

The int8 body is 31.2 ms measured against a 26.9 ms floor at the 91 TOPS the convs report, and the
card's measured int8 peak is ~147 TOPS with B in the K-major layout. Nothing else on the board
closes that gap: fusion does not help a compute-bound GEMM, and the epilogue is worth 1.40x at most
(E27). So the question is whether ANY builder configuration raises the per-convolution rate above
~120 TOPS, and if not, what the honest ceiling is.

What it measures, per configuration: total ms/frame, per-layer ms from trt.IProfiler, the tactic
name of every convolution, and the achieved TOPS of each (2*C*C*9*W*H / ms). Interleaved rounds
where more than one plan is alive.

Usage (one GPU job at a time; a build at opt5/iters8 on 1080p costs ~10-20 min):
  F=~/.cache/opencode/venvs/trt310/bin/python
  # measure an existing plan: no build, ~1 min
  flock -n /var/tmp/opencode/gpu.lock $F scripts/probe_int8_tactics.py --plan /tmp/opencode/compact2x_1080p_int8.plan
  # rebuild from the same QDQ graph at another tiling level
  flock -n /var/tmp/opencode/gpu.lock $F scripts/probe_int8_tactics.py \
      --qdq /tmp/opencode/compact2x_1080p.qdq.onnx --tiling none --build-only
"""
from __future__ import annotations

import argparse
import hashlib
import re
import json
import os
import statistics
import sys
import time

import torch

def build_plan(qdq: str, out: str, *, tiling: str, fp16: bool = True, workspace_gib: int = 8,
               iters: int = 8) -> dict:
    import onnx
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # EXPLICIT_BATCH only: int8 comes from the Q/DQ nodes in the graph. Setting BuilderFlag.INT8
    # would ask for implicit quantisation with no calibrator, which is a different (and wrong) path.
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(net, logger)
    with open(qdq, "rb") as fh:
        model = onnx.load(qdq)
    if not parser.parse(model.SerializeToString()):
        return {"error": [f"parse: {parser.get_error(i)}" for i in range(parser.num_errors)]}
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib << 30)
    if fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.builder_optimization_level = 5
    cfg.avg_timing_iterations = iters
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    level = {"none": trt.TilingOptimizationLevel.NONE,
             "fast": trt.TilingOptimizationLevel.FAST,
             "moderate": trt.TilingOptimizationLevel.MODERATE,
             "full": trt.TilingOptimizationLevel.FULL}[tiling]
    cfg.tiling_optimization_level = level
    t0 = time.time()
    blob = builder.build_serialized_network(net, cfg)
    if blob is None:
        return {"error": ["build returned None"]}
    data = bytes(blob)
    with open(out, "wb") as fh:
        fh.write(data)
    return {"bytes": len(data), "build_s": round(time.time() - t0, 1),
            "sha256": hashlib.sha256(data).hexdigest(), "tiling": tiling}


def measure(plan_path: str, w: int, h: int, iters: int = 20, rounds: int = 3) -> dict:
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.ERROR)
    with open(plan_path, "rb") as fh:
        engine = trt.Runtime(logger).deserialize_cuda_engine(fh.read())
    if engine is None:
        raise SystemExit(f"{plan_path}: the runtime refused this plan")
    ctx = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    for n in names:
        dt = engine.get_tensor_dtype(n)
        tdt = {trt.DataType.HALF: torch.float16, trt.DataType.INT8: torch.int8,
               trt.DataType.FLOAT: torch.float32}.get(dt, torch.float32)
        t = torch.zeros(tuple(engine.get_tensor_shape(n)), dtype=tdt, device="cuda")
        ctx.set_tensor_address(n, t.data_ptr())

    class P(trt.IProfiler):
        def __init__(self) -> None:
            trt.IProfiler.__init__(self)
            self.ms: dict[str, float] = {}

        def report_layer_time(self, name, ms) -> None:      # noqa: N802 - TRT API
            self.ms[name] = self.ms.get(name, 0.0) + float(ms)

    prof = P()
    ctx.profiler = prof
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(6):
            ctx.execute_async_v3(stream.cuda_stream)
        torch.cuda.synchronize()
        times = []
        for _ in range(rounds):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(iters):
                ctx.execute_async_v3(stream.cuda_stream)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e) / iters)

    insp = engine.create_engine_inspector()
    info = insp.get_engine_information(trt.LayerInformationFormat.JSON)
    parsed = json.loads(info) if isinstance(info, str) else info
    rows = parsed.get("Layers", parsed) if isinstance(parsed, dict) else parsed
    layers = []
    for r in rows:
        if not isinstance(r, dict):
            layers.append({"name": str(r), "tactic": ""})
            continue
        layers.append({"name": str(r.get("Name", "")), "tactic": str(r.get("TacticName", "")),
                       "precision": str(r.get("Precision", ""))})
    per_layer = {k: v / (iters * rounds) for k, v in prof.ms.items() if v > 0}
    return {"rounds": [round(t, 2) for t in times],
            "median": round(statistics.median(times[1:]) if rounds > 1 else times[0], 2),
            "layers": layers, "per_layer": per_layer,
            "total_layers": len(layers),
            "fused_mylin": sum(l["name"].startswith("__myl_") for l in layers),
            "reformat": sum("Reformat" in l["name"] for l in layers),
            "pointwise": sum("PointWise" in l["name"] or l["name"].startswith("PWN") for l in layers),
            "sha256": hashlib.sha256(open(plan_path, "rb").read()).hexdigest()}


CONV_RX = re.compile(r"conv2d_\d+|/conv/Conv|_conv_|Conv2d", re.I)
NON_CONV_RX = re.compile(r"PWN|PointWise|Reformat|MaxMin|Cast|TranResh|MoveAdd|Slice|Concat", re.I)


def conv_rate(name: str, ms: float, w: int, h: int) -> float | None:
    """TOPS of one layer, only if the layer IS a 64->64 3x3 convolution at frame size.

    The first version of this filter only excluded Reformat, which put the int8 PReLU passes and fused
    arithmetic layers on the same list and produced 26886 TOPS from a pointwise pass - a wrong number
    that looked like a finding. A pointwise layer does not have the conv's FLOPs, so it cannot be
    divided by them.
    """
    if not ms or NON_CONV_RX.search(name) or not CONV_RX.search(name):
        return None
    return 2.0 * 64 * 64 * 9 * w * h / (ms * 1e-3) / 1e12


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="")
    ap.add_argument("--qdq", default="")
    ap.add_argument("--tiling", default="full", choices=("none", "fast", "moderate", "full"))
    ap.add_argument("--out-plan", default="")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--emit", default="")
    args = ap.parse_args()

    built = {}
    if args.qdq:
        out = args.out_plan or f"/tmp/opencode/int8_tiling-{args.tiling}.plan"
        info = build_plan(args.qdq, out, tiling=args.tiling)
        print(f"build tiling={args.tiling}: {info}")
        if "error" in info:
            return 1
        built[args.tiling] = out
        if args.build_only:
            return 0

    if args.plan:
        built["as-built"] = args.plan
    if not built:
        raise SystemExit("give --plan and/or --qdq")

    results = {}
    for label, path in built.items():
        m = measure(path, args.width, args.height, args.iters, args.rounds)
        results[label] = m
        body = {n: ms for n, ms in m["per_layer"].items() if conv_rate(n, ms, args.width,
                                                                      args.height) is not None}
        rates = sorted((conv_rate(n, ms, args.width, args.height), n) for n, ms in body.items())
        print(f"\n== {label}: {m['median']:.2f} ms/frame  rounds {m['rounds']}")
        print(f"   layers {m['total_layers']}: myelin {m['fused_mylin']}, reformat "
              f"{m['reformat']}, pointwise {m['pointwise']}")
        print(f"   convs with a rate: {len(rates)}  "
              f"min {rates[0][0]:.1f} TOPS  max {rates[-1][0]:.1f} TOPS  "
              f"sum {sum(r for r, _ in rates):.1f}" if rates else "   no per-layer conv times")
        for r, n in rates[:4]:
            print(f"   {r:6.1f} TOPS  {n[:70]}")
        tactics = sorted({l["tactic"] for l in m["layers"] if l["tactic"]})
        for t in tactics[:6]:
            print(f"   tactic: {t[:110]}")

    print(f"\nfloor: the 16-conv body at the shipped 91 TOPS is 26.9 ms; at 120 TOPS it is 20.4 ms; "
          f"at the card's measured 147.4 TOPS it is 16.6 ms")
    print("NOT checked: whether a faster tactic keeps the picture (this plan's QDQ is the accuracy "
          "instrument and it is not re-certified here); the app's own build path")

    if args.emit:
        sys.path.insert(0, "scripts")
        from harness import Population, Record, Store
        import tensorrt as trt
        store = Store()
        for label, m in results.items():
            pop = Population.of("plan repeats", [f"round-{i + 1}" for i in range(len(m["rounds"]))],
                                label=f"int8 plan {label}: {args.width}x{args.height}, one process, "
                                      f"no encoder")
            store.emit(Record(value=m["median"], unit="ms", statistic="median", population=pop,
                              instrument=f"torch.cuda.Event around {args.iters} executions, one GPU "
                                         f"job at a time",
                              config=f"trt {trt.__version__}, {torch.cuda.get_device_name(0)}, "
                                     f"int8 QDQ, tiling={label}, opt 5, iters {args.iters}",
                              session=store.session, run=f"E33-{label}", artifact=m["sha256"],
                              extra={"of": "frame_time", "quantity": "speed",
                                     "artifact_sha256": m["sha256"], "rounds": m["rounds"],
                                     "graph": {"total": m["total_layers"],
                                               "fused_mylin": m["fused_mylin"],
                                               "reformat": m["reformat"],
                                               "pointwise": m["pointwise"]}}))
            body = {n: ms for n, ms in m["per_layer"].items()
                    if conv_rate(n, ms, args.width, args.height) is not None}
            if body:
                tops = statistics.median(conv_rate(n, ms, args.width, args.height)
                                         for n, ms in body.items())
                store.emit(Record(value=round(tops, 1), unit="TOPS", statistic="median",
                                  population=Population.of("body convolutions", list(body),
                                                           label=f"per-conv rate, {label}"),
                                  instrument="trt.IProfiler per-layer times",
                                  config=f"trt {trt.__version__}, int8 64->64 3x3 at "
                                         f"{args.width}x{args.height}, tiling={label}",
                                  session=store.session, run=f"E33-{label}",
                                  artifact=m["sha256"],
                                  extra={"of": "per_conv_rate", "artifact_sha256": m["sha256"]}))
        store.dump(args.emit)
        print(f"records -> {args.emit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
