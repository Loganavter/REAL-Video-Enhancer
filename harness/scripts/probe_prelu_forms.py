"""B3: at C=32 the PReLU is left outside the convolution. Does a ReLU-shaped rewrite fuse?

Three forms of the same activation, same width, same builder settings, same graph geometry:

  prelu    nn.PReLU(ch)                  what the model ships
  affine   a*x + (1-a)*relu(x)           E17's rewrite, measured at C=64 and unmeasured at 32
  leaky    nn.LeakyReLU(slope)           scalar-slope control: the shape an int8 epilogue can take

For each: fused myelin layer count, reformat/pointwise/conv counts, total ms/frame over interleaved
rounds, and the tactic name of the body convolutions. The question is not "is the rewrite better" but
"does ANY of the three get the epilogue inside the kernel at 32 channels", because the fusion break -
not the arithmetic - is what eats the MAC saving (E30a, reproduced four times).

The whole 16-block body is used, not one block: a single-block probe chooses a different tactic family
than the engine does (2.06x apart at C=32, B2 confound in queue.yaml), so a block verdict does not
transfer. Geometry and export path are `make_width_probe.py`'s, so a number here is comparable to the
recorded C=64/48/32 builds (50.2-56.5 / 42.4 / 40.6-41.4 ms).

Usage (one GPU job at a time):
  flock -n /var/tmp/opencode/gpu.lock ~/.cache/opencode/venvs/trt310/bin/python \
      harness/scripts/probe_prelu_forms.py --channels 32 --emit /tmp/opencode/b3-records.jsonl

  --channels 64        the control width, where fusion already works
  --forms prelu,affine,leaky
  --rounds 3 --iters 25
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_act(form: str, ch: int) -> nn.Module:
    if form == "prelu":
        return nn.PReLU(ch)
    if form == "leaky":
        return nn.LeakyReLU(0.25)
    if form == "affine":
        return Affine(ch)
    raise SystemExit(f"unknown form {form}")


class Affine(nn.Module):
    """PReLU written as a*x + (1-a)*relu(x): the same function, a different ONNX shape."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.full((1, ch, 1, 1), 0.25))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a * x + (1.0 - self.a) * torch.relu(x)


class Probe(nn.Module):
    """head conv -> n body convs (C->C, act) -> tail conv -> pixelshuffle -> + nearest."""

    def __init__(self, ch: int, body: int, scale: int, form: str) -> None:
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(3, ch, 3, 1, 1)
        self.act0 = make_act(form, ch)
        self.body = nn.ModuleList()
        for _ in range(body):
            self.body.append(nn.Conv2d(ch, ch, 3, 1, 1))
            self.body.append(make_act(form, ch))
        self.tail = nn.Conv2d(ch, 3 * scale * scale, 3, 1, 1)
        self.upsampler = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clamp(0.0, 1.0)
        out = self.act0(self.head(out))
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(self.tail(out))
        return out + F.interpolate(x, scale_factor=self.scale, mode="nearest")


def export(form: str, ch: int, body: int, h: int, w: int, path: str) -> None:
    torch.manual_seed(0)
    m = Probe(ch, body, 2, form).eval()
    dummy = torch.zeros(1, 3, h, w)
    with torch.no_grad():
        torch.onnx.export(m, dummy, path, opset_version=17,
                          input_names=["input"], output_names=["output"])


def build(onnx_path: str, plan_path: str, h: int, w: int, fp16: bool = True,
          workspace_gib: int = 8) -> dict:
    import onnx
    import tensorrt as trt

    with open(onnx_path, "rb") as fh:
        model = onnx.load(onnx_path)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(net, logger)
    if not parser.parse(model.SerializeToString()):
        return {"error": [f"parse: {parser.get_error(i)}" for i in range(parser.num_errors)]}

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib << 30)
    if fp16 and builder.platform_has_fast_fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    cfg.builder_optimization_level = 5
    cfg.avg_timing_iterations = 8

    t0 = time.time()
    blob = builder.build_serialized_network(net, cfg)
    if blob is None:
        return {"error": ["build returned None"]}
    data = bytes(blob)
    with open(plan_path, "wb") as fh:
        fh.write(data)
    return {"bytes": len(data), "build_s": round(time.time() - t0, 1),
            "sha256": hashlib.sha256(data).hexdigest()}


def load_engine(plan_path: str):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.ERROR)
    with open(plan_path, "rb") as fh:
        engine = trt.Runtime(logger).deserialize_cuda_engine(fh.read())
    if engine is None:
        raise SystemExit(f"{plan_path}: runtime refused the plan")
    return engine


def graph(engine) -> dict:
    """Layer census + the tactic of every body convolution, from the DETAILED inspector."""
    import tensorrt as trt
    insp = engine.create_engine_inspector()
    info = insp.get_engine_information(trt.LayerInformationFormat.JSON)
    parsed = json.loads(info) if isinstance(info, str) else info
    rows = parsed.get("Layers", parsed) if isinstance(parsed, dict) else parsed
    names = [str(r.get("Name", "")) if isinstance(r, dict) else str(r) for r in rows]
    tactics = [str(r.get("TacticName", "")) for r in rows if isinstance(r, dict)]
    conv_tactics = sorted({t for t in tactics if "xmma" in t or "scudnn" in t or "winograd" in t
                           or "__myl_" in t})
    return {"total": len(names),
            "fused_mylin": sum(n.startswith("__myl_") for n in names),
            "reformat": sum("Reformat" in n for n in names),
            "pointwise": sum("PointWise" in n or n.startswith("PWN") for n in names),
            "conv": sum(("Conv" in n or "Gemm" in n or "xmma" in n.lower()) for n in names),
            "conv_tactics": conv_tactics[:6]}


def time_plan(engine, h: int, w: int, iters: int, profiler_out: dict | None = None) -> float:
    import tensorrt as trt
    ctx = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    io = {}
    for n in names:
        io[n] = torch.zeros(tuple(engine.get_tensor_shape(n)), dtype=torch.float16, device="cuda") \
            if engine.get_tensor_dtype(n) == trt.DataType.HALF else \
            torch.zeros(tuple(engine.get_tensor_shape(n)), device="cuda")
        ctx.set_tensor_address(n, io[n].data_ptr())

    class P(trt.IProfiler):
        def __init__(self) -> None:
            trt.IProfiler.__init__(self)
            self.ms: dict[str, float] = {}

        def report_layer_time(self, name, ms) -> None:      # noqa: N802 - TRT API
            self.ms[name] = self.ms.get(name, 0.0) + float(ms)

    prof = P()
    if profiler_out is not None:
        ctx.profiler = prof
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(6):
            ctx.execute_async_v3(stream.cuda_stream)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            ctx.execute_async_v3(stream.cuda_stream)
        e.record()
        torch.cuda.synchronize()
    if profiler_out is not None:
        profiler_out.update({k: v / iters for k, v in prof.ms.items() if v > 0})
    return s.elapsed_time(e) / iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--forms", default="prelu,affine,leaky")
    ap.add_argument("--body", type=int, default=16)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--workspace-gib", type=int, default=8)
    ap.add_argument("--out", default="/tmp/opencode/b3")
    ap.add_argument("--emit", default="")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    forms = [f.strip() for f in args.forms.split(",") if f.strip()]
    built: dict[str, dict] = {}

    for form in forms:
        stem = f"{args.out}/c{args.channels}_{form}"
        onnx_path, plan_path = stem + ".onnx", stem + ".plan"
        if not os.path.exists(plan_path):
            export(form, args.channels, args.body, args.height, args.width, onnx_path)
            info = build(onnx_path, plan_path, args.height, args.width,
                         workspace_gib=args.workspace_gib)
            if "error" in info:
                print(f"{form}: BUILD FAILED {info['error'][:2]}")
                continue
        else:
            info = {"bytes": os.path.getsize(plan_path), "reused": True,
                    "sha256": hashlib.sha256(open(plan_path, "rb").read()).hexdigest()}
        engine = load_engine(plan_path)
        g = graph(engine)
        built[form] = {**info, "plan": plan_path, "engine": engine, "graph": g}
        print(f"{form}: built {info.get('build_s', 'reused')}s  graph {g['total']} layers, "
              f"fused {g['fused_mylin']}, reformat {g['reformat']}, pointwise {g['pointwise']}")

    if not built:
        return 1

    rounds: dict[str, list[float]] = {f: [] for f in built}
    profiles: dict[str, dict] = {}
    for r in range(1, args.rounds + 1):
        for form, b in built.items():
            prof = {} if r == 1 else None          # per-layer from the first round only
            ms = time_plan(b["engine"], args.height, args.width, args.iters, prof)
            rounds[form].append(ms)
            if prof:
                profiles[form] = prof
            print(f"round {r}: {form:7s} {ms:7.2f} ms/frame")

    print()
    print(f"{'form':7s} {'ms (median r>=2)':>18s} {'fused':>6s} {'pw':>4s} {'rf':>4s} {'conv':>5s}")
    summary = {}
    for form in built:
        vals = rounds[form]
        ms = statistics.median(vals[1:]) if len(vals) > 1 else vals[0]
        summary[form] = ms
        g = built[form]["graph"]
        print(f"{form:7s} {ms:18.2f} {g['fused_mylin']:6d} {g['pointwise']:4d} "
              f"{g['reformat']:4d} {g['conv']:5d}   {g['conv_tactics'][:1]}")

    # per-layer: where the body time goes, for the widest gap between forms
    for form, prof in profiles.items():
        body = sorted(((ms, n) for n, ms in prof.items() if "Reformat" not in n),
                      reverse=True)[:5]
        print(f"\n{form}: top layers (ms/frame)")
        for ms, n in body:
            print(f"  {ms:6.2f}  {n[:100]}")

    import tensorrt as trt

    print(f"\ninstrument: trt {trt.__version__}, torch {torch.__version__}, "
          f"{torch.cuda.get_device_name(0)}, one GPU job, {args.width}x{args.height}, "
          f"fp16, opt 5, iters {args.iters}, workspace {args.workspace_gib} GiB")
    print(f"NOT checked: picture (random weights); the app's own build path; "
          f"per-channel slope values of the shipped model")

    if args.emit:
        import os as _bos, sys as _bsys
        _HARNESS_ROOT = _bos.path.dirname(_bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
        if _HARNESS_ROOT not in _bsys.path:
            _bsys.path.insert(0, _HARNESS_ROOT)
        from harness import Population, Record, Store
        import tensorrt as trt
        store = Store()
        for form, b in built.items():
            vals = rounds[form]
            pop = Population.of("bench rounds", [f"round-{i + 1}" for i in range(len(vals))],
                                label=f"C={args.channels} {form}: whole 16-block body, "
                                      f"{args.width}x{args.height}, synthesized random weights")
            store.emit(Record(
                value=round(statistics.median(vals[1:]) if len(vals) > 1 else vals[0], 2),
                unit="ms", statistic="median", population=pop,
                instrument=f"torch.cuda.Event around {args.iters} executions after 6 warmups, "
                           f"one GPU job at a time",
                config=f"trt {trt.__version__}, {torch.cuda.get_device_name(0)}, fp16, opt 5, "
                       f"avg_timing_iters 8, workspace {args.workspace_gib} GiB, static "
                       f"{args.width}x{args.height}, form={form}, C={args.channels}",
                session=store.session, run=f"B3-C{args.channels}",
                artifact=b.get("sha256", ""),
                extra={"of": "frame_time", "quantity": "speed", "graph": b["graph"],
                       "artifact_sha256": b.get("sha256", ""),
                       "rounds": [round(v, 2) for v in vals]}))
        store.dump(args.emit)
        print(f"records -> {args.emit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
