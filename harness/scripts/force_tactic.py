"""B2: is there ANY tactic for Conv(C->C, 3x3) + PReLU that myelin will fuse at C=32?

The selector cannot filter by kernel name - IAlgorithmVariant exposes `implementation` (an int) and a
64-bit tactic hash, no name - but it can restrict the build to ONE choice index at a time, and the
built engine names its layers: `__myl_*` / `CorrMulLtSele` when the PReLU went into the same kernel,
a bare `Conv` plus a `PWN(...PRelu)` when it did not. So the fused variant is found by scanning
indices and reading the names, not by guessing.

Two graph sizes, on purpose:
  --graph block  one Conv+PReLU at 1920x1080. Cheap, but a single block chooses a DIFFERENT tactic
                 family than the engine does at C=32 (winograd vs indexed_wo_smem, 2.06x apart),
                 so a verdict from it alone does not transfer - see the confound note in queue.yaml.
  --graph body   N body blocks of the same shape. Detects fusion per layer; 4 blocks build ~4x
                 faster than 16, and every body block has the same shape, so the scan is run at 4
                 and the winner confirmed at 16.

Usage (one GPU job at a time):
  F=~/.cache/opencode/venvs/trt310/bin/python
  flock -n /var/tmp/opencode/gpu.lock $F harness/scripts/force_tactic.py --channels 32 --graph block --scan
  flock -n /var/tmp/opencode/gpu.lock $F harness/scripts/force_tactic.py --channels 32 --graph body --body 4 --scan
  flock -n /var/tmp/opencode/gpu.lock $F harness/scripts/force_tactic.py --channels 32 --graph body --body 16 --index 7
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

OUT = "/tmp/opencode/b2"


def make_onnx(kind: str, channels: int, body: int, h: int, w: int) -> str:
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{kind}_c{channels}_b{body}_{h}x{w}.onnx"
    if os.path.exists(path):
        return path

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)
            self.act = nn.PReLU(channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.act(self.conv(x))

    class Body(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Conv2d(3, channels, 3, padding=1)
            self.blocks = nn.ModuleList([Block() for _ in range(body)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.head(x)
            for b in self.blocks:
                out = b(out)
            return out

    torch.manual_seed(0)
    m = (Block() if kind == "block" else Body()).eval()
    with torch.no_grad():
        torch.onnx.export(m, (torch.rand(1, 3 if kind == "body" else channels, h, w),), path,
                          opset_version=17, input_names=["input"], output_names=["output"],
                          do_constant_folding=True, dynamo=False)
    return path


class Pick:
    """Allow every choice except on the target layer, where only `index` is allowed.

    Built as a plain object then registered, because trt.IAlgorithmSelector is an abstract base
    whose SWIG trampoline is picky about attribute names in older bindings.
    """

    def __init__(self, index: int | None) -> None:
        self.index = index

    def select_algorithms(self, context, choices):          # noqa: N802 - TRT API
        if self.index is None or self.index >= len(choices):
            return list(range(len(choices)))
        return [self.index]

    def report_algorithms(self, context, choices):          # noqa: N802 - TRT API
        pass


def build(onnx_path: str, index: int | None, workspace_gib: int = 8, fp16: bool = True):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(net, logger)
    with open(onnx_path, "rb") as fh:
        if not parser.parse(fh.read()):
            return None, [f"parse: {parser.get_error(i)}" for i in range(parser.num_errors)]
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib << 30)
    if fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.builder_optimization_level = 5
    cfg.avg_timing_iterations = 8
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if index is not None:
        sel = Pick(index)
        # trt.IAlgorithmSelector cannot be instantiated directly; wrap via the trampoline
        cfg.algorithm_selector = _Selector(sel)
    try:
        blob = builder.build_serialized_network(net, cfg)
    except Exception as exc:                                # noqa: BLE001 - a tactic that cannot
        return None, [f"build raised: {exc}"]               # run is a legitimate answer
    if blob is None:
        return None, ["build returned None"]
    return bytes(blob), []


def _Selector(pick: Pick):
    import tensorrt as trt

    class Selector(trt.IAlgorithmSelector):
        def select_algorithms(self, context, choices):       # noqa: N802
            return pick.select_algorithms(context, choices)

        def report_algorithms(self, context, choices):       # noqa: N802
            pass

    return Selector()


def describe(blob: bytes, h: int, w: int, iters: int = 15) -> dict:
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.ERROR)
    engine = trt.Runtime(logger).deserialize_cuda_engine(blob)
    if engine is None:
        raise RuntimeError("runtime refused a plan the builder produced")
    insp = engine.create_engine_inspector()
    ctx = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    i_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    o_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
    dtype = torch.float16 if engine.get_tensor_dtype(i_name) == trt.DataType.HALF else torch.float32
    x = torch.zeros(tuple(engine.get_tensor_shape(i_name)), dtype=dtype, device="cuda")
    oshape = tuple(engine.get_tensor_shape(o_name))
    y = torch.zeros(oshape, dtype=dtype, device="cuda")
    ctx.set_tensor_address(i_name, x.data_ptr())
    ctx.set_tensor_address(o_name, y.data_ptr())
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(5):
            ctx.execute_async_v3(stream.cuda_stream)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            ctx.execute_async_v3(stream.cuda_stream)
        e.record()
        torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters

    info = insp.get_engine_information(trt.LayerInformationFormat.JSON)
    parsed = json.loads(info) if isinstance(info, str) else info
    rows = parsed.get("Layers", parsed) if isinstance(parsed, dict) else parsed
    layers = []
    for r in rows:
        if isinstance(r, dict):
            layers.append((str(r.get("Name", "")), str(r.get("TacticName", ""))))
        else:
            layers.append((str(r), ""))
    fused = sum(n.startswith("__myl_") or "CorrMulLtSele" in n or "CorrMulLtSele" in t
                for n, t in layers)
    pw = sum("PointWise" in n or n.startswith("PWN") or "PRelu" in n or "PRelu" in t
             for n, t in layers)
    conv_t = sorted({t for n, t in layers if t and ("xmma" in t or "scudnn" in t or "myl" in t)})
    return {"ms": round(ms, 3), "layers": len(layers), "fused": fused, "pointwise": pw,
            "conv_tactics": conv_t}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--graph", choices=("block", "body"), default="block")
    ap.add_argument("--body", type=int, default=4)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--limit", type=int, default=40, help="scan indices 0..limit-1")
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--emit", default="")
    args = ap.parse_args()

    onnx_path = make_onnx(args.graph, args.channels, args.body, args.height, args.width)
    print(f"graph: {onnx_path}")

    if args.index is not None:
        blob, errs = build(onnx_path, args.index)
        if blob is None:
            print(f"index {args.index}: build failed: {errs[:1]}")
            return 1
        d = describe(blob, args.height, args.width)
        print(f"index {args.index}: {d['ms']:.3f} ms  fused={d['fused']} pw={d['pointwise']} "
              f"layers={d['layers']}  {d['conv_tactics'][:2]}")
        return 0

    blob, errs = build(onnx_path, None)
    if blob is None:
        print(f"default build failed: {errs[:1]}")
        return 1
    d = describe(blob, args.height, args.width)
    print(f"default: {d['ms']:.3f} ms  fused={d['fused']} pw={d['pointwise']} layers={d['layers']}")
    for t in d["conv_tactics"]:
        print(f"   tactic {t}")

    rows = []
    for index in range(args.limit):
        blob, errs = build(onnx_path, index)
        if blob is None:
            rows.append({"index": index, "error": errs[0][:120]})
            print(f"index {index:3d}: build failed ({errs[0][:60]})")
            continue
        try:
            d = describe(blob, args.height, args.width, iters=10)
        except Exception as exc:                            # noqa: BLE001
            rows.append({"index": index, "error": f"ran but failed: {exc}"})
            print(f"index {index:3d}: ran but failed ({exc})")
            continue
        rows.append({"index": index, **d,
                     "sha256": hashlib.sha256(blob).hexdigest()})
        tag = "FUSED" if d["fused"] else "split"
        print(f"index {index:3d}: {d['ms']:7.3f} ms {tag:5s} pw={d['pointwise']} "
              f"{(d['conv_tactics'][0] if d['conv_tactics'] else '')[:70]}")

    fused_rows = [r for r in rows if r.get("fused")]
    print(f"\nscan: {len(rows)} indices, {len(fused_rows)} fused, "
          f"{sum(1 for r in rows if 'error' in r)} failed")
    if fused_rows:
        best = min(fused_rows, key=lambda r: r["ms"])
        print(f"best fused: index {best['index']} {best['ms']:.3f} ms "
              f"{best['conv_tactics'][:2]}")
    print(f"\nNOT checked: whether a forced tactic survives the app's own build path; "
          f"per-conv time split on the body graph (profiler is off in this scan)")

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
        pop = Population.of("forced tactic builds",
                            [f"index-{r['index']}" for r in rows if "ms" in r],
                            label=f"C={args.channels} {args.graph} b{args.body}, one tactic index "
                                  f"forced per build, {args.width}x{args.height}, fp16")
        for r in rows:
            if "ms" not in r:
                continue
            store.emit(Record(value=float(r["ms"]), unit="ms", statistic="single",
                              population=pop,
                              instrument="torch.cuda.Event around 10 executions after 5 warmups, "
                                         "one GPU job at a time",
                              config=f"trt {trt.__version__}, {torch.cuda.get_device_name(0)}, "
                                     f"forced choice index {r['index']}, opt 5, workspace 8 GiB",
                              session=store.session, run=f"B2-C{args.channels}-{args.graph}",
                              artifact=r.get("sha256", ""),
                              extra={"of": "frame_time", "quantity": "speed",
                                     "graph": {"total": r["layers"], "fused_mylin": r["fused"],
                                               "pointwise": r["pointwise"]},
                                     "conv_tactics": r["conv_tactics"]}))
        store.dump(args.emit)
        print(f"records -> {args.emit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
