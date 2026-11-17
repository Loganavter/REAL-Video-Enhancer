"""Export the compact SR net to ONNX at a real frame size and build a TensorRT plan from it.

This is the `--engine` half of `bench_engine.py` without torch_tensorrt: the ledger has a row for
exactly this path ("same, built from ONNX instead of torch-tensorrt: 50.3 ms at 1920x1080,
tiling none"), so a number taken here is comparable to a recorded one instead of being a new
instrument.

The model is rebuilt from the checkpoint's own keys (`body.N.weight`), which is the same rule
`scripts/measure_residual.py` uses: no spandrel, no safetensors, no assumptions about layer order.

Usage:
  python3 scripts/build_plan.py <model.pth> <out.plan> [--width 1920] [--height 1080]
                                [--fp16] [--no-fp16] [--workspace-gib 8]
"""
from __future__ import annotations

import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class Compact(nn.Module):
    def __init__(self, state: dict) -> None:
        super().__init__()
        self.body = nn.ModuleList()
        i = 0
        while (key := f"body.{i}.weight") in state:
            w = state[key]
            self.body.append(nn.Conv2d(w.shape[1], w.shape[0], 3, 1, 1) if w.dim() == 4
                             else nn.PReLU(num_parameters=w.shape[0]))
            i += 1
        self.scale = int(round((state[f"body.{i - 1}.weight"].shape[0] / 3) ** 0.5))
        self.upsampler = nn.PixelShuffle(self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clamp(0.0, 1.0)
        for layer in self.body:
            out = layer(out)
        return self.upsampler(out) + F.interpolate(x, scale_factor=self.scale, mode="nearest")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("plan")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--workspace-gib", type=int, default=8)
    ap.add_argument("--onnx", default="")
    ap.add_argument("--dynamic", action="store_true", help="export with dynamic H/W (no fusion-friendly static shape)")
    ap.add_argument("--parse", default="", help="skip the export and parse this ONNX instead (e.g. a QDQ model)")
    ap.add_argument("--detailed", action="store_true", help="build with DETAILED profiling verbosity (tactic names)")
    args = ap.parse_args()

    if args.parse:
        return build_from_onnx(args.parse, args.plan, args.width, args.height, args.fp16,
                               args.workspace_gib, args.dynamic, args.detailed)

    state = torch.load(args.model, map_location="cpu", weights_only=True)
    state = state.get("params", state)
    model = Compact(state)
    model.load_state_dict(state, strict=True)
    model.eval()

    dummy = torch.zeros(1, 3, args.height, args.width)
    onnx_path = args.onnx or (args.plan + ".onnx")
    with torch.no_grad():
        dyn = ({"input": {2: "h", 3: "w"}, "output": {2: "oh", 3: "ow"}} if args.dynamic else None)
        torch.onnx.export(model, dummy, onnx_path, opset_version=17,
                          input_names=["input"], output_names=["output"], dynamic_axes=dyn)
    print(f"onnx -> {onnx_path} (input 1x3x{args.height}x{args.width})")

    return build_from_onnx(onnx_path, args.plan, args.width, args.height, args.fp16,
                           args.workspace_gib, args.dynamic, args.detailed)


def build_from_onnx(onnx_path: str, plan_path: str, width: int, height: int, fp16: bool,
                    workspace_gib: int, dynamic: bool, detailed: bool = False) -> int:
    import onnx
    import tensorrt as trt
    m = onnx.load(onnx_path)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(net, logger)
    if not parser.parse(m.SerializeToString()):
        for e in range(parser.num_errors):
            print("onnx parse error:", parser.get_error(e))
        return 1

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib << 30)
    if fp16 and builder.platform_has_fast_fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)
    # the app and the ledger's builds run at optimization level 5 with 8 timing iterations; a build
    # without them picks worse tactics and looks 2x slower than the recorded engine
    if detailed:
        cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    cfg.builder_optimization_level = 5
    cfg.avg_timing_iterations = 8
    # pin the shape: the ONNX is dynamic so one plan serves any frame, but the ledger's numbers are
    # at a fixed 1920x1080 and an unconstrained profile lets the builder choose bad tactics
    if dynamic:
        prof = builder.create_optimization_profile()
        prof.set_shape("input", (1, 3, height, width), (1, 3, height, width), (1, 3, height, width))
        cfg.add_optimization_profile(prof)

    t0 = __import__("time").time()
    plan = builder.build_serialized_network(net, cfg)
    if plan is None:
        print("build failed")
        return 1
    data = bytes(plan)
    with open(plan_path, "wb") as fh:
        fh.write(data)
    print(f"plan -> {plan_path}  {len(data)} bytes  {__import__('time').time() - t0:.1f}s  "
          f"fp16={fp16 and builder.platform_has_fast_fp16}  trt {trt.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
