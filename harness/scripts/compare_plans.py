"""Run two plans on the same frame and report the picture difference in levels.

The ledger's int8 claim has two halves: time (68.11 against 55.54 ms) and picture ("0.000% of pixels
above 8 levels against the FP16 engine, peak 4, PSNR 54.1 dB"). `bench_plan.py` does the first; this
does the second, with the same definition the project uses everywhere: the share of pixels whose WORST
channel is more than 8 levels off, plus the peak and PSNR.

The frame is named by its sha256, so the number belongs to a population instead of to a sentence.

Usage:
  python3 harness/scripts/compare_plans.py <ref.plan> <cand.plan> <frame.raw> --width W --height H [--emit FILE]
"""
from __future__ import annotations

import argparse
import hashlib
import sys

import numpy as np
import torch
import tensorrt as trt


def load(path: str, logger):
    with open(path, "rb") as fh:
        engine = trt.Runtime(logger).deserialize_cuda_engine(fh.read())
    if engine is None:
        raise SystemExit(f"{path}: the runtime refused the plan")
    return engine


def run(engine, x: torch.Tensor, width: int, height: int) -> torch.Tensor:
    ctx = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    ins = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    outs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
    dt = {trt.DataType.HALF: torch.float16, trt.DataType.FLOAT: torch.float32}[
        engine.get_tensor_dtype(ins[0])]
    inp = x.to(dt).contiguous()
    shape = tuple(engine.get_tensor_shape(outs[0]))
    out = torch.empty(shape, dtype=torch.float32 if engine.get_tensor_dtype(outs[0]) == trt.DataType.FLOAT
                      else torch.float16, device="cuda")
    ctx.set_tensor_address(ins[0], inp.data_ptr())
    ctx.set_tensor_address(outs[0], out.data_ptr())
    ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return out.float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("cand")
    ap.add_argument("frame")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--emit", default="")
    args = ap.parse_args()

    raw = np.fromfile(args.frame, dtype=np.uint8)
    if raw.size != args.width * args.height * 3:
        raise SystemExit(f"{args.frame}: {raw.size} bytes is not {args.width}x{args.height}x3")
    sha = hashlib.sha256(raw.tobytes()).hexdigest()
    x = torch.from_numpy(raw.reshape(args.height, args.width, 3).transpose(2, 0, 1)).float().div(255.0)[None].cuda()

    logger = trt.Logger(trt.Logger.ERROR)
    a = run(load(args.ref, logger), x, args.width, args.height)
    b = run(load(args.cand, logger), x, args.width, args.height)
    if a.shape != b.shape:
        raise SystemExit(f"output shapes differ: {tuple(a.shape)} vs {tuple(b.shape)}")

    d = (a - b).abs().mul(255.0)
    worst = d.amax(dim=1)                            # worst channel per pixel
    share = float((worst > 8).float().mean() * 100.0)
    peak = float(d.max())
    mean = float(d.mean())
    mse = float(((a - b) ** 2).mean())
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
    print(f"{args.frame} sha256 {sha[:16]}  {args.width}x{args.height}")
    print(f"share above 8 levels (worst channel): {share:.3f}%   peak {peak:.2f} levels   "
          f"mean {mean:.2f}   PSNR {psnr:.2f} dB")

    if args.emit:
        import os as _bos, sys as _bsys
        _HARNESS_ROOT = _bos.path.dirname(_bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
        if _HARNESS_ROOT not in _bsys.path:
            _bsys.path.insert(0, _HARNESS_ROOT)
        from harness import Population, Record, Store
        store = Store()
        pop = Population.of("pixels of one frame", [sha[:16]],
                            label=f"{args.frame.split('/')[-1]}: {args.cand.split('/')[-1]} vs "
                                  f"{args.ref.split('/')[-1]}, {args.width}x{args.height}")
        common = dict(population=pop, instrument="TRT execution of both plans, worst channel in levels",
                      config=f"trt {trt.__version__}, {torch.cuda.get_device_name(0)}",
                      session=store.session, run=f"run-compare-{args.cand.split('/')[-1][:-5]}",
                      extra={"artifact_sha256": sha, "of": "share", "quantity": "accuracy"})
        store.emit(Record(value=round(share, 4), unit="%", statistic="single", **common))
        store.emit(Record(value=round(peak, 2), unit="levels", statistic="max",
                          **{**common, "extra": {**common["extra"], "of": "peak"}}))
        store.emit(Record(value=round(psnr, 2), unit="dB", statistic="single",
                          **{**common, "extra": {**common["extra"], "of": "quality"}}))
        store.dump(args.emit)
        print(f"records -> {args.emit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
