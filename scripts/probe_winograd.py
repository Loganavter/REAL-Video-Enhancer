"""E26: is Winograd available and faster than implicit GEMM for our body convolution?

One process, one mode, one shape. Run it once per mode from the driver shell so no
cuDNN cache or benchmark result survives between modes (the app builds a fresh
engine per run, so a fresh process is the honest unit).

The shape is the model's own body layer: 3x3, 64 -> 64, 1920x1080, fp16 NHWC, and
the ledger's measured cost for it is ~3.05 ms per convolution (16 of them make up
48.72 ms of the 56.8 ms frame).

Modes:
  heuristic  torch default: cuDNN picks with cudnnGetConvolutionForwardAlgorithm
  search     cudnnFindConvolutionForwardAlgorithmEx (tries every algorithm,
             Winograd included); if this is faster than heuristic, the default
             choice is not the best one available
  fp32       NCHW fp32 control: Winograd's advantage is largest here, so if it
             does not show up in fp32 either, the algorithm is not in play at all

Usage: probe_winograd.py <mode> [--height 1080] [--width 1920] [--iters 30]
Prints one line: "mode=<m> ms=<x> gflops=<y> matches_default=<bool>"
"""

import argparse
import statistics
import sys

import torch
import torch.nn.functional as F


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["heuristic", "search", "fp32", "fp32-search"])
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float32 if args.mode.startswith("fp32") else torch.float16
    memory_format = torch.channels_last if dtype == torch.float16 else torch.contiguous_format

    # one layer of the engine: 3x3, C -> C, no bias (the builder folds the bias into PReLU)
    x = torch.randn(1, args.channels, args.height, args.width, device=device, dtype=dtype)
    w = torch.randn(args.channels, args.channels, 3, 3, device=device, dtype=dtype) * 0.05
    x = x.to(memory_format=memory_format)
    w = w.to(memory_format=memory_format)

    search = args.mode.endswith("search")
    torch.backends.cudnn.benchmark = search
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = False

    def run():
        return F.conv2d(x, w, padding=1)

    reference = run().float().clone()
    # 0.6 s of load, not five iterations: an idle Ada part sits at 210 MHz and the
    # first hundred milliseconds of a short loop are clock ramp, not convolution
    import time
    ramp = time.time()
    while time.time() - ramp < 0.6:
        run()
    torch.cuda.synchronize()

    samples = []
    for _ in range(args.iters):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        out = run()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
        del out

    # fp16 Winograd rounds differently from implicit GEMM; report it instead of hiding it
    drift = (run().float() - reference).abs().max().item()
    ms = statistics.median(samples)
    flops = 2.0 * args.channels * args.channels * 9 * args.height * args.width
    print(f"mode={args.mode} ms={ms:.3f} min={min(samples):.3f} gflops={flops / ms / 1e9:.1f} "
          f"max_abs_drift_vs_first={drift:.4f} dtype={dtype} layout={memory_format}")
    if args.profile:
        # the NAME of the kernel is the answer to "was Winograd even considered": a timing
        # alone cannot tell a fast Winograd from a well-tuned implicit GEMM
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                run()
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=8))
    return 0


if __name__ == "__main__":
    sys.exit(main())
