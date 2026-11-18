"""Is there 2x in the tail conv's shape, or is it at its floor? Answers in seconds, not hours.

The tail convolution is 64 -> 12, 3x3, 1080p, fp16, and it runs at ~5 TFLOPS where the card offers ~44
(fp16 CUDA cores) and ~176 (tensor cores). The suspicion is tensor-core tile underfill: C_out = 12 does not
fill a 16-wide tile. This script needs no clip, no engine and no model weights - a conv's latency does not
depend on the values - so it is a smoke test to run BEFORE funding a row, not the row itself.

Arms: C_out in {12, 16, 32, 64} x {weights NCHW, weights channels_last} x {input NCHW, input NHWC-view}.
The C_out sweep is the discriminator: if 12 -> 16 is a large jump, the tile fill is the lever; if the curve
is flat, the 5 TFLOPS is the shape's floor at this size and the row should close.

Usage: python scripts/tail_conv_probe.py [--height 1080] [--width 1920] [--iters 10]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402


def timeit(fn, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--cin", type=int, default=64)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    dev = "cuda"
    torch.backends.cudnn.benchmark = True   # let cudnn pick; the shipped path does not set it, noted below
    x_nchw = torch.randn(1, args.cin, args.height, args.width, device=dev, dtype=torch.float16)
    x_nhwc = x_nchw.to(memory_format=torch.channels_last)
    flop_per_out = args.cin * 9 * 2

    rows = []
    print(f"{'C_out':>5} {'weights':>14} {'input':>11} {'ms':>8} {'TFLOPS':>8} {'ms/C_out':>9}")
    for cout in (12, 16, 32, 64):
        for wmode in ("nchw", "channels_last"):
            conv = torch.nn.Conv2d(args.cin, cout, 3, padding=1, bias=True).to(dev).half()
            if wmode == "channels_last":
                conv = conv.to(memory_format=torch.channels_last)
            for imode, x in (("nchw", x_nchw), ("nhwc-view", x_nhwc)):
                with torch.no_grad():
                    ms = timeit(lambda: conv(x), iters=args.iters)
                tflops = args.height * args.width * cout * flop_per_out / (ms * 1e9)
                rows.append(dict(cout=cout, weights=wmode, input=imode, ms=round(ms, 3),
                                 tflops=round(tflops, 2)))
                print(f"{cout:>5} {wmode:>14} {imode:>11} {ms:>8.3f} {tflops:>8.2f} {ms/cout:>9.4f}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"height": args.height, "width": args.width, "cin": args.cin,
                       "take": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": rows}, fh, indent=1)
        print(f"# -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
