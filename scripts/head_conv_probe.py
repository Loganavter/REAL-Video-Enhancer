"""Where do 4.65 ms go for a 7.17 GFLOP head convolution? Sweep the things a conv of this shape responds to.

The head is 3 -> 64, 3x3, at 1080p fp16: 7.17 GFLOP, and the app's phase counter puts it at 4.65-5.2 ms,
i.e. **1.5 TFLOPS** - 3.5% of this card's fp16 CUDA-core peak - while its own traffic floor (265 MB out +
12 MB in) is 0.96 ms. It is the second largest single line in the int8 frame after the sixteen hops and has
never been timed in isolation.

Arms: input as a contiguous NCHW tensor against a channels_last VIEW of the same memory (the lever that gave
the tail 2.5x), weights NCHW against channels_last, and the 3 -> 8 padded input the engine build already
uses so myelin can fuse the head with its PReLU. Latency does not depend on the values, so a random tensor is
the whole instrument.

Usage: python scripts/head_conv_probe.py [--height 1080] [--width 1920] [--iters 10]
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
    ap.add_argument("--out", type=int, default=64)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out-file", default="")
    args = ap.parse_args(argv)

    dev = "cuda"
    px = args.height * args.width
    flop = px * args.out * 27 * 2 / 1e9                    # GFLOP, 3 -> out, 3x3
    traffic = (px * args.out * 2 + px * 3 * 2) / 1e6        # MB, out fp16 + in 3ch fp16
    floor_ms = traffic / 288e3 * 1000
    print(f"head conv 3->{args.out} 3x3 at {args.width}x{args.height}: {flop:.2f} GFLOP, "
          f"{traffic:.0f} MB -> traffic floor {floor_ms:.2f} ms")
    print(f"{'input':>14} {'weights':>14} {'ms':>8} {'TFLOPS':>8} {'x floor':>8}")

    rows = []
    for cin in (3, 8):                                     # 8 = the engine's padded head
        x_nchw = torch.randn(1, cin, args.height, args.width, device=dev, dtype=torch.float16)
        x_nhwc = x_nchw.to(memory_format=torch.channels_last)
        for wmode in ("nchw", "channels_last"):
            conv = torch.nn.Conv2d(cin, args.out, 3, padding=1, bias=True).to(dev).half()
            if wmode == "channels_last":
                conv = conv.to(memory_format=torch.channels_last)
            for imode, x in (("nchw", x_nchw), ("nhwc-view", x_nhwc)):
                with torch.no_grad():
                    ms = timeit(lambda: conv(x), iters=args.iters)
                tflops = flop / ms
                rows.append(dict(cin=cin, weights=wmode, input=imode, ms=round(ms, 3),
                                 tflops=round(tflops, 2)))
                print(f"{imode + ' c' + str(cin):>14} {wmode:>14} {ms:>8.3f} {tflops:>8.2f} {ms/floor_ms:>8.1f}")
    best = min(rows, key=lambda r: r["ms"])
    print(f"\nbest: {best}")
    if args.out_file:
        with open(args.out_file, "w") as fh:
            json.dump({"take": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "height": args.height, "width": args.width, "out_channels": args.out,
                       "flop_gflop": round(flop, 2), "traffic_mb": round(traffic, 1),
                       "floor_ms": round(floor_ms, 3), "rows": rows}, fh, indent=1)
        print(f"# -> {args.out_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
