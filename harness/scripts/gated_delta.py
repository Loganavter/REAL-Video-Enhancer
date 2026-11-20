"""The gated statistic: how much of F changes on the frames the gate actually renders.

The earlier pass used `max|pixel difference|` on a stride-2 grid as a stand-in for the gate, which
renders 97.7% of a clip the real gate reuses 66% of - the proxy is an order of magnitude more sensitive
because one changed pixel raises it. The gate is not that: it pools luma into 32x32 blocks at 1080p and
thresholds the MAX over blocks against the certified level, with a chain limit on how far back a
reference may be.

So this reproduces the gate's own arithmetic, then reports the change rate of F = mask+18 conditional on
the frames it renders - the selection effect that decides whether "update only what changed" is worth
anything. The mask grid is 2x2 (threshold x dilation) so the two factors are not varied together.

Usage: gated_delta.py <clip> [--threshold 4] [--chain 2] [--width 1920] [--height 1080]
"""
import argparse
import subprocess
import sys

import torch
import torch.nn.functional as TF

HALO = 18
THIN = 1 / 2.2
ENGINE_SHARE = 0.646
FF = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/bin/ffmpeg"


def frames(clip, w, h):
    proc = subprocess.Popen([FF, "-v", "error", "-i", clip, "-f", "rawvideo", "-pix_fmt", "gray",
                             "-s", f"{w}x{h}", "-"], stdout=subprocess.PIPE)
    size = w * h
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        yield torch.frombuffer(bytearray(buf), dtype=torch.uint8).reshape(1, 1, h, w).float().cuda()
    proc.stdout.close()
    proc.wait()


def grow(m, radius):
    return TF.max_pool2d(m, 2 * radius + 1, stride=1, padding=radius)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--threshold", type=float, default=4.0)
    ap.add_argument("--chain", type=int, default=2)
    ap.add_argument("--block", type=int, default=32)
    args = ap.parse_args()

    configs = [(2.0, 2), (2.0, 18), (8.0, 2), (8.0, 18)]
    changed = {c: [] for c in configs}
    prev_F = {}
    ref_blocks = None
    chain = 0
    total = rendered = 0

    for gray in frames(args.clip, args.width, args.height):
        blocks = TF.avg_pool2d(gray / 255.0, args.block, ceil_mode=True) * 255.0
        delta = float("inf") if ref_blocks is None else float((blocks - ref_blocks).abs().max())
        draw = ref_blocks is None or delta > args.threshold or chain >= args.chain

        if ref_blocks is not None:
            total += 1
            if draw:
                rendered += 1
        # masks are independent of the gate decision; computed once per frame
        mag_x = torch.zeros_like(gray)
        mag_y = torch.zeros_like(gray)
        mag_x[:, :, :, 1:-1] = (gray[:, :, :, 2:] - gray[:, :, :, :-2]).abs() * 0.5
        mag_y[:, :, 1:-1, :] = (gray[:, :, 2:, :] - gray[:, :, :-2, :]).abs() * 0.5
        mag = (mag_x * mag_x + mag_y * mag_y).sqrt()
        # the mask is thresholds on the gradient dilated by `radius`; F is that grown by the HALO,
        # and both factors have to be used or the 2x2 grid collapses into two thresholds
        current = {c: grow(grow((mag > c[0]).float(), c[1]), HALO) for c in configs}

        if ref_blocks is not None and draw:
            for c in configs:
                if c in prev_F:
                    xor = (current[c].bool() ^ prev_F[c].bool()).float()
                    changed[c].append(float(grow(xor, HALO).mean()))
        prev_F = current

        if draw:
            ref_blocks = blocks
            chain = 0
        else:
            chain += 1

    print(f"frames {total}, gate renders {rendered} ({rendered / max(total, 1) * 100:.1f}%), "
          f"threshold {args.threshold}, chain {args.chain}")
    for c in configs:
        vals = changed[c]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        p90 = sorted(vals)[int(0.9 * (len(vals) - 1))]
        engine = 1 / (THIN + mean)
        wall = 1 / (ENGINE_SHARE / engine + (1 - ENGINE_SHARE))
        print(f"thr {c[0]:4.1f} r{c[1]:<3}: delta on rendered frames {mean * 100:5.1f}% "
              f"(p90 {p90 * 100:5.1f}%) -> engine {engine:.2f}x, wall {wall:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
