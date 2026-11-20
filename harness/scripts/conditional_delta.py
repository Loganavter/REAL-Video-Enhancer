"""The number that decides the routing lane: how much does F change on the frames the gate RENDERS?

Frames the gate reuses never reach this mechanism, so the average change over all pairs is the wrong
statistic - it is diluted by the ~73% of pairs the gate throws away. What matters is the conditional one:
given that the frame was rendered, how much of F = mask+18 has to be recomputed.

Both quantities are measured here per consecutive pair, on the source clip:

  lr_delta   - max abs luma difference on a stride-2 grid, the quantity the gate thresholds on
  F_changed  - |F_t xor F_{t-1}|, and the same after growing it by the receptive halo
  content    - max abs luma difference inside F, sampled, because the mask can stand still while the
               pixels under it drift (grain, slow brightness, dithering)

Then the report separates all pairs from the pairs whose lr_delta clears the gate's threshold, and
prints the cost model with that conditional change rate.

Usage: conditional_delta.py <clip> [--threshold 4] [--masks "8:2,2:18"] [--limit 0]
"""
import argparse
import subprocess
import sys

import torch
import torch.nn.functional as TF

HALO = 18
THIN = 1 / 2.2
FF = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/bin/ffmpeg"


def frames(clip, w, h, limit):
    argv = [FF, "-v", "error", "-i", clip, "-f", "rawvideo", "-pix_fmt", "gray",
            "-s", f"{w}x{h}", "-"]
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE)
    size = w * h
    n = 0
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        # 2 Mpx with a 37x37 max-pool per stage is minutes on CPU and milliseconds on the card
        yield torch.frombuffer(bytearray(buf), dtype=torch.uint8).reshape(1, 1, h, w).float().cuda()
        n += 1
        if limit and n >= limit:
            break
    proc.stdout.close()
    proc.wait()


def grow(m, radius):
    return TF.max_pool2d(m, 2 * radius + 1, stride=1, padding=radius)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--threshold", type=float, default=4.0, help="gate lr_delta threshold")
    ap.add_argument("--masks", default="8:2,2:18")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    configs = [tuple(int(v) for v in m.split(":")) for m in args.masks.split(",")]
    stats = {c: {"all": [], "rendered": [], "content": []} for c in configs}
    prev_gray = prev_F = None
    total = rendered = 0

    for gray in frames(args.clip, args.width, args.height, args.limit):
        mag_x = torch.zeros_like(gray)
        mag_y = torch.zeros_like(gray)
        mag_x[:, :, :, 1:-1] = (gray[:, :, :, 2:] - gray[:, :, :, :-2]).abs() * 0.5
        mag_y[:, :, 1:-1, :] = (gray[:, :, 2:, :] - gray[:, :, :-2, :]).abs() * 0.5
        mag = (mag_x * mag_x + mag_y * mag_y).sqrt()

        if prev_gray is not None:
            lr_delta = float((gray[:, :, ::2, ::2] - prev_gray[:, :, ::2, ::2]).abs().max())
            total += 1
            drawn = lr_delta > args.threshold
            rendered += drawn
            current = {}
            for thr, radius in configs:
                f = grow((mag > thr).float(), HALO)
                current[(thr, radius)] = f
                if prev_F is not None and (thr, radius) in prev_F:
                    xor = (f.bool() ^ prev_F[(thr, radius)].bool()).float()
                    changed = float(xor.mean())
                    grown = float(grow(xor, HALO).mean())
                    inside = f.bool()
                    drift = float((gray - prev_gray).abs()[inside].mean()) if inside.any() else 0.0
                    stats[(thr, radius)]["all"].append(grown)
                    if drawn:
                        stats[(thr, radius)]["rendered"].append(grown)
                        stats[(thr, radius)]["content"].append(drift)
            prev_F = current
        prev_gray = gray

    print(f"pairs {total}, gate would render {rendered} ({rendered / max(total, 1) * 100:.1f}%)")
    for (thr, radius) in configs:
        a = stats[(thr, radius)]["all"]
        r = stats[(thr, radius)]["rendered"]
        c = stats[(thr, radius)]["content"]
        if not r:
            continue
        mean_all = sum(a) / len(a)
        mean_rendered = sum(r) / len(r)
        p90 = sorted(r)[int(0.9 * (len(r) - 1))]
        cost_all = 1 / (THIN + mean_all)
        cost_rendered = 1 / (THIN + mean_rendered)
        print(f"mask thr {thr} r{radius}: change (xor+halo) all pairs {mean_all * 100:5.1f}% | "
              f"rendered {mean_rendered * 100:5.1f}% (p90 {p90 * 100:5.1f}%) | "
              f"cost {cost_all:.2f}x / {cost_rendered:.2f}x rendered | "
              f"drift inside F {sum(c) / len(c):.2f} levels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
