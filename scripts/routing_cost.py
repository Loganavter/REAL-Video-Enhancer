"""What does the routing lane actually cost? Measured on areas, not on the mask.

The composite test measured the mask area, but the full engine needs its INPUT to cover mask+18 to
produce a correct answer inside the mask. So the area that matters is the dilated one, and the cost is

    thin everywhere + full on F            cost = c + |F|
    each region with its own halo         cost = |F| + c * |T + 18|

with c = 1/2.2 (thin runs 2.2x faster than full) and F = mask + 18, T = complement of F.

Also measured here: the temporal variant - the LR gradient mask is mostly scene contours, so it is
nearly static between frames; if the full answer on F can be reused, the per-frame cost is
c + |difference of F between frames|.

And the artefact the bar cannot see: at the mask boundary the composite steps from the full answer to
the thin one, so the step is exactly the thin model's own error along that contour.

Usage: routing_cost.py <frame.raw> <W> <H> [<frame2.raw>]
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

HALO = 18
THIN = 1 / 2.2


def mask_from(img, thr, radius):
    gray = img.mean(dim=1, keepdim=True) * 255.0
    gx = torch.zeros_like(gray)
    gy = torch.zeros_like(gray)
    gx[:, :, :, 1:-1] = (gray[:, :, :, 2:] - gray[:, :, :, :-2]) * 0.5
    gy[:, :, 1:-1, :] = (gray[:, :, 2:, :] - gray[:, :, :-2, :]) * 0.5
    mag = (gx * gx + gy * gy).sqrt()
    m = (mag > thr).float()
    if radius:
        m = F.max_pool2d(m, 2 * radius + 1, stride=1, padding=radius)
    return m


def grow(m, radius):
    return F.max_pool2d(m, 2 * radius + 1, stride=1, padding=radius)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("frame")
    ap.add_argument("width", type=int)
    ap.add_argument("height", type=int)
    ap.add_argument("second", nargs="?", default=None)
    args = ap.parse_args()

    def load(path):
        raw = np.fromfile(path, dtype=np.uint8)
        img = torch.from_numpy(raw.reshape(args.height, args.width, 3).astype(np.float32) / 255.0)
        return img.permute(2, 0, 1)[None].cuda()

    x = load(args.frame)
    total = x.shape[2] * x.shape[3]
    print(f"{'mask':<22} {'mask':>7} {'F=mask⊕18':>10} {'T⊕18':>7} {'c+|F|':>8} {'|F|+c|T⊕18|':>13}")

    rows = []
    for thr in (2.0, 4.0, 8.0):
        for radius in (2, 4, 18):
            m = mask_from(x, thr, radius)
            f = grow(m, HALO)
            t = 1.0 - f
            td = grow(t, HALO)
            fa = float(f.mean())
            tda = float(td.mean())
            a = 1 / (THIN + fa)
            b = 1 / (fa + THIN * tda)
            rows.append((thr, radius, fa, tda, a, b))
            print(f"thr {thr:4.1f} r{radius:<3}        {float(m.mean()) * 100:6.1f}% {fa * 100:9.1f}% "
                  f"{tda * 100:6.1f}% {a:7.2f}x {b:12.2f}x")

    if args.second:
        x2 = load(args.second)
        for thr, radius in ((4.0, 2), (8.0, 2), (2.0, 18)):
            f1 = grow(mask_from(x, thr, radius), HALO).bool()
            f2 = grow(mask_from(x2, thr, radius), HALO).bool()
            xor = (f1 ^ f2).float()
            changed = float(xor.mean())
            # the recomputed region needs its own halo, or the seams are wrong for the same reason
            grown = float(grow(xor, HALO).mean())
            print(f"temporal thr {thr} r{radius}: |F| {float(f1.float().mean()) * 100:5.1f}%  "
                  f"xor {changed * 100:4.1f}%  xor⊕18 {grown * 100:5.1f}%  "
                  f"cost {1 / (THIN + changed):.2f}x / {1 / (THIN + grown):.2f}x with its halo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
