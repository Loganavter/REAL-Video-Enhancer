"""Oracle routing: is "thin everywhere, full where it matters" a real lever or a nice statistic?

The diagnostics split the pruning error cleanly - flat pixels 0.060% above 8 levels (at the bar),
edge pixels 26.1% - and that split is a licence for conditional compute, not for a global cut. The
cheapest way to find out whether the licence is worth anything is to build the composite from outputs
that already exist:

    out = where(M, full, thin)

and measure it with the acceptance rule against `full`. Inside M the composite *is* full, so the only
error that can appear is thin's error outside the mask. Two masks are reported:

  edge  - what a router could actually compute: gradient on the LR frame, dilated by the receptive
          radius (18 LR pixels), upscaled to HR
  error - the oracle: where thin actually exceeds 8 levels. Not implementable, but it bounds the prize
          and names the gap the router has to close

Usage: oracle_routing.py <frame.raw> <W> <H> [--keep 32] [--radius 18]
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts")
from cut_channels_naive import cut_model, load, statistic  # noqa: E402


def edge_mask(lr, radius):
    """Gradient magnitude on LR luma, thresholded, then dilated by `radius` LR pixels."""
    gray = lr.mean(dim=1, keepdim=True) * 255.0
    gx = torch.zeros_like(gray)
    gy = torch.zeros_like(gray)
    gx[:, :, :, 1:-1] = (gray[:, :, :, 2:] - gray[:, :, :, :-2]) * 0.5
    gy[:, :, 1:-1, :] = (gray[:, :, 2:, :] - gray[:, :, :-2, :]) * 0.5
    mag = (gx * gx + gy * gy).sqrt()
    mask = (mag > 2.0).float()
    k = 2 * radius + 1
    mask = F.max_pool2d(mask, k, stride=1, padding=radius)
    return mask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("frame")
    ap.add_argument("width", type=int)
    ap.add_argument("height", type=int)
    ap.add_argument("--keep", type=int, default=32)
    ap.add_argument("--radius", type=int, default=18)
    ap.add_argument("--model", default="/tmp/opencode/models2/futsuu.pth")
    args = ap.parse_args()

    raw = np.fromfile(args.frame, dtype=np.uint8)
    img = torch.from_numpy(raw.reshape(args.height, args.width, 3).astype(np.float32) / 255.0)
    x = img.permute(2, 0, 1)[None].cuda().half()

    full, scale, _ = load(args.model)
    full = full.cuda().half().eval()
    with torch.no_grad():
        ref = full(x).float()
    thin, _ = cut_model(full, args.keep)
    thin = thin.cuda().half().eval()
    with torch.no_grad():
        got = thin(x).float()

    base_share, base_peak, _ = statistic(ref, got)
    print(f"thin C={args.keep} alone: >8 {base_share:.3f}%  peak {base_peak:.0f}")

    err = (got - ref).abs().amax(dim=1, keepdim=True) * 255.0
    oracle = (err > 8).float()
    print(f"oracle mask (thin error > 8) covers {float(oracle.mean()) * 100:.2f}% of HR pixels")

    m = edge_mask(x.float(), args.radius)
    m_hr = F.interpolate(m, scale_factor=scale, mode="nearest")
    cov = float(m_hr.mean()) * 100
    # what the router must fix: the share of thin's error pixels the edge mask actually covers
    covered = float((oracle.bool() & m_hr.bool()).sum()) / max(float(oracle.sum()), 1.0) * 100
    print(f"edge mask (grad>2, radius {args.radius} LR, x{scale} HR) covers {cov:.2f}% of HR pixels, "
          f"{covered:.1f}% of the error pixels")

    # a router is only useful if some threshold/radius combination covers the error cheaply, so
    # sweep instead of reporting one arbitrary point
    for thr in (2.0, 4.0, 8.0, 16.0, 32.0):
        for radius in (18, 8, 4, 2):
            gray = x.float().mean(dim=1, keepdim=True) * 255.0
            gx = torch.zeros_like(gray)
            gy = torch.zeros_like(gray)
            gx[:, :, :, 1:-1] = (gray[:, :, :, 2:] - gray[:, :, :, :-2]) * 0.5
            gy[:, :, 1:-1, :] = (gray[:, :, 2:, :] - gray[:, :, :-2, :]) * 0.5
            mag = (gx * gx + gy * gy).sqrt()
            mm = F.max_pool2d((mag > thr).float(), 2 * radius + 1, stride=1, padding=radius)
            mm_hr = F.interpolate(mm, scale_factor=scale, mode="nearest")
            c = torch.where(mm_hr.bool(), ref, got)
            s, pk, _ = statistic(ref, c)
            area = float(mm_hr.mean()) * 100
            covd = float((oracle.bool() & mm_hr.bool()).sum()) / max(float(oracle.sum()), 1.0) * 100
            cost = (area / 100) * 1.15 + (1 - area / 100) / 2.2
            print(f"  thr {thr:5.1f} r{radius:<3} area {area:5.1f}%  covers {covd:5.1f}% of error  "
                  f">8 {s:6.3f}%  cost {1 / cost:4.2f}x")

    comp = torch.where(m_hr.bool(), ref, got)
    share, peak, median = statistic(ref, comp)
    print(f"composite where(edge, full, thin): >8 {share:.3f}% (bar 0.1)  peak {peak:.0f}  median {median:.2f}")
    comp_oracle = torch.where(oracle.bool(), ref, got)
    share_o, peak_o, _ = statistic(ref, comp_oracle)
    print(f"composite where(oracle, full, thin): >8 {share_o:.3f}%  peak {peak_o:.0f}")

    # time model: full = 1.0, thin = 1/2.2 (measured 2.07-2.36x), halo/call tax 1.15 on the masked part
    e = cov / 100.0
    eo = float(oracle.mean())
    thin_cost = 1 / 2.2
    for label, area in (("edge mask", e), ("oracle mask", eo)):
        for tax in (1.0, 1.15):
            cost = area * tax + (1 - area) * thin_cost
            print(f"  cost with {label:<11} area {area * 100:5.1f}% tax {tax:.2f}: {1 / cost:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
