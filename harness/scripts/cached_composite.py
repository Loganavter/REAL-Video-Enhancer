"""The configuration the cost model actually describes: a CACHED full region, not a fresh one.

Everything measured so far compared compositing against a full render of the SAME frame, where the
full region is perfect by construction. The cost of 1/2.2 + delta assumes something else: the full
answer on F is kept from earlier frames and only delta is recomputed, so 82% of F is stale by up to
several frames. That staleness is invisible in the fresh measurement and it is exactly the mechanism
that made the tile path measure 0.287% without dilation and 0.071% with it.

Simulated offline, no engine:

  cache    = full output, updated only where delta = (F_t xor F_t-1) grown by the halo
  composite = where(F, cache, thin_t)
  truth     = full_t

`--stride` skips frames, because the gate renders roughly one in three: a stride-1 run freshens the
cache every frame and is optimistic; stride 3 approximates what the gate leaves.

Usage: cached_composite.py <clip> [--frames 24] [--stride 1] [--mask 8:18]
"""
import argparse
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as TF

import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)
from cut_channels_naive import cut_model, load, statistic  # noqa: E402

HALO = 18
FF = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/bin/ffmpeg"


def frames(clip, w, h, want, stride):
    proc = subprocess.Popen([FF, "-v", "error", "-i", clip, "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{w}x{h}", "-"], stdout=subprocess.PIPE)
    size = w * h * 3
    index = 0
    taken = 0
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        if index % stride == 0:
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).astype(np.float32) / 255.0
            yield torch.from_numpy(arr).permute(2, 0, 1)[None]
            taken += 1
            if taken >= want:
                break
        index += 1
    proc.stdout.close()
    proc.wait()


def grow(m, radius):
    return TF.max_pool2d(m, 2 * radius + 1, stride=1, padding=radius)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--mask", default="8:18")
    ap.add_argument("--keep", type=int, default=32)
    ap.add_argument("--model", default="/tmp/opencode/models2/futsuu.pth")
    args = ap.parse_args()
    thr_s, radius_s = args.mask.split(":")
    thr, radius = float(thr_s), int(radius_s)

    full, scale, _ = load(args.model)
    full = full.cuda().half().eval()
    thin, _ = cut_model(full, args.keep)
    thin = thin.cuda().half().eval()

    cache = None
    prev_F = None
    fresh_shares, cached_shares = [], []
    for i, img in enumerate(frames(args.clip, args.width, args.height, args.frames, args.stride)):
        x = img.cuda().half()
        with torch.no_grad():
            ref = full(x).float()
            got = thin(x).float()

        gray = x.float().mean(dim=1, keepdim=True) * 255.0
        gx = torch.zeros_like(gray)
        gy = torch.zeros_like(gray)
        gx[:, :, :, 1:-1] = (gray[:, :, :, 2:] - gray[:, :, :, :-2]) * 0.5
        gy[:, :, 1:-1, :] = (gray[:, :, 2:, :] - gray[:, :, :-2, :]) * 0.5
        mag = (gx * gx + gy * gy).sqrt()
        mask_lr = grow(grow((mag > thr).float(), radius), HALO)
        F = TF.interpolate(mask_lr, scale_factor=scale, mode="nearest").bool()

        fresh = torch.where(F, ref, got)
        s_fresh, p_fresh, _ = statistic(ref, fresh)
        fresh_shares.append(s_fresh)

        if cache is None:
            cache = ref.clone()
            delta = torch.ones_like(F)
        else:
            xor = (mask_lr.bool() ^ prev_F.bool()).float()
            delta = TF.interpolate(grow(xor, HALO), scale_factor=scale, mode="nearest").bool()
            cache = torch.where(delta, ref, cache)
        composite = torch.where(F, cache, got)
        s_cached, p_cached, _ = statistic(ref, composite)
        cached_shares.append(s_cached)
        print(f"frame {i:>2}: F {float(F.float().mean()) * 100:5.1f}%  delta {float(delta.float().mean()) * 100:5.1f}%  "
              f"fresh >8 {s_fresh:6.3f}% peak {p_fresh:4.0f}  cached >8 {s_cached:6.3f}% peak {p_cached:5.0f}")
        prev_F = mask_lr

    print(f"\nfresh : median {np.median(fresh_shares):.3f}%  worst {max(fresh_shares):.3f}%")
    print(f"cached: median {np.median(cached_shares):.3f}%  worst {max(cached_shares):.3f}%  "
          f"(bar 0.1%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
