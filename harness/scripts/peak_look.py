"""Where does the peak actually live? Find the worst pixel of the shipped int8 default against the model.

The peak bar (20 levels) is argued about in a number nobody has looked at: the ledger carries a 29-55 range for
the int8 kernel FAMILY, and the products of the two adoptions are deltas between int8 configurations (6 and 13
levels). This script produces the thing the argument is about - the worst frame, the worst pixel, and what the
neighbourhood of that pixel looks like:

  * the reference (the model's own fp16 output), the shipped default, and the same region with the difference
    amplified, side by side, as one PNG;
  * how the large errors are DISTRIBUTED - isolated single pixels, or patches - because that is the difference
    between "invisible" and "visible", and the peak alone cannot say it;
  * the share at the 8-level threshold, so the area statistic and the extreme sit in one place.

Frames are the spread sample of the corpus: 8 calibration frames are excluded, the held-out set is scored.

Usage: python harness/scripts/peak_look.py [--frames 29,44,100,169,274,400,460] [--half 48] [--gain 8]
                                   [--out-dir docs/wsp/verified/logs/peak-look]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "backend"))
import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)

from cut_channels_naive import load  # noqa: E402
from src.pytorch.Int8Body import Int8Body  # noqa: E402
from src.pytorch.TorchUtils import TorchUtils  # noqa: E402
from tail_int8_emulation import CALIB, CLIP, H, W, decode_yuv10  # noqa: E402

APP = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2"
MODEL = f"{APP}/custom_models/2x-Futsuu-Anime(1).pth"


def to_img(t: torch.Tensor) -> np.ndarray:
    """(1,3,h,w) float 0..1 -> uint8 HWC."""
    a = t.detach().float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return (a * 255.0 + 0.5).astype(np.uint8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="29,44,100,169,274,400,460")
    ap.add_argument("--half", type=int, default=48, help="half-size of the crop, in output pixels")
    ap.add_argument("--gain", type=int, default=8, help="amplification of the difference panel")
    ap.add_argument("--out-dir", default="docs/wsp/verified/logs/peak-look")
    ap.add_argument("--out-file", default="")
    args = ap.parse_args(argv)

    dev = torch.device("cuda", 0)
    tu = TorchUtils(width=W, height=H, device_type="cuda", hdr_mode=True, gpu_id=0, high_bitdepth=True)
    stream = torch.cuda.Stream(device=dev)

    def frame(idx: int) -> torch.Tensor:
        return tu.yuv420p10_to_tensor(decode_yuv10(idx), stream, dev, torch.float16)

    model, _, _ = load(MODEL)
    model = model.eval().cuda().half()
    body = Int8Body(model)
    with torch.no_grad():
        body.calibrate([frame(i) for i in CALIB])
        body._build()
    print(f"config: head_fused={body.head_fused} tail_int8={body.tail_int8} "
          f"fold={body.fold} (the shipped default)")

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for f in [int(s) for s in args.frames.split(",")]:
        x = frame(f)
        with torch.no_grad():
            ref = model(x).float()
            got = body(x).float()
        d = (got - ref).abs() * 255.0                     # (1,3,h,w) in levels
        dmax = d.amax(dim=1)[0]                           # worst channel per pixel
        peak = float(dmax.max())
        iy, ix = np.unravel_index(int(dmax.argmax().item()), tuple(dmax.shape))
        # how the large errors are distributed: isolated or clustered?
        for thr in (8, 20, 40):
            m = dmax > thr
            rows.append(dict(frame=f, threshold=thr, pixels=int(m.sum()),
                             share_pct=100.0 * float(m.sum()) / dmax.numel()))
        # a patch test at the peak's own level: how many pixels sit within 20% of the peak
        m_peak = dmax > 0.8 * peak
        # largest connected component at the 8-level threshold, in this frame
        lab = None
        try:
            import cv2
            n, lab = cv2.connectedComponents((dmax > 8).byte().cpu().numpy())
            sizes = np.bincount(lab.ravel())[1:] if n > 1 else np.array([0])
            largest = int(sizes.max()) if sizes.size else 0
            ncomp = int(n - 1)
        except Exception:
            largest, ncomp = -1, -1

        # crop around the LARGEST 8-level blob when there is one: a peak of 17 on one pixel is not what
        # anyone could see, a 19-pixel patch is - the two are different questions and both get a picture
        hh, ww = dmax.shape
        by, bx = iy, ix
        if largest > 0 and lab is not None:
            k = int(np.argmax(sizes)) + 1
            yy, xx = np.where(lab == k)
            by, bx = int(yy.mean()), int(xx.mean())
        y0, y1 = max(0, by - args.half), min(hh, by + args.half)
        x0, x1 = max(0, bx - args.half), min(ww, bx + args.half)
        ref_c = to_img(ref[:, :, y0:y1, x0:x1])
        got_c = to_img(got[:, :, y0:y1, x0:x1])
        diff_c = np.clip(d[:, :, y0:y1, x0:x1].amax(dim=1)[0].cpu().numpy() * args.gain, 0,
                         255).astype(np.uint8)
        diff_rgb = np.stack([diff_c, diff_c // 3, diff_c // 3], axis=-1)
        # the mask: which pixels are >8 levels, drawn white on a dimmed reference - the area statistic,
        # visible as area instead of as a number
        mask = (d[:, :, y0:y1, x0:x1].amax(dim=1)[0] > 8).cpu().numpy()
        mask_rgb = (ref_c.astype(np.float32) * 0.35).astype(np.uint8)
        mask_rgb[mask] = np.array([255, 40, 40], np.uint8)
        sep = np.full((ref_c.shape[0], 6, 3), 255, np.uint8)
        panel = np.concatenate([ref_c, sep, got_c, sep, diff_rgb, sep, mask_rgb], axis=1)
        scale = 4
        panel = np.repeat(np.repeat(panel, scale, axis=0), scale, axis=1)
        name = f"peak_frame{f:04d}_crop_y{by}_x{bx}_peak{peak:.0f}_blob{largest}.png"
        Image.fromarray(panel).save(os.path.join(args.out_dir, name))
        rows.append(dict(frame=f, peak=round(peak, 1), y=int(iy), x=int(ix),
                         near_peak_pixels=int(m_peak.sum()), largest_component_8=int(largest),
                         components_8=int(ncomp), crop=name))
        print(f"frame {f:>4}: peak {peak:5.1f} levels at (y={iy}, x={ix})  "
              f"pixels >8: {100.0 * float((dmax > 8).sum()) / dmax.numel():.5f}%  "
              f">20: {int((dmax > 20).sum())}  >40: {int((dmax > 40).sum())}  "
              f"largest 8-level blob: {largest} px in {ncomp} components")

    print(f"\nPNGs (reference | shipped default | difference x{args.gain}) -> {args.out_dir}/")
    if args.out_file:
        with open(args.out_file, "w") as fh:
            json.dump({"clip": os.path.basename(CLIP), "calib": list(CALIB), "rows": rows}, fh, indent=1)
        print(f"# -> {args.out_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
