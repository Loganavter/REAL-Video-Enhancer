"""Where the ink lift comes from: the body in RGB, the colour pipe, or the decode.

`int8-scales-bd.txt` found the rendered file reading +4.40/+6.60 levels on line art against the model while the
body in RGB read -0.5, and left it open with "whatever shifts the ink by ~5 levels happens between the body's
output and the decoded file". This probe puts three points on ONE frame, each against the same model output:

  1  the body's RGB tensor                       (the quantiser)
  2  the same tensor through yuv420p and back     (the colour pipe: BT.709 limited, 4:2:0 chroma)
  3  the decoded frame of a render                (not here - a render is needed, and this probe stops at 2)

If (2) already carries the lift, the pipe is the cause and the decode is innocent.

Usage:
  RVE_TAIL_YUV=0 RVE_INT8_NO_EVENTS=1 python3 harness/scripts/ink_lift_probe.py \
      --calib /tmp/opencode/calib-frames --frame /tmp/opencode/eval3/violet_5.rgb
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

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

MODEL = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/custom_models/2x-Futsuu-Anime(1).pth"


def frame_tensor(path: str, h: int, w: int) -> torch.Tensor:
    raw = np.fromfile(path, dtype=np.uint8).reshape(h, w, 3)
    return torch.from_numpy(raw).permute(2, 0, 1).float().div_(255.0)[None].contiguous().cuda().half()


def yuv420p_to_rgb(packed: np.ndarray, h: int, w: int) -> torch.Tensor:
    """Inverse of TorchUtils.tensor_to_yuv420p: BT.709 limited, nearest chroma upsample."""
    n = h * w
    y = packed[:n].reshape(h, w).astype(np.float32)
    u = packed[n:n + (h // 2) * (w // 2)].reshape(h // 2, w // 2).astype(np.float32)
    v = packed[n + (h // 2) * (w // 2):].reshape(h // 2, w // 2).astype(np.float32)
    y = torch.from_numpy((y - 16.0) / 219.0).cuda()
    u = torch.from_numpy((u - 128.0) / 224.0).cuda().repeat_interleave(2, 0).repeat_interleave(2, 1)
    v = torch.from_numpy((v - 128.0) / 224.0).cuda().repeat_interleave(2, 0).repeat_interleave(2, 1)
    r = y + 1.5748 * v
    g = y - 0.1873 * u - 0.4681 * v
    b = y + 1.8556 * u
    return torch.stack([r, g, b])[None]


def stats(out: torch.Tensor, ref: torch.Tensor) -> dict:
    d = (out.float() - ref.float()) * 255.0                       # signed, in 8-bit levels
    luma = (0.2126 * ref[0, 0] + 0.7152 * ref[0, 1] + 0.0722 * ref[0, 2]).float()
    ink = luma < 0.20                                             # the dark line core
    return {"mean_abs": round(float(d.abs().mean()), 4),
            "mean_signed": round(float(d.mean()), 4),
            "ink_mean_signed": round(float(d[:, :, ink].mean()), 4),
            "ink_mean_abs": round(float(d[:, :, ink].abs().mean()), 4),
            "share_gt8_pct": round(float((d.abs() > 8).float().mean() * 100), 6),
            "ink_share": round(float(ink.float().mean() * 100), 3)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--calib", default="/tmp/opencode/calib-frames")
    ap.add_argument("--frame", default="/tmp/opencode/eval3/violet_5.rgb")
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--decoded", default="", help="raw RGB24 of the rendered file's frame at the same timestamp")
    ap.add_argument("--out-file", default="/tmp/opencode/ink-lift.json")
    a = ap.parse_args(argv)

    model, scale, _ = load(a.model)
    model = model.eval().cuda().half()
    calib = [frame_tensor(p, a.height, a.width) for p in sorted(glob.glob(os.path.join(a.calib, "*.rgb")))]
    x = frame_tensor(a.frame, a.height, a.width)
    body = Int8Body(model)
    with torch.no_grad():
        body.calibrate(calib)
        body._build()
        ref = model(x).float()
        rgb = body(x).float()
    h, w = rgb.shape[-2], rgb.shape[-1]
    tu = TorchUtils(width=w, height=h, device_type="cuda", hdr_mode=False, gpu_id=0, high_bitdepth=False)
    packed = tu.tensor_to_yuv420p(rgb)
    back = yuv420p_to_rgb(np.asarray(packed), h, w).float()

    res = {"model": os.path.basename(a.model), "frame": os.path.basename(a.frame), "scale": scale,
           "body_rgb": stats(rgb, ref), "yuv_roundtrip": stats(back, ref),
           "rgb_vs_roundtrip": stats(back, rgb)}
    if a.decoded:
        dh, dw = ref.shape[-2], ref.shape[-1]
        raw = np.fromfile(a.decoded, dtype=np.uint8)
        if raw.size != dh * dw * 3:
            raise SystemExit(f"{a.decoded}: {raw.size} bytes is not {dw}x{dh}x3")
        dec = torch.from_numpy(raw.reshape(dh, dw, 3)).permute(2, 0, 1)[None].float().div_(255.0).cuda()
        res["decoded_file"] = stats(dec, ref)
        res["decoded_vs_roundtrip"] = stats(dec, back)
    for k in ("body_rgb", "yuv_roundtrip", "rgb_vs_roundtrip", "decoded_file", "decoded_vs_roundtrip"):
        if k not in res:
            continue
        print(f"{k:18s} mean|d| {res[k]['mean_abs']:7.3f}  signed {res[k]['mean_signed']:+7.3f}  "
              f"ink signed {res[k]['ink_mean_signed']:+7.3f}  ink|d| {res[k]['ink_mean_abs']:6.3f}  "
              f">8 {res[k]['share_gt8_pct']:.4f}%")
    with open(a.out_file, "w") as f:
        json.dump(res, f, indent=1)
    print(f"# -> {a.out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
