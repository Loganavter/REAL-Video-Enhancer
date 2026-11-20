"""Does the 0.27% engine-vs-model really belong to the engine, or is it the delivered colour pipe?

The int8 lane's whole "fails the bar" number was measured against the fp16 RENDER, i.e. after the pipe; the
model's own output has no pipe. So before blaming the engine, both sides are put through the same 4:2:0
round trip and compared again. Whatever survives that is the engine's own deviation.

frame 29 is the int8 worst; 169 is the frame the owner looked at.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as TF

REPO = "/home/jorj/Загрузки/projects/REAL-Video-Enhancer"
import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)
from cut_channels_naive import load  # noqa: E402

FF = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/bin/ffmpeg"
W, H = 3840, 2160


def model_frame(path: str, idx: int) -> np.ndarray:
    raw = np.fromfile(f"/var/tmp/rve-rem/frames/v{idx}.raw", dtype=np.uint8)
    x = torch.from_numpy(raw.reshape(1080, 1920, 3).astype(np.float32) / 255.0)
    t = x.permute(2, 0, 1)[None].cuda().half()
    model, _, _ = load(path)
    model = model.eval().cuda().half()
    with torch.no_grad():
        out = model(t).float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return (out * 255.0 + 0.5).astype(np.uint8)


def to_420_and_back(rgb: np.ndarray) -> np.ndarray:
    """Push an RGB frame through the delivered colour path and read it back as RGB."""
    p = subprocess.run([FF, "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-s", f"{W}x{H}", "-i", "-", "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"],
                       input=rgb.tobytes(), capture_output=True, check=True)
    q = subprocess.run([FF, "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p",
                        "-s", f"{W}x{H}", "-i", "-", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                       input=p.stdout, capture_output=True, check=True)
    return np.frombuffer(q.stdout[:W * H * 3], dtype=np.uint8).reshape(H, W, 3)


def render_frame(ref: str, idx: int) -> np.ndarray:
    raw = subprocess.run([FF, "-loglevel", "error", "-i", ref, "-vf", f"select=eq(n\\,{idx})",
                          "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw[:W * H * 3], dtype=np.uint8).reshape(H, W, 3)


def stat(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
    return float((d > 8).mean() * 100), float(d.max())


def main(argv=None) -> int:
    model_path = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/custom_models/2x-Futsuu-Anime(1).pth"
    ref = "/var/tmp/rve-rem/ref_fp16_y8.mkv"
    print("frame   engine-vs-model (raw)   both through 4:2:0   pipe alone")
    for f in (29, 169):
        m = model_frame(model_path, f)
        e = render_frame(ref, f)
        s_raw, p_raw = stat(e, m)
        m2 = to_420_and_back(m)
        s_pipe, p_pipe = stat(e, m2)
        s_self, _ = stat(to_420_and_back(m), m)
        print(f"{f:>5}   {s_raw:8.5f}% (max {p_raw:3.0f})   {s_pipe:8.5f}% (max {p_pipe:3.0f})   "
              f"{s_self:8.5f}%")
    print("\\n'pipe alone' is the model against itself through the same round trip - the floor this control "
          "cannot go below.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
