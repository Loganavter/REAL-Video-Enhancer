"""What the weights themselves produce, on the frame the owner is looking at.

The lane compares renders to each other; this writes the model's own fp16 output as a picture so the three
things can be seen side by side: the source, what the weights compute (torch fp16), and the two renders.

Files, per frame:
  model_f<N>.png                      the model's output, 3840x2160, no colour pipe
  crop_src_model_engine_int8_f<N>.png 1:1 crop of the frame's worst 64 px block, four panels
  crop_model_vs_engine_f<N>.png       model | engine | difference x16
  crop_model_vs_int8_f<N>.png         model | int8 | difference x16
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

REPO = "/home/jorj/Загрузки/projects/REAL-Video-Enhancer"
sys.path.insert(0, os.path.join(REPO, "scripts"))
from cut_channels_naive import load  # noqa: E402

FF = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/bin/ffmpeg"
W, H = 3840, 2160


def model_output(path: str, idx: int, model) -> np.ndarray:
    raw = np.fromfile(f"/var/tmp/rve-rem/frames/v{idx}.raw", dtype=np.uint8)
    x = torch.from_numpy(raw.reshape(1080, 1920, 3).astype(np.float32) / 255.0)
    t = x.permute(2, 0, 1)[None].cuda().half()
    with torch.no_grad():
        out = model(t).float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return (out * 255.0 + 0.5).astype(np.uint8)


def source_crop(idx: int, x: int, y: int, cw: int, ch: int) -> np.ndarray:
    raw = np.fromfile(f"/var/tmp/rve-rem/frames/v{idx}.raw", dtype=np.uint8)
    full = raw.reshape(1080, 1920, 3)
    sub = full[y // 2:y // 2 + ch // 2, x // 2:x // 2 + cw // 2]
    return np.asarray(Image.fromarray(sub).resize((cw, ch), Image.NEAREST))


def render_frame(path: str, idx: int) -> np.ndarray:
    raw = subprocess.run([FF, "-loglevel", "error", "-i", path, "-vf", f"select=eq(n\\,{idx})",
                          "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw[:W * H * 3], dtype=np.uint8).reshape(H, W, 3)


def worst_block(model_img: np.ndarray, other: np.ndarray, block: int = 64) -> tuple[int, int]:
    d = np.abs(model_img.astype(np.int16) - other.astype(np.int16)).max(axis=2)
    hh, ww = (H // block) * block, (W // block) * block
    s = d[:hh, :ww].reshape(hh // block, block, ww // block, block).mean(axis=(1, 3))
    by, bx = np.unravel_index(int(s.argmax()), s.shape)
    return int(bx * block), int(by * block)


def crop(img: np.ndarray, x: int, y: int, cw: int, ch: int) -> np.ndarray:
    return img[y:y + ch, x:x + cw]


def diff_panel(a: np.ndarray, b: np.ndarray, x: int, y: int, cw: int, ch: int) -> np.ndarray:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
    amp = np.clip(d * 16, 0, 255).astype(np.uint8)
    return np.stack([amp] * 3, axis=2)[y:y + ch, x:x + cw]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=169)
    ap.add_argument("--outdir", default="/home/jorj/eye-check/2026-11-19-int8-per-channel")
    ap.add_argument("--model", default="/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/"
                                       "custom_models/2x-Futsuu-Anime(1).pth")
    ap.add_argument("--engine", default="/var/tmp/rve-rem/ref_fp16_y8.mkv")
    ap.add_argument("--int8", dest="int8_path", default="/var/tmp/rve-rem/int8_pt.mkv")
    args = ap.parse_args(argv)

    model, _, _ = load(args.model)
    model = model.eval().cuda().half()
    f = args.frame
    m = model_output(args.model, f, model)
    Image.fromarray(m).save(f"{args.outdir}/model_f{f}.png")

    e = render_frame(args.engine, f)
    i8 = render_frame(args.int8_path, f)

    bx, by = worst_block(m, i8)
    cw, ch = 1024, 576
    x = int(min(max(0, bx + 32 - cw // 2), W - cw))
    y = int(min(max(0, by + 32 - ch // 2), H - ch))

    src = source_crop(f, x, y, cw, ch)
    panels = [src, crop(m, x, y, cw, ch), crop(e, x, y, cw, ch), crop(i8, x, y, cw, ch)]
    Image.fromarray(np.hstack(panels)).save(f"{args.outdir}/crop_src_model_engine_int8_f{f}.png")
    Image.fromarray(np.hstack([crop(m, x, y, cw, ch), crop(e, x, y, cw, ch),
                               diff_panel(m, e, x, y, cw, ch)])).save(
        f"{args.outdir}/crop_model_vs_engine_f{f}.png")
    Image.fromarray(np.hstack([crop(m, x, y, cw, ch), crop(i8, x, y, cw, ch),
                               diff_panel(m, i8, x, y, cw, ch)])).save(
        f"{args.outdir}/crop_model_vs_int8_f{f}.png")

    print(f"frame {f}: crop ({x},{y}) {cw}x{ch}, centred on the worst model-vs-int8 block")
    for n, p in (("model (full frame)", f"model_f{f}.png"),
                 ("source | model | engine | int8", f"crop_src_model_engine_int8_f{f}.png"),
                 ("model | engine | diff", f"crop_model_vs_engine_f{f}.png"),
                 ("model | int8 | diff", f"crop_model_vs_int8_f{f}.png")):
        print(f"  {n:<34} {args.outdir}/{p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
