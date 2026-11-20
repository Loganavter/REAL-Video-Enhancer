"""The eye's half of the verdict: 1:1 crops at the worst block, an amplified difference, a zoomed clip.

The decision statistic is a share, and a share is exactly the wrong instrument for "does this look like
a blob to a person" - the rejected 35 mm artifact sat at a 7.57 block mean while its share was under the
bar. So every cell that decides something gets three artefacts a human can judge:

  * `*_gated.png` / `*_ungated.png` - a 1:1 crop around the worst 64 px block (no resampling, so what
    you see is the pixel the statistic counted);
  * `*_diff_x16.png` - the channel-max delta multiplied by 16, the same region;
  * `*_zoom.mp4` - the same crop, gated on the left and ungated on the right, a few frames either side,
    H.264 CRF 12 so the encoding itself does not add what we are asked to judge.

Usage:
  python harness/scripts/eye_check.py GATED UNGATED FRAME OUTPREFIX [--clip-frames 48] [--width 1024]
  python harness/scripts/eye_check.py --sheet        # regenerate the checklist table from a manifest (see main)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

FF = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/bin/ffmpeg"
W, H = 3840, 2160
SIZE = W * H * 3


def frame_at(path: str, idx: int) -> np.ndarray:
    raw = subprocess.run([FF, "-loglevel", "error", "-i", path, "-vf", f"select=eq(n\\,{idx})",
                          "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    if len(raw) < SIZE:
        raise SystemExit(f"no frame {idx} in {path}")
    return np.frombuffer(raw[:SIZE], dtype=np.uint8).reshape(H, W, 3).copy()


def png(arr: np.ndarray, path: str) -> None:
    h, w = arr.shape[:2]
    subprocess.run([FF, "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{w}x{h}", "-i", "-", "-y", path], input=arr.tobytes(), check=True)


def worst_block(d: np.ndarray, block: int = 64) -> tuple[int, int, float]:
    hh, ww = (H // block) * block, (W // block) * block
    s = d[:hh, :ww].reshape(hh // block, block, ww // block, block).mean(axis=(1, 3))
    y, x = np.unravel_index(int(s.argmax()), s.shape)
    return y * block, x * block, float(s.max())


def frames(args) -> None:
    gated, ungated, idx, prefix = args.gated, args.ungated, args.frame, args.prefix
    a, b = frame_at(gated, idx), frame_at(ungated, idx)
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
    by, bx, bm = worst_block(d)
    # CENTRE the crop on the worst block instead of extending from it: a block at the right edge used
    # to yield a 192 px sliver, which is not a picture a person can judge.
    ch, cw = min(576, H), min(args.width, W)
    y = int(min(max(0, by + 32 - ch // 2), H - ch))
    x = int(min(max(0, bx + 32 - cw // 2), W - cw))
    diff = np.clip(d * 16, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    left, right = (args.names.split(",") + ["left", "right"])[:2]
    png(a[y:y + ch, x:x + cw], f"{prefix}_{left}.png")
    png(b[y:y + ch, x:x + cw], f"{prefix}_{right}.png")
    png(np.stack([diff] * 3, axis=2)[y:y + ch, x:x + cw], f"{prefix}_{left}_minus_{right}_x16.png")
    print(f"frame {idx}: share {(d > 8).mean() * 100:.5f}%  peak {int(d.max())}  "
          f"worst block ({bx},{by}) mean {bm:.2f}  crop ({x},{y}) {cw}x{ch}")

    if args.clip_frames:
        half = args.clip_frames // 2
        start = max(0, idx - half)
        crop = "null" if args.full else f"crop={cw}:{ch}:{x}:{y}"
        subprocess.run([
            FF, "-loglevel", "error", "-y",
            "-ss", f"{start / 23.976:.3f}", "-i", gated, "-ss", f"{start / 23.976:.3f}", "-i", ungated,
            "-filter_complex",
            f"[0:v]{crop},scale=960:-2,setpts=PTS-STARTPTS[l];"
            f"[1:v]{crop},scale=960:-2,setpts=PTS-STARTPTS[r];"
            f"[l][r]hstack=inputs=2,format=yuv420p[v]",
            "-map", "[v]", "-frames:v", str(args.clip_frames), "-r", "23.976",
            "-c:v", "libx264", "-crf", "12", "-preset", "veryfast",
            f"{prefix}_{'full' if args.full else 'zoom'}.mp4"], check=True)
        tag = "full" if args.full else "zoom"
        print(f"  clip {args.clip_frames} frames -> {prefix}_{tag}.mp4 (gated left, ungated right)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gated")
    ap.add_argument("ungated")
    ap.add_argument("frame", type=int)
    ap.add_argument("prefix")
    ap.add_argument("--clip-frames", type=int, default=48)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--names", default="a,b",
                    help="panel names for the first and second input, e.g. int8,fp16 - 'gated,ungated' "
                         "is a claim about the gate and has mislabelled an int8-vs-fp16 pair before")
    ap.add_argument("--full", action="store_true",
                    help="whole frame instead of a 1:1 crop - what the eye sees at normal viewing")
    args = ap.parse_args(argv)
    frames(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
