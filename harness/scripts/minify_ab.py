#!/usr/bin/env python3
"""Compare the two minification arms against a Lanczos3 reference, and their phase pair against each other.

Why this exists (owner, 2026-11-25): the fit view minifies a 1440x1080 frame into a 421-wide pane (3.42x) through one
bilinear tap. The fix is a mip chain + trilinear (`RVE_MIP_MINIFY`), and the acceptance is a NUMBER, not an opinion:
PSNR and SSIM of each arm against a proper downscale, plus how far the picture moves when the sampling phase shifts by
half an output pixel (the shimmer that no single-frame score can see).

The pictures come from `harness/tests/test_preview_minify_rhi.py`, which grabs the pane IN PROCESS (`grabFrame()`) under
a leased X display - the control channel cannot capture a QRhiWidget (both `grab()` and `grabFrame()` time out there).
The reference is built with the repo's own `bin/ffmpeg` from the same chart file the pane was fed.

Usage:
    python3 harness/scripts/minify_ab.py [--dir harness/docs/wsp/verified/artifacts/preview-minify] [--json]
Exit: 0 with the table, 1 when an arm's pictures are missing (a missing arm is never a silent pass).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: The fit geometry: a 421-wide pane and a 1440x1080 frame give a 421x316 picture centred in a 421x512 pane.
PANE_W, PANE_H = 421, 512
DRAWN_W, DRAWN_H = 421, 316


def _reference(chart: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
    """Lanczos3 421x316 from the chart, with the repo's ffmpeg (the reference is the only thing ffmpeg produces here:
    an ffmpeg-made ARM would be a different resampler and would flatter the comparison)."""
    ffmpeg = ROOT / "bin" / "ffmpeg"
    subprocess.run([str(ffmpeg), "-y", "-hide_banner", "-loglevel", "error", "-i", str(chart),
                    "-vf", f"scale={DRAWN_W}:{DRAWN_H}:flags=lanczos+accurate_rnd,format=rgb24",
                    str(target)], check=True)
    return target


def _rgb(path: pathlib.Path):
    """`(width, height, list of (r,g,b) rows)` as flat lists - Pillow keeps this dependency-free and exact."""
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return rgb.size, list(rgb.tobytes())


def psnr(a: pathlib.Path, b: pathlib.Path) -> float:
    size_a, data_a = _rgb(a)
    size_b, data_b = _rgb(b)
    assert size_a == size_b, f"{a.name} is {size_a}, {b.name} is {size_b}"
    squares = 0
    for index in range(0, len(data_a)):
        diff = data_a[index] - data_b[index]
        squares += diff * diff
    mse = squares / max(1, len(data_a))
    if mse == 0:
        return float("inf")
    import math

    return 10.0 * math.log10((255.0 * 255.0) / mse)


def ssim(a: pathlib.Path, b: pathlib.Path, window: int = 8) -> float:
    """Mean SSIM over non-overlapping `window`x`window` blocks, computed on luma.

    A global SSIM would be dominated by the chart's flat areas; the blocks keep the fine structure - the part a
    minification kernel either preserves or aliases - in the score.
    """
    size_a, data_a = _rgb(a)
    size_b, data_b = _rgb(b)
    width, height = size_a
    assert size_a == size_b
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    scores: list[float] = []
    for top in range(0, height - window + 1, window):
        for left in range(0, width - window + 1, window):
            values_a: list[float] = []
            values_b: list[float] = []
            for y in range(top, top + window):
                row = y * width * 3
                for x in range(left, left + window):
                    offset = row + x * 3
                    values_a.append(0.299 * data_a[offset] + 0.587 * data_a[offset + 1] + 0.114 * data_a[offset + 2])
                    values_b.append(0.299 * data_b[offset] + 0.587 * data_b[offset + 1] + 0.114 * data_b[offset + 2])
            count = len(values_a)
            mean_a = sum(values_a) / count
            mean_b = sum(values_b) / count
            var_a = sum((value - mean_a) ** 2 for value in values_a) / count
            var_b = sum((value - mean_b) ** 2 for value in values_b) / count
            cov = sum((va - mean_a) * (vb - mean_b) for va, vb in zip(values_a, values_b)) / count
            numerator = (2 * mean_a * mean_b + c1) * (2 * cov + c2)
            denominator = (mean_a * mean_a + mean_b * mean_b + c1) * (var_a + var_b + c2)
            scores.append(numerator / denominator)
    return sum(scores) / max(1, len(scores))


def phase_delta(fit: pathlib.Path, phase: pathlib.Path) -> float:
    """Worst per-channel difference between the two phase grabs, over the picture area they share.

    The second grab is the same picture at a scale 1.0005x larger (half an output pixel). With a footprint-aware filter
    the two agree closely; with one bilinear tap the difference IS the aliasing error.
    """
    from PIL import Image

    with Image.open(fit) as first, Image.open(phase) as second:
        first = first.convert("RGB")
        second = second.convert("RGB")
        width = min(first.width, second.width)
        height = min(DRAWN_H, second.height)          # the picture band, both centred the same way
        top = max(0, (min(first.height, second.height) - height) // 2)
        a = first.load()
        b = second.load()
        worst = 0
        total = 0
        count = 0
        for y in range(top, top + height):
            for x in range(width):
                for channel in range(3):
                    difference = abs(a[x, y][channel] - b[x, y][channel])
                    worst = max(worst, difference)
                    total += difference
                    count += 1
        return {"max": worst, "mean": total / max(1, count)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=str(ROOT / "harness" / "docs" / "wsp" / "verified" / "artifacts" / "preview-minify"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    directory = pathlib.Path(args.dir)
    chart = directory / "chart-1440x1080.png"
    reference = directory / "ref-lanczos3-421x316.png"
    missing = [name for name in ("bilinear-fit.png", "mips-fit.png", "bilinear-phase.png", "mips-phase.png")
               if not (directory / name).exists()]
    if missing or not chart.exists():
        print(f"minify_ab: missing {missing or ['chart-1440x1080.png']} in {directory} - run the RHI gate for both "
              f"arms first (RVE_MIP_MINIFY=0 and =1)", file=sys.stderr)
        return 1
    _reference(chart, reference)

    # Both grabs are the WHOLE pane (421x512); the picture sits in a 421x316 band, centred. Crop before scoring, or the
    # letterbox would dominate every metric.
    from PIL import Image

    crops: dict[str, pathlib.Path] = {}
    for arm in ("bilinear", "mips"):
        source = directory / f"{arm}-fit.png"
        with Image.open(source) as image:
            rgb = image.convert("RGB")
            top = (rgb.height - DRAWN_H) // 2
            crop = rgb.crop((0, top, DRAWN_W, top + DRAWN_H))
            out = directory / f"{arm}-fit-crop.png"
            crop.save(out)
            crops[arm] = out

    rows = {}
    for arm in ("bilinear", "mips"):
        phase = phase_delta(directory / f"{arm}-fit.png", directory / f"{arm}-phase.png")
        rows[arm] = {
            "psnr_db": psnr(crops[arm], reference),
            "ssim": ssim(crops[arm], reference),
            "phase_max_delta": phase["max"],
            "phase_mean_delta": phase["mean"],
        }
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(f"{'arm':>9} {'PSNR dB':>9} {'SSIM':>8} {'phase max':>10} {'phase mean':>11}")
        for arm in ("bilinear", "mips"):
            row = rows[arm]
            print(f"{arm:>9} {row['psnr_db']:9.2f} {row['ssim']:8.4f} {row['phase_max_delta']:10.1f} "
                  f"{row['phase_mean_delta']:11.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
