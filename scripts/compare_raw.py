"""Compare two rendered frames stored as raw fp16 RGB, in levels.

The pair this exists for: the Triton int8 pipeline (`int8_pipeline.py --dump`) against the fp16 model's
own output on the same frame. That is the half of E9' the ledger marks as unmeasured ("the accuracy of
the design is unmeasured, not good"), and it needs no TensorRT at all - just two files and the
definition the project uses everywhere: the share of pixels whose WORST channel is more than 8 levels
off, the peak, and PSNR.

Usage: python3 scripts/compare_raw.py ref.raw cand.raw --width 1920 --height 1080 [--scale 2] [--emit F]
"""
from __future__ import annotations

import argparse
import hashlib
import sys

import numpy as np


def load(path: str, h: int, w: int, scale: int) -> np.ndarray:
    a = np.fromfile(path, dtype=np.float16)
    want = h * scale * w * scale * 3
    if a.size != want:
        raise SystemExit(f"{path}: {a.size} halves, expected {want} ({w*scale}x{h*scale}x3)")
    return a.reshape(h * scale, w * scale, 3).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("cand")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--emit", default="")
    args = ap.parse_args()

    a, b = load(args.ref, args.height, args.width, args.scale), load(args.cand, args.height, args.width, args.scale)
    d = np.abs(a - b) * 255.0                                   # levels
    worst = d.max(axis=2)                                       # worst channel per pixel
    share = float((worst > 8).mean() * 100.0)
    peak = float(d.max())
    mean = float(d.mean())
    mse = float(((a - b) ** 2).mean())
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
    sha = hashlib.sha256(open(args.cand, "rb").read()).hexdigest()[:16]
    print(f"{args.cand} ({sha}) vs {args.ref}: {args.width*args.scale}x{args.height*args.scale}")
    print(f"share above 8 levels (worst channel): {share:.3f}%   peak {peak:.2f} levels   "
          f"mean {mean:.2f}   PSNR {psnr:.2f} dB")

    if args.emit:
        sys.path.insert(0, "scripts")
        from harness import Population, Record, Store
        store = Store()
        pop = Population.of("pixels of one frame", [sha],
                            label=f"{args.cand.split('/')[-1]} vs {args.ref.split('/')[-1]} "
                                  f"at {args.width*args.scale}x{args.height*args.scale}")
        common = dict(population=pop, instrument="raw fp16 frames, worst channel in levels",
                      config=f"{args.width}x{args.height} -> x{args.scale}",
                      session=store.session, run=f"run-raw-diff-{sha}",
                      extra={"artifact_sha256": sha, "of": "share", "quantity": "accuracy"})
        store.emit(Record(value=round(share, 4), unit="%", statistic="single", **common))
        store.emit(Record(value=round(peak, 2), unit="levels", statistic="max",
                          **{**common, "extra": {**common["extra"], "of": "peak"}}))
        store.emit(Record(value=round(psnr, 2), unit="dB", statistic="single",
                          **{**common, "extra": {**common["extra"], "of": "quality"}}))
        store.dump(args.emit)
        print(f"records -> {args.emit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
