"""The acceptance statistic on a render pair, with its population and its provenance.

The decision statistic of this project is local and in RGB: the share of pixels whose worst channel is
more than 8 levels off, on the worst frame of the sample, bar 0.1%. This script is the instrument for
it: it decodes two lossless renders frame by frame, emits the per-frame table, and (with ``--emit``)
emits the records the claim cites, so the probe, the record and the claim are the same object.

Two controls it prints on every run, because both have been the cause of a withdrawn conclusion:
  * the identity row - a pair that is the same file must read 0.00000% and peak 0;
  * the uncovered scope - what this statistic does not see (no temporal judgement, no perceptual
    model, no check that the reference itself is right).

Usage:
  python scripts/gate_sweep.py GATED UNGATED --out stats.json [--block 64] [--emit records.jsonl]
         [--clip-name bd_violet20s.mkv] [--label "..."] [--config "..."] [--instrument "..."]
         [--session s-...] [--artifact docs/wsp/verified/logs/x.log] [--ffmpeg PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

BAR = 8.0  # levels; the acceptance bar's own threshold


def _frame_iter(path: str, ffmpeg: str, w: int, h: int):
    cmd = [ffmpeg, "-loglevel", "error", "-i", path, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 2)
    size = w * h * 3
    try:
        while True:
            raw = proc.stdout.read(size)
            if len(raw) < size:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.wait()


def sweep(gated: str, ungated: str, ffmpeg: str, w: int, h: int, block: int) -> list[dict]:
    hh, ww = (h // block) * block, (w // block) * block
    rows: list[dict] = []
    for i, (a, b) in enumerate(zip(_frame_iter(gated, ffmpeg, w, h),
                                  _frame_iter(ungated, ffmpeg, w, h))):
        d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
        blk = d[:hh, :ww].reshape(hh // block, block, ww // block, block).mean(axis=(1, 3))
        rows.append({"frame": i,
                     "share": float((d > BAR).sum()) / (w * h) * 100.0,
                     "peak": int(d.max()),
                     "block_mean": float(blk.max()),
                     "sum_abs": int(d.sum())})
    return rows


def summary(rows: list[dict], block: int, bar: float = 0.1) -> dict:
    if not rows:
        raise SystemExit("no frames decoded - the pair has no common frames")
    w = max(rows, key=lambda r: r["share"])
    m = max(rows, key=lambda r: r["block_mean"])
    return {"frames": len(rows), "block": block,
            "worst_share": w["share"], "worst_share_frame": w["frame"],
            "worst_peak": max(r["peak"] for r in rows),
            "frames_above_bar": sum(1 for r in rows if r["share"] > bar),
            "worst_block_mean": m["block_mean"], "worst_block_mean_frame": m["frame"],
            "identity": all(r["sum_abs"] == 0 for r in rows)}


def emit_records(rows: list[dict], s: dict, args, digest: str) -> list[dict]:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from harness.core import Population, Record

    items = [f"{args.clip_name}#frame{r['frame']}" for r in rows]
    pop = Population.of("frames", items, args.label)
    common = dict(population=pop, session=args.session, run=args.run,
                  artifact=args.artifact, extra={"artifact_sha256": digest})
    recs = [
        Record(value=s["worst_share"], unit="%", instrument=f"{args.instrument}, {s['block']}px blocks",
               config=args.config + f" (worst frame {s['worst_share_frame']})", statistic="worst_of",
               **common),
        Record(value=float(s["frames_above_bar"]), unit="count",
               instrument=args.instrument, config=args.config + f" (bar {args.bar}%)",
               statistic="single", **common),
        Record(value=s["worst_block_mean"], unit="levels", instrument=args.instrument,
               config=args.config + f" (worst block mean, frame {s['worst_block_mean_frame']})",
               statistic="max", **common),
        Record(value=float(s["worst_peak"]), unit="levels", instrument=args.instrument,
               config=args.config + " (peak channel delta on any frame)", statistic="max", **common),
    ]
    out = []
    for i, r in enumerate(recs, 1):
        r.id = f"{args.id_prefix}{i:03d}"
        out.append(r.to_dict())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gated")
    ap.add_argument("ungated")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ffmpeg", default="/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/bin/ffmpeg")
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--height", type=int, default=2160)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--bar", type=float, default=0.1)
    ap.add_argument("--emit", default="")
    ap.add_argument("--clip-name", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--instrument", default="channel-max |delta| in RGB")
    ap.add_argument("--config", default="")
    ap.add_argument("--session", default="")
    ap.add_argument("--run", default="")
    ap.add_argument("--artifact", default="")
    ap.add_argument("--id-prefix", default="RB")
    args = ap.parse_args(argv)

    rows = sweep(args.gated, args.ungated, args.ffmpeg, args.width, args.height, args.block)
    s = summary(rows, args.block, args.bar)
    json.dump({"summary": s, "rows": rows}, open(args.out, "w"))

    print(f"{args.gated} vs {args.ungated}")
    print(f"  {s['frames']} frames, block {s['block']}")
    print(f"  worst share {s['worst_share']:.5f}% (frame {s['worst_share_frame']}), "
          f"frames above {args.bar}%: {s['frames_above_bar']}")
    print(f"  worst block mean {s['worst_block_mean']:.2f} (frame {s['worst_block_mean_frame']}), "
          f"peak {s['worst_peak']}")
    if s["identity"]:
        print("  IDENTITY: every frame is bit-identical between the two files")
    print("  NOT covered: temporal judgement beyond the pair, perceptual models, whether the ungated "
          "reference is right, and whether this clip is representative of the title.")

    if args.emit:
        # The digest is of the file the record NAMES: the per-frame table, not the log prose.
        target = args.artifact if args.artifact and os.path.exists(args.artifact) else args.out
        digest = hashlib.sha256(open(target, "rb").read()).hexdigest()
        recs = emit_records(rows, s, args, digest)
        with open(args.emit, "w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        print(f"  emitted {len(recs)} records -> {args.emit} (artifact sha256 {digest[:16]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
