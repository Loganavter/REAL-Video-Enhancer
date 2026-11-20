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
  python harness/scripts/gate_sweep.py GATED UNGATED --out stats.json [--block 64] [--emit records.jsonl]
         [--clip-name bd_violet20s.mkv] [--label "..."] [--config "..."] [--instrument "..."]
         [--session s-...] [--artifact harness/docs/wsp/verified/logs/x.log] [--ffmpeg PATH]
"""

from __future__ import annotations
import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

BAR = 8.0  # levels; the acceptance bar's own threshold

# Which side is the reference, and what the pair can therefore say. The rule exists because a comparison
# against an FP16 ENGINE RENDER was used once as if it measured the quantiser: the engine deviates from the
# model by 0.055-0.087% and the colour pipe alone moves 0.21% of pixels past the 8-level bar on delivered
# 4:2:0 material (int8-scales-bd.txt), while the quantiser's own error against the MODEL is 0.004-0.019%.
# So the tool asks, the records carry the answer, and `harness/scripts/check_reference_declarations.py` keeps the
# undeclared ones visible instead of silently forgiven.
REFERENCE_NOTES = {
    "model": "reference is the MODEL's own fp16 output (torch): the only reference that measures a "
             "renderer's own error",
    "engine-render": "reference is a render of the FP16 ENGINE: the difference includes the engine "
                     "(0.055-0.087% against the model) and the colour pipe (0.21% on its own), so it "
                     "measures the PAIR, not the quantiser",
    "config": "reference is the same renderer in another CONFIGURATION: the difference is that "
              "configuration's effect, not an error",
}


def declare_reference(kind: str, force: bool = False) -> str:
    """Refuse to measure a pair without saying what the second file IS."""
    if kind in REFERENCE_NOTES:
        return REFERENCE_NOTES[kind]
    if not force:
        raise SystemExit(
            "refusing to compare two files without saying what the reference is.\n"
            "  Pass --reference-kind {model|engine-render|config}.\n"
            "  model         - the model's own fp16 output; the only one that measures an error\n"
            "  engine-render - a render of the fp16 ENGINE; includes engine + colour pipe, measures the pair\n"
            "  config        - the same renderer in another configuration; measures that change\n"
            "  (--force-undeclared writes nothing usable: the record would carry no reference)")
    return "UNDECLARED reference (forced): this number cannot say what it measures"


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
    reference_note = declare_reference(args.reference_kind, getattr(args, "force_undeclared", False))
    instrument = f"{args.instrument} (reference: {args.reference_kind})"
    common = dict(population=pop, session=args.session, run=args.run,
                  artifact=args.artifact,
                  extra={"artifact_sha256": digest, "reference_kind": args.reference_kind,
                         "reference_note": reference_note})
    recs = [
        Record(value=s["worst_share"], unit="%", instrument=f"{instrument}, {s['block']}px blocks",
               config=args.config + f" (worst frame {s['worst_share_frame']})", statistic="worst_of",
               **common),
        Record(value=float(s["frames_above_bar"]), unit="count",
               instrument=instrument, config=args.config + f" (bar {args.bar}%)",
               statistic="single", **common),
        Record(value=s["worst_block_mean"], unit="levels", instrument=instrument,
               config=args.config + f" (worst block mean, frame {s['worst_block_mean_frame']})",
               statistic="max", **common),
        Record(value=float(s["worst_peak"]), unit="levels", instrument=instrument,
               config=args.config + " (peak channel delta on any frame)", statistic="max", **common),
    ]
    out = []
    for i, r in enumerate(recs, 1):
        r.id = f"{args.id_prefix}{i:03d}"
        out.append(r.to_dict())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    # with --from-json the pair is already decoded; the two positional paths are not needed
    ap.add_argument("gated", nargs="?")
    ap.add_argument("ungated", nargs="?")
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
    ap.add_argument("--from-json", default="",
                    help="emit from a saved per-frame table instead of re-decoding the pair")
    ap.add_argument("--reference-kind", default="undeclared",
                    choices=("undeclared", "model", "engine-render", "config"),
                    help="what the SECOND file is - see REFERENCE_NOTES; undeclared is refused")
    ap.add_argument("--force-undeclared", action="store_true",
                    help="run anyway; the output says the number cannot say what it measures")
    args = ap.parse_args(argv)
    reference_note = declare_reference(args.reference_kind, args.force_undeclared)

    if args.from_json:
        saved = json.load(open(args.from_json))
        rows, s = saved["rows"], saved["summary"]
    else:
        rows = sweep(args.gated, args.ungated, args.ffmpeg, args.width, args.height, args.block)
        s = summary(rows, args.block, args.bar)
    if not args.from_json:
        json.dump({"summary": s, "rows": rows}, open(args.out, "w"))

    print(f"{args.gated} vs {args.ungated}")
    print(f"  reference ({args.reference_kind}): {reference_note}")
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
        from harness.core import Record as _R
        for r in recs:
            print("  " + _R.from_dict(r).claims_line())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
