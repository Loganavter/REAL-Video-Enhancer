"""Emit the numbers a render LOG states, as records - the tuner's floor, the gate's decision, the time.

A render log is the product's own instrument: the tuner prints the noise floor and the certified chain,
the gate prints how many frames it reused, and the loop prints ms/frame. Reading them into records by
hand is how a number loses its population, so this parser is the only path from a log to a claim.

Usage:
  python harness/harness/scripts/emit_render_records.py LOG [LOG ...] --clip bd_violet20s.mkv \
      --artifact docs/wsp/verified/logs/bd/vio_luma.log --config "fp16, auto luma gate" \
      --out docs/wsp/verified/records/gate-bd.jsonl --id-prefix B
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)

from harness.core import Population, Record  # noqa: E402

FLOOR_OK = re.compile(r"auto gate: threshold ([\d.]+)/255, chain (\d+), veto (\S+), noise floor ([\d.]+), "
                      r"would reuse ([\d.]+)%")
FLOOR_REFUSE = re.compile(r"auto gate: noise floor ([\d.]+) is above the ([\d.]+) limit")
REUSE = re.compile(r"reference reuse: (\d+)/(\d+) frames reused \(([\d.]+)%\)")
LOOP = re.compile(r"render loop: [\d.]+ s for (\d+) frames \(([\d.]+) ms/frame\)")
PROBE = re.compile(r"int8 probe: engine ([\d.]+)%\s+composition ([\d.]+)%\s+peak (\d+)\s+\((\d+) pairs")
COMPOSITION = re.compile(r"composition d1 ([\d.]+)% / d2 ([\d.]+)%")


def read(path: str) -> str:
    # the app writes progress with \r, so a line-oriented read would miss the gate lines
    return open(path, errors="replace").read().replace("\r", "\n")


def facts(text: str) -> dict:
    out: dict = {}
    if m := FLOOR_OK.search(text):
        out.update(threshold=float(m.group(1)), chain=int(m.group(2)), noise_floor=float(m.group(4)),
                   would_reuse=float(m.group(5)), refused=False)
    elif m := FLOOR_REFUSE.search(text):
        out.update(noise_floor=float(m.group(1)), refuse_limit=float(m.group(2)), refused=True)
    if m := REUSE.search(text):
        out.update(reused=int(m.group(1)), frames=int(m.group(2)), reuse_share=float(m.group(3)))
    if m := LOOP.search(text):
        out.update(frames_rendered=int(m.group(1)), ms_per_frame=float(m.group(2)))
    if m := PROBE.search(text):
        out.update(engine=float(m.group(1)), composition=float(m.group(2)),
                   peak=int(m.group(3)), pairs=int(m.group(4)))
    if m := COMPOSITION.search(text):
        out.update(comp_d1=float(m.group(1)), comp_d2=float(m.group(2)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--instrument-prefix", default="")
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--session", default="s-2026-11-18T08:00:00Z")
    ap.add_argument("--id-prefix", default="B")
    ap.add_argument("--calibration-frames", type=int, default=16)
    ap.add_argument("--with-time", action="store_true",
                    help="emit ms/frame even when the log name does not say 'null' - only for runs whose "
                         "output really was the null sink")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--start", type=int, default=1, help="first record number")
    args = ap.parse_args(argv)

    digest = hashlib.sha256(open(args.artifact, "rb").read()).hexdigest()
    cal = Population.of("frames", [f"{args.clip}#frame{i}" for i in range(args.calibration_frames)],
                        f"the first {args.calibration_frames} rendered frames (the tuner's window)")
    pop_clip = Population.of("clips", [args.clip], args.label)
    recs, seq = [], args.start - 1
    specs = [
        ("noise_floor", "levels", "single", "the app's auto gate tuner, median of the calmest third of "
                                       "the observed LR block-mean deltas", cal),
        ("reuse_share", "%", "single", "the gate's own counter: reused frames / frames", pop_clip),
        ("ms_per_frame", "ms", "mean", "render loop wall clock / frames (the output sink decides what "
                                       "this contains)", pop_clip),
        ("engine", "%", "worst_of", "the app's int8 probe: int8(t) against fp16(t) on its pairs", pop_clip),
        ("composition", "%", "worst_of", "the app's int8 probe: int8(t-1) against fp16(t)", pop_clip),
        ("peak", "levels", "max", "the app's int8 probe: peak channel delta on its pairs", pop_clip),
    ]
    for log in args.logs:
        f = facts(read(log))
        for key, unit, stat, instrument, pop in specs:
            if key not in f:
                continue
            if key == "ms_per_frame" and not (args.with_time or "null" in os.path.basename(log).lower()):
                continue  # a file sink measures the ffv1 writer; only the null sink measures the render
            seq += 1
            recs.append(Record(value=float(f[key]), unit=unit, population=pop,
                               instrument=(args.instrument_prefix + instrument),
                               config=f"{args.config} ({os.path.basename(log)})",
                               session=args.session, statistic=stat, tier="measured",
                               run=f"rve-backend.py, log {os.path.basename(log)}",
                               artifact=args.artifact, extra={"artifact_sha256": digest,
                                                              "reuse_refused": f.get("refused", None),
                                                              "pairs": f.get("pairs")},
                               id=f"{args.id_prefix}{seq:03d}"))
    mode = "a" if args.append else "w"
    with open(args.out, mode) as fh:
        for r in recs:
            r.validate()
            fh.write(json.dumps(r.to_dict()) + "\n")
    for r in recs:
        print(r.claims_line())
    print(f"# {len(recs)} records -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
