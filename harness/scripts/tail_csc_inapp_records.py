"""Records for the in-app half of the tail-CSC row: the three-arm A/B, the phase split, and the pair.

The isolated probe (``tail_csc_probe.py``, records N001-N003) measured the kernel against the old
conversion inside one arm. This reader takes the numbers that decide the row, in the app:

* the render loop per arm (fp16 / conv / yuv), median of rounds 2+, one session, interleaved;
* the ratio against the bar - a quotient of two records, never a typed number;
* the two phase lines the arms move (tail and pack+queue), so the saving is attributed and not just
  asserted;
* the picture: ``gate_sweep``'s per-frame table over the 494-frame ffv1 pair, different-arithmetic
  (the chroma rounding order moves), reference-kind config.

Usage:
  python harness/scripts/tail_csc_inapp_records.py \
      --logs docs/wsp/verified/logs/tail-csc-ratio \
      --pair docs/wsp/verified/logs/tail-csc-pair/pair.json \
      --out docs/wsp/verified/records/int8-tail-csc-inapp.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys

import os as _bos, sys as _bsys
_HARNESS_ROOT = _bos.path.dirname(_bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
if _HARNESS_ROOT not in _bsys.path:
    _bsys.path.insert(0, _HARNESS_ROOT)

from harness.core import Population, Record, Store  # noqa: E402

CLIP = "bd_violet20s.mkv"          # the label
CLIPFILE = "bd_violet20s"          # the log-name stem ("${clip%.mkv}")
STAMP = re.compile(r"^# wall_ab (\S+)", re.M)
LOOP = re.compile(r"render loop: [\d.]+ s for (\d+) frames \(([\d.]+) ms/frame\)")
TAIL = re.compile(r"int8 phases: head [\d.]+\s+quantise [\d.]+\s+hops [\d.]+\s+tail ([\d.]+)")
PACK = re.compile(r"profile over \d+ frames: .*?pack\+queue ([\d.]+) ms")


def read_arm(logdir: str, arm: str, rounds: list[int], sink: str = "null") -> dict:
    """Per-round loop ms, and the last tail / pack line of each round."""
    out = {"ms": {}, "tail": {}, "pack": {}, "session": None, "path": None}
    for r in rounds:
        path = os.path.join(logdir, f"{CLIPFILE}_{arm}_r{r}_{sink}.log")
        if not os.path.exists(path):
            raise SystemExit(f"missing log: {path}")
        text = open(path, errors="replace").read()
        if out["session"] is None and (s := STAMP.search(text)):
            out["session"] = f"s-{s.group(1)}"
        m = LOOP.search(text)
        if not m:
            raise SystemExit(f"no loop line in {path}")
        out["ms"][r] = float(m.group(2))
        t = TAIL.findall(text)
        p = PACK.findall(text)
        if t:
            out["tail"][r] = float(t[-1])
        if p:
            out["pack"][r] = float(p[-1])
        if out["path"] is None:
            out["path"] = path
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="docs/wsp/verified/logs/tail-csc-ratio")
    ap.add_argument("--pair", default="docs/wsp/verified/logs/tail-csc-pair/pair.json")
    ap.add_argument("--pair-records", default="docs/wsp/verified/logs/tail-csc-pair/pair-records.jsonl")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-tail-csc.jsonl",
                    help="ONE file per row: the isolated N-records are carried over, the pair's "
                         "TC-records appended, so a single --records validates the whole doc")
    ap.add_argument("--first-round", type=int, default=2)
    ap.add_argument("--nvenc-logs", default="docs/wsp/verified/logs/tail-csc-nvenc",
                    help="the same two arms with hevc_nvenc in the loop: the product's own number")
    args = ap.parse_args(argv)

    arms = ("fp16", "conv", "yuv")
    rounds = [1, 2, 3]
    data = {a: read_arm(args.logs, a, rounds) for a in arms}
    kept = [r for r in rounds if r >= args.first_round]
    session = data["yuv"]["session"] or "unknown"

    store = Store(session=session)
    # the isolated probe's records (N001-N003) live in the same file and must survive the rewrite
    if os.path.exists(args.out) and os.path.abspath(args.out) != os.path.abspath(args.pair_records):
        for line in open(args.out):
            if line.strip():
                rec = Record.from_dict(json.loads(line))
                if not rec.id.startswith(("JC", "TC")):
                    store.records[rec.id] = rec
    store.seq = len(store.records)
    out: list[Record] = []

    pop = Population.of("rounds", [f"{CLIP}#null#round-{r}" for r in kept],
                        f"{CLIP}, null sink, rounds {kept[0]}-{kept[-1]} of one session, arms "
                        f"interleaved (round 1 dropped: the draw right after engine load is warmup)")
    instr = "the app's own render loop: wall clock / frames, null sink (rawvideo + null muxer)"
    config = (f"{CLIP}, null sink, reuse off (--no_temporal_reuse), one process per arm; "
              f"yuv = tail writes yuv420p (RVE_TAIL_YUV=1), conv = tail writes fp16 RGB and the pack "
              f"converts (RVE_TAIL_YUV=0)")

    ids: dict[str, str] = {}
    for arm in arms:
        rec = Record(value=round(statistics.median(data[arm]["ms"][r] for r in kept), 2), unit="ms",
                     population=pop, instrument=instr, config=config, session=session,
                     statistic="median", tier="measured", id=f"JC{len(out) + 1:03d}",
                     run="rve-backend.py, logs from scripts/wall_ab.sh", artifact=data[arm]["path"],
                     extra={"of": "frame_time", "arm": arm,
                            "artifact_sha256": hashlib.sha256(
                                open(data[arm]["path"], "rb").read()).hexdigest()})
        store.emit(rec)
        out.append(rec)
        ids[arm] = rec.id

    def ratio(a: str, b: str, note: str) -> None:
        # Store.ratio emits under an automatic id; rename in place (a second emit would store a
        # second copy of the same quotient under the new id)
        rec = store.ratio(ids[a], ids[b], instrument=instr, note=note)
        store.records.pop(rec.id, None)
        rec.id = f"JC{len(out) + 1:03d}"
        store.records[rec.id] = rec
        out.append(rec)

    ratio("fp16", "yuv", f"{CLIP}, null sink: the lane's ratio against bar_int8_wall")
    ratio("fp16", "conv", f"{CLIP}, null sink: the same ratio before this change")
    ratio("conv", "yuv", f"{CLIP}, null sink: what the tail CSC itself is worth")

    # the two phase lines the change moves, per arm
    for arm in ("conv", "yuv"):
        for phase, series in (("tail", data[arm]["tail"]), ("pack", data[arm]["pack"])):
            if not all(r in series for r in kept):
                continue
            rec = Record(value=round(statistics.median(series[r] for r in kept), 2), unit="ms",
                         population=pop,
                         instrument=("the body's own phase counter (last 8 frames) for tail; the "
                                     "profile summary for pack+queue"),
                         config=config, session=session, statistic="median", tier="measured",
                         id=f"JC{len(out) + 1:03d}",
                         run="rve-backend.py, logs from scripts/wall_ab.sh", artifact=data[arm]["path"],
                         extra={"of": "frame_time", "phase": phase, "arm": arm,
                                "artifact_sha256": hashlib.sha256(
                                    open(data[arm]["path"], "rb").read()).hexdigest()})
            store.emit(rec)
            out.append(rec)

    # with the encoder in the loop: the product number (the bar is "wall clock under the gate")
    if os.path.isdir(args.nvenc_logs):
        nvrounds = set()
        for f in os.listdir(args.nvenc_logs):
            mm = re.search(r"_r(\d+)_nvenc\.log$", f)
            if mm:
                nvrounds.add(int(mm.group(1)))
        nvrounds = sorted(nvrounds)
        nvkept = [r for r in nvrounds if r >= args.first_round]
        if nvkept and os.path.exists(os.path.join(args.nvenc_logs,
                                                  f"{CLIPFILE}_fp16_r{nvkept[0]}_nvenc.log")):
            nv = {a: read_arm(args.nvenc_logs, a, nvrounds, sink="nvenc") for a in ("fp16", "yuv")}
            nvsession = nv["yuv"]["session"] or "unknown"
            nvpop = Population.of("rounds", [f"{CLIP}#nvenc#round-{r}" for r in nvkept],
                                  f"{CLIP}, hevc_nvenc sink, rounds {nvkept[0]}-{nvkept[-1]} of one "
                                  f"session, arms interleaved")
            nvinstr = "the app's own render loop: wall clock / frames with hevc_nvenc encoding alongside"
            nvconfig = (f"{CLIP}, hevc_nvenc sink, reuse off (--no_temporal_reuse), one process per arm; "
                        f"yuv = tail writes yuv420p (RVE_TAIL_YUV=1), conv = the RGB tail")
            nvids = {}
            for arm in ("fp16", "yuv"):
                rec = Record(value=round(statistics.median(nv[arm]["ms"][r] for r in nvkept), 2),
                             unit="ms", population=nvpop, instrument=nvinstr, config=nvconfig,
                             session=nvsession, statistic="median", tier="measured",
                             id=f"JC{len(out) + 1:03d}", run="rve-backend.py, logs from scripts/wall_ab.sh",
                             artifact=nv[arm]["path"],
                             extra={"of": "frame_time", "arm": arm,
                                    "artifact_sha256": hashlib.sha256(
                                        open(nv[arm]["path"], "rb").read()).hexdigest()})
                store.emit(rec)
                out.append(rec)
                nvids[arm] = rec.id
            rec = store.ratio(nvids["fp16"], nvids["yuv"], instrument=nvinstr,
                              note=f"{CLIP}, hevc_nvenc: the product's ratio against bar_int8_wall")
            store.records.pop(rec.id, None)
            rec.id = f"JC{len(out) + 1:03d}"
            store.records[rec.id] = rec
            out.append(rec)

    # the picture: gate_sweep's own records for the pair (ids TC001..TC004 there), and the summary
    pair = json.load(open(args.pair))
    if os.path.exists(args.pair_records):
        for line in open(args.pair_records):
            out.append(Record.from_dict(json.loads(line)))

    store.dump(args.out)
    if os.path.exists(args.pair_records):
        # the pair's own records (ids TC001..TC004) are emitted by gate_sweep from the same table;
        # carry them into this file so a claim can cite them next to the timings
        with open(args.out, "a") as fh:
            fh.write(open(args.pair_records).read())
    for rec in out:
        print(rec.claims_line())
    print(f"# {len(out)} records -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
