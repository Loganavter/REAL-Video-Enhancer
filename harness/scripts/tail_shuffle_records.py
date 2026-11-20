"""Records for the tail's epilogue fold: the isolated pair, the app A/B, and the identity on 494 frames.

Usage: python harness/scripts/tail_shuffle_records.py [--probe .../probe.json] [--logs .../ab]
                                              [--pair .../pair-records.jsonl]
                                              [--out docs/wsp/verified/records/int8-tail-shuffle.jsonl]
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

LOOP = re.compile(r"render loop: [\d.]+ s for (\d+) frames \(([\d.]+) ms/frame\)")
STAMP = re.compile(r"^# wall_ab (\S+)", re.M)


def sha256(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="docs/wsp/verified/logs/tail-shuffle/probe.json")
    ap.add_argument("--logs", default="docs/wsp/verified/logs/tail-shuffle/ab")
    ap.add_argument("--pair", default="docs/wsp/verified/records/int8-tail-shuffle-pair.jsonl")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-tail-shuffle.jsonl")
    args = ap.parse_args(argv)

    pr = json.load(open(args.probe))
    pdig, psess = sha256(args.probe), f"s-{pr['take']}"
    clip = pr["clip"]
    pop = Population.of("frames", (f"{clip}#frame{pr['frame']}",),
                        f"{clip} frame {pr['frame']}, tail conv + shuffle + residual at 1080p -> 4K")
    inst = (f"torch.cuda.Event around {pr['iters']} calls behind 3 warm-ups, {pr['rounds']} "
            f"interleaved rounds, medians of rounds 2+")
    cfg = "int8 body, tail = int8 conv (12 ch) + nn.PixelShuffle(2) + nearest residual"
    st = Store(session=psess)
    for rid, arm, label in (("Q001", "today", "conv, then PixelShuffle, then the residual add"),
                            ("Q002", "conv", "the int8 tail conv alone"),
                            ("Q003", "fused", "one kernel: conv + shuffle + residual")):
        st.emit(Record(value=round(pr["median"][arm], 3), unit="ms",
                       population=Population(kind=pop.kind, items=pop.items,
                                             label=f"{pop.label}, arm {arm}: {label}"),
                       instrument=inst, config=cfg, session=psess, statistic="median", tier="measured",
                       id=rid, run="scripts/tail_shuffle_probe.py", artifact=args.probe,
                       extra={"artifact_sha256": pdig, "arm": arm, "of": "tail_time"}))
    st.delta("Q003", "Q001", note="the fold's saving on the tail, one frame")
    st.emit(Record(value=float(pr["identity"]["differing"]), unit="count", population=pop,
                   instrument="element-wise equality between the folded kernel's output and the "
                              "three-step path's output, same codes, same frame",
                   config=cfg, session=psess, statistic="single", tier="measured", id="Q004",
                   run="scripts/tail_shuffle_probe.py", artifact=args.probe,
                   extra={"artifact_sha256": pdig, "of": "identity",
                          "total_values": pr["identity"]["total"],
                          "note": "0 means the fold computes the same two additions on the same 12 "
                                  "values in the same order - a layout change, not arithmetic"}))

    loops: dict[str, list[float]] = {}
    session = "unknown"
    for name in sorted(os.listdir(args.logs)):
        m = re.match(r"^(.+)_(shuf0|shuf1)_r(\d+)_null\.log$", name)
        if not m:
            continue
        text = open(os.path.join(args.logs, name), errors="replace").read().replace("\r", "\n")
        if session == "unknown" and (s0 := STAMP.search(text)):
            session = f"s-{s0.group(1)}"
        lm = LOOP.search(text)
        if lm:
            loops.setdefault(m.group(2), []).append(float(lm.group(2)))
    if loops:
        pop_ab = Population.of("frames", (f"{clip}#first494",),
                               f"{clip}, 494-frame render, null sink, shuffle folded vs separate, arms "
                               f"alternating in one session")
        cfg_ab = "int8 fit, null sink, arms alternating in one session"
        ids = {}
        for rid, arm in (("Q005", "shuf0"), ("Q006", "shuf1")):
            art = f"{args.logs}/{clip.removesuffix('.mkv')}_{arm}_r1_null.log"
            st.emit(Record(value=round(statistics.median(loops[arm]), 2), unit="ms",
                           population=Population(kind=pop_ab.kind, items=pop_ab.items,
                                                 label=f"{pop_ab.label}, arm {arm}"),
                           instrument="the render loop line, ms/frame", config=cfg_ab, session=session,
                           statistic="median", tier="measured", id=rid,
                           run=f"rve-backend.py, logs {args.logs}/*_{arm}_r*_null.log", artifact=art,
                           extra={"artifact_sha256": sha256(art), "of": "frame_time",
                                  "per_round": loops[arm]}))
            ids[arm] = rid
        st.delta(ids["shuf1"], ids["shuf0"], note="the fold's effect on the frame")

    if args.pair and os.path.exists(args.pair):
        for rid, rec in Store.load(args.pair).records.items():
            st.emit(rec)

    st.dump(args.out)
    print(st.claims_text().strip())
    print(f"# -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
