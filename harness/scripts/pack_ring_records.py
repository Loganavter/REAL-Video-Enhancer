"""Records for the pack-ring falsifier: the loop, the pack line, and the payload identity.

Usage: python harness/harness/scripts/pack_ring_records.py [--logs docs/wsp/verified/logs/pack-ring/ab]
                                           [--pair docs/wsp/verified/records/int8-pack-ring-pair.jsonl]
                                           [--out docs/wsp/verified/records/int8-pack-ring.jsonl]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import statistics
import sys

import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)

from harness.core import Population, Record, Store  # noqa: E402

LOOP = re.compile(r"render loop: [\d.]+ s for (\d+) frames \(([\d.]+) ms/frame\)")
PACK = re.compile(r"pack\+queue ([\d.]+)")
STAMP = re.compile(r"^# wall_ab (\S+)", re.M)


def sha256(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="docs/wsp/verified/logs/pack-ring/ab")
    ap.add_argument("--sink", default="null", help="null | nvenc - the sink decides what the loop contains")
    ap.add_argument("--id-base", type=int, default=0,
                    help="offset added to the record numbers, so two sinks can share one file")
    ap.add_argument("--pair", default="docs/wsp/verified/records/int8-pack-ring-pair.jsonl")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-pack-ring.jsonl")
    args = ap.parse_args(argv)

    loops: dict[str, list[float]] = {}
    packs: dict[str, list[float]] = {}
    session = "unknown"
    n_frames = 0
    for name in sorted(os.listdir(args.logs)):
        m = re.match(rf"^(.+)_(ring0|ring1)_r(\d+)_{args.sink}\.log$", name)
        if not m:
            continue
        text = open(os.path.join(args.logs, name), errors="replace").read().replace("\r", "\n")
        if session == "unknown" and (s := STAMP.search(text)):
            session = f"s-{s.group(1)}"
        lm = LOOP.search(text)
        if lm:
            n_frames = max(n_frames, int(lm.group(1)))
            if int(m.group(3)) >= 2:
                loops.setdefault(m.group(2), []).append(float(lm.group(2)))
        pm = PACK.search(text)
        if pm and int(m.group(3)) >= 2:
            packs.setdefault(m.group(2), []).append(float(pm.group(1)))

    clip = "bd_violet20s.mkv"
    pop = Population.of("rounds", [f"{clip}#{args.sink}#round-{r}" for r in (2, 3)],
                        f"{clip}, {args.sink} sink, rounds 2-3 of one session, {n_frames} frames")
    cfg = f"int8 fit, {args.sink} sink, reuse off, arms alternating in one session"
    st = Store(session=session)
    ids = {}
    def rid(n: int) -> str:
        return f"K{args.id_base + n:03d}"

    for ridx, arm, what, src, unit in ((1, "ring0", "loop", loops, "ms"),
                                       (2, "ring1", "loop", loops, "ms"),
                                       (4, "ring0", "pack", packs, "ms"),
                                       (5, "ring1", "pack", packs, "ms")):
        art = f"{args.logs}/{clip.removesuffix('.mkv')}_{arm}_r2_{args.sink}.log"
        label = ("per-frame bytes payload (tobytes)" if arm == "ring0" else "pinned ring slots")
        st.emit(Record(value=round(statistics.median(src[arm]), 2), unit=unit,
                       population=Population(kind=pop.kind, items=pop.items,
                                             label=f"{pop.label}, arm {arm}: {label}"),
                       instrument=("the render loop line, ms/frame" if what == "loop"
                                   else "the profile's pack+queue line, ms/frame"),
                       config=cfg, session=session, statistic="median", tier="measured", id=rid(ridx),
                       run=f"rve-backend.py, logs {args.logs}/*_{arm}_r*_{args.sink}.log", artifact=art,
                       extra={"artifact_sha256": sha256(art), "of": what, "per_round": src[arm],
                              "sink": args.sink}))
        ids[(what, arm)] = rid(ridx)
    st.delta(ids[("loop", "ring1")], ids[("loop", "ring0")], note="the ring's effect on the frame")
    st.delta(ids[("pack", "ring1")], ids[("pack", "ring0")], note="the ring's effect on pack+queue")

    if args.pair and os.path.exists(args.pair):
        for rid, rec in Store.load(args.pair).records.items():
            st.emit(rec)

    st.dump(args.out)
    print(st.claims_text().strip())
    print(f"# -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
