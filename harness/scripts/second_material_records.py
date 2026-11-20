"""Records for the second-material replay: each adopted change reverted, per clip, never pooled.

Usage: python harness/scripts/second_material_records.py [--root docs/wsp/verified/logs/second-material]
                                                 [--out docs/wsp/verified/records/int8-defaults-material.jsonl]
"""
from __future__ import annotations

import argparse
import hashlib
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
NAME = re.compile(r"^(?P<clip>.+)_(?P<arm>head[01]|tail[01])_r(?P<r>\d+)_null\.log$")
CHANGE = {"head": "head as one fused kernel (h1) against cuDNN head + quantise (h0)",
          "tail": "int8 tail kernel + shuffle fold (t1) against dequant + cuDNN tail + shuffle (t0)"}


def sha256(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="docs/wsp/verified/logs/second-material")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-defaults-material.jsonl")
    args = ap.parse_args(argv)

    st = Store()
    n = 0
    for d in sorted(os.listdir(args.root)):
        path = os.path.join(args.root, d)
        if not os.path.isdir(path) or "-" not in d:
            continue
        change = d.rsplit("-", 1)[1]          # head | tail
        if change not in CHANGE:
            continue
        arms: dict[str, list[float]] = {}
        frames: set[int] = set()
        session = "unknown"
        for name in sorted(os.listdir(path)):
            m = NAME.match(name)
            if not m:
                continue
            text = open(os.path.join(path, name), errors="replace").read().replace("\r", "\n")
            if session == "unknown" and (s := STAMP.search(text)):
                session = f"s-{s.group(1)}"
            lm = LOOP.search(text)
            if not lm:
                continue
            frames.add(int(lm.group(1)))
            if int(m.group("r")) < 2:          # round 1 is dropped: it carries the load ramp
                continue
            arms.setdefault(m.group("arm"), []).append(float(lm.group(2)))
        if len(arms) != 2:
            print(f"# skip {d}: arms {sorted(arms)}", file=sys.stderr)
            continue
        clip = d.rsplit("-", 1)[0] + ".mkv"
        nfr = max(frames) if frames else 0
        pop = Population.of("rounds", [f"{clip}#null#round-{r}" for r in (2, 3)],
                            f"{clip}, null sink, rounds 2-3 of one session, {nfr} frames")
        cfg = "int8 fit, null sink, reuse off, arms alternating in one session"
        lo, hi = sorted(arms)                  # head0/head1, tail0/tail1
        ids = {}
        for arm in (lo, hi):
            n += 1
            rid = f"M{n:03d}"
            rid_out = rid
            art = os.path.join(path, [f for f in os.listdir(path) if f"_{arm}_r2_" in f][0])
            st.emit(Record(value=round(statistics.median(arms[arm]), 2), unit="ms",
                           population=Population(kind=pop.kind, items=pop.items,
                                                 label=f"{pop.label}, arm {arm}"),
                           instrument="the render loop line, ms/frame", config=cfg, session=session,
                           statistic="median", tier="measured", id=rid,
                           run=f"rve-backend.py via scripts/wall_ab.sh, logs {path}", artifact=art,
                           extra={"artifact_sha256": sha256(art), "of": "frame_time",
                                  "change": change, "per_round": arms[arm], "clip": clip}))
            ids[arm] = rid_out
        n += 1
        st.delta(ids[hi], ids[lo], note=f"{CHANGE[change]} on {clip}")

    st.dump(args.out)
    print(st.claims_text().strip())
    print(f"# -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
