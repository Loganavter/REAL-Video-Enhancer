"""Records for the Fast/Best question on a SECOND model family, and the decision they support.

The row asked whether the Fast preset's speed-up is a property of this card and one model, or of the path.
A second card is impossible here (one GPU in this machine); a second model family is not, and the app's own
body carries the answer: `realesr-animevideov3` is a 64-channel Compact with a 4x tail, which the int8 body
serves.

Usage: python harness/scripts/fast_preset_records.py [--logs docs/wsp/verified/logs/fast-preset/animevideov3]
                                             [--out docs/wsp/verified/records/fast-preset.jsonl]
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
MODEL = re.compile(r"^# model (\S+)$", re.M)
SANITY = re.compile(r"int8 body first-frame sanity: ([\d.]+)% of pixels >8 levels against the fp16 model")


def sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="docs/wsp/verified/logs/fast-preset/animevideov3")
    ap.add_argument("--out", default="docs/wsp/verified/records/fast-preset.jsonl")
    args = ap.parse_args(argv)

    loops: dict[str, list[float]] = {}
    frames, session, model = set(), "unknown", "unknown"
    sanity: list[float] = []
    for name in sorted(os.listdir(args.logs)):
        m = re.match(r"^(.+)_(fp16|int8)_r(\d+)_null\.log$", name)
        if not m:
            continue
        text = open(os.path.join(args.logs, name), errors="replace").read().replace("\r", "\n")
        if session == "unknown" and (s := STAMP.search(text)):
            session = f"s-{s.group(1)}"
        if model == "unknown" and (mm := MODEL.search(text)):
            model = os.path.basename(mm.group(1))
        lm = LOOP.search(text)
        if lm:
            frames.add(int(lm.group(1)))
            loops.setdefault(m.group(2), []).append(float(lm.group(2)))
        san = SANITY.search(text)
        if san:
            sanity.append(float(san.group(1)))

    n = max(frames) if frames else 0
    pop = Population.of("frames", (f"{model}#first{n}",),
                        f"BD Violet first {n} frames, model {model} (a SECOND model family: 4x, 48-channel "
                        f"tail), null sink, fp16 against int8")
    inst = "the render loop line, ms/frame"
    cfg = f"model {model}, 1080p -> 4x, null sink, reuse off"
    st = Store(session=session)
    ids = {}
    for rid, arm in (("F001", "fp16"), ("F002", "int8")):
        art = f"{args.logs}/bd_violet20s_{arm}_r1_null.log"
        st.emit(Record(value=round(statistics.median(loops[arm]), 2), unit="ms", population=pop,
                       instrument=inst, config=cfg, session=session, statistic="median",
                       tier="measured", id=rid,
                       run=f"rve-backend.py via scripts/wall_ab.sh MODEL=<second family>, logs {args.logs}",
                       artifact=art,
                       extra={"artifact_sha256": sha256(art), "of": "frame_time",
                              "per_round": loops[arm], "model": model}))
        ids[arm] = rid
    st.ratio(ids["fp16"], ids["int8"], note="Fast against Best on the second model family")

    if sanity:
        st.emit(Record(value=round(max(sanity), 5), unit="%", population=pop,
                       instrument="the body's own first-frame check: pixels >8 levels against the fp16 "
                                  "MODEL, one frame",
                       config=cfg, session=session, statistic="max", tier="measured", id="F003",
                       run=f"rve-backend.py via scripts/wall_ab.sh, logs {args.logs}",
                       artifact=f"{args.logs}/bd_violet20s_int8_r1_null.log",
                       extra={"artifact_sha256": sha256(f"{args.logs}/bd_violet20s_int8_r1_null.log"),
                              "of": "share", "note": "a gross-error gate on one frame, NOT an "
                                                     "acceptance number"}))

    st.dump(args.out)
    print(st.claims_text().strip())
    print(f"# -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
