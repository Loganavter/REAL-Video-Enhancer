"""Per-arm medians and the ratio from any `wall_ab.sh` log directory - one reader, any arm names.

`wall_ab_records.py` is deliberately narrow: it wants the `fp16` / `int8` pair and reads the phase lines of the
production model. A pair of VARIANTS (a grid, a width, a hop count) has different arm names and often a different
model, and the numbers still have to be records rather than prose. This is that reader.

The arm is the variable under test, so it lives in `run` and `extra`, never in `config` - the harness refuses a
ratio whose configurations differ, and putting the arm there would make the ratio unrepresentable.

Usage:
  python harness/scripts/ab_records.py --clip bd_violet20s --logdir docs/wsp/verified/logs/ultra-null2 \
      --arms fp16,k2 --sink null --model "2x-sudo-UltraCompact" --out records/int8-ultra.jsonl
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


def read(logdir: str, clip: str, arm: str, sink: str, rounds: list[int]) -> tuple[dict, str, str]:
    ms, session, path = {}, None, None
    for r in rounds:
        p = os.path.join(logdir, f"{clip}_{arm}_r{r}_{sink}.log")
        if not os.path.exists(p):
            continue
        text = open(p, errors="replace").read()
        if session is None and (s := STAMP.search(text)):
            session = f"s-{s.group(1)}"
        m = LOOP.search(text)
        if m:
            ms[r] = float(m.group(2))
            path = path or p
    return ms, session or "unknown", path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--arms", required=True, help="comma-separated arm names as wall_ab.sh wrote them")
    ap.add_argument("--sink", default="null")
    ap.add_argument("--model", default="")
    ap.add_argument("--rounds", default="1,2,3")
    ap.add_argument("--first-round", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    rounds = [int(r) for r in args.rounds.split(",") if r.strip()]
    kept = [r for r in rounds if r >= args.first_round]
    data = {a: read(args.logdir, args.clip, a, args.sink, rounds) for a in arms}
    session = next((d[1] for d in data.values() if d[1] != "unknown"), "unknown")
    store = Store(session=session)
    out: list[Record] = []

    pop = Population.of("rounds", [f"{args.clip}#{args.sink}#round-{r}" for r in kept],
                        f"{args.clip}, {args.sink} sink, rounds {kept[0]}-{kept[-1]} of one session, arms "
                        f"interleaved")
    instr = "the app's own render loop: wall clock / frames"
    cfg = (f"{args.clip}, {args.sink} sink, reuse off, one process per arm"
           + (f", model {args.model}" if args.model else ""))
    ids = {}
    for arm in arms:
        ms, _s, path = data[arm]
        if not all(r in ms for r in kept):
            print(f"# skip {arm}: rounds {kept} missing", file=sys.stderr)
            continue
        rec = Record(value=round(statistics.median(ms[r] for r in kept), 2), unit="ms", population=pop,
                     instrument=instr, config=cfg, session=session, statistic="median", tier="measured",
                     run=f"rve-backend.py, arm {arm}, logs from scripts/wall_ab.sh", artifact=path,
                     extra={"of": "frame_time", "arm": arm,
                            "artifact_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()})
        rec.id = f"AB{len(out) + 1:03d}"
        store.emit(rec)
        out.append(rec)
        ids[arm] = rec.id
    if len(ids) == 2:
        a, b = arms
        rec = store.ratio(ids[a], ids[b], instrument=instr, note=f"{args.clip}, {args.sink}: {a} / {b}")
        store.records.pop(rec.id, None)
        rec.id = f"AB{len(out) + 1:03d}"
        store.records[rec.id] = rec
        out.append(rec)

    store.dump(args.out)
    for rec in out:
        print(rec.claims_line())
    print(f"# {len(out)} records -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
