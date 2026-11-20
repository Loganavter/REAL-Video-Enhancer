"""Read the int8 tail's four steps out of the app's log: dequant / conv / shuffle / residual.

Why: `int8-window.txt` splits the tail as 2.1 / 3.6 / 2.4 ms, measured on a different model at a different
output size, and the row that proposed fusing the tail's passes rested on that split. The split is now
printed per run by the body itself (`int8 tail: ...`), so the row can be judged on the shipped path.

Usage: python harness/scripts/tail_split.py --logs /tmp/opencode/tail --out harness/docs/wsp/verified/records/int8-tail-split.jsonl
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

from emit_render_records import read  # noqa: E402
from harness.core import Population, Record, Store  # noqa: E402

TAIL = re.compile(r"int8 tail: dequant ([\d.]+)\s+conv ([\d.]+)\s+shuffle ([\d.]+)\s+residual ([\d.]+)"
                  r"\s+SUM ([\d.]+) ms")
PHASE = re.compile(r"int8 phases: .*?tail ([\d.]+)")
NAME = re.compile(r"^(?P<clip>.+)_(?P<arm>[a-z0-9]+)_r(?P<round>\d+)_(?P<sink>\w+)\.log$")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="/tmp/opencode/tail")
    ap.add_argument("--out", default="harness/docs/wsp/verified/records/int8-tail-split.jsonl")
    ap.add_argument("--session", default="s-2026-11-18T19:05:00Z")
    args = ap.parse_args(argv)

    store = Store(session=args.session)
    out: list[Record] = []
    n = 0
    for name in sorted(os.listdir(args.logs)):
        m = NAME.match(name)
        if not m:
            continue
        path = os.path.join(args.logs, name)
        text = read(path)
        tails = TAIL.findall(text)
        if not tails:
            print(f"# skip {name}: no tail line", file=sys.stderr)
            continue
        dequant, conv, shuffle, residual, total = (float(v) for v in tails[-1])
        phase = PHASE.findall(text)
        pop = Population.of("clips", [f"{m.group('clip')}.mkv"],
                            f"{m.group('clip')}.mkv, int8 body, reuse off, last 8 frames of one run")
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
        for label, value, stat, unit in (("total", total, "single", "ms"), ("dequant", dequant, "single", "ms"),
                                         ("conv", conv, "single", "ms"), ("shuffle", shuffle, "single", "ms"),
                                         ("residual", residual, "single", "ms"),
                                         ("phase tail", float(phase[-1]) if phase else float("nan"),
                                          "single", "ms")):
            if value != value:  # nan
                continue
            n += 1
            rec = Record(value=round(value, 2), unit=unit, population=pop,
                         instrument="the body's own CUDA events around each tail step (last 8 frames)",
                         config=f"int8 body, arm {m.group('arm')}, {m.group('sink')} sink, step {label}",
                         session=args.session, statistic=stat, tier="measured",
                         run=f"rve-backend.py, log {name}", artifact=path,
                         extra={"artifact_sha256": digest, "of": "frame_time", "step": label})
            rec.id = f"T{n:03d}"
            store.emit(rec)
            out.append(rec)

    outpath = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), args.out)
    with open(outpath, "w") as fh:
        for rec in out:
            fh.write(json.dumps(rec.to_dict()) + "\n")
    for rec in out:
        print(rec.claims_line())
    print(f"# {len(out)} records -> {outpath}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
