"""Read the app's own per-frame profile lines into records, and check them against the render loop.

The question this answers: on 2026-11-18 the profile printed "frame total 50.40 ms" against named parts
summing to 44.17, and the 6.2 ms gap was quoted as if it were an unknown cost. It was not unknown - it was
three stages the line did not print (pack+queue, the wrapper around the body, read wait). This script emits
what the instrument actually says now, plus the one number that shows whether the instrument can be summed
at all: the profile's frame total divided against the render loop's wall clock.

Usage: python scripts/frame_accounting.py --logs /tmp/opencode/acct2 --out docs/wsp/verified/records/int8-frame-accounting.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from emit_render_records import read  # noqa: E402
from harness.core import Population, Record, Store  # noqa: E402

LOOP = re.compile(r"render loop: [\d.]+ s for (\d+) frames \(([\d.]+) ms/frame\)")
PROFILE = re.compile(r"profile over (\d+) frames: gate ([\d.]+) ms .*?model wall ([\d.]+) ms "
                     r"\(self-reported ([\d.]+), engine ([\d.]+), frame handover ([\d.]+)\).*?"
                     r"input convert ([\d.]+) ms, preview ([\d.]+) ms, pack\+queue ([\d.]+) ms, "
                     r"read wait ([\d.]+) ms, named ([\d.]+) of frame total ([\d.]+) ms, "
                     r"unaccounted (-?[\d.]+) ms")
BODY = re.compile(r"int8 phases: .*?BODY-TOTAL ([\d.]+) ms")
NAME = re.compile(r"^bd_(?P<clip>\w+?)_(?P<arm>fp16|int8)_r\d+_null\.log$")


def parse(text: str) -> dict:
    out: dict = {}
    if m := LOOP.search(text):
        out.update(frames=int(m.group(1)), loop=float(m.group(2)))
    if m := PROFILE.search(text):
        # groups: 11 = the NAMED sum, 12 = the frame total, 13 = the leftover - do not swap them, the
        # first version of this script did and emitted the named sum as if it were the frame
        out.update(named=float(m.group(11)), profile_total=float(m.group(12)),
                   unaccounted=float(m.group(13)),
                   model_wall=float(m.group(3)), self_reported=float(m.group(4)),
                   engine=float(m.group(5)), pack=float(m.group(9)), preview=float(m.group(8)),
                   read=float(m.group(10)), gate=float(m.group(2)), convert=float(m.group(7)))
    # the LAST phase line, not the first: the first is the body's cold frame, and `wall_ab_records.py`
    # already takes the last for the same reason (steady state)
    if found := BODY.findall(text):
        out.update(body=float(found[-1]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="/tmp/opencode/acct2")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-frame-accounting.jsonl")
    ap.add_argument("--session", default="")
    ap.add_argument("--first-clip-only", action="store_true",
                    help="keep one clip (the repeats of the same clip are a level check, not new data)")
    args = ap.parse_args(argv)

    session = args.session or "s-2026-11-18T18:10:00Z"
    store = Store(session=session)
    rows = []
    for name in sorted(os.listdir(args.logs)):
        m = NAME.match(name)
        if not m:
            continue
        path = os.path.join(args.logs, name)
        d = parse(read(path))
        if "loop" not in d or "profile_total" not in d:
            print(f"# skip {name}: no loop or profile line", file=sys.stderr)
            continue
        rows.append((m.group("clip"), m.group("arm"), path, d))

    digest = {p: hashlib.sha256(open(p, "rb").read()).hexdigest() for _, _, p, _ in rows}
    out: list[Record] = []
    n = 0

    def add(rec: Record) -> Record:
        nonlocal n
        n += 1
        rec.id = f"A{n:03d}"
        store.emit(rec)
        out.append(rec)
        return rec

    for clip, arm, path, d in rows:
        pop = Population.of("clips", [f"bd_{clip}20s.mkv"], f"bd_{clip}20s.mkv, {arm}, reuse off, "
                                                          f"{d['frames']} frames of one run")
        common = dict(unit="ms", population=pop, config=f"{clip}, {arm}, reuse off, one run",
                      session=session, tier="measured", run=f"rve-backend.py, log {os.path.basename(path)}",
                      artifact=path, extra={"artifact_sha256": digest[path]})
        loop = add(Record(value=d["loop"], statistic="mean", instrument="the app's render loop: wall clock / "
                                                          "frames", **common))
        add(Record(value=d["profile_total"], statistic="mean",
                   instrument="the app's profile: the frame window, per-iteration spans", **common))
        add(Record(value=d["named"], statistic="mean",
                   instrument="the app's profile: gate + model wall + preview + pack + read, summed",
                   **common))
        if "body" in d:
            add(Record(value=d["body"], statistic="mean",
                       instrument="the body's own phase counter (last 8 frames of the run)", **common))
        add(Record(value=d["pack"], statistic="mean",
                   instrument="perf_counter around the pinned-slot fill + writeQueue.put", **common))
        # No delta record for "profile total - loop": the harness refuses it, and it is right - the two are
        # different measurement points (per-iteration spans vs wall clock), and their difference is an
        # OBSERVATION about the instrument, not a derived quantity. The doc cites both ids and says so.

    with open(args.out, "w") as fh:
        for rec in out:
            rec.validate()
            fh.write(json.dumps(rec.to_dict()) + "\n")
    for rec in out:
        print(rec.claims_line())
    print(f"# {len(out)} records -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
