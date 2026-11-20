"""Read the wall A/B logs into records: per-round draws, per-arm medians of rounds 2+, and the ratio.

Why a second reader next to ``emit_render_records.py``: that one turns ONE log into records and calls the
population the clip. The A/B needs the complement - the same arm ACROSS rounds, statistic "median", and
the population is the ROUNDS, because run-to-run drift here is 10-15% and a 3-run median of an arm is a
different claim from "this clip took 57.7 ms/frame once".

The number the row decides on is the ratio, so every ratio here is a quotient of two records
(``Store.ratio``) rather than a number typed into prose. The arm is the variable that differs: it lives in
the population label and in ``run``, while ``config``/``instrument``/``session``/``statistic`` are held
equal, which is what ``Store._relation`` demands of a ratio.

Usage:
  python harness/scripts/wall_ab_records.py --logs docs/wsp/verified/logs/wall-ab \
      --out docs/wsp/verified/records/int8-wall-ab.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

import os as _bos, sys as _bsys
_HARNESS_ROOT = _bos.path.dirname(_bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
if _HARNESS_ROOT not in _bsys.path:
    _bsys.path.insert(0, _HARNESS_ROOT)

from emit_render_records import facts, read  # noqa: E402  one parser for the loop line
from harness.core import Population, Record, Store  # noqa: E402

NAME = re.compile(r"^(?P<clip>.+)_(?P<arm>[a-z0-9]+)_r(?P<round>\d+)_(?P<sink>null|nvenc)\.log$")
PHASES = re.compile(r"int8 phases: head ([\d.]+)\s+quantise ([\d.]+)\s+hops ([\d.]+)\s+tail ([\d.]+)"
                    r"\s+SUM ([\d.]+)\s+BODY-TOTAL ([\d.]+)")
STAMP = re.compile(r"^# wall_ab (\S+)", re.M)
COMPONENTS = ("head", "quantise", "hops", "tail", "body_total")


def read_logs(logdir: str) -> tuple[dict, str]:
    """{(clip, sink): {arm: {round: (ms_per_frame, phases)}}}, first run's timestamp"""
    out: dict = {}
    session = "unknown"
    for path in sorted(os.listdir(logdir)):
        m = NAME.match(path)
        if not m:
            continue
        text = read(os.path.join(logdir, path))
        if session == "unknown" and (s := STAMP.search(text)):
            session = f"s-{s.group(1)}"
        f = facts(text)
        if "ms_per_frame" not in f:
            continue
        ph = [tuple(float(g) for g in mm.groups()) for mm in PHASES.finditer(text)]
        phases = dict(zip(("head", "quantise", "hops", "tail", "sum", "body_total"), ph[-1])) if ph else {}
        out.setdefault((m["clip"], m["sink"]), {}).setdefault(m["arm"], {})[int(m["round"])] = (
            f["ms_per_frame"], phases)
    return out, session


def median(vals: list[float]) -> float:
    vals = sorted(vals)
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="docs/wsp/verified/logs/wall-ab")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-wall-ab.jsonl")
    ap.add_argument("--first-round", type=int, default=2,
                    help="first round kept in the median (2 drops the draw taken right after engine load)")
    args = ap.parse_args(argv)

    data, session = read_logs(args.logs)
    store = Store(session=session)
    out: list[Record] = []
    n = 0

    def add(rec: Record) -> Record:
        nonlocal n
        n += 1
        rec.id = f"W{n:03d}"
        store.emit(rec)
        out.append(rec)
        return rec

    digests: dict[str, str] = {}

    def digest(path: str) -> str:
        if path not in digests:
            digests[path] = hashlib.sha256(open(path, "rb").read()).hexdigest()
        return digests[path]

    for (clip, sink), arms in sorted(data.items()):
        if set(arms) != {"fp16", "int8"}:
            print(f"# skip {clip} {sink}: only arms {sorted(arms)}", file=sys.stderr)
            continue
        config = f"{clip}, {sink} sink, reuse off (--no_temporal_reuse), one process per arm"
        instrument = "the app's own render loop: wall clock / frames (the sink decides what it contains)"
        per_round: dict[tuple[str, int], Record] = {}
        for arm in ("fp16", "int8"):
            for r in sorted(arms[arm]):
                ms, _ = arms[arm][r]
                log = os.path.join(args.logs, f"{clip}_{arm}_r{r}_{sink}.log")
                per_round[(arm, r)] = add(Record(
                    value=round(ms, 2), unit="ms",
                    population=Population.of("rounds", [f"{clip}#{sink}#round-{r}"],
                                             f"{clip}, {arm}, {sink} sink, round {r}: one draw, n=1"),
                    instrument=instrument, config=config, session=session, statistic="single",
                    tier="measured", run=f"rve-backend.py, log {os.path.basename(log)}",
                    artifact=log, extra={"of": "frame_time", "arm": arm,
                                         "artifact_sha256": digest(log)}))
        kept = [r for r in sorted(arms["fp16"]) if r >= args.first_round]
        if len(kept) < 2:
            print(f"# skip {clip} {sink}: {len(kept)} rounds past r{args.first_round - 1}",
                  file=sys.stderr)
            continue
        # the CLIP and the SINK are in the item set, not only in the label: otherwise the sha is the
        # same for every clip and a cross-clip ratio would pass the gate (the label-vs-population trap)
        pop = Population.of("rounds", [f"{clip}#{sink}#round-{r}" for r in kept],
                            f"{clip}, {sink} sink, rounds {kept[0]}-{kept[-1]} of one session "
                            f"(round 1 dropped: the first draw after engine load is warmup)")
        med: dict[str, Record] = {}
        for arm in ("fp16", "int8"):
            log = os.path.join(args.logs, f"{clip}_{arm}_r{kept[0]}_{sink}.log")
            med[arm] = add(Record(
                value=round(median([arms[arm][r][0] for r in kept]), 2), unit="ms", population=pop,
                instrument=instrument, config=config, session=session, statistic="median",
                tier="measured", run=f"rve-backend.py, logs {clip}_{arm}_r*_{sink}.log",
                artifact=log, extra={"of": "frame_time", "arm": arm,
                                     "artifact_sha256": digest(log)}))
        ratio = store.ratio(med["fp16"].id, med["int8"].id, instrument=instrument,
                            note=f"{clip}, {sink}: median of rounds {kept[0]}-{kept[-1]}")
        n += 1
        ratio.id = f"W{n:03d}"
        out.append(ratio)
        for r in kept:  # a pairwise ratio per round, so the reader sees the round-to-round spread
            pair = store.ratio(per_round[("fp16", r)].id, per_round[("int8", r)].id,
                               instrument=instrument, note=f"{clip}, {sink}, round {r}")
            n += 1
            pair.id = f"W{n:03d}"
            out.append(pair)
        for comp in COMPONENTS:
            vals = [arms["int8"][r][1].get("sum" if comp == "body_total" else comp) for r in kept]
            if any(v is None for v in vals):
                continue
            log = os.path.join(args.logs, f"{clip}_int8_r{kept[0]}_{sink}.log")
            add(Record(
                value=round(median(vals), 2), unit="ms", population=pop,
                instrument="the body's own phase counter, last 8 frames of each run",
                config=f"{clip}, int8 body, {sink} sink, reuse off", session=session,
                statistic="median", tier="measured",
                run=f"rve-backend.py, logs {clip}_int8_r*_{sink}.log", artifact=log,
                extra={"of": "frame_time", "phase": comp, "artifact_sha256": digest(log)}))

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
