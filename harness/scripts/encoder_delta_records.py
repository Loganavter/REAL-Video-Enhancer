"""Records for the encoder question: is the ~10 ms encoder penalty a constant or a property of content?

The artifact's numbers are taken with a `null` sink; the product runs with an encoder. Two clips, both sinks,
so the answer is not a one-clip claim:

  clip, null        the artifact's condition
  clip, hevc_nvenc  the product's condition
  delta             nvenc - null, same arm: what the encoder actually costs on THAT clip
  ratio             fp16 / int8-yuv under each sink

Usage: python harness/scripts/encoder_delta_records.py [--out ...]
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
STAMP = re.compile(r"^# wall_ab (\S+)", re.M)
CLIPS = {
    "bd_violet20s": {"null": "harness/docs/wsp/verified/logs/tail-csc-ratio",
                     "nvenc": "harness/docs/wsp/verified/logs/tail-csc-nvenc"},
    "bd_csm_same": {"null": "harness/docs/wsp/verified/logs/tail-csc-action-null",
                    "nvenc": "harness/docs/wsp/verified/logs/tail-csc-nvenc-action"},
}
ARMS = ("fp16", "yuv")
ROUNDS = (1, 2, 3)
KEPT = (2, 3)


def read(logdir: str, clip: str, arm: str, sink: str) -> tuple[dict, str, str]:
    ms, session, path = {}, None, None
    for r in ROUNDS:
        p = os.path.join(logdir, f"{clip}_{arm}_r{r}_{sink}.log")
        if not os.path.exists(p):
            raise SystemExit(f"missing log: {p}")
        text = open(p, errors="replace").read()
        if session is None and (s := STAMP.search(text)):
            session = f"s-{s.group(1)}"
        m = LOOP.search(text)
        if not m:
            raise SystemExit(f"no loop line in {p}")
        ms[r] = float(m.group(2))
        path = path or p
    return ms, session or "unknown", path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="harness/docs/wsp/verified/records/int8-encoder-delta.jsonl")
    args = ap.parse_args(argv)

    store = Store(session="s-encoder-delta-2026-11-20")
    out: list[Record] = []

    def add(rec: Record) -> Record:
        rec.id = f"ED{len(out) + 1:03d}"
        store.emit(rec)
        out.append(rec)
        return rec

    for clip, dirs in CLIPS.items():
        by: dict[tuple[str, str], Record] = {}
        for sink in ("null", "nvenc"):
            # ONE session per sink: the arms are interleaved inside one run set, so the A/B's session is the
            # run set's stamp (its first log), not each process's own - the per-process stamp would turn an
            # interleaved A/B into a cross-session comparison and the harness would (rightly) refuse it
            ms_fp16, session, _p = read(dirs[sink], clip, "fp16", sink)
            for arm in ARMS:
                ms, _s, path = read(dirs[sink], clip, arm, sink)
                pop = Population.of("rounds", [f"{clip}#{sink}#round-{r}" for r in KEPT],
                                    f"{clip}, {sink} sink, rounds 2-3 of one session")
                by[(arm, sink)] = add(Record(
                    value=round(statistics.median(ms[r] for r in KEPT), 2), unit="ms",
                    population=pop,
                    instrument="the app's own render loop: wall clock / frames",
                    config=f"{clip}, {sink} sink, reuse off, one process per arm"
                           + ("" if sink == "null" else " (hevc_nvenc encoding alongside)"),
                    session=session, statistic="median", tier="measured",
                    run=f"rve-backend.py --int8_fit {arm}, logs from harness/scripts/wall_ab.sh",
                    artifact=path,
                    extra={"of": "frame_time", "arm": arm, "sink": sink,
                           "artifact_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}))

        # what the encoder itself costs on this clip, per arm. A DERIVED record, not a relation: the two
        # arms differ in config (the sink), which `_relation` refuses by design - the harness recomputes
        # "<nvenc> - <null>" from the inputs instead.
        for arm in ARMS:
            null, nv = by[(arm, "null")], by[(arm, "nvenc")]
            # emitted by hand, not through Store.derived: that one is tier "derived" and gets its arithmetic
            # recomputed, which is wrong for a delta across two run sets. This one says so in its tier.
            add(Record(
                value=round(nv.value - null.value, 6), unit="ms", population=null.population,
                instrument="derived", config=null.config, session=null.session,
                statistic="median", tier="assumed", formula=f"{nv.id}-{null.id}",
                inputs=(nv.id, null.id),
                note=f"{clip}, arm {arm}: hevc_nvenc minus null - the two sinks were separate run sets, so "
                     f"this is a cross-session delta"))

        # and the product's ratio under each sink
        for sink in ("null", "nvenc"):
            rec = store.ratio(by[("fp16", sink)].id, by[("yuv", sink)].id,
                              instrument=by[("fp16", sink)].instrument,
                              note=f"{clip}, {sink} sink: fp16 / int8-yuv")
            store.records.pop(rec.id, None)
            rec.id = f"ED{len(out) + 1:03d}"
            store.records[rec.id] = rec
            out.append(rec)

    store.dump(args.out)
    for rec in out:
        print(rec.claims_line())
    print(f"# {len(out)} records -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
