"""Records for the epilogue-fold row: two isolated timings, two chain timings, the in-app pair, the codes.

The isolated arms are one hop and the sixteen-hop chain, both epilogues, same codes and weights. The in-app
arms are the same configuration with `RVE_INT8_EPILOGUE_FOLD` 0/1 alternating inside one session, read from
the body's own phase counter (`hops ...`) and the render loop line.

Usage: python harness/harness/scripts/epilogue_records.py [--probe docs/wsp/verified/logs/epilogue-fold/probe.json]
                                          [--logs docs/wsp/verified/logs/epilogue-fold/ab]
                                          [--out docs/wsp/verified/records/int8-epilogue-fold.jsonl]
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
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)

from harness.core import Population, Record, Store  # noqa: E402

PHASES = re.compile(r"int8 phases: head ([\d.]+)\s+quantise ([\d.]+)\s+hops ([\d.]+)\s+tail ([\d.]+)"
                    r"\s+SUM ([\d.]+)\s+BODY-TOTAL ([\d.]+)")
LOOP = re.compile(r"render loop: [\d.]+ s for (\d+) frames \(([\d.]+) ms/frame\)")
STAMP = re.compile(r"^# wall_ab (\S+)", re.M)


def sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def read(path: str) -> str:
    return open(path, errors="replace").read().replace("\r", "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="docs/wsp/verified/logs/epilogue-fold/probe-r8.json")
    ap.add_argument("--logs", default="docs/wsp/verified/logs/epilogue-fold/ab")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-epilogue-fold.jsonl")
    ap.add_argument("--pair", default="",
                    help="records emitted by gate_sweep for the render pair; appended, not re-derived")
    args = ap.parse_args(argv)

    probe = json.load(open(args.probe))
    pdig = sha256(args.probe)
    psess = f"s-{probe['take']}"
    clip = probe["clip"]
    pop = Population.of("frames", (f"{probe['width']}x{probe['height']}#{probe['frame']}",),
                        f"{clip} frame {probe['frame']} through the app's own yuv420p10_to_tensor, "
                        f"{probe['width']}x{probe['height']}")
    inst = (f"torch.cuda.Event around {probe['iters']} launches behind 3 warm-ups, "
            f"{probe['rounds']} interleaved rounds, medians of rounds 2+")
    cfg = "int8 body, block_m 128, 4 warps, the body's own weights and grids"

    st = Store(session=psess)
    ids = {}
    for rid, kind, fold in (("E001", "hop", "0"), ("E002", "hop", "1"),
                            ("E003", "chain", "0"), ("E004", "chain", "1")):
        st.emit(Record(value=round(probe["median"][kind][fold], 3), unit="ms",
                       population=Population(kind=pop.kind, items=pop.items,
                                             label=f"{pop.label}, arm {kind} fold={fold}"),
                       instrument=inst, config=cfg, session=psess,
                       statistic="median", tier="measured", id=rid, run="harness/scripts/hop_epilogue_probe.py",
                       artifact=args.probe,
                       extra={"artifact_sha256": pdig, "kind": kind, "fold": int(fold)}))
        ids[(kind, fold)] = rid
    st.delta(ids[("hop", "1")], ids[("hop", "0")], note="one hop, folded epilogue minus shipped")
    st.delta(ids[("chain", "1")], ids[("chain", "0")], note="sixteen hops, folded minus shipped")

    # the codes the two epilogues disagree on, on the same input
    c = probe["codes"]
    st.emit(Record(value=round(100.0 * c["chain_diff"] / c["chain_total"], 5), unit="%", population=pop,
                   instrument="int8 codes compared element-wise after sixteen hops on the same input",
                   config=cfg, session=psess, statistic="single",
                   tier="measured", id="E005", run="harness/scripts/hop_epilogue_probe.py", artifact=args.probe,
                   extra={"artifact_sha256": pdig, "max_abs_diff": c["chain_max"],
                          "hop1_diff": c["hop_diff"], "of": "codes"}))

    # in-app: the phase counter and the loop line, one session, arms alternating
    hops: dict[str, list[float]] = {}
    loops: dict[str, list[float]] = {}
    session = "unknown"
    for name in sorted(os.listdir(args.logs)):
        m = re.match(r"^(?P<clip>.+)_(?P<arm>[a-z]+)_r(?P<r>\d+)_null\.log$", name)
        if not m:
            continue
        text = read(os.path.join(args.logs, name))
        if session == "unknown" and (s := STAMP.search(text)):
            session = f"s-{s.group(1)}"
        vals = [float(x.group(3)) for x in PHASES.finditer(text)]
        if vals:
            hops.setdefault(m.group("arm"), []).extend(vals)
        lm = LOOP.search(text)
        if lm:
            loops.setdefault(m.group("arm"), []).append((int(lm.group(1)), float(lm.group(2))))

    if hops:
        frames = sorted({n for v in loops.values() for n, _ in v})
        pop_ab = Population.of("frames", (f"{clip}#first{max(frames) if frames else 0}",),
                               f"{clip}, 494-frame render, null sink, arms alternating in one session")

        def ab_pop(arm_value: str) -> Population:
            return Population(kind=pop_ab.kind, items=pop_ab.items,
                              label=f"{pop_ab.label}, arm {arm_value}")
        iid = {}
        for rid, arm in (("E006", "nofold"), ("E007", "fold")):
            st.emit(Record(value=round(statistics.median(hops[arm]), 2), unit="ms",
                           population=ab_pop(arm),
                           instrument="the body's own CUDA events, 'int8 phases: hops' line, last 8 frames",
                           config="int8 fit, null sink, arms alternating in one session",
                           session=session, statistic="median", tier="measured", id=rid,
                           run=f"rve-backend.py, logs {args.logs}/*_{arm}_r*_null.log",
                           artifact=f"{args.logs}/{clip.removesuffix('.mkv')}_{arm}_r1_null.log",
                           extra={"artifact_sha256": sha256(
                               f"{args.logs}/{clip.removesuffix('.mkv')}_{arm}_r1_null.log"),
                               "of": "hops_phase", "draws": len(hops[arm]),
                               "per_round": [round(v, 2) for v in
                                             (statistics.mean(hops[arm][i:i + 61])
                                              for i in range(0, len(hops[arm]), 61))]}))
            iid[arm] = rid
        st.delta(iid["fold"], iid["nofold"], note="in-app hops phase, folded minus shipped")
        if loops:
            lid = {}
            for rid, arm in (("E008", "nofold"), ("E009", "fold")):
                st.emit(Record(value=round(statistics.median([ms for _, ms in loops[arm]]), 2), unit="ms",
                               population=ab_pop(arm), instrument="the render loop line, ms/frame",
                               config="int8 fit, null sink, arms alternating in one session",
                               session=session, statistic="median", tier="measured", id=rid,
                               run=f"rve-backend.py, logs {args.logs}/*_{arm}_r*_null.log",
                               artifact=f"{args.logs}/{clip.removesuffix('.mkv')}_{arm}_r1_null.log",
                               extra={"artifact_sha256": sha256(
                                   f"{args.logs}/{clip.removesuffix('.mkv')}_{arm}_r1_null.log"),
                                   "of": "frame_time",
                                   "per_round": [ms for _, ms in loops[arm]]}))
                lid[arm] = rid
            st.delta(lid["fold"], lid["nofold"], note="in-app render loop, folded minus shipped")

    if args.pair:
        # the render pair's own records (EF001-EF004) come from the instrument that decoded the frames;
        # re-typing them here would be a second author for one number
        st2 = Store.load(args.pair)
        for rid, rec in st2.records.items():
            st.emit(rec)

    st.dump(args.out)
    print(st.claims_text().strip())
    print(f"# -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
