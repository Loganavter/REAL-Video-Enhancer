"""Records for the fused head: the isolated probe, the in-app A/B, and the render pair.

Usage: python scripts/head_fusion_records.py [--probe .../probe.json] [--logs .../ab]
                                             [--pair .../pair-records.jsonl]
                                             [--out docs/wsp/verified/records/int8-head-fusion.jsonl]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness.core import Population, Record, Store  # noqa: E402

LOOP = re.compile(r"render loop: [\d.]+ s for (\d+) frames \(([\d.]+) ms/frame\)")
PHASES = re.compile(r"int8 phases: head ([\d.]+)\s+quantise ([\d.]+)\s+hops ([\d.]+)\s+tail ([\d.]+)"
                    r"\s+SUM ([\d.]+)\s+BODY-TOTAL ([\d.]+)")
STAMP = re.compile(r"^# wall_ab (\S+)", re.M)


def sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="docs/wsp/verified/logs/head-fusion/probe.json")
    ap.add_argument("--logs", default="docs/wsp/verified/logs/head-fusion/ab")
    ap.add_argument("--pair", default="docs/wsp/verified/records/int8-head-fusion-pair.jsonl")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-head-fusion.jsonl")
    args = ap.parse_args(argv)

    pr = json.load(open(args.probe))
    pdig, psess = sha256(args.probe), f"s-{pr['take']}"
    clip = pr["clip"]
    w, hh = pr.get("width", 1920), pr.get("height", 1080)
    pop = Population.of("frames", (f"{w}x{hh}#frame{pr['frame']}",),
                        f"{clip} frame {pr['frame']}, 3 -> 64 head at {w}x{hh}")
    inst = (f"torch.cuda.Event around {pr['iters']} calls behind 3 warm-ups, {pr['rounds']} "
            f"interleaved rounds, medians of rounds 2+")
    cfg = f"int8 body head 3->64 3x3 fp16 at {w}x{hh}, BLOCK_M {pr['block_m']}"
    st = Store(session=psess)
    for rid, arm, label in (("H001", "today", "head + PReLU + quantise as shipped"),
                            ("H002", "head", "cuDNN head conv alone"),
                            ("H003", "quantise", "head conv + quantise pass (no separate PReLU)"),
                            ("H004", "fused", "one kernel: conv + bias + PReLU + requant, writes int8")):
        st.emit(Record(value=round(pr["median"][arm], 3), unit="ms",
                       population=Population(kind=pop.kind, items=pop.items,
                                             label=f"{pop.label}, arm {arm}: {label}"),
                       instrument=inst, config=cfg, session=psess, statistic="median", tier="measured",
                       id=rid, run="scripts/head_fusion_probe.py", artifact=args.probe,
                       extra={"artifact_sha256": pdig, "arm": arm, "of": "head_time"}))
    st.delta("H004", "H001", note="the fused head against the shipped three-step head")
    st.emit(Record(value=round(100.0 * pr["codes"]["n_diff"] / pr["codes"]["total"], 5), unit="%",
                   population=pop, instrument="int8 codes compared element-wise on the same frame",
                   config=cfg, session=psess, statistic="single", tier="measured", id="H005",
                   run="scripts/head_fusion_probe.py", artifact=args.probe,
                   extra={"artifact_sha256": pdig, "max_abs_diff": pr["codes"]["max"], "of": "codes"}))

    loops: dict[str, list[float]] = {}
    phases: dict[str, list[tuple]] = {}
    session = "unknown"
    for name in sorted(os.listdir(args.logs)):
        m = re.match(r"^(.+)_(plain|fused)_r(\d+)_null\.log$", name)
        if not m:
            continue
        text = open(os.path.join(args.logs, name), errors="replace").read().replace("\r", "\n")
        if session == "unknown" and (s0 := STAMP.search(text)):
            session = f"s-{s0.group(1)}"
        lm = LOOP.search(text)
        if lm:
            loops.setdefault(m.group(2), []).append(float(lm.group(2)))
        phases.setdefault(m.group(2), []).extend(
            (float(t.group(1)), float(t.group(2))) for t in PHASES.finditer(text))
    if loops:
        pop_ab = Population.of("frames", (f"{clip}#first494",),
                               f"{clip}, 494-frame render, null sink, head fused vs shipped, arms "
                               f"alternating in one session")
        lcfg = "int8 fit, null sink, arms alternating in one session"
        ids = {}
        for rid, arm in (("H006", "plain"), ("H007", "fused")):
            art = f"{args.logs}/{clip.removesuffix('.mkv')}_{arm}_r1_null.log"
            st.emit(Record(value=round(statistics.median(loops[arm]), 2), unit="ms",
                           population=Population(kind=pop_ab.kind, items=pop_ab.items,
                                                 label=f"{pop_ab.label}, arm {arm}"),
                           instrument="the render loop line, ms/frame", config=lcfg, session=session,
                           statistic="median", tier="measured", id=rid,
                           run=f"rve-backend.py, logs {args.logs}/*_{arm}_r*_null.log", artifact=art,
                           extra={"artifact_sha256": sha256(art), "of": "frame_time",
                                  "per_round": loops[arm]}))
            ids[arm] = rid
        st.delta(ids["fused"], ids["plain"], note="the fused head's effect on the frame")
        for rid, arm in (("H008", "plain"), ("H009", "fused")):
            art = f"{args.logs}/{clip.removesuffix('.mkv')}_{arm}_r1_null.log"
            st.emit(Record(value=round(statistics.median([h + q for h, q in phases[arm]]), 2), unit="ms",
                           population=Population(kind=pop_ab.kind, items=pop_ab.items,
                                                 label=f"{pop_ab.label}, arm {arm}"),
                           instrument="the body's own CUDA events, head + quantise phase lines",
                           config=lcfg, session=session, statistic="median", tier="measured", id=rid,
                           run=f"rve-backend.py, logs {args.logs}/*_{arm}_r*_null.log", artifact=art,
                           extra={"artifact_sha256": sha256(art), "of": "head_phase",
                                  "head": round(statistics.median([h for h, _ in phases[arm]]), 2),
                                  "quantise": round(statistics.median([q for _, q in phases[arm]]), 2),
                                  "draws": len(phases[arm])}))
        st.delta("H009", "H008", note="head + quantise on the phase counter, fused against shipped")

    if args.pair and os.path.exists(args.pair):
        for rid, rec in Store.load(args.pair).records.items():
            st.emit(rec)

    st.dump(args.out)
    print(st.claims_text().strip())
    print(f"# -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
