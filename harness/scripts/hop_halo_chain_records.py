"""Records for the inline-halo chain: 16 hops, padded buffers, one border zeroing per frame."""
from __future__ import annotations
import argparse, hashlib, json, os, re, statistics, sys
import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)
from harness.core import Population, Record, Store  # noqa: E402

LOOP = re.compile(r"render loop: [\d.]+ s for (\d+) frames \(([\d.]+) ms/frame\)")
STAMP = re.compile(r"^# wall_ab (\S+)", re.M)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="docs/wsp/verified/logs/hop-halo-chain/probe.json")
    ap.add_argument("--logs", default="docs/wsp/verified/logs/hop-halo/ab")
    ap.add_argument("--pair", default="docs/wsp/verified/records/int8-hop-halo-pair.jsonl")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-hop-halo.jsonl")
    a = ap.parse_args(argv)
    pr = json.load(open(a.probe))
    dig = hashlib.sha256(open(a.probe, "rb").read()).hexdigest()
    pop = Population.of("frames", (f"{pr['clip']}#frame{pr['frame']}",),
                        f"{pr['clip']} frame {pr['frame']}, SIXTEEN hops chained, buffer layout varied")
    st = Store(session=f"s-{pr['take']}")
    i = 0
    for arm, ms in pr["median"].items():
        i += 1
        st.emit(Record(value=round(ms, 3), unit="ms",
                       population=Population(kind=pop.kind, items=pop.items,
                                             label=f"{pop.label}, arm {arm}"),
                       instrument="torch.cuda.Event around the whole 16-launch chain, 3 interleaved rounds",
                       config="int8 body, 16 hops, buffer layout is the arm", session=st.session,
                       statistic="median",
                       tier="measured", id=f"H{i:03d}", run="harness/scripts/hop_halo_chain_probe.py",
                       artifact=a.probe, extra={"artifact_sha256": dig, "arm": arm,
                                                "codes_differ": pr["codes_differ"]}))
    i += 1
    st.emit(Record(value=round(pr["zero_ms"], 3), unit="ms", population=pop,
                   instrument="torch.cuda.Event, 20 border zeroings of three padded buffers",
                   config="three (h+2, w+2, C) buffers, border zeroing", session=st.session,
                   statistic="median", tier="measured", id=f"H{i:03d}",
                   run="harness/scripts/hop_halo_chain_probe.py", artifact=a.probe,
                   extra={"artifact_sha256": dig, "of": "border_zero"}))
    # the chain gain as a verifiable delta (H001 - H002); the NET of it against the border zeroing (H003) is
    # stated in the doc, because the harness recomputes only two-input subtraction and inventing a third input
    # would be a derived number nothing can check
    st.delta("H001", "H002", note="the chain gain before the border zeroing")

    loops: dict[str, list[float]] = {}
    session = "unknown"
    if os.path.isdir(a.logs):
        for name in sorted(os.listdir(a.logs)):
            m = re.match(r"^(.+)_(halo0|halo1)_r(\d+)_null\.log$", name)
            if not m:
                continue
            text = open(os.path.join(a.logs, name), errors="replace").read().replace("\r", "\n")
            if session == "unknown" and (s := STAMP.search(text)):
                session = f"s-{s.group(1)}"
            lm = LOOP.search(text)
            if lm and int(m.group(3)) >= 2:
                loops.setdefault(m.group(2), []).append(float(lm.group(2)))
    if len(loops) == 2:
        pop_ab = Population.of("frames", (f"{pr['clip']}#first494",),
                               f"{pr['clip']}, 494-frame render, null sink, halo vs plain, arms alternating")
        ids = {}
        for k, arm in enumerate(("halo0", "halo1"), start=i + 1):
            art = f"{a.logs}/{pr['clip'].removesuffix('.mkv')}_{arm}_r1_null.log"
            st.emit(Record(value=round(statistics.median(loops[arm]), 2), unit="ms",
                           population=Population(kind=pop_ab.kind, items=pop_ab.items,
                                                 label=f"{pop_ab.label}, arm {arm}"),
                           instrument="the render loop line, ms/frame",
                           config="int8 fit, null sink, arms alternating in one session",
                           session=session, statistic="median", tier="measured", id=f"H{k:03d}",
                           run=f"rve-backend.py, logs {a.logs}/*_{arm}_r*_null.log", artifact=art,
                           extra={"artifact_sha256": hashlib.sha256(open(art, "rb").read()).hexdigest(),
                                  "per_round": loops[arm], "of": "frame_time"}))
            ids[arm] = f"H{k:03d}"
        st.delta(ids["halo1"], ids["halo0"], note="the inline halo's effect on the frame")
    if a.pair and os.path.exists(a.pair):
        for rid, rec in Store.load(a.pair).records.items():
            st.emit(rec)
    st.dump(a.out)
    print(st.claims_text().strip())
    print(f"# -> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
