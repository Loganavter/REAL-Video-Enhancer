"""Records for the occupancy sweep and the epilogue decomposition - what the hop's 1.55 ms is made of."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)
from harness.core import Population, Record, Store  # noqa: E402

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="docs/wsp/verified/logs/hop-occupancy/probe.json")
    ap.add_argument("--epi", default="docs/wsp/verified/logs/hop-epilogue/probe.json")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-hop-occupancy.jsonl")
    a = ap.parse_args(argv)
    sw, ep = json.load(open(a.sweep)), json.load(open(a.epi))
    d1, d2 = hashlib.sha256(open(a.sweep, "rb").read()).hexdigest(), \
             hashlib.sha256(open(a.epi, "rb").read()).hexdigest()
    pop = Population.of("frames", ("bd_violet20s.mkv#frame169",),
                        "one hop of the int8 body on its own codes, launch parameters varied")
    cfg = "int8 hop 64->64 3x3, 1920x1080, FOLD=0"
    st = Store(session=f"s-{sw['take']}")
    i = 0
    for arm, ms in sw["median"].items():
        i += 1
        bm, wp = arm.replace("BM", "").split("/w")
        st.emit(Record(value=round(ms, 3), unit="ms", population=pop,
                       instrument="torch.cuda.Event, 3 interleaved rounds, medians of rounds 2+",
                       config=f"{cfg}, BLOCK_M {bm}, {wp} warps", session=st.session, statistic="median",
                       tier="measured", id=f"O{i:03d}", run="harness/scripts/hop_occupancy_probe.py",
                       artifact=a.sweep, extra={"artifact_sha256": d1, "arm": arm,
                                                "block_m": int(bm), "warps": int(wp)}))
    st.delta("O001", "O001", note="placeholder") if False else None
    j = i
    for arm, ms in ep["median"].items():
        j += 1
        st.emit(Record(value=round(ms, 3), unit="ms", population=pop,
                       instrument="torch.cuda.Event, 3 interleaved rounds, medians of rounds 2+",
                       config=f"{cfg}, BLOCK_M 128, 4 warps, epilogue = {arm}", session=st.session,
                       statistic="median", tier="measured", id=f"O{j:03d}",
                       run="harness/scripts/hop_epilogue_probe.py", artifact=a.epi,
                       extra={"artifact_sha256": d2, "arm": arm, "of": "epilogue"}))
    st.dump(a.out)
    print(st.claims_text().strip())
    print(f"# -> {a.out}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
