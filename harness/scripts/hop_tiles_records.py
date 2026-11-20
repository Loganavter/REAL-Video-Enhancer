"""Records for the two literature-driven hop experiments: the halo and the 2-D tile."""
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
    ap.add_argument("--halo", default="harness/docs/wsp/verified/logs/hop-halo/probe.json")
    ap.add_argument("--tile", default="harness/docs/wsp/verified/logs/hop-tile2d/probe.json")
    ap.add_argument("--out", default="harness/docs/wsp/verified/records/int8-hop-tiles.jsonl")
    a = ap.parse_args(argv)
    ha, ti = json.load(open(a.halo)), json.load(open(a.tile))
    d1 = hashlib.sha256(open(a.halo, "rb").read()).hexdigest()
    d2 = hashlib.sha256(open(a.tile, "rb").read()).hexdigest()
    pop = Population.of("frames", ("bd_violet20s.mkv#frame169",),
                        "one hop of the int8 body, buffer layout and CTA tile shape varied")
    cfg = "int8 hop 64->64 3x3, BLOCK_M 128, 4 warps; arms differ in layout or tile shape only"
    st = Store(session=f"s-{ti['take']}")
    i = 0
    for arm, ms in ha["median"].items():
        i += 1
        st.emit(Record(value=round(ms, 3), unit="ms", population=pop,
                       instrument="torch.cuda.Event, 3 interleaved rounds, medians of rounds 2+",
                       config=f"{cfg}, buffer = {arm}", session=st.session, statistic="median",
                       tier="measured", id=f"T{i:03d}", run="harness/scripts/hop_halo_probe.py", artifact=a.halo,
                       extra={"artifact_sha256": d1, "arm": arm, "of": "hop_time",
                              "halo_build_ms": round(ha["halo_copy_ms"], 3),
                              "codes_differ": ha["codes_differ"]}))
    for arm, ms in ti["median"].items():
        i += 1
        st.emit(Record(value=round(ms, 3), unit="ms", population=pop,
                       instrument="torch.cuda.Event, 3 interleaved rounds, medians of rounds 2+",
                       config=f"{cfg}, CTA tile = {arm}", session=st.session, statistic="median",
                       tier="measured", id=f"T{i:03d}", run="harness/scripts/hop_tile2d_probe.py", artifact=a.tile,
                       extra={"artifact_sha256": d2, "arm": arm, "of": "hop_time"}))
    st.dump(a.out)
    print(st.claims_text().strip())
    print(f"# -> {a.out}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
