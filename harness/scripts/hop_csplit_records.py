"""Records for the output-channel split: the last occupancy lever Triton can express."""
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
    ap.add_argument("--probe", default="docs/wsp/verified/logs/hop-csplit/probe.json")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-hop-csplit.jsonl")
    a = ap.parse_args(argv)
    pr = json.load(open(a.probe))
    dig = hashlib.sha256(open(a.probe, "rb").read()).hexdigest()
    pop = Population.of("frames", (f"{pr['clip']}#frame{pr['frame']}",),
                        f"{pr['clip']} frame {pr['frame']}, one hop, output channels split across CTAs")
    st = Store(session=f"s-{pr['take']}")
    for i, (arm, ms) in enumerate(pr["median"].items(), start=1):
        st.emit(Record(value=round(ms, 3), unit="ms", population=pop,
                       instrument="torch.cuda.Event, 3 interleaved rounds, medians of rounds 2+",
                       config=f"int8 hop 64->64 3x3, BLOCK_M 128, output channels per CTA = {arm}",
                       session=st.session, statistic="median", tier="measured", id=f"X{i:03d}",
                       run="harness/scripts/hop_csplit_probe.py", artifact=a.probe,
                       extra={"artifact_sha256": dig, "arm": arm, "tops": round(152.9 / ms, 1)}))
    st.dump(a.out)
    print(st.claims_text().strip())
    print(f"# -> {a.out}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
