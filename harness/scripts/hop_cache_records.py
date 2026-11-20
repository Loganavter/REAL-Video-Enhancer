"""Records for the cache-policy sweep on the hop: five policies, one kernel."""
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
    ap.add_argument("--probe", default="docs/wsp/verified/logs/hop-cache/probe.json")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-hop-cache.jsonl")
    a = ap.parse_args(argv)
    pr = json.load(open(a.probe))
    dig = hashlib.sha256(open(a.probe, "rb").read()).hexdigest()
    pop = Population.of("frames", (f"{pr['clip']}#frame{pr['frame']}",),
                        f"{pr['clip']} frame {pr['frame']}, one hop, cache policy varied")
    st = Store(session=f"s-{pr['take']}")
    for i, (arm, ms) in enumerate(pr["median"].items(), start=1):
        st.emit(Record(value=round(ms, 3), unit="ms", population=pop,
                       instrument="torch.cuda.Event, 3 interleaved rounds, medians of rounds 2+",
                       config=f"int8 hop, BLOCK_M 128, 4 warps, cache policy = {arm}",
                       session=st.session, statistic="median", tier="measured", id=f"C{i:03d}",
                       run="harness/scripts/hop_cache_probe.py", artifact=a.probe,
                       extra={"artifact_sha256": dig, "arm": arm,
                              "gbps": round(250.6 / ms, 1)}))
    st.dump(a.out)
    print(st.claims_text().strip())
    print(f"# -> {a.out}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
