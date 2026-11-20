"""Records for the M x 2/3 batching falsifier."""
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
    ap.add_argument("--probe", default="harness/docs/wsp/verified/logs/hop-batch/probe.json")
    ap.add_argument("--out", default="harness/docs/wsp/verified/records/int8-hop-batch.jsonl")
    a = ap.parse_args(argv)
    pr = json.load(open(a.probe))
    dig = hashlib.sha256(open(a.probe, "rb").read()).hexdigest()
    pop = Population.of("frames", ("bd_violet20s.mkv#frames169-172",),
                        "four consecutive frames' codes, one launch carrying 1, 2 or 3 of them")
    st = Store(session=f"s-{pr['take']}")
    for i, (arm, ms) in enumerate(pr["median_per_frame"].items(), start=1):
        st.emit(Record(value=round(ms, 3), unit="ms", population=pop,
                       instrument="torch.cuda.Event, 3 interleaved rounds, medians of rounds 2+, per FRAME",
                       config=f"int8 hop over a flat B*H*W run, {arm}", session=st.session,
                       statistic="median", tier="measured", id=f"Y{i:03d}",
                       run="harness/scripts/hop_batch_probe.py", artifact=a.probe,
                       extra={"artifact_sha256": dig, "arm": arm,
                              "hops16_ms": round(16 * ms, 2)}))
    st.dump(a.out)
    print(st.claims_text().strip())
    print(f"# -> {a.out}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
