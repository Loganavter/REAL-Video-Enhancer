"""Records for the tail CSC: the payload agreement and the isolated timing."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
import os as _bos, sys as _bsys
_HARNESS_ROOT = _bos.path.dirname(_bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
if _HARNESS_ROOT not in _bsys.path:
    _bsys.path.insert(0, _HARNESS_ROOT)
from harness.core import Population, Record, Store  # noqa: E402

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="docs/wsp/verified/logs/tail-csc/probe.json")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-tail-csc.jsonl")
    a = ap.parse_args(argv)
    pr = json.load(open(a.probe))
    dig = hashlib.sha256(open(a.probe, "rb").read()).hexdigest()
    pop = Population.of("frames", (f"{pr['clip']}#frame{pr['frame']}",),
                        f"{pr['clip']} frame {pr['frame']}, the tail plus the colour conversion to yuv420p")
    st = Store(session=f"s-{pr['take']}")
    for i, (arm, ms) in enumerate(pr["median"].items(), start=1):
        st.emit(Record(value=round(ms, 3), unit="ms", population=pop,
                       instrument="torch.cuda.Event, 3 interleaved rounds, medians of rounds 2+",
                       config=f"tail 64->12 + shuffle + residual + yuv420p, {arm}", session=st.session,
                       statistic="median", tier="measured", id=f"N{i:03d}",
                       run="scripts/tail_csc_probe.py", artifact=a.probe,
                       extra={"artifact_sha256": dig, "arm": arm}))
    st.emit(Record(value=float(pr["max_code_diff"]), unit="levels", population=pop,
                   instrument="byte-wise |delta| between the two payloads, 12 441 600 bytes",
                   config="fused tail CSC against the shipped fp16-RGB + pack conversion", session=st.session,
                   statistic="max", tier="measured", id="N003", run="scripts/tail_csc_probe.py",
                   artifact=a.probe,
                   extra={"artifact_sha256": dig, "of": "payload_delta",
                          "share_gt1_pct": pr["share_gt1_pct"]}))
    st.dump(a.out)
    print(st.claims_text().strip())
    print(f"# -> {a.out}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
