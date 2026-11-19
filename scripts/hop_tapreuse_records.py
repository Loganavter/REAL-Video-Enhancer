"""Records for the tap-reuse attempt in Triton."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness.core import Population, Record, Store  # noqa: E402

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="docs/wsp/verified/logs/hop-tapreuse/probe.json")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-hop-tapreuse.jsonl")
    a = ap.parse_args(argv)
    pr = json.load(open(a.probe))
    dig = hashlib.sha256(open(a.probe, "rb").read()).hexdigest()
    pop = Population.of("frames", (f"{pr['clip']}#frame{pr['frame']}",),
                        f"{pr['clip']} frame {pr['frame']}, one hop, three rows loaded once then "
                        f"nine taps formed in registers")
    st = Store(session=f"s-{pr['take']}")
    for i, (arm, ms) in enumerate(pr["median"].items(), start=1):
        st.emit(Record(value=round(ms, 3), unit="ms", population=pop,
                       instrument="torch.cuda.Event, 3 interleaved rounds, medians of rounds 2+",
                       config=f"int8 hop 64->64 3x3, {arm}", session=st.session, statistic="median",
                       tier="measured", id=f"U{i:03d}", run="scripts/hop_tapreuse_probe.py",
                       artifact=a.probe, extra={"artifact_sha256": dig, "arm": arm,
                                                "tops": round(152.9 / ms, 1)}))
    st.dump(a.out)
    print(st.claims_text().strip())
    print(f"# -> {a.out}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
