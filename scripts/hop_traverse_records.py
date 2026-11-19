"""Records for the reverse-traversal falsifier."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness.core import Population, Record, Store  # noqa: E402

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="docs/wsp/verified/logs/hop-traverse/probe.json")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-hop-traverse.jsonl")
    a = ap.parse_args(argv)
    pr = json.load(open(a.probe))
    dig = hashlib.sha256(open(a.probe, "rb").read()).hexdigest()
    pop = Population.of("frames", (f"{pr['clip']}#frame{pr['frame']}",),
                        f"{pr['clip']} frame {pr['frame']}, sixteen hops chained, CTA traversal order varied")
    st = Store(session=f"s-{pr['take']}")
    for i, (arm, ms) in enumerate(pr["median"].items(), start=1):
        st.emit(Record(value=round(ms, 3), unit="ms", population=pop,
                       instrument="torch.cuda.Event around the whole 16-launch chain, 3 rounds",
                       config=f"16 hops, traversal = {arm}", session=st.session, statistic="median",
                       tier="measured", id=f"V{i:03d}", run="scripts/hop_traverse_probe.py",
                       artifact=a.probe, extra={"artifact_sha256": dig, "arm": arm}))
    st.dump(a.out)
    print(st.claims_text().strip())
    print(f"# -> {a.out}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
