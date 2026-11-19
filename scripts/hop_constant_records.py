"""Records for the scaling test: is the hop's cost per launch or per work?"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness.core import Population, Record, Store  # noqa: E402

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="docs/wsp/verified/logs/hop-constant/probe.json")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-hop-constant.jsonl")
    a = ap.parse_args(argv)
    pr = json.load(open(a.probe))
    dig = hashlib.sha256(open(a.probe, "rb").read()).hexdigest()
    st = Store(session=f"s-{pr['take']}")
    for i, r in enumerate(pr["rows"], start=1):
        pop = Population.of("frames", (f"{r['H']}x{r['W']}#int8-codes",),
                            f"one hop over a synthetic {r['H']}x{r['W']} code buffer, {r['px']/1e6:.2f} Mpx")
        st.emit(Record(value=r["ms"], unit="ms", population=pop,
                       instrument="torch.cuda.Event, 3 draws of 20 launches, median",
                       config=f"int8 hop 64->64 3x3, BLOCK_M 128, 4 warps, {r['H']}x{r['W']}",
                       session=st.session, statistic="median", tier="measured", id=f"S{i:03d}",
                       run="scripts/hop_constant_probe.py", artifact=a.probe,
                       extra={"artifact_sha256": dig, "tops": r["tops"], "ms_per_mpx": r["ms_per_mpx"]}))
    st.emit(Record(value=pr["implied_constant_ms"], unit="ms",
                   population=Population.of("frames", ("256x256#int8-codes", "3840x2160#int8-codes"),
                                            "the per-launch term implied by the two extreme frame sizes"),
                   instrument="derived from the two ends of the scaling sweep",
                   config="int8 hop, 0.07 Mpx against 8.29 Mpx", session=st.session, statistic="single",
                   tier="measured", id=f"S{len(pr['rows'])+1:03d}",
                   run="scripts/hop_constant_probe.py", artifact=a.probe,
                   extra={"artifact_sha256": dig, "of": "launch_constant"}))
    st.dump(a.out)
    print(st.claims_text().strip())
    print(f"# -> {a.out}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
