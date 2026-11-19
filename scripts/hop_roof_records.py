"""Records for the hop's decomposition: taps, store and the write floor, without ncu."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness.core import Population, Record, Store  # noqa: E402

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="docs/wsp/verified/logs/hop-roof/probe.json")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-hop-roof.jsonl")
    a = ap.parse_args(argv)
    pr = json.load(open(a.probe))
    dig, sess = hashlib.sha256(open(a.probe, "rb").read()).hexdigest(), f"s-{pr['take']}"
    clip = pr["clip"]
    pop = Population.of("frames", (f"{clip}#frame{pr['frame']}",),
                        f"{clip} frame {pr['frame']}, one hop of the int8 body, decomposed")
    inst = (f"torch.cuda.Event around {pr['iters']} launches behind 3 warm-ups, {pr['rounds']} "
            f"interleaved rounds, medians of rounds 2+")
    cfg = "int8 hop 64->64 3x3, BLOCK_M 128, 4 warps, 16200 CTAs; arms differ only in taps/store"
    st = Store(session=sess)
    labels = {"full 9 taps": "the shipped kernel", "taps=1": "one centre tap only",
              "taps=3": "three taps", "store-only": "epilogue + store over a zero accumulator",
              "no-store": "INVALID: no store means dead-code elimination (0.018 ms measures nothing)"}
    for i, (arm, label) in enumerate(labels.items(), start=1):
        st.emit(Record(value=round(pr["median"][arm], 3), unit="ms",
                       population=Population(kind=pop.kind, items=pop.items,
                                             label=f"{pop.label}, arm {arm}: {label}"),
                       instrument=inst, config=cfg, session=sess, statistic="median",
                       tier="measured", id=f"R{i:03d}", run="scripts/hop_roof_probe.py",
                       artifact=a.probe,
                       extra={"artifact_sha256": dig, "arm": arm, "of": "hop_time"}))
    st.dump(a.out)
    print(st.claims_text().strip())
    print(f"# -> {a.out}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
