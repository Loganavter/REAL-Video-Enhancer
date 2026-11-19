"""Records for the per-hop K falsifier."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness.core import Population, Record, Store  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="docs/wsp/verified/logs")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-grid-k-perlayer.jsonl")
    a = ap.parse_args(argv)
    def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
    st = Store(session="s-2026-11-19T16:20:00+03:00")
    n = 1
    for fname in ("grid-perlayer-futsuu.json", "grid-perlayer-realesr.json"):
        path = os.path.join(a.dir, fname)
        d = json.load(open(path))
        pop = Population.of("pixels of one frame", (d["model"],), f"{d['model']}, three held-out frames, mean share")
        for label, val in ((f"best single K={d['best_single']}", d["single"][d["best_single"]]),
                           ("per-hop greedy K", d["per_hop_score"])):
            st.emit(Record(value=float(val), unit="%", population=pop,
                           instrument="channel-max |delta| against the fp16 model, mean over 3 held-out frames",
                           config=f"{d['model']} at {d['scale']}x, 1920x1080, crops calibration; {label}",
                           session=st.session, statistic="mean", tier="measured", id=f"PK{n:03d}",
                           run="scripts/grid_k_perlayer_probe.py", artifact=path,
                           extra={"artifact_sha256": sha(path), "policy": label}))
            n += 1
    st.dump(a.out)
    print(st.claims_text().strip())
    print(f"# -> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
