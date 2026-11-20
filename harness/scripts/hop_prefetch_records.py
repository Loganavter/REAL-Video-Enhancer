"""Records for the test of the smem theory: reordering, and the scalar microbenchmark."""
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
    ap.add_argument("--probe", default="harness/docs/wsp/verified/logs/hop-prefetch/probe.json")
    ap.add_argument("--bench", default="harness/docs/wsp/verified/logs/hop-smem-theory/bench.json")
    ap.add_argument("--out", default="harness/docs/wsp/verified/records/int8-hop-prefetch.jsonl")
    a = ap.parse_args(argv)
    pr = json.load(open(a.probe))
    dig = hashlib.sha256(open(a.probe, "rb").read()).hexdigest()
    pop = Population.of("frames", (f"{pr['clip']}#frame{pr['frame']}",),
                        f"{pr['clip']} frame {pr['frame']}, one hop, load ORDER varied (arithmetic unchanged)")
    st = Store(session=f"s-{pr['take']}")
    i = 0
    for arm, ms in pr["median"].items():
        i += 1
        st.emit(Record(value=round(ms, 3), unit="ms", population=pop,
                       instrument="torch.cuda.Event, 3 interleaved rounds, medians of rounds 2+",
                       config=f"int8 hop, 64->64 3x3, BLOCK_M 128, 4 warps, loads = {arm}",
                       session=st.session, statistic="median", tier="measured", id=f"P{i:03d}",
                       run="harness/scripts/hop_prefetch_probe.py", artifact=a.probe,
                       extra={"artifact_sha256": dig, "arm": arm, "tops": round(152.9 / ms, 1)}))
    if os.path.exists(a.bench):
        b = json.load(open(a.bench))
        bd = hashlib.sha256(open(a.bench, "rb").read()).hexdigest()
        bpop = Population.of("frames", ("1080x1920#synthetic-int8",),
                             "a scalar CUDA microbenchmark of the hop's address pattern: nine dependent "
                             "gathers against a shared-memory window, no MMA")
        for arm, ms in b.items():
            i += 1
            st.emit(Record(value=round(ms, 3), unit="ms", population=bpop,
                           instrument="cudaEvent around 10 launches of a hand-written CUDA kernel",
                           config=f"scalar int8 sums, 64 channels per pixel, arm {arm}",
                           session=st.session, statistic="median", tier="measured", id=f"P{i:03d}",
                           run="logs/hop-smem-theory/smem_drive.py", artifact=a.bench,
                           extra={"artifact_sha256": bd, "arm": arm, "of": "l2_vs_smem"}))
    st.dump(a.out)
    print(st.claims_text().strip())
    print(f"# -> {a.out}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
