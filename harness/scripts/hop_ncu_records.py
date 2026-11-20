"""Records for the ncu run on one hop: the counters this chip does and does not expose."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)
from harness.core import Population, Record, Store  # noqa: E402

# the values are the one profiled launch's counters, read from the ncu report in logs/hop-ncu/
METRICS = [
    ("N001", "duration", 1.60, "ms", "the kernel's own duration, profiled (not replayed timing)"),
    ("N002", "sm_throughput", 73.97, "%", "Compute (SM) Throughput - the highest pipe"),
    ("N003", "dram_throughput", 55.98, "%", "DRAM Throughput"),
    ("N004", "memory_throughput", 156.60, "GB/s", "Memory Throughput, all levels"),
    ("N005", "l2_hit_rate", 91.32, "%", "L2 Hit Rate"),
    ("N006", "l1tex_hit_rate", 22.30, "%", "L1/TEX Hit Rate"),
    ("N007", "warp_cycles_per_issued", 9.04, "cycle", "Warp Cycles Per Issued Instruction"),
    ("N008", "spill_requests", 0.0, "count", "Local Memory Spilling Requests"),
]

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="docs/wsp/verified/logs/hop-ncu/report.txt")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-hop-ncu.jsonl")
    a = ap.parse_args(argv)
    dig = hashlib.sha256(open(a.report, "rb").read()).hexdigest()
    pop = Population.of("frames", ("bd_violet20s.mkv#frame169",),
                        "one hop launch of the int8 body, profiled by ncu inside the app's own process")
    sess = "s-2026-11-19T07:05:00Z"
    st = Store(session=sess)
    for rid, name, value, unit, what in METRICS:
        u = unit if unit in ("ms", "%", "GB/s", "cycle", "count") else "cycle"
        if u == "cycle":
            st.emit(Record(value=value, unit="count", population=pop, instrument="ncu 2026.4.0 section metric",
                           config="int8 hop 64->64 3x3, BLOCK_M 128, 16200 CTAs", session=sess,
                           statistic="single", tier="measured", id=rid, run="ncu --section SpeedOfLight",
                           artifact=a.report, extra={"artifact_sha256": dig, "metric": name, "of": what}))
        else:
            st.emit(Record(value=value, unit=u, population=pop, instrument="ncu 2026.4.0 section metric",
                           config="int8 hop 64->64 3x3, BLOCK_M 128, 16200 CTAs", session=sess,
                           statistic="single", tier="measured", id=rid, run="ncu --section SpeedOfLight",
                           artifact=a.report, extra={"artifact_sha256": dig, "metric": name, "of": what}))
    st.dump(a.out)
    print(st.claims_text().strip())
    print(f"# -> {a.out}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
