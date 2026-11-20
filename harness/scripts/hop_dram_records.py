"""Records for the hop's DRAM and L2 traffic: counters, not constructions."""
from __future__ import annotations
import argparse, hashlib, os, sys
import os as _bos, sys as _bsys
_HARNESS_ROOT = _bos.path.dirname(_bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
if _HARNESS_ROOT not in _bsys.path:
    _bsys.path.insert(0, _HARNESS_ROOT)
from harness.core import Population, Record, Store  # noqa: E402

# read from logs/hop-dram/report.txt (ncu 2026.4.0, one launch)
ROWS = [
    ("D001", "dram__bytes_read.sum", 132.84, "MB", "the compulsory read of one hop's codes"),
    ("D002", "dram__bytes_write.sum", 117.63, "MB", "the write, minus what was still in L2 at kernel end"),
    ("D003", "lts__t_sectors.sum x 32 B", 1528.0, "MB", "L2 traffic - 6.1x the DRAM traffic"),
    ("D004", "lts__t_sector_hit_rate.pct", 91.25, "%", "L2 hit rate"),
    ("D005", "dram__bytes.sum.per_second", 156.67, "GB/s", "DRAM throughput during the launch"),
    ("D006", "gpu__time_duration.sum", 1.60, "ms", "the profiled launch's duration"),
]

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="docs/wsp/verified/logs/hop-dram/report.txt")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-hop-dram.jsonl")
    a = ap.parse_args(argv)
    dig = hashlib.sha256(open(a.report, "rb").read()).hexdigest()
    pop = Population.of("frames", ("bd_violet20s.mkv#frame169",),
                        "one hop launch of the int8 body, DRAM and L2 counters")
    sess = "s-2026-11-19T07:40:00Z"
    st = Store(session=sess)
    for rid, metric, value, unit, what in ROWS:
        st.emit(Record(value=value, unit=unit, population=pop,
                       instrument="ncu 2026.4.0 counter, one launch inside a real body run",
                       config="int8 hop 64->64 3x3, BLOCK_M 128, 16200 CTAs", session=sess,
                       statistic="single", tier="measured", id=rid, run="ncu --metrics dram__bytes*",
                       artifact=a.report, extra={"artifact_sha256": dig, "metric": metric, "of": what}))
    st.dump(a.out)
    print(st.claims_text().strip())
    print(f"# -> {a.out}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
