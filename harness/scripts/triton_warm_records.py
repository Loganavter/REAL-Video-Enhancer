"""Records for the Triton warm-up and the single engine load."""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys
import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)
from harness.core import Population, Record, Store  # noqa: E402


def total(path):
    for line in open(path):
        m = re.search(r"startup: first frame enqueued ([\d.]+) s", line)
        if m:
            return float(m.group(1))
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="docs/wsp/verified/logs")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-triton-warm.jsonl")
    a = ap.parse_args(argv)
    def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
    st = Store(session="s-2026-11-19T17:20:00+03:00")
    pop = Population.of("process starts", ("bd_violet20s#triton-cold", "bd_violet20s#triton-warm"),
                        "one cold backend process each, same clip and model")
    for i, (name, label) in enumerate((("startup_triton_cold.log", "Triton cache empty (a machine's first render)"),
                                       ("startup_triton_warm.log", "Triton cache warm (every later render)")),
                                      start=1):
        path = os.path.join(a.dir, name)
        t = total(path)
        if t is None:
            continue
        st.emit(Record(value=t, unit="s", population=pop,
                       instrument="the app's own stdout, seconds since process start",
                       config=label + ", crops + hybrid decode, fixed K=2", session=st.session,
                       statistic="single", tier="measured", id=f"TW{i:03d}", run="/tmp/opencode/timed_run.py",
                       artifact=path, extra={"artifact_sha256": sha(path)}))
    st.dump(a.out)
    print(st.claims_text().strip())
    print(f"# -> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
