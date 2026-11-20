"""Records for the K-pick cost: the redundant build removed, and the price of asking for the sweep."""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys
import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)
from harness.core import Population, Record, Store  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", default="docs/wsp/verified/logs/startup-timed-hybrid.log")
    ap.add_argument("--pick", default="docs/wsp/verified/logs/startup-timed-pick.log")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-grid-pick-cost.jsonl")
    a = ap.parse_args(argv)
    def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
    st = Store(session="s-2026-11-19T17:00:00+03:00")
    pop = Population.of("process starts", ("bd_violet20s#fixed-K2", "bd_violet20s#auto-pick"),
                        "one cold backend process each, same clip and model")
    for i, (path, label) in enumerate(((a.fixed, "fixed K=2 (the shipped default)"),
                                       (a.pick, "RVE_INT8_GRID_K=auto, K=2 clean so the sweep is skipped")), start=1):
        total = None
        for line in open(path):
            m = re.search(r"startup: first frame enqueued ([\d.]+) s", line)
            if m:
                total = float(m.group(1))
        if total is None:
            continue
        st.emit(Record(value=total, unit="s", population=pop,
                       instrument="the app's own stdout, seconds since process start",
                       config=label + ", crops + hybrid decode", session=st.session, statistic="single",
                       tier="measured", id=f"KP{i:03d}", run="/tmp/opencode/timed_run.py", artifact=path,
                       extra={"artifact_sha256": sha(path), "arm": label}))
    st.dump(a.out)
    print(st.claims_text().strip())
    print(f"# -> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
