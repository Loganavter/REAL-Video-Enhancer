"""Records for the start-up split: where the seconds before the first upscaled frame go.

Input is a timestamped run of the backend (`/tmp/opencode/timed_run.py` prefixes every line with seconds since
process start), so the split is read from the app's own log rather than reconstructed from memory.

The three blocks are read by marker, and the markers are the app's own messages:
  interpreter  process start -> the first "head convolution body.0" line (imports, CLI, ffprobe, CUDA init)
  probe        "Loading TensorRT engine" -> "int8 probe:" (engine deserialize + the calibration's 16 frames)
  body         "int8 calibration frames:" -> "int8 body first-frame sanity" (Triton JIT + 24 frames + first call)
  total        "startup: first frame enqueued N s after process start"
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys

import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)
from harness.core import Population, Record, Store  # noqa: E402

LINE = re.compile(r"^\s*([\d.]+)\s+(.*)$")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="/tmp/opencode/startup_timed.log")
    ap.add_argument("--out", default="docs/wsp/verified/records/startup-split.jsonl")
    ap.add_argument("--clip", default="bd_violet20s.mkv")
    ap.add_argument("--prefix", default="ST")
    ap.add_argument("--config", default=("calibration CACHE MISS (16 frames measured), full frames"))
    a = ap.parse_args(argv)

    rows = []
    for line in open(a.log):
        m = LINE.match(line)
        if m:
            rows.append((float(m.group(1)), m.group(2)))
    if not rows:
        raise SystemExit(f"{a.log}: not a timestamped log")

    def first(marker: str) -> float:
        for t, text in rows:
            if marker in text:
                return t
        raise SystemExit(f"marker not found: {marker!r}")

    t_head = first("head convolution body.0")
    t_engine = first("Loading TensorRT engine")
    t_probe = first("int8 probe:")
    t_frames = first("int8 calibration frames:")
    t_body = first("int8 body first-frame sanity")
    total = None
    for t, text in rows:
        m = re.search(r"startup: first frame enqueued ([\d.]+) s", text)
        if m:
            total = float(m.group(1))
    if total is None:
        raise SystemExit("the log has no 'startup: first frame enqueued' line")

    digest = hashlib.sha256(open(a.log, "rb").read()).hexdigest()
    pop = Population.of("process starts", (f"{a.clip}#cold-run",),
                        f"{a.clip}, one cold backend process, int8 arm, null sink")
    st = Store(session="s-2026-11-19T15:10:00+03:00")
    instr = "the app's own stdout, each line stamped with seconds since process start"
    cfg = (f"{a.clip}, 1920x1080, backend tensorrt, int8_fit int8, no temporal reuse, null sink, "
           f"opt-5 tiling-full engine, {a.config}")
    for rid, name, value in (
        (f"{a.prefix}001", "start-up to the first enqueued frame", total),
        (f"{a.prefix}002", "interpreter + CLI + ffprobe + CUDA init (process start -> model load)", t_head),
        (f"{a.prefix}003", "engine deserialize + the int8 probe's 16 calibration frames", t_probe - t_engine),
        (f"{a.prefix}004", "Triton JIT + 24 calibration frames + first body call", t_body - t_frames),
    ):
        st.emit(Record(value=round(value, 3), unit="s", population=pop, instrument=instr, config=cfg,
                       session=st.session, statistic="single", tier="measured", id=rid,
                       run="/tmp/opencode/timed_run.py", artifact=a.log,
                       extra={"artifact_sha256": digest, "block": name}))
    st.dump(a.out)
    print(st.claims_text().strip())
    print(f"# -> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
