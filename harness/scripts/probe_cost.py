"""What the probe costs today, in each of its two modes: calibration only, and calibration + certificate.

Why this exists: the decision "the certification is not worth its wall clock in the shipped path" needs the
clock, and the only recorded split (`probe-cost.txt`, 2026-11-17) was measured on another configuration
(4 of 8 test pairs) with the certificate on. Each mode runs in its OWN process, because that is what a render
pays: a fresh interpreter, a fresh CUDA context, a fresh model load. Rounds alternate so a warm cache or a
neighbour's load lands on both modes.

Usage:
  python harness/scripts/probe_cost.py --rounds 2 --out harness/docs/wsp/verified/records/int8-probe-cost.jsonl
"""

from __future__ import annotations
import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from harness.core import Population, Record, Store  # noqa: E402

GPU_LOCK = "/var/tmp/opencode/gpu.lock"
SECONDS = re.compile(r"PROBE_SECONDS ([\d.]+)")
CHILD = """\
import sys, time
from src.utils.Int8Probe import probe
t0 = time.perf_counter()
result = probe(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
               certify=sys.argv[5] == "1")
print(f"PROBE_SECONDS {time.perf_counter() - t0:.3f}")
print(result.stats())
"""


def one(clip: str, model: str, width: int, height: int, certify: bool, log) -> float:
    cmd = ["flock", "-n", GPU_LOCK, sys.executable, "-c", CHILD,
           clip, model, str(width), str(height), "1" if certify else "0"]
    out = subprocess.run(cmd, cwd=os.path.join(ROOT, "backend"), capture_output=True, text=True,
                         timeout=1800)
    log.write(f"# mode={'certify' if certify else 'calibration'} exit={out.returncode}\n")
    log.write(out.stdout + out.stderr + "\n")
    if out.returncode != 0:
        raise SystemExit(f"probe failed ({out.returncode}); see the log")
    match = SECONDS.search(out.stdout)
    if not match:
        raise SystemExit("the child did not print PROBE_SECONDS")
    return float(match.group(1))


def _measure(args, log_path: str, taken: dict, session: str) -> dict:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as log:
        log.write(f"# probe cost {session} clip={args.clip} model={os.path.basename(args.model)} "
                  f"width={args.width} height={args.height} rounds={args.rounds}\n")
        for r in range(1, args.rounds + 1):
            order = [("calibration", False), ("certify", True)]
            if r % 2 == 0:
                order.reverse()
            for name, certify in order:
                seconds = one(args.clip, args.model, args.width, args.height, certify, log)
                taken[name].append(seconds)
                log.write(f"# round={r} mode={name} seconds={seconds:.3f}\n")
                print(f"round {r}  {name:12} {seconds:7.3f} s")
    return taken


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="/tmp/opencode/bd_violet20s.mkv")
    ap.add_argument("--model", default="/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/"
                                       "custom_models/2x-Futsuu-Anime(1).pth")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", default="harness/docs/wsp/verified/records/int8-probe-cost.jsonl")
    ap.add_argument("--log", default="harness/docs/wsp/verified/logs/probe-cost-2026-11-19.log")
    ap.add_argument("--session", default="")
    ap.add_argument("--from-log", action="store_true",
                    help="rebuild the records from an existing log instead of re-running the probes")
    args = ap.parse_args(argv)

    log_path = os.path.join(ROOT, args.log) if not os.path.isabs(args.log) else args.log
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    session = args.session or time.strftime("s-%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    taken: dict[str, list[float]] = {"calibration": [], "certify": []}
    if args.from_log:
        done = re.compile(r"^# round=(\d+) mode=(\w+) seconds=([\d.]+)$", re.M)
        text = open(log_path).read()
        for _r, _m, _s in done.findall(text):
            taken[_m].append(float(_s))
        session = (re.search(r"^# probe cost (\S+)", text, re.M) or [None, session])[1]
        print(f"# rebuilt from {args.log}: " + ", ".join(f"{k}={v}" for k, v in taken.items()),
              file=sys.stderr)
    else:
        taken = _measure(args, log_path, taken, session)
    with open(log_path, "a" if args.from_log else "w") as log:
        log.write(f"# probe cost {session} clip={args.clip} model={os.path.basename(args.model)} "
                  f"width={args.width} height={args.height} rounds={args.rounds}\n")
        pass

    out_path = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out
    digest = hashlib.sha256(open(log_path, "rb").read()).hexdigest()
    store = Store(session=session)
    pop = Population.of("rounds", [f"{os.path.basename(args.clip)}#round-{i + 1}"
                                   for i in range(args.rounds)],
                        f"{args.clip}, probe cost, rounds 1-{args.rounds}, one process per run")
    # The mode is the VARIABLE under test, so it lives in the population label, `run` and `extra` - not in
    # `config`: Store.delta refuses a pair whose config differs, and rightly so (a changed config is a new
    # measurement, not a delta).
    ids = {}
    for name, label in (("calibration", "calibration only (decode + scales)"),
                        ("certify", "calibration + certificate (reference forwards + quantised twin)")):
        mode_pop = Population.of(pop.kind, pop.items,
                                 f"{args.clip}, probe cost, {label}, rounds 1-{args.rounds}")
        ids[name] = store.emit(Record(
            value=round(max(taken[name]), 3), unit="s", population=mode_pop,
            instrument="wall clock around Int8Probe.probe in a fresh process, flock on the GPU",
            config=f"Int8Probe.probe at {args.width}x{args.height}, one process per run",
            session=session, statistic="max", tier="measured",
            run=f"harness/scripts/probe_cost.py --mode {name}", artifact=args.log,
            extra={"artifact_sha256": digest, "of": "frame_time", "mode": name}))
    ids["delta"] = store.delta(ids["certify"].id, ids["calibration"].id,
                               instrument="difference of two measured modes, same rounds, same session",
                               note="what the certificate costs on top of the calibration it cannot exist without")
    with open(out_path, "w") as fh:
        for rec in store.records.values():
            fh.write(json.dumps(rec.to_dict()) + "\n")
    for rec in store.records.values():
        print(rec.claims_line())
    print(f"# {len(store.records)} records -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
