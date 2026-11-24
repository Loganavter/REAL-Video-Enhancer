#!/usr/bin/env python3
"""Join the pane's seek log with the backend's, so a seek's latency is a table instead of a mystery.

Why this exists (owner, 2026-11-25: *"seeking is extremely slow"*, and then *"we have a profiling problem again"*):
each side had numbers and neither had the middle. The pane knew `turnaround_ms` - a SUM of everything - and the
backend knew `backend_total` from the moment IT acted. Between them sat two invisible stretches: the request waiting
for the renderer's loop to reach it, and the published picture waiting for the pane's next 100 ms poll.

Both processes are on one host, so `time.time()` is the clock that can cross between them (monotonic cannot). The pane
stamps every seek line with `ts=`; the backend stamps each answered request with `pickup_wall=`. Joined on the seek
GENERATION, the five stages are arithmetic:

    press --(press_to_pickup)--> backend picked --(reader/decode/bytes/model/publish)--> published
          --(publish_to_paint)--> pane painted

Usage:
    python3 harness/scripts/seek_profile.py                      # ./preview-log.txt + ./log.prev.txt (or log.txt)
    python3 harness/scripts/seek_profile.py --pane A --backend B [--json]
Exit: 0 with a table, 1 when neither side yields a single seek (a silent empty answer is a lie), 2 when the pane side
has seeks and the backend side does not (then the join is impossible and that IS the finding).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import sys

#: `preview: seek decide press=7 route=display target_s=5.000 … generation=3` (with or without `ts=`)
_PANE_DECIDE = re.compile(r"^preview: seek decide .*?press=(?P<press>\d+).*?generation=(?P<gen>\d+)(?:.*?ts=(?P<ts>[\d.]+))?")
#: `preview: seek turnaround_ms=578.2 target_frame=46 press=7 … target_gen=3 landed_gen=3 ts=…`
_PANE_CLOSE = re.compile(r"^preview: seek turnaround_ms=(?P<ms>[\d.]+).*?press=(?P<press>\d+)"
                         r".*?target_gen=(?P<gen>\d+)(?:.*?ts=(?P<ts>[\d.]+))?")
#: `seek: displayed the frame at 5.972s (generation 1, not written to the export); stages reader=… decode=…`
_BACKEND = re.compile(r"seek: displayed the frame at (?P<at>[\d.]+)s \(generation (?P<gen>\d+)[^)]*\); stages "
                      r"(?P<stages>.*?)pickup_wall=(?P<wall>[\d.]+)")

_STAGE = re.compile(r"(?P<name>[a-z_]+)=(?P<value>[\d.]+)ms")


def parse_pane(lines) -> tuple[dict[int, dict], dict[int, dict]]:
    """`generation -> decide` and `generation -> close`, each with the wall stamp when the line carries one."""
    decides: dict[int, dict] = {}
    closes: dict[int, dict] = {}
    for line in lines:
        match = _PANE_DECIDE.match(line)
        if match:
            gen = int(match.group("gen"))
            decides[gen] = {"press": int(match.group("press")),
                            "ts": None if match.group("ts") is None else float(match.group("ts"))}
            continue
        match = _PANE_CLOSE.match(line)
        if match:
            gen = int(match.group("gen"))
            closes[gen] = {"press": int(match.group("press")),
                           "turnaround_ms": float(match.group("ms")),
                           "ts": None if match.group("ts") is None else float(match.group("ts"))}
    return decides, closes


def parse_backend(lines) -> dict[int, dict]:
    """`generation -> the pickup and the stage split the renderer measured for it."""
    out: dict[int, dict] = {}
    for line in lines:
        match = _BACKEND.search(line)
        if not match:
            continue
        stages = {item.group("name"): float(item.group("value")) for item in _STAGE.finditer(match.group("stages"))}
        out[int(match.group("gen"))] = {"at": float(match.group("at")),
                                        "pickup_wall": float(match.group("wall")),
                                        "stages": stages}
    return out


def join(decides: dict[int, dict], closes: dict[int, dict], backend: dict[int, dict]) -> list[dict]:
    """One row per generation that BOTH sides know; a one-sided generation is reported, never dropped silently."""
    rows: list[dict] = []
    for gen in sorted(set(decides) | set(closes) | set(backend)):
        decide = decides.get(gen, {})
        close = closes.get(gen, {})
        served = backend.get(gen, {})
        row: dict = {"generation": gen, "press": decide.get("press") or close.get("press")}
        press_ts = decide.get("ts")
        publish_wall = served.get("pickup_wall")
        stages = served.get("stages", {})
        row.update({name: stages.get(name) for name in
                    ("reader", "decode", "bytes", "model", "publish", "backend_total")})
        row["press_to_pickup_ms"] = (
            None if press_ts is None or publish_wall is None else (publish_wall - press_ts) * 1000.0)
        row["publish_to_paint_ms"] = (
            None if close.get("ts") is None or publish_wall is None or "backend_total" not in stages
            else (close["ts"] - publish_wall) * 1000.0 - stages["backend_total"])
        row["turnaround_ms"] = close.get("turnaround_ms")
        row["unmatched"] = [side for side, has in
                            (("decide", bool(decide)), ("backend", bool(served)), ("close", bool(close))) if not has]
        rows.append(row)
    return rows


def _fmt(value) -> str:
    return "-" if value is None else f"{value:.1f}"


def render(rows: list[dict]) -> str:
    head = (f"{'gen':>4} {'press':>5} {'press→pick':>10} {'reader':>7} {'decode':>7} {'bytes':>6} "
            f"{'model':>7} {'publish':>8} {'backend':>8} {'incl. paint':>12} {'total':>8}  unmatched")
    out = [head, "-" * len(head)]
    for row in rows:
        paint = row["publish_to_paint_ms"]
        backend = row.get("backend_total")
        total = row["turnaround_ms"]
        out.append(
            f"{row['generation']:>4} {str(row['press']):>5} {_fmt(row['press_to_pickup_ms']):>10} "
            f"{_fmt(row.get('reader')):>7} {_fmt(row.get('decode')):>7} {_fmt(row.get('bytes')):>6} "
            f"{_fmt(row.get('model')):>7} {_fmt(row.get('publish')):>8} {_fmt(backend):>8} "
            f"{_fmt(paint):>12} {_fmt(total):>8}  {','.join(row['unmatched']) or ''}")
    measured = [row for row in rows if row.get("turnaround_ms") is not None]
    if measured:
        out.append("")
        for name, key in (("press→pickup", "press_to_pickup_ms"), ("backend_total", "backend_total"),
                          ("publish→paint", "publish_to_paint_ms"), ("total", "turnaround_ms"),
                          ("model", "model")):
            values = [row[key] for row in measured if row.get(key) is not None]
            if values:
                out.append(f"{name:>14}: median {statistics.median(values):6.1f} ms   max {max(values):6.1f} ms "
                           f"(n={len(values)})")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pane", default="preview-log.txt")
    parser.add_argument("--backend", default=None,
                        help="the backend log (default: log.txt, the current run; falls back to log.prev.txt)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    pane = pathlib.Path(args.pane)
    backend_path = pathlib.Path(args.backend) if args.backend else None
    if backend_path is None:
        # The CURRENT run writes `log.txt`; `log.prev.txt` is the one before it. Preferring the rotated file measured
        # the wrong run and reported "the backend log has none" while the answer sat in log.txt (2026-11-25).
        backend_path = pathlib.Path("log.txt")
        if not backend_path.exists():
            backend_path = pathlib.Path("log.prev.txt")
    if not pane.exists():
        print(f"seek_profile: no pane log at {pane}", file=sys.stderr)
        return 1
    bad = 0
    for path in (pane, backend_path):
        if not path.exists():
            print(f"seek_profile: no backend log at {path} - the join needs both sides", file=sys.stderr)
            bad = 2

    decides, closes = parse_pane(pane.read_text(encoding="utf-8", errors="replace").splitlines())
    backend = parse_backend(backend_path.read_text(encoding="utf-8", errors="replace").splitlines()) \
        if backend_path.exists() else {}
    rows = join(decides, closes, backend)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(render(rows))
    if not rows:
        print("seek_profile: neither log yielded a seek - nothing to join", file=sys.stderr)
        return 1
    if bad:
        return bad
    if closes and not backend:
        print("seek_profile: the pane logged seeks and the backend log has none - the join is impossible, and that "
              "is a finding, not an empty answer", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
