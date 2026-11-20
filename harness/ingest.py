"""Turn a backend run report into records, so a render's cost is a number with a commit and a card.

The backend writes the report (`--report path.json`); this reads it and emits one record per stage plus the reuse
statistics. That is the difference between "it felt slower" and "model wall went 31.7 -> 40.2 ms at commit X on this
card", which is the whole point of the harness: a claim that cannot be compared across commits is a memory.

    python3 -m harness.ingest run.json [--out harness/docs/wsp/verified/records/run-reports.jsonl] [--render RU100]

Record ids default to the next free RU number in the target file, so ingesting twice does not collide.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .core import Population, Record

STAGES = ("gate", "up_wall", "model", "pack", "read", "preview", "conv", "unadjusted")


def _next_id(path: Path) -> int:
    highest = 0
    if path.exists():
        for line in path.read_text().splitlines():
            match = re.match(r'.*"id":\s*"RU(\d+)"', line)
            if match:
                highest = max(highest, int(match.group(1)))
    return highest + 1


def records_from(report: dict, start: int = 1000) -> list[Record]:
    frames = report.get("run", {}).get("frames") or 0
    prov = report.get("provenance", {}) or {}
    settings = report.get("settings", {}) or {}
    gate = report.get("gate") or {}
    phases = report.get("phases_ms_per_frame", {}) or {}
    population = Population.of("frames", range(max(frames, 1)),
                               label=f"the frames of the run ({report.get('run', {}).get('input')})")
    config = (f"{report.get('run', {}).get('backend')} {report.get('run', {}).get('model')} "
              f"{report.get('run', {}).get('precision')}; reuse "
              f"{'on' if settings.get('temporal_reuse') else 'off'}"
              + (f", correction {'on' if settings.get('temporal_reuse_correction') else 'off'}"))
    base = dict(population=population, config=config, session=prov.get("measured_at", ""),
                tier="measured", run="backend/rve-backend.py --report",
                measured_at=prov.get("measured_at", ""), commit=prov.get("commit") or "",
                dirty=prov.get("dirty"), device=prov.get("device", ""), capability=prov.get("capability", ""),
                driver=prov.get("driver", ""), torch=prov.get("torch") or "", triton=prov.get("triton") or "",
                tensorrt=prov.get("tensorrt") or "", model_sha=prov.get("model_sha") or "")
    out: list[Record] = []
    number = start
    stage_names = {"up_wall": "model wall (upload to frame handover)", "model": "engine GPU time",
                   "pack": "pack + queue", "read": "read wait", "preview": "preview", "gate": "the gate"}
    for key, value in phases.items():
        if value is None or key not in STAGES:
            continue
        out.append(Record(value=float(value), unit="ms", statistic="mean", id=f"RU{number:03d}",
                          instrument=f"the backend's phase profile, {stage_names.get(key, key)} per frame",
                          note=f"phase '{key}' of the render loop, ms/frame, from a --report run", **base))
        number += 1
    if gate.get("reuse_pct") is not None:
        out.append(Record(value=float(gate["reuse_pct"]), unit="%", statistic="mean", id=f"RU{number:03d}",
                          instrument="the reuse gate's own counters (frames reused / frames processed)",
                          note=f"reuse pct, depth histogram {gate.get('depth_histogram')}", **base))
        number += 1
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m harness.ingest")
    parser.add_argument("report")
    parser.add_argument("--out", default="harness/docs/wsp/verified/records/run-reports.jsonl")
    parser.add_argument("--render", default="", help="record id for the first emitted record, e.g. RU100")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    report = json.loads(Path(args.report).read_text())
    out_path = Path(args.out)
    if args.render:
        start = int(re.sub(r"\D", "", args.render) or 0)
    else:
        start = _next_id(out_path)
    records = records_from(report, start)

    for record in records:
        print(f"{record.id}  {record.value:9.3f} {record.unit:3s}  {record.instrument[:70]}")
    if args.dry_run:
        print(f"(dry run: {len(records)} records not written)")
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict() if hasattr(record, "to_dict") else record.__dict__) + "\n")
    print(f"wrote {len(records)} records to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
