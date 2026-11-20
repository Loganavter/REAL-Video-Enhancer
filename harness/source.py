"""Sources: read what a measurement's INPUT actually is, and refuse a false population label.

Why this exists: from 2026-11-15 to 2026-11-18 every content number in this project was measured on
`/tmp/opencode/{violet,csm}20s.mkv`, which the prose called "the BD remaster". They were 4.04 and
3.98 Mbps h264 re-encodes of a 37.8 Mbps BDRemux - a 9.4x re-encode. The harness hashed the files
correctly and never asked whether the label was true; it says so in its own scope note ("whether a
population label is true ... stays a human judgement"). The noise floors, the gate's acceptance share,
the ROC, the channel-max threshold and every image handed to the owner inherited the false label.

`env-check` was the same idea one level down: a claim about the machine is checkable, so it is
checked. A claim about the input file is checkable the same way - `ffprobe` reads it - and the
stream-copy claim has an exact test that does not need to trust anyone:

  * the clip's video **packet sizes** must appear as a contiguous subsequence of the source's, which
    is only true when the bytes were copied, not decoded and re-encoded;
  * the bit rate is then compared against the manifest's threshold for the label the clip carries;
  * the fields the acceptance statistic depends on (pix_fmt, colour tags, frame count) are recorded,
    because a missing matrix or a different frame count is a population change, not a detail.

Nothing here decides whether a population is the one a DECISION needs. It decides whether the file is
what the label says it is, which is the cheaper question and the one that was wrong.

Usage:
  python -m harness.cli source-check docs/wsp/sources.yaml
  python -m harness.cli source-check docs/wsp/sources.yaml --records docs/wsp/verified/records
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
from dataclasses import dataclass, field

FACT_FIELDS = ("codec_name", "profile", "pix_fmt", "width", "height", "r_frame_rate",
               "color_range", "color_space", "color_transfer", "color_primaries",
               "field_order", "has_b_frames")


@dataclass
class Facts:
    bit_rate_mbps: float
    packets: int
    fields: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"bit_rate_mbps": round(self.bit_rate_mbps, 3), "packets": self.packets,
                **{k: self.fields.get(k, "") for k in FACT_FIELDS}}


def probe(path: str, ffprobe: str = "ffprobe") -> Facts:
    """Read the container facts and count packets without decoding a frame."""
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets",
           "-show_entries",
           "stream=" + ",".join(FACT_FIELDS) + ",nb_read_packets:format=bit_rate",
           "-of", "json", path]
    raw = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    st = raw["streams"][0]
    br = raw.get("format", {}).get("bit_rate")
    if not br or float(br) <= 0:
        raise ValueError(f"{path}: the container declares no bit rate; a label cannot be checked")
    return Facts(bit_rate_mbps=float(br) / 1e6, packets=int(st.get("nb_read_packets", 0)),
                 fields={k: str(st.get(k, "")) for k in FACT_FIELDS})


def packet_sizes(path: str, ffprobe: str = "ffprobe") -> list[int]:
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0",
           "-show_entries", "packet=size", "-of", "csv=p=0", path]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return [int(x.strip().rstrip(",")) for x in out.split() if x.strip().rstrip(",")]


def stream_copy_proof(clip: str, source: str, head: int = 40,
                      ffprobe: str = "ffprobe") -> dict:
    """The exact test for `-c copy`: the clip's packets ARE a contiguous slice of the source's.

    A re-encode produces different packet sizes (and a different number of them), so no run of the
    source matches. `offset` is the source packet index of the clip's FIRST video packet; `compared`
    is how many packets were matched end to end, which includes the first one.
    """
    c, s = packet_sizes(clip, ffprobe), packet_sizes(source, ffprobe)
    if len(c) < head:
        return {"ok": False, "reason": f"clip has {len(c)} packets, fewer than the {head} needed"}
    needle = c[:head]
    for i in range(len(s) - len(needle) + 1):
        if s[i:i + head] == needle:
            n = min(len(c), len(s) - i)
            ok = s[i:i + n] == c[:n]
            return {"ok": ok, "offset": i, "compared": n, "packets": len(c)}
    return {"ok": False, "offset": None, "compared": 0, "packets": len(c)}


def check(manifest: dict, clip_root: str = "", probe_fn=probe,
          proof_fn=stream_copy_proof, prove: bool = False,
          read_files: bool = True) -> tuple[list[str], list[dict]]:
    """Problems for the manifest as written, plus the facts re-read from the files.

    A clip is checked only when its file is reachable; a missing file is reported as NOT CHECKED,
    never as a pass - the same rule the harness applies to an uncovered kind.
    """
    problems: list[str] = []
    floor = float(manifest.get("min_bitrate_mbps", 0.0) or 0.0)
    rows: list[dict] = []
    for clip in manifest.get("clips", []):
        name = clip.get("file", "")
        path = os.path.join(clip_root, name) if clip_root else name
        row = {"file": name, "label": clip.get("label", ""), "checked": False,
               "contaminated": bool(clip.get("contaminated", False))}
        if not read_files or not os.path.exists(path):
            row["status"] = ("NOT CHECKED: probing disabled" if not read_files
                             else "NOT CHECKED: file not present")
            if read_files and clip.get("must_exist", False):
                problems.append(f"{name}: must_exist is set but the file is absent")
            if clip.get("contaminated"):
                row["contamination"] = clip.get("contamination", "")
            rows.append(row)
            continue
        try:
            facts = probe_fn(path)
        except Exception as exc:                                   # noqa: BLE001
            row["status"] = f"NOT CHECKED: probe failed ({exc})"
            problems.append(f"{name}: probe failed - {exc}")
            rows.append(row)
            continue
        row["facts"] = facts.as_dict()
        row["checked"] = True
        claim = clip.get("facts") or {}
        for key in ("bit_rate_mbps", "packets", "pix_fmt", "color_range", "has_b_frames"):
            if key in claim and str(claim[key]) != str(row["facts"].get(key)):
                problems.append(f"{name}: manifest says {key}={claim[key]}, file reads "
                                f"{row['facts'].get(key)}")
        if clip.get("label_requires_bitrate", True) and not clip.get("contaminated"):
            clip_floor = float(clip.get("min_bitrate_mbps", floor) or 0.0)
            if clip_floor and facts.bit_rate_mbps < clip_floor:
                problems.append(f"{name}: {facts.bit_rate_mbps:.2f} Mbps is below the "
                                f"{clip_floor:.1f} Mbps floor for the label "
                                f"'{clip.get('label', '')}'")
        cut = clip.get("cut") or {}
        if cut.get("mode") == "stream_copy":
            src = clip.get("source", "")
            if not prove:
                # The packet subsequence is the only exact test, and it reads the whole source. The
                # recorded offset is checked for internal consistency here; re-running the proof is
                # opt-in (`--prove`) because it is minutes per clip.
                ok = (cut.get("offset") is not None and cut.get("compared") and cut.get("packets")
                      and cut["compared"] >= cut["packets"] - 1)
                row["proof"] = (f"recorded offset={cut.get('offset')} "
                                f"compared={cut.get('compared')}/{cut.get('packets')}, not re-run")
                if not ok:
                    problems.append(f"{name}: claims -c copy but the cut block is incomplete or "
                                    f"inconsistent (offset/compared/packets)")
            elif not src or not os.path.exists(src):
                row["proof"] = "NOT CHECKED: source not present"
                problems.append(f"{name}: claims -c copy but the source is not reachable")
            else:
                proof = proof_fn(path, src)
                row["proof"] = proof
                if not proof.get("ok"):
                    problems.append(f"{name}: claims -c copy but the packet sequence does not "
                                    f"appear in the source")
                elif cut.get("offset") is not None and cut["offset"] != proof.get("offset"):
                    problems.append(f"{name}: manifest says offset {cut['offset']}, packets say "
                                    f"{proof.get('offset')}")
        elif clip.get("contaminated"):
            row["contamination"] = clip.get("contamination", "")
        rows.append(row)
    return problems, rows


def contaminated_records(manifest: dict, records_glob: str) -> list[tuple[str, int, str]]:
    """Records whose population items name a contaminated file, by (path, lineno, record id).

    This is the mechanical half of `contaminated_by`: the item-set hash already names the file, so
    propagation needs no rewriting of history - only a lookup.
    """
    bad = [c["file"] for c in manifest.get("clips", []) if c.get("contaminated")]
    if not bad:
        return []
    # A substring test would flag `bd_violet20s.mkv` for the contaminated `violet20s.mkv`: the BD cut's
    # name CONTAINS the re-encode's. Require the basename at a boundary.
    pat = re.compile(r"(?<![\w.-])(" + "|".join(re.escape(b) for b in bad) + r")(?![\w.-])")
    hits: list[tuple[str, int, str]] = []
    paths = sorted(glob.glob(records_glob)) or ([records_glob] if os.path.isfile(records_glob) else [])
    for path in paths:
        for lineno, line in enumerate(open(path), 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            items = (rec.get("population") or {}).get("items") or []
            named = sorted({m.group(1) for it in items for m in [pat.search(it)] if m})
            if named:
                hits.append((path, lineno, f"{rec.get('id', '?')} ({', '.join(named)})"))
    return hits


def main(argv: list[str] | None = None) -> int:
    import argparse
    import yaml

    ap = argparse.ArgumentParser(prog="python -m harness.cli source-check")
    ap.add_argument("manifest")
    ap.add_argument("--clip-root", default="")
    ap.add_argument("--records", default="", help="records dir/glob to scan for contaminated items")
    ap.add_argument("--no-probe", action="store_true", help="check the manifest alone, do not read files")
    ap.add_argument("--prove", action="store_true",
                    help="re-run the -c copy packet-subsequence proof (minutes per clip)")
    args = ap.parse_args(argv)

    manifest = yaml.safe_load(open(args.manifest)) or {}
    problems, rows = check(manifest, args.clip_root, prove=args.prove,
                           read_files=not args.no_probe)

    for row in rows:
        tag = "CONTAMINATED" if row["contaminated"] else "ok"
        bits = row.get("facts") or {}
        line = f"{row['file']:<28} {tag:<13}"
        if not row.get("checked"):
            line += " " + row.get("status", "")
        elif bits:
            line += (f" {bits.get('bit_rate_mbps', '?')} Mbps {bits.get('codec_name', '')} "
                     f"{bits.get('pix_fmt', '')} matrix={bits.get('color_space') or 'undeclared'} "
                     f"range={bits.get('color_range') or 'undeclared'} "
                     f"frames={bits.get('packets')}")
        if row.get("proof"):
            pr = row["proof"]
            line += f" | -c copy: {pr}" if not isinstance(pr, dict) else (
                f" | -c copy offset={pr.get('offset')} compared={pr.get('compared')}/{pr.get('packets')}")
        print(line)
    for p in problems:
        print(f"REFUSED: {p}")
    if args.records:
        hits = contaminated_records(manifest, args.records)
        print(f"\nrecords naming a contaminated input: {len(hits)}")
        for path, lineno, rid in hits[:40]:
            print(f"  {os.path.basename(path)}:{lineno} {rid}")
        if len(hits) > 40:
            print(f"  ... {len(hits) - 40} more")
    print("\nNOT checked here: whether this population is the one a DECISION needs; whether the "
          "colour matrix the app assumes equals the one the file implies; whether a frame index "
          "from another cut names the same picture.")
    return 1 if problems else 0
