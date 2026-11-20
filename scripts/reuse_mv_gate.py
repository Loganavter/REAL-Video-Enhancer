"""Resolve which frame the codec's motion vectors actually predict from, and ask whether the field
reduces the residual on the frames the gate reused - where the correction is applied.

Why this exists: applying ffmpeg's exported motion vectors to `t-1` made the residual worse
(RU035), which reads as "the vectors are wrong". They are not. `AVMotionVector.source` is a
*reference-list* tag (-1 = list0, +1 = list1), not a frame distance, and this stream's P anchors
are two display frames apart, so a P frame's vectors point at the previous anchor. The reference
is recovered here by residual search over the last four display-order frames, per source list, and
the chosen distance is reported per frame - no assumption about the codec's list order.

The second half is the question the correction lane actually needs answered: on the frames the
gate reused, does the field reduce the residual in the region that fails? The region is measured
two ways - an 8 px border band and a high-gradient mask - because on the held-out clip the failing
pixels sit almost entirely in the former (see docs/wsp/verified/reuse-mv-reference.txt).

Usage:
  python3 scripts/reuse_mv_gate.py --clip cut.mkv --depth-log gate.log [--band 8] [--grad 20]
      [--grid 4] [--emit out.json]

Requires PyAV with motion-vector side data: the decoder option `flags2=+export_mvs` must be set
BEFORE decoding (`stream.codec_context.options = {...}`); without it frames carry timecode only,
and ffprobe prints the side-data type but not its contents.
"""
from __future__ import annotations

import argparse
import collections
import json
import re

import cv2
import numpy as np

FF_WIDTH, FF_HEIGHT = 1920, 1080


def motion_fields(path: str, width: int, height: int) -> list[dict]:
    """Per decoded frame (display order): the displacement field of each reference list.

    Block geometry: `dst_x/dst_y` is the block's position in this frame, `src_x/src_y` its
    position in the reference, `motion_x/motion_y` the displacement in `motion_scale` units
    (quarter-pel on H.264). Blocks with no coded vector stay at zero displacement, which is what
    a warp by an empty field does anyway.
    """
    import av

    out: list[dict] = []
    container = av.open(path)
    stream = container.streams.video[0]
    stream.codec_context.options = {"flags2": "+export_mvs"}
    for frame in container.decode(stream):
        by_source: dict[int, list[tuple]] = collections.defaultdict(list)
        for key, side in frame.side_data.items():
            if "MOTION" not in str(key):
                continue
            for m in side:
                scale = float(m.motion_scale) or 1.0
                by_source[int(m.source)].append(
                    (int(m.dst_x), int(m.dst_y), int(m.w), int(m.h),
                     float(m.motion_x) / scale, float(m.motion_y) / scale))
        fields = {}
        for source, vectors in by_source.items():
            dx = np.zeros((height, width), np.float32)
            dy = np.zeros((height, width), np.float32)
            for x, y, w, h, mx, my in vectors:
                dx[y:y + h, x:x + w] = mx
                dy[y:y + h, x:x + w] = my
            fields[source] = (dx, dy)
        out.append({"fields": fields, "type": int(frame.pict_type),
                    "vectors": sum(len(v) for v in by_source.values())})
    container.close()
    return out


class Gray:
    def __init__(self, path: str, width: int, height: int):
        import subprocess

        self.size = width * height
        self.proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={width}:{height}",
             "-pix_fmt", "gray", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)

    def read(self) -> np.ndarray | None:
        b = self.proc.stdout.read(self.size)
        if not b or len(b) < self.size:
            return None
        return np.frombuffer(b, np.uint8).reshape(FF_HEIGHT, FF_WIDTH).astype(np.float32)

    def close(self) -> None:
        self.proc.stdout.close()
        self.proc.terminate()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--depth-log", required=True)
    ap.add_argument("--band", type=int, default=8, help="border band side at full resolution of the cut")
    ap.add_argument("--grad", type=float, default=20.0, help="high-gradient threshold on the luma")
    ap.add_argument("--grid", type=int, default=4, help="how many display-order frames back the search goes")
    ap.add_argument("--emit", default="")
    a = ap.parse_args()

    depth = {}
    for line in open(a.depth_log, errors="ignore"):
        m = re.search(r"reuse frame=(\d+) depth=(\d+)", line)
        if m:
            depth[int(m.group(1))] = int(m.group(2))

    print(f"motion vectors: {a.clip}", flush=True)
    frames = motion_fields(a.clip, FF_WIDTH, FF_HEIGHT)
    with_field = sum(1 for f in frames if f["fields"])
    print(f"  {len(frames)} frames decoded, {with_field} carry motion vectors")

    gx, gy = np.meshgrid(np.arange(FF_WIDTH, dtype=np.float32), np.arange(FF_HEIGHT, dtype=np.float32))
    band = np.zeros((FF_HEIGHT, FF_WIDTH), bool)
    band[:a.band, :] = band[-a.band:, :] = True
    band[:, :a.band] = band[:, -a.band:] = True

    gray = Gray(a.clip, FF_WIDTH, FF_HEIGHT)
    history: list[np.ndarray] = []
    rows = []
    while True:
        cur = gray.read()
        if cur is None:
            break
        history.append(cur)
        i = len(history) - 1
        k = depth.get(i)
        if not k or len(history) < a.grid + 1:
            continue
        if i >= len(frames):
            continue
        grad = (np.abs(np.diff(cur, axis=0, append=cur[-1:, :])) +
                np.abs(np.diff(cur, axis=1, append=cur[:, -1:])))
        hot = grad > a.grad
        candidates = []
        for source, (dx, dy) in frames[i]["fields"].items():
            moving = (np.abs(dx) + np.abs(dy)) > 0.5
            if moving.sum() < 2000:
                continue
            for d in range(1, a.grid + 1):
                ref = history[-1 - d]
                pred = cv2.remap(ref, gx + dx, gy + dy, interpolation=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
                entry = {"source": source, "distance": d}
                for name, sel in (("full", np.ones_like(band)), ("band", band), ("hot", hot),
                                  ("interior", ~band)):
                    if sel.sum() < 100:
                        continue
                    rw = float(np.abs(cur - pred)[sel].mean())
                    rp = float(np.abs(cur - ref)[sel].mean())
                    entry[name] = {"warped": round(rw, 3), "plain": round(rp, 3),
                                   "gain_pct": round(100 * (1 - rw / max(rp, 1e-9)), 1)}
                entry["moving_pct"] = round(100 * float(moving.mean()), 1)
                candidates.append(entry)
        if candidates:
            best = min(candidates, key=lambda c: c["full"]["warped"])
            at_own = [c for c in candidates if c["distance"] == 2 and c["source"] == -1]
            at_depth = [c for c in candidates if c["distance"] == k and c["source"] == -1]
            rows.append({"frame": i, "depth": k, "pict_type": frames[i]["type"],
                         "vectors": frames[i]["vectors"], "best": best,
                         "hot_px_pct": round(100 * float(hot.mean()), 2),
                         "at_distance_2": at_own[0] if at_own else None,
                         "at_reuse_depth": at_depth[0] if at_depth else None,
                         "candidates": candidates})
    gray.close()

    print(f"reused frames analysed: {len(rows)}")
    if rows:
        print("  best distance:", dict(sorted(collections.Counter(
            r["best"]["distance"] for r in rows).items())))
        print("  best source:", dict(sorted(collections.Counter(
            r["best"]["source"] for r in rows).items())))
    def median_of(key: str, field: str = "band"):
        vals = [r[key][field]["gain_pct"] for r in rows
                if r.get(key) and field in r[key]]
        return round(float(np.median(vals)), 1) if vals else None
    print(f"  gain in the band, at the frame's own reuse depth: {median_of('at_reuse_depth')}% median")
    print(f"  gain in the band, at the codec's P anchor (distance 2): {median_of('at_distance_2')}% median")
    print(f"  gain in the interior, at the codec's P anchor: {median_of('at_distance_2', 'interior')}% median")

    summary = {
        "clip": a.clip, "frames": len(frames), "with_field": with_field,
        "reused": len(rows), "band_px_pct": round(100 * float(band.mean()), 2),
        "hot_px_pct": None,  # filled below
        "best_distance": dict(sorted(collections.Counter(r["best"]["distance"] for r in rows).items())),
        "best_source": dict(sorted(collections.Counter(r["best"]["source"] for r in rows).items())),
        "band_gain_at_reuse_depth_median": median_of("at_reuse_depth"),
        "band_gain_at_distance_2_median": median_of("at_distance_2"),
        "interior_gain_at_distance_2_median": median_of("at_distance_2", "interior"),
    }
    if rows:
        summary["hot_px_pct"] = round(float(np.median([r["hot_px_pct"] for r in rows])), 2)
    if a.emit:
        json.dump({"summary": summary, "rows": rows}, open(a.emit, "w"))
        print("wrote", a.emit)
    print("NOT checked: bi-predicted blocks (two vectors for one block) weighted by the codec's own "
          "weights are used as two separate fields, and occlusion is not detected - a vector with no "
          "matching source content simply shows as a negative gain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
