"""Which per-frame statistic ranks the frames where temporal reuse damaged the picture?

The gate's failure, in the owner's words: it only behaves on flat/binary content, refuses on digital noise,
and when it does reuse the result looks wrong. The lane's own diagnosis is that the decision statistic
(block mean of the absolute difference on the LR frame) cannot separate noise from motion. This tool tests
that directly on the population that already failed: it computes several candidate statistics per frame pair
and ranks them against the per-frame damage measured by ``harness/scripts/gate_sweep.py`` on the same clip.

Candidates (all on the same pair, same t-1 reference, same frame count):

  block_maxabs   current gate statistic: max over 32 px blocks of mean(|cur - prev|), 8-bit levels
  block_signed   the retired statistic:    max over 32 px blocks of |mean(cur) - mean(prev)|
  mctf_p99       motion-compensated: DIS flow prev->cur, warp, residual |cur - warp(prev)|, 99th percentile
  mctf_share     the same residual as a share of pixels above 3 x a robust sigma (MAD / 0.6745)
  mctf_mean      the same residual, mean

The ranking statistic is the AUC (Mann-Whitney) against "this frame is one of the frames above the 0.1%
acceptance bar", plus the Spearman correlation with the measured share. A statistic that ranks better is a
candidate decision variable; one that ranks the same is not the fix, whatever its threshold.

Usage:
  python3 scripts/reuse_stats.py --clip bd_csm_same.mkv --damage aa_csm_same.json [--block 32] [--limit N]
      [--emit out.json]

NOT checked: the gate serves a reference up to `max_chain` frames old while every statistic here compares
against t-1; a threshold chosen on this table is a fit until it is re-evaluated on the other half of the clip.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability a positive outranks a negative, ties counted half (Mann-Whitney)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def block_stats(cur: np.ndarray, prev: np.ndarray, block: int) -> tuple[float, float]:
    """The gate's two block statistics on one pair, 8-bit luma levels."""
    diff = cur.astype(np.float32) - prev.astype(np.float32)
    h, w = diff.shape
    hb, wb = h // block, w // block
    d = diff[:hb * block, :wb * block]
    # mean of |diff| per block
    maxabs = np.abs(d).reshape(hb, block, wb, block).mean(axis=(1, 3)).max()
    # |mean(cur) - mean(prev)| per block
    signed = np.abs(d.reshape(hb, block, wb, block).mean(axis=(1, 3))).max()
    return float(maxabs), float(signed)


def mctf_stats(cur: np.ndarray, prev: np.ndarray, flow) -> tuple[float, float, float]:
    """Motion-compensated residual: p99, share above 3 sigma, mean."""
    h, w = cur.shape
    sw, sh = w // 2, h // 2
    a = cv2.resize(prev, (sw, sh), interpolation=cv2.INTER_AREA)
    b = cv2.resize(cur, (sw, sh), interpolation=cv2.INTER_AREA)
    f = flow.calc(a, b, None)                      # displacement in half-res pixels
    grid_x, grid_y = np.meshgrid(np.arange(sw, dtype=np.float32), np.arange(sh, dtype=np.float32))
    warped = cv2.remap(a.astype(np.float32), grid_x + f[..., 0], grid_y + f[..., 1],
                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # residual and its statistics at half resolution: a ranking statistic, not an acceptance number
    r = np.abs(b.astype(np.float32) - warped)
    mad = np.median(np.abs(r - np.median(r)))
    sigma = max(mad / 0.6745, 1e-6)
    return float(np.percentile(r, 99)), float((r > 3 * sigma).mean() * 100.0), float(r.mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--damage", required=True, help="gate_sweep --out json: per-frame share of pixels >8 levels")
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--emit", default="")
    a = ap.parse_args()

    dmg = json.load(open(a.damage))
    rows = dmg["rows"] if isinstance(dmg, dict) else dmg
    share = {int(r["frame"]): float(r["share"]) for r in rows}

    cap = cv2.VideoCapture(a.clip)
    if not cap.isOpened():
        raise SystemExit(f"{a.clip}: cannot open")
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow.setUseSpatialPropagation(True)

    prev, out, idx = None, [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            mx, sg = block_stats(cur, prev, a.block)
            p99, sh, mn = mctf_stats(cur, prev, flow)
            out.append({"frame": idx, "block_maxabs": mx, "block_signed": sg,
                        "mctf_p99": p99, "mctf_share": sh, "mctf_mean": mn,
                        "damage": share.get(idx, 0.0)})
        prev = cur
        idx += 1
        if a.limit and idx >= a.limit:
            break
    cap.release()

    labelled = [r for r in out if r["frame"] in share]
    d = np.array([r["damage"] for r in labelled])
    print(f"clip {os.path.basename(a.clip)}: {idx} frames, {len(labelled)} joined, "
          f"{int((d > 0.1).sum())} above the 0.1% bar")
    keys = ["block_maxabs", "block_signed", "mctf_p99", "mctf_share", "mctf_mean"]
    pos_mask = d > 0.1
    table = {}
    for k in keys:
        v = np.array([r[k] for r in labelled])
        table[k] = {"auc": round(auc(v[pos_mask], v[~pos_mask]), 4),
                    "spearman": round(spearman(v, d), 4),
                    "median_pos": round(float(np.median(v[pos_mask])), 3) if pos_mask.any() else None,
                    "median_neg": round(float(np.median(v[~pos_mask])), 3)}
    print(json.dumps(table, indent=1))
    if a.emit:
        json.dump({"clip": a.clip, "damage": a.damage, "frames": out, "table": table}, open(a.emit, "w"))
        print("wrote", a.emit)
    print("NOT checked: the t-1 reference against the gate's chain, a held-out threshold, and the "
          "temporal-texture criterion the 0.1% share cannot see")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
