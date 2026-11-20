"""The Palantir error DAG as a predictor, without the network: err = TC + sum(w * err_ref).

Palantir (arXiv 2408.06152) models the SR error of a patch as its texture complexity plus the weighted sum of the
errors of the patches it references, with the texture term estimated from the LR video alone. That is the shape of
a certificate: a predictor of the SR error that needs no network run. This script builds the predictor on our
material and asks the only question that matters here - does it rank the reuses that actually damage the picture?

  TC(block)   = || upsample(downsample(LR_t)) - LR_t ||   (their Eq. 17 approximation, block mean)
  w(block)    = 1 to the reference block its flow points at (one reference, not three: our clips are not VP9)
  err_t       = TC_t + err_ref[mapped block]              (err = 0 on frames the gate rendered: anchors)
  frame score = mean over blocks of err_t

Usage:
  python3 harness/scripts/reuse_dag.py --clip cut.mkv --depth-log gate.log --damage damage.json [--block 32] [--emit f]
"""
from __future__ import annotations

import argparse
import json
import re

import cv2
import numpy as np


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    v = np.concatenate([pos, neg])
    o = v.argsort()
    r = np.empty(len(v))
    r[o] = np.arange(1, len(v) + 1)
    _, inv, c = np.unique(v, return_inverse=True, return_counts=True)
    s = np.zeros(len(c))
    np.add.at(s, inv, r)
    r = (s / c)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--depth-log", required=True)
    ap.add_argument("--damage", required=True)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--emit", default="")
    a = ap.parse_args()

    depth = {}
    for line in open(a.depth_log, errors="ignore"):
        m = re.search(r"reuse frame=(\d+) depth=(\d+)", line)
        if m:
            depth[int(m.group(1))] = int(m.group(2))
    dmg = {int(r["frame"]): float(r["share"]) for r in json.load(open(a.damage))["rows"]}

    cap = cv2.VideoCapture(a.clip)
    if not cap.isOpened():
        raise SystemExit(f"{a.clip}: cannot open")
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    hist_lr, hist_err, hist_small, rows, i = [], [], [], [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        h, w = g.shape
        hb, wb = h // a.block, w // a.block
        small = cv2.resize(g, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        rt = cv2.resize(cv2.resize(g, (w // 2, h // 2), interpolation=cv2.INTER_AREA), (w, h),
                        interpolation=cv2.INTER_LINEAR)
        tc = np.abs(g.astype(np.float32) - rt.astype(np.float32))[:hb * a.block, :wb * a.block]
        tc = tc.reshape(hb, a.block, wb, a.block).mean(axis=(1, 3))

        k = depth.get(i)
        if k is None:
            err = np.zeros((hb, wb), dtype=np.float32)                  # anchor: assumption #2
        else:
            ref_small = hist_small[-1 - k] if len(hist_small) > k else hist_small[0]
            ref_err = hist_err[-1 - k] if len(hist_err) > k else hist_err[0]
            u = flow.calc(ref_small, small, None)
            ys = (np.arange(hb) * a.block + a.block // 2) // 2
            xs = (np.arange(wb) * a.block + a.block // 2) // 2
            ux = u[ys][:, xs][..., 0] * 2.0 / a.block          # displacement in blocks
            uy = u[ys][:, xs][..., 1] * 2.0 / a.block
            by = np.clip(np.round(np.arange(hb)[:, None] - uy).astype(int), 0, hb - 1)
            bx = np.clip(np.round(np.arange(wb)[None, :] - ux).astype(int), 0, wb - 1)
            err = tc + ref_err[by, bx]
        hist_lr.append(g)
        hist_err.append(err)
        hist_small.append(small)
        if k is not None:
            rows.append({"frame": i, "depth": k, "err": float(err.mean()),
                         "err_max": float(err.max()), "damage": dmg.get(i, 0.0)})
        i += 1
        if len(hist_small) > 8:
            hist_small.pop(0)
            hist_err.pop(0)
            hist_lr.pop(0)
    cap.release()

    labelled = [r for r in rows if r["frame"] in dmg]
    pos = [r for r in labelled if r["damage"] > 0.1]
    neg = [r for r in labelled if r["damage"] <= 0.1]
    out = {"clip": a.clip, "reused": len(labelled), "positives": len(pos),
           "auc_mean_err": round(auc(np.array([r["err"] for r in pos]), np.array([r["err"] for r in neg])), 3),
           "auc_max_err": round(auc(np.array([r["err_max"] for r in pos]), np.array([r["err_max"] for r in neg])), 3)}
    print(json.dumps(out, indent=1))
    if a.emit:
        json.dump({"summary": out, "rows": rows}, open(a.emit, "w"))
        print("wrote", a.emit)
    print("NOT checked: one reference instead of three, flow instead of codec motion vectors, TC from an 8-bit "
          "luma round trip rather than the codec's own residual")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
