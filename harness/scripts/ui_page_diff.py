"""Diff two page-capture sets (stock vs migrated) into the table the migration is judged by.

Pairs files by name, so `nav3.png` in the stock directory is compared with `nav3.png` in the migrated one. For each
page: mean level difference, share of pixels more than 8 levels off, and the four worst row bands - the last one is
what turns "5.69%" into "the threshold row and the two labels above it".

The threshold defaults are the ones the migration is judged by (0.1% is the acceptance bar for a RENDER, not for a
layout; a layout diff is a parity measure, so the budget here is a fraction of the page, not a level of a picture).

Usage:
  python3 harness/scripts/ui_page_diff.py STOCK_DIR MIGRATED_DIR [--budget 0.05] [--json out.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np


def load(path: str) -> np.ndarray:
    import cv2
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read {path}")
    return image


def compare(stock: np.ndarray, migrated: np.ndarray) -> dict:
    if stock.shape != migrated.shape:
        return {"shape_mismatch": [list(stock.shape), list(migrated.shape)]}
    diff = np.abs(stock.astype(np.int16) - migrated.astype(np.int16)).max(axis=2)
    share = float((diff > 8).mean()) * 100.0
    bands: list[tuple[int, float]] = []
    for start in range(0, diff.shape[0], 20):
        bands.append((start, float(diff[start:start + 20].mean())))
    bands.sort(key=lambda item: -item[1])
    return {
        "mean": round(float(diff.mean()), 3),
        "over8_pct": round(share, 3),
        "max": int(diff.max()),
        "worst_bands": [y for y, _ in bands[:4]],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stock")
    ap.add_argument("migrated")
    ap.add_argument("--budget", type=float, default=0.05,
                    help="per-page budget in %% of pixels over 8 levels (default 0.05)")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    stock_files = {os.path.basename(p): p for p in glob.glob(os.path.join(args.stock, "*.png"))}
    migrated_files = {os.path.basename(p): p for p in glob.glob(os.path.join(args.migrated, "*.png"))}
    if not stock_files:
        raise SystemExit(f"no captures in {args.stock}")
    shared = sorted(set(stock_files) & set(migrated_files))
    only_stock = sorted(set(stock_files) - set(migrated_files))
    only_migrated = sorted(set(migrated_files) - set(stock_files))

    rows = []
    failed = []
    for name in shared:
        result = compare(load(stock_files[name]), load(migrated_files[name]))
        rows.append((name, result))
        if result.get("shape_mismatch") or result.get("over8_pct", 100.0) > args.budget:
            failed.append(name)

    width = max((len(name) for name, _ in rows), default=10)
    print(f"{'page'.ljust(width)}  {'mean':>7}  {'>8 %':>7}  {'max':>4}  worst row bands")
    for name, result in rows:
        if result.get("shape_mismatch"):
            print(f"{name.ljust(width)}  shape mismatch {result['shape_mismatch']}")
            continue
        print(f"{name.ljust(width)}  {result['mean']:7.2f}  {result['over8_pct']:7.3f}  {result['max']:4d}  "
              f"{result['worst_bands']}")
    if only_stock:
        print(f"\nmissing in the migrated run: {', '.join(only_stock)}")
    if only_migrated:
        print(f"extra in the migrated run:   {', '.join(only_migrated)}")
    print(f"\n{len(rows) - len(failed)} of {len(rows)} pages inside the {args.budget}% budget"
          + (f"; over: {', '.join(failed)}" if failed else ""))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"pages": dict(rows), "budget": args.budget, "failed": failed}, handle, indent=1)
        print(f"json -> {args.json}")
    return 0 if not failed and not only_stock else 1


if __name__ == "__main__":
    sys.exit(main())
