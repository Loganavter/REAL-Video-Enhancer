"""Diff two builds widget by widget: the table that answers "which control looks different".

Why this exists: the page table reports a share of the PAGE, and a page is ~800 000 pixels - a button is 7 000 of
them, so a button that lost its icon and clipped its label reads as "1.2% of the page" and drowns next to everything
else that is fine. That is exactly how a visibly wrong button passed as "close" while the owner kept pointing at it.

This script crops every *named* widget out of both screenshots (using the same layout dumps the structural harness
writes) and reports the share of the widget's OWN pixels that differ. A button whose icon is gone scores tens of
percent, not one, and the table names it.

Usage:
  python3 harness/scripts/ui_widget_diff.py DUMPS_STOCK DUMPS_NEW PNG_STOCK PNG_NEW [--budget 2.0] [--limit 25]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np


def _load_png(path: str):
    import cv2

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read {path}")
    return image


def _walk(node, out, ox=0, oy=0):
    """Collect (name, class, absolute rect) for every named node, parent-local geometry summed up the tree."""
    x, y, w, h = node.get("geometry", [0, 0, 0, 0])
    ax, ay = ox + x, oy + y
    if node.get("name"):
        out.append((node["name"], node.get("class", ""), (ax, ay, w, h), bool(node.get("visible"))))
    for child in node.get("children", []):
        _walk(child, out, ax, ay)
    return out


def _share(a, b) -> float:
    if a.shape != b.shape:
        return 100.0
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
    return float((diff > 8).mean()) * 100.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps_stock")
    ap.add_argument("dumps_new")
    ap.add_argument("png_stock")
    ap.add_argument("png_new")
    ap.add_argument("--budget", type=float, default=2.0, help="per-widget budget in %% of its own pixels (default 2)")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    rows = []
    pages = sorted(os.path.basename(path) for path in glob.glob(os.path.join(args.dumps_stock, "*.json")))
    for page in pages:
        stock_json = os.path.join(args.dumps_stock, page)
        new_json = os.path.join(args.dumps_new, page)
        stock_png = os.path.join(args.png_stock, page.replace(".json", ".png"))
        new_png = os.path.join(args.png_new, page.replace(".json", ".png"))
        if not (os.path.exists(new_json) and os.path.exists(stock_png) and os.path.exists(new_png)):
            continue
        stock_img, new_img = _load_png(stock_png), _load_png(new_png)
        widgets = _walk(json.load(open(stock_json, encoding="utf-8"))["root"], [])
        for name, cls, (x, y, w, h), visible in widgets:
            if not visible or w < 4 or h < 4:
                continue
            if x < 0 or y < 0 or x + w > stock_img.shape[1] or y + h > stock_img.shape[0]:
                continue
            share = _share(stock_img[y:y + h, x:x + w], new_img[y:y + h, x:x + w])
            rows.append({"page": page, "widget": name, "class": cls, "rect": [x, y, w, h],
                         "off_pct": round(share, 2)})

    rows.sort(key=lambda row: -row["off_pct"])
    width = max((len(row["widget"]) for row in rows), default=10)
    print(f"{'page'.ljust(14)}  {'widget'.ljust(width)}  {'class'.ljust(14)}  {'size':>13}  {'off %':>6}")
    for row in rows[: args.limit]:
        w, h = row["rect"][2], row["rect"][3]
        print(f"{row['page'][:-5].ljust(14)}  {row['widget'].ljust(width)}  {row['class'][:14].ljust(14)}  "
              f"{f'{w}x{h}':>13}  {row['off_pct']:6.2f}")
    over = [row for row in rows if row["off_pct"] > args.budget]
    print(f"\n{len(rows) - len(over)} of {len(rows)} widgets inside the {args.budget}% budget"
          + (f"; worst: {over[0]['widget']} ({over[0]['off_pct']}%)" if over else ""))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"widgets": rows, "budget": args.budget}, handle, indent=1)
        print(f"json -> {args.json}")
    return 0 if not over else 1


if __name__ == "__main__":
    sys.exit(main())
