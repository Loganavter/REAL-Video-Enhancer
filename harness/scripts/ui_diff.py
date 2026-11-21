"""Compare two builds of the interface: pages for the summary, widgets for the verdict.

One command, one model (`harness/ui_compare`). The page share is how a whole-screen regression is caught; the widget
table is how a control that lost its icon, its colour or its font is caught - the thing the first version of this
harness could not see.

Usage:
  python3 harness/scripts/ui_diff.py compare \
      --stock-dumps /tmp/opencode/lay-designer --new-dumps /tmp/opencode/lay-w4n \
      --stock-png   /tmp/opencode/cap-stock-4   --new-png   /tmp/opencode/cap-w4n \
      [--views widgets|pages|both] [--budget-widget 2.0] [--budget-page 0.05] \
      [--class-alias QtLabel=QLabel ...] [--prune-under NAME ...] [--ignore-text SUBSTR ...] [--json out.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from harness.ui_compare import (  # noqa: E402
    current_page_root, flatten, format_page_table, format_widget_table, load, page_deltas, parse_aliases, prune_under,
    widget_deltas,
)


def _image(path: str):
    if not path or not os.path.exists(path):
        return None
    import cv2

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    return image


def _page_pixels(stock_img, new_img) -> dict:
    import numpy as np

    if stock_img is None or new_img is None or stock_img.shape != new_img.shape:
        return {"mean": float("nan"), "over8_pct": float("nan"), "max": 0, "shape_mismatch": True}
    diff = np.abs(stock_img.astype(np.int16) - new_img.astype(np.int16)).max(axis=2)
    return {"mean": round(float(diff.mean()), 3), "over8_pct": round(float((diff > 8).mean()) * 100.0, 3),
            "max": int(diff.max())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["compare"])
    ap.add_argument("--stock-dumps", required=True)
    ap.add_argument("--new-dumps", required=True)
    ap.add_argument("--stock-png", default="")
    ap.add_argument("--new-png", default="")
    ap.add_argument("--views", choices=["widgets", "pages", "both"], default="both")
    ap.add_argument("--budget-widget", type=float, default=2.0, help="per-widget ink budget, %% of the widget")
    ap.add_argument("--budget-page", type=float, default=0.05, help="per-page pixel budget, %% of the page")
    ap.add_argument("--pages", default="",
                    help="comma-separated page stems to compare (default: every page in the stock dumps). Useful with "
                         "the live capture server, which only re-captures the pages a change touched")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--class-alias", action="append", default=[], metavar="NEW=OLD")
    ap.add_argument("--prune-under", action="append", default=[], metavar="NAME")
    ap.add_argument("--ignore-text", action="append", default=[])
    ap.add_argument("--include-hidden", action="store_true",
                    help="compare widgets that are hidden on either side; by default only widgets visible in BOTH "
                         "builds count, because a hidden widget keeps stale geometry and its page may simply not be "
                         "the current one in one of the dumps")
    ap.add_argument("--no-dedupe", action="store_true",
                    help="keep one row per (page, widget); by default a widget is reported once, on the page where "
                         "it differs most (every dump contains the whole window, so one widget appears 13 times)")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    aliases = parse_aliases(args.class_alias)
    pages = sorted(os.path.basename(path) for path in glob.glob(os.path.join(args.stock_dumps, "*.json")))
    if args.pages:
        wanted = {part.strip() for part in args.pages.split(",") if part.strip()}
        pages = [page for page in pages if page.replace(".json", "") in wanted]
    if not pages:
        raise SystemExit(f"no dumps in {args.stock_dumps}")

    widgets_all = []
    page_rows = []
    tree_rows = []
    missing_pages = []
    for name in pages:
        stock_path = os.path.join(args.stock_dumps, name)
        new_path = os.path.join(args.new_dumps, name)
        if not os.path.exists(new_path):
            missing_pages.append(name)
            continue
        stock_tree = load(stock_path)
        new_tree = load(new_path)
        if args.prune_under:
            prune_under(stock_tree, set(args.prune_under))
        stock_img = _image(os.path.join(args.stock_png, name.replace(".json", ".png"))) if args.stock_png else None
        new_img = _image(os.path.join(args.new_png, name.replace(".json", ".png"))) if args.new_png else None
        page_root = current_page_root(stock_tree)
        rows = widget_deltas(name, stock_tree, new_tree, stock_img, new_img, aliases, args.ignore_text)
        if page_root:
            marker = f"/{page_root}/"
            rows = [row for row in rows if marker in row.path or row.path.endswith(f"/{page_root}")]
        for row in rows:
            row.page = name
        widgets_all.extend(rows)
        summary = _page_pixels(stock_img, new_img)
        summary["page"] = name
        page_rows.append(summary)
        tree_rows.append(page_deltas(name, stock_tree, new_tree, aliases))

    if not args.include_hidden:
        widgets_all = [row for row in widgets_all if not row.visibility]
    if not args.no_dedupe:
        best: dict[str, object] = {}
        for row in widgets_all:
            current = best.get(row.path.rsplit("/", 1)[-1])
            if current is None or row.severity > current.severity:
                best[row.path.rsplit("/", 1)[-1]] = row
        widgets_all = list(best.values())
    widgets_all.sort(key=lambda row: -row.severity)
    exit_code = 0
    if args.views in {"widgets", "both"}:
        print(format_widget_table(widgets_all, args.limit, args.budget_widget))
        print()
    if args.views in {"pages", "both"}:
        print(format_page_table(page_rows))
        over = [row for row in page_rows if row.get("over8_pct", 100.0) > args.budget_page]
        print(f"\n{len(page_rows) - len(over)} of {len(page_rows)} pages inside the {args.budget_page}% budget"
              + (f"; over: {', '.join(row['page'] for row in over)}" if over else ""))
        print("\ntree: " + "  ".join(
            f"{row['page'][:-5]} nodes {row['nodes'][0]}/{row['nodes'][1]} geom {row['geometry']} "
            f"icon {row['icon']}" for row in tree_rows[:4]) + ("" if len(tree_rows) <= 4 else "  ..."))
        if over or missing_pages:
            exit_code = 1
    if missing_pages:
        print(f"\nmissing in the new run: {', '.join(missing_pages)}")
        exit_code = 1
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({
                "widgets": [row.__dict__ for row in widgets_all],
                "pages": page_rows,
                "trees": tree_rows,
                "budget_widget": args.budget_widget,
                "budget_page": args.budget_page,
            }, handle, indent=1, default=list)
        print(f"\njson -> {args.json}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
