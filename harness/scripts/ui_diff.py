"""Compare two builds of the interface: pages for the summary, widgets for the verdict.

One command, one model (`harness/ui_compare`). The page share is how a whole-screen regression is caught; the widget
table is how a control that lost its icon, its colour or its font is caught - the thing the first version of this
harness could not see.

**The page gate is a ratchet, per page.** A single flat `--budget-page 0.05` was a gate that could not be green: the
Process page re-probes the machine on every boot, so two captures of the SAME tree differ there by several percent
(`src/ui/page_capture.py` measured `nav2-tab0` at 3.58 and 5.93 with an identical tree), while every other page sat at
0.00 in the 2026-11-21 shell run (`harness/docs/wsp/verified/logs/ui-shell-pixels-2026-11-21.json`). Charging both the
same tolerance makes the gate red on day one, and a red gate is a gate nobody reads. `PAGE_BASELINE` records each page's
own tolerance, `compare` fails only when a page **grows past its recorded number**, and `--record-baseline` rewrites the
table under the same guard the other ratchets use: it refuses a page that would grow and names it.

Usage:
  python3 harness/scripts/ui_diff.py compare \
      --stock-dumps /tmp/opencode/lay-a --new-dumps /tmp/opencode/lay-b \
      --stock-png   /tmp/opencode/cap-a --new-png   /tmp/opencode/cap-b \
      [--views widgets|pages|both] [--budget-widget 2.0] [--budget-page 0.05] \
      [--no-baseline] [--record-baseline] \
      [--class-alias QtLabel=QLabel ...] [--prune-under NAME ...] [--ignore-text SUBSTR ...] [--json out.json]

`--budget-page` is the fallback for a page `PAGE_BASELINE` never named (a new page): it is not the gate's default
tolerance, it is the tolerance of a page nobody has measured yet.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from harness.ui_compare import (  # noqa: E402
    current_page_root, flatten, format_page_table, format_widget_table, load, page_deltas, parse_aliases, prune_under,
    widget_deltas,
)

#: The page gate's floor: a page measured at 0.00 records 0.05, not 0.00. Antialiasing and a different GPU/driver pair
#: are not a regression, and a zero baseline would charge them as one on the next machine. It is also the value the
#: 2026-11-21 shell run met with 0.00 on every page, so the floor is measured, not guessed.
BASELINE_FLOOR = 0.05

#: Per-page pixel tolerance, in percent of the page, keyed by the page stem `ui_diff` prints (`nav0.png`, ...). The
#: keys are spelled with `.png` because that is how the captures and the committed evidence name a page; the lookup
#: matches on the stem, so a row keyed `nav0.json` (the dump it was read from) finds the same entry.
#:
#: Where each number comes from - no entry is invented:
#:
#: * `nav2.png` / `nav2-tab0.png` are the same page (the Process page is the only page with a sub-tab). Its start-up
#:   probe picks the encoder, and two boots of the SAME tree measured 3.58 and 5.93 there (`src/ui/page_capture.py`).
#:   The state pin (`RVE_UI_STABLE_STATE=1`) narrows that, but it is a measurement, not a guarantee, so the tolerance
#:   is the documented worst case rounded up to 6.0.
#: * every other page records `BASELINE_FLOOR`: the 2026-11-21 shell run measured 0.00 on all thirteen pages at a 0.05
#:   budget (`harness/docs/wsp/verified/logs/ui-shell-pixels-2026-11-21.json`, `failed: []`). The 0.05 is what that run
#:   already met, so the floor is measured too - and it leaves room for a driver's antialiasing, which a zero-tolerance
#:   entry would charge as a regression on the next machine.
#:
#: The same 2026-11-21 tree also has a *page-by-page* run with larger numbers (nav0 0.168, nav3-tab0 3.137,
#: `ui-page-diff-2026-11-21.json`). Those are not this table's baseline: that comparison was against the Designer's
#: compiled pages, and the Designer stopped being the reference (its fixture is deleted). Returning a number from it
#: would re-baseline a comparison this gate no longer runs.
#:
#: **It only shrinks.** A page that gets cheaper lowers its entry in the same change; a page that grows never does -
#: `--record-baseline` refuses and names it, exactly like `ui_gates.py --rekey`. Re-record after a deliberate change to
#: the page, by lowering the entry first. An empty table is a gate that accepts anything, so `compare` treats a missing
#: entry as the `--budget-page` fallback and says so.

PAGE_BASELINE: dict[str, float] = {
    "nav0.png": 0.05,
    "nav1.png": 0.05,
    "nav2-tab0.png": 6.0,
    "nav2-tab1.png": 0.05,
    "nav2-tab2.png": 0.05,
    "nav2-tab3.png": 0.05,
    "nav2-tab4.png": 0.05,
    "nav2.png": 6.0,
    "nav3-tab0.png": 0.05,
    "nav3-tab1.png": 0.05,
    "nav3-tab2.png": 0.05,
    "nav3.png": 0.05,
    "nav4.png": 0.05,
}


def _image(path: str):
    """Read a PNG as a BGR array. `cv2` on the app host, Pillow on the portable one (where `cv2` is absent).

    Both loaders are used only through this function, so a comparison never mixes channel orders: the metric is a
    per-channel maximum, which is order-free as long as both sides arrive the same way.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        import cv2
    except ImportError:
        from PIL import Image
        import numpy as np

        with Image.open(path) as handle:
            rgb = np.asarray(handle.convert("RGB"))
        return rgb[:, :, ::-1].copy()                   # RGB -> BGR, the array layout the rest of this file assumes
    return cv2.imread(path, cv2.IMREAD_COLOR)


def _page_pixels(stock_img, new_img, mask_rects=()) -> dict:
    import numpy as np

    if stock_img is None or new_img is None or stock_img.shape != new_img.shape:
        return {"mean": float("nan"), "over8_pct": float("nan"), "max": 0, "shape_mismatch": True}
    stock = stock_img
    new = new_img
    if mask_rects:
        # Panes whose CONTENT is environment, not chrome (the release notes render the bundled CHANGELOG.md through
        # markdown, the system-info pane prints the host's runtime). The structural dump already freezes them
        # (RVE_UI_STABLE_TEXT); without the same mask here the pixel gate reports the same file rendered from a
        # different bundle as a chrome regression - measured: nav0 3.12 % where every row above the pane was identical.
        stock = stock_img.copy()
        new = new_img.copy()
        for x, y, w, h in mask_rects:
            stock[max(0, y):y + h, max(0, x):x + w] = 0
            new[max(0, y):y + h, max(0, x):x + w] = 0
    diff = np.abs(stock.astype(np.int16) - new.astype(np.int16)).max(axis=2)
    return {"mean": round(float(diff.mean()), 3), "over8_pct": round(float((diff > 8).mean()) * 100.0, 3),
            "max": int(diff.max())}


def _rects_for(tree: dict, names: set[str]) -> list[tuple[int, int, int, int]]:
    """Widget rectangles by object name, in page coordinates (the dump's geometry space)."""
    rects = []
    for path, node in flatten(tree).items():
        if path.rsplit("/", 1)[-1] in names:
            geometry = node.get("geometry")
            if geometry and len(geometry) == 4:
                rects.append(tuple(int(value) for value in geometry))
    return rects


# --- the page ratchet -------------------------------------------------------------------------------------

def _stem(page: str) -> str:
    """`nav0.json` and `nav0.png` are the same page: a row is keyed by the dump it came from, the table by the image."""
    return page.rsplit(".", 1)[0] if "." in page.rsplit("/", 1)[-1] else page


def _page_key(page: str) -> str:
    """The table's key for a page row: `nav0.json` -> `nav0.png`. One page, one entry, whichever side named it."""
    return f"{_stem(page)}.png"


def recorded_budget(page: str, baseline: dict[str, float] | None) -> float | None:
    """The tolerance `baseline` records for `page`, or None when the table never named it."""
    if baseline is None:
        return None
    stem = _stem(page)
    for key in (page, f"{stem}.png", f"{stem}.json"):
        if key in baseline:
            return baseline[key]
    return None


def page_budget(page: str, baseline: dict[str, float] | None, fallback: float) -> float:
    """The tolerance one page is compared against: its recorded entry, or the fallback for an unmeasured page."""
    recorded = recorded_budget(page, baseline)
    return fallback if recorded is None else recorded


def over_budget(page_rows: list[dict], baseline: dict[str, float] | None,
                fallback: float) -> list[tuple[str, float, float]]:
    """`(page, measured, budget)` for every page that grew past its own recorded number.

    A page whose image was missing or whose shape did not match has `over8_pct` `nan`; it is NOT listed here - it is
    the `uncompared` list's job, and a comparison that never ran must never read as "inside the budget". Callers fail
    on `uncompared` separately (measured 2026-11-22: a `nan` page printed "inside the budget").
    """
    grown: list[tuple[str, float, float]] = []
    for row in page_rows:
        value = row.get("over8_pct")
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            continue
        budget = page_budget(row["page"], baseline, fallback)
        if value > budget:
            grown.append((_page_key(row["page"]), float(value), budget))
    return grown


def record_baseline(page_rows: list[dict], baseline: dict[str, float] | None,
                    fallback: float, floor: float = BASELINE_FLOOR) -> tuple[dict[str, float], list[tuple[str, float, float]]]:
    """The table `--record-baseline` would write, and every page that would GROW doing it.

    The guard is per page, not per total - the same lesson as `ui_gates.py --rekey`: a fix on one page plus a
    regression on another leaves the total unchanged while the baseline silently absorbs the regression. A page the
    table never named is compared against `fallback`, so its first measurement is growth exactly like any other.

    The written value is the measurement floored at `floor`: a page measured at 0.00 keeps the harness's own noise
    allowance instead of becoming a zero-tolerance pin that fails on the next driver.
    """
    current = dict(baseline or {})
    grown: list[tuple[str, float, float]] = []
    recorded: dict[str, float] = {}
    for row in page_rows:
        value = row.get("over8_pct")
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            continue                     # no image: a page that was never compared cannot be recorded
        page = _page_key(row["page"])
        budget = page_budget(page, baseline, fallback)
        if value > budget:
            grown.append((page, float(value), budget))
            continue
        recorded[page] = max(round(float(value), 3), floor)
    if grown:
        return current, grown
    return {**current, **recorded}, []


def _baseline_literal(baseline: dict[str, float]) -> str:
    lines = ["PAGE_BASELINE: dict[str, float] = {"]
    lines.extend(f"    {page!r}: {value:g}," for page, value in sorted(baseline.items()))
    lines.append("}")
    return "\n".join(lines)


def rewrite_baseline(path: str | os.PathLike[str], baseline: dict[str, float]) -> bool:
    """Replace the `PAGE_BASELINE` literal in `path` with `baseline`. False when the block is not in the expected shape."""
    text = Path(path).read_text(encoding="utf-8")
    pattern = re.compile(r"PAGE_BASELINE: dict\[str, float\] = \{.*?\n\}", re.S)
    if not pattern.search(text):
        return False
    Path(path).write_text(pattern.sub(_baseline_literal(baseline), text, count=1), encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["compare"])
    ap.add_argument("--stock-dumps", required=True)
    ap.add_argument("--new-dumps", required=True)
    ap.add_argument("--stock-png", default="")
    ap.add_argument("--new-png", default="")
    ap.add_argument("--views", choices=["widgets", "pages", "both"], default="both")
    ap.add_argument("--budget-widget", type=float, default=2.0, help="per-widget ink budget, %% of the widget")
    ap.add_argument("--budget-page", type=float, default=0.05,
                    help="per-page pixel budget for a page PAGE_BASELINE never named (%% of the page)")
    ap.add_argument("--no-baseline", action="store_true",
                    help="judge every page against --budget-page; for an A/B run where the numbers are the point, "
                         "not the verdict")
    ap.add_argument("--record-baseline", action="store_true",
                    help="rewrite PAGE_BASELINE from this run; refuses (exit 2) if any page would grow past its entry")
    ap.add_argument("--pages", default="",
                    help="comma-separated page stems to compare (default: every page in the stock dumps). Useful with "
                         "the live capture server, which only re-captures the pages a change touched")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--class-alias", action="append", default=[], metavar="NEW=OLD")
    ap.add_argument("--prune-under", action="append", default=[], metavar="NAME")
    ap.add_argument("--ignore-text", action="append", default=[])
    ap.add_argument("--mask-widgets", default="",
                    help="comma-separated object names whose CONTENT is environment, not chrome (release notes, "
                         "system info): their rectangles are blanked in both images for the page metric, the same "
                         "freeze RVE_UI_STABLE_TEXT applies to the structural side")
    ap.add_argument("--include-hidden", action="store_true",
                    help="compare widgets that are hidden on either side; by default only widgets visible in BOTH "
                         "builds count, because a hidden widget keeps stale geometry and its page may simply not be "
                         "the current one in one of the dumps")
    ap.add_argument("--no-dedupe", action="store_true",
                    help="keep one row per (page, widget); by default a widget is reported once, on the page where "
                         "it differs most (every dump contains the whole window, so one widget appears 13 times)")
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)

    baseline = None if args.no_baseline else dict(PAGE_BASELINE)
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
    missing_pages: list[str] = []
    uncompared: list[str] = []
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
        # A capture writes the dump and the PNG into ONE directory, so an omitted `--*-png` means "next to the dumps",
        # not "no pixels". Treating a missing directory as None made the page table print `nan` and still report
        # "inside the budget" - a green verdict on a comparison that never ran (measured 2026-11-22).
        stock_png = args.stock_png or args.stock_dumps
        new_png = args.new_png or args.new_dumps
        stock_img = _image(os.path.join(stock_png, name.replace(".json", ".png")))
        new_img = _image(os.path.join(new_png, name.replace(".json", ".png")))
        if stock_img is None or new_img is None:
            uncompared.append(name)
        page_root = current_page_root(stock_tree)
        rows = widget_deltas(name, stock_tree, new_tree, stock_img, new_img, aliases, args.ignore_text)
        if page_root:
            marker = f"/{page_root}/"
            rows = [row for row in rows if marker in row.path or row.path.endswith(f"/{page_root}")]
        for row in rows:
            row.page = name
        widgets_all.extend(rows)
        mask_rects = []
        if args.mask_widgets:
            names = {part.strip() for part in args.mask_widgets.split(",") if part.strip()}
            mask_rects = _rects_for(new_tree, names) + _rects_for(stock_tree, names)
        summary = _page_pixels(stock_img, new_img, mask_rects)
        if summary.get("shape_mismatch") and name not in uncompared:
            uncompared.append(name)
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

    if args.record_baseline:
        if baseline is None:
            print("REFUSED: --record-baseline cannot run with --no-baseline (there is no table to guard)")
            return 2
        recorded, grown = record_baseline(page_rows, baseline, args.budget_page)
        if grown:
            for page, measured, budget in grown:
                print(f"REFUSED: {page} would grow {budget:g} -> {measured:g}: retire the regression, "
                      "do not re-baseline it")
            return 2
        if not rewrite_baseline(__file__, recorded):
            print("REFUSED: the PAGE_BASELINE block is not in the expected shape")
            return 2
        measured_count = sum(1 for row in page_rows
                             if isinstance(row.get("over8_pct"), (int, float))
                             and math.isfinite(row["over8_pct"]))
        print(f"recorded {measured_count} page(s) from this run; the table only shrinks from here")
        return 0

    widgets_all.sort(key=lambda row: -row.severity)
    exit_code = 0
    if args.views in {"widgets", "both"}:
        print(format_widget_table(widgets_all, args.limit, args.budget_widget))
        print()
    grown_rows: list[tuple[str, float, float]] = []
    if args.views in {"pages", "both"}:
        print(format_page_table(page_rows))
        grown_rows = [entry for entry in over_budget(page_rows, baseline, args.budget_page)
                      if entry[0] not in uncompared]
        compared = [row for row in page_rows if row.get("page") not in uncompared]
        unrecorded = sorted(row["page"] for row in compared
                            if baseline is not None and recorded_budget(row["page"], baseline) is None)
        print(f"\n{len(compared) - len(grown_rows)} of {len(page_rows)} pages inside their recorded budget"
              + (f"; over: {', '.join(page for page, *_ in grown_rows)}" if grown_rows else "")
              + (f"; NOT COMPARED (no image): {', '.join(uncompared)}" if uncompared else ""))
        if unrecorded:
            print(f"no recorded entry (judged at the {args.budget_page:g}% fallback): {', '.join(unrecorded)}")
        for page, measured, budget in grown_rows:
            print(f"  {page}: {measured:g}% > {budget:g}% - the page grew; fix it or lower the entry deliberately")
        print("\ntree: " + "  ".join(
            f"{row['page'][:-5]} nodes {row['nodes'][0]}/{row['nodes'][1]} geom {row['geometry']} "
            f"icon {row['icon']}" for row in tree_rows[:4]) + ("" if len(tree_rows) <= 4 else "  ..."))
        if grown_rows or missing_pages or uncompared:
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
                "page_baseline": baseline,
                "over": [{"page": page, "measured": measured, "budget": budget}
                         for page, measured, budget in grown_rows],
            }, handle, indent=1, default=list)
        print(f"\njson -> {args.json}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
