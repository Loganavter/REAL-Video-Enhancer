"""Diff two layout dumps (stock vs rebuilt) into the structural table a code rebuild is judged by.

Why not only pixels: `ui_page_diff.py` measures what the screen looks like; this one measures what the interface
*is* - which widgets exist, where they sit, what they say, whether they show. A page rebuilt in code aims at the same
tree, and "the screenshot is close" is not the same statement as "label_20 is at the same place with the same text".

Pairing: files by name (`nav3-tab0.json` vs `nav3-tab0.json`), nodes by their path through the tree - ancestor object
names, with `~class[index]` where a widget has no name (the Designer names nearly everything, and RVE's glue binds by
name, so a rebuild keeps them). A node present on one side only is reported as missing/extra, which is how a runtime
swap shows up: it hides the stock widget and inserts a second one in the same slot.

Usage:
  python3 harness/scripts/ui_layout_diff.py STOCK_DIR REBUILT_DIR [--tolerance-px 0] [--limit 8] [--json out.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _name(node: dict, index: int) -> str:
    name = node.get("name") or ""
    return name if name else f"~{node.get('class', '?')}[{index}]"


def flatten(tree: dict) -> dict[str, dict]:
    """path -> node, path built from names so two runs of the same build produce the same keys."""
    flat: dict[str, dict] = {}

    def walk(node: dict, path: str, index: int) -> None:
        key = f"{path}/{_name(node, index)}" if path else _name(node, index)
        if key in flat:            # two unnamed siblings of the same class: disambiguate by position
            suffix = 2
            while f"{key}#{suffix}" in flat:
                suffix += 1
            key = f"{key}#{suffix}"
        flat[key] = node
        for child_index, child in enumerate(node.get("children", [])):
            walk(child, key, child_index)

    walk(tree.get("root", {}), "", 0)
    return flat


def _state(node: dict) -> dict:
    return {k: node[k] for k in ("currentIndex", "value", "checked") if k in node}


def compare(stock: dict, rebuilt: dict, tolerance_px: int, ignore_text: list[str] | None = None) -> dict:
    ignore_text = ignore_text or []
    left, right = flatten(stock), flatten(rebuilt)
    missing = sorted(set(left) - set(right))
    extra = sorted(set(right) - set(left))
    geometry: list[tuple[str, list[int], list[int]]] = []
    text: list[tuple[str, str, str]] = []
    state: list[tuple[str, dict, dict]] = []
    visibility: list[tuple[str, bool, bool]] = []
    ignored_text = 0
    for path in sorted(set(left) & set(right)):
        a, b = left[path], right[path]
        if a.get("geometry") != b.get("geometry"):
            ga, gb = a.get("geometry", [0, 0, 0, 0]), b.get("geometry", [0, 0, 0, 0])
            if max(abs(x - y) for x, y in zip(ga, gb)) > tolerance_px:
                geometry.append((path, ga, gb))
        if a.get("text", "") != b.get("text", ""):
            if any(pattern in path for pattern in ignore_text):
                ignored_text += 1
            else:
                text.append((path, a.get("text", ""), b.get("text", "")))
        if a.get("visible") != b.get("visible"):
            visibility.append((path, bool(a.get("visible")), bool(b.get("visible"))))
        if _state(a) != _state(b):
            state.append((path, _state(a), _state(b)))
    return {
        "nodes": [len(left), len(right)],
        "missing": missing,
        "extra": extra,
        "geometry": geometry,
        "text": text,
        "visibility": visibility,
        "state": state,
        "ignored_text": ignored_text,
        "clean": not (missing or extra or geometry or text or visibility or state),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stock")
    ap.add_argument("rebuilt")
    ap.add_argument("--tolerance-px", type=int, default=0,
                    help="allowed geometry delta per coordinate, in px (default 0: a rebuild matches exactly)")
    ap.add_argument("--limit", type=int, default=8, help="mismatch lines printed per category (default 8)")
    ap.add_argument("--ignore-text", action="append", default=[],
                    help="skip text comparison for paths containing this substring (repeatable; e.g. a label that "
                         "embeds free disk space)")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    stock_files = {os.path.basename(p): p for p in glob.glob(os.path.join(args.stock, "*.json"))}
    rebuilt_files = {os.path.basename(p): p for p in glob.glob(os.path.join(args.rebuilt, "*.json"))}
    if not stock_files:
        raise SystemExit(f"no dumps in {args.stock}")
    shared = sorted(set(stock_files) & set(rebuilt_files))
    only_stock = sorted(set(stock_files) - set(rebuilt_files))
    only_rebuilt = sorted(set(rebuilt_files) - set(stock_files))

    results: dict[str, dict] = {}
    failed: list[str] = []
    width = max((len(name) for name in shared), default=10)
    print(f"{'page'.ljust(width)}  {'nodes':>11}  {'miss':>4}  {'extra':>5}  {'geom':>4}  {'text':>4}  {'vis':>3}  {'state':>5}")
    for name in shared:
        result = compare(load(stock_files[name]), load(rebuilt_files[name]), args.tolerance_px, args.ignore_text)
        results[name] = result
        print(f"{name.ljust(width)}  {result['nodes'][0]:5d}/{result['nodes'][1]:<5d}  "
              f"{len(result['missing']):4d}  {len(result['extra']):5d}  {len(result['geometry']):4d}  "
              f"{len(result['text']):4d}  {len(result['visibility']):3d}  {len(result['state']):5d}")
        if not result["clean"]:
            failed.append(name)
            shown = 0
            for path in result["missing"]:
                if shown >= args.limit:
                    break
                print(f"    missing  {path}")
                shown += 1
            for path in result["extra"]:
                if shown >= args.limit:
                    break
                print(f"    extra    {path}")
                shown += 1
            for path, ga, gb in result["geometry"][: args.limit]:
                delta = [b - a for a, b in zip(ga, gb)]
                print(f"    geom     {path}  {ga} -> {gb}  delta {delta}")
            for path, ta, tb in result["text"][: args.limit]:
                print(f"    text     {path}  {ta!r} -> {tb!r}")
            for path, va, vb in result["visibility"][: args.limit]:
                print(f"    visible  {path}  {va} -> {vb}")
            for path, sa, sb in result["state"][: args.limit]:
                print(f"    state    {path}  {sa} -> {sb}")

    if only_stock:
        print(f"\nmissing in the rebuilt run: {', '.join(only_stock)}")
    if only_rebuilt:
        print(f"\nextra in the rebuilt run:   {', '.join(only_rebuilt)}")
    print(f"\n{len(shared) - len(failed)} of {len(shared)} pages structurally identical"
          + (f"; differing: {', '.join(failed)}" if failed else ""))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"pages": results, "tolerance_px": args.tolerance_px, "failed": failed}, handle, indent=1)
        print(f"json -> {args.json}")
    return 0 if not failed and not only_stock else 1


if __name__ == "__main__":
    sys.exit(main())
