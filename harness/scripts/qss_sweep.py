"""Which QSS rules in a page's stylesheet can go, because no stock widget they match is left.

Layer C deletes stylesheets page by page, and the dangerous direction is deleting a rule that still paints a stock
widget. This reports, per page: the stylesheet owner, every selector in it, and whether anything in that page still
builds a stock class the selector can match. Read-only - it prints, it never edits.

Usage:
  python3 harness/scripts/qss_sweep.py [--page tab_2] [--json out.json]
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAGES = os.path.join(_ROOT, "src", "ui", "pages")

#: Selector class -> the stock classes a page can still build. A toolkit adapter (`Qt*`) matches none of them.
STOCK = {"QLabel", "QPushButton", "QCheckBox", "QComboBox", "QLineEdit", "QSpinBox", "QDoubleSpinBox",
         "QProgressBar", "QListWidget", "QTextEdit", "QScrollArea", "QTabWidget", "QSlider", "QRadioButton",
         "QFrame", "QWidget"}
ADAPTER = re.compile(r"^Qt[A-Z]")


def _built(path: str) -> dict[str, str]:
    tree = ast.parse(open(path, encoding="utf-8").read())
    built: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            target = node.targets[0]
            if isinstance(target, ast.Attribute) and getattr(target.value, "id", "") == "self":
                func = node.value.func
                built[target.attr] = func.id if isinstance(func, ast.Name) else "?"
    return built


def _stylesheets(path: str) -> list[tuple[str, str]]:
    tree = ast.parse(open(path, encoding="utf-8").read())
    found: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "setStyleSheet" or not node.args:
            continue
        owner = node.func.value
        if not (isinstance(owner, ast.Attribute) and getattr(owner.value, "id", "") == "self"):
            continue
        css = node.args[0].value if isinstance(node.args[0], ast.Constant) else ""
        if isinstance(css, str) and css.strip():
            found.append((owner.attr, css))
    return found


def _selectors(css: str) -> list[str]:
    return [match.group(1).strip() for match in re.finditer(r"([A-Za-z#][\w:#.\- ]*?)\s*\{", css)]


def _descendants(page: str, seen: set[str] | None = None) -> set[str]:
    """The page plus every page module it imports from `src.ui.pages`.

    A page's stylesheet applies to its whole subtree, and a page holds its child pages as objects built by their own
    modules (`settingsPage` hosts `tab_2/3/4`). Judging a rule by the parent's own constructions alone would delete a
    `QScrollArea` rule while a child tab still has a stock scroll area.
    """
    seen = seen if seen is not None else set()
    path = os.path.join(PAGES, f"{page}.py")
    if page in seen or not os.path.exists(path):
        return seen
    seen.add(page)
    source = open(path, encoding="utf-8").read()
    for match in re.finditer(r"from src\.ui\.pages\.(\w+) import", source):
        _descendants(match.group(1), seen)
    return seen


def analyse(page: str) -> dict:
    path = os.path.join(PAGES, f"{page}.py")
    built: dict[str, str] = {}
    for module in _descendants(page):
        built.update(_built(os.path.join(PAGES, f"{module}.py")))
    adapters = {name for name, cls in built.items() if ADAPTER.match(cls)}
    stock = {cls for cls in built.values() if cls in STOCK}
    rows = []
    for owner, css in _stylesheets(path):
        for selector in _selectors(css):
            head = re.split(r"[:#.\s]", selector.strip())[0]
            if selector.strip().startswith("#"):
                # an #id selector matches one widget by name: safe to keep only if that widget is stock
                name = selector.strip().lstrip("#")
                matched = "stock" if built.get(name, "") in STOCK else "toolkit/unknown"
                rows.append({"owner": owner, "selector": selector, "match": matched})
                continue
            if head in adapters:
                rows.append({"owner": owner, "selector": selector, "match": "toolkit"})
            elif head in stock:
                rows.append({"owner": owner, "selector": selector, "match": "stock"})
            else:
                rows.append({"owner": owner, "selector": selector, "match": "unmatched"})
    deletable = [row for row in rows if row["match"] in {"toolkit", "unmatched"}]
    return {"page": page, "stylesheets": len(_stylesheets(path)), "rules": len(rows),
            "deletable": len(deletable), "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    pages = [args.page] if args.page else sorted(
        name[:-3] for name in os.listdir(PAGES) if name.endswith(".py") and name != "__init__.py")
    results = [analyse(page) for page in pages]
    print(f"{'page'.ljust(24)} {'rules':>5} {'can go':>6}")
    for result in results:
        print(f"{result['page'].ljust(24)} {result['rules']:5d} {result['deletable']:6d}")
        for row in result["rows"]:
            if row["match"] in {"toolkit", "unmatched"}:
                print(f"    {row['match']:9s} {row['selector'][:60]:60s} (on {row['owner']})")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=1)
        print(f"json -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
