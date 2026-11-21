"""Does the shell produce the Designer's window? (W3's whole-window gate, offscreen)

The page gate (`ui_pages_identity.py`) compares pages built one at a time. This one compares the assembled window: the
same top-level chrome, the same five nav pages, the same attribute surface RVE's glue binds.

Two things are checked separately because they fail differently:

* **surface**: every attribute the generated class assigned (476 of them) must exist on the shell-built window -
  otherwise glue written against `Ui_MainWindow` breaks with `AttributeError` at startup, which is the kind of break
  that looks like "the app does not start" rather than "a widget moved";
* **tree**: the widget tree must be the Designer's, node for node.

Usage:
  <app venv>/bin/python harness/scripts/ui_shell_identity.py [--geometry] [--limit 8]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

TOOLKIT_ALIASES = {"QtLabel": "QLabel", "QtButton": "QPushButton", "QtCheckBox": "QCheckBox",
                   "QtComboBox": "QComboBox", "QtLineEdit": "QLineEdit"}
INTERNAL_SUBSTRINGS = ("qt_scrollarea", "qt_tabwidget")


def _diff_module():
    path = os.path.join(_ROOT, "harness", "scripts", "ui_layout_diff.py")
    spec = importlib.util.spec_from_file_location("ui_layout_diff", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _designer_ui():
    """The Designer baseline, from the fixture kept after the .ui was deleted (W3).

    `harness/fixtures/designer_mainwindow.py` is the compiled output the split was made from, so the identity gate still
    has a reference to compare against without the generated file living in the tree.
    """
    path = os.path.join(_ROOT, "harness", "fixtures", "designer_mainwindow.py")
    spec = importlib.util.spec_from_file_location("designer_mainwindow", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.Ui_MainWindow


def _migrated_names() -> set[str]:
    """Widgets the pages build with a toolkit adapter - their private children are the toolkit's, not the Designer's."""
    import re

    names: set[str] = set()
    pages_dir = os.path.join(_ROOT, "src", "ui", "pages")
    for entry in os.listdir(pages_dir):
        if entry.endswith(".py"):
            names |= set(re.findall(r"self\.(\w+) = Qt\w+\(",
                                    open(os.path.join(pages_dir, entry), encoding="utf-8").read()))
    return names


def _dump_pages(ui, out_dir: str, app) -> None:
    """The Designer build's own dumps, page by page - the baseline for a live dump taken by the same serializer.

    The page walker reads `ui.stackedWidget`: in the Designer's shape the widgets live on the `Ui_MainWindow` instance,
    not on the window (the shell's `expose()` is what moves them onto the window in the live app).
    """
    import json

    from src.ui import layout_dump, page_capture

    os.makedirs(out_dir, exist_ok=True)
    steps = page_capture.steps(ui)
    # Two passes: the first lays every page out (a page's children have stale geometry until it has been current),
    # the second writes the dumps. One pass produced a baseline where hidden pages were 100x30 boxes (measured).
    for _name, action in steps:
        action()
        app.processEvents()
    for name, action in steps:
        action()
        app.processEvents()
        target = ui.centralwidget.window()
        page_capture.stabilise(target)
        with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as handle:
            json.dump(layout_dump.dump(target), handle, indent=1, sort_keys=True)
    print(f"baseline: {len(page_capture.steps(ui))} pages")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", action="store_true",
                    help="also compare rectangles; Qt's private scroll/tab internals are excluded and named")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--dump", default="",
                    help="write the Designer build's per-page dumps to this directory (a baseline the live dump can "
                         "be compared against with the same serializer)")
    args = ap.parse_args()

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QMainWindow

    app = QApplication.instance() or QApplication([])
    from src.ui import toolkit_theme
    toolkit_theme.install(app)
    import resources_rc  # noqa: F401
    from src.ui import layout_dump
    from src.ui.shell.window import MainWindowShell

    designer_window = QMainWindow()
    designer = _designer_ui()()
    designer.setupUi(designer_window)
    designer_window.show()
    app.processEvents()

    shell_window = QMainWindow()
    MainWindowShell().build(shell_window)
    shell_window.resize(designer_window.size())
    shell_window.show()
    app.processEvents()

    design_names = {name for name in vars(designer) if not name.startswith("_")}
    shell_names = {name for name in vars(shell_window) if not name.startswith("_")}
    missing = sorted(design_names - shell_names)
    print(f"surface: {len(design_names)} Designer attributes, {len(missing)} missing on the shell window")
    if missing:
        print("   ", missing[: args.limit])

    diff = _diff_module()
    reference = {"root": layout_dump._node(designer_window)}
    diff.prune_under(reference, _migrated_names())
    candidate = {"root": layout_dump._node(shell_window)}
    if args.geometry:
        reference["root"]["geometry"] = list(candidate["root"]["geometry"])
    result = diff.compare(reference, candidate, tolerance_px=0, class_aliases=TOOLKIT_ALIASES)
    internals = [entry for entry in result["geometry"]
                 if any(marker in entry[0] for marker in INTERNAL_SUBSTRINGS)]
    if not args.geometry:
        result["geometry"] = []
    else:
        result["geometry"] = [entry for entry in result["geometry"] if entry not in internals]
    result["clean"] = not (result["missing"] or result["extra"] or result["geometry"] or result["text"]
                           or result["visibility"] or result["state"])
    clean = result["clean"] and not missing
    print(f"tree: nodes {result['nodes'][0]}/{result['nodes'][1]} missing {len(result['missing'])} "
          f"extra {len(result['extra'])} geom {len(result['geometry'])} text {len(result['text'])} "
          f"vis {len(result['visibility'])} state {len(result['state'])}")
    for path in result["missing"][: args.limit]:
        print(f"    missing {path}")
    for path in result["extra"][: args.limit]:
        print(f"    extra   {path}")
    for path, ga, gb in result["geometry"][: args.limit]:
        print(f"    geom    {path} {ga} -> {gb}")
    for path, ta, tb in result["text"][: args.limit]:
        print(f"    text    {path} {ta!r} -> {tb!r}")
    for path, va, vb in result["visibility"][: args.limit]:
        print(f"    visible {path} {va} -> {vb}")
    for path, sa, sb in result["state"][: args.limit]:
        print(f"    state   {path} {sa} -> {sb}")
    if internals:
        print(f"    (qt-internal rectangles not compared: {len(internals)})")
    if args.dump:
        # Last, because the walk moves the Designer window from page to page: done before the comparison it would
        # leave the two windows showing different pages and every visibility flag would read as a difference.
        _dump_pages(designer, args.dump, app)
        print(f"dumped the Designer build's pages -> {args.dump}")
    print("\nOK" if clean else "\nDIFF")
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
