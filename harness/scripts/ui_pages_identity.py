"""Do the split-out page modules build the same tree as the Designer's own pages? (W3's gate, offscreen)

The splitter moves statements; it does not re-implement Qt's compiler. So the check can be exact: build the Designer
window (`Ui_MainWindow.setupUi`) and each generated page side by side, serialize both with the same serializer the live
harness uses (`src/ui/layout_dump.py`), and compare with the same diff (`harness/scripts/ui_layout_diff.py`).

The generated page is docked into a host of the Designer page's size, so geometry is comparable too: a page laid out in
the same box must produce the same rectangle for every widget.

Needs PySide6 (the app venv), not the portable harness interpreter. Offscreen is enough - this is a layout question,
not a painting one; the live pixel run is the second gate.

Usage:
  <app venv>/bin/python harness/scripts/ui_pages_identity.py [--no-geometry] [--limit 6]
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# page widget name -> module in src/ui/pages
PAGES = {
    "homePage": "home_page",
    "morePage": "more_page",
    "procPage": "proc_page",
    "generalSettings": "general_settings",
    "tab_6": "tab_6",
    "renderQueueTab": "render_queue_tab",
    "advancedSettings": "advanced_settings",
    "tab_5": "tab_5",
    "settingsPage": "settings_page",
    "tab_2": "tab_2",
    "tab_3": "tab_3",
    "tab_4": "tab_4",
    "downloadPage": "download_page",
}


def _diff_module():
    path = os.path.join(_ROOT, "harness", "scripts", "ui_layout_diff.py")
    spec = importlib.util.spec_from_file_location("ui_layout_diff", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _show_page(widget) -> None:
    """Make a Designer page current through every stack/tab above it, exactly as the page walker does."""
    from PySide6.QtWidgets import QStackedWidget, QTabWidget

    chain = []
    node = widget
    while node is not None:
        parent = node.parentWidget()
        if isinstance(parent, QTabWidget):
            chain.append((parent, parent.indexOf(node)))
        elif isinstance(parent, QStackedWidget):
            chain.append((parent, parent.indexOf(node)))
        node = parent
    for container, index in reversed(chain):
        container.setCurrentIndex(index)


def _class_name(module: str, widget: str, page_module) -> str:
    for name in dir(page_module):
        if name.endswith("Page") and getattr(page_module, name).__module__ == page_module.__name__:
            cls = getattr(page_module, name)
            if getattr(cls, "page_id", None) == widget:
                return name
    raise SystemExit(f"{module}: no page class with page_id={widget!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", action="store_true",
                    help="also compare rectangles; off by default because the offscreen build path lays out "
                         "QAbstractScrollArea internals differently from the Designer path - the live dump, where "
                         "the whole window is laid out once, is the geometry gate")
    ap.add_argument("--limit", type=int, default=6)
    args = ap.parse_args()

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout

    app = QApplication.instance() or QApplication([])
    import resources_rc  # noqa: F401  - icons resolve through the compiled resource
    from mainwindow import Ui_MainWindow
    from src.ui import layout_dump

    window = QMainWindow()
    designer = Ui_MainWindow()
    designer.setupUi(window)
    window.show()
    app.processEvents()

    diff = _diff_module()
    failures = 0
    for widget, module_name in PAGES.items():
        page_module = importlib.import_module(f"src.ui.pages.{module_name}")
        cls = getattr(page_module, _class_name(module_name, widget, page_module))
        built = cls()
        root = built.build()

        original = getattr(designer, widget)
        _show_page(original)
        app.processEvents()
        box = original.size()
        host = QWidget()
        host.resize(box)
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(root)
        host.show()
        app.processEvents()

        # No origin shift: `QWidget.geometry()` is parent-local, so a page's position inside its container never
        # propagates to its children. Shifting them was a bug in this script (it moved every rectangle by the stack's
        # padding and made an identical page look 20 rectangles off).
        reference = {"root": layout_dump._node(original)}
        candidate = {"root": layout_dump._node(root)}
        # The page's own position is the container's business (the shell decides where a page sits); with geometry on,
        # the root is excluded and Qt's private QAbstractScrollArea/QTabWidget internals are named separately - they are
        # laid out by Qt on show/resize, and the live dump (whole window laid out once) is their gate.
        reference["root"]["geometry"] = list(candidate["root"]["geometry"])
        result = diff.compare(reference, candidate, tolerance_px=0)
        internals = [entry for entry in result["geometry"]
                     if "qt_scrollarea" in entry[0] or "qt_tabwidget" in entry[0]]
        result["geometry"] = [entry for entry in result["geometry"] if entry not in internals]
        result["clean"] = not (result["missing"] or result["extra"] or result["geometry"] or result["text"]
                               or result["visibility"] or result["state"])
        if not args.geometry:
            result["geometry"] = []
            result["clean"] = not (result["missing"] or result["extra"] or result["text"]
                                   or result["visibility"] or result["state"])
        stats = (f"nodes {result['nodes'][0]}/{result['nodes'][1]} miss {len(result['missing'])} "
                 f"extra {len(result['extra'])} geom {len(result['geometry'])} text {len(result['text'])} "
                 f"vis {len(result['visibility'])} state {len(result['state'])}")
        print(f"{widget:20s} {'OK ' if result['clean'] else 'DIFF'} {stats}")
        if not result["clean"]:
            failures += 1
            for path in result["missing"][: args.limit]:
                print(f"    missing {path}")
            for path in result["extra"][: args.limit]:
                print(f"    extra   {path}")
            for path, ga, gb in result["geometry"][: args.limit]:
                print(f"    geom    {path} {ga} -> {gb}")
            if internals:
                print(f"    (qt-internal rectangles not compared: {len(internals)})")
            for path, ta, tb in result["text"][: args.limit]:
                print(f"    text    {path} {ta!r} -> {tb!r}")
            for path, va, vb in result["visibility"][: args.limit]:
                print(f"    visible {path} {va} -> {vb}")
            for path, sa, sb in result["state"][: args.limit]:
                print(f"    state   {path} {sa} -> {sb}")
        host.close()

    print(f"{len(PAGES) - failures} of {len(PAGES)} pages identical to the Designer build")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
