"""The structural diff must not report a page as equal when it is not.

`ui_layout_diff.py` is the instrument a code rebuild is judged by, and an instrument that says "identical" too easily
is worse than none: it would certify a page that lost a widget or moved a row. These cases pin the categories it
reports, using hand-written trees so the test does not need an app, a display or a captured dump.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

TOOL = Path(__file__).resolve().parents[2] / "harness" / "scripts" / "ui_layout_diff.py"


def _load():
    spec = importlib.util.spec_from_file_location("ui_layout_diff", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _tree(*children: dict) -> dict:
    root = {"class": "QMainWindow", "name": "MainWindow", "geometry": [0, 0, 800, 600],
            "visible": True, "enabled": True}
    if children:
        root["children"] = list(children)
    return {"geometry_space": "parent-local", "root": root}


def _label(name: str, geometry: list[int], text: str = "x", **extra) -> dict:
    node = {"class": "QLabel", "name": name, "geometry": geometry, "visible": True, "enabled": True, "text": text}
    node.update(extra)
    return node


def test_identical_trees_are_clean() -> None:
    tool = _load()
    tree = _tree(_label("label_1", [10, 20, 100, 20]))
    result = tool.compare(tree, tree, tolerance_px=0)
    assert result["clean"], result


def test_moved_widget_is_geometry() -> None:
    tool = _load()
    a = _tree(_label("label_1", [10, 20, 100, 20]))
    b = _tree(_label("label_1", [12, 24, 100, 20]))
    result = tool.compare(a, b, tolerance_px=0)
    assert not result["clean"]
    assert [path for path, _ga, _gb in result["geometry"]] == ["MainWindow/label_1"]


def test_tolerance_absorbs_subpixel_layout_noise() -> None:
    tool = _load()
    a = _tree(_label("label_1", [10, 20, 100, 20]))
    b = _tree(_label("label_1", [11, 21, 100, 20]))
    assert tool.compare(a, b, tolerance_px=1)["clean"]
    assert not tool.compare(a, b, tolerance_px=0)["clean"]


def test_lost_or_added_widget_is_reported() -> None:
    tool = _load()
    a = _tree(_label("label_1", [0, 0, 10, 10]), _label("label_2", [0, 10, 10, 10]))
    b = _tree(_label("label_1", [0, 0, 10, 10]))
    result = tool.compare(a, b, tolerance_px=0)
    assert result["missing"] == ["MainWindow/label_2"]
    assert not result["clean"]


def test_text_change_is_reported_and_can_be_ignored() -> None:
    tool = _load()
    a = _tree(_label("systemInfoText", [0, 0, 10, 10], text="Available Disk Space: 58.24GB"))
    b = _tree(_label("systemInfoText", [0, 0, 10, 10], text="Available Disk Space: 58.23GB"))
    assert not tool.compare(a, b, tolerance_px=0)["clean"]
    assert tool.compare(a, b, tolerance_px=0, ignore_text=["systemInfoText"])["clean"]


def test_state_change_is_reported() -> None:
    tool = _load()
    a = _tree(_label("combo", [0, 0, 10, 10], currentIndex=0))
    b = _tree(_label("combo", [0, 0, 10, 10], currentIndex=1))
    result = tool.compare(a, b, tolerance_px=0)
    assert not result["clean"]
    assert result["state"], "a different current index changes what the page shows"


def test_visibility_flip_is_reported() -> None:
    tool = _load()
    a = _tree(_label("label_1", [0, 0, 10, 10]))
    b = _tree(_label("label_1", [0, 0, 10, 10]))
    b["root"]["children"][0]["visible"] = False
    result = tool.compare(a, b, tolerance_px=0)
    assert result["visibility"], "the runtime swap hides the source widget - that must be visible to the diff"


def test_unnamed_siblings_do_not_collide() -> None:
    tool = _load()
    twins = [{"class": "QScrollBar", "name": "", "geometry": [0, 0, 5, 50], "visible": True, "enabled": True}]
    a = _tree(*twins, dict(twins[0]))
    b = _tree(*twins, dict(twins[0]))
    paths = list(_load().flatten(a))
    assert len(paths) == 3, paths
    assert _load().compare(a, b, tolerance_px=0)["clean"]
