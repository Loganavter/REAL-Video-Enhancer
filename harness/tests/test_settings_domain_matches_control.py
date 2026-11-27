"""A control's range and the store's domain must be the same set.

Settings audit (2026-11-27, F3): the threshold spinbox is `setRange(1.0, 10.0)` with one decimal, while `allowedSettings`
carried `str(num / 10) for num in range(1, 100)` - 0.1 .. 9.9. Both ends disagreed: the control offered a value the
writer dropped (10.0 never persisted) and the store accepted values the control cannot show (0.1-0.9). Neither side
was wrong on its own; nobody compared them.

This file compares them. The page's side is read with `ast` rather than by constructing the page (a whole window for
one number), so a change to either half fails here instead of surfacing as "my threshold resets itself".
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[2] / "src" / "ui" / "pages" / "tab_2.py"

KEY = "scene_change_detection_threshold"


def _page_range(widget: str) -> tuple[float, float]:
    tree = ast.parse(PAGE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) != 2:
            continue
        func = node.func
        if getattr(func, "attr", "") != "setRange":
            continue
        owner = getattr(func.value, "attr", "")
        if owner != widget:
            continue
        low, high = (ast.literal_eval(arg) for arg in node.args)
        return float(low), float(high)
    raise AssertionError(f"{PAGE.name} has no `{widget}.setRange(a, b)`; the pin below is stale")


def test_the_domain_equals_the_control_range(tmp_path):
    from src.ui.settings_store import Settings

    settings = Settings(str(tmp_path / "settings.txt"))
    domain = {float(value) for value in settings.allowedSettings[KEY]}
    low, high = _page_range(KEY)

    assert min(domain) == pytest.approx(low), "a value the control offers below the domain is silently dropped"
    assert max(domain) == pytest.approx(high), "a value the control offers above the domain is silently dropped"
    assert len(domain) == round((high - low) * 10) + 1, "the domain has holes the control can walk into"
