"""The look has one owner, and this test keeps it honest.

The checks live in `harness/scripts/ui_theme_audit.py` because PySide6 is installed in the app's venv, not in the
portable interpreter the harness suite runs on. Here they are pinned to the suite for any environment that does have
Qt: the table must agree with `QTstyle.Palette()`, cover every hex in `QTstyle.styleSheet()`, name only tokens the
toolkit palette defines, and add no state to the palette it derives.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Skipped on the portable interpreter (no PySide6) - the script is the record of that run, not a silent pass.
pytest.importorskip("PySide6.QtGui")

AUDIT = Path(__file__).resolve().parents[2] / "harness" / "scripts" / "ui_theme_audit.py"


def _audit_module():
    spec = importlib.util.spec_from_file_location("ui_theme_audit", AUDIT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_tokens_agree_with_rve_palette_and_stylesheet() -> None:
    problems = _audit_module().audit()
    assert not problems, "\n".join(problems)


def test_one_token_cannot_mean_two_colours(monkeypatch) -> None:
    from src.ui import toolkit_theme

    conflicting = dict(toolkit_theme.LOOK)
    conflicting["QTstyle-test:rule-a"] = ("dialog.text", "#ffffff")
    conflicting["QTstyle-test:rule-b"] = ("dialog.text", "#000000")
    monkeypatch.setattr(toolkit_theme, "LOOK", conflicting)
    with pytest.raises(ValueError, match="dialog.text"):
        toolkit_theme._build_tokens()
