"""A page fade must not leave a QGraphicsOpacityEffect on the stack.

The effect makes Qt paint the whole stack into an offscreen surface and blit it; a page switch under it is not
invalidated, so the newly shown page stayed black until an input forced a repaint (owner report 2026-11-22: "the
interface appears only when you click it").
"""
from __future__ import annotations

import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from src.ui.AnimationHandler import AnimationHandler  # noqa: E402


def _wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_a_fade_removes_its_effect_when_it_finishes():
    app = QApplication.instance() or QApplication([])
    widget = QWidget()
    widget.resize(200, 100)
    widget.show()
    app.processEvents()

    handler = AnimationHandler()

    handler.fadeInAnimation(widget)
    assert widget.graphicsEffect() is not None, "the fade should install its effect"
    assert _wait_until(lambda: widget.graphicsEffect() is None), (
        "the opacity effect outlived the fade-in - a page switch under it shows a stale frame"
    )

    handler.fadeOutAnimation(widget)
    assert widget.graphicsEffect() is not None
    assert _wait_until(lambda: widget.graphicsEffect() is None), (
        "the opacity effect outlived the fade-out"
    )
    app.processEvents()
