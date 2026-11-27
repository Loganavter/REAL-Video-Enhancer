"""An empty pane promises nothing: no pan, no hand cursor, no request for an original it cannot show.

Owner, 2026-11-26: "при пустом экране можно зажать пустое место и курсор реально сменится - это неверно". The compare
press armed a pan and set `ClosedHandCursor` regardless of whether the pane held a picture, and `pan_by` then clamped
to zero - a gesture that claims something the pane cannot do, the same class as the checked ring on an empty pane.

The other half is equally important: with a picture, the gesture is unchanged - compare modes still pan on LMB, single
mode still asks for the original on LMB, and the middle button still pans everywhere.
"""

from __future__ import annotations

import sys
import types

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest

if "resources_rc" not in sys.modules:
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")


@pytest.fixture()
def pane(qapp):
    from src.ui.widgets.video_preview import VideoPreviewSurface

    widget = VideoPreviewSurface()
    widget.resize(400, 300)
    widget.show()
    yield widget
    widget.close()


def _picture(widget) -> None:
    image = QImage(64, 64, QImage.Format.Format_RGBA8888)
    image.fill(Qt.GlobalColor.red)
    widget.set_frame(image, index=1, media_time=0.0, generation=1)


def test_an_empty_compare_pane_arms_no_pan_and_shows_no_hand(pane):
    pane.set_view_mode("sbs")

    QTest.mousePress(pane, Qt.MouseButton.LeftButton, pos=QPoint(120, 120))
    QTest.mouseRelease(pane, Qt.MouseButton.LeftButton, pos=QPoint(120, 120))

    assert pane._panning is False, "an empty pane armed a pan it cannot perform"
    assert pane.cursor().shape() != Qt.CursorShape.ClosedHandCursor, "the cursor claimed a picture that is not there"


def test_an_empty_single_pane_does_not_ask_for_an_original(pane):
    pane.set_view_mode("single")
    asked: list[float] = []
    pane.originalRequested.connect(asked.append)

    QTest.mousePress(pane, Qt.MouseButton.LeftButton, pos=QPoint(120, 120))
    QTest.mouseRelease(pane, Qt.MouseButton.LeftButton, pos=QPoint(120, 120))

    assert asked == [], "the pane asked the host to decode a frame for a picture it does not have"
    assert pane._showing_original is False


def test_the_middle_button_does_not_pan_an_empty_pane(pane):
    pane.set_view_mode("sbs")

    QTest.mousePress(pane, Qt.MouseButton.MiddleButton, pos=QPoint(120, 120))
    QTest.mouseRelease(pane, Qt.MouseButton.MiddleButton, pos=QPoint(120, 120))

    assert pane._panning is False


def test_a_picture_keeps_the_gesture(pane):
    pane.set_view_mode("sbs")
    _picture(pane)

    QTest.mousePress(pane, Qt.MouseButton.LeftButton, pos=QPoint(120, 120))

    assert pane._panning is True, "the compare gesture must be unchanged with a picture on screen"
    assert pane.cursor().shape() == Qt.CursorShape.ClosedHandCursor
    QTest.mouseRelease(pane, Qt.MouseButton.LeftButton, pos=QPoint(120, 120))
