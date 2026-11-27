"""The draw line's `half=` is a claim about the two pictures, and it must be derived from them.

Owner, 2026-11-27, after seeing the upscale vanish: "затерся, а апскейл справа больше не появлялся". The log for that
minute said `half=pair` on every compare paint - including two whose `result_fp` was `-`, i.e. paints with no right
half at all. The instrument asserted a pair it did not have, which is how a real defect reads as "nothing happened".
"""

from __future__ import annotations

import sys
import types

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage

if "resources_rc" not in sys.modules:
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")


@pytest.fixture()
def pane(qapp):
    from src.ui.widgets.video_preview import VideoPreviewSurface

    widget = VideoPreviewSurface()
    widget.resize(400, 300)
    yield widget
    widget.close()


@pytest.fixture()
def draws(monkeypatch):
    from src.ui.video import telemetry

    captured: list[dict] = []

    def fake_event(kind, **fields):
        if kind == "draw":
            captured.append(fields)

    monkeypatch.setattr(telemetry, "event", fake_event)
    return captured


def _image(fill=Qt.GlobalColor.red) -> QImage:
    image = QImage(32, 32, QImage.Format.Format_RGBA8888)
    image.fill(fill)
    return image


def test_both_pictures_are_a_pair(pane, draws):
    pane.set_view_mode("sbs")
    pane._log_draw_two(_image(), _image(Qt.GlobalColor.blue), present=True)

    assert draws and draws[-1]["half"] == "pair"


def test_a_missing_result_is_named_as_such(pane, draws):
    pane.set_view_mode("sbs")
    # `present` is the CALLER's planes-aware verdict on the result half (see `_on_rhi_draw`), so "no result" is
    # `present=False` - a YUV result has no QImage and is still a picture.
    pane._log_draw_two(None, _image(), present=False)

    assert draws[-1]["half"] == "lone:source", "the line said 'pair' while the right half was missing"


def test_a_missing_original_is_named_as_such(pane, draws):
    pane.set_view_mode("sbs")
    pane._log_draw_two(_image(), None, present=True)

    assert draws[-1]["half"] == "lone:result"


def test_a_picture_with_no_committed_pair_is_named_held(pane, draws):
    """Entering a compare mode KEEPS the picture that was on screen while it waits for the first pair.

    Calling that retained picture "lone:result" reported a defect the user cannot see (the pane is still showing a
    whole picture, just not yet a pair of this mode) - and the blank alternative made "ни одного кадра нет"
    (owner, 2026-11-27). The line says `half=held`.
    """
    pane.set_view_mode("sbs")
    pane._log_draw_two(_image(), _image(Qt.GlobalColor.blue), present=True, held=True)

    assert draws and draws[-1]["half"] == "held"
