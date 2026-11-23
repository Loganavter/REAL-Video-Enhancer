"""The strip's reach: the produced edge for a live tail, the whole timeline for an export's display feed.

Measured live (2026-11-24, a 495-frame export whose produced edge stood at 7.5 %): a strip drag from 55 % to 30 %
moved nothing. The playhead kept advancing and the backend logged no `seek: requested` line, because
`VideoPreviewSurface.mousePressEvent` refuses a press past `_produced_fraction()` - correct for a tail (nothing exists
past the edge) and wrong for the display feed, which renders any frame of the timeline on demand through
`ProcessTab._on_preview_seek`'s third route.

What is pinned here, offscreen and without a GPU:

* with no ceiling the pane keeps today's behaviour - a press past produced 10 % is refused (`seekBlocked`, no
  `seekRequested`), and a drag is clamped at the same edge;
* with a ceiling of 1.0 the same press emits `seekRequested` with that fraction, and a drag reaches the whole strip;
* `ProcessTab._prepare_display_state` sets 1.0 only when the feed owns the pane (an export with a usable block) and
  leaves `None` for a preview (its tail owns the edge); `_close_display_state` restores `None`;
* the widget's own `set_seek_ceiling(None)` restores the produced-edge refusal.

The `ProcessTab` tests build the object with `object.__new__` (the house style of `test_display_wiring.py`): the wiring
is a handful of plain methods, and constructing the window would only make the test slower. `DisplayReader` is faked
so no shared-memory block is allocated.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = str(Path(__file__).resolve().parents[2])
# BOTH the app and the backend live in a `src` directory (they are namespace packages). The suite's conftest puts the
# backend first, which makes `src.constants` the backend's module - the app's `ProcessTab` needs ITS `src.constants`
# (it owns `has_network` and the pid-scoped names), so the app root goes first for this module. Backend modules do not
# exist under the app root, so they still resolve.
_ORIGINAL_SYS_PATH = list(sys.path)
if _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)
sys.modules.pop("src.constants", None)

# The app's optional runtime imports: stub them ONLY when the interpreter does not have them, so this module never
# shadows a real install for the rest of the suite.
try:  # pragma: no cover - the branch taken depends on the interpreter
    import cpuinfo  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    _cpuinfo = types.ModuleType("cpuinfo")
    _cpuinfo.get_cpu_info = lambda *args, **kwargs: {}
    sys.modules["cpuinfo"] = _cpuinfo
try:  # pragma: no cover
    import pypresence  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    _presence = types.ModuleType("pypresence")
    _presence.Presence = object
    sys.modules["pypresence"] = _presence

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from src.ui.ProcessTab import ProcessTab  # noqa: E402
from src.ui.widgets.video_preview import VideoPreviewSurface  # noqa: E402

# Leave the suite's path order as it was: these modules are imported (and bound) now, while backend `src.utils`
# resolves from the backend root as before.
sys.path[:] = _ORIGINAL_SYS_PATH

PANE_SIZE = (800, 600)
FRAME_SIZE = (1600, 900)


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _picture(width: int, height: int, colour: str = "#204060") -> QImage:
    image = QImage(width, height, QImage.Format.Format_RGBA8888)
    image.fill(QColor(colour))
    return image


@pytest.fixture()
def surface():
    """A fresh surface per test: the harness sweeps top-level widgets after each test."""
    _app()
    widget = VideoPreviewSurface()
    widget.resize(*PANE_SIZE)
    widget.set_frame(_picture(*FRAME_SIZE))
    return widget


def _strip_point(surface, fraction: float) -> QPointF:
    strip = surface._strip_rect()
    assert strip.width() > 0, "the strip must be laid out for this test"
    return QPointF(strip.left() + strip.width() * fraction, strip.center().y())


def _press(widget, point: QPointF) -> None:
    event = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(point), QPointF(point),
                        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                        Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, event)


def _move(widget, point: QPointF) -> None:
    event = QMouseEvent(QEvent.Type.MouseMove, QPointF(point), QPointF(point),
                        Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
                        Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, event)


def _release(widget, point: QPointF) -> None:
    event = QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(point), QPointF(point),
                        Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, event)


# --- the widget's rule -----------------------------------------------------------------------------

def test_no_ceiling_keeps_the_produced_edge_refusal(surface):
    """(a) Today's behaviour stays for the live tail: produced 10 %, a press at 30 % is refused."""
    surface.set_progress(10.0, 100.0)
    assert surface._produced_fraction() == pytest.approx(0.10)
    blocked: list[float] = []
    seeks: list[float] = []
    surface.seekBlocked.connect(blocked.append)
    surface.seekRequested.connect(seeks.append)

    point = _strip_point(surface, 0.30)
    _press(surface, point)
    _release(surface, point)

    assert blocked == [pytest.approx(0.30)], "a press past the produced edge reports the refusal"
    assert seeks == [], "and nothing moves over a picture that does not exist"
    assert surface._scrubbing is False, "a refused press never starts a scrub"


def test_a_ceiling_of_one_lets_the_press_through(surface):
    """(b) The export's feed renders any frame, so the same press is a seek - and a drag reaches the whole strip."""
    surface.set_progress(10.0, 100.0)
    surface.set_seek_ceiling(1.0)
    blocked: list[float] = []
    seeks: list[float] = []
    surface.seekBlocked.connect(blocked.append)
    surface.seekRequested.connect(seeks.append)

    point = _strip_point(surface, 0.30)
    _press(surface, point)
    assert seeks == [pytest.approx(0.30)], "the press reaches the host instead of being refused"
    assert blocked == []
    assert surface._scrubbing is True

    _move(surface, _strip_point(surface, 0.90))
    assert seeks == [pytest.approx(0.30), pytest.approx(0.90)], (
        "a drag follows the whole timeline the feed can render, not the produced band")
    _release(surface, _strip_point(surface, 0.90))
    assert surface._scrubbing is False


def test_the_widget_can_put_the_produced_edge_back(surface):
    """(d) `set_seek_ceiling(None)` restores the tail's rule on the same widget."""
    surface.set_progress(10.0, 100.0)
    surface.set_seek_ceiling(1.0)
    surface.set_seek_ceiling(None)
    blocked: list[float] = []
    seeks: list[float] = []
    surface.seekBlocked.connect(blocked.append)
    surface.seekRequested.connect(seeks.append)

    point = _strip_point(surface, 0.30)
    _press(surface, point)
    _release(surface, point)

    assert blocked == [pytest.approx(0.30)]
    assert seeks == []


# --- the page's rule -------------------------------------------------------------------------------

class _Signal:
    """A Qt signal's `connect` as `_wire_preview_strip` uses it."""

    def __init__(self) -> None:
        self.slots: list = []

    def connect(self, slot) -> None:
        self.slots.append(slot)


class _Pane:
    """The pane API `_prepare_display_state` reads, with the ceiling calls recorded."""

    def __init__(self) -> None:
        self.seekRequested = _Signal()
        self.originalRequested = _Signal()
        self.ceilings: list[float | None] = []

    def width(self) -> int:  # noqa: N802 - Qt spelling
        return PANE_SIZE[0]

    def height(self) -> int:  # noqa: N802 - Qt spelling
        return PANE_SIZE[1]

    def devicePixelRatioF(self) -> float:  # noqa: N802 - Qt spelling
        return 1.0

    def set_seek_ceiling(self, fraction: float | None) -> None:
        self.ceilings.append(fraction)


class _FakeReader:
    """`DisplayReader` without a block: attached, with the signal and release the page uses."""

    def __init__(self, name, box=(0, 0), seek_name=None) -> None:
        self.name = name
        self.box = box
        self.seek_name = seek_name
        self.attached = True
        self.display_frame = _Signal()
        self.released = False

    def release(self) -> None:
        self.released = True


def _tab(*, pane: _Pane, preview: bool) -> ProcessTab:
    tab = object.__new__(ProcessTab)
    tab.page_handles = SimpleNamespace(proc=SimpleNamespace(previewLabel=pane))
    tab.isPreview = bool(preview)
    tab._display = None
    tab._preview_strip_wired_for = None
    return tab


def test_an_exports_feed_lifts_the_ceiling_and_a_preview_leaves_the_edge(monkeypatch):
    """(c) The page says who paints the pane: the export's feed reaches the whole timeline, a tail does not."""
    module = sys.modules[ProcessTab.__module__]
    monkeypatch.setattr(module, "DisplayReader", _FakeReader)

    export_pane = _Pane()
    export_tab = _tab(pane=export_pane, preview=False)
    export_tab._prepare_display_state()
    assert export_tab._display_owns_pane() is True
    assert export_pane.ceilings[-1] == pytest.approx(1.0), (
        "an export whose block is usable seeks the whole timeline: the feed renders any frame on demand")

    preview_pane = _Pane()
    preview_tab = _tab(pane=preview_pane, preview=True)
    preview_tab._prepare_display_state()
    assert preview_tab._display is not None, "the block is allocated for a preview too (the command carries it)"
    assert preview_tab._display_owns_pane() is False
    assert preview_pane.ceilings[-1] is None, (
        "a preview's tail is the picture and only has what the renderer wrote: the produced edge stays")


def test_closing_the_block_restores_the_produced_edge(monkeypatch):
    """(c) The feed's reach must not outlive the feed: the closing pass clears the ceiling."""
    module = sys.modules[ProcessTab.__module__]
    monkeypatch.setattr(module, "DisplayReader", _FakeReader)

    pane = _Pane()
    tab = _tab(pane=pane, preview=False)
    tab._prepare_display_state()
    assert pane.ceilings[-1] == pytest.approx(1.0)
    reader = tab._display

    tab._close_display_state()
    assert reader.released is True
    assert tab._display is None
    assert pane.ceilings[-1] is None


def test_an_export_without_a_usable_block_keeps_the_tail_as_its_reach(monkeypatch):
    """The fallback export (no block) is painted by a tail, so it must keep the produced edge - not inherit a ceiling."""
    module = sys.modules[ProcessTab.__module__]

    class _Unattached(_FakeReader):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.attached = False

    monkeypatch.setattr(module, "DisplayReader", _Unattached)
    pane = _Pane()
    tab = _tab(pane=pane, preview=False)
    tab._prepare_display_state()

    assert tab._display is None
    assert pane.ceilings[-1] is None
