"""The strip's reach: who decides it, now that the pane no longer has a strip.

Measured live (2026-11-24, a 495-frame export whose produced edge stood at 7.5 %): a strip drag from 55 % to 30 %
moved nothing. The playhead kept advancing and the backend logged no `seek: requested` line, because
`VideoPreviewSurface.mousePressEvent` refused a press past `_produced_fraction()` - correct for a tail (nothing exists
past the edge) and wrong for the display feed, which renders any frame of the timeline on demand through
`ProcessTab._on_preview_seek`'s third route.

**The pane's half of that is deleted** (audit 2026-11-27 §6.6): the painted strip, `seekBlocked`, the fraction-valued
`seekRequested` and the widget-side `set_seek_ceiling` are gone, so the widget no longer decides a reach at all. What
is pinned here is the half that SURVIVES, the page's:

* `ProcessTab._prepare_display_state` sets 1.0 whenever the pane's block is usable - an export's feed paints any frame
  on demand, and a preview's same block answers a press past the produced band with that one frame (the tail alone
  keeps the produced edge, see `test_preview_frame_on_demand.py`); `_close_display_state` restores `None`;
* the value that reaches the BAR is the same fact, published through `PreviewState.seek_ceiling`
  (`ProcessTab._push_preview_state`) and clamped by `PreviewTimeline`, which is why deleting the widget-side copy loses
  no reach.

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

from src.ui.ProcessTab import ProcessTab  # noqa: E402

# Leave the suite's path order as it was: these modules are imported (and bound) now, while backend `src.utils`
# resolves from the backend root as before.
sys.path[:] = _ORIGINAL_SYS_PATH

PANE_SIZE = (800, 600)


# --- the page's rule -------------------------------------------------------------------------------

class _Signal:
    """A Qt signal's `connect` as `_wire_preview_strip` uses it."""

    def __init__(self) -> None:
        self.slots: list = []

    def connect(self, slot) -> None:
        self.slots.append(slot)


class _Pane:
    """The pane API `_prepare_display_state` reads, with the ceiling calls recorded.

    The SHIPPED surface no longer answers `set_seek_ceiling` (its strip, and with it the widget-side reach, is deleted -
    audit 2026-11-27 §6.6), so the page's call is a no-op for it. It is kept in this double because the page's DECISION
    is what must not regress while `ProcessTab.py` belongs to another lane: "a usable block means the whole timeline is
    reachable", and its surviving consumer is the bar (`_preview_seek_ceiling` -> `PreviewState.seek_ceiling`).
    """

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


def test_a_usable_block_lifts_the_ceiling_for_both_job_kinds(monkeypatch):
    """(c) The page says what can serve a frame: with a usable block both panes reach the whole timeline.

    A preview's *painter* is still its tail (the owner's `may_paint` stays False until a strip press asks the block
    for one frame), but the block answers a press past the produced band with that requested frame, so the strip must
    accept that press (see `test_preview_frame_on_demand.py`).
    """
    from src.ui.preview.owner import owner_of

    module = sys.modules[ProcessTab.__module__]
    monkeypatch.setattr(module, "DisplayReader", _FakeReader)

    export_pane = _Pane()
    export_tab = _tab(pane=export_pane, preview=False)
    export_tab._prepare_display_state()
    assert owner_of(export_tab).may_paint is True, "the export's feed owns its pane"
    assert owner_of(export_tab).may_seek_any is True, (
        "the fact the published reach reads: the feed renders any frame on demand")
    assert export_pane.ceilings[-1] == pytest.approx(1.0), (
        "an export whose block is usable seeks the whole timeline: the feed renders any frame on demand")

    preview_pane = _Pane()
    preview_tab = _tab(pane=preview_pane, preview=True)
    preview_tab._prepare_display_state()
    assert preview_tab._display is not None, "the block is allocated for a preview too (the command carries it)"
    assert owner_of(preview_tab).may_paint is False, "the tail is still the preview's painter"
    assert owner_of(preview_tab).may_seek_any is True, "the block answers one frame on demand"
    assert preview_pane.ceilings[-1] == pytest.approx(1.0), (
        "a press past the produced band is answered by the block with that one frame, so the strip may ask")


def test_a_preview_without_a_usable_block_keeps_the_produced_edge(monkeypatch):
    """The other side: no block means the tail is all there is, and the produced edge stays the honest reach."""
    from src.ui.preview.owner import owner_of

    module = sys.modules[ProcessTab.__module__]

    class _Unattached(_FakeReader):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.attached = False

    monkeypatch.setattr(module, "DisplayReader", _Unattached)
    pane = _Pane()
    tab = _tab(pane=pane, preview=True)
    tab._prepare_display_state()

    assert tab._display is None
    assert pane.ceilings[-1] is None
    assert owner_of(tab).may_seek_any is False, "nothing can serve a press past the produced edge"


def test_closing_the_block_restores_the_produced_edge(monkeypatch):
    """(c) The feed's reach must not outlive the feed: the closing pass clears the ceiling."""
    from src.ui.preview.owner import owner_of

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
    assert owner_of(tab).may_seek_any is False, "the reach went with the block"


def test_an_export_without_a_usable_block_keeps_the_tail_as_its_reach(monkeypatch):
    """The fallback export (no block) is painted by a tail, so it must keep the produced edge - not inherit a ceiling."""
    from src.ui.preview.owner import owner_of

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
    assert owner_of(tab).may_seek_any is False, "nothing can serve a press past the produced edge"
