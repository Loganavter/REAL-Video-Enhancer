"""The strip's reach: who decides it, now that the pane no longer has a strip.

Measured live (2026-11-24, a 495-frame export whose produced edge stood at 7.5 %): a strip drag from 55 % to 30 %
moved nothing. The playhead kept advancing and the backend logged no `seek: requested` line, because
`VideoPreviewSurface.mousePressEvent` refused a press past `_produced_fraction()` - correct for a tail (nothing exists
past the edge) and wrong for the display feed, which renders any frame of the timeline on demand through
`ProcessTab._on_preview_seek`'s third route.

**The pane's half of that is deleted** (audit 2026-11-27 §6.6): the painted strip, `seekBlocked`, the fraction-valued
`seekRequested` and the widget-side `set_seek_ceiling` are gone, so the widget no longer decides a reach at all.

**The page's compatibility push is deleted too** (board row `rve-ceiling-port-retarget-2026-11-27`): the page used to
derive a fraction and call `pane.set_seek_ceiling(1.0 | None)` from `_apply_seek_ceiling`. That method had no receiver
on the shipped surface and a second copy of the reach was the defect (CODE_PATTERNS §11). What SURVIVES, and what is
pinned here, is the one SECONDS authority:

* `ProcessTab._preview_seek_ceiling(owner, duration)` answers the whole clip whenever the owner may seek any frame
  (a usable display block, a finished file, a queue item) and the produced edge in seconds for a growing tail;
* the value that reaches the BAR is that fact, published through `PreviewState.seek_ceiling`
  (`ProcessTab._push_preview_state`); `PreviewState.seekable_ceiling()` falls back to `produced` when it is `None`,
  which is why the deleted `None` ("keep the produced edge") and today's produced seconds are the same reach.

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

#: 5 s of clip, 12 % produced: the tail's honest reach is 0.6 s, the whole timeline is 5.0 s.
CLIP_SECONDS = 5.0
PRODUCED_FRACTION = 0.12
PRODUCED_SECONDS = CLIP_SECONDS * PRODUCED_FRACTION


# --- the page's rule -------------------------------------------------------------------------------

class _Signal:
    """A Qt signal's `connect` as `_wire_preview_strip` uses it."""

    def __init__(self) -> None:
        self.slots: list = []

    def connect(self, slot) -> None:
        self.slots.append(slot)


class _Pane:
    """The pane API `_prepare_display_state` reads.

    The SHIPPED surface answers no seek port at all (its strip, and with it the widget-side reach, is deleted - audit
    2026-11-27 §6.6), so this double deliberately has no `set_seek_ceiling`: the page's reach is pinned on
    `_preview_seek_ceiling`, never on a push into the widget.
    """

    def __init__(self) -> None:
        self.seekRequested = _Signal()
        self.originalRequested = _Signal()

    def width(self) -> int:  # noqa: N802 - Qt spelling
        return PANE_SIZE[0]

    def height(self) -> int:  # noqa: N802 - Qt spelling
        return PANE_SIZE[1]

    def devicePixelRatioF(self) -> float:  # noqa: N802 - Qt spelling
        return 1.0


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
    # `owner_of` reads these raw inputs; the produced edge comes from the render counters, as on a live tab.
    tab._queue_pane_owns = False
    tab._queue_pane_decoder = None
    tab._queue_pane_duration = 0.0
    tab._live_tail = None
    tab._preview_decoder = None
    tab._preview_paused = True
    tab._display_seek_pending = False
    tab.renderProcess = None
    tab.currentFrame = PRODUCED_FRACTION * 100.0
    tab.max_value = 100.0
    return tab


def _reach(tab: ProcessTab, duration: float = CLIP_SECONDS) -> float:
    """The one authority's answer for the tab's current owner, in seconds."""
    from src.ui.preview.owner import owner_of

    return tab._preview_seek_ceiling(owner_of(tab), duration)


def test_the_deleted_push_has_no_receiver_left():
    """The compat push is gone: no pane port to call, and the page carries no second authority.

    This is what makes the retarget a deletion instead of a rename - the old route answered a fraction on a pane whose
    strip (and with it the widget-side reach) no longer exists.
    """
    assert not hasattr(ProcessTab, "_apply_seek_ceiling")
    assert not hasattr(_Pane(), "set_seek_ceiling")


def test_a_usable_block_lifts_the_ceiling_for_both_job_kinds(monkeypatch):
    """(c) The page says what can serve a frame: with a usable block both panes reach the whole timeline.

    A preview's *painter* is still its tail (the owner's `may_paint` stays False until a strip press asks the block
    for one frame), but the block answers a press past the produced band with that requested frame, so the reach is the
    whole clip (see `test_preview_frame_on_demand.py`).
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
    assert _reach(export_tab) == pytest.approx(CLIP_SECONDS), (
        "an export whose block is usable seeks the whole timeline: the feed renders any frame on demand")

    preview_pane = _Pane()
    preview_tab = _tab(pane=preview_pane, preview=True)
    preview_tab._prepare_display_state()
    assert preview_tab._display is not None, "the block is allocated for a preview too (the command carries it)"
    assert owner_of(preview_tab).may_paint is False, "the tail is still the preview's painter"
    assert owner_of(preview_tab).may_seek_any is True, "the block answers one frame on demand"
    assert _reach(preview_tab) == pytest.approx(CLIP_SECONDS), (
        "a press past the produced band is answered by the block with that one frame, so the strip may ask")


def test_a_preview_without_a_usable_block_keeps_the_produced_edge(monkeypatch):
    """The other side: no block means the tail is all there is, and the produced edge stays the honest reach (in
    seconds - `PreviewState.seekable_ceiling()` is where a `None` reading falls back to exactly this number)."""
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
    assert _reach(tab) == pytest.approx(PRODUCED_SECONDS), (
        "12 % of a 5 s clip produced: 0.6 s is the furthest a press may reach")
    assert owner_of(tab).may_seek_any is False, "nothing can serve a press past the produced edge"


def test_closing_the_block_restores_the_produced_edge(monkeypatch):
    """(c) The feed's reach must not outlive the feed: the closing pass puts the produced edge back."""
    from src.ui.preview.owner import owner_of

    module = sys.modules[ProcessTab.__module__]
    monkeypatch.setattr(module, "DisplayReader", _FakeReader)

    pane = _Pane()
    tab = _tab(pane=pane, preview=False)
    tab._prepare_display_state()
    assert _reach(tab) == pytest.approx(CLIP_SECONDS)
    reader = tab._display

    tab._close_display_state()
    assert reader.released is True
    assert tab._display is None
    assert _reach(tab) == pytest.approx(PRODUCED_SECONDS), "the reach went back with the block"
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
    assert _reach(tab) == pytest.approx(PRODUCED_SECONDS), "no block, no whole-clip reach: the produced edge stands"
    assert owner_of(tab).may_seek_any is False, "nothing can serve a press past the produced edge"


def test_a_queue_item_reaches_its_whole_file():
    """The deleted branch this retarget must not lose: a queue item is a finished file, so all of it is reachable.

    `_apply_seek_ceiling` used to answer 1.0 for `QUEUE_FILE` (and `None` for an empty field). The seconds authority
    answers the SAME reach through `PreviewState.seekable_ceiling()`: the item's own length.
    """
    tab = _tab(pane=_Pane(), preview=True)
    tab._queue_pane_owns = True
    tab._queue_pane_decoder = SimpleNamespace(duration=42.0)
    tab._queue_pane_duration = 42.0

    assert _reach(tab, 42.0) == pytest.approx(42.0), "every frame of a finished queue item is reachable"
    assert _reach(tab, 0.0) == pytest.approx(0.0), "an empty field has no axis, so there is nothing to seek to"
