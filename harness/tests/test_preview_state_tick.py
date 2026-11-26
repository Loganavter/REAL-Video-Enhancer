"""The 100 ms tick must publish the preview state, not only a painted frame.

Measured live 2026-11-25 (owner's screenshot): a paused preview showed "produced 20.6 s · 100%" above a strip whose
produced band stood at `f 316 / 493` and whose thumb ignored presses. The bar is driven by `PreviewState`, and the
state was pushed only from `_on_preview_frame` (once per painted frame) and from transport changes - so a paused pane
with nothing to paint froze both the band and the thumb, while the renderer kept producing.

The tick is the pane's only clock while paused. It is already the owner of the transport, the held progress and the
display feed, so it publishes the state too; `Controller._emit` drops an unchanged state, so an idle pane repaints
nothing.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

_ROOT = str(Path(__file__).resolve().parents[2])
# The app and the backend both own a `src` namespace package; the suite's conftest puts the backend first and
# `ProcessTab` needs the APP's `src.constants` (same pattern as `test_process_readouts_language.py`).
_ORIGINAL_SYS_PATH = list(sys.path)
if _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)
sys.modules.pop("src.constants", None)

# The app's optional runtime imports, stubbed only when the interpreter lacks them.
for _name, _attrs in (("cpuinfo", {"get_cpu_info": lambda *a, **k: {}}),
                      ("pypresence", {"Presence": object})):
    try:  # pragma: no cover - the branch depends on the interpreter
        __import__(_name)
    except ModuleNotFoundError:  # pragma: no cover
        _module = types.ModuleType(_name)
        for _attr, _value in _attrs.items():
            setattr(_module, _attr, _value)
        sys.modules[_name] = _module

from src.ui.ProcessTab import ProcessTab  # noqa: E402

sys.path[:] = _ORIGINAL_SYS_PATH


def _tab() -> ProcessTab:
    tab = object.__new__(ProcessTab)
    tab.page_handles = SimpleNamespace(proc=SimpleNamespace(previewLabel=None))
    # `None` keeps the tick off the readout block: this file tests the pane's publish, and the readouts have their own
    # tests (`test_process_readouts_language.py`) over the real page.
    tab.renderTextOutputList = None
    tab.max_value = 493
    tab.currentFrame = 316
    tab.fps = 0
    tab.eta = 0
    tab.status = ""
    tab._render_started_mono = None
    tab.currentRenderOptions = None
    for name in ("_ensure_live_tail", "_apply_seek_ceiling", "_sync_preview_transport",
                 "_refresh_held_progress", "_tick_display", "_tick_second_surface",
                 "syncRenderConsole", "refreshQueueRows"):
        setattr(tab, name, lambda *args, **kwargs: None)
    return tab


def test_the_tick_publishes_the_preview_state():
    calls: list[str] = []
    tab = _tab()
    tab._push_preview_state = lambda: calls.append("state")
    tab.updateProcessTab()
    assert calls == ["state"], (
        "the tick must publish the preview state: a paused pane has no other clock, and its band/thumb froze "
        f"without it (observed calls: {calls})"
    )


def test_the_tick_publishes_after_the_held_progress_so_both_agree():
    """The pane's own strip is refreshed first, then the bar reads the same numbers - one order, no drift."""
    order: list[str] = []
    tab = _tab()
    tab._refresh_held_progress = lambda: order.append("held")
    tab._push_preview_state = lambda: order.append("state")
    tab.updateProcessTab()
    assert order == ["held", "state"]


def test_a_live_export_band_shows_progress_not_the_whole_clip():
    """The blue half of the strip is "upscaled", so a live export at 14 % must be 14 % blue, not all of it.

    The DISPLAY_FEED pane used to report `produced = duration` unconditionally, so an export's strip was fully blue
    from the first frame and the owner's rule ("серая - ещё не проапскейленное") never showed (2026-11-26, screenshot of
    a render at frame 67/493).
    """
    from src.ui.preview.state import PreviewBackendKind

    tab = _tab()
    tab.currentFrame, tab.max_value = 67, 493
    tab.fps = 24

    # A live export: the band is the render's progress.
    assert tab._produced_for_band(PreviewBackendKind.DISPLAY_FEED, 20.0,
                                  fraction=67 / 493, rendering=True) == pytest.approx(20.0 * 67 / 493)
    # The same pane after the render finished is a finished file: nothing left to produce.
    assert tab._produced_for_band(PreviewBackendKind.DISPLAY_FEED, 20.0,
                                  fraction=67 / 493, rendering=False) == pytest.approx(20.0)
    # A preview's tail is always its progress.
    assert tab._produced_for_band(PreviewBackendKind.LIVE_TAIL, 20.0,
                                  fraction=0.25, rendering=True) == pytest.approx(5.0)
    # A reopened finished file is fully produced.
    assert tab._produced_for_band(PreviewBackendKind.FINISHED_FILE, 20.0,
                                  fraction=0.0, rendering=False) == pytest.approx(20.0)
