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


def test_a_frame_already_in_flight_does_not_answer_the_press():
    """The press closes on the frame that ANSWERS it, not on the first one that arrives.

    A tail emits frames decoded before the press; closing on such a frame moved the marker back to the old position
    for a few frames and then forward again (owner, 2026-11-26: "скачет назад-вперёд пару раз несколько кадров
    подряд"). The timeout is the escape hatch: a seek that cannot land still settles, with the delta in the log.
    """
    import time as _time

    tab = _tab()
    tab.currentRenderOptions = SimpleNamespace(videoFps=24.0)
    tab._preview_playhead = 1.5          # a frame from before the press
    tab._seek_pressed_at = _time.perf_counter()
    assert tab._frame_answers_watch(10.28) is False, "a stale frame must not close the press"

    tab._preview_playhead = 10.28        # the frame the press asked for
    assert tab._frame_answers_watch(10.28) is True

    tab._preview_playhead = 1.5          # a seek that never lands must still settle eventually
    tab._seek_pressed_at = _time.perf_counter() - (tab.SEEK_WATCH_TIMEOUT_S + 0.01)
    assert tab._frame_answers_watch(10.28) is True


def test_the_landing_tolerance_is_two_frames_with_a_floor():
    tab = _tab()
    tab.currentRenderOptions = SimpleNamespace(videoFps=24.0)
    assert tab._seek_landing_tolerance() == pytest.approx(2.0 / 24.0)
    tab.currentRenderOptions = SimpleNamespace(videoFps=0.0)
    tab.fps = 0
    assert tab._seek_landing_tolerance() == pytest.approx(0.1)
