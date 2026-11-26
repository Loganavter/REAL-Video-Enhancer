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
import threading
import time
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

from src.ui.QTcustom import UpdateGUIThread  # noqa: E402
from src.ui.ProcessTab import ProcessTab  # noqa: E402

sys.path[:] = _ORIGINAL_SYS_PATH


def _pump(qapp, predicate, seconds: float = 5.0) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _settle(qapp, seconds: float = 0.4) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)


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


# -- the tick's thread contract (plan P6, 2026-11-26) --------------------------------------------------
#
# `ProcessTab` is not a `QObject`, so `self.workerThread.tick.connect(self.updateProcessTab)` binds a plain callable
# and there is no receiver whose affinity the page could read from itself. The question P6 asked was therefore
# empirical: does that body run on `UpdateGUIThread` or on the GUI thread? These two tests answer it on this host
# and keep the answer honest - the first drives the real thread into the real body, the second pins WHO owns the
# clock (an `UpdateGUIThread`, not a second `QTimer`), so a future "just connect it directly" edit fails here
# instead of as a widget write off the GUI thread in the render.


def test_the_tick_body_runs_on_the_gui_thread(qapp):
    """The 100 ms tick's body is a GUI-thread body - measured, not assumed.

    A plain-callable connection has no receiver QObject, and PySide resolves the connection type from the proxy it
    builds for the callable: measured here (PySide6 6.12.0 / Qt 6.12.0), the emit in `UpdateGUIThread.run()` does not
    call the slot at all, and every call lands on `qApp.thread()`. Both halves are asserted: nothing runs before the
    GUI loop turns (a direct connection would), and everything that runs does so on the GUI thread.

    The two hooks are installed by the test on the instance, so the recording happens INSIDE the real
    `updateProcessTab` at the two places P6 also cares about: the tick's first act (`_tick_press_deadline`, plan P4's
    deadline) and its pane publish.
    """
    main = threading.get_ident()          # the conftest builds the one QApplication in this thread
    seen: list[tuple[str, int]] = []
    tab = _tab()
    tab._tick_press_deadline = lambda: seen.append(("deadline", threading.get_ident()))
    tab._push_preview_state = lambda: seen.append(("state", threading.get_ident()))
    tab._tick_second_surface = lambda: None
    thread = UpdateGUIThread(parent=tab)
    thread.tick.connect(tab.updateProcessTab)          # the app's exact shape (ProcessTab.py::startGUIUpdate)
    try:
        thread.start()
        # `run()` emits immediately and has no event loop of its own; give it a full second of emits with no event
        # processing. A DirectConnection would have called the body already, and this is the discriminator.
        quiet = time.time() + 1.0
        while not seen and time.time() < quiet:
            time.sleep(0.01)
        assert seen == [], (
            "the tick ran at emit time, before the GUI loop turned: the connection is no longer queued to the GUI "
            f"thread (observed: {seen})"
        )
        assert _pump(qapp, lambda: len(seen) >= 4), "the tick never reached the page after the GUI loop turned"
    finally:
        thread.stop()
        thread.quit()
        thread.wait(2000)

    off_thread = [entry for entry in seen if entry[1] != main]
    assert not off_thread, (
        "updateProcessTab, its press deadline and its publish must run on the GUI thread - it writes widgets; "
        f"off-thread calls: {off_thread} (GUI thread is {main})"
    )
    assert {name for name, _ in seen} == {"deadline", "state"}, (
        f"both hooks sit inside the tick's body and must be reached by the same tick (reached: {seen})"
    )


def test_the_tick_is_owned_by_the_update_thread_and_stops_with_it(qapp):
    """The page's clock is `UpdateGUIThread` and nothing else: 100 ms, the page's own method, gone when it stops.

    No window is needed for this half - the wiring is the object graph. The spy replaces the instance attribute BEFORE
    `startGUIUpdate` reads it, so what is asserted is the production wiring (whatever `startGUIUpdate` connects), not a
    copy of it. The stop assertion is the one a second `QTimer` would fail: with the thread stopped and the queue
    drained, no further tick may arrive.
    """
    main = threading.get_ident()
    calls: list[int] = []
    tab = _tab()
    tab.updateProcessTab = lambda: calls.append(threading.get_ident())
    tab.guiChangesOnRenderCompletion = lambda *args, **kwargs: None
    tab.startGUIUpdate()
    thread = tab.workerThread
    try:
        assert isinstance(thread, UpdateGUIThread), f"the tick's owner is {type(thread)!r}, not UpdateGUIThread"
        assert UpdateGUIThread.TICK_SECONDS == pytest.approx(0.1), (
            "the period is the contract the readouts were measured at; P6 moves the tick, never its interval"
        )
        assert _pump(qapp, lambda: len(calls) >= 2), "startGUIUpdate did not connect the page's updateProcessTab"
        _settle(qapp, 0.35)                       # three own intervals: the clock is actually running
        assert len(calls) >= 4, f"the tick is slower than its interval (got {len(calls)} calls in ~0.45 s)"
    finally:
        thread.stop()
        thread.quit()
        thread.wait(2000)
        qapp.processEvents()                      # drain what was already emitted before the stop
    settled = len(calls)
    _settle(qapp, 0.4)                            # four more intervals with no thread alive
    assert len(calls) == settled, (
        "ticks kept arriving after UpdateGUIThread stopped: something else owns the page's clock "
        f"({len(calls) - settled} extra calls)"
    )
    assert all(ident == main for ident in calls), f"a tick ran off the GUI thread: {sorted(set(calls))} vs {main}"
