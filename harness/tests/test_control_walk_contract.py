"""`=1` is the diagnosis channel, and "no page walk" is a behaviour, not a parse result.

`test_control_channel_optin.py` pins the mode table (`=1` -> walk: no), but it only asks `mode()`. The walk itself is
scheduled at the END of `install()` (`QTimer.singleShot(delay, lambda: warm(0))`), outside every `always_on` branch,
so on a real boot the bare flag walked every page forty seconds in - switching pages under the owner's hands, wiping
the `ready` file and contradicting the module's own comment ("It polls immediately and only warms a page when a request
asks for it").

The control audit (2026-11-27, F4) and the doc-truth audit (LIVE-TESTING.md:61-65) both reported it as a doc error.
It is a code error, and this file is the missing half of the contract: the parser says which mode, and `install()`
must obey it.
"""

from __future__ import annotations

import sys
import types

import pytest

if "resources_rc" not in sys.modules:
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")


@pytest.fixture()
def walk_spy(monkeypatch, tmp_path):
    """A page walk with one recorded step, in a throwaway control root."""
    from src.ui import capture_server, page_capture

    calls: list[str] = []
    monkeypatch.setattr(page_capture, "steps", lambda window: [("page-1", lambda: calls.append("p1"))])
    monkeypatch.setattr(page_capture, "walk_state", lambda window: {"page": 0, "tab": 0})
    monkeypatch.setattr(page_capture, "restore_walk_state", lambda window, before: None)
    monkeypatch.setattr(capture_server, "control_root", lambda: str(tmp_path / "control"))
    monkeypatch.setenv("RVE_UI_CAPTURE_DELAY_MS", "30")
    monkeypatch.setenv("RVE_UI_CAPTURE_STEP_MS", "5")
    return capture_server, calls


def test_a_bare_flag_never_walks_even_after_the_startup_delay(walk_spy, monkeypatch, qtbot):
    capture_server, calls = walk_spy
    monkeypatch.setenv("RVE_UI_CAPTURE_SERVER", "1")

    capture_server.install(object())
    qtbot.wait(250)

    assert calls == [], "a bare `=1` channel walked a page; the mode says it must not"


def test_a_directory_still_walks_once(walk_spy, monkeypatch, tmp_path, qtbot):
    capture_server, calls = walk_spy
    monkeypatch.setenv("RVE_UI_CAPTURE_SERVER", str(tmp_path / "capture"))
    capture_server.install(object())
    qtbot.wait(250)

    assert calls == ["p1"], "the capture mode lost its warm-up walk"
