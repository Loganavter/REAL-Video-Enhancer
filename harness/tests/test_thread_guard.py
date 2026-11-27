"""Every pipe thread is guarded: a raising reader or writer must end the JOB, not leave the process alive.

Backend audit 2026-11-27, 5.10: only the render loop was wrapped in `guard_render_thread`. A reader or writer that
raised left the backend ALIVE with the pipe wedged - the GUI polls a live process at 0 FPS forever (the 2026-11-22
"render that never ends"). The guard's `path` argument was also accepted and never printed, so once four threads go
through it a failure line could not say whose it was.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def logged(monkeypatch):
    from src.utils import Util

    lines: list[str] = []
    monkeypatch.setattr(Util, "log", lambda message, *a, **k: lines.append(str(message)))
    return lines


def test_a_failing_body_exits_non_zero_and_names_the_thread(logged):
    from src.utils.Util import guard_render_thread

    exits: list[int] = []

    def body():
        raise RuntimeError("boom")

    code = guard_render_thread(body, path="reader", exit_process=exits.append)

    assert code == 1
    assert exits == [1], "the process must leave non-zero in the same call, or it stays alive with a wedged pipe"
    assert any("RENDER FAILED" in line for line in logged), "the GUI keys on this prefix"
    assert any("reader" in line for line in logged), "the line must name the thread that died"


def test_a_healthy_body_exits_nothing(logged):
    from src.utils.Util import guard_render_thread

    exits: list[int] = []
    assert guard_render_thread(lambda: None, path="writer", exit_process=exits.append) == 0
    assert exits == []
    assert logged == []
