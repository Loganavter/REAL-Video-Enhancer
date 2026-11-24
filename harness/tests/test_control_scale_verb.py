"""`scale:<factor>`: the channel can move the toolkit's process-wide UI scale, so a look can be judged on ONE window.

The scale is a single process-wide factor (`sli_ui_toolkit.ui.managers.UiScale`), so "does the text grow" and "do the
pages still fit" can only be judged before/after on the same running window: a re-run in another process is a
different instrument (a different window, different fonts, a different capture). Before this verb the only way to
raise the factor was to edit code and restart, which also reset every page to its first state.

The answer echoes the factor the toolkit actually kept (it clamps), and the resolved application-font pixel size,
so a probe can separate "the number moved" from "the number moved and the text followed". Page overflow is the
existing `audit:` verb's answer, taken right after.
"""

from __future__ import annotations

import json
import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QWidget  # noqa: E402

from src.ui import capture_server, page_capture  # noqa: E402


def _request(root, verb: str, request_id: str, qapp) -> dict:
    """Publish one request the way the channel's clients do, and wait for ITS answer (never the shared `done`)."""
    (root / "requests").mkdir(parents=True, exist_ok=True)
    (root / "requests" / request_id).write_text(verb, encoding="utf-8")
    answer = root / "answers" / f"{request_id}.json"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if answer.exists():
            return json.loads(answer.read_text(encoding="utf-8"))
        QTest.qWait(10)
    raise AssertionError(f"no answer for {verb!r} - the channel did not dispatch it")


@pytest.fixture()
def armed(tmp_path, monkeypatch, qapp):
    """A served window with no pages to walk: `install` needs a page list, the scale verb needs only the scale."""
    monkeypatch.setenv("RVE_UI_CAPTURE_SERVER", str(tmp_path))
    monkeypatch.setenv("RVE_UI_CAPTURE_DELAY_MS", "0")
    monkeypatch.setenv("RVE_UI_CAPTURE_STEP_MS", "0")
    monkeypatch.setattr(page_capture, "steps", lambda window: [("only", lambda: None)])
    window = QWidget()
    capture_server.install(window)
    ready = tmp_path / "ready"
    deadline = time.monotonic() + 10
    while not ready.exists() and time.monotonic() < deadline:
        QTest.qWait(10)
    assert ready.exists(), "the server never armed - cannot judge the verb"
    yield tmp_path
    window.close()


def test_scale_moves_the_toolkit_factor_and_reports_what_it_kept(armed, qapp):
    from sli_ui_toolkit.ui.managers import MAX_FACTOR, UiScale

    manager = UiScale.get_instance()
    before = manager.factor()
    try:
        answer = _request(armed, "scale:1.5", "s1", qapp)
        assert answer["kind"] == "scale"
        assert answer["error"] is None, answer["error"]
        assert answer["requested"] == "1.5"
        assert answer["before"] == pytest.approx(before), "the answer must name the factor it found, not assume 1.0"
        assert answer["factor"] == pytest.approx(1.5)
        assert manager.factor() == pytest.approx(1.5), "the verb must move the real manager, not only its report"
        assert max(answer["applicationFontPx"], answer["applicationFontPt"]) > 0, (
            "Qt reports exactly one channel for the app face (-1 for the other); the answer must carry a real size")
    finally:
        manager.set_factor(before)


def test_scale_asks_the_toolkit_for_the_clamp_and_a_bad_number_is_an_error(armed, qapp):
    from sli_ui_toolkit.ui.managers import MAX_FACTOR, MIN_FACTOR, UiScale

    manager = UiScale.get_instance()
    before = manager.factor()
    try:
        answer = _request(armed, "scale:99", "s2", qapp)
        assert answer["error"] is None
        assert answer["factor"] == pytest.approx(MAX_FACTOR), (
            "the toolkit owns the range; the verb must report the clamped value instead of pretending 99")
        assert manager.factor() == pytest.approx(MAX_FACTOR)

        answer = _request(armed, "scale:0.0001", "s3", qapp)
        assert answer["error"] is None
        assert answer["factor"] == pytest.approx(MIN_FACTOR)

        answer = _request(armed, "scale:bigger", "s4", qapp)
        assert answer["error"] and "not a number" in answer["error"], (
            f"an unparseable factor must be answered, not silently ignored: {answer}")
        assert manager.factor() == pytest.approx(MIN_FACTOR), "a rejected request must not move the scale"
    finally:
        manager.set_factor(before)
