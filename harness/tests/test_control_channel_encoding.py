"""Every request the control channel READS is either dispatched or answered with an error - never eaten silently.

Three defects, all measured on a leased instance on 2026-11-24 (`src/ui/capture_server.py`):

1. `poll()` opened the request with a plain utf-8 `open(...)`. A request carrying a Cyrillic path raised
   `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb7` INSIDE the timer callback, so `poll` never
   rescheduled: every later request got no answer, which from outside is indistinguishable from a hung app.
2. `poll()` tested `os.path.exists`, then opened and unlinked. A client writing in place truncates first, and a
   read in that window saw `""` - the file was consumed, nothing was dispatched and nothing answered. Worse, the
   empty verb fell through `handle()` into `capture("")`, which means "every page": a lost request ran a batch.
3. ONE `request` slot, many writers: a probe polling `state:` while a `render:`/`preview:` request is in flight
   overwrote it inside one 120 ms tick, and the long verb never ran (no `failed:` line either - the handler was
   never called).

The invariant, stated once: **a request that reaches the intake is dispatched once or rejected visibly, and the
intake survives a bad byte, an empty file in the write window, and one handler that raises.** The reader, the
drain and the fallthrough verdict are module-level, so this file needs no window; what genuinely needs one is
listed as LIVE-PROOF WANTED in the close-out.

The red-before is the absence of the helper itself: `capture_server.drain_requests` and
`capture_server.page_request_error` did not exist when this file was written.
"""

from __future__ import annotations

import json
import time

import pytest

from src.ui import capture_server

GRACE = capture_server.EMPTY_REQUEST_GRACE_S


class _Recorder:
    """What the loop would have dispatched: `(verb, request_id)` in arrival order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def __call__(self, verb: str, request_id: str | None) -> None:
        self.calls.append((verb, request_id))


def _write(path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


# -- mechanism 1: one non-UTF-8 byte must not take the intake down ---------------------------------


def test_a_non_utf8_byte_is_replaced_and_the_next_tick_still_serves(tmp_path):
    legacy = tmp_path / "request"
    _write(legacy, b"state:previewLabel\xb7bad")
    record = _Recorder()

    assert capture_server.drain_requests(str(tmp_path), record, now=10.0) == 1, (
        "a request with an undecodable byte must still be a request - not an exception into the loop")
    verb, request_id = record.calls[0]
    assert verb.startswith("state:previewLabel") and "\ufffd" in verb, (
        f"the bad byte must arrive as a replacement character, got {verb!r}")
    assert request_id is None
    assert not legacy.exists(), "a request that was read must not be handed over twice"

    # The loop is alive: the very next tick serves a valid request. This is the assertion the measured
    # `UnicodeDecodeError` broke - every later request went unanswered.
    _write(legacy, b"state:ok#next1")
    assert capture_server.drain_requests(str(tmp_path), record, now=10.1) == 1
    assert record.calls[1] == ("state:ok#next1", None), (
        "the legacy file's `verb#id` is parsed by handle(); drain passes it through untouched")
    assert not legacy.exists()


def test_a_non_utf8_byte_in_a_request_file_is_bound_to_its_filename_id(tmp_path):
    _write(tmp_path / "requests" / "r7", b"\xb7")
    record = _Recorder()

    assert capture_server.drain_requests(str(tmp_path), record, now=0.0) == 1
    assert record.calls == [("\ufffd", "r7")], (
        "the name of the file in requests/ is the id: it must survive decoding, and the verb is what was written")


# -- mechanism 2: the truncate-then-write window is not a request yet ------------------------------


def test_an_empty_request_file_is_left_for_the_next_tick_instead_of_consumed(tmp_path):
    legacy = tmp_path / "request"
    _write(legacy, b"")
    record = _Recorder()

    assert capture_server.drain_requests(str(tmp_path), record, now=100.0) == 0
    assert record.calls == []
    assert legacy.exists(), (
        "an empty file is a client mid-write (it truncates first): consuming it loses the request in silence")

    _write(legacy, b"nav4")                      # the client finished writing
    assert capture_server.drain_requests(str(tmp_path), record, now=100.12) == 1
    assert record.calls == [("nav4", None)]
    assert not legacy.exists()


def test_an_empty_request_file_is_kept_then_served_once_it_is_written(tmp_path):
    staged = tmp_path / "requests" / "r9"
    _write(staged, b"")
    record = _Recorder()

    assert capture_server.drain_requests(str(tmp_path), record, now=0.0) == 0
    assert staged.exists()
    _write(staged, b"state:ok")
    assert capture_server.drain_requests(str(tmp_path), record, now=0.05) == 1
    assert record.calls == [("state:ok", "r9")]


def test_an_empty_request_that_never_fills_is_answered_not_swallowed(tmp_path):
    legacy = tmp_path / "request"
    _write(legacy, b"")
    record = _Recorder()
    state: dict = {}

    assert capture_server.drain_requests(str(tmp_path), record, state=state, now=0.0) == 0
    expired = capture_server.drain_requests(str(tmp_path), record, state=state, now=GRACE + 0.01)
    assert expired == 1, "past the grace window the empty file is a request the loop must answer"
    assert record.calls == [("", None)], (
        "the empty verb is handed to handle(), whose fallthrough answers it with an error - it must never "
        "reach capture('') and photograph every page")
    assert not legacy.exists(), "an answered empty file is removed, or it would answer again every tick"


def test_the_empty_and_unknown_verb_are_errors_not_a_full_capture():
    known = {"nav1": None, "nav2": None}

    empty = capture_server.page_request_error("", known)
    assert empty is not None and "empty" in empty, f"an empty verb needs its own sentence, got {empty!r}"
    bogus = capture_server.page_request_error("bogus", known)
    assert bogus is not None and "bogus" in bogus, f"an unknown verb must name itself, got {bogus!r}"
    assert "nav1" in bogus, "the rejection should say what IS understood"

    assert capture_server.page_request_error("all", known) is None
    assert capture_server.page_request_error(" nav1 , nav2 ", known) is None
    # A partly known list is still a capture request: `capture()` answers with the pages it could name, and its
    # `done` payload lists them - the client sees which one was dropped (the pre-2026-11-24 behaviour).
    assert capture_server.page_request_error("nav1,bogus", known) is None


# -- mechanism 3: one file per request, so two writers cannot clobber each other -------------------


def test_two_requests_written_in_the_same_tick_each_get_their_own_answer(tmp_path):
    _write(tmp_path / "requests" / "a1", b"state:previewLabel")
    _write(tmp_path / "requests" / "b2", b"render:/home/x/clip.mkv")
    record = _Recorder()

    assert capture_server.drain_requests(str(tmp_path), record, now=5.0) == 2, (
        "both requests were on disk before the tick: a single slot would have kept only the last one")
    assert sorted(record.calls) == [("render:/home/x/clip.mkv", "b2"), ("state:previewLabel", "a1")]
    assert list((tmp_path / "requests").iterdir()) == [], "each answered request file is removed"


def test_a_request_file_takes_its_verb_verbatim_so_an_inline_hash_is_not_an_id(tmp_path):
    """`preview:<clip>#model=futsuu` uses `#` itself; only the legacy single file may read it as an id."""
    _write(tmp_path / "requests" / "p1", b"preview:/home/x/clip.mkv#model=futsuu")
    record = _Recorder()

    assert capture_server.drain_requests(str(tmp_path), record, now=0.0) == 1
    assert record.calls == [("preview:/home/x/clip.mkv#model=futsuu", "p1")]


def test_the_legacy_single_file_still_dispatches_alongside_the_directory(tmp_path):
    """The supervisor's probes, `ui_live.py shot` and `live.sh ask` write the single file; they keep working."""
    _write(tmp_path / "request", b"state:window#legacy1")
    _write(tmp_path / "requests" / "d1", b"where:")
    record = _Recorder()

    assert capture_server.drain_requests(str(tmp_path), record, now=1.0) == 2
    assert ("state:window#legacy1", None) in record.calls, (
        "the legacy request must keep its `verb#id` form, or its answer stops being bound")
    assert ("where:", "d1") in record.calls
    assert not (tmp_path / "request").exists()
    assert not (tmp_path / "requests" / "d1").exists()


def test_a_writers_staging_names_are_never_consumed(tmp_path):
    """`os.replace(tmp, target)` is the sanctioned write: a `.tmp` stage and a hidden name are not requests."""
    requests = tmp_path / "requests"
    requests.mkdir()
    (requests / "r1.tmp").write_bytes(b"state:x")
    (requests / ".hidden").write_bytes(b"state:x")
    record = _Recorder()

    assert capture_server.drain_requests(str(tmp_path), record, now=0.0) == 0
    assert record.calls == []
    assert (requests / "r1.tmp").exists() and (requests / ".hidden").exists()


# -- the intake survives one handler that raises ---------------------------------------------------


def test_one_failing_handler_does_not_eat_its_neighbours(tmp_path):
    _write(tmp_path / "requests" / "a", b"boom:")
    _write(tmp_path / "requests" / "b", b"state:ok")
    dispatched: list[tuple[str, str | None]] = []

    def dispatch(verb: str, request_id: str | None) -> None:
        if verb.startswith("boom"):
            raise RuntimeError("handler exploded")
        dispatched.append((verb, request_id))

    assert capture_server.drain_requests(str(tmp_path), dispatch, now=0.0) == 2
    assert dispatched == [("state:ok", "b")], (
        "the second request was already read off the disk: an exception must not take the rest of the batch down")


# -- the real loop: the measured defect was that `poll` never came back ---------------------------------


def _wait_for(path, timeout: float = 10.0) -> bool:
    pytest.importorskip("PySide6")
    from PySide6.QtTest import QTest

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        QTest.qWait(20)
    return path.exists()


def test_the_real_poll_answers_a_non_utf8_request_and_still_serves_the_next(qapp, tmp_path, monkeypatch):
    """The measured shape was "every later request got no answer" - a byte the decoder refuses ended the timer.

    Here the whole path runs: `install()` on the smallest window that has pages, the real 120 ms `poll`, the real
    `handle`. One request carries `0xb7`; its answer must arrive, and a valid request written after it must be
    answered too.
    """
    from PySide6.QtWidgets import QStackedWidget, QWidget

    class _Window(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.setObjectName("fakeWindow")
            self.stack = QStackedWidget(self)
            self.stack.setObjectName("stackedWidget")
            for index in range(2):
                page = QWidget(self.stack)
                page.setObjectName(f"page{index}")
                self.stack.addWidget(page)

    window = _Window()
    monkeypatch.setenv("RVE_UI_CAPTURE_SERVER", str(tmp_path))
    monkeypatch.setenv("RVE_UI_CAPTURE_DELAY_MS", "0")
    monkeypatch.setenv("RVE_UI_CAPTURE_STEP_MS", "0")
    try:
        capture_server.install(window)
        assert _wait_for(tmp_path / "ready"), "the warm-up walk never finished - the channel was never armed"

        (tmp_path / "request").write_bytes(b"state:noSuchWidget\xb7x#bad1")
        answer = tmp_path / "answers" / "bad1.json"
        assert _wait_for(answer), (
            "a request with an undecodable byte got no answer: this is the measured silence - the loop died on it")
        payload = json.loads(answer.read_text(encoding="utf-8"))
        assert payload["request_id"] == "bad1", "the answer must be bound to the request that asked"
        assert payload["error"] == "widget not found", payload
        assert payload["target"] == "noSuchWidget\ufffdx", (
            f"the undecodable byte must reach the handler as a replacement character, got {payload['target']!r}")

        (tmp_path / "request").write_bytes(b"state:fakeWindow#ok2")
        assert _wait_for(tmp_path / "answers" / "ok2.json"), (
            "the request after the bad one was not served - the intake did not survive")
        second = json.loads((tmp_path / "answers" / "ok2.json").read_text(encoding="utf-8"))
        assert second["kind"] == "state" and second["error"] is None, second
    finally:
        (tmp_path / "quit").write_text("", encoding="utf-8")
        window.close()
