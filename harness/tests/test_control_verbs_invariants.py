"""The control channel's three invariants: the JOB finishes a render, the DISPATCH owns the modal gate, one answer per
request.

Why one file (2026-11-27, the incident that produced the audit): the three findings live in different corners of
`src/ui/capture_server.py` and they are one contract - a verb answers, the answer belongs to its request, and nothing a
verb does may park the GUI thread. Read together they are the channel; read separately each one looks like a local bug.

    F1  `render:` reported `finished: true` 250-800 ms after the press, from `startRenderButton.isEnabled()` - a button
        `ProcessTab.RENDER_CONTROLS` keeps usable in `idle`, `running` AND `paused` - so a click that did nothing read
        as a finished render and `harness/scripts/ui_render_gate.py` failed every real one.
    F2  The overwrite modal was refused in ONE verb (`start_render_job`). `click:` (a general presser, Start included)
        and `render:`'s own press reached the same button unguarded, and `ProcessTab.run` -> `checkForOverwrite` ->
        `askForOverwrite` -> `confirm` parked the GUI thread: every later verb timed out.
    F5  `_finish` read a single mutable `state["request_id"]`, so a verb that answers from a timer wrote its answer under
        whichever request was current THEN; `capture` never answered an id at all.

The channel is run FOR REAL here: `install()` on a minimal window, the shipped 120 ms `poll`/`drain_requests`/`handle`
and the verb's own `QTimer` chain, driven through `requests/<id>` and judged at `answers/<id>.json` - the file a client
actually reads. The doubles supply only the window attributes the live code touches.

Red-before (base commit `d0e7fa7d`, quoted in `DONE.md`): every test below was written and run against the pre-fix tree
FIRST. What it answered there:

* `test_render_answers_the_overwrite_refusal_instead_of_a_finished_job` -> `{'started': True, 'finished': True,
  'error': None, 'output': ...}` and the Start control had been pressed once (`presses == 1`).
* `test_a_job_that_died_is_finished_false` -> `{'finished': True, 'error': None}` 250 ms in, no exit code anywhere.
* `test_a_start_click_that_did_nothing_is_a_refusal` -> `{'started': True, 'finished': True, 'error': None}`.
* `test_a_click_that_would_open_the_overwrite_question_is_refused_at_dispatch` -> no refusal: `error` was either null
  or the click path's own exception.
* `test_each_answer_lands_under_the_request_that_asks` -> `answers/q1.json` never appeared, and `answers/s2.json`
  (the `state:` probe's file) was overwritten by the `queue:` payload.
* `test_the_answer_id_is_not_read_from_a_shared_slot` -> `'state["request_id"]'` was still in `install()`.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import sys
import time
import types
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).resolve().parents[2])
# The app and the backend both live in a `src` namespace package; the suite's conftest puts the backend first. The
# channel's own overwrite gate imports the app's `ProcessTab` (its sentence and its predicate), so the app root goes
# first for this module - the preamble `test_control_start_refusal.py` uses.
_ORIGINAL_SYS_PATH = list(sys.path)
if _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)
sys.modules.pop("src.constants", None)

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

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QProgressBar, QPushButton, QStackedWidget, QWidget  # noqa: E402

from src.ui.ProcessTab import OVERWRITE_QUESTION  # noqa: E402
from src.ui import capture_server  # noqa: E402

sys.path[:] = _ORIGINAL_SYS_PATH


# --- the doubles: only the attribute paths the live code touches ---------------------------------


class _Process:
    """`subprocess.Popen` as the completion rule reads it: `poll()` is the exit code, None while the child is alive."""

    def __init__(self, code: int | None = None) -> None:
        self.code = code

    def poll(self):
        return self.code


class _Row:
    """One render-queue row: `outputPath` (what the modal predicate reads) and `output` (what `render:` reads)."""

    def __init__(self, output: str, input_file: str = "") -> None:
        self.output = output
        self.outputPath = output
        self.inputFile = input_file


class _Queue:
    def __init__(self, rows=None) -> None:
        self.queue = list(rows or [])

    def getQueue(self):
        return list(self.queue)


class _Field:
    def __init__(self) -> None:
        self._text = ""

    def setText(self, text) -> None:
        self._text = str(text)

    def text(self) -> str:
        return self._text


class _Settings:
    def __init__(self) -> None:
        self.settings: dict = {}

    def writeSetting(self, key, value) -> None:
        self.settings[key] = value


class _Probe:
    """`window._batchProbe`: while it is "running" the queue verb reschedules its add instead of adding."""

    def __init__(self, running_for: int = 4) -> None:
        self.running_for = running_for
        self.calls = 0

    def isRunning(self) -> bool:
        self.calls += 1
        return self.calls <= self.running_for


class _Check:
    """An option box: `isChecked` for `select_a_model`, `setChecked` for the queue verb's "no model" fallback."""

    def __init__(self, checked: bool = False) -> None:
        self.checked = bool(checked)

    def isChecked(self) -> bool:
        return self.checked

    def setChecked(self, value) -> None:
        self.checked = bool(value)


class _Job:
    """A render as the APP produces it: a child that lives, writes its file, exits, and lets the glue go idle.

    The test never waits on a real backend: `press()` plants the child the completion rule must follow, and `_end()`
    is the app's own order of events (`wait()` returns, the file holds bytes, `guiChangesOnRenderCompletion` runs) on
    a `QTimer` so it lands between two polls of the verb.
    """

    def __init__(self, window, output: str, *, exit_code: int = 0, writes: bool = True,
                 completes: bool = True, duration_ms: int = 120) -> None:
        self.window = window
        self.output = output
        self.exit_code = exit_code
        self.writes = writes
        self.completes = completes
        self.duration_ms = duration_ms
        self.child = _Process(None)

    def press(self) -> None:
        tab = self.window.tab
        tab.renderProcess = self.child
        tab._render_state = "running"
        QTimer.singleShot(self.duration_ms, self._end)

    def _end(self) -> None:
        if self.writes:
            with open(self.output, "wb") as output_file:
                output_file.write(b"x" * 2048)
        self.child.code = self.exit_code
        if self.completes:
            self.window.tab._render_state = "idle"


class _Window(QWidget):
    """The smallest window `install()` can walk and the verbs can read: a page stack, a General page and the transport."""

    def __init__(self, *, rows=None, job: _Job | None = None, probe: _Probe | None = None) -> None:
        super().__init__()
        self.setObjectName("invariantWindow")
        self.resize(420, 320)
        self.stack = QStackedWidget(self)
        self.stack.setObjectName("stackedWidget")
        self.stack.setGeometry(0, 0, 420, 320)
        for index in range(2):
            page = QWidget(self.stack)
            page.setObjectName(f"page{index}")
            self.stack.addWidget(page)

        # The transport, inside a page panel: the panel is the shape `click:` uses to reach it by coordinates.
        self.container = QWidget(self.stack)
        self.container.setObjectName("processSettingsContainer")
        self.container.setGeometry(0, 0, 200, 60)
        self.start = QPushButton("start", self.container)
        self.start.setObjectName("startRenderButton")
        self.start.setGeometry(5, 5, 80, 24)
        self.presses = 0
        self.job = job
        self.start.clicked.connect(self._on_start)

        # A control that is NOT the transport, so "an unrelated click is not refused" has something to click.
        self.other = QPushButton("play", self.stack)
        self.other.setObjectName("preview_play")
        self.other.setGeometry(300, 250, 60, 24)

        self.progress = QProgressBar(self.container)
        self.progress.setObjectName("progressBar")
        self.input_field = _Field()
        self.output_field = _Field()
        self.add_button = _AddButton(self)
        self.general = types.SimpleNamespace(
            inputFileText=self.input_field,
            outputFileText=self.output_field,
            backendComboBox=None,
            addToRenderQueueButton=self.add_button,
            upscaleCheckBox=_Check(True),
            interpolateCheckBox=_Check(False),
            deblurCheckBox=_Check(False),
            denoiseCheckBox=_Check(False),
            decompressCheckBox=_Check(False),
            upscaleModelComboBox=None,
        )
        self.queue_widget = types.SimpleNamespace(count=lambda: len(self.renderQueue.queue))
        self.row_to_add: _Row | None = None
        self.tab = types.SimpleNamespace(renderProcess=None, _render_state="idle", status="")
        # The channel reads `window.processTab`; the test also wants a shorter name for the same object.
        self.processTab = self.tab
        self.renderQueue = _Queue(rows)
        self.page_handles = types.SimpleNamespace(
            menu=types.SimpleNamespace(stackedWidget=self.stack),
            proc=types.SimpleNamespace(startRenderButton=self.start, progressBar=self.progress),
            general=self.general,
            queue=types.SimpleNamespace(renderQueueListWidget=self.queue_widget),
            # `page_capture.stabilise` (the capture verb's settle step) pins the encoder by text; a window without that
            # handle is a window that cannot be photographed by this channel.
            encoder=types.SimpleNamespace(encoder=None),
        )
        self.settings = _Settings()
        self.videoLength = 5
        self._batchProbe = probe
        self.loads: list[str] = []
        self.adds = 0

    def _on_start(self) -> None:
        self.presses += 1
        if self.job is not None:
            self.job.press()

    def loadVideo(self, path, multi_file: bool = False) -> None:
        self.loads.append(str(path))

    def addToRenderQueue(self) -> None:
        self.adds += 1
        if self.row_to_add is not None:
            self.renderQueue.queue.append(self.row_to_add)


class _AddButton:
    def __init__(self, window: _Window) -> None:
        self.window = window

    def click(self) -> None:
        self.window.addToRenderQueue()


# --- the channel, driven through the files a client writes ---------------------------------------


def _wait_for(path: Path, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        QTest.qWait(20)
    return path.exists()


def _channel(window: _Window, tmp_path: Path, monkeypatch) -> Path:
    """Arm the REAL channel on `window`: a directory mode (so `install` walks and then polls) in `tmp_path`."""
    monkeypatch.setenv("RVE_UI_CAPTURE_SERVER", str(tmp_path))
    monkeypatch.setenv("RVE_UI_CAPTURE_DELAY_MS", "0")
    monkeypatch.setenv("RVE_UI_CAPTURE_STEP_MS", "0")
    window.show()
    QTest.qWait(20)
    capture_server.install(window)
    assert _wait_for(tmp_path / "ready"), "the channel never armed - `install()` did not reach `ready`"
    return tmp_path


def _ask(root: Path, request_id: str, verb: str, timeout: float = 20.0) -> dict:
    """Publish one request the sanctioned way (`write tmp; os.replace`) and read the answer bound to ITS id."""
    requests = root / "requests"
    requests.mkdir(exist_ok=True)
    stage = requests / f".{request_id}.tmp"
    stage.write_text(verb, encoding="utf-8")
    os.replace(stage, requests / request_id)
    answer = root / "answers" / f"{request_id}.json"
    if not _wait_for(answer, timeout):
        pytest.fail(f"no answer for {request_id} ({verb!r}) within {timeout} s - the verb said nothing")
    return json.loads(answer.read_text(encoding="utf-8"))


def _stop(root: Path, window: QWidget) -> None:
    (root / "quit").write_text("", encoding="utf-8")
    window.close()


@pytest.fixture
def window_factory():
    made: list[QWidget] = []

    def make(**kwargs) -> _Window:
        window = _Window(**kwargs)
        made.append(window)
        return window

    yield make
    for window in made:
        window.close()


# --- F1 + F2: `render:` over a would-be overwrite answers the refusal, not a finished job ----------


def test_render_answers_the_overwrite_refusal_instead_of_a_finished_job(qapp, window_factory, tmp_path, monkeypatch):
    """The mandate's row: the queue's output exists, so the press would open the modal - refuse it, do not press.

    Base commit: `presses == 1` and `{'started': True, 'finished': True, 'error': None, 'output_bytes': 2048}`. On a
    leased display that press enters `ProcessTab.run`'s modal and the channel stops answering for good.
    """
    existing = tmp_path / "render-out.mkv"
    existing.write_bytes(b"old render")
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")

    window = window_factory()
    window.row_to_add = _Row(str(existing))          # `Add to Render Queue` adds the row whose output exists
    root = _channel(window, tmp_path, monkeypatch)

    payload = _ask(root, "r1", f"render:{clip}")

    assert payload["kind"] == "render", payload
    assert payload["finished"] is False, ("a click that cannot start a job must not read as a finished render: "
                                          "F1's rule is the job's completion, not the transport's look")
    assert payload["started"] is False, "nothing was started - the press was refused"
    assert payload["error"] == OVERWRITE_QUESTION, "the refusal must carry the dialog's OWN sentence"
    assert payload["output"] == str(existing)
    assert payload["refused"] == "overwrite", "a refusal names its reason instead of leaving a hole"
    assert window.presses == 0, ("the press is what runs `ProcessTab.run` synchronously and parks the GUI thread in "
                                 "the modal")
    _stop(root, window)


def test_a_job_that_died_is_finished_false(qapp, window_factory, tmp_path, monkeypatch):
    """F1: `finished` needs the backend child's exit, its file, and the app's completion - never an enabled button.

    Base commit: `{'started': True, 'finished': True, 'error': None}` 250 ms in, from a button the app keeps enabled
    in every render state.
    """
    output = tmp_path / "job-out.mkv"
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")

    window = window_factory()
    window.row_to_add = _Row(str(output))
    window.job = _Job(window, str(output), exit_code=1, writes=False, duration_ms=120)
    root = _channel(window, tmp_path, monkeypatch)

    payload = _ask(root, "r2", f"render:{clip}")

    assert payload["finished"] is False, "the backend exited with an error: this is not a finished render"
    assert payload["started"] is True, "the press DID start a job; the job failed"
    assert "exited with code 1" in (payload["error"] or ""), (
        f"the answer must carry the child's exit code, got {payload['error']!r}")
    _stop(root, window)


def test_a_start_click_that_did_nothing_is_a_refusal(qapp, window_factory, tmp_path, monkeypatch):
    """F1: `started` is set BEFORE the click, so a press the app ignored used to answer `finished: true`.

    Base commit: `{'started': True, 'finished': True, 'error': None}` in ~0.8 s with no output file at all.
    """
    output = tmp_path / "nothing-out.mkv"
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")

    window = window_factory(job=None)                 # the press plants no child: a click that did nothing
    window.row_to_add = _Row(str(output))
    root = _channel(window, tmp_path, monkeypatch)

    payload = _ask(root, "r3", f"render:{clip}", timeout=30.0)

    assert payload["finished"] is False
    assert payload["started"] is False, "no child ever appeared and the glue stayed idle: nothing started"
    assert "refused" in (payload["error"] or ""), (
        f"a press the app ignored is the app's refusal, answered with its own words; got {payload['error']!r}")
    assert not output.exists(), "no job ran, so there is no output"
    _stop(root, window)


def test_a_render_that_really_ran_is_finished_with_its_output(qapp, window_factory, tmp_path, monkeypatch):
    """The other side of F1: when the job DOES run, the answer waits for it and names its file."""
    output = tmp_path / "real-out.mkv"
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")

    window = window_factory()
    window.row_to_add = _Row(str(output))
    window.job = _Job(window, str(output), exit_code=0, writes=True, duration_ms=500)
    root = _channel(window, tmp_path, monkeypatch)

    payload = _ask(root, "r4", f"render:{clip}")

    assert payload["finished"] is True, payload
    assert payload["started"] is True and payload["error"] is None
    assert payload["output"] == str(output)
    assert payload.get("output_bytes", 0) > 0, "a finished render names the bytes it wrote"
    _stop(root, window)


# --- F2: the modal gate belongs to the dispatch ---------------------------------------------------


def test_a_click_that_would_open_the_overwrite_question_is_refused_at_dispatch(qapp, window_factory, tmp_path,
                                                                                monkeypatch):
    """`click:` is a general presser: Start included. The gate must answer before any delivery.

    Base commit: no refusal - the click path ran and `presses` became 1 (on a display, that is `ProcessTab.run` and
    the modal; the audit's incident is one `click:startRenderButton@x,y`).
    """
    existing = tmp_path / "queued-out.mkv"
    existing.write_bytes(b"old")
    window = window_factory(rows=[_Row(str(existing))])
    root = _channel(window, tmp_path, monkeypatch)

    payload = _ask(root, "c1", "click:startRenderButton@10,10")

    assert payload["kind"] == "click", payload
    assert payload["error"] == OVERWRITE_QUESTION
    assert payload["refused"] == "overwrite"
    assert payload["output"] == str(existing)
    assert window.presses == 0, "the press never happened: the gate answered first"
    assert not payload.get("delivered"), "the answer must come before any event is delivered to the button"
    _stop(root, window)


def test_a_click_on_a_container_that_holds_start_is_refused_too(qapp, window_factory, tmp_path, monkeypatch):
    """`click:processSettingsContainer@x,y` can land on the button inside it; a point-based gate cannot know, so the
    ancestor form is refused as well (the refusal is recoverable, a wedged GUI thread is not)."""
    existing = tmp_path / "queued-out.mkv"
    existing.write_bytes(b"old")
    window = window_factory(rows=[_Row(str(existing))])
    root = _channel(window, tmp_path, monkeypatch)

    payload = _ask(root, "c2", "click:processSettingsContainer@10,10")

    assert payload["error"] == OVERWRITE_QUESTION and payload["refused"] == "overwrite"
    assert window.presses == 0
    _stop(root, window)


def test_a_click_that_cannot_reach_start_is_not_refused(qapp, window_factory, tmp_path, monkeypatch):
    """The gate must not over-refuse: an ordinary click on another control stays a click, output on disk or not."""
    existing = tmp_path / "queued-out.mkv"
    existing.write_bytes(b"old")
    window = window_factory(rows=[_Row(str(existing))])
    root = _channel(window, tmp_path, monkeypatch)

    payload = _ask(root, "c3", "click:preview_play@10,10")

    assert payload["kind"] == "click", payload
    assert payload.get("refused") is None, f"an unrelated click was refused: {payload}"
    assert payload["error"] != OVERWRITE_QUESTION
    _stop(root, window)


def test_a_running_render_is_not_refused_by_the_gate(qapp, window_factory, tmp_path, monkeypatch):
    """In `running`/`paused` the same button is pause/resume: no `run()`, no `checkForOverwrite`, no modal."""
    existing = tmp_path / "queued-out.mkv"
    existing.write_bytes(b"old")
    window = window_factory(rows=[_Row(str(existing))])
    window.tab._render_state = "running"
    root = _channel(window, tmp_path, monkeypatch)

    payload = _ask(root, "c4", "click:startRenderButton@10,10")

    assert payload.get("refused") is None, "a pause press opens no question and must not be refused"
    assert payload["error"] != OVERWRITE_QUESTION
    _stop(root, window)


def test_the_channel_owns_no_second_copy_of_the_overwrite_predicate():
    """One predicate, one owner: `would_ask_for_overwrite`/`overwrite_prompt_path` live in `ProcessTab` only."""
    source = inspect.getsource(capture_server)
    assert "def would_ask_for_overwrite" not in source
    assert "def overwrite_prompt_path" not in source
    assert "os.path.isfile" not in inspect.getsource(capture_server.start_press_refusal), (
        "the dispatch gate must ask the predicate, not restate `isfile`")


# --- F5: one answer per request ------------------------------------------------------------------


def test_each_answer_lands_under_the_request_that_asks(qapp, window_factory, tmp_path, monkeypatch):
    """`queue:` answers from a timer; the id it writes under is ITS OWN, not the request that arrived meanwhile.

    Base commit: `answers/q1.json` never appeared and `answers/s2.json` - the `state:` probe's own file - was
    overwritten by the queue payload (the audit's observed C6b: "the answer for q1 was written to answers/other.json
    with request_id: other").
    """
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    window = window_factory(probe=_Probe(running_for=4))
    window.row_to_add = _Row(str(tmp_path / "queued-out.mkv"), input_file=str(clip))
    root = _channel(window, tmp_path, monkeypatch)

    requests = root / "requests"
    requests.mkdir(exist_ok=True)
    (requests / "q1").write_text(f"queue:{clip}", encoding="utf-8")
    QTest.qWait(200)                                 # one tick: `q1` dispatched, its add deferred by the probe
    (requests / "s2").write_text("state:invariantWindow", encoding="utf-8")

    state_answer = _ask_wait(root, "s2", 10.0)
    assert state_answer["kind"] == "state", state_answer

    queue_answer = _ask_wait(root, "q1", 10.0)
    assert queue_answer["request_id"] == "q1", f"the queue answer is bound to another request: {queue_answer}"
    assert queue_answer["kind"] == "queue" and queue_answer["error"] is None, queue_answer
    assert queue_answer["rows"] == 1, f"the queue add did not run: {queue_answer}"

    assert state_answer == _ask_wait(root, "s2", 1.0), "the state probe's answer was overwritten by another request"
    _stop(root, window)


def _ask_wait(root: Path, request_id: str, timeout: float) -> dict:
    """Read an answer a test published itself (the id is already on disk)."""
    answer = root / "answers" / f"{request_id}.json"
    if not _wait_for(answer, timeout):
        pytest.fail(f"no answer file for {request_id!r} within {timeout} s")
    return json.loads(answer.read_text(encoding="utf-8"))


def test_capture_answers_its_request_id(qapp, window_factory, tmp_path, monkeypatch):
    """`capture:` wrote `done` and no answer at all: a client publishing `requests/<id>` waited for ever."""
    window = window_factory()
    root = _channel(window, tmp_path, monkeypatch)

    payload = _ask(root, "cap1", "nav0")

    assert payload["kind"] == "capture", payload
    assert payload["request_id"] == "cap1"
    assert payload["pages"] == ["nav0"]
    assert (tmp_path / payload["stamp"] / "nav0.png").exists(), "the capture still writes its pictures"
    assert json.loads((root / "done").read_text(encoding="utf-8"))["stamp"] == payload["stamp"], (
        "the legacy `done` mailbox keeps its shape for `ui_live.py shot`")
    _stop(root, window)


def test_the_answer_id_is_not_read_from_a_shared_slot():
    """The mechanism, pinned: no verb answers under "whichever request is current when the timer fires"."""
    source = inspect.getsource(capture_server.install)
    assert 'state["request_id"]' not in source, (
        "a single mutable slot is read at answer time; a deferred verb must carry its OWN id")
    for verb in ("queue", "render", "preview", "capture"):
        signature = re.search(rf"def {verb}\(([^)]*)\)", source)
        assert signature is not None, f"{verb} is gone from the channel"
        assert "request_id" in signature.group(1), (
            f"`{verb}` answers from a timer and must bind the id the dispatch handed it")
