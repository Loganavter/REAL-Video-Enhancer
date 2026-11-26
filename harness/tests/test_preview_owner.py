"""One owner: the eight painter decisions are one table, and the page reads it once per snapshot (plan P2).

Why this file exists. "Who paints this pane" used to be eight predicates on `ProcessTab`
(`_display_paints_pane`, `_display_owns_pane`, `_preview_backend_kind`, `_tail_should_run`, `_display_should_want`,
`_display_can_seek_any_frame` and the two frame gates). Each answered a slice of the same question from the same
handful of raw facts, and the painter races the audit records (two painters `3791f9e8`, the thumbnail `48455ff9`,
the queue hand-back `f17cbfff`, the one-frame loan `b3e4b7b1`) are places where two of them disagreed for a frame.
`preview.owner.owner_of(page)` is now the one answer, and this file is its table:

* **the table** - a page shape in, an `Owner` out (`kind` / `may_paint` / `may_want` / `may_seek_any`), no Qt and no
  window: the six shapes the deleted gates could tell apart, including the two races (a playing tail under an
  attached export block; the queue item owning the pane) and the loan (a held preview asking for one frame);
* **purity** - the decision reads attributes and calls nothing: a pane whose every method raises still decides, and
  the one widget fact it may not carry (visibility) is ANDed on by the page;
* **one owner per snapshot** - `_push_preview_state` builds exactly ONE `Owner` and reads `kind`, the band and the
  reach off it, so a new gate has nowhere to be added but into `owner_of`;
* **no gate left** - the deleted methods' names are absent from the `ProcessTab` class body (parsed, not grepped, so
  a docstring mention cannot pass for a method);
* **the painter feeds the session** - a painted frame settles the page's own press line FIRST and reports to
  `PreviewSession.observe(Painted)` SECOND, so the press telemetry stays byte-identical (plan P2, P4 retires the
  page's landing watch).
"""

from __future__ import annotations

import ast
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.ui.preview.owner import Owner, owner_of
from src.ui.preview.session import Painted
from src.ui.preview.state import PreviewBackendKind, PreviewState

_ROOT = Path(__file__).resolve().parents[2]

#: The five predicates this phase deleted, plus the reach reader that became `may_seek_any`. A name from this list
#: anywhere in the `ProcessTab` class body is a second copy of the decision.
_DELETED_GATES = (
    "_display_paints_pane",
    "_display_owns_pane",
    "_preview_backend_kind",
    "_tail_should_run",
    "_display_should_want",
    "_display_can_seek_any_frame",
)


# --- the page shapes the deleted gates could tell apart ------------------------------------------------


class _Reader:
    """`DisplayReader` as the owner reads it: the block's attachment and nothing else."""

    def __init__(self, attached: bool = True) -> None:
        self.attached = bool(attached)


class _Process:
    """`subprocess.Popen` as the owner reads it: `poll()` is None while the renderer can answer."""

    def __init__(self, alive: bool = True) -> None:
        self.alive = bool(alive)

    def poll(self):
        return None if self.alive else 0


class _NeverCalledPane:
    """The pane: `owner_of` may RETURN it as the surface, but must never ask it anything."""

    def __getattr__(self, name):  # pragma: no cover - any touch is the failure this class exists for
        raise AssertionError(f"owner_of called the pane's {name!r}: the decision is a pure read of the page")


def _page(**fields) -> SimpleNamespace:
    """A page with the raw inputs the owner reads, at the defaults of an idle, unowned pane."""
    values = dict(
        _queue_pane_owns=False,
        _queue_pane_decoder=None,
        _live_tail=None,
        _preview_decoder=None,
        _display=None,
        _feed=None,
        isPreview=False,
        _preview_paused=True,
        _display_seek_pending=False,
        renderProcess=None,
        page_handles=SimpleNamespace(proc=SimpleNamespace(previewLabel=_NeverCalledPane())),
    )
    values.update(fields)
    return SimpleNamespace(**values)


def _owner(**fields) -> Owner:
    return owner_of(_page(**fields))


# (shape name, page fields, kind, may_paint, may_want, may_seek_any)
TABLE = (
    (
        "the queue tab owns the pane (f17cbfff): its own finished file and clock, the feed stages nothing",
        dict(_queue_pane_owns=True, _queue_pane_decoder=object(), _live_tail=object(), _display=_Reader(),
             isPreview=True, _preview_paused=False, renderProcess=_Process(alive=True)),
        PreviewBackendKind.QUEUE_FILE,
        False, False, True,
    ),
    (
        "a live tail over a growing file (a preview, held): the tail is the picture and nothing may want",
        dict(_live_tail=object(), _display=_Reader(), isPreview=True, _preview_paused=True,
             renderProcess=_Process(alive=True)),
        PreviewBackendKind.LIVE_TAIL,
        False, False, True,
    ),
    (
        "a finished preview file: its own decoder answers the transport",
        dict(_preview_decoder=object(), isPreview=True, _preview_paused=False,
             renderProcess=_Process(alive=False)),
        PreviewBackendKind.FINISHED_FILE,
        False, False, False,
    ),
    (
        "an export the display block feeds (3791f9e8's other half): the feed paints and may want",
        dict(_display=_Reader(), isPreview=False, _preview_paused=False, renderProcess=_Process(alive=True)),
        PreviewBackendKind.DISPLAY_FEED,
        True, True, True,
    ),
    (
        "an export with a block AND a playing tail (the two-painter race): the tail is the timeline, the feed stands",
        dict(_display=_Reader(), _live_tail=object(), isPreview=False, _preview_paused=False,
             renderProcess=_Process(alive=True)),
        PreviewBackendKind.LIVE_TAIL,
        False, False, True,
    ),
    (
        "a preview whose block is attached, held and not asking: the tail owns the pane, the block only extends reach",
        dict(_display=_Reader(), isPreview=True, _preview_paused=True, renderProcess=_Process(alive=True)),
        PreviewBackendKind.NONE,
        False, False, True,
    ),
    (
        "the one-frame loan (b3e4b7b1): the same held preview asked for exactly one frame",
        dict(_display=_Reader(), isPreview=True, _preview_paused=True, _display_seek_pending=True,
             renderProcess=_Process(alive=True)),
        PreviewBackendKind.NONE,
        True, True, True,
    ),
    (
        "a held export with a block: the feed paints, but a held picture costs the renderer nothing",
        dict(_display=_Reader(), isPreview=False, _preview_paused=True, renderProcess=_Process(alive=True)),
        PreviewBackendKind.DISPLAY_FEED,
        True, False, True,
    ),
    (
        "nothing at all: no block, no owner, no render",
        dict(),
        PreviewBackendKind.NONE,
        False, False, False,
    ),
    (
        "an export whose block never attached: the tail fallback is its only painter, and nothing may be staged",
        dict(_display=_Reader(attached=False), isPreview=False, _preview_paused=False,
             renderProcess=_Process(alive=True)),
        PreviewBackendKind.NONE,
        False, False, False,
    ),
)


@pytest.mark.parametrize("shape,fields,kind,may_paint,may_want,may_seek_any", TABLE,
                         ids=[row[0] for row in TABLE])
def test_the_table_of_page_shapes(shape, fields, kind, may_paint, may_want, may_seek_any):
    owner = _owner(**fields)
    assert owner.kind is kind, shape
    assert owner.may_paint is may_paint, shape
    assert owner.may_want is may_want, shape
    assert owner.may_seek_any is may_seek_any, shape


def test_the_surface_is_the_pane_exactly_when_the_feed_paints_it():
    """`surface` names where a picture may go: the pane when it may paint, nothing otherwise."""
    pane = object()
    page = _page(_display=_Reader(), _preview_paused=False, renderProcess=_Process(alive=True),
                 page_handles=SimpleNamespace(proc=SimpleNamespace(previewLabel=pane)))
    assert owner_of(page).surface is pane
    page.isPreview = True          # a preview's tail paints: the feed may not hand it a picture
    page._live_tail = object()
    assert owner_of(page).surface is None


def test_the_queue_items_decoder_is_its_clock_and_its_pane_is_the_surface():
    """The queue pane is its own owner: the item's decoder answers the transport and paints the pane there."""
    pane, decoder = object(), object()
    page = _page(_queue_pane_owns=True, _queue_pane_decoder=decoder,
                 page_handles=SimpleNamespace(proc=SimpleNamespace(previewLabel=pane)))
    owner = owner_of(page)
    assert owner.kind is PreviewBackendKind.QUEUE_FILE
    assert owner.clock is decoder, "the object that answers the queue item's transport is its own decoder"
    assert owner.surface is pane, "the queue item paints the pane through that decoder"
    assert owner.may_paint is False, "`may_paint` is the display feed's permission, and the feed paints nothing here"
    assert owner.may_want is False, "the renderer stages nothing for a queue item"
    assert owner.may_seek_any is True, "every frame of a finished item is reachable, even without the block"
    assert owner_of(_page(_queue_pane_owns=True)).clock is None, "an empty field has no clock but still owns the pane"


def test_the_clock_is_the_object_that_answers_the_transport():
    """`clock` is the transport's own object for the kind it named - not a second adapter built here."""
    tail, decoder, reader = object(), object(), _Reader()
    queue_spool = object()
    assert owner_of(_page(_live_tail=tail, isPreview=True)).clock is tail
    assert owner_of(_page(_preview_decoder=decoder, isPreview=True)).clock is decoder
    assert owner_of(_page(_display=reader, _preview_paused=False)).clock is reader
    assert owner_of(_page(_queue_pane_owns=True, _queue_pane_decoder=queue_spool)).clock is queue_spool
    assert owner_of(_page()).clock is None, "nobody answers the transport, so there is no clock"


def test_the_decision_calls_nothing_on_the_pane():
    """The table above already runs every row through `_NeverCalledPane`; this names the property.

    A widget read in here would make the decision untestable without a window and would put the page's visibility and
    frame logic back into the "who paints" answer. `may_want` is the transport's want; the page ANDs visibility on.
    """
    owner = _owner(_display=_Reader(), _preview_paused=False, renderProcess=_Process(alive=True))
    assert owner.may_want is True


# --- one owner per snapshot, and no gate left in the page -----------------------------------------------


class _RecordingController:
    """The controller's three entry points that `_push_preview_state` uses."""

    def __init__(self) -> None:
        self.kinds = []
        self.backends = []
        self.observed = None

    def register_backend(self, kind, backend) -> None:
        self.kinds.append(kind)
        self.backends.append(backend)

    def set_backend(self, kind, backend=None) -> None:
        self.kinds.append(kind)

    def observe(self, **fields) -> None:
        self.observed = fields


class _SnapshotPane:
    """The pane calls `_push_preview_state` makes, with a frame on it."""

    def has_frame(self) -> bool:
        return True

    def view_mode(self) -> str:
        return "single"


def _import_process_tab():
    """`ProcessTab` with the interpreter's missing optional imports stubbed first (the harness's house pattern)."""
    pytest.importorskip("PySide6")
    for name, attrs in (("cpuinfo", {"get_cpu_info": lambda *a, **k: {}}),
                        ("pypresence", {"Presence": object})):
        try:  # pragma: no cover - depends on the interpreter
            __import__(name)
        except ModuleNotFoundError:  # pragma: no cover
            module = types.ModuleType(name)
            for attr, value in attrs.items():
                setattr(module, attr, value)
            sys.modules[name] = module
    from src.ui.ProcessTab import ProcessTab

    return ProcessTab


def _snapshot_tab(controller) -> object:
    tab = object.__new__(_import_process_tab())
    tab._preview_controller = controller
    tab.page_handles = SimpleNamespace(proc=SimpleNamespace(previewLabel=_SnapshotPane()))
    tab.currentRenderOptions = SimpleNamespace(videoFrameCount=120, videoFps=24, interpolateTimes=1)
    tab.max_value = 120
    tab.currentFrame = 12
    tab.fps = 24
    tab.isPreview = True
    tab._preview_paused = True
    tab._press = None
    tab._preview_playhead = 0.5
    tab._live_tail = None
    tab._preview_decoder = None
    tab._queue_pane_decoder = None
    tab.renderProcess = None
    tab._display = None
    tab._feed = None
    return tab


def test_the_page_builds_exactly_one_owner_per_snapshot(monkeypatch):
    """One snapshot, one decision: `_push_preview_state` reads `owner_of` once and hands the result down.

    This is the guard the plan asks for ("a phase that adds a `may_paint` short-circuit must add a row"): a new gate
    added beside the snapshot would either call `owner_of` a second time (this fails) or read the one object.
    """
    import sys as _sys

    page_module = _sys.modules[_import_process_tab().__module__]

    calls: list[object] = []
    real = page_module.owner_of

    def counting(page):
        owner = real(page)
        calls.append(owner)
        return owner

    monkeypatch.setattr(page_module, "owner_of", counting)
    controller = _RecordingController()
    tab = _snapshot_tab(controller)

    tab._push_preview_state()

    assert len(calls) == 1, f"one snapshot, one owner (got {len(calls)})"
    assert controller.kinds == [calls[0].kind, calls[0].kind], (
        "the backend registered and selected is the owner's kind, read off the one decision")
    assert controller.observed is not None
    assert controller.observed["seek_ceiling"] == pytest.approx(0.5), (
        "the reach is the owner's `may_seek_any`, not a second reader of the block")


# --- the queue item's own numbers (plan P3) ---------------------------------------------------------


def test_the_page_clip_duration_is_the_queue_items_own_length():
    """The page's one duration derivation reads the queue item's length while its tab owns the pane."""
    tab = _snapshot_tab(_RecordingController())
    tab._queue_pane_owns = True
    tab._queue_pane_duration = 42.0
    tab._queue_pane_decoder = SimpleNamespace(duration=42.0)
    assert tab._clip_duration() == pytest.approx(42.0)

    tab._queue_pane_owns = False
    assert tab._clip_duration() == pytest.approx(5.0), "the probe's 120 frames / 24 fps once the queue lets go"


def test_a_queue_file_snapshot_reports_its_own_length_and_the_previews_is_untouched():
    """Two shapes on one page: the queue item's snapshot is its own finished file; the preview keeps its numbers."""
    controller = _RecordingController()
    tab = _snapshot_tab(controller)
    tab._queue_pane_owns = True
    tab._queue_pane_decoder = SimpleNamespace(duration=12.5)
    tab._queue_pane_duration = 12.5
    tab._queue_pane_position = 3.0

    tab._push_preview_state()

    assert controller.observed["duration"] == pytest.approx(12.5), "the queue item's own length is the bar's axis"
    assert controller.observed["produced"] == pytest.approx(12.5), "a finished queue item is fully produced"
    assert controller.observed["seek_ceiling"] == pytest.approx(12.5), "every frame of it is reachable"
    assert controller.observed["playhead"] == pytest.approx(3.0), "the thumb is the queue item's picture time"
    queue_state = PreviewState(backend=PreviewBackendKind.QUEUE_FILE,
                               duration=controller.observed["duration"],
                               produced=controller.observed["produced"])
    assert queue_state.finished() is True, "a queue item's band is its whole file, so it is finished"

    # The PREVIEW's own shape on the same page: letting the queue go must not leave its numbers behind.
    tab._queue_pane_owns = False
    tab._queue_pane_duration = 0.0
    tab._preview_playhead = 0.5
    tab._push_preview_state()

    assert controller.observed["duration"] == pytest.approx(5.0), "the probe's 120 frames / 24 fps, not 12.5"
    assert controller.observed["produced"] == pytest.approx(0.5), "frame 12 of 120 = 10 % of 5 s"
    preview_state = PreviewState(duration=controller.observed["duration"],
                                 produced=controller.observed["produced"])
    assert preview_state.finished() is False, "a render at 10 % is not finished, whatever the queue item was"


def test_no_gate_method_survives_in_the_page():
    """The eight predicates are DELETED, not wrapped: the class body carries none of their names.

    Parsed rather than grepped: a docstring may still name a deleted predicate (it explains the history), a method may
    not - and a renamed wrapper with the old body would show up as a new name holding the old decision, which the
    table above would then have to grow a row for.
    """
    tree = ast.parse((_ROOT / "src" / "ui" / "ProcessTab.py").read_text(encoding="utf-8"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ProcessTab"]
    assert classes, "the page class must be in the module it is expected in"
    methods = {node.name for node in classes[0].body if isinstance(node, ast.FunctionDef)}
    seized = sorted(methods & set(_DELETED_GATES))
    assert seized == [], f"the deleted gates are still methods of ProcessTab: {seized}"


# --- the painter reports the picture to the session (plan P2's seam into P1) ---------------------------


def _frame_slot_tab(pts: float, index: int, order: list[str]):
    """The page `_on_preview_frame` needs, with the press settle and the session both recording."""
    ProcessTab = _import_process_tab()
    painted: list[tuple] = []

    class _Session:
        def observe(self, event) -> None:
            order.append("session")
            painted.append(event)

    class _Pane:
        def set_frame(self, image, **kwargs) -> None:
            pass

        def set_frame_planes(self, *args, **kwargs) -> None:
            pass

        def set_progress(self, **kwargs) -> None:
            pass

        def shown(self):
            return ("frame", index)

    tab = object.__new__(ProcessTab)
    tab.page_handles = SimpleNamespace(proc=SimpleNamespace(previewLabel=_Pane()))
    tab._display = None
    tab._feed = None
    tab.isPreview = True
    tab._preview_paused = False
    tab._display_seek_pending = False
    tab._live_tail = SimpleNamespace(media_origin=None)
    tab._preview_decoder = None
    tab._preview_playhead = 0.0
    tab._preview_duration = 1.0
    tab.currentRenderOptions = SimpleNamespace(videoFps=24.0)
    tab.renderProcess = None
    tab.currentFrame = 1
    tab.max_value = 1
    tab._seek_watch = float(pts)
    tab._seek_pressed_at = time.perf_counter()
    tab._seek_target_frame = index
    tab._press = None
    tab._tail_origin_logged = True
    tab._preview_session = lambda: _Session()
    tab._push_preview_state = lambda: None
    return tab, painted


def test_a_painted_frame_settles_the_page_line_first_and_the_session_second(monkeypatch):
    """The one ordering that matters: the press line is the page's, the session's record comes after it.

    The page's settle block (`seek: press wanted ... landed at ...`) already carries the press's identity, its
    generation and its turnaround; feeding the session BEFORE it would let the session close the ticket first and the
    page's line would then be a second, thinner record of the same press (P1's "one line per press").
    """
    pytest.importorskip("PySide6")
    order: list[str] = []
    monkeypatch.setattr("src.ui.video.telemetry.note_seek", lambda ms, **kw: order.append("page"))
    tab, painted = _frame_slot_tab(pts=1.5, index=7, order=order)
    frame = SimpleNamespace(buffer=bytes(12), width=2, height=2, stride=6, format="rgb24", pts=1.5, index=7)

    tab._on_preview_frame(frame)

    assert order == ["page", "session"], f"the page's press line must settle first (got {order})"
    assert len(painted) == 1, "the painter reports the picture exactly once per painted frame"
    assert isinstance(painted[0], Painted)
    assert painted[0].media_time == pytest.approx(1.5)
    assert painted[0].index == 7, "the session names the picture the way the producer numbered it"
