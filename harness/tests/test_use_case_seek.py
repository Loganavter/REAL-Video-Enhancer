"""The seek/transport/job use cases, proven with a namespace owner (no window, no GPU, no render).

Why these pins exist. The three modules under test were carved out of a 4,193-line tab class, and the split is only
safe if the behaviour moved with the bodies. Two 2026-11-27 fixes live inside those bodies and are pinned here by
effect, not by comment:

* a REFUSED strip press is a no-op (audit §6.3, `b33df4c1`) - the result generation and the compare register are
  untouched, and an accepted press mints the ONE token exactly once;
* the decoder's picture arrival writes the compare's `arrivals` counter (audit §6 / `rules.md` §6).

Every extracted function also has one test that fails if its body is replaced by `pass`, so the delegation cannot go
silently hollow.

The owner is a `SimpleNamespace` carrying only the attributes the function reads, plus stub methods for the helpers
the tab keeps (the shape of `test_display_wiring.py`'s `_tab()` helper, without constructing a page).
"""

from __future__ import annotations

import pathlib
import sys
import types
from types import SimpleNamespace

import pytest

_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
_ORIGINAL_SYS_PATH = list(sys.path)
if _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)
# The app and the backend both live in a `src` namespace package; the app's `src.constants` is the one the use cases
# and `ProcessTab` resolve (same pattern as `test_display_wiring.py`).
sys.modules.pop("src.constants", None)

# The app's optional runtime imports: stub them ONLY when the interpreter does not have them. The use cases import
# `src.Util` lazily (for the log), so a namespace-owner call reaches these.
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

from PySide6.QtCore import QThread  # noqa: E402

from src.ui.preview.use_cases import job, seek, transport  # noqa: E402

sys.path[:] = _ORIGINAL_SYS_PATH


# --- the owner and its collaborators ---------------------------------------------------------------

class _Register:
    """The compare register's two facts here: what was offered, and that a clear was asked for."""

    def __init__(self, held: int = 0) -> None:
        self.held = held
        self.offers: list[tuple] = []
        self.clears = 0

    def __len__(self) -> int:
        return self.held

    def clear(self) -> None:
        self.clears += 1
        self.held = 0

    def offer(self, moment, image, origin, *, generation):
        self.offers.append((float(moment), origin, int(generation)))
        self.held += 1
        return object()          # an arrival, as `SourceRegister.offer` returns a picture handle


class _Ticket:
    def __init__(self) -> None:
        self.route = None
        self.refusals: list[str] = []
        self.open = True

    def decide(self, **fields) -> None:
        self.route = fields.get("route")

    def refused(self, reason: str) -> None:
        self.refusals.append(reason)
        self.open = False

    def watch(self, seconds) -> None:
        pass


class _Session:
    def __init__(self) -> None:
        self.tickets: list[_Ticket] = []
        self.watches: list[float] = []

    def press(self, **_fields) -> _Ticket:
        ticket = _Ticket()
        self.tickets.append(ticket)
        return ticket

    def watch(self, seconds) -> None:
        self.watches.append(float(seconds))


def _owner(**overrides) -> SimpleNamespace:
    """The tab shape the three use cases read: no window, no page, only the attributes they touch."""
    owner = SimpleNamespace(
        page_handles=SimpleNamespace(proc=SimpleNamespace(previewLabel=None)),
        isPreview=False,
        _preview_paused=True,
        _preview_decoder=None,
        _queue_pane_decoder=None,
        _live_tail=None,
        _display=None,
        renderProcess=None,
        _result_generation=0,
        _source_register=_Register(),
        currentFrame=0,
        max_value=0,
        _press=None,
        _press_session=_Session,
        _clip_duration=lambda owner=None: 0.0,
        _live_produced_fraction=lambda: 0.0,
    )
    owner.__dict__.update(overrides)
    return owner


class _Chain:
    def __init__(self) -> None:
        self.seeks: list[float] = []
        self.wants: list[bool] = []

    def request_seek(self, seconds: float) -> int:
        self.seeks.append(float(seconds))
        return 4

    def set_wanted(self, wanted: bool) -> None:
        self.wants.append(bool(wanted))


class _Process:
    def __init__(self, alive: bool = True) -> None:
        self._alive = alive

    def poll(self):
        return None if self._alive else 0


# --- the refused press is a no-op (§6.3) ----------------------------------------------------------

def test_a_refused_press_leaves_the_generation_and_the_register_untouched():
    """The preview has no usable block, so the press cannot be served; nothing identity-bearing may move.

    Before the fix the bump ran at the top of the handler, so a refusal took generation 7 -> 8 and the register
    3 -> 0 (audit §6.3, probe B), and every source offer was then refused as the wrong generation.
    """
    register = _Register(held=3)
    session = _Session()
    bumped = []

    def next_generation():
        bumped.append(1)
        return 99

    tab = _owner(
        isPreview=True, _result_generation=7, _source_register=register,
        _press_session=lambda: session, _next_result_generation=next_generation,
    )

    seek.on_preview_seek(tab, 0.4)

    assert tab._result_generation == 7, "a refused press must not mint a token"
    assert bumped == [], "the generation mint was reached by a refusal"
    assert register.clears == 0 and len(register) == 3, "a refused press destroyed the compare register"
    assert session.tickets and session.tickets[-1].refusals == ["preview-without-a-block"]


def test_an_accepted_press_mints_the_token_exactly_once_per_press():
    """A drag emits many presses; each accepted one mints ONE token, and no press mints more than one."""
    session = _Session()
    bumped = []

    def next_generation():
        bumped.append(1)
        tab._result_generation += 1
        return tab._result_generation

    tail = SimpleNamespace(produced_seconds=5.0, seeks=[], seek=lambda target: tail.seeks.append(float(target)))

    def live_produced_fraction() -> float:
        return 0.5

    tab = _owner(
        _live_tail=tail,
        _result_generation=0,
        _source_register=_Register(held=2),
        _press_session=lambda: session,
        _next_result_generation=next_generation,
        _clip_duration=lambda owner=None: 10.0,
        _live_timeline=lambda edge: 10.0,
        _live_produced_fraction=live_produced_fraction,
        currentFrame=50,
        max_value=100,
    )

    for fraction in (0.1, 0.2, 0.3):
        seek.on_preview_seek(tab, fraction)

    assert len(bumped) == 3, "one mint per accepted press (a drag is three presses, not one and not six)"
    assert tab._result_generation == 3
    assert [ticket.route for ticket in session.tickets] == ["tail", "tail", "tail"]
    assert tail.seeks == pytest.approx([1.0, 2.0, 3.0])
    assert session.watches == pytest.approx([1.0, 2.0, 3.0])


# --- the display seek refuses without an answerer, writes nothing ---------------------------------

def test_request_display_seek_returns_false_and_writes_nothing_without_a_process_or_a_clip():
    chain = _Chain()

    no_block = _owner(_display=None)
    assert seek.request_display_seek(no_block, 0.5, hold=True) is False

    dead = _owner(_display=SimpleNamespace(attached=True), renderProcess=_Process(alive=False),
                  _display_chain=lambda: chain)
    assert seek.request_display_seek(dead, 0.5, hold=True) is False

    unknown_clip = _owner(_display=SimpleNamespace(attached=True), renderProcess=_Process(alive=True),
                          _clip_duration=lambda owner=None: 0.0, _display_chain=lambda: chain)
    assert seek.request_display_seek(unknown_clip, 0.5, hold=True) is False

    assert chain.seeks == [], "a request nobody can answer must not touch the seek slot"
    assert chain.wants == []
    assert getattr(unknown_clip, "_seek_generation", None) is None
    assert getattr(unknown_clip, "_display_seek_pending", False) is False


def test_request_display_seek_asks_the_chain_and_parks_a_held_pane():
    """The accepted request writes the newest-wins slot and the generation; a held pane asks for that one frame."""
    chain = _Chain()
    tab = _owner(
        _display=SimpleNamespace(attached=True),
        renderProcess=_Process(alive=True),
        _clip_duration=lambda owner=None: 10.0,
        _display_chain=lambda: chain,
        _preview_paused=True,
        _seek_pressed_at=1.0,
    )

    assert seek.request_display_seek(tab, 0.5, hold=True) is True

    assert chain.seeks == [5.0], "the share must become seconds once, against the clip"
    assert chain.wants == [True], "a held pane must ask the renderer for exactly the requested frame"
    assert tab._seek_generation == 4
    assert tab._display_seek_pending is True
    assert tab._preview_playhead == pytest.approx(5.0)


# --- the pause toggle must not drop an in-flight request (audit §6.8) -----------------------------

@pytest.mark.xfail(strict=True, reason=(
    "audit §6.8 is still open: `_on_preview_pause_toggled` clears `_display_seek_pending` on ANY transport change, "
    "so a pause during an in-flight display seek drops the one-frame paint permission `owner_of` derives from it "
    "and the press waits for its deadline instead of showing the requested frame."))
def test_a_pause_toggle_leaves_an_in_flight_display_request_pending():
    pending: list[bool] = []
    tab = _owner(
        _preview_paused=False,
        _display_seek_pending=True,
        _playback_tail=False,
        currentRenderOptions=None,
        _stop_live_tail=lambda: None,
        _start_live_tail=lambda path: None,
        _ensure_live_tail=lambda: None,
        _resume_tail_at_the_pane=lambda tail: None,
        _update_display_wanted=lambda: pending.append(True),
        _sync_preview_transport=lambda: None,
    )

    transport.on_preview_pause_toggled(tab, True)

    assert tab._display_seek_pending is True, "the request that is in flight was cleared by a pause"


# --- one pass-replacement test per extracted function ---------------------------------------------

class _Button:
    def __init__(self) -> None:
        self.checked = []
        self.tooltips: list[str] = []
        self.enabled: list[bool] = []

    def thread(self):
        return QThread.currentThread()

    def setChecked(self, value, **_kwargs) -> None:      # noqa: N802 - Qt spelling
        self.checked.append(bool(value))

    def setToolTip(self, text: str) -> None:             # noqa: N802 - Qt spelling
        self.tooltips.append(str(text))

    def setEnabled(self, enabled: bool) -> None:         # noqa: N802 - Qt spelling
        self.enabled.append(bool(enabled))


def test_sync_preview_transport_paints_the_button_and_its_reach(qapp):
    button = _Button()
    pushes = []
    tab = _owner(
        page_handles=SimpleNamespace(proc=SimpleNamespace(previewPauseButton=button, previewLabel=None)),
        _preview_paused=True,
        renderProcess=_Process(alive=True),
        _push_preview_state=lambda: pushes.append(1),
    )

    transport.sync_preview_transport(tab)

    assert button.checked == [True]
    assert "Paused" in button.tooltips[-1]
    assert button.enabled == [True], "a running render owns a picture the transport can hold"
    assert pushes == [1]


def test_on_original_frame_offers_the_decoder_picture_and_counts_the_arrival():
    """The offer names the decoder origin and the generation; the counter is written where a picture arrives."""
    register = _Register()
    arrivals = []
    synced = []
    pane = SimpleNamespace(
        shown_time=lambda: 1.0,
        shown=lambda: "result",
        set_source=lambda image: None,
        view_mode=lambda: "single",
    )
    frame = SimpleNamespace(buffer=bytes(4 * 2 * 3), width=4, height=2, stride=12, pts=1.0, index=0)
    tab = _owner(
        page_handles=SimpleNamespace(proc=SimpleNamespace(previewLabel=pane)),
        _source_register=register,
        _result_generation=3,
        _source_crop=None,
        _source_decoder=None,
        _compare_watch=None,
        _compare_fill_note_arrival=lambda: arrivals.append(1),
        _sync_compare_source=lambda: synced.append(1),
        _log_source_crop_unknown=lambda reason: None,
        _source_fill=lambda playhead: None,
    )

    transport.on_original_frame(tab, frame)

    assert register.offers == [(1.0, "decoder", 3)]
    assert len(arrivals) == 1, "the compare's arrivals counter lost its only write site once already"
    assert synced == [1], "the arrived picture must be presented through the one presentation path"


def test_next_result_generation_mints_one_token_and_clears_the_register():
    register = _Register(held=2)
    tab = _owner(_result_generation=7, _source_register=register)

    assert job.next_result_generation(tab) == 8
    assert tab._result_generation == 8
    assert register.clears == 1 and len(register) == 0, "a new generation makes every held source stale"


def test_on_job_finished_hands_a_finished_export_to_the_pane(tmp_path):
    output = tmp_path / "finished.mkv"
    output.write_bytes(b"not empty")
    calls = []
    tab = _owner(
        currentRenderOptions=SimpleNamespace(isPreview=False, outputPath=str(output)),
        return_codes=[0],
        userKilled=False,
        _preview_paused=False,
        _restoreQueueAfterPreview=lambda: calls.append("restore"),
        _start_preview_playback=lambda path: calls.append(("play", path)),
        _sync_preview_transport=lambda: calls.append("sync"),
    )

    job.on_job_finished(tab)

    assert calls == ["restore", ("play", str(output)), "sync"]
    assert tab._preview_paused is True, "the finished file is held at the render's position"


def test_on_job_finished_starts_a_preview_job_from_its_own_file():
    calls = []
    tab = _owner(
        currentRenderOptions=SimpleNamespace(isPreview=True, outputPath="/out/preview.mkv"),
        return_codes=[],
        userKilled=False,
        _restoreQueueAfterPreview=lambda: calls.append("restore"),
        _start_preview_playback=lambda path: calls.append(("play", path)),
        _sync_preview_transport=lambda: calls.append("sync"),
    )

    job.on_job_finished(tab)

    assert calls == ["restore", ("play", "/out/preview.mkv")]
