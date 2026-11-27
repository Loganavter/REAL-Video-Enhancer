"""The compare's use cases driven by a `SimpleNamespace` owner: no window, no tab class, no decoder.

`ProcessTab` is being split into plain functions whose first argument is the tab (`harness/docs/CODE_PATTERNS.md` §1,
plan `plan-rve-preview-use-cases-2026-11-27` §P2). This module is the contract for the first half of that split -
`present`, `supply`, `modes` - and it is written the way §9 asks: every test here FAILS when the body it covers is
replaced by `pass`, and the three behaviours that have regressed four times are pinned by SHAPE, not by arithmetic:

* the presenter asks the PANE for the moment (`PREVIEWER_RULES.md` §1). A stub pane whose `shown_moment()` disagrees
  with every tab clock must win - the tab clock must not even be consulted, because a fallback that is read first is
  how `_result_playhead` sent the supply to 0.0 while the pane showed 0.167 (2026-11-27, `asks=0` for a whole run);
* a moment with no register entry is handed over as a GAP (`set_source(None)`), never as the nearest picture. The
  register's bounded `picture_for` is the rule (`source_register.py`), and a presenter that tolerates a neighbour
  draws the desync instead of a hole;
* a PAUSED playhead is never asked for a decode lead, a moving one is (§5, the owner's "the original blinks a couple
  of times with playback never started").

How the owner is built: a `SimpleNamespace` carrying ONLY the attributes the function under test touches, so a body
that reaches for something else fails here instead of in the app. The pane is a stub of its public API
(`shown_moment`/`view_mode`/`set_source`/`set_original`/`set_view_mode`); the register is the REAL `SourceRegister`,
because "a picture for a moment or nothing" is the rule being tested, not a fixture.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = str(Path(__file__).resolve().parents[2])
# Both the app and the backend live in a `src` namespace package; the suite's conftest puts the backend first, which
# would make `src.constants` - and therefore `src.Util`, which `modes.py` logs through - the backend's module. The app
# root goes first for this module (the same house pattern as `test_display_wiring.py`), and the cached entry is dropped.
_ORIGINAL_SYS_PATH = list(sys.path)
if _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)
sys.modules.pop("src.constants", None)

# The app's optional runtime imports: stub them ONLY when the interpreter does not have them.
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

from src.ui.preview.source_register import SourceRegister  # noqa: E402
from src.ui.preview.use_cases import modes, present, supply  # noqa: E402

# Leave the suite's path order as it was (the house pattern): the app modules are imported and bound now.
sys.path[:] = _ORIGINAL_SYS_PATH

#: One frame at 24 fps, and the lead the supply keeps ahead of the playhead (`ProcessTab.COMPARE_LEAD_FRAMES` = 3).
FRAME = 1.0 / 24.0
LEAD = 3.0 * FRAME


class _Pane:
    """The pane's PUBLIC API these use cases read: a clock, a mode, and the two source setters. No widget."""

    def __init__(self, *, moment: float | None = None, mode: str = "single") -> None:
        self.moment = moment
        self.mode = mode
        self.sources: list = []          # every `set_source` payload, in order (None is a gap)
        self.legacy_sources: list = []   # every `set_original` call, for a pane from before the register API
        self.view_modes: list = []       # every `set_view_mode` argument

    def shown_moment(self) -> float | None:
        """The moment the pane is showing - the ONLY answer to "now" (`PREVIEWER_RULES.md` §1)."""
        return self.moment

    def view_mode(self) -> str:
        return self.mode

    def set_source(self, snapshot) -> None:
        self.sources.append(snapshot)

    def set_original(self, image, *, media_time: float | None = None, index: int | None = None) -> None:
        self.legacy_sources.append((image, media_time))

    def set_view_mode(self, mode: str) -> None:
        self.view_modes.append(mode)
        self.mode = str(mode)


class _LegacyPane:
    """A pane from before `set_source` existed: only `set_original`, which `present` keeps as a compatibility call."""

    def __init__(self, *, moment: float | None = None) -> None:
        self.moment = moment
        self.legacy_sources: list = []

    def shown_moment(self) -> float | None:
        return self.moment

    def set_original(self, image, *, media_time: float | None = None, index: int | None = None) -> None:
        self.legacy_sources.append((image, media_time))


class _Register(SourceRegister):
    """The real register, plus the questions it was asked - so a test can prove which moment was looked up."""

    def __init__(self) -> None:
        super().__init__(24)
        self.lookups: list[tuple[float, float, int | None]] = []

    def picture_for(self, moment, tolerance_s, generation=None):
        self.lookups.append((float(moment), float(tolerance_s), None if generation is None else int(generation)))
        return super().picture_for(moment, tolerance_s, generation=generation)


def _owner(*, pane=None, register=None, generation: int = 0, paused: bool = True, playhead: float = 1.0,
           fps: float = 24.0, input_file: str = "/tmp/clip.mkv", preview: bool = True,
           decoder=None) -> SimpleNamespace:
    """A tab shape carrying ONLY what `present`/`supply`/`modes` read (the delegators are bound to the real bodies)."""
    pane = _Pane() if pane is None else pane
    owner = SimpleNamespace(
        page_handles=SimpleNamespace(proc=SimpleNamespace(previewLabel=pane)),
        isPreview=bool(preview),
        _preview_paused=bool(paused),
        _preview_playhead=0.0,
        renderProcess=None,
        # `owner_of()`'s raw inputs: nothing owns the pane but the preview's own tail (which is None here).
        _queue_pane_owns=False,
        _queue_pane_decoder=None,
        _live_tail=None,
        _preview_decoder=None,
        _display=None,
        _display_seek_pending=False,
        currentRenderOptions=SimpleNamespace(inputFile=input_file),
        _source_register=_Register() if register is None else register,
        _result_generation=int(generation),
        _source_fill_pending=None,
        _compare_fill_stats={"asks": 0, "arrivals": 0, "at": 0.0},
        _compare_from_tick=False,
        _source_decoder=decoder,
        _source_warm_tried=False,
        _asked=[],
        _opened=[],
    )
    owner._on_original_requested = lambda seconds, span=False: owner._asked.append((float(seconds), bool(span)))
    owner._open_source_decoder = lambda: owner._opened.append(True) or True
    owner._compare_playhead = lambda: float(playhead)
    owner._compare_fps = lambda: float(fps)
    owner._compare_lead = lambda: 3.0 / float(fps)
    # The tab's own delegator in the tree: presentation asks the TAB for the tolerance, and the tab asks `supply`.
    owner._source_tolerance = lambda: supply.source_tolerance(owner)
    return owner


def _register_with(moments: dict[float, str], *, origin: str = "decoder", generation: int = 0) -> _Register:
    register = _Register()
    for moment, picture in moments.items():
        register.offer(moment, picture, origin, generation=generation)
    return register


# --- present.sync_compare_source ------------------------------------------------------------------------------------

def test_the_presenter_asks_the_pane_for_the_moment():
    """The pane's `shown_moment()` wins, and a tab clock is not even consulted: `PREVIEWER_RULES.md` §1.

    Pins the first of the three recurrences: `_supply_compare_source`'s guard read the transport clock (`0.0` after a
    render handed the finished file over) instead of the pane's own, and the source half was never asked for at all
    (`compare fill asks=0 playhead=0.167` for minutes, live 2026-11-27). A presenter that reads a tab clock first - or
    that prefers it when the pane has an answer - draws the moment of a picture the user is not looking at.
    """
    pane = _Pane(moment=1.0, mode="sbs")
    register = _register_with({1.0: "P1.0", 7.0: "P7.0"})
    owner = _owner(pane=pane, register=register)
    clock_calls: list[float] = []
    owner._compare_playhead = lambda: clock_calls.append(7.0) or 7.0

    present.sync_compare_source(owner)

    assert clock_calls == [], "the presenter consulted a tab clock while the pane named its own moment"
    assert len(pane.sources) == 1
    assert pane.sources[0] is not None and pane.sources[0].picture == "P1.0", (
        f"the pane shows 1.0 but was handed {pane.sources[0]}")
    assert register.lookups == [(pytest.approx(1.0), pytest.approx(0.5 / 24.0), 0)], (
        "the lookup is the pane's moment, the tab's tolerance, the tab's generation")


def test_a_moment_with_no_picture_is_a_gap_not_a_neighbour():
    """A hole in the register is handed over as `set_source(None)`, never as the nearest picture.

    Pins `source_register.picture_for`'s bounded rule from the caller's side: with the pane on 2.0 and pictures held at
    1.9 and 2.1, the honest answer is a gap. A presenter that falls back to "the closest one" draws a one-frame desync
    and calls it a picture; the tolerance is the tab's half frame (`_source_tolerance`), not an opinion of this module.
    """
    pane = _Pane(moment=2.0, mode="sbs")
    register = _register_with({1.9: "left-neighbour", 2.1: "right-neighbour"})

    present.sync_compare_source(_owner(pane=pane, register=register))

    assert pane.sources == [None], f"a neighbour was handed over as the picture of 2.0: {pane.sources}"

    # And the tolerance IS the tab's: a wide one makes the neighbour the nearest picture within half a frame.
    wide = _owner(pane=_Pane(moment=2.0, mode="sbs"), register=register)
    wide._source_tolerance = lambda: 0.2
    present.sync_compare_source(wide)
    assert wide.page_handles.proc.previewLabel.sources[0].picture == "right-neighbour"


def test_an_unnamed_pane_falls_back_to_the_tab_clock_not_to_a_gap():
    """A pane that names no moment (a queue item, a compare opened before its first frame) still has a playhead.

    The fallback is documented in the body and is the reason the presenter can serve a pane whose result is unnamed; it
    is only ever a FALLBACK - the test above pins that the pane's own answer comes first.
    """
    pane = _Pane(moment=None, mode="sbs")
    register = _register_with({7.0: "P7.0"})
    owner = _owner(pane=pane, register=register, playhead=7.0)
    owner._preview_playhead = 3.0

    present.sync_compare_source(owner)

    assert pane.sources[0] is not None and pane.sources[0].picture == "P7.0"
    assert register.lookups[0][0] == pytest.approx(7.0), "the tab's playhead, not the tab's last painted memo"


def test_a_picture_from_another_generation_is_not_an_answer():
    """The register is asked with the tab's generation: a picture from the previous file/render is a gap."""
    pane = _Pane(moment=5.0, mode="sbs")
    register = _register_with({5.0: "previous-generation"}, generation=2)

    present.sync_compare_source(_owner(pane=pane, register=register, generation=3))

    assert pane.sources == [None], "a source from the previous generation was painted"
    assert register.lookups == [(pytest.approx(5.0), pytest.approx(0.5 / 24.0), 3)]


def test_a_legacy_pane_gets_the_same_bounded_answer():
    """A pane from before `set_source` (an old harness double) is served through `set_original`, same moment check."""
    pane = _LegacyPane(moment=5.0)
    register = _register_with({5.0: "P5.0"})

    present.sync_compare_source(_owner(pane=pane, register=register))

    assert pane.legacy_sources == [("P5.0", pytest.approx(5.0))], pane.legacy_sources

    gap = _LegacyPane(moment=6.0)
    present.sync_compare_source(_owner(pane=gap, register=register))
    assert gap.legacy_sources == [(None, None)], "a gap must clear the legacy pane's source half"


def test_paint_pair_is_the_same_one_path():
    """`paint_pair` hands over the source of the moment that was just painted - a forced playhead is only a read.

    Both paint routes call it after `set_result` so the two halves change in ONE repaint ("one drawn, then the other",
    owner 2026-11-26). The `playhead` argument is deliberately not a way to assert a picture exists: the pane's own
    moment still decides which picture is looked up.
    """
    pane = _Pane(moment=1.0, mode="sbs")
    register = _register_with({1.0: "P1.0", 9.0: "P9.0"})
    owner = _owner(pane=pane, register=register)

    present.paint_pair(owner, 9.0)

    assert pane.sources[0] is not None and pane.sources[0].picture == "P1.0", (
        f"the forced playhead overrode the pane's own moment: {pane.sources}")
    assert register.lookups[0][0] == pytest.approx(1.0)


# --- supply.supply_compare_source -----------------------------------------------------------------------------------

def test_single_warms_the_decoder_and_starts_no_standing_fill():
    """`single` keeps the LMB flash: warm the decoder, ask for nothing.

    The warm must happen BELOW the single return, not above it: during a render the pane is `single`, so a warm-up after
    the return never ran and the first switch to a compare mode paid the decoder's open (~0.8 s of the owner's wait,
    2026-11-26).
    """
    pane = _Pane(moment=5.0, mode="single")
    owner = _owner(pane=pane, register=_register_with({5.0: "P5.0"}), paused=False, playhead=5.0)

    supply.supply_compare_source(owner)

    assert owner._opened == [True], "single did not warm the compare decoder"
    assert owner._source_warm_tried is True
    assert owner._asked == [], "single started a standing fill"


def test_a_compare_mode_asks_the_decoder_at_the_pane_moment():
    """`sbs`/`split`: warm, then ask for the lead at the moment the pane shows - the pane's clock, not the transport's."""
    pane = _Pane(moment=5.0, mode="sbs")
    owner = _owner(pane=pane, register=_register_with({5.0: "P5.0"}), paused=False, playhead=5.0)

    supply.supply_compare_source(owner)

    assert owner._opened == [True]
    assert owner._asked == [(pytest.approx(5.0 + LEAD), True)], f"asked {owner._asked}"


def test_a_queue_file_asks_for_nothing_at_all():
    """A queue item IS the source: the compare decoder must not be opened over it."""
    pane = _Pane(moment=5.0, mode="sbs")
    owner = _owner(pane=pane, register=_register_with({5.0: "P5.0"}), paused=False, playhead=5.0)
    owner._queue_pane_owns = True

    supply.supply_compare_source(owner)

    assert owner._opened == [], "the compare decoder was warmed for a queue item"
    assert owner._asked == []


def test_a_zero_playhead_is_not_asked_about():
    """A pane that names nothing to start from (`<= 0.0`) is not asked: the ask would decode a moment nobody shows."""
    pane = _Pane(moment=None, mode="sbs")
    owner = _owner(pane=pane, register=_register_with({0.5: "P0.5"}), paused=False, playhead=0.0)

    supply.supply_compare_source(owner)

    assert owner._asked == []


# --- supply.source_fill ---------------------------------------------------------------------------------------------

def test_a_paused_playhead_is_never_asked_for_a_lead_but_a_moving_one_is():
    """A still playhead holds its picture; a moving one is kept a lead ahead of.

    Pins `PREVIEWER_RULES.md` §5: the paused pane used to be walked a lead ahead every tick, and because eviction was
    "oldest first" the walk dropped exactly the picture on screen - the original blinked a couple of times with playback
    never started (owner, 2026-11-27). The same fill must still lead a MOVING playhead, or the pair arrives late.
    """
    register = _register_with({5.0: "P5.0"})
    owner = _owner(pane=_Pane(moment=5.0, mode="sbs"), register=register, paused=True)

    supply.source_fill(owner, 5.0)
    assert owner._asked == [], f"a paused playhead asked the decoder for {owner._asked}"
    assert owner._source_fill_pending is None

    owner._preview_paused = False
    supply.source_fill(owner, 5.0)
    assert owner._asked == [(pytest.approx(5.0 + LEAD), True)], (
        f"a moving playhead was not kept a lead ahead: {owner._asked}")
    assert owner._compare_fill_stats["asks"] == 1
    assert owner._compare_fill_stats["newest"] == pytest.approx(5.0)
    assert owner._compare_fill_stats["target"] == pytest.approx(5.0 + LEAD)
    assert owner._compare_fill_stats["span"] == 1


def test_a_paused_playhead_with_nothing_held_asks_for_one_picture_not_a_lead():
    """A jump while paused (a seek) asks for the ONE frame at the new moment - a seek's answer, not a burst."""
    owner = _owner(pane=_Pane(moment=6.0, mode="sbs"), register=_register_with({5.0: "P5.0"}), paused=True)

    supply.source_fill(owner, 6.0)

    assert owner._asked == [(pytest.approx(6.0), False)], f"asked {owner._asked}"
    assert owner._source_fill_pending is not None and owner._source_fill_pending[0] == pytest.approx(6.0)


def test_only_one_ask_is_in_flight_at_a_time():
    """The arrival of each frame calls the fill again; a burst must not start a second span (the decoder keeps one ask)."""
    owner = _owner(pane=_Pane(moment=5.0, mode="sbs"), register=_register_with({5.0: "P5.0"}), paused=False)

    supply.source_fill(owner, 5.0)
    supply.source_fill(owner, 5.0)
    assert len(owner._asked) == 1, f"a second span was started for the same moment: {owner._asked}"

    # The answer arriving for the pending target retires the ask, so the next tick may lead again.
    owner._source_register.offer(5.0 + LEAD, "P5.125", "decoder")
    supply.source_fill(owner, 5.0)
    assert owner._source_fill_pending is None
    assert len(owner._asked) == 1, "the lead was already full but the fill asked again"


def test_an_unanswered_ask_is_re_sent_after_the_window():
    """One ask in flight is a retry deadline, not a throttle: past `COMPARE_FILL_WINDOW_S` the fill asks again."""
    owner = _owner(pane=_Pane(moment=5.0, mode="sbs"), register=_register_with({5.0: "P5.0"}), paused=False)
    owner._source_fill_pending = (5.0 + LEAD, 0.0)          # asked at the epoch: the window is long gone

    supply.source_fill(owner, 5.0)

    assert owner._asked == [(pytest.approx(5.0 + LEAD), True)], f"a dead ask blocked the fill for ever: {owner._asked}"


def test_the_block_owns_the_moment_and_the_decoder_stays_silent():
    """The renderer's block publishing THIS moment's source means no decode: that is the cost the block removes."""
    owner = _owner(pane=_Pane(moment=5.0, mode="sbs"),
                   register=_register_with({5.0: "P5.0"}, origin="block"), paused=False)

    supply.source_fill(owner, 5.0)

    assert owner._asked == [], "the compare decoded the moment the block already owns"
    assert owner._source_fill_pending is None


def test_a_moving_playhead_with_nothing_held_asks_at_its_own_moment():
    """A scrub where the register answers nothing: ONE picture at the new moment, not a burst from the decoder's stand.

    With a picture held at the playhead the fill tops up the lead (the test above); with nothing held the same call
    asks at the playhead itself, `span=False`. That is the seek's answer and the reason a jump does not replay the
    stretch between the old position and the new one.
    """
    owner = _owner(pane=_Pane(moment=9.0, mode="sbs"), register=_register_with({5.0: "P5.0"}), paused=False)

    supply.source_fill(owner, 9.0)

    assert owner._asked == [(pytest.approx(9.0), False)], f"asked {owner._asked}"


# --- supply.warm_source_decoder -------------------------------------------------------------------------------------

def test_the_warm_open_is_guarded_and_needs_a_file():
    """One open per job: no file means no open, and a second tick never re-opens (a failing file is not retried 10x/s)."""
    no_file = _owner(pane=_Pane(moment=None, mode="single"), input_file="")
    supply.warm_source_decoder(no_file)
    assert no_file._opened == [] and no_file._source_warm_tried is False, "a job without an input file was warmed"

    owner = _owner(pane=_Pane(moment=None, mode="single"))
    supply.warm_source_decoder(owner)
    supply.warm_source_decoder(owner)
    assert owner._opened == [True], f"the decoder was opened {len(owner._opened)} times"
    assert owner._source_warm_tried is True

    already = _owner(pane=_Pane(moment=None, mode="single"), decoder=object())
    supply.warm_source_decoder(already)
    assert already._opened == [], "an open decoder was opened again"


# --- supply.source_tolerance ----------------------------------------------------------------------------------------

def test_the_tolerance_is_half_a_frame_at_the_decoder_rate():
    """The matching slack is measured in frames, and a nonsense fps cannot widen it past a whole frame."""
    assert supply.source_tolerance(_owner(fps=24.0)) == pytest.approx(0.5 / 24.0)
    assert supply.source_tolerance(_owner(fps=50.0)) == pytest.approx(0.5 / 50.0)
    assert supply.source_tolerance(_owner(fps=0.5)) == pytest.approx(0.5), "the max(1.0, fps) clamp is gone"
    assert isinstance(supply.source_tolerance(_owner(fps=24.0)), float)


# --- modes ----------------------------------------------------------------------------------------------------------

def test_set_view_mode_forwards_to_the_pane():
    """The pane owns the mode; this module owns the policy and the one call that changes it."""
    pane = _Pane(moment=5.0, mode="single")
    owner = _owner(pane=pane)

    modes.set_view_mode(owner, "sbs")

    assert pane.view_modes == ["sbs"], f"the pane was handed {pane.view_modes}"
    assert pane.mode == "sbs"


def test_an_unknown_mode_is_refused_before_the_pane_changes():
    """A mode the pane cannot paint is refused by the policy, not half-applied and then failed on the renderer."""
    pane = _Pane(moment=5.0, mode="single")
    owner = _owner(pane=pane)

    with pytest.raises(ValueError):
        modes.set_view_mode(owner, "quad")

    assert pane.view_modes == [], "the pane was changed by a mode that does not exist"


def test_pair_required_follows_the_pane_own_mode():
    """`sbs`/`split` paint a pair; `single` does not; a host with no compare pane requires nothing."""
    for mode, required in (("sbs", True), ("split", True), ("single", False)):
        pane = _Pane(moment=5.0, mode=mode)
        assert modes.pair_required(_owner(pane=pane)) is required, f"mode {mode}"

    assert modes.pair_required(_owner(pane=_LegacyPane(moment=5.0))) is False, (
        "a pane without a view mode claims a pair requirement")
    assert modes.pair_required(SimpleNamespace(page_handles=None)) is False


def test_playback_asks_ahead_but_a_jump_asks_for_one_frame():
    """A CONTINUOUS playhead asks the LEAD; a JUMP asks one frame. Both are `paused=False`, so motion is the difference.

    Measured live 2026-11-27 in the owner's session: during a preview render the source lagged 21 frames behind the
    result, 29 results were refused as older than the picture, the hold expired 41 times and 41 sbs draws carried no
    original. A single-frame ask cannot win that race - by the time the answer lands the playhead is at the next moment -
    so playback must ask ahead while a scrub keeps its one picture.
    """
    owner = _owner(pane=_Pane(moment=5.0, mode="sbs"), register=_register_with({}), paused=False)
    interval = 1.0 / 24.0

    supply.source_fill(owner, 5.0)                       # nothing before it: a jump, one frame
    assert owner._asked == [(pytest.approx(5.0), False)], f"asked {owner._asked}"

    owner._source_fill_pending = None
    supply.source_fill(owner, 5.0 + interval)            # a tick later: playback
    assert owner._asked[-1] == (pytest.approx(5.0 + interval + LEAD), True), (
        f"a continuous playhead was not kept a lead ahead: {owner._asked}")

    owner._source_fill_pending = None
    supply.source_fill(owner, 9.0)                       # a jump of four seconds: one frame at the new moment
    assert owner._asked[-1] == (pytest.approx(9.0), False), f"a jump asked for a burst: {owner._asked}"
