"""The pair writer: a compare mode paints a pair or holds the last one, and one hard limit says otherwise.

Why this file exists (harness/docs/CODE_PATTERNS.md §2, §5, §7, §9, §12): the pane used to answer "what is on screen" with two
queues, a valve timer and a predicate that could each disagree with the others, and the renderer was handed the newest
result ALONE whenever its source was missing - which is the normal state at every refused source, so the compare flashed
"the upscale, then the original" and the strip drag blanked the left half for ~1.3 s (31 draws in the owner's log).

Every row here fails on the behaviour the refactor removed: the module-level rows assert on the sink (the renderer's own
`set_frame`/`set_original` calls), the pane-level rows assert that a real `VideoPreviewSurface` never hands a compare mode
half a picture. The red-before proof is in each docstring - the guard was reverted in place and the row watched to fail.
"""

from __future__ import annotations

import re

from contextlib import contextmanager
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QColor, QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from src.ui.preview.pair import MAX_PAIR_STALE_S, CommittedPair, Half, PairWriter  # noqa: E402
from src.ui.preview.source_register import ResultPicture, SourcePicture  # noqa: E402
from src.ui.video import telemetry  # noqa: E402
from src.ui.widgets.video_preview import VideoPreviewSurface  # noqa: E402

#: The tolerance the pane states to the writer: half a frame at the scene's 24 fps (`SOURCE_HALF_FRAME_S`).
HALF_FRAME_S = 0.5 / 24.0


class Sink:
    """The renderer's two ingestion calls, recorded. A frame is any object here - the writer treats it as opaque."""

    def __init__(self) -> None:
        self.result = "none"
        self.source = "none"
        self.calls: list[tuple[str, object]] = []

    def set_frame(self, frame) -> None:
        self.result = frame
        self.calls.append(("frame", frame))

    def set_original(self, frame) -> None:
        self.source = frame
        self.calls.append(("orig", frame))


class Clock:
    """A monotonic clock the test drives, so `MAX_PAIR_STALE_S` costs no wall time."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


def _half(moment: float, *, generation: int = 1, tag: str | None = None, image=None) -> Half:
    payload = ("frame", tag if tag is not None else f"{moment:.4f}")
    return Half(frame=payload, moment=moment, generation=generation, image=image, name=("frame", 0))


def _writer(*, mode: str = "sbs") -> tuple[PairWriter, Sink, Clock]:
    sink = Sink()
    clock = Clock()
    writer = PairWriter(sink, clock=clock)
    writer.set_view_mode(mode)
    writer.set_tolerance(HALF_FRAME_S)
    return writer, sink, clock


def _offer_result(writer: PairWriter, half: Half) -> bool:
    writer.offer_result(half)
    return writer.paint()


def _offer_source(writer: PairWriter, half: Half | None) -> bool:
    writer.offer_source(half)
    return writer.paint()


@contextmanager
def _captured():
    """Capture the telemetry lines a policy decision writes (the writer must SAY which half and why)."""
    records: list[str] = []
    telemetry.set_writer(lambda line, _which: records.append(line))
    try:
        yield records
    finally:
        telemetry.set_writer(None)


def _stale(lines: list[str]) -> list[str]:
    return [line for line in lines if line.startswith("preview: pair stale ")]


def _refused(lines: list[str]) -> list[str]:
    return [line for line in lines if line.startswith("preview: pair refused ")]


# --- the drag row: a late source keeps the previous pair painted --------------------------------------


def test_a_late_source_keeps_the_previous_pair_painted():
    """The owner's drag row, in one row: a result whose source is missing must NOT reach the renderer alone.

    Red before: with the old gate (`_commit_newest` on every `_original is None`) the second, third and fourth results
    were each handed over as `set_frame(new)` + `set_original(None)`, so the left half went black and came back when a
    source finally arrived. Reverting `paint()` to the lone-result branch makes this row fail on the first hold.
    """
    writer, sink, clock = _writer()
    assert _offer_result(writer, _half(1.0)) is False, "a result with no source paints nothing in a compare mode"
    writer.offer_source(_half(1.0))
    assert writer.paint() is True
    assert sink.result == ("frame", "1.0000") and sink.source == ("frame", "1.0000")

    painted_before = writer.painted()
    calls_before = list(sink.calls)

    # The drag: results keep arriving, every source answer is one frame behind and is refused as a neighbour.
    for offset in (1.0 / 24.0, 2.0 / 24.0, 3.0 / 24.0):
        clock.advance(0.02)
        assert _offer_result(writer, _half(1.0 + offset)) is False, "the pane holds the last matched pair"
        _offer_source(writer, None)                                  # the gap the drag produces
    assert sink.calls == calls_before, "the renderer was handed nothing while the source was missing"
    assert writer.painted() is painted_before, "the pair on screen is the last one that belonged together"
    assert sink.source is not None and sink.result is not None

    # The source for the newest moment lands: the pair advances as ONE hand-over of two halves.
    writer.offer_source(_half(1.0 + 3.0 / 24.0))
    assert writer.paint() is True
    assert sink.calls[-2:] == [("frame", ("frame", "1.1250")), ("orig", ("frame", "1.1250"))]
    assert writer.painted().lone_half is None


def test_the_pane_never_hands_a_compare_mode_a_lone_result():
    """The same row at the pane level, on the real widget: no compare-mode paint may carry a result without a source.

    Red before: `_request_frame_paint`'s `_original is None` branch called `_commit_newest()`, so `set_result` alone
    produced a `set_frame` + `set_original(None)` pair - which is the flash the owner reported.
    """
    QApplication.instance() or QApplication([])
    surface = VideoPreviewSurface()
    surface.resize(800, 600)
    surface.set_view_mode("sbs")
    hands: list[tuple[str, object]] = []
    surface._renderer.set_frame = lambda frame: hands.append(("frame", frame))
    surface._renderer.set_original = lambda source: hands.append(("orig", source))

    surface.set_result(ResultPicture(moment=2.0, picture=_picture(), generation=1))
    assert hands == [], "a lone result is not handed over, not even alone and not even first"
    for moment in (2.0 + 1.0 / 24.0, 2.0 + 2.0 / 24.0):
        surface.set_result(ResultPicture(moment=moment, picture=_picture(), generation=1))
        assert hands == [], "and it stays held while the source is missing"

    surface.set_source(SourcePicture(moment=2.0 + 2.0 / 24.0, picture=_picture(), origin="decoder", generation=1))
    kinds = [kind for kind, _frame in hands]
    assert kinds[-2:] == ["frame", "orig"], "the pair is what reaches the renderer, together"


# --- the one hard limit ------------------------------------------------------------------------------


def test_the_hard_limit_paints_a_half_and_says_which_and_why():
    """`MAX_PAIR_STALE_S` is the ONLY way a compare mode shows half a picture, and it must be greppable.

    Red before: with no hard limit at all the pane is frozen for ever by a dead decoder; with the limit's condition
    changed to "always" the first row paints a lone result immediately. Both directions fail here.
    """
    writer, sink, clock = _writer()
    writer.offer_source(_half(1.0))
    assert writer.paint() is True                      # the source is the only picture: the documented lone-source case
    assert sink.source is not None

    with _captured() as lines:
        assert _offer_result(writer, _half(2.0)) is False, "held: the source for 2.0 has not arrived yet"
        clock.advance(MAX_PAIR_STALE_S - 0.01)
        assert _offer_result(writer, _half(2.0 + 1.0 / 24.0)) is False, "one notch under the limit still holds"
        assert _stale(lines) == [], "nothing may be called stale before the limit"

        clock.advance(0.02)
        assert _offer_result(writer, _half(2.0 + 2.0 / 24.0)) is True
    assert line_has(_stale(lines), half="result", why="no-source-within-hold"), (
        f"the log must say which half and why: {lines}")
    painted = writer.painted()
    assert painted is not None and painted.result is not None and painted.source is None
    assert painted.lone_half == "result", "the pair names the exception instead of pretending to be a pair"
    assert painted.moment == pytest.approx(2.0 + 2.0 / 24.0)

    # The limit is paid once: the newest result keeps painting alone until a pair completes again.
    calls = len(sink.calls)
    assert _offer_result(writer, _half(2.0 + 3.0 / 24.0)) is True
    assert len(sink.calls) == calls + 1 and sink.source is None
    writer.offer_source(_half(2.0 + 3.0 / 24.0))
    assert writer.paint() is True
    assert writer.painted().lone_half is None and sink.source is not None


def test_the_pane_honours_the_hard_limit_with_a_dead_source():
    """The same limit through the pane: after it, the result paints and the log names the half (`pair stale`)."""
    QApplication.instance() or QApplication([])
    surface = VideoPreviewSurface()
    surface.resize(800, 600)
    surface.set_view_mode("split")
    clock = Clock()
    surface._pair._clock = clock
    hands: list[tuple[str, object]] = []
    surface._renderer.set_frame = lambda frame: hands.append(("frame", frame))
    surface._renderer.set_original = lambda source: hands.append(("orig", source))

    with _captured() as lines:
        surface.set_result(ResultPicture(moment=3.0, picture=_picture(), generation=1))
        assert hands == []
        clock.advance(MAX_PAIR_STALE_S + 0.01)
        surface.set_result(ResultPicture(moment=3.0 + 1.0 / 24.0, picture=_picture(), generation=1))
    # The half that paints after the limit is the RESULT; the source half stays empty. (The hand-over is not required
    # to touch BOTH textures: `_paint` only re-hands a half whose payload changed, so an already-empty source is
    # skipped - the STATE is the assertion, not the call shape.)
    assert any(kind == "frame" for kind, _ in hands), f"after the limit the result must paint: {hands}"
    assert all(frame is None for kind, frame in hands if kind == "orig"), "and the source half stays empty"
    painted = surface._pair.painted()
    assert painted is not None and painted.lone_half == "result"
    assert line_has(_stale(lines), half="result", why="no-source-within-hold")


# --- entering a compare mode ---------------------------------------------------------------------------


def test_entering_a_compare_mode_paints_no_half_and_the_first_pair_arrives_with_its_source():
    """A switch from `single` must not paint the upscale alone - the owner's own complaint (2026-11-26).

    The order "the upscale, then the original" is what this module exists to remove, and the mode switch was the one
    place still producing it: `single` holds a result with no source, and handing that picture to the compare drew
    `half=lone:result` until the source landed. Measured 2026-11-27 on the owner's gesture: 1-9 such paints per switch,
    each corrected only 0.24-1.1 s later.

    The pane now waits EMPTY for the first pair. RED-BEFORE: with the old branch the pane hands the single-mode result
    as the first comparison picture (`assert [("frame", ...)] == []`).
    """
    writer, sink, clock = _writer(mode="single")
    sink.calls.clear()
    writer.offer_result(_half(5.0))                 # the single-mode picture on screen
    assert writer.paint() is True
    assert ("frame", ("frame", "5.0000")) in sink.calls, f"single hands the result: {sink.calls}"

    sink.calls.clear()
    writer.set_view_mode("sbs")                     # the switch
    assert sink.calls == [("frame", None), ("orig", None)], (
        f"a compare mode starts empty, never on one half: {sink.calls}")
    assert writer.painted() is None, "an empty pane is no pair at all - not an empty pair object"

    # The first result is HELD, not painted alone...
    sink.calls.clear()
    writer.offer_result(_half(5.5))
    assert writer.paint() is False and sink.calls == [], f"a lone result must not paint: {sink.calls}"

    # ...and the pair paints the moment its source arrives.
    writer.offer_source(_half(5.5))
    assert writer.paint() is True
    painted = writer.painted()
    assert painted is not None and painted.lone_half is None
    assert painted.result is not None and painted.source is not None and painted.moment == pytest.approx(5.5)


def test_after_a_mode_entry_a_source_alone_waits_for_its_result():
    """The mirror image of the mode-switch fix: entering a compare mode DROPS the single-mode result, so the source that
    arrives next has no partner yet - and painting it alone shows the original with an empty right half (measured
    2026-11-27, the owner's screenshot after a seek). A source alone paints only on a pane that never had a result.
    """
    writer, sink, clock = _writer(mode="sbs")
    writer.offer_result(_half(4.0))
    writer.offer_source(_half(4.0))
    assert writer.paint() is True and writer.painted().lone_half is None

    writer.set_view_mode("single")
    writer.set_view_mode("sbs")                       # the entry drops the single-mode picture
    sink.calls.clear()

    writer.offer_source(_half(4.5))                   # the source of the moment this seek asked for
    assert writer.paint() is False, "a source with no result is held, not painted"
    assert sink.calls == [], f"nothing may reach the renderer yet: {sink.calls}"

    writer.offer_result(_half(4.5))
    assert writer.paint() is True
    assert writer.painted().lone_half is None and writer.painted().moment == pytest.approx(4.5)


def test_a_pane_that_never_had_a_result_still_shows_its_source():
    """A compare opened before its first result (a queue item): the source IS the picture there is."""
    writer, sink, clock = _writer(mode="sbs")
    writer.offer_source(_half(1.0))
    assert writer.paint() is True
    painted = writer.painted()
    assert painted is not None and painted.lone_half == "source", "the one picture in hand is shown"


# --- a clear is a clear ------------------------------------------------------------------------------


def test_a_cleared_pane_stays_empty_and_refuses_a_late_source():
    """`clear()` drops the pair, empties the renderer, and a source from before the clear is refused (f1ed720e).

    Red before: the pane kept its result queue across the clear, so a late source re-keyed itself by its own moment
    (`shown_moment()` was None) and painted half a picture onto the empty pane (audit §6.1b, probe G2).
    """
    writer, sink, _clock = _writer()
    writer.offer_source(_half(1.0))
    writer.paint()
    assert sink.source is not None, "a picture was on screen before the clear"

    writer.clear()
    assert (sink.result, sink.source) == (None, None), "the renderer is emptied"
    painted = writer.painted()
    assert painted is None, "a cleared pane has no pair on screen at all - not an empty one"

    with _captured() as lines:
        assert _offer_source(writer, _half(0.5)) is False, "a source from before the clear may not paint"
    assert line_has(_refused(lines), half="source", why="cleared-pane")
    assert (sink.result, sink.source) == (None, None)
    assert writer.cleared is True

    # A NEW picture re-opens the pane, with its own source.
    assert _offer_result(writer, _half(2.0)) is False
    assert _offer_source(writer, _half(2.0)) is True
    painted = writer.painted()
    assert painted is not None and painted.moment == pytest.approx(2.0) and painted.source is not None
    assert writer.cleared is False


# --- a source is a pair only at the same moment, from the same generation ------------------------------


def test_a_source_outside_half_a_frame_is_not_a_pair_and_keeps_the_painted_one():
    """Half a frame is the pane's tolerance; outside it the snapshot is a neighbour, not this moment's picture.

    Red before: the queue keyed pictures in milliseconds and the gate fell back to the newest of each half, so a source
    two frames off was paired with the result and the draw line reported it as a pair (the `pair_sub` rows).
    """
    writer, sink, _clock = _writer()
    writer.offer_source(_half(1.0))
    writer.paint()
    assert _offer_result(writer, _half(1.0)) is True
    painted = writer.painted()
    assert painted.source is not None and painted.source.moment == pytest.approx(1.0)
    calls = list(sink.calls)

    # Two frames away: not a pair, and the painted pair is what stays.
    writer.offer_source(_half(1.0 + 2.0 / 24.0))
    assert writer.paint() is False
    assert sink.calls == calls, "the neighbour never reached the renderer"
    assert writer.painted() is painted

    # Half a frame away IS the same picture (the drift the two independently derived timestamps produce).
    fresh = _half(1.0 + HALF_FRAME_S / 2.0)
    writer.offer_source(fresh)
    assert writer.paint() is True
    assert writer.painted().source is fresh


def test_a_pending_result_does_not_pair_with_a_neighbouring_source():
    """The forward direction: a source offered BEFORE its result may only pair within the same half frame."""
    writer, sink, _clock = _writer()
    writer.offer_source(_half(5.0))
    assert writer.paint() is True                      # no result at all yet: the source is the one picture
    sink.calls.clear()

    assert _offer_result(writer, _half(5.0 + 2.0 / 24.0)) is False, "two frames off is the flicker: hold"
    assert sink.calls == []
    assert writer.painted().source is not None and writer.painted().result is None, "the source-only picture stays"

    assert _offer_result(writer, _half(5.0 + HALF_FRAME_S / 2.0)) is True
    painted = writer.painted()
    assert painted.result is not None and painted.source is not None, "a pair, not a lone result"
    assert abs(painted.moment - painted.source.moment) <= HALF_FRAME_S
    assert [kind for kind, _frame in sink.calls] == ["frame"], (
        "only the RESULT half changed: the source the renderer already holds is the same picture")


def test_a_picture_from_an_older_generation_is_refused_and_the_pair_survives():
    """Identity is not only the moment: a previous render/file/seek's picture is never an answer for this one."""
    writer, sink, _clock = _writer()
    writer.offer_source(_half(4.0, generation=2))
    writer.offer_result(_half(4.0, generation=2))
    assert writer.paint() is True
    painted = writer.painted()
    calls = list(sink.calls)

    with _captured() as lines:
        writer.offer_source(_half(4.0, generation=1))
        assert writer.paint() is False
    assert line_has(_refused(lines), half="source", why="older-generation")
    assert sink.calls == calls
    assert writer.painted() is painted

    # A NEWER generation is a new picture: the old pair is not "in sync" with it, it is a different picture.
    writer.offer_result(_half(4.0 + 3.0, generation=3))
    assert writer.paint() is False, "the new generation's result has no source yet: hold"
    assert writer.painted() is painted


# --- one surface has no pair --------------------------------------------------------------------------


def test_single_forwards_the_result_and_the_flash_and_never_holds():
    """`single` is one surface: the result is the picture and the LMB flash is the source beside it. No hold applies."""
    writer, sink, _clock = _writer(mode="single")
    assert _offer_result(writer, _half(1.0)) is True
    assert sink.result is not None and sink.source is None

    flash = _half(1.0 + 4.0 / 24.0)                     # the host answers the ask; it is NOT a pair by construction
    assert _offer_source(writer, flash) is True
    assert sink.source is flash.frame
    assert writer.painted().lone_half is None, "single is one surface, not a lone half"

    assert _offer_source(writer, None) is True
    assert sink.source is None, "releasing the flash returns to the result at once"


# --- one writer, proved mechanically ------------------------------------------------------------------


def test_the_pair_writer_is_the_only_code_that_feeds_the_renderer():
    """A second writer is the defect; this fails on it EXISTING, not on it misbehaving (code-patterns §2).

    Two calls are the renderer's whole ingestion (`set_frame`/`set_original`). In `src/` only `preview/pair.py` may make
    them; the other allowed receivers are the PANE's public adapters (`pane.set_original` from the tab, `self.set_*`
    inside the pane), which are the API and route through the same guards. A new module that starts feeding the
    renderer has to be added here on purpose - that is the point.
    """
    root = Path(__file__).resolve().parents[2] / "src"
    allowed_receivers = {"self", "pane", "surface", "widget"}
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "pair.py":
            continue                              # the one writer
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("def ") or stripped.startswith("they are"):
                continue
            for match in re.finditer(r"([\w.]+)\.(set_frame|set_original)\(", stripped):
                if match.group(1) in allowed_receivers:
                    continue
                offenders.append(f"{path.relative_to(root)}:{number}: {stripped}")
    assert offenders == [], f"the renderer has a second writer: {offenders}"

    pair_source = (root / "ui" / "preview" / "pair.py").read_text()
    assert "_sink.set_frame(" in pair_source and "_sink.set_original(" in pair_source, (
        "the writer is what hands the halves over")


def test_committed_pair_states_its_two_halves_and_its_one_exception():
    """The pair's shape is the policy: both halves of one moment, and `lone_half` only where the policy allows it."""
    pair = CommittedPair(result=_half(1.0), source=_half(1.0), moment=1.0, generation=1)
    assert pair.lone_half is None
    assert pair.result.moment == pair.source.moment == pair.moment
    lone = CommittedPair(result=_half(1.0), source=None, moment=1.0, generation=1, lone_half="result")
    assert lone.lone_half == "result"


# --- helpers ------------------------------------------------------------------------------------------


def _picture(colour: str = "#204060") -> QImage:
    image = QImage(320, 180, QImage.Format.Format_RGBA8888)
    image.fill(QColor(colour))
    return image


def line_has(lines: list[str], **fields) -> bool:
    """Whether one captured line carries every `key=value` pair. Telemetry lines are `key=value` on purpose."""
    return any(all(f"{key}={value}" in line for key, value in fields.items()) for line in lines)
