"""The pane's feed: one picture, its identity, its drop count and its seek slot - without a render.

Four facts that only make sense together are pinned here, in the order the module's docstring states them:

* **latest-wins**: an accepted picture that nobody read before the next one arrived is one drop, and the feed
  holds exactly one picture (nothing is queued, so a slow pane never replays an old frame);
* **the identity is the feed's**: `shown`/`shown_time` name the picture `image()` returns, captured from the
  reader at the moment it was accepted - moving the reader's counter afterwards must not rename it;
* **no block, no crash**: a feed that never attached answers `False`/`None` and writes nothing;
* **the seek slot is the reader's**: `request_seek`/`seek_generation` are the reader's newest-wins generation.

The fakes are a stub reader (identity attributes, one pending picture, the seek slot) for the rules, plus one
hand-written block read by the real `DisplayReader` so "the identity comes from the block the writer fills" is
not only a property of the stub.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).resolve().parents[2])
# The suite puts the backend root first, which makes `src` the backend package. The app owns its own `src` (the
# feed lives under it), so for this module the app root goes first - the same dance `test_display_wiring.py` does.
_ORIGINAL_SYS_PATH = list(sys.path)
if _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)
# The app's optional runtime imports: stub them ONLY when the interpreter does not have them, so this module never
# shadows a real install for the rest of the suite (the page imports the whole window module).
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

from PySide6.QtGui import QColor, QImage  # noqa: E402

from src.ui import shared_flag  # noqa: E402
from src.ui.ProcessTab import ProcessTab  # noqa: E402
from src.ui.video.channel import (  # noqa: E402
    DISPLAY_FMT_RGB888,
    DISPLAY_HEADER,
    DISPLAY_HEADER_SIZE,
    DISPLAY_TIME_UNSET,
)
from src.ui.video.feed import PreviewFeed  # noqa: E402

sys.path[:] = _ORIGINAL_SYS_PATH

_SEGMENT_COUNTER = 0


def _segment_name(slug: str) -> str:
    """A fresh segment name; the pid and a counter keep parallel runs out of each other's blocks."""
    global _SEGMENT_COUNTER
    _SEGMENT_COUNTER += 1
    return f"/rve-preview-feed-{slug}-{os.getpid()}-{_SEGMENT_COUNTER}"


def _drop_block(block) -> None:
    """Close a test block the reader may already have unlinked, without upsetting the resource tracker."""
    if block is None:
        return
    try:
        block.close()
    except Exception:
        pass
    try:
        block.unlink()
    except FileNotFoundError:
        pass


def _picture(width: int = 2, height: int = 2) -> QImage:
    return QImage(width, height, QImage.Format.Format_RGB888)


class _StubReader:
    """`DisplayReader` as the feed reads it: the identity attributes, one pending picture, the seek slot.

    `publish` stands for the writer's publish plus the reader's next poll: the stub names the picture it is
    about to hand over (`seq`, `media_time`) exactly where the real reader took those two fields off the header.
    """

    def __init__(self, *, attached: bool = True, generation: int = 0) -> None:
        self.attached = bool(attached)
        self.seq = 0
        self.media_time = None
        self.generation = int(generation)
        self.wants: list[bool] = []
        self.seeks: list[float] = []
        self.released = False
        self._next = None

    def publish(self, image, *, seq: int | None = None, media_time: float | None = None) -> None:
        if seq is not None:
            self.seq = int(seq)
        self.media_time = media_time
        self._next = image

    def poll(self):
        image, self._next = self._next, None
        return image

    def set_wanted(self, wanted: bool) -> None:
        self.wants.append(bool(wanted))

    def request_seek(self, seconds: float) -> int:
        self.seeks.append(float(seconds))
        self.generation = (self.generation + 1) & 0xFFFFFFFF
        return self.generation

    def seek_generation(self) -> int:
        return self.generation

    def release(self) -> None:
        self.released = True


def _feed(reader) -> PreviewFeed:
    feed = PreviewFeed(_segment_name("stub"), box=(2, 2), reader=reader)
    assert feed.attach() is True
    return feed


# --- latest-wins -----------------------------------------------------------------------------------


def test_a_polled_picture_is_the_one_picture_and_carries_its_identity():
    reader = _StubReader()
    feed = _feed(reader)
    picture = _picture()

    reader.publish(picture, seq=41, media_time=1.25)

    assert feed.poll() is True
    assert feed.image() is picture
    assert feed.shown() == ("seq", 41)
    assert feed.shown_time() == pytest.approx(1.25)
    assert feed.dropped() == 0, "nothing was replaced: the first picture is not a drop"


def test_a_picture_replaced_before_anyone_took_it_is_one_drop():
    reader = _StubReader()
    feed = _feed(reader)
    first, second = _picture(), _picture()

    reader.publish(first, seq=1, media_time=0.5)
    assert feed.poll() is True
    reader.publish(second, seq=2, media_time=1.0)
    assert feed.poll() is True

    assert feed.dropped() == 1, "the picture nobody read was replaced: one drop"
    assert feed.image() is second, "latest wins: the feed holds the newest picture, not a queue"
    assert feed.shown() == ("seq", 2)
    assert feed.poll() is False, "nothing else is waiting - the older picture was not kept behind the newer one"
    assert feed.dropped() == 1


def test_a_taken_picture_is_not_a_drop_when_the_next_one_replaces_it():
    reader = _StubReader()
    feed = _feed(reader)

    reader.publish(_picture(), seq=1, media_time=0.0)
    assert feed.poll() is True
    assert feed.image() is not None, "reading the picture is taking it"

    reader.publish(_picture(), seq=2, media_time=0.5)
    assert feed.poll() is True

    assert feed.dropped() == 0, "a picture somebody already had is replaced, not dropped"


def test_the_drop_count_is_a_count_of_the_run_and_not_of_the_hold():
    reader = _StubReader()
    feed = _feed(reader)
    for seq in (1, 2, 3):
        reader.publish(_picture(), seq=seq, media_time=None)
        assert feed.poll() is True
    assert feed.dropped() == 2, "two of the three were replaced before anyone read them"

    assert feed.image() is not None
    assert feed.dropped() == 2, "reading the picture does not clear the drop bookkeeping"


def test_the_picture_changed_signal_fires_once_per_accepted_picture():
    reader = _StubReader()
    feed = _feed(reader)
    seen = []
    feed.picture_changed.connect(lambda: seen.append(True))

    reader.publish(_picture(), seq=1, media_time=None)
    assert feed.poll() is True
    assert len(seen) == 1

    assert feed.poll() is False, "nothing new is not an accepted picture"
    assert len(seen) == 1

    reader.publish(_picture(), seq=2, media_time=None)
    assert feed.poll() is True
    assert len(seen) == 2


# --- the identity belongs to the feed --------------------------------------------------------------


def test_the_identity_is_captured_when_the_picture_is_accepted():
    reader = _StubReader()
    feed = _feed(reader)

    reader.publish(_picture(), seq=7, media_time=2.5)
    assert feed.poll() is True

    # The reader moves on (the writer published again) before the pane reads the picture: the identity must still
    # name the picture the pane is holding, never the one that replaced it in the block.
    reader.seq = 99
    reader.media_time = 9.0
    assert feed.shown() == ("seq", 7)
    assert feed.shown_time() == pytest.approx(2.5)


def test_before_the_first_picture_the_feed_names_nothing():
    reader = _StubReader()
    feed = _feed(reader)
    assert feed.image() is None
    assert feed.shown() is None
    assert feed.shown_time() is None
    assert feed.dropped() == 0


# --- no reader, no crash ---------------------------------------------------------------------------


def test_a_feed_that_never_attached_polls_nothing_and_names_nothing():
    feed = PreviewFeed(_segment_name("none"), box=(2, 2))
    assert feed.attached() is False
    assert feed.poll() is False
    assert feed.image() is None
    assert feed.shown() is None
    assert feed.shown_time() is None
    assert feed.dropped() == 0
    assert feed.request_seek(3.0) == 0, "no block, no request - never a write into somebody else's slot"
    assert feed.seek_generation() == 0


def test_a_reader_that_cannot_attach_is_refused_rather_than_used():
    reader = _StubReader(attached=False)
    feed = PreviewFeed(_segment_name("refused"), box=(2, 2), reader=reader)

    assert feed.attach() is False
    assert feed.attached() is False
    reader.publish(_picture(), seq=3, media_time=None)
    assert feed.poll() is False, "a block that is not usable is not polled"
    assert feed.image() is None


# --- the seek slot is the reader's -----------------------------------------------------------------


def test_the_seek_request_is_the_readers_newest_wins_generation():
    reader = _StubReader(generation=7)
    feed = _feed(reader)

    assert feed.seek_generation() == 7
    assert feed.request_seek(3.5) == 8
    assert reader.seeks == [pytest.approx(3.5)], "the feed writes through the reader's block"
    assert feed.seek_generation() == 8
    assert feed.request_seek(5.25) == 9


def test_the_wanted_byte_goes_to_the_reader():
    reader = _StubReader()
    feed = _feed(reader)
    assert feed.wanted() is False
    feed.set_wanted(True)
    assert reader.wants[-1] is True
    assert feed.wanted() is True
    feed.set_wanted(False)
    assert reader.wants[-1] is False


def test_detaching_releases_the_reader_and_forgets_the_picture():
    reader = _StubReader()
    feed = _feed(reader)
    reader.publish(_picture(), seq=1, media_time=None)
    assert feed.poll() is True

    feed.detach()
    assert reader.released is True
    assert feed.attached() is False
    assert feed.image() is None
    assert feed.poll() is False
    feed.detach()                       # safe twice


# --- the pane paints from the feed -----------------------------------------------------------------


class _PaneStub:
    """The pane's two calls the frame handler makes, without a window."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def set_frame(self, image, *, gui_ms=None, index=None, seq=None, media_time=None) -> None:
        self.calls.append((image, seq, media_time))

    def set_progress(self, **values) -> None:
        pass


def test_the_panes_frame_handler_takes_the_picture_and_identity_from_the_feed():
    """`picture_changed` carries nothing: the slot must read both off the feed, and its default argument is what
    lets a no-argument signal reach a handler that also serves a caller holding the image already."""
    reader = _StubReader()
    feed = _feed(reader)
    pane = _PaneStub()
    tab = types.SimpleNamespace(
        _feed=feed,
        _display=reader,
        page_handles=types.SimpleNamespace(proc=types.SimpleNamespace(previewLabel=pane)),
        _display_paints_pane=lambda: True,
        _preview_paused=False,
        _display_seek_pending=False,
        _seek_pressed_at=None,
        currentFrame=0,
        max_value=0,
        _result_playhead=lambda: 0.0,
        _clip_duration=lambda: 0.0,
    )
    feed.picture_changed.connect(types.MethodType(ProcessTab._on_display_frame, tab))

    picture = _picture()
    reader.publish(picture, seq=12, media_time=1.5)
    assert feed.poll() is True

    assert pane.calls, "the feed's signal reached the pane's handler"
    image, seq, media_time = pane.calls[-1]
    assert image is picture
    assert seq == 12
    assert media_time == pytest.approx(1.5)

    # The reader moving on must not rename the picture the pane is holding.
    reader.seq = 40
    reader.media_time = 4.0
    handler = types.MethodType(ProcessTab._on_display_frame, tab)
    handler()
    assert pane.calls[-1][1:] == (12, pytest.approx(1.5))


# --- the same rules over a real block ---------------------------------------------------------------


def test_the_identity_is_read_off_the_block_the_writer_fills():
    """No stub reader: a hand-written block, read by the real `DisplayReader`."""
    width, height = 4, 2
    name = _segment_name("block")
    block = shared_flag.create(name, DISPLAY_HEADER_SIZE + 2 * width * height * 3)
    feed = PreviewFeed(name, box=(width, height), seek_name=name + "_seek")
    try:
        assert feed.attach() is True
        pixels = bytes(((index * 17) & 0xFF) for index in range(width * height * 3))
        _write_frame(block, seq=1, width=width, height=height, pixels=pixels, media_time=2.5)
        feed.set_wanted(True)

        assert feed.poll() is True
        image = feed.image()
        assert isinstance(image, QImage)
        assert (image.width(), image.height()) == (width, height)
        assert image.pixelColor(0, 0) == QColor(pixels[0], pixels[1], pixels[2])
        assert feed.shown() == ("seq", 1)
        assert feed.shown_time() == pytest.approx(2.5)
        assert feed.dropped() == 0

        # A streaming publish names no time: the identity follows the NEW picture, and the time goes back to None.
        second = bytes(((index * 31 + 5) & 0xFF) for index in range(width * height * 3))
        _write_frame(block, seq=2, width=width, height=height, pixels=second, media_time=None)
        assert feed.poll() is True
        assert feed.shown() == ("seq", 2)
        assert feed.shown_time() is None
        assert feed.image().pixelColor(0, 0) == QColor(second[0], second[1], second[2])
        assert feed.dropped() == 0, "the first picture was read before the second landed"

        assert feed.poll() is False, "an unchanged sequence is no picture"
    finally:
        feed.detach()
        _drop_block(block)


def _write_frame(block, *, seq: int, width: int, height: int, pixels: bytes,
                 media_time: float | None) -> None:
    """Write a header and one payload the way the renderer does: payload into `(seq - 1) & 1`, then the header.

    A `media_time` of None writes the sentinel the writer packs when it cannot name the picture.
    """
    slot = (seq - 1) & 1
    slot_size = width * height * 3
    start = DISPLAY_HEADER_SIZE + slot * slot_size
    assert len(pixels) == slot_size
    block.buf[start:start + slot_size] = pixels
    stamp = DISPLAY_TIME_UNSET if media_time is None else float(media_time)
    DISPLAY_HEADER.pack_into(block.buf, 0, seq, width, height, DISPLAY_FMT_RGB888, 0, 0, 0, stamp)
