"""The announced box: the pane asks, the renderer answers with what the block really holds.

Why this exists (owner, 2026-11-25: "при ресайзе окна разрешение не меняется, хотя должно было быть у превьювера"):
the block was sized from the pane's box BEFORE the job started, the box travelled on the command line, and a resize
only re-created the pane's texture - so the payload kept its old pixel size and the picture was only scaled. The fix
follows what the professional sources do (mpv's `resize()` vs `reconfig()`, libplacebo's `pl_swapchain_resize` returning
the size it ACTUALLY set, xpra's clamped reply): the pane announces a box in a small record, the renderer applies it at a
frame boundary and ANSWERS with the size it really fits.

These tests drive both sides in one process against one segment, which is exactly the round trip the live app performs.
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for entry in (str(ROOT), str(ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from src.ui.video.channel import BOX_FAILED, BOX_READY, BOX_REQUESTED, DisplayBoxWriter  # noqa: E402
from src.utils.DisplayChannel import DisplayBoxState  # noqa: E402


def _name(suffix: str) -> str:
    return f"/display_box_test_{suffix}_{os.getpid()}"


def _pair(suffix: str):
    name = _name(suffix)
    writer = DisplayBoxWriter(name)
    assert writer.attached, "the pane could not allocate the record"
    reader = DisplayBoxState(name)
    assert reader.attached, "the renderer could not attach to the record"
    return writer, reader, name


def test_the_pane_announces_and_the_writer_answers_with_what_it_fits():
    writer, reader, name = _pair("roundtrip")
    try:
        generation = writer.announce(800, 600)
        assert generation == 1

        pending = reader.pending()
        assert pending == (1, 800, 600), "the renderer sees the request once"
        assert reader.pending() is None, "and never acts on the same generation twice"

        reader.answer(1, 800, 600, 640, 480)
        seen = writer.answer()
        assert (seen[0], seen[1], seen[2], seen[3], seen[4]) == (1, 800, 600, 640, 480)
        assert seen[5] == BOX_FAILED, "a smaller answer than the request must be visible as such to the pane"
    finally:
        writer.release()
        reader.close()


def test_an_honoured_request_is_answered_ready_and_is_never_acted_on_twice():
    writer, reader, name = _pair("honoured")
    try:
        writer.announce(420, 512)
        assert reader.pending() == (1, 420, 512)
        reader.answer(1, 420, 512, 420, 512)
        assert writer.answer()[5] == BOX_READY

        # A resize inside the allocated maximum is another request, not a reallocation.
        writer.announce(900, 700)
        assert reader.pending() == (2, 900, 700)
        assert reader.pending() is None, "one action per generation, however often the loop asks"
        reader.answer(2, 900, 700, 900, 700)
        seen = writer.answer()
        assert (seen[0], seen[3], seen[4], seen[5]) == (2, 900, 700, BOX_READY)
    finally:
        writer.release()
        reader.close()


def test_odd_sizes_are_rounded_to_even_so_the_payload_stays_aligned():
    writer, reader, name = _pair("odd")
    try:
        writer.announce(801, 601)
        assert reader.pending() == (1, 800, 600)
        assert writer.answer()[5] == BOX_REQUESTED, "still a request until the renderer answers"
    finally:
        writer.release()
        reader.close()


def test_a_block_too_small_for_the_request_is_clamped_with_the_aspect_kept():
    """The renderer's answer is the size it FITS, and the pane can read the difference.

    This is the case libplacebo documents and xpra implements: the requester does not get to dictate a size the resource
    cannot hold, and a silent smaller payload is the failure mode every surveyed project avoids.
    """
    from types import SimpleNamespace

    from src.utils.DisplayChannel import DisplayChannel

    channel = SimpleNamespace(slot_size=421 * 512 * 3, fit_announced_box=DisplayChannel.fit_announced_box)
    real_w, real_h = DisplayChannel.fit_announced_box(channel, 1920, 1080)
    assert real_w * real_h <= channel.slot_size // 3
    assert real_w % 2 == 0 and real_h % 2 == 0
    assert abs((real_w / real_h) - (1920 / 1080)) < 0.05, "the aspect survives the clamp"

    # A request the block can hold comes back untouched.
    assert DisplayChannel.fit_announced_box(channel, 421, 512) == (420, 512)
