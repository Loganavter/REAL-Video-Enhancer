"""The render-status slot: the record both processes must spell the same way, and its torn-read guard.

The pane used to learn FPS, ETA, the frame count and the phase by parsing four regexes out of the
backend's stdout. This file pins the replacement: a second, pid-scoped shared-memory segment
(``/status_state<pid>``) whose 24-byte record is a seqlock.

Three contracts are asserted here, and none of them needs a render, torch or a GPU:

* **the two copies agree** - the backend writer (`backend/src/utils/DisplayChannel.py`) and the pane's
  reader (`src/ui/video/channel.py`) declare the same struct, offsets, phase/engine tables and name;
* **write -> read round trip** - a hand-written block read through the shipped `DisplayReader.status`,
  field for field, and a partial update leaves the fields it did not name alone;
* **a half-written record answers `None`** - the writer's odd generation is what marks a record in
  flight, and a write that completes between the reader's two probes is refused rather than reported.

It reorders `sys.path` the way `test_display_wiring.py` does: this repository has two `src` packages, and
the backend modules (`src.utils`) and the pane's modules (`src.ui`) have to be importable in one process.
"""

from __future__ import annotations

import os
import struct
import sys

import pytest

_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
for _path in (_ROOT, os.path.join(_ROOT, "backend")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
# Both `src` trees are namespace packages here; the app's `src.constants` must not be shadowed by the
# backend's same-named module when the pane's modules are imported.
sys.modules.pop("src.constants", None)

pytest.importorskip("PySide6")

from src.utils import DisplayChannel as DC  # noqa: E402
from src.ui.video import channel as CH  # noqa: E402
from src.ui.video.feed import PreviewFeed  # noqa: E402

_counter = 0


def _name(slug: str) -> str:
    """A fresh segment name: the pid and a counter keep parallel runs out of each other's blocks."""
    global _counter
    _counter += 1
    return f"/rve-status-test-{slug}-{os.getpid()}-{_counter}"


# --- the two copies --------------------------------------------------------------------------------


def test_the_writer_and_the_reader_spell_one_record():
    """A field at a different offset does not raise - it reports a counter that is a phase byte."""
    assert DC.STATUS.format == CH.STATUS.format == "<IIIffBB2x"
    assert DC.STATUS.size == CH.STATUS_SIZE == DC.STATUS_SIZE == 24
    assert DC.STATUS_FIELDS.format == CH.STATUS_FIELDS.format == "<IIffBB2x"
    for name in (
        "STATUS_OFFSET_GENERATION",
        "STATUS_OFFSET_FRAMES_RENDERED",
        "STATUS_OFFSET_FRAMES_TOTAL",
        "STATUS_OFFSET_ETA",
        "STATUS_OFFSET_FPS",
        "STATUS_OFFSET_PHASE",
        "STATUS_OFFSET_ENGINE",
    ):
        assert getattr(DC, name) == getattr(CH, name), name
    assert (DC.STATUS_OFFSET_GENERATION, DC.STATUS_OFFSET_FRAMES_RENDERED,
            DC.STATUS_OFFSET_FRAMES_TOTAL, DC.STATUS_OFFSET_ETA, DC.STATUS_OFFSET_FPS,
            DC.STATUS_OFFSET_PHASE, DC.STATUS_OFFSET_ENGINE) == (0, 4, 8, 12, 16, 20, 21)
    assert CH.STATUS_FIELDS_OFFSET == DC.STATUS_FIELDS_OFFSET == 4


def test_the_phase_and_engine_tables_are_the_same_words_on_both_sides():
    assert DC.PHASE_NAMES == CH.PHASE_NAMES
    assert DC.ENGINE_NAMES == CH.ENGINE_NAMES
    for code in (DC.PHASE_IDLE, DC.PHASE_PREPARING, DC.PHASE_BUILDING_ENGINE, DC.PHASE_RENDERING,
                 DC.PHASE_FINALISING, DC.PHASE_DONE, DC.PHASE_FAILED, DC.PHASE_PAUSED):
        assert code in DC.PHASE_NAMES and code in CH.PHASE_NAMES
    assert DC.PHASE_NAMES[DC.PHASE_BUILDING_ENGINE] == "Building engine…"
    assert DC.PHASE_NAMES[DC.PHASE_RENDERING] == "Rendering"
    assert DC.engine_code("tensorrt") == CH.ENGINE_TENSORRT
    assert DC.engine_code("pytorch") == CH.ENGINE_PYTORCH
    assert DC.engine_code("ncnn") == CH.ENGINE_NCNN
    assert DC.engine_code("no-such-backend") == CH.ENGINE_UNKNOWN


def test_the_block_name_is_pid_scoped_and_identical_on_both_sides():
    assert DC.status_state_id(4242) == CH.status_state_id(4242) == "/status_state4242"
    assert CH.STATUS_STATE_SHARED_MEMORY_ID == "/status_state" + str(os.getpid())


def test_the_pane_formats_an_eta_the_way_the_backend_prints_one():
    # `convertTime` in backend/src/FFmpeg.py: hours unpadded, minutes and seconds two digits.
    assert CH.format_eta(0) == "0:00:00"
    assert CH.format_eta(83) == "0:01:23"
    assert CH.format_eta(5023) == "1:23:43"
    assert CH.format_eta(None) == "0:00:00"


# --- the round trip --------------------------------------------------------------------------------


def _block(slug: str):
    """A pane's reader (which owns the status segment) and the backend's writer attached to it."""
    name = _name(slug)
    status_name = _name(slug + "-status")
    reader = CH.DisplayReader(name, box=(2, 2), seek_name=name + "_seek", status_name=status_name)
    assert reader.attached, "the reader creates its frame block"
    writer = DC.StatusChannel(status_name, engine=DC.ENGINE_TENSORRT)
    assert writer.attached, "the writer attaches to the pane's block"
    assert not writer.created, "the pane owns the name; the renderer must never create it"
    return reader, writer


def test_a_written_status_reads_back_field_for_field():
    reader, writer = _block("roundtrip")
    try:
        generation = writer.set(
            frames_rendered=120,
            frames_total=500,
            eta_seconds=42.5,
            fps=30.0,
            phase=DC.PHASE_RENDERING,
        )
        assert generation and generation % 2 == 0, "a completed record has an even generation"
        got = reader.status()
        assert got is not None
        assert got["generation"] == generation
        assert got["frames_rendered"] == 120
        assert got["frames_total"] == 500
        assert got["eta_seconds"] == pytest.approx(42.5)
        assert got["fps"] == pytest.approx(30.0)
        assert got["phase"] == DC.PHASE_RENDERING
        assert got["phase_name"] == "Rendering"
        assert got["engine"] == DC.ENGINE_TENSORRT
        assert got["engine_name"] == "tensorrt"
    finally:
        writer.close()
        reader.release()


def test_a_partial_update_leaves_the_fields_it_did_not_name_alone():
    reader, writer = _block("partial")
    try:
        first = writer.set(frames_rendered=10, frames_total=100, eta_seconds=90.0, fps=10.0,
                           phase=DC.PHASE_RENDERING)
        writer.set(frames_rendered=20)          # a frame tick names no ETA/FPS/phase
        got = reader.status()
        assert got is not None
        assert got["generation"] == (first + 2) & DC.UINT32_MAX
        assert got["frames_rendered"] == 20
        assert got["frames_total"] == 100, "the total survived a tick that did not name it"
        assert got["eta_seconds"] == pytest.approx(90.0)
        assert got["fps"] == pytest.approx(10.0)
        assert got["phase"] == DC.PHASE_RENDERING
    finally:
        writer.close()
        reader.release()


def test_an_unset_eta_reads_as_none_rather_than_a_negative_number():
    reader, writer = _block("unset-eta")
    try:
        writer.set(frames_rendered=0, frames_total=50, eta_seconds=DC.ETA_UNSET, fps=0.0,
                   phase=DC.PHASE_PREPARING)
        got = reader.status()
        assert got is not None
        assert got["eta_seconds"] is None
        assert got["phase_name"] == "Preparing"
    finally:
        writer.close()
        reader.release()


# --- the torn read ---------------------------------------------------------------------------------


def test_a_half_written_record_is_refused_not_mixed():
    """The writer turned the generation odd and filled the counters; the reader must answer None.

    A reader that ignored the marker would report the new frame count beside the old phase and ETA -
    exactly the mixed record this record exists to prevent.
    """
    reader, writer = _block("torn")
    try:
        writer.set(frames_rendered=10, frames_total=100, eta_seconds=90.0, fps=10.0,
                   phase=DC.PHASE_RENDERING)
        buf = reader._status_shm.buf
        # What `StatusChannel.set` does first: mark the record in flight, then write the fields.
        struct.pack_into("<I", buf, CH.STATUS_OFFSET_GENERATION, 11)   # odd
        CH.STATUS_FIELDS.pack_into(buf, CH.STATUS_FIELDS_OFFSET,
                                   99, 100, 90.0, 10.0, CH.PHASE_FINALISING, CH.ENGINE_UNKNOWN)
        assert reader.status() is None, "a record in flight must not be reported, even partially"
    finally:
        writer.close()
        reader.release()


def test_a_write_that_completes_between_the_two_probes_is_refused(monkeypatch):
    """The writer finishing between the reader's `before` probe and its field copy moves the generation.

    A real thread cannot be interleaved exactly there, so the `before` probe is made to answer two
    generations behind the block on every attempt (what a reader sees when the writer keeps completing
    between its probes): the fields are internally consistent but the generation moved, so the answer is
    `None` rather than a record that belongs to neither generation.
    """
    reader, writer = _block("race")
    try:
        writer.set(frames_rendered=10, frames_total=100, eta_seconds=90.0, fps=10.0,
                   phase=DC.PHASE_RENDERING)
        real_unpack = CH.struct.unpack_from
        probes = {"n": 0}

        def moving_generation(format_, buffer, offset=0):
            values = real_unpack(format_, buffer, offset)
            if format_ == "<I" and offset == 0:
                probes["n"] += 1
                if probes["n"] % 2 == 0:       # the "after" probe: a whole record landed in between
                    return ((values[0] + 2) & 0xFFFFFFFF,)
            return values

        monkeypatch.setattr(CH, "struct", type("FakeStruct", (),
                                               {"unpack_from": staticmethod(moving_generation)}))
        assert reader.status() is None
    finally:
        writer.close()
        reader.release()


def test_a_fresh_block_is_a_valid_idle_record():
    """A zeroed block has generation 0 (even) and phase 0: readable, and the pane sees "nothing happened"."""
    reader = CH.DisplayReader(_name("no-status"), box=(2, 2), seek_name=_name("no-status-seek"),
                              status_name=_name("no-status-block"))
    try:
        assert reader.attached
        got = reader.status()
        assert got is not None
        assert got["generation"] == 0
        assert got["frames_rendered"] == 0
        assert got["phase"] == CH.PHASE_IDLE
        assert got["eta_seconds"] is None
    finally:
        reader.release()


# --- the feed forwards it --------------------------------------------------------------------------


def test_the_feed_forwards_the_status_and_survives_a_reader_without_one():
    class Stub:
        attached = True

        def status(self):
            return {"generation": 6, "phase_name": "Rendering"}

    class Broken:
        attached = True

        def status(self):
            raise RuntimeError("no block")

    class Mute:
        attached = True

    feed = PreviewFeed(_name("feed"), box=(2, 2), reader=Stub())
    assert feed.status() == {"generation": 6, "phase_name": "Rendering"}
    assert PreviewFeed(_name("broken"), box=(2, 2), reader=Broken()).status() is None
    assert PreviewFeed(_name("mute"), box=(2, 2), reader=Mute()).status() is None
    assert PreviewFeed(_name("detached"), box=(2, 2)).status() is None
