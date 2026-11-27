"""The renderer's picture names: `<output>.srcpts`, one SOURCE time per OUTPUT frame.

Why this exists (owner, 2026-11-27): the export is CFR, so a source gap is filled with copies of the previous picture
(493 source pictures -> 495 output pictures on the owner's clip). The file's own PTS is then the muxer's 24 fps grid,
not the source moment the picture carries, and the compare's source half is named by the source's PTS - at the gap the
two names drifted two frames apart and the pair was refused, which read as "the last frame has no source".

The renderer records the answer while it writes (duplicates included) and the tail and the finished-file decoder read
it. These rows pin the format and the readers' fallback, which is the part that must never invent a name.
"""

from __future__ import annotations

import pytest

from src.ui.video.frame_names import SIDECAR_SUFFIX, names_path, read_names, write_names


def test_the_sidecar_lives_beside_the_output_and_round_trips(tmp_path):
    output = str(tmp_path / "clip.mkv")
    path = write_names(output, [0.0, 0.042, 20.520833, 20.520833, 20.604167])
    assert path == output + SIDECAR_SUFFIX == names_path(output)
    assert read_names(output) == pytest.approx([0.0, 0.042, 20.520833, 20.520833, 20.604167])


def test_a_held_picture_keeps_the_time_of_the_picture_it_holds(tmp_path):
    """The two extra output pictures of the owner's clip duplicate the source frame before its gap: same name, twice."""
    output = str(tmp_path / "clip.mkv")
    write_names(output, [20.479167, 20.604000, 20.604000])
    names = read_names(output)
    assert names[-1] == names[-2] == pytest.approx(20.604), (
        "a duplicate carries the source time it duplicates, not the muxer's grid")


def test_no_sidecar_is_not_an_error(tmp_path):
    missing = str(tmp_path / "nothing.mkv")
    assert read_names(missing) is None
    assert read_names("") is None
    assert write_names(str(tmp_path / "empty.mkv"), []) is None, "an empty list writes no file"


def test_a_still_writing_list_is_read_as_far_as_it_goes(tmp_path):
    output = str(tmp_path / "clip.mkv")
    write_names(output, [0.0, 0.042])
    names = read_names(output)
    assert names == pytest.approx([0.0, 0.042]), (
        "the tail reads what exists and names the rest by the container's PTS")


def test_the_tail_names_an_output_picture_by_the_renderers_time(tmp_path):
    """The tail's own lookup: the list wins for the frames it covers, the container's PTS past its end."""
    from src.ui.video.tail import TailReader

    output = str(tmp_path / "clip.mkv")
    write_names(output, [0.0, 0.042, 20.604])
    reader = TailReader.__new__(TailReader)          # no Qt, no container: the lookup is all this row needs
    reader._path = output
    reader._names = None

    assert reader._name_for(2) == pytest.approx(20.604)
    assert reader._name_for(3) is None, "past the list the caller keeps the container's own PTS"
