"""The compare's source half is handed over for the pane's OWN moment, or as a gap - never a neighbour.

This is the guard the precedent catalogue `compare-sync-frozen-source-half-already-supplied` only PROPOSED
(`projects/rve/docs/dev/qt/sweep/precedents.yaml`: guard "proposed guard-frozen-source-half", verdict "unverified").
The 2026-11-27 defect is the same class in a new corner: the register filled and nothing presented it, so every result
was held while the pane painted a stale pair. The rule the entry states - "never cache 'supplied' beside the widget;
re-derive it from what the pane holds" - is now enforced here, through the one presentation path
(`preview/use_cases/present.py`) and the register's own bounded lookup.
"""

from __future__ import annotations

import types

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QImage  # noqa: E402

from src.ui.preview.source_register import SourceRegister  # noqa: E402
from src.ui.preview.use_cases import present  # noqa: E402


def _picture(tag: int) -> QImage:
    image = QImage(4, 4, QImage.Format.Format_RGBA8888)
    image.fill(tag)
    return image


class Pane:
    """The pane's side of the contract: its own moment, and the one source API."""

    def __init__(self, moment: float | None) -> None:
        self._moment = moment
        self.offered: list[object] = []

    def shown_moment(self) -> float | None:
        return self._moment

    def set_source(self, snapshot) -> None:
        self.offered.append(snapshot)


def _tab(pane: Pane, register: SourceRegister, *, playhead: float = 0.0, generation: int = 0,
         tolerance: float = 0.02) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        page_handles=types.SimpleNamespace(proc=types.SimpleNamespace(previewLabel=pane)),
        _source_register=register,
        _result_generation=generation,
        _source_tolerance=lambda: tolerance,
        _compare_playhead=lambda: playhead,
        _preview_playhead=playhead,
    )


def test_the_pane_is_handed_the_picture_of_its_own_moment():
    register = SourceRegister(capacity=8)
    held = register.offer(1.0, _picture(1), "decoder")
    pane = Pane(moment=1.0)

    present.sync_compare_source(_tab(pane, register))

    assert pane.offered == [held]


def test_a_moment_without_a_picture_is_a_gap_not_a_neighbour():
    """The register holds a picture of ANOTHER moment; handing it over would be the desync this exists to prevent."""
    register = SourceRegister(capacity=8)
    register.offer(1.5, _picture(1), "decoder")
    pane = Pane(moment=1.0)

    present.sync_compare_source(_tab(pane, register, tolerance=0.02))

    assert pane.offered == [None], "a neighbour's picture is not an answer"


def test_an_unnamed_result_still_gets_its_playhead_answered():
    register = SourceRegister(capacity=8)
    held = register.offer(2.0, _picture(1), "block")
    pane = Pane(moment=None)

    present.sync_compare_source(_tab(pane, register, playhead=2.0))

    assert pane.offered == [held]


def test_a_picture_from_another_generation_is_not_served():
    register = SourceRegister(capacity=8)
    register.offer(1.0, _picture(1), "decoder", generation=0)
    pane = Pane(moment=1.0)

    present.sync_compare_source(_tab(pane, register, generation=1))

    assert pane.offered == [None], "the previous render's picture answered for this one"


def test_the_asked_moment_survives_a_fill_ahead_of_it():
    """A paused pane asks for one moment repeatedly while the fill runs ahead; eviction must keep the asked picture."""
    register = SourceRegister(capacity=3)
    asked = register.offer(10.0, _picture(1), "decoder")
    assert register.picture_for(10.0, 0.02) is asked

    for step in (10.1, 10.2, 10.3):
        register.offer(step, _picture(int(step * 10)), "decoder")

    assert register.picture_for(10.0, 0.02) is asked, "the cache dropped the answer it was being asked for"


def test_a_miss_names_the_filter_that_refused_it():
    """A hold must be attributable: the generation and the tolerance are different bugs with different fixes."""
    register = SourceRegister(capacity=8)
    register.offer(6.298, _picture(1), "block", generation=3)

    assert register.picture_for(1.585, 0.02, generation=4) is None
    refusal = register.last_refusal
    assert refusal["why"] == "generation", refusal
    assert refusal["nearest_moment"] == 6.298 and refusal["nearest_generation"] == 3

    register.offer(1.60, _picture(2), "block", generation=4)
    assert register.picture_for(1.585, 0.01, generation=4) is None
    assert register.last_refusal["why"] == "tolerance", register.last_refusal


def test_a_hit_clears_the_refusal():
    register = SourceRegister(capacity=8)
    register.offer(1.0, _picture(1), "decoder")
    assert register.picture_for(1.0, 0.02) is not None
    assert register.last_refusal is None


def test_a_source_one_frame_behind_is_the_same_picture_and_two_are_not():
    """The grid error of an at-or-before producer is [0, 1) frame; half a frame refused every answer (2026-11-27)."""
    frame = 1.0 / 24.0
    register = SourceRegister(capacity=8)
    register.offer(2.0 - frame, _picture(1), "decoder")

    assert register.picture_for(2.0, 1.0 * frame) is not None, "one frame behind is the same moment's picture"
    assert register.picture_for(2.0, 0.5 * frame) is None, "half a frame cannot contain the producer's grid error"

    far = SourceRegister(capacity=8)
    far.offer(2.0 - 2 * frame, _picture(3), "decoder")
    assert far.picture_for(2.0, 1.0 * frame) is None, "two frames is a different moment"
