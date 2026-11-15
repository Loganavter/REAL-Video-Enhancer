"""Tests for TileProbe: the pre-render decision "do tiles pay on this source?".

The decision layer is numpy-only on purpose, so these run wherever the app's tests run,
including a cell with no torch and no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

np = pytest.importorskip("numpy")

from src.utils import TileProbe as TP  # noqa: E402

BLOCK = 8


def _still(h=64, w=128, level=120.0):
    return np.full((h, w), level, dtype=np.float64)


def _changed(h=64, w=128, level=120.0, patch=(0, 0, 16, 16), delta=40.0):
    frame = _still(h, w, level)
    y0, x0, y1, x1 = patch
    frame[y0:y1, x0:x1] += delta
    return frame


def test_block_means_averages_like_ceil_mode_pooling():
    frame = np.zeros((10, 10), dtype=np.float64)
    frame[8:, :] = 4.0          # the tail block holds 2 rows of 4.0
    grid = TP.block_means(frame, 4)
    assert grid.shape == (3, 3)
    assert grid[2, 0] == pytest.approx(4.0)      # only existing rows averaged
    assert grid[0, 0] == pytest.approx(0.0)


def test_block_means_rejects_a_bad_block():
    with pytest.raises(ValueError):
        TP.block_means(_still(), 0)


def test_area_tax_matches_the_documented_shape():
    # 1920x1080 fitted at 512 gives 480x540 cores; 8 windows of 516x576
    assert TP.area_tax(540, 480, 1080, 1920, 18) == pytest.approx(1.147, abs=5e-3)
    # a square grid that does not divide the frame pays the clipped tail
    assert TP.area_tax(512, 512, 1080, 1920, 18) > 1.7


def test_a_still_source_reuses_almost_every_tile():
    still = _still()
    result = TP.predict([[still] * 6], tile_h=32, tile_w=32, halo=2,
                        gate_threshold=5.0, tile_threshold=0.35, block=BLOCK,
                        gate_max_chain=1)
    # with a chain cap of 1 the gate renders every second frame, and those frames have
    # nothing changed in them either
    assert result.frames_gated == 3
    assert result.use_tiles is True
    # three rendered frames: the first has no reference, the other two reuse every tile
    assert result.tile_reuse == pytest.approx(16 / 24)


def test_the_gate_chain_forces_renders_and_tiles_still_pay_for_them():
    """A still source is not "gate skips everything": the chain cap forces a render every
    few frames, and those renders have nothing changed in them, which is exactly the case the
    tile path is for."""
    still = _still()
    result = TP.predict([[still] * 6], tile_h=32, tile_w=32, halo=2,
                        gate_threshold=5.0, tile_threshold=0.35, block=BLOCK,
                        gate_max_chain=4)
    assert result.frames_gated == 4        # renders at frame 0 and frame 5
    assert result.tile_reuse == pytest.approx(0.5)
    assert result.use_tiles is True


def test_a_moving_source_rejects_tiles():
    """Every frame changes everywhere: tiles would add the halo tax for nothing."""
    frames = [_changed(level=120.0 + 60.0 * i, patch=(0, 0, 64, 128)) for i in range(5)]
    result = TP.predict([frames], tile_h=32, tile_w=32, halo=2,
                        gate_threshold=5.0, tile_threshold=0.35, block=BLOCK)
    assert result.frames_gated == 0
    assert result.tile_reuse < result.break_even
    assert result.use_tiles is False


def test_one_moving_patch_keeps_tiles_worthwhile():
    """Only one tile of four changes: the tax is paid once, three quarters are reused."""
    frames = [_still() for _ in range(4)]
    frames = [f.copy() for f in frames]
    frames[3][0:16, 0:16] += 80.0        # one 32x32 tile of a 2x4 grid
    result = TP.predict([frames], tile_h=32, tile_w=32, halo=2,
                        gate_threshold=5.0, tile_threshold=0.35, block=BLOCK)
    # two rendered frames (the first has no reference, the third changed) x 8 tiles
    assert result.tiles_planned == 16       # 64x128 frame -> 2x4 tiles of 32
    assert result.tile_reuse == pytest.approx(7 / 16)
    assert result.use_tiles is True


def test_tile_delta_does_not_let_local_changes_cancel():
    """A checkerboard change inside one tile must count as change.

    Taking the difference of the tile's two means lets a +40 patch and a -40 patch cancel and
    reports "unchanged"; taking the mean of the absolute block deltas does not. This is the
    difference between predicting 38% reuse and the 9.5% the runtime achieved on film.
    """
    base = _still()
    changed = base.copy()
    changed[0:32, 0:32] += 60.0
    changed[0:32, 32:64] -= 60.0          # same tile row, opposite signs
    result = TP.predict([[base, changed]], tile_h=32, tile_w=32, halo=2,
                        gate_threshold=5.0, tile_threshold=0.35, block=BLOCK)
    # the gate saw a real change, so this frame was rendered; the two tiles that moved must
    # be counted as recomputed. With |mean(a) - mean(b)| they would have cancelled and the
    # prediction would have been 8 reused instead of 6.
    assert result.frames_gated == 0
    # 8 tiles from the first frame (no reference yet, all rendered) plus the 2 that moved in
    # the second one. With the difference of means both would have cancelled and this would
    # read 8.
    assert result.tiles_planned - result.tiles_reused == 10


def test_break_even_follows_the_tax():
    frames = [_still() for _ in range(4)]
    result = TP.predict([frames], tile_h=32, tile_w=32, halo=2,
                        gate_threshold=5.0, tile_threshold=0.35, block=BLOCK)
    assert result.break_even == pytest.approx(1.0 - 1.0 / result.area_tax)


def test_a_seek_breaks_continuity():
    """Each run starts fresh: the first frame of a run is never reused against the last of
    the previous one, because the runtime has no such reference after a seek."""
    run_a = [_still(level=10.0)] * 3
    run_b = [_still(level=10.0)] * 3       # identical pixels, but a different position
    result = TP.predict([run_a, run_b], tile_h=32, tile_w=32, halo=2,
                        gate_threshold=5.0, tile_threshold=0.35, block=BLOCK)
    assert result.frames_sampled == 6
    # two runs of three: one frame per run cannot be gated (no reference yet)
    assert result.frames_gated == 4


def test_scene_cut_prevents_reuse():
    dark = _still(level=10.0)
    bright = _still(level=200.0)
    result = TP.predict([[dark, bright, bright]], tile_h=32, tile_w=32, halo=2,
                        gate_threshold=5.0, tile_threshold=0.35, block=BLOCK)
    assert result.frames_gated == 1        # the second `bright` only


def test_empty_input_is_refused():
    with pytest.raises(ValueError):
        TP.predict([], tile_h=32, tile_w=32, halo=2, gate_threshold=5.0,
                   tile_threshold=0.35, block=BLOCK)
    with pytest.raises(ValueError):
        TP.predict([[]], tile_h=32, tile_w=32, halo=2, gate_threshold=5.0,
                   tile_threshold=0.35, block=BLOCK)


def test_sample_luma_splits_runs_and_parses_frames():
    """The seek/parse layer, driven by a fake runner: no ffmpeg, no encoder."""
    width, height = 8, 4
    frame = bytes([7]) * (width * height)

    calls = []

    def runner(argv, expected):
        calls.append(argv)
        return frame * 2          # two frames per call

    runs = TP.sample_luma("in.mkv", width=width, height=height, positions=2,
                          per_position=2, runner=runner)
    assert len(runs) == 2 and all(len(r) == 2 for r in runs)
    assert runs[0][0].shape == (height, width)
    assert float(runs[0][0].mean()) == 7.0
    assert any("-ss" in argv for argv in calls)


def test_sample_luma_drops_a_short_read():
    """A truncated seek (end of file) yields fewer frames, never a broken frame."""
    width, height = 8, 4
    frame = bytes([9]) * (width * height)

    def runner(argv, expected):
        return frame            # one frame when two were asked for

    runs = TP.sample_luma("in.mkv", width=width, height=height, positions=1,
                          per_position=2, runner=runner)
    assert len(runs) == 1 and len(runs[0]) == 1


def test_seek_points_spread_and_never_overlap():
    points = TP._seek_points(1000, 4, 2)
    assert len(points) == 4
    assert points == sorted(points)
    assert all(b - a >= 2 for a, b in zip(points, points[1:]))
    # percentages when the frame count is unknown
    assert TP._seek_points(None, 3, 2) == [0.0, 0.25, 0.5]


def test_probe_reports_a_decodable_source(monkeypatch):
    """`probe()` wires the fit + the sampler + the decision together."""
    width, height = 128, 64
    frame = np.full((height, width), 100.0)

    def fake_sample(path, *, width, height, positions, per_position, ffmpeg_path,
                    expected_frames=None, runner=None):
        return [[frame.copy() for _ in range(4)]]

    monkeypatch.setattr(TP, "sample_luma", fake_sample)
    result = TP.probe("in.mkv", width=width, height=height, tile_size=32, halo=2,
                      gate_threshold=5.0, tile_threshold=0.35, block=BLOCK,
                      gate_max_chain=1)
    assert result.frames_sampled == 4
    assert result.use_tiles is True
    assert "tiles ON" in result.stats()


def test_probe_with_no_frames_says_so(monkeypatch):
    monkeypatch.setattr(TP, "sample_luma", lambda *a, **k: [])
    result = TP.probe("in.mkv", width=64, height=64, tile_size=32, halo=2)
    assert result.use_tiles is False
    assert "no frames" in result.reason
