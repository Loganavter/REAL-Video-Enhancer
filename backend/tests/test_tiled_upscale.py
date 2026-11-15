"""Tests for ``backend/src/pytorch/TiledUpscale.py`` (tile-level temporal reuse).

Three layers, because the cell that has to verify this ships without torch (the bundled
interpreter lives outside the worker jail):

* **geometry / mask** - no torch, no numpy: tile grid, halo window, replicate padding
  (clamped index vector), HR write/read slices, per-tile decision, scene cut. A tile layout
  bug is a geometry bug long before it is a tensor bug, so this is the layer that must be
  green everywhere;
* **numpy composite** - the same framing applied to a stand-in "network" (three box-blur
  passes + nearest x-scale, receptive radius 6 LR px) compared against the whole-frame
  reference: this is where the seam error and the load-bearingness of the halo are
  measured, and where the skip rate of the decision logic is measured on a synthetic
  sequence;
* **torch** - the real class (``plan``/``compose``, in-place compositing, chain limit,
  scene cut), skipped when torch is not installed (``pytest.importorskip``).
"""

from __future__ import annotations

import math
import os
import sys

import pytest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from src.pytorch import TiledUpscale as TU  # noqa: E402

SEAM_TOL = 1.0 / 255.0          # the acceptance tolerance from the task spec
DUMMY_RADIUS = 6                # receptive radius of the stand-in network (LR px)
TILE_LR = 64
HALO_LR = 18                    # production halo (18 >= DUMMY_RADIUS)


# --------------------------------------------------------------------------------------
# stand-in network (numpy): 3 box-blur passes (radius 2 each) + nearest x-scale
# --------------------------------------------------------------------------------------
def _box2d(array, kernel: int):
    """Mean filter with edge (replicate) padding, via a summed-area table."""
    np = pytest.importorskip("numpy")
    radius = kernel // 2
    padded = np.pad(array, radius, mode="edge")
    cum = None
    cum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    cum = np.pad(cum, ((1, 0), (1, 0)), mode="constant")
    height, width = array.shape
    window = (cum[kernel:kernel + height, kernel:kernel + width]
              - cum[0:height, kernel:kernel + width]
              - cum[kernel:kernel + height, 0:width]
              + cum[0:height, 0:width])
    return window / float(kernel * kernel)


def np_net(array, scale: int = 2):
    """Stand-in SR network: receptive radius DUMMY_RADIUS, pointwise upscale."""
    np = pytest.importorskip("numpy")
    out = np.asarray(array, dtype=np.float64)
    for _ in range(3):
        out = _box2d(out, 5)
    if scale != 1:
        out = np.repeat(np.repeat(out, scale, axis=0), scale, axis=1)
    return out


def np_composite(frame, scale: int, tile_lr: int, halo: int, given_cores=None,
                 tile_w=None):
    """Render ``given_cores`` (default: the whole grid) through ``np_net`` and paste back.

    Uses the module's own geometry helpers - the point of the test is that those helpers
    place every core tile exactly where the whole-frame reference expects it.
    """
    np = pytest.importorskip("numpy")
    height, width = frame.shape
    cores = (TU.tile_grid(height, width, tile_lr, tile_w) if given_cores is None
             else given_cores)
    out = np.zeros((height * scale, width * scale), dtype=np.float64)
    written = np.zeros((height * scale, width * scale), dtype=np.int32)
    for core in cores:
        window = TU.tile_window(core, halo, height, width, tile_lr, tile_w)
        tile_in = frame[np.ix_(list(window.rows), list(window.cols))]
        tile_out = np_net(tile_in, scale)
        read_y, read_x = TU.hr_read_slice(window, scale)
        write_y, write_x = TU.hr_write_slice(core, scale)
        out[write_y, write_x] = tile_out[read_y, read_x]
        written[write_y, write_x] += 1
    return out, written


def _interior(height: int, width: int, margin: int):
    """True away from the frame border (the materialised-replicate region)."""
    np = pytest.importorskip("numpy")
    mask = np.ones((height, width), dtype=bool)
    if margin > 0:
        mask[:margin] = False
        mask[-margin:] = False
        mask[:, :margin] = False
        mask[:, -margin:] = False
    return mask


# --------------------------------------------------------------------------------------
# geometry: tile grid
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("height,width,tile_lr", [
    (1080, 1920, 512),      # the production case: not divisible by 512
    (137, 203, 64),         # small, both axes partial
    (512, 512, 512),        # exactly one tile
    (37, 41, 16),           # smaller than one window would be
])
def test_tile_grid_partitions_frame(height, width, tile_lr):
    cores = TU.tile_grid(height, width, tile_lr)

    assert len(cores) == math.ceil(height / tile_lr) * math.ceil(width / tile_lr)
    for y0, y1, x0, x1 in cores:
        assert 0 <= y0 < y1 <= height
        assert 0 <= x0 < x1 <= width
        assert y1 - y0 <= tile_lr and x1 - x0 <= tile_lr

    # no overlap, complete coverage: every pixel belongs to exactly one core
    counts = [0] * (height * width)
    for y0, y1, x0, x1 in cores:
        for y in range(y0, y1):
            row = y * width
            for x in range(x0, x1):
                counts[row + x] += 1
    assert min(counts) == 1
    assert max(counts) == 1

    # row-major order (plan() reports cores in this order)
    assert cores[0][0] == 0 and cores[0][2] == 0
    assert cores == sorted(cores, key=lambda c: (c[0], c[2]))


# --------------------------------------------------------------------------------------
# geometry: the divisor grid (cores that divide the frame exactly)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("size,target,expected", [
    (1080, 512, 540),       # 2 rows: the user's 480x540 grid for 1920x1080
    (1920, 512, 480),       # 4 columns
    (1440, 512, 480),       # 3 columns
    (1080, 96, 108),        # nearest tile *count* wins: 10 tiles of 108
    (1081, 512, 1081),      # prime height: one tile, never a clipped tail
    (2160, 512, 540),
])
def test_divisor_tile_lands_on_a_divisor(size, target, expected):
    side = TU.divisor_tile(size, target)
    assert side == expected
    assert size % side == 0, f"{side} does not divide {size}"


def test_divisor_grid_removes_the_tail_area():
    """1920x1080 at a 512 target: 8 exact windows of 516x576 instead of 12 of 548x548."""
    th, tw = TU.divisor_tile(1080, 512), TU.divisor_tile(1920, 512)
    cores = TU.tile_grid(1080, 1920, th, tw)
    assert len(cores) == 8
    assert all(c[1] - c[0] == th and c[3] - c[2] == tw for c in cores)
    windows = [TU.tile_window(c, HALO_LR, 1080, 1920, th, tw) for c in cores]
    assert {w.window_height for w in windows} == {th + 2 * HALO_LR}
    assert {w.window_width for w in windows} == {tw + 2 * HALO_LR}
    exact = len(windows) * (th + 2 * HALO_LR) * (tw + 2 * HALO_LR)
    square = len(TU.tile_grid(1080, 1920, 512)) * (512 + 2 * HALO_LR) ** 2
    assert exact / (1080 * 1920) < 1.16, exact / (1080 * 1920)
    assert square / (1080 * 1920) > 1.7, square / (1080 * 1920)
    print(f"\n[area] divisor grid {exact / (1080 * 1920):.3f}x vs square-512 "
          f"{square / (1080 * 1920):.3f}x the frame")


def test_rectangular_window_is_fixed_shape():
    window = TU.tile_window((100, 260, 40, 200), HALO_LR, 1080, 1920, 160, 160 + 24)
    assert window.window_height == 160 + 2 * HALO_LR
    assert window.window_width == 184 + 2 * HALO_LR
    assert window.core_height == 160 and window.core_width == 160


def test_tile_grid_rejects_empty_input():
    with pytest.raises(ValueError):
        TU.tile_grid(0, 10, 8)
    with pytest.raises(ValueError):
        TU.tile_grid(10, 10, 0)


# --------------------------------------------------------------------------------------
# geometry: halo window + replicate padding
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("height,width,tile_lr,halo", [
    (1080, 1920, 512, 18),
    (137, 203, 64, 18),
    (37, 41, 16, 18),       # frame smaller than the window: pure replicate on two sides
    (64, 64, 64, 0),        # no halo at all
])
def test_window_is_fixed_shape_with_core_at_offset(height, width, tile_lr, halo):
    window_size = tile_lr + 2 * halo
    for core in TU.tile_grid(height, width, tile_lr):
        y0, y1, x0, x1 = core
        win = TU.tile_window(core, halo, height, width, tile_lr)

        # rule 1: the network always sees (1, 3, T, T)
        assert len(win.rows) == window_size
        assert len(win.cols) == window_size
        # rows/cols are valid frame indices (that is the replicate padding: clamped, not 0)
        assert all(0 <= r < height for r in win.rows)
        assert all(0 <= c < width for c in win.cols)
        assert list(win.rows) == sorted(win.rows)
        assert list(win.cols) == sorted(win.cols)
        # the core sits inside the window at (top, left) and is exactly its pixels
        assert win.rows[win.top] == y0 and win.rows[win.top + (y1 - y0) - 1] == y1 - 1
        assert win.cols[win.left] == x0 and win.cols[win.left + (x1 - x0) - 1] == x1 - 1
        # halo is clipped, never wrapped
        assert 0 <= win.top <= halo and 0 <= win.left <= halo


def test_replicate_window_matches_edge_padding():
    np = pytest.importorskip("numpy")
    plane = np.arange(1, 6, dtype=np.float64)      # 1..5
    padded = np.pad(plane, 16, mode="edge")        # the reference: real edge padding
    for start in (-16, -3, -1, 0, 2, 4, 6, 8):
        indices = TU.replicate_window(start, 7, 5)
        assert indices == [min(max(i, 0), 4) for i in range(start, start + 7)]
        assert np.array_equal(plane[indices], padded[start + 16:start + 16 + 7])


def test_hr_slices_cover_output_exactly():
    np = pytest.importorskip("numpy")
    scale = 2
    for height, width, tile_lr in ((120, 300, 64), (1080, 1920, 512), (37, 41, 16)):
        counts = np.zeros((height * scale, width * scale), dtype=np.int32)
        for core in TU.tile_grid(height, width, tile_lr):
            write_y, write_x = TU.hr_write_slice(core, scale)
            counts[write_y, write_x] += 1
            assert write_y.stop - write_y.start == (core[1] - core[0]) * scale
            assert write_x.stop - write_x.start == (core[3] - core[2]) * scale
        assert counts.min() == 1 and counts.max() == 1


def test_hr_read_slice_matches_core_size():
    for height, width, tile_lr, halo in ((137, 203, 64, 18), (1080, 1920, 512, 18)):
        for core in TU.tile_grid(height, width, tile_lr):
            win = TU.tile_window(core, halo, height, width, tile_lr)
            read_y, read_x = TU.hr_read_slice(win, 2)
            assert read_y.stop - read_y.start == win.core_height * 2 <= win.window_size * 2
            assert read_x.stop - read_x.start == win.core_width * 2 <= win.window_size * 2
            assert read_y.start >= 0 and read_x.start >= 0
            assert read_y.stop <= win.window_size * 2
            assert read_x.stop <= win.window_size * 2


# --------------------------------------------------------------------------------------
# geometry: low-pass block mapping
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("height,width,downscale", [(1080, 1920, 8), (137, 203, 8), (41, 37, 4)])
def test_block_bounds_cover_every_block(height, width, downscale):
    blocks_h, blocks_w = TU.block_grid(height, width, downscale)
    assert blocks_h == math.ceil(height / downscale)
    assert blocks_w == math.ceil(width / downscale)
    counts = [[0] * blocks_w for _ in range(blocks_h)]
    for core in TU.tile_grid(height, width, 64):
        by0, by1, bx0, bx1 = TU.block_bounds(core, downscale, blocks_h, blocks_w)
        assert 0 <= by0 < by1 <= blocks_h
        assert 0 <= bx0 < bx1 <= blocks_w
        for by in range(by0, by1):
            for bx in range(bx0, bx1):
                counts[by][bx] += 1
    for by in range(blocks_h):
        for bx in range(blocks_w):
            assert counts[by][bx] >= 1     # complete coverage (sharing is allowed)


def test_block_bounds_are_disjoint_for_aligned_tiles():
    blocks_h, blocks_w = TU.block_grid(1024, 1024, 8)
    counts = [[0] * blocks_w for _ in range(blocks_h)]
    for core in TU.tile_grid(1024, 1024, 128):      # 128 % 8 == 0
        by0, by1, bx0, bx1 = TU.block_bounds(core, 8, blocks_h, blocks_w)
        for by in range(by0, by1):
            for bx in range(bx0, bx1):
                counts[by][bx] += 1
    assert all(counts[by][bx] == 1 for by in range(blocks_h) for bx in range(blocks_w))


# --------------------------------------------------------------------------------------
# mask decision
# --------------------------------------------------------------------------------------
def test_decide_tiles_static_frame_reuses_everything():
    assert TU.decide_tiles([0.0] * 6, threshold=0.35, chains=[0] * 6, max_chain=2) == \
        [False] * 6


def test_decide_tiles_changed_tile_is_recomputed():
    deltas = [0.0, 0.0, 4.0, 0.1, 0.0, 0.0]
    mask = TU.decide_tiles(deltas, threshold=0.35, chains=[0] * 6, max_chain=2)
    assert mask == [False, False, True, False, False, False]


def test_decide_tiles_threshold_is_inclusive_of_small_noise():
    # 0.25-0.35 levels of low-pass delta predict a sub-1-level SR change: skip it
    mask = TU.decide_tiles([0.34, 0.36], threshold=0.35, chains=[0, 0], max_chain=2)
    assert mask == [False, True]


def test_decide_tiles_chain_limit_forces_recompute():
    # a tile may be served from cache for at most max_chain consecutive frames
    assert TU.decide_tiles([0.0], 0.35, chains=[0], max_chain=2) == [False]
    assert TU.decide_tiles([0.0], 0.35, chains=[1], max_chain=2) == [False]
    assert TU.decide_tiles([0.0], 0.35, chains=[2], max_chain=2) == [True]
    assert TU.decide_tiles([0.0], 0.35, chains=[7], max_chain=2) == [True]


def test_decide_tiles_max_chain_zero_always_recomputes():
    assert TU.decide_tiles([0.0, 0.0], 0.35, chains=[0, 0], max_chain=0) == [True, True]


def test_decide_tiles_rejects_mismatched_chain_list():
    with pytest.raises(ValueError):
        TU.decide_tiles([0.0, 0.0], 0.35, chains=[0], max_chain=2)


def test_scene_cut_threshold():
    assert TU.is_scene_cut(100.0, 140.0, 32.0) is True
    assert TU.is_scene_cut(140.0, 100.0, 32.0) is True
    assert TU.is_scene_cut(100.0, 110.0, 32.0) is False
    assert TU.is_scene_cut(100.0, 132.0, 32.0) is True      # boundary is inclusive


# --------------------------------------------------------------------------------------
# numpy: seam error of a composited frame vs the whole-frame reference
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("height,width,tile_lr", [
    (137, 203, 64),         # both axes partial, 3x4 tiles
    (128, 128, 64),         # exactly 2x2, no partial tiles
    (300, 517, 128),
])
def test_composite_matches_whole_frame_within_one_level(height, width, tile_lr):
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(20240517)
    frame = rng.random((height, width))

    whole = np_net(frame, scale=2)
    composite, written = np_composite(frame, scale=2, tile_lr=tile_lr, halo=HALO_LR)

    assert written.min() == 1 and written.max() == 1     # every HR pixel written once
    error = np.abs(composite - whole)
    interior = _interior(*error.shape, margin=DUMMY_RADIUS * 2)   # margin in HR px
    interior_error = float(error[interior].max())
    border_error = float(error[~interior].max())
    # halo (18) > receptive radius (6): no seam anywhere between tiles, exactly 0
    assert interior_error <= SEAM_TOL, f"seam error {interior_error * 255:.3f}/255"
    print(f"\n[seam] {height}x{width} tile={tile_lr} halo={HALO_LR}: "
          f"interior max |composite - whole-frame| = {interior_error * 255:.4f}/255, "
          f"frame-border band ({DUMMY_RADIUS} LR px) = {border_error * 255:.2f}/255")


def test_rectangular_tiles_composite_without_seams():
    """Non-square cores (a divisor grid) must place tiles exactly like square ones."""
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(4242)
    height, width = 192, 320
    frame = rng.random((height, width))

    whole = np_net(frame, scale=2)
    composite, written = np_composite(frame, scale=2, tile_lr=48, tile_w=80, halo=HALO_LR)

    assert written.min() == 1 and written.max() == 1
    error = np.abs(composite - whole)
    interior = _interior(*error.shape, margin=DUMMY_RADIUS * 2)
    interior_error = float(error[interior].max())
    assert interior_error <= SEAM_TOL, f"seam error {interior_error * 255:.3f}/255"
    print(f"\n[seam] 192x320 rectangular 48x80 halo={HALO_LR}: interior max error "
          f"{interior_error * 255:.4f}/255")


def test_halo_is_load_bearing():
    """The same composite with halo=0 must show seams - otherwise the test above proves
    nothing about padding, only about a pointwise network."""
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(7)
    frame = rng.random((137, 203))

    whole = np_net(frame, scale=2)
    composite, _ = np_composite(frame, scale=2, tile_lr=64, halo=0)

    error = np.abs(composite - whole)
    interior = _interior(*error.shape, margin=DUMMY_RADIUS * 2)
    max_error = float(error[interior].max())
    assert max_error > SEAM_TOL, (
        f"halo=0 produced {max_error * 255:.3f}/255 in the interior - the stand-in "
        f"network has no receptive field, so the seam test would be vacuous")
    print(f"\n[seam] halo=0 control: interior max |composite - whole-frame| = "
          f"{max_error * 255:.2f}/255 (>{SEAM_TOL * 255:.2f}/255, as expected)")


def test_partial_recompute_writes_only_inside_the_changed_tile():
    """Overlaying one recomputed tile leaves every other HR pixel bit-identical."""
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(11)
    height, width, tile_lr, scale = 192, 320, 64, 2
    before = rng.random((height, width))
    after = before.copy()
    after[70:110, 10:50] += 0.4                   # one changed rectangle, inside core 5

    full_before, _ = np_composite(before, scale=scale, tile_lr=tile_lr, halo=HALO_LR)
    cores = TU.tile_grid(height, width, tile_lr)
    target = cores[5]
    tile_after, _ = np_composite(after, scale=scale, tile_lr=tile_lr, halo=HALO_LR,
                                 given_cores=[target])

    out = full_before.copy()                      # == prev_hr handed to compose()
    write_y, write_x = TU.hr_write_slice(target, scale)
    inside = np.zeros_like(out, dtype=bool)
    inside[write_y, write_x] = True
    out[write_y, write_x] = tile_after[write_y, write_x]

    assert np.array_equal(out[~inside], full_before[~inside])   # untouched, bit-identical
    assert not np.array_equal(out[inside], full_before[inside])  # the changed tile moved


# --------------------------------------------------------------------------------------
# numpy: skip rate of the decision logic on a synthetic sequence
# --------------------------------------------------------------------------------------
def _synthetic_sequence(frames: int, height: int, width: int):
    """Textured static background + a moving box, a freeze and a scene cut."""
    np = pytest.importorskip("numpy")
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    background = 0.5 + 0.15 * np.sin(xx / 17.0) + 0.12 * np.cos(yy / 23.0)
    rng = np.random.default_rng(4242)
    frozen = background.copy()
    frozen[40:136, 40:136] += 0.2
    for index in range(frames):
        if index == frames // 2:                     # a "still" moment: exact duplicates
            frame = frozen.copy()
        elif index > frames // 2 and index <= frames // 2 + 4:
            frame = frozen.copy()
        elif index > 3 * frames // 4:                 # hard scene cut (+60 levels)
            frame = 0.30 * background + 0.60 + 0.10 * np.sin(xx / 5.0)
        else:
            offset = 6 * index
            frame = background.copy()
            y = 32 + (offset % (height - 160))
            x = 32 + (offset % (width - 160))
            frame[y:y + 96, x:x + 96] += 0.2
        # codec-like dither: small, everywhere, below the threshold
        frame = frame + 0.0004 * rng.standard_normal((height, width))
        yield np.clip(frame, 0.0, 1.0)


@pytest.mark.parametrize("max_chain", [2, 10])
def test_skip_rate_on_synthetic_sequence(max_chain):
    np = pytest.importorskip("numpy")
    height, width, downscale = 512, 768, 8
    tile_lr, halo, threshold = 128, HALO_LR, 0.35

    blocks_h, blocks_w = TU.block_grid(height, width, downscale)
    cores = TU.tile_grid(height, width, tile_lr)
    chains = [0] * len(cores)
    prev_blocks = None

    frames = reused_tiles = total_tiles = full_reuse = scene_cuts = 0
    for frame in _synthetic_sequence(60, height, width):
        frames += 1
        luma = frame[0:blocks_h * downscale, 0:blocks_w * downscale]
        blocks = luma.reshape(blocks_h, downscale, blocks_w, downscale).mean(axis=(1, 3)) * 255.0
        if prev_blocks is None or prev_blocks.shape != blocks.shape:
            mask = [True] * len(cores)
        else:
            cut = TU.is_scene_cut(float(prev_blocks.mean()), float(blocks.mean()), 32.0)
            if cut:
                scene_cuts += 1
                chains = [0] * len(cores)
                mask = [True] * len(cores)
            else:
                deltas = []
                for core in cores:
                    by0, by1, bx0, bx1 = TU.block_bounds(core, downscale, blocks_h, blocks_w)
                    deltas.append(float(np.abs(blocks[by0:by1, bx0:bx1]
                                               - prev_blocks[by0:by1, bx0:bx1]).max()))
                mask = TU.decide_tiles(deltas, threshold, chains, max_chain)
        chains = [0 if must else chain + 1 for must, chain in zip(mask, chains)]
        prev_blocks = blocks
        total_tiles += len(mask)
        reused_tiles += sum(1 for must in mask if not must)
        if not any(mask):
            full_reuse += 1

    skip_pct = 100.0 * reused_tiles / total_tiles
    print(f"\n[skip rate] {frames} synthetic frames {height}x{width}, tile={tile_lr} "
          f"(grid {len(cores)}), threshold={threshold}/255, max_chain={max_chain}: "
          f"{skip_pct:.1f}% tiles reused, {full_reuse}/{frames} frames fully reused, "
          f"{scene_cuts} scene cut(s)")

    assert scene_cuts == 1                              # the hard cut is detected
    # max_chain caps how long a tile may live in cache, so it also caps the skip rate:
    # 1 - 1/(max_chain + 1) of the tiles are available for reuse at best
    assert skip_pct > 50.0, f"skip rate {skip_pct:.1f}% is implausibly low"
    assert skip_pct <= 100.0 * max_chain / (max_chain + 1) + 20.0
    assert full_reuse >= 1, "the frozen stretch should allow whole-frame reuse"


# --------------------------------------------------------------------------------------
# torch: the real class
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def torch_mod():
    return pytest.importorskip("torch")


def _net(torch, scale: int):
    """3 box-blur passes (radius 2 each, replicate padding) + nearest x-scale."""
    F = torch.nn.functional

    def upscale(x):
        y = x
        for _ in range(3):
            y = F.pad(y, (2, 2, 2, 2), mode="replicate")
            y = F.avg_pool2d(y, 5, stride=1)
        if scale != 1:
            y = F.interpolate(y, scale_factor=scale, mode="nearest")
        return y

    return upscale


def _frame(torch, height: int, width: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((1, 3, height, width), generator=generator, dtype=torch.float32)


def test_torch_static_sequence_reuses_the_whole_frame(torch_mod):
    torch = torch_mod
    upscaler = TU.TiledUpscaler(tile_lr=64, halo_lr=HALO_LR, threshold=0.35, max_chain=2)
    lr = _frame(torch, 128, 192)

    first = upscaler.plan(lr)
    assert first == TU.tile_grid(128, 192, 64)
    hr = upscaler.compose(2, _net(torch, 2), None, lr)
    assert hr.shape == (1, 3, 256, 384)

    assert upscaler.plan(lr.clone()) is None            # identical frame -> whole-frame reuse
    assert upscaler.plan(lr.clone()) is None            # second reuse is still inside the chain
    assert upscaler.plan(lr.clone()) == TU.tile_grid(128, 192, 64)   # chain limit (2) hit
    assert upscaler.frames_reused == 2


def test_torch_changed_rectangle_recomputes_that_tile_and_its_neighbours(torch_mod):
    """Dilation: a tile next to a recomputed one is recomputed too.

    Without it, neighbouring tiles reference frames of different ages and an object crossing a tile
    boundary is cut in half - measured as 0.287% of pixels above 8 levels against 0.071% with the
    mask dilated. `dilate=False` gives the old behaviour and is pinned separately.
    """
    torch = torch_mod
    upscaler = TU.TiledUpscaler(tile_lr=96, halo_lr=HALO_LR, threshold=0.35, max_chain=2)
    lr = _frame(torch, 160, 288)
    upscaler.plan(lr)
    hr = upscaler.compose(2, _net(torch, 2), None, lr)

    changed = lr.clone()
    changed[:, :, 20:60, 110:150] = 1.0                 # inside core (0, 96, 96, 192)
    rects = upscaler.plan(changed)
    assert (0, 96, 96, 192) in rects, "the changed tile itself"
    assert len(rects) > 1, "its neighbours came along"

    calls = []

    def counting(x):
        calls.append(tuple(x.shape))
        return _net(torch, 2)(x)

    out = upscaler.compose(2, counting, hr, changed)
    assert calls == [(1, 3, 132, 132)] * len(rects)      # T = 96 + 2*18, one call per tile
    assert out is hr                                     # composed in place, no clone


def test_torch_dilation_can_be_switched_off(torch_mod):
    torch = torch_mod
    upscaler = TU.TiledUpscaler(tile_lr=96, halo_lr=HALO_LR, threshold=0.35, max_chain=2,
                                dilate=False)
    lr = _frame(torch, 160, 288)
    upscaler.plan(lr)
    upscaler.compose(2, _net(torch, 2), None, lr)

    changed = lr.clone()
    changed[:, :, 20:60, 110:150] = 1.0
    assert upscaler.plan(changed) == [(0, 96, 96, 192)]

    write_y, write_x = TU.hr_write_slice((0, 96, 96, 192), 2)
    inside = torch.zeros_like(hr, dtype=torch.bool)
    inside[:, :, write_y, write_x] = True
    assert torch.equal(out[~inside], before[~inside])    # untouched tiles: bit-identical
    assert not torch.equal(out[inside], before[inside])  # changed tile: recomputed


def test_torch_threshold_zero_matches_whole_frame(torch_mod):
    torch = torch_mod
    upscaler = TU.TiledUpscaler(tile_lr=64, halo_lr=HALO_LR, threshold=0.0, max_chain=2)
    lr = _frame(torch, 137, 203, seed=3)

    rects = upscaler.plan(lr)
    assert rects == TU.tile_grid(137, 203, 64)           # threshold 0 -> everything
    composite = upscaler.compose(2, _net(torch, 2), None, lr)
    whole = _net(torch, 2)(lr)

    margin = DUMMY_RADIUS * 2                            # margin in HR pixels
    error = (composite - whole).abs()
    interior = error[:, :, margin:-margin, margin:-margin].max().item()
    border = error.max().item()
    print(f"\n[seam/torch] 137x203 tile=64 halo={HALO_LR}: interior error "
          f"{interior * 255:.4f}/255, frame-border band {border * 255:.2f}/255")
    assert interior <= SEAM_TOL


def test_torch_non_divisible_frame_uses_fixed_tile_shape(torch_mod):
    torch = torch_mod
    tile_lr, halo = 512, HALO_LR
    upscaler = TU.TiledUpscaler(tile_lr=tile_lr, halo_lr=halo, threshold=0.35, max_chain=2)
    lr = _frame(torch, 1080, 1920)

    shapes = []

    def recording(x):
        shapes.append(tuple(x.shape))
        return x                                      # scale 1: identity is enough here

    rects = upscaler.plan(lr)
    assert len(rects) == 12                            # 3 x 4 tiles, 1080x1920
    assert rects[-1] == (1024, 1080, 1536, 1920)       # clipped last row/column
    out = upscaler.compose(1, recording, None, lr)

    assert out.shape == (1, 3, 1080, 1920)
    assert shapes == [(1, 3, 548, 548)] * 12           # T = 512 + 2*18, static shape
    assert torch.equal(out, lr)                        # identity net, full coverage


def test_torch_rejects_wrong_output_shape(torch_mod):
    torch = torch_mod
    upscaler = TU.TiledUpscaler(tile_lr=64, halo_lr=HALO_LR, threshold=0.35)
    lr = _frame(torch, 64, 64)
    upscaler.plan(lr)
    with pytest.raises(ValueError):
        upscaler.compose(2, lambda x: x, None, lr)     # forgot the x2 upscale


def test_torch_scene_cut_and_reset(torch_mod):
    torch = torch_mod
    upscaler = TU.TiledUpscaler(tile_lr=64, halo_lr=HALO_LR, threshold=0.35, max_chain=2)
    lr = _frame(torch, 128, 128, seed=5)
    upscaler.plan(lr)
    hr = upscaler.compose(2, _net(torch, 2), None, lr)
    assert upscaler.plan(lr.clone()) is None

    cut = (lr + 0.5).clamp(0.0, 1.0)                   # global brightness jump
    assert upscaler.plan(cut) == TU.tile_grid(128, 128, 64)
    assert upscaler.scene_cuts == 1

    upscaler.reset()
    assert upscaler.can_reuse is False
    assert upscaler.plan(lr.clone()) == TU.tile_grid(128, 128, 64)


def test_torch_resolution_change_recovers(torch_mod):
    torch = torch_mod
    upscaler = TU.TiledUpscaler(tile_lr=64, halo_lr=HALO_LR, threshold=0.35)
    small = _frame(torch, 128, 128)
    upscaler.plan(small)
    hr = upscaler.compose(2, _net(torch, 2), None, small)

    bigger = _frame(torch, 192, 256)
    rects = upscaler.plan(bigger)
    assert rects == TU.tile_grid(192, 256, 64)
    # prev_hr has the wrong shape: compose must start a fresh buffer, not crash or wrap
    out = upscaler.compose(2, _net(torch, 2), hr, bigger)
    assert out.shape == (1, 3, 384, 512)
