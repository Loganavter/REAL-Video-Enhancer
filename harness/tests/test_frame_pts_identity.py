"""A frame built from another frame must carry the other frame's identity.

The 2026-11-27 backend audit found the export's media time silently reverting to the pipe counter: `FFmpegBuffers.py:224`
is the only writer of `Frame.pts`, and every upscaler built a fresh `Frame` without it, so `RenderVideo.py:1655` always
took the `self._stream_media_time(frames_rendered)` fallback and the CFR rebuild below it never fired while the reader
ran `-fps_mode passthrough`. A gapped source then exported short, and the pane's media time was a counter again.

The defect is structural, not a forgotten line: each stage had its own reason to believe it was making a new picture, and
nothing named the other half - that the picture is the SAME one and must carry the same name. So the fix has a seam
(`Frame.carry_identity_from`) and this file has two halves: the seam behaves, and no construction site can skip it
silently. A site that legitimately synthesises a new time says so in a `# identity:` comment; that is the escape hatch,
and it has to name its reason.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

BACKEND_SRC = Path(__file__).resolve().parents[2] / "backend" / "src"

#: A construction that derives one frame from another. Assignment-shaped on purpose: the reader's own `Frame(...)` is
#: included, and it names the picture six lines later (`frame.pts = self._take_pts()`), which the scan accepts.
CONSTRUCTION = re.compile(r"=\s*Frame\(")

#: How a derived frame proves it kept the identity: a copy, the seam, or a reason for synthesising a new one.
CARRIES = ("carry_identity_from(", ".pts =", "# identity:")

WINDOW = 12


def _frame(pts: float | None = None):
    from src.utils import Frame as FrameModule

    frame = FrameModule.Frame(
        backend="ncnn", width=4, height=4, device=None, gpu_id=0,
        hdr_mode=False, dtype=None, high_bitdepth=False,
    )
    frame.pts = pts
    return frame


def _construction_sites():
    sites = []
    for path in sorted(BACKEND_SRC.rglob("*.py")):
        if "spandrel" in path.parts or path.name == "Frame.py":
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for index, line in enumerate(lines):
            if not CONSTRUCTION.search(line):
                continue
            if line.lstrip().startswith("#"):
                continue
            window = "\n".join(lines[index + 1:index + 1 + WINDOW])
            sites.append((path, index + 1, any(token in window for token in CARRIES)))
    return sites


def test_the_scan_finds_the_construction_sites():
    """A scan that finds nothing would pass vacuously - pin that it sees the real derivations."""
    sites = _construction_sites()
    assert len(sites) >= 8, [(str(p), n) for p, n, _ in sites]
    assert any(p.name == "UpscaleNCNN.py" for p, _, _ in sites)
    assert any(p.name == "UpscaleTorch.py" for p, _, _ in sites)


def test_every_derived_frame_carries_its_source_time():
    missing = [f"{path.relative_to(BACKEND_SRC.parents[1])}:{line}"
               for path, line, ok in _construction_sites() if not ok]
    assert not missing, (
        "these frames are built from another frame but do not carry its pts "
        f"(use Frame.carry_identity_from, or say why with a `# identity:` comment): {missing}"
    )


def test_clone_keeps_the_source_time():
    original = _frame(1.5)
    assert original.clone().pts == 1.5


def test_clone_of_an_unnamed_frame_stays_unnamed():
    assert _frame(None).clone().pts is None


def test_carry_identity_from_copies_the_source_time():
    source = _frame(2.25)
    derived = _frame(None)
    assert derived.carry_identity_from(source) is derived
    assert derived.pts == pytest.approx(2.25)


def test_carry_identity_from_refuses_a_non_frame():
    with pytest.raises(TypeError):
        _frame(None).carry_identity_from(object())
