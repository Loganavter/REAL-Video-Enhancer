"""The preview's numbers are derived in ONE place (`src/ui/preview/facts.py`), and this pins each derivation.

The owner's 2026-11-26 screenshot is the case: an export at frame 67 of 493 with a fully blue strip and "finished"
above it, because the pane reported the whole clip as produced for a `DISPLAY_FEED` while the render stood at 14 %.
The class behind it is "the same number, two definitions"; the fix is not a branch on `rendering` somewhere else but
ONE function per number, called by the page.

What is pinned here, with no window and no render:

* the clip length: the queue item's own length when it is known, else the opened file's, else frames / fps;
* the produced band: a finished file is fully produced, a live render is its progress - and 67/493 of a 20 s clip is
  NOT the whole clip, whether the pane follows the display feed (export) or a live tail;
* the reach: a finished file, a feed and a preview with a usable block reach the whole clip; a tail reaches the
  produced edge; an unknown length reaches nothing;
* the frame readout: `f <done> / <total>` with `done` clamped into the probe's packet set, and a dash without one;
* the module itself: `facts` imports on an interpreter where PySide6 CANNOT be imported (a subprocess with a
  meta-path finder that refuses it), which is the honest form of "no Qt import" - asserted by really importing it
  there, not by grepping the source.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = str(Path(__file__).resolve().parents[2])
# BOTH the app and the backend live in a `src` directory (namespace packages). The suite's conftest puts the backend
# first, which would make `src.ui` the backend's - `facts` lives in the APP's tree. The house pattern, kept identical
# to `test_preview_state_tick.py` so module order cannot change the answer.
_ORIGINAL_SYS_PATH = list(sys.path)
if _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)

from src.ui.preview.facts import (  # noqa: E402
    clip_duration,
    frame_ratio,
    produced_seconds,
    reach_seconds,
)
from src.ui.preview.state import PreviewBackendKind, PreviewState  # noqa: E402

sys.path[:] = _ORIGINAL_SYS_PATH

_PROBE = SimpleNamespace(videoFrameCount=493, videoFps=24.0)
_CLIP = 493 / 24.0                      # 20.5416... s, the owner's clip


# -- duration ------------------------------------------------------------------------------------


def test_the_opened_files_length_wins_over_the_probe():
    """A decoder that knows the real length answers it: the probe's frames/fps is only the fallback."""
    assert clip_duration(_PROBE, SimpleNamespace(duration=20.0)) == pytest.approx(20.0)


def test_the_length_falls_back_to_the_probes_frames_over_fps():
    assert clip_duration(_PROBE) == pytest.approx(_CLIP)
    assert clip_duration(_PROBE, SimpleNamespace(duration=0.0)) == pytest.approx(_CLIP)


def test_a_queue_item_is_measured_by_its_own_length():
    """The queue pane's length, when known, is the axis its bar and thumb are a share of."""
    assert clip_duration(_PROBE, SimpleNamespace(duration=20.0), queue_duration=12.5) == pytest.approx(12.5)


def test_a_length_nobody_knows_is_zero():
    assert clip_duration(None) == 0.0
    assert clip_duration(SimpleNamespace(videoFrameCount=0, videoFps=24.0)) == 0.0
    assert clip_duration(SimpleNamespace(videoFrameCount=493, videoFps=0.0)) == 0.0


# -- produced ------------------------------------------------------------------------------------


def test_a_finished_file_is_fully_produced():
    assert produced_seconds(PreviewBackendKind.FINISHED_FILE, fraction=0.0, duration=20.0,
                            rendering=False) == pytest.approx(20.0)


def test_a_live_export_band_is_its_progress_not_the_whole_clip():
    """The regression pin: frame 67 of 493 is 14 % blue, not a full bar with "finished" above it."""
    duration = 20.0
    band = produced_seconds(PreviewBackendKind.DISPLAY_FEED, fraction=67 / 493, duration=duration,
                            rendering=True)
    assert band == pytest.approx(duration * 67 / 493)
    assert band == pytest.approx(2.718, abs=1e-3)
    assert band < duration - 1.0, "a live export's band must never read as the whole clip"
    state = PreviewState(backend=PreviewBackendKind.DISPLAY_FEED, duration=duration, produced=band)
    assert state.finished() is False


def test_the_same_export_after_the_render_is_fully_produced():
    """The render is over: the feed shows the finished file and nothing is left to produce."""
    assert produced_seconds(PreviewBackendKind.DISPLAY_FEED, fraction=67 / 493, duration=20.0,
                            rendering=False) == pytest.approx(20.0)


def test_a_preview_tails_band_is_its_progress():
    assert produced_seconds(PreviewBackendKind.LIVE_TAIL, fraction=0.25, duration=20.0,
                            rendering=True) == pytest.approx(5.0)


def test_the_band_is_clamped_into_the_clip():
    assert produced_seconds(PreviewBackendKind.LIVE_TAIL, fraction=1.5, duration=20.0,
                            rendering=True) == pytest.approx(20.0)
    assert produced_seconds(PreviewBackendKind.LIVE_TAIL, fraction=-0.25, duration=20.0,
                            rendering=True) == pytest.approx(0.0)
    assert produced_seconds(PreviewBackendKind.LIVE_TAIL, fraction=0.5, duration=0.0,
                            rendering=True) == pytest.approx(0.0)


# -- reach ---------------------------------------------------------------------------------------


def test_a_finished_file_reaches_the_whole_clip():
    assert reach_seconds(PreviewBackendKind.FINISHED_FILE, duration=20.0, produced=2.7,
                         any_frame=False) == pytest.approx(20.0)


def test_a_feed_reaches_the_whole_clip():
    assert reach_seconds(PreviewBackendKind.DISPLAY_FEED, duration=20.0, produced=2.7,
                         any_frame=False) == pytest.approx(20.0)


def test_a_tail_reaches_the_produced_edge():
    """A growing file's tail only has what the renderer has written - the band is its reach, not the clip."""
    assert reach_seconds(PreviewBackendKind.LIVE_TAIL, duration=20.0, produced=2.7,
                         any_frame=False) == pytest.approx(2.7)
    assert reach_seconds(PreviewBackendKind.NONE, duration=20.0, produced=0.0,
                         any_frame=False) == pytest.approx(0.0)


def test_a_preview_with_a_usable_block_reaches_the_whole_clip():
    """`any_frame`: the host renders ANY timestamp on demand, so a press past the band is answerable."""
    assert reach_seconds(PreviewBackendKind.LIVE_TAIL, duration=20.0, produced=2.7,
                         any_frame=True) == pytest.approx(20.0)


def test_no_length_means_nothing_to_reach():
    assert reach_seconds(PreviewBackendKind.FINISHED_FILE, duration=0.0, produced=0.0,
                         any_frame=True) == pytest.approx(0.0)


def test_the_band_and_the_reach_are_two_quantities():
    """The two numbers cannot be collapsed: at 67/493 of a live export the band is 2.7 s, the reach is the clip."""
    band = produced_seconds(PreviewBackendKind.DISPLAY_FEED, fraction=67 / 493, duration=20.0,
                            rendering=True)
    reach = reach_seconds(PreviewBackendKind.DISPLAY_FEED, duration=20.0, produced=band, any_frame=True)
    assert band == pytest.approx(2.718, abs=1e-3)
    assert reach == pytest.approx(20.0)


# -- frame readout -------------------------------------------------------------------------------


def test_the_frame_readout_is_the_source_packet_set():
    assert frame_ratio(493, 250) == "f 250 / 493"


def test_the_frame_readout_clamps_into_the_probe():
    """A counter sampled one packet past the probe's total must not read as a second truth."""
    assert frame_ratio(493, 1000) == "f 493 / 493"
    assert frame_ratio(493, -5) == "f 0 / 493"


def test_an_unknown_frame_total_is_a_dash():
    assert frame_ratio(0, 10) == "f \u2014"


def test_the_readout_matches_the_states_own_decision():
    """The state still prints its own line (P0 may not touch `state.py`); the two must not drift."""
    for total, done in ((493, 250), (493, 1000), (493, -5), (0, 10), (0, 0), (495, 495)):
        state = PreviewState(source_frames=total, frames_done=done)
        assert state.frame_ratio() == frame_ratio(total, done), (total, done)


# -- the module itself ---------------------------------------------------------------------------

_NO_QT_SCRIPT = """
import sys

class _NoPySide6:
    \"\"\"Refuse the module at find time: an already-imported PySide6 would prove nothing.\"\"\"

    def find_spec(self, name, path=None, target=None):
        if name == "PySide6" or name.startswith("PySide6."):
            raise ImportError("PySide6 is blocked for this proof")
        return None

sys.meta_path.insert(0, _NoPySide6())
import src.ui.preview.facts as facts
print(facts.__file__)
print(facts.produced_seconds.__name__, facts.reach_seconds.__name__)
"""


def test_the_facts_module_imports_without_qt():
    """Prove it by importing it where PySide6 CANNOT be imported - not by grepping for `import PySide6`."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [_ROOT, os.path.join(_ROOT, "src"), env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run([sys.executable, "-c", _NO_QT_SCRIPT], cwd=_ROOT, env=env,
                               capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines()[0].endswith(os.path.join("src", "ui", "preview", "facts.py"))
    assert completed.stdout.splitlines()[1] == "produced_seconds reach_seconds"
