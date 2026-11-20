"""The GUI's half of the preview contract: header payload -> QImage, for both formats.

The backend is tested where it builds the planes; this file pins what the window does with
them, because that half is where a wrong assumption silently draws garbage:

* an RGB888 payload becomes an 8x6 image with the exact pixels (no channel swap);
* a sampled yuv420 payload is decoded with the BT.709 limited matrix - 128/128/128 must come
  out neutral grey, not black and not a shifted green.

It runs the import in a subprocess on purpose: this repository has two `src` packages (the
window's `src/` and the backend's `backend/src/`), and `harness/tests/conftest.py` puts the
backend one first. Importing the window's module here would either pick the wrong `src` or
depend on collection order, so the test gives the subprocess the root it needs and nothing
else.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CHILD = r"""
import json, os, sys, types
os.environ["QT_QPA_PLATFORM"] = "offscreen"
root = sys.argv[1]
sys.path.insert(0, root)                      # the window's src, not the backend's
try:
    from PySide6 import QtGui
except ImportError:
    print("NOQT"); raise SystemExit(0)
# the window's package imports py-cpuinfo at module level for an unrelated feature; stub it
# rather than install a dependency for a test of one static method
cpuinfo = types.ModuleType("cpuinfo")
cpuinfo.get_cpu_info = lambda: {}
sys.modules["cpuinfo"] = cpuinfo
app = QtGui.QGuiApplication([])
from src.ui.QTcustom import UpdateGUIThread as T

w, h = 8, 6
rgb = bytes([10, 20, 30] * (w * h))
img = T._to_image(rgb, w, h, T.PREVIEW_FMT_RGB888)

n = w * h
ch, cw = (w + 1) // 2, (h + 1) // 2
yuv = bytes([128] * n) + bytes([128] * (ch * cw)) * 2
img2 = T._to_image(yuv, w, h, T.PREVIEW_FMT_YUV420)

print(json.dumps({
    "rgb_size": [img.width(), img.height()],
    "rgb_px": list(img.pixelColor(0, 0).getRgb()[:3]),
    "yuv_size": [img2.width(), img2.height()],
    "yuv_px": list(img2.pixelColor(0, 0).getRgb()[:3]),
}))
"""


def _render():
    proc = subprocess.run([sys.executable, "-c", _CHILD, ROOT],
                          capture_output=True, text=True, timeout=120)
    if "NOQT" in proc.stdout:
        pytest.skip("PySide6 is not installed for this interpreter")
    assert proc.returncode == 0, f"the window module failed to load:\n{proc.stderr[-1500:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_an_rgb_payload_keeps_its_pixels_and_geometry():
    got = _render()
    assert got["rgb_size"] == [8, 6]
    assert got["rgb_px"] == [10, 20, 30], "no channel swap, no scaling on the GUI side"


def test_a_neutral_yuv_payload_decodes_to_neutral_grey():
    got = _render()
    assert got["yuv_size"] == [8, 6]
    r, g, b = got["yuv_px"]
    # BT.709 limited: 1.164*(128-16) = 130.4. Anything near 0 is the reader treating YUV as RGB.
    assert 126 <= r <= 134 and 126 <= g <= 134 and 126 <= b <= 134, got["yuv_px"]
    assert max(r, g, b) - min(r, g, b) <= 2, "neutral in must be neutral out"


_TICK_CHILD = r"""
import json, os, sys, types
os.environ["QT_QPA_PLATFORM"] = "offscreen"
root = sys.argv[1]
sys.path.insert(0, root)
try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:
    print("NOQT"); raise SystemExit(0)
cpuinfo = types.ModuleType("cpuinfo"); cpuinfo.get_cpu_info = lambda: {}; sys.modules["cpuinfo"] = cpuinfo
from src.ui.QTcustom import UpdateGUIThread as T

app = QtWidgets.QApplication([])
reader = T(None, "/never_created_block")   # no block: the no-preview case
seen = []
reader.latestPreviewPixmap.connect(lambda img: seen.append(img.isNull()))
reader.start()

def done():
    reader.stop(); reader.wait(1000)
    print(json.dumps({"ticks": len(seen), "all_null": all(seen)}))
    app.quit()

QtCore.QTimer.singleShot(600, done)
app.exec()
"""


def test_the_reader_ticks_without_a_preview_block():
    """The panel's log box, FPS/ETA and status are painted by the preview slot. With no
    preview target the slot was never called, so a render ran with a frozen panel - the
    "logs do not work without preview" report. The reader now emits a null image per tick."""
    proc = subprocess.run([sys.executable, "-c", _TICK_CHILD, ROOT],
                          capture_output=True, text=True, timeout=120)
    if "NOQT" in proc.stdout:
        pytest.skip("PySide6 is not installed for this interpreter")
    assert proc.returncode == 0, proc.stderr[-1500:]
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert got["ticks"] >= 3, f"expected ~6 ticks in 600 ms at 10 Hz, got {got['ticks']}"
    assert got["all_null"], "a tick carries no picture - the slot must skip the pixmap branch"
