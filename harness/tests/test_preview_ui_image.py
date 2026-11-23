"""The process page's 100 ms clock, and the changelog panel.

The clock used to carry a 640x360 thumbnail out of shared memory. That channel is gone (the pane is fed at full
resolution from the preview file), so what this file still pins is the part a render depends on: the panel's log box,
FPS/ETA and status are repainted on a tick that carries nothing - with no picture to show, the panel used to freeze.

It runs the import in a subprocess on purpose: this repository has two `src` packages (the window's `src/` and the
backend's `backend/src/`), and `harness/tests/conftest.py` puts the backend one first. Importing the window's module
here would either pick the wrong `src` or depend on collection order, so the test gives the subprocess the root it
needs and nothing else.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_TICK_CHILD = r"""
import json, os, sys, types
os.environ["QT_QPA_PLATFORM"] = "offscreen"
root = sys.argv[1]
sys.path.insert(0, root)
try:
    from PySide6 import QtCore, QtWidgets
except ImportError:
    print("NOQT"); raise SystemExit(0)
cpuinfo = types.ModuleType("cpuinfo"); cpuinfo.get_cpu_info = lambda: {}; sys.modules["cpuinfo"] = cpuinfo
from src.ui.QTcustom import UpdateGUIThread as T

app = QtWidgets.QApplication([])
clock = T(None)
seen = []
clock.tick.connect(lambda: seen.append(1))
clock.start()

def done():
    clock.stop(); clock.wait(1000)
    print(json.dumps({"ticks": len(seen)}))
    app.quit()

QtCore.QTimer.singleShot(600, done)
app.exec()
"""


def test_the_panel_ticks_with_nothing_to_show():
    """The log box, FPS/ETA and status are painted by the tick. When the picture came from the (now removed)
    thumbnail channel, a render without a viewer never called the slot and the panel froze - "logs do not work
    without preview". The tick carries no payload at all."""
    proc = subprocess.run([sys.executable, "-c", _TICK_CHILD, ROOT],
                          capture_output=True, text=True, timeout=120)
    if "NOQT" in proc.stdout:
        pytest.skip("PySide6 is not installed for this interpreter")
    assert proc.returncode == 0, proc.stderr[-1500:]
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert got["ticks"] >= 3, f"expected ~6 ticks in 600 ms at 10 Hz, got {got['ticks']}"


_HOME_CHILD = r"""
import json, os, sys, types
os.environ["QT_QPA_PLATFORM"] = "offscreen"
root, cache = sys.argv[1], sys.argv[2]
sys.path.insert(0, root)
try:
    from PySide6 import QtWidgets
except ImportError:
    print("NOQT"); raise SystemExit(0)
cpuinfo = types.ModuleType("cpuinfo"); cpuinfo.get_cpu_info = lambda: {}; sys.modules["cpuinfo"] = cpuinfo
app = QtWidgets.QApplication([])
from src.ui import HomeTab as HT
HT.CHANGELOG_CACHE = cache
HT.NotesFetcher.run = lambda self: None          # no network in a test

class P(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        # The window surface after the typed handles: `HomeTab` reads `parent.page_handles`, never `parent.<widget>`.
        # `name_of()` is the page the stack shows - "home" is what `MainWindow` sets at startup.
        github, kofi, log = QtWidgets.QPushButton(self), QtWidgets.QPushButton(self), QtWidgets.QTextBrowser(self)
        self.page_handles = types.SimpleNamespace(
            home=types.SimpleNamespace(githubBtn=github, kofiBtn=kofi, changeLogText=log),
            menu=types.SimpleNamespace(pages=types.SimpleNamespace(name_of=lambda: "home")),
        )

p = P()
log = p.page_handles.home.changeLogText
tab = HT.HomeTab(p)
before = log.toPlainText()
tab.applyFetchedNotes("# 9.9.9\nfrom the api")
print(json.dumps({
    "shipped_chars": len(tab.getChangelog()),
    "panel_before_head": before.splitlines()[:1],
    "panel_after_head": log.toPlainText().splitlines()[:2],
    "cache_written": open(cache).read().splitlines()[:2] if os.path.exists(cache) else None,
}))
"""


def test_the_changelog_panel_opens_on_the_shipped_file():
    """The window used to fetch these notes on every start, twice, on the UI thread and with
    no cache - so a spent anonymous quota (60/h per address) left the panel blank. The repo
    ships the same notes; they must be on screen before any network call is made."""
    import tempfile
    cache = os.path.join(tempfile.mkdtemp(prefix="rve-home-"), "cache.md")
    proc = subprocess.run([sys.executable, "-c", _HOME_CHILD, ROOT, cache],
                          capture_output=True, text=True, timeout=120)
    if "NOQT" in proc.stdout:
        pytest.skip("PySide6 is not installed for this interpreter")
    assert proc.returncode == 0, proc.stderr[-1500:]
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert got["shipped_chars"] > 1000, f"the shipped changelog is {got['shipped_chars']} chars"
    assert got["panel_before_head"] == ["RVE 2.4.1"], got["panel_before_head"]
    assert got["panel_after_head"][0] == "9.9.9", "a fetched update replaces the panel"
    assert got["cache_written"] == ["# 9.9.9", "from the api"], "and it is cached for the next start"
