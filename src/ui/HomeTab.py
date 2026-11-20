import os

import requests
from PySide6 import QtCore

from ..Util import openLink
from ..constants import CWD, TEMP_DOWNLOAD_PATH

RELEASES_URL = "https://api.github.com/repos/tntwise/real-video-enhancer/releases"
CHANGELOG_CACHE = os.path.join(TEMP_DOWNLOAD_PATH, "changelog_cache.md")
SHIPPED_CHANGELOG = os.path.join(CWD, "CHANGELOG.md")


def read_notes(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""


def fetch_release_notes(timeout: float = 10.0) -> str:
    """The five latest releases from the anonymous API, or "" when they cannot be had.

    One request, no retries, never raises: the panel always has the local changelog to
    show, so a rate limit, an offline start or a slow link must not reach the window.
    """
    try:
        response = requests.get(RELEASES_URL, timeout=timeout)
        releases = response.json()
        if not isinstance(releases, list) or not releases:
            return ""
        notes = ""
        for release in releases[:5]:
            notes += "\n# " + release.get("tag_name", "")
            # a real CRLF, not the two-character escape the old line removed
            notes += "\n" + (release.get("body") or "").replace("\r\n", "\n")
        return notes
    except Exception:
        return ""


class NotesFetcher(QtCore.QThread):
    """The live fetch, off the UI thread: the window opens on the shipped notes."""

    fetched = QtCore.Signal(str)

    def run(self):
        notes = fetch_release_notes()
        if notes:
            self.fetched.emit(notes)


class HomeTab:
    def __init__(self, parent):
        self.parent = parent
        self._fetcher = None
        self.QConnect()

    def getChangelog(self) -> str:
        """What the panel can show without the network: the shipped changelog, then cache.

        The window used to fetch this from the GitHub API on every start, twice (a
        `networkCheck` HEAD plus the fetch), on the UI thread and with no cache: the panel
        was blank whenever the anonymous quota (60/h per address) was spent, and a slow
        network froze the window before it appeared. The repository ships the same notes in
        `CHANGELOG.md`, so the network is an optional update, never the source.
        """
        return read_notes(SHIPPED_CHANGELOG) or read_notes(CHANGELOG_CACHE)

    def applyFetchedNotes(self, notes: str) -> None:
        try:
            os.makedirs(os.path.dirname(CHANGELOG_CACHE), exist_ok=True)
            with open(CHANGELOG_CACHE, "w", encoding="utf-8") as fh:
                fh.write(notes)
        except OSError:
            pass
        self.parent.changeLogText.setMarkdown(notes)

    def QConnect(self):
        self.parent.githubBtn.clicked.connect(
            lambda: openLink("https://github.com/tntwise/REAL-Video-Enhancer")
        )
        self.parent.kofiBtn.clicked.connect(
            lambda: openLink("https://ko-fi.com/tntwise")
        )
        self.parent.changeLogText.setVisible(True)
        self.parent.changeLogText.setMarkdown(self.getChangelog() or "_No release notes bundled._")
        self._fetcher = NotesFetcher(self.parent)
        self._fetcher.fetched.connect(self.applyFetchedNotes)
        self._fetcher.start()
