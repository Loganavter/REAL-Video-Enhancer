import os

from ..Util import openLink, networkCheck
from ..constants import TEMP_DOWNLOAD_PATH
import requests

RELEASES_URL = "https://api.github.com/repos/tntwise/real-video-enhancer/releases"
CHANGELOG_CACHE = os.path.join(TEMP_DOWNLOAD_PATH, "changelog_cache.md")


class HomeTab:
    def __init__(self, parent):
        self.parent = parent
        self.QConnect()

    def getChangelog(self):
        """Release notes for the home panel: live, then cached, then a visible reason.

        The anonymous GitHub API allows 60 requests an hour per address. When that is
        exhausted the endpoint answers 403 with a dict, `release["tag_name"]` raises, and
        the previous code swallowed the exception into an empty panel - which reads exactly
        like a broken window (measured 2026-11-20: `API rate limit exceeded ... 83.94.67.33`,
        remaining 0 of 60). A cached copy survives the limit, and a message beats a blank box.
        """
        cached = ""
        try:
            with open(CHANGELOG_CACHE, encoding="utf-8") as fh:
                cached = fh.read()
        except OSError:
            pass
        try:
            response = requests.get(RELEASES_URL, timeout=10)
            releases = response.json()
            if not isinstance(releases, list) or not releases:
                reason = releases.get("message") if isinstance(releases, dict) else "unexpected response"
                raise RuntimeError(reason)
            changeLog = ""
            for release in releases[:5]:
                changeLog += "\n# " + release.get("tag_name", "")
                changeLog += "\n" + (release.get("body") or "").replace(r"\r\n", "")
            try:
                os.makedirs(os.path.dirname(CHANGELOG_CACHE), exist_ok=True)
                with open(CHANGELOG_CACHE, "w", encoding="utf-8") as fh:
                    fh.write(changeLog)
            except OSError:
                pass
            return changeLog
        except Exception as exc:
            if cached:
                return cached + "\n\n---\n*Release notes from cache; the live fetch failed:* " + str(exc)
            return ("# Changelog unavailable\n\n" + str(exc) + "\n\n"
                    "*The releases page has the same notes:* "
                    "[github.com/tntwise/REAL-Video-Enhancer/releases]"
                    "(https://github.com/tntwise/REAL-Video-Enhancer/releases)")

    def QConnect(self):
        self.parent.githubBtn.clicked.connect(
            lambda: openLink("https://github.com/tntwise/REAL-Video-Enhancer")
        )
        self.parent.kofiBtn.clicked.connect(
            lambda: openLink("https://ko-fi.com/tntwise")
        )
        # The panel is filled either way: `networkCheck` only decides whether the live
        # fetch is attempted, and a failure now says why instead of leaving a blank box.
        self.parent.changeLogText.setVisible(True)
        self.parent.changeLogText.setMarkdown(self.getChangelog())
