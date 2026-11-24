"""The `input:<path>` verb: a live probe can select the file the file dialog would, without the dialog.

Why it exists (measured 2026-11-24, on a lease): with no input file set, `render:<path>` and `preview:<path>` answer
NOTHING at all - no payload, no error, and no line in `app.log` - while `state:` requests around them are served
normally. A disabled Render Preview button with no file behind it reads exactly like a dead control channel, and it cost
a live-proof attempt before anyone looked at the field.

The verb's contract is the honest one: it answers what the APP ended up with. `MainWindow.loadVideo` refuses a file its
probe cannot read and leaves the input field empty, so `loaded` stays None and `error` says why - a request that was
"accepted" but did nothing is exactly the failure this verb must not repeat.

These tests cover the pure validator and the answer shape; the happy path needs a real video and a window, so it is
proved on a lease and named in the close-out (`rve-readouts-live-proof`).
"""

from __future__ import annotations

import pytest

from src.ui import capture_server


def test_an_empty_path_is_answered_not_ignored():
    error, resolved = capture_server.validate_input_path("   ")
    assert error == "empty path"
    assert resolved is None


def test_a_missing_file_is_answered_with_its_resolved_path():
    error, resolved = capture_server.validate_input_path("/nope/definitely-not-a-clip.mkv")
    assert resolved is None
    assert error and error.startswith("no such file:")
    assert "/nope/definitely-not-a-clip.mkv" in error, "the answer must name the path the probe actually looked at"


def test_an_existing_file_resolves_relative_to_the_process(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mkv"
    clip.write_bytes(b"not really a video - the validator only states existence")
    error, resolved = capture_server.validate_input_path("clip.mkv")
    assert error is not None, "a relative path from another directory must not resolve silently"
    monkeypatch.chdir(tmp_path)
    error, resolved = capture_server.validate_input_path("clip.mkv")
    assert error is None
    assert resolved == str(clip.resolve()), "the verb answers with the absolute path it handed to the app"


def test_a_tilde_path_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "clip.mp4").write_bytes(b"")
    error, resolved = capture_server.validate_input_path("~/clip.mp4")
    assert error is None
    assert resolved == str((tmp_path / "clip.mp4").resolve())
