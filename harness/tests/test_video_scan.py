"""The folder scan behind "open a folder of videos": what it accepts, and what one backend call returns.

Picking a folder used to probe every entry with its own interpreter on the GUI thread - images and subtitle files
included, since the validity check came after the expensive part. The scan now filters by extension first and reads the
whole folder in ONE backend process; these tests pin both halves of that contract. Measured on 20 copies of a 2 s clip:
0.53 s in one process against 3.2 s for the same files one process each.
"""

from __future__ import annotations

import os
import sys

import pytest

from src.ui import video_scan


def _write(path, text: str = "x") -> str:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return str(path)


def test_only_video_extensions_are_listed_and_in_a_stable_order(tmp_path):
    for name in ("b.MKV", "a.mp4", "notes.txt", "cover.png", "c.webm", "clip.Mov"):
        _write(tmp_path / name)
    (tmp_path / "subfolder.mp4").mkdir()

    found = video_scan.video_files(str(tmp_path))

    assert [os.path.basename(path) for path in found] == ["a.mp4", "b.MKV", "c.webm", "clip.Mov"]
    assert all(os.path.isfile(path) for path in found), "a directory named like a video is not a video"


def test_a_missing_folder_is_empty_not_an_exception():
    assert video_scan.video_files("/definitely/not/a/folder") == []


def test_probe_many_reads_one_json_record_per_line_from_a_single_process(tmp_path, monkeypatch):
    """The contract with the backend: one line per file, and the process is started ONCE for the whole list."""
    stub = _write(
        tmp_path / "rve-backend.py",
        "import json, sys\n"
        "paths = sys.argv[sys.argv.index('--print_video_info_batch') + 1:]\n"
        "for path in paths:\n"
        "    print('BACKEND: noise the parser must skip', flush=True)\n"
        "    print(json.dumps({'path': path, 'valid': True, 'width': 1920}), flush=True)\n",
    )
    launches = []
    real_popen = video_scan.subprocess.Popen

    def counting_popen(command, *args, **kwargs):
        launches.append(list(command))
        return real_popen(command, *args, **kwargs)

    monkeypatch.setattr(video_scan.subprocess, "Popen", counting_popen)

    seen = []
    records = video_scan.probe_many(
        ["one.mp4", "two.mkv"], python=sys.executable, backend=str(tmp_path), ffmpeg="ffmpeg",
        on_record=seen.append,
    )

    assert len(launches) == 1, "one process for the whole folder - that is the point of the batch call"
    assert [record["path"] for record in records] == ["one.mp4", "two.mkv"]
    assert len(seen) == 2, "the caller sees records as they arrive, so the page can show progress"
    assert all(record["width"] == 1920 for record in records)


def test_probe_many_of_nothing_starts_nothing(monkeypatch):
    def explode(*_args, **_kwargs):
        raise AssertionError("no files must not launch a backend")

    monkeypatch.setattr(video_scan.subprocess, "Popen", explode)
    assert video_scan.probe_many([], python=sys.executable, backend="/nowhere", ffmpeg="ffmpeg") == []
