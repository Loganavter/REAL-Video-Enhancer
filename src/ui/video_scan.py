"""Reading a folder of videos without freezing the window - the Qt-free half.

Picking a folder used to run, per file, a whole `rve-backend.py --print_video_info` process (a Python interpreter, cv2
and an ffmpeg probe) **on the GUI thread** - measured at 0.16 s per file here, so a 200-file folder is half a minute of
a frozen window with no progress and no way to cancel. It also probed every entry in the folder, images and subtitles
included, since the validity check ran after the expensive part.

Two changes, both measured (20 copies of a 2 s clip): one batch process reads the whole folder in 0.53 s against 3.2 s
for the same 20 files one process at a time, and the work runs on a thread, so the window keeps painting, the wheel
keeps scrolling, and each record arrives as progress.

This half has no Qt at all, so `harness/tests/test_video_scan.py` can drive it on the portable interpreter; the
`QThread` wrapper lives in `batch_probe.py`.
"""

from __future__ import annotations

import json
import os
import subprocess

#: Where the backend lives is the APP's business, and it is passed in: the harness suite puts `backend/` first on
#: sys.path, so the two `src` trees merge into one namespace package and an import of `src.constants` from here resolves
#: to the backend's module (measured: ImportError on BACKEND_PATH). Keeping this file import-free makes it testable.

#: What the file dialog offers; the batch folder walk uses the same set so a folder cannot bring in what "Open file"
#: would have refused.
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".wmv", ".flv", ".ts", ".m2ts", ".mpg", ".mpeg")


def video_files(folder: str) -> list[str]:
    """Video-looking files in `folder`, sorted - the cheap filter that runs before any probe."""
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return []
    return [
        os.path.join(folder, name)
        for name in names
        if name.lower().endswith(VIDEO_EXTENSIONS) and os.path.isfile(os.path.join(folder, name))
    ]


def probe_many(paths: list[str], *, python: str, backend: str, ffmpeg: str, on_record=None) -> list[dict]:
    """One backend process for the whole list; `on_record` sees each record as it arrives."""
    if not paths:
        return []
    command = [
        python,
        "-W",
        "ignore",
        os.path.join(backend, "rve-backend.py"),
        # The option goes BEFORE the list: `--print_video_info_batch` takes `nargs="+"`, so a trailing option would be
        # one more path for anything that parses the argv by hand.
        "--ffmpeg_path",
        f"{ffmpeg}",
        "--print_video_info_batch",
        *paths,
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    records: list[dict] = []
    try:
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            records.append(record)
            if on_record is not None:
                on_record(record)
    finally:
        try:
            process.stdout.close()
        except Exception:
            pass
        process.wait()
    return records
