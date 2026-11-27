"""The picture's own name: which SOURCE moment each output frame carries.

Why this file exists (2026-11-27). The export is CFR: where the source has a gap, `RenderVideo` holds the previous
picture for the missing slots, so a 493-frame source with a three-frame hole near its end writes 495 output frames. The
two extra pictures are DUPLICATES of a source frame, so they have an exact partner - but the tail named every output
frame on the muxer's own 24 fps grid (`k/24`), while the compare's source half is named by the source's PTS. At the gap
the two names drifted two frames apart and the pair was refused, which read as "no source frame for the last frame".

The renderer knows the answer: `Frame.pts` is the source presentation time of every picture it writes, duplicates
included. It writes that list next to the output (`<output>.srcpts`, one float per line, in OUTPUT order), and the tail
and the finished-file decoder name frames from it when it is there. Nothing else changes: a name is still a name, the
pair gate is still the pane's half-frame rule - only the number it compares is the picture's OWN source time.

Format: plain text, one `%.6f` per line, no header, no index - the line NUMBER is the output frame ordinal (0-based).
A missing or short file is not an error: the readers fall back to the container's PTS, which is what they did before.
"""

from __future__ import annotations

__all__ = ["SIDECAR_SUFFIX", "read_names", "write_names", "names_path"]

#: Appended to the output file's name: `clip.mkv` -> `clip.mkv.srcpts`.
SIDECAR_SUFFIX = ".srcpts"


def names_path(output_file: str) -> str:
    """Where the source-time list for `output_file` lives."""
    return f"{output_file}{SIDECAR_SUFFIX}"


def write_names(output_file: str, names) -> str | None:
    """Write one source time per output frame; returns the path, or None when nothing was written."""
    values = [float(v) for v in names if v is not None]
    if not values:
        return None
    path = names_path(output_file)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            for value in values:
                handle.write(f"{value:.6f}\n")
    except OSError:
        return None
    return path


def read_names(output_file: str) -> list[float] | None:
    """The source times for `output_file`, or None when there is no usable list.

    A short list is returned as-is: the caller uses a name only for the frames it has, and falls back to the
    container's own PTS past the end (a render still writing has fewer lines than the file has frames).
    """
    try:
        with open(names_path(output_file), encoding="utf-8") as handle:
            values = []
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    values.append(float(line))
                except ValueError:
                    return None
            return values or None
    except OSError:
        return None
