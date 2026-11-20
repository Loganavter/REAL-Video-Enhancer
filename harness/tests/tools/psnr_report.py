#!/usr/bin/env python3
"""PSNR between two videos, with the colour format stated and the worst frames named.

Why this exists: every quality number in ``harness/docs/`` was produced by a hand-typed
``ffmpeg ... -lavfi psnr`` piped through a one-liner that parsed ``psnr_avg``, and a second
one-liner that sorted the frames to find the worst. Two ways that goes wrong silently:

* the two inputs are converted to *different* pixel formats (or one of them not at all), so
  the comparison crosses a colour conversion nobody wrote down - the failure mode behind
  C3 in ``harness/docs/RESEARCH-LEDGER.md``, where a missing matrix cost 5 dB and nobody saw it;
* the two inputs do not have the same number of frames. ``psnr`` pairs frames in decode
  order, stops at the shorter one and prints an average over whatever it paired, which is
  not a number about either file. Here that is an error and an exit code, never a warning.

Usage::

    python3 harness/tests/tools/psnr_report.py REFERENCE.mkv CANDIDATE.mkv [options]

Exit codes: ``0`` the comparison ran and the mean is at or above ``--fail-under`` (or no
threshold was given); ``1`` the comparison ran and the mean is below it; ``2`` the
comparison would not be meaningful and was refused - different frame counts, an unreadable
or garbled stats file, or ffmpeg/ffprobe missing or failing.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

# ``yuv420p10le`` first and default: that is what the 10-bit pipeline emits
# (``TorchUtils.tensor_to_yuv420p10`` packs 10-bit limited-range BT.709 samples), so
# comparing a render against its source *without* saying which format each side was
# converted to is exactly the mistake this tool exists to prevent.
SUPPORTED_FORMATS = ("yuv420p10le", "yuv420p", "rgb24")
DEFAULT_FORMAT = "yuv420p10le"
DEFAULT_WORST = 3

# FFmpeg writes one line per paired frame into ``stats_file``:
#   n:1 mse_avg:0.11 psnr_y:51.70 psnr_u:55.80 psnr_v:56.37 psnr_avg:52.73
_FIELD_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*):(\S+)")


class PsnrError(RuntimeError):
    """The comparison could not be made, or would not mean anything."""


class StatsParseError(PsnrError):
    """A line of the ffmpeg stats file is not a per-frame PSNR record."""


class FrameCountMismatch(PsnrError):
    """The two inputs do not hold the same number of frames."""


@dataclass(frozen=True)
class FrameStats:
    """One frame's record from the ffmpeg ``psnr`` stats file."""

    index: int                                  # ``n:`` in the stats file, 1-based
    psnr_avg: float
    psnr_y: float | None = None
    psnr_u: float | None = None
    psnr_v: float | None = None
    mse_avg: float | None = None


@dataclass(frozen=True)
class Summary:
    """Distribution of the per-frame ``psnr_avg`` values."""

    frames: int
    mean: float
    median: float
    worst: float
    worst_frames: tuple[FrameStats, ...]
    identical: int = 0            # frames byte identical to the reference (per-frame dB inf)
    mean_over_mse: float = float("nan")  # PSNR of the mean MSE: ffmpeg's own "average" line


@dataclass(frozen=True)
class Report:
    """Everything a reader needs to reproduce the number."""

    reference: str
    candidate: str
    pixel_format: str
    summary: Summary
    command: tuple[str, ...]
    stats_path: str | None = None
    keep_stats: bool = False

    @property
    def command_line(self) -> str:
        return shlex.join(self.command)


# ------------------------------------------------------------------------------------
# stats file parsing - a plain function over the text ffmpeg writes
# ------------------------------------------------------------------------------------
def _float_field(fields: dict[str, str], name: str, where: str) -> float | None:
    if name not in fields:
        return None
    try:
        return float(fields[name])
    except ValueError as exc:
        raise StatsParseError(f"{where}: {name}={fields[name]!r} is not a number") from exc


def parse_stats_line(line: str, line_number: int = 0) -> FrameStats:
    """Parse one ``n:.. psnr_avg:..`` record.

    Raises ``StatsParseError`` rather than skipping: a dropped frame changes the mean, and
    an average over the records that happened to parse is not the average of the file.
    """
    where = f"stats line {line_number}" if line_number else "stats line"
    fields = dict(_FIELD_RE.findall(line))
    if "n" not in fields:
        raise StatsParseError(f"{where}: no frame index (n:) in {line.strip()!r}")
    if "psnr_avg" not in fields:
        raise StatsParseError(f"{where}: no psnr_avg in {line.strip()!r}")
    try:
        index = int(fields["n"])
    except ValueError as exc:
        raise StatsParseError(
            f"{where}: frame index {fields['n']!r} is not an integer") from exc

    average = _float_field(fields, "psnr_avg", where)
    if average is None:                       # unreachable: the key was checked above
        raise StatsParseError(f"{where}: no psnr_avg in {line.strip()!r}")
    stats = FrameStats(
        index=index,
        psnr_avg=average,
        psnr_y=_float_field(fields, "psnr_y", where),
        psnr_u=_float_field(fields, "psnr_u", where),
        psnr_v=_float_field(fields, "psnr_v", where),
        mse_avg=_float_field(fields, "mse_avg", where),
    )
    # A NaN would slip through every "is the mean below the threshold" comparison as
    # False and report a pass on a number that does not exist.
    if stats.psnr_avg != stats.psnr_avg:
        raise StatsParseError(f"{where}: psnr_avg is NaN")
    return stats


def parse_stats_file(text: str) -> list[FrameStats]:
    """Parse a whole stats file. Empty text yields an empty list (no frames compared)."""
    frames: list[FrameStats] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        frames.append(parse_stats_line(line, number))
    return frames


def _peak_squared(records: Sequence[FrameStats]) -> float | None:
    """Reconstruct the code value range from any usable frame: peak^2 = mse * 10^(psnr/10).

    Needed because the report has no business assuming 8 or 10 bits: `psnr_avg` carries the
    format's peak already, so a finite frame gives it back exactly.
    """
    for record in records:
        if record.mse_avg and record.mse_avg > 0 and math.isfinite(record.psnr_avg):
            return record.mse_avg * 10.0 ** (record.psnr_avg / 10.0)
    return None


def summarize(frames: Iterable[FrameStats], worst_n: int = DEFAULT_WORST) -> Summary:
    """Mean, median, worst and the ``worst_n`` worst frames (ties: lowest index first).

    Two aggregates, because they answer different questions and one of them is the number in
    ``harness/docs/``:

    * ``mean`` averages the per-frame dB over the *finite* ones, so one bad frame moves it.
      A frame byte identical to the reference scores ``inf``: it is counted in ``identical``
      instead of being averaged in, which is why the naive average of a gated render reads
      ``inf`` (measured: 502 frames, 73% reused, 18 of them identical).
    * ``mean_over_mse`` is the PSNR of the mean MSE. That is what ffmpeg's own ``average:``
      line reports and the figure quoted in ``harness/docs/MODEL.md``; on the same pair it read
      52.731 against 52.810 for the per-frame average, so they are close but not the same.
    """
    records = list(frames)
    if not records:
        raise PsnrError("no per-frame statistics: ffmpeg compared nothing")
    finite = [record for record in records if math.isfinite(record.psnr_avg)]
    identical = len(records) - len(finite)
    values = [record.psnr_avg for record in finite]
    if not values:
        mean = median = worst = float("inf")
        ordered: list[FrameStats] = []
        mean_over_mse = float("inf")
    else:
        # Sorting by (value, index) makes the tie-break deterministic instead of depending on
        # the order ffmpeg happened to write equal values in.
        ordered = sorted(finite, key=lambda record: (record.psnr_avg, record.index))
        mean = statistics.fmean(values)
        median = statistics.median(values)
        worst = min(values)
        peak_squared = _peak_squared(records)
        # Over ALL frames, identical ones included: their mse is 0 and it belongs in the mean,
        # which is exactly how ffmpeg computes its "average" line. Averaging only the finite
        # frames would quietly inflate the number the docs quote.
        mses = [record.mse_avg for record in records]
        if peak_squared is not None and all(mse is not None for mse in mses):
            mean_mse = statistics.fmean(float(mse) for mse in mses)
            mean_over_mse = (10.0 * math.log10(peak_squared / mean_mse)
                             if mean_mse > 0 else float("inf"))
        else:
            mean_over_mse = mean
    return Summary(
        frames=len(records),
        mean=mean,
        median=median,
        worst=worst,
        worst_frames=tuple(ordered[:max(0, worst_n)]),
        identical=identical,
        mean_over_mse=mean_over_mse,
    )


# ------------------------------------------------------------------------------------
# frame counts - the guard against an average over a mismatched pair
# ------------------------------------------------------------------------------------
FrameCounter = Callable[[str], int]
"""``path -> number of decoded video frames``; injected so the logic is testable."""


def ffprobe_frame_count(ffprobe: str, path: str) -> int:
    """Decoded frame count of the first video stream, via ``ffprobe -count_frames``.

    Decode rather than container metadata (``nb_frames``): mkv usually does not carry it,
    and this is the count the ``psnr`` filter will actually pair.
    """
    command = [ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
               "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path]
    try:
        proc = subprocess.run(command, capture_output=True, text=True)
    except OSError as exc:
        raise PsnrError(f"cannot run {ffprobe!r}: {exc}") from exc
    if proc.returncode != 0:
        raise PsnrError(f"ffprobe failed on {path!r}: {proc.stderr.strip()}")
    raw = proc.stdout.strip().splitlines()
    try:
        return int(raw[0].strip().rstrip(","))
    except (IndexError, ValueError) as exc:
        raise PsnrError(
            f"ffprobe reported no frame count for {path!r}: {proc.stdout!r}") from exc


def check_frame_counts(reference_count: int, candidate_count: int,
                       reference: str = "reference", candidate: str = "candidate") -> int:
    """Return the common frame count, or raise if the two inputs disagree."""
    if reference_count != candidate_count:
        raise FrameCountMismatch(
            f"frame count mismatch: {reference} has {reference_count} frames, "
            f"{candidate} has {candidate_count}. ffmpeg's psnr filter would pair frames in "
            f"decode order and average over the shorter file, which is not a number about "
            f"either of them - trim both to the same range and re-run")
    if reference_count <= 0:
        raise PsnrError(f"no video frames in {reference} / {candidate}")
    return reference_count


# ------------------------------------------------------------------------------------
# the ffmpeg invocation
# ------------------------------------------------------------------------------------
def _quote_filter_value(value: str) -> str:
    """Quote a filtergraph option value so a path cannot end the value early."""
    return "'" + value.replace("\\", "\\\\").replace("'", "'\\''") + "'"


def build_command(ffmpeg: str, reference: str, candidate: str, pixel_format: str,
                  stats_file: str) -> list[str]:
    """The exact argv: both inputs converted to ``pixel_format`` before any comparison."""
    if pixel_format not in SUPPORTED_FORMATS:
        raise PsnrError(f"unsupported pixel format {pixel_format!r}; "
                        f"choose one of {', '.join(SUPPORTED_FORMATS)}")
    graph = (f"[0:v]format={pixel_format}[ref];"
             f"[1:v]format={pixel_format}[dist];"
             f"[ref][dist]psnr=stats_file={_quote_filter_value(stats_file)}")
    return [ffmpeg, "-hide_banner", "-nostdin", "-v", "error",
            "-i", reference, "-i", candidate,
            "-lavfi", graph, "-an", "-f", "null", "-"]


def run_psnr(command: Sequence[str]) -> None:
    """Run the ffmpeg command, raising ``PsnrError`` with its stderr if it fails."""
    try:
        proc = subprocess.run(list(command), capture_output=True, text=True)
    except OSError as exc:
        raise PsnrError(f"cannot run {command[0]!r}: {exc}") from exc
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-20:]
        raise PsnrError("ffmpeg failed (exit {}):\n{}".format(
            proc.returncode, "\n".join(tail)))


PsnrRunner = Callable[[Sequence[str]], None]


# ------------------------------------------------------------------------------------
# the comparison
# ------------------------------------------------------------------------------------
def compare(reference: str, candidate: str, *, ffmpeg: str, counter: FrameCounter,
            pixel_format: str = DEFAULT_FORMAT, worst_n: int = DEFAULT_WORST,
            keep_stats: str | None = None,
            runner: PsnrRunner = run_psnr) -> Report:
    """Measure PSNR, refusing anything whose average would not mean what it says.

    ``counter`` and ``runner`` are the two side-effecting edges (frame counting and the
    ffmpeg call), injected so the whole path is exercised without ffmpeg installed.
    """
    count = check_frame_counts(counter(reference), counter(candidate), reference, candidate)

    temporary: Path | None = None
    if keep_stats:
        stats_file = Path(keep_stats)
        stats_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        handle = tempfile.NamedTemporaryFile(prefix="psnr-", suffix=".log", delete=False)
        handle.close()
        temporary = Path(handle.name)
        stats_file = temporary

    try:
        command = build_command(ffmpeg, reference, candidate, pixel_format, str(stats_file))
        runner(command)
        try:
            text = stats_file.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise PsnrError(f"ffmpeg left no stats file at {stats_file}: {exc}") from exc
        frames = parse_stats_file(text)
        if len(frames) != count:
            raise PsnrError(
                f"ffmpeg compared {len(frames)} frame(s) but the inputs hold {count}: the "
                f"stats file is truncated, so the average would be over a partial file")
        indices = [frame.index for frame in frames]
        if indices != list(range(1, count + 1)):
            raise PsnrError(
                f"stats indices are not 1..{count} (first {indices[:5]}, last "
                f"{indices[-3:]}) - frames were dropped or renumbered")
        report = Report(
            reference=reference,
            candidate=candidate,
            pixel_format=pixel_format,
            summary=summarize(frames, worst_n),
            command=tuple(command),
            stats_path=str(stats_file) if keep_stats else None,
            keep_stats=bool(keep_stats),
        )
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return report


# ------------------------------------------------------------------------------------
# output
# ------------------------------------------------------------------------------------
def format_report(report: Report, fail_under: float | None = None) -> str:
    """Human-readable block: counts, statistics, worst frames, the command, the verdict."""
    summary = report.summary
    lines = [
        f"reference : {report.reference}",
        f"candidate : {report.candidate}",
        f"format    : {report.pixel_format} (both inputs converted before comparison)",
        f"frames    : {summary.frames} (both inputs)",
        f"mean      : {summary.mean:.2f} dB (per-frame, over "
        f"{summary.frames - summary.identical} finite frames)",
        f"average   : {summary.mean_over_mse:.2f} dB (PSNR of the mean MSE = ffmpeg's own "
        f"\"average\", the figure harness/docs/ quotes)",
        f"median    : {summary.median:.2f} dB",
    ]
    if summary.identical:
        lines.append(f"identical : {summary.identical} frame(s) byte identical to the "
                     f"reference (per-frame dB infinite, counted here, not averaged)")
    if summary.worst_frames:
        # The worst frame is the first entry of the sorted list by construction; read the
        # index off it rather than searching the values again (ties pick the lowest index).
        lines.append(f"worst     : {summary.worst:.2f} dB "
                     f"(frame {summary.worst_frames[0].index})")
        lines.append(f"worst {len(summary.worst_frames)} of {summary.frames}:")
        for frame in summary.worst_frames:
            lines.append(f"  frame {frame.index:<6d} {frame.psnr_avg:8.2f} dB")
    else:
        lines.append(f"worst     : {summary.worst:.2f} dB")
    if report.keep_stats and report.stats_path:
        lines.append(f"stats     : {report.stats_path}")
    lines.append("command   :")
    lines.append(f"  {report.command_line}")
    if fail_under is not None:
        verdict = "PASS" if summary.mean >= fail_under else "FAIL"
        relation = ">=" if summary.mean >= fail_under else "<"
        lines.append(f"verdict   : {verdict} (mean {summary.mean:.2f} dB {relation} "
                     f"{fail_under:g} dB)")
    return "\n".join(lines)


def build_payload(report: Report, fail_under: float | None = None) -> dict:
    """The ``--json`` payload: the same facts, with full precision."""
    summary = report.summary
    payload = {
        "reference": report.reference,
        "candidate": report.candidate,
        "format": report.pixel_format,
        "frames": summary.frames,
        "identical": summary.identical,
        "mean": summary.mean,
        "mean_over_mse": summary.mean_over_mse,
        "median": summary.median,
        "worst": summary.worst,
        "worst_frames": [{"index": frame.index, "psnr_avg": frame.psnr_avg}
                         for frame in summary.worst_frames],
        "command": list(report.command),
        "command_line": report.command_line,
        "stats_path": report.stats_path,
    }
    if fail_under is not None:
        payload["fail_under"] = fail_under
        payload["passed"] = summary.mean >= fail_under
    return payload


# ------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------
def default_ffmpeg() -> str:
    """``./bin/ffmpeg`` when this repo ships one, else whatever is on PATH."""
    local = Path("bin") / "ffmpeg"
    if local.is_file():
        return str(local)
    found = shutil.which("ffmpeg")
    if not found:
        raise PsnrError("no ffmpeg: ./bin/ffmpeg does not exist and none is on PATH")
    return found


def default_ffprobe(ffmpeg: str) -> str:
    """The ffprobe next to the chosen ffmpeg (a bundled build ships both), else PATH."""
    sibling = Path(ffmpeg).with_name("ffprobe")
    if sibling.is_file():
        return str(sibling)
    found = shutil.which("ffprobe")
    if not found:
        raise PsnrError(f"no ffprobe next to {ffmpeg!r} and none is on PATH")
    return found


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="psnr_report",
        description="Measure PSNR between two videos with a stated pixel format, and name "
                    "the worst frames. Mismatched frame counts are refused.")
    parser.add_argument("reference", help="the file to compare against (the source)")
    parser.add_argument("candidate", help="the file under test (the render)")
    parser.add_argument("--ffmpeg", default=None,
                        help="ffmpeg binary (default: ./bin/ffmpeg if present, else PATH)")
    parser.add_argument("--ffprobe", default=None,
                        help="ffprobe binary (default: next to --ffmpeg, else PATH)")
    parser.add_argument("--format", default=DEFAULT_FORMAT, choices=SUPPORTED_FORMATS,
                        help=f"pixel format both inputs are converted to before comparison "
                             f"(default: {DEFAULT_FORMAT}, the format the 10-bit pipeline "
                             f"emits)")
    parser.add_argument("--fail-under", type=float, default=None, metavar="DB",
                        help="exit 1 when the mean PSNR is below DB")
    parser.add_argument("--worst", type=int, default=DEFAULT_WORST, metavar="N",
                        help=f"how many worst frames to list (default: {DEFAULT_WORST})")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--keep-stats", default=None, metavar="FILE",
                        help="keep the raw ffmpeg stats file at FILE")
    args = parser.parse_args(argv)
    if args.worst < 0:
        parser.error("--worst must be >= 0")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        ffmpeg = args.ffmpeg or default_ffmpeg()
        ffprobe = args.ffprobe or default_ffprobe(ffmpeg)

        def counter(path: str) -> int:
            return ffprobe_frame_count(ffprobe, path)

        report = compare(args.reference, args.candidate, ffmpeg=ffmpeg, counter=counter,
                         pixel_format=args.format, worst_n=args.worst,
                         keep_stats=args.keep_stats)
    except PsnrError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(build_payload(report, args.fail_under), indent=2))
    else:
        print(format_report(report, args.fail_under))
    if args.fail_under is not None and report.summary.mean < args.fail_under:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
