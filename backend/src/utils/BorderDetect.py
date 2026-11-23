"""Border detection: nine seeked previews and a per-side median vote.

The old shape of this file decoded the whole input through ``cropdetect`` and kept the crop with the largest
area, which costs one full-file pass before a render starts (minutes on a 2 hour source) and is decided by the
single brightest/noisiest frame in the file. This is HandBrake's sampled model (``libhb/scan.c`` master
2026-11-23): K previews at ``(i+1)/(K+1)`` of the duration, one frame each, reject previews that crop a quarter
of the frame away on any side, then take the per-side **median** - and only the per-side minimum when at least
``crop_threshold_frames`` previews disagree by more than ``crop_threshold_pixels`` pixels (letterbox that changes
between scenes). The decision is made in pure functions (:func:`sides_from_crop`, :func:`sample_valid`,
:func:`decide`) so it can be pinned without media, an ffmpeg binary or a GPU; the subprocess work lives in the
thin ``_probe*`` methods.

``RVE_BORDER_SAMPLE=0`` (or ``false``/``no``/``off``) restores the old whole-file pass. It is also the
fallback when the container cannot be probed at all - no readable header, or not one preview returned a
``crop=`` line - and that fallback is logged as such.
"""

from __future__ import annotations

import os
import re
import statistics
import subprocess

from ..utils.Util import log, subprocess_popen_without_terminal

#: One preview per tenth of the duration, positions (i+1)/(K+1): 10%, 20%, ... 90%.
PROBES = 9

#: ``scan.c``: when this many previews report a side as more than ``crop_threshold_pixels`` smaller than the
#: median, the crop is not stable across the file and the per-side minimum is used instead.
DEFAULT_CROP_THRESHOLD_FRAMES = 4
DEFAULT_CROP_THRESHOLD_PIXELS = 9

_CROP_RE = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_RESOLUTION_RE = re.compile(r"Video:.*?(\d{2,5})x(\d{2,5})")


def sampling_enabled(environ=None):
    """Whether the sampled path is on; ``RVE_BORDER_SAMPLE`` defaults to on and ``0``/``false`` turns it off."""
    raw = (environ if environ is not None else os.environ).get("RVE_BORDER_SAMPLE", "1")
    return str(raw).strip().lower() not in ("0", "false", "no", "off")


def probe_positions(duration, count=PROBES):
    """The preview timestamps in seconds: ``(i+1)/(count+1)`` of the duration, spread over the whole file."""
    if duration <= 0 or count <= 0:
        return []
    return [duration * (i + 1) / (count + 1) for i in range(count)]


def parse_header(text):
    """``(duration_seconds, frame_width, frame_height)`` from ffmpeg's input header, or ``None``.

    ffmpeg prints both on stderr when it opens the input (``Duration: 00:01:23.45`` and the video stream's
    ``1920x1080``), so one cheap ``-i`` call answers everything the seeked previews need.
    """
    if not text:
        return None
    duration_match = _DURATION_RE.search(text)
    resolution_match = _RESOLUTION_RE.search(text)
    if duration_match is None or resolution_match is None:
        return None
    hours, minutes, seconds = duration_match.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    width, height = int(resolution_match.group(1)), int(resolution_match.group(2))
    if duration <= 0 or width <= 0 or height <= 0:
        return None
    return duration, width, height


def parse_crop_line(text):
    """``(w, h, x, y)`` from the last ``crop=w:h:x:y`` in ``text``; ``None`` when there is no usable line.

    ``cropdetect`` prints the accumulated crop once per decoded frame, so a one-frame preview has one line and
    a whole-file pass has thousands - the last one is the settled value either way.
    """
    if not text:
        return None
    matches = _CROP_RE.findall(text)
    if not matches:
        return None
    width, height, x, y = (int(value) for value in matches[-1])
    if width <= 0 or height <= 0:
        return None
    return width, height, x, y


def sides_from_crop(crop, frame_w, frame_h):
    """``(top, bottom, left, right)`` black borders implied by ``crop = (w, h, x, y)`` in a frame."""
    width, height, x, y = crop
    return (
        max(0, y),
        max(0, frame_h - height - y),
        max(0, x),
        max(0, frame_w - width - x),
    )


def sample_valid(sides, frame_w, frame_h):
    """Whether a preview is representative; ``False`` for a crop that reaches a quarter of the frame.

    A title card, a fade through black or a credit roll crops far into the picture; letting one of those vote
    moves the median by hundreds of pixels. ``scan.c`` discards such a preview instead.
    """
    top, bottom, left, right = sides
    if frame_w <= 0 or frame_h <= 0:
        return False
    if min(top, bottom, left, right) < 0:
        return False
    if frame_w - left - right <= 0 or frame_h - top - bottom <= 0:
        return False
    return not (
        top * 4 >= frame_h
        or bottom * 4 >= frame_h
        or left * 4 >= frame_w
        or right * 4 >= frame_w
    )


def _median(values):
    """Integer median of non-negative sides, half up (``[3, 4, 4]`` -> 4, ``[3, 4]`` -> 4)."""
    return int(statistics.median(values) + 0.5)


def decide(samples, frame_w, frame_h, crop_threshold_frames=DEFAULT_CROP_THRESHOLD_FRAMES,
           crop_threshold_pixels=DEFAULT_CROP_THRESHOLD_PIXELS):
    """The border decision for already-validated samples.

    ``samples`` is a list of ``(top, bottom, left, right)``. The answer is a dict so a caller can log every
    stage: ``crop`` (``w, h, x, y``) or ``None``, the ``median`` it came from, whether ``loose`` mode was
    taken, how many ``samples`` voted, and a human ``reason`` when there is no crop.
    """
    decision = {
        "samples": len(samples),
        "median": None,
        "loose": False,
        "crop": None,
        "reason": "",
        "probes": 0,
        "fallback": False,
        "log": "",
    }
    if frame_w <= 0 or frame_h <= 0:
        decision["reason"] = f"invalid frame {frame_w}x{frame_h}"
        return decision
    if len(samples) < 3:
        decision["reason"] = f"fewer than 3 valid samples ({len(samples)})"
        return decision

    median = tuple(_median([sample[i] for sample in samples]) for i in range(4))
    decision["median"] = median
    disagreements = sum(
        1
        for sample in samples
        if any(expected - side > crop_threshold_pixels for expected, side in zip(median, sample))
    )
    loose = disagreements >= crop_threshold_frames
    decision["loose"] = loose
    sides = tuple(min(sample[i] for sample in samples) for i in range(4)) if loose else median

    # Even sides: the crop filter is fed yuv420p, where an odd offset/dimension is not addressable.
    top, bottom, left, right = (max(0, side) & ~1 for side in sides)
    width = frame_w - left - right
    height = frame_h - top - bottom
    if width <= 0 or height <= 0:
        decision["reason"] = f"crop {width}x{height} leaves nothing"
        return decision
    if width * height * 2 < frame_w * frame_h:
        decision["reason"] = f"crop {width}x{height} leaves less than half the frame"
        return decision

    decision["crop"] = (width, height, left, top)
    return decision


def format_decision(decision):
    """One log line per decision, in the shape ``detecting borders: 9 probes, 8 valid, median 4/4/0/0, ...``."""
    median = decision.get("median")
    median_text = "/".join(str(side) for side in median) if median else "n/a"
    prefix = (
        f"detecting borders: {decision.get('probes', 0)} probes, "
        f"{decision.get('samples', 0)} valid, median {median_text}, "
        f"loose={str(bool(decision.get('loose'))).lower()}"
    )
    crop = decision.get("crop")
    if crop is None:
        return f"{prefix}, no crop: {decision.get('reason') or 'undecided'}"
    width, height, x, y = crop
    return f"{prefix}, crop {width}x{height}+{x}+{y}"


def decide_probes(crops, frame_w, frame_h, crop_threshold_frames=DEFAULT_CROP_THRESHOLD_FRAMES,
                  crop_threshold_pixels=DEFAULT_CROP_THRESHOLD_PIXELS):
    """From raw probe crops (``(w, h, x, y)`` or ``None`` per preview) to the decision.

    ``fallback`` is set when not one preview produced a crop line: that is a container this ffmpeg cannot seek
    (broken index, non-seekable stream), not a file without borders, so the caller uses the whole-file pass.
    """
    samples = []
    for crop in crops:
        if crop is None:
            continue
        sides = sides_from_crop(crop, frame_w, frame_h)
        if sample_valid(sides, frame_w, frame_h):
            samples.append(sides)
    decision = decide(samples, frame_w, frame_h, crop_threshold_frames, crop_threshold_pixels)
    decision["probes"] = len(crops)
    decision["fallback"] = not any(crop is not None for crop in crops)
    if decision["fallback"]:
        decision["reason"] = "no preview returned a crop line"
    decision["log"] = format_decision(decision)
    return decision


class BorderDetect:
    def __init__(self, inputFile, ffmpeg_path="./bin/ffmpeg", probes=PROBES,
                 crop_threshold_frames=DEFAULT_CROP_THRESHOLD_FRAMES,
                 crop_threshold_pixels=DEFAULT_CROP_THRESHOLD_PIXELS):
        self.inputFile = inputFile
        self.ffmpeg_path = ffmpeg_path
        self.probes = probes
        self.crop_threshold_frames = crop_threshold_frames
        self.crop_threshold_pixels = crop_threshold_pixels
        #: True once a run had to fall back to the whole-file pass (the corpus check reads this).
        self.used_fallback = False
        #: The last :func:`decide` dict, for logs and for tests.
        self.last_decision = None

    # -- subprocess, thin --------------------------------------------------

    def _run(self, command):
        """``(returncode, stderr_text)``; ``(-1, "")`` when the binary itself cannot be started."""
        try:
            process = subprocess_popen_without_terminal(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                universal_newlines=True,
            )
        except (OSError, subprocess.SubprocessError) as e:
            log(f"border detect: ffmpeg could not be started: {e}")
            return -1, ""
        try:
            _, stderr = process.communicate()
        except subprocess.SubprocessError as e:
            log(f"border detect: ffmpeg probe failed: {e}")
            return -1, ""
        return process.returncode, stderr or ""

    def _probeHeader(self):
        """Duration and frame size from one input-header read (no decoding)."""
        _, stderr = self._run([self.ffmpeg_path, "-hide_banner", "-nostdin", "-i", self.inputFile])
        # ffmpeg exits 1 here ("at least one output file"), the header is already on stderr.
        return parse_header(stderr)

    def _probeFrame(self, timestamp):
        """One frame at ``timestamp``, cropdetected; ``(w, h, x, y)`` or ``None``.

        ``skip=0`` is load-bearing: cropdetect's default is ``skip=2`` (two initial frames are not reported),
        so with ``-frames:v 1`` the filter would print no ``crop=`` line at all - measured on ffmpeg n9.1,
        where the one-frame probe with the default emitted zero lines.
        """
        command = [
            self.ffmpeg_path,
            "-hide_banner",
            "-nostdin",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            self.inputFile,
            "-frames:v",
            "1",
            "-vf",
            "cropdetect=round=1:reset=1:skip=0",
            "-f",
            "null",
            "-",
        ]
        code, stderr = self._run(command)
        if code != 0:
            return None
        return parse_crop_line(stderr)

    # -- old whole-file pass, kept as the fallback and for the A/B -------

    def processBorders(self):
        command = [
            f"{self.ffmpeg_path}",
            "-i",
            f"{self.inputFile}",
            "-vf",
            "cropdetect",
            "-f",
            "null",
            "-",
        ]
        try:
            process = subprocess_popen_without_terminal(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                universal_newlines=True,
            )
            output = process.communicate()
            return output
        except (OSError, subprocess.SubprocessError) as e:
            log(f"Error during subprocess execution: {e}")
            return None, None

    def processOutput(self, output):
        if output is None or output[1] is None:
            return None

        borders = []
        for line in output[1].split("\n"):
            if "crop=" in line:
                crop_value = line.split("crop=")[1].split(" ")[0]
                borders.append(crop_value)

        if borders:

            def parse_crop(crop_str):
                # Expected format: "width:height:x:y"
                try:
                    width, height, x, y = map(int, crop_str.split(":"))
                    if width <= 0 or height <= 0:
                        log(f"Invalid crop dimensions: {crop_str}")
                        return None
                    return width, height, x, y
                except ValueError:
                    log(f"Invalid crop format: {crop_str}")
                    return None

            # Parse all crop values and filter out any invalid entries
            parsed_crops = [parse_crop(crop) for crop in borders]
            parsed_crops = [crop for crop in parsed_crops if crop is not None]

            if not parsed_crops:
                log("No valid crop values found.")
                return None

            # Determine the least cropped crop (i.e., largest area)
            least_cropped = max(parsed_crops, key=lambda dims: dims[0] * dims[1])
            least_cropped_str = f"{least_cropped[0]}:{least_cropped[1]}:{least_cropped[2]}:{least_cropped[3]}"

            return least_cropped_str

        return None

    # -- orchestration -----------------------------------------------------

    def _getBordersSampled(self):
        """The sampled decision as ``(width, height, x, y)``, or ``None`` when the file must be passed whole."""
        header = self._probeHeader()
        if header is None:
            log("border detect: the input header could not be read, falling back to the whole-file pass")
            self.used_fallback = True
            return None
        duration, frame_w, frame_h = header

        crops = [self._probeFrame(timestamp) for timestamp in probe_positions(duration, self.probes)]
        decision = decide_probes(
            crops,
            frame_w,
            frame_h,
            self.crop_threshold_frames,
            self.crop_threshold_pixels,
        )
        self.last_decision = decision
        if decision["fallback"]:
            log(
                "border detect: no preview returned a crop line (unseekable or broken index), "
                "falling back to the whole-file pass"
            )
            self.used_fallback = True
            return None

        log(decision["log"])
        if decision["crop"] is None:
            # "No crop" is the full frame, not `None`: the reader still needs a crop filter to run.
            return frame_w & ~1, frame_h & ~1, 0, 0
        width, height, x, y = decision["crop"]
        return width, height, x, y

    def _getBordersWholeFile(self):
        output = self.processBorders()
        output = self.processOutput(output)
        if output is None:
            log("No valid borders detected.")
            return None, None, None, None
        width, height, borderX, borderY = map(int, output.split(":"))
        return width, height, borderX, borderY

    def getBorders(self):
        if sampling_enabled():
            borders = self._getBordersSampled()
            if borders is not None:
                return borders
        else:
            log("border detect: sampling disabled (RVE_BORDER_SAMPLE=0), using the whole-file cropdetect pass")
        return self._getBordersWholeFile()
