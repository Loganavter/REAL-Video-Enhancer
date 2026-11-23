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

The decision is cached in ``.crop.json`` beside the render's report (or output) file, so re-rendering the
same input performs no probes at all. The cache key is ``(input path, size, mtime_ns, probe params)``: a
different file, a different size/mtime or different probe parameters is a miss. A cached "no crop" (the
full frame) is a decision too and is reused as-is. A corrupt or partial file is ignored as a miss, never a
crash, and ``RVE_BORDER_CACHE=0`` (or ``false``/``no``/``off``) disables the cache for A/B runs.
"""

from __future__ import annotations

import json
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


# -- decision cache ------------------------------------------------------------

#: The cache file name. It lives beside the render's report/output, never beside the input: sources are
#: routinely mounted read-only, so a cache in the input directory could not be written at all.
CACHE_FILENAME = ".crop.json"

#: Bumped when the payload shape changes; an older file then reads as a miss instead of a bad decision.
CACHE_VERSION = 1


def cache_enabled(environ=None):
    """Whether the decision cache is on; ``RVE_BORDER_CACHE`` defaults to on and ``0``/``false`` turns it off."""
    raw = (environ if environ is not None else os.environ).get("RVE_BORDER_CACHE", "1")
    return str(raw).strip().lower() not in ("0", "false", "no", "off")


def cache_location(output_path=None, report_path=None):
    """``.crop.json`` beside the run report (preferred) or the output file; ``None`` when neither is known.

    The report is preferred because it names the run, while a preview/render output can be a temporary
    container; both live in the writable destination tree, unlike the input.
    """
    for candidate in (report_path, output_path):
        if not candidate:
            continue
        directory = os.path.dirname(os.path.abspath(os.fspath(candidate)))
        return os.path.join(directory, CACHE_FILENAME)
    return None


def cache_key(input_path, probes=PROBES, crop_threshold_frames=DEFAULT_CROP_THRESHOLD_FRAMES,
              crop_threshold_pixels=DEFAULT_CROP_THRESHOLD_PIXELS, sampled=None):
    """The identity of a detection: input file state plus the parameters that change the probes.

    ``sampled`` is the whole-file/sampled choice and is part of the key: the two paths run different
    probes, so an A/B toggle must not read the other mode's answer. Returns ``None`` when the input
    cannot be stat()ed - there is nothing to key on, so the run is a miss.
    """
    try:
        stat = os.stat(input_path)
    except OSError:
        return None
    if sampled is None:
        sampled = sampling_enabled()
    return {
        "input": os.path.abspath(os.fspath(input_path)),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "probes": int(probes),
        "crop_threshold_frames": int(crop_threshold_frames),
        "crop_threshold_pixels": int(crop_threshold_pixels),
        "sampled": bool(sampled),
    }


def _valid_borders(borders):
    """Four ints ``(width, height, x, y)`` with a positive frame; ``None`` and partial answers are not cached."""
    if not isinstance(borders, (tuple, list)) or len(borders) != 4:
        return False
    if any(type(value) is not int for value in borders):
        return False
    width, height, x, y = borders
    return width > 0 and height > 0 and x >= 0 and y >= 0


def read_cache(path, key):
    """The cached answer for ``key`` as ``{"borders", "used_fallback"}``, or ``None`` (a miss).

    A missing, unreadable, truncated, wrong-version or wrong-key file is a miss. It must never raise: a
    cache is an optimisation, and a damaged one has to cost a re-probe, not a render.
    """
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != CACHE_VERSION:
        return None
    if payload.get("key") != key:
        return None
    borders = payload.get("borders")
    if not _valid_borders(borders):
        return None
    used_fallback = payload.get("used_fallback", False)
    if not isinstance(used_fallback, bool):
        return None
    return {"borders": tuple(borders), "used_fallback": used_fallback}


def write_cache(path, key, borders, used_fallback):
    """Store ``borders`` under ``key``; returns whether it was written.

    The write is a temp file plus ``os.replace`` in the same directory, so a concurrent reader sees the
    old file or the new one, never a half-written one. Any OS error is logged and swallowed: a destination
    that is full or read-only must not take a render down.
    """
    if not path or key is None or not _valid_borders(borders):
        return False
    payload = {
        "version": CACHE_VERSION,
        "key": key,
        "borders": list(borders),
        "used_fallback": bool(used_fallback),
    }
    temporary = f"{path}.{os.getpid()}.tmp"
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
        return True
    except OSError as exc:
        log(f"borders: cache write failed ({path}): {exc}")
        try:
            os.unlink(temporary)
        except OSError:
            pass
        return False


def _describe_key(key):
    """The one-line identity the cache logs: file, size, mtime and the probe parameters."""
    return (
        f"{os.path.basename(key['input'])} {key['size']}B mtime_ns={key['mtime_ns']} "
        f"probes={key['probes']} thresholds={key['crop_threshold_frames']}/{key['crop_threshold_pixels']} "
        f"sampled={str(key['sampled']).lower()}"
    )


def _describe_borders(borders):
    width, height, x, y = borders
    return f"{width}x{height}+{x}+{y}"


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
                 crop_threshold_pixels=DEFAULT_CROP_THRESHOLD_PIXELS,
                 cache_path=None):
        self.inputFile = inputFile
        self.ffmpeg_path = ffmpeg_path
        self.probes = probes
        self.crop_threshold_frames = crop_threshold_frames
        self.crop_threshold_pixels = crop_threshold_pixels
        #: Where the decision is cached (beside the report/output); ``None`` disables caching for this run.
        self.cachePath = cache_path
        #: True once a run had to fall back to the whole-file pass (the corpus check reads this).
        self.used_fallback = False
        #: The last :func:`decide` dict, for logs and for tests. ``None`` on a cache hit: no decision ran.
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
        """The crop, from the cache when this exact input+params was decided before, else from the probes.

        The cache is checked before anything is spawned, so a re-render of the same input performs zero
        probes (``borders: cache hit``). A miss logs its reason and writes the fresh decision afterwards.
        """
        if not cache_enabled():
            log("borders: cache disabled (RVE_BORDER_CACHE=0)")
            return self._decideBorders()

        key = cache_key(
            self.inputFile,
            self.probes,
            self.crop_threshold_frames,
            self.crop_threshold_pixels,
            sampled=sampling_enabled(),
        )
        if key is None:
            log(f"borders: cache miss (the input {self.inputFile!r} could not be stat()ed)")
            return self._decideBorders()

        cached = read_cache(self.cachePath, key)
        if cached is not None:
            self.used_fallback = cached["used_fallback"]
            log(
                f"borders: cache hit {self.cachePath} "
                f"({_describe_key(key)} -> {_describe_borders(cached['borders'])})"
            )
            return cached["borders"]

        log(f"borders: cache miss {self.cachePath or '(no report/output path)'} ({_describe_key(key)})")
        borders = self._decideBorders()
        # A None/partial answer means "nothing could be decided"; caching it would freeze a failed probe
        # into every later run, so only a real decision (including the full frame) is stored.
        if _valid_borders(borders):
            write_cache(self.cachePath, key, borders, self.used_fallback)
        return borders

    def _decideBorders(self):
        """The uncached decision: the sampled path, or the whole-file pass when that falls back."""
        if sampling_enabled():
            borders = self._getBordersSampled()
            if borders is not None:
                return borders
        else:
            log("border detect: sampling disabled (RVE_BORDER_SAMPLE=0), using the whole-file cropdetect pass")
        return self._getBordersWholeFile()
