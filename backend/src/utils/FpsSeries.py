"""The run's speed over time, as the Properties window's chart reads it.

Why a series and not the per-phase averages: `phases_ms_per_frame` is one number per stage, averaged over the whole
run. Averaging is exactly what erases the thing a curve is for - a model swap mid-run, a stall while the writer
drains, a section where the pipeline fell apart - and the owner asked for the shape of the run instead of its mean
(2026-11-28: "а в чем великий смысл тут у колонки Phases? Показали бы уж лучше какой-нибудь график обработки").

What is stored
--------------
The render loop appends one ``(frame, seconds)`` point every ``sample_stride()`` output frames
(``RenderVideo._sample_fps``) plus one at the end. This module turns that sparse record into the report's
``fps_series``: ``frames`` / ``seconds`` / ``fps`` as three equal-length lists, starting at ``(0, 0.0, 0.0)``.

``fps`` is a WINDOW rate - ``delta_frames / delta_seconds`` between consecutive stored points - not the run's
global average. That is the point: if the last minute of the run fell to 3 fps, the window says 3, while the
average says 20.

Bounded by construction: for a run of N output frames the loop samples every ``N // 240`` frames, so a run stores
about 240 points regardless of its length; ``MAX_POINTS`` (256) is the hard ceiling, and the thinning here keeps
the start, evenly spaced middles and the final point if a run outlived its own frame estimate. One tuple per
sampled frame costs one list append and one ``perf_counter()`` read per sample, so the loop's own cost is
unchanged in any way a user can see (the profile timings around it are milliseconds; this is microseconds per
240 frames).

Pure by design: no torch, no clock, no rendering - the backend's test interpreter has none of them, so this is the
one part of the series the unit tests can construct out of a synthetic profile.
"""

from __future__ import annotations

import math

#: The most points the window's chart may be handed. 240 sampled frames + the synthetic start + the final point
#: stays under it, and the thinning below is the hard guarantee for a run whose frame estimate was too low.
MAX_POINTS = 256

#: How many samples a whole run is divided into before the endpoints are added.
TARGET_POINTS = 240


def sample_stride(total_frames) -> int:
    """Every how many OUTPUT frames the render loop stores one speed sample.

    At least one: a short run samples every frame (and is under ``TARGET_POINTS`` anyway), a long one samples
    every ``total // 240``. A missing/unusable estimate reads as 1, which is the safe side: the report-time
    thinning still bounds the series, and ``_sample_fps`` widens the stride if the list reaches the cap.
    """
    try:
        total = int(total_frames or 0)
    except (TypeError, ValueError):
        total = 0
    return max(1, total // TARGET_POINTS)


def _thin(points: list, limit: int) -> list:
    """``points`` reduced to at most ``limit``, keeping the first, evenly spaced middles and the LAST point.

    The last point is what says where the run ended up; a plain slice would drop it whenever the count is not a
    multiple of the step, and a curve that stops 3 % short of the end hides the part where the encode drained.
    """
    if len(points) <= limit:
        return points
    step = max(1, math.ceil((len(points) - 1) / float(limit - 1)))
    kept = points[0:len(points) - 1:step]
    kept.append(points[-1])
    return kept


def fps_series(profile) -> dict | None:
    """The report's ``fps_series`` from the render loop's ``_profile`` dict, or None when there is nothing to draw.

    ``profile["fps_samples"]`` is the list ``RenderVideo._sample_fps`` appends to: ``(frame, seconds)`` pairs.
    Non-monotonic input is dropped rather than trusted - frames must not go backwards and neither may the clock,
    because either one produces a negative window and a rate that is arithmetic, not a measurement.

    None (not an empty series) for an empty run: the Properties window then prints the reason row, and no chart is
    built from a curve with no points in it.
    """
    if not isinstance(profile, dict):
        return None
    raw = profile.get("fps_samples")
    if not isinstance(raw, (list, tuple)) or not raw:
        return None

    points: list[tuple[int, float]] = [(0, 0.0)]
    for sample in raw:
        try:
            frame = int(sample[0])
            seconds = float(sample[1])
        except (TypeError, ValueError, IndexError):
            continue
        if frame <= points[-1][0] or seconds <= points[-1][1]:
            continue
        points.append((frame, seconds))
    if len(points) < 2:  # only the synthetic origin: no window, nothing to draw
        return None

    points = _thin(points, MAX_POINTS)

    frames = [point[0] for point in points]
    seconds = [round(point[1], 3) for point in points]
    rates = [0.0]
    for index in range(1, len(points)):
        delta_frames = frames[index] - frames[index - 1]
        delta_seconds = seconds[index] - seconds[index - 1]
        # The start point has no rate, and a zero-length window has no rate either (a clock that did not move is
        # not an infinite speed): both stay 0.0, which the chart draws at the floor.
        rates.append(round(delta_frames / delta_seconds, 3) if delta_seconds > 0 else 0.0)
    return {"frames": frames, "seconds": seconds, "fps": rates}
