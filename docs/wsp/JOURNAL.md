# Journal

Every measurement and every decision that changes a quoted number, newest first. Each entry carries
the commit, the card and the library versions it was taken on: a number without those describes a
build that may no longer exist (`harness/provenance.py`).

Entries: 2

## 2026-11-20T12:31:44+03:00 - 0feb0d1c0587 (dirty tree) - NVIDIA GeForce RTX 4060 Ti (cc 8.9)
*triton 3.6.0*

The 4x path measured, which the unit tests could only algebraically: gate reuses 27/41 at threshold 5 on the
4x animevideov3; with the correction the worst frame is 0.05626% and the MEDIAN is 0.00000% (verbatim 0.05389%
and 0.00407%), both under the bar. Cost at 4320p is the finding: 40.0 ms/frame corrected against 24.9 verbatim
and 55.9 ungated, i.e. 1.40x instead of 2.25x, because a corrected frame is new and has to be converted for
the writer while verbatim re-emitted the cached bytes.

Records: RU061, RU062, RU063, RU064, RU065

## 2026-11-20T12:11:59+03:00 - 6f8641157076 (dirty tree) - NVIDIA GeForce RTX 4060 Ti (cc 8.9)
*triton 3.6.0*

int8 wall re-measured, interleaved, on two clips: 1.65x end to end (null and nvenc sinks), 1.14x and 0.93x on
a lossless sink where the writer is the bottleneck. The 2026-11-18 medians W009/W025/W041/W057 are superseded:
that body ran 44.6-47.9 ms/frame, today's runs 26.5, so the recorded 1.09-1.14x described a build that no
longer exists. Picture, A/A identical in both arms: lossless worst frame 0.00048% (w1040) and 0.08491% (w960),
0 frames over the bar on either clip; through the delivery encode 0.3471% with 90/205 frames over (w1040) and
0.09176% with 0/201 (w960) - the encoder, not the body, moves those pixels.

Records: RU053, RU054, RU055, RU056, RU057, RU058, RU059, RU060

Supersedes: W009, W025, W041, W057
