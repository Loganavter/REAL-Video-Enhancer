# Finding: the source's edge boil does not block reuse — its motion does

Date: 2026-11-20. Status: measured, **not** emitted as records (the commands below are the
provenance). Read this before reopening "make the material reusable": four mechanisms were tried
and priced, and the premise they were built on turned out to be wrong.

## What was asked

Reuse works badly on `Untitled78.mov` (Pink Plasma, denoised, 1440x1080 10-bit): the gate refuses
the whole run. The suspicion was the denoiser — it re-renders the cel's edges frame to frame, so
the picture "boils", and a frame has nothing stable to freeze.

## The correction: the boil is 1 level, and it is not what the gate sees

The denoised file's true source is `078 - Pink Plasma (1975)_nxl1.mp4` (same length to 0.01 s;
correlation 0.9997, mean difference 1.08 levels). Comparing the two frame for frame, the
denoiser's own effect changes between frames by **mean ~1 level, p99 3-5 levels, and more than 8
levels on 0.01-0.02% of pixels**.

The gate's statistic, on the same 16-frame windows, for the clean source and the denoised one:

| window | clean `nxl1` | denoised `Untitled78` |
|---|---|---|
| t=0 (black/static opening) | floor 0.00, worst pair 0.03 | floor 0.00, worst pair 0.06 |
| t=60 | floor 10.34, worst pair 123.21 | floor 10.00, worst pair 122.76 |
| t=120 | floor 7.38, worst pair 95.72 | floor 6.98, worst pair 96.13 |

The denoise *lowers* the floor slightly. The 7-10 it reports is **motion** in the window (and
scene cuts), and at t=0 — where the material is static — the floor is 0.00 and reuse engages.
So: reuse already works on this material wherever the content holds still, and is refused where
it moves. The boil is below the 8-level acceptance bar and below the gate's 2-4 threshold; it was
never the blocker.

## The four mechanisms, priced

| mechanism | result |
|---|---|
| whole-frame reuse, forced at the maximum threshold (4) | 9/62 frames reused, 6 frames over the bar, worst **0.534%** (bar 0.1%) |
| residual compute, frame-level (`out(t-1) + out(dt)`) | 0.46-6.46% against a full pass - no better than plain reuse |
| residual compute, per-layer (Skip-Convolutions recursion) | **exact** without a gate (0.00000%, peak 0), and 27% of positions skip at eps 16 with peak 4; but a dense hop is 6.66 ms, the gate alone 4.30 ms, and the sparse path costs 10.4 ms at the skip fractions it can justify - **1.6x slower** |
| AABB stabiliser (TAA history + 3x3 clip), alpha 0.1 | 0.99 ms/frame, gate engages (floor 1.45, 34/60 reused), render **0.813%** over the bar - 8x |
| MCTF with OpenCV Farneback flow | floor 3.07 -> 2.51, drift peaks 146 levels - flow is estimated on the same edges that need fixing |
| Lai et al. blind consistency (PWC-Net reimplementation, trained weights) | loads and runs under torch 2.9 (0 missing tensors) and removes synthetic brightness jitter 11x, but on every real pair it changes almost nothing (floor 6.98 -> 6.78, drift 1%) |

The last row is the useful negative: the model is trained for **appearance** inconsistency
(brightness, colour, contrast), not for local structure being re-rendered, so a defect of ~1
level is invisible to it. It is not the wrong wiring - the synthetic control proves the wiring.

## What this closes

- "Stabilise the source so the gate engages" is closed: the gate is not blocked by a fixable
  defect, and the only stabiliser that made it engage (AABB) did so by averaging motion and
  missed the bar by 8x.
- "Compute on the residual" is closed for this card and this model width: see the cost line.
- The auto tuner now certifies on the acceptance statistic (`RenderVideo`, 2026-11-20), so the
  proxy that once certified that AABB configuration can no longer do it.

## What remains open

- A learned stabiliser **fine-tuned on this defect class**. There is a usable training pair on
  disk (`078_nxl1` clean against `Untitled78` denoised, aligned to 0.9997), which is what such a
  fine-tune would need - but the defect it would learn to remove is 1 level, i.e. below the bar;
  the honest reason to do it would be picture quality, not reuse.
- Fixing the denoiser's temporal consistency upstream, in the restoration pipeline rather than in
  RVE.

## Second pass: calibration does not save it, and the ceiling is the real number

Asked after the first write-up: if the calibration is the problem, does reuse grow? Measured, on
a 30 s run started on the calm opening (which is what a per-scene calibration would look like):

| | |
|---|---|
| gate decision | engaged, threshold 4, chain 4, floor 0.01, acceptance 0.00% on the 16 calibration frames |
| reuse | **81/720 = 11.2%** |
| damage on the run | **0.482% worst frame, 46 of 540 frames above the 0.1% bar** |

So reuse grows from 0% to ~11% and the picture fails by 4.8x. The calibration sample is calmer
than the run it decides - the documented ~5x sample bias, here unbounded (0.00 against 0.482).
Calibration is not the lever.

The ceiling says why. Sampled at native 24 fps over the episode (`--clip`):

    <= 2 levels:  6-8% of frame pairs      <= 8 levels: 37%
    <= 4 levels:  8-10%                    median delta 11-13, 21-35% of pairs are cuts

No mechanism can exceed the ceiling, and the ceiling at the gate's thresholds is ~10%.

## Why the pairs differ on what should be held cels - open

An animation shot on twos should give ~50% identical consecutive frames, and this gives 8%.
Phase correlation over several windows shows the shift between consecutive frames is
**sub-pixel** (median 0.05-0.15 px, max ~1 px) - so no camera pan - and compensating it removes
10-50% of the delta on some windows (t=200: 20.5 -> 9.3) and almost nothing on others (t=60:
55.4 -> 55.0, cuts). What is left is not a global move.

**Answered for the source**: there are no identical frames to reuse. Over 47 consecutive pairs of
the clean `nxl1` (before any denoise), exactly **one** pair is bit-identical and one is within 0.5
levels; the median pair differs by 15.0 levels. The drawing may well be held, but the film scan
does not hold: grain, dust and sub-pixel weave make every frame unique at pixel level (the denoised
file reads the same: 1 of 47, median 14.9). So the 8-10% ceiling is a property of the material, not
a measurement error - those are the pairs where grain and weave happen to be small, and they reuse
safely (0.096% at the opening, against the 0.1% bar).

Still unmeasured: how much of that per-frame difference is grain, how much is scan weave, and how
much the dust-removal adds.

## The lever that is left: decide on the structure, not on the pixels

If only the drawing has to survive - grain, dust and weave are not content - then the reuse
decision should compare the structure. Measured on the clean `nxl1`, block-mean delta after a
2-px Gaussian low-pass and after compensating the sub-pixel shift:

| window | raw pixels, <=4 | low-pass, <=4 | low-pass + shift, <=4 |
|---|---|---|---|
| t=120 | 4.3% (median 11.84) | 30.4% | **47.8%** (median 4.08) |
| t=200 | 8.7% (12.92) | 26.1% | **34.8%** (4.83) |
| t=280 (fast motion) | 4.3% (26.70) | 4.3% | 8.7% (8.26) |

So in calm shots the share of pairs that carry the same drawing is **5-10x the share the pixel
statistic sees**, and it still refuses the fast-motion window - the statistic separates the two
cases, which the raw block mean cannot.

**The blocker is the acceptance bar, not the gate.** It is written on pixels ("share of pixels
more than 8 levels off, worst frame, 0.1%"), so it counts the grain difference of a reused frame
as damage - exactly the difference the owner says need not be preserved. With a pixel bar, a
structural reuse cannot pass however correct the drawing is. To take this lever, the bar has to be
redefined for this material (compare after the same low-pass, and keep the pixel bar for the
frames that are freshly rendered), which is the owner's call - the picture judgement is his, and
this note can only price what each choice costs:

* structure bar: ~35-48% of frames reusable in calm shots, grain frozen on those frames (a pulsing
  grain pattern between reused and fresh frames is the visible risk);
* pixel bar: 8-10% ceiling, and only where grain and weave happen to be small.

A third path removes the conflict: denoise the grain away *before* the upscale, which is the one
thing the current dust-removal does not do (the denoised file reads the same per-frame difference
as the clean one: 1 exact duplicate in 47 pairs, median 14.9 against 15.0).

The deeper measurement problem: a fraction-of-a-pixel shift of line art moves edge pixels by tens
of levels, so the gate's statistic (max over blocks of the mean |difference|) and the acceptance
statistic (share of pixels >8 levels) both read invisible sub-pixel shimmer as motion. Nothing in
the current instrument set separates "invisible" from "moved".

## Harness: the guard against repeating this

`harness/scripts/reuse_ceiling.py` - one command answers the question the day was spent on:

* `--clip CLIP` reports the reuse ceiling (share of frame pairs under 2/4/8 levels, cut share,
  median) sampled at **native frame rate**. The 1 fps sampling that made this material look static
  is not possible through this instrument.
* `--pair A B` prints correlation, mean difference and share >8 levels before any model is fed a
  pair. The blind-consistency check first ran on the wrong "original" and read as "the model does
  nothing"; the true pair correlates 0.9997 and differs by 1.1 levels.
* `--wobble CLIP` prints the per-pair shift and the delta before and after shift compensation,
  so the next investigator sees whether the obstacle is translatable.
* A **control runs on every invocation**: identical frames must read exactly 0 and a known
  +8-level difference must read >= 8. A broken instrument now fails loudly instead of producing a
  plausible number.

## Reproduce

```bash
# the gate's own statistic on a window, clean against denoised
python3 harness/scripts/temporal_stabilize_probe.py CLIP --frames 16 --start 120

# residual compute: accuracy curve, then the cost floor on this card
python3 harness/scripts/residual_reuse_probe.py CLIP --frames 8 --scale 0.5 --cost

# MCTF with the installed flow
python3 harness/scripts/mctf_stabilize_probe.py CLIP --frames 12

# the ceiling, the pair check and the wobble - start here, not with a mechanism
python3 harness/scripts/reuse_ceiling.py --clip CLIP --duration 20
python3 harness/scripts/reuse_ceiling.py --pair ORIGINAL PROCESSED --start 120
python3 harness/scripts/reuse_ceiling.py --wobble CLIP --start 200 --frames 14

# the AABB stabiliser, end to end (encode stabilised -> render -> gate_sweep)
# see the commit of 2026-11-20 for the exact commands; the numbers above are its output
```

## Uncovered scope

One clip, one model, one card. The Lai check ran the PWC-Net reimplementation's weights
(trained on the W3_D1_C1_I1 task mix), not the unavailable original ECCV18 checkpoint; and it ran
at 1440x1080 with the original supplied from the aligned nxl1 file. The acceptance numbers were
taken with `ffv1` renders; the product encoder changes the fixed cost, not the ratio between the
arms.
