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

## Reproduce

```bash
# the gate's own statistic on a window, clean against denoised
python3 harness/scripts/temporal_stabilize_probe.py CLIP --frames 16 --start 120

# residual compute: accuracy curve, then the cost floor on this card
python3 harness/scripts/residual_reuse_probe.py CLIP --frames 8 --scale 0.5 --cost

# MCTF with the installed flow
python3 harness/scripts/mctf_stabilize_probe.py CLIP --frames 12

# the AABB stabiliser, end to end (encode stabilised -> render -> gate_sweep)
# see the commit of 2026-11-20 for the exact commands; the numbers above are its output
```

## Uncovered scope

One clip, one model, one card. The Lai check ran the PWC-Net reimplementation's weights
(trained on the W3_D1_C1_I1 task mix), not the unavailable original ECCV18 checkpoint; and it ran
at 1440x1080 with the original supplied from the aligned nxl1 file. The acceptance numbers were
taken with `ffv1` renders; the product encoder changes the fixed cost, not the ratio between the
arms.
