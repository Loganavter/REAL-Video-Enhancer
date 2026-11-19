# Review request: can temporal frame reuse be made safe on real (noisy) material?

Written 2026-11-19 for an outside reader. The numbers cited here live in
`docs/wsp/verified/reuse-decision-statistic.txt` (records RU001-RU011), `gate-bd.txt` and `gate-roc.txt`;
this page is the problem statement, not the evidence. Nothing here is a proposal to ship: frame skipping is
**off by default** and stays off until the questions at the end have answers.

## The system, in one paragraph

A video super-resolution pipeline runs a 2x convolutional network on every frame. Consecutive frames are often
near-identical (animation on twos, calm shots, held cels), so the pipeline can *skip the network* and re-emit
the last super-resolved frame - the "temporal reuse gate". The win is large: on animation the measured
end-to-end speed-up is 3.3-4.3x, and skipping is the only lever of that size left (the engine itself is
1.6-1.9x over the upstream baseline, and the int8 body another 1.4-1.7x). The gate is also the only lever that
is content-dependent, and that is where it fails.

## The two failures, as measured

**(1) The decision statistic cannot separate noise from motion.** The gate decides per 32 px block on the LR
frame: a block is "unchanged" when the block mean of |difference| against the last kept frame stays under a
threshold (2-4 of 255). Digital noise is temporally independent, so on real material the statistic reads a
noise floor that has nothing to do with content motion. Measured across 18 eight-second windows of one
Blu-ray episode: noise floor 0.35-51.5 at the same nominal settings. Where the floor is above the 1.5 refusal
limit the gate refuses - and the material where it refuses is exactly where reuse would pay most (grainy
film gets 1.85x from the engine, clean animation 3.3-4.3x).

**(2) Where it reuses, the output stops living.** Two distinct artifacts, and the acceptance statistic sees
only the first:

- *Per-pixel error against the ungated render*: the share of pixels more than 8 levels off in any channel.
  On the 1040 s window: 130 of 205 frames reused, **37 of those 130 reuses above the 0.1% bar**, worst
  0.44121% at frame 122.
- *Frozen texture*: a reused frame carries the reference's noise, so on flat static areas the output stops
  changing from frame to frame. Measured as the block temporal |diff| in the gated render over the same in
  the ungated one: **130 reused frames median 0.000 (91.5% below 0.5), 74 rendered frames median 1.026
  (0% below 0.5)**. Of the 93 reused frames that *pass* the 0.1% bar, **82 are frozen**. The bar sees less
  than a third of the artifact class.

A human looking at the crops describes (2) as "the grain died" and (1) as "the line holds an older pose".
They are not the same defect and they do not have the same instrument.

## What we have already ruled out

- **Threshold tuning.** The block-mean statistic at any threshold in the tuner's clamp (2.0-4.0) leaves
  frames above the bar on the failing segment. Widening the chain, a pixel veto at 40 levels, a block-max
  second condition, and a channel-max share were each measured; channel-max passes the share and concentrates
  the damage (worst 64 px block mean 20.90 against 7.57 on the artifact that was rejected by eye).
- **The retired signed statistic** (|mean(a) - mean(b)| per block) - ranks worse than the shipped one
  (AUC 0.723 against 0.844 on the same 130 reuses).
- **Motion-compensated ranking as a drop-in fix.** DIS optical flow, warp, residual: the mean of the residual
  ranked the 1040 s damage at AUC 0.906 (against 0.844), but on a **held-out** clip (960 s, 123 reuses, 7 above
  the bar) the same statistic at the same calibrated threshold collapsed to AUC 0.489 while the shipped block
  statistic reached 0.994. The advantage was a property of one segment.
- **Noise-normalising the residual** (share of pixels above 3 x MAD-sigma) - worse than chance (AUC 0.384):
  on flat regions both the residual and its sigma are tiny, so a little motion reads as a large share.

## What the measurements now say about the shape of the problem

1. **The failure is non-stationary.** The statistic that ranks one segment's damaged reuses almost perfectly
   ranks the next segment's at chance, in both directions. There is no global threshold - and, more
   importantly, no *fixed* statistic has yet been shown to be a decision variable at all.
2. **The acceptance instrument is the wrong one for half the artifact.** A per-pixel share at one frame
   cannot express "the texture stopped moving". We can measure the freeze (numbers above) but we have no
   accepted threshold for it, and no perceptual calibration of what ratio a viewer accepts.
3. **Decision and judgement use different references.** The gate decides against the last *kept* frame; the
   bar judges against the *ungated render of the same build*; the artifact the eye rejects is the difference
   from the *source's* grain. Three references, one pipeline.
4. **The cheap routes have been tried and are documented as dead ends** so a reviewer does not have to
   suggest them: exact-duplicate gating (0% bit-identical on encoded video), luma-only statistics (a clipped
   highlight differing 11 levels in R only reads 2.75 in luma), tile-level reuse (halo tax eats the saving),
   and a per-frame cascade of two models (the saving is conditional on the gate rendering, which removes it).

## Candidate directions, each with what would falsify it

- **A decision statistic with a bound, not a threshold.** The gate's job is to predict a per-pixel error on
  the HR output from LR evidence. The empirical HR/LR amplification on this material is median 2.04, p90 9.50,
  max 35.4 (measured over 100 reused frames), so a criterion of the form `delta_LR * L_p90 <= bar` is
  computable and would be a *certificate* rather than a proxy. Falsifier: on the two labelled clips, does a
  bound-based criterion at a fixed confidence reject the damaged reuses without rejecting most of the clean
  ones? (Our current numbers say the LR-side statistics cannot even rank them consistently across segments,
  which is what makes us ask whether the bound is the right frame at all.)
- **Split the artifact from the signal: denoise, reuse the clean plate, re-synthesise grain per frame.**
  This is the codec industry's answer (AV1 film-grain synthesis; ITU-T H Suppl.21, 2025; deep FG removal and
  synthesis, IEEE TIP 2023): grain is temporally independent, so it is removed before coding and regenerated
  at the decoder from an AR template plus an intensity-vs-luma curve. Applied here, a reused frame would get
  fresh synthesised grain instead of the reference's frozen noise. Falsifier: does a re-synthesised output
  pass the texture criterion by construction, and what does it cost per frame; does it make the *per-pixel*
  error worse (synthesised grain is not the source's grain).
- **Reuse with a correction instead of reuse as-is.** Video coding never re-emits a prediction without a
  residual; leaky prediction bounds drift with a leak factor. Here: warp the previous SR output by flow and
  add a cheap residual (a small convolution on the LR difference, or a blend with alpha < 1) instead of a hard
  skip. Falsifier: the residual path must be materially cheaper than the model call (our body is 26-30 ms per
  frame at 1080p; a residual path above ~8 ms is not worth it) and must reduce both the per-pixel error and
  the freeze.
- **Per-segment calibration.** If the failure class is non-stationary, the decision may need a per-shot
  calibration of whatever statistic is chosen - which is what the gate's auto-tuner does today for one
  scalar, and what it demonstrably fails to do across two segments.

## Questions for the reviewer

1. Is there published work on **temporal reuse with a guaranteed bound** on the reconstructed frame's error
   (as opposed to a heuristic similarity threshold)? We would expect this in video coding under "drift" and
   "leaky prediction", but we have not found a formulation that carries over to a learned upscaler whose
   response to an input change is not linear.
2. Is the **two-instrument split** (per-pixel error vs temporal texture) recognised in the literature, and is
   there an accepted metric for "the texture stopped moving"? Our freeze ratio is home-made.
3. Has anyone applied **film-grain re-synthesis to a frame-reuse pipeline** (as opposed to a codec)? What
   breaks when the grain model is estimated from the source but the plate comes from a model's output?
4. Is **non-stationarity of the failure** a known phenomenon in this class of gate (e.g. adaptive computation
   for video), and how is it usually handled - per-shot calibration, uncertainty, or a different family of
   statistic entirely?
5. We would rather be told the problem is mis-framed than receive another statistic to try. If you know a
   reason why "predict an HR per-pixel error from LR evidence" cannot work for a learned upscaler, that is the
   most useful answer this page can get.

## What this page does not establish

One episode (Chainsaw Man ep.01), one model (2x SRVGGNetCompact), one GPU, 8-second windows, the gate's
reference approximated by t-1 in the offline statistics, no perceptual study. The clip the lane was originally
closed on (539 frames, 17 frames above the bar) has **no packet-level provenance** and was lost with `/tmp`;
every number here is on `-c copy` windows cut from the BDRemux at recorded offsets.
