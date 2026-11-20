# Review request 2, 2026-11-20: the trade-off looks like policy - we think it is an unseparated correction

Follow-up to the reply of 2026-11-19. Self-contained. The numbers below are records in
`harness/docs/wsp/verified/reuse-decision-statistic.txt` (RU001-RU024); the artifacts are scratch renders of one
provable cut (Chainsaw Man ep.01, 1040 s, 205 frames, ffv1, 2x SR, 1920x1080 -> 3840x2160).

## 1. What your review changed (thank you - one of the three was a bug in our numbers)

1. **The `t-1` approximation was ours and it mattered.** Recomputed against the reference the gate actually
   served, the shipped statistic ranks the damaged reuses at **AUC 0.991 and 0.988** on the two labelled clips
   instead of 0.844 and 0.994. So "the failure class is non-stationary" was substantially our artefact and is
   withdrawn: the *ranker* is fine.
2. **Your flow mechanism is confirmed**: the seven damaged reuses at 960 s sit at mean flow magnitude **2.26**
   against 0.48 on the clean ones (coherent motion, which warping removes); at 1040 s it is the reverse,
   **0.057** against 0.266 (local redraw at almost no motion). Two mechanisms, opposite signatures.
3. **Your proposed fixes were tested and do not close it.** Adding a flow term (product, max, and a
   two-branch OR/AND rule with thresholds calibrated on one clip) never beat the block branch alone. The
   reference-free rho1 of the high-frequency residual does not separate on this material (fresh 0.918-0.931,
   frozen 1.000) because the render's HF energy is static line detail, not grain.
4. **A reader bug of ours, retracted**: the first residual-reuse numbers (0.26648%, texture 0.989) were
   computed with three misaligned streams (ffmpeg emitted yuv420p while the reader assumed rgb24, so every read
   consumed two frames). Corrected below. That is the second reader bug in this lane in two days.

## 2. The certification rule, when written down, is not a certificate

Skip when the statistic (against the served reference) exceeds the threshold picked on clip A (2.172 levels),
count a violation when a skipped frame is past the 0.1% bar:

| clip | skipped | violations | rate |
|---|---|---|---|
| A, 1040 s (calibration) | 130 | 37 | **28.46%** |
| B, 960 s (held out) | 123 | 6 | **4.88%** |

A per-chunk audit controller (10% of each 20-reuse chunk rendered and scored, threshold moved 5% up or 2%
down) does not converge at two audited frames per chunk: the threshold drifts up from 2.172 to 2.485 and the
violations stay. The binomial arithmetic agrees: zero violations in `n` audits bound the rate at
`1 - 0.05^(1/n)` - 5.8% at n=50, **0.77% at n=388**, i.e. more audits than reuses.

## 3. The part that looks like a policy choice, and why we think it is not

Reuse with a correction instead of verbatim - output = gated frame + `gain * nearest-2x(LR_t - LR_ref)` - on
the same 130 reuses:

| gain | worst share | frames over the 0.1% bar | texture ratio | frames frozen |
|---|---|---|---|---|
| 0 (verbatim) | 0.44121% | 37 / 130 | 0.000 | 91.5% |
| **1.0** | **0.09329%** | **0 / 130** | 0.388 | 91.5% |
| 2.0 | 0.67232% | 51 / 130 | 0.775 | 31.5% |
| 2.58 | 3.11501% | 117 / 130 | **1.000** | **0%** |

Two independent instruments, pulled in opposite directions by one number. The eye agrees: the amplified
difference at frame 122 is a displaced contour for the verbatim skip, nearly black at gain 1.0, and pure noise
at gain 2.58; the frame-minus-temporal-mean view shows flat areas for the verbatim skip, partial speckle at
gain 1.0, and too much at gain 2.58.

And your architecture (motion-rejected 3-tap bilateral on LR, gate on the clean plate, grain re-added on all
frames) lands on the same curve: texture **0.9738** with 0% frozen - the half the bar cannot see is solved -
but worst share **0.79771%** and **42 of 204** frames over the bar, because the clean plate drops the noise
floor (0.87 -> 0.65), the gate engages more (66.2% against 63.4%), and its structural errors are no longer
masked by noise.

**We do not think this is a genuine policy trade-off. We think the knob is one gain applied to a difference
that contains two physically different components**, and that separating them should decouple the two
instruments:

- `d_t = LR_t - LR_ref` carries (a) **fresh noise** - temporally independent, exactly what the texture
  instrument wants restored, and which the network amplifies by `alpha` ~ 2.04 (our measured noise gain), so it
  must be added at ~`alpha`; and (b) **displacement / motion** - coherent, which for a *non-warped* skip is a
  structural error that must be corrected at gain 1 (adding it twice makes the error worse, not better).
- One scalar gain cannot satisfy both: at 1 the noise is under-restored (texture 0.388 ~ 1/alpha), at `alpha`
  the displacement is double-counted (share 3.1%). That is exactly what the table shows.
- The candidate correction is therefore two-term:
  `out = gated + g_n * (LR_t - warp(LR_ref)) + g_m * (warp(LR_ref) - LR_ref)`,
  with `g_n` the measured noise gain and `g_m` = 1 (or the displacement handled by warping the SR output
  instead of adding it linearly). Falsifier: at `g_n = alpha` and `g_m = 1`, does the texture reach ~1 **while**
  the share stays at or below the unit-gain share (0.09329%, 0/130)? If yes, the "choice" was an artefact of
  the single gain and the problem is a measurement. If the share climbs anyway, the two instruments are
  genuinely coupled and we are back to a policy decision - which we would then take to the product owner with
  this table.

A second, independent route to the same decoupling: re-synthesise the grain **with the network's own gain**
(measure `alpha` per block from a rendered pair, `SR(raw) - SR(clean)`, and inject at that amplitude) instead
of passing the source grain through at 1x. Our denoise+regrain prototype used unit gain and that is one reason
its pixel share is poor; the same prototype at `alpha` would test whether the regrain route can hold both.

## 4. Questions for the reviewer

1. Is the two-term split (`g_n` on the motion-compensated residual, `g_m` on the displacement) the right
   decomposition, or is there a known reason it fails - e.g. the network's response to noise and to
   displacement not being separable at the output at all?
2. How would you estimate the local noise gain `alpha` robustly? We have one estimate (median 2.04 from the
   HR/LR amplification of reused pairs) and a per-block version is not built; per-scene would be cheap, per-block
   needs a decision about the block size of the measurement, not of the decision.
3. Is there prior art where a **non-warped** frame reuse is corrected by an additive low-resolution residual
   (as opposed to warping the HR output)? Our reading says the VSR caching systems (NEMO, NeuroScaler) warp and
   add a residual, but they warp the *high*-resolution output; we are adding an LR-band term to a non-warped HR
   frame, which is why the spectrum question (LR band at 4K) keeps coming back.
4. If the two-term split works, the remaining question is the **certificate**, not the statistic: with a ranker
   at 0.99 and a rate-based bar, is conformal risk control the right frame given that the gate itself selects
   which frames would be audited?

## 5. Scope

One episode, one 8 s window per clip, one model (2x SRVGGNetCompact), one GPU, texture measured as a
differential against the same pipeline's ungated render (the reference-free variant does not separate, see §1.3),
no perceptual study. The clip the lane was originally closed on has no packet-level provenance and was lost with
`/tmp`; every number here is on `-c copy` windows cut from the BDRemux at recorded offsets.
