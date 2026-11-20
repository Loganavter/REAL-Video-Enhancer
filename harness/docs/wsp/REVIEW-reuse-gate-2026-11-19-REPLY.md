# Reply to the review of 2026-11-19: what we accepted, what we tested, what changed

Written for the same outside reader, self-contained. The reply is ordered the way the review was: the framing
answer first, then the three suspected artefacts, then the proposals we could test within a day.

## 1. The main answer was right, and we are acting on it

**Verbatim skip on noisy material is unfixable** - agreed, and our own numbers already said it: of 93 reuses
that pass the per-pixel bar, 82 are frozen in the flat areas (block temporal |diff| 0.000 against 1.026 on the
same frames rendered fresh). No detector can rescue an action that re-emits the reference's noise. The two
contracts (A: transparent drop-in for `SR(raw)`, B: a pipeline that owns the final look) are now written into
the problem statement, and under contract A the answer is "refuse on noisy material", which is what ships.

## 2. The three suspected artefacts: two confirmed, and one of them changed a conclusion

**(a) The `t-1` approximation was a real bug, and it withdrew a claim.** Every offline AUC compared a frame with
`t-1` while the gate serves the last KEPT frame, up to two back. Recomputed against the reference the gate
actually served (the per-frame depth is logged), the shipped statistic reads:

| statistic | clip | vs t-1 | **vs the served reference** |
|---|---|---|---|
| block mean of \|diff\| (shipped) | 1040 s | 0.844 | **0.991** |
| block mean of \|diff\| (shipped) | 960 s, held out | 0.994 | **0.988** |
| motion-compensated residual mean | 1040 s | 0.906 | 0.972 |
| motion-compensated residual mean | 960 s, held out | 0.489 | **0.488** |

So **"the failure class is non-stationary" was largely our own artefact**: against the served reference the
shipped statistic ranks damaged reuses at 0.99 on both clips. That claim is withdrawn in the record.

**(b) The flow explanation for the DIS collapse is confirmed.** The seven damaged reuses at 960 s sit at a mean
flow magnitude of **2.26** against **0.48** on the 116 clean ones - coherent motion that warping removes. On
1040 s it is the opposite: **0.057** on the 37 damaged against **0.266** on the clean - local redraw at almost
no global motion. Two mechanisms, and a warp-blind statistic is blind exactly where the skip is blind.

**(c) The MAD-normalisation failure** is explained as proposed (no floor on local sigma, share punishes a soft
background more than a sharp contour), and we do not intend to revisit that form.

## 3. What we tested from the proposals, and what happened

**Accounting for `||u||` in the decision.** Tested as four variants over the same 130/123 reused frames:

| statistic | AUC, 1040 s | AUC, 960 s |
|---|---|---|
| block mean (shipped) | 0.844 | 0.994 |
| block mean x (1 + 2\|u\|) | 0.735 | 0.961 |
| max(block mean, 3\|u\|) | 0.844 | 0.929 |
| flow magnitude alone | 0.361 | 0.626 |

and as a two-branch rule with thresholds calibrated on one clip and applied to the other: `OR(block, mctf)` and
`AND(block, mctf)` both caught 3/7 with 5/116 false alarms on 960 s and 27/37 with 10-12/93 on 1040 s - never
better than the block branch alone. Any motion term trades the low-flow failure class against the coherent-motion
one; a linear combination cannot hold both. (Those numbers are the `t-1` ones; the correction in §2a lifts the
block branch to 0.99 but does not make the motion terms better.)

**rho1 of the high-frequency residual as a reference-free texture criterion.** Tested on the 1040 s pair, on
flat static blocks selected from the LR source, with and without removing the temporal mean:

| material | rho1, raw | rho1, mean removed |
|---|---|---|
| reused (frozen) | 1.000 | 1.000 |
| rendered (fresh) | 0.938 | 0.918 |

It **does not separate** here: the output's high-frequency energy is dominated by static line detail, not by
grain, so a fresh frame reads 0.92-0.94 rather than ~0. Our differential ratio (gated temporal |diff| over the
same in the ungated render) separates cleanly - 0.000 against 1.026 - precisely because it cancels that static
component. So we agree rho1 is the right *family* (and we want a runtime-usable criterion), but on this material
it has to be computed on the **grain residual after denoising**, not on the render's HF.

**The residual-reuse algebra** (a residual leaves `(alpha - 1) * n_ref` frozen, alpha being the net's noise
gain, median 2.04 here) - accepted as the reason candidate 3 cannot work without denoising, and it is next on
the list to measure rather than argue: build `SR_ref + bilinear(LR_t - LR_ref)` on the 130 reused frames and
measure both the per-pixel share and the freeze ratio. If the freeze ratio stays high, the candidate is dead
for the price of an afternoon.

## 4. Where the problem now stands (after the correction)

1. **The ranker is not the missing piece.** With the served reference, the shipped statistic predicts the
   damaged reuses at AUC 0.99 on both labelled clips. What is missing is a **certification rule** that turns a
   ranker into a threshold: today it passes 37 of 130 reuses, and 82 of those it passes are frozen.
2. **The texture criterion is the missing half**, and the review's point that a per-pixel bar cannot express it
   is now measured, not asserted.
3. **The verbatim-skip objection stands** and is what we will tell the product owner: on noisy material the gate
   must refuse, which is exactly where reuse would pay most.

## 5. What we will run next (and what would falsify each)

- **Denoise-then-reuse prototype** on the same two clips: a motion-rejected 3-tap temporal filter on LR, the gate
  on the clean plate, and the source grain re-added on **all** frames (rendered and reused) as
  `HR_clean + beta * up(LR_t - LR_clean_t)`. Falsifier: the plain-plate share against the clean ungated render
  stays inside 0.1% while the freeze ratio falls to the fresh level, and the added grain does not make the
  per-pixel share against the *raw* reference worse than the current bar allows (the perception-distortion
  conflict the review names - if it does, the bar has to be redefined against the clean plate, which is a
  policy decision, not a measurement).
- **Certification on non-stationary content**: we have two labelled clips and no more, so the question is which
  framework can certify with two. The review's conformal-risk-control suggestion is the candidate we will read
  first; the open objection is that the gate *selects* which frames to audit (non-i.i.d. selection), and the
  audit costs the very speed-up it certifies.
- **Grain model at LR or HR**: the review's point 3.3 (AR model estimated at 1080p, applied at 4K) decides whether
  the re-synthesised grain is the right size; we will measure the PSD of the source grain at LR and of the
  re-added grain at HR before choosing.

## 6. Two questions we still cannot answer ourselves

1. Given a ranker at AUC 0.99 and two labelled clips, **what is the right way to certify a threshold** for
   non-stationary content without a per-shot label? Per-shot recalibration is what the gate's tuner does today,
   and it is what failed; conformal control needs the audit it pays for.
2. **What is the accepted number for texture freshness** - is there a published threshold on any of the
   SpEED-QA / ST-RRED family that was validated against viewers on *synthesised* grain, or is the field still at
   "PSD + subjective viewing" as the ITU supplement suggests?

Numbers cited here live in `harness/docs/wsp/verified/reuse-decision-statistic.txt` (records RU001-RU014) and
`gate-bd.txt`; the instruments are `harness/scripts/reuse_stats.py`, `harness/scripts/reuse_texture.py`, `harness/scripts/reuse_rho1.py`.
