# Prior art, 2026-11-20: the reuse correction we proposed is NEMO's baseline, and the certificate is Palantír's DAG

Read after the second review request (`REVIEW-reuse-gate-2026-11-20.md`). Every claim below is from the cited
source, opened today; nothing here is inferred from our own measurements unless it says so.

## 1. The published systems, and what they actually do

**NEMO (MobiCom'20, Yeo et al.)** - an SR-integrated codec that applies the network only to *anchor* frames and
reconstructs the rest by "reusing-based SR": each inter-coded block is taken from the cached **SR** reference and
the codec's **residual** is added after **bilinear upscaling**, with the motion vector scaled by the SR ratio:

```
b_inter.SR = b_inter.ref.SR + interp(b_inter.res, scale)      (their Eq. 2)
```

Their ablation is the number that matters to us: excluding the residual costs **1.6 dB** PSNR, excluding the
motion vector costs **9.9 dB**. So the two terms are not optional, and the mechanism is exactly the two-term
correction this lane was about to "propose" - the proposal was a rediscovery. Anchor selection is greedy over a
**heavy offline profiling pass: 59.6 s for a 4 s segment**, i.e. the guarantee comes from measuring the content
before streaming, not from a threshold.
Sources: <https://dl.acm.org/doi/pdf/10.1145/3372224.3419185>, <https://github.com/kaist-ina/nemo>

**NeuroScaler (SIGCOMM'22, Yeo et al.)** - frame-level reuse at scale for live streaming; it models the SR error
propagation as a **linked list** (each frame's error = previous frame's error + residual), prioritises keyframes
and AltRefs, and returns to **heuristic thresholds** for real-time selection. Their own follow-up (Palantír)
criticises the linked-list model as too simplified because a frame can reference up to three others.
Sources: <https://dl.acm.org/doi/pdf/10.1145/3544216.3544218>,
<https://jaykim305.github.io/assets/pdf/neuroscaler-sigcomm22.pdf>

**Palantír (arXiv 2408.06152, 2024)** - patch-level scheduling with a **DAG over the coding reference graph**:

```
P.error ≈ P.TC + Σ_i W^i · P^i.error
```

where `P.TC` is the texture complexity of the patch's HR content or residual, and the weights are the fraction of
a patch's pixels referencing each predecessor patch. Their key move for us: **`TC` is estimated from the LR
video alone**, so the error of any anchor set can be predicted *without running the network*. They report a 20x
(60x) reduction in SR inference overhead while preserving 54.0-82.6% (32.8-64.0%) of the quality gain, and
selection latency under 5.7% of the end-to-end budget after vectorising the DAG as sparse matrix-matrix products.
Source: <https://arxiv.org/html/2408.06152>

**What none of them has**: temporally independent noise. NEMO and Palantír evaluate on clean 4K YouTube content
downsampled to 480p - there is no film grain to freeze. That is exactly our axis, and it is why their correction
(Eq. 2, residual at unit gain) works for them and leaves our texture at 0.388.

## 2. What this changes in our plan

1. **The two-term correction is not our idea** - it is NEMO's Eq. 2, with the motion vector present and the
   residual upscaled. Our version omitted the warp (a non-warped skip plus an LR residual) and that is precisely
   the term NEMO's ablation prices at 9.9 dB. The experiment to run is therefore theirs: **warp the cached SR by
   the scaled motion vector, then add the upscaled residual**, not "add a residual to a non-warped frame".
   Our measured curve (unit gain: 0/130 over the bar, texture 0.388) is the non-warped variant, which no
   published system uses.
2. **The certificate we lack is their DAG, not a better statistic.** Our own numbers already show the signature:
   the shipped statistic ranks damage at 0.844 when compared with `t-1` and at **0.991** when compared with the
   reference the gate actually served. Palantír's model says why in one line - the error of a frame is its
   texture complexity plus the *weighted sum of its references' errors*, so a pairwise difference is the first
   term only. Copying the DAG (block weights from the codec's motion vectors or our flow, `TC` from the LR
   residual's own down/up round trip) gives a predictor that can be evaluated **without running the network**,
   which is what a certificate needs to be cheap.
3. **The texture instrument has a published home.** SpEED-QA (Bampis et al., IEEE SPL 2017) computes **local
   entropic differencing** spatially and, for video, **on frame differences** - its temporal component is the
   published instrument for "the texture stopped moving". A public implementation exists.
   Sources: <https://github.com/christosbampis/SpEED-QA_release>, <https://vmafx.dev/metrics/speed_qa>
   Our home-made ratio (gated temporal diff over ungated) separates cleanly on our data (0.000 against 1.026) but
   needs an ungated render; SpEED-QA's temporal term is the candidate for a runtime-usable version, and the
   reviewer's rho1 is its cheap proxy (ours did not separate because the render's HF energy is static detail, not
   grain - an issue the entropic differencing is designed around).
4. **The audit-based certificate has a literature, and a limit.** Conformal risk control (Angelopoulos et al.,
   arXiv 2208.02814) bounds the **expected** risk of a selective predictor given an **exchangeable** calibration
   set; "non-exchangeable CRC" and "anytime-valid CRC" cover ordered data. Two caveats from the sources:
   the guarantee is in expectation over the calibration distribution (not per clip), and exchangeability is the
   assumption a gate violates when it chooses which frames to audit. Our arithmetic stands beside it: a
   zero-violation certificate needs ~388 audits for a 0.77% rate, while CRC can certify a *rate* with far fewer
   samples - as long as the audit is not selected by the gate.
   Sources: <https://arxiv.org/abs/2208.02814>, <https://openreview.net/forum?id=j511LaqEeP>,
   <https://arxiv.org/html/2602.04364v1>


## 2b. The practice line, found on the second sweep: anime Blu-ray, not synthetic

The first sweep covered VSR systems; this one went after practitioners. Two documents change the picture:

**EISAI (Chen & Zwicker, ECCV 2022)** - the closest published precedent, and it is on real material: a frame
removal step for 2D animation interpolation, features are a **linear regression over the mean AND the maximum L2
LAB colour difference** between consecutive frames, trained on a small hand-labelled set, removing **49.17 % of
8.24 M Blu-ray anime frames**. Their coefficients ship with the code. That is the "practice, not synthetic" paper
this lane was looking for, and it is a two-feature linear model - the baseline any decision statistic here has to
beat. Source: <https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770275.pdf>

**"Duplicate Frame Detection for 2D Anime" (nullcel, 2025 design study + measurement campaign)** - written for
exactly our consumer ("AI upscaling and frame-interpolation pipelines"), and its findings converge with ours
independently:

- **The task is two-axis, not binary**: `same_cel` (is this the same drawing - drives interpolation) versus
  `pixel_reusable` (can the *upscaled* output be reused, and where - drives skip-upscale). A static cel with snow
  moving is `same_cel = true` and `pixel_reusable = false`. That is our structure/texture split, arrived at from the
  other direction.
- **No scalar threshold separates the binding cases**: the hardest true duplicate (static cel, only particles
  moving) and the weakest true motion (a 0.2 px/frame pan) are separated by **1.18x** on block statistics, and the
  pan produces *less* block residual than the snow. "There is no magnitude threshold, on any scalar, that separates
  them."
- **Per-region, never per-frame**: every efficient-inference method it surveys (Deep Feature Flow, Clockwork
  Convnets, Skip-Convolutions, SkipVSR, MIA-VSR) makes the reuse decision per region.
- **Grain, stated plainly** (their 5.18): "Real film grain and sensor noise are temporally independent... For other
  codecs the `denoise -> detect/upscale -> re-grain` pipeline removes grain from the decision *and* fixes
  frozen-grain on reuse." With the two caveats we hit ourselves: temporal denoisers are motion-compensated
  (circular), spatial ones attack line art and banding.
- **No benchmark exists**: the four major anime video datasets (ATD-12K, Sakuga-42M, AVC/AnimeSR, RAL) each
  *delete* near-duplicates during construction, so the decision boundary is exactly what the data removes. That is
  the same wall we hit when the labelled clip turned out to be `cut: {mode: not_proved}`.
- Also free, and useful: AV1 signals a repeated frame with `show_existing_frame` and reproduces it bit-exactly
  *including film grain*, and decoding an AV1 source with grain synthesis disabled yields a grain-free master for
  nothing.
  Source: <https://github.com/NevermindNilas/nullcel/blob/main/anime-duplicate-frame-detection.md>

Community tooling confirms the practice: TheAnimeScripter ships `--dedup` (flownets) and `--smooth_dedup` for
frames held on twos/threes, Framegen detects on-twos frames on the GPU, and the Topaz forum has a thread on
removing duplicates *before* enhancing.

**What this does to the claim.** The architecture we were converging on (separate the plate, reuse on the plate,
re-grain on every frame) is already the community's recommendation, and the two-axis split is already named. What
is still ours is the *quantification*: the measured trade-off curve between the structural share and the texture
ratio on the same 130 reuses (0 / 130 with texture 0.388 at unit gain, 6 / 130 with 0.838 warped, 37 / 130 with
0.000 verbatim), the audit-certification arithmetic (388 audits for a 0.77 % rate), and the fact that the
published correction's texture gain comes from remapping the noise. None of the three documents has a number for
"how much texture is left" as a function of the correction.

## 3. The three experiments this ordering suggests

1. **The published correction, done as published**: warp the cached SR by the scaled motion vector, add the
   bilinear-upscaled residual, and measure our two instruments on the same 130 reuses. Expected: the structural
   class improves beyond our non-warped unit-gain result, the texture stays at ~1/alpha - i.e. the frozen-grain
   failure is *not* fixed by the published recipe, which is the finding worth writing down.
2. **The Palantír DAG as a predictor**: `err = TC + Σ w·err_ref` with `TC` from the LR round trip and weights
   from the block reference map. Test: does it rank the damaged reuses better than the pairwise statistic, and -
   more important - can it *certify* a configuration without rendering? Falsifier for the certificate: if the
   predicted error and the measured damage disagree on the two labelled clips, the DAG needs the grain term.
3. **SpEED-QA's temporal component** on the gated/ungated pair and on the gain curve, to see whether the published
   instrument reproduces our ratio's ordering (0.000 / 0.388 / 0.775 / 1.000) without needing an ungated render.

## 4. What we still cannot answer

- Whether the frozen-grain failure of these systems is reported anywhere: our searches found the VFX term
  (*frozen grain* / *grain pulsing*) and the codec literature on grain synthesis, but no evaluation of a VSR
  reuse system on grainy material. If it is unreported, that is the gap this lane has data for.
- Whether a certificate can be built on `TC + Σ w·err_ref` when `TC` must include the *noise* term, which is
  exactly what the pairwise statistics cannot predict without the network.

## 5. Sources opened today

| what | where |
|---|---|
| NEMO mechanism, Eq. 2, ablation 1.6/9.9 dB, 59.6 s per 4 s | <https://dl.acm.org/doi/pdf/10.1145/3372224.3419185> |
| NEMO code | <https://github.com/kaist-ina/nemo> |
| NeuroScaler, linked-list error model, heuristics at real time | <https://jaykim305.github.io/assets/pdf/neuroscaler-sigcomm22.pdf> |
| Palantír, DAG error model, TC from LR, 20x/60x, <5.7% latency | <https://arxiv.org/html/2408.06152> |
| SpEED-QA (entropic differencing, temporal component) | <https://github.com/christosbampis/SpEED-QA_release> |
| Conformal risk control | <https://arxiv.org/abs/2208.02814> |
| Non-exchangeable CRC | <https://openreview.net/forum?id=j511LaqEeP> |
