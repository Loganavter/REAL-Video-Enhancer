# Prior art, 2026-11-20 (second sweep): the failure class left after the correction is the one the graphics literature says it cannot fix

Read after the geometry axis closed (`reuse-mv-reference.txt`). The question was narrow: the correction's remaining
failing pixels are 4K detail that has to *move*, which a 1080p residual cannot express — has anyone solved that, and
with what mechanism? Four sources were opened in full (the TAA survey and the k-DOP paper as PDF, RLSP and RelayVSR
as HTML/PDF), plus one transfer test run on our own four failing frames.

## 1. The graphics line: output reuse with a clamp, and its documented limit

**A Survey of Temporal Antialiasing Techniques** (Yang, Liu, Salvi, CGF 39(2), 2020) is the reference text for
"reuse the previous output and correct it". Its pipeline is exactly ours minus the network: reproject the history
buffer with motion vectors, **validate** the history, **accumulate** it with the current sample, clamp.

- History rectification approximates the colour gamut of the current 3×3 neighbourhood with an AABB and clips or
  clamps the history colour into it, in RGB or YCoCg (`Karis 2014`, `Lottes 2011`).
- **Variance clipping** (`Salvi 2016`) replaces the AABB extent with `Cmin = µ − γσ`, `Cmax = µ + γσ` over the local
  neighbourhood, `γ` typically 0.75-1.25, to stop dark/bright outliers inflating the box.
- The limitation is stated in the survey itself, in the section on ghosting: *"the effect of history rectification
  techniques like history clipping or clamping is often compromised near high-contrast object boundaries"* — dense
  high-contrast edges make the bounding box large, and the invalid history passes through it (their Figure 10). The
  same section names the deeper case: an undersampled current frame cannot reliably tell valid history from stale.
- The cost is the scale to beat: the TAA pass is **0.14 / 0.24 / 0.52 ms** at 1920×1080 / 2560×1440 / 3840×2160 on
  an RTX 2080 Ti (their Table 1). Our correction is 2.428 ms, i.e. this class of technique is 5x cheaper, and still
  does not eliminate the artifact.
- The survey's own forward look: learn the rectification instead of engineering it (`Salvi 2017` replaces variance
  clipping with a recurrent convolutional autoencoder). That is a small network per frame, not a pixel rule.

Source: <http://behindthepixels.io/assets/files/TemporalAA.pdf>

**k-DOP Clipping: Robust Ghosting Mitigation in Temporal Antialiasing** (Ikkala et al., SIGGRAPH Asia 2024 TC) is
the newest attempt at the same failure. It replaces the AABB with a 16- or 32-DOP: extents over k/2 arbitrary axes,
a tighter approximation of the convex hull of the neighbourhood colours.

- Ghosting (FLIP, lower is better) on "Grass": RGB AABB 0.0499, YCoCg 0.0538, **32-DOP+Var 0.0336**; shimmer rises
  0.0346 → 0.0568, the usual trade.
- Cost: **+0.06-0.20 ms** over the AABB methods (0.45-0.66 ms total at 1080p, GTX 1080 Ti).
- And its own limitation section concedes the case we measured: *"Neighborhood clipping on its own is not enough to
  completely remove all ghosting; even if a convex hull were used, the ghosting color may lie inside the
  neighborhood's color gamut and go undetected."*

Source: <https://webpages.tuni.fi/vga/publications/k_DOP_clipping_Ikkala2024.pdf>

That sentence is our failure, stated in advance: the stale 4K detail is *inside* what the 1080p neighbourhood allows,
so no colour-space clamp can see it. Section 5 measures exactly that on our frames.

## 2. The VSR line: nobody skips the network, they make the per-frame network cheap

**RLSP** (Fuoli, Gu, Timofte, ICCVW 2019) is the efficient-recurrence reference. The network works in **LR space**
and expands to HR only at the last layer (pixel-shuffle), feeding the previous HR output back **shuffled down to LR**
and concatenated with the hidden state; 7 layers, 3×3, 48-256 filters, 15-92 ms for a 1080p frame.

- Its ablation prices the two kinds of reuse we have been arguing about: adjacent frames +1.3 dB, **previous output
  feedback +0.23 dB**, **hidden state +0.55 dB**. The output is the weakest of the three, and the hidden state - the
  part we cannot have without retraining - is the strongest.
- Crucially, **the network still runs on every frame**. Reuse here means "carry state", not "skip the pass".

Source: <https://arxiv.org/html/1909.08080v1>

**RelayVSR** (Wang et al., arXiv 2609.37850, 2026) is the current version of the same answer for generative VSR: a
large model produces reference latents for **sparse keyframes only**, and a **trained lightweight VSR network**
(Dual-Memory Video Transformer) super-resolves every frame from the LR video plus those latents. 29.29 FPS at 1080p
on one A100, 13.82 GB, at a 15-frame keyframe interval.

Its interval sweep is the number that matters to us — what reuse across frames costs *even with a per-frame network
trained for it*:

| keyframe interval | LPIPS (lower better) | DOVER (higher better) |
|---|---|---|
| 5 | 0.2115 | 0.5530 |
| 15 | 0.2171 | 0.5448 |
| 25 | **0.2370** | **0.5140** |

So the cost of reuse is monotone in the interval and does not vanish with training; it is *managed*, by keeping the
per-frame path and by paying for it. Their ablation also shows what part matters: reference memory alone (no rolling
video memory) reaches MUSIQ 63.65, and video memory alone (no references) 56.80 — the temporal state carries the
quality, exactly like RLSP's hidden state.

Sources: <https://arxiv.org/pdf/2609.37850>, <https://github.com/kopperx/RelayVSR>

## 3. The transfer test: variance clipping on our four failing frames

The one mechanism from Section 1 that needs no training is the clamp, so it was measured on our population. Target:
the split correction at the served reference (`reuse_split.py`), the four frames over the bar of the 960 s cut
(18, 21, 22, 56), and four frames that pass (17, 54, 59, 60). `L` is the current frame's own picture at 4K
(nearest-2× of the 1080p luma), `µ` and `σ` from a box filter over it, and the output clamped to `µ ± (σ + m)`,
which is variance clipping with `γ = 1` plus a margin `m` to let the network's sharpening overshoot survive.

| frame | raw | 3×3, m=8 | 3×3, m=16 | 3×3, m=32 | 7×7, m=32 |
|---|---|---|---|---|---|
| 18 | 0.1044% | 0.1138% | 0.1106% | 0.1073% | 0.1049% |
| 21 | 0.1082% | 0.1137% | 0.1132% | 0.1124% | 0.1114% |
| 22 | 0.1924% | 0.1943% | 0.1942% | 0.1940% | 0.1940% |
| 56 | 0.1659% | 0.1888% | 0.1662% | 0.1659% | 0.1659% |
| 17 (passes) | 0.0867% | 0.0832% | 0.0874% | 0.0835% | 0.0842% |
| 54 (passes) | 0.0115% | 0.0115% | 0.0115% | 0.0115% | 0.0115% |
| 59 (passes) | 0.0066% | **0.3008%** | 0.1297% | 0.0095% | 0.0114% |
| 60 (passes) | 0.0082% | **0.3056%** | 0.1347% | 0.0113% | 0.0133% |

**It does not help where the bar is exceeded, and it destroys frames that pass.** The four failures move by at most
0.003% in either direction; frames 59 and 60 go from 0.007-0.008% to 0.30% because the network's edge overshoot sits
outside the 1080p neighbourhood and is clipped away. Raising the margin to 32 recovers them to 0.0095-0.0133% —
still 1.4-1.6x worse than doing nothing — and by then the clamp does nothing at all on the failures. This is the
survey's Fig. 10 and the k-DOP paper's limitation, reproduced on our material and on our statistic.

## 4. What this says about the lane

- **Not solved, and not for lack of trying.** The graphics field's best current answer (32-DOP + variance clipping,
  2024) still ghosts, and says why: the stale colour is inside the neighbourhood gamut. Our failing frames are that
  case by construction — frame 56's LR change never exceeds 17 levels and never puts a pixel above 16, yet 0.15% of
  the 4K output moves by more than 8.
- **The VSR field's answer is architectural, and it is what we already do when we do not reuse:** run a network on
  every frame, and carry the detail in the *state* rather than in the output. Both efficient systems that do reuse
  (RLSP 2019, RelayVSR 2026) keep the per-frame pass and pay for it; RelayVSR's own sweep shows quality falling
  monotonically as the reuse interval grows.
- **No free lever was found.** For this repo the options are: ship the measured state (0/130 + 1.001, 3/124 + 1.021,
  2.428 ms), render the frames a cheap test distrusts (no such test separated the three — measured), or move the
  reuse into the feature domain, which is a retraining project, not a perf change.

## Sources

- Yang, Liu, Salvi, *A Survey of Temporal Antialiasing Techniques*, CGF 39(2), 2020 —
  <http://behindthepixels.io/assets/files/TemporalAA.pdf>
- Ikkala, Lauttia, Jääskeläinen, Mäkitalo, *k-DOP Clipping: Robust Ghosting Mitigation in Temporal Antialiasing*,
  SIGGRAPH Asia 2024 Technical Communications — <https://webpages.tuni.fi/vga/publications/k_DOP_clipping_Ikkala2024.pdf>
- Fuoli, Gu, Timofte, *Efficient Video Super-Resolution through Recurrent Latent Space Propagation*, ICCVW 2019 —
  <https://arxiv.org/abs/1909.08080>
- Wang et al., *RelayVSR: Large-Small Model Collaboration for Efficient Real-World Video Super-Resolution*,
  arXiv 2609.37850, 2026 — <https://arxiv.org/abs/2609.37850>
