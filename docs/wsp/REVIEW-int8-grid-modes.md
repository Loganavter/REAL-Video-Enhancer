# int8 grid modes: why per-channel helps some models and hurts others

A write-up for an outside reader. Everything below is either a number from a record/probe named in place, or
explicitly marked as an unmeasured hypothesis. Nothing here is a claim about picture quality.

## 1. What the setting is

The body runs sixteen 3×3 convolutions at 64 channels in int8, with a Triton kernel. Between layers the
activations are re-quantised to 8-bit codes; the grid that maps a float activation to a code is a per-layer
choice, and it is the only knob in this document.

Two modes exist in the code (`backend/src/pytorch/Int8Body.py`, `_build`):

**per-tensor** (shipped default). One scalar grid per layer: `s_t = max_c s_in[c]`, where `s_in[c]` is the
per-channel activation maximum measured during calibration. Input codes are `x / s_t`. Because the per-channel
scale is *not* folded into the weights, the epilogue restores it: the int32 accumulator is multiplied by
`m[c] = round(s_in[c] / s_t * 2^20)` and shifted back, then dequantised by `w_sc * s_t`.

**per-channel** (`RVE_INT8_PER_CHANNEL=1`). The scale is folded into the weights: `w' = w * s_in[c]`, so each
input channel's codes use its own full ±127 range. The epilogue's multiplier is the identity (no multiply, no
shift), and the dequantisation is `w_sc` alone.

So the mode trades **input code resolution** (per-channel should be better: 64 tight boxes instead of one wide
one) against **weight quantisation** (per-channel quantises the *folded* weights, whose dynamic range across
inputs is wider).

## 2. What was measured

21 models from the user's collection that the pipeline can run (all Compact/SRVGG, 64 channels, 2x; 1x and 4x
where noted). Per model, both modes, one process, the same calibration frames, three held-out frames that the
calibration never saw. The statistic is the app's own start-up gate: percent of pixels more than 8 levels off
in any channel against the model's own fp16 output, worst of the three frames.

Records: `docs/wsp/verified/records/model-fit.jsonl` (MF001-MF105, `grid: per_tensor|per_channel` beside each).
Raw: `logs/model-fit/all.json`, `logs/model-fit/all-perchannel.json`.

The split is not marginal:

| model | per-tensor | per-channel | |
|---|---|---|---|
| `1x-Epsilon-one-compact` | 1.5394% | 0.0979% | 16x better |
| `2x_GarfieldJr_net_g` | 0.8763% | 0.0436% | 20x better |
| `2x_Ani4K_Compact` | 0.3021% | 0.0140% | 22x better |
| `realesr-animevideov3` (4x) | 0.1410% | 0.0288% | 5x better |
| `2x_AniSD_AC_G6i2a` | 0.0982% | 0.0021% | 47x better |
| `2xHFA2k_LUDVAE_compact` | 0.0761% | **0.2789%** | **3.7x worse** |
| `2xken-v1-evA-01` | 0.0664% | **0.1759%** | **2.6x worse** |
| `2x-Futsuu-Anime` (production) | 0.0114% | **0.0309%** | **2.7x worse** |

(Full table in `docs/MODEL-FIT.md`.)

## 3. The predictor we filed the row with, and its refutation

The hypothesis was that the calibration grid's dynamic range — `max / median` of the per-channel activation
maxima — would predict which models improve. It does not:

```
rescued models' spreads:  2.71  3.09  4.03  4.20  4.34  4.74  4.96  4.99  5.15  6.33  6.66  6.93
damaged models' spreads:  4.25  4.31  4.42  4.48  4.55
```

The ranges overlap completely. A model with spread 4.5 improved and a model with spread 4.48 got worse. So
there is no threshold on the calibration spread, and the "obvious" story — wide grid, therefore per-channel
helps — is wrong.

## 4. The two first-order terms, measured

Both terms were measured directly on the same six models (three rescued, three damaged), same frames, one
process: `scripts/int8_grid_modes_probe.py`, raw `logs/grid-modes/probe.json`.

**Input quantisation error** — `||a - round(a/g)·g|| / ||a||` over all activations of all 16 layers, on held-out
frames, against each mode's own grid `g`.

**Weight quantisation error** — `||w - dequant(q(w))|| / ||w||`, on the folded weights for per-channel and the
raw weights for per-tensor, i.e. in the domain the kernel multiplies codes in.

| model | input err, pt → pc | weight err, pt → pc | outcome |
|---|---|---|---|
| Futsuu | 0.149 → 0.046 (**3.2x better**) | 0.0208 → 0.0292 (1.40x worse) | **worse** |
| ken-v1 | 0.135 → 0.041 (3.3x better) | 0.0204 → 0.0269 (1.32x worse) | **worse** |
| LUDVAE | 0.088 → 0.033 (2.7x better) | 0.0215 → 0.0258 (1.20x worse) | **worse** |
| Garfield | 0.155 → 0.042 (3.7x better) | 0.0200 → 0.0256 (1.28x worse) | **better (20x)** |
| AniSD | 0.228 → 0.059 (3.9x better) | 0.0162 → 0.0204 (1.26x worse) | **better (47x)** |
| AniScale | 0.160 → 0.046 (3.5x better) | 0.0201 → 0.0271 (1.35x worse) | slightly worse (both ~0) |

This is the part we cannot explain yet, and it is the question for the reader:

> **Both first-order terms move in the same direction for every model — input error 2.7–3.9x better, weight
> error 1.2–1.4x worse — yet the picture outcome flips sign, and it does not track the size of either term.**
> Futsuu and Garfield have nearly identical numbers (3.2x/1.40x versus 3.7x/1.28x) and opposite outcomes.
> LUDVAE has the *smallest* weight cost and the *smallest* input gain, and is the most damaged.

## 5. Hypotheses, with status

**(a) Weight folding costs precision.** CONFIRMED as a systematic cost: per-channel is 1.20–1.40x worse on
relative weight error for every model measured. Insufficient as an explanation — the cost is similar across
models while the outcome is not.

**(b) Headroom / clipping on held-out frames.** REFUTED for these models: the share of activations whose code
exceeds ±127 is 0.00000–0.00001% in both modes, on all three held-out frames. A tight per-channel box is not
clipping here, so the classic resolution-vs-headroom trade is not what decides the sign.

**(c) Channel-wise weight collapse.** NOT MEASURED — this is the measurement we would run next. The relative
weight error above is an aggregate. Folding `w * s_in[c]` and then normalising per *output* channel means a
low-range input channel contributes tiny folded weights, which can quantise to one or two integer levels:
a per-channel failure, invisible in an aggregate L2. The discriminating measurement is per *input* channel:
the distribution of `||w[:, c] - dequant(q(w'[:, c]))|| / ||w[:, c]||` in both modes, and the count of channels
whose relative error exceeds, say, 0.5. If the damaged models have a handful of destroyed channels and the
rescued ones do not, that is the mechanism; if both look alike, it is not.

**(d) The epilogue's extra rounding.** NOT MEASURED. Per-tensor mode multiplies the accumulator by `m[c]` and
shifts (`(q*m + half) >> 20`), which is an extra truncation that per-channel does not perform. Whether that
helps (smoothing) or hurts is a matter for measurement; it is also the only structural difference in the
accumulator path, so a reader who suspects it is not being fanciful.

## 6. What the pipeline does meanwhile

Not a threshold rule, and not a global default. The implementable version is: **build both grids, evaluate both
on a held-out slice of the calibration frames, keep the better one**, and let the start-up gate certify on a
disjoint slice — otherwise the gate would pass by construction. Cost: one extra build (~2 s) and one extra
frame pass per render; benefit: the six models above stop looking unquantisable when the grid was the problem,
and the three damaged ones keep the shipped grid.

## 7. What is not claimed

- These are three held-out frames of one clip (BD Violet, 1080p, 10-bit source). The acceptance statistic of
  this project is a worst frame of a 494-frame pair at 0.1%; **none of the models here has one**, and the 1%
  start-up gate is a smoke test, not a verdict on a picture.
- The fp16 reference is eager PyTorch, not the app's TensorRT engine. That is right for comparing the two
  grid modes (the reference is identical in both arms) and wrong for quoting a product ratio.
- The statistic is thresholded (pixels >8 levels), which is deliberately nonlinear — a mode can improve a mean
  error and still move more pixels across a threshold. That is a property of the criterion, not a defect, but
  it is why "which error term is bigger" does not settle the question.

## 8. Reproduce

```bash
# both modes, all compatible models, three held-out frames each
RVE_INT8_PER_CHANNEL=0 <install-python> scripts/model_fit_probe.py \
    --out-file docs/wsp/verified/logs/model-fit/all.json
RVE_INT8_PER_CHANNEL=1 <install-python> scripts/model_fit_probe.py \
    --out-file docs/wsp/verified/logs/model-fit/all-perchannel.json
PYTHONPATH=scripts python3 scripts/model_fit_records.py
PYTHONPATH=scripts python3 scripts/model_fit_report.py

# the two first-order terms, per layer, both modes, one process
flock -n /var/tmp/opencode/gpu.lock <install-python> scripts/int8_grid_modes_probe.py \
    --out-file docs/wsp/verified/logs/grid-modes/probe.json
```
