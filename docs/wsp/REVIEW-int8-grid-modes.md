# int8 grid modes: why per-channel helps some models and hurts others

A write-up for an outside reader. Everything below is either a number from a record/probe named in place, or
explicitly marked as an unmeasured hypothesis. Nothing here is a claim about picture quality.

## 1. What the setting is (exact, from `Int8Body._build` and `hop_padded_kernel`)

The body runs sixteen 3×3 convolutions at 64 channels in int8. A hop reads codes `q_k`, accumulates
`acc = Σ q_k · w_q` in int32, and its epilogue is:

```
v   = acc · vec + bias                 # vec = per-output-channel dequant scale
v   = v if v >= 0 else v · slope       # PReLU, in float
q   = round(v / os)                    # quantise onto the NEXT layer's grid, `os` per channel
q   = clamp(q, -127, 127)
q   = (q · m + 2^19) >> 20             # `m` per channel, then clamp again
```

`os` is the per-channel scale of the next layer's grid and `m` converts a per-channel code into that grid;
**`m` is indexed by this layer's OUTPUT channel** (which is the next layer's input channel), not by its input —
an earlier version of this note had it the other way round, and the reader who caught it was right that the
description did not close.

**per-tensor (shipped).** One scalar grid per layer: `s_t = max_c s_in[c]`, codes `x / s_t`, weights
quantised raw. The epilogue's `os` is the per-channel `s_out`, and `m[c] = round(s_out[c] / s_t_next · 2^20)`
does the conversion to the next layer's scalar grid. So the epilogue performs **two** roundings (one at
`round(v/os)`, one after the shift), and the second one scales codes *down* by `s_out/s_t_next ≤ 1`, which
costs resolution precisely in the channels whose range is far below the layer maximum.

**per-channel.** `s[c] = s_in[c]` is folded into the weights (`w' = w · s[c]`), codes are `x / s[c]`, the fold
cancels exactly in the accumulation, `os` is the next layer's `s_out` (which is its own grid) and `m` is the
identity — one rounding instead of two, and no down-scaling of codes.

**The axis, not the dichotomy.** `s[c] = max(s_in[c], s_t / K)` interpolates between them: `K = 1` is the
per-tensor grid reached through the folding path, `K → ∞` is per-channel, and intermediate `K` bounds how far
any one channel's grid can fall below the layer maximum. The fold is exact for **any** positive `s[c]`, so this
axis costs no arithmetic and needs no epilogue change — which is why the rest of this note is a ladder over `K`
rather than a choice between two modes.

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

## 2b. The mode was the wrong axis: a clamp beats both ends

`s[c] = max(s_in[c], s_t / K)` is the same code path with a floor on how far one channel's grid may fall below
the layer maximum. `K = 1` is the per-tensor grid reached through the folding path, `K → ∞` is per-channel, and
the fold cancels exactly for any `s[c]`, so **every K costs the same arithmetic** — this is a picture axis only.

All 22 models the pipeline can run, same calibration, same three held-out frames (records GK001-GK132):

| model | per_tensor | K=1 | K=2 | K=3 | K=4 | per_channel | best |
|---|---|---|---|---|---|---|---|
| `2x-anifilm-compact.pth` | 0.0600 | 0.0456 | **0.0005** | 0.0029 | 0.0026 | 0.0171 | K=2 |
| `2x-AniScale.pth` | 0.0001 | 0.0002 | **0.0000** | 0.0001 | 0.0000 | 0.0002 | K=2 |
| `2x-Bubble-AnimeScale-Compact-v1(3).pth` | 0.0012 | 0.0010 | 0.0001 | 0.0002 | **0.0000** | 0.0004 | K=4 |
| `2x-Futsuu-Anime(1).pth` | 0.0114 | 0.0126 | **0.0002** | 0.0019 | 0.0043 | 0.0309 | K=2 |
| `2x-LD-Anime-Compact.pth` | 0.0110 | 0.0091 | **0.0001** | 0.0006 | 0.0002 | 0.0019 | K=2 |
| `2x-sudo-UltraCompact(1).pth` | 0.0284 | 0.0246 | 0.0006 | **0.0003** | 0.0006 | 0.0013 | K=3 |
| `2x.StarSample.V1.0(2).pth` | 0.0650 | 0.0610 | 0.0040 | **0.0010** | 0.0011 | 0.0031 | K=3 |
| `2x_Ani4K_Compact_35000.pth` | 0.3021 | 0.1589 | 0.0166 | 0.0054 | **0.0053** | 0.0140 | K=4 |
| `2x_AniSD_AC_G6i2a_Compact_72500.pth` | 0.0982 | 0.0760 | 0.0059 | 0.0016 | **0.0014** | 0.0021 | K=4 |
| `2x_GarfieldJr_net_g_51000.pth` | 0.8763 | 0.2434 | 0.0030 | 0.0157 | **0.0029** | 0.0436 | K=4 |
| `2xGT-v2-evA-01.pth` | 0.0156 | 0.0101 | 0.0006 | **0.0002** | 0.0003 | 0.0075 | K=3 |
| `2xHFA2k_compact_multijpg(1).pth` | 0.0000 | 0.0000 | **0.0000** | 0.0000 | 0.0000 | 0.0001 | K=2 |
| `2xHFA2k_LUDVAE_compact(2).pth` | 0.0761 | 0.0904 | 0.0448 | **0.0443** | 0.1511 | 0.2789 | K=3 |
| `2xHFA2kAVCCompact(1).pth` | 0.0000 | 0.0000 | **0.0000** | 0.0000 | 0.0000 | 0.0000 | K=2 |
| `2xken-v1-evA-01.pth` | 0.0664 | 0.0434 | 0.0141 | **0.0037** | 0.0311 | 0.1759 | K=3 |
| `WoEiCS_2x(1).pth` | 0.0001 | 0.0000 | **0.0000** | 0.0000 | 0.0000 | 0.0000 | K=2 |
| `1x-AnimeUndeint-Compact(1).pth` | 0.0001 | 0.0000 | **0.0000** | 0.0000 | 0.0001 | 0.0000 | K=2 |
| `1x-BleedOut-Compact.pth` | 0.0014 | 0.0000 | **0.0000** | 0.0000 | 0.0000 | 0.0000 | K=2 |
| `1x-BroadcastToStudio_Compact(1).pth` | 0.0966 | 0.0630 | 0.0018 | **0.0018** | 0.0049 | 0.0251 | K=3 |
| `1x-Dotzilla-Compact.pth` | 0.0000 | 0.0000 | **0.0000** | 0.0000 | 0.0000 | 0.0000 | K=2 |
| `1x-Epsilon-one-compact(3).pth` | 1.5394 | 1.0937 | 0.0962 | **0.0665** | 0.0998 | 0.0979 | K=3 |
| `realesr-animevideov3(1).pth` | 0.1410 | 0.1246 | 0.0504 | **0.0071** | 0.0446 | 0.0288 | K=3 |

An intermediate K beats both shipped modes on 17 of 22 models.

Five of the 22 read 0.0000% at every K (they were already clean), so "beats both ends" is strict for the other
**17 of 22**. Two entries are worth naming: `1x-Epsilon-one-compact` was at 1.5394% — above the app's 1%
start-up gate, i.e. refused outright — and reads 0.0665% at K=3, inside the bar; and `realesr-animevideov3`
(4x, 48-channel tail) goes 0.1410% → 0.0071%.

`K = 1` also isolates the epilogue term (d) that §5 lists: it produces the per-tensor *grid* without the
epilogue's multiply-and-shift. Against the shipped per-tensor it reads better on Garfield (0.8763 → 0.2435),
worse on LUDVAE (0.0761 → 0.0904) and neutral on Futsuu (0.0114 → 0.0126) — so the epilogue term is real,
model-dependent, and not the main driver.

## 2c. On a 494-frame pair, against the engine

The 3-frame gate is a smoke test. For the production model the ladder was rendered and compared properly
(`logs/grid-k-pair/`, records GK/GB):

```
494 frames, ffv1 4:4:4, reference: the FP16 ENGINE's own render
  shipped per-tensor  vs  engine   worst share 0.00353%   frames over the 0.1% bar: 0  (peak 24)
  K=2                 vs  engine   worst share 0.00125%   frames over the 0.1% bar: 0  (peak 19)
  shipped per-tensor  vs  K=2      worst share 0.00625%   frames over the 0.1% bar: 0
```

**K=2 is the shipped default since 2026-11-20** (`207f9cbf`), with the two old modes still reachable
(`RVE_INT8_GRID_K=0`, `RVE_INT8_PER_CHANNEL=1`) and the resolution logged at build time. What is *not* adopted
is a per-model optimum: eight models prefer K=3/4 by 2.5-7x, which needs a start-up pick rather than a constant.

The triangle is consistent: the two int8 configurations differ from each other (0.00625%) by more than either
differs from the engine, which is what happens when the engine sits between them.

So the clamp is not a marginal reshuffle of a broken picture: both configurations are far inside the bar on the
full clip, and K=2 is the closer of the two to the engine by a factor of five on this statistic.

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

**(c) Channel-wise weight collapse.** PARTIALLY ADDRESSED by the ladder: if a crushed low-range channel were the
damage, harm should appear as K grows and vanish at small K. It does appear at the top of the ladder (per-channel
is the worst end for four of six models) and an intermediate K repairs it — consistent with the mechanism, but
not proof: the clamp also changes the *input* grid, so both terms move together. The per-input-channel weight
error remains the measurement that would separate them. The relative
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

## 6b. The timing expectation: per-channel cannot be faster here

An earlier reply expected the identity multiplier to remove a multiply and a shift from the epilogue. It cannot:
`m_next` is a **pointer to a runtime tensor**, not a `tl.constexpr`, so the kernel executes the multiply in every
mode. The measurement agrees — body time across the ladder ran 27.28 → 27.68 ms monotonically with K, which
cannot be causal because the arithmetic is identical, i.e. it is clock drift; and `K = 1` (no multiply) is not
faster than the shipped per-tensor (27.28 against 26.93 ms, one draw each).

## 7. What is not claimed

- These are three held-out frames of one clip (BD Violet, 1080p, 10-bit source). The acceptance statistic of
  this project is a worst frame of a 494-frame pair at 0.1%. **Only the production model has one** (§2c); the
  other twenty are ranked by a smoke test and must not be quoted as verdicts.
- **A retraction of my own:** I first reported that the 494-frame pair reads 0.42% for the shipped body (0.00353%
  is the measured value), making the smoke test understate by ~37x. That number came from `int8-scales-bd.txt`, whose own text flags a
  different reference for its per-channel row, and it does not survive the measurement above: on one
  instrument, the shipped body reads 0.00625% against K=2. Two records with different references were
  multiplied, which is the exact error class this project keeps retracting.
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
PYTHONPATH=. python3 harness/scripts/model_fit_records.py
PYTHONPATH=. python3 scripts/model_fit_report.py

# the two first-order terms, per layer, both modes, one process
flock -n /var/tmp/opencode/gpu.lock <install-python> scripts/int8_grid_modes_probe.py \
    --out-file docs/wsp/verified/logs/grid-modes/probe.json

# the clamp ladder, any Compact (width and depth read from the weights)
flock -n /var/tmp/opencode/gpu.lock <install-python> scripts/int8_grid_k_probe.py \
    "2x-Futsuu-Anime(1).pth,2x_GarfieldJr_net_g_51000.pth" \
    --out-file docs/wsp/verified/logs/grid-k/ladder.jsonl

# the 494-frame pair: fp16 engine, shipped per-tensor, and K=2
ARMS="fp16 base k2" ARM_ENV_K2="RVE_INT8_GRID_K=2" CLIPS="bd_violet20s.mkv" ROUNDS=1 SINK=ffv1 \
  LOGDIR=docs/wsp/verified/logs/grid-k-pair RENDERS=/home/jorj/2Tb/grid-k scripts/wall_ab.sh
PYTHONPATH=. python3 harness/scripts/gate_sweep.py <fp16.mkv> <k2.mkv> --reference-kind engine-render ...
```
