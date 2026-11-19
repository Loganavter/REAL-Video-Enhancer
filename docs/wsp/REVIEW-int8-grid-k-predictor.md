# Choosing the int8 grid clamp K without measuring: three attempts, one negative result, one cheap win

A write-up for an outside reader. Every number is from a record or a probe named in place; the unmeasured parts
are marked as such.

## 1. What the predictor would replace

The body quantises each layer onto a grid `s[c] = max(s_in[c], s_t/K)`, where `K` trades input resolution
against weight precision. The best K is model-dependent (K=2 for some models, K=3/4 for others, by 2.5-7x), so
the shipped path builds the body at K=1..4, scores each with the app's own gross gate on a slice of the
calibration frames, and keeps the winner **only if it beats K=2 by 2x** - a guard against chasing noise on a
handful of frames. That costs about 3 s on top of a 6.5 s start-up.

The question: can the winner be read off the weights instead?

## 2. The target

22 models, six grid values each, three held-out frames (`logs/grid-k/ladder.jsonl`). Split by the guard's own
criterion: **movers** are the 14 models where some K beats K=2 by >=2x, **stayers** the 8 where it does not
(three tie at 1.00x, Epsilon sits at 1.44x).

## 3. Attempt 1 - the calibration grid's dynamic range. Refuted.

`max/median` of the per-channel activation maxima, the obvious "the grid is wide, so per-channel should help"
statistic:

```
movers  2.71  3.09  4.03  4.20  4.34  4.74  4.96  4.99  5.15  6.33  6.66  6.93
stayers 4.25  4.31  4.42  4.48  4.55
```

The ranges overlap completely - a model with spread 4.5 improved and one with 4.48 got worse.

## 4. Attempt 2 - four weight-only statistics. All overlap.

Computed per model from the state dict alone, no forward pass, no calibration:

| statistic | what it would capture |
|---|---|
| `row_cv` | coefficient of variation of the per-output-channel weight energy: are a few filters dominant |
| `row_spread` | max/median of the same |
| `out_spread` | max/median of the per-input-position energy: does the layer read a narrow set of taps |
| `log_cond` | log10 median condition number of the folded weight matrix: does the layer amplify error |

```
row_cv      movers 0.074 .. 0.344   stayers 0.106 .. 0.300   OVERLAP
row_spread  movers 1.22  .. 1.643   stayers 1.196 .. 1.546   OVERLAP
out_spread  movers 1.561 .. 2.839   stayers 1.736 .. 2.404   OVERLAP
log_cond    movers 0.50  .. 0.77    stayers 0.52  .. 0.67    OVERLAP
```

Not marginal overlap: a mover reads `row_cv` 0.074 while a stayer reads 0.300; a mover reads `out_spread` 2.839
while a stayer reads 1.736.

## 5. Why they fail, mechanistically

The two first-order terms of the trade were measured per model, per layer, on held-out frames:

- **input** quantisation error: per-channel grids are 2.7-3.9x **better** on *every* model;
- **weight** quantisation error (on the folded weights): 1.20-1.40x **worse** on *every* model.

Both move the same way for all 22 models, yet the picture outcome flips sign - Futsuu (3.2x better input, 1.40x
worse weights) gets 2.7x worse; Garfield (3.7x / 1.28x) gets 20x better. So the sign is not set by either term's
magnitude; it is set by how each network **amplifies** a per-layer perturbation into a thresholded statistic
(pixels more than 8 levels off on the worst frame). A summary of the weights cannot see the amplifier, and a
summary of the calibration activations cannot either.

## 6. What would capture amplification, and what it costs

- **Per-layer ablation** - quantise one layer, leave the rest fp16, read the gate: 16 x 6 x 3 forwards per model,
  i.e. *more expensive than the sweep it would replace*, and valid only for the model it ran on.
- **A learned rule** over weight statistics: needs a population far larger than 22, and the guard means the label
  is "beats K=2 by 2x", which is noisy at the boundary (Epsilon at 1.44x could cross it on a larger sample).
- **A proxy for the amplifier** - e.g. the response of the final layer to a perturbation of the size the
  quantiser actually introduces. This is the one idea that would be cheap (one extra forward) and it is
  **unmeasured**; it is also close to just running the sweep, so it needs a reason to be cheaper before it is
  worth building.

## 7. The cheap win that does exist: two candidates, not four

The sweep's cost is proportional to the number of candidates. From the ladder, `K=2` and `K=3` capture the
outcome for 21 of 22 models:

```
keeping only {K=2, K=3} instead of {1,2,3,4}:
  median cost vs the best of {K=3, K=4}        0.0%
  worst absolute loss                         Garfield 0.0029 -> 0.0157 (K=4 is the only good one there)
  worst ratio loss (on already-clean models)  Bubble 0.0000 -> 0.0002
```

Garfield is the one real case, and 0.0157% is still six times inside the 0.1% bar. So the sweep can be halved
without losing a decision - which is a better answer than a predictor that does not exist. `K=1` can also go: it
was never the winner on any model.

## 8. An open problem, and it is not about prediction

Wired into the app, the pick's own scores came back at **99.98% for every K**, while `certify()` - which uses the
same expression over the same frames - read **0.00001%** on the same model, immediately after. In isolation the
same pick reproduces the ladder exactly (Futsuu K=2 at 0.0005, realesr K=3, Ani4K K=4, Epsilon K=2 under the
guard). So something about the app's context makes `self.model(f)` and `self(f)` disagree by a factor of ~10^4,
and the pick now **refuses** in that case rather than choosing (it compares its own default score against
`certify()` and raises when they differ by 10x, keeping K=2).

Candidates, none measured yet:

- the app's model is the wrapper's - a `SRVGGNetCompact` whose head is widened to 8 input channels and wrapped in
  a `PadInputConv`; `certify` and the pick both call `self.model(...)`, so this should cancel, but it is the
  first thing to check;
- the calibration frames come from `Int8Probe`, not from the render loop, and may differ in layout, scale or
  resolution from the frame `certify` sees;
- the body's `out_yuv` / `tail_yuv` state during the pick (it is forced off, but the flag and the effective value
  are two different attributes).

A reader who has seen this shape before will recognise it: two components agreeing in isolation and disagreeing
in the app is usually a *context* difference - a flag, a dtype, a frame source - not arithmetic.

## 9. Not covered

- Four weight-only statistics are not "all possible statistics". They are the ones implied by the two measured
  mechanisms; a predictor from a different mechanism would need a different argument.
- The targets are three-frame smoke tests, so the mover/stayer split is not stable at the boundary (Epsilon).
- The 3 s cost is measured on one model at 1080p; it scales with candidates and frame size, not with the model.
- The app-side discrepancy in §8 is unexplained, and until it is, the pick is opt-in and refuses rather than
  guesses.

## 10. Reproduce

```bash
<install-python> scripts/grid_k_predict_probe.py --out-file docs/wsp/verified/logs/grid-k-predict/rows.json
<install-python> scripts/int8_grid_modes_probe.py --out-file docs/wsp/verified/logs/grid-modes/probe.json
```
