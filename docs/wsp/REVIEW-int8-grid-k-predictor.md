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

## 6. Attempt 3 - the arithmetic argmin with a calibrated beta. Partly right, still 4 of 22.

A reader proposed the better formulation: do not predict the mover/stayer label (which is about *transfer between
models*), predict `argmin_K` inside one model, from arithmetic rather than from a forward per candidate:

```
input_err(K)  = ||a - round(a/s[c](K)) * s[c](K)|| / ||a||          over all layers, all calibration frames
weight_err(K) = ||w' - dequant(q(w'))|| / ||w'||,  w' = w * s[c](K)
predict       = argmin_K [ input_err(K) + beta * weight_err(K) ]
```

Both terms are exact functions of K from the calibration activations and the weights, so the cost is the fp16
pass that produces the activations - which calibration does anyway. Measured on all 22 models
(`logs/grid-k-beta/rows.json`), sweeping beta:

```
beta 1.0 -> 6.0 :  4/22 agreement with the ladder's argmin, at every beta
```

**What the attempt did establish, and it is worth having.** The two terms have *different shapes*:

- `input_err(K)` falls monotonically with K (median 2.88x from K=1 to K=4) - wider K means finer per-channel grids;
- `weight_err(K)` has an **interior minimum**: 0.021 at K=1, 0.009 at K=4, and 0.029 at per-channel. Folding the
  per-channel scale into the weights first *balances* the columns (a channel with large activations gets its
  weights scaled up, which compresses the row's dynamic range), and only past that point does the clamp stop
  protecting the small channels and the error rises again.

That interior minimum in the weight term is the reason the picture's optimum K is interior - it is not a
mysterious property of the network. But the *sum* still does not rank the argmin, at any beta, because the
quantity being predicted is a thresholded count of pixels rather than a norm: the same error energy lands in
different numbers of pixels depending on where in the range it sits, and that mapping differs per model.

**Why the two-point fallback does not rescue it.** Fitting `A * input + B * weight` from two scored candidates
uses two equations for two unknowns - exact at those two points, and the prediction for the other two is
unconstrained with only four K values. That is the sweep with extra steps: two scored candidates *are* half the
sweep, and what it would buy is the other half.

## 7. What was adopted instead: two candidates, six frames, and no sweep when it cannot matter

The predictor does not exist, so the sweep stays - but two of its costs were removable, both on the review's
argument and both measured in the app.

**Two candidates, not four.** `{K=2, K=3}`: K=1 never won on any of the 22 models, and K=3 captures the outcome
wherever K=4 does except Garfield, where dropping K=4 costs 0.0029 -> 0.0157 - still six times inside the bar.
The holdout is six frames, so this halves the pass count while keeping the label on six frames rather than three.

**No sweep when K=2 is already clean.** If K=2's own share is <= 0.002%, the sweep is skipped after the first
build. The threshold comes from the ladder's largest gain (7.1x): 0.002% x 7.1 = 0.014%, seven times inside the
bar. It is deliberately *not* derived from a 3-frame-to-494-frame multiplier, because none exists between the two
references - the smoke test compares against the model, the 494-frame pair against the engine - and inventing one
is the same error class as the retracted x37.

In the app on the production model both fire, and together with the calibration cache the start-up falls:

```
cold, first run of a clip   calibration measured + sweep skipped     ~33.5 s
warm, cache hit             28.48 s  (fp16 arm 26.6 s -> the int8 premium is now ~1.9 s, was 6.5 s)
log:  int8 calibration: CACHE HIT ... (18 frames not measured)
      int8 grid PICKED K=2 (K=2 already clean, sweep skipped) (scores {'K=2': 0.0018}; build 1.21 s, score 0.43 s)
```

For a model whose K=2 is not clean (realesr at 0.0504%, Ani4K at 0.0166%) the sweep still runs and still finds
K=3 and K=4 - which is what it exists for.

## 8. The app-side discrepancy, resolved: the body trusted its caller's layout

Wired into the app, the pick's scores came back at **99.98% for every K** while `certify()` - the same expression
over the same frames - read **0.00001%** on the render frame. The diagnostic line showed the frames identical in
shape, dtype and device (24 x `(1,3,1080,1920)` fp16 cuda, render frame the same), so the difference was not the
geometry.

It was the **layout**. `Int8Probe` builds its frames as `torch.from_numpy(img).permute(2, 0, 1)[None]` - a view
with stride `(3, 1, 5760, 3)` - while the render path's `frame_to_tensor` ends with `.contiguous()`. The head
kernel indexes the frame as `c*H*W + y*W + x`, so on a view it reads garbage codes. Calibration never noticed
because `_measure` runs the *torch* model, which handles strides; only the body's kernel needs the layout, and only
the pick ever fed it those frames.

The body now makes its input contiguous at the top of `__call__` and logs when it had to. After that the pick
works in the app on the production model:

```
int8 grid PICKED K=2 (scores {'K=1': 0.0042, 'K=2': 0.0018, 'K=3': 0.0019, 'K=4': 0.0046};
                     guard x2 over K=2, chosen on 6 frames, certificate on another slice)
int8 body first-frame sanity: 0.00001%
```

K=2 wins, and K=3 is only 1.05x better, so the guard keeps the default - which is what the ladder said for this
model.

**Cost, measured twice.** The first working version started in 39.84 s against 33.08 s without the pick, because
it calibrated on all 24 frames and then again on the 18-frame slice - 4.8 s of eager forwards for nothing. With
the split (calibrate on 3/4, pick on 1/4) the pick costs **0.77 s** (33.85 s against 33.08 s). A calibration
cache now also removes the remaining ~5 s on a re-render of the same clip, since the grids depend on (model, clip)
and nothing else.

This is the third bug of one class in this lane - the body trusting a caller's shape or layout (the padded width,
the tail's channel dim, now the stride) - and in all three the fix belongs at the boundary, not inside the kernel.

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
