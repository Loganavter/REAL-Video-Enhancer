# Plan: int8 body under a CUDA graph (2026-11-15)

Delivery plan for the int8 lane. The research is closed; what is left is integration, and every stage ends
in a number or a refusal. The order is not negotiable: it is arranged so that silent failures are caught
before anything is built on top of them.

Adapted to this machine at the end of each stage (`Tools`).

## Facts the plan rests on

| | measured |
|---|---|
| block 64->64 at 1920x1080, int8, per-channel epilogue | **1.97 ms, 77.6 TOPS** (fp16 fused block: 2.93 ms) |
| own defect, no reuse: anime / Bebop / Violet | 0.005-0.017% / 0.016% / 0.028-0.031% of pixels above 8 levels (bar 0.1%) |
| composition under the gate: anime / Violet | 0.017% / 0.258% (Violet over the bar -> fp16 fallback) |
| engine share of a frame: anime gated / film / Bebop | 0.646 / 0.866 / 0.904 |
| register frontier of this codegen | BM 256, 255 registers, 8 warps is the best; caps 128/168 cost 11.7% / 1.8 KB spill |
| graph capture | bit-exact (max abs diff 0) with 16 launches, stream read **inside** the capture |

## Stage 0 - reference and contracts (one hour, no GPU timings)

The reference has already caught two bugs (`permute(1,0,2)`, a weight scale declared and not applied) and
broken once itself. Until it agrees with a third implementation, any mismatch reads as "the kernel is wrong".

**Do:**
1. Reference against `torch._int_mm` **with the scales applied** and against fp32 `conv2d` on dequantised
   weights. Three implementations must agree to zero on one frame.
2. Write the contract sheet out of the PTX and the metadata, not from memory.

| # | contract | value | failure type |
|---|---|---|---|
| 1 | grid | `H*W/BLOCK` (8100 at BM 256, 16200 at BM 128) | silent: part of a frame |
| 2 | block | 128 threads (`.reqntid` stripped, the arithmetic still assumes it) | silent |
| 3 | `sharedMemBytes` | `metadata.shared` = 12288 at BM 128 / 20480 at BM 256 | silent: reads/writes outside staging |
| 4 | scratch | 0 **for this kernel and this Triton version** | silent |
| 5 | stream | read **inside** `with torch.cuda.graph(g)` | empty graph that reports OK |
| 6 | context | `cuCtxGetCurrent`, never a second one | hard failure (400) |

**Gate (refined by the first run):** the two integer implementations must agree **exactly**; the float
reference must agree to within **1 level on unsaturated pixels** and may diverge where the output clamps.

First run: kernel against shift+`_int_mm` = **0 exactly**; kernel against fp32 `conv2d` on dequantised weights
= **max 8 on 2.2% of pixels**, and the two references differ from each other by the same 8, so that gap is
between the references and not attributable to the kernel. It is expected in kind - the integer path sums 576
terms exactly and scales once, while the float path scales the weights first and accumulates in float32 - but
"expected in kind" is not a measurement, so the gate above replaces the absolute zero the plan started with.
The next check is the same comparison restricted to pixels that do not saturate; if it is <= 1 there, the
float reference is usable as a sanity check and not as a bit-exact standard.

A third permutation bug was caught getting this far: `w.permute(2, 1, 0)` is [cout, cin, tap] and reshapes
into a valid conv2d weight, while `permute(2, 0, 1)` gives [cout, tap, cin] and reshapes silently into a
*wrong* weight of the same element count. Three permutation mistakes in one investigation (twice in a
reference, once in a probe), all of them producing plausible numbers rather than errors.

**Tools:** `scripts/int8_block_probe.py` (`reference()`, `--check-size`), `scripts/launch_cubin.py`.

## Stage 1 - build the pipeline eagerly, bit-exact (one day)

Not "an int8 TRT plan": TensorRT cannot load a Triton cubin without a plugin, and the plugin path is E9'/QDP,
closed by measurement. The architecture is what the graph contract already proved:

```
head engine (fp16, TRT) -> clamp+quantize -> 16 Triton launches -> dequantize -> tail engine (fp16, TRT)
```

The tail segment includes PixelShuffle and the residual add. `clamp` happens **before** the residual
(`SRVGG.py:75` against `:82`), so one buffer serves both - but it must be applied before quantisation or the
[0, 1] scale clips.

Boundary conversions are plain torch ops on this pass. The goal is correctness, not speed.

**Measure:** the full frame against the shipping fp16 engine, RGB channel-max, worst of >= 3 frames on two
titles.

**Gate:** the build reproduces the **simulation's numbers** - anime 0.000-0.005%, Violet 0.028-0.031% - not
merely "inside the bar". Anything else means the scale order, the boundary dtype or the slope indexing
diverges, and which side is right is then unknown.

**Kills it:** build/simulation disagree by more than 2x on the share. Stop and find out, do not optimise.

**Tools:** `int8_accuracy_sim.py` is the reference implementation to reproduce; `TensorRTHandler.py` is where
the three-segment dispatch goes. `scripts/int8_pipeline.py` is the build.

**State (2026-11-16): the pipeline runs end to end and its gate FAILS.** Sixteen kernel launches with real
handoffs, head and tail in torch: `>8 100.000%, peak 236, median 113 levels` against a bar of 0.1%. Two
confounds are already fixed and did not change the number - the calibration mutating the model it is given
(now runs on a copy, so head and tail keep their original weights), and a debug script that derived the
output scale from the PReLU *slope* instead of the PReLU *output* range.

A median of 113 levels is the signature of a saturated picture rather than a scale that is slightly off: the
tail receives a dequantised body output on a grid that is too coarse and clamps. The next diagnostic is to
print, per layer, the ratio of the kernel's dequantised output to the fp16 reference's activation - the
first layer whose ratio leaves 1.0 names the layer where the scale bookkeeping breaks - before touching
anything else.

Per the stage gate this stops the work here: build/simulation disagree far beyond 2x, so the cause is found
first, not optimised around.

**Third pass: the layout is settled and the pipeline is out of the garbage regime, but the gate still fails.**
The diagnostic that settled it passed ONE tensor to both sides with the same epilogue: `[tap, cin, cout]`
matches `_int_mm` exactly (max 0), the transpose differs by 254. Built that way the pipeline reads

    >8 3.589%  peak 58  median 4.82 levels      (was 100% / 113 levels)

200x off the simulation's 0.017% rather than nonsensical, which moves the cause from "wrong data" to "wrong
grid". The remaining gap is explained by the one thing the build cannot express yet: the simulation quantises
each PReLU output on a **per-channel** grid, while the pipeline uses a **per-tensor** grid because the kernel
takes one scalar-activation scale. Chaining per-channel grids needs the kernel to apply a per-input-channel
scale inside the dot (the weight scale is per-output-channel, so the product is rank-1 and will not fold into
the vector the kernel loads today) - the same E9' mechanism, one level down.

That is the next change, and it is a kernel change with a re-time attached, not a bookkeeping fix.

**Tested, and refuted: the grid was not the cause.** Implemented per-channel activation grids with each
layer's input scale folded into its weights (so the dot stays int8: `acc * s_w[cout] = sum_cin x_q[cin] *
(w*s_in)[cin,cout]`, the real value). The pipeline reads **3.680%** against 3.589% for the per-tensor
variant - a 2.5% relative change on a number that is 200x over the bar. So the grid granularity is not the
cause, and neither is the fold.

Which leaves one unambiguous discriminator, and it is cheap: compare the **pipeline's** intermediate tensor
against the **simulation's** intermediate tensor for the same layer, in the same units. That splits "the
kernel is fed the wrong data" from "the kernel's maths is wrong", which the fp16 comparison cannot do because
both causes look identical there. It needs the simulation to expose its per-layer output, ~20 lines.

**And the kernel is exonerated against the simulation at layer level.** Fed the same quantised input and the
same quantised weights, kernel and simulation agree to **0.0002 of the signal** (max 0.21 levels, mean 0.000):

| | range | mean |
|---|---|---|
| kernel | -8.18 .. 26.80 | 1.32 |
| simulation | -8.18 .. 26.80 | 1.32 |

So the layer maths is right, the epilogue is right, and the 200x gap lives in **the chaining or in the head
and tail** - not in the kernel. The next test follows from that and is one comparison: run the pipeline and
compare it against the *chained* simulation on the same frame, rather than against fp16. If the two chains
agree, both are correct and the fp16 comparison was measuring the int8 chain's own accumulated error (which
would then be a finding about 16 hops, not a bug); if they diverge, the difference is in how the pipeline
passes grids between layers - most likely the per-layer `s_out` coming from fp16 calibration while the chained
activations drift away from it and clip.

Three things are now excluded by measurement: the kernel's dot orientation (exact against `_int_mm`), the
scale granularity (3.680 against 3.589), and the kernel's maths against the simulation (0.0002).

**And the comparison the plan asked for says the simulation is not a reachable target.** On the same 512x512
frame the chained simulation reads **0.000% / peak 4 / median 0.65**, which is essentially exact, while the
pipeline reads 1.693-3.589%. The reason is structural, not a bug:

- the simulation quantises the **input of each conv per-tensor** on a *float* tensor that already carries the
  previous layer's **per-channel** grid. That transition is free in float and is what makes it accurate;
- a real int8 chain must pick one grid per hop. Per-channel output grids (1.693% with the scale folded into
  the weights at alpha = 1) keep the ranges but wreck the weight conditioning - `w * s_in[cin]` spans 30x
  across input channels and is then quantised per output channel, which is exactly the trade the correlation
  measurement (+0.45) predicted;
- per-tensor output grids chain cheaply but read 3.589%;
- the third option, an explicit requant between grids, costs 266 MB per hop (0.92 ms x 16 = 14.7 ms), which
  erases the 1.53x the block wins.

So stage 1's gate as written - "reproduce the simulation's numbers" - cannot be met by any chainable int8
body: the simulation's 0.000% is bought with a free float requant that the hardware does not have. The gate
has to be restated against what a chain can do, and 1.693% is what it does.

**The discriminating variable is the INPUT grid, not the fold.** Same float harness, same crop, three
configurations:

| chain | input grid | output grid | share |
|---|---|---|---|
| the simulation (`quantised_forward`) | **per-tensor** | per-channel | **0.000%** |
| hand-rolled chain, no fold | per-channel | per-channel | **2.038%** |
| hand-rolled chain, fold alpha=1 | per-channel | per-channel | **2.765%** |

So the fold costs a factor of 1.36 (2.765 against 2.038) and the per-channel *input* grid costs a factor of
infinity (2.038 against 0.000). The simulation's recipe is **per-tensor in, per-channel out**, and that is
what a chain must reproduce - which is exactly what the prologue requant does: the layer writes int8 on its
per-channel grid, and the NEXT layer's prologue requants per-channel -> per-tensor in registers, touching the
operand before `tl.dot` and leaving the int32 accumulator alone.

That also explains the pipeline's 3.589%: it chains per-channel output grids into per-channel inputs, which
is the 2.038% configuration plus the head/tail.

The plan's earlier claim that "no chainable int8 body can meet the gate" was wrong, and it was wrong for the
reason this file keeps recording: it compared the pipeline against the simulation without isolating which of
the two variables differed. The gate stands.

**Built, and it does not close the gap.** `scripts/int8_prologue_kernel.py` requants per-channel ->
per-tensor on load with an integer multiplier and shift, leaving the accumulator int32:

    ptxas: Used 176 registers, 0 bytes spill, smem 12288    <- no hot spill, so the variant is viable
    pipeline on the same frame: >8 1.870%  peak 34  median 2.47

Against the simulation's 0.000% that is still 100x, so the prologue requant - correct as it is - is not the
missing piece either. The discriminators are now exhausted one by one: dot orientation (exact), scale
granularity (3.680 vs 3.589), kernel maths against the simulation (0.0002), the fold (2.765 vs 2.038), the
input grid with a float intermediate (0.000 vs 2.038), and now the same input grid with an integer prologue
(1.870).

Which means the difference is not in any single mechanism the last six tests varied, and the next step is
the one the plan named two passes ago and that keeps getting deferred: a **per-layer comparison of the
pipeline's chain against the simulation's chain**, layer by layer, on the same frame and in the same units.
Six hypotheses have been tested against fp16 in isolation and five of them died; comparing the two chains to
each other is the only test that cannot be satisfied by a coincidence of the final number.

**Localised on the full frame: the error follows the content, not the border.** Pipeline against simulation on
1920x1080, 8x8 tile map of the share of pixels above 8 levels:

```
 0.19  0.30  0.01  0.25  0.98  0.63  1.22  0.10      <- top strip: quiet
 0.08  0.00  0.01  0.05  0.93  1.34  1.17  0.20
 0.14  1.44  0.03  0.03  0.20  1.84  0.64  0.60
 1.56  5.70  7.99  7.34  6.26  5.11  3.19  0.81      <- the busy middle
 1.08  4.60  6.96  5.27  5.18  4.13  2.11  1.47
 2.01  2.30  4.85 10.28  5.25  4.02  2.22  0.39
 0.32  0.87  4.35  6.32  4.86  6.16  1.14  0.56
 0.05  0.26  2.38  3.98  3.64  1.36  0.00  0.00
```

Only 0.5% of the failing pixels lie in the outer 16 px band (which is 2.3% of the image), and the mean error is
*lower* at the border (2.78) than inside (3.30). So no clamp-at-the-edge story and no padding story: the
pipeline diverges where the picture is busy, i.e. where the activations are large.

Which points at saturation: every grid is a maximum over the calibration pass, and the chained activations
drift away from the fp16 ones those maxima were taken on. A large activation that the fp16 chain keeps inside
the range lands outside it after sixteen hops, clips at +-127, and the error appears exactly in the busy
tiles.

**Measured, and it is saturation from the first hop:**

| layer | saturated, simulation | saturated, pipeline |
|---|---|---|
| 1 | **0.00%** | **12.63%** |
| 2 | 0.00% | 5.26% |
| 3 | 0.00% | 3.58% |
| 5 | 0.00% | 1.55% |
| 8 | 0.00% | 1.15% |

Twelve percent of the first layer's quantised values sit at the clamp, against none in the simulation, and the
chain then carries that clipping forward. So the defect is not the grid, the requant or the size: the
pipeline's own output is **larger than the range it is being quantised against**, from the first layer onward.

Which makes sense of the whole stage: every grid is a maximum taken on the **fp16** model's activations
(`outputs[key]` from a calibration pass), while the pipeline quantises the values its **own chain** produces.
Those two distributions agree at the head and part company immediately after, because the chain re-quantises
and the fp16 activations never see that. A grid fitted to the wrong distribution clips by construction.

**Built and measured: chain recalibration converges and barely moves the share.**

    calibration pass 1: max grid drift 10.24%
    calibration pass 2:                 4.72%
    calibration pass 3:                 2.59%
    pipeline: >8 1.569%  peak 34  median 1.77    (was 1.910% / median 2.54)

So the grids are now fitted to the chain's own maxima, the iteration converges at 2.6% per pass, the **median
falls from 2.54 to 1.77 levels** - and the share stays at 1.57%. The saturation hypothesis explains the bulk of
the *typical* error and almost none of the *tail* the bar measures.

**STAGE 1 PASSES ON BOTH POPULATIONS. The 6.72x and the 2.90x were my instrument, not the build.**

Dumping both images and comparing them to **one** reference exposed it:

| | vs the true fp16 reference | vs a quantised-weight reference |
|---|---|---|
| build, Violet 3840x2160 | **0.03065%** | 0.00355% |
| simulation, same frame | **0.05550%** | 0.01059% |

`simnum.py` computed `ref = model(x)` **after** `calibrate(model, ...)` had replaced the weights with their int8
rounding, so it was scoring the simulation against a quantised-weight picture while the build was scored against
the true fp16 one. `int8_accuracy_sim.main()` computes the reference *before* calibrate (line 145 against 152) -
my harness had the two lines the other way round. Against one reference the build is **1.81x better than the
simulation at 4K**, not 2.9x worse, and the ratio the plan's kill rule cares about is **0.552x**.

That retires every 4K number recorded before this pass: 6.72x, 2.90x, and the "Violet 0.028-0.031% band" in the
gate all came from comparisons of two different populations. What actually holds, measured now on the same frame
and the same reference:

| | build | simulation |
|---|---|---|
| anime 1920x1080 | 0.00000% (peak 7, median 1.08) | 0.00000% (peak 8, median 1.23) |
| Violet 3840x2160 | 0.03065% (peak 17, median 1.39) | 0.05550% (peak 21, median 1.62) |

`claim_check` now reads **PASS** on `docs/claims-int8-stage1.txt`, the anime gate and the 2x rule both green.

Two artefacts were added so this comparison is repeatable rather than reconstructed: `--dump` on the build and on
the simulation, so the two images can be differenced directly. Doing that showed the build and the simulation
differ on 8200 of 33 177 600 pixels at 4K, spread evenly across the frame (29 of 8200 fall in the 8-pixel border,
the rest on content), which is what a rounding difference looks like and not what an edge or block bug looks like.

Also measured and refuted in this pass: per-tensor activation grids, which is the simulation's own choice, are
worse at 4K (0.04100% against 0.03065%) - so the build's per-channel grids plus the prologue requant stay.

**The per-hop instrument is rebuilt and it agrees with its own end-to-end result; and what it exposed first is a
unit error in my own reading.** The new instrument marches both chains in lockstep, keeps the chain state as int8
codes plus a grid (so a 4K frame fits), and processes the frame in row strips with a one-row halo - exact for each
strip's interior because a hop is a 3x3 conv. Validated at 1080p, where the answer is known: build vs fp16
0.00000%, simulation vs fp16 0.00000%, build vs simulation 0.00001%.

What it settled: **body levels are not image levels.** The body's mean absolute activation is 0.0032 while the
image's is 0.0904, and a deliberately injected 40-level body perturbation lands as 17.8 image levels. So the
"22 levels at hop 1" I reported two passes ago was a unit error - a large number in a small unit - not damage.
Any future per-hop number in this document must be quoted in propagated image levels or it is meaningless.

**And the 4K defect is the chain self-calibration itself.** Removing it - the grids come from the calibration on
the fp16 activations, exactly as the simulation does - improves both populations:

| | with self-calibration | without |
|---|---|---|
| anime 1080p | 0.00043% | **0.00000%** (peak 7, median 1.08; the simulation is 0.00000%, peak 8, median 1.21) |
| Violet 4K | 0.07115% | **0.03075%** (peak 17, median 1.39) |

So the build is now **bit-exact with the simulation on the anime gate** and 6.72x -> **2.90x** on 4K Violet. That
reverses the pct-99 reading from two passes ago, which had kept the self-calibration in: that comparison ran with
the broken rounding shift, so it was measuring the wrong thing - the same confound, one level up.

Three more candidates were closed at 4K: percentiles are worse, not better (99.9 -> 0.102%, 99.5 -> 0.447%,
99 -> 0.767%, against 0.03075% for the true maximum, which is also what the simulation uses); snapping the head's
input to the simulation's 127-level grid is worse (0.03379%); and the prologue's channel collapse does not happen
where it would matter - 0 of 1024 channel-slots are below the 1/254 threshold at 4K, and 8 of 1024 are at 1080p,
where the build is exact.

**Open: 2.90x against the plan's 2x rule.** The absolute gap is 418 pixels of 8 310 400 (0.03075% vs 0.01059%),
both inside the anime gate band and both 3x under the bar, so the rule is being violated by a small number - but
it is violated, and the next pass starts from the 4K frame with the instrument that now agrees with itself.

**STAGE 1 PASSES ON THE ANIME POPULATION: the rounding shift was implemented wrong, and that was the 5x.**
The "sign-aware rounding" I tested twice and rejected twice was not a rounding shift at all. Adding `-half` for
negative products rounds **away from zero**:

| prod | true | floor | round (prod + half) | what I called "sign-aware" |
|---|---|---|---|---|
| -436 | -3.406 | -4 | **-3** | -4 |
| -300 | -2.344 | -3 | **-2** | -3 |
| -200 | -1.562 | -2 | **-2** | -3 |
| 200 | 1.562 | 1 | **2** | 2 |

So the 4.990% and 2.773% readings measured a shift that was worse than flooring by construction, and the
conclusion "the truncation is compensating something" was an artefact of my own instrument. With ONE constant for
both signs - `((a * m) + (1 << (shift - 1))) >> shift` - the same build reads:

    anime 1920x1080, pct 100, recalibration on:  >8 0.00043%  peak 10  median 1.72 levels
    simulation, same frame:                      >8 0.00000%  peak  8  median 1.21 levels

Against the gate's own anime band (0.000-0.005%): **passed**, and the peak sits at 10 against a guidance of 20.
That is 1.569% -> 0.00043%, and 9 pixels of 2 073 600 above the 8-level bar.

**The Violet half of the gate does not reproduce, and the build loses there.** The recorded band is 0.028-0.031%,
but the simulation on a 4K Violet frame (t = 10 s, 3840x2160) reads **0.01059%**, so the band belongs to a
different population and cannot be judged against. Comparing build to simulation on the same frame:

| | share |
|---|---|
| build, Violet 3840x2160 | 0.07115% |
| simulation, same frame | 0.01059% |
| ratio | **6.72x** |

which is the plan's own kill condition ("build/simulation disagree by more than 2x - stop and find out"). So
stage 1 holds where the gate is defined and does not hold at 4K; the 4K gap is the next thing to close, not a
rounding question.

Context for the 4K number, because the share alone misleads: this plan's own composition table already expects
Violet at **0.258%**, over the 0.1 bar, with the fp16 fallback covering it. The build now reads 0.07115% there -
**under the bar and 3.6x better than the composition table** - so the open item is not "Violet is unusable", it is
"the build is 6.7x the simulation on the same frame". And the 4K run has no saturation at any hop either (0.00%
from layer 5 up), so it is accumulation, not a grid that is too tight; the per-hop instrument needed to localise
it has to be rebuilt first, because the one built in this stage disagrees with its own end-to-end result.

Two side results: at pct 100 the grid now uses `amax` instead of `quantile` (identical value, no sort buffer,
which is what let the 4K frame run at all instead of OOM), and measuring the grids on the quantised model instead
of the fp16 one moves 1080p 0.00043 -> 0.00047 and 4K 0.07115 -> 0.06436, so the grid source is not the mechanism.

**Cost of the fix: none in registers, and the prologue is what costs time.** `cuobjdump -res-usage` on the
compiled cubin with the bias in the epilogue: **REG 241, STACK 0, LOCAL 0** (against 255 before), so the bias add
does not touch occupancy and the accumulator is still int32 - int8-MMA is intact. Timing on the full 1920x1080
frame with the prologue requant in the kernel:

| block | ms per layer |
|---|---|
| BM 128, 4 warps | **3.09** (49.39 ms for 16 hops) |
| BM 256, 8 warps | 3.60 |
| BM 128, 8 warps | 4.34 |

So the 1.97 ms in the earlier record belongs to the bare int8 block without the prologue; the requant version
costs ~3.1 ms per layer, and the prologue - nine masked gathers, an integer multiply and a clamp per element -
is where the difference goes. That is a real number for stage 2 rather than an assumption.

**After the bias fix: the chain reproduces the simulation exactly, and the remainder is frame-size dependent.**

With the bias carried, the same harness at 256x256 and 512x512 reproduces the simulation **bit-exactly**
(0.0000%, peak 7) whichever grid source is used, and the script agrees at 512x512 (0.000%). So the kernel chain
*is* the simulation's chain. At full frame it is not:

| size | pipeline vs fp16 |
|---|---|
| 640x640 | 0.001% |
| 768x768 | 0.747% |
| 1024x1024 | 1.194% |
| 1920x1080 | 1.045% |

Smooth growth, mean error 1.6-2.1 levels per tile-row uniformly across the frame against the simulation's 1.26-1.33,
no saturation anywhere (0.00% at every hop, ranges matching to three decimals at hop 1: 50.629 both). So it is a
small per-hop deficit that accumulates with the activation dynamic range, not a boundary or block artefact.

Two arithmetic findings, both measured at full frame:

| configuration | share |
|---|---|
| flooring shift, no recalibration, pct 100 | 1.045% |
| **rounding shift, same settings** | **4.990%** |
| recalibration on, pct 100 | 0.929% |
| **recalibration on, pct 99** | **0.091%** |
| recalibration off, pct 99 | 0.797% |

The rounding shift is worse by 5x in every configuration tested, so the truncation is *compensating* a systematic
positive bias of about half a code per element per hop somewhere else in the hop - which is now a specific,
testable statement rather than a mystery. And the chain self-calibration I tried to remove as a fixed point is
worth 0.7 percentage points at pct 99, so it stays (opt-out via `--no-recal`).

**THE DEFECT: the pipeline dropped the conv bias.** The A/A said my hop returned 3.3x the simulation's
magnitude at layer 1 with the same input and the same weights. Instrumenting the module itself:

    module: Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    bias is None: False   |bias| mean 4.28992
    module(captured input) |mean| 1.5006   == captured 1.5006   (max|diff| 0.000000)
    my conv2d, no bias     |mean| 3.8041
    my conv2d + bias       |mean| 1.4987

Every one of the 18 body convs carries a bias, and the bias is not a rounding detail: its mean absolute value is
2.1-4.3 while the activations are ~0.5, so **dropping it is a different network**, not a quantisation choice.
`grep -n bias scripts/int8_pipeline.py scripts/int8_prologue_kernel.py` returned **nothing** before this pass -
the whole int8 lane was measuring an unbiased model.

Carrying the bias in the epilogue (after dequantisation, before PReLU - one extra per-channel load and add, the
accumulator stays int32 so int8-MMA is untouched):

| pct | share |
|---|---|
| 100 | 1.569% -> **0.789%** |
| 99.9 | 0.180% |
| **99** | **0.089% - first time under the 0.1 bar** |
| 98 | 0.186% |
| 97 | 0.270% |
| 95 | 0.422% |
| 90 | 0.607% |

And the A/A magnitudes now agree: my hop 0.578 against the simulation's 0.526 at layer 1 (was 1.754 vs 0.526).

The rounding shift was re-tested **with** the bias in place, because every knob sweep in this stage had been run
against an unbiased chain and was therefore invalid: 4.091% at pct 100 and 0.172% at pct 99, both worse than the
flooring shift's 0.789/0.089. Flooring stays.

**A/A run: the two hops diverge at the FIRST layer, by 3.3x in magnitude.**

| layer | mean abs, my hop | mean abs, simulation | relative |
|---|---|---|---|
| 1 | **1.754** | **0.526** | 3.65 |
| 2 | 0.894 | 0.711 | 1.53 |
| 3 | 0.984 | 0.825 | 1.48 |
| 6 | 1.062 | 0.850 | 1.48 |

The first layer is fed the *same* activation, the *same* grid and weights that differ only by int8 rounding, and
my hop returns 3.3x the magnitude. So my hand-built hop is not the simulation's hop - and since the kernel was
shown bit-identical to my hop (0.0000%), the disagreement is between **the harness and the simulation**, not
between the kernel and anything.

That is the end of what this stage can conclude without fixing the harness: the pipeline's 1.569%, the 0.45-0.49%
at 512x512, and every "the chain is wrong" statement in this file are measured against a reference my own code
cannot reproduce, so they are not results. The 3.3x factor at layer 1 is the thing to chase, and it is a
harness bug - one layer, one set of inputs, printable - not a property of the pipeline.

**And the comparison itself fails A/A: my harness cannot reproduce the simulation it is comparing against.**

| chain | vs fp16 |
|---|---|
| my torch chain, chain grid, int conv (variant A) | 0.4795% |
| my torch chain, **the simulation's own hop**, fp16 grid, float conv (variant B) | 0.4865% |
| `quantised_forward` itself | **0.0000%** |
| `quantised_forward` with the model's weights quantised too | **0.0000%** |

Variant B is built to be the simulation's hop verbatim - per-tensor input grid from `inputs[key]`, per-channel
output grid from `outputs[key]`, PReLU, one quantisation per hop - and it still reads 0.4865% where the
simulation reads 0.0000%. So the harness and the thing it measures differ in something not yet identified, and
**every comparison of the pipeline against `quantised_forward` in this stage inherits that error**, including
the 1.569% headline: it is a pipeline measured against a reference my own harness cannot reproduce.

That is the pattern this file already names twice (the empty graph that reported OK, the reference that ignored
its weight scale): an instrument that disagrees with itself by 0.48 percentage points is not a measuring stick.

The next step is therefore not another hypothesis about the kernel - that is already exonerated bit-for-bit -
but an A/A on the harness: run variant B and `quantised_forward` on the *same* activation and compare their
per-layer outputs, layer by layer, until the first divergence names the line where my hop differs. Only after
that is the 1.569% interpretable at all.

**The separating test: the kernel and a torch chain with the same bookkeeping agree exactly.**

| chain | vs fp16 |
|---|---|
| torch, sixteen hops, prologue requant in float | 0.4901% / peak 25 / median 1.11 |
| kernel, same grids | 0.4535% / peak 25 / median 1.21 |
| **kernel against the torch chain** | **0.0000% / peak 4** |

So the kernel does exactly what the bookkeeping says, and it is exonerated by the test the plan named. Whatever
separates 0.45% from the simulation's 0.000% is a bookkeeping choice, not an implementation one - and it can now
be iterated in **torch at full speed**, with no kernel rebuild and no GPU launch, which is the practical gain of
this pass.

Two things this also fixes in the reading of the earlier sweeps: at 512x512 the chain reads 0.45-0.49% while the
full frame reads 1.569%, so the frame-to-frame spread is content, and every "the kernel does something odd"
hypothesis was testing an implementation that is bit-identical to its own specification.

**The grid's SOURCE is not the difference either.** Switching the prologue target to the simulation's own
per-tensor calibration (`inputs[key]/127`, measured on the fp16 activation) reads **1.581%** against 1.569% for
the chain-derived `max(per-channel grid)`. No difference, so the structural statement recorded last pass is
refuted as well, and the claims file now carries both numbers.

That closes the eleventh suspect. The list, all measured, all negative:

| suspect | test | result |
|---|---|---|
| dot orientation | one tensor into both paths | exact (0) |
| scale granularity | per-tensor / per-channel / fold | 3.589 / 1.693 / 2.765 |
| prologue arithmetic | layer 1 vs reference, no clipping | 0.956 |
| floor shift | rounding shift | 2.773% |
| grid source | simulation's calibration | 1.581% |
| grid value | percentile 100/99.9/99/95 | 1.910 / 1.783 / 1.838 / 2.064 |
| grid margin | 1.5 / 2.0 / 3.0 | 2.476 / 2.905 / 3.121 |
| collection strategy | float walk / kernel atomic max | 74% / 96% |
| requant precision | SHIFT 12 / 16 / 20 | 1.910 / 1.926 / 1.935 |
| frame size | 256 / 512 / 1024 / full | 0.000 / 0.452 / 2.276 / 1.783 |
| head and tail | tail fed the simulation's body output | **bit-exact 0.0000%** |

Eleven measurements, and the only thing still true is the sentence from two passes ago: one layer matches, the
sixteen-hop chain does not, and the tail is exact. What has never been done is the one comparison that would
separate "the chain is right and the simulation is measuring something else" from "the chain is wrong": run the
**simulation's** sixteen hops with the pipeline's own quantise-dequantise at every hop but *without* the kernel,
i.e. the same arithmetic in torch, and compare it to the pipeline. If torch-chain and kernel-chain agree, the
kernel is fine and the simulation's 0.000% is measuring a different pipeline; if they differ, the kernel's hop
is not doing what the pipeline's bookkeeping says.

**The tail is exact, and a grid margin makes things worse.** Two results that close the remaining suspects.

**Tail: exonerated.** The simulation's own body output pushed through the pipeline's tail code (the same three
lines the pipeline uses) reproduces the simulation's final image **bit-exactly**:

    captured body output (1, 64, 512, 512) fp16
    pipeline tail vs simulation, same body output: >8 0.0000%  peak 0  median 0.006 levels

So PixelShuffle, the residual, and the dequantisation of the last grid are all correct. The 1.569% is entirely
inside the sixteen-hop body chain.

**Margin: refuted in the pipeline too** (it was already refuted in the simulation):

| grid margin | share |
|---|---|
| 1.0 | **1.569%** |
| 1.5 | 2.476% |
| 2.0 | 2.905% |
| 3.0 | 3.121% |

Coarsening the grids costs more than the clipping it removes, exactly as the simulation's margin sweep said.
And with the tail clean and every knob refuted, the remaining difference is a single structural one: the
pipeline's hop quantises **twice** per layer - int8 on the per-channel grid, then the prologue requant to a
per-tensor grid - while the simulation's hop also quantises twice but between different grids: its input grid
is a per-tensor maximum measured on the **fp16** activation, the pipeline's is `max(per-channel grid)` of the
**chain's** output. Those two agree at the first layer and diverge after it, and no amount of margin,
percentile or collection strategy changes that because they all move the grid, not its source.

That is where stage 1 stops: the mechanism is narrow enough to state in one sentence - the hop's input grid is
derived from the chain rather than measured on the material the bar judges - and it has not been tested,
because every test of it so far changed the grid's *value* instead of its *source*.

**The rounding shift was tried and is worse: 2.773% against 1.569%.** Sign-aware rounding in the prologue
(`(q*m + sign*2^(SHIFT-1)) >> SHIFT`) raises the share to 2.773% and the median to 2.96 levels, so the floor
truncation was not the defect - it was partially cancelling something else in the chain, and removing it
exposed more. Reverted to the truncating version, which stays the best measured: 1.569%.

So both named suspects from the last pass are now resolved against:

| candidate | test | result |
|---|---|---|
| prologue arithmetic (double counting) | layer 1 against the shift reference, no clipping | **0.956** - correct |
| floor-shift bias | sign-aware rounding shift | **2.773%** - worse, reverted |
| per-hop grid choice | three collection strategies | all worse or equal (1.569 / 74 / 96) |
| head + tail | not yet compared in one harness | **open** |

That leaves the head and the tail as the only untested part of the pipeline. The next measurement puts the
**simulation's own body output** through the pipeline's head and tail code and compares it against the
simulation's final image: if they differ, the tail (PixelShuffle, residual, the dequantisation of the last
grid) is the remaining 1.569%; if they agree, the body chain is still wrong despite the layer-1 agreement, and
the difference is between "one layer right" and "sixteen hops right".

**Discriminated: the prologue is numerically right, and the 1.8x was my measurement.** Layer 1 fed the
identical quantised buffer, with the output grid set 20x too large so nothing clips:

    s_in[:3] = 0.01219, 0.03368, 0.01622     s_t = 0.04930     median s_in/s_t = 0.250
    m_int[:3] = 259200, 716276, 345055       (SHIFT is 20 in this build)
    |kernel| 1.8240   |reference| 1.9074   ratio 0.956

So the prologue's algebra - rescale the buffer from its per-channel grid to `s_t`, then dequantise with
`w_scale * s_t` - reproduces the reference to **4.4%**, not with a 1.8x gain. The 1.809 figure came from my
magnitude print, which compared the chain's dequantised output against the fp16 activation *on the full frame*
whereas the earlier per-layer test had compared against the simulation's activation; two different denominators,
one claim. That is the same class of error the claims file exists for and it was committed without one.

The 4.4% deficit is the floor shift: `(q*m) >> SHIFT` truncates toward -inf, and 3 of 4 channels have
`s_in/s_t` near 0.25, so the multiplier is ~0.25 x 2^20 and the truncation is systematic. A rounding shift is
the next one-line test.

What this does *not* explain is the pipeline's 1.569%: with the prologue correct, the chain's remaining error is
in how the grids are chosen per hop (`outs` from fp16 calibration, then from the chain's own clipped maxima) -
the fixed point already recorded - or in the head/tail, which have not been compared in the same harness.

**The gain is present at the first layer, so it is a scale factor and not accumulation.**

| layer | 1 | 2 | 3 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| mean abs of the chain / of fp16 | **1.809** | 2.184 | 1.864 | 2.032 | 2.411 | 3.048 |

Layer 1's *input* is bit-identical to the fp16 head, so a 1.8x ratio there cannot come from the chain drifting:
the first layer alone computes about twice the value it should. That rules out accumulation and puts the defect
in the kernel's scale bookkeeping, with two candidates:

- the prologue's `(q * m) >> SHIFT` **floors** toward -inf for negative values, adding a -1 bias per element
  per hop; over 576 accumulated terms that is a systematic magnitude increase, and it is the kind of bias that
  shows as a gain rather than as noise;
- or `s_t` vs `s_in` is applied twice somewhere. The algebra says it should not be: the prologue maps the
  buffer from the `s_in` grid to `s_t`, and the epilogue multiplies by `w_scale * s_t`, which is the true value
  `sum (a*s_t) * (w_q*w_scale)`. If the prologue were skipped, the ratio would be exactly `s_t/s_in` >= 1,
  which on this frame is the 1.8-3x observed - so the failure looks like *the prologue not applying*, not like
  it applying twice.

The discriminating test for the first candidate is one line: replace `>> shift` with a rounding shift
(`(q*m + half) >> shift` with the sign handled) and see whether the ratio moves toward 1.0. If it does, the
bias is the cause; if it does not, the prologue is not being applied and the multiplier is wrong.

**All three collection strategies now tried, and the last is the most informative.**

| collection | share |
|---|---|
| clipped self-calibration (grid from the kernel's own output) | **1.569%** |
| float walk, pre-clamp maxima | 74.451% |
| **kernel writes its own pre-clamp per-channel maximum (atomic max)** | **96.585%** |

The third gives grid drift 79% -> 66% -> 324% across passes and a median of 23.81 levels: the feedback
diverges. That is the informative part - in a chain that is already clipping, the *pre-clamp* maximum is the
value the clamp was hiding, so the grid it produces is enormous, the next hop quantises against that enormous
step, and the activations grow rather than settle.

Which means the defect is not the choice of calibration at all: **the pipeline's chain is unstable in
magnitude**, growing hop by hop, while the simulation's is stable at 0.000%. Calibration strategies only
decide how fast the instability is discovered. That is where the next investigation starts, and it is a
different question from every one asked in this stage: not "which grid" but "why does the quantised chain have
gain > 1 when the float one has gain = 1".

**Pre-clamp collection: tried, and it made things worse (74.451%).** Replacing the collect pass with a float
walk that records each layer's pre-clamp maximum read `>8 74.451% peak 90` against 1.569% for the clipped
self-calibration, and saturation rose to 23% by layer 3. The float walk quantises with a per-channel grid
derived from its own current activation while the kernel chain uses the previous layer's grid, so the two
chains diverge and the grids are fitted to the wrong one - the same class of error as the fp16 calibration,
one level down. Reverted to the committed 1.569% version.

So the fixed point stands and the honest reading is that stage 1 has no verified fix: the clipped
self-calibration converges to a clipped solution (1.569%), and the unclipped walk diverges (74%). What has not
been tried is the third measurement route - a pre-clamp running maximum written to a side buffer by the kernel
itself, which would see the chain's own distribution without any second walk.

**And the saturation counter explains why recalibration could not help: clipping is a fixed point.**

| layer | saturated after recalibration |
|---|---|
| 1 | **18.52%** |
| 2 | 8.34% |
| 3 | 9.62% |
| 8 | 0.13% |

The grid for a layer is set from that layer's **measured** maximum - but once the output clips, the measured
maximum *is* the clamp (`127 x s_out`), so the grid stops growing and the clipping reproduces itself exactly.
Self-calibration is therefore a fixed point that a clipped chain can never leave, which is why the drift
converged (10.24 -> 4.72 -> 2.59%) while the share did not move: it converged *to the clipped solution*.

The fix is to measure the distribution **before** the clamp. Two ways, both cheap: run the collect pass with
the quantisation replaced by an identity (float conv in torch for that pass, ~16 convs of fp16 - what the
simulation already does), or have the kernel write a per-channel running maximum of the pre-clamp value to a
side buffer during the collect pass. The first is a two-line change to the collect path; the second needs one
extra atomic per block.

Which means the bar's 1.57% is carried by a minority of pixels that the grid cannot fix: if twelve percent of a
layer's values sit at the clamp, the grid widening should have removed them, and the median says it did; what
survives is something else. The next measurement is the same saturation counter after recalibration - if it is
still 1-12%, the clamps are being hit for a different reason (a grid that is correct on average but too small
per channel); if it is near zero, the tail has another source and the saturation story closes.

The fix is therefore not another knob but a different calibration: fit the grids to the **chain's** activations,
not the fp16 model's - i.e. run the fp16 pass to seed a first set of grids, then recalibrate from the
quantised chain's own outputs (one extra pass per frame), or simply widen each grid by the ratio the chain
actually reaches. The measurement that decides between those two: the ratio of the chain's per-channel maximum
to the fp16 maximum, per layer, which is the amount the fp16 calibration was too tight.

**The size sweep was read wrong, and the requant precision is refuted too.**

| crop | share |
|---|---|
| 256x256 (top-left) | **0.000%** |
| 512x512 | 0.452% |
| 1024x1024 | **2.276%** |
| 1920x1080 (full) | 1.783% |

1024x1024 reads *worse* than the full 1920x1080, so this is not monotone in size and therefore not a size
effect at all: the 256x256 crop is a smooth patch and the 1024x1024 crop reaches the busy part of the frame.
The variable is **content**, and my reading of the crop as "the pipeline is correct where it was measured" was
wrong for the same reason the whole stage has been circling: the crop was not a sample of the frame, it was a
different picture.

And the integer requant precision does not matter either: `SHIFT` 12 / 16 / 20 gives 1.910 / 1.926 / 1.935%.

So on hard content the pipeline sits at ~1.8-2.3% while the simulation reads 0.000% on the same frame. That is
the defect, it is content-dependent rather than size-dependent, and every mechanical knob tried so far - grid
granularity, grid source, requant precision - leaves it where it was.

**Percentile grids: refuted.** `--pct` sets every grid from a percentile instead of the maximum, and the
full frame barely moves:

| grids from | share |
|---|---|
| max (p100) | 1.910% |
| p99.9 | 1.783% |
| p99.0 | 1.838% |
| p95.0 | 2.064% |

Flat within 15%, so the coarsening that a full-frame maximum causes is not the mechanism - the crop's 0.000%
is not bought by finer grids. That kills the only explanation offered so far and leaves the frame-size effect
itself as the open question: the same chain code passes at 256x256 and fails at 1920x1080, and nothing about
the grids distinguishes them.

Next: a size sweep (256, 512, 1024, 1920x1080) on the same chain. If the failure appears at a threshold rather
than growing smoothly, the mechanism is size-dependent (an index, a block count, the 4x tail at 3840x2160); if
it grows smoothly, it is accumulation and the crop was simply lucky. That read distinguishes the two remaining
stories in one run.

**The pipeline PASSES on a 256x256 crop and fails on the full frame, so the metric is frame-dependent.**
Same chain code, same kernel, same calibration procedure, only the frame changes:

| frame | pipeline vs fp16 | simulation vs fp16 | pipeline vs simulation |
|---|---|---|---|
| 256x256 crop | **0.000% / peak 5 / median 1.09** | 0.000% / peak 3 / 0.59 | 0.000% / peak 5 / 1.16 |
| 1920x1080 | **1.870% / peak 34** | 0.000% | - |

And on the crop the error is not a border artefact either: 0.0% of the >8 pixels lie in the outer 16 px band,
mean error 1.25 levels interior against 1.30 at the border.

So the chain is correct where it was measured and fails where the frame is 8x larger, with no localised
structure. The only thing that changes with the frame is the **calibration maxima**: `max|activation|` over
2073600 pixels is larger than over 65536, so every grid is coarser by whatever the tail of the distribution
contributes, and the quantisation error scales with the grid. That is the mechanism the earlier variants were
all fighting without naming: the grids are set by a maximum over the whole frame, and a full frame has a
longer tail than a crop.

The next step follows directly: decide the grids from a **percentile over the frame** rather than its maximum,
or from a maximum over a spread of frames, and measure whether the full frame returns to the crop's 0.000%.
That is a one-line change to the calibration and it is the first hypothesis in this stage that explains both
observations at once.

**Layer-by-layer, the chains part at the FIRST hop and drift.** Relative error of the pipeline's chain
against the simulation's, per layer, same 256x256 crop:

| layer | 1 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|
| rel error | **0.046** | 0.111 | 0.176 | 0.167 | 0.258 |

Since the kernel matches the simulation to 0.0002 when handed *identical* input, layer 1's 4.6% is entirely
in the handoff, not in the arithmetic. The difference is visible in the two recipes:

    simulation:  fp16 activation -> one quantisation onto a PER-TENSOR grid from calibration -> dot
    pipeline:    fp16 activation -> quantisation onto a PER-CHANNEL grid -> prologue requant to per-tensor

The pipeline quantises twice and the simulation once, and the grids come from different places: the pipeline
derives both from the activation's own range, the simulation takes `inputs[key]` (per-tensor) and
`outputs[key]` (per-channel) from a calibration pass on fp16 activations. Sixteen hops of a slightly different
grid is enough to turn 4.6% into 26%.

**The grids were already aligned, and the ratio is misleading.** Rebuilding with the simulation's own
`s_t = max|input|/127` changes not a digit - 0.0462, 0.0939, 0.1110 reproduce exactly - because the pipeline's
`max(per-channel grid)` is that same global maximum. So the two chains quantise to identical grids and still
differ by 4.6% at the first hop.

The reason the ratio looks alarming is the denominator: the layer's activation has `|sim| mean 0.64` while its
quantisation step is `max/127`, and on this crop the step is comparable to the mean. A relative error against
a tensor whose mean is one step wide is dominated by rounding, not by a mechanism - which is consistent with
the *final image* being nearly exact in the simulation while the intermediate ratio reads 4.6%.

So the layer-wise ratio is the wrong instrument here, and the honest conclusion is the one the final comparison
already gave: pipeline 1.870% against simulation 0.000% on the same frame, both chains quantising to the same
grids, the kernel exact on identical input. Something in the handoff still differs and the layer-level ratio
cannot localise it because its denominator is the noise floor.

What can localise it: compare the two chains' **final images** pixel-wise and look at where the pipeline's
error concentrates - if it is a thin set of pixels (as the 21-level localisation was for the single-block
case), it is a boundary or a clamp; if it is everywhere, it is a grid. That is a diagnostic with a
discriminating read, unlike the intermediate ratio.

Steps 2-6 remain blocked on this gate.

Next: build the kernel with the prologue requant - integer requant `(q * M[c] + R[c]) >> S[c]` on load, one
int32 temporary per element, no fp32 expansion - and check `ptxas -v` first: the kernel already sits at 255
registers, so a hot spill would kill the variant in a minute, exactly as the 168-register cap did.

**Second pass, still failing, and the layout is the suspect.** Per-layer isolation on 256x256 says every
layer is wrong on its own (ratio 0.12-3.7 against the fp16 activation, relative error 1.1-4.3), so it is not
the chaining. The kernel was verified bit-exact in stage 0 only on **synthetic** weights already in the
kernel's `[tap, cin, cout]` layout, so stage 0 never exercised the real conversion, and five attempts at that
conversion have now disagreed with the reference:

| conversion | vs shift reference on real weights |
|---|---|
| `permute(2,3,0,1)` | 254 on 88.8% |
| `permute(2,3,1,0)` | 203-207 on 87-90% |
| explicit `stack([w_q[:,:,i,j]])` (`[tap, cout, cin]`) | 254 on 90.5% |
| the same with `.t()` (`[tap, cin, cout]`) | 207 on 87.1% |

Both sides were agreeing with each other and disagreeing with the truth at different points, which is the
signature of hand-built layout code rather than a numerical issue. The next diagnostic is not another
permutation: build the kernel's operand **from the same tensor the reference uses** (`wtap[tap].contiguous()`
passed to both) and compare against `_int_mm` on that same pair - if they differ, the kernel's dot orientation
is not `a @ b` with `b[cin, cout]` and the PTX tells you which operand it transposes; if they agree, the shift
reference is the broken side and the pipeline's conversion is fine.

## Stage 2 - the first real numbers (half a day)

`16 * 1.97 + 6.2 = 37.7 ms` stops being a projection here.

**State (2026-11-16, third measurement): THE GATE PASSES, 1.492x idle and 1.504x under NVENC.**

The 2.00 ms came back from two places, and the smaller one was the block sweep:

| BLOCK_M | warps | ms per launch |
|---|---|---|
| **64** | **2** | **2.771** |
| 64 | 4 | 4.004 |
| 128 | 4 | 2.876 |
| 256 | 8 | 3.337 |
| 512 | 16 | 3.427 |

The recorded 1.97 ms for "BM 256 / 8 warps" does **not** reproduce with the prologue inside the kernel - that
number belongs to the bare int8 block without the requant, and at 1080p the prologue's best configuration is
BM 64 / 2 warps, not the one in the record. The sweep is worth doing rather than inheriting.

The larger part came from fusing the dequantise into the tail: one Triton kernel reads the last hop's int8 codes
NHWC and writes fp16 NCHW directly, instead of dequantising into a 530 MB float32 tensor, permuting it and casting
it to half. That is **4.00 ms** (57.92 -> 53.92), eight times the 0.49 ms the quantise fusion bought - the
asymmetry is the tell: the expensive side was the buffer round-trip, not the arithmetic.

Final gate, three interleaved rounds each, clocks logged:

| condition | fp16 engine | int8 cycle | ratio | margin under the gate |
|---|---|---|---|---|
| NVENC idle | 80.47 ms | 53.92 ms | **1.492x** | 3.56 ms |
| NVENC alongside | 91.73 ms | 60.98 ms | **1.504x** | 4.54 ms |

Accuracy unchanged by any of it: int8 vs the engine picture `>8 0.00004%, peak 8`, int8 vs the torch fp16 model
`>8 0.00001%`. `docs/claims-int8-stage2.txt` reads **PASS**. NVENC costs the engine 13.99% and the int8 cycle
13.09% - still no extra encoder sensitivity in the Triton body.

**State (2026-11-16, second measurement): the gate is taken and it FAILS by 2.00 ms.**

`scripts/int8_app_gate.py` loads the **cached fp16 TensorRT engine the app itself loads**
(`torch.jit.load` on `2x-Futsuu-Anime(1).pth_fp16_..._1920x1080.engine`), keeps both cached in one process,
encodes h264 with NVENC concurrently through a real ffmpeg loop, runs three interleaved rounds with the order
alternating, and logs clocks before and after each condition:

| condition | fp16 engine | int8 cycle | ratio |
|---|---|---|---|
| NVENC idle | 80.43 ms | 59.45 ms | **1.353x** |
| NVENC alongside | 91.92 ms | 67.23 ms | **1.367x** |

Gate 1.40x: **failed, short by 2.00 ms idle / 1.57 ms under NVENC** (`docs/claims-int8-stage2.txt` encodes that as
the shortfall, which must be <= 0). Kill 1.15x: not triggered. The engine is a valid baseline - it agrees with the
torch fp16 model to `>8 0.00000%` on the same input - and the int8 cycle reproduces the engine's picture to
`>8 0.00004%, peak 8`.

NVENC costs the engine 14.3% and the int8 cycle 13.1%, so the Triton body is **not** more encoder-sensitive than
the engine - the worry in condition 3 does not materialise.

**Instrument finding worth more than the number:** the first run of this gate read a perfectly plausible 59.72 ms
for a cycle that produced a 73.4% wrong picture. `s_act` had been left in float16, so the requant multiplier
`s_in/s_t * 2^SHIFT` overflowed half (max 65504) to inf and cast to `INT32_MAX`, and every hop became garbage -
with unchanged timing. Only the image check caught it. This is the sharpest instance so far of the rule that a
speed number without a picture number does not survive, and the reason the gate script now prints
`engine vs torch`, `int8 vs torch` and `int8 vs engine` before it starts the clock.

**Where the 2 ms is.** Fusing the quantise pass into a single Triton kernel (NCHW float in, NHWC int8 out) bought
0.49 ms - 59.54 -> 59.05. The rest sits in the body block and the dequantise: 2.88 ms per launch against the 1.97
recorded for the bare block is 14.6 ms of headroom on 16 launches. Decomposing the cycle against the fixed fp16
head and tail (5.5 + 14.9 ms, which both sides pay) puts the **body-only ratio at about 1.54x** - so the body
already clears the gate and the fp16 boundary work is what dilutes it. That is stage 6's sentence, now measured
rather than suspected.

**State (2026-11-16, first measurement): the segments exist and the boundaries are 3x their budget.** `scripts/int8_timing.py` builds every launch argument before the clock, then measures
six segments plus the fp16 baseline in the same process, in three interleaved rounds, with clocks in the log
(`2805 MHz / 8751 MHz / 49 C / 65.89 W` before, `2775 / 8751 / 57 / 143.05 W` after). Medians at 1920x1080:

| segment | ms |
|---|---|
| head (fp16 conv0 + PReLU) | 5.507 |
| quantise | **10.041** |
| 16 int8 launches | **46.007** (2.88 per launch) |
| dequantise | 6.807 |
| tail (fp16 conv34 + upsampler + residual) | 14.867 |
| full int8 cycle | 60.197 |

The plan budgeted 6.2 ms for **all** boundaries; quantise + dequantise alone are **16.8 ms**, i.e. 28% of the
60.2 ms cycle, and they are two element-wise passes over a 64-channel 1080p buffer that this build does in torch
(float round-and-clamp). That is precisely the trap the plan's condition 2 names - boundaries measured inside the
pipeline rather than as standalone kernels - only in the other direction: they are more expensive than projected,
not less. Head and tail are unchanged fp16 work and not on the critical path of this stage.

**The baseline is torch fp16, not the app's engine, so the ratio below is NOT the stage-2 gate number.** Measured
in the same session against torch fp16 convolution (cudnn benchmark on, fixed shapes):

| baseline | body | full cycle |
|---|---|---|
| fp16 NCHW | 192.176 ms | 199.184 ms |
| fp16 NHWC (channels_last, the layout the app's engines use) | 126.891 ms | 132.188 ms |

so the body is **2.758x** and the cycle **2.203x** faster in int8 - but the plan's gate is `engine >= 1.4x` against
the **fp16 TensorRT engine**, and torch fp16 conv is not that engine: 126.9 ms for 16 convs is 2.7x the 46.9 ms
that the recorded 2.93 ms-per-block implies, so a torch baseline overstates the ratio. The gate number has to come
from the app, where the fp16 engine and NVENC both exist, and it has not been taken.

**Measure separately, inside the built pipeline:** head engine, quantize, 16 launches, dequantize, tail
engine, full cycle.

Three conditions, each with its own precedent:
1. **fp16 baseline in the same session**, both engines cached, >= 3 interleaved rounds, clocks in the log,
   nothing else on the card (E22 retracted 7.6% for skipping this; drift here is 13-30%).
2. **Boundaries measured inside the pipeline, not as standalone kernels** (an isolated convolution once read
   7.90 ms against ~3 in the engine because it paid two 530 MB border reformats).
3. **With NVENC alongside**: the fp16 engine pays 9% there (51.37 -> 55.99). The Triton body's sensitivity
   has never been measured, and on material without a gate that is ~8% of wall.

**Gate:** engine >= 1.4x. Below 1.3x the boundaries or NVENC ate the prize.
**Kills it:** engine <= 1.15x - the lane does not pay for integration.

**Tools:** `scripts/probe_int8_fit.py` supplies the frames and the calibration; timing uses the app, not the
probe, because only the app has the encoder alongside.

## Stage 3 - the graph (half a day)

**State (2026-11-16, closed): the graph is BIT-EXACT, binds by POINTER, and buys nothing - stage 3 is not a lever.**

The bit-exactness failure was mine, and it was the same two-buffer bug the gate harness had. The ping-pong
`src, dst = dst, src` hands `inp` back as the *destination* on the second hop, so hop 2 writes over the source and
every run after the first starts from a previous run's output; a captured graph then re-reads that garbage as its
input. A destination cycle between the two scratch buffers - never back to the source - fixes it:

    graph vs eager, codes:  max diff 0, differing 0 of 132 710 400
    graph replay:           bit-exact on the whole body, dequantise included

With a correct harness the plan's second question answers itself:

    capture with calibration A; overwrite m/vec/os buffers on device with B; replay
    graph vs eager-on-B: max diff 0        graph vs eager-on-A: max diff 145
    verdict: POINTERS - one capture serves any calibration

So the scales are bound by address, not by value: a per-video calibration does **not** need a re-capture, and the
`set_tensor_address` concern carries over to the TRT segments only as a buffer-rotation question.

And the stopwatch still says no:

| | ms |
|---|---|
| eager | 47.25 |
| graph replay | 47.35 |
| ratio | **0.998x** |

Sixteen launches of 2.771 ms are 44.3 ms of GPU work; there is at most ~3 ms of launch overhead to remove and the
graph removes none of it. **Stage 3 closes as "the graph is not a lever at 1920x1080"** - it is correct, it is
capturable, and it is worth nothing here. It may still earn its keep in the app's render path, where the per-frame
Python around these launches is not hidden under 44 ms of GPU work, and that belongs to stage 4.

The single-hop isolation that found the bug is worth keeping as a habit: a one-hop capture on an explicit side
stream compared against eager was bit-exact while the sixteen-hop version was not, which located the fault in the
orchestration rather than in the capture mechanism.

**State (2026-11-16, first graph measurement): the stopwatch says 1.00x, so the graph is not a lever for this body.** `scripts/int8_graph.py` warms five launches
(the first enqueue does lazy initialisation), captures the 16 hops plus the fused dequantise, replays, and then
times. At 1920x1080:

| | ms |
|---|---|
| eager | 47.16 |
| graph replay | 47.47 |
| ratio | **0.994x** |

Sixteen launches take 16 x 2.771 = 44.3 ms of GPU work, so the launch overhead being removed is at most ~3 ms and
the graph removes none of it in practice - the body is **compute-bound, not launch-bound**. The plan's stage-3
premise (that launch overhead is what a graph buys) does not hold at this size.

The bit-exactness check **fails and is not yet explained**: replay is deterministic (replay-to-replay max diff 0)
but differs from eager by up to 127 codes on 56.4 M of 132.7 M elements, and the fault is at the first hop, so it is
a bind or argument issue rather than nondeterminism. Two candidate causes are named and neither is tested: Triton's
launcher using the default stream during capture (which would execute rather than record), or a temporary created
inside the capture taking its storage with it. This must be resolved before any graph number is entered in the
ledger - but it cannot change the 0.994x, which is measured with 16 real kernels executing either way.

One trap was found and fixed in the harness itself, and it is the same trap the gate script had: the ping-pong
writes over its own input, so a second run starts from the first run's last hop. The graph harness now keeps an
immutable source buffer. The first comparison read 35 M differing codes purely because of it.

The empty graph is already found. One silent trap of the same class remains.

**Scale binding: pointer or value?**
```
capture with scales A; overwrite the device buffer with B; replay; compare against eager on B
```
Matches B -> pointers, per-video calibration works. Matches A -> values baked in, capture per video (fine, once
per video) or move the scales into device memory.

**Same for the TRT segments:** `set_tensor_address` binds addresses at capture. If the app rotates output
buffers per frame, the graph writes to the captured one. Either fixed buffers under the graph plus a copy out,
or re-capture.

**Also:** the first `enqueueV3` does lazy initialisation - warm up before capturing. Keep the output Frame
construction and the sync outside the graph.

**Gate:** graph against eager, bit-exact, on the whole three-segment pipeline (16 launches alone are already
verified), then the stopwatch.

**Tools:** `/var/tmp/opencode/graph_bitexact.py` is the harness; `launch_cubin.py` has the launcher.

## Stage 4 - calibration and the probe in the render path (one day)

**The decision now lands in the render path, and porting it found two more instruments lying.**

`backend/src/utils/Int8Probe.py` is the probe the render can actually call: torch-only, with the calibration
and the quantised forward **vendored** rather than imported from `scripts/`, because `scripts/` is not shipped with
the app and a probe that imports its measuring stick from a directory that is not there fails in the field. It
returns a verdict plus both numbers; every exception path returns "no", so a probe failure costs only its own
runtime.

Two of my own bugs were caught by porting it, and both would have decided on garbage:

* the `-vf scale=W:H` on the decoder, the same one the offline tool was missing - without it the vendored reader
  takes the first `w*h*3` bytes of a 4K frame, exactly the 500x error recorded above;
* the **gate filter**. A pair only reaches the composition if the gate would reuse it (its 32x32 block-mean
  statistic has to clear the threshold). Without that filter the probe measured scene cuts: `composition 99.749%,
  peak 255` on `a5_violet` and `0.223%, peak 210` on `a5_chainsaw`, against real values of 0.102% and 0.002%.

With both in place the module reproduces the offline verdicts on every title tested:

| title | engine | composition | verdict | offline verdict |
|---|---|---|---|---|
| a5_chainsaw | 0.001% | 0.002% | int8 | int8 |
| adv_chainsaw | 0.001% | 0.015% | int8 | int8 |
| c10_p010 | 0.002% | 0.002% | int8 | int8 |
| a5_violet | 0.011% | **0.102%** | fp16 | fp16 |

`RenderVideo.setupUpscale()` calls `_probe_int8_fit()` once, right after the model is loaded and before the first
frame, and stores `self.int8Fit`. That placement is the "never switch mid-render" requirement made structural: the
verdict is computed exactly once per render. The log says both halves of the truth - the probe's answer, and that
the int8 body is **not wired into the render path yet**, so the fp16 engine runs this pass. A log that only said
"int8" would be a promise the render does not keep.

**State (2026-11-16): the probe runs, it discriminates, and it had been reading mangled frames.**

`scripts/probe_int8_fit.py` assumes 1920x1080 input, but these clips are **3840x2160**: it took the first
`w*h*3` bytes of a 4K frame and reinterpreted them as 1080p. That is not a short frame, it is a scrambled one, and
it read **4.0-4.2% engine error on every title** - 500x the same clip's real number. First run of the fixed reader
on one frame of `a5_chainsaw`: simulation **0.00858%**, build **0.00616%**. One missing `-vf scale=W:H` would have
returned P(pass) = 0 and killed the lane.

With the reader fixed, the probe separates the two statistics the plan says must be separated:

| title | engine (int8(t) vs fp16(t)) | composition (int8(t-1) vs fp16(t)) | verdict |
|---|---|---|---|
| a5_chainsaw | 0.009% | 0.010% | int8 |
| adv_chainsaw | 0.010% | 0.022% | int8 |
| c10_p010 | 0.010% | 0.011% | int8 |
| a5_violet | 0.057% | **0.170%** | fp16 fallback |
| adv_violet | 0.058% | **0.165%** | fp16 fallback |

**P(pass) = 3 of 5 readable titles = 0.6**, against a kill threshold of 0.3 - the lane survives, and it survives
as an opt-in, which is what the plan's kill rule describes.

The Violet pair is the useful part: both pass on the engine's own defect (0.057%, 0.058%) and fail on composition
(0.170%, 0.165%). Certifying on the engine's own error would have shipped them. That is the plan's warning
measured on real titles rather than argued.

Five `bebop*` titles produced no output from the probe (decoder refused them); they are **not** in the denominator
above, and that is a hole in the sample rather than a pass.

**Scales:** the head's input is the domain constant [0, 1] and needs no calibration. The rest come from
spread frames plus deliberately bright ones (p99.9 LR luma), ~1 s of forwards, reusing the frames the probe
already decodes.

**Certification must be compositional, not own-defect.** Measured: the tuner reads int8(t) against int8(t-1)
= 0.242% where the bar measures int8(t-1) against fp16(t) = 0.146%. On a **static** pair int8(t) == int8(t-1),
so the tuner reads zero and is blind to the engine's error precisely on the frames it certifies. Both engines
on the 16 calibration frames, statistic against fp16; own defect stays as a cheap pre-filter.

**Probe:** decides int8/fp16 **before the first frame**, never switches mid-render (the gate's HR history would
mix two engines).

**Gate:** P(pass) over 8-10 digital titles. Locally 1 of 2, an interval that means nothing. The probe needs no
renders - ~15 s of GPU for ten titles.

**Kills it:** P < 0.3 - int8 becomes a narrow opt-in and `T > 4.2/P` stops paying on ordinary content.

**Tools:** `probe_int8_fit.py` already applies the gate filter and answers per clip; `RenderVideo.py` is where
the decision must land.

## Stage 5 - shapes and models (half a day)

**State (2026-11-16): the sweep names the coverage boundary, and it is narrower than the plan assumed.**

| model | feat | scale | shape | image vs engine | fp16 engine | int8 cycle | ratio |
|---|---|---|---|---|---|---|---|
| 2x-Futsuu-Anime(1) | 64 | 2 | 1920x1080 | 0.00004% | 80.47 ms | 54.03 ms | **1.489x** (pass) |
| realesr-animevideov3(1) | 64 | 4 | 1920x1080 | 0.00058% | 87.60 ms | 64.27 ms | **1.363x** (misses by 1.70 ms) |
| Archivist_AntiLines | **48** | 1 | 1440x1080 | - | - | - | not covered |
| 2x_ModernSpanimationV1 | - | - | 1440x1080 | - | - | - | not covered |

Two coverage walls, both structural rather than tunable:

* the kernel's channel count is a `tl.constexpr` of 64, so a 48-channel model (`Archivist_AntiLines`) cannot use
  it at all;
* `2x_ModernSpanimationV1` has no `body.N.weight` keys - it is not a Compact architecture - so the sixteen-hop
  assumption does not describe it.

And the gate itself is **model-dependent**: the same body on the 4x model reads 1.363x against the 2x model's
1.489x, because the 4x tail and upsampler are four times the area and both sides pay them. So "int8 pays" is a
statement about a model and a shape, not about the lane - the same shape of answer the probe gives per video.

Claims: `docs/claims-int8-stage5.txt` fails on exactly the 4x shortfall, which is the finding.

## Stage 5 - the original plan text



| what | why |
|---|---|
| 1440x1080 | grid 6075 at BM 256 - divides, but verify |
| arbitrary resolutions | a partial tail block needs the masked loads, which are untested at the edge |
| x4 | tail 64->48 and a 199 MB fp16 output against 49.8 - different kernel, different grid |
| the 78-model collection | the kernel covers SRVGGNetCompact 64ch; SPAN (11 files) has `Conv3XC` summing a 1x1 into a 3x3; RRDB is 12 channels |

**Gate:** a list of covered models with their grid, and fp16 fallback for the rest. Without it "it works" means
"it works on one clip".

**Tools:** `scripts/rebuild_width.py` and `force_tactic.py` already build synthesized variants per width;
`scripts/residual_scale.py` and `cut_channels_naive.py` show how to instantiate the architecture.

## Stage 6 - own head and tail, only after stage 2

**State (2026-11-16, first half): the body is in the app and wired into the upscale path, and it REFUSES to
serve. No int8 frame has been rendered, and the 1.489x has NOT been confirmed inside the render.**

`backend/src/pytorch/Int8Kernels.py` vendors the three kernels (the app does not ship `scripts/`), and
`backend/src/pytorch/Int8Body.py` is a drop-in inference helper: clamp -> head conv + PReLU in fp16 -> sixteen
int8 hops -> dequantise to fp16 -> the model's own `tail_conv` and `upsampler` -> plus the residual.

`UpscaleTorch._call_single` consults `_int8_body()`, which calibrates on the **first frame of the render** and then
runs `certify()` - the body must reproduce the fp16 model on that same frame within the bar before it serves a
single pixel. Straight-line timing was already banked from the plan's stage 2 (1.489x idle, 1.505x under NVENC,
0.00004% picture agreement) so the guard is what decides whether that win is real in this render.

It is not, yet:

| frame | Int8Body vs the fp16 model |
|---|---|
| 256x256 | **0.00000%** (peak 8) |
| 512x512 | 1.22709% |
| 1920x1080 | **3.53381%** |

against `scripts/int8_pipeline.py` reading **0.00000%** on the same 1920x1080 frame. So the app-side body differs
from the validated pipeline in something the pipeline gets right, and the difference grows with resolution - the
same signature the pipeline had before the rounding-shift fix, which means the body is reproducing older
behaviour, not new behaviour.

One real bug was found and fixed on the way here, and it is the third instance of the same class: `_measure()`
returned the calibration grids in **float16** because the model runs in half, so the requant multiplier
`s_in/s_t * 2^SHIFT` overflowed half to infinity and cast to `INT32_MAX`. The first certification read 99.58%.
Keeping the grids in float32 took it to 3.53%; the remaining factor is unexplained.

**The hop-by-hop comparison was run, and it exonerated every part while the composite still disagrees.**

Five separate comparisons, all in one process, all on the same 512x512 frame:

| component | result |
|---|---|
| activation grids (`s_act`) | **identical** (`torch.equal`) |
| launch parameters, all six per hop, with the same grids | **identical** (no field differs at any of the 16 hops) |
| `quantise_nchw_to_nhwc` vs torch round/clamp/cast/permute | differs on 0.0023% of elements by **1 code** (half-way rounding) |
| `dequantise_nhwc_to_nchw` vs torch dequantise | **bit-identical** (max diff 0.00000000) |
| both tails from the same codes | **bit-identical**, and both read 1.23053% |

And yet, in that same process, with the same grids and the same launches:

| | share at 512x512 |
|---|---|
| the pipeline's recipe, written out inline | **0.00000%** (median 0.36) |
| `Int8Body(x)` | **1.27525%** (median 1.57) |

So the defect is not the grids, not the launches, not the kernels, not the conversions and not the tail - it is in
`__call__`'s own assembly. The one structural difference left untested is that `__call__` reuses two ping-pong
buffers while the recipe allocates a fresh destination per hop; the next probe is to make `__call__` do exactly what
the recipe does (fresh buffer per hop) and see whether the number collapses. Everything else on the list above is
now measured rather than assumed, which is why this is a two-line experiment rather than another sweep.

Next probe, named rather than guessed: run `__call__` with a per-hop destination and compare, then bisect what is
left of the assembly. Until that reads 0, the
render keeps the fp16 engine, because enabling this body would trade a real 1.489x for a wrong picture and the
picture is the thing the bar measures.

## Stage 6 - the original text


Prize is fixed: **+7% of wall on anime** (1.360x against 1.271x), stable against whatever the boundaries eat.
Floors 0.50 and 0.68 ms against the current ~3.9 and ~3.35.

Cost: four new correctness surfaces at once - clamp, edge padding, PixelShuffle, residual gather. Doing that on
top of an unvalidated launcher means six suspects when the comparison fails instead of two.

**Decide by stage 2's number, not before it.**

## Verify explicitly, because these are silent

1. **Clamp before quantisation** - the [0, 1] scale assumes a clamped input.
2. **Zero padding at frame edges** - masked loads must give zeros where torch gives zeros, tested with content
   at the very border.
3. **Slope slicing** - 1088 = 17 x 64: head is index 0, body 1..16. A wrong slice gives a plausible, wrong
   picture.
4. **Scale order** - `acc * s_act * s_w[k]` -> PReLU -> `* 1/s_out[k]` -> clamp [-128, 127], matching the
   simulation literally including clamp bounds.
5. **TRT buffer addresses under the graph** - see stage 3.

All five are caught by the bit-exact comparison; checking the PTX and the code before running is cheaper than
debugging the difference afterwards.

## Before a number enters the ledger: `scripts/claim_check.py`

Nine conclusions in this investigation were wrong the same way - a number computed on one population, scale
or column and applied to another. That class is mechanical, so it is now checked mechanically. Claims go into
a plain text table next to the measurement:

    @claim  name  value  unit  "population"
    @ratio  numerator  denominator  claimed  ["population"]
    @below  name  limit          # the value must sit under its bar
    @sum    X%  @of  a + b       # shares that must add up

The checker fails on: arithmetic that does not reproduce, unit families that differ (share vs delta, TOPS vs
TFLOP/s), a ratio whose two operands were measured on different populations, a ratio whose declared population
is neither operand's, and a value that sits above its own bar. Exit code 1, so it can gate an edit.

Its first run on real claims from this session caught seven issues, including the published Chainsaw margin
(an 18.6 pp claim that is 0.280 of the reuse percentage it was derived from, in a different unit family) and
the cross-frame TOPS comparison (67.6 at 1080x1080 against 77.6 at 1920x1080). The fixture is
`docs/claims-example.txt`; it fails on purpose and is the self-test.

Rule: a measurement is not a result until its claims file passes.

## Machine notes (tools, paths, rules)

- **One GPU job at a time**: wrap every run in `flock -n /var/tmp/opencode/gpu.lock`.
- **Interpreter**: `/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/python/python/bin/python3`.
  The repo has no venv; `pip install` into the shared environment is forbidden.
- **Scratch**: `/var/tmp/opencode` only. `/tmp` is tmpfs and **was wiped by a reboot this session**, taking the
  model copies and the probe scripts with it - the frames under `/var/tmp/opencode/titles` survived.
- **Models**: `/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/custom_models/` (the `/tmp/opencode/models2`
  copies are gone).
- **Delegation**: `agent-clone.sh "/home/jorj/Загрузки/projects/REAL-Video-Enhancer" <task-id>` accepts a path, but
  this repo is not one of the three guarded ones - no `pre-push` hook, no repo guard config, so "never push
  `main`" rests on the prompt and on my check of the branch tip. Cells cannot run the model (no weights inside
  the jail, no bundled interpreter), so they write kernels and unit-test them offscreen; every measurement is
  the supervisor's.
- **Reporting rule**: before a number is called a result, name the population it was computed on and the
  population the decision applies to. Nine conclusions in RESEARCH-LEDGER were wrong the same way.
