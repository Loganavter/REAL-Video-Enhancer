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

The next test is therefore not another kernel variant but a **grid alignment**: rebuild the pipeline's chain
using exactly the simulation's grids - `s_t = inputs[key]/127` for the prologue target and
`s_out = outputs[key]` for the write - and see whether layer 1's 4.6% collapses. If it does, the pipeline was
correct all along and only disagreed about which numbers to quantise to; if it does not, the difference is in
the double quantisation itself, which is a design constraint rather than a bug.

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
