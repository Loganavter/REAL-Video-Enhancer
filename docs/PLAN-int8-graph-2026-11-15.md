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
