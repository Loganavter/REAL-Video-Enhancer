# Investigation: what "temporal reuse" actually costs in picture

Date: 2026-11-15. Rig: RTX 4060 Ti 16 GB, TensorRT 10.12, torch 2.9, one GPU job at a time.

## Why this started

Two features were added to speed up rendering: a duplicate-frame gate (re-emit the previous
upscaled frame when the current one has not changed) and tile-level reuse (re-run the model only
for the tiles of a frame that changed). Together they took 502 anime frames from 38.85 s to
7.68 s, and the numbers used to justify them were PSNR against an ungated render, quoted as
"52.7 dB average, 48.2 dB worst frame, invisible".

The owner then looked at the worst frame of the gated render and found a fist in a different pose
than in the ungated one. That report is the only reason any of the following exists.

## What was tried, in order, and what each step ruled out

### 1. A/A control (the step that should have come first)

Two renders, same input, same flags: **502 of 502 frames byte identical**. The pipeline is
deterministic. Without this, nothing below means anything; with it, every difference between
configurations is real and measurable.

### 2. Global alignment

Frame *i* of one render against frames *i-2..i+2* of the other. The peak is at offset 0 in both
clips (anime 55.18 dB at 0 against 52.53 at -1 and 53.69 at +1; film: identical frames dominate at
0). **Not a one-frame time shift.**

### 3. Patch probe at the reported region

PSNR of a 128 px patch is the same at every offset from -5 to +5 (31.7 across the board), i.e. the
content matches no neighbouring reference frame either. So it is not "one frame late" locally
either. Absolute value matters too: 31.7 dB in that patch is ~6.6 levels of average error.

### 4. Tiles against the full-frame render (same 60-frame clip, gate off)

| halo | average | worst frame |
|---|---|---|
| 18 LR px | 51.34 dB | 49.84 dB |
| 24 | 51.36 | 49.94 |
| 32 | 51.36 | 49.94 |

A 2D shift search of the tile render against the full-frame one peaks at (0,0) with 49.53 dB
(±1 px gives 40.28, ±2 px 35.15). The error at tile boundaries (0.496 levels) is indistinguishable
from inside the tiles (0.477). Frame peak 9 levels, mean 0.478.

So: **the tile path is geometrically exact**, its halo of 18 is not a limitation, there are no
seams, and its difference from a full-frame render is fp16-level noise. Two plausible theories
(hole in the halo, seam at the border) died here.

### 5. Chain depth

Same frame 383, share of pixels by error magnitude, 8-bit levels:

| threshold | chain 4 (shipped) | chain 1 |
|---|---|---|
| > 2 | 10.87% | 8.91% |
| > 4 | 1.13% | 0.39% |
| > 8 | 0.29% | **0.00%** |
| > 16 | 0.12% | **0.00%** |
| > 32 | 0.01% | **0.00%** |
| worst 64 px block | mean 7.57, peak 100 | mean 2.29, peak **12** |

The film clip's worst frame, same measurement at chain 4: > 2 levels on 20.31% of pixels, > 4 on
5.88%, > 8 on 2.35%, > 32 on 0.40%.

At this point the artifact looked like the chain: the gate may serve a frame from a reference up
to `max_chain` frames old, and four frames is long enough for a fist to move. That turned out to be
half the answer - the tile path has its own staleness and, as the next section shows, the two mask
each other.

## The two measurement mistakes underneath it

**The tuner scored the wrong quantity.** `AutoGateTuner` measured the upscaled frame against its
*immediate predecessor* and picked a threshold from a 2-level budget. The gate, however, serves a
frame from a reference up to `max_chain` frames old. On static anime the two agree; on moving
content the four-step error is an order of magnitude larger (anime: one-step 1.65 levels, four-step
9.26). It now records one delta per chain depth and certifies the depth itself, with the same
strict budget at every depth, and a candidate that certifies nothing falls back to "do not reuse"
in the quality-first preset.

**Whole-frame metrics cannot see this defect.** The same pairs, PSNR and VMAF:

| pair | PSNR avg | PSNR worst | VMAF mean | VMAF min |
|---|---|---|---|---|
| anime, gate + fitted tiles, chain 4 | 51.71 | 48.64 | 95.50 | 93.65 |
| film, gate, chain 4 | 50.34 | 39.83 | 96.70 | 83.79 |
| P010 fixed vs a libx265 render | 52.01 | 50.40 | 96.24 | 95.63 |

The film row has the *best* mean of the three and a minimum 10 points below the others, and the
anime row above was the one a viewer rejected over a moved fist: a fist is a fraction of a percent
of a 4K frame, so both averages and the 1080p VMAF min pass it. The number that tracks the eye is
local and absolute - what share of pixels is off by more than how many levels, and what the worst
block looks like.

## What else came out of the same session

**The 10-bit pipe was feeding ffmpeg the wrong format.** `_pipe_input_pixel_format()` derived the
pipe format from the encoder tag ("NVENC wants P010") while the packer emits planar
`yuv420p10le` unless `RVE_P010` is set. Every 10-bit source rendered with NVENC therefore handed
ffmpeg planar samples labelled P010: **12.68 dB** against a reference, silently, because both are
16-bit words at 3 bytes per pixel. The command now declares what the packer produced. With
`RVE_P010` and a consistent declaration the same content measures 52.01 dB at 10-bit output.

Consequence for previous numbers: the film quality figures in this repo's docs (71 dB "vs ungated")
were two equally corrupted renders agreeing with each other. They are replaced.

**10-bit output is not free.** Keeping 10 bits at the encoder buys ~6 dB (52.01 against 46.11) and
costs about twice the NVENC wall clock on this machine (8.01 s against 3.84 s for 94 frames of
2160p), so it is opt-in.

**Kernel-level work is closed by measurement.** Four engine builds of the same ONNX (explicit 8 GB
workspace, baseline, cuDNN-only, JIT-only) with three interleaved timing rounds each:

| variant | median ms/frame | vs baseline |
|---|---|---|
| baseline | 50.09 | 1.000 |
| workspace 8 GB | 49.84 | 0.995 |
| cuDNN only | 55.43 | 1.107 |
| JIT convolutions only | 49.97 | 0.998 |

An explicit workspace changes the tactic mix (one tail conv moves from sm80 to sm75) and buys
nothing; restricting to JIT sources picks the same `sm75` tactic, so no sm80/86/89 alternative is
being skipped; cuDNN alone is slower, which does not reproduce the Winograd counter-precedent.

**Grain does not stop frame reuse, it stops tile reuse.** Clean anime with synthetic grain
(`noise=alls=S:allf=t+u`), gate on: the reuse share stays at 70.9% from clean to grain 16 because
the statistic is a 32 px block mean and grain averages out inside a block; the quality cost grows
(52.14 -> 42.10 dB) and the ungated render jumps from 17.6 s to 36.6 s because the tile probe
refuses tiles past a noise floor of ~0.45.

## Rules adopted

1. **Acceptance is local, in RGB, and on the worst frame of the clip** - three qualifiers that each
   cost a wrong conclusion to learn:
   - **local**: a fist is a fraction of a percent of a 4K frame, so PSNR (mean or worst) and VMAF at
     1080p pass a frame a viewer rejects. The number is the share of pixels more than 8 levels off.
   - **in RGB, not luma**: the moved fist measures 0.29% by channel maximum and 0.036% in luma,
     because the error lands in chroma; a luma-only rule passes exactly the frame this whole
     investigation exists for.
   - **worst frame of the clip, not one frame**: a configuration that looked 30x better on frame 383
     was 4x worse on frame 360.

   The bar is one number: **under 0.1% of pixels above 8 levels (RGB) on the worst of a spread
   sample**. Measured against it: 0.045% for the shipping configuration, 0.045% for no reuse at all,
   0.287% for the rejected tile configuration, 1.174% for the worst-block tile judge. A peak under
   20 levels accompanies a pass. VMAF (min >= 90) is reported as a second opinion and is explicitly
   **not** sufficient - the rejected configuration scores min 93.65.
2. **One GPU job at a time** means no other render or benchmark; `nvidia-smi` utilisation is not a
   usable idle check on this host (17-47% with nothing of ours running).
3. **Ratios only within a session.** The same ungated configuration measured 30.8 s and 38.85 s on
   different days; the difference is the machine, not the code.
4. **A/A before A/B.** If two identical runs disagree, the measurement is the problem.
5. **The tuner certifies the depth it can prove**, and refuses rather than reusing "the least bad"
   when nothing certifies.
6. **A pipe format must be declared by the packer, never inferred from the encoder.**

## Colour tags on the wrong side cost 62 ms/frame and five levels of chroma

Found while checking why a 10-bit render looked twice as slow as an 8-bit one of the same clip.
Bisecting the exact command the app logs, at 2880x2160 10-bit:

| command variant | ms/frame |
|---|---|
| input section + encoder | 16.6 |
| + second input, `-map`, audio/subtitle copy | 15.0 |
| + `-colorspace bt709 -color_range tv` on the **output** | **79.0** |
| the same tags on the **input** | **16.6** |
| no tags at all | 15.3 |

`-colorspace` after the input tells ffmpeg the *output* must be that space, so it inserts a
swscale matrix conversion on every frame. Before the input it describes the raw pipe and no
conversion happens. The container tags are identical either way (`color_range=tv`,
`color_space=bt709`), so this was 62 ms/frame - 4.8x the encoder's own cost - paid for a metadata
declaration.

And it was not only time: mean chroma against the source, 10-bit scale (0..1023), first 20 frames:

| stream | U | V |
|---|---|---|
| source | 468.96 | 565.72 |
| tags on input (now) | **468.88** | **565.51** |
| tags on output (before) | 474.22 | 563.61 |

The old path was converting already-bt709 samples as if they were bt601, which moved U by 5.3
levels. So every 10-bit render on the NVENC path had a chroma shift, and the two renders differ by
38 dB between themselves - a difference one would otherwise be tempted to attribute to the model.

Consequence for the defaults: 10 bits at the encoder is free now (4.67 s against 4.64 s on the
film clip), so a 10-bit source keeps its depth unless asked otherwise, and the "10-bit is opt-in
for speed" row in PERFORMANCE.md is withdrawn.

## The statistic has to be a box filter over |difference|, not a box compared across frames

The gate compared `max |box(I) - box(I_ref)|`. That is invariant to a permutation of pixels inside
a block, so a fist changing pose inside one 32 px block barely moves it - the same blindness as the
`|mean(a) - mean(b)|` error that was already fixed once at the tile level. Swapping the order of
operations to `max box(|I - I_ref|)` costs one subtraction, one abs and one pool per frame
(**+0.1-0.3 ms**, gate 4.64 against 4.31 ms) and changes the numbers:

| statistic | threshold | reuse | share above 8 levels (worst reused frame) |
|---|---|---|---|
| `max |box - box|` | 2 | 70.2% | 0.004% |
| `max |box - box|` | 4 | 70.2% | 0.004% |
| `max |box - box|` | 8 | 70.2% | **0.460%** |
| `max box(|diff|)` | 2 | 70.2% | **0.000%** |
| `max box(|diff|)` | 4 | 70.2% | 0.004% |
| `max box(|diff|)` | 8 | 70.2% | 0.004% |

So the new statistic is both more accurate at the same reuse and, more usefully, its safe range is
three times wider: the old one falls off a cliff between 4 and 8, the new one is still fine at 8.
That margin is what the tuner actually needs - the failure this investigation started with was a
threshold that had drifted to 12.

Whole-clip effect, 19 sampled frames, RGB, chip clip: worst frame goes from **0.045% / peak 13** to
**0.009% / peak 14**, at 11.7 s against 11.3 s (3.7% slower for a 5x smaller defect area). The
scene-cut logic still runs on the level grid, which is why `observe()` now takes the decision grid
separately: the error field is already relative to the reference and must not be differenced twice.

## The two staleness sources mask each other

The chain-depth conclusion above is only half the answer, and the way it was reached hid the
other half. Reuse has two independent staleness sources: the frame gate (a whole frame served
from a reference up to `max_chain` frames old) and the tile path (single tiles served from the
same kind of reference, judged by their own mean). Measured in the metric that matches a viewer -
share of pixels more than 8 levels off in any RGB channel, worst of 19 sampled frames:

| configuration | median | worst frame | peak | time |
|---|---|---|---|---|
| tile reuse off, gate threshold 4, chain 4 | 0.001% | **0.045%** (f480) | **13** | 11.3 s |
| tile reuse off, gate threshold 12, chain 4 | 0.001% | 0.517% (f370) | 95 | 10.9 s |
| tile reuse on, gate chain 1 | 0.002% | 0.105% (f390) | 20 | 11.5 s |
| tile reuse on, gate chain 4, threshold 4 | 0.025% | 0.287% (f383) | 98 | 9.2 s |
| tile reuse on, gate chain 4, threshold 12 | 0.024% | 0.288% (f383) | 100 | 8.4 s |
| tile judge = worst block, threshold 2 | 0.024% | 1.174% (f360) | 42 | 8.9 s |

Read down the columns and the trap is visible: **with tile reuse on, the gate's own threshold
makes no difference at all** (0.288% against 0.287%), because the tile artifact is an order of
magnitude larger and masks it. Turn tile reuse off and the same gate threshold becomes a tenfold
difference (0.517% against 0.045%). A quarter of a day went into "the threshold does not matter"
and "chain depth is the cause" conclusions that were both artefacts of measuring two sources at
once.

What each knob actually costs, isolated:

- **threshold 12 -> 4** (tiles off): 0.517% -> 0.045%, peaks 95 -> 13.
- **chain 4 -> 1** (tiles on): 0.288% -> 0.105%, peaks 100 -> 20.
- **tile reuse on -> off** (threshold 4, chain 4): 0.287% -> 0.045%, peaks 98 -> 13, at the price of
  9.2 -> 11.3 s.
- **tile judge mean -> worst block**: worse, 0.288% -> 1.174%. The worst block fires on grain and
  takes neighbouring tiles out of sync frame to frame; it moved the artifact instead of removing
  it.
- **tile threshold 0.35 -> 0.10**: reuse collapses from 57.7% to 3.0% and the render becomes
  *slower* than not using the tile path at all (13.3 s against 11.4 s), so the tile statistic
  cannot be tightened into safety - it either pays for itself with the artifact or pays nothing.

## Result

| configuration | 502 anime frames | worst frame, RGB > 8 levels |
|---|---|---|
| ungated, full frame | 38.85 s | reference |
| **default now: gate (thr 4, chain 4), no tile reuse** | **11.3 s** | **0.045%, peak 13** |
| gate, chain 4, tile reuse | 9.2 s | 0.287%, peak 98 |
| gate, chain 4, tile reuse, threshold 12 | 8.4 s | 0.288%, peak 100 |
| tile judge = worst block | 8.9 s | 1.174%, peak 42 |
| film, any of the above | 4.3-4.6 s against 4.66 s ungated | the gate refuses at `best`; tile reuse off |

`best` is the default and takes the certified depth - chain 4, once the chain budget is expressed
on the proxy's own scale instead of the one-step bar; `fast` keeps the full chain unconditionally
and pays the moved fist when tile reuse is also on. Tile reuse is off by default: it buys 2.1 s of 11.3 and costs six times the defect area, and
its per-tile mean cannot be made safe without giving the win back entirely (`--tile_reuse_mode auto`
and `on` stay available for material where the owner accepts the trade). The honest headline is
**3.4x** (38.85 -> 11.3 s); 5.06x was the tile-reuse number, i.e. the configuration that moves the
fist.

Caveat on this whole framing: "the last 1.5x is the price of not moving objects" describes a
*decision rule*, not a property of the problem. The gate reduces its per-pixel difference into a
32 px block mean, which is exactly the reduction that also averages grain away - a fist of ~50 px
survives a block mean almost intact, so the statistic cannot see what it is being asked to police,
and the remedy (a shorter chain) is applied to the whole frame although the defect is one tile. A
box-filtered intermediate statistic is being measured precisely to see whether that 1.5x can come
back without the artifact; the rules below are therefore provisional.

## Still open

- **INT8 (explicit Q/DQ)** is the only remaining numeric lever with 2x in its arithmetic, and it now
  has an acceptance rule to be judged against. `onnxruntime` cannot be installed (no DNS for PyPI
  or GitHub on this host); calibration can be done with torch hooks on the real model instead.
- **The input conversion costs 2.80 ms/frame** (19% of the gated path): one H2D plus roughly
  fifteen kernels that each materialise a full-resolution intermediate, against a ~0.1 ms
  bandwidth floor. Two candidates are recorded (planar `gbrp` from the decoder, and one
  preallocated buffer for the 10-bit conversion); neither is measured.
- **Film does not justify the gate as it stands.** 1.08x (4.33 against 4.66 s) for 2.35% of pixels
  above 8 levels is a bad trade, and `best` refuses it automatically; `fast` still offers it. Either
  the local-statistic work above makes film cheap too, or film simply runs ungated.
- **The tile probe reads 16 frames** and mispredicts in both directions (anime 33% predicted
  against 58% achieved, film 15% against 9.5%), held together by a margin rule. Needs two or three
  more titles with different grain.
- **The GUI never passes any reuse flag**, so its renders inherit the backend defaults; the quality
  preset is being wired into the window separately.
