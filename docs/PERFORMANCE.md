# Performance notes (this fork)

Measured on: RTX 4060 Ti 16 GB, Ryzen 5 5600 (6c/12t), TensorRT 10.12, torch 2.9, on a
1440x1080 10-bit HEVC clip (94 frames) and a 40 s excerpt of the same film. Numbers are
wall-clock for the whole render unless stated otherwise.

## What is on by default

| knob | default | what it does | measured |
|---|---|---|---|
| `--tensorrt_opt_profile` | 5 (was 3) | TRT builder optimization level | engine only: 94.3 -> 52.7 ms/frame (1.79x) |
| `--trt_timing_iters` | 8 (was 4) | tactic timing iterations | included above |
| `--trt_tiling_level` | `full` (was `moderate`) | keeps per-layer tiles in L2 across consecutive kernels | `moderate` -> `full` measured on anime, two runs each: 11.01 s -> 10.60/10.59 s (3.8%) |
| `--quality_preset` | **best** (also exposed in the window: "Encoder Options" → *Quality preset*, `Best quality` / `Fast`) | decides how stale a reference the gate may serve. `best`: the tuner certifies the **chain depth** (a p90 budget at every depth the gate can reach) and, when nothing certifies, switches reuse off for the whole render. `fast`: keeps the full chain and reuses as little as the threshold range allows | anime `best`: chain 4, **11.7 s**, **0.009%** of pixels above 8 levels (RGB) on the worst of 19 sampled frames, peak 14; two digital BDs: chain 2, **6.61 s against 13.50 s (2.04x)** at 0.005% and **6.90 s against 13.31 s (1.93x)** at 0.096%; film `best`: refuses to reuse, and the result is **byte-identical** to an ungated render (94/94 frames, 4.67 s against 4.63 s) - the refusal is honest, not a partial one. The chain stops at 2 on a 16-frame calibration window because deeper depths run out of *reachable* samples (a depth-`k` reuse needs `k-1` consecutive passes), and forcing 4 by hand buys only 12% on the faster title while pushing the other past the bar |
| `--temporal_reuse_threshold` | chosen by the tuner (seed 5.0) | max 32 px-block mean delta vs the last KEPT frame; the tuner scores every candidate at every chain depth the gate can reach | anime 362/498 reused; two digital BDs 146/220 and 142/219; the film and 35 mm clips **refuse** on the noise floor (they used to read 18.1%) |
| `--temporal_reuse_max_chain` | 4 (was 2) | cap on consecutive reuses | |
| `--no_temporal_reuse_auto` | off (i.e. auto is ON) | automatic threshold and chain depth: during the first 16 rendered frames the gate records its own LR delta paired with **the acceptance statistic itself** - the share of pixels more than 8 levels off in any RGB channel, between the frame and each candidate reference - and picks the deepest chain and widest threshold whose **worst** share stays inside the 0.1% bar. A block-mean p90 (2 levels at depth 1, 12 deeper) is still measured and logged, but it no longer decides: certifying one instrument while judging by another is what produced every false "price of quality" in the ledger. Two caveats, both measured: a depth is only scored where the gate could reach it (serving a `k`-frame-old reference requires the reuse of the frames in between), and the calibration sample reads lower than the whole clip - Violet Evergarden certifies at 0.02% on 16 calibration frames and measures 0.096% on the clip, a factor of ~5. **Widening the window does not fix that, and it is not a proven improvement either way**: the same clip at 48 calibration frames reads chain 1 (117/219) at 0.097% on the clip, against chain 2 (142/219) at 0.096% on 16 frames - more conservative, not more accurate, and its depth-2 sample values are contaminated by scene cuts (99.69% of pixels differ, which is a cut and not a reuse). The 16-frame window stays because it reproduces. The sample number is an estimate with a ~5x bias measured on one title, not a guarantee; a tighter sample bar (~0.05%) would change nothing for any configuration measured so far - the bar needs a *failing* case on the fixed code to be calibrated, and that is the open question rather than a proposed number. During those 16 frames the gate is forced to render, so the cost is ~0.7 s once per video. The threshold is clamped to 2..4/255 (`--temporal_reuse_threshold` can narrow it) and reuse is refused outright when the measured noise floor is above **1.5**. The limit **separates the last passing measurement from the first failing one; it is not a calibrated boundary**: 0.95 -> 0.005% of pixels above 8 levels at the worst frame, 1.32 -> 0.097%, 1.84 -> 0.105%, 2.27 -> 0.470% at every threshold. The bar is crossed between 1.32 and 1.84, so the two cases straddle 1.5 with 0.008% of defect between them, and a limit of 2.0 (the first value) shipped a configuration that measured 0.105%. **The two guards share a failure mode**: a 48-frame Violet Evergarden run passes this floor (1.32) *and* the acceptance sample, and still lands at 0.121% on the clip | |
| `--temporal_reuse_pixel_veto` | 0 (off) | full-res max delta that rescues a would-be duplicate. 40 is TheAnimeScripter's value and is right for clean anime; on grainy film a single pixel exceeds it and reuse collapses to **1.1%** (measured: 43 vetoes in 94 frames). The "from 46.8%" this row used to carry is the withdrawn figure from the C4-corrupted pairs | |
| 10-bit path | auto for >=10-bit sources | YUV420p10 packed on the GPU, colour matrix from the source, and the ffmpeg command declares the format the packer actually produced | colour parity vs a libx265 render: 46.11 dB at 8-bit output, 52.01 dB at 10-bit output |
| output depth | **the source's depth** | a 10-bit source now stays 10-bit, and it is free: 4.67 s against 4.64 s for 94 frames of 2160p, ~6 dB better. The old "10-bit costs 2x NVENC" was measured and is wrong - the encoder costs 1.47x (9.4 against 6.4 ms/frame) and the rest was swscale, inserted because the colour tags were declared on the *output* side. Declared on the input side they produce the same container tags with no conversion, and the chroma now matches the source to 0.2 levels where the old path was 5 levels off | |

End-to-end on the clip with `x265_nvenc` (so the encoder does not hide the model):
**6.26 s (old defaults) -> 4.27 s (new defaults) = 1.47x**. The gate on top of that, measured on
502 anime frames at 1080p -> 2160p: **38.85 s ungated -> 11.7 s gated = 3.3x**, with 70.9% of
frames reused and **0.017%** of pixels above 8 levels on the worst of 13 sampled frames (an earlier
19-frame sample of the same configuration read 0.009%; both are inside the bar, and the difference is
which frames were sampled, not which configuration ran). (Earlier
revisions put "62% frame reuse -> 3.84 s" here; 3.84 s is the 94-frame 2160p clip at 8-bit output
from the depth comparison in the investigation, not the 502-frame run.)

### One instrument decides, the other is a log line

Two measurements exist side by side in this file, and until recently the tuner decided on the one the
pictures are *not* judged by:

- the **acceptance statistic** - the share of pixels more than 8 levels off in any RGB channel, on
  the worst frame of the sample, bar **0.1%**. This is now what the tuner certifies against, computed
  on the same calibration frames by the app;
- the **block-mean proxy** - p90 of the HR delta, 2 levels at depth 1 and 12 at depth >= 2. It is
  still measured and printed (`d1 p90 0.3 (limit 2)`), and it still decides when a caller supplies
  no acceptance data (the unit tests do), but the app no longer lets it overrule the picture.

Why that mattered: the mapping between the two was only ever measured in one point - at depth 4 the
proxy reads **9.3** while the same reuses realise **0.00%** of pixels above 8 levels and a worst
64 px block of 2.0 levels - and a single point cannot certify a rule. Every false "price of quality"
in the ledger came from that gap: the statistic's order of operations, a one-step budget applied to a
four-step chain, and depths the gate could not reach.

The residue is not zero, and it is stated rather than hidden: the calibration sample reads lower than
the clip (Violet: 0.02% against 0.096%), a longer window makes the verdict *worse* rather than
better, and a sample bar with a margin (~0.05%) is the open calibration question. The noise-floor
limit above remains the cheap pre-filter that stops hopeless material before any of this runs.

The one time this *was* a fault is worth naming, because it is the reason the budget is 2 and not 1:
at a 1-level budget `best` disabled the gate even on clean anime (18.1 s against 7.65 s, the gate
doing nothing at all), and the symptom read exactly like a preset-wiring bug.

## The single most important operational fact

With `libx265 -preset medium` the same clip takes **11.2 s**, of which the encoder alone is
**9.6 s** and the model **4.95 s**. The pipeline is encoder-bound, so model work is invisible
until the encoder is off the critical path. Use `x265_nvenc`/`av1_nvenc` (or a faster x265
preset) before tuning anything else.

## Opt-in

| knob | default | why not default |
|---|---|---|
| `--tile_reuse_mode` | **off** | tile-level reuse with its mask dilated (every tile adjacent to a recomputed one is recomputed too) takes **10.0 s** against 11.7 s on 502 anime frames at **0.071%** of pixels above 8 levels (RGB, worst frame, peak 23) against 0.009% without tiles. That passes the 0.1% bar but with eight times less margin, for 14% of wall clock, so it stays opt-in. Its per-tile mean cannot be tightened into safety: at threshold 0.10 reuse collapses from 57.7% to 3.0% and the render becomes slower than not using the tile path at all. `auto` keeps the pre-render probe (`utils/TileProbe.py`: 16 frames at 4 positions, tiles only when the predicted cost beats the halo tax) for material where the trade is wanted |
| `--tile_reuse` | off | forces tiles on, equivalent to `--tile_reuse_mode on`; the core grid is still fitted to the frame (`divisor_tile`: 480x540 for 1920x1080, 8 windows, 1.147x the frame's area) |
| `--tile_reuse_size` / `--tile_reuse_threshold` | 512 / 0.35 | the tile path's own knobs: LR core target and the worst low-passed delta under which a tile is reused |
| `RVE_P010=1` | off | the "diverges from the planar reference at 31 dB, unexplained" that used to be here was **three separate faults, all found and fixed** (ledger C4): P010 missing from the "already YUV" branch, `_pipe_input_pixel_format()` deriving the format from the *encoder* tag while the packer emitted planar `yuv420p10le`, and the colour tags declared on the output side. Corrected, an opt-in P010 render measures 52.01 dB (10-bit output) / 46.11 dB (8-bit) against a libx265 reference. It stays **off**, and that is now measured rather than left open: on a 10-bit source (`yuv420p10le`, 220 frames, same preset) the two paths are **byte-identical in picture (0.000% of pixels above 8 levels)** and P010 costs **+0.5 ms/frame** - frame total 28.16 against 27.65 ms, render 6.40 against 6.29 s. The 1.54 ms repack it was supposed to avoid was already gone once C4 stopped deriving the pipe format from the encoder tag, so packing on the GPU now adds a conversion and removes nothing |
| `--trt_streams` | 1 | >1 falls back to 1 with a warning: two `torch.jit.load` modules of one engine do not give independent contexts and the output corrupts (25 dB). |

## Measured and rejected in this fork

| item | result |
|---|---|
| pooled pinned output payload | wired as specified, a full render comes out 12.6 dB against the reference (slots rewritten before ffmpeg has the bytes) and slower (4.83 vs 4.21 s) - disabled. The later writer instrumentation shows the queue never fills (depth 4 of 25), so "a whole pool behind" cannot be the mechanism: a slot was released before its last byte had been copied |

## Measured dead ends - do not re-litigate

| idea | result |
|---|---|
| FP8 on consumer Ada | no speedup vs FP16 (NVIDIA issue #4081; L40S gets 1.7-1.9x, Ada does not) |
| INT8 PTQ for this net | **measured twice, and the second one is the informative one.** (a) Implicit with fp16 fallback, i.e. mmdeploy's ESRGAN run: 73.2 and 94.3 ms against 48-51 ms fp16 (two runs that overlapped on the card, so the magnitude is indicative only). (b) Explicit Q/DQ, per-channel int8 weights, per-tensor activation scales calibrated on 8 of our own frames: **essentially lossless** (0.000% of pixels above 8 levels against the fp16 engine, peak 4 levels, PSNR 54.1 dB) and **slower, 68.11 against 55.54 ms**. The per-layer profile says why, and it is not the kernel: the 16 quantised convs run `sm80_xmma_fprop_implicit_gemm_interleaved_i8f16_i8i32_f32_nchw_vect_c_...`, the interleaved int8 tensor-core layout, at **1.56-1.88 ms each (91 TFLOP/s against the 52 TFLOP/s the fused fp16 block achieves, i.e. 1.74x)**. The cost is the boundary: a per-channel PReLU (measured slopes -1.39..1.70) cannot go into an int8 epilogue, so it runs as a **separate 1.75 ms pass between two int8 convs**. In fp16 the same block is one fused myelin kernel (`__myl_CorrMulLtSele` = conv+PRelu) at 2.9 ms; int8 pays 1.68 + 1.80 = 3.5 ms for that block. The engine holds **21 reformat layers of 78**, and profiling them by group they cost **5.6 ms** in total - 8% of the engine, against **32.7 ms for the 17 PReLU passes**. 60 is the number of non-convolution layers, which is where the "60 reformats" in earlier revisions of this file came from. Quantising the conv output as well, so the PReLU reads int8 instead of half: **78.36 ms**, 36 reformats and 16 `DQout` passes at 1.64 ms each - the boundary is charged per hop, so a second hop makes it worse. (That build's picture metric is not quoted: the extra output scales were mis-calibrated and its output contained NaN; the timing does not depend on scale values). A third build answers "how much would the fusion be worth": with the PReLU replaced by a **scalar** `LeakyRelu` - the only form TRT puts in an int8 epilogue - the engine runs **40.24 ms against 56.28 (1.40x)** with 4 reformat layers and one pointwise pass instead of 21 and 17. That form is not this function (99.323% of pixels above 8 levels, peak 255, -2.02 dB, because the slopes span -1.39..1.70), so the 1.4x is reachable only through a plugin that implements its own int8 convolution with a per-channel PReLU epilogue |
| 2:4 structured sparsity | needs pruned weights (retraining) and TRT itself picks sparse tactics for 22 of 53 eligible layers |
| dynamic activation sparsity | no GPU win without custom kernels |
| CUDA graphs on the engine | ~0.3% at 35-45 ms/frame |
| L2 persistence window | nothing to pin: **1.2 MB of weights against 32 MB of L2** (AD106, not the 24 MB of a 4060 without Ti). The "265 MB activation stream" that used to justify this row predates the myelin-fusion finding - the engine is 25 top-level layers, not 18 kernels each paying a full round trip - so the traffic that remains is unmeasured; the verdict rests on the weights |
| SVD/low-rank post-training compression | energy at rank 32 is 77-91%; layer output error 28.7 dB - unusable |
| hand-written conv kernels | **not established either way, and not in this table's sense.** One public attempt measured 39.8 TFLOP/s against a 5090's 186.7 (CORE-TAG), but for a 3x3 conv cuDNN chose Winograd at 74 ms where TRT's best tactic was 156 ms - TRT's tactic set is not the set of all algorithms. Recorded as untested in ledger E10, i.e. do not assume either direction |
| exact-duplicate frame gating | 0% bit-identical frames on encoded video |
| concurrent engine contexts (this app) | no gain and corrupt output |

## Where the remaining time is

Per frame at 2880x2160 output: engine ~35 ms, **D2H 3.4 ms** (an 18.7 MB RGB frame over the link
below), colour conversion ~1.3 ms, the rest is reading/writing and the encoder.

Inside the engine, 8.11 ms of 56.8 ms is not the body GEMMs. The biggest single piece is now
**fixed**: padding the head convolution to 8 input channels puts it into myelin and removes the
separate PReLU pass, **57.51 -> 54.15 ms (-5.8%)** with an unchanged picture (0.000% of pixels above
8 levels, 78.69 dB), and it needs no change outside the engine because the pad is internal and the
input stays 3-channel. What remains of that 8.11 ms: tail convolution tactics and the upsampler plus
myelin's reshape/add, **2.35 ms**, and the layout kernels **0.81 ms** (ledger E20-E21, E18).

### The link is PCIe 3.0 x8, and pinning is not the lever

`nvidia-smi` reports `pcie.link.gen.current = 3`, `width = 8` - that is both the current and the
maximum, so the card sits in an x8 3.0 slot (7.9 GB/s theoretical). Measured here, 60 iterations
per cell, pageable against pinned (`.pin_memory()` + `non_blocking`), CUDA events:

| size | H2D pageable | H2D pinned | gain | D2H pageable | D2H pinned | gain |
|---|---|---|---|---|---|---|
| 6.2 MB (1080p10 input planes) | 1.10 ms | 1.00 ms | 1.10x | 1.19 ms | 1.04 ms | 1.15x |
| 3.1 MB | 0.65 | 0.50 | 1.30x | 0.59 | 0.52 | 1.14x |
| 18.7-24.9 MB (4K RGB frame) | 4.44 | 4.53 | 0.98x | 4.43 | 4.15 | 1.07x |
| 12.4 MB | 2.24 | 2.20 | 1.02x | 2.13 | 2.04 | 1.04x |

Both directions land at **5.1-6.0 GB/s in every row**, i.e. at the x8 3.0 link, not at a
pageable-memory limit. So a pinned staging buffer is worth **0.1 ms on the 6.2 MB input and 0.3 ms
on the 4K output** - roughly 1% of the frame, not the ~2.3% a "pinned is 2x pageable" estimate
would predict. The input conversion measured at 1.42 ms is therefore what the link costs for
6.2 MB, and the only way to move the output faster is to stop sending a full RGB frame back at
all (10-bit packing, or feeding NVENC from the GPU).

### The encoder already overlaps the model (measured)

The writer thread reports its own split now. On the 502-frame anime render it is **blocked in
`write()` 12.34 ms/frame** (ffmpeg draining at NVENC speed) and **idle 10.63 ms/frame** waiting
for frames, with the queue at depth 4 of 25: the loop (22.72 ms/frame) is the producer and the
encoder has slack. So there is nothing to win by moving encoding in-process or by
double-buffering harder - the "~13 ms/frame not covered by the render loop" that the budget
table used to show is **fixed startup** (engine build/deserialize, ffmpeg + NVENC init,
flush), which is 1.5 s on a 94-frame clip and invisible on 502 frames. The log prints
`startup:` and `render loop:` so the split is visible per run.

Note on this host: `nvidia-smi` reports 17-47% utilisation while nothing of ours runs, because
the desktop's GPU clients (gnome-remote-desktop, sunshine, VS Code) submit small copies. They
do not take meaningful throughput (23-26 W idle against 164 W under load), so "one GPU job at
a time" means "no other render or benchmark process", not "utilisation 0".

### In-process encoding: measured, not worth it in this stack

- PyPI `av` 19.0.2 installs and exposes `libx265`/`hevc_nvenc`, but the wheel has **no
  `av.CudaContext`** and `hevc_nvenc` refuses to open (`avcodec_open2` -> EINVAL) for both
  `yuv420p10le` and `p010le`, so CUDA-frame encoding is unreachable through PyAV here.
- Dropping the subprocess and its pipe, same codec: **+2%** on real content - libx265
  in-process 10.35 s vs ffmpeg CLI + pipe 10.58 s for 94 frames at 2880x2160 10-bit. The pipe
  write overlaps the encode, so it costs ~2 ms/frame, not the 3-8 ms often quoted.
- The only true zero-copy route left is PyNvVideoCodec (P010 from a CUDA tensor), which moves
  muxing, timestamps and audio into our code: ~18% of a 45 ms frame for days of work and
  permanent maintenance. Not recommended.
