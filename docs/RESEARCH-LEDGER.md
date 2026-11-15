# Research ledger

Everything below was either measured on this machine or read from a named source. Numbers
without a source are ours. If an idea is not in this file, it was not considered.

Measurement rig: RTX 4060 Ti 16 GB, Ryzen 5 5600 (6c/12t), TensorRT 10.12, torch 2.9,
PyAV 19.0.2.

**Measurement protocol - one GPU job at a time.** Two engine builds running next to a render
inflate every number: in this session a `full_l2` engine measured 67.4 ms and 103.2 ms in two
"identical" runs, and an app render measured 20.4 s against 11.5 s for the same configuration,
purely because a second process was building engines on the same card. Check for a running render/benchmark process before quoting anything, and repeat the pair
(same-session ratios only - the same ungated configuration measured 30.8 s in one session and
38.85 s in another). Note that `nvidia-smi` utilisation is **not** a usable idle check on this
host: the desktop's own GPU clients (gnome-remote-desktop, sunshine, VS Code) submit small
copies and keep the counter at 17-47% while nothing of ours runs. They do not take meaningful
throughput - idle draw is 23-26 W against 164 W under load - so the rule "one GPU job at a
time" means no other render or benchmark, not "utilisation 0". Two
clips: a 1440x1080 10-bit HEVC film excerpt (94 frames, "the clip") and a 40 s window of the
same film ("the window"); anime work uses a 502-frame 1920x1080 clip ("anime"). Unless stated,
wall-clock for the whole render with `x265_nvenc` (so the encoder is not the bottleneck).

Status: **adopted** (on by default), **opt-in** (code exists, off by default), **rejected**
(measured, do not retry *in that configuration* - the reason says which configuration),
**open** (not tested, or tested in a way that does not settle it).

## The model itself: what is left

Per frame the network is 1.85 TFLOP at 1440x1080 (2.48 at 1920x1080) and runs at 35.1 /
45.7 ms, i.e. **53-54 TFLOP/s**, or ~58 TFLOP/s at the `tiling=full` default (derived, see
MODEL.md §2). That is ~78% of a *borrowed* 74 TF/s sustained fp16-accumulate figure, and we
do **not** have a trustworthy ceiling for this conv shape - MODEL.md §3 explains why the
standalone-conv experiments cannot supply one. So "kernel-level tuning is finished" is not
something the numbers say. What the numbers say is narrower, and still useful:

- the model runs faster than any of its own convolutions measured standalone, and the only
  compute-dominated standalone conv (256->256, 49 TF/s) is *slower* than the whole network;
- five kernel-level ideas were measured as dead ends (below), two are open (E14, E16);
- every per-layer time is spread evenly (2.2-3.9 ms over 18 subgraphs), so no single layer is
  the lever, and the remaining gap needs `nsys` to localise (compute vs DRAM).

The ways to make a frame cheaper are therefore: do fewer frames, do fewer pixels per frame,
or run the same pixels in a cheaper numeric format.

| lever | status | measured |
|---|---|---|
| skip frames that are duplicates of the last rendered one | **adopted** | 46.8% of frames skipped on the clip, 71.1 dB vs the ungated render; anime 73.5% (auto threshold) |
| recompute only changed tiles, grid fitted to the frame | **open** (was "opt-in, not recommended") | the 512 px grid was a loss because 512 does not divide 1080 (T5); with a fitted 480x540 grid the same mechanism is a win - see T5/T5' |
| batching 2-4 frames per enqueue | rejected | the model is compute-bound at ~78% of a borrowed sustained figure, so batching is at best linear, and the measured dead end was *concurrent contexts*, not batching: no gain and corrupt output (E13). `nvidia-smi` "util 96-100%" was used as the argument for "already saturated" and does not mean that - it means at least one kernel is alive |
| lower precision (INT8/FP8) | **open for explicit Q/DQ**, closed for implicit and for FP8 | E9, E8 |
| a cheaper published model | not applicable | the owner keeps his ESRGAN model set |

## Engine / kernels

| id | idea | source | expected | measured | status |
|---|---|---|---|---|---|
| E1 | builder optimization level 5 instead of 3 | TRT docs; NVIDIA benchmarking page | 1.3-1.6x | isolated engine 94.3 -> 57.5 ms | **adopted** |
| E2 | `num_avg_timing_iters` 8 instead of 4 | torch-tensorrt defaults | few % | included in E1 | **adopted** |
| E3 | `tiling_optimization_level=moderate` | TRT "tiling optimization" docs | 5-20% | 57.5 -> 52.7 ms | **adopted** |
| E4 | clamp the engine to a static shape per resolution | torch-tensorrt docs | removes dynamic-shape tactic loss | used everywhere | **adopted** |
| E5 | `workspace_size` as the reason the body convs got `sm75_` tactics | our tactic dump + the OOM in the standalone builds | - | **measured: it changes the tactic mix and not the time.** Four builds of the same ONNX (8 GB explicit workspace, baseline, cuDNN-only, JIT-only), then three interleaved timing rounds per plan. 8 GB shifts one conv from sm80 to sm75 (17+1 instead of 16+2) and measures 49.84 ms against the baseline's 50.09 (0.995x, inside noise). So the default workspace does truncate the search, and the truncation is worth nothing measurable - the conv that moves is the tail one at 1% of the FLOPs | **closed** (measured) |
| E5' | explicit workspace around the build (the standalone-conv builds hit `CUDA error 2` while autotuning) | our measurement | - | covered by E5 above: the search does change, the speed does not. The OOM was real in the *standalone* engines, where the workspace is shared with the reformat buffers; the full engine has room | **closed** (measured) |
| E6 | CUDA-graph replay of the engine | TAS `upscale/tensorrt.py`; vs-mlrt `trt_utils.h` | 2-4 ms claimed | ~0.3% at 35-45 ms/frame | **rejected** |
| E7 | 2:4 structured sparsity | TRT docs; NVIDIA engineer on GitHub | 2x theoretical | needs pruned weights; TRT picks sparse tactics for 22/53 layers; up to 5% slower in fp16 | **rejected** |
| E8 | FP8 on consumer Ada | TRT issue #4081; OwLite report | 1.4-1.8x | no gain vs FP16 on 4090/4060 Ti; L40S gets 1.7-1.9x. Consumer parts are firmware-capped | **rejected** |
| E9 | INT8 PTQ, implicit with fp16 fallback | mmdeploy's ESRGAN benchmark | 2x ceiling | **slower**: 73.2 and 94.3 ms against 48-51 (the two runs overlapped on the card, so the magnitude is indicative) | **rejected as implemented** - this configuration is a reformat-heavy graph. Explicit Q/DQ NHWC per-channel is E9' |
| E9' | INT8 with explicit Q/DQ, NHWC, per-channel | TRT PTQ docs | 2x of the arithmetic | not measured. The mmdeploy source (12.45 vs 12.42 ms) is RRDB/ESRGAN, 32x32 patches, T4, TRT 7.2 - a launch-bound regime on a different architecture, i.e. not evidence about this net. Locally only the implicit-with-fallback format was tried. On this part INT8 is not firmware-capped (unlike FP8) and C=64 is a multiple of 32, so the tensor-core alignment requirement is satisfied | open - the only precision row with a real 2x in its arithmetic |
| E10 | hand-written conv kernel / plugin | CORE-TAG/rtx50-realesrgan-video | - | 39.8 TFLOP/s vs the engine's 186.7 on a 5090. Counter-precedent: for a 3x3 conv cuDNN chose Winograd at 74 ms where TRT's best tactic was 156 ms, so TRT's tactic set is not the set of all algorithms | **untested** (was "rejected": the source is one attempt on another card, and it does not measure this GPU) |
| E11 | L2 persistence window | blog measurement (H100/3090) | 1.02-1.20x | nothing to pin: 1.2 MB of weights. The old justification ("265 MB activation stream") predates the myelin fusion finding - the engine is 25 top-level layers, not 18 separate kernels with a full round trip each - so how much traffic actually remains is unknown; the verdict rests on the weights being 1.2 MB against 24 MB of L2, not on the traffic count | **rejected** (justification corrected) |
| E12 | per-layer precision constraints (force fp16 accumulate) | TRT docs | unknown | closed by measurement, not by an API limit: the DETAILED tactic dump shows all 16 body convs already running `f16f16_f16f16` (fp16 accumulation, NHWC). There is no fp32-accumulate share to convert | **closed (already in effect)** |
| E13 | concurrent engine contexts (batch/streams) | vs-mlrt wiki: 1.65 -> 2.25x on a 2nd stream | up to 2x | no gain (4.25 -> 4.27 s) and output corrupt (25 dB) - because two `torch.jit.load` modules of one engine file are not independent contexts | **rejected as wired** (a wiring fault, not a GPU property) |
| E14 | alternative tactics for the 16 body convs | our tactic dump: 17 of 18 convs get `sm75_` kernels on an sm89 part | unknown | **measured and closed.** With the tactic sources restricted to JIT convolutions only (no cuDNN, no cuBLAS) the winner is the *same* `sm75_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize256x64x32`, at 49.97 ms against the baseline's 50.09 (0.998x). So there is no sm80/86/89 alternative being skipped for this shape at K=576 in this TRT build; the `sm75_` name is the name of the winner, not a symptom. Restricting to cuDNN alone is 10.7% **slower** (55.43 ms, stable across three rounds), so TRT's own kernels beat cuDNN here | **closed** (measured) |
| E15 | crop the active picture area (letterbox bars) | - | up to 1.33x if 25% of the frame is black | **n/a on our material**: the film clip (1440x1080) and the anime clip (1920x1080) both have zero rows darker than 6/255, minimum row mean 59.9 and 124.8. Worth one row-histogram check per new source before rendering | open (source-dependent) |
| E16 | Winograd F(2,3)/F(4,3) | cuDNN/CORE-TAG counter-precedent | F(2,3) cuts MACs 2.25x, inflates tensors ~4x | not measured. The ridge is **128 MAC/byte** at the borrowed 74 TF/s and 288 GB/s (257 FLOP/byte), not 42: 42 corresponds to a ~24 TF/s fp32-shader peak, so the earlier figure was off by 3x in the wrong direction. The conclusion is unchanged - Winograd inflates tensors ~4x, which drops intensity below the ridge - but it is a weaker argument than it looked, and cuDNN-only builds measuring 10.7% *slower* also weakens the counter-precedent | open (untested) |
| E17 | rewrite PReLU as `a*x + (1-a)*Relu(x)` | TRT supported-op patterns | maybe changes tactic eligibility | not measured. PReLU is generic pointwise; the rewrite moves it into the natively supported Conv+ReLU / Conv+Scale / ElementWise patterns with identical maths. Cheap (an hour), expected near zero, but it also A/Bs the sm75 hypothesis | open (low confidence) |

## Precision / colour (10-bit)

| id | idea | source | expected | measured | status |
|---|---|---|---|---|---|
| C1 | pass the source's colour matrix/range to ffmpeg | ffmpeg/swscale docs | removes chroma shift | \|dU\| 6.64 -> 0.2 codes; 8-bit path 40.25 -> 44.06 dB | **adopted** |
| C2 | keep 10 bits end to end (pack YUV420p10 on the GPU) | waifu2x postproc shader uses a GPU colour stage | parity with ffmpeg | 49.69/55.34/54.84 dB vs the ffmpeg path | **adopted** |
| C3 | pipe `rgb48le` (the app's old "16-bit" path) | - | better than 8-bit | **worse**: 39.0 dB and a -5.9 level shift, because of the missing matrix | **rejected** |
| C4 | pipe P010 for NVENC (avoids a 1.54 ms/frame repack) | ffmpeg encoder caps | ~12% of the NVENC path | **two separate faults, both found and fixed.** (a) P010 was not in the "already YUV" branch, so it got `scale=in_range=pc:out_color_matrix=...`: a full-range reinterpretation of a limited-range stream plus a second matrix. (b) worse, `_pipe_input_pixel_format()` derived the format from the *encoder* tag ("NVENC wants P010") while the packer emits planar `yuv420p10le` unless `RVE_P010` is set, so **every 10-bit source rendered with NVENC fed planar samples labelled P010 and came out at 12.68 dB**. With the command following the packer and P010 opt-in: 52.01 dB against a libx265 render at 10-bit output, 46.11 dB at 8-bit output. A format disagreement cannot fail loudly here, both are 16-bit words at 3 bytes/pixel - which is why it survived a session. (c) the colour tags were declared on the *output* side, which makes ffmpeg convert every frame to the declared space: 79.0 ms/frame against 16.6 ms/frame with the tags on the input, and a chroma shift of 5.3 levels in U because already-bt709 samples were converted from bt601. Same tags in the container either way. See the investigation for the bisection | **fixed** (measured); P010 itself stays opt-in |
| C5 | convert YUV->RGB on the GPU on the input side | same as C2 | halves the input pipe | works, colour matches; costs ~1.2 ms/frame | **adopted** |

## Temporal / dedup

| id | idea | source | expected | measured | status |
|---|---|---|---|---|---|
| T1 | frame-level reuse vs the previous frame, 0.35 threshold | our first attempt | 20-40% | 24.7% reused, 0.5 s saved | superseded by T3 |
| T2 | exact-duplicate (bit-identical) gating | video2x issue #270 discussion | large | 0% bit-identical frames on encoded video | **rejected** |
| T3 | reuse vs the **last kept** frame, 32 px block means, threshold chosen automatically, chain 4 | TAS `dedup/dedup.py` + `PARAMETERS.MD` (sens 35 / 7) | 20-40% frames | **anime**: 70.9% reused, worst frame 48.64 dB against ungated. **film**: 18.1% reused, average 50.34 dB and worst 39.83 dB against ungated, 4.33 s against 4.66 s (1.08x). The earlier "46.8% reused, 71 dB" was measured against a *broken* pipe (see C4): both compared renders were equally corrupted, so the pair agreed with itself | **adopted** |
| T4 | full-res per-pixel veto at 40 levels | TAS `PixelDiffGuard` | rescues mouth flaps | on grainy film a single pixel exceeds it: 43 vetoes, reuse 1.1%, render 5.08 s | opt-in (`--temporal_reuse_pixel_veto 40` for anime) |
| T5 | tile-level reuse with an 18 px halo, **square 512 px cores** | TAS/waifu2x tiling; our halo measurement (radius 18 LR px) | 2-4x on static content | 65.8% of tiles reused on anime and *still* slower than the full-frame call (21.37 s vs 30.8 ungated, but 10.6 s for the frame gate); film 27.7% reused, 6.57 s vs 4.27 | superseded by T5' - the loss was the grid, see below |
| T5' | same mechanism, but the core grid **divides the frame** | this ledger's area model, checked against T5's own timings | 2.55x on the model at 65.8% reuse, instead of 1.68x | **18.14 s** ungated against 21.37 s for the 512 grid: the geometry fix alone is 1.18x on the tile path. On top of the frame gate it reads 7.68-9.15 s against 11.3 s for the gate alone = up to **1.4x**, but every one of those runs has tile reuse enabled, which is what the artifact section shows is visible (0.287% of pixels above 8 levels, peaks of 98). Tile reuse falls from 65.6% to 57.4% inside the gate, because the frames the gate leaves are exactly the ones that changed | **opt-in, off by default.** With the frame gate out of the way the trade is explicit: 9.2 s against 11.3 s on 502 anime frames, at **0.287%** of pixels above 8 levels (RGB, worst frame, peaks of 98) against 0.045%. Its per-tile mean cannot be tightened into safety - at threshold 0.10 reuse falls from 57.7% to 3.0% and the render ends up slower than not using the tile path at all - and switching the judge to the worst block made the clip worse (1.174%). `--tile_reuse_mode auto` keeps the pre-render probe for material where the trade is wanted |
| T6 | blend or discard the tile overlap | vs-mlrt discards, chaiNNer blends sinusoidally | - | we crop (discard), which is exact thanks to the halo | **adopted** |
| T7 | per-slot HR memo: copies of one held frame share one upscale | TAS `pipeline/executor.py` | - | our gate already skips the whole model for a duplicate | equivalent |

**Why T5 lost, and what T5' measured.** The cost of the tile path is `area x (1 - reuse)`,
with `area` the halo-inflated grid area over the frame area:

| grid | tiles | area with halo 18 | break-even reuse |
|---|---|---|---|
| 512x512 @1920x1080 | 4x3 = 12 | **1.738x** | 42.5% |
| 1024x1024 @1920x1080 | 2x2 = 4 | 2.167x | 53.9% |
| 1024x1024 @1440x1080 | 2x2 = 4 | 2.890x | 65.4% |
| **480x540 @1920x1080** | 4x2 = 8 | **1.147x** | **12.8%** |
| **480x540 @1440x1080** | 3x2 = 6 | **1.147x** | **12.8%** |

That model, with no free parameters, reproduces the measured tile renders: the 512 px grid
predicts 21.49 s and the render took 21.37 s (0.6%), the 1024 px grid predicts 24.76 s against
26.14 s (5%), and the fitted grid predicts 16.9 s against 18.14 s measured. So there is **no
per-tile launch penalty worth modelling** - the entire cost of T5 was the clipped tail of a
grid that does not divide the frame (512 does not divide 1080: the third row is a 56 px core
inside a full 548 px window). `divisor_tile()` in `backend/src/pytorch/TiledUpscale.py` picks
the core that divides each axis.

Two reasons the combination is 1.5x rather than the multiplicative 1.5-2x: the tile share of
reuse drops from 65.6% to 57.4% once the gate runs in front (the frames left to render are the
ones that changed), and 36.9% of those frames still come out fully reused by the tile path. Those
figures are for a chain of 4; the artifact section below explains why the default now certifies a
chain of 1 and what that costs.
The measured composition, 502 anime frames, same session:

| configuration | wall | loop | frames rendered | tiles reused |
|---|---|---|---|---|
| ungated, full frame | 38.85 s | - | 100% | - |
| tiles only, fitted grid | 18.14 s | 35.06 ms | 100% (56% of them fully reused) | 65.6% |
| gate only | 11.66 / 11.74 / 11.84 s | 22.6-22.9 ms | 27% | - |
| gate + fitted tiles, chain 4 | **7.64 / 8.42 / 8.45 / 8.48 / 8.65 s** | 14.6-15.4 ms | 27% | 57.4% |
| gate only, no tile reuse (the default now) | 11.3 s | 22.5 ms | 27% | - |

Median against median is 11.74 -> 8.45 s = **1.39x**; the best pair in the same window is
7.64 against 11.66 = 1.54x. Those runs all used tile reuse, which is now off by default because
of the artifact in the investigation: the configuration that ships is the gate alone at **11.3 s**
(3.4x over ungated) with 0.045% of pixels above 8 levels. The spread inside one configuration is 10%, which is this
machine's run-to-run noise, so the honest statement is "1.4x, up to 1.5x".

On the moving film clip (94 frames, 1440x1080) the same combination is **neutral**: 4.07 s for
the gate against 4.14 s with tiles, tile reuse only 9.5%, quality 68.46 dB average against
69.29 (worst frame 65.2 against 65.3). That is the selection effect in its purest form: the
frames the gate renders on this clip have essentially no unchanged area left.

That is why the default is a measurement rather than a flag: `TileProbe` decodes 16 frames at
4 spread positions, replays the gate's and the tile path's statistics on them, and reports the
predicted cost of the tile path as a fraction of one full-frame call. Two things make it
dependable. The tile statistic has to be the mean of the *absolute* block deltas, not the
difference of the tile's two means: the latter lets local changes cancel and predicted 38%
reuse where the render then achieved 9.5%. And equality is not a safe threshold, because a
16-frame sample mispredicts by a few points in both directions (anime 33% predicted / 58%
achieved, film 15% / 9.5%), so tiles must clear `1 / area_tax` - a saving at least as large as
the halo tax - instead of the bare break-even.

## Pipeline / IO

| id | idea | source | expected | measured | status |
|---|---|---|---|---|---|
| P1 | remove the double clone of the output (`UpscaleModelWrapper` + `Frame`) | our measurement | 4.6 ms/frame at 4x | 1.5 ms at 2880p, bit-identical output | **adopted** |
| P2 | stop building previews nobody consumes | our measurement | large on 16-bit frames | 10 s of 21 s on the naive 10-bit path | **adopted** |
| P3 | drop the extra `numpy -> bytes` copy of the frame | TAS keeps frames as CUDA tensors | 2-3 ms | 0.79 ms at 2880p; output bit-identical | **adopted** |
| P4 | pooled pinned buffers with a semaphore, writer releases the slot | TAS writer; our design | removes a full copy | 12.6 dB against the reference (slots rewritten before ffmpeg consumed them) and slower (4.83 vs 4.21 s) | **rejected as wired** - and the overlap it was meant to buy already exists, see P4' |
| P4' | overlap the encode with the model (writer thread + bounded queue) | our writer instrumentation | 1.44x if the encoder serialised behind the loop | **already overlapped**: on the 502-frame anime render the writer spends 12.34 ms/frame blocked in `write()` (ffmpeg draining) and **10.63 ms/frame waiting for frames** - it is idle half the time, queue depth max 4, and its 6.20 s of busy time sit inside the 11.74 s render. The render loop is 22.72 ms/frame and the writer has slack | **not a lever** - the "13 ms not covered by the render loop" in the budget table is fixed startup on a 94-frame clip, not serialisation |
| P5 | remove the ffmpeg subprocess, encode in-process | TAS uses in-process NeLux | 45 -> 20-25 ms | PyAV 19 wheel has no `av.CudaContext`; `hevc_nvenc` will not open; libx265 in-process is only +2% (10.35 vs 10.58 s) | **rejected** (the +2% bound is the whole upside, since the pipe already overlaps) |
| P6 | drop the device-wide `torch.cuda.synchronize()` per frame | TAS orders with `wait_stream` only | 2-4 ms | within noise here (not model-bound) | **adopted** (harmless) |
| P7 | one full-res pass per frame instead of frame-on-disk | video2x 4.8.1 does 4 passes | large | we already pipe raw frames | n/a |
| P8 | reflect-pad to a bucket instead of one engine per resolution | TAS `upscale/tensorrt.py`; chaiNNer `remove_padding` | avoids rebuilds | not measured | open |
| P9 | write tiles straight into the destination plane | waifu2x `postproc.comp` | saves a pass | we composite on GPU already; per-plane write untested | open |
| P10 | `F_SETPIPE_SZ` 1 MB, one `os.write` per frame | pipe throughput reports | 2-5 ms claimed | with ffmpeg as the reader: +6% (noise) | **rejected** |
| P11 | in-app model time vs the isolated benchmark | our profile | - | **split measured**: with `RVE_TRT_SPLIT=1` the engine alone costs **14.67 ms/frame** averaged over all frames = **54.3 ms on the 27% that run**, against 42.7-45.7 ms isolated for the same shape (+19-27%). Without the forced sync the same block measures 16.10 ms/frame = 59.6 ms, i.e. the output Frame's construction and handover add ~5 ms and **overlap with the kernels**, which is why the reported "inference" was never the engine time. `last_inference_ms` = enqueue + ctor + sync (engine 0.30 ms at enqueue); `last_model_ms`/`last_frame_ms` now separate them | open - the 19-27% gap is the largest unexplained number in the pipeline; next: same engine with an NVENC encode alongside, clocks sampled both ways |
| P12 | cheaper gate: no clone, block means without a full-frame fp32 luma | our profile | 1-2 ms/frame | **rejected as a lever.** The stage reads 4.37 ms/frame, but 2.80 ms of that is `get_frame_tensor()` (the 10-bit upload + colour stage), which every configuration pays on its first read of the frame; the gate's own statistic measures **0.80 ms** standalone (clone 0.017, luma fp32 0.43, pool 0.40, `max().item()` 0.03). So the clone is worth ~0.02 ms and the whole optimisation ~0.7 ms of a 22.6 ms frame = 3%. The number that looked like a 19% block was mostly the input pipe | **rejected** (measured) |

## Encoder

| id | idea | source | expected | measured | status |
|---|---|---|---|---|---|
| N1 | use NVENC instead of `libx265 medium` | our measurement | removes the bottleneck | **11.18 -> 4.27 s** on the clip (encode alone 9.6 -> 1.2 s) | **adopted** (advice) |
| N2 | `--frame-threads 4..6` for x265 | x265 threading docs | -4% | measured -4% | open (only if x265 is used) |
| N3 | anime flag set instead of `--tune animation` (`aq-mode 3`, `psy-rd 1.0`, `bframes 8`, `deblock -1:-1`) | JET guide, kokomins | same time, less banding | `--tune animation` measured +8% time, -6.5% bits and is wrong for flat gradients | open |
| N4 | `av1_nvenc` | IEEE Access 2025 BD-rate table | quality/size, not speed | -34% BD-rate vs hevc_nvenc | open (quality, not time) |
| N5 | p010 for NVENC to skip the repack | ffmpeg caps | 1.54 ms/frame | see C4 | blocked on C4 |

## Current per-frame budget (measured by the built-in profiler)

Two configurations, because they answer different questions.

**Anime, 502 frames, 1920x1080 -> 3840x2160, gate at 73.5% reuse, `x265_nvenc`:**

| stage | ms/frame | note |
|---|---|---|
| duplicate gate | 4.37 | of which **2.80 is `get_frame_tensor()`**: the 10-bit upload + colour stage, paid once per frame by whoever reads it first. The gate's own statistic is 0.80 ms standalone (P12) |
| model + stream sync | 16.31 | over all frames, i.e. **61.5 ms on the 26.5% that run** |
| read wait | 0.26 | the decoder is not a factor |
| render loop total | **22.72** | |
| writer blocked in `write()` | 12.34 | concurrent with the loop, 10.63 ms/frame idle - not on the critical path |
| process start -> first frame | ~? | measured by the `startup:` line now in the log |

**Film, 94 frames, 1440x1080 -> 2880x2160, gate at 46.8% reuse:** gate 3.5, model 22.7
(43 ms on the 53% that run), pack + D2H + queue ~3, read wait 0.5.

**Where the 22.7 ms of "model" actually goes (anime, gate only, `RVE_TRT_SPLIT=1`):** engine
14.67 ms/frame over all frames, handover 0.01, and the difference to the 16.10 ms of the
unsplit run is the output Frame's construction, which overlaps the kernels. Per rendered frame
that is 54.3 ms of engine against 42.7-45.7 ms isolated.

The old table added a "~13 ms not covered by the render loop" row and read it as encoder
serialisation. It is not: `3.5 + 22.7 + 3 + 0.5 = 29.7 ms` of loop against 45.4 ms/frame of
wall clock on a 94-frame clip is 94 x 29.7 = 2.79 s of loop inside a 4.27 s render, and the
difference is fixed cost (engine build/deserialize, ffmpeg + NVENC init, flush) amortised over
only 94 frames. The 502-frame run shows it directly: 22.72 x 502 = 11.40 s of loop inside an
11.74 s render.

## Backlog, ranked by expected gain on our pipeline

Nothing in the model or the gate is a measured loss any more; what is left is either a
precision experiment or an input-path rewrite.

1. **E9' - explicit Q/DQ INT8.** The only remaining precision row with 2x in its arithmetic and
   no local measurement. Needs a Q/DQ ONNX (per-channel weights, per-tensor activations), a TRT
   INT8 build, and an accuracy budget on real frames. Half a day, and the accuracy risk is the
   whole question - the implicit-with-fallback attempt was *slower*, but that configuration is a
   reformat-heavy graph, not a test of the format.
2. **The input conversion - 2.80 ms/frame** of the gated anime path, paid once per frame by
   whichever stage reads the frame first. The gate's own arithmetic is 0.80 ms, so the lever is
   the input pipe. Where it goes, per `TorchUtils`: one H2D of the packed frame (6.2 MB at 1080p
   8-bit, 12.4 MB at 10-bit) followed by roughly fifteen elementwise kernels that each
   materialise a full-resolution intermediate (`sub_`/`div_`, two bilinear chroma upsamples,
   `r`, `b`, `g` computed through `r` and `b`, a three-channel `stack`, `clamp_`, the dtype cast,
   and in the 8-bit path a `reshape -> permute -> contiguous` layout copy). The bandwidth floor
   for that traffic is ~0.1 ms, so it is launch- and materialisation-bound by roughly 30x. Two
   candidates: give ffmpeg's decoder `gbrp` so the planar frame arrives in the layout the tensor
   wants (kills the permute copy, 8-bit path only; the preview and cv2 consumers would need the
   interleaved view), and rewrite the 10-bit conversion to write into one preallocated
   3-channel buffer with `g` computed directly from `y,u,v`. Neither is measured.
3. **E16 Winograd / E17 PReLU rewrite.** Both cheap, both expected near zero. E16's
   counter-precedent is now weaker: restricting the build to cuDNN alone made the engine 10.7%
   *slower*, so this TRT build's own kernels beat cuDNN's for this shape, whatever way cuDNN
   wins a single-conv race elsewhere.
4. **P8 / P9 / N2 / N3 / N4** - quality-of-life and encoder-side, only if x265 or per-plane tile
   writes are kept.

## Which metric answers "is it visible" (2026-11-15)

PSNR alone was the wrong instrument for a reuse decision, and the numbers show why. The same
three pairs, compared with `libvmaf` (model v0.6.1neg, both inputs scaled to 1080p because that
is what the model is calibrated for):

| pair | PSNR avg | PSNR worst | VMAF mean | VMAF p5 | VMAF min |
|---|---|---|---|---|---|
| anime, gate + fitted tiles | 51.71 | 48.64 | 95.50 | 94.18 | 93.65 |
| film, gate | 50.34 | 39.83 | 96.70 | 88.81 | 83.79 |
| P010 after the fix vs a libx265 render | 52.01 | 50.40 | 96.24 | 95.65 | 95.63 |

The film row is the lesson: its *mean* is the best of the three (96.7) while its minimum is the
worst by 10 points, and the minimum is what a viewer sees on the frame where a reused edge landed
in the wrong place. The pixel-level peak agrees: the worst block of that film frame is **196
levels** off (mean 1.74), against 100 levels for the anime frame (mean 0.98).

VMAF is **necessary but not sufficient**, and this is measured, not assumed: the anime
configuration a viewer rejected (a moved fist, chain 4 with tile reuse) scores **min 93.65 / p5
94.18**, i.e. it passes a `min >= 90` floor. Whole-frame metrics are blind here because the defect
is a fraction of a percent of a 4K frame, and VMAF at 1080p downscale makes it smaller still.

The deciding number is local and in RGB: **share of pixels more than 8 levels off in any channel, on
the worst frame of a spread sample, must stay under 0.1%**. Measured: 0.045% for the shipping
configuration, 0.287% for the rejected one, 0.045% for no reuse at all. VMAF min/p5 and PSNR are
reported next to it as second opinions, and the worst frame's PNGs are kept for eyeballing
(`remastering/compare_frames/`).

## What the reuse artifact actually is (2026-11-15)

Chased from a viewer's report: on the worst frame of the gated anime render a fist sits in a
different pose than in the ungated one. Five measurements, in the order that narrowed it:

1. **A/A control.** Two runs, same input, same flags: **502 of 502 frames byte identical**. The
   pipeline is deterministic, so every difference between configurations is real.
2. **Global alignment.** Frame i against frames i-2..i+2 of the other render: the peak is at
   offset 0 in both clips (anime 55.2 dB at 0 against 52.5 at -1). The pair is not time-shifted.
3. **Patch-level probe** at the reported region: nearly identical PSNR at every offset from -5 to
   +5, i.e. the content matches no neighbouring reference frame - so it is not "the frame is one
   late" either.
4. **Tiles against full frame**, same short clip, halo 18 / 24 / 32: 51.34 / 51.36 / 51.36 dB,
   worst frame 49.84 / 49.94 / 49.94, a 2D shift search peaking at (0,0) with 49.53 dB, and the
   error at tile boundaries (0.496 levels) indistinguishable from inside the tiles (0.477). So the
   tile path is geometrically exact; its difference from the full-frame render is fp16-level noise
   with no seams, and the halo is not the limit.
5. **Chain depth**, same frame 383, share of pixels by error magnitude:

| threshold | chain 4 (what shipped) | chain 1 (what the tuner now certifies) |
|---|---|---|
| > 2 levels | 10.87% | 8.91% |
| > 4 | 1.13% | 0.39% |
| > 8 | 0.29% | **0.00%** |
| > 16 | 0.12% | **0.00%** |
| peak in the worst block | 100 levels | **12 levels** |

So the artifact is the **chain depth**: a reference four frames old moves a fist. The threshold and
the tile path were both innocent, which is why "lower the threshold" and "raise the halo" were
dead ends. `best` now certifies the depth itself (chain 4 once the chain budget is expressed on the
proxy's own scale, chain 1 when it is held to the one-step bar) and `fast` keeps the full chain;
the price is 11.3 s against 9.2 s on 502 anime frames (tile reuse off), and that is what the moved
fist costs.

Acceptance for this family of defects is therefore local and absolute, not a mean: **nothing above
8 levels, and under 0.5% of pixels above 4**. Whole-frame PSNR and VMAF both pass a frame that
fails this, because a fist is a fraction of a percent of the frame.

## Grain sweep (synthetic, 2026-11-15)

The clean anime clip with `noise=alls=S:allf=t+u` layered on, then three renders per variant:
ungated, `--quality_preset fast`, `--quality_preset best`. One GPU job at a time, same session.

| variant | noise floor | fast | reuse | avg dB vs ungated |
|---|---|---|---|---|
| clean | 0.18 | 8.30 s | 70.9% | 52.14 |
| grain 4 | 0.40 | 8.96 s | 70.9% | 50.21 |
| grain 8 | 0.51 | 13.99 s | 70.9% | 47.60 |
| grain 12 | 0.67 | 12.54 s | 70.9% | 44.75 |
| grain 16 | 0.96 | 13.47 s | 70.9% | 42.10 |

Two results worth keeping:

- **Grain does not stop frame reuse.** The share stays at 70.9% from clean to grain 16, because
  the statistic is a 32 px block mean and grain averages out inside a block. What degrades is the
  *quality* of the reuse: 52.1 -> 42.1 dB, because a reused frame suppresses the grain's own
  frame-to-frame variation.
- **Grain stops tile reuse.** The ungated render jumps from 17.6 s at grain 4 to 36.6 s at grain 8
  and stays there: the tile probe refuses tiles on grainier material, so the render falls back to
  the full-frame model. The boundary sits between noise floor 0.40 and 0.51 on this scale.

`best` refused to reuse on every variant including the clean one, which cannot be right when
`fast` certifies 81% of the same calibration frames at the same budget - that is a bug in the
preset wiring, not a verdict about grain, and it is being chased before any of these rows are
used to choose a default.

## Open questions

- **Is the film gate worth its quality cost?** Measured: 18.1% reuse, 4.33 s against 4.66 s
  ungated (1.08x), average 50.34 dB and **worst frame 39.83 dB**. That worst frame is not
  transparent. The automation now picks the strictest threshold on this clip, so this is the best
  the mechanism does here, not a tuning miss. The owner has to say whether 7% is worth a frame
  at 40 dB, or whether film should run ungated.
- **Does the tile probe generalise?** It reads 16 frames at 4 positions. On our two clips it
  classified both correctly after the margin rule, but the prediction is off in both directions
  (anime 33% predicted against 58% achieved, film 15% against 9.5%) and the sample is two titles.
  Three more anime clips with different grain would settle whether "cost must clear 1/area_tax"
  is the right bar or luck.
- **The GUI still passes its own `--video_pixel_format` default**, so the 10-bit output fix
  reaches the CLI only. The GUI path keeps quantising 10-bit sources at the encoder.
- **The pooled payload corruption is explained but unverified.** The writer instrumentation
  shows the queue never fills (depth 4 of 25), so the mechanism must be a slot released before
  its last byte was copied. The feature is off; re-verifying it would buy 0.79 ms/frame.
- **Does the chain-aware tuner over-tighten?** With the one-step budget applied at every depth it
  certified only chain 1 on anime (11.5 s); with the chain budget on the proxy's own scale it
  certifies chain 4 (11.3 s). The gap between those two budgets is what "over-tighten" means here,
  and it is a calibration question rather than a correctness one.
- ~~Why does the P010 pipe diverge (31 dB)?~~ Answered: two faults, see C4. The packing was never
  the problem; the command declared a format the packer did not produce, and P010 additionally
  fell through to the RGB branch.
- ~~Is the app's per-frame model time really higher than the isolated benchmark?~~ Answered: no,
  or almost. 51.37 ms standalone against 54.3 in-app at the same shape, with an NVENC encode
  alongside costing 9% in a controlled A/B (51.37 -> 55.99). The earlier 23-43% "gap" was a
  cross-session comparison against a session in which this machine was 26% faster.
