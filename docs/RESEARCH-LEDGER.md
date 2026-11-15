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
| E5 | `workspace_size`, `max_aux_streams`, `use_fast_partitioner` | TRT docs | unknown | not measured. `use_fast_partitioner=False` changes the subgraph composition and therefore the tactic set, so "expected 0" was never justified; the workspace is the better lead because the standalone-conv builds hit `CUDA error 2 allocating 4246732800-byte buffer` **during autotuning**, i.e. their tactic search was truncated (E5') | open (**not** low value) |
| E5' | workspace / partitioner as the *reason* the body convs got `sm75_` tactics | our tactic dump + the OOM in the standalone builds | - | if the same truncation happens when the full engine is built, the sm75 choice is an artefact of a constrained search, not a property of Ada. Test: rebuild with an explicit workspace on a quiet card, diff the tactic dump | open |
| E6 | CUDA-graph replay of the engine | TAS `upscale/tensorrt.py`; vs-mlrt `trt_utils.h` | 2-4 ms claimed | ~0.3% at 35-45 ms/frame | **rejected** |
| E7 | 2:4 structured sparsity | TRT docs; NVIDIA engineer on GitHub | 2x theoretical | needs pruned weights; TRT picks sparse tactics for 22/53 layers; up to 5% slower in fp16 | **rejected** |
| E8 | FP8 on consumer Ada | TRT issue #4081; OwLite report | 1.4-1.8x | no gain vs FP16 on 4090/4060 Ti; L40S gets 1.7-1.9x. Consumer parts are firmware-capped | **rejected** |
| E9 | INT8 PTQ, implicit with fp16 fallback | mmdeploy's ESRGAN benchmark | 2x ceiling | **slower**: 73.2 and 94.3 ms against 48-51 (the two runs overlapped on the card, so the magnitude is indicative) | **rejected as implemented** - this configuration is a reformat-heavy graph. Explicit Q/DQ NHWC per-channel is E9' |
| E9' | INT8 with explicit Q/DQ, NHWC, per-channel | TRT PTQ docs | 2x of the arithmetic | not measured. The mmdeploy source (12.45 vs 12.42 ms) is RRDB/ESRGAN, 32x32 patches, T4, TRT 7.2 - a launch-bound regime on a different architecture, i.e. not evidence about this net. Locally only the implicit-with-fallback format was tried. On this part INT8 is not firmware-capped (unlike FP8) and C=64 is a multiple of 32, so the tensor-core alignment requirement is satisfied | open - the only precision row with a real 2x in its arithmetic |
| E10 | hand-written conv kernel / plugin | CORE-TAG/rtx50-realesrgan-video | - | 39.8 TFLOP/s vs the engine's 186.7 on a 5090. Counter-precedent: for a 3x3 conv cuDNN chose Winograd at 74 ms where TRT's best tactic was 156 ms, so TRT's tactic set is not the set of all algorithms | **untested** (was "rejected": the source is one attempt on another card, and it does not measure this GPU) |
| E11 | L2 persistence window | blog measurement (H100/3090) | 1.02-1.20x | nothing to pin: 1.2 MB weights, 265 MB activation stream | **rejected** |
| E12 | per-layer precision constraints (force fp16 accumulate) | TRT docs | unknown | closed by measurement, not by an API limit: the DETAILED tactic dump shows all 16 body convs already running `f16f16_f16f16` (fp16 accumulation, NHWC). There is no fp32-accumulate share to convert | **closed (already in effect)** |
| E13 | concurrent engine contexts (batch/streams) | vs-mlrt wiki: 1.65 -> 2.25x on a 2nd stream | up to 2x | no gain (4.25 -> 4.27 s) and output corrupt (25 dB) - because two `torch.jit.load` modules of one engine file are not independent contexts | **rejected as wired** (a wiring fault, not a GPU property) |
| E14 | alternative tactics for the 16 body convs | our tactic dump: 17 of 18 convs get `sm75_` kernels on an sm89 part | unknown | not measured. Editable timing cache: does an sm80/86/89 variant exist for `tilesize256x64x32` at K=576, how does it time, A/B both | open |
| E15 | crop the active picture area (letterbox bars) | - | up to 1.33x if 25% of the frame is black | **n/a on our material**: the film clip (1440x1080) and the anime clip (1920x1080) both have zero rows darker than 6/255, minimum row mean 59.9 and 124.8. Worth one row-histogram check per new source before rendering | open (source-dependent) |
| E16 | Winograd F(2,3)/F(4,3) | cuDNN/CORE-TAG counter-precedent | F(2,3) cuts MACs 2.25x, inflates tensors ~4x | not measured. The intensity argument says it loses at the ridge (288 GB/s, ~42 MAC/byte break-even); the counter-precedent says TRT may not be offering the algorithm at all | open (untested) |
| E17 | rewrite PReLU as `a*x + (1-a)*Relu(x)` | TRT supported-op patterns | maybe changes tactic eligibility | not measured. PReLU is generic pointwise; the rewrite moves it into the natively supported Conv+ReLU / Conv+Scale / ElementWise patterns with identical maths. Cheap (an hour), expected near zero, but it also A/Bs the sm75 hypothesis | open (low confidence) |

## Precision / colour (10-bit)

| id | idea | source | expected | measured | status |
|---|---|---|---|---|---|
| C1 | pass the source's colour matrix/range to ffmpeg | ffmpeg/swscale docs | removes chroma shift | \|dU\| 6.64 -> 0.2 codes; 8-bit path 40.25 -> 44.06 dB | **adopted** |
| C2 | keep 10 bits end to end (pack YUV420p10 on the GPU) | waifu2x postproc shader uses a GPU colour stage | parity with ffmpeg | 49.69/55.34/54.84 dB vs the ffmpeg path | **adopted** |
| C3 | pipe `rgb48le` (the app's old "16-bit" path) | - | better than 8-bit | **worse**: 39.0 dB and a -5.9 level shift, because of the missing matrix | **rejected** |
| C4 | pipe P010 for NVENC (avoids a 1.54 ms/frame repack) | ffmpeg encoder caps | ~12% of the NVENC path | packing is bit-exact in isolation, but a full render diverges (31 dB) | **open** - 31 dB for a bit-exact packing is the signature of a missing matrix/range, not of the packing (C3 lost 39.0 dB to exactly that, and here swscale drops out of the path entirely so the tags must reach the encoder). Check `-colorspace/-color_primaries/-color_trc/-color_range` and drop `-hwaccel cuda` from the rawvideo input, which is meaningless there |
| C5 | convert YUV->RGB on the GPU on the input side | same as C2 | halves the input pipe | works, colour matches; costs ~1.2 ms/frame | **adopted** |

## Temporal / dedup

| id | idea | source | expected | measured | status |
|---|---|---|---|---|---|
| T1 | frame-level reuse vs the previous frame, 0.35 threshold | our first attempt | 20-40% | 24.7% reused, 0.5 s saved | superseded by T3 |
| T2 | exact-duplicate (bit-identical) gating | video2x issue #270 discussion | large | 0% bit-identical frames on encoded video | **rejected** |
| T3 | reuse vs the **last kept** frame, 32 px block means, threshold chosen automatically (seed 5.0), chain 4 | TAS `dedup/dedup.py` + `PARAMETERS.MD` (sens 35 / 7) | 20-40% frames | **46.8%** reused on film, 3.97 s vs 4.92 s, 71.1 dB; anime 73.5% at the auto threshold, 10.6 s vs 30.8 s ungated, 52.7 dB avg / 48.2 dB worst frame | **adopted** |
| T4 | full-res per-pixel veto at 40 levels | TAS `PixelDiffGuard` | rescues mouth flaps | on grainy film a single pixel exceeds it: 43 vetoes, reuse 1.1%, render 5.08 s | opt-in (`--temporal_reuse_pixel_veto 40` for anime) |
| T5 | tile-level reuse with an 18 px halo, **square 512 px cores** | TAS/waifu2x tiling; our halo measurement (radius 18 LR px) | 2-4x on static content | 65.8% of tiles reused on anime and *still* slower than the full-frame call (21.37 s vs 30.8 ungated, but 10.6 s for the frame gate); film 27.7% reused, 6.57 s vs 4.27 | superseded by T5' - the loss was the grid, see below |
| T5' | same mechanism, but the core grid **divides the frame** | this ledger's area model, checked against T5's own timings | 2.55x on the model at 65.8% reuse, instead of 1.68x | **18.14 s** ungated against 21.37 s for the 512 grid, and **8.45 s** on top of the frame gate against 11.74 s for the gate alone = **1.39x**; 51.6 dB average / 47.5 dB worst frame against the ungated render (gate alone 52.7 / 48.2). Tile reuse falls from 65.6% to 57.4% inside the gate, because the frames the gate leaves are exactly the ones that changed | **adopted as `--tile_reuse_mode auto`** (the default): a pre-render probe samples 16 frames at 4 spread positions, replays both detectors' own statistics, and builds the tile engine only when the predicted cost beats the halo tax. Verdicts on our two clips: anime 33% predicted / 58% achieved -> ON (8.01 s); film 15% predicted / 9.5% achieved -> OFF (3.79 s, no tile engine). `--tile_reuse` still forces it on |
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

Two reasons the combination is 1.39x rather than the multiplicative 1.5-2x: the tile share of
reuse drops from 65.6% to 57.4% once the gate runs in front (the frames left to render are the
ones that changed), and 36.9% of those frames still come out fully reused by the tile path.
The measured composition, 502 anime frames, same session:

| configuration | wall | loop | frames rendered | tiles reused |
|---|---|---|---|---|
| ungated, full frame | 38.85 s | - | 100% | - |
| tiles only, fitted grid | 18.14 s | 35.06 ms | 100% (56% of them fully reused) | 65.6% |
| gate only | 11.66 / 11.74 / 11.84 s | 22.6-22.9 ms | 27% | - |
| gate + fitted tiles | **7.64 / 8.42 / 8.45 / 8.48 / 8.65 s** | 14.6-15.4 ms | 27% | 57.4% |

Median against median is 11.74 -> 8.45 s = **1.39x**; the best pair in the same window is
7.64 against 11.66 = 1.54x. The spread inside one configuration is 10%, which is this
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

0. **T5' is measured, not a plan any more**: on anime the fitted grid plus the gate is 1.39x
   (median) to 1.54x (best pair) over the gate alone, and on the moving film clip it is
   **neutral in time and -0.8 dB in quality** (4.14 s against 4.07 s, tile reuse 9.5%, 68.46 dB
   average against 69.29, worst frame identical at 65.2-65.3). So it stays opt-in: it pays on
   content the gate already likes, and on film it is a small quality loss for nothing. A
   content-adaptive default (enable tiles once the gate's own reuse share passes ~60% during
   calibration) is the way to get it without asking the user, and it is unbuilt.
1. **The input conversion - 2.80 ms/frame** of the gated anime path, paid once per frame by
   whichever stage reads the frame first (10-bit upload + GPU colour stage). The gate's own
   arithmetic is 0.8 ms, so the lever is the input pipe, not the gate.
2. **E14 / E5' - the `sm75_` tactics.** Same experiment answers both: a quiet rebuild with an
   explicit workspace, then diff the tactic dump and A/B the alternatives through an editable
   timing cache. This is the only remaining kernel-level lead.
3. **E9' - explicit Q/DQ INT8.** The only remaining precision row with 2x in its arithmetic.
   Needs a calibration set and a real accuracy check; expect a non-trivial quality budget.
4. **C4 - the P010 divergence.** 1.54 ms/frame, and the 31 dB smells like tags, not packing.
5. **P8 / P9 / N3 / N4** - quality-of-life and encoder-side, only if x265 or per-plane tile
   writes are kept.
6. **Tile reuse as a default?** Measured: anime 1.39-1.54x, film neutral in time (+1.7%) and
   -0.8 dB in quality. Turning it on for everyone buys the anime case and costs the film case a
   little fidelity, so it stays a flag until the owner says otherwise or a content-adaptive
   switch exists (enable tiles when the gate's calibration reuse share passes ~60%).
7. **P11 is closed** (engine 51.4 ms standalone against 54.3 in-app, NVENC accounting for 9%
   of that), and **P4'/P5' are closed too** (the writer is idle half the time), so neither is
   on this list any more.

## In flight (delegated, 2026-11-15)

Two cells, disjoint files, no GPU work in either (the card is busy with measurements here):

| branch | task | files |
|---|---|---|
| `agent/rve-bench/work` | one reproducible benchmark harness: timing with repeat, NVENC contention A/B, clock sampling, tactic dump, TFLOP/s with its basis printed | `backend/tests/tools/bench_engine.py`, `backend/tests/test_bench_tool.py` |
| `agent/rve-quality/work` | PSNR report for render comparisons with the pixel format stated, worst frames with indices, fail-under, frame-count mismatch as an error | `backend/tests/tools/psnr_report.py`, `backend/tests/test_psnr_report.py` |

Both replace /tmp scripts that produced the numbers in this ledger and are now lost, so the next
reader can re-take them. Neither touches `backend/src`, and neither is a speed task: the tile
default question below is the supervisor's, because flipping it changes output quality.

## Open questions

- Why does the P010 pipe diverge end to end (31 dB) while the packing is bit-exact against the
  planar decode? Leading hypothesis now: the colour matrix/range never reaches the encoder,
  the same failure that cost C3 5 dB (check the `-colorspace/-color_primaries/-color_trc/
  -color_range` tags and the meaningless `-hwaccel cuda` on the raw input).
- Why did the pooled payload corrupt frames (12.6 dB) with `writeQueue.maxsize=25` and 4 slots?
  The writer instrumentation shows the queue never fills (max depth 4 of 25), so "a whole pool
  behind" cannot be the mechanism; a slot released before its last byte was copied is.
- Is the app's per-frame model time really higher than the isolated benchmark (P11)? Answered:
  no, or almost. The engine measures 51.37 ms standalone and 54.3 ms in-app at the same shape,
  and an NVENC encode running alongside costs 9% in a controlled A/B (51.37 -> 55.99). The
  earlier 23-43% "gap" was a cross-session comparison against a figure from a session in which
  this machine was 26% faster.
