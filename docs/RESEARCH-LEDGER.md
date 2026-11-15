# Research ledger

Everything below was either measured on this machine or read from a named source. Numbers
without a source are ours. If an idea is not in this file, it was not considered.

Measurement rig: RTX 4060 Ti 16 GB, Ryzen 5 5600 (6c/12t), TensorRT 10.12, torch 2.9,
PyAV 19.0.2.

**Measurement protocol - one GPU job at a time.** Two engine builds running next to a render
inflate every number: in this session a `full_l2` engine measured 67.4 ms and 103.2 ms in two
"identical" runs, and an app render measured 20.4 s against 11.5 s for the same configuration,
purely because a second process was building engines on the same card. Check `pgrep -x python3`
and `nvidia-smi` before quoting anything, and repeat the pair (same-session ratios only). Two clips: a 1440x1080 10-bit HEVC film excerpt (94 frames, "the clip") and a
40 s window of the same film ("the window"). Unless stated, wall-clock for the whole render
with `x265_nvenc` (so the encoder is not the bottleneck).

Status: **adopted** (on by default), **opt-in** (code exists, off by default), **rejected**
(measured, do not retry), **open** (not tested).

## The model itself: what is left

Per frame the network is 1.85 TFLOP at 1440x1080 for the x2 compact model, and it runs at
35 ms = **53 TFLOP/s, about 60% of this card's fp16 peak**. Kernel-level tuning is therefore
finished: the ledger below contains five measured dead ends in that direction. The only
remaining ways to make the model cheaper are to *do less work per frame*:

| lever | status | measured |
|---|---|---|
| skip frames that are duplicates of the last rendered one | **adopted** | 46.8% of frames skipped on the clip, output 71.1 dB vs the ungated render |
| recompute only changed tiles (512 px + 18 halo) | opt-in | 27.7% of tiles reused but slower: the 548 px halo window costs +15% area (6.57 s vs 4.27 s) |
| batching 2-4 frames per enqueue | rejected | the GPU is already at 96-100% util and batching is at best linear |
| lower precision (INT8/FP8) | rejected | see below |
| a cheaper published model | not applicable | the owner keeps his ESRGAN model set |

## Engine / kernels

| id | idea | source | expected | measured | status |
|---|---|---|---|---|---|
| E1 | builder optimization level 5 instead of 3 | TRT docs; NVIDIA benchmarking page | 1.3-1.6x | isolated engine 94.3 -> 57.5 ms | **adopted** |
| E2 | `num_avg_timing_iters` 8 instead of 4 | torch-tensorrt defaults | few % | included in E1 | **adopted** |
| E3 | `tiling_optimization_level=moderate` | TRT "tiling optimization" docs | 5-20% | 57.5 -> 52.7 ms | **adopted** |
| E4 | clamp the engine to a static shape per resolution | torch-tensorrt docs | removes dynamic-shape tactic loss | used everywhere | **adopted** |
| E5 | `workspace_size`, `max_aux_streams`, `use_fast_partitioner` | TRT docs | none for a single fully-supported block | not measured, expected 0 | open (low value) |
| E6 | CUDA-graph replay of the engine | TAS `upscale/tensorrt.py`; vs-mlrt `trt_utils.h` | 2-4 ms claimed | ~0.3% at 35-45 ms/frame | **rejected** |
| E7 | 2:4 structured sparsity | TRT docs; NVIDIA engineer on GitHub | 2x theoretical | needs pruned weights; TRT picks sparse tactics for 22/53 layers; up to 5% slower in fp16 | **rejected** |
| E8 | FP8 on consumer Ada | TRT issue #4081; OwLite report | 1.4-1.8x | no gain vs FP16 on 4090/4060 Ti; L40S gets 1.7-1.9x | **rejected** |
| E9 | INT8 PTQ | mmdeploy benchmark (same 16-conv class) | 2x ceiling | 12.45 ms vs 12.42 ms fp16 | **rejected** |
| E10 | hand-written conv kernel / plugin | CORE-TAG/rtx50-realesrgan-video | - | 39.8 TFLOP/s vs the engine's 186.7 on a 5090 | **rejected** |
| E11 | L2 persistence window | blog measurement (H100/3090) | 1.02-1.20x | nothing to pin: 1.2 MB weights, 265 MB activation stream | **rejected** |
| E12 | per-layer precision constraints (force fp16 accumulate) | TRT docs | unknown | TRT does not expose accumulator precision (issue #1237) | **rejected** |
| E13 | concurrent engine contexts (batch/streams) | vs-mlrt wiki: 1.65 -> 2.25x on a 2nd stream | up to 2x | no gain (4.25 -> 4.27 s) and output corrupt (25 dB): two `torch.jit.load` modules are not independent contexts | **rejected** |

## Precision / colour (10-bit)

| id | idea | source | expected | measured | status |
|---|---|---|---|---|---|
| C1 | pass the source's colour matrix/range to ffmpeg | ffmpeg/swscale docs | removes chroma shift | \|dU\| 6.64 -> 0.2 codes; 8-bit path 40.25 -> 44.06 dB | **adopted** |
| C2 | keep 10 bits end to end (pack YUV420p10 on the GPU) | waifu2x postproc shader uses a GPU colour stage | parity with ffmpeg | 49.69/55.34/54.84 dB vs the ffmpeg path | **adopted** |
| C3 | pipe `rgb48le` (the app's old "16-bit" path) | - | better than 8-bit | **worse**: 39.0 dB and a -5.9 level shift, because of the missing matrix | **rejected** |
| C4 | pipe P010 for NVENC (avoids a 1.54 ms/frame repack) | ffmpeg encoder caps | ~12% of the NVENC path | packing is bit-exact in isolation, but a full render diverges (31 dB) | opt-in (`RVE_P010`) |
| C5 | convert YUV->RGB on the GPU on the input side | same as C2 | halves the input pipe | works, colour matches; costs ~1.2 ms/frame | **adopted** |

## Temporal / dedup

| id | idea | source | expected | measured | status |
|---|---|---|---|---|---|
| T1 | frame-level reuse vs the previous frame, 0.35 threshold | our first attempt | 20-40% | 24.7% reused, 0.5 s saved | superseded by T3 |
| T2 | exact-duplicate (bit-identical) gating | video2x issue #270 discussion | large | 0% bit-identical frames on encoded video | **rejected** |
| T3 | reuse vs the **last kept** frame, 32 px block means, threshold 5.0, chain 4 | TAS `dedup/dedup.py` + `PARAMETERS.MD` (sens 35 / 7) | 20-40% frames | **46.8%** reused, 3.97 s vs 4.92 s, 71.1 dB vs ungated | **adopted** |
| T4 | full-res per-pixel veto at 40 levels | TAS `PixelDiffGuard` | rescues mouth flaps | on grainy film a single pixel exceeds it: 43 vetoes, reuse 1.1%, render 5.08 s | opt-in (`--temporal_reuse_pixel_veto 40` for anime) |
| T5 | tile-level reuse with halo | TAS/waifu2x tiling; our halo measurement (radius 18 LR px) | 2-4x on static content | 27.7% of tiles reused but slower: halo window +15% area; at 1024 px tiles 30.3% and 12.92 s | opt-in (`--tile_reuse`) |
| T6 | blend or discard the tile overlap | vs-mlrt discards, chaiNNer blends sinusoidally | - | we crop (discard), which is exact thanks to the halo | **adopted** |
| T7 | per-slot HR memo: copies of one held frame share one upscale | TAS `pipeline/executor.py` | - | our gate already skips the whole model for a duplicate | equivalent |

## Pipeline / IO

| id | idea | source | expected | measured | status |
|---|---|---|---|---|---|
| P1 | remove the double clone of the output (`UpscaleModelWrapper` + `Frame`) | our measurement | 4.6 ms/frame at 4x | 1.5 ms at 2880p, bit-identical output | **adopted** |
| P2 | stop building previews nobody consumes | our measurement | large on 16-bit frames | 10 s of 21 s on the naive 10-bit path | **adopted** |
| P3 | drop the extra `numpy -> bytes` copy of the frame | TAS keeps frames as CUDA tensors | 2-3 ms | 0.79 ms at 2880p; output bit-identical | **adopted** |
| P4 | pooled pinned buffers with a semaphore, writer releases the slot | TAS writer; our design | removes a full copy | 12.6 dB against the reference (slots rewritten before ffmpeg consumed them) and slower (4.83 vs 4.21 s) | **rejected** as wired |
| P5 | remove the ffmpeg subprocess, encode in-process | TAS uses in-process NeLux | 45 -> 20-25 ms | PyAV 19 wheel has no `av.CudaContext`; `hevc_nvenc` will not open; libx265 in-process is only +2% (10.35 vs 10.58 s) | **rejected** |
| P6 | drop the device-wide `torch.cuda.synchronize()` per frame | TAS orders with `wait_stream` only | 2-4 ms | within noise here (not model-bound) | **adopted** (harmless) |
| P7 | one full-res pass per frame instead of frame-on-disk | video2x 4.8.1 does 4 passes | large | we already pipe raw frames | n/a |
| P8 | reflect-pad to a bucket instead of one engine per resolution | TAS `upscale/tensorrt.py`; chaiNNer `remove_padding` | avoids rebuilds | not measured | open |
| P9 | write tiles straight into the destination plane | waifu2x `postproc.comp` | saves a pass | we composite on GPU already; per-plane write untested | open |
| P10 | `F_SETPIPE_SZ` 1 MB, one `os.write` per frame | pipe throughput reports | 2-5 ms claimed | with ffmpeg as the reader: +6% (noise) | **rejected** |

## Encoder

| id | idea | source | expected | measured | status |
|---|---|---|---|---|---|
| N1 | use NVENC instead of `libx265 medium` | our measurement | removes the bottleneck | **11.18 -> 4.27 s** on the clip (encode alone 9.6 -> 1.2 s) | **adopted** (advice) |
| N2 | `--frame-threads 4..6` for x265 | x265 threading docs | -4% | measured -4% | open (only if x265 is used) |
| N3 | anime flag set instead of `--tune animation` (`aq-mode 3`, `psy-rd 1.0`, `bframes 8`, `deblock -1:-1`) | JET guide, kokomins | same time, less banding | `--tune animation` measured +8% time, -6.5% bits and is wrong for flat gradients | open |
| N4 | `av1_nvenc` | IEEE Access 2025 BD-rate table | quality/size, not speed | -34% BD-rate vs hevc_nvenc | open (quality, not time) |
| N5 | p010 for NVENC to skip the repack | ffmpeg caps | 1.54 ms/frame | see C4 | blocked on C4 |

## Current per-frame budget (measured by the built-in profiler)

94 frames, 1440x1080 -> 2880x2160, `x265_nvenc`, gate at 46.8% reuse. Averaged over all frames:

| stage | ms/frame | note |
|---|---|---|
| duplicate gate | 3.5 | luma + 32px block means + full-res diff of the LR frame (includes a 4.6 MB clone from `get_frame_tensor`) |
| model | 22.7 | i.e. 43 ms on the 53% of frames that actually run; 53 TFLOP/s ~ 60% of fp16 peak |
| colour pack + D2H + queue | ~3 | 4.07 ms on the frames that run |
| read wait | 0.5 | the decoder is not a factor |
| not covered by the render loop | ~13 | the writer/encoder side; the loop itself is 29.3 ms/frame |

Two conclusions: the model is the single largest block and it is already near the card's
arithmetic peak, so making the model cheaper means doing less of it (T3/T5), and the *next*
largest block is the encoder again - `RVE_NVENC_PRESET=p1` measured 3.82 s vs 4.01 s (~5%).

## Backlog, ranked by expected gain on our pipeline

1. **Nothing large is left in the model.** 53 TFLOP/s of 88 peak; every kernel-level idea is
   a measured dead end. Further speed must come from doing less work (T3/T5) or from the
   encoder side.
2. **T5 on anime content.** 27.7% tile reuse was a loss on a moving film; on held cels the
   arithmetic changes. Needs one anime clip to settle (30 min).
3. **P8 reflect-pad buckets** - quality-of-life: one engine instead of one per resolution.
4. **N3 anime x265 flags** - only if x265 is kept.
5. **P9 per-plane tile writes** - only together with T5.

## Open questions

- Why does the P010 pipe diverge end to end (31 dB) while the packing is bit-exact against the
  planar decode? Suspect `-hwaccel cuda` on the raw input.
- Why did the pooled payload corrupt frames (12.6 dB) with `writeQueue.maxsize=25` and 4 slots?
  Most likely the kernel pipe buffer holding the tail of a frame while the slot was recycled.
- Is the app's per-frame model time (see `--profile` output) higher than the isolated 35 ms
  because of contention with the encoder thread?
