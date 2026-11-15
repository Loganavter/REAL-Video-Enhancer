# Performance notes (this fork)

Measured on: RTX 4060 Ti 16 GB, Ryzen 5 5600 (6c/12t), TensorRT 10.12, torch 2.9, on a
1440x1080 10-bit HEVC clip (94 frames) and a 40 s excerpt of the same film. Numbers are
wall-clock for the whole render unless stated otherwise.

## What is on by default

| knob | default | what it does | measured |
|---|---|---|---|
| `--tensorrt_opt_profile` | 5 (was 3) | TRT builder optimization level | engine only: 94.3 -> 52.7 ms/frame (1.79x) |
| `--trt_timing_iters` | 8 (was 4) | tactic timing iterations | included above |
| `--trt_tiling_level` | `moderate` | keeps per-layer tiles in L2 | included above; tiling is the larger part |
| `--temporal_reuse_threshold` | 5.0 (was 0.35) | max 32px-block mean delta vs the last KEPT frame | 62.4% of frames reused |
| `--temporal_reuse_max_chain` | 4 (was 2) | cap on consecutive reuses | |
| `--temporal_reuse_pixel_veto` | 0 (off) | full-res max delta that rescues a would-be duplicate. 40 is TheAnimeScripter's value and is right for clean anime; on grainy film a single pixel exceeds it and reuse collapses from 46.8% to 1.1% (measured, 43 vetoes in 94 frames) | |
| 10-bit path | auto for >=10-bit sources | YUV420p10 packed on the GPU, colour matrix from the source | colour parity vs ffmpeg: Y-PSNR 49.7 dB |

End-to-end on the clip with `x265_nvenc` (so the encoder does not hide the model):
**6.26 s (old defaults) -> 4.27 s (new defaults) = 1.47x**, and 62% frame reuse takes it to
**3.84 s**.

## The single most important operational fact

With `libx265 -preset medium` the same clip takes **11.2 s**, of which the encoder alone is
**9.6 s** and the model **4.95 s**. The pipeline is encoder-bound, so model work is invisible
until the encoder is off the critical path. Use `x265_nvenc`/`av1_nvenc` (or a faster x265
preset) before tuning anything else.

## Opt-in

| knob | why not default |
|---|---|
| `--tile_reuse` (+ `--tile_reuse_size`, `--tile_reuse_threshold`) | recomputing a tile needs the 548px halo window (+15% area), which ate the 27.7% it saved on moving footage: 6.57 s vs 4.27 s. Expect it to pay only on mostly-static content. |
| `RVE_P010=1` | the P010 packing is bit-exact in isolation but a full `hevc_nvenc` render diverges from the planar reference (31 dB) - unexplained, so off. |
| `--trt_streams` | >1 falls back to 1 with a warning: two `torch.jit.load` modules of one engine do not give independent contexts and the output corrupts (25 dB). |

## Measured and rejected in this fork

| item | result |
|---|---|
| pooled pinned output payload | wired as specified, a full render comes out 12.6 dB against the reference (slots rewritten before ffmpeg has the bytes) and slower (4.83 vs 4.21 s) - disabled |

## Measured dead ends - do not re-litigate

| idea | result |
|---|---|
| FP8 on consumer Ada | no speedup vs FP16 (NVIDIA issue #4081; L40S gets 1.7-1.9x, Ada does not) |
| INT8 PTQ for this net | 12.45 ms vs 12.42 ms fp16 (mmdeploy, same 16-conv class) |
| 2:4 structured sparsity | needs pruned weights (retraining) and TRT itself picks sparse tactics for 22 of 53 eligible layers |
| dynamic activation sparsity | no GPU win without custom kernels |
| CUDA graphs on the engine | ~0.3% at 35-45 ms/frame |
| L2 persistence window | nothing to pin: 1.2 MB of weights, 265 MB activation stream |
| SVD/low-rank post-training compression | energy at rank 32 is 77-91%; layer output error 28.7 dB - unusable |
| hand-written conv kernels | 39.8 TFLOP/s vs the engine's 186.7 on a 5090 (CORE-TAG) |
| exact-duplicate frame gating | 0% bit-identical frames on encoded video |
| concurrent engine contexts (this app) | no gain and corrupt output |

## Where the remaining time is

Per frame at 2880x2160 output: engine ~35 ms, colour pack + D2H + bytes ~10 ms, the rest is
reading/writing and the encoder. Ideas with sources are in the session notes: in-process
encoding from CUDA frames (PyAV 19 / PyNvVideoCodec) is the only large remaining lever
(estimates 45 -> 20-25 ms/frame), at the cost of owning muxing and audio.
