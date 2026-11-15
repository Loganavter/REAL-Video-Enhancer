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
| `--quality_preset` | **best** | decides how stale a reference the gate may serve. `best`: the tuner certifies the **chain depth** (a p90 budget at every depth the gate can reach) and, when nothing certifies, switches reuse off for the whole render. `fast`: keeps the full chain and reuses as little as the threshold range allows | anime `best`: chain 4, **11.7 s**, **0.009%** of pixels above 8 levels (RGB) on the worst of 19 sampled frames, peak 14; film `best`: refuses to reuse, 4.6 s against 4.66 s ungated |
| `--temporal_reuse_threshold` | chosen by the tuner (seed 5.0) | max 32 px-block mean delta vs the last KEPT frame; the tuner scores every candidate at every chain depth the gate can reach | anime 70.9% reused; film 18.1% |
| `--temporal_reuse_max_chain` | 4 (was 2) | cap on consecutive reuses | |
| `--no_temporal_reuse_auto` | off (i.e. auto is ON) | automatic threshold and chain depth: during the first 16 rendered frames the gate records its own LR delta paired with the HR error that reusing that frame would cost **at every chain depth it can reach**, then picks the deepest chain and widest threshold whose p90 stays inside a 2-level budget. During those 16 frames the gate is forced to render, so the cost is ~0.7 s once per video. The threshold is clamped to 2..4/255 (`--temporal_reuse_threshold` can narrow it): with the tile path out of the way, 12 against 4 measures 0.517% against 0.045% of pixels above 8 levels |
| `--temporal_reuse_pixel_veto` | 0 (off) | full-res max delta that rescues a would-be duplicate. 40 is TheAnimeScripter's value and is right for clean anime; on grainy film a single pixel exceeds it and reuse collapses from 46.8% to 1.1% (measured, 43 vetoes in 94 frames) | |
| 10-bit path | auto for >=10-bit sources | YUV420p10 packed on the GPU, colour matrix from the source, and the ffmpeg command declares the format the packer actually produced | colour parity vs a libx265 render: 46.11 dB at 8-bit output, 52.01 dB at 10-bit output |
| output depth | **the source's depth** | a 10-bit source now stays 10-bit, and it is free: 4.67 s against 4.64 s for 94 frames of 2160p, ~6 dB better. The old "10-bit costs 2x NVENC" was measured and is wrong - the encoder costs 1.47x (9.4 against 6.4 ms/frame) and the rest was swscale, inserted because the colour tags were declared on the *output* side. Declared on the input side they produce the same container tags with no conversion, and the chroma now matches the source to 0.2 levels where the old path was 5 levels off |

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
| `--tile_reuse_mode` | **off** | tile-level reuse buys 9.2 s against 11.3 s on 502 anime frames and costs **0.287%** of pixels above 8 levels (RGB, worst frame, peaks of 98) against 0.045% without it, so it is off by default. Its per-tile mean cannot be tightened into safety: at threshold 0.10 reuse collapses from 57.7% to 3.0% and the render becomes slower than not using the tile path at all. `auto` keeps the pre-render probe (`utils/TileProbe.py`: 16 frames at 4 positions, tiles only when the predicted cost beats the halo tax) for material where the trade is wanted |
| `--tile_reuse` | off | forces tiles on, equivalent to `--tile_reuse_mode on`; the core grid is still fitted to the frame (`divisor_tile`: 480x540 for 1920x1080, 8 windows, 1.147x the frame's area) |
| `--tile_reuse_size` / `--tile_reuse_threshold` | 512 / 0.35 | the tile path's own knobs: LR core target and the worst low-passed delta under which a tile is reused |
| `RVE_P010=1` | the P010 packing is bit-exact in isolation but a full `hevc_nvenc` render diverges from the planar reference (31 dB) - unexplained, so off. |
| `--trt_streams` | >1 falls back to 1 with a warning: two `torch.jit.load` modules of one engine do not give independent contexts and the output corrupts (25 dB). |

## Measured and rejected in this fork

| item | result |
|---|---|
| pooled pinned output payload | wired as specified, a full render comes out 12.6 dB against the reference (slots rewritten before ffmpeg has the bytes) and slower (4.83 vs 4.21 s) - disabled. The later writer instrumentation shows the queue never fills (depth 4 of 25), so "a whole pool behind" cannot be the mechanism: a slot was released before its last byte had been copied |

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
reading/writing and the encoder.

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
