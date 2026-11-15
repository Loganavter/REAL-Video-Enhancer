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
| `--temporal_reuse_threshold` | chosen by the tuner (seed 5.0) | max 32 px-block mean delta vs the last KEPT frame. The tuner now scores a candidate at every chain depth the gate can reach, and falls back to the **strictest** candidate when nothing qualifies, instead of keeping the seed | anime 70.9% reused, worst frame 48.64 dB against ungated; film 18.1% reused, average 50.34 / worst 39.83 dB, 4.33 s against 4.66 s |
| `--temporal_reuse_max_chain` | 4 (was 2) | cap on consecutive reuses | |
| `--no_temporal_reuse_auto` | off (i.e. auto is ON) | automatic threshold: during the first 16 rendered frames the gate records both its own LR delta and the real HR delta it caused, then picks the widest candidate whose p90 HR error stays under 2 levels. During those 16 frames the gate is forced to render (otherwise the sample is biased to already-changed frames), so the cost is ~0.7 s once per video; measured on the same clip: 11.38/11.20 manual vs 11.40/11.32 auto. On anime it then picks 8.0 (73.5% reuse, 52.7 dB avg / 48.2 dB worst vs ungated) and on grainy film it keeps the 5.0 seed |
| `--temporal_reuse_pixel_veto` | 0 (off) | full-res max delta that rescues a would-be duplicate. 40 is TheAnimeScripter's value and is right for clean anime; on grainy film a single pixel exceeds it and reuse collapses from 46.8% to 1.1% (measured, 43 vetoes in 94 frames) | |
| 10-bit path | auto for >=10-bit sources | YUV420p10 packed on the GPU, colour matrix from the source, and the ffmpeg command declares the format the packer actually produced | colour parity vs a libx265 render: 46.11 dB at 8-bit output, 52.01 dB at 10-bit output |
| output depth | the source's depth only if asked | `--video_pixel_format yuv420p` by default even for 10-bit sources: keeping 10 bits at the encoder is +6 dB but costs NVENC roughly twice the wall clock (3.84 s against 8.01 s on 94 frames of 2160p) | opt-in |

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
| `--tile_reuse_mode` | **auto** | decides from the source, before any engine is built: decode 16 frames at 4 spread positions and replay the gate's and the tile path's own statistics on them (`backend/src/utils/TileProbe.py`). Tiles are enabled only when the predicted cost is below `1 / area_tax`, i.e. when the saving beats the halo tax instead of merely breaking even. Measured verdicts: **anime** 33% predicted reuse -> cost 0.76 of a full-frame call against a bar of 0.87 -> ON, render 8.01 s against 11.66 s for the gate alone (1.46x), 57.7% tiles reused at runtime, 51.6 dB average / 47.5 dB worst frame against ungated (gate alone 52.7 / 48.2); **moving film** 15% predicted -> 0.97 against 0.87 -> OFF, render 3.79 s and no tile engine built at all. Probe cost is one decode of 16 frames, about a second, once per video |
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
