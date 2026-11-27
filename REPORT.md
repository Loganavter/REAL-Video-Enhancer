# The Logs block: emitter audit and policy

The Logs tab (`src/ui/pages/tab_5.py` -> `ConsoleView`) is a **pure consumer**. `ProcessTab.renderToPipeThread`
reads the backend child's merged stdout/stderr line by line (`src/ui/ProcessTab.py:1874`), runs every line through
`src/ui/render_log.py:RenderLogFilter`, and appends what survives to `renderLogLines`; `syncRenderConsole`
(`src/ui/ProcessTab.py:3706`) paints them. The app's own `src/Util.log()` writes `log.txt`, not the block. So a
policy for the block belongs at the backend emitters, and that is where this change puts it.

One shared module, `backend/src/utils/LogPolicy.py`, is imported by both processes (`src/Util.py` loads it by path
next to the backend it launches). It is pure Python - no Qt, no torch, no numpy.

## Levels and cadence

| rule | value | why |
|---|---|---|
| default level | `info` | lifecycle, phases and per-job summaries are what a person reads |
| detail level | `debug` | per-frame / per-call rows; off unless `RVE_LOG_DEBUG=1` (`true`/`yes`/`on` also count) |
| progress cadence | one line per **second** of wall clock (`DEFAULT_PROGRESS_INTERVAL_S = 1.0`) | frame rate spans ~3-1000 fps here, so a percent cadence is hundreds of lines/s on a short fast clip and looks stalled on a long slow one; seconds bound the block at ~1 line/s whatever the render is doing |
| once per job | `JobPhases.first(key)` | one `RenderVideo` per job; a repeated mark in a loop cannot print a phase twice |

Nothing is deleted: the same call prints again with `RVE_LOG_DEBUG=1`.

## What the block was actually showing

`src/ui/render_log.py` already dropped the live progress line, `host timeline:`, `fp16 engine GPU:` and
`int8 body engine GPU:`. It did **not** match the int8 body's per-8-frame rows, so at ~30 fps the pane received
`int8 phases: ...` and `int8 tail: ...` about 3.75 times a second each - the "full per-frame timing breakdown every
second" in the report. `int8 body GPU:` (a different prefix from the one the filter matches) leaked as well.

## Emitter matrix

Frequency legend: **once** = per process; **job** = once per render job; **pass** = once per render pass; **req** =
per seek/preview request; **2/s** = the 0.5 s progress writer; **frame** = every rendered frame; **8f** = every 8th
frame/body call; **250f** = every 250 rendered frames.

### `backend/src/RenderVideo.py` (the render job's lifecycle)

| line | what it prints | frequency | level now |
|---|---|---|---|
| 89/101 | `borders: ...` / machine `BORDERS ...` record | job | info |
| 278/281/285 | working directory, invalid input | job | info |
| 325 | **new** `phase: video probe N ms (WxH, frames, fps)` | job | info |
| 326/348/351 | 10-bit / 8-bit pipe decisions | job | info |
| 396 | `Detecting borders` | job | info |
| 408 | `Detected borders: ...` | job | info |
| 418 | `Using backend: ...` | job | info |
| 427 | **new** `phase: model load upscale '<name>' N ms` | job | info |
| 429 | `Using Upscaling Model: ...` | job | info |
| 441/444 | `Using Extra Restoration Model` + **new** `phase: model load restoration '<name>' N ms` | job, per model | info |
| 451/453 | **new** `phase: model load interpolation '<name>' N ms` + `Using Interpolation Model` | job | info |
| 457 | `debug: after hotReload ...` (was `DEBUG after hotReload`) | job | **debug** |
| 423-430 | override/model scale/config lines | job | info |
| 515 | **new** `phase: reader open N ms` | job | info |
| 555 | **new** `phase: writer/encoder init N ms` | job | info |
| 519-545 | display / seek channel setup | job | info |
| 573-635 | reuse gate, auto-gate and correction config | job | info |
| 786 | **new** `phase: backend init N ms (backend, device, precision); N s since process start` | job | info |
| 725-730 | tile probe stats / reason / failure | job | info |
| 764-779 | int8 probe stats / verdict / failure | job | info |
| 839/976/987/1001/1010/1021 | auto-gate calibration and tuner summaries | job | info |
| 1042/1046/1374/1382 | reuse/correction/calibration failures | job (on failure) | info |
| 1115/1117 | run report written / failed | job | info |
| 1165/1226/1232/1252/1254 | seek request/preview lines | req | info |
| 1788 | `startup: first frame enqueued N s after process start` | job (guarded by `JobPhases`) | info |
| 1796 | `reuse: ...` progress | 250f, **now rate-limited to <=1/s** | info |
| 1822 | `render loop: N s for M frames (N ms/frame), total ...` | job (guarded) | info |
| 1465/1467/1487/1503 | temporal-reuse, tuner, tile and seek-reader summaries | job | info |
| 1834 | `profile over N frames: gate ... model wall ... unaccounted ...` | job | **debug** |
| 1512 | pyinstrument dump (inside a `"""` block, dead) | never | - |
| 1938/1946/2051 | `Setting up ...` | job | info |
| 1611/1617 | int8 calibration frames / forced fit | job | info |
| 1637/1648/1651 | determinism diagnostics (`RVE_DIAG_*`) | job, env-gated | info |
| 2024 | `debug: after assign ...` (was `DEBUG after assign`) | job | **debug** |
| 1692/1695 | scene detection on/off | job | info |

### `backend/src/utils/Util.py`

| line | what it prints | frequency | level now |
|---|---|---|---|
| 67 | `warnAndLog()` -> `WARNING: ...` | on warning | info |
| 76 | `log_error()` -> `ERROR: ...` | on error | info |
| 134 | `log(message, show_backend, level)` - the one backend emitter | every backend line | info; **debug gated here** |
| 15-20 | `PROCESS_START` / `process_uptime()` (new, not a line) | - | - |

### `backend/src/FFmpeg.py`

| line | what it prints | frequency | level now |
|---|---|---|---|
| 146 | `FPS: N Current Frame: N ETA: h:mm:ss` (in-place `\r`) | 2/s | info (the pane drops the copy and shows FPS/ETA from the same line) |

### `backend/src/pytorch/compact/int8_body.py`

| line | what it prints | frequency | level now |
|---|---|---|---|
| 231/323/328/346 | calibration cache / layout notes | job | info |
| 474/478/486 | calibration/grid notes | job | info |
| 628/637 | non-contiguous input, `RVE_DEBUG_INPUT` dump | once, env-gated | info |
| 661 | `int8 body GPU: N ms/frame (last 8), frame N` | 8f | **debug** |
| 679 | `int8 phases: head ... quantise ... hops ... tail ... SUM ... BODY-TOTAL ...` | 8f | **debug** |
| 690 | `int8 tail: dequant ... conv ... shuffle ... residual ... SUM ... ms` | 8f (per tail call) | **debug** |
| 707 | `int8 head FUSED: ...` | once | info |
| 723/776/818/826/843/851 | branch notes / certification / refusals | job | info |

### `backend/src/pytorch/TensorRTHandler.py`

| line | what it prints | frequency | level now |
|---|---|---|---|
| 181 | `TensorRT engine built in N s (raw builder).` | per engine build | info, **now `BACKEND:`-prefixed** |
| 193 | `Building TensorRT engine <name>. This may take a while...` | per engine build | info, **now `BACKEND:`-prefixed** |
| 246 | `TensorRT engine built in N s.` | per engine build | info, **now `BACKEND:`-prefixed** |
| 258 | `raw TensorRT plan kept at ...` | per engine save | info |
| 281 | `TensorRT engine already loaded in this process: ...` | per cached load | info |
| 286/288 | **new** `Loading raw TensorRT plan from ...` + `TensorRT engine loaded in N s (...)` | per engine load | info |
| 292/294 | **new** `Loading TensorRT engine from ...` + `TensorRT engine loaded in N s (...)` | per engine load | info |
| 299 | `TensorRT partitioning (loaded): N sub-engine(s)` | per load | info |

The four build/load prints used bare `print(..., file=sys.stderr)`. They reached the pane (stderr is merged) but
without the `BACKEND: ` prefix the pane's phase colouring and any prefix-keyed reader miss them; they now go through
`log()`, and load has a duration where before only build did.

### `backend/src/pytorch/UpscaleTorch.py` (out of this task's file scope - listed for completeness)

| line | what it prints | frequency | pane filter | policy verdict |
|---|---|---|---|---|
| 563 | `fp16 INPUT shape/dtype/mean...` | once, `RVE_DEBUG_INPUT=1` | not dropped | debug-only already |
| 597 | `fp16 engine GPU: N ms/frame (last 8); alloc_retries N` | 8f | dropped (`_PER_FRAME`) | candidate for `debug` |
| 646 | `host timeline: upload ... model-call ... total ... ms` | first 3 + 8f | dropped (`_PER_FRAME`) | candidate for `debug` |
| 662-767 | `int8: start/built/refused/calibration/grid/sanity` | job | not dropped | info |

These per-frame rows never reach the block today because the pane filter drops them. They are the remaining
emitter-level debt: the same `debug` gate should be applied when that file is next touched.

### `backend/src/utils/TemporalReuse.py`

| line | what it prints | frequency | level now |
|---|---|---|---|
| 128/675 | `reuse frame=N depth=N share=...` | frame, only `RVE_DEBUG_DEPTH=1` | debug-only already |
| 1034-1040 | `[tuner] ...` | job, only `RVE_TUNER_DEBUG=1` | debug-only already |

### Other backend prints that can reach the pane

| file:line | what | frequency |
|---|---|---|
| `backend/src/utils/SceneDetect.py:150` | `Transition Mismatch ... skipping` | per mismatch |
| `backend/src/FFmpegBuffers.py:811` | `Failed to read FFmpeg log file` | job failure |
| `backend/src/utils/Frame.py:39` | `Initialized Frame PyTorch utils` | once |
| `backend/src/utils/Util.py:63` | `Failed to remove file!` | on failure |
| `backend/src/utils/GetFFMpeg.py:18-30` | download progress | startup only |
| `backend/src/utils/VideoInfo.py:338-343` | `--print_video_info` CLI output | CLI mode, not a render |

### App side (`src/`)

| file:line | what it prints | frequency | destination |
|---|---|---|---|
| `src/Util.py:223` `log()` | every app event (downloads, settings, render notices) | event | `log.txt`; now `level=`-gated (`debug` needs `RVE_LOG_DEBUG=1`) |
| `src/ui/ProcessTab.py:871` | the backend command line | job | `log.txt` + block (via `log`) |
| `src/ui/ProcessTab.py:972` | each backend line the pane kept | per kept line | `log.txt` |
| `src/ui/ProcessTab.py:964/999/1001` | render failed / all completed / return codes | pass | `log.txt` |
| `src/ui/ProcessTab.py:1019/1154/1157/1167/1195/1212/1225/1229/1235/1241` | preview/queue pane events | event/req | `log.txt` |
| `src/ui/ProcessTab.py:1311/1330/1332/1344/1394/1477/1479/1492/1499/1512` | preview/queue decoder events | event | `log.txt` |
| `src/ui/ProcessTab.py:1582/1590/1656/1663/1673/1678/1683/1734` | seek events | req | `log.txt` |
| `src/ui/ProcessTab.py:1803/1813/1816/1821/1852/1863/1912` | compare/source-decode events | event | `log.txt` |
| `src/ui/render_log.py` | not an emitter - the consumer filter that classifies what survives | per line | block |

The pane itself needed no change: it paints whatever text it is given and already maps `phase`/`warn`/`error` to
colours. One optional follow-up is recorded below.

## Phases added (one line each, at `info`, with a duration)

| phase | where the timer lives | line |
|---|---|---|
| process start -> backend ready | `Util.PROCESS_START` (import time) to the end of `RenderVideo.__init__` | `phase: backend init ...; N s since process start` |
| video probe (input open/metadata) | around `OpenCVInfo` | `phase: video probe N ms (WxH, frames, fps)` |
| model load, upscale (by name) | around `setupUpscale()` + `hotUnload()` | `phase: model load upscale '<name>' N ms` |
| model load, restoration (by name) | around each `setupExtraRestoration()` | `phase: model load restoration '<name>' N ms` |
| model load, interpolation (by name) | around `setupInterpolate()` | `phase: model load interpolation '<name>' N ms` |
| TensorRT build / load | existing `start_time` in `build_engine`, new timer in `load_engine` | `TensorRT engine built/loaded in N s ...` |
| reader open | around `FFmpegRead(...)` | `phase: reader open N ms` |
| writer/encoder init | around `FFmpegWrite(...)` | `phase: writer/encoder init N ms` |
| first frame | existing `_T0` mark, now `JobPhases`-guarded | `startup: first frame enqueued N s ...` |
| total and ms/frame | existing render-loop summary | `render loop: N s for M frames (N ms/frame), total ...` |

Still not timed because the code has no such point: **device/context creation** (CUDA context and cuDNN handle are
created lazily by the first model call inside `setupUpscale`, so it is inside the model-load phase rather than a
separate line); **ffmpeg/NVENC encoder open** (the writer process is spawned in `FFmpegWrite.__init__`, but its
first frame is only accepted after `start()`, which this file never calls separately - the writer-init phase covers
the spawn and the pipe setup); **the encoder's own first-frame latency** (reported only as part of first-frame ->
render-loop).

## Follow-up the pane would need (not done here)

`src/ui/render_log.py` drops `host timeline:` / `fp16 engine GPU:` unconditionally, so `RVE_LOG_DEBUG=1` restores the
int8 and profile rows in the block but not those two (they are emitted by `UpscaleTorch.py`, outside this task's
scope). When that file is gated, the filter should also read `RVE_LOG_DEBUG` and stop dropping the per-frame class.
