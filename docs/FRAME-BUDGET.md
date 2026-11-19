# The frame budget, as it stands

**Generated** by `scripts/frame_budget.py` - do not edit. Every number is parsed from the render log
named below or resolved by id from `docs/wsp/verified/records/`. The prose says what each line IS and
what its floor is, so a reader who was never in this lane can challenge any of it.

## The machine, the model, the build

| | |
|---|---|
| card | NVIDIA GeForce RTX 4060 Ti, 34 SMs, **176 dense int8 TOPS** and **288 GB/s** (the two roofs every number below is measured against) |
| ridge | 176 TOPS / 288 GB/s = **611 FLOP/byte** - and the hop sits at 610, i.e. exactly on it |
| clip | `bd_violet20s` - BD Violet, 494 frames, 1920x1080 in, 4K out |
| model | `2x-Futsuu-Anime(1).pth`, SRVGGNetCompact 2x, 64 body channels, 18 convolutions |
| precision | fp16 engine vs the int8 body: 8-bit codes between the 16 body hops, fp16 head and tail weights where noted |
| build | all four opt-ins ON: fused head kernel, int8 tail kernel, tail shuffle+residual inside that tail kernel, zero-halo hop buffers |
| the bar | `bar_int8_wall` = **1.2x** wall clock against the fp16 engine; the picture bar is the share of pixels more than 8 levels off on the worst frame of a spread sample, 0.1% |

| commit | `9b814a0d` - the tree these numbers were taken on |

The pipeline, in the order a frame is paid for: **decode** (ffmpeg, outside the loop) -> **reader** (yuv420p10 -> a tensor) -> **head** (3 -> 64 channels) -> **sixteen hops** (the body) -> **tail** (64 -> 12 channels, pixel shuffle to 3 at 2x) -> **residual** (add the upsampled input) -> **pack** (4K payload to the host) -> **writer** (ffmpeg stdin). The render loop below is the wall clock of that chain; the body is what the model costs, measured by the app's own events.

## The body, from the app's own phase counter

| line | ms | share | what it is |
|---|---|---|---|
| head | 0.60 | 2% | the first convolution, 3 -> 64 channels, 3x3, fused with its PReLU and with the quantisation into int8 codes (one kernel) |
| quantise | 0.00 | 0% | 0.00 by construction: the head writes codes directly, so this pass no longer exists |
| hops | 24.30 | 94% | the body proper: sixteen 64 -> 64 int8 convolutions, 3x3, one Triton kernel launched sixteen times |
| tail | 1.00 | 4% | the last convolution 64 -> 12 channels, its pixel shuffle to 3 channels at 2x, and the residual add - all three in one kernel now |
| **BODY TOTAL** | **25.90** | 100% | measured with the app's own CUDA events, window by window |

Tail split on the same run: dequant 0.00, conv 1.01, shuffle 0.00, residual 0.00, sum 1.02 ms. The zeroes are the second adoption: the dequant pass was folded into the tail's weights and the shuffle+residual into its epilogue.

## The hop against its two roofs - why every single lever came out flat

| quantity | value |
|---|---|
| one hop, measured | 1.519 ms |
| its compute floor (152.9 GFLOP at 176 TOPS) | 0.869 ms = **57% of the roof** |
| its DRAM floor (250.6 MB at 288 GB/s, counter-measured) | 0.870 ms = **57% of the roof** |
| the two floors agree | that is what 'on the ridge' means: 610 FLOP/byte against a ridge of 611 |
| ncu's own reading | Compute (SM) 74%, DRAM 56%, LSU the busiest pipe (73.9% by instruction count), L2 6.1x the DRAM traffic |

The 74% and the 57% are not a contradiction: ncu measures **pipe saturation** (including the load/store unit), the 57% measures **arithmetic against the int8 roof**. Both say the same thing - the kernel presses on compute and on memory at the same time, in equal measure - and that is why a change that improves only one of the two reads as flat noise. Thirty configurations in seven families were tried on this line; every
one lost (the table three sections down).

## The frame

| what | ms | floor, and where the floor comes from |
|---|---|---|
| body (above) | 25.90 | **13.9 ms** for the hops alone at the 176 TOPS roof (152.9 GFLOP per hop x 16); DRAM floor **13.9 ms** (250.6 MB per hop x 16 at 288 GB/s) - the two agree, which is the ridge again |
| pack + queue | 4.36 | **1.9-2.4 ms**: the 4K yuv payload is 12.44 MB and the link is **PCIe gen 3 x8** - measured under load at 6.55 GB/s round-trip, 25 of 25 samples, against 7.88 GB/s theoretical. The card is a PCIe 4.0 part but the slot negotiates gen 3, so this floor is a property of the motherboard |
| **render loop** | **33.58** | what the user waits for; `null` sink, so no encoder |
| everything else | +3.32 | reader, the wrapper around the model call, and the writer's back-pressure. The input path inside it is measured at 1.38 ms, of which 0.07 was removable |

## What is in this frame because it was measured and adopted

| change | saved on the loop | records |
|---|---|---|
| head as one fused kernel | -5.27 ms (11.9%) | int8-head-fusion:H006 / H007 |
| tail as a 17th int8 hop | -3.16 ms (6.9%) | int8-tail-hop:C001 / C002 |
| tail shuffle + residual folded in | -0.80 ms (2.0%) | int8-tail-shuffle:Q005 / Q006 |
| zero-halo hop buffers (mask-free) | -0.98 ms (2.8%) | int8-hop-halo:H004 / H005 |

## What was measured and deliberately NOT adopted

An outsider reading this list is the intended use: every entry is a lever someone expected to work.

| lever | what it was supposed to buy | what it actually read | against | record |
|---|---|---|---|---|
| epilogue divide folded into a and b | 1.3-2.7 ms: one fp32 divide per element, in the kernel all 16 hops run | **-0.655 ms** | 16 hops chained, 29.3 ms before it | `docs/wsp/verified/int8-epilogue-fold.txt` : R002 |
| 6-bit codes between hops | 25% of the hop's 265 MB of traffic | **2.71431 %** | picture bar 0.1% of pixels | `docs/wsp/verified/int8-sub8-codes.txt` : S005 |
| vectorised loads + weight hoist | the hop is at 52% of both roofs, so its schedule | **1.772 ms** | shipped hop 1.756 ms | `docs/wsp/verified/int8-hop-loads.txt` : L002 |
| persistent-CTA grid | the same schedule, one CTA per SM looping over tiles | **3.709 ms** | shipped hop 1.703 ms | `docs/wsp/verified/int8-hop-persistent.txt` : N002 |
| cache eviction / modifier policy | L1 hit rate is only 22%, L2 traffic 6.1x DRAM | **1.594 ms** | shipped hop 1.580 ms | `docs/wsp/verified/int8-hop-cache.txt` : C002 |
| 2-D CTA tiles | halve the window amplification (3.05x -> 1.59x) | **1.55 ms** | shipped tile 1.504 ms | `docs/wsp/verified/int8-hop-tiles.txt` : T004 |
| output channels split across CTAs | halve the accumulator to double resident warps | **2.632 ms** | shipped hop 1.591 ms | `docs/wsp/verified/int8-hop-csplit.txt` : X002 |
| M x 2 batching | amortise the 0.59 ms per-launch constant over two frames | **1.764 ms** | batch 1 of the same kernel 1.769 ms/frame | `docs/wsp/verified/int8-hop-batch.txt` : Y002 |
| pack ring (pinned slots) | the 12.44 MB payload copy is synchronous | **34.36 ms** | ring off 34.42 ms loop | `docs/wsp/verified/int8-pack-ring.txt` : K002 |
| reverse tile order on alternate hops | the previous hop's tail is still in L2 (32 MB of 132) | **26.577 ms** | forward 26.551 ms over 16 hops | `docs/wsp/verified/int8-hop-traverse.txt` : V002 |

## What is still open, and what it costs

| idea | expected | what it needs | state |
|---|---|---|---|
| fuse two hops per launch (cascade) | **+3.1 ms/frame** at T=32: 1.064x of work against ~250 MB of DRAM per fused boundary removed (the launch-constant argument is dead) | a CUDA kernel; the intermediate is 34^2 x 64 B = 74 KB, which fits 100 KB of shared memory | `rve-hop-fuse2-cascade`, P2, unmeasured |
| shared-memory staging of the 3-row window | the 0.59 ms first-tap cost, if it is a round trip (it is NOT a per-launch constant: the same kernel at 256x256 takes 0.057 ms, so the implied launch term is 0.011 ms - `int8-hop-constant.txt`) | the same CUDA kernel; measured negative in its two cheap forms (reordered loads -101%, a CUDA microbenchmark -26%) | closed, `int8-hop-prefetch.txt` |
| **overlap the IO with the body** (three streams, double buffer) | **up to 7.68 ms** - read, body, pack and write are strictly sequential today | software only, no new kernel | `rve-frame-pipeline-overlap`, P1, unmeasured |
| a PCIe 4.0 slot | ~1.2 ms off `pack`, since the link is gen 3 x8 today | hardware | not software |
| skipping duplicate frames / static regions | large on anime, and the owner's call forbids it | - | closed by policy, `rve-reuse-gate-forbidden` |
| cut the model (fewer channels or hops) | 2.9-4.5 ms per hop removed | 14-37% per-layer residual, 1.58% of pixels over the picture bar | closed by measurement |

## Which side of the ratio a millisecond lands on

The bar is a RATIO, `fp16 / int8`, so a millisecond saved is not worth the same everywhere:

| a millisecond saved in | int8 becomes | fp16 becomes | ratio |
|---|---|---|---|
| the int8 body only (a kernel) | 32.58 | 52.12 | **1.600** |
| a stage BOTH arms pay (pack, reader, writer) | 32.58 | 51.12 | 1.569 |

So a body-side millisecond is worth ~2.8x a shared one against this bar, and the largest wall-clock win in
the budget can be the smallest ratio win. That is why `rve-hop-fuse2-cascade` (body) and the pack (shared)
must not be reported as if their milliseconds were interchangeable.

## The ratio this produces

Measured last in `logs/wall-final` (Violet, null sink, 3 rounds, the shipped default):
`int8-wall-final.jsonl` W009 = **1.552x** against `bar_int8_wall` = 1.2x. Older, pre-halo:
`int8-wall-defaults.txt` 1.399x / 1.481x null and 1.511x / 1.506x under `hevc_nvenc`.

## What this file is NOT

- **Null sink.** No encoder is in the loop; under `hevc_nvenc` the same comparison read 1.51x, and the
  loop itself is ~10 ms higher on that class of frame. The product runs with an encoder.
- **One clip, one model, one card.** The gains were replayed on three clips (1920x1080 8-bit, native
  1440x1080 10-bit, and a same-shot cut) and hold within 20% of their size; other cards are untested and
  the lane has external evidence that int8's advantage is a property of the card.
- **Not a picture claim.** The picture is judged elsewhere, on 494-frame lossless pairs, and the two
  boundary changes are byte-identical (`int8-tail-hop.txt`, `int8-hop-halo-chain.txt`).
- **Not the whole story of the hop.** The hop has its own record: 30 configurations in seven families,
  ncu counters, DRAM/L2 traffic, and the page of levers that lost (`int8-hop-occupancy.txt`,
  `int8-hop-dram.txt`, `int8-hop-ncu.txt`).

## Regenerate

```bash
python scripts/frame_budget.py --logs docs/wsp/verified/logs/wall-final --arm int8
```

