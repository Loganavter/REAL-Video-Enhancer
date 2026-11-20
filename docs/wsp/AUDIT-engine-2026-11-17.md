# Engine/kernel queue audit — 9 items against the record

Cell: `rve-audit-engine`, read-only, branch `agent/rve-audit-engine/work`, HEAD `03acc346`.
Question per item: is it answered by the record, partly answered, genuinely open, or not needed.

Read, in this order: `docs/backlog/queue.yaml` (the 9 full entries), `docs/RESEARCH-LEDGER.md`
(all E/B sections, "Backlog, ranked", "What the literature closes", "Open questions"),
`docs/MODEL.md` §4-6, `docs/REPORT-int8-lane-2026-11-16.md` (the later measurement — 2026-11-16,
which most of these queue notes predate), `git log` after 2026-11-15, and the probes:
`scripts/{force_tactic,inspect_plan,make_width_probe,bench_plan,probe_channel_width}.py`,
`backend/src/pytorch/{RawTRTEngine,Int8Kernels,Int8Body}.py`.

**The one fact that reorganises this audit:** the int8 lane was *built and run* on 2026-11-16,
after every one of these items was filed on 2026-11-15. Three of the nine are answered by that
build, not by the ledger section their own note cites.

| id | verdict | citation | one sentence | what would close it |
|---|---|---|---|---|
| rve-int8-prelu-plugin (E27) | **a** | `docs/RESEARCH-LEDGER.md:1019-1046`; `docs/REPORT-int8-lane-2026-11-16.md:121-145`, `:8-30`; `backend/src/pytorch/Int8Kernels.py:31-65` | The plugin-equivalent exists as one Triton kernel that does the 9-tap int8 conv, the per-channel PReLU and the requant (`tl.dot` → `v*slope` → `q*scale>>shift`, one kernel), it meets the block budget at 1.753 ms/hop in the requant-in-epilogue scheme and 2.198 ms in the harness against 2.93 fp16, and the frame lands at **0.991x, not 1.40x**. | Nothing on the kernel. The remaining work is the product question the report answers negatively (§1): serve the body under the gate instead of disabling the gate (3.3x) to serve it. |
| rve-int8-prelu-pass (E32) | **a** | `RESEARCH-LEDGER.md:811-818`; `REPORT-int8-lane-2026-11-16.md:121-145` | The pass is at its traffic floor and the later build removed it: requant moved from the prologue (9× per element) to the epilogue (1×) took a hop from 2.518 to 1.753 ms and 16 hops from 40.3 to 28.0 ms. | Nothing. Keep the row as documentation only, so the 42.7 ms arithmetic is not re-derived a third time. |
| rve-int8-gemm-efficiency (E33) | **b** | `RESEARCH-LEDGER.md:838-876`, `:1837-1851`, `:2038-2039`; `MODEL.md:325-334` | Measured: the card's int8 peak (147.4 TOPS), TRT's int8 conv at 91 TOPS = 62% of it, and the arithmetic that 2x needs 120 TOPS; not measured: **any** tiling or source sweep of the int8 *plan* (E14 did this for fp16 only), and our own kernel's ceiling was attributed to registers/occupancy (255 regs, 16.5%), not to tactics. | 3 tiling levels × 2 source sets on the existing Q/DQ ONNX, read with `harness/harness/scripts/bench_plan.py --profile` (added in `39bd227b`), 3 interleaved rounds ≈ 2 h GPU — worth spending only if the int8 lane gets a product path (`REPORT:8-30`). |
| rve-welder-onnx (E28) | **d** | `RESEARCH-LEDGER.md:853-859`, `:775-779`, `:260-262`; `MODEL.md:284-291` | At fp16 the whole board is 1.34x, measured from the peak; Welder's own paper reports *losing* on conv-dominated stacks; and the axis it optimises (whole-graph traffic) is not the measured binding constraint (registers/occupancy at int8, 69.2% SM throughput with both pipes held down). | Nothing. A day of building a TVM-era research artifact cannot beat a measured hardware peak, and it addresses neither the per-hop boundary nor the register frontier. |
| rve-l2-pyramid (E29) | **b** | `MODEL.md:266-291`; `RESEARCH-LEDGER.md:1816-1828`; `queue.yaml:250-260` | Measured: the traffic decomposition *as a bound* (9.37 GiB logical, 34.9 ms if all of it reached DRAM, 55-57 measured, slope 2.46 against 3.73 ms/GiB, "about a third of the logical traffic never leaves L2"); not measured: the per-boundary buffer budget (0.49 MB × 16 = 7.9 MB exists **only** in the queue note), the real DRAM split on the fp16 plan, and the 4K case. | ncu on the shipped fp16 plan (counters unblocked, `RESEARCH-LEDGER.md:1818`) for `dram__bytes` + L2 hit per body kernel, plus a 20-line boundary-budget script at 1080p and 3840×2160 — one GPU job, hours. |
| rve-stage-pipelining (E35) | **c** | `MODEL.md:229-236`, `:299-301`; `RESEARCH-LEDGER.md:1830-1835` | Nothing in the record enumerates stage variants — the only evidence is the `_stage1_` suffix inside a kernel name, and MODEL.md already warns that "the name is a label, not a capability list"; the adjacent Triton sweep has `num_stages` 1/3/5 within 1%. | `harness/harness/scripts/force_tactic.py --channels 64`: one-conv graph, scan the choice indices and grep the resulting tactic names for `stage[2-9]`. ~2 h, and the item's own refutes (stage1 is the whole menu) closes it either way. |
| rve-fusion-break-map (B1) | **d** | `RESEARCH-LEDGER.md:956`, `:961-964`, `:1714-1717` | Fusion does not break by width — 64/48/40/32 all report "16 fused, 0 separate PReLU" — and every width below 64 is already excluded on picture by E30b, so a 36/44/56 table cannot change a shippable decision. | Nothing worth 2 h. If the family switch must be *documented*, 36/44/56 via `make_width_probe.py` + `inspect_plan.py` is 1 h, and it buys a table, not a lever. |
| rve-force-epilogue-tactic (B2) | **a** | `RESEARCH-LEDGER.md:1769-1777`, `:956` | The acceptance is met by measurement — 16 fused myelin layers and zero separate PReLU passes in all three builds, body convs at 1.17-1.42 ms, frame 22.41-24.88 ms = 2.07-2.36x — although the trigger was an 8 GB workspace pool, not a forced tactic index. | Nothing on timing. The unclosed half is *attribution* (pool vs. tactic-selection instability, "not yet separated"), and at C=64 E5 already measured an 8 GB workspace at 0.995x, so there is no shipped-engine prize behind it. |
| rve-prelu-rewrite-at-32 (B3) | **a** | `RESEARCH-LEDGER.md:956`, `:1772-1774`, `:92`, `:1809-1812` | PReLU *does* fuse at C=32 in a working build (16 fused, 0 separate, 3/3), and when it does not, the winning narrow-width families take no epilogue at all — so a rewrite of the activation has nothing to win; E17's mechanism (one pass traded for two) is unchanged. | Nothing. C=32 is excluded on picture regardless (E30b), so the 1 h three-form probe would produce a table with no consumer. |

Counts: **a = 4, b = 2, c = 1, d = 2.**

## Quotes behind the (a) and (d) verdicts

**E27 (a)** — `RESEARCH-LEDGER.md:1040-1042`:
> "**So the budget is met**: <=1.9-2.0 ms and >=76-80 TOPS, achieved at 1.91-1.93 ms and 79-80 TOPS.
> Against the fp16 fused block's 2.93 ms, a block is 1.53x cheaper while carrying the per-channel slope
> and a per-channel output scale - the two things TensorRT's epilogue could not take."

and the later build, `REPORT-int8-lane-2026-11-16.md:143-145`:
> "То есть полоса **вышла в паритет с движком** и это ещё до двух названных фиксов … проекция
> `51.54 − 15.5 = 36.0` мс → **1.42x** в продуктовых числах, то есть выше ворот 1.40 ещё без выигрыша
> от интеграции под гейт."
> (= the lane reached parity with the engine, and the 1.42x is a *projection* from two unfixed items,
> not a measured frame.)

The decision this fed, `queue.yaml:938-943`:
> "the only remaining path through that ceiling is a plugin with its own int8 convolution
> (rve-int8-prelu-plugin, E27), which is a different project."

**E32 (a)** — `RESEARCH-LEDGER.md:816-818`:
> "The PReLU pass is already at its own traffic floor: 530.8 MB at 288 GB/s is 1.84 ms against 1.71-2.37
> measured. There is nothing to win there - the pass has to disappear into the convolution, which is the
> plugin."

and the pass disappearing, `REPORT-int8-lane-2026-11-16.md:129-131`:
> "| переквант в прологе (было) | 2.518 | 40.3 |
> | **переквант в эпилоге (стало)** | **1.753** | **28.0** |"

**B2 (a)** — `RESEARCH-LEDGER.md:1772-1777`:
> "The same synthesized C=32 model that measured 45.69 ms with the PReLU split out rebuilt three times
> with an explicit 8 GB workspace pool: **24.88, 24.06, 22.41 ms, sixteen fused myelin layers and zero
> separate PReLU passes in all three.** The body convolutions come in at 1.17-1.42 ms each, i.e. at the
> 1.3 ms bar that makes the frame 2x. Against 56.68 ms at C=64 that is **2.28-2.53x**, and the spread
> across three identical builds (1.11x) is smaller than the effect."

**B3 (a)** — `RESEARCH-LEDGER.md:961-964`:
> "so the "fusion break" row is retired, and the honest statement is that a build can land in either
> family, while the fused one reproduces 3/3 for C=32 in this run."

with the width table at `:953-956` reporting "16 fused, 0 separate PReLU" for C=64, C=48, C=40 and C=32,
and the mechanism at `:1809-1811`:
> "the tactic family changes, and both families that win at narrow widths - `scudnn_winograd` here,
> `sm80_xmma_..._indexed_wo_smem_...` in the full model - take no epilogue, so myelin leaves the PReLU
> outside."

**E28 (d)** — `RESEARCH-LEDGER.md:856-859`:
> "That closes the "the ceiling is unknown" question and it makes fp16 arithmetic final: the body is
> 2.446 TFLOP, so even at a perfect 71.6 the floor is **34.2 ms**, i.e. 42 ms for the frame, i.e.
> **1.34x is the entire fp16 headroom**. 2x is unreachable without leaving fp16."

and `:775-776`:
> "So the literature's claim does not transfer, and the reason is visible in the numbers: Welder's
> authors compared against TensorRT 8.x on an V100/3090 where an fp16 Winograd kernel existed and won."

**B1 (d)** — `RESEARCH-LEDGER.md:1714-1717`:
> "So the pruning lane closes on a measurement rather than on a disappointment: **C=32 buys 2.07-2.36x of
> time and cannot pay for it in features; C=48 buys 1.2x and has the same picture as C=32** (the error
> does not scale with the cut, and 48 is the same cut on this metric). Nothing between them is worth the
> attempt."

## Queue notes that are stale (contradict the ledger or the report)

1. **`rve-int8-prelu-plugin` (queue:104-116)** — `blocked_on: "an int8 GEMM we control: the TRT python wheel
   ships no NvInfer.h"` and `"BLOCKER 2026-11-17: effort … a CUDA/CUTLASS kernel … is weeks"`.
   Contradicted by `backend/src/pytorch/Int8Kernels.py:31-65` — the kernel exists in Triton, not CUTLASS,
   and is measured. Its `measured:` field also still sells the 1.40x ceiling as the target, while the
   delivered frame is 0.991x (`REPORT:143-145`). **Most expensive stale note on the board**: read
   literally it sends the next agent into weeks of work that is already done and already measured
   not to pay.

2. **`rve-force-epilogue-tactic` (queue:525-547)** — `refutes: "No tactic in the epilogue-capable family
   can be built for C=32"` and the note's open question "whether the selector can identify a tactic by
   family". Both superseded: the fused tactic builds at C=32 with no selector at all
   (`RESEARCH-LEDGER.md:1769-1777`).

3. **`rve-fusion-break-map` (queue:499-505)** — `note: "at C=32 it runs sm80_xmma_..._indexed_wo_smem_...
   and the PReLU becomes 16 separate passes worth 23 ms"`. The ledger retires exactly this reading:
   `"the \"fusion break\" row is retired"` (`:963`), and `:996-997` records the 45.69 ms build as *one*
   build, with 16 fused convs.

4. **`rve-prelu-rewrite-at-32` (queue:574-577)** — `note: "E17's negative result was measured at C=64 …
   so it says nothing about C=32, where the tactic family is different."` Contradicted: the same family
   test at C=32 shows fusion working (`:956`, `:1772`), so no width-specific gap is left for the rewrite.

5. **`rve-l2-pyramid` (queue:256-260)** — self-contradictory inside one note: `"BLOCKER 2026-11-17: needs
   ncu counters (see rve-ncu-access)"` followed immediately by `"RUNNABLE NOW 2026-11-17: ncu is unblocked
   on this host"`. `rve-ncu-access` is `done` (commit `03acc346` verified `RmProfilingAdminOnly: 0`).

6. **`rve-int8-gemm-efficiency` (queue:179-183)** — `"Closing that 55-60% -> 75% gap is the only path from
   40 ms to 28 ms"`. The 40 ms build is the *scalar-slope* one, which the ledger itself labels
   "a **measured ceiling of the wrong function**" (`:1006`, picture −2.02 dB), and the delivered int8
   frame is at parity (`REPORT:121-145`). The target frame is stale even though the TOPS question is not.

7. **`rve-stage-pipelining` (queue:434-439)** — `"The chosen kernel is tilesize128x64x32_stage1_…"` and
   `"an editable timing cache or a forced tactic is a cheap 10-20%"`. "The chosen kernel" is not a stable
   fact: the same builder picks two different families depending on the build (`:1769-1782`,
   `"a build can land in either family"`), and E14 measured source restriction at 0.998x. The row's
   *question* is still open; its *premise* is not.

Record staleness outside the queue, for whoever reconciles next: `docs/MODEL.md:403-404` still ends
with "`ncu` needs `RmProfilingAdminOnly=0` (one line plus a reboot), and until then the DRAM split stays
a bound" — counters have been unblocked since 2026-11-15 (`RESEARCH-LEDGER.md:1818-1828`); and
`MODEL.md:211` still lists Winograd as "no / not measured", which `:1793-1807` retracts.

## What this audit changes

The engine/kernel backlog is close to exhausted: four of nine are answered, two cannot change a
decision, two have one named piece missing, and exactly one (E35) is genuinely open. The plan's 2x
question has moved out of the queue and into `docs/REPORT-int8-lane-2026-11-16.md`, which the ledger
does not yet cite — **that gap is the main finding**: the queue still describes E27 as weeks of unbuilt
CUTLASS work, the ledger still describes int8 as "closed as a TRT configuration", and the actual state
is a delivered Triton body at parity with no product path.

Three items worth doing next, in order:

1. **E35** — 2 h, the one genuinely open row, on the engine that actually ships.
2. **Reconcile the record** — fold `REPORT-int8-lane-2026-11-16.md` into `RESEARCH-LEDGER.md` and fix the
   seven stale queue notes above. Two of them (E27's blocker, B2's selector question) actively mislead.
3. **E33's sweep** — 2 h, real and cheap, but only after the int8 lane's product question is reopened
   (`REPORT:716-721`: serve the body under the gate); otherwise it measures a kernel nobody ships.
