# Queue audit, 2026-11-17: four pipeline items and sixteen closed ones

Research cell, read-only. Base commit `03acc346` (clean tree). Read: all 31 rows of
`docs/backlog/queue.yaml`, all of `docs/RESEARCH-LEDGER.md` (2362 lines), `docs/MODEL.md`, the
`docs/wsp/verified/` records and the harness.

Mechanical checks used below (all reproducible in the cell, no GPU, no network):

- `PYTHONPATH=. python3 -m harness.cli below-report docs/claims-*.txt --bars docs/claims-bars.json`
  → `29 @below assertions: 24 under the bar, 2 OVER, 2 against a literal, 1 unknown`.
- `pytest --collect-only` on the harness pair → 63 tests at HEAD, **62 at `1f908c45`** (the commit
  that closed `rve-harness-redteam-residual`; one test was added afterwards).
- `claims-bars.json` at `07d63c36` (the commit that closed `rve-bars-populations`) → **8 bars**
  (10 now; two were added by later items).
- `docs/wsp/lean/Statistic.lean`: 87 lines, zero `sorry`.
- `harness/tests/test_harness_evidence.py` + `test_harness_redteam.py` at `1f908c45`: 48 + 14 = 62.

Those checks are in the report because each one *looked* like a contradiction and is not.
That distinction is the whole value of the exercise.

Not observable from a cell: the real `fork`/`origin` remote name (a cell's `origin` is the local
source checkout), whether `~/.elan/bin/lean` exists, `.venv` (gitignored, absent in every cell),
and the GPU. Nothing below depends on those except where marked unverified.

---

## Table A — pipeline items

| id | verdict | citation | code path (file:function) | what would close it |
|---|---|---|---|---|
| `rve-cone-recompute-2026-11-15` (E36) | **PARTLY** | The cone arithmetic already exists for the **1-D** case: ledger 1555-1557 — "For a linear change of width 2 px and length L, the cylinder costs 16\*38\*L and the cone L\*sum(2+2k) = 304L - exactly 2x less work, i.e. 26.1% -> ~13% and 1.40x -> 1.71x." The measured cylinder cost it would improve is ledger 1580-1597 and 1623-1637. The queue asks for a different geometry — radius r ∈ {5,10,20,40}, 2-D, plus "the per-frame launch cost of the extra regions" (queue 463-466) — and no r-table exists anywhere in the ledger or MODEL. | Cylinder: `backend/src/pytorch/TiledUpscale.py:tile_window` (one 18 px halo for the whole engine) and `harness/scripts/routing_cost.py:main` (HALO = 18 constant, `mask_from`/`grow`). Cone: **none**. Per-layer execution does not exist in fp16 — `backend/src/pytorch/RawTRTEngine.py:forward` executes one whole-plan context; the only layer-by-layer path in the repo is int8 and 64-channel-only: `backend/src/pytorch/Int8Body.py:Int8Body.__call__` (16 Triton hops). | Phase 1: add halo `k` at layer `k` to `routing_cost.py` and print the work ratio + region count for r ∈ {5,10,20,40}. Decision rule already exists: must beat the tile path's **1.17x wall at 0.071%** (ledger 1633-1636). Its only consumer, `rve-spatial-routing`, is dropped — but the ledger calls the cone "the only lever on this lane's numerator" (1557), so a 2x on that numerator would re-open it. Phase 2 is blocked on an engine architecture the repo does not have (per-layer regions), not on arithmetic. |
| `rve-shot-cascade-2026-11-15` (A) | **GENUINELY OPEN** | The 78-model collection has been touched only at load/forward/ONNX-export: ledger 599-604 — "0 load failures, 0 forward failures, 0 ONNX export failures" — and the raw-path correctness check on two architectures (608-615). No line anywhere compares the *time* of two collection members, and none compares their pictures on one shot. The nearest per-shot sentence is an instrument recommendation for the int8 gate probe (1405: "the decision is per video (or per shot)"), not a measurement. The seam danger the item reasons about is recorded at tile granularity (1546-1553, "the seam creeps") — that is a tile result and supports the item's own choice of shot granularity rather than answering it. | Shots exist, routing does not. `backend/src/utils/SceneDetect.py:NPMeanSCDetect.sceneDetect` (also `NPMeanSegmentedSCDetect`), consumed in `backend/src/RenderVideo.py` (658-662) only for interpolation (`transition=sceneDetect`). One model per render: `RenderVideo.__init__` sets `self.upscaleModel` (136) and loads one engine (285, 528). No collection loader, no per-shot model switch. | Two numbers, in this order: (a) ms/frame of the cheapest collection member against the owner's model (one GPU job — the ledger has none), then (b) the acceptance statistic for that pair over 2-3 frames of one shot. If (b) passes, the plumbing is small: the gate already exposes `ReferenceFrameGate.last_scene_cut` and the renderer already resets state on it (570-581). The refutes condition (cheap model over the bar, or a visible texture change at the cut) is decided by (b) alone — offline, no engine build. |
| `rve-hr-reference-cache-2026-11-15` (E37) | **PARTLY** — its primary premise is already answered, and the answer is "storage was not the limit" | The motivation is the 1.5x lost to a chain of depth 1 on Chainsaw Man (queue 669-674). The ledger measured that loss and **recovered it without any cache**: 351-366 — "the depth was not free because the rule was wrong, and fixing it paid 1.5x"; 83/220 at 9.94 s → **146/220 at 6.56 s**, defect unchanged 0.005%, via `AutoGateTuner._decide` filtering unreachable depths plus the dead scene-cut guard. The item's own refutes condition is the ledger's measurement at 359-363: "calm frames read 0.2 at depth 1 and 16-35 at depths 2-4" — i.e. depth 1 *is* the best available reference on that title. Forcing deeper depths buys 6.56 → 5.82 s on one title and makes Violet worse (0.124%, 379-381). So the first-order claim ("reachability is a property of the storage, and it cost 1.5x") is answered negatively. What is genuinely open is the second-order tile-coherence argument (queue 674-678), which only affects the tile path — shipped off by default at 0.071% (ledger 113). | One reference only: `backend/src/utils/TemporalReuse.py:ReferenceFrameGate.__init__` keeps `self._ref` (block grid) and `self._ref_luma` (264-265); `decide_reference(delta, ...)` at 185 takes a scalar delta against that one ref. A grid history exists but is capped at the chain length and stores no HR frames: `RenderVideo._hr_history` (570-612) and `TemporalReuse.worst_chain_delta(grid, history, depth)` (460-475). No signature search, no N-reference selection. | Only the tile half is worth measuring: keep ≥2 HR refs, search by block-mean signature, tile path on, measured against the *current* chain-2 baseline **146/220 6.56 s / 142/219 6.92 s** (ledger 371-374 — the queue's 6.61/6.90 is MODEL's older pair). If the reused tiles can all come from one cached reference, the 4x-dilation tax is removable and the 0.071% path gets cheaper; if not, the item closes on its refutes condition, which the ledger has already met for the chain itself. |
| `rve-strip-concurrency-2026-11-15` (C) | **GENUINELY OPEN** (and one of its premises is stale) | The only stream experiment in the ledger is E13 (line 88): "no gain (4.25 -> 4.27 s) and output corrupt (25 dB) - because two `torch.jit.load` modules of one engine file are not independent contexts", status "**rejected as wired** (a wiring fault, not a GPU property)". So the GPU question was never answered, exactly as the item says. The numbers in its own refutes line (2 strips 1.067x, 3 strips 1.10x, 4 strips 1.133x, queue 700) are the **halo-area tax**, not a measurement, and they appear nowhere in the ledger. The note's "~53% of DRAM and ~61% of the fp16 peak" (705-707) is also not in the ledger; the counters that exist are for the int8 block (59.4% DRAM, 51.2% tensor pipe, 90% L2 — ledger 1818-1827). | **None.** `backend/src/pytorch/RawTRTEngine.py:forward` runs one execution context on `torch.cuda.current_stream` (73-77); `TiledUpscale` composes tiles sequentially (`TiledUpscaler.plan`/`compose`); `backend/src/pytorch/TorchUtils.py:init_stream` exists but the render loop never takes a second stream for the upscaler. | The acceptance run itself: 1/2/3/4 strips, halo 18, one context and one stream per strip, against the single-stream build, with the area tax as the null. `ncu` is available again (ledger 1818) and would say in advance whether either pipe has room — the item's refutes line still says "they are still blocked", which is the stale premise. |

Two notes that apply to all four: (1) each item is blocked on *pipeline-level* code, not on
arithmetic, and only E36's phase 1 is hours of work; (2) for E36 and C the cheapest step is a
script extension that runs on one frame, so "pipeline-level" is a reason to sequence them last, not
to leave the note as "BLOCKER".

---

## Table B — closed items

Sixteen rows have status `done` or `dropped` (10 + 6). "Contradiction" means a later measurement in
`docs/RESEARCH-LEDGER.md`, `docs/MODEL.md` or the verified records contradicts the resolution or a
number it asserts.

| id | status | contradiction | the two citations | what the right status is |
|---|---|---|---|---|
| `rve-ledger-reopen-e16-2026-11-15` | done | **no** | queue 3-11 (reopen E16 and record the literature closures) vs ledger 91 (E16 row now "**closed (negative, measured)**") | done. The action was right and E26 then closed the lever — but the *title* ("E16 Winograd is the largest untested lever") is stale: it is closed. |
| `rve-winograd-tactics-2026-11-15` | done | **no** | queue 37-40 (cuDNN autotune 3.67 vs engine 3.05, negative) vs ledger 741-752 (same numbers) | done. Its "nsys 2025.1.3 refuses CUDA 13" is true of the *archive* only, not of the timeline (see stale note 6). |
| `rve-channel-width-time-2026-11-15` | done | **YES** | queue 266-270 — "C=32 gives the same 1.23x ... because myelin stops fusing the PReLU at that width and 16 separate passes (1.4-1.5 ms) put the gain back" vs ledger 951-957 — C=32 row: "24.88, 24.06, 22.41 \| 16 fused, 0 separate \| **2.07-2.36x**" — and ledger 1769-1774 ("C=32 fuses and runs at 22.4-24.9 ms: the 'fusion break' was the builder"). The ledger even retires the queue's exact numbers at 961-963: "The earlier readings for 48 and 32 ... (46.02 and 46.10 ms with the PReLU split out) were one build each of a choice that mostly does fuse". The same file still prints the retired 46.10 sentence at 974-977, so E30a contradicts itself. | done, resolution text wrong. Correct text: C=48 41.8-44.3 ms (1.16-1.27x); C=32 22.4-24.9 ms (**2.07-2.36x**, 16 fused, 0 separate); C=40 48.5 ms is the non-monotonic point. The 1.23x "C=32 buys nothing" reading is a torn build (workspace pool), not a width bound. |
| `rve-pruning-bounds-2026-11-15` | dropped | **no** | queue 302-309 (14-37% fit residual, 1.58% output, 0.060%/26.1%, 1.573→1.553) vs ledger 1679-1716 (same numbers) | dropped. It also correctly says C=32 "buys 2.07-2.36x of time and cannot pay for it in features" (1714), i.e. this row already knew the E30a resolution was wrong. |
| `rve-spatial-routing-2026-11-15` | dropped | **no** | queue 329-335 (fresh 0.026-0.034% at 1.10-1.26x engine / 1.06-1.15x wall; cached 0.42-0.65%) vs ledger 1623-1637 (same table, plus "the only passing configuration ... is slightly slower than the tile path at 1.17x") | dropped. |
| `rve-film-gate-codec-2026-11-15` | dropped | **no** | queue 359-364 (sigma=16, 72.4% of pixels differ by >8) vs ledger 880-881 (same number) | dropped (owner decision). |
| `rve-residual-scale-2026-11-15` | done | **yes — the values, not the formula** | queue 387-390 — "on a real 1080p frame: median 0.9 ... max 44.5; 98.43% of pixels under the 8-level bar" vs `docs/wsp/verified/residual-scale.jsonl` R001-R006 (2026-11-16, named frame `frame1080p.raw`, sha256 in the record): 1.16 / 4.83 / 14.35 / 44.67 / **109.03**, **96.52%** under 8. `docs/wsp/README.md:63` states the relation: "a DIFFERENT, named frame: 1.3-2.3x away, which is what an unnamed population costs". | done. The ledger row (894-901) should carry the population caveat and the pinned record; the queue's absolute values are uncited and the only replayable frame is 2.4x higher at the max. This changes no decision (the licence it grants is unchanged, 922-928) but it is exactly the "unnamed population" defect the ledger's own rule names. |
| `rve-base-grain-split-2026-11-15` | dropped | **no** | queue 619-624 (film 1.85x vs anime 4.3x; noise floor 2.27) vs MODEL §7 420-421 (noise floor 2.27 > 1.5 limit, gate refuses) | dropped (owner decision). |
| `rve-ncu-access-2026-11-15` | done | **no — the resolution is right; MODEL.md is the stale one** | queue 734-743 (params read 0, ledger has the 2026-11-15 counters) vs `docs/MODEL.md:133`, `:319-321`, `:403-404` ("ncu is blocked on this host", "RmProfilingAdminOnly: 1", "ncu needs RmProfilingAdminOnly=0") — while ledger 1818 reads "`RmProfilingAdminOnly` is 0 after the owner's modprobe line and reboot, so `ncu` runs" | done. The queue was fixed on 2026-11-17; **MODEL.md was not**, and it is the file a reader checks first. |
| `rve-doc-literature-closures-2026-11-15` | done | **no** | queue 750 ("folded into rve-ledger-reopen-e16") vs ledger 2274+ (the three closures, written down) | done. |
| `rve-bars-populations-2026-11-16` | done | **no** | queue 894-900 ("all 8 bars", "29 @below assertions ... 24 under, 2 OVER") vs the reproduced run (29 @below; 24 under / 2 OVER / 2 against a literal / 1 unknown) and `claims-bars.json` at `07d63c36` = 8 bars | done. Verified, not merely plausible. Its "16 legacy bars" in the refutes line means pre-dedup bars; the file has 8 at close, 10 today. |
| `rve-slope-epilogue-refit-2026-11-16` | dropped | **YES — two numbers** | (a) queue 937 and 941-943 — "every other measured lever tops out at 1.23x" / "the one item whose prize was larger than every measured lever" vs ledger 951-957 and 1714 — C=32 is **2.07-2.36x** of measured time. (b) queue 935-936 — "the lossless int8 build already spent peak 4 of the 8-level budget" vs `docs/wsp/verified/engine-int8.txt` (2026-11-17, same TRT 10.12.0.36): the picture does **not** reproduce — 0.010% / 0.291% / **4.389%** above 8 levels and 38-48 dB across three calibrations of the same recipe against the recorded 0.000% / peak 4 / 54.1 dB. | dropped (the owner's "weights cannot change" stands, and neither number changes it) — but both sentences are wrong, and (b) means the ledger's E9' accuracy row (73) and MODEL §6 item 5 (336-337) are now unsupported as stated. |
| `rve-functional-headroom-map-2026-11-16` | dropped | **no** | queue 975-978 (consumer is an edit; E30b's bound answers it) vs ledger 1679-1716 | dropped (owner decision). |
| `rve-lean-statistic-spike-2026-11-16` | done | **no** | queue 1014-1025 ("compiles with no `sorry`", "~110 lines") vs `docs/wsp/lean/Statistic.lean` (87 lines, 0 `sorry`) | done. The line count is the only drift ("~110" vs 87) and it changes nothing. |
| `rve-agent-crew-adoption-2026-11-16` | done | **unverified** | queue 1069-1075 (`pushDefault=fork`, `pre-push`, `post-checkout` verified) vs the cell, where `origin` is the local source checkout by construction | done, pending one check outside a cell (`git -C <real checkout> remote -v` + `git config remote.pushDefault`). `scripts/autopush.sh` and `packaging/systemd/rve-autopush.{service,timer}` exist as claimed. |
| `rve-harness-redteam-residual-2026-11-16` | done | **no** | queue 1123-1124 ("62 tests in the pair", "all 23 repros now raise") vs 48 + 14 = **62** test functions at `1f908c45` (63 today — one test was added afterwards) | done. Checked because it is the kind of number that drifts by one; it did not, at the commit that claimed it. |

**Counts:** 16 closed rows reviewed, **2 contradictions with the ledger** (`rve-channel-width-time`,
`rve-slope-epilogue-refit`), **1 number-level contradiction from a verified record**
(`rve-residual-scale`), **1 doc contradiction of a correct resolution** (`rve-ncu-access`), **1
unverified** (`rve-agent-crew-adoption`), **11 clean**. Four apparent contradictions were disproved
by checking the closing commit and are listed at the top.

The pattern the owner asked for — "a note that says blocked/untested while the record says it ran" —
appears once more, and it is the same shape as the ncu case: **the TensorRT runtime**. `MODEL.md`
and two queue notes say the host has no TRT; `docs/wsp/verified/engine-fp16.txt` and
`engine-int8.txt` record both engines rebuilt on 2026-11-17 with **TensorRT 10.12.0.36**.

---

## Stale notes in `docs/backlog/queue.yaml` (and in the two docs), by id

1. **`rve-channel-width-time-2026-11-15`** — "C=32 gives the same 1.23x ... because myelin stops
   fusing the PReLU at that width and 16 separate passes (1.4-1.5 ms) put the gain back" (queue
   266-270). Wrong on both halves: C=32 fuses 16/16 and runs 22.4-24.9 ms (ledger 951-957,
   1769-1774). This is the single most decision-bearing stale note in the file: it says the best
   measured lever on the board is worth 1.23x when the record says 2.07-2.36x.
2. **`rve-fusion-break-map-2026-11-15` (B1)** — "at C=32 it runs `sm80_xmma_..._indexed_wo_smem_`
   ... and the PReLU becomes 16 separate passes worth 23 ms" (queue 499-505). Same error, and it is
   the *opposite* of what B1 exists to map: ledger 1769-1774 says the split was the workspace pool
   ("build three times with an explicit 8 GB workspace pool: 24.88, 24.06, 22.41 ms, sixteen fused
   myelin layers and zero separate PReLU passes"), and `RawTRTEngine.build_plan` (80-105) never sets
   one — which, if the pool is the cause, is a speedup available to the *shipped* engine with no
   width cut at all.
3. **`rve-slope-epilogue-refit-2026-11-16`** — "every other measured lever tops out at 1.23x"
   (937) and "the lossless int8 build already spent peak 4 of the 8-level budget" (935-936). Both
   false (Table B). The dropped status is still right; the reasoning is not.
4. **`rve-residual-scale-2026-11-15`** — "it also reorders the plan: a channel cut at fp16 is now the
   shortest route to 2x" (404-407). E30b closed that route on a representational bound (ledger
   1679-1716; queue 302-309).
5. **`rve-strip-concurrency-2026-11-15`** — "ncu counters would say in advance whether anything is
   there to hide, and they are still blocked" (699-702). ncu has been running since 2026-11-15
   (ledger 1818-1827). The item is still open, for a different reason (no code path).
6. **`rve-winograd-tactics-2026-11-15`** — "nsys 2025.1.3 refuses CUDA 13" (queue 39-40, echoed at
   ledger 767). True of the **archive** subcommand; the **timeline** runs and produced the kernel
   table in MODEL §6 293-309 (48.72 ms `sm75_xmma_..._stage1_...`). As written it reads as "nsys is
   unusable", which the record does not support.
7. **`rve-harness-adoption-2026-11-16`** — "engine probes still need python-tensorrt" (808-809) — and
   **`rve-findings-rederivation-2026-11-16`** — "the host has no TensorRT runtime at all (only
   tensorrt-debug symbols; no libnvinfer, no trtexec) ... A PyPI wheel installs 11.4.0.106" (857-863).
   Contradicted by four other queue rows in the same file ("RUNNABLE NOW 2026-11-17: TensorRT
   10.12.0.36 and torch are installed", 150-153, 183-186, 548-551, 577-580) and by
   `docs/wsp/verified/engine-fp16.txt` ("Rebuilt 2026-11-17 here ... TensorRT 10.12.0.36", 25 layers,
   four sessions) and `engine-int8.txt` ("the same TensorRT **10.12.0.36** the ledger used", 64.3-70.4
   ms). Both notes describe a host state that no longer exists.
8. **`docs/wsp/README.md:70-73`** — "**The engine lane cannot follow.** This host has no TensorRT
   runtime ... nothing in this repository can substitute for it." Three lines above it, the same
   table lists `engine-fp16.txt` and `engine-int8.txt` as PASSing re-derivations of exactly that
   lane. The paragraph was not updated when the table row was added.
9. **`docs/MODEL.md:133, 319-321, 403-404`** — ncu blocked / `RmProfilingAdminOnly: 1` / "ncu needs
   RmProfilingAdminOnly=0 ... until then the DRAM split stays a bound". Ledger 1818 and queue 734-743
   say the opposite, with a measured counter table. Note the internal tell at `MODEL.md:251`: "the
   counter is read (§6, item 2)" — the file contradicts itself in the same section.
10. **`docs/MODEL.md:206`** — "Explicit Q/DQ, fully-int8 NHWC, per-channel: **not tried**". It is
    tried, in the same file (§6 item 5, 335-368: 68.11 vs 55.54 ms) and in ledger E9' (73). And the
    verified record now also says the *accuracy* half of that row does not reproduce
    (`engine-int8.txt`).
11. **`docs/MODEL.md:211`** — "Winograd | **no** | - |". E26 measured it (ledger 733-752) and the
    C=32 probe found TRT *selecting* a Winograd tactic (1793-1800). The row also still carries the
    intensity objection the ledger withdrew on 2026-11-15 (ledger 91).
12. **`rve-hr-reference-cache-2026-11-15`** — acceptance baseline "146/220 at 6.61 s and 142/219 at
    6.90 s" (665-668) vs ledger 371-374 "146/220, 6.56 s" and "142/219, 6.92 s". MODEL §7 (418-419)
    carries the 6.61/6.90 pair. Small, but an acceptance number that two docs disagree on is the
    kind of thing that later reads as drift.
13. **`rve-ledger-reopen-e16-2026-11-15`** — title: "E16 Winograd is the largest untested lever".
    E26 closed E16 negative (ledger 91, 733-752). The row is `done` for a reason that no longer
    reads: the lever was tested and lost.
14. **`rve-agent-crew-adoption-2026-11-16`** — "DECISION needed before (3): this repo's push remote
    is named `fork` ... not `origin`" (1061-1064). The resolution decided it (`pushDefault=fork`);
    the note still asks the question.
15. **`rve-ncu-access-2026-11-15`** — the note still opens "RmProfilingAdminOnly=1 blocks the
    counters ... BLOCKER 2026-11-17: owner: one line ... plus a reboot" (729-733), directly above a
    resolution saying the opposite. Kept as history by design, but it is the exact note a future
    reader will quote; consider prefixing it `SUPERSEDED:`.

**The three most stale, if only three get fixed:** the `rve-channel-width-time` resolution (1.23x
where the record says 2.07-2.36x — it is cited as the ceiling of every other lever), the TensorRT
blocker trio (queue 857-863, queue 808-809, `docs/wsp/README.md:70-73` — the lane has already
run), and `docs/MODEL.md`'s three ncu-blocked passages (the counters have been read since
2026-11-15, and the file is the front door).

---

## Method limits

- Every citation above is a line I opened; no number here is quoted from a summary.
- `docs/wsp/verified/engine-int8.txt` is a *record*, not a ledger row: the ledger and MODEL §6 still
  carry the superseded 0.000% / peak 4 picture. That gap is itself a finding — the ledger is the
  stated source of truth and it has not absorbed the 2026-11-17 re-derivation.
- I did not run a render, a build or `ncu`; the cell has no GPU and no `tensorrt`/`torch` on the
  system interpreter (`.venv` is gitignored and absent here). The two claims I could not verify —
  the `fork` remote and the Lean toolchain — are marked as such rather than assumed.
