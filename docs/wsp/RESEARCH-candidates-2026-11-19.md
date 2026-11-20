# Optimisation candidates the int8 lane has not tried

Written 2026-11-19 by a scout. **This is a reading list with falsifiers, not evidence.** It carries no
`@claim`, it lives outside `docs/wsp/verified/`, and every number in it is either taken from a named record,
marked `derived` (arithmetic on those records), or marked `ESTIMATE` (a ceiling no one has measured).
Nothing here was measured: no GPU job was run.

What it is for: the lane's own portfolio (`docs/PLAN-int8-envelope-2026-11-19.md` §3) is mostly spent - two
of its seven rows were closed by measurement the same day, two more are filed on the queue, and the three
that remain have no owner - and its reserve (§3b) is a list of unknowns. This file adds the literature
angle: what other fields do with a memory-bound narrow-channel convolution, a 12-channel output, a
3-channel input, a synchronous 4K readback. For each candidate it states the part of the frame attacked,
what would kill it, and whether the corpus already knows the answer.

## 0. The state I read, and one correction to the plan

| what | value | record |
|---|---|---|
| frame, int8 (NHWC tail) | 47.96 (Violet) / 45.70 ms (Chainsaw) | `int8-tail-layout.txt` W025, W009 |
| frame, fp16 same session | 58.54 / 56.22 ms | same file |
| ratio | **1.221x / 1.230x**, median of rounds 2-3 | same file |
| the bar | 1.2x, `bar_int8_wall` | `docs/claims-bars.json` |
| hops | 26.95 ms, 152.9 GFLOP per hop, 91 TOPS | `int8-wall-ab.txt` W046 |
| head + quantise | 4.65 + 1.60 ms | W044, W045 |
| tail (NHWC) | dequant 1.58, conv **2.33-2.40**, shuffle 0.78-1.05, residual 0.89-1.55 | `int8-tail-layout.txt` |
| pack + queue | 3.87 ms | `int8-frame-accounting.txt` A018 |
| wrapper + read wait | ~2.4 ms | same file |

**Correction.** The plan's §2 table (1.141x, "gap 2.50 ms") predates the tail-layout adoption the same day.
With the adopted layout the median clears the bar: **margin 0.82 ms on Violet and 1.15 ms on Chainsaw**
(derived: 58.54/1.2 - 47.96 = 0.82; 56.22/1.2 - 45.70 = 1.15), and one round of Violet is 0.6 ms short
(1.178x). So the question is no longer "can it reach 1.2x" but "what buys margin, and what is left that is
worth more than its probe". Everything below is ranked on that.

**The one arithmetic that frames the whole list.** A hop takes 1.684 ms. Its arithmetic floor is 0.87 ms
(152.9 GFLOP at the card's ~176 dense int8 TOPS) and its traffic floor is 0.92 ms (265 MB at ~288 GB/s),
both derived from records above. It runs at **91 TOPS and 157 GB/s - about 52-55% of both roofs at once**.
A kernel at 52% of two roofs is not bound by either; it is bound by its schedule. That is a different
diagnosis from "the hops are at the practical int8 rate", and candidate C3 is what follows from it.

## 1. How the ranking was made, and what it refuses

Expected value per falsifier hour, where "expected value" is the *measured* part the candidate attacks, and
"falsifier" is the cheapest experiment that would kill it. No probability column: a number I cannot measure
is an opinion in a number's clothes (the plan says the same in its §3 preamble, and it is right).

Every ceiling below is an **estimate derived from a record**, never a measurement, and is marked so. Where
the corpus has already measured the mechanism negative, that is stated in the evidence line and the
candidate is kept only because the *variant* differs (C3 is the clear case).

Ordering: the two boundary convolutions first (largest measured parts with a certain landing, if the
picture holds), then the hop's schedule (largest part, cheapest falsifier, prior negatives attached), then
the non-body stages.

**Tally, so the next reader can check it.** Thirteen candidates: **five genuinely new** (C1 int8 tail,
C5 epilogue divide, C9 input clone/sync, C12 packed codes, and C3 as a new variant of a partly-tried
mechanism), **five reserve rows that no one has measured** (C2, C4, C8, C10, C11), and **three that extend
work already filed or done** (C6 graph, C7 tail transpose, C13 equalisation). Fifteen rows are in the
do-not-propose table, each with the file that closed it.

---

## 2. Candidates

### C1. The tail as a 17th int8 hop, with the dequant scale folded into its weights

**What.** The last hop writes int8 codes on a known per-channel grid; those codes are dequantised to fp16
(1.58 ms) and fed to a cuDNN fp16 conv (2.33-2.40 ms). Fold the dequant scale into the tail's weights
offline (`w'[k,c] = w[k,c] * s_k`) and quantise `w'` per output channel, and the tail is an ordinary int8
convolution whose input is already codes - no dequant pass, one kernel instead of two. Padding 12 outputs
to 16 with zeros contributes nothing and therefore keeps the arithmetic, which is the property the lane
requires of an adoptable variant (`rve-tail-conv-utilization-2026-11-19` says the same in its note). Source:
the standard per-output-channel-scale / per-input-channel-activation fold of int8 convolution; the repo's
own kernel already performs exactly this fold between hops (`backend/src/pytorch/Int8Kernels.py:53-63`).

**Where it lands.** Dequant 1.58 + conv 2.33-2.40 = **3.9 ms**, 8% of the frame. Once the tail is our
kernel, `pixel_shuffle` (0.78-1.05) and the residual (0.89-1.55) can be written in its epilogue - the
shuffle is a write-side index map, the residual is a nearest upsample read of the 6 MB LR input - for
another 1.7-2.6 ms of headroom.

**Falsifier.** Staged, cheapest first: (a) emulate the folded-scale int8 tail against today's fp16 tail on
the four held-out BD frames with `harness/scripts/int8_kernel_emulation.py` (the bit-exact emulation the corpus
already trusts) - **1-2 h, no GPU**; if the share or the peak moves past its margin, stop. (b) one Triton
conv 576->16 at 1080p with zero-padded outputs, timed against 2.35 ms - **1 h**.

**Evidence check.** `grep -rn "int8 tail" docs/` returns the diagnosis (`int8-tail-split.txt`) and nothing
else: the tail has only ever been attacked as a *layout* problem, never as a *quantisation* problem. It is
not refuted: `PLAN-int8-graph-2026-11-15.md:1257` names "replacing the head/tail convolutions with int8
kernels too" as the one lever left "with a measured size", and no queue row or verdict has taken it.
**NEW.**

**Ceiling.** ESTIMATE 3.9 ms of the frame from the conv and dequant alone; 5.5-6.0 with the shuffle and
residual in the epilogue.

---

### C2. The head as an int8 hop with the quantise fused into its epilogue

**What.** The head is 3->64, 3x3, fp16, 4.65 ms at **1.5 TFLOPS** - 4.8x its own 0.96 ms traffic floor.
Its input is 8-bit-derived RGB, so an asymmetric int8 input grid (zero point 128, scale 1/255) is
essentially lossless for 8-bit material; gate the feature on `not high_bitdepth` exactly as `yuv8_output`
already is (`backend/src/pytorch/UpscaleTorch.py:167`). K = 27 padded to 32, per-output-channel weights,
PReLU and requant in the epilogue as the body already does - and because the head then *writes int8 codes*,
the separate 1.60 ms quantise pass disappears. Sources: the int8 tensor-core alignment rule (multiples of 16,
NVIDIA's matrix-multiplication guide) and the repo's own reserve row.

**Where it lands.** Head + quantise = **6.25 ms**, the second largest line in the frame after the hops.
It also removes the 265 MB fp16 write that the quantise pass exists to consume.

**Falsifier.** The reserve row's own first step, arithmetic first, no GPU: what grid do three input channels
need, and what does quantising the head's 27 weights do to the four held-out frames
(`harness/harness/scripts/int8_accuracy_sim.py` / the emulation) - **1-2 h**. Only then one kernel, **1 h**.

**Evidence check.** `grep -rn "int8 head" docs/PLAN-int8-envelope-2026-11-19.md` - it is a reserve row whose
open question is "whether an int8 head is even expressible here". `rve-head-conv-2026-11-19` is filed and
covers isolating the fp16 head and fusing it with the quantise; it does **not** cover quantising it. The
fp16 path already pads 3->8 input channels for myelin fusion (`RVE_NO_HEAD_PAD`, -3.5% engine time) - that
is the zero-padding route, measured, and it pads for *fusion*, not for an int8 K. No verdict covers this.
**RESERVE, unknown.**

**Ceiling.** ESTIMATE 0.7-1.1 ms (traffic floor ~0.5 ms for the 132 MB int8 write plus the 6 MB read, at
20-90 TOPS on 7.17 GFLOP); save 5.1-5.5 ms.

---

### C3. The hop's schedule: it is at 52% of both roofs, and the load path is the suspect

**What.** Nine taps per block, each a `tl.load` on a computed gather pointer with a mask and per-element
index arithmetic (`Int8Kernels.py:43-54`), no vectorised 128-bit access, and the weights re-read per CTA -
36 KB per CTA x 16.2k CTAs is ~0.58 GB of L2 traffic per hop (derived). The literature's levers for this
shape: block-pointer / 128-bit vectorised loads and shared-memory staging of the three-row window instead of
nine gathers (CUTLASS's implicit-GEMM conv recommends 128-bit aligned NHWC and channel counts that are
multiples of 32); a **weight-stationary CTA** that loops over several M tiles so the weights load once,
which is the same mechanism the BLOCK_M curve already shows (1.82 ms at 256 against 2.38 at 128, 512
spilling to 79 ms - `int8-body-hop`); and, for reduced precision specifically, packing and layout overhead
rather than arithmetic (arXiv:2202.06819).

**Where it lands.** 26.95 ms, the largest line in the frame by 4x, at 91 of ~176 TOPS.

**Falsifier.** The plan's own 1-hour probe, with the mechanism named: one hop, three arms - today, a
block-pointer/vectorised load, and weights hoisted out of a four-tile M loop - compare TOPS. If none of the
three moves 91, the hop is at its floor on this shape and the whole row closes.

**Evidence check.** **PARTLY TRIED, and the negatives are specific.** `Int8Kernels.py:40-42` records that
unrolling all nine taps costs registers (2.919 vs 2.198 ms) and that Triton pipelining at 2-3 stages made it
*worse* (2.497 / 2.635), "because the gathers are already L2-resident" - so staged pipelining as tried is
measured-negative, and only the vectorised-load and weight-hoisting variants are open. `int8-body-hop`
(current verdict) swept BLOCK_M and warps only: 1.82 ms at 256/4. The plan's "hops, MMA utilisation" row is
the parent of this one. Not refuted, not tried.

**Ceiling.** ESTIMATE 26.95 -> 20-22 ms, i.e. 111-122 TOPS on the hop - which is the plan's "~20 ms"
figure, and also about 85% of the traffic roof (at 150 TOPS the hop would need 260 GB/s of the card's 288).
**The traffic roof is what stops this lever**, not the arithmetic: save 5-7 ms, the largest single ceiling
in this file.

---

### C4. The 3.87 ms pack is synchronous, pageable, and already 12.4 MB

**What.** `get_frame_bytes()` builds the deliverable payload on the GPU and then copies it into a host
`bytes` object *inside the frame window* (`RenderVideo.py:836-844`); `writeQueue.put` follows. A ring of
pinned slots filled on a copy stream, with release guarded by a CUDA event and the wait deferred into the
writer, takes the copy off the critical path. Source: ordinary CUDA async-copy practice (event-guarded
pinned ring, `cudaMemcpyAsync` on a non-default stream); the repo's own pinned table
(`docs/PERFORMANCE.md`, the PCIe 3.0 x8 section) already measures the link at 5.1-6.0 GB/s in both
directions.

**Where it lands.** 3.87 ms of the frame. **And there is arithmetic in the target itself:** at the measured
1.5 bytes/px payload a 4K frame is 12.44 MB, which the measured link moves in **2.44 ms** (derived) - the
stage is already ~1.4 ms above its own floor before any overlap is added.

**Falsifier.** **1 h**, null sink: fill the payload into a pinned buffer on a second stream with
`non_blocking=True`, record an event, and watch the loop line. If the loop does not move, the 3.87 is the
writer's back-pressure and the row closes with that.

**Evidence check.** RESERVE (plan §3 "pack + queue"). Two things the next reader needs, both from reading
the tree rather than the docs: (a) `RVE_YUV8` defaults to **on** (`UpscaleTorch.py:167`), so the plan's
"output colour path" reserve row - whose premise is that "the writer's rgb24 payload is twice yuv420p's
bytes" - is already satisfied in the shipped path; the payload is 1.5 B/px, and the 3.87 is therefore not a
payload problem. (b) `PERFORMANCE.md` records a pooled-pinned attempt that came out at 12.6 dB and was
"disabled" - that was a **bug** (a slot released before its last byte had been copied) and the same file
says so; it is not a refutation of the idea, and presenting it as one would be wrong. The preview stage is
already a sentinel in headless renders (`RenderVideo.py:519-540`), so it is not part of the 3.87.

**Ceiling.** ESTIMATE 2.4-3.9 ms off the loop; 0 if the queue is the cost.

---

### C5. Fold the output scale into the incoming scale and bias: the epilogue has a per-element divide

**What.** The hop's epilogue computes `v = acc*w_scale + bias`, applies PReLU, then `q = floor(v/out_scale
+ 0.5)` - a **per-element fp32 divide** over 132 M elements per hop (`Int8Kernels.py:55-60`). PReLU is
homogeneous of degree one, so a positive rescale commutes with it: precompute `a = w_scale/out_scale` and
`b = bias/out_scale + 0.5` per channel and the epilogue becomes one FMA and a floor. Source: the fold-into-
weights identity that SmoothQuant is built on (arXiv:2211.10438 §4) applied at the epilogue instead of
between layers; standard int8 epilogue practice.

**Where it lands.** **All sixteen** hops in the shipped body run this one kernel (`Int8Body.py:385-395`
loops `body_prologue_kernel` over the sixteen weight sets) - the "6 of 16 on the three-accumulator kernel"
split belongs to `harness/harness/scripts/int8_pipeline.py`, a different object (`int8-window-measured`), and must not be
quoted here. If the divide is 5-10% of a hop it is 1.3-2.7 ms of the frame.

**Falsifier.** **30-60 min**, one hop, two arms, timed and code-diffed on a real frame. The rewrite is exact
in real arithmetic but not bit-identical in fp32 (a 1-ulp difference can flip a floor at an exact .5), so it
must be recorded as a **different-arithmetic** change and judged with `gate_sweep --reference-kind config`,
not waved through as a layout change.

**Evidence check.** The `w_scale`/`bias` fold is already there (`:55`); the divide at `:59` is not folded,
and no file in the corpus mentions it. `grep -rn "out_scale" docs/` finds only the kernel description.
**NEW.**

**Ceiling.** ESTIMATE 1.3-2.7 ms (5-10% of 26.95) if the divide is what those hops spend it on; 0 if the
kernel is not ALU-limited, which the same one-hop timing settles.

---

### C6. Capture the whole body, not just the 16 hops

**What.** `int8-graph-launcher` (current verdict) captures the sixteen int8 launches "with head and tail
leading out of the graph". So per frame the head, quantise, dequant, shuffle and residual still cross the
host path, and the wrapper measures ~2.3-2.4 ms that belongs to no named part. One capture over
head->quantise->hops->tail->shuffle->residual, with the buffers preallocated per resolution, is the same
technique extended to the whole call. Sources: Triton's own launch-overhead RFC (triton-lang/triton#10140:
11-27 us per call on the full Python path, 3-6 us with the fast cache); arXiv:2609.37241 on residual host
orchestration being invisible to a graph.

**Where it lands.** The wrapper (~2.3-2.4 ms) and ~16-20 launches at 5-27 us, which is 0.1-0.5 ms of it.

**Falsifier.** The plan's own cheapest probe first - **1 h**: time the wrapper with the body replaced by a
memcpy. If that reads ~0, the wrapper is launch cost and the graph is the fix; if it reads 2 ms of copies,
the graph cannot help and the row closes.

**Evidence check.** **EXTENSION of a current verdict** (`int8-graph-launcher`, 16 launches, head and tail
outside). `PERFORMANCE.md`'s "CUDA graphs ~0.3%" is about the TensorRT engine, a different object, and must
not be quoted here. The wrapper itself is a reserve row (plan §3b, row 1).

**Ceiling.** ESTIMATE 0.3-0.6 ms of host launch cost; up to 2.3 only if the probe shows the wrapper is host
time rather than copies.

---

### C7. The tail as a transposed tall-skinny GEMM, in fp16, with no arithmetic change

**What.** Today the tail is solved as M = pixels, N = 12. The transpose puts C_out in M (12 -> 16 padded)
and pixels in N, which is the shape the tall-and-skinny GEMM literature optimises: an outer-product
formulation, shared-memory staging of the small operand, and a model that switches between compute-bound,
memory-bound and latency-bound regimes. Source: TSM2X, arXiv:2002.03258 (GPU tall-and-skinny GEMM).

**Where it lands.** 2.33-2.40 ms for 28.7 GFLOP = **12.2 TFLOPS** (derived), against ~44 fp16 CUDA-core and
~176 tensor TFLOPS. It is the fallback if C1's picture fails, because it changes no quantisation.

**Falsifier.** **2 h**: extend the existing `harness/scripts/tail_conv_probe.py` (which already sweeps C_out
12/16/32/64 x layouts) with the transposed arm - a conv on a permuted input, or an unfolded matmul - and
time against 2.35. A different algorithm means the picture test, not byte-identity.

**Evidence check.** **FILED, partly**: `rve-tail-conv-utilization-2026-11-19` includes the 12->16 padding
arm but not the transpose, and `harness/scripts/tail_conv_probe.py` confirms that in its own docstring. Not refuted.

**Ceiling.** ESTIMATE 2.35 -> 1.0-1.5 ms; save 0.9-1.4.

---

### C8. Two frames through the hops (M x 2)

**What.** The kernel already carries a batch dimension and the app feeds one (`x.shape[0]`). Doubling M
doubles the MMA work behind each weight load and each launch; the product pays one frame of added latency.

**Where it lands.** The 26.95 ms of hops, if the loss is occupancy rather than traffic.

**Falsifier.** The plan's own **1 h**: the same hop at M x 2, compare TOPS and GB/s. Two numbers decide it -
if the achieved TOPS and bandwidth both stay at 91 / 157, batching buys nothing.

**Evidence check.** RESERVE (plan §3b), explicitly "unmeasured either way". **NEW as a measurement.**

**Ceiling.** ESTIMATE 1-3 ms of 26.95; 0 if the hop is not occupancy-limited - which is the point of the
probe, not a detail.

---

### C9. The input path synchronises and clones per frame

**What.** `Frame.get_frame_tensor()` returns `self._tensor.clone()` on every call, with a comment that says
it is there "so the frame won't be overwritten? have to test later" (`backend/src/utils/Frame.py:322`), and
`yuv420p10_to_tensor` ends with an explicit `sync_stream` (`backend/src/pytorch/TorchUtils.py:243-244`).
Both are inside the "wrapper" the frame accounting could not attribute (2.3-2.4 ms with the body's phases
named). Source: CUDA/PyTorch asynchronous-copy practice - a per-frame stream sync forfeits the overlap the
rest of the pipeline is built for.

**Where it lands.** The unattributed wrapper, ~2.3-2.4 ms, of which this is a part that has never been
itemised. It also shrinks if C2 lands: an int8 head consumes codes, so the float RGB conversion and the
clone can be replaced by one quantise of the yuv planes.

**Falsifier.** **30 min**, no clip: time `get_frame_tensor` alone on a decoded frame, with and without the
clone and the sync, by CUDA event. If it is under 0.1 ms, close it.

**Evidence check.** **NEW** (code reading, not a document). Nothing in the corpus names the clone; the plan's
reserve row for the wrapper says "never isolated", which this is the first step of.

**Ceiling.** ESTIMATE 0.2-0.5 ms; likely small, and cheap enough to settle before anything else on this list
is funded.

---

### C10. A swizzled / channel-blocked int8 buffer layout

**What.** The literature's layout warning for reduced-precision MMA is that an NHWC tensor whose channel
group is only 16 bytes wide defeats 32-byte transactions, and it recommends an `NHWCnc` blocked variant
(arXiv:2202.06819). Our intermediate already has 64 contiguous channels - two 128-bit words per pixel - so
the *alignment* half of that argument does not apply here. What remains is pixel-order swizzling (Morton or
tile-major) so a CTA's three-row window overlaps its neighbour's in L2.

**Where it lands.** The hops' 157 GB/s of read+write, if the layout is what costs it.

**Falsifier.** **1 h**, one hop with a swizzled index, compare TOPS and GB/s.

**Evidence check.** RESERVE (plan §3b, "int8 buffer layout"). The honest prior is that the strongest part of
the literature's argument (16-byte channel groups) is already satisfied at C=64, so this is the weakest of
the hop levers. Listed so nobody re-derives it from the paper.

**Ceiling.** ESTIMATE 0-2 ms; most likely 0.

---

### C11. Pipeline depth / a second frame in flight

**What.** One frame is in flight. The app's profile total already exceeds the render loop's wall clock by
27-30% on the int8 arms, i.e. inference already overlaps the loop and the instrument counts it twice
(`int8-frame-accounting.txt`).

**Falsifier.** Plan's **1 h**: vary the queue depth and watch the loop line.

**Evidence check.** RESERVE (plan §3b). The prior is small room *and* an entangled instrument.

**Ceiling.** ESTIMATE 0-1.5 ms.

---

### C12. Packed sub-8-bit codes between hops

**What.** The hop moves 265 MB and reads+writes at 157 GB/s. Packing the codes to 6 bits would cut that
traffic by 25% at the cost of unpack ALU and a non-tensor-core operand layout. The idea is not from a paper
that shows a GPU win for it; it is arithmetic on our own traffic - the two compression lines I found
(arXiv:2201.08830 APack, arXiv:1905.10830 feature-map transform coding) are accelerator/FPGA designs and
report energy and bandwidth, not a tensor-core speedup.

**Where it lands.** The hops' 157 GB/s, if traffic is the binding half of the 52%.

**Falsifier.** **30 min, no GPU**: quantise the body's activations to 6 bits with per-channel scales in the
bit-exact emulation and read the share and peak on the same four held-out frames. `bar_share` is 0.1% and
today's body sits at 0.003% (post-fix pair), so there is room - but 2 bits is a lot.

**Evidence check.** **NEW.** Listed low because the picture risk is the whole question and the win is
conditional on a diagnosis (52% of both roofs) that has two other candidates ahead of it (C3, C5).

**Ceiling.** ESTIMATE 2-4 ms if the hop is traffic-bound and the picture holds; I would not spend a GPU hour
before the emulation says it does.

---

### C13. DFQ cross-layer equalisation - a picture lever, not a speed lever

**What.** Rescale each layer's input channels against the next layer's weight rows to equalise ranges
(Nagel et al., "Data-Free Quantization Through Weight Equalization and Bias Correction", ICCV 2019). Here it
would *not* buy frame time: the epilogue is already fused and the scales are already per-channel. Its only
route to speed is to make a per-tensor or scalar grid survive, and that is the route the closed list already
spent (scalar-slope int8: 1.40x ceiling, picture destroyed).

**Falsifier.** **1 h** in the emulation, if the goal is a picture question. Not a speed probe.

**Evidence check.** **HALF DONE.** The bias-correction half is implemented behind `RVE_INT8_BIAS_CORR=1` and
measured (+9% of an already-tiny share) in `int8-scales-bd.txt`, which also names equalisation as "worth
reading" and notes that SmoothQuant-style migration "is what folding s_in into the weights already does
here". So this is a picture lever already at its useful end.

**Ceiling.** ESTIMATE 0 ms of frame time.

---

## 3. Refuted or closed - do not propose

Listed so the next scout does not spend searches on them. Each has the file that closed it.

| technique | why it is closed | where |
|---|---|---|
| int8 implicit + FP8 on this graph | measured slower / no Ada speedup vs fp16 | `PERFORMANCE.md`; `summary-lower-precision-closed` |
| explicit Q/DQ per-channel | lossless and **slower** (68.11 vs 55.54) | `engine-int8.txt`; `trt-int8-two-pass` superseded |
| scalar-slope int8 | 1.40x ceiling, 99.3% of pixels past the bar | `RESEARCH-LEDGER.md` E9' |
| body width C=32/48 | 14-37% residual per layer; 48 buys nothing | `rve-channel-width-time`, `rve-pruning-bounds` |
| Winograd via cuDNN (fp16) | E26 negative | `rve-winograd-tactics-2026-11-15` |
| **int8 Winograd** | not "closed by measurement" here, but closed by the literature: int8 Winograd is numerically unstable without Winograd-aware *training* or PTQ-aware transforms (WACV 2024 "Wino Vidi Vici"; NeurIPS 2023 PAW/FSQ; ACM TACO 2024). Retraining is out of scope - the model is the product | this file, §5 |
| activation sparsity | PReLU output non-zero everywhere | `PLAN-int8-envelope` §6 |
| 2:4 structured sparsity | needs pruned weights + TRT already picks sparse tactics for 22 of 53 layers | `PERFORMANCE.md` |
| merging 3x3 into 5x5 | +39% MAC | `PLAN-int8-envelope` §6 |
| Strassen | at N=64 | same |
| fusing the 16 hops (megakernel, any width) | the boundary is already overlapped: 91 TOPS in-app vs 84 isolated | `rve-megakernel-body-2026-11-19`; `PLAN-int8-envelope` §6 |
| a smaller / different network | owner decision: the model menu is the product | `rve-model-class-question-2026-11-19` |
| frame or tile reuse | shelved by the owner; further investigation forbidden | `reuse-gate-shelved` |
| zero-copy decode (PyNvVideoCodec, P010 from a CUDA tensor) | priced at ~18% of a 45 ms frame for days of work and permanent maintenance - rejected on cost, not refuted by measurement | `PERFORMANCE.md`, "In-process encoding" |
| second stream / strips | 1.001x, SM at 100% | `streams-no-overlap` |

## 4. Searches that found nothing new

Each of these was run and produced no candidate that survives its own filter; the reason is given so the
next scout can skip it.

- **Winograd in the 8-bit domain.** Every route that keeps our weights needs Winograd-aware training (WACV
  2024, NeurIPS 2023 PAW/FSQ, ACM TACO 2024 all say the default transforms are unusable in int8). Training
  is out of scope, and the fp16 route is already E26-negative. Closed.
- **"Width folding" for the 3-channel head** (arXiv:2601.11608, post-training, semantics-preserving,
  aligns small channel counts to tensor-core multiples). Does **not** apply as written: the transformation is
  exact only for a kernel that does not convolve along the folded axis, and it is stated for `Cin = 1`. Our
  head convolves along width, so folding would mix neighbours unless the 3x3 is first split into three 1x3
  convolutions and the folded kernel made banded across the fold blocks - and its own reported "3x" has no
  public measurement behind it (the paper withholds results, pointing at a patent). The *actionable* residue
  is C2's K = 27 -> 32, which needs no reformulation.
- **Tall-and-skinny / split-K / Stream-K.** The hop already has ~16k CTAs, so there is no tail effect to
  balance and split-K's extra reduction pass buys nothing; Stream-K is a work-decomposition fix for too few
  tiles. The only live part of that literature is the *transposed* tail (C7).
- **Persistent / warp-specialised kernels (TMA, WGMMA producer-consumer).** Hopper features; Ada (this card)
  has neither, and the Triton `num_stages` variant is measured negative on this kernel
  (`Int8Kernels.py:40-42`).
- **Thread-block rasterisation for L2 reuse (CUTLASS swizzles, "grouped ordering").** Worth 6.7x of DRAM
  traffic in the Triton write-up I read, because a GEMM's A and B tiles are shared by many CTAs. Our hop has
  no cross-CTA reuse beyond a one-pixel halo, so the mechanism has nothing to exploit.
- **Lossless activation compression on the copy path** (APack arXiv:2201.08830; PlaneZip arXiv:2610.25033;
  feature-map transform coding arXiv:1905.10830). All three are accelerator/PCIe-swap designs whose win is
  energy or link time at low bandwidth; the codec runs on the same SMs the hop needs, and at 288 GB/s the
  unpack cost lands before the saving does. Nothing actionable on a 4060 Ti.
- **Compressing the D2H payload.** Real-time lossless video codecs on a GPU run at 1-2 GB/s, below the
  measured 5.1 GB/s link. Nothing.
- **CPU offload of the head** (AMX/oneDNN). 7.17 GFLOP at ~50 GFLOPS of CPU is 140 ms. Nothing.
- **bf16 instead of fp16 for the tail or the hops.** Same tensor cores, same rate, worse mantissa; changes
  the picture for no time.
- **Variable-rate rendering / checkerboard / temporal upsampling** (the real-time-rendering field). These are
  the frame-reuse class the owner shelved; not re-entered.

## 5. What I could not verify

- The width-folding paper's 3x is **not** a number anyone can check: results are withheld, cited to a patent
  application. I say so rather than paraphrase it (§4).
- The int8-Winograd claims are about *accuracy*, on edge hardware (Mali, Cortex-A), not about this card's
  tensor-core rate. The conclusion I draw - "needs training, so out of scope" - is the safe direction, but I
  did not find a GPU paper that measures int8 Winograd against an int8 implicit-GEMM conv on Ada.
- No GPU job was run for this file. Every "ceiling" is arithmetic on a record and is marked `ESTIMATE`. The
  derived quantities are labelled `derived` where they appear: the 52%-of-both-roofs figure, the 12.2
  TFLOPS tail, the 2.44 ms pack floor, and the 0.82 / 1.15 ms bar margins.
- The two code-level findings (C4's `RVE_YUV8` default and C9's per-call clone + stream sync) are **reads**,
  not measurements; they are offered as the cheapest first probes, not as wins.
