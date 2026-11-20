# Draft — issue to TNTwise/REAL-Video-Enhancer (NOT SENT)

Title: **An evidence harness, a frame-reuse implementation of `skip_render_if`, and a few pipeline fixes**

---

Hi. We run a fork of RVE and spent about a week on it with a small fleet of AI agents. The part we think is worth
your time is not a feature — it is that none of our numbers are "trust me".

**The harness (this is the piece we would most like to give back).** Every number we produce is a *record*: the
value, the unit, the population it was computed on (as an item set with a hash, not a label), the instrument, the
config, the session, and since today the commit, the GPU and the library versions it was taken on. A prose claim
cites a record id and cannot restate the value; a check fails if it does not match. On top of that: bars with an
explicit statistic, A/A before A/B (a refusal is only valid against a base that proved deterministic), three-state
probes (measured / refused / broken - never "errored equals fine"), a ledger over records and verdicts, and a dated
journal.

Why it might matter to you specifically: you carry 32 open issues of the "it is slow" / "it broke on my card" kind,
and a refactor makes all of them unreproducible at once. With the harness "it got slower" becomes a record with a
commit and a card, and a fix that is not a fix fails mechanically instead of by review. It is ~3.2k lines, standard
library plus `pyyaml`, with ~6k lines of its own tests, and it touches nothing in the renderer - so it can land or be
ignored independently of everything below. Our own record corpus stays with us; the machinery is what we would send.

**Frame reuse — your `TOOD.md` sentence.** The one asking for a `Frame` attribute `skip_render_if` that filters and
interpolation can consult. We built that as a gate and measured it: if a frame barely changed, do not run the model
again, and do not emit the stale picture either - add back what changed, split by frequency (low-pass at gain 1,
high-pass at 1.35 on flat static blocks, per channel, applied recursively on the previous emitted frame). On two
Blu-ray cuts at 1080p→4K: 63% of frames reused on the calm cut with **0 of 130** frames above the 0.1%-of-pixels bar;
**1 of 124** on a held-out action cut with a cheap veto (skip the reuse when more than ~0.2% of pixels changed by
more than 8 levels); about **1.9x** on the render loop where the content allows it, 2.4 ms per corrected frame. It
refuses on its own where it cannot be trusted - dense moving line art is refused outright rather than rendered
wrong. We would rather re-express it as your attribute than drop a gate object into the render loop.

**Pipeline fixes that are not about reuse**, and that any user on the stock TensorRT engine feels: pack `yuv420p` on
the GPU instead of `rgb24` (−2.3 ms/frame, picture-neutral); pooled pinned host payloads, removing a second
full-frame host copy per frame (~10 ms at 4320p); 10-bit surviving the reader and the packer; the engine no longer
loaded twice at startup (−2.3 s); frame accounting that adds up (it overshot the loop by 30% before).

**Separately, an int8 body** (Triton kernels, head and tail in fp16, Compact models): 1.65x end to end against the
stock fp16 engine, measured interleaved on two clips, 0 of 205 and 0 of 201 frames above the bar on the pipeline's
own output. It is a second inference path rather than a patch, so it is offered only if you want it; it needs a card
with int8 tensor cores.

What we are **not** claiming: everything was measured on one card (4060 Ti, sm_89), so nothing here speaks to another
GPU, and every timing in our notes names the session it came from.

If none of this fits where the refactor is going, that is a fine answer - we will keep it in the fork. If it does,
tell us which piece to bring first and in which shape (`backend/src/io/`, PEP 8 module names, whatever else you are
converging on), and we will open it against `refactor`.

---

## Notes for us before sending

- Order matters: the harness first, because it is the piece that does not compete with their plans; reuse second,
  because it is their own TODO; int8 last, because it is the risky one.
- Say plainly that our record corpus is not part of the offer - only the machinery. Otherwise it reads as dumping.
- Do not promise review timelines; the line is frozen by the maintainer's own notice.
- One question at the end, not four. If they answer "not now", that is the end of it.
