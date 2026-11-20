# Draft — issue to TNTwise/REAL-Video-Enhancer (NOT SENT)

Title: **Frame reuse (`skip_render_if`) and a few pipeline fixes — offered against the `refactor` line**

---

Hi. We run a fork of RVE and spent about a week on it with a small fleet of AI agents, mostly on measurement rather
than features: an evidence harness where every number is a record carrying its population, instrument, config, the
commit it was taken on and the card. Two questions: is any of this interesting to you, and if it is, which shape do
you want it in?

Short version of what we have:

**Frame reuse.** If a frame has not changed since the last one we rendered, don't run the model again — but don't
emit the stale picture either; add back what changed, split by frequency: the low-frequency part of `LR_t - LR_ref`
at gain 1, the high-frequency part at 1.35 on flat static blocks, per channel, applied recursively on the previous
emitted frame. Measured on two Blu-ray cuts at 1080p→4K:

- calm cut: 63% of frames reused, **0 of 130** above the 0.1%-of-pixels bar, texture 1.001;
- action cut, held out: **1 of 124** with a cheap veto (skip the reuse when more than ~0.2% of pixels changed by
  more than 8 levels), 3 of 124 without it;
- about **1.9x** on the render loop where the content allows it, and 2.4 ms per corrected frame at 2x;
- it refuses on its own where it cannot be trusted: the auto gate refuses dense moving line art outright (8 of 8
  reuses would be visibly doubled, so it does not reuse there).

**This is your `TOOD.md` sentence.** The one that asks for a Frame attribute `skip_render_if` that filters and
interpolation can use. We would rather implement it that way, as an attribute the pipeline consults, than drop our
gate object into the render loop — say the word and the first PR is that attribute plus the correction, small and
self-contained, against `refactor`.

**Pipeline fixes that are not about reuse** and that any user on the stock TensorRT engine feels:

- 8-bit pipe: pack `yuv420p` on the GPU instead of `rgb24` — **−2.3 ms/frame**, picture-neutral;
- pooled pinned host payloads — removes a second full-frame host copy per frame (worth ~10 ms at 4320p);
- 10-bit survives the reader and the packer;
- the engine was being loaded twice at startup — **−2.3 s**;
- frame accounting: the profile now adds up (it overshot the loop by 30% before).

**And, separately, an int8 body** (Triton kernels, head/tail in fp16, Compact models): **1.65x end-to-end** against
the stock fp16 engine, measured interleaved on two clips, with the picture 0/205 and 0/201 frames above the bar on
the pipeline's own output. This one is a second inference path rather than a patch, so it is offered only if you want
it — and it needs a card with int8 tensor cores (Turing and later).

What we are **not** claiming: everything above was measured on one card (4060 Ti, sm_89); nothing speaks to another
GPU, and any timing in our notes says which session it came from.

If none of this fits where you are taking the refactor, that is a completely fine answer — we will keep it in the
fork and stop taking your time. If it does, tell us which piece to bring first and in which shape, and we will open
the PR against `refactor` with the conventions it uses now (`backend/src/io/`, PEP 8 module names).

---

## Notes for us before sending

- The issue must **not** promise a PR review timeline; the line is frozen by the maintainer's own notice.
- Link the numbers to our evidence only if asked; do not dump the harness in the issue.
- Check: does the repo have issues enabled for non-collaborators? (It does — 32 open, some from strangers.)
- Ask for the shape, not for permission to work. One question, not four.
