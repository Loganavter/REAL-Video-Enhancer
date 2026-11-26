# Code patterns for this repository

How to structure the code in REAL-Video-Enhancer so that the next fix does not recreate the last one. Written 2026-11-27
after a zero-trust audit of the previewer (15 modules, 147-method tab class, four generation counters and five clocks for one
picture) and a night of repairs that found three more instances of the same defect while fixing the first.

The rules are borrowed from BlinkTable, which is several times larger and does not fall apart the same way — but only the
cheap half: its own architecture documents (`docs/dev/CODE_PATTERNS.md`, `CODE_MASS_REDUCTION.md`, `CONTRACTS.md`) and its
141 contract-test modules. The Store/Dispatcher/plugin/tab machinery is **not** copied; it is sized for four workspace
tabs and a persisted session model, and a single-window upscaler does not need a Redux runtime.

Rule of thumb: **one fact, one owner, one writer.** Everything below is a consequence.

## 1. Thin owner + `use_cases/` as plain functions

**Symptom.** A class past ~500 lines or ~40 methods, mixing orthogonal concerns: a `git blame` or a method-name scan shows
one object doing loading *and* export *and* drag-and-drop *and* settings sync. Every new feature adds methods to it because
that is where the existing ones live.

**Rule.** Each orthogonal concern moves into its own module under `use_cases/` (for logic the tab/controller owns) or
`ui/` (for widget-side logic) as **plain functions whose first argument is the owner** — not a second class. The owner
keeps:

- construction and wiring (signal connections, subscriptions);
- the instance state those functions read and write;
- thin delegators for the names Qt or the tests look up by name (`dropEvent`, `eventFilter`, `_some_private` a test
  reaches) — a delegator does nothing but call the function with the same arguments.

```python
# ProcessTab.py  — the owner: state + wiring + delegators carrying the OLD names
def _supply_compare_source(self) -> None:
    supply.supply_compare_source(self)

# src/ui/preview/use_cases/supply.py — the logic, testable with a SimpleNamespace owner
def supply_compare_source(tab) -> None:
    ...
```

**Keep the method names when splitting.** Tests and callers reach `_private` methods directly; a rename for "cleanliness"
breaks call sites for no behavioural gain. The split is about *where the body lives*.

**When not to split.** A large single-responsibility state machine is not this problem. If every method reads and writes
the same tightly-coupled state toward one job (one async pipeline, one resource lifetime, one render pass's
`initialize→apply→record→release`), pulling pieces out just moves state back and forth. Sign you are in this case: you
cannot name the split modules without one being "misc"/"helpers", or every function needs half the instance's fields as
parameters.

**Who owns the state.** If the concern owns its own lifecycle (a queue with timers, a worker with an abort predicate), make
it a small collaborator object. If the state genuinely lives on the owner (a pending-paths list, a selection), keep the
function-taking-owner shape.

## 2. One writer per fact, and a test that proves it

A second writer is the defect; wrong values are only its symptom. Wherever one datum decides what a user sees or what the
renderer does, exactly one function may write it, and a test must fail when a second appears.

- The pane's painted pair: `src/ui/preview/pair.py` (planned; today `VideoPreviewSurface._commit_*`) — nothing else calls
  `renderer.set_frame`/`set_original`.
- The compare's source cache: `SourceRegister` — producers only `offer`, the presenter only asks.
- The generation token: minted by the thing that actually changed (a job, a file, a seek that was *asked for*), never by an
  attempt that was refused.
- The transport: `preview/controls.py` builds the one bar from `preview/actions.py` specs, and `preview/timeline.py`
  is the one scrubber. The pane used to paint a second strip and emit its own fraction-valued `seekRequested`; both
  were deleted (audit §6.6), so a seek has exactly one shape — the bar's `SEEK_TO` in seconds through the
  controller.

Cheap enforcement, in the spirit of BlinkTable's `test_strip_single_renderer.py` / `test_single_upload_queue.py`: grep-based
or AST tests that fail on the second owner existing at all, not on it misbehaving.

## 3. A token at landing sites, never at attempt sites

A refused action must be a no-op. Measured here (2026-11-27): a press the pane could not serve bumped the result generation
and cleared the compare register, so every source offer was then refused as the wrong generation — the blank half after a
click. The fix was to mint the token where a route actually asks.

## 4. One clock: the pane's

`VideoPreviewSurface.shown_moment()` is the only answer to "what moment is on screen". Three other clocks existed in the
same tab (transport, pair queue, preview playhead) and each pair that disagreed produced a gap, a neighbour or a decoder
that refilled a moment nobody showed.

Corollary: a function that needs "now" takes it as an argument or asks the pane. It never re-derives it from
`currentFrame`, a produced share, or a queue key.

## 5. One named render gate

`VideoPreviewSurface._request_frame_paint` is the gate through which a picture reaches the renderer. Every recurrence of
the compare defects came from a path that bypassed it. If you add a second painter, you have removed the invariant — add a
branch to the gate instead, and a matrix row for the state it creates.

## 6. A cache may not evict the answer it is being asked for

`SourceRegister` dropped the oldest moment when full; a paused pane asks for one moment every tick while the supply walks a
decode lead ahead of it, so the lead frames made the pane's own picture the oldest and the next walk dropped it: paint, gap,
paint, gap. Eviction is measured from the last lookup, and a paused pane is never asked for a lead.

Rule of reading: any `while len(...) > capacity: del ...[oldest]` in this tree deserves the question "can what the user is
looking at be the oldest entry?".

## 7. A counter nobody writes is worse than no counter

`compare fill asks=N arrivals=0` is the line that separates "never asked" from "asked and never answered". Its only write
site was deleted in a refactor and the line kept printing zero, so every run read as the frozen-pool bug. When telemetry
moves, the commit must show the line changing in a real run.

## 8. The state matrix, not a green suite

The previewer's 462 tests were green while the worst defect (a compare mode painting half a picture) was live: they pin
arithmetic and wiring, never states. Every pane change adds a row to
`harness/tests/test_preview_state_matrix.py` — state | expectation — and, where pixels are needed, one step in
`harness/scripts/live_matrix.sh`:

```
paused, sbs, pair            both halves of one moment, asks=0 over a quiet window, no gap
paused, sbs, no source       left half empty and stays empty; the result is never painted alone
playing, sbs, steady         every draw carries the source; delta_frames=0
playing, sbs, strip drag     the last matched pair holds; the gap is bounded; no lone-half flashes
seek on a paused pane        a gap, then the new moment; left and right are the same moment
render end -> finished file  no frame of the previous file; the first pair of the new generation is whole
cleared pane                 empty at 2 s and at 10 s
refused press                generation and register unchanged
export, sbs                  both halves from one owner; held_block only while the render is alive
```

## 9. Red-before by reverting, not by describing

A test that cannot go red pins nothing. The accepted proof in this repo: put the old behaviour back into the file, run the
test, watch it fail, restore, watch it pass. Six fixes on 2026-11-27 carry that proof; the docstrings say so.

## 10. Qt specifics (this app)

- The 100 ms tick arrives on the GUI thread; delegated functions stay synchronous there.
- A widget write from a worker thread uses `QTimer.singleShot(0, widget, fn)` — the three-argument form with the widget as
  context. A `QTimer`/`QThread` created off the GUI thread never fires.
- `QT_QPA_PLATFORM=offscreen` has no QRhi: paint/grab cannot be proven there. Prove behaviour offscreen, prove pixels on a
  leased display (`harness/scripts/live.sh`), and write `LIVE-PROOF WANTED:` lines for what a cell could not observe.
- `state:` on a hidden widget reports its last geometry — a click on a hidden control silently does nothing. Assert the
  effect (`checked`, a log line), never the absence of an error.
- The seek-reader tests rewrite tracked `scratch/seek-reader-*.txt`; restore them before committing.

## 11. Ask the pane; do not add a flag

Known live cases of a datum asserting a conclusion about a picture: `_shown_time` (accepted, not painted), `_result_gap`,
`pair_in_sync` with no source, `_seek_ceiling` from "a block exists", `may_paint = pending`, `_result_generation` bumped by a
refused press. A new boolean that describes the picture is the defect, not the fix. If the fact is not in the pane's own
state, add it to the pane.

## 12. A clear is a clear

Telling a pane to be empty must empty it: the picture, its queued frames, its source half, and the right to paint a
leftover picture back. Two live defects of this shape: `set_result(None)` kept the queued results, so a late source
re-keyed itself by its own moment (the pane's moment is None, so the moment guard had nothing to compare against) and
painted half a picture onto an empty pane; and a theme flip reset the "told to be empty" latch. After a clear the writer's
`painted()` answers `None` - not an object whose halves happen to be None, which a caller reads as "a picture is there".

## 13. What not to copy from BlinkTable

Its Store/Dispatcher/Reducer/undo runtime (~16.9k lines), the 44-member `TabContract` framework, the plugin protocol with
startup tiers, and the QRhi tiled-pass package protocol. All of them solve problems this app does not have; adopting them
for one preview pane is a second framework on top of the fix.
