"""The pooled pinned output payload (``backend/src/utils/Frame.py`` + ``TorchUtils``).

What is being tested, and why at this level: the output path used to convert the tensor
into a fresh host allocation and then copy it once more with ``.tobytes()`` because the
write queue holds ``bytes``. The pool replaces the second copy with a rotating set of
pinned host buffers. So the properties that matter are all about *slot lifetime*, not about
pixel values:

* a slot is allocated once per geometry and reused;
* a slot is never handed to the next frame while its payload is still being written
  (pool exhaustion blocks - the queue alone would let the render thread lap the writer and
  emit a torn frame);
* the payload is byte-identical to the unpooled path (same bytes, different storage);
* releasing twice is harmless and cannot over-release the pool.

Three layers, because the verification cell ships without torch (the bundled interpreter
lives outside the worker jail):

* **pool mechanics** - numpy only, allocator injected;
* **frame sizing / non-pooled behaviour** - numpy only, backend without torch;
* **torch** - the real conversions into a slot, ``pytest.importorskip("torch")``.
"""

from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np
import pytest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from src.utils.Frame import (  # noqa: E402
    PINNED_POOL_SLOTS,
    Frame,
    PinnedSlotPool,
    _as_byte_view,
)
from src.utils.PreviewChannel import FMT_YUV420 as PREVIEW_FMT_YUV420  # noqa: E402

# 4320x7680 is the geometry the task is written against (8K 16:9)
W8K, H8K = 7680, 4320
SLOT_BYTES_8K = W8K * H8K * 3  # 99_532_800 == 99.5 MB


def _numpy_pool(count: int, numel: int, dtype=np.uint8) -> PinnedSlotPool:
    """A pool whose slots are plain numpy arrays - no torch, no page locking."""
    return PinnedSlotPool(count, numel, dtype, allocator=lambda n, d: np.zeros(n, d))


def _frame(width: int = W8K, height: int = H8K, **kwargs) -> Frame:
    """A Frame on a backend that never touches torch (geometry + numpy paths only)."""
    options = dict(
        backend="ncnn",
        width=width,
        height=height,
        device="cpu",
        gpu_id=0,
        hdr_mode=False,
        dtype="float32",
    )
    options.update(kwargs)
    return Frame(**options)


# ======================================================================================
# pool mechanics (numpy only)
# ======================================================================================


def test_acquire_returns_index_and_flat_byte_view():
    pool = _numpy_pool(2, 32)
    slot, view = pool.acquire()
    assert slot in (0, 1)
    assert len(view) == 32, "len() of the payload must be the payload size in bytes"
    assert view.format == "B" and view.itemsize == 1
    view[:] = b"x" * 32  # writable: the conversion fills it through this view
    assert bytes(view) == b"x" * 32
    pool.release(slot)


def test_uint16_slot_view_is_byte_addressed():
    """The 10-bit payloads are uint16 elements: len(view) must still be bytes."""
    pool = _numpy_pool(1, 8, np.uint16)
    slot, view = pool.acquire()
    assert len(view) == 16
    pool.release(slot)


def test_slots_are_allocated_once_per_geometry():
    calls = []

    def allocator(numel, dtype):
        calls.append(numel)
        return np.zeros(numel, dtype)

    pool = PinnedSlotPool(4, 16, np.uint8, allocator=allocator)
    for _ in range(5):  # five rounds of four frames through a four-slot pool
        slots = [pool.acquire()[0] for _ in range(4)]
        for slot in slots:
            pool.release(slot)
    assert len(calls) == 4, f"slots must be allocated once and reused, got {len(calls)}"
    assert pool.allocated == 4


def test_lazy_allocation_only_pays_for_slots_actually_used():
    pool = _numpy_pool(4, 16)
    slot, _view = pool.acquire()
    assert pool.allocated == 1
    pool.release(slot)


def test_buffer_and_view_share_storage():
    """Production fills the slot through ``buffer(slot)`` (a torch tensor) and pipes the
    view: they must be the same memory."""
    pool = _numpy_pool(1, 4)
    slot, view = pool.acquire()
    pool.buffer(slot)[:] = np.array([1, 2, 3, 4], np.uint8)
    assert bytes(view) == b"\x01\x02\x03\x04"
    pool.release(slot)


def test_release_is_idempotent_and_never_over_releases():
    pool = _numpy_pool(1, 8)
    slot, _view = pool.acquire()
    assert pool.release(slot) is True
    assert pool.release(slot) is False, "a duplicate release must be refused"
    assert pool.release(slot) is False

    # exactly one permit is available: one acquire succeeds, a second must block
    again, _ = pool.acquire()
    got = threading.Event()

    def waiter():
        other, _v = pool.acquire()
        got.set()
        pool.release(other)

    thread = threading.Thread(target=waiter, daemon=True)
    thread.start()
    assert not got.wait(0.15), "an over-released pool would hand out a second live slot"
    pool.release(again)
    assert got.wait(5.0)
    thread.join(5.0)


def test_pool_exhaustion_blocks_instead_of_overwriting():
    pool = _numpy_pool(2, 8)
    first, _ = pool.acquire()
    second, _ = pool.acquire()

    got = threading.Event()
    acquired: dict = {}

    def waiter():
        slot, _view = pool.acquire()
        acquired["slot"] = slot
        got.set()
        pool.release(slot)

    thread = threading.Thread(target=waiter, daemon=True)
    thread.start()
    assert not got.wait(0.15), "acquire() handed out a live slot instead of blocking"
    pool.release(first)
    assert got.wait(5.0), "releasing a slot must wake the waiting producer"
    assert acquired["slot"] == first, "the waiter must get the freed slot"
    thread.join(5.0)
    assert pool.blocked >= 1
    assert pool.max_in_use == 2
    pool.release(second)
    assert pool.in_use == 0


def test_reused_slot_never_tears_a_payload_being_written():
    """The slow-writer case the pool exists for: the render thread must not fill a slot
    while the writer is still reading it."""
    pool = _numpy_pool(1, 4096)
    first = bytes(range(256)) * 16
    slot, view = pool.acquire()
    view[:] = first
    seen: dict = {}

    def slow_consumer():
        # stands in for the pipe write of an encoder that fell behind
        time.sleep(0.05)
        seen["payload"] = bytes(view)
        pool.release(slot)

    consumer = threading.Thread(target=slow_consumer, daemon=True)
    consumer.start()

    started = time.monotonic()
    new_slot, new_view = pool.acquire()  # blocks until the consumer is done with the slot
    waited = time.monotonic() - started
    assert "payload" in seen, "acquire() returned while the payload was still being read"
    assert seen["payload"] == first, "the live payload was overwritten by the next frame"
    assert waited >= 0.04, f"acquire() did not actually wait (waited {waited:.3f}s)"
    assert new_slot == slot
    assert bytes(new_view) == first, "the reused slot must still hold the previous payload"
    new_view[:] = b"\xAA" * 4096
    assert set(bytes(new_view)) == {0xAA}
    consumer.join(5.0)
    pool.release(new_slot)


def test_two_producers_two_consumers_keep_payloads_distinct():
    """A rotating pool under concurrent fill/read must not mix frames up."""
    pool = _numpy_pool(3, 256)
    per_producer = 20
    produced = 2 * per_producer
    handoffs: list = []
    handoff_lock = threading.Lock()
    consumer_results: list = []
    stop = threading.Event()
    done_consuming = threading.Event()

    def producer(tag):
        for _ in range(per_producer):
            slot, view = pool.acquire()
            payload = bytes([tag]) * 256
            view[:] = payload
            with handoff_lock:
                handoffs.append((slot, view, payload))

    def consumer():
        while not stop.is_set() or handoffs:
            with handoff_lock:
                if not handoffs:
                    time.sleep(0.001)
                    continue
                slot, view, expected = handoffs.pop(0)
            consumer_results.append((bytes(view), expected))
            pool.release(slot)
        done_consuming.set()

    consumer_thread = threading.Thread(target=consumer, daemon=True)
    consumer_thread.start()
    producers = [
        threading.Thread(target=producer, args=(1,), daemon=True),
        threading.Thread(target=producer, args=(2,), daemon=True),
    ]
    for thread in producers:
        thread.start()
    for thread in producers:
        thread.join(20.0)
    stop.set()
    assert done_consuming.wait(20.0)
    assert len(consumer_results) == produced, f"{len(consumer_results)} of {produced} payloads"
    for payload, expected in consumer_results:
        assert payload == expected, "a consumer read a payload that had been recycled"
    assert pool.in_use == 0


def test_stats_report_the_pool_and_its_backpressure():
    pool = _numpy_pool(2, 8)
    slot, _view = pool.acquire()
    text = pool.stats()
    assert "1/2 slots in use" in text
    assert "elems" in text and "waits on a slow writer" in text
    pool.release(slot)


def test_byte_view_rejects_a_pool_of_an_unexpected_shape():
    """A pool whose allocator disagrees about the payload size must be a loud error, not a
    silently short (or overlong) payload. The first slot defines the size; the second is
    checked against it."""
    sizes = []

    def odd_allocator(numel, dtype):
        sizes.append(numel + len(sizes))
        return np.zeros(sizes[-1], dtype)

    pool = PinnedSlotPool(2, 8, np.uint8, allocator=odd_allocator)
    first, view = pool.acquire()
    assert len(view) == 8
    with pytest.raises(RuntimeError):
        pool.acquire()  # 9 bytes against the 8 the pool already handed out
    assert pool.in_use == 1, "the failed acquisition must not leak its permit"
    pool.release(first)
    assert pool.in_use == 0


def test_a_pool_needs_at_least_one_slot():
    with pytest.raises(ValueError):
        PinnedSlotPool(0, 8, np.uint8)


def test_as_byte_view_flattens_a_multidimensional_buffer():
    view = _as_byte_view(np.zeros((3, 4), np.uint16))
    assert len(view) == 24 and view.ndim == 1 and view.format == "B"


# ======================================================================================
# frame sizing and the non-pooled path (numpy only)
# ======================================================================================


def test_default_pool_size_is_documented_and_small():
    assert 4 <= PINNED_POOL_SLOTS <= 6, "the task asks for a default of 4-6 slots"
    frame = _frame()
    assert frame.pooled_output is False, "pooling must be opt-in per frame"
    assert frame.pool_slots == PINNED_POOL_SLOTS


def test_pooled_slot_size_at_4320x7680():
    """Documents the memory cost: 99.5 MB per slot, for rgb24 and for yuv420p10le."""
    frame = _frame()
    assert frame._pooled_numel() == SLOT_BYTES_8K
    frame.yuv10_output = True
    assert frame._pooled_numel() == SLOT_BYTES_8K // 2, "uint16 elements: half as many"
    assert frame._pooled_numel() * 2 == SLOT_BYTES_8K


def test_pooled_numel_uses_the_tensor_geometry():
    """An upscaled frame's width/height can lag the tensor it holds; the slot must follow
    the tensor or the copy would overrun/underrun."""

    class StandIn:
        shape = (1, 3, 7, 11)

    frame = _frame(width=4, height=3)
    frame._tensor = StandIn()
    assert frame._pooled_numel() == 7 * 11 * 3


def test_non_pooled_numpy_path_is_unchanged():
    array = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    frame = _frame(width=6, height=4)
    frame.set_frame_np(array)
    plain = frame.get_frame_bytes()
    assert type(plain) is bytes
    assert plain == array.tobytes()

    pooled = _frame(width=6, height=4)
    pooled.set_frame_np(array)
    pooled.pooled_output = True  # ignored: this backend has no torch conversion
    assert pooled.get_frame_bytes() == array.tobytes()
    pooled.release_payload()


def test_release_payload_is_a_no_op_without_a_pool():
    frame = _frame(width=6, height=4)
    frame.set_frame_np(np.zeros((4, 6, 3), np.uint8))
    frame.get_frame_bytes()
    frame.release_payload()
    frame.release_payload()  # idempotent
    assert frame._pool is None and frame._pool_slot is None


# ======================================================================================
# frame-side pooling orchestration, with a stand-in for TorchUtils (numpy only)
# ======================================================================================


class _FakeTorchUtils:
    """The parts of TorchUtils that Frame touches, with numpy instead of torch.

    This is where the Frame half of the fix is verified: which pool it asks for, that the
    conversion lands in the slot, that the slot comes back on every path (including the
    one where the conversion raises) and that the payload is the view the writer gets.
    """

    def __init__(self, convert=None):
        self.high_bitdepth = False
        self.pools: dict = {}
        self.fills = 0
        self._convert = convert

    def pinned_slots(self, n, numel, dtype, allocator=None):
        key = (int(n), int(numel), str(dtype))
        if key not in self.pools:
            self.pools[key] = PinnedSlotPool(
                n, numel, dtype, allocator=lambda count, dt: np.zeros(count, dt)
            )
        return self.pools[key]

    def _fill(self, tensor, dest):
        if self._convert is not None:
            self._convert(tensor, dest)
        dest[...] = np.asarray(tensor).reshape(-1)
        self.fills += 1

    def tensor_to_frame(self, tensor, dest=None):
        self._fill(tensor, dest)
        return dest

    def tensor_to_yuv420p10(self, tensor, dest=None):
        self._fill(tensor, dest)
        return dest

    def sync_all_streams(self):
        pass


@pytest.fixture
def fake_torch_env():
    from src.utils import Frame as FrameModule

    saved = (FrameModule._torch, FrameModule._torch_utils)
    utils = _FakeTorchUtils()
    FrameModule._torch = None
    FrameModule._torch_utils = utils
    try:
        yield FrameModule, utils
    finally:
        FrameModule._torch, FrameModule._torch_utils = saved


def _pooled_frame(FrameModule, width, height, high_bitdepth=False, fill=None):
    frame = FrameModule.Frame(
        backend="pytorch", width=width, height=height, device="cpu", gpu_id=0,
        hdr_mode=high_bitdepth, dtype="float32", high_bitdepth=high_bitdepth,
    )
    frame.pooled_output = True
    numel = frame._pooled_numel()
    dtype = np.uint16 if high_bitdepth else np.uint8
    data = np.arange(numel, dtype=dtype) if fill is None else fill
    frame._tensor = data.reshape(1, 3, height, width)
    return frame, data


def test_frame_fills_a_pool_slot_and_hands_out_a_view(fake_torch_env):
    FrameModule, utils = fake_torch_env
    frame, data = _pooled_frame(FrameModule, 8, 6)

    payload = frame.get_frame_bytes()
    assert isinstance(payload, memoryview)
    assert len(payload) == data.size
    assert bytes(payload) == data.tobytes(), "the payload must be the converted frame"
    assert utils.fills == 1
    assert frame._pool is not None and frame._tensor is None

    pool = frame._pool
    assert pool.in_use == 1
    frame.release_payload()
    assert pool.in_use == 0
    assert frame._pool is None


def test_pooled_payload_is_cached_not_reconverted(fake_torch_env):
    FrameModule, utils = fake_torch_env
    frame, _data = _pooled_frame(FrameModule, 8, 6)
    first = frame.get_frame_bytes()
    second = frame.get_frame_bytes()
    assert first is second
    assert utils.fills == 1, "asking twice must not spend a second slot or conversion"
    assert frame._pool.in_use == 1
    frame.release_payload()
    frame.release_payload()  # idempotent


def test_a_released_pooled_frame_refuses_to_be_emitted_again(fake_torch_env):
    """Temporal reuse remembers a frame and emits the same object again; a pooled payload is
    a one-shot buffer, so that must fail loudly instead of writing another frame's pixels."""
    FrameModule, utils = fake_torch_env
    frame, _data = _pooled_frame(FrameModule, 8, 6)
    frame.get_frame_bytes()
    frame.release_payload()
    with pytest.raises(RuntimeError, match="pooled frame cannot be emitted twice"):
        frame.get_frame_bytes()
    assert next(iter(utils.pools.values())).in_use == 0


def test_frame_pool_dtype_follows_the_data_path(fake_torch_env):
    FrameModule, utils = fake_torch_env
    eight_bit, _ = _pooled_frame(FrameModule, 8, 6)
    eight_bit.get_frame_bytes()
    sixteen_bit, data = _pooled_frame(FrameModule, 8, 6, high_bitdepth=True)
    payload = sixteen_bit.get_frame_bytes()
    assert len(payload) == data.size * 2, "uint16 payload: two bytes per element"
    assert len(utils.pools) == 2, "8-bit and 16-bit payloads need separate pools"
    eight_bit.release_payload()
    sixteen_bit.release_payload()


def test_frame_releases_the_slot_when_the_conversion_fails(fake_torch_env):
    FrameModule, utils = fake_torch_env
    utils._convert = lambda tensor, dest: (_ for _ in ()).throw(RuntimeError("CUDA OOM"))
    frame, _data = _pooled_frame(FrameModule, 8, 6)
    with pytest.raises(RuntimeError):
        frame.get_frame_bytes()
    assert frame._pool is None and frame._pool_slot is None
    pool = next(iter(utils.pools.values()))
    assert pool.in_use == 0, "a failed conversion must not eat a slot forever"


def test_frame_pool_blocks_a_second_frame_while_the_first_is_live(fake_torch_env):
    """The failure mode the task asks about: with one slot, frame N+1 waits for the writer
    instead of writing into frame N's buffer."""
    FrameModule, _utils = fake_torch_env
    first, first_data = _pooled_frame(FrameModule, 8, 6)
    first.pool_slots = 1
    first_view = first.get_frame_bytes()

    second, second_data = _pooled_frame(FrameModule, 8, 6, fill=np.full(8 * 6 * 3, 7, np.uint8))
    second.pool_slots = 1
    done = threading.Event()

    def render_next():
        second.get_frame_bytes()
        done.set()

    thread = threading.Thread(target=render_next, daemon=True)
    thread.start()
    assert not done.wait(0.15), "the second frame must wait for the first payload"
    assert bytes(first_view) == first_data.tobytes(), "the live payload must be intact"
    first.release_payload()
    assert done.wait(5.0)
    assert bytes(second.get_frame_bytes()) == second_data.tobytes()
    second.release_payload()
    thread.join(5.0)


def test_dropping_a_pooled_frame_returns_its_slot(fake_torch_env):
    """The seatbelt: an aborted render must not shrink the pool to zero and hang on
    acquire()."""
    FrameModule, utils = fake_torch_env
    frame, _data = _pooled_frame(FrameModule, 8, 6)
    frame.pool_slots = 1
    frame.get_frame_bytes()
    pool = frame._pool
    assert pool.in_use == 1
    del frame
    assert pool.in_use == 0, "Frame.__del__ must release a slot nobody released"


# The writer does `Popen(...).stdin.buffer.write(payload)` with `text=True` (see
# FFmpegBuffers.write_out_frames): the payload must be a buffer object, not a wrapper.
_PIPE_READER = (
    "import hashlib, sys;"
    " data = sys.stdin.buffer.read();"
    " print(len(data), hashlib.sha256(data).hexdigest())"
)


def _pipe_reader():
    import subprocess

    return subprocess.Popen(
        [sys.executable, "-c", _PIPE_READER],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )


def test_payload_is_accepted_by_a_real_pipe():
    import hashlib

    payload_bytes = bytes(range(256)) * 4
    pool = _numpy_pool(1, len(payload_bytes))
    slot, view = pool.acquire()
    view[:] = payload_bytes

    reader = _pipe_reader()
    reader.stdin.buffer.write(view)  # exactly the writer's call
    reader.stdin.close()
    length, digest = reader.stdout.read().split()
    reader.wait(timeout=10)
    pool.release(slot)

    assert int(length) == len(payload_bytes)
    assert digest == hashlib.sha256(payload_bytes).hexdigest()


def test_writer_sequence_pipes_a_pooled_frame_and_releases_the_slot(fake_torch_env):
    """The INTEGRATION sequence, verbatim: take the item off the queue, write its payload,
    release the slot. The pipe gets the frame's bytes and the pool is empty afterwards."""
    import hashlib

    FrameModule, utils = fake_torch_env
    frame, data = _pooled_frame(FrameModule, 8, 6)
    frame.pool_slots = 1
    queue = [frame]

    reader = _pipe_reader()
    item = queue.pop(0)
    payload = item.get_frame_bytes() if hasattr(item, "get_frame_bytes") else item
    reader.stdin.buffer.write(payload)
    if hasattr(item, "release_payload"):
        item.release_payload()
    reader.stdin.close()

    length, digest = reader.stdout.read().split()
    reader.wait(timeout=10)
    assert int(length) == data.size
    assert digest == hashlib.sha256(data.tobytes()).hexdigest()
    assert item._pool is None
    assert next(iter(utils.pools.values())).in_use == 0



# ======================================================================================
# torch: the real conversions into a slot
# ======================================================================================


def _torch_env(high_bitdepth: bool = False, width: int = 8, height: int = 6):
    """Seed ``utils.Frame``'s lazily-created singletons without BackendDetect's device
    probing, which needs an accelerator host and is irrelevant on CPU."""
    torch = pytest.importorskip("torch")
    try:
        from src.pytorch.TorchUtils import TorchUtils
    except Exception as error:  # pragma: no cover - host without an accelerator stack
        pytest.skip(f"TorchUtils unavailable: {error}")

    from src.utils import Frame as FrameModule

    FrameModule._torch = torch
    FrameModule._pytorch_device = torch.device("cpu")
    FrameModule._pytorch_dtype = torch.float32
    FrameModule._pytorch_stream = None
    if FrameModule._torch_utils is None or FrameModule._torch_utils.high_bitdepth != high_bitdepth:
        FrameModule._torch_utils = TorchUtils(
            width=width,
            height=height,
            device_type="cpu",
            hdr_mode=high_bitdepth,
            gpu_id=0,
            high_bitdepth=high_bitdepth,
        )
    return torch, FrameModule


def _torch_frame(FrameModule, torch, width, height, high_bitdepth=False, **kwargs):
    frame = FrameModule.Frame(
        backend="pytorch",
        width=width,
        height=height,
        device="cpu",
        gpu_id=0,
        hdr_mode=high_bitdepth,
        dtype="float32",
        high_bitdepth=high_bitdepth,
    )
    frame.pooled_output = True
    frame._tensor = kwargs.get("tensor")
    return frame


def test_pinned_slots_are_shared_per_geometry():
    torch, FrameModule = _torch_env()
    utils = FrameModule._torch_utils
    first = utils.pinned_slots(3, 64, torch.uint8)
    again = utils.pinned_slots(3, 64, torch.uint8)
    assert first is again, "a Frame per input frame must share one pool"
    other = utils.pinned_slots(4, 64, torch.uint8)
    assert other is not first, "pool size is part of the identity"
    slot, view = first.acquire()
    buffer = first.buffer(slot)
    assert isinstance(buffer, torch.Tensor)
    assert buffer.dtype == torch.uint8 and buffer.device.type == "cpu"
    if torch.cuda.is_available():
        assert buffer.is_pinned(), "host buffers must be pinned out of the GPU"
    assert len(view) == 64
    first.release(slot)
    assert "slots in use" in utils.pinned_pool_stats()


def test_pooled_rgb_payload_is_byte_identical_and_releases():
    torch, FrameModule = _torch_env()
    width, height = 8, 6
    tensor = torch.rand(1, 3, height, width)

    plain = _torch_frame(FrameModule, torch, width, height, tensor=tensor)
    plain.pooled_output = False
    expected = plain.get_frame_bytes()

    frame = _torch_frame(FrameModule, torch, width, height, tensor=tensor.clone())
    payload = frame.get_frame_bytes()
    assert isinstance(payload, memoryview), "the writer needs a writable buffer object"
    assert len(payload) == width * height * 3
    assert bytes(payload) == expected.tobytes()
    # the GPU tensor is dropped once the payload owns the conversion
    assert frame._tensor is None
    assert frame._pool is not None and frame._pool_slot is not None
    frame.release_payload()
    assert frame._pool is None and frame._pool_slot is None
    frame.release_payload()  # idempotent
    with pytest.raises(RuntimeError):
        frame.get_frame_bytes()  # the payload was consumed; nothing left to hand out


def test_pooled_rgb48_payload_is_byte_identical():
    torch, FrameModule = _torch_env(high_bitdepth=True)
    width, height = 6, 4
    tensor = torch.rand(1, 3, height, width)

    plain = FrameModule.Frame(
        backend="pytorch", width=width, height=height, device="cpu", gpu_id=0,
        hdr_mode=True, dtype="float32", high_bitdepth=True,
    )
    plain.set_frame_tensor(tensor.clone())
    expected = plain.get_frame_bytes()

    frame = _torch_frame(FrameModule, torch, width, height, high_bitdepth=True,
                         tensor=tensor.clone())
    payload = frame.get_frame_bytes()
    assert len(payload) == width * height * 3 * 2, "rgb48le is 6 bytes/pixel"
    assert bytes(payload) == expected.tobytes()
    frame.release_payload()


def test_pooled_yuv420p10_payload_is_byte_identical():
    torch, FrameModule = _torch_env(high_bitdepth=True)
    width, height = 8, 6
    tensor = torch.rand(1, 3, height, width)

    frame = _torch_frame(FrameModule, torch, width, height, high_bitdepth=True,
                         tensor=tensor.clone())
    frame.yuv10_output = True
    payload = frame.get_frame_bytes()
    expected = FrameModule._torch_utils.tensor_to_yuv420p10(tensor)
    # the converter hands back a freshly allocated numpy array (one host copy less); compare
    # through a common byte view so the test does not care which of the two it is
    expected_bytes = expected.tobytes() if hasattr(expected, "tobytes") else bytes(expected)
    assert len(bytes(payload)) == len(expected_bytes), "3 bytes/pixel: Y + chinroma planes"
    assert bytes(payload) == expected_bytes
    frame.release_payload()


def test_a_slow_writer_through_frames_never_tears_a_payload():
    """The end-to-end shape of the fix: frames are queued as handles, the writer releases
    each slot after it has copied the payload out, and one slot is enough to stay correct
    (the second frame waits for the first)."""
    torch, FrameModule = _torch_env()
    width, height = 8, 6
    tensors = [torch.rand(1, 3, height, width) for _ in range(4)]
    expected = []
    for tensor in tensors:
        probe = FrameModule.Frame(
            backend="pytorch", width=width, height=height, device="cpu", gpu_id=0,
            hdr_mode=False, dtype="float32",
        )
        probe.set_frame_tensor(tensor.clone())
        expected.append(probe.get_frame_bytes())

    queue: list = []
    results: list = []
    finished = threading.Event()

    def writer():
        index = 0
        while index < len(tensors):
            if not queue:
                time.sleep(0.001)
                continue
            item = queue.pop(0)
            payload = item.get_frame_bytes()
            time.sleep(0.02)  # an encoder that lags
            results.append(bytes(payload))
            item.release_payload()
            index += 1
        finished.set()

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()

    pool = None
    for tensor in tensors:
        frame = _torch_frame(FrameModule, torch, width, height, tensor=tensor)
        frame.pool_slots = 1  # one slot: every frame must wait for the writer
        frame.get_frame_bytes()  # acquires (and blocks when the writer is behind)
        pool = frame._pool
        queue.append(frame)

    assert finished.wait(20.0), "the writer never drained the queue"
    assert results == [bytes(expect) for expect in expected]
    assert pool is not None and pool.max_in_use == 1
    assert pool.blocked >= len(tensors) - 1, "the pool must have applied backpressure"
    assert pool.in_use == 0, "every slot must be back in the pool"


# ======================================================================================
# the payload the int8 tail wrote itself (RVE_TAIL_YUV)
#
# The body can hand the frame a finished yuv420p payload instead of an RGB tensor: PixelShuffle(2)
# makes the 4:2:0 chroma average register-local, so the tail's epilogue produces the six bytes per
# input pixel directly (row rve-tail-csc-yuv-2026-11-20). The frame's job then is to COPY those
# bytes, and the failure this guards against is the pack running the colour conversion over samples
# that are already yuv - silent, plausible-looking, and wrong.
# ======================================================================================


def _payload_frame(FrameModule, torch, width, height, pooled):
    """A frame holding a packed yuv420p payload of the given OUTPUT geometry."""
    n = width * height
    chroma = (height // 2) * (width // 2)
    data = torch.arange(n + 2 * chroma, dtype=torch.uint8)
    frame = FrameModule.Frame(
        backend="pytorch", width=width, height=height, device="cpu", gpu_id=0,
        hdr_mode=False, dtype="float32",
    )
    frame.pooled_output = pooled
    frame.yuv8_output = True
    frame._tensor = data
    return frame, data


def test_a_packed_yuv420p_payload_is_recognised_and_sized_by_the_payload():
    torch, FrameModule = _torch_env()
    width, height = 8, 6
    frame, data = _payload_frame(FrameModule, torch, width, height, pooled=True)
    assert frame._is_raw_payload(), "a flat uint8 CUDA buffer is the pipe payload, not an image"
    assert frame._pooled_numel() == data.numel(), \
        "the slot must follow the payload: the image formula would ask for 3 bytes/pixel"


def test_a_packed_payload_is_copied_into_the_slot_not_converted():
    """The production copy path: a real pinned slot, a real `copy_`, and 1.5 bytes/pixel of payload.

    A conversion here would be arithmetic over samples that are already yuv; the check is that the
    slot holds exactly the payload's bytes (the colour conversion would not reproduce them)."""
    torch, FrameModule = _torch_env()
    width, height = 8, 6
    frame, data = _payload_frame(FrameModule, torch, width, height, pooled=True)
    payload = frame.get_frame_bytes()
    assert isinstance(payload, memoryview)
    assert len(payload) == data.numel() == width * height + 2 * (height // 2) * (width // 2)
    assert bytes(payload) == data.numpy().tobytes(), "the bytes must arrive unchanged"
    assert frame._pool is not None and frame._tensor is None
    frame.release_payload()
    assert frame._pool is None and frame._pool_slot is None


def test_a_packed_payload_without_the_ring_is_handed_on_as_bytes():
    torch, FrameModule = _torch_env()
    width, height = 8, 6
    frame, data = _payload_frame(FrameModule, torch, width, height, pooled=False)
    out = frame.get_frame_bytes()
    assert isinstance(out, np.ndarray) and out.dtype == np.uint8
    assert out.tobytes() == data.numpy().tobytes()


def test_a_packed_payload_preview_is_sampled_yuv():
    """The information channel takes the payload's own planes at thumbnail size: Y at
    `w*h`, U and V at `(w+1)//2` x `(h+1)//2`. Grey in, grey out pins shape and range, and
    the assertion that it is *not* RGB pins the format the reader must decode."""
    torch, FrameModule = _torch_env()
    width, height = 8, 6
    frame, _data = _payload_frame(FrameModule, torch, width, height, pooled=False)
    frame._tensor = torch.full((width * height + 2 * (height // 2) * (width // 2),), 128,
                               dtype=torch.uint8)
    payload, fmt = frame.preview_small(width, height)
    assert fmt == PREVIEW_FMT_YUV420
    expected = width * height + 2 * ((width + 1) // 2) * ((height + 1) // 2)
    assert len(payload) == expected, "sampled planes, not an rgb picture"
    levels = np.frombuffer(payload, np.uint8)
    assert levels.min() == 128 and levels.max() == 128, "128 stays 128 through the sampler"


def test_an_image_tensor_is_never_mistaken_for_a_payload():
    torch, FrameModule = _torch_env()
    frame = FrameModule.Frame(
        backend="pytorch", width=8, height=6, device="cpu", gpu_id=0,
        hdr_mode=False, dtype="float32",
    )
    frame._tensor = torch.rand(1, 3, 6, 8)
    assert not frame._is_raw_payload()
    frame._tensor = torch.zeros(1, 3, 4, 4, dtype=torch.uint16)
    assert not frame._is_raw_payload(), "a 16-bit frame is an image, not a payload"
