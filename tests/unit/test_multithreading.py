import threading

import pytest
import torch
import importlib

import triton
import triton.language as tl

import tilelens
from tilelens.clients.profiler.profiler import Profiler
from tilelens.core.config import config as cfg

trace_state = importlib.import_module("tilelens.core.trace")


@pytest.fixture
def one_sm():
    previous = tilelens.config.num_sms
    tilelens.config.num_sms = 1
    yield
    tilelens.config.num_sms = previous


@pytest.fixture
def two_sms():
    previous = tilelens.config.num_sms
    tilelens.config.num_sms = 2
    yield
    tilelens.config.num_sms = previous


@pytest.fixture(autouse=True)
def _clear_traces():
    tilelens.core.clear()
    yield
    tilelens.core.clear()


@tilelens.trace("tracer")
@triton.jit
def _producer_consumer(x, out):
    pid = tl.program_id(0)
    if pid == 0:
        counter = 0.0
        max_spin = 10_000.0
        # Spin until PID1 updates the output, but bail out after max_spin to avoid hangs.
        while tl.atomic_cas(out, 0.0, 0.0) == 0.0 and counter < max_spin:
            counter += 1.0
        tl.atomic_add(out, counter)
    else:
        # Do a bit of work, then publish a non-zero value for PID0 to observe.
        arange = tl.arange(0, 32)
        s = 0.0
        for i in range(4):
            s += tl.sum(tl.load(x + i * 32 + arange))
        tl.atomic_add(out, s + 1.0)


def test_producer_consumer_hang(one_sm):
    """With only one SM, PID0 should spin to the cap or hang because blocks run sequentially."""
    out = torch.zeros((), dtype=torch.float32)
    x = torch.zeros((128,), dtype=torch.float32)

    thread = threading.Thread(target=lambda: _producer_consumer[(2,)](x, out))
    thread.start()

    # note: don't use timeout=2 for this, it'll cause the next
    # test to run before this thread unpatches tilelens ->
    # this thread will unpatch while the next test
    # needs patched ops, causing weird failures
    thread.join()
    assert (
        out.item() >= 10_000
    ), "producer_consumer should hang or PID 0 spin to max limit (10K) when blocks run sequentially"


def test_producer_consumer_converges(two_sms):
    """With two SMs, PID1 should run concurrently so PID0 exits the spin loop before the cap."""
    out = torch.zeros((), dtype=torch.float32)
    x = torch.zeros((128,), dtype=torch.float32)

    thread = threading.Thread(target=lambda: _producer_consumer[(2,)](x, out))
    thread.start()
    thread.join()
    assert (
        0 < out.item() < 10_000
    ), "producer_consumer should complete and PID 0 shouldn't spin to max limit (10K) when blocks run concurrently"


@tilelens.trace("tracer")
@triton.jit
def _racing_threads(x, out):
    pid = tl.program_id(0)
    arange = tl.arange(0, 8)
    if pid == 0:
        s = 0.0
        for i in range(256):
            s += tl.sum(tl.load(x + i * 8 + arange))
        tl.atomic_cas(out, 0.0, 256.0 + s)
    else:
        s = tl.sum(tl.load(x + arange))
        tl.atomic_cas(out, 0.0, 1.0 + s)


def test_racing_threads_prefers_first_block(one_sm):
    """Sequential execution: slower PID0 wins the atomic race when blocks are serialized."""
    x = torch.zeros((8 * 256,), dtype=torch.float32)
    out = torch.zeros((), dtype=torch.float32)
    _racing_threads[(2,)](x, out)
    assert out.item() == pytest.approx(
        256.0
    ), "PID 0 in racing_threads should win when blocks run sequentially"


def test_racing_threads_prefers_fast_block(two_sms):
    """Concurrent execution: faster PID1 should win the atomic race when blocks run in parallel."""
    x = torch.zeros((8 * 256,), dtype=torch.float32)
    out = torch.zeros((), dtype=torch.float32)
    _racing_threads[(2,)](x, out)
    assert out.item() == pytest.approx(
        1.0
    ), "PID 1 in racing_threads should win when blocks run concurrently"


@tilelens.trace("tracer")
@triton.jit
def _write_pid(out):
    pid = tl.program_id(0)
    tl.store(out + pid, pid)


@triton.jit
def _profiler_load_store(x, out):
    pid = tl.program_id(0)
    arange = tl.arange(0, 8)
    offset = pid * 8 + arange
    vals = tl.load(x + offset)
    tl.store(out + offset, vals)


@tilelens.trace("sanitizer")
@triton.jit
def _sanitizer_two_blocks(x, out):
    pid = tl.program_id(0)
    x_slice = tl.load(x) + tl.cast(pid, tl.float32)
    tl.store(out + pid, x_slice)


def test_tracer_records_all_blocks(two_sms):
    """Tracer should record a Grid entry for every launched block when multiple blocks run."""
    out = torch.zeros((4,), dtype=torch.float32)
    _write_pid[(4,)](out)
    assert trace_state.launches, "trace launch should be recorded"
    grid_records = [r for r in trace_state.launches[-1].records if hasattr(r, "idx")]
    seen = {tuple(record.idx) for record in grid_records}
    assert seen == {(i, 0, 0) for i in range(4)}


def test_tracer_records_are_thread_safe(two_sms):
    """Run many blocks concurrently and ensure every block is recorded once."""
    out = torch.zeros((32,), dtype=torch.float32)
    _write_pid[(32,)](out)
    assert trace_state.launches, "trace launch should be recorded"
    grid_records = [r for r in trace_state.launches[-1].records if hasattr(r, "idx")]
    seen = {tuple(record.idx) for record in grid_records}
    assert seen == {(i, 0, 0) for i in range(32)}


def test_sanitizer_handles_concurrent_blocks(two_sms):
    """Sanitizer should complete and report no OOB even when blocks run concurrently."""
    x = torch.tensor(3.0)
    out = torch.zeros((2,), dtype=torch.float32)
    thread = threading.Thread(target=lambda: _sanitizer_two_blocks[(2,)](x, out))
    thread.start()
    thread.join(timeout=3)
    assert not thread.is_alive(), "sanitizer execution should complete under timeout"
    # Symbolic sanitizer may not perform concrete stores; just ensure no OOB records
    assert not trace_state.launches[-1].records, "sanitizer should not report OOB"


def test_sanitizer_handles_serial_blocks(one_sm):
    """Sanitizer should complete and report no OOB even when blocks run sequentially."""
    x = torch.tensor(3.0)
    out = torch.zeros((2,), dtype=torch.float32)
    thread = threading.Thread(target=lambda: _sanitizer_two_blocks[(2,)](x, out))
    thread.start()
    thread.join(timeout=3)
    assert not thread.is_alive(), "sanitizer execution should complete under timeout"
    assert not trace_state.launches[-1].records, "sanitizer should not report OOB"


@pytest.fixture
def _no_profiler_sampling():
    """Disable profiler block sampling and load/store skipping for the test."""
    prev_block_sampling = cfg.profiler_enable_block_sampling
    prev_skip = cfg.profiler_enable_load_store_skipping
    cfg.profiler_enable_block_sampling = False
    cfg.profiler_enable_load_store_skipping = False
    yield
    cfg.profiler_enable_block_sampling = prev_block_sampling
    cfg.profiler_enable_load_store_skipping = prev_skip


def _run_profiler_load_store():
    profiler = Profiler()
    traced = tilelens.trace(profiler)(_profiler_load_store)
    x = torch.ones((16,), dtype=torch.float32)
    out = torch.zeros_like(x)
    traced[(2,)](x, out)
    return profiler.load_mask_total_count, profiler.store_mask_total_count


def test_profiler_counts_serial(one_sm, _no_profiler_sampling):
    """Profiler load/store mask counts should be stable when blocks run sequentially."""
    load_total, store_total = _run_profiler_load_store()
    assert load_total == 16
    assert store_total == 16


def test_profiler_counts_concurrent(two_sms, _no_profiler_sampling):
    """Profiler load/store mask counts should match concurrent execution."""
    load_total, store_total = _run_profiler_load_store()
    assert load_total == 16
    assert store_total == 16
