import inspect

import pytest
import torch

import triton
import triton.language as tl

import tilelens
from tilelens.clients import RaceType
from tilelens.clients.race_detector.race_detector import SymbolicRaceDetector
from tilelens.clients.race_detector.data import AccessEventRecord, RaceRecord
from tilelens.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)
from tilelens.clients.symbolic_engine import SymbolicExpr
import tilelens.core.trace as trace_module
from tilelens.core.config import config as cfg
from tilelens.core.data import Load, Store


@pytest.fixture
def _isolate_race_detector_cfg():
    saved = cfg.enable_race_detector
    cfg.enable_race_detector = True
    yield
    cfg.enable_race_detector = saved


@triton.jit
def _basic_kernel(x_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs)
    tl.store(x_ptr + offs, v + 1)


@triton.jit
def _load_store_loop_kernel(x_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    for _i in tl.range(0, 4):
        v = tl.load(x_ptr + offs)
        tl.store(x_ptr + offs, v + 1)


@triton.jit
def _loop_dedup_kernel(x_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for _i in tl.range(0, 4):
        acc += tl.load(x_ptr + offs)
    tl.store(x_ptr + offs, acc)


@triton.jit
def _loop_premises_kernel(x_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    for _i in tl.range(0, 4):
        v = tl.load(x_ptr + offs)
        tl.store(x_ptr + offs, v)


@triton.jit
def _dispatch_kernel(x_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(x_ptr + offs, tl.load(x_ptr + offs))


def test_basic_capture(_isolate_race_detector_cfg):
    detector = SymbolicRaceDetector()
    traced = tilelens.trace(client=detector)(_basic_kernel)

    x = torch.zeros(16, dtype=torch.float32)
    traced[(1,)](x, BLOCK=16)

    modes = [r.access_mode for r in detector.records]
    op_types = [r.op_type for r in detector.records]
    assert modes == ["read", "write"], f"expected [read, write], got {modes}"
    assert op_types == [Load, Store], f"expected [Load, Store], got {op_types}"

    for record in detector.records:
        assert record.tensor is not None, "base tensor should be resolvable"
        assert record.tensor_name == "x_ptr"
        assert isinstance(record, AccessEventRecord)
    # Each lane rewrites its own element: no race.
    assert detector.last_status == "ok" and detector.last_reports == []


def test_loop_load_and_store_same_addr_not_merged(_isolate_race_detector_cfg):
    detector = SymbolicRaceDetector()
    traced = tilelens.trace(client=detector)(_load_store_loop_kernel)

    x = torch.zeros(16, dtype=torch.float32)
    traced[(1,)](x, BLOCK=16)

    modes = sorted(r.access_mode for r in detector.records)
    assert modes == [
        "read",
        "write",
    ], f"load+store at same addr should stay as two events; got {modes}"


def test_loop_repeated_access_deduped(_isolate_race_detector_cfg):
    detector = SymbolicRaceDetector()
    traced = tilelens.trace(client=detector)(_loop_dedup_kernel)

    x = torch.zeros(16, dtype=torch.float32)
    traced[(1,)](x, BLOCK=16)

    reads = [r for r in detector.records if r.access_mode == "read"]
    assert (
        len(reads) == 1
    ), f"loop-body load should dedupe to a single event, got {len(reads)}"


def test_loop_event_premises_include_iterator(_isolate_race_detector_cfg):
    detector = SymbolicRaceDetector()
    traced = tilelens.trace(client=detector)(_loop_premises_kernel)

    x = torch.zeros(16, dtype=torch.float32)
    traced[(1,)](x, BLOCK=16)

    assert detector.records
    for record in detector.records:
        premise_strs = " ".join(str(p) for p in record.premises)
        assert "loop_i_" in premise_strs, (
            f"in-loop event premises must carry a loop iterator constraint; "
            f"got premises: {premise_strs}"
        )


def test_string_dispatch_and_manager_lookup(_isolate_race_detector_cfg):
    traced = tilelens.trace("race_detector")(_dispatch_kernel)

    x = torch.zeros(8, dtype=torch.float32)
    traced[(1,)](x, BLOCK=8)

    rd = traced.client_manager.clients["race_detector"]
    assert isinstance(rd, SymbolicRaceDetector)
    assert len(rd.records) == 2


def test_flag_off_does_not_swallow_explicit_instance():
    """An explicitly constructed detector instance reflects a deliberate user
    choice and must keep tracing regardless of the global flag."""
    saved = cfg.enable_race_detector
    try:
        cfg.enable_race_detector = False
        detector = SymbolicRaceDetector()
        traced = tilelens.trace(client=detector)(_dispatch_kernel)

        from tilelens.core.trace import TritonTrace

        assert isinstance(
            traced, TritonTrace
        ), "explicit instance must be traced even when the env flag is off"

        x = torch.zeros(8, dtype=torch.float32)
        traced[(1,)](x, BLOCK=8)
        assert len(detector.records) == 2
    finally:
        cfg.enable_race_detector = saved


def test_repeat_launches_are_consistent(_isolate_race_detector_cfg):
    """Launching the same traced kernel twice must produce deterministic,
    identical block-execution counts. The previous grid-enumeration cache
    could skip block 0 on a cache hit and then run block 1 instead —
    different blocks run, different side effects, events captured from a
    non-deterministic representative. This regression asserts every launch
    behaves the same."""
    detector = SymbolicRaceDetector()
    traced = tilelens.trace(client=detector)(_dispatch_kernel)

    x1 = torch.zeros(8, dtype=torch.float32)
    traced[(2,)](x1, BLOCK=8)
    after_first = list(detector.records)

    detector.records.clear()
    x2 = torch.zeros(8, dtype=torch.float32)
    traced[(2,)](x2, BLOCK=8)
    after_second = list(detector.records)

    assert len(after_first) == len(after_second), (
        f"repeat launches must capture the same number of events; "
        f"got {len(after_first)} then {len(after_second)}"
    )
    assert [r.access_mode for r in after_first] == [r.access_mode for r in after_second]
    assert [r.source_location for r in after_first] == [
        r.source_location for r in after_second
    ]


# ======== Verdicts ========


def _run(kernel, grid, *args):
    detector = SymbolicRaceDetector()
    tilelens.trace(client=detector)(kernel)[grid](*args)
    assert all(getattr(SymbolicExpr, slot) is None for slot in SymbolicExpr._HOOK_SLOTS)
    return detector


def _line(kernel, needle: str) -> int:
    lines, start = inspect.getsourcelines(kernel.fn)
    return start + next(i for i, line in enumerate(lines) if needle in line)


def _race(detector, race_type, kernel, first: str, second: str) -> RaceRecord:
    assert detector.last_status == "ok"
    assert trace_module.launches[-1].records == detector.last_reports
    (race,) = detector.last_reports
    assert race.race_type == race_type
    assert race.first.source_location[1] == _line(kernel, first)
    assert race.second.source_location[1] == _line(kernel, second)
    return race


def _unsupported(detector, kind: str) -> None:
    assert (detector.last_status, detector.last_reports) == ("unsupported", [])
    (verdict,) = trace_module.launches[-1].records
    assert (verdict.status, verdict.refusal.kind) == ("unsupported", kind)


def _ints(n: int) -> torch.Tensor:
    return torch.zeros(n, dtype=torch.int32)


@triton.jit
def _waw_kernel(out, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * (BLOCK - 1) + tl.arange(0, BLOCK)  # overlaps by 1
    tl.store(out + offs, 1)


@triton.jit
def _raw_kernel(data, out):
    pid = tl.program_id(0)
    tl.store(data, 1, mask=pid == 0)
    tl.store(out + pid, tl.load(data, mask=pid == 1, other=0))


@triton.jit
def _war_kernel(data, out):
    pid = tl.program_id(0)
    tl.store(out + pid, tl.load(data, mask=pid == 1, other=0))
    tl.store(data, 1, mask=pid == 0)


@triton.jit
def _atomic_vs_plain_kernel(counter, out):
    pid = tl.program_id(0)
    tl.atomic_add(counter, 1)
    tl.store(out + pid, tl.load(counter))


def test_race_types(capsys):
    detector = _run(_waw_kernel, (2,), _ints(16), 8)
    race = _race(detector, RaceType.WAW, _waw_kernel, "tl.store", "tl.store")
    assert race.first.writes and race.second.writes
    # Block 0's last element is block 1's first.
    assert {race.programs[0], race.programs[1]} == {(0, 0, 0), (1, 0, 0)}
    assert race.first.element == race.second.element == 7
    detector = _run(_raw_kernel, (2,), _ints(1), _ints(2))
    race = _race(detector, RaceType.RAW, _raw_kernel, "tl.store(data", "tl.load")
    assert race.first.writes and not race.second.writes
    detector = _run(_war_kernel, (2,), _ints(1), _ints(2))
    race = _race(detector, RaceType.WAR, _war_kernel, "tl.load", "tl.store(data")
    assert not race.first.writes and race.second.writes
    detector = _run(_atomic_vs_plain_kernel, (2,), _ints(1), _ints(2))
    kernel = _atomic_vs_plain_kernel
    race = _race(detector, RaceType.RAW, kernel, "tl.atomic_add", "tl.load")
    assert (race.first.op, race.second.op) == ("atomic_rmw", "load")
    out = capsys.readouterr().out
    assert out.count("Data Race Detected") == 4
    assert "store (write) of out[7], program (1, 0, 0)" in out


@triton.jit
def _flag_mask_kernel(flag, out):
    tl.store(out, tl.program_id(0), mask=tl.load(flag) == 1)


@triton.jit
def _other_mask_kernel(flag, out):
    pid = tl.program_id(0)
    v = tl.load(flag + pid, mask=pid == 0, other=42)  # other=0 is dropped
    tl.store(out, pid, mask=v == 42)


def test_loaded_values_drive_masks():
    # The load-value provider gives each program the snapshot's value.
    for value, racy in ((1, True), (0, False)):
        detector = _run(_flag_mask_kernel, (2,), _ints(1) + value, _ints(1))
        assert detector.last_status == "ok" and bool(detector.last_reports) == racy
    detector = _run(_other_mask_kernel, (3,), _ints(1), _ints(1))
    race = _race(detector, RaceType.WAW, _other_mask_kernel, "tl.store", "tl.store")
    assert {race.programs[0][0], race.programs[1][0]} == {1, 2}


@triton.jit
def _disjoint_kernel(x, out, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out + offs, tl.load(x + offs) + 1)


@triton.jit
def _atomic_only_kernel(counter):
    tl.atomic_add(counter, 1)


@pytest.mark.parametrize(
    "kernel, args",
    [(_disjoint_kernel, (_ints(32), _ints(32), 8)), (_atomic_only_kernel, (_ints(1),))],
)
def test_race_free_proof(kernel, args):
    detector = _run(kernel, (4,), *args)
    assert detector.last_status == "ok"
    assert trace_module.launches[-1].records == detector.last_reports == []


@triton.jit
def _old_value_in_address_kernel(head, buf):
    tl.store(buf + tl.atomic_add(head, 1, sem="relaxed"), 1)


@triton.jit
def _cas_old_value_in_address_kernel(head, buf):
    tl.store(buf + tl.atomic_cas(head, 0, 1), 1)


@triton.jit
def _winner_kernel(flag, out, WINNERS: tl.constexpr):
    old = tl.atomic_add(flag, 1, sem="relaxed")
    tl.store(out, 1, mask=old < WINNERS)


@triton.jit
def _cas_guard_kernel(flag, data, out, SEM: tl.constexpr):
    pid = tl.program_id(0)
    tl.store(data, 1, mask=pid == 0)
    tl.debug_barrier()
    old = tl.atomic_cas(flag, tl.where(pid == 0, 0, 1), 1, sem=SEM)
    tl.debug_barrier()
    tl.store(out + pid, tl.load(data, mask=(pid == 1) & (old == 1), other=0))


def test_atomic_old_values():
    # An old value in an address is refused; in a mask it is modeled.
    for kernel in (_old_value_in_address_kernel, _cas_old_value_in_address_kernel):
        with pytest.warns(RuntimeWarning, match="data-dependent memory address"):
            detector = _run(kernel, (4,), _ints(1), _ints(4))
        _unsupported(detector, "unsupported")
    for winners, racy in ((1, False), (2, True)):
        detector = _run(_winner_kernel, (4,), _ints(1), _ints(1), winners)
        assert detector.last_status == "ok" and bool(detector.last_reports) == racy
    for sem, racy in (("acq_rel", False), ("relaxed", True)):
        detector = _run(_cas_guard_kernel, (2,), _ints(1), _ints(1), _ints(2), sem)
        assert detector.last_status == "ok" and bool(detector.last_reports) == racy


@triton.jit
def _reversed_reload_kernel(x, y, BLOCK: tl.constexpr, FENCE: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(x + offs, offs)
    if FENCE:
        tl.debug_barrier()
    tl.store(y + offs, tl.load(x + (BLOCK - 1 - offs)))


def test_debug_barrier_orders_one_program():
    kernel = _reversed_reload_kernel
    detector = _run(kernel, (1,), _ints(8), _ints(8), 8, False)
    race = _race(detector, RaceType.RAW, kernel, "tl.store(x", "tl.store(y")
    assert race.programs == ((0, 0, 0), (0, 0, 0))
    assert "Missing source fence" in race.reason
    detector = _run(kernel, (1,), _ints(8), _ints(8), 8, True)
    assert detector.last_status == "ok" and detector.last_reports == []


@triton.jit
def _barrier_loop_kernel(x, y, BLOCK: tl.constexpr, UNROLL: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    if UNROLL:
        tl.store(x + offs, 0)
        tl.debug_barrier()
        tl.store(y + offs, tl.load(x + (BLOCK - 1 - offs)))
        tl.store(x + offs, 1)
        tl.debug_barrier()
        tl.store(y + offs, tl.load(x + (BLOCK - 1 - offs)))
    else:
        for i in range(2):
            tl.store(x + offs, i)
            tl.debug_barrier()
            tl.store(y + offs, tl.load(x + (BLOCK - 1 - offs)))


@triton.jit
def _first_iteration_barrier_kernel(x, y, BLOCK: tl.constexpr, UNROLL: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    for i in range(2):
        tl.store(x + offs, i)
        if i == 0:
            tl.debug_barrier()
    tl.store(y + offs, tl.load(x + (BLOCK - 1 - offs)))


@triton.jit
def _back_edge_barrier_kernel(x, y, BLOCK: tl.constexpr, UNROLL: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    for i in range(2):
        tl.store(x + offs, i)
        tl.debug_barrier()
        tl.store(y + offs, tl.load(x + (BLOCK - 1 - offs)))
        if UNROLL:  # here: the barrier on the back edge
            tl.debug_barrier()


@triton.jit
def _butterfly_kernel(x, y, BLOCK: tl.constexpr, UNROLL: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(x + offs)
    for _ in range(2):
        for stage in tl.static_range(2):  # one store per stage, same address
            tl.store(y + offs, v)
            tl.debug_barrier()
            v += tl.load(y + (offs ^ (1 << stage)))
            tl.debug_barrier()
    tl.store(x + offs, v)


@pytest.mark.parametrize(
    "kernel, unroll, racy",
    [
        (_barrier_loop_kernel, True, True),
        (_barrier_loop_kernel, False, True),
        (_first_iteration_barrier_kernel, False, True),
        (_back_edge_barrier_kernel, True, False),
        (_butterfly_kernel, False, False),
    ],
)
def test_debug_barrier_in_a_loop(kernel, unroll, racy):
    # A loop's barrier must separate every iteration's accesses: without
    # one on the back edge it misses one iteration's load against the next
    # one's store, and one run in the first iteration only misses the last
    # store against the load after the loop.
    detector = _run(kernel, (1,), _ints(8), _ints(8), 8, unroll)
    assert detector.last_status == "ok" and bool(detector.last_reports) == racy


@triton.jit
def _own_element_kernel(x, M: tl.constexpr, N: tl.constexpr, SHIFT: tl.constexpr):
    p = tl.make_block_ptr(x, (M, N), (N, 1), (0, 0), (M, N), (1, 0))
    v = tl.load(p)
    if SHIFT:  # each row stores the next row's elements
        p = tl.advance(p, (1, 0))
    tl.store(p, v + 1, boundary_check=(0,))


def test_load_and_store_of_its_value_are_dependency_ordered():
    # Each block-pointer access has its own aranges: the same element is
    # still the same position, so only the shifted store races.
    for shift, racy in ((False, False), (True, True)):
        detector = _run(_own_element_kernel, (1,), _ints(32), 4, 8, shift)
        assert detector.last_status == "ok" and bool(detector.last_reports) == racy


@triton.jit
def _published_old_value_kernel(z, flag, B: tl.constexpr, SEM: tl.constexpr):
    # Program 0 lane p publishes flag[p] = z[p]'s old value + 100; program
    # 1 lane q stores z[q] once it acquires flag[q]. One arange for every
    # access: lane q's atomic is ordered before the publication of flag[q].
    pid = tl.program_id(0)
    offs = tl.arange(0, B)
    old = tl.atomic_add(z + offs, 1, mask=pid == 0, sem="relaxed")
    tl.atomic_xchg(flag + offs, old + 100, mask=pid == 0, sem=SEM)
    f = tl.atomic_add(flag + offs, 0, mask=pid == 1, sem="acquire")
    tl.store(z + offs, 5, mask=(pid == 1) & (f >= 100))


@triton.jit
def _published_reversed_old_value_kernel(z, flag, B: tl.constexpr):
    # As above, but lane p publishes z[B - 1 - p]'s old value, each access
    # through its own arange: the atomic on z[q] is lane B - 1 - q's, which
    # nothing orders before program 1's store to z[q].
    pid = tl.program_id(0)
    o1 = tl.arange(0, B)
    old = tl.atomic_add(z + (B - 1 - o1), 1, mask=pid == 0, sem="relaxed")
    o2 = tl.arange(0, B)
    tl.atomic_xchg(flag + o2, old + 100, mask=pid == 0, sem="release")
    o3 = tl.arange(0, B)
    f = tl.atomic_add(flag + o3, 0, mask=pid == 1, sem="acquire")
    o4 = tl.arange(0, B)
    tl.store(z + o4, 5, mask=(pid == 1) & (f >= 100))


@triton.jit
def _published_transposed_old_value_kernel(z, flag, B: tl.constexpr):
    # Position (i, j) publishes flag[j, i]: flag[i, j] follows the atomic
    # on z[j, i], not on z[i, j].
    pid = tl.program_id(0)
    rows = tl.arange(0, B)
    cols = tl.arange(0, B)  # its own line: its own arange variable
    offs = rows[:, None] * B + cols[None, :]
    old = tl.atomic_add(z + offs, 1, mask=pid == 0, sem="relaxed")
    tl.atomic_xchg(tl.trans(flag + offs), old + 100, mask=pid == 0, sem="release")
    f = tl.atomic_add(flag + offs, 0, mask=pid == 1, sem="acquire")
    tl.store(z + offs, 5, mask=(pid == 1) & (f >= 100))


@triton.jit
def _published_one_address_old_value_kernel(z, flag, B: tl.constexpr):
    # B lanes without an arange, all on z[0] and flag[0]: program 1
    # acquires one lane's publication, which orders only that lane's
    # atomic on z[0] before its store.
    pid = tl.program_id(0)
    zero = tl.zeros([B], dtype=tl.int32)
    old = tl.atomic_add(z + zero, 1, mask=pid == 0, sem="relaxed")
    tl.atomic_xchg(flag + zero, old + 100, mask=pid == 0, sem="release")
    f = tl.atomic_add(flag, 0, mask=pid == 1, sem="acquire")
    tl.store(z, 5, mask=(pid == 1) & (f >= 100))


@triton.jit
def _published_scalar_old_value_kernel(z, flag):
    pid = tl.program_id(0)
    old = tl.atomic_add(z, 1, mask=pid == 0, sem="relaxed")
    tl.atomic_xchg(flag, old + 100, mask=pid == 0, sem="release")
    f = tl.atomic_add(flag, 0, mask=pid == 1, sem="acquire")
    tl.store(z, 5, mask=(pid == 1) & (f >= 100))


def test_dependency_on_a_block_without_positions_gives_no_order():
    # A block anchor with no lane position (lanes on one address) orders no
    # dependent lane after it; a scalar anchor, one lane, still does.
    kernel = _published_one_address_old_value_kernel
    detector = _run(kernel, (2,), _ints(1), _ints(1), 4)
    race = _race(detector, RaceType.WAW, kernel, "tl.atomic_add(z", "tl.store")
    assert race.programs == ((0, 0, 0), (1, 0, 0))
    detector = _run(_published_scalar_old_value_kernel, (2,), _ints(1), _ints(1))
    assert detector.last_status == "ok" and not detector.last_reports


def test_dependency_orders_a_lane_after_its_own_position_only():
    # A dependency orders the anchor's lane at the dependent lane's own
    # position, not every lane of it: with a release publication the
    # shared arange is ordered, a relaxed one is not, and separate aranges
    # that permute the positions (or a transposed pointer) are a race.
    for sem, racy in (("release", False), ("relaxed", True)):
        detector = _run(_published_old_value_kernel, (2,), _ints(4), _ints(4), 4, sem)
        assert detector.last_status == "ok" and bool(detector.last_reports) == racy
    for kernel, n in (
        (_published_reversed_old_value_kernel, 4),
        (_published_transposed_old_value_kernel, 16),
    ):
        detector = _run(kernel, (2,), _ints(n), _ints(n), 4)
        race = _race(detector, RaceType.WAW, kernel, "tl.atomic_add(z", "tl.store")
        assert race.programs == ((0, 0, 0), (1, 0, 0))
        assert race.first.element == race.second.element


@triton.jit
def _pid_branch_kernel(out):
    if tl.program_id(0) == 0:
        tl.store(out, 1)


@triton.jit
def _spin_on_flag_kernel(flag, out):
    while tl.load(flag) == 0:
        pass
    tl.store(out + tl.program_id(0), 1)


@triton.jit
def _block_and_kernel(out):
    offs = tl.arange(0, 8)
    tl.store(out + offs, 1, mask=(offs < 2) and (offs < 4))


@triton.jit
def _float_rmw_mask_kernel(acc, out):
    old = tl.atomic_add(acc, 1.0)
    tl.store(out, 1, mask=old > 0.5)


@triton.jit
def _pid_loop_bound_kernel(out):
    for i in range(tl.program_id(0)):
        tl.store(out + i, 1)


@triton.jit
def _inline_asm_kernel(out):
    offs = tl.arange(0, 8)
    y = tl.inline_asm_elementwise(
        "mov.b32 $0, $1;", "=r,r", [offs], dtype=tl.int32, is_pure=True, pack=1
    )
    tl.store(out + y, 1)


@pytest.mark.parametrize(
    "kernel, args, kind, reason",
    [
        (_pid_branch_kernel, (_ints(1),), "unsupported", "host-side control"),
        (_spin_on_flag_kernel, (_ints(1) + 1, _ints(2)), "unsupported", "host-side"),
        (
            _float_rmw_mask_kernel,
            (torch.zeros(1), _ints(1)),
            "unsupported",
            "atomic_rmw return",
        ),
        (_pid_loop_bound_kernel, (_ints(2),), "unsupported", "loop bound"),
        (_block_and_kernel, (_ints(8),), "block-truthiness", "triton/pull/12150"),
        (_inline_asm_kernel, (_ints(8),), "inline-asm", "inline_asm"),
    ],
)
def test_unsupported_launch(kernel, args, kind, reason):
    with pytest.warns(RuntimeWarning, match=r"\[RaceDetector\] launch unsupported"):
        detector = _run(kernel, (2,), *args)
    _unsupported(detector, kind)
    assert reason in detector.unsupported_reason


def test_solver_timeout_is_unsupported(monkeypatch):
    monkeypatch.setattr(TwoCopySymbolicHBSolver, "TIMEOUT_MS", 1)

    @triton.jit
    def nonlinear_kernel(x, n):
        pid = tl.program_id(0)
        offs = tl.arange(0, 8)
        tl.store(x + (pid * pid * 7919 + offs * offs * 31 + pid * offs) % n, 1)

    with pytest.warns(RuntimeWarning, match="timeout"):
        detector = _run(nonlinear_kernel, (4,), _ints(64), 61)
    _unsupported(detector, "timeout")


@triton.jit
def _raising_kernel(out):
    tl.store(out, 1)
    1 // 0


@triton.jit
def _cas_spin_lock_kernel(lock, out):
    while tl.atomic_cas(lock, 0, 1) == 1:
        pass
    tl.store(out, 1)


def test_abort_launch_reclaims_hook_slots():
    # A launch marked unsupported before the engine raised stays unsupported.
    detector = SymbolicRaceDetector()
    with pytest.raises(NotImplementedError, match="atomic_cas"):
        tilelens.trace(client=detector)(_cas_spin_lock_kernel)[(2,)](_ints(1), _ints(1))
    assert detector.last_status == "unsupported"
    assert "host-side control flow" in detector.unsupported_reason
    assert all(getattr(SymbolicExpr, slot) is None for slot in SymbolicExpr._HOOK_SLOTS)
    with pytest.raises(ZeroDivisionError):
        tilelens.trace(client=detector)(_raising_kernel)[(1,)](_ints(1))
    assert (detector.last_status, detector.solver) == ("aborted", None)
    assert detector.unsupported_reason is None
    assert all(getattr(SymbolicExpr, slot) is None for slot in SymbolicExpr._HOOK_SLOTS)


def test_race_survives_save_and_load(tmp_path, monkeypatch):
    detector = _run(_waw_kernel, (2,), _ints(16), 8)
    monkeypatch.setattr(trace_module, "launches", trace_module.launches[-1:])
    tilelens.load(tilelens.save(tmp_path / "trace.zip"))
    assert trace_module.launches[-1].records == detector.last_reports
    assert trace_module.launches[-1].records[0].race_type == RaceType.WAW
