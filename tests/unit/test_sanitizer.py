import numpy as np
import pytest
import torch
import triton
import triton.language as tl
from typing import cast
from z3 import Int

import tilelens
from tilelens.core.callbacks import ForLoopCallbacks, OpCallbacks
from tilelens.core.client import Client, ClientManager
from tilelens.core.config import config as cfg
from tilelens.core.data import Load, Store
from tilelens.core.symbolic_metadata import (
    INT1,
    INT32,
    SymbolicTensorValue,
    TensorDescriptorAccess,
    pointer_type,
    type_spec,
)
from tilelens.clients import Sanitizer
from tilelens.clients.sanitizer.report import _classify_layout_and_segments
from tilelens.clients.sanitizer.sanitizer import (
    NullSanitizer,
    SymbolicSanitizer,
    _fn_symbolic_cache_set,
)
from tilelens.clients.sanitizer.range_summary import IntRange, access_interval_summary
from tilelens.clients.symbolic_engine import LoopContext, PendingCheck, SymbolicExpr
from tilelens.clients.symbolic_engine import symbolic_tensor_descriptor_access
from tilelens.core.patch import LoopSite, loop_file_token


# ======== Init Tests ===========


@pytest.fixture
def _isolate_sanitizer_cfg():
    """Save and restore cfg.enable_sanitizer around every test."""
    saved = cfg.enable_sanitizer
    yield
    cfg.enable_sanitizer = saved


def test_sanitizer_init(_isolate_sanitizer_cfg):
    cfg.enable_sanitizer = True
    assert isinstance(Sanitizer(), SymbolicSanitizer)

    cfg.enable_sanitizer = False
    assert isinstance(Sanitizer(), NullSanitizer)


# ======== Launch-lifecycle regression tests ===========
#
# These guard the two spots that break most easily when SymbolicClient's
# launch-lifecycle machinery is reshuffled:
#   1. the cross-launch fn cache (sanitizer's pre_run_callback short-circuit)
#   2. the post_run_callback guard that must NOT wipe launch state mid-
#      full-grid enumeration.


@pytest.fixture
def _isolate_fn_symbolic_cache():
    """Reset the cross-launch sanitizer fn-cache around each test."""
    saved = set(_fn_symbolic_cache_set)
    _fn_symbolic_cache_set.clear()
    yield _fn_symbolic_cache_set
    _fn_symbolic_cache_set.clear()
    _fn_symbolic_cache_set.update(saved)


def _populate_for_launch(sanitizer: SymbolicSanitizer, grid: tuple[int, ...]) -> None:
    """Prime a sanitizer as if ``arg_callback`` + ``grid_callback`` had fired.

    Uses a fake cache_args payload instead of real tensors so the test stays
    focused on pre_run_callback's cache-hit logic.
    """
    sanitizer.cache_args = [("fake_shape",)]
    sanitizer.cache_grid = grid


def test_pre_run_callback_cache_hit_on_replay(_isolate_fn_symbolic_cache):
    """Identical (fn, grid, args) launched twice: 1st runs, 2nd is cached-skip."""

    def kernel():
        pass

    first = SymbolicSanitizer(abort_on_error=False)
    _populate_for_launch(first, (2, 1, 1))
    assert first.pre_run_callback(kernel) is True
    # pre_run_callback clears the cache after hashing, matching production flow.
    assert first.cache_grid is None and first.cache_args == []
    assert len(_isolate_fn_symbolic_cache) == 1

    second = SymbolicSanitizer(abort_on_error=False)
    _populate_for_launch(second, (2, 1, 1))
    assert second.pre_run_callback(kernel) is False
    assert len(_isolate_fn_symbolic_cache) == 1


def test_pre_run_callback_cache_miss_on_grid_change(_isolate_fn_symbolic_cache):
    """Changing the grid must NOT trigger a cache hit — it's part of the key."""

    def kernel():
        pass

    a = SymbolicSanitizer(abort_on_error=False)
    _populate_for_launch(a, (2, 1, 1))
    assert a.pre_run_callback(kernel) is True

    b = SymbolicSanitizer(abort_on_error=False)
    _populate_for_launch(b, (4, 1, 1))
    assert b.pre_run_callback(kernel) is True
    assert len(_isolate_fn_symbolic_cache) == 2


def test_vetoed_block_does_not_count_as_active_sanitizer_block():
    class VetoClient(Client):
        NAME = "veto"

        def pre_run_callback(self, fn):
            return False

        def post_run_callback(self, fn):
            return True

        def arg_callback(self, name, arg, arg_cvt):
            return None

        def grid_callback(self, grid):
            return None

        def grid_idx_callback(self, grid_idx):
            return None

        def register_op_callback(self, op_type, *args, **kwargs):
            return OpCallbacks()

        def register_for_loop_callback(self):
            return ForLoopCallbacks()

        def finalize(self):
            return []

        def pre_warmup_callback(self, jit_fn, *args, **kwargs):
            return False

        def post_warmup_callback(self, jit_fn, ret):
            return None

    sanitizer = SymbolicSanitizer(abort_on_error=False)
    manager = ClientManager([sanitizer, VetoClient()])
    manager.grid_callback((1, 1, 1))
    manager.grid_idx_callback((0, 0, 0))

    assert manager.pre_run_callback(lambda: None) is False
    assert sanitizer._active_blocks == 0


@triton.jit
def _masked_copy_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)


def test_fn_cache_skipped_launch_drops_its_state(_isolate_fn_symbolic_cache):
    """No block of the cache-skipped repeat reaches post_run_callback; its
    tensors must still go, or x's full range hides the x[:64] OOB read."""
    sanitizer = SymbolicSanitizer(abort_on_error=False)
    kernel = tilelens.trace(client=sanitizer)(_masked_copy_kernel)
    x, y = torch.arange(128, dtype=torch.float32), torch.empty(128)
    kernel[(1,)](x, y, 128, BLOCK=128)
    kernel[(1,)](x, y, 128, BLOCK=128)
    assert sanitizer.tensors == []
    kernel[(1,)](x[:64], y, 128, BLOCK=128)
    assert [record.op_type for record in sanitizer.records] == [Load]


def test_post_run_callback_preserves_state_mid_full_grid():
    """When need_full_grid=True and we're NOT on the last block, launch state
    (tensors / tensor_addrs / tensor_names / cache_*) must survive
    post_run_callback so later blocks still see registered tensors."""
    sanitizer = SymbolicSanitizer(abort_on_error=False)

    # Simulate state populated during arg_callback + grid_callback.
    sentinel_tensor = object()
    sanitizer.tensors.append(cast(object, sentinel_tensor))  # type: ignore[arg-type]
    sanitizer.tensor_addrs.append((0, 0, cast(object, sentinel_tensor)))  # type: ignore[arg-type]
    sanitizer.tensor_names[id(sentinel_tensor)] = {"X"}
    sanitizer.cache_args = [("fake",)]
    sanitizer.cache_grid = (4, 1, 1)

    sanitizer.last_grid = (3, 0, 0)  # last block is (3,0,0) in a 4-block grid
    sanitizer.grid_idx = (1, 0, 0)  # we're mid-enumeration
    sanitizer.need_full_grid = True

    ret = sanitizer.post_run_callback(lambda: None)

    assert ret is True
    assert sanitizer.tensors == [sentinel_tensor]
    assert sanitizer.tensor_addrs == [(0, 0, sentinel_tensor)]
    assert sanitizer.tensor_names == {id(sentinel_tensor): {"X"}}
    assert sanitizer.cache_args == [("fake",)]
    assert sanitizer.cache_grid == (4, 1, 1)


def test_post_run_callback_clears_state_on_last_block():
    """On the final block we DO want launch state cleared."""
    sanitizer = SymbolicSanitizer(abort_on_error=False)

    sentinel_tensor = object()
    sanitizer.tensors.append(cast(object, sentinel_tensor))  # type: ignore[arg-type]
    sanitizer.tensor_addrs.append((0, 0, cast(object, sentinel_tensor)))  # type: ignore[arg-type]
    sanitizer.tensor_names[id(sentinel_tensor)] = {"X"}
    sanitizer.cache_args = [("fake",)]
    sanitizer.cache_grid = (4, 1, 1)

    sanitizer.last_grid = (3, 0, 0)
    sanitizer.grid_idx = (3, 0, 0)  # final block
    sanitizer.need_full_grid = True

    sanitizer.post_run_callback(lambda: None)

    assert sanitizer.tensors == []
    assert sanitizer.tensor_addrs == []
    assert sanitizer.tensor_names == {}
    assert sanitizer.cache_args == []
    assert sanitizer.cache_grid is None


def _make_concrete_ptr_and_mask(
    addrs: list[int],
    mask_values: list[bool],
) -> tuple[SymbolicExpr, SymbolicExpr]:
    ptr_dtype = pointer_type(INT32)
    ptr_sym = SymbolicExpr.create(
        "const",
        SymbolicTensorValue(np.asarray(addrs, dtype=np.uint64), ptr_dtype),
        type_spec(ptr_dtype, (len(addrs),)),
    )
    mask_sym = SymbolicExpr.create(
        "const",
        SymbolicTensorValue(np.asarray(mask_values, dtype=bool), INT1),
        type_spec(INT1, (len(mask_values),)),
    )
    return ptr_sym, mask_sym


def test_concrete_ptr_access_respects_masked_lanes_for_load_and_store():
    sanitizer = SymbolicSanitizer(abort_on_error=False)
    tensor = torch.empty(4, dtype=torch.int32)
    base = tensor.data_ptr()
    end = base + tensor.numel() * tensor.element_size() - 1
    sanitizer.tensors.append(tensor)
    sanitizer.tensor_addrs.append((base, end, tensor))

    ptr_sym, inactive_oob_mask = _make_concrete_ptr_and_mask(
        [base, end + tensor.element_size()],
        [True, False],
    )

    def fail_symbolic_path(*_args, **_kwargs):
        raise AssertionError("concrete pointer access should not hit symbolic path")

    sanitizer._handle_access_check = fail_symbolic_path  # type: ignore[method-assign]

    sanitizer._op_store_overrider(ptr_sym, 0, inactive_oob_mask)
    sanitizer._op_load_overrider(ptr_sym, inactive_oob_mask, None)
    assert sanitizer.records == []

    _ptr_sym, active_oob_mask = _make_concrete_ptr_and_mask(
        [base, end + tensor.element_size()],
        [True, True],
    )

    sanitizer._op_store_overrider(ptr_sym, 0, active_oob_mask)
    assert len(sanitizer.records) == 1
    assert sanitizer.records[0].op_type is Store
    assert sanitizer.records[0].violation_address == end + tensor.element_size()

    sanitizer.records.clear()
    sanitizer._op_load_overrider(ptr_sym, active_oob_mask, None)
    assert len(sanitizer.records) == 1
    assert sanitizer.records[0].op_type is Load
    assert sanitizer.records[0].violation_address == end + tensor.element_size()


def test_concrete_ptr_router_uses_symbolic_path_inside_loops():
    sanitizer = SymbolicSanitizer(abort_on_error=False)
    tensor = torch.empty(4, dtype=torch.int32)
    base = tensor.data_ptr()
    end = base + tensor.numel() * tensor.element_size() - 1
    sanitizer.tensors.append(tensor)
    sanitizer.tensor_addrs.append((base, end, tensor))
    ptr_sym, mask_sym = _make_concrete_ptr_and_mask([base], [True])

    calls = []

    def record_symbolic_path(expr, op_type, access_mode):
        calls.append((expr.op, op_type, access_mode))

    sanitizer._handle_access_check = record_symbolic_path  # type: ignore[method-assign]
    sanitizer.loop_stack.append(object())  # type: ignore[arg-type]

    sanitizer._op_load_overrider(ptr_sym, mask_sym, None)

    assert calls == [("load", Load, "read")]
    assert sanitizer.records == []


class _HostTensorDescriptor:
    def __init__(self, base, shape, strides, block_shape):
        self.base = base
        self.shape = shape
        self.strides = strides
        self.block_shape = block_shape


def test_symbolic_tensor_descriptor_access_reports_block_oob():
    sanitizer = SymbolicSanitizer(abort_on_error=False)
    tensor = torch.empty((4, 4), dtype=torch.int32)
    desc = _HostTensorDescriptor(
        tensor,
        shape=(4, 4),
        strides=(4, 1),
        block_shape=(2, 2),
    )
    sanitizer.arg_callback("input_desc", desc, None)
    sanitizer.grid_callback((1, 1, 1))

    ptr = symbolic_tensor_descriptor_access(desc, (3, 0))
    assert ptr.op == "descriptor_access"
    assert not ptr.has_op("addptr")
    sanitizer._op_load_overrider(ptr, None, None)

    assert len(sanitizer.records) == 1
    assert sanitizer.records[0].op_type is Load
    assert sanitizer.records[0].tensor is tensor
    assert sanitizer.records[0].violation_address == (
        tensor.data_ptr() + 12 * tensor.element_size()
    )


def test_tensor_descriptor_access_record_reports_block_oob():
    sanitizer = SymbolicSanitizer(abort_on_error=False)
    tensor = torch.empty((4, 4), dtype=torch.int32)
    desc = _HostTensorDescriptor(
        tensor,
        shape=(4, 4),
        strides=(4, 1),
        block_shape=(2, 2),
    )
    sanitizer.arg_callback("input_desc", desc, None)
    sanitizer.grid_callback((1, 1, 1))

    sanitizer._op_load_overrider(TensorDescriptorAccess(desc, (3, 0)), None, None)

    assert len(sanitizer.records) == 1
    assert sanitizer.records[0].op_type is Load
    assert sanitizer.records[0].tensor is tensor


def test_tensor_descriptor_created_in_kernel_reports_block_oob():
    class _GluonDescriptor:
        pass

    sanitizer = SymbolicSanitizer(abort_on_error=False)
    tensor = torch.empty((4, 4), dtype=torch.int32)
    desc = _GluonDescriptor()
    sanitizer.arg_callback("input_desc", tensor, None)
    sanitizer.grid_callback((1, 1, 1))
    sanitizer._op_allocate_after_callback(
        desc,
        tensor,
        (4, 4),
        (4, 1),
        (2, 2),
    )

    sanitizer._op_load_overrider(TensorDescriptorAccess(desc, (3, 0)), None, None)

    assert len(sanitizer.records) == 1
    assert sanitizer.records[0].op_type is Load
    assert sanitizer.records[0].tensor is tensor


def test_tensor_descriptor_access_record_ignores_inactive_predicate():
    sanitizer = SymbolicSanitizer(abort_on_error=False)
    tensor = torch.empty((4, 4), dtype=torch.int32)
    desc = _HostTensorDescriptor(
        tensor,
        shape=(4, 4),
        strides=(4, 1),
        block_shape=(2, 2),
    )
    sanitizer.arg_callback("input_desc", desc, None)
    sanitizer.grid_callback((1, 1, 1))

    ret = sanitizer._op_load_overrider(
        TensorDescriptorAccess(desc, (3, 0), pred=False),
        False,
        None,
    )

    assert ret.op == "load"
    assert sanitizer.records == []


def test_symbolic_tensor_descriptor_access_allows_in_bounds_block():
    sanitizer = SymbolicSanitizer(abort_on_error=False)
    tensor = torch.empty((4, 4), dtype=torch.int32)
    desc = _HostTensorDescriptor(
        tensor,
        shape=(4, 4),
        strides=(4, 1),
        block_shape=(2, 2),
    )
    sanitizer.arg_callback("input_desc", desc, None)
    sanitizer.grid_callback((1, 1, 1))

    ptr = symbolic_tensor_descriptor_access(desc, (2, 2))
    sanitizer._op_load_overrider(ptr, None, None)

    assert sanitizer.records == []


def test_concrete_ptr_router_uses_concrete_path_for_large_multi_range_access():
    sanitizer = SymbolicSanitizer(abort_on_error=False)
    tensor = torch.empty(128, dtype=torch.int32)
    base = tensor.data_ptr()
    stride_bytes = tensor.element_size() * 2
    ranges = [
        (base + i * stride_bytes, base + i * stride_bytes + tensor.element_size() - 1)
        for i in range(64)
    ]
    sanitizer.tensors.append(tensor)
    sanitizer.tensor_addrs.extend((start, end, tensor) for start, end in ranges)
    ptr_sym, mask_sym = _make_concrete_ptr_and_mask(
        [start for start, _end in ranges],
        [True] * len(ranges),
    )

    def fail_symbolic_path(*_args, **_kwargs):
        raise AssertionError("large concrete access should not hit symbolic path")

    sanitizer._handle_access_check = fail_symbolic_path  # type: ignore[method-assign]

    sanitizer._op_store_overrider(ptr_sym, 0, mask_sym)

    assert sanitizer.records == []


def _make_loop_affine_access(
    base: int,
    element_size: int,
    start: int,
    stop: int,
    step: int,
) -> tuple[LoopContext, SymbolicExpr]:
    idx_z3 = Int("loop_i_123")
    idx_sym = SymbolicExpr.create("const", 0, INT32)
    ctx = LoopContext(
        loop_site=LoopSite(123, loop_file_token(__file__)),
        length=len(range(start, stop, step)),
        idx=idx_sym,
        idx_z3=idx_z3,
        start=start,
        stop=stop,
        step=step,
    )
    idx_sym.loop_ctx = ctx

    base_sym = SymbolicExpr.create("const", base, pointer_type(INT32))
    scale_sym = SymbolicExpr.create("const", element_size, INT32)
    offset_sym = SymbolicExpr.create("mul", idx_sym, scale_sym)
    ptr_sym = SymbolicExpr.create("add", base_sym, offset_sym)
    access_expr = SymbolicExpr.create("load", ptr_sym, None, None)
    return ctx, access_expr


def test_loop_range_summary_skips_z3_for_proven_safe_access():
    sanitizer = SymbolicSanitizer(abort_on_error=False)
    tensor = torch.empty(4, dtype=torch.int32)
    base = tensor.data_ptr()
    end = base + tensor.numel() * tensor.element_size() - 1
    sanitizer.tensors.append(tensor)
    sanitizer.tensor_addrs.append((base, end, tensor))
    sanitizer.grid = (1, 1, 1)

    ctx, access_expr = _make_loop_affine_access(
        base,
        tensor.element_size(),
        start=0,
        stop=tensor.numel(),
        step=1,
    )
    pending = PendingCheck(access_expr, Int("addr"), None)

    def fail_z3_path(*_args, **_kwargs):
        raise AssertionError("range summary should skip the Z3 check")

    sanitizer._check_range_satisfiable = fail_z3_path  # type: ignore[method-assign]

    sanitizer._process_pending_check(ctx, pending, [])


def test_loop_range_summary_skips_z3_for_addptr_access():
    sanitizer = SymbolicSanitizer(abort_on_error=False)
    tensor = torch.empty(4, dtype=torch.int32)
    base = tensor.data_ptr()
    end = base + tensor.numel() * tensor.element_size() - 1
    sanitizer.tensors.append(tensor)
    sanitizer.tensor_addrs.append((base, end, tensor))
    sanitizer.grid = (1, 1, 1)

    ctx, _access_expr = _make_loop_affine_access(
        base,
        tensor.element_size(),
        start=0,
        stop=tensor.numel(),
        step=1,
    )
    ptr_sym = SymbolicExpr.create(
        "addptr",
        SymbolicExpr.create("const", base, pointer_type(INT32)),
        ctx.idx,
    )
    access_expr = SymbolicExpr.create("load", ptr_sym, None, None)
    pending = PendingCheck(access_expr, Int("addr"), None)

    def fail_z3_path(*_args, **_kwargs):
        raise AssertionError("range summary should skip the Z3 check")

    sanitizer._check_range_satisfiable = fail_z3_path  # type: ignore[method-assign]

    sanitizer._process_pending_check(ctx, pending, [])


def test_loop_range_summary_skips_z3_for_fma_affine_access():
    sanitizer = SymbolicSanitizer(abort_on_error=False)
    tensor = torch.empty(4, dtype=torch.int32)
    base = tensor.data_ptr()
    end = base + tensor.numel() * tensor.element_size() - 1
    sanitizer.tensors.append(tensor)
    sanitizer.tensor_addrs.append((base, end, tensor))
    sanitizer.grid = (1, 1, 1)

    ctx, access_expr = _make_loop_affine_access(
        base,
        tensor.element_size(),
        start=0,
        stop=tensor.numel(),
        step=1,
    )
    idx_sym = ctx.idx
    scale_sym = SymbolicExpr.create("const", tensor.element_size(), INT32)
    base_sym = SymbolicExpr.create("const", base, pointer_type(INT32))
    ptr_sym = SymbolicExpr.create("fma", idx_sym, scale_sym, base_sym)
    access_expr = SymbolicExpr.create("load", ptr_sym, None, None)
    pending = PendingCheck(access_expr, Int("addr"), None)

    def fail_z3_path(*_args, **_kwargs):
        raise AssertionError("range summary should skip the Z3 check")

    sanitizer._check_range_satisfiable = fail_z3_path  # type: ignore[method-assign]

    sanitizer._process_pending_check(ctx, pending, [])


def test_loop_range_summary_falls_back_when_interval_can_escape_tensor():
    sanitizer = SymbolicSanitizer(abort_on_error=False)
    tensor = torch.empty(4, dtype=torch.int32)
    base = tensor.data_ptr()
    end = base + tensor.numel() * tensor.element_size() - 1
    sanitizer.tensors.append(tensor)
    sanitizer.tensor_addrs.append((base, end, tensor))
    sanitizer.grid = (1, 1, 1)

    ctx, access_expr = _make_loop_affine_access(
        base,
        tensor.element_size(),
        start=0,
        stop=tensor.numel() + 1,
        step=1,
    )
    pending = PendingCheck(access_expr, Int("addr"), None)
    calls = []

    def record_z3_path(*args, **_kwargs):
        calls.append(args)

    sanitizer._check_range_satisfiable = record_z3_path  # type: ignore[method-assign]

    sanitizer._process_pending_check(ctx, pending, [])

    assert len(calls) == 1


def test_loop_range_summary_handles_abs_without_sign_flip():
    sanitizer = SymbolicSanitizer(abort_on_error=False)
    tensor = torch.empty(4, dtype=torch.int32)
    base = tensor.data_ptr()
    end = base + tensor.numel() * tensor.element_size() - 1
    sanitizer.tensors.append(tensor)
    sanitizer.tensor_addrs.append((base, end, tensor))
    sanitizer.grid = (1, 1, 1)

    ctx, _access_expr = _make_loop_affine_access(
        base,
        tensor.element_size(),
        start=0,
        stop=tensor.numel(),
        step=1,
    )
    base_sym = SymbolicExpr.create("const", base, pointer_type(INT32))
    negative_offset = SymbolicExpr.create("const", -tensor.element_size(), INT32)
    abs_offset = SymbolicExpr.create("abs", negative_offset)
    ptr_sym = SymbolicExpr.create("sub", base_sym, abs_offset)
    access_expr = SymbolicExpr.create("load", ptr_sym, None, None)
    pending = PendingCheck(access_expr, Int("addr"), None)
    calls = []

    assert access_interval_summary(access_expr, sanitizer.grid) == IntRange(
        base - tensor.element_size(),
        base - tensor.element_size(),
    )

    def record_z3_path(*args, **_kwargs):
        calls.append(args)

    sanitizer._check_range_satisfiable = record_z3_path  # type: ignore[method-assign]

    sanitizer._process_pending_check(ctx, pending, [])

    assert len(calls) == 1


def test_loop_range_summary_falls_back_for_forbidden_unary_address_op():
    sanitizer = SymbolicSanitizer(abort_on_error=False)
    tensor = torch.empty(4, dtype=torch.int32)
    base = tensor.data_ptr()
    end = base + tensor.numel() * tensor.element_size() - 1
    sanitizer.tensors.append(tensor)
    sanitizer.tensor_addrs.append((base, end, tensor))
    sanitizer.grid = (1, 1, 1)

    ctx, _access_expr = _make_loop_affine_access(
        base,
        tensor.element_size(),
        start=0,
        stop=tensor.numel(),
        step=1,
    )
    floor_offset = SymbolicExpr.create(
        "floor",
        SymbolicExpr.create("const", tensor.element_size(), INT32),
    )
    ptr_sym = SymbolicExpr.create(
        "add",
        SymbolicExpr.create("const", base, pointer_type(INT32)),
        floor_offset,
    )
    access_expr = SymbolicExpr.create("load", ptr_sym, None, None)
    pending = PendingCheck(access_expr, Int("addr"), None)
    calls = []

    def record_z3_path(*args, **_kwargs):
        calls.append(args)

    sanitizer._check_range_satisfiable = record_z3_path  # type: ignore[method-assign]

    sanitizer._process_pending_check(ctx, pending, [])

    assert len(calls) == 1


def test_loop_range_summary_falls_back_for_cast_address_op():
    sanitizer = SymbolicSanitizer(abort_on_error=False)
    tensor = torch.empty(4, dtype=torch.int32)
    base = tensor.data_ptr()
    end = base + tensor.numel() * tensor.element_size() - 1
    sanitizer.tensors.append(tensor)
    sanitizer.tensor_addrs.append((base, end, tensor))
    sanitizer.grid = (1, 1, 1)

    ctx, access_expr = _make_loop_affine_access(
        base,
        tensor.element_size(),
        start=0,
        stop=tensor.numel(),
        step=1,
    )
    ptr_sym = SymbolicExpr.create(
        "bitcast",
        access_expr.ptr,
        pointer_type(INT32),
    )
    access_expr = SymbolicExpr.create("load", ptr_sym, None, None)
    pending = PendingCheck(access_expr, Int("addr"), None)
    calls = []

    def record_z3_path(*args, **_kwargs):
        calls.append(args)

    sanitizer._check_range_satisfiable = record_z3_path  # type: ignore[method-assign]

    sanitizer._process_pending_check(ctx, pending, [])

    assert len(calls) == 1


def test_loop_range_summary_falls_back_for_left_shift_address_op():
    sanitizer = SymbolicSanitizer(abort_on_error=False)
    tensor = torch.empty(4, dtype=torch.int32)
    base = tensor.data_ptr()
    end = base + tensor.numel() * tensor.element_size() - 1
    sanitizer.tensors.append(tensor)
    sanitizer.tensor_addrs.append((base, end, tensor))
    sanitizer.grid = (1, 1, 1)

    ctx, _access_expr = _make_loop_affine_access(
        base,
        tensor.element_size(),
        start=0,
        stop=tensor.numel(),
        step=1,
    )
    shifted = SymbolicExpr.create(
        "left_shift",
        SymbolicExpr.create("const", -1, INT32),
        SymbolicExpr.create("const", 2, INT32),
    )
    offset = SymbolicExpr.create(
        "add",
        shifted,
        SymbolicExpr.create("const", tensor.element_size(), INT32),
    )
    ptr_sym = SymbolicExpr.create(
        "add",
        SymbolicExpr.create("const", base, pointer_type(INT32)),
        offset,
    )
    access_expr = SymbolicExpr.create("load", ptr_sym, None, None)
    pending = PendingCheck(access_expr, Int("addr"), None)
    calls = []

    def record_z3_path(*args, **_kwargs):
        calls.append(args)

    sanitizer._check_range_satisfiable = record_z3_path  # type: ignore[method-assign]

    sanitizer._process_pending_check(ctx, pending, [])

    assert len(calls) == 1


# ======== Report Layout Classification Tests ===========


def _normalize(layout_segments, tensor):
    """Translate ``(layout, segments)`` to ``(layout, relative_segments)``.

    Segment endpoints are made relative to ``tensor.data_ptr()`` so the asserts
    do not depend on PyTorch's allocator returning specific addresses.
    """
    layout, segments = layout_segments
    base = tensor.data_ptr() if tensor.numel() else 0
    return layout, [(s - base, e - base) for s, e in segments]


def test_classify_contiguous_1d():
    t = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    assert _normalize(_classify_layout_and_segments(t), t) == (
        "contiguous",
        [(0, 15)],
    )


def test_classify_dim0_contiguous_transpose():
    # ``t.T`` alone is storage-contiguous (just F-order), so it falls into the
    # single-segment path. Slice it to force the dim-0-contiguous bucket:
    # ``t.T[1:]`` has strides (1, 8) and a non-zero storage_offset.
    t = torch.arange(32, dtype=torch.int32).reshape(4, 8).T[1:]  # shape (7, 4)
    layout, rel = _normalize(_classify_layout_and_segments(t), t)
    assert layout == "dim-0-contiguous"
    # 4 segments (outer dim 1 size = 4), each covering 7 int32 elements = 28 bytes,
    # separated by stride[1]=8 elements = 32 bytes.
    assert len(rel) == 4
    assert rel[0] == (0, 27)
    assert rel[1] == (32, 59)


def test_classify_dim1_contiguous_slice():
    # ``t[:, 2:6]``: storage-discontiguous but inner dim 1 still has stride 1.
    base_t = torch.arange(32, dtype=torch.int32).reshape(4, 8)
    t = base_t[:, 2:6]
    layout, rel = _normalize(_classify_layout_and_segments(t), t)
    assert layout == "dim-1-contiguous"
    # 4 row segments, each 4 int32 wide = 16 bytes; rows separated by 32 bytes.
    assert len(rel) == 4
    assert rel[0] == (0, 15)
    assert rel[1] == (32, 47)


def test_classify_per_element_strided_view():
    base_t = torch.tensor([10, 99, 11, 99, 12], dtype=torch.int32)
    view = base_t[0::2]
    assert _normalize(_classify_layout_and_segments(view), view) == (
        "per-element",
        [(0, 3), (8, 11), (16, 19)],
    )


def test_classify_per_element_nonzero_storage_offset():
    base_t = torch.tensor([99, 10, 99, 11, 99, 12], dtype=torch.int32)
    view = base_t[1::2]
    assert view.storage_offset() == 1
    assert _normalize(_classify_layout_and_segments(view), view) == (
        "per-element",
        [(0, 3), (8, 11), (16, 19)],
    )


def test_classify_empty_tensor():
    t = torch.empty((0,), dtype=torch.int32)
    assert _classify_layout_and_segments(t) == ("empty", [])
