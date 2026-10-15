import pytest
import torch
import numpy as np

import triton
import triton.language as tl

import tilelens
from tilelens.core.data import Load, RawLoad, Store
from tilelens.clients.symbolic_engine import (
    SymbolicExpr,
    Z3Expr,
    RangeWrapper,
    _range_to_iterator_constraint,
)
from tilelens.core.symbolic_metadata import is_pointer_dtype
from tilelens.clients.sanitizer.data import UndecidedAccessRecordZ3
from tilelens.clients.sanitizer.sanitizer import SymbolicSanitizer
from tilelens.core.callbacks import ForLoopCallbacks
from tilelens.core.config import config
from tilelens.core.patch import LoopSite, loop_file_token
from z3 import unknown
from z3.z3 import BoolRef

from .loop_site_cross_file_kernel import (
    cross_file_loop_sanitizer,
    cross_file_outer_loop_kernel,
)


def _loop_var_name(lineno: int) -> str:
    """Expected symbolic iterator name for a loop in this test file."""
    token = loop_file_token(_loop_var_name.__code__.co_filename)
    return f"loop_i_{LoopSite(lineno, token)}"


@pytest.fixture
def _isolate_virtual_memory():
    """Save and restore config.virtual_memory around a test."""
    saved = config.virtual_memory
    yield
    config.virtual_memory = saved


# ======== Helpers ===========


class LoadIndexChecker(SymbolicSanitizer):
    """
    Record all offsets, then union into a set.
    """

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self._offset_lists: list[list[int]] = list()

    @property
    def observed_offsets(self) -> list[list[int]]:
        return self._offset_lists

    def register_op_callback(self, op_type):
        op_callbacks = super().register_op_callback(op_type)
        if op_type not in (Load, RawLoad) or op_callbacks.op_overrider is None:
            return op_callbacks

        orig_overrider = op_callbacks.op_overrider

        def _sum_offsets_from_addptr(expr):
            offsets = []

            cur = expr
            while cur.op == "addptr":
                off = cur.offset
                if (
                    off.op == "const"
                ):  # If any offset is not constant, we cannot sum it.
                    offsets.append(off.to_py())
                cur = cur.ptr

            if len(offsets) == 0:
                return None
            return np.sum(offsets, axis=0)

        def _offsets_from_const_pointer(expr):
            if expr.op != "const" or not is_pointer_dtype(expr.dtype):
                return None

            ptrs = expr.to_py()
            if not isinstance(ptrs, list):
                return None

            tensor = self._resolve_tensor(expr)
            if tensor is None:
                return None

            return (
                np.asarray(ptrs, dtype=np.int64) - tensor.data_ptr()
            ) // tensor.element_size()

        def _new_load_overrider(ptr, *args, **kwargs):
            # exec original overrider
            load_expr = orig_overrider(ptr, *args, **kwargs)
            p = load_expr.ptr
            offs = _sum_offsets_from_addptr(p)
            if offs is None:
                offs = _offsets_from_const_pointer(p)
            if offs is not None:
                self._offset_lists.append(offs.tolist())
            return load_expr

        # Return OpCallbacks with the new overrider, preserving other callbacks
        from tilelens.core.callbacks import OpCallbacks

        return OpCallbacks(
            before_callback=op_callbacks.before_callback,
            after_callback=op_callbacks.after_callback,
            op_overrider=_new_load_overrider,
        )


load_index_checker: LoadIndexChecker = LoadIndexChecker(abort_on_error=False)


class LoopBoundsChecker(SymbolicSanitizer):
    """
    Record concretized loop bounds.
    """

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self._bounds: list[tuple[int, int, int]] = list()

    @property
    def observed_bounds(self) -> list[tuple[int, int, int]]:
        return self._bounds

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        callbacks = super().register_for_loop_callback()
        orig_before = callbacks.before_loop_callback

        def _before_loop(loop_site, iterable):
            if orig_before is not None:
                orig_before(loop_site, iterable)
            if isinstance(iterable, RangeWrapper):
                self._bounds.append((iterable.start, iterable.stop, iterable.step))

        return ForLoopCallbacks(
            range_wrapper_factory=callbacks.range_wrapper_factory,
            range_type_callback=callbacks.range_type_callback,
            before_loop_callback=_before_loop,
            loop_iter_overrider=callbacks.loop_iter_overrider,
            loop_iter_listener=callbacks.loop_iter_listener,
            after_loop_callback=callbacks.after_loop_callback,
        )


loop_bounds_checker: LoopBoundsChecker = LoopBoundsChecker(abort_on_error=True)


class LoopDeferredCheckRecorder(SymbolicSanitizer):
    """
    Record when deferred checks are executed after loop exit.
    """

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self.after_loop_pending: list[int] = []
        self.check_inside_loop: list[tuple[Z3Expr, BoolRef | None]] = []
        self.iterator_constraints: list[BoolRef] = []

    def _check_range_satisfiable(
        self,
        access_addr: Z3Expr,
        expr_constraints: BoolRef | None,
        symbolic_expr: SymbolicExpr,
        source_location: tuple[str, int, str] | None = None,
    ) -> None:
        self.check_inside_loop.append((access_addr, expr_constraints))
        super()._check_range_satisfiable(
            access_addr, expr_constraints, symbolic_expr, source_location
        )

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        callbacks = super().register_for_loop_callback()
        orig_after = callbacks.after_loop_callback

        def _after_loop(loop_site: LoopSite) -> None:
            pending = 0
            if self.loop_stack and self.loop_stack[-1].loop_site == loop_site:
                ctx = self.loop_stack[-1]
                pending = len(ctx.pending_checks)
                self.iterator_constraints.append(
                    _range_to_iterator_constraint(
                        ctx.idx_z3, start=ctx.start, stop=ctx.stop, step=ctx.step
                    )
                )
            if orig_after is not None:
                orig_after(loop_site)
            self.after_loop_pending.append(pending)

        return ForLoopCallbacks(
            range_wrapper_factory=callbacks.range_wrapper_factory,
            range_type_callback=callbacks.range_type_callback,
            before_loop_callback=callbacks.before_loop_callback,
            loop_iter_overrider=callbacks.loop_iter_overrider,
            loop_iter_listener=callbacks.loop_iter_listener,
            after_loop_callback=_after_loop,
        )


loop_deferred_check_recorder: LoopDeferredCheckRecorder = LoopDeferredCheckRecorder(
    abort_on_error=False
)
sort_pointer_sanitizer: SymbolicSanitizer = SymbolicSanitizer(abort_on_error=False)
tuple_pointer_cast_checker: SymbolicSanitizer = SymbolicSanitizer(abort_on_error=False)
tuple_pointer_item_checker: SymbolicSanitizer = SymbolicSanitizer(abort_on_error=False)


# ======== Kernels ===========


@tilelens.trace(client=load_index_checker)
@triton.jit
def indirect_load_kernel(idx_ptr, src_ptr, dst_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    indices = tl.load(idx_ptr + offsets)
    out_val = tl.load(src_ptr + indices)
    tl.store(dst_ptr + offsets, out_val)


@tilelens.trace(client=loop_bounds_checker)
@triton.jit
def loop_bounds_kernel(start_ptr, stop_ptr, out_ptr):
    start = tl.load(start_ptr)
    stop = tl.load(stop_ptr)
    for _ in range(start, stop):
        pass
    tl.store(out_ptr, start)


@tilelens.trace(client=loop_bounds_checker)
@triton.jit
def loop_bounds_pid_kernel(out_ptr):
    pid = tl.program_id(0)
    start = pid
    stop = pid + 2
    for _ in range(start, stop):
        pass
    tl.store(out_ptr + pid, start)


@tilelens.trace(client=loop_deferred_check_recorder)
@triton.jit
def loop_deferred_check_kernel(out_ptr):
    for i in range(0, 4):
        idx = i + 1
        tl.store(out_ptr + idx, idx)


@tilelens.trace(client=load_index_checker)
@triton.jit
def loop_deferred_check_simplify_kernel(out_ptr):
    pid = tl.program_id(0)
    num_blocks = tl.num_programs(0) + 1
    for i in range(0, num_blocks):
        idx = pid + 1
        tl.store(out_ptr + idx, idx)


@tilelens.trace(client=sort_pointer_sanitizer)
@triton.jit
def sort_pointer_oob_kernel(out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    sorted_offsets = tl.sort(offs, 0, descending=True)
    tl.store(out_ptr + sorted_offsets, offs, mask=offs == 0)


@tilelens.trace(client=tuple_pointer_cast_checker)
@triton.jit
def pointer_tuple_cast_where_store_kernel(peer_ptrs):
    pid = tl.program_id(0)
    offs = tl.arange(0, 1)
    peer_ptr = peer_ptrs[0].to(tl.int64, bitcast=True)
    zero_ptr = tl.zeros((1,), dtype=tl.int64)
    selected_ptr = tl.where(pid == 0, peer_ptr, zero_ptr)
    selected_ptr = selected_ptr.to(peer_ptrs[0].dtype, bitcast=True)
    tl.store(selected_ptr + offs, tl.full((1,), 1.0, dtype=tl.float32))


@tilelens.trace(client=tuple_pointer_item_checker)
@triton.jit
def pointer_tuple_item_select_store_kernel(peer_ptrs, rank_ptr):
    dst_rank = tl.load(rank_ptr)
    dst_ptr = tl.zeros((1,), dtype=tl.int64).item()
    for i in tl.static_range(2):
        if dst_rank == i:
            dst_ptr = peer_ptrs[i].to(tl.int64, bitcast=True)
    dst_ptr = dst_ptr.to(peer_ptrs[0].dtype, bitcast=True)
    tl.store(dst_ptr, 1.0)


# ======== Indirect Load/Store Tests ===========


def test_indirect_load():
    load_index_checker.observed_offsets.clear()

    idx = torch.arange(128, dtype=torch.int32)
    src = torch.rand(128)
    dst = torch.empty_like(src)

    grid = lambda META: (triton.cdiv(128, META["BLOCK_SIZE"]),)
    indirect_load_kernel[grid](idx, src, dst, BLOCK_SIZE=32)

    expected_offsets = idx.cpu().numpy().tolist()  # Ground truth
    observed_offsets = [
        x for sublist in load_index_checker.observed_offsets for x in sublist
    ]  # Flatten the list of lists
    assert (
        expected_offsets == observed_offsets
    ), "Observed offsets do not match expected offsets."


# ======== Loop Tests ===========


def test_loop_bounds_from_load():
    loop_bounds_checker.observed_bounds.clear()
    start = torch.tensor([2], dtype=torch.int32)
    stop = torch.tensor([6], dtype=torch.int32)
    out = torch.empty((1,), dtype=torch.int32)

    loop_bounds_kernel[(1,)](start, stop, out)

    assert loop_bounds_checker.observed_bounds == [(2, 6, 1)]


def test_loop_bounds_from_pid():
    loop_bounds_checker.observed_bounds.clear()
    out = torch.empty((2,), dtype=torch.int32)

    loop_bounds_pid_kernel[(2,)](out)

    assert loop_bounds_checker.observed_bounds == [(0, 2, 1), (1, 3, 1)]


def test_loop_deferred_checks_after_context():
    loop_deferred_check_recorder.after_loop_pending.clear()
    loop_deferred_check_recorder.check_inside_loop.clear()
    loop_deferred_check_recorder.iterator_constraints.clear()
    loop_deferred_check_recorder.records.clear()

    out = torch.empty((2,), dtype=torch.int32)

    loop_deferred_check_kernel[(1,)](out)

    assert loop_deferred_check_recorder.after_loop_pending == [1]
    addr_expr, _ = loop_deferred_check_recorder.check_inside_loop[0]
    loop_var = _loop_var_name(2)
    assert f"4*{loop_var}" in str(addr_expr)
    iterator_constraints_str = " ".join(
        str(c) for c in loop_deferred_check_recorder.iterator_constraints
    )
    assert f"{loop_var} >= 0" in iterator_constraints_str
    assert f"{loop_var} < 4" in iterator_constraints_str
    assert loop_deferred_check_recorder.records


def test_cross_file_same_relative_lineno_loops_detect_oob():
    """A cross-file nested-loop OOB must be reported regardless of how loop
    identity is keyed.

    The outer loop (i in [0, 8), kernel file) and the inner loop (j in [0, 2),
    helper file) both sit at function-relative line 2. The store touches
    elements 2*i + j, up to 15, while ``out`` has only 8 elements — a real
    OOB.

    Today device jit functions run un-rewritten (``_jit_function_call``
    executes the raw fn), so j stays concrete and each unrolled address is
    checked against the outer iterator alone — the OOB is found. If device
    function loops are ever wired into the symbolic loop hooks, bare-lineno
    identity would collapse both iterators into one variable v < 2, turning
    the address into 3*v (max element 3) and silently missing the OOB;
    LoopSite identity keeps the iterators distinct. Either way this test must
    keep reporting the OOB.
    """
    cross_file_loop_sanitizer.records.clear()

    out = torch.empty((8,), dtype=torch.int32)

    cross_file_outer_loop_kernel[(1,)](out)

    assert cross_file_loop_sanitizer.records


def test_loop_deferred_checks_simplify():
    load_index_checker.observed_offsets.clear()

    out = torch.empty((3,), dtype=torch.int32)

    loop_deferred_check_simplify_kernel[(2,)](out)

    assert load_index_checker.observed_offsets == []


def test_sort_used_in_pointer_falls_back_to_eager_offsets():
    sort_pointer_sanitizer.records.clear()

    out = torch.empty((7,), dtype=torch.int32)
    sort_pointer_oob_kernel[(1,)](out, BLOCK=8)

    assert sort_pointer_sanitizer.records


def test_tuple_pointer_int_cast_uses_registered_tuple_ranges():
    tuple_pointer_cast_checker.records.clear()

    out0 = torch.empty((1,), dtype=torch.float32)
    out1 = torch.empty((1,), dtype=torch.float32)

    pointer_tuple_cast_where_store_kernel[(1,)]((out0, out1))

    assert tuple_pointer_cast_checker.records == []


def test_tuple_pointer_item_selection_uses_registered_tuple_ranges():
    tuple_pointer_item_checker.records.clear()

    out0 = torch.empty((1,), dtype=torch.float32)
    out1 = torch.empty((1,), dtype=torch.float32)
    rank = torch.tensor([1], dtype=torch.int32)

    pointer_tuple_item_select_store_kernel[(1,)]((out0, out1), rank)

    assert tuple_pointer_item_checker.records == []


# Dedicated sanitizer for nested loop regression test
nested_loop_checker = SymbolicSanitizer()


@tilelens.trace(client=nested_loop_checker)
@triton.jit
def nested_loop_outer_dep_kernel(out_ptr):
    for i in range(0, 3):
        for j in range(0, 4):
            idx = i * 4 + j
            tl.store(out_ptr + idx, idx)


def test_nested_loop_no_false_positive():
    nested_loop_checker.records.clear()

    # Buffer size = 12, exactly fits i*4+j where i in [0,3), j in [0,4)
    out = torch.empty((12,), dtype=torch.int32)
    nested_loop_outer_dep_kernel[(1,)](out)

    assert (
        len(nested_loop_checker.records) == 0
    ), f"False positive OOB detected. Records: {nested_loop_checker.records}"


# ======== Line Number Reporting Tests ===========


# Create a dedicated sanitizer for line number tests
# abort_on_error=False is on purpose: so OOB violations are recorded
line_number_checker: SymbolicSanitizer = SymbolicSanitizer(abort_on_error=False)


@tilelens.trace(client=line_number_checker)
@triton.jit
def oob_in_loop_kernel(ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Kernel where OOB occurs inside a loop at the tl.load line."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    for _ in range(2):
        # This line should be reported in the traceback when OOB occurs
        val = tl.load(ptr + offsets + 1000)  # OOB access due to +1000 offset
        tl.store(ptr + offsets, val)


def test_loop_oob_reports_correct_line_number():
    """
    Test that sanitizer reports the correct line number for OOB errors in loops.

    Previously, the sanitizer would report the function definition line instead
    of the actual tl.load/tl.store line that caused the error. This was because
    traceback was captured at loop exit time rather than when the memory
    operation was executed.
    """
    line_number_checker.records.clear()

    data = torch.zeros((16,), dtype=torch.float32)

    oob_in_loop_kernel[(1,)](data, N=16, BLOCK_SIZE=16)

    # Verify that OOB was detected
    assert len(line_number_checker.records) > 0, "Expected OOB to be detected"

    record = line_number_checker.records[0]
    assert len(record.user_code_tracebacks) > 0, "Expected traceback info"

    tb_info = record.user_code_tracebacks[0]

    # The error should point to the tl.load line, not the function definition
    assert "tl.load" in tb_info.line_of_code, (
        f"Expected traceback to point to tl.load line, "
        f"but got: {tb_info.line_of_code!r}"
    )

    # Verify the line contains the OOB offset
    assert (
        "+1000" in tb_info.line_of_code or "1000" in tb_info.line_of_code
    ), f"Expected line to contain the OOB offset, but got: {tb_info.line_of_code!r}"

    # Verify function name
    assert (
        tb_info.func_name == "oob_in_loop_kernel"
    ), f"Expected func_name to be 'oob_in_loop_kernel', but got: {tb_info.func_name!r}"


# ======== Z3 unknown ===========


class SolverGivesUpChecker(SymbolicSanitizer):
    """Z3 answers unknown for every access query: a resource limit of one
    unit stands in for any query it gives up on, deterministically."""

    def grid_callback(self, grid):
        super().grid_callback(grid)
        assert self.solver is not None
        self.solver.set("rlimit", 1)


class SolverInterruptedChecker(SymbolicSanitizer):
    """Z3 answers unknown because a Ctrl+C reached it during check()."""

    def grid_callback(self, grid):
        super().grid_callback(grid)
        solver = self.solver
        assert solver is not None
        solver.check = lambda *args: unknown
        solver.reason_unknown = lambda: "interrupted from keyboard"


solver_gives_up_checker = SolverGivesUpChecker(abort_on_error=False)
solver_gives_up_aborting_checker = SolverGivesUpChecker(abort_on_error=True)
solver_interrupted_checker = SolverInterruptedChecker(abort_on_error=False)


@tilelens.trace(client=solver_gives_up_checker)
@triton.jit
def oob_undecided_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs + 1)  # out of bounds in the last program
    for i in range(2):
        tl.store(y_ptr + offs + i, x)  # out of bounds in the last program


@tilelens.trace(client=solver_gives_up_aborting_checker)
@triton.jit
def oob_undecided_store_kernel(y_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(y_ptr + offs + 1, offs)


@tilelens.trace(client=solver_interrupted_checker)
@triton.jit
def interrupted_store_kernel(y_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(y_ptr + offs, offs)


def test_solver_unknown_records_access_as_unchecked():
    """An access Z3 cannot decide is recorded as unchecked, never taken for
    in bounds (both accesses are out of bounds; neither has a witness)."""
    solver_gives_up_checker.records.clear()

    x = torch.zeros((32,), dtype=torch.float32)
    y = torch.zeros((32,), dtype=torch.float32)
    oob_undecided_kernel[(2,)](x, y, BLOCK=16)

    records = solver_gives_up_checker.records
    assert [type(r) for r in records] == [UndecidedAccessRecordZ3] * 2
    load, store = records
    assert (load.op_type, load.tensor_name) == (Load, "x_ptr")
    assert (store.op_type, store.tensor_name) == (Store, "y_ptr")
    assert load.reason and store.reason
    # The deferred loop check keeps the location captured at the store.
    assert "tl.load(x_ptr" in load.user_code_tracebacks[-1].line_of_code
    assert "tl.store(y_ptr" in store.user_code_tracebacks[-1].line_of_code


def test_solver_unknown_warns_without_aborting(capsys):
    solver_gives_up_aborting_checker.records.clear()

    y = torch.zeros((16,), dtype=torch.int32)
    oob_undecided_store_kernel[(1,)](y, BLOCK=16)  # no SystemExit

    out = capsys.readouterr().out
    assert "UNCHECKED MEMORY ACCESS" in out
    assert "Z3 could not decide whether this Store stays in bounds" in out
    assert "Tensor Arg: y_ptr" in out
    assert solver_gives_up_aborting_checker.records == []


def test_solver_interrupted_by_ctrl_c_raises_keyboard_interrupt():
    solver_interrupted_checker.records.clear()

    y = torch.zeros((16,), dtype=torch.int32)
    with pytest.raises(KeyboardInterrupt):
        interrupted_store_kernel[(1,)](y, BLOCK=16)
    assert solver_interrupted_checker.records == []


# ---------------------------------------------------------------------------
# gemm_oob call-stack extraction test
# ---------------------------------------------------------------------------
# Runs the actual examples/sanitizer/gemm_oob.py and verifies that the
# sanitizer output (Code Context, Call Stack) matches docs/index.html.


def test_gemm_oob_call_stack():
    """
    Run examples/sanitizer/gemm_oob.py and verify the reported call-stack
    matches what is documented in docs/index.html:

      ━━━ Code Context ━━━
      File: tilelens/examples/sanitizer/gemm_oob.py
      Function: gemm_kernel
      Line 25:
        22 │     for k_block in range(K // TILE_SIZE):
        ...
      → 25 │         A_tile = tl.load(A + A_off + 1) # Out-Of-Bounds Access HERE!
      ━━━ Call Stack ━━━
      #1 gemm_kernel at gemm_oob.py:25
         └─ A_tile = tl.load(A + A_off + 1) # Out-Of-Bounds Access HERE!
    """
    import pathlib
    import subprocess
    import sys

    example_file = (
        pathlib.Path(__file__).resolve().parents[2]
        / "examples"
        / "sanitizer"
        / "gemm_oob.py"
    )
    assert example_file.exists(), f"Example file not found: {example_file}"

    import re

    result = subprocess.run(
        [sys.executable, str(example_file)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    # Strip ANSI escape codes so plain-text assertions work
    output = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout + result.stderr)

    # ---- Code Context checks (matching docs/index.html) ----
    assert "Code Context" in output, "Missing 'Code Context' section in output"
    assert "gemm_oob.py" in output, "Missing filename 'gemm_oob.py' in output"
    assert (
        "Function: gemm_kernel" in output
    ), "Missing 'Function: gemm_kernel' in output"
    assert "Line 25:" in output, "Missing 'Line 25:' in output"
    assert (
        "tl.load(A + A_off + 1)" in output
    ), "Missing 'tl.load(A + A_off + 1)' in Code Context"

    # ---- Call Stack checks ----
    assert (
        "#1 gemm_kernel at gemm_oob.py:25" in output
    ), "Missing expected call stack entry '#1 gemm_kernel at gemm_oob.py:25'"


# ======== Block Tensor (Block Pointer) Tests ===========


# abort_on_error=False is on purpose: so OOB violations are recorded
block_sanitizer = SymbolicSanitizer(abort_on_error=False)


@tilelens.trace(client=block_sanitizer)
@triton.jit
def block_tensor_1d_load_kernel(ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(N,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    tl.load(block_ptr, boundary_check=(0,))


@tilelens.trace(client=block_sanitizer)
@triton.jit
def block_tensor_1d_load_oob_kernel(ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(N,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    # No boundary_check -> OOB is undefined behavior
    tl.load(block_ptr)


@tilelens.trace(client=block_sanitizer)
@triton.jit
def block_tensor_1d_store_kernel(ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(N,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    val = tl.full((BLOCK_SIZE,), value=1.0, dtype=tl.float32)
    tl.store(block_ptr, val, boundary_check=(0,))


@tilelens.trace(client=block_sanitizer)
@triton.jit
def block_tensor_1d_store_oob_kernel(ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(N,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    val = tl.full((BLOCK_SIZE,), value=1.0, dtype=tl.float32)
    # No boundary_check -> OOB is undefined behavior
    tl.store(block_ptr, val)


@tilelens.trace(client=block_sanitizer)
@triton.jit
def block_tensor_2d_load_kernel(
    ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.load(block_ptr, boundary_check=(0, 1))


@tilelens.trace(client=block_sanitizer)
@triton.jit
def block_tensor_2d_load_oob_kernel(
    ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    # No boundary_check -> OOB is undefined behavior
    tl.load(block_ptr)


@tilelens.trace(client=block_sanitizer)
@triton.jit
def block_tensor_boundary_check_kernel(ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Block extends past tensor end but boundary_check masks it."""
    pid = tl.program_id(0)
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(N,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    # boundary_check=(0,) means hardware masks OOB positions
    tl.load(block_ptr, boundary_check=(0,))


@tilelens.trace(client=block_sanitizer)
@triton.jit
def block_tensor_loop_advance_kernel(ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(N,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    for _ in range(N // BLOCK_SIZE):
        tl.load(block_ptr, boundary_check=(0,))
        block_ptr = tl.advance(block_ptr, (BLOCK_SIZE,))


@tilelens.trace(client=block_sanitizer)
@triton.jit
def block_tensor_loop_advance_oob_kernel(
    ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(N,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    # One extra iteration -> OOB, NO boundary_check
    for _ in range(N // BLOCK_SIZE + 1):
        tl.load(block_ptr)
        block_ptr = tl.advance(block_ptr, (BLOCK_SIZE,))


@tilelens.trace(client=block_sanitizer)
@triton.jit
def block_tensor_2d_loop_advance_kernel(
    ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """2D block pointer advanced along K (N) dimension in a loop."""
    pid_m = tl.program_id(0)
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    for _ in range(N // BLOCK_N):
        tl.load(block_ptr, boundary_check=(0, 1))
        block_ptr = tl.advance(block_ptr, (0, BLOCK_N))


def test_block_tensor_1d_load_non_oob():
    block_sanitizer.records.clear()
    N = 64
    BLOCK_SIZE = 32
    data = torch.randn(N, dtype=torch.float32)
    grid = (N // BLOCK_SIZE,)
    block_tensor_1d_load_kernel[grid](data, N=N, BLOCK_SIZE=BLOCK_SIZE)
    assert (
        len(block_sanitizer.records) == 0
    ), f"Expected no OOB records, got {len(block_sanitizer.records)}"


def test_block_tensor_1d_store_non_oob():
    block_sanitizer.records.clear()
    N = 64
    BLOCK_SIZE = 32
    data = torch.zeros(N, dtype=torch.float32)
    grid = (N // BLOCK_SIZE,)
    block_tensor_1d_store_kernel[grid](data, N=N, BLOCK_SIZE=BLOCK_SIZE)
    assert (
        len(block_sanitizer.records) == 0
    ), f"Expected no OOB records, got {len(block_sanitizer.records)}"


def test_block_tensor_2d_load_non_oob():
    block_sanitizer.records.clear()
    M, N = 4, 4
    BLOCK_M, BLOCK_N = 2, 2
    data = torch.randn(M, N, dtype=torch.float32).contiguous()
    grid = (M // BLOCK_M, N // BLOCK_N)
    block_tensor_2d_load_kernel[grid](data, M=M, N=N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    assert (
        len(block_sanitizer.records) == 0
    ), f"Expected no OOB records, got {len(block_sanitizer.records)}"


def test_block_tensor_boundary_check_masks_oob():
    block_sanitizer.records.clear()
    # N=48 is not divisible by BLOCK_SIZE=32, so last block extends past end.
    # boundary_check=(0,) means hardware masks the OOB positions.
    N = 48
    BLOCK_SIZE = 32
    data = torch.randn(N, dtype=torch.float32)
    grid = (2,)  # 2 blocks: [0,32) and [32,64) -- second extends past N=48
    block_tensor_boundary_check_kernel[grid](data, N=N, BLOCK_SIZE=BLOCK_SIZE)
    assert (
        len(block_sanitizer.records) == 0
    ), f"Expected no OOB records with boundary_check, got {len(block_sanitizer.records)}"


def test_block_tensor_1d_load_oob():
    block_sanitizer.records.clear()
    N = 48
    BLOCK_SIZE = 32
    data = torch.randn(N, dtype=torch.float32)
    grid = (2,)  # Second block [32,64) goes past N=48, no boundary_check
    block_tensor_1d_load_oob_kernel[grid](data, N=N, BLOCK_SIZE=BLOCK_SIZE)
    assert len(block_sanitizer.records) > 0, "Expected OOB to be detected"


def test_block_tensor_1d_store_oob():
    block_sanitizer.records.clear()
    N = 48
    BLOCK_SIZE = 32
    data = torch.zeros(N, dtype=torch.float32)
    grid = (2,)  # Second block [32,64) goes past N=48, no boundary_check
    block_tensor_1d_store_oob_kernel[grid](data, N=N, BLOCK_SIZE=BLOCK_SIZE)
    assert len(block_sanitizer.records) > 0, "Expected OOB to be detected"


def test_block_tensor_2d_load_oob():
    block_sanitizer.records.clear()
    M, N = 4, 4
    BLOCK_M, BLOCK_N = 2, 2
    data = torch.randn(M, N, dtype=torch.float32).contiguous()
    # 3x2 grid but matrix is only 4x4 with BLOCK_M=2 -> pid_m=2 goes OOB
    grid = (3, 2)  # pid_m in [0,3), but M//BLOCK_M=2, so pid_m=2 is OOB
    block_tensor_2d_load_oob_kernel[grid](
        data, M=M, N=N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    assert len(block_sanitizer.records) > 0, "Expected OOB to be detected"


def test_block_tensor_loop_advance_non_oob():
    block_sanitizer.records.clear()
    N = 64
    BLOCK_SIZE = 32
    data = torch.randn(N, dtype=torch.float32)
    block_tensor_loop_advance_kernel[(1,)](data, N=N, BLOCK_SIZE=BLOCK_SIZE)
    assert (
        len(block_sanitizer.records) == 0
    ), f"Expected no OOB records, got {len(block_sanitizer.records)}"


def test_block_tensor_loop_advance_oob():
    block_sanitizer.records.clear()
    N = 64
    BLOCK_SIZE = 32
    data = torch.randn(N, dtype=torch.float32)
    block_tensor_loop_advance_oob_kernel[(1,)](data, N=N, BLOCK_SIZE=BLOCK_SIZE)
    assert len(block_sanitizer.records) > 0, "Expected OOB to be detected"


def test_block_tensor_2d_loop_advance_non_oob():
    block_sanitizer.records.clear()
    M, N = 4, 8
    BLOCK_M, BLOCK_N = 4, 4
    data = torch.randn(M, N, dtype=torch.float32).contiguous()
    # 1 program, loops N//BLOCK_N=2 times
    block_tensor_2d_loop_advance_kernel[(1,)](
        data, M=M, N=N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    assert (
        len(block_sanitizer.records) == 0
    ), f"Expected no OOB records, got {len(block_sanitizer.records)}"


def test_cli_code_context_points_to_kernel():
    """
    Run a minimal OOB kernel via ``tile-sanitizer`` CLI and verify the
    Code Context section points to the actual kernel line, not the CLI
    entry-point wrapper.
    """
    import os
    import re
    import subprocess
    import sys
    import tempfile
    import textwrap

    kernel_src = textwrap.dedent(
        """\
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def oob_kernel(ptr, BLOCK: tl.constexpr):
            offs = tl.arange(0, BLOCK)
            tl.load(ptr + offs)  # OOB: BLOCK > tensor size, no mask

        oob_kernel[(1,)](torch.zeros(4), BLOCK=16)
    """
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(kernel_src)
        tmp_path = tmp.name

    try:
        sanitizer = os.path.join(os.path.dirname(sys.executable), "tile-sanitizer")
        result = subprocess.run(
            [sanitizer, tmp_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout + result.stderr)

        # Code Context must reference the kernel, not the CLI wrapper
        assert "Code Context" in output, "Missing 'Code Context' section"
        assert (
            "Function: oob_kernel" in output
        ), f"Code Context should point to 'oob_kernel', got:\n{output}"
        assert (
            "tl.load(ptr + offs)" in output
        ), f"Code Context should show the tl.load line, got:\n{output}"
    finally:
        os.unlink(tmp_path)


# ======== Reduce with return_indices Tests ===========

reduce_indices_sanitizer = SymbolicSanitizer(abort_on_error=False)


@tilelens.trace(client=reduce_indices_sanitizer)
@triton.jit
def max_return_indices_kernel(inp_ptr, out_val_ptr, out_idx_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    inp = tl.load(inp_ptr + offs)
    max_val, max_idx = tl.max(inp, axis=0, return_indices=True)
    tl.store(out_val_ptr, max_val)
    tl.store(out_idx_ptr, max_idx)


@tilelens.trace(client=reduce_indices_sanitizer)
@triton.jit
def min_return_indices_kernel(inp_ptr, out_val_ptr, out_idx_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    inp = tl.load(inp_ptr + offs)
    min_val, min_idx = tl.min(inp, axis=0, return_indices=True)
    tl.store(out_val_ptr, min_val)
    tl.store(out_idx_ptr, min_idx)


def test_tl_max_return_indices():
    """tl.max with return_indices=True should stay on the symbolic path."""
    reduce_indices_sanitizer.records.clear()

    N = 32
    inp = torch.arange(N, dtype=torch.float32)
    out_val = torch.empty(1, dtype=torch.float32)
    out_idx = torch.empty(1, dtype=torch.int32)

    max_return_indices_kernel[(1,)](inp, out_val, out_idx, N=N)

    assert (
        len(reduce_indices_sanitizer.records) == 0
    ), f"Sanitizer reported {len(reduce_indices_sanitizer.records)} error(s)"


def test_tl_min_return_indices():
    """tl.min with return_indices=True should stay on the symbolic path."""
    reduce_indices_sanitizer.records.clear()

    N = 32
    inp = torch.arange(N, dtype=torch.float32)
    out_val = torch.empty(1, dtype=torch.float32)
    out_idx = torch.empty(1, dtype=torch.int32)

    min_return_indices_kernel[(1,)](inp, out_val, out_idx, N=N)

    assert (
        len(reduce_indices_sanitizer.records) == 0
    ), f"Sanitizer reported {len(reduce_indices_sanitizer.records)} error(s)"


# ======== Reduce + Broadcast Tests ===========

reduce_broadcast_sanitizer = SymbolicSanitizer()


@tilelens.trace(client=reduce_broadcast_sanitizer)
@triton.jit
def reduce_broadcast_kernel(in_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    row = tl.program_id(0) * 1 + tl.arange(0, 1)[:, None]
    cols = tl.arange(0, N)[None, :]
    x = tl.load(in_ptr + row * N + cols)
    s = tl.sum(x, axis=1)[:, None]  # reduce axis=1, re-expand
    result = x - s  # broadcast back to [1, N]
    tl.store(out_ptr + row * N + cols, result)


def test_reduce_broadcast():
    """
    Verify the symbolic engine handles reduce + broadcast without raising
    ValueError: Cannot broadcast, rank mismatch.

    Pattern: 2D load -> tl.sum(axis=1) -> [:, None] -> arithmetic with 2D tensor.
    """
    reduce_broadcast_sanitizer.records.clear()

    M, N = 1, 16
    inp = torch.randn(M, N, dtype=torch.float32)
    out = torch.empty_like(inp)

    reduce_broadcast_kernel[(M,)](inp, out, M=M, N=N)

    # No OOB expected — the kernel accesses exactly M*N elements
    assert (
        len(reduce_broadcast_sanitizer.records) == 0
    ), f"Expected no OOB records, got {len(reduce_broadcast_sanitizer.records)}"


# ======== Fake Tensor (Virtual Memory) OOB Tests ===========

fake_tensor_sanitizer = SymbolicSanitizer(abort_on_error=True)


@tilelens.trace(client=fake_tensor_sanitizer)
@triton.jit
def fake_tensor_oob_kernel(x_ptr, out_ptr, N: tl.constexpr):
    # Intentionally read out-of-bounds: offset N is beyond the valid range [0, N)
    val = tl.load(x_ptr + N)
    tl.store(out_ptr, val)


def test_oob_with_fake_tensor(_isolate_virtual_memory):
    fake_tensor_sanitizer.records.clear()

    config.virtual_memory = True
    x = torch.randn(8)
    out = torch.empty(1)
    with pytest.raises(SystemExit):
        fake_tensor_oob_kernel[(1,)](x, out, N=8)


@tilelens.trace(client=SymbolicSanitizer())
@triton.jit
def block_ptr_sum_kernel(
    s_ptr,
    z_ptr,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    o_i = tl.arange(0, BT)
    m = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)
    b_z = tl.zeros([BS], dtype=tl.float32)
    p_s = tl.make_block_ptr(s_ptr, (T, S), (S, 1), (0, 0), (BT, BS), (1, 0))
    p_z = tl.make_block_ptr(z_ptr, (T, S), (S, 1), (0, 0), (BT, BS), (1, 0))
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_c = b_z[None, :] + tl.dot(m, b_s, allow_tf32=False)
    tl.store(p_z, b_c.to(p_z.dtype.element_ty), boundary_check=(0, 1))
    b_z += tl.sum(b_s, 0)


def test_reduce_symbolic_core_dtype():
    """tl.sum on cast block_ptr data must preserve block_type dtype."""
    s = torch.randn(16, 16, device="cpu")
    z = torch.empty_like(s)
    block_ptr_sum_kernel[(1,)](s, z, T=16, S=16, BT=16, BS=16)


@tilelens.trace(client=SymbolicSanitizer())
@triton.jit
def softmax_kernel(output_ptr, input_ptr, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(input_ptr + row * N + offs, mask=mask, other=-float("inf"))
    x_max = tl.max(x, 0)
    exp_x = tl.exp(x - x_max)
    sum_exp = tl.sum(exp_x, 0)
    tl.store(output_ptr + row * N + offs, exp_x / sum_exp, mask=mask)


def test_reduce_symbolic_nonetype():
    """Softmax kernel: tl.exp -> tl.max / tl.sum must propagate dtype correctly."""
    x = torch.randn(4, 64, device="cpu")
    out = torch.empty_like(x)
    softmax_kernel[(4,)](out, x, 64, BLOCK=64)


@tilelens.trace(client=SymbolicSanitizer())
@triton.jit
def exp_expand_kernel(x_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    w = tl.exp(x)
    m = w[None, :] * tl.load(x_ptr + offs)[:, None]
    tl.store(out_ptr + offs, tl.sum(m, axis=1))


def test_expand_dims_scalar_attr():
    """tl.exp followed by expand_dims must propagate dtype correctly."""
    x = torch.randn(8, device="cpu")
    out = torch.empty(8, device="cpu")
    exp_expand_kernel[(1,)](x, out, N=8)


# ======== Non-contiguous Expanded Tensor Regression Test ===========


@tilelens.trace(client=SymbolicSanitizer())
@triton.jit
def read_expanded_kernel(inp, out, stride_row, stride_col, M, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    ptrs = inp + row * stride_row + offs * stride_col
    x = tl.load(ptrs, mask=mask, other=0)
    tl.store(out + row * N + offs, x, mask=mask)


def test_non_contiguous_expanded_tensor():
    """expand()-ed tensors (stride-0, non-contiguous) must not crash the sanitizer."""
    M, N = 4, 8
    row = torch.arange(N, device="cpu", dtype=torch.float32)
    x = row.unsqueeze(0).expand(M, N)  # shape (4,8), strides (0, 1)
    assert not x.is_contiguous()
    out = torch.empty(M, N, device="cpu")
    read_expanded_kernel[(M,)](x, out, x.stride(0), x.stride(1), M, N, BLOCK_N=8)


# ======== Sliced View with Nonzero storage_offset Regression Test ===========


inner_stride1_offset_sanitizer = SymbolicSanitizer(abort_on_error=False)


@tilelens.trace(client=inner_stride1_offset_sanitizer)
@triton.jit
def inner_stride1_offset_no_oob_kernel(x_ptr, out_ptr, L: tl.constexpr):
    offs = tl.arange(0, 4)
    x = tl.load(x_ptr + offs, mask=offs < L, other=0)
    tl.store(out_ptr + offs, x, mask=offs < L)


def test_inner_stride1_nonzero_storage_offset_no_oob():
    """Sliced view with inner-stride 1 and nonzero storage_offset: reading
    the inner-contiguous first row is legal and must not be reported as OOB.
    Regression for get_physical_addr_from_tensor_slice() double-counting
    storage_offset."""
    inner_stride1_offset_sanitizer.records.clear()

    base = torch.arange(20, dtype=torch.int32).reshape(4, 5)
    x = base[1:3, 1:4]
    # shape=(2, 3), stride=(5, 1), storage_offset=6
    # First row is physically contiguous: x_ptr + {0, 1, 2}

    out = torch.empty((4,), dtype=torch.int32)

    inner_stride1_offset_no_oob_kernel[(1,)](x, out, L=3)

    assert len(inner_stride1_offset_sanitizer.records) == 0


cumsum_sanitizer = SymbolicSanitizer(abort_on_error=False)


@tilelens.trace(client=cumsum_sanitizer)
@triton.jit
def cumsum_indexed_store_kernel(active_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    active = tl.load(active_ptr + offs)
    write_idx = tl.cumsum(active, 0) - 1
    tl.store(out_ptr + write_idx, active)


def test_sanitizer_supports_data_dependent_cumsum_index():
    cumsum_sanitizer.records.clear()

    active = torch.ones((8,), dtype=torch.int32)
    out = torch.empty_like(active)

    cumsum_indexed_store_kernel[(1,)](active, out, BLOCK=active.numel())

    assert len(cumsum_sanitizer.records) == 0


ashr_index_sanitizer = SymbolicSanitizer(abort_on_error=False)


@tilelens.trace(client=ashr_index_sanitizer)
@triton.jit
def ashr_index_store_kernel(out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    write_idx = offs >> 1
    tl.store(out_ptr + write_idx, offs)


def test_sanitizer_supports_ashr_in_symbolic_address():
    ashr_index_sanitizer.records.clear()

    out = torch.empty((4,), dtype=torch.int32)

    ashr_index_store_kernel[(1,)](out, BLOCK=out.numel())

    assert len(ashr_index_sanitizer.records) == 0


where_index_sanitizer = SymbolicSanitizer(abort_on_error=False)


@tilelens.trace(client=where_index_sanitizer)
@triton.jit
def where_index_store_kernel(out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    write_idx = tl.where(offs % 2 == 0, offs, BLOCK - 1 - offs)
    tl.store(out_ptr + write_idx, offs)


def test_sanitizer_supports_where_in_symbolic_address():
    where_index_sanitizer.records.clear()

    out = torch.empty((4,), dtype=torch.int32)

    where_index_store_kernel[(1,)](out, BLOCK=out.numel())

    assert len(where_index_sanitizer.records) == 0


masked_compaction_index_sanitizer = SymbolicSanitizer(abort_on_error=False)


@tilelens.trace(client=masked_compaction_index_sanitizer)
@triton.jit
def masked_compaction_index_store_kernel(bitmask_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    word = tl.load(bitmask_ptr + offs * 0)
    active = (word >> offs) & 1
    exc_cumsum = tl.cumsum(active, 0) - active
    active_flags = active.to(tl.int1)
    rev_arange = tl.where(active_flags, 0, BLOCK - 1 - offs)
    write_idx = exc_cumsum + rev_arange
    tl.store(out_ptr + write_idx, active)


def test_sanitizer_materializes_loaded_masked_compaction_store_indices():
    masked_compaction_index_sanitizer.records.clear()

    bitmask = torch.tensor([0b0101], dtype=torch.int32)
    out = torch.empty((4,), dtype=torch.int32)

    masked_compaction_index_store_kernel[(1,)](bitmask, out, BLOCK=out.numel())

    assert len(masked_compaction_index_sanitizer.records) == 0


# ======== Data-Dependent Loop Bound (Integer Division) Tests ===========


data_dep_div_sanitizer = SymbolicSanitizer(abort_on_error=False)


@tilelens.trace(client=data_dep_div_sanitizer)
@triton.jit
def data_dep_loop_div_kernel(Lens, Out, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    seq_len = tl.load(Lens + pid)
    # integer division of a loaded value produces a SymbolicExpr
    num_blocks = (seq_len + BLOCK - 1) // BLOCK
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(0, num_blocks):
        acc += 1.0
    tl.store(Out + pid * BLOCK + tl.arange(0, BLOCK), acc)


def test_data_dependent_loop_bound_div():
    """
    Data-dependent loop bound via integer division of a loaded value
    must not crash the symbolic engine (e.g. 'to_py must be implemented
    by subclasses').
    """
    data_dep_div_sanitizer.records.clear()

    N = 4
    BLOCK = 16
    lens = torch.tensor([48, 32, 64, 16], dtype=torch.int32)
    out = torch.empty(N, BLOCK)

    data_dep_loop_div_kernel[(N,)](lens, out, BLOCK=BLOCK)

    assert (
        len(data_dep_div_sanitizer.records) == 0
    ), f"Expected no OOB records, got {len(data_dep_div_sanitizer.records)}"


# ======== Data-Dependent Loop Bound (tl.cdiv) Tests ===========


cdiv_loop_sanitizer = SymbolicSanitizer(abort_on_error=False)


@tilelens.trace(client=cdiv_loop_sanitizer)
@triton.jit
def cdiv_loop_bound_kernel(
    X, Out, seqlen, chunk_size, BLOCK_CS: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    # min of runtime values — produces a symbolic expr involving pid
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    # tl.cdiv on that value → idiv in symbolic tree
    num_iters = tl.cdiv(chunk_size_limit, BLOCK_CS)
    offs = tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for cs in range(0, num_iters):
        acc += tl.load(X + offs, mask=offs < BLOCK_N)
    tl.store(Out + pid_m * BLOCK_N + offs, acc)


def test_data_dependent_cdiv_loop_bound():
    """
    tl.cdiv on a runtime value used as a loop bound must not crash the
    symbolic engine with 'NotImplementedError: Concretize for op idiv'.
    """
    cdiv_loop_sanitizer.records.clear()

    seqlen = 64
    chunk_size = 32
    BLOCK_CS = 16
    BLOCK_N = 16
    nchunks = seqlen // chunk_size
    x = torch.ones(BLOCK_N)
    out = torch.empty(2, BLOCK_N)

    cdiv_loop_bound_kernel[(2, nchunks)](
        x, out, seqlen, chunk_size, BLOCK_CS=BLOCK_CS, BLOCK_N=BLOCK_N
    )

    assert (
        len(cdiv_loop_sanitizer.records) == 0
    ), f"Expected no OOB records, got {len(cdiv_loop_sanitizer.records)}"


# ======== TensorWrapper Regression Test ===========


@tilelens.trace(client=SymbolicSanitizer())
@triton.jit
def copy_kernel(src, dst, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(src + offs, mask=mask)
    tl.store(dst + offs, x, mask=mask)


# ======== Reduce on Dot Result Tests ===========


reduce_dot_sanitizer = SymbolicSanitizer()


@tilelens.trace(client=reduce_dot_sanitizer)
@triton.jit
def dot_row_max_kernel(
    Q,
    K,
    Out,
    stride_qm,
    stride_kn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    q = tl.load(Q + offs_m[:, None] * stride_qm + offs_d[None, :])
    k = tl.load(K + offs_d[:, None] + offs_n[None, :] * stride_kn)

    qk = tl.dot(q, k)  # [BLOCK_M, BLOCK_N]
    row_max = tl.max(qk, axis=1)  # reduce axis=1 -> [BLOCK_M]

    tl.store(Out + offs_m, row_max)


def test_reduce_on_dot_result():
    """tl.max on the result of tl.dot must not crash the symbolic engine.

    Regression: ReduceSymbolicExpr previously raised
    'expects block input with non-empty shape' when reducing a dot product.
    """
    reduce_dot_sanitizer.records.clear()

    M, N, D = 16, 16, 16
    q = torch.randn(M, D, dtype=torch.float16)
    k = torch.randn(D, N, dtype=torch.float16)
    out = torch.empty(M, dtype=torch.float32)

    dot_row_max_kernel[(1,)](
        q,
        k,
        out,
        q.stride(0),
        k.stride(1),
        BLOCK_M=M,
        BLOCK_N=N,
        D=D,
    )

    assert (
        len(reduce_dot_sanitizer.records) == 0
    ), f"Expected no OOB records, got {len(reduce_dot_sanitizer.records)}"


@tilelens.trace(client=reduce_dot_sanitizer)
@triton.jit
def batched_dot_row_max_kernel(
    A,
    B,
    Out,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_ob,
    stride_om,
    B_DIM: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    offs_b = tl.arange(0, B_DIM)
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)

    a = tl.load(
        A
        + offs_b[:, None, None] * stride_ab
        + offs_m[None, :, None] * stride_am
        + offs_k[None, None, :] * stride_ak
    )
    b = tl.load(
        B
        + offs_b[:, None, None] * stride_bb
        + offs_k[None, :, None] * stride_bk
        + offs_n[None, None, :] * stride_bn
    )

    c = tl.dot(a, b)  # [B_DIM, M, N]
    row_max = tl.max(c, axis=2)  # reduce axis=2 -> [B_DIM, M]

    tl.store(Out + offs_b[:, None] * stride_ob + offs_m[None, :] * stride_om, row_max)


def test_reduce_on_batched_dot_result():
    """tl.max on the result of a 3D batched tl.dot must not crash the symbolic engine."""
    reduce_dot_sanitizer.records.clear()

    B, M, N, K = 2, 16, 16, 16
    a = torch.randn(B, M, K, dtype=torch.float16)
    b = torch.randn(B, K, N, dtype=torch.float16)
    out = torch.empty(B, M, dtype=torch.float32)

    batched_dot_row_max_kernel[(1,)](
        a,
        b,
        out,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        out.stride(0),
        out.stride(1),
        B_DIM=B,
        M=M,
        N=N,
        K=K,
    )

    assert (
        len(reduce_dot_sanitizer.records) == 0
    ), f"Expected no OOB records, got {len(reduce_dot_sanitizer.records)}"


def test_reinterpret_tensor_wrapper():
    """triton.reinterpret() produces a TensorWrapper; sanitizer must handle it."""
    N = 64
    x = torch.ones(N, dtype=torch.float16, device="cpu")
    y = torch.empty(N, dtype=torch.float16, device="cpu")
    copy_kernel[(1,)](triton.reinterpret(x, tl.float16), y, N, BLOCK=64)


# ======== Strided Tensor View OOB Tests (triton-lang/triton#10336) ===========


strided_view_sanitizer = SymbolicSanitizer(abort_on_error=False)


@tilelens.trace(client=strided_view_sanitizer)
@triton.jit
def strided_view_oob_load_kernel(
    vals_ptr,
    bins_ptr,
    out_ptr,
    L: tl.constexpr,
    SV: tl.constexpr,
    SB: tl.constexpr,
):
    offs = tl.arange(0, 8)
    v = tl.load(vals_ptr + offs * SV, mask=offs < L, other=0)
    b = tl.load(bins_ptr + offs * SB, mask=offs < L, other=-1)  # OOB HERE!
    tl.store(out_ptr + offs, v * 100 + b, mask=offs < L)


@pytest.fixture
def _isolate_per_element_threshold():
    """Save and restore the per-element warning threshold."""
    saved = config.symbolic_per_element_warn_threshold
    yield
    config.symbolic_per_element_warn_threshold = saved


def test_strided_view_oob_load():
    """
    Regression for triton-lang/triton#10336.

    A masked tl.load with ``mask = offs < L`` reads past the underlying
    storage when L exceeds the strided view's element count. The JIT path
    silently performs the wild pointer read; the sanitizer should detect it.

    Setup:
      - bins_base has 5 int32 elements (20 bytes of storage).
      - bins = bins_base[0::2] is a view of 3 elements with stride 2.
      - L=4 makes the mask true for offs=3, so the kernel issues a load at
        bins_ptr + 3*2 = bins_base[6], past the underlying storage.
    """
    strided_view_sanitizer.records.clear()

    vals_base = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    bins_base = torch.tensor([10, 99, 11, 99, 12], dtype=torch.int32)

    vals = vals_base
    bins = bins_base[0::2]  # visible values [10, 11, 12], stride=2
    out = torch.empty((4,), dtype=torch.int32)

    strided_view_oob_load_kernel[(1,)](
        vals, bins, out, L=4, SV=vals.stride(0), SB=bins.stride(0)
    )

    assert len(strided_view_sanitizer.records) > 0, "Expected OOB to be detected"


def test_strided_view_correct_mask_no_oob():
    """Correctly-masked access on a stride-2 view must not produce OOB."""
    strided_view_sanitizer.records.clear()

    vals_base = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    bins_base = torch.tensor([10, 99, 11, 99, 12], dtype=torch.int32)

    bins = bins_base[0::2]
    out = torch.empty((4,), dtype=torch.int32)

    # L=3 matches bins.numel(); mask excludes the would-be OOB offset.
    strided_view_oob_load_kernel[(1,)](
        vals_base, bins, out, L=3, SV=vals_base.stride(0), SB=bins.stride(0)
    )

    assert len(strided_view_sanitizer.records) == 0


def test_strided_view_warns_on_large_numel(_isolate_per_element_threshold):
    """Per-element enumeration emits UserWarning above the threshold."""
    strided_view_sanitizer.records.clear()
    config.symbolic_per_element_warn_threshold = 1  # force a warning

    vals_base = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    bins_base = torch.tensor([10, 99, 11, 99, 12], dtype=torch.int32)
    bins = bins_base[0::2]
    out = torch.empty((4,), dtype=torch.int32)

    with pytest.warns(UserWarning, match="non-unit inner stride"):
        strided_view_oob_load_kernel[(1,)](
            vals_base, bins, out, L=3, SV=vals_base.stride(0), SB=bins.stride(0)
        )

    # Warning path must not change correctness -- no OOB with correct mask.
    assert len(strided_view_sanitizer.records) == 0
