"""Unit tests for the shared SymbolicClient substrate.

These tests are client-agnostic — they exercise pieces of
``tilelens/clients/symbolic_engine.py`` (the ``_range_to_iterator_constraint``
Z3 helper, the ``SymbolicExpr`` tree and its subclasses, reduce-shape
inference, block-pointer construction, load/store dtype derivation) that
every symbolic client relies on. Sanitizer- and race-detector-specific
tests live next to their respective clients.
"""

import os
import pytest
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np

from tilelens.clients.symbolic_engine import (
    SymbolicClient,
    SymbolicExpr,
    ConstSymbolicExpr,
    RangeWrapper,
    ReduceSymbolicExpr,
    LoadSymbolicExpr,
    StoreSymbolicExpr,
    _range_to_iterator_constraint,
)
from tilelens.core.data import Sort
from tilelens.core.patch import LoopSite, loop_file_token
from tilelens.core.symbolic_metadata import (
    FLOAT16,
    FLOAT32,
    INT1,
    INT32,
    UINT32,
    SymbolicTensorValue,
    SymbolicTypeSpec,
    block_type,
    pointer_type,
)
from tilelens.utils.traceback_utils import _triton_frame_dirs
from z3.z3 import ArithRef, BoolRef, IntNumRef
from z3 import Solver, Int, sat


class _LoopSiteSymbolicClient(SymbolicClient):
    NAME = "loop_site_symbolic_client"

    def pre_warmup_callback(self, jit_fn, *args, **kwargs) -> bool:
        return True

    def post_warmup_callback(self, jit_fn, ret) -> None:
        pass


# ======== Range Constraint Tests ===========


def test_symbolic_engine_imports_without_loading_triton():
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(root)
        if not env.get("PYTHONPATH")
        else f"{root}{os.pathsep}{env['PYTHONPATH']}"
    )
    code = """
import builtins
original_import = builtins.__import__
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "triton" or name.startswith("triton."):
        raise RuntimeError(f"blocked import: {name}")
    return original_import(name, globals, locals, fromlist, level)
builtins.__import__ = guarded_import
import tilelens.clients.symbolic_engine
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_range_to_iterator_constraint():
    def assert_constraint_allows(constraint: BoolRef, var: ArithRef, val: int) -> None:
        solver = Solver()
        solver.add(constraint)
        solver.add(var == val)
        assert solver.check() == sat

    def assert_constraint_forbids(constraint: BoolRef, var: ArithRef, val: int) -> None:
        solver = Solver()
        solver.add(constraint)
        solver.add(var == val)
        assert solver.check() != sat

    # Positive step: range(1, 10, 2) -> [1, 3, 5, 7, 9]
    i = Int("i")
    constraint = _range_to_iterator_constraint(i, start=1, stop=10, step=2)
    assert_constraint_allows(constraint, i, 1)
    assert_constraint_allows(constraint, i, 3)
    assert_constraint_allows(constraint, i, 9)
    assert_constraint_forbids(constraint, i, 2)
    assert_constraint_forbids(constraint, i, 10)

    # Negative step: range(4, -1, -2) -> [4, 2, 0]
    i = Int("i")
    constraint = _range_to_iterator_constraint(i, start=4, stop=-1, step=-2)
    assert_constraint_allows(constraint, i, 4)
    assert_constraint_allows(constraint, i, 2)
    assert_constraint_allows(constraint, i, 0)
    assert_constraint_forbids(constraint, i, 1)
    assert_constraint_forbids(constraint, i, -1)


def test_loop_sites_in_different_files_get_distinct_iterator_state():
    client = _LoopSiteSymbolicClient()
    iterable = RangeWrapper(range(3), length=3, start=0, stop=3, step=1)
    site_a = LoopSite(5, loop_file_token("/src/kernel_a.py"))
    site_b = LoopSite(5, loop_file_token("/src/helper_b.py"))

    client._loop_hook_before(site_a, iterable)
    ctx_a = client.loop_stack[-1]
    client._loop_hook_after(site_a)

    client._loop_hook_before(site_b, iterable)
    ctx_b = client.loop_stack[-1]
    client._loop_hook_after(site_b)

    assert ctx_a.idx_z3.decl().name() == f"loop_i_{site_a}"
    assert ctx_b.idx_z3.decl().name() == f"loop_i_{site_b}"
    assert ctx_a.idx_z3.decl().name() != ctx_b.idx_z3.decl().name()
    assert (site_a, 0, 3, 1) in client.loop_iterator_constraint_cache
    assert (site_b, 0, 3, 1) in client.loop_iterator_constraint_cache
    # A cache keyed by lineno alone would hand site_b the constraint built
    # for site_a's variable, leaving site_b's own iterator unconstrained.
    assert f"loop_i_{site_a}" in str(ctx_a.iterator_constraint)
    assert f"loop_i_{site_b}" in str(ctx_b.iterator_constraint)


# ======== Reduce Operations Tests =========


@pytest.mark.parametrize("op", ["max", "min", "sum"])
@pytest.mark.parametrize("data", [[1, 5, 3, 2], [42]])
def test_reduce_expr_eval(op: str, data):
    import numpy as np
    import builtins

    input_arr = SymbolicExpr.create(
        "const", np.array(data), block_type(INT32, [len(data)])
    )
    reduce_expr = SymbolicExpr.create(op, input_arr, None, False)

    result, _ = reduce_expr.eval(simplify_constraints=False)
    # Use reflection to call the matching Python builtin, e.g. builtins.max([1,5,3,2]) -> 5
    assert cast(IntNumRef, result).as_long() == getattr(builtins, op)(data)


# ======== Basic Symbolic Expr Operations Tests =========


@pytest.mark.parametrize("value", [(1, 2, 3), 4])
def test_basic_expr_const_eval(value):
    # Test that "const" SymbolicExpr correctly evaluates both scalar (e.g. 4)
    # and vector (e.g. (1, 2, 3)) constants.
    const_expr = SymbolicExpr.create("const", value, INT32)
    result, constraints = const_expr.eval(simplify_constraints=False)
    if isinstance(value, (list, tuple)):
        # Vector constants return a list of Z3 ints.
        assert [cast(IntNumRef, v).as_long() for v in cast(list, result)] == list(value)
    else:
        # Scalar constants return a single Z3 int.
        assert cast(IntNumRef, result).as_long() == value
    # Constants produce no constraints.
    assert constraints is None


def test_numpy_scalar_const_eval():
    const_expr = SymbolicExpr.from_value(np.int32(7))
    assert isinstance(const_expr, ConstSymbolicExpr)

    result, constraints = const_expr.eval(simplify_constraints=False)

    assert cast(IntNumRef, result).as_long() == 7
    assert constraints is None


@pytest.mark.parametrize(
    "axis,expected_pid",
    [
        (0, SymbolicExpr.PID0),
        (1, SymbolicExpr.PID1),
        (2, SymbolicExpr.PID2),
    ],
)
def test_basic_expr_pid_eval(axis, expected_pid):
    # Test that "pid" expr returns the corresponding predefined symbolic variable (PID0/PID1/PID2).
    pid_expr = SymbolicExpr.create("pid", axis)
    result, constraints = pid_expr.eval(simplify_constraints=False)
    assert result == expected_pid
    assert constraints is None


@pytest.mark.parametrize("start,end", [(4, 8), (0, 4)])
def test_basic_expr_arange_eval(start, end):
    # Test that arange expr produces a named symbolic variable with range
    # constraints; the name is suffixed with the creation site.
    arange_expr = SymbolicExpr.create("arange", INT32, start, end)
    result, constraints = arange_expr.eval(simplify_constraints=False)
    result = cast(ArithRef, result)
    assert result.decl().name().startswith(f"arange_{start}_{end}_l")
    assert constraints is not None
    constraints_str = str(constraints)
    assert f"{result} >= {start}" in constraints_str
    assert f"{result} < {end}" in constraints_str


def test_arange_interned_by_creation_site():
    # A square tile's row and column aranges must not share one var; one
    # line re-executed (a loop) keeps reusing its var.
    rows = SymbolicExpr.create("arange", INT32, 0, 4)
    cols = SymbolicExpr.create("arange", INT32, 0, 4)
    looped = [SymbolicExpr.create("arange", INT32, 0, 4) for _ in range(2)]
    names = [str(e.eval()[0]) for e in (rows, cols, *looped)]
    assert names[0] != names[1] and names[2] == names[3]


def test_cumsum_overrider_defaults_axis_like_tl_cumsum():
    # A bare tl.cumsum(x) reaches the overrider with one positional arg.
    client = _LoopSiteSymbolicClient()
    x = SymbolicExpr.create("const", np.array([1, 0, 1]), block_type(INT32, [3]))
    assert int(client._op_cumsum_overrider(x).axis.to_py()) == 0


# ======== Unary Symbolic Expr Operations Tests =========


@pytest.mark.parametrize(
    "op,value,expected",
    [
        ("abs", -3, 3),
        ("fabs", -7, 7),
    ],
)
def test_unary_expr_eval(op: str, value: int, expected: int):
    # Test that unary ops (abs, fabs) correctly evaluate on constant inputs.
    arg = SymbolicExpr.create("const", value, INT32)
    expr = SymbolicExpr.create(op, arg)
    result, constraints = expr.eval(simplify_constraints=False)
    assert cast(IntNumRef, result).as_long() == expected
    assert constraints is None


# ======== Binary Symbolic Expr Operations Tests =========


@pytest.mark.parametrize(
    "op,lhs,rhs,expected",
    [
        ("add", 2, 3, 5),
        ("sub", 7, 4, 3),
        ("mul", 3, 5, 15),
        ("idiv", 8, 2, 4),
        ("mod", 7, 4, 3),
        ("less", 2, 9, True),
        ("less_equal", 3, 3, True),
        ("greater", 5, 2, True),
        ("greater_equal", 5, 7, False),
        ("equal", 4, 4, True),
        ("not_equal", 4, 4, False),
        ("maximum", 2, 9, 9),
        ("minimum", 2, 9, 2),
        ("bitwise_and", 6, 3, 2),
        ("bitwise_or", 6, 3, 7),
        ("bitwise_xor", 6, 3, 5),
    ],
)
def test_binary_expr_eval(op: str, lhs: int, rhs: int, expected):
    # Test arithmetic, comparison, and bitwise binary ops on constant inputs.
    lhs_expr = SymbolicExpr.create("const", lhs, INT32)
    rhs_expr = SymbolicExpr.create("const", rhs, INT32)
    expr = SymbolicExpr.create(op, lhs_expr, rhs_expr)
    result, constraints = expr.eval(simplify_constraints=False)
    if isinstance(expected, bool):
        assert str(result) == str(expected)
    else:
        assert cast(IntNumRef, result).as_long() == expected
    assert constraints is None


def test_ashr_expr_eval_scalar_constants():
    lhs = SymbolicExpr.create("const", -8, INT32)
    rhs = SymbolicExpr.create("const", 1, INT32)
    expr = SymbolicExpr.create("ashr", lhs, rhs)

    result, constraints = expr.eval(simplify_constraints=False)

    assert cast(IntNumRef, result).as_long() == -4
    assert constraints is None


def test_unsigned_shift_expr_eval_scalar_constants():
    lhs = SymbolicExpr.create("const", 0xFFFF0008, UINT32)
    rhs = SymbolicExpr.create("const", 16, UINT32)
    expr = SymbolicExpr.create("right_shift", lhs, rhs)

    result, constraints = expr.eval(simplify_constraints=False)

    assert cast(IntNumRef, result).as_long() == 0xFFFF
    assert constraints is None

    lhs = SymbolicExpr.create("const", 2, UINT32)
    rhs = SymbolicExpr.create("const", 4, UINT32)
    expr = SymbolicExpr.create("left_shift", lhs, rhs)

    result, constraints = expr.eval(simplify_constraints=False)

    assert cast(IntNumRef, result).as_long() == 32
    assert constraints is None


def test_bitwise_bool_expr_eval():
    """
    Test short circuiting behavior of bitwise_and and bitwise_or operators which do not
    go through bitvector conversion like other binary operators.
    """
    a = SymbolicExpr.create("const", 1, INT32)
    b = SymbolicExpr.create("const", 2, INT32)
    cond_true = SymbolicExpr.create("less", a, b)
    cond_false = SymbolicExpr.create("greater", a, b)

    and_expr = SymbolicExpr.create("bitwise_and", cond_true, cond_false)
    result, constraints = and_expr.eval(simplify_constraints=False)
    assert isinstance(result, BoolRef)
    assert str(result) == "False"
    assert constraints is None

    or_expr = SymbolicExpr.create("bitwise_or", cond_true, cond_false)
    result, constraints = or_expr.eval(simplify_constraints=False)
    assert isinstance(result, BoolRef)
    assert str(result) == "True"
    assert constraints is None

    where_expr = SymbolicExpr.create(
        "where",
        and_expr,
        SymbolicExpr.create("const", 1, INT32),
        SymbolicExpr.create("const", 0, INT32),
    )
    result, constraints = where_expr.eval(simplify_constraints=False)
    assert isinstance(result, list)
    assert cast(IntNumRef, result[0]).as_long() == 0
    assert constraints is None


def test_where_expr_concretize_scalar_constants():
    cond = SymbolicExpr.create("const", True, INT1)
    lhs = SymbolicExpr.create("const", 10, INT32)
    rhs = SymbolicExpr.create("const", -1, INT32)
    expr = SymbolicExpr.create(
        "where",
        cond,
        lhs,
        rhs,
    )

    concrete = expr.concretize()

    assert isinstance(concrete, SymbolicTensorValue)
    np.testing.assert_array_equal(concrete.data, np.array([10], dtype=np.int32))


def test_concrete_ashr_does_not_mutate_shared_tensor_value():
    original_data = np.array([0x80000000], dtype=np.uint32)
    value = SymbolicTensorValue(original_data.copy(), UINT32)
    shared = SymbolicExpr.create("const", value, UINT32)
    rhs = SymbolicExpr.create("const", 1, UINT32)
    cond = SymbolicExpr.create("const", True, INT1)
    expr = SymbolicExpr.create(
        "where",
        cond,
        shared,
        SymbolicExpr.create("ashr", shared, rhs),
    )

    concrete = expr.concretize()

    assert isinstance(concrete, SymbolicTensorValue)
    assert value.data.dtype == np.uint32
    np.testing.assert_array_equal(value.data, original_data)
    np.testing.assert_array_equal(concrete.data, original_data)


def test_unary_expr_concretize_vector_constants():
    value = SymbolicTensorValue(np.array([1.0, 4.0], dtype=np.float32), FLOAT32)
    arg = SymbolicExpr.create("const", value, block_type(FLOAT32, [2]))
    expr = SymbolicExpr.create("sqrt", arg)

    concrete = expr.concretize()

    assert isinstance(concrete, SymbolicTensorValue)
    np.testing.assert_allclose(concrete.data, np.array([1.0, 2.0], dtype=np.float32))


# ======== Pointer Symbolic Expr Operations Tests =========


def test_pointer_expr_addptr_eval():
    # Test that addptr scales offset by element size (100 + 3 * 4 = 112).
    base = SymbolicExpr.create("const", 100, pointer_type(INT32))
    offset = SymbolicExpr.create("const", 3, INT32)
    expr = SymbolicExpr.create("addptr", base, offset)
    result, constraints = expr.eval(simplify_constraints=False)
    assert cast(IntNumRef, result).as_long() == 112
    assert constraints is None


# ======== Reshape Symbolic Expr Operations Tests =========


@pytest.mark.parametrize(
    "op,extra",
    [
        ("splat", block_type(INT32, [2])),
        ("expand_dims", 0),
        ("broadcast", (2,)),
        ("reshape", (2,)),
        ("trans", (0,)),
    ],
)
def test_reshape_expr_eval(op: str, extra):
    # Test that reshape ops (splat, expand_dims, broadcast, reshape, trans) preserve scalar value.
    arg = SymbolicExpr.create("const", 5, INT32)
    if op == "splat":
        expr = SymbolicExpr.create(op, extra, arg)
    else:
        expr = SymbolicExpr.create(op, arg, extra)
    result, constraints = expr.eval(simplify_constraints=False)
    assert cast(IntNumRef, result).as_long() == 5
    assert constraints is None


# ======== Regression Tests: expand_dims/broadcast Shape Fix =========


def test_expand_dims_updates_shape():
    # Scalar constant has empty shape
    arg = SymbolicExpr.create("const", 5, INT32)
    assert arg.shape == ()

    # expand_dims at axis 0 should insert dimension of size 1
    expr = SymbolicExpr.create("expand_dims", arg, 0)
    assert expr.shape == (1,), f"Expected shape (1,), got {expr.shape}"

    # Test with tensor input and positive axis
    tensor_arg = SymbolicExpr.create("arange", INT32, 0, 8)
    assert tensor_arg.shape == (8,)
    expr2 = SymbolicExpr.create("expand_dims", tensor_arg, 0)
    assert expr2.shape == (1, 8), f"Expected shape (1, 8), got {expr2.shape}"
    expr3 = SymbolicExpr.create("expand_dims", tensor_arg, 1)
    assert expr3.shape == (8, 1), f"Expected shape (8, 1), got {expr3.shape}"

    # Test with negative axis
    expr4 = SymbolicExpr.create("expand_dims", tensor_arg, -1)
    assert expr4.shape == (8, 1), f"Expected shape (8, 1), got {expr4.shape}"
    expr5 = SymbolicExpr.create("expand_dims", tensor_arg, -2)
    assert expr5.shape == (1, 8), f"Expected shape (1, 8), got {expr5.shape}"


def test_broadcast_updates_shape():
    # Scalar constant has empty shape
    arg = SymbolicExpr.create("const", 5, INT32)
    assert arg.shape == ()

    # broadcast to shape (4,)
    expr = SymbolicExpr.create("broadcast", arg, (4,))
    assert expr.shape == (4,), f"Expected shape (4,), got {expr.shape}"


def test_shape_expr_concretize_vector_constants():
    value = SymbolicTensorValue(np.arange(4, dtype=np.int32), INT32)
    arg = SymbolicExpr.create("const", value, block_type(INT32, [4]))

    expanded = SymbolicExpr.create("expand_dims", arg, 0).concretize()
    assert isinstance(expanded, SymbolicTensorValue)
    np.testing.assert_array_equal(expanded.data, np.arange(4, dtype=np.int32)[None, :])

    reshaped_expr = SymbolicExpr.create(
        "reshape",
        arg,
        (
            SymbolicExpr.create("const", 2, INT32),
            SymbolicExpr.create("const", 2, INT32),
        ),
    )
    reshaped = reshaped_expr.concretize()
    assert isinstance(reshaped, SymbolicTensorValue)
    np.testing.assert_array_equal(
        reshaped.data, np.arange(4, dtype=np.int32).reshape(2, 2)
    )

    transposed = SymbolicExpr.create("trans", reshaped_expr, (1, 0)).concretize()
    assert isinstance(transposed, SymbolicTensorValue)
    np.testing.assert_array_equal(
        transposed.data, np.arange(4, dtype=np.int32).reshape(2, 2).T
    )

    scalar = SymbolicExpr.create("const", 7, INT32)
    broadcast = SymbolicExpr.create("broadcast", scalar, (2,)).concretize()
    assert isinstance(broadcast, SymbolicTensorValue)
    np.testing.assert_array_equal(broadcast.data, np.array([7, 7], dtype=np.int32))


# ======== Block Pointer Symbolic Expr Tests =========


def test_make_block_ptr_symbolic_expr_creation():
    """Create MakeBlockPtrSymbolicExpr with concrete values, verify children and attributes."""
    base = SymbolicExpr.create("const", 1000, pointer_type(FLOAT32))
    shape_list = [SymbolicExpr.create("const", 64, INT32)]
    stride_list = [SymbolicExpr.create("const", 1, INT32)]
    offset_list = [SymbolicExpr.create("const", 0, INT32)]
    block_shape_vals = [32]
    order_vals = [0]

    expr = SymbolicExpr.create(
        "make_block_ptr",
        base,
        shape_list,
        stride_list,
        offset_list,
        block_shape_vals,
        order_vals,
    )
    assert expr.op == "make_block_ptr"
    assert expr.ndim == 1
    assert expr.block_shape_values == [32]
    assert expr.order_values == [0]
    assert expr.base is base
    assert expr.shape_keys == ["shape_0"]
    assert expr.stride_keys == ["stride_0"]
    assert expr.offset_keys == ["offset_0"]
    assert getattr(expr, "shape_0") is not None
    assert getattr(expr, "stride_0") is not None
    assert getattr(expr, "offset_0") is not None


def test_advance_symbolic_expr_creation():
    """Create Advance wrapping MakeBlockPtr, verify children."""
    base = SymbolicExpr.create("const", 1000, pointer_type(FLOAT32))
    shape_list = [SymbolicExpr.create("const", 64, INT32)]
    stride_list = [SymbolicExpr.create("const", 1, INT32)]
    offset_list = [SymbolicExpr.create("const", 0, INT32)]

    mbp = SymbolicExpr.create(
        "make_block_ptr",
        base,
        shape_list,
        stride_list,
        offset_list,
        [32],
        [0],
    )

    delta = SymbolicExpr.create("const", 32, INT32)
    adv = SymbolicExpr.create("advance", mbp, [delta])
    assert adv.op == "advance"
    assert adv.ndim == 1
    assert adv.delta_keys == ["delta_0"]
    assert adv.ptr is mbp
    assert getattr(adv, "delta_0") is delta


def test_tensor_pointer_load_z3_eval():
    """Create make_block_ptr -> tensor_pointer_load, call .eval(), verify Z3 expression."""
    base = SymbolicExpr.create("const", 1000, pointer_type(FLOAT32))
    shape_list = [SymbolicExpr.create("const", 64, INT32)]
    stride_list = [SymbolicExpr.create("const", 1, INT32)]
    offset_list = [SymbolicExpr.create("const", 0, INT32)]

    mbp = SymbolicExpr.create(
        "make_block_ptr",
        base,
        shape_list,
        stride_list,
        offset_list,
        [32],
        [0],
    )

    load_expr = SymbolicExpr.create("tensor_pointer_load", mbp, (0,))
    z3_expr, constraints = load_expr.eval()

    # Should have Z3 expression with base address (1000) and block index variable
    z3_str = str(z3_expr)
    assert "1000" in z3_str, f"Expected base address in Z3 expr, got: {z3_str}"
    assert "blk_k_0" in z3_str, f"Expected block index variable, got: {z3_str}"

    # Constraints should include block index bounds and boundary check
    assert constraints is not None
    constr_str = str(constraints)
    assert "blk_k_0" in constr_str


def test_resolve_block_ptr_through_advance_chain():
    """Create make_block_ptr -> advance -> advance, verify accumulated offsets."""
    from tilelens.clients.symbolic_engine import TensorPointerLoadSymbolicExpr

    base = SymbolicExpr.create("const", 1000, pointer_type(FLOAT32))
    shape_list = [SymbolicExpr.create("const", 128, INT32)]
    stride_list = [SymbolicExpr.create("const", 1, INT32)]
    offset_list = [SymbolicExpr.create("const", 0, INT32)]

    mbp = SymbolicExpr.create(
        "make_block_ptr",
        base,
        shape_list,
        stride_list,
        offset_list,
        [32],
        [0],
    )

    delta1 = SymbolicExpr.create("const", 32, INT32)
    adv1 = SymbolicExpr.create("advance", mbp, [delta1])

    delta2 = SymbolicExpr.create("const", 32, INT32)
    adv2 = SymbolicExpr.create("advance", adv1, [delta2])

    # Create a load to use its _resolve_block_ptr_components
    load_expr = SymbolicExpr.create("tensor_pointer_load", adv2, ())
    assert isinstance(load_expr, TensorPointerLoadSymbolicExpr)

    (
        resolved_base,
        shapes,
        strides,
        offsets,
        bs,
    ) = load_expr._resolve_block_ptr_components(adv2)
    assert resolved_base is base
    assert bs == [32]

    # Offsets should be accumulated: 0 + 32 + 32 = 64
    off_z3, _ = offsets[0].eval()
    assert cast(IntNumRef, off_z3).as_long() == 64


# ======== ReduceSymbolicExpr Shape Tests ===========


def _make_block_expr(scalar_ty, shape):
    """Helper: create a ConstSymbolicExpr with the given block dtype."""
    return ConstSymbolicExpr("const", value=0, dtype=block_type(scalar_ty, shape))


def test_dtype_is_always_scalar():
    """dtype must always be a scalar type, never a type spec."""
    expr = _make_block_expr(FLOAT32, [4, 8])
    assert not isinstance(expr.dtype, SymbolicTypeSpec)
    assert expr.dtype == FLOAT32
    assert expr.shape == (4, 8)
    full = block_type(expr.dtype, list(expr.shape))
    assert isinstance(full, SymbolicTypeSpec)


def test_reduce_shape_2d_axis1():
    """tl.sum(x, axis=1) on [4, 8] -> [4]."""
    inp = _make_block_expr(FLOAT32, [4, 8])
    assert inp.shape == (4, 8)

    reduced = ReduceSymbolicExpr("sum", inp, axis=1)
    assert reduced.shape == (4,)


def test_reduce_shape_2d_axis0():
    """tl.sum(x, axis=0) on [4, 8] -> [8]."""
    inp = _make_block_expr(FLOAT32, [4, 8])
    reduced = ReduceSymbolicExpr("sum", inp, axis=0)
    assert reduced.shape == (8,)


def test_reduce_shape_negative_axis():
    """tl.sum(x, axis=-1) on [4, 8] -> [4]."""
    inp = _make_block_expr(FLOAT32, [4, 8])
    reduced = ReduceSymbolicExpr("sum", inp, axis=-1)
    assert reduced.shape == (4,)


def test_reduce_shape_2d_axis1_keepdims():
    """tl.sum(x, axis=1, keepdims=True) on [4, 8] -> [4, 1]."""
    inp = _make_block_expr(FLOAT32, [4, 8])
    reduced = ReduceSymbolicExpr("sum", inp, axis=1, keepdims=True)
    assert reduced.shape == (4, 1)


def test_reduce_shape_2d_axis0_keepdims():
    """tl.sum(x, axis=0, keepdims=True) on [4, 8] -> [1, 8]."""
    inp = _make_block_expr(FLOAT32, [4, 8])
    reduced = ReduceSymbolicExpr("sum", inp, axis=0, keepdims=True)
    assert reduced.shape == (1, 8)


def test_reduce_shape_negative_axis_keepdims():
    """tl.sum(x, axis=-1, keepdims=True) on [4, 8] -> [4, 1]."""
    inp = _make_block_expr(FLOAT32, [4, 8])
    reduced = ReduceSymbolicExpr("sum", inp, axis=-1, keepdims=True)
    assert reduced.shape == (4, 1)


def test_reduce_shape_1d_to_scalar():
    """tl.sum(x, axis=0) on [8] -> scalar ()."""
    inp = _make_block_expr(FLOAT32, [8])
    assert inp.shape == (8,)

    reduced = ReduceSymbolicExpr("sum", inp, axis=0)
    assert reduced.shape == ()


def test_reduce_shape_all_axes():
    """tl.sum(x) with axis=None on [4, 8] -> scalar ()."""
    inp = _make_block_expr(FLOAT32, [4, 8])
    reduced = ReduceSymbolicExpr("sum", inp, axis=None)
    assert reduced.shape == ()


def test_reduce_shape_preserves_scalar_type():
    """Reduced dtype keeps the original scalar type (e.g. float16)."""
    inp = _make_block_expr(FLOAT16, [4, 8])
    reduced = ReduceSymbolicExpr("sum", inp, axis=1)
    assert reduced.shape == (4,)
    assert reduced.dtype == FLOAT16


def test_reduce_shape_max_min():
    """tl.max and tl.min compute the same output shape as tl.sum."""
    inp = _make_block_expr(FLOAT32, [4, 8])

    for op in ("max", "min"):
        reduced = ReduceSymbolicExpr(op, inp, axis=1)
        assert reduced.shape == (4,), f"op={op}: expected (4,), got {reduced.shape}"


def test_symbolic_sort_overrider_preserves_input_shape_and_dtype():
    class DummySymbolicClient(SymbolicClient):
        def pre_warmup_callback(self, jit_fn, *args, **kwargs) -> bool:
            return False

        def post_warmup_callback(self, jit_fn, ret) -> None:
            pass

    inp = _make_block_expr(FLOAT32, [4, 8])
    callback = DummySymbolicClient().register_op_callback(Sort).op_overrider

    assert callback is not None
    result = callback(inp, dim=1, descending=True, stable=True)

    assert result is not inp
    assert result.has_op("sort")
    assert result.dtype == FLOAT32
    assert result.shape == (4, 8)


# ======== LoadSymbolicExpr dtype Tests ===========


def test_load_dtype_block_of_pointers():
    """tl.load on a block of pointers should produce a block of the pointed-to type.

    ptr dtype: block_type(pointer<fp32>, [1, 16])
    expected load dtype: block_type(fp32, [1, 16])
    """
    ptr = ConstSymbolicExpr(
        "const", value=0, dtype=block_type(pointer_type(FLOAT32), [1, 16])
    )
    load = LoadSymbolicExpr("load", ptr)
    assert load.shape == (1, 16), f"Expected shape (1, 16), got {load.shape}"
    assert load.dtype == FLOAT32, f"Expected FLOAT32, got {load.dtype}"


def test_store_dtype_block_of_pointers():
    """tl.store on a block of pointers derives the pointee dtype and shape.

    ptr dtype: block_type(pointer<fp32>, [1, 16])
    expected store dtype: fp32
    """
    ptr = ConstSymbolicExpr(
        "const", value=0, dtype=block_type(pointer_type(FLOAT32), [1, 16])
    )
    value = ConstSymbolicExpr("const", value=0, dtype=block_type(FLOAT32, [1, 16]))
    store = StoreSymbolicExpr("store", ptr, value)
    assert store.dtype == FLOAT32, f"Expected FLOAT32, got {store.dtype}"
    assert store.shape == (1, 16), f"Expected shape (1, 16), got {store.shape}"


# ======== Truthiness and Hook Slot Tests ===========


def test_frontend_plumbing_truthiness_is_object_truthiness():
    # semantic.py's `other.handle if other else None` must keep a falsy
    # user-provided other=0; kernel code still branches on the value.
    falsy = ConstSymbolicExpr("const", value=0, dtype=INT32).data
    for plumbing_file in _triton_frame_dirs()[2]:
        namespace: dict = {}
        probe = compile("def probe(x):\n    return bool(x)\n", plumbing_file, "exec")
        exec(probe, namespace)
        assert namespace["probe"](falsy) is True
    assert bool(falsy) is False


def test_scalar_concretize_observer_sees_every_scalar_read(monkeypatch):
    seen: list = []
    monkeypatch.setattr(SymbolicExpr, "_scalar_concretize_observer", seen.append)
    data = ConstSymbolicExpr("const", value=3, dtype=INT32).data
    assert (int(data), int(data.squeeze()), int(data[0]), bool(data)) == (3, 3, 3, True)
    assert seen == [data.symbolic_expr] * 4


def test_value_providers_dispatch(monkeypatch):
    ptr = SymbolicExpr.create("const", 0, pointer_type(INT32))
    one = SymbolicExpr.create("const", 1, INT32)

    def exprs():
        return (
            SymbolicExpr.create("load", ptr),
            SymbolicExpr.create("atomic_cas", ptr, one, one),
            SymbolicExpr.create("atomic_rmw", ptr, one, None),
        )

    load, cas, rmw = exprs()
    assert load._to_z3() == (0, None)  # legacy pointer-as-value
    for atomic in (cas, rmw):
        with pytest.raises(NotImplementedError):
            atomic._to_z3()
    value, old = Int("value"), Int("old")
    monkeypatch.setattr(SymbolicExpr, "_load_value_provider", lambda e: (value, None))
    monkeypatch.setattr(SymbolicExpr, "_atomic_value_provider", lambda e: (old, None))
    load, cas, rmw = exprs()
    assert [e._to_z3()[0] for e in (load, cas, rmw)] == [value, old, old]


def test_hook_slot_reclaim(monkeypatch):
    client = _LoopSiteSymbolicClient()
    slots = SymbolicExpr._HOOK_SLOTS

    def install():
        client.loop_stack.append(object())  # a dead launch's loop context
        for slot, owner in zip(slots, (id(client), id(client), id(object()))):
            monkeypatch.setattr(SymbolicExpr, slot, print)
            monkeypatch.setattr(SymbolicExpr, f"{slot}_owner", owner)

    install()
    client.abort_launch(RuntimeError())  # only the aborting client's slots
    assert [getattr(SymbolicExpr, s) for s in slots] == [None, None, print]
    install()
    client.grid_callback((1, 1, 1))  # every slot, whoever installed it
    assert client.loop_stack == []
    assert {getattr(SymbolicExpr, s) for s in slots} == {None}
    assert {getattr(SymbolicExpr, f"{s}_owner") for s in slots} == {None}
