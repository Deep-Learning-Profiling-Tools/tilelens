import pytest
import numpy as np
import triton.language as tl
from triton.runtime.interpreter import TensorHandle

from tilelens.core.callbacks import OpCallbacks
from tilelens.core.client import Client, ClientManager
from tilelens.core.data import (
    AddPtr,
    AtomicCas,
    AtomicRMW,
    BinaryOp,
    CastImpl,
    Dot,
    IntToPtr,
    Load,
    ProgramId,
    PtrToInt,
    ReduceSum,
    Store,
    TernaryOp,
    UnaryOp,
)

from tilelens.core.frontend.base import AdapterResult, get_frontend
from tilelens.core.frontend.nki import HAS_NKI
from tilelens.core.symbolic_metadata import (
    FLOAT32,
    FLOAT8_E4B8,
    FLOAT8_E5B16,
    INT32,
    SymbolicTensorValue,
    SymbolicTypeSpec,
    TensorDescriptorAccess,
    pointer_type,
)

from tilelens.core.patch import PatchOp

TRITON_FRONTEND = get_frontend("triton")
TRITON_ADAPTERS = TRITON_FRONTEND.adapters
NKI_ADAPTERS = get_frontend("nki").adapters
GLUON_FRONTEND = get_frontend("gluon")
GLUON_ADAPTERS = GLUON_FRONTEND.adapters


def test_adapter_result_kwargs_copy():
    """Test that AdapterResult SHALLOW-copies kwargs."""
    kwargs = {"named": "value"}
    result = AdapterResult(1, **kwargs)
    kwargs[
        "named"
    ] = "updated"  # result.kwargs["named"] is immutable str so it doesn't change from this
    assert result.args == (1,)
    assert result.kwargs == {"named": "value"}

    mutable_object = ["before"]
    kwargs = {"named": mutable_object}
    result = AdapterResult(1, **kwargs)
    kwargs["named"][
        0
    ] = "after"  # result.kwargs["named"] is mutable so it DOES change from this
    assert result.args == (1,)
    assert result.kwargs["named"][0] == "after"


def test_patch_op_preserves_descriptor_binding():
    seen = []

    class Target:
        def op(self, value):
            return ("ret", self, value)

    def adapter(instance, value):
        return AdapterResult(instance, value)

    original = Target.op
    callbacks = OpCallbacks(
        before_callback=lambda instance, value: seen.append((instance, value))
    )
    Target.op = PatchOp(
        original,
        ProgramId,
        callbacks,
        adapter,
        run_op_overrider=lambda op, _op_type, _op_overrider, args, kwargs: op(
            *args,
            **kwargs,
        ),
        maybe_yield_for_multism=lambda: None,
    )
    try:
        instance = Target()
        assert instance.op(7) == ("ret", instance, 7)
        assert seen == [(instance, 7)]
    finally:
        Target.op = original


def test_triton_store_adapter_handles_keys():
    """Adapter mirrors tl.store(ptr, value, mask=..., cache_modifier=..., eviction_policy=..., keys=...)."""
    adapter = TRITON_ADAPTERS[Store]
    ptr, value, mask, keys = object(), object(), object(), object()

    cache_modifier = "wb"
    eviction_policy = "evict_last"

    full = adapter(
        ptr,
        value,
        mask,
        cache_modifier=cache_modifier,
        eviction_policy=eviction_policy,
        keys=keys,
    )
    assert isinstance(full, AdapterResult)
    assert full.args == (ptr, mask, keys)
    assert full.kwargs == {}

    minimal = adapter(ptr, value, mask)
    assert minimal.args == (ptr, mask, None)
    assert minimal.kwargs == {}


def test_triton_load_adapter_keys_passthrough():
    """Adapter mirrors tl.load(ptr, mask=..., other=..., keys=...)."""
    adapter = TRITON_ADAPTERS[Load]
    ptr, mask, other, keys = object(), object(), object(), object()

    cache_modifier = "wb"
    eviction_policy = "evict_last"
    is_volatile = False

    full = adapter(
        ptr,
        mask,
        other,
        cache_modifier=cache_modifier,
        eviction_policy=eviction_policy,
        is_volatile=is_volatile,
        keys=keys,
    )
    assert full.args == (ptr, mask, keys)
    assert full.kwargs == {}

    minimal = adapter(ptr, mask, other)
    assert minimal.args == (ptr, mask, None)
    assert minimal.kwargs == {}


def test_triton_reduce_adapter_supports_positional_and_keyword_axis():
    """Adapter retains axis/keep_dims regardless of positional or keyword usage."""
    adapter = TRITON_ADAPTERS[ReduceSum]
    tensor = object()
    axis = 1
    keep_dims = True

    positional = adapter(tensor, axis, keep_dims)
    assert positional.args == (tensor, axis, keep_dims)
    assert positional.kwargs == {}

    keyword = adapter(tensor, axis=2, keep_dims=False)
    assert keyword.args == (tensor, 2, False)
    assert keyword.kwargs == {}


def test_triton_addptr_adapter_orders_arguments():
    """Adapter maps tl.addptr(ptr, offset) to (ptr, offset)."""
    adapter = TRITON_ADAPTERS[AddPtr]
    ptr = object()
    offset = object()
    result = adapter(ptr, offset)
    assert result.args == (ptr, offset)
    assert result.kwargs == {}


def test_triton_atomic_rmw_adapter_returns_ptr_and_mask():
    """Adapter maps create_atomic_rmw(rmw_op, ptr, val, mask, sem, scope) to (ptr, mask)."""
    adapter = TRITON_ADAPTERS[AtomicRMW]
    rmw_op, ptr, val, mask, sem, scope = (object() for _ in range(6))
    result = adapter(rmw_op, ptr, val, mask, sem, scope)
    assert result.args == (ptr, mask)
    assert result.kwargs == {}


def test_triton_atomic_cas_adapter_returns_ptr_only():
    """Adapter maps create_atomic_cas(ptr, cmp, val, sem, scope) to (ptr,)."""
    adapter = TRITON_ADAPTERS[AtomicCas]
    ptr, cmp, val, sem, scope = (object() for _ in range(5))
    result = adapter(ptr, cmp, val, sem, scope)
    assert result.args == (ptr,)
    assert result.kwargs == {}


def test_program_id_adapter_returns_axis_only():
    """Adapter maps tl.program_id(axis) to (axis,)."""
    adapter = TRITON_ADAPTERS[ProgramId]
    program_id = 0
    result = adapter(program_id)
    assert result.args == (program_id,)
    assert result.kwargs == {}


def test_gluon_descriptor_memory_adapters_accept_async_signatures():
    load = GLUON_ADAPTERS[Load]
    store = GLUON_ADAPTERS[Store]
    from tilelens.core.frontend import gluon as gluon_frontend

    normal_descriptor_load = gluon_frontend._gluon_descriptor_load_adapter(
        3,
        descriptor_kwarg="tensor_desc",
        coords_kwarg="coord",
    )
    im2col_descriptor_load = gluon_frontend._gluon_descriptor_load_adapter(
        4,
        descriptor_kwarg="tensor_desc",
        coords_kwarg="coord",
    )
    tdm_descriptor_load = gluon_frontend._gluon_descriptor_load_adapter(
        2,
        descriptor_kwarg="src",
        coords_kwarg="offsets",
    )
    desc = type("TensorDescriptor", (), {"block_shape": (1,)})()

    def assert_descriptor_access(result, coords, mask=None):
        access, result_mask, other = result.args
        assert access.descriptor is desc
        assert access.coords == coords
        assert access.pred == mask
        assert result_mask == mask
        assert other is None

    assert load("ptr", "mask").args == ("ptr", "mask", None)
    assert_descriptor_access(
        normal_descriptor_load(desc, "coord", "bar", "result"),
        "coord",
    )
    assert_descriptor_access(
        normal_descriptor_load(desc, "coord", "bar", "result", "pred"),
        "coord",
        "pred",
    )
    assert_descriptor_access(
        im2col_descriptor_load(
            desc,
            "coord",
            "offsets",
            "bar",
            "result",
            "pred",
        ),
        "coord",
        "pred",
    )
    assert_descriptor_access(
        tdm_descriptor_load(desc, "offsets", "dest"),
        "offsets",
    )
    assert_descriptor_access(
        tdm_descriptor_load(desc, "offsets", "dest", "pred"),
        "offsets",
        "pred",
    )
    assert_descriptor_access(
        normal_descriptor_load(desc, "coord", "bar", "result", pred="pred"),
        "coord",
        "pred",
    )
    assert_descriptor_access(
        normal_descriptor_load(desc, coord="kw_coord", barrier="bar", result="result"),
        "kw_coord",
    )
    assert_descriptor_access(
        normal_descriptor_load(
            tensor_desc=desc,
            coord="kw_coord",
            barrier="bar",
            result="result",
            pred="kw_pred",
        ),
        "kw_coord",
        "kw_pred",
    )
    assert_descriptor_access(
        tdm_descriptor_load(src=desc, offsets="kw_offsets", dest="dest"),
        "kw_offsets",
    )

    assert store("ptr", "value", "mask").args == ("ptr", "mask", None)
    assert_descriptor_access(store(desc, "coord", "src"), "coord")
    assert_descriptor_access(
        store(desc, "values", "indices", "axis", "mask"),
        "values",
    )
    assert_descriptor_access(
        store(desc, "values", "indices", "axis", mask="mask"),
        "values",
        "mask",
    )


def test_gluon_descriptor_load_adapters_honor_positional_predicates():
    from tilelens.core.frontend import gluon as gluon_frontend

    desc = type("TensorDescriptor", (), {"block_shape": (1,)})()
    normal_load = gluon_frontend._gluon_descriptor_load_adapter(
        3,
        descriptor_kwarg="tensor_desc",
        coords_kwarg="coord",
    )
    im2col_load = gluon_frontend._gluon_descriptor_load_adapter(
        4,
        descriptor_kwarg="tensor_desc",
        coords_kwarg="coord",
    )
    tdm_load = gluon_frontend._gluon_descriptor_load_adapter(
        2,
        descriptor_kwarg="src",
        coords_kwarg="offsets",
    )

    access, mask, other = normal_load(desc, "coord", "barrier", "result", False).args
    assert access.pred is False
    assert mask is False
    assert other is None

    access, mask, other = im2col_load(
        desc, "coord", "offsets", "barrier", "result"
    ).args
    assert access.pred is None
    assert mask is None
    assert other is None

    access, mask, other = im2col_load(
        desc, "coord", "offsets", "barrier", "result", False
    ).args
    assert access.pred is False
    assert mask is False
    assert other is None

    access, mask, other = tdm_load(desc, "offsets", "dest", False).args
    assert access.pred is False
    assert mask is False
    assert other is None


def test_gluon_shared_memory_descriptor_class_is_not_an_op():
    from triton.experimental.gluon.language import _core as gluon_core

    assert "shared_memory_descriptor" not in GLUON_FRONTEND.namespaces[gluon_core]


def test_gluon_load_store_adapters_keep_first_positional_mask():
    load = GLUON_ADAPTERS[Load]
    store = GLUON_ADAPTERS[Store]

    assert load("ptr", "mask", "other", "boundary_check").args == (
        "ptr",
        "mask",
        None,
    )
    assert store("ptr", "value", "mask", "boundary_check").args == (
        "ptr",
        "mask",
        None,
    )


def test_gluon_load_adapter_does_not_treat_pred_as_mask():
    from tilelens.core.frontend import gluon as gluon_frontend

    assert gluon_frontend._gluon_load_adapter("ptr", pred="pred").args == (
        "ptr",
        None,
        None,
    )


def test_gluon_frontend_routes_tensor_descriptor_access_to_overrider(monkeypatch):
    from tilelens.core.frontend import gluon as gluon_frontend

    class FakeTensor:
        dtype = "float32"

        def data_ptr(self):
            return 4096

    class TensorDescriptor:
        base = FakeTensor()
        shape = (4, 4)
        strides = (4, 1)
        block_shape = (2, 2)

    calls = []

    def original(desc, coords):
        raise AssertionError("run_op_overrider should not call the original op")

    def overrider(ptr, mask, other):
        calls.append((ptr, mask, other))
        return "overridden"

    monkeypatch.setitem(
        gluon_frontend.GLUON_CALLABLE_ADAPTERS,
        original,
        gluon_frontend._gluon_descriptor_load_adapter(
            1,
            descriptor_kwarg="desc",
            coords_kwarg="coords",
        ),
    )

    desc = TensorDescriptor()
    result = GLUON_FRONTEND.run_op_overrider(
        original,
        Load,
        overrider,
        (desc, (3, 0)),
        {},
    )

    assert result == "overridden"
    assert len(calls) == 1
    ptr, mask, other = calls[0]
    assert isinstance(ptr, TensorDescriptorAccess)
    assert ptr.descriptor is desc
    assert ptr.coords == (3, 0)
    assert ptr.pred is None
    assert mask is None
    assert other is None


def test_gluon_binary_op_adapter_binds_original_callable():
    from triton.experimental.gluon import language as ttgl

    calls = []

    def overrider(lhs, rhs, op):
        calls.append((lhs, rhs, op))
        return "overridden"

    result = GLUON_FRONTEND.run_op_overrider(
        ttgl.add,
        BinaryOp,
        overrider,
        ("lhs", "rhs"),
        {},
    )

    assert result == "overridden"
    assert calls == [("lhs", "rhs", np.add)]


def test_gluon_unary_op_adapter_binds_original_callable():
    from triton.experimental.gluon import language as ttgl

    calls = []

    def overrider(arg, op):
        calls.append((arg, op))
        return "overridden"

    result = GLUON_FRONTEND.run_op_overrider(
        ttgl.exp,
        UnaryOp,
        overrider,
        ("arg",),
        {},
    )

    assert result == "overridden"
    assert calls == [("arg", np.exp)]


def test_gluon_semantic_binary_adapter_drops_bound_semantic():
    from triton.experimental.gluon.language import _semantic as gluon_semantic

    calls = []

    def overrider(lhs, rhs):
        calls.append((lhs, rhs))
        return "overridden"

    result = GLUON_FRONTEND.run_op_overrider(
        gluon_semantic.GluonSemantic.add,
        AddPtr,
        overrider,
        (object(), "lhs", "rhs", True),
        {},
    )

    assert result == "overridden"
    assert calls == [("lhs", "rhs")]


def test_gluon_semantic_comparison_adapter_binds_original_callable():
    from triton.experimental.gluon.language import _semantic as gluon_semantic

    calls = []

    def overrider(lhs, rhs, op):
        calls.append((lhs, rhs, op))
        return "overridden"

    result = GLUON_FRONTEND.run_op_overrider(
        gluon_semantic.GluonSemantic.less_than,
        BinaryOp,
        overrider,
        (object(), "lhs", "rhs"),
        {},
    )

    assert result == "overridden"
    assert calls == [("lhs", "rhs", np.less)]


def test_gluon_where_adapter_binds_original_callable():
    from triton.experimental.gluon import language as ttgl

    calls = []

    def overrider(condition, lhs, rhs, op):
        calls.append((condition, lhs, rhs, op))
        return "overridden"

    result = GLUON_FRONTEND.run_op_overrider(
        ttgl.where,
        TernaryOp,
        overrider,
        ("condition", "lhs", "rhs"),
        {},
    )

    assert result == "overridden"
    assert calls == [("condition", "lhs", "rhs", np.where)]


def test_gluon_dot_adapter_preserves_accumulator_for_overriders():
    from triton.experimental.gluon import language as ttgl

    calls = []

    def overrider(a, b, acc, input_precision, max_num_imprecise_acc):
        calls.append((a, b, acc, input_precision, max_num_imprecise_acc))
        return "overridden"

    result = GLUON_FRONTEND.run_op_overrider(
        ttgl.dot_fma,
        Dot,
        overrider,
        ("a", "b", "acc"),
        {},
    )

    assert result == "overridden"
    assert calls == [("a", "b", "acc", None, None)]


def test_gluon_frontend_wraps_symbolic_concrete_fn():
    calls = []

    def concrete_fn(arg, *, named):
        calls.append((arg, named))
        return [arg, named]

    arg = object()
    named = object()
    wrapped = GLUON_FRONTEND.wrap_symbolic_concrete_fn(concrete_fn)

    assert wrapped(arg, named=named) == [arg, named]
    assert calls == [(arg, named)]


def test_gluon_language_ops_patch_and_use_simulation_callable():
    from triton.experimental.gluon import language as ttgl
    from tilelens.core.simulation import gluon as gluon_sim

    seen_axes = []
    original_program_id = ttgl.program_id

    class RecordingClient(Client):
        NAME = "recording"

        def pre_run_callback(self, fn):
            return True

        def post_run_callback(self, fn):
            return True

        def arg_callback(self, name, arg, arg_cvt):
            pass

        def grid_callback(self, grid):
            pass

        def grid_idx_callback(self, grid_idx):
            pass

        def register_op_callback(self, op_type):
            if op_type is ProgramId:
                return OpCallbacks(before_callback=lambda axis: seen_axes.append(axis))
            return OpCallbacks()

        def register_for_loop_callback(self):
            from tilelens.core.callbacks import ForLoopCallbacks

            return ForLoopCallbacks()

        def finalize(self):
            return []

        def pre_warmup_callback(self, jit_fn, *args, **kwargs):
            return False

        def post_warmup_callback(self, jit_fn, ret):
            pass

    def kernel():
        pass

    gluon_sim.gluon_builder.set_grid_dim(1, 1, 1)
    gluon_sim.gluon_builder.set_grid_idx(0, 0, 0)
    manager = ClientManager([RecordingClient()])
    with manager.patch_run(kernel, frontend_name="gluon"):
        assert ttgl.program_id is not original_program_id
        result = ttgl.program_id(2)

    assert ttgl.program_id is original_program_id
    assert seen_axes == [2]
    assert result.handle.data.tolist() == [0]


def test_triton_pointer_cast_aliases_have_distinct_op_types():
    frontend = get_frontend("triton")
    builder_ops = frontend.namespaces[frontend.builder]

    assert builder_ops["cast_impl"] is CastImpl
    assert builder_ops["create_ptr_to_int"] is PtrToInt
    assert builder_ops["create_int_to_ptr"] is IntToPtr


def test_triton_frontend_normalizes_block_pointer_type():
    frontend = get_frontend("triton")

    spec = frontend.normalize_symbolic_value(
        tl.block_type(tl.pointer_type(tl.float32), [1, 16])
    )

    assert isinstance(spec, SymbolicTypeSpec)
    assert spec.dtype == pointer_type(FLOAT32)
    assert spec.shape == (1, 16)


@pytest.mark.parametrize(
    "triton_dtype,symbolic_dtype",
    [
        (tl.float8e4b8, FLOAT8_E4B8),
        (tl.float8e5b16, FLOAT8_E5B16),
    ],
)
def test_triton_frontend_normalizes_other_fp8_dtypes(triton_dtype, symbolic_dtype):
    frontend = get_frontend("triton")

    assert frontend.normalize_symbolic_value(triton_dtype) == symbolic_dtype

    spec = frontend.normalize_symbolic_value(
        tl.block_type(tl.pointer_type(triton_dtype), [2, 4])
    )
    assert isinstance(spec, SymbolicTypeSpec)
    assert spec.dtype == pointer_type(symbolic_dtype)
    assert spec.shape == (2, 4)

    handle = TensorHandle(np.array([1, 2], dtype=np.uint8), triton_dtype)
    normalized = frontend.normalize_symbolic_value(handle)
    assert isinstance(normalized, SymbolicTensorValue)
    assert normalized.dtype == symbolic_dtype
    np.testing.assert_array_equal(normalized.data, handle.data)

    round_tripped = frontend.to_frontend_symbolic_value(normalized)
    assert isinstance(round_tripped, TensorHandle)
    assert round_tripped.dtype == triton_dtype
    np.testing.assert_array_equal(round_tripped.data, handle.data)


def test_triton_frontend_normalizes_tensor_and_round_trips():
    frontend = get_frontend("triton")
    handle = TensorHandle(np.array([7], dtype=np.int32), tl.int32)
    tensor = tl.core.tensor(handle, tl.int32)

    normalized = frontend.normalize_symbolic_value(tensor)

    assert isinstance(normalized, SymbolicTensorValue)
    assert normalized.dtype == INT32
    np.testing.assert_array_equal(normalized.data, np.array([7], dtype=np.int32))

    round_tripped = frontend.to_frontend_symbolic_value(normalized)
    assert isinstance(round_tripped, TensorHandle)
    assert round_tripped.dtype == tl.int32
    np.testing.assert_array_equal(round_tripped.data, handle.data)


def test_triton_frontend_wraps_symbolic_concrete_fn():
    frontend = get_frontend("triton")

    def concrete_fn(value, dtype):
        assert isinstance(value, TensorHandle)
        assert dtype == tl.float32
        return TensorHandle(value.data.astype(np.float32), dtype)

    wrapped = frontend.wrap_symbolic_concrete_fn(concrete_fn)
    result = wrapped(
        SymbolicTensorValue(np.array([3], dtype=np.int32), INT32),
        FLOAT32,
    )

    assert isinstance(result, SymbolicTensorValue)
    assert result.dtype == FLOAT32
    np.testing.assert_array_equal(result.data, np.array([3], dtype=np.float32))


@pytest.mark.skipif(not HAS_NKI, reason="NKI extras not installed")
def test_nki_load_store_adapters_align_with_clients():
    """NKI adapters normalize masked load/store to (tensor, mask, keys)."""
    src = object()
    dst = object()
    keys = object()
    mask = object()
    value = object()

    load_result = NKI_ADAPTERS[Load](src, keys, mask=mask)
    assert load_result.args == (src, mask, keys)
    assert load_result.kwargs == {}

    store_result = NKI_ADAPTERS[Store](dst, keys, value, mask=mask)
    assert store_result.args == (dst, mask, keys)
    assert store_result.kwargs == {}


def test_patchop_uses_adapter_for_callbacks():
    """PatchOp must feed the adapted arguments into before/after callbacks."""
    before_log: list[tuple[str, tuple, dict]] = []
    after_log: list[tuple[str, tuple, dict]] = []

    def before_callback(*args, **kwargs):
        before_log.append(("before", args, kwargs))

    def after_callback(ret, *args, **kwargs):
        after_log.append(("after", args, kwargs))

    adapter = TRITON_ADAPTERS[Store]
    callbacks = OpCallbacks(
        before_callback=before_callback, after_callback=after_callback
    )
    patch_op = PatchOp(
        op=lambda *_a, **_k: "return-value",
        op_type=Store,
        callbacks=callbacks,
        adapter=adapter,
        run_op_overrider=TRITON_FRONTEND.run_op_overrider,
        maybe_yield_for_multism=TRITON_FRONTEND.maybe_yield_for_multism,
    )

    ptr = object()
    mask = object()
    patch_op(ptr, "value", mask)

    expected_args = (ptr, mask, None)
    assert before_log == [("before", expected_args, {})]
    assert after_log == [("after", expected_args, {})]
