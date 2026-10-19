import inspect
from typing import Any

import numpy as np
import pytest
import triton_viz

try:
    from triton_viz.core.frontend.base import _LangPatchScope
    from triton_viz.clients import Tracer
    from triton_viz.core.data import Dot, Transfer
    from triton_viz.core.trace import launches
    from triton_viz.tools.nki_trace_dump import records_to_events
    import nki.isa as nisa
    import nki.language as nl
    import triton_viz.core.simulation.nki_beta2 as b2
    from triton_viz.utils.dtypes import STORAGE_DTYPES
except ModuleNotFoundError:
    pytest.skip(
        "NeuronX dependencies are missing. Install triton-viz[nki] to run these tests.",
        allow_module_level=True,
    )

pytestmark = pytest.mark.nki


@pytest.fixture
def patched_scope():
    """Patch beta2 interpreter symbols for one test."""
    scope = _LangPatchScope()
    b2.nki_patch_lang(scope)
    yield scope
    b2.nki_unpatch_lang(scope)


def _nd(value, dtype=None, buffer="sbuf"):
    """Create an interpreter NDArray from Python data."""
    array = np.asarray(value, dtype=dtype) if dtype is not None else np.asarray(value)
    return b2.NDArray(value=array, buffer=buffer)


def _zeros(shape, dtype=None, buffer="sbuf"):
    """Create a zero-initialized interpreter NDArray."""
    array = np.zeros(shape, dtype=dtype) if dtype is not None else np.zeros(shape)
    return b2.NDArray(value=array, buffer=buffer)


def _assert_low_precision_dma_copy(dtype_name, expected):
    """Assert dma_copy casts source values to dtype-specific storage."""
    logical = np.array([[0.1, 0.9, 1.24], [1.51, 2.9, -5.2]], dtype=np.float64)
    dtype = getattr(nl, dtype_name, dtype_name)
    dst = b2.NDArray(shape=logical.shape, dtype=dtype, buffer="sbuf")
    b2.dma_copy(dst=dst, src=_nd(logical))
    assert dst.data.dtype == STORAGE_DTYPES[dtype_name]
    assert np.array_equal(dst.data, np.array(expected, dtype=dst.data.dtype))


def _core_inputs():
    """Build canonical tensors used by core dataflow tests."""
    a = _nd(np.arange(8, dtype=float).reshape(2, 4), buffer="hbm")
    b = _nd(np.arange(8, dtype=float).reshape(2, 4) + 10, buffer="hbm")
    out = _zeros((2, 4), buffer="sbuf")
    return a, b, out


def _act_input():
    """Build canonical activation input tensor."""
    return _nd(np.array([[1.0, 2.0, 3.0], [4.0, 0.0, -1.0]]))


def _dtype_token(name):
    """Resolve a dtype name to nl token or raw string fallback."""
    return getattr(nl, name, name)


def _buffer_token_name(token: Any) -> str:
    """Resolve a buffer token to its stable string name."""
    return getattr(token, "buffer", getattr(token, "name", str(token)))


def _is_float_dtype(dtype):
    """Return whether dtype should be compared as floating-point values."""
    return "float" in str(dtype) or "bfloat" in str(dtype)


def _assert_tensor_equal(actual, expected):
    """Assert equality with float-aware tolerance handling."""
    if _is_float_dtype(actual.dtype):
        assert np.allclose(
            np.asarray(actual, dtype=np.float64),
            np.asarray(expected, dtype=np.float64),
            rtol=1e-2,
            atol=1e-2,
            equal_nan=True,
        )
        return
    assert np.array_equal(actual, expected)


def _matrix(shape, dist, dtype_name, seed):
    """Generate deterministic matrix inputs across requested distributions."""
    size = int(np.prod(shape))
    if dist == "linspace":
        data = np.linspace(-3.0, 3.0, num=size, dtype=np.float64)
    elif dist == "arange":
        data = np.arange(size, dtype=np.float64) - size // 2
    elif dist == "randn":
        data = np.random.default_rng(seed).standard_normal(size)
    else:  # pragma: no cover - defensive fallback
        raise ValueError(f"Unknown distribution: {dist}")
    if dtype_name == "int16":
        data = np.rint(data * 64.0)
    return np.asarray(data.reshape(shape), dtype=STORAGE_DTYPES[dtype_name])


def _typed_ndarray(shape, dtype_name, buffer="sbuf", dist="arange", seed=0):
    """Create an NDArray with a documented dtype token and deterministic payload."""
    return b2.NDArray(
        value=_matrix(shape, dist, dtype_name, seed),
        dtype=_dtype_token(dtype_name),
        buffer=buffer,
    )


SHAPE_CASES = ((), (1,), (17,), (19, 23), (3, 5, 7, 11, 13))
NL_DTYPE_CASES = (
    "bool_",
    "int8",
    "int16",
    "int32",
    "uint8",
    "uint16",
    "uint32",
    "float16",
    "float32",
    "bfloat16",
    "tfloat32",
    "float8_e4m3",
    "float8_e5m2",
    "float8_e4m3fn",
    "float8_e5m2fn",
)


def test_dma_copy_bfloat16_quantization():
    """dma_copy should quantize values to bfloat16 precision."""
    _assert_low_precision_dma_copy(
        "bfloat16",
        [[0.10009765625, 0.8984375, 1.2421875], [1.5078125, 2.90625, -5.1875]],
    )


def test_dma_copy_float8_e4m3_quantization():
    """dma_copy should quantize values to float8_e4m3 precision."""
    _assert_low_precision_dma_copy(
        "float8_e4m3",
        [[0.1015625, 0.875, 1.25], [1.5, 3.0, -5.0]],
    )


def test_dma_copy_float8_e5m2_quantization():
    """dma_copy should quantize values to float8_e5m2 precision."""
    _assert_low_precision_dma_copy(
        "float8_e5m2",
        [[0.09375, 0.875, 1.25], [1.5, 3.0, -5.0]],
    )


def test_dma_copy_float8_e4m3fn_quantization():
    """dma_copy should quantize values to float8_e4m3fn precision."""
    _assert_low_precision_dma_copy(
        "float8_e4m3fn",
        [[0.1015625, 0.875, 1.25], [1.5, 3.0, -5.0]],
    )


def test_dma_copy_float4_e2m1fn_quantization():
    """dma_copy should quantize values to float4_e2m1fn precision."""
    _assert_low_precision_dma_copy(
        "float4_e2m1fn",
        [[0.0, 1.0, 1.0], [1.5, 3.0, -6.0]],
    )


def test_unsupported_float8_e5m2fn_dtype_raises():
    """float8_e5m2fn should be unsupported by beta2 dtype mapping."""
    dtype = getattr(nl, "float8_e5m2fn", "float8_e5m2fn")
    with pytest.raises(TypeError, match="Unsupported dtype"):
        b2.NDArray(shape=(1, 1), dtype=dtype, buffer="sbuf")


def test_unsupported_tfloat32_dtype_raises():
    """tfloat32 should be unsupported by beta2 dtype mapping."""
    dtype = getattr(nl, "tfloat32", "tfloat32")
    with pytest.raises(TypeError, match="Unsupported dtype"):
        b2.NDArray(shape=(1, 1), dtype=dtype, buffer="sbuf")


def test_ndarray_has_no_name_attr():
    """Beta2 NDArray should not expose a name attribute."""
    assert not hasattr(b2.NDArray(value=np.array([1])), "name")


def test_patch_surface_and_signatures(patched_scope):
    """Patched surface should match this repository's beta2 dialect."""
    del patched_scope
    nl_callable_names = (
        "ndarray",
        "zeros",
        "program_id",
        "affine_range",
        "ds",
    )
    nl_value_names = ("tile_size", "sbuf", "hbm", "psum")
    nisa_names = (
        "nc_matmul",
        "nc_transpose",
        "activation",
        "exponential",
        "reciprocal",
        "dma_copy",
        "tensor_copy",
        "tensor_tensor",
        "tensor_reduce",
        "tensor_scalar",
    )

    for name in nl_callable_names:
        assert hasattr(nl, name)
    for name in nl_value_names:
        assert hasattr(nl, name)
    for name in nisa_names:
        assert hasattr(nisa, name)

    for name in nl_callable_names:
        assert str(inspect.signature(getattr(nl, name))) == str(
            inspect.signature(getattr(b2, name))
        )

    for name in nisa_names:
        assert str(inspect.signature(getattr(nisa, name))) == str(
            inspect.signature(getattr(b2, name))
        )


def test_trace_records_beta2_nc_matmul():
    """beta2 tracing should record nc_matmul via the builder surface."""
    triton_viz.clear()

    def kernel(lhsT, rhs, out):
        lhs_tile = nl.ndarray((128, 128), dtype=lhsT.dtype, buffer=nl.sbuf)
        rhs_tile = nl.ndarray((128, 512), dtype=rhs.dtype, buffer=nl.sbuf)
        res_psum = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
        out_tile = nl.ndarray((128, 512), dtype=out.dtype, buffer=nl.sbuf)
        nisa.dma_copy(lhs_tile, lhsT)
        nisa.dma_copy(rhs_tile, rhs)
        nisa.nc_matmul(dst=res_psum, stationary=lhs_tile, moving=rhs_tile)
        nisa.tensor_copy(out_tile, res_psum)
        nisa.dma_copy(out, out_tile)

    traced = triton_viz.trace(client=Tracer(), frontend="nki_beta2")(kernel)
    lhs = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
    rhs = np.arange(128 * 512, dtype=np.float32).reshape(128, 512)
    out = np.empty((128, 512), dtype=np.float32)
    traced[(1,)](lhs, rhs, out)

    grid_records = [
        record for record in launches[-1].records if type(record).__name__ == "Grid"
    ]
    assert grid_records[0].idx == (0, 0, 0)
    assert any(isinstance(record, Dot) for record in launches[-1].records)


def test_trace_records_beta2_transfers():
    """beta2 tracing should record dma_copy/tensor_copy as Transfer ops."""
    triton_viz.clear()

    # check tracer records Transfer ops
    def kernel(src, out):
        src_tile = nl.ndarray((128, 128), dtype=src.dtype, buffer=nl.sbuf)
        psum_tile = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
        out_tile = nl.ndarray((128, 128), dtype=out.dtype, buffer=nl.sbuf)
        nisa.dma_copy(src_tile, src)
        nisa.tensor_copy(psum_tile, src_tile)
        nisa.tensor_copy(out_tile, psum_tile)
        nisa.dma_copy(out, out_tile)

    traced = triton_viz.trace(client=Tracer(), frontend="nki_beta2")(kernel)
    src = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
    out = np.empty((128, 128), dtype=np.float32)
    traced[(1,)](src, out)

    transfers = [
        record for record in launches[-1].records if isinstance(record, Transfer)
    ]
    assert [(record.mem_src, record.mem_dst) for record in transfers] == [
        ("hbm", "sbuf"),
        ("sbuf", "psum"),
        ("psum", "sbuf"),
        ("sbuf", "hbm"),
    ]

    # check the visualizer interface can handle tracer-produced Transfer records
    from triton_viz.visualizer.draw import get_visualization_data

    viz_data = get_visualization_data()
    ops = viz_data["visualization_data"]["(0, 0, 0)"]
    transfer_ops = [op for op in ops if op["type"] == "Transfer"]
    assert transfer_ops
    assert all(tuple(op["global_shape"]) == (128, 128) for op in transfer_ops)


def test_trace_records_beta2_transfer_bytes_for_mixed_dtypes():
    """beta2 tracing should size transfer bytes from the destination dtype."""
    triton_viz.clear()

    def kernel(src, out):
        src_tile = nl.ndarray((128, 128), dtype=src.dtype, buffer=nl.sbuf)
        psum_tile = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
        out_tile = nl.ndarray((128, 128), dtype=out.dtype, buffer=nl.sbuf)
        nisa.dma_copy(src_tile, src)
        nisa.tensor_copy(psum_tile, src_tile)
        nisa.tensor_copy(out_tile, psum_tile)
        nisa.dma_copy(out, out_tile)

    traced = triton_viz.trace(client=Tracer(), frontend="nki_beta2")(kernel)
    src = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
    out = np.empty((128, 128), dtype=np.float16)
    traced[(1,)](src, out)

    transfers = [
        record for record in launches[-1].records if isinstance(record, Transfer)
    ]
    assert [record.bytes for record in transfers] == [
        src.nbytes,
        src.nbytes,
        out.nbytes,
        out.nbytes,
    ]


def test_beta2_transfer_events_carry_hbm_access_geometry():
    """Runtime DMA-packet accounting needs concrete HBM access geometry.

    beta2 kernels record memory movement as ``Transfer`` records, so the JSONL
    events must expose the same active-access/stride features the legacy Load/
    Store path already provides. On-chip PSUM/SBUF copies must not be mistaken
    for HBM accesses.
    """
    triton_viz.clear()

    def kernel(src, out):
        src_tile = nl.ndarray((128, 128), dtype=src.dtype, buffer=nl.sbuf)
        psum_tile = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
        out_tile = nl.ndarray((128, 128), dtype=out.dtype, buffer=nl.sbuf)
        nisa.dma_copy(src_tile, src)
        nisa.tensor_copy(psum_tile, src_tile)
        nisa.tensor_copy(out_tile, psum_tile)
        nisa.dma_copy(out, out_tile)

    traced = triton_viz.trace(client=Tracer(), frontend="nki_beta2")(kernel)
    src = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
    out = np.empty((128, 128), dtype=np.float32)
    traced[(1,)](src, out)
    events = records_to_events(launches[-1].records)
    transfers = [event for event in events if event["op"] == "transfer"]
    hbm = [event for event in transfers if "hbm" in str(event["mem_src"]).lower() or "hbm" in str(event["mem_dst"]).lower()]
    on_chip = [event for event in transfers if "hbm" not in str(event["mem_src"]).lower() and "hbm" not in str(event["mem_dst"]).lower()]
    assert len(hbm) == 2 and len(on_chip) == 2
    assert all(event.get("active_access_count") == 128 * 128 for event in hbm)
    assert all(event.get("item_bytes") == 4 for event in hbm)
    assert all(event.get("free_stride_items") == 1 for event in hbm)
    assert all(event.get("active_access_count") is None for event in on_chip)


def test_trace_no_grid_needed():
    """beta2 tracing should not need an SPMD grid to run."""
    triton_viz.clear()

    def kernel(lhsT, rhs, out):
        lhs_tile = nl.ndarray((128, 128), dtype=lhsT.dtype, buffer=nl.sbuf)
        rhs_tile = nl.ndarray((128, 512), dtype=rhs.dtype, buffer=nl.sbuf)
        res_psum = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
        out_tile = nl.ndarray((128, 512), dtype=out.dtype, buffer=nl.sbuf)
        nisa.dma_copy(lhs_tile, lhsT)
        nisa.dma_copy(rhs_tile, rhs)
        nisa.nc_matmul(dst=res_psum, stationary=lhs_tile, moving=rhs_tile)
        nisa.tensor_copy(out_tile, res_psum)
        nisa.dma_copy(out, out_tile)

    traced = triton_viz.trace(client=Tracer(), frontend="nki_beta2")(kernel)
    lhs = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
    rhs = np.arange(128 * 512, dtype=np.float32).reshape(128, 512)
    out = np.empty((128, 512), dtype=np.float32)
    traced(lhs, rhs, out)

    grid_records = [
        record for record in launches[-1].records if type(record).__name__ == "Grid"
    ]
    assert grid_records[0].idx == (0, 0, 0)
    assert any(isinstance(record, Dot) for record in launches[-1].records)


def test_language_helpers_and_ops(patched_scope):
    """Patched language helpers should run with expected semantics."""
    del patched_scope
    arr = nl.ndarray((2, 2), dtype=float, buffer=nl.sbuf)
    z = nl.zeros((2, 2), dtype=float)
    assert arr.shape == (2, 2)
    assert np.array_equal(z.data, np.zeros((2, 2)))

    b2.nki_builder.grid_dims = (2, 3, 4)
    b2.nki_builder.grid_idx = [1, 2, 3]
    assert nl.program_id(0) == 1
    assert list(nl.affine_range(4)) == [0, 1, 2, 3]


def test_par_dim_marks_nonleading_partition_axis(patched_scope):
    """par_dim should preserve a partition axis beyond shape[0]."""
    del patched_scope
    tile = nl.ndarray((3, nl.par_dim(8), 16), dtype=nl.float32, buffer=nl.sbuf)
    assert tile.shape == (3, 8, 16)
    assert tile.partition_axis == 1
    assert b2._partition_and_free(tile) == (8, 48)
    view = tile[1]
    assert view.shape == (8, 16)
    assert view.partition_axis == 0
    assert b2._partition_and_free(view) == (8, 16)
    assert list(nl.affine_range(1, 6, 2)) == [1, 3, 5]
    assert list(nl.affine_range(start=1, stop=6, step=2)) == [1, 3, 5]
    assert nl.ds(2, 3) == slice(2, 5, None)
    assert nl.tile_size.pmax == 128


@pytest.mark.parametrize(
    "op_name,lhs,rhs",
    (
        (
            "add",
            _nd([[1.0, 2.0]], dtype=np.float32),
            _nd([[3.0, 4.0]], dtype=np.float32),
        ),
        (
            "multiply",
            _nd([[1.0, 2.0]], dtype=np.float32),
            _nd([[3.0, 4.0]], dtype=np.float32),
        ),
        (
            "maximum",
            _nd([[1.0, 2.0]], dtype=np.float32),
            _nd([[3.0, 4.0]], dtype=np.float32),
        ),
        ("logical_or", _nd([[1, 0]], dtype=np.int32), _nd([[0, 1]], dtype=np.int32)),
        ("bitwise_and", _nd([[1, 3]], dtype=np.int32), _nd([[1, 1]], dtype=np.int32)),
        (
            "power",
            _nd([[2.0, 3.0]], dtype=np.float32),
            _nd([[4.0, 2.0]], dtype=np.float32),
        ),
    ),
)
def test_nl_binary_op_tokens_raise_on_direct_tensor_math(
    patched_scope, op_name, lhs, rhs
):
    """Direct nl binary math on tensors should be rejected in beta2."""
    del patched_scope
    op = getattr(nl, op_name)
    with pytest.raises(TypeError, match="binary operators on tensors not supported"):
        op(lhs, rhs)


def test_dma_copy(patched_scope):
    """dma_copy should copy source data into destination."""
    del patched_scope
    a, _, out = _core_inputs()
    nisa.dma_copy(out, a)
    assert np.array_equal(out.data, a.data)


def test_dma_copy_dst_rmw_op(patched_scope):
    """dma_copy should apply dst_rmw_op when provided."""
    del patched_scope
    a, b, out = _core_inputs()
    nisa.dma_copy(out, a)
    nisa.dma_copy(out, b, dst_rmw_op=nl.add)
    assert np.array_equal(out.data, a.data + b.data)


@pytest.mark.parametrize(
    "oob_mode,dge_mode,unique_indices",
    (
        (nisa.oob_mode.error, nisa.dge_mode.none, True),
        (nisa.oob_mode.skip, nisa.dge_mode.none, False),
    ),
)
def test_dma_copy_accepts_doc_kwargs_for_static_copy(
    patched_scope, oob_mode, dge_mode, unique_indices
):
    """dma_copy should accept documented keyword args for static copies."""
    del patched_scope
    src = _nd(np.arange(8, dtype=np.float32).reshape(2, 4), buffer="hbm")
    dst = _zeros((2, 4), dtype=np.float32, buffer="sbuf")
    nisa.dma_copy(
        dst,
        src,
        dst_rmw_op=nl.add,
        oob_mode=oob_mode,
        dge_mode=dge_mode,
        unique_indices=unique_indices,
    )
    assert np.array_equal(dst.data, src.data)


def test_dma_copy_rejects_non_add_dst_rmw_op(patched_scope):
    """dma_copy should only allow nl.add as the documented rmw op."""
    del patched_scope
    src = _nd(np.arange(8, dtype=np.float32).reshape(2, 4), buffer="hbm")
    dst = _zeros((2, 4), dtype=np.float32, buffer="sbuf")
    with pytest.raises(ValueError, match="only .*nl.add"):
        nisa.dma_copy(dst, src, dst_rmw_op=nl.multiply)


def test_dma_copy_rejects_total_element_mismatch(patched_scope):
    """dma_copy should reject src/dst tensors with different total elements."""
    del patched_scope
    src = _nd(np.arange(8, dtype=np.float32).reshape(2, 4), buffer="hbm")
    dst = _zeros((2, 3), dtype=np.float32, buffer="sbuf")
    with pytest.raises(ValueError, match="same number of elements|must match"):
        nisa.dma_copy(dst, src)


def test_tensor_copy(patched_scope):
    """tensor_copy should copy tensors elementwise."""
    del patched_scope
    a = _nd(np.arange(8, dtype=np.float32).reshape(2, 4), buffer="sbuf")
    out = _zeros((2, 4), dtype=np.float32, buffer="sbuf")
    nisa.tensor_copy(out, a)
    assert np.array_equal(out.data, a.data)


@pytest.mark.parametrize(
    "src_buffer,dst_buffer",
    (
        ("sbuf", "sbuf"),
        ("sbuf", "psum"),
        ("psum", "sbuf"),
        ("psum", "psum"),
    ),
)
def test_tensor_copy_same_dtype_is_bit_accurate(patched_scope, src_buffer, dst_buffer):
    """tensor_copy should preserve bits when src/dst dtypes match."""
    del patched_scope
    src_data = np.array(
        [[0x12345678, -7, 0, 0x7FFFFFFF], [5, -11, 42, -99]],
        dtype=np.int32,
    )
    src = _nd(src_data, buffer=src_buffer)
    dst = b2.NDArray(shape=src.shape, dtype=_dtype_token("int32"), buffer=dst_buffer)
    nisa.tensor_copy(dst, src)
    assert np.array_equal(dst.data, src_data)


@pytest.mark.parametrize(
    "src_buffer,dst_buffer,engine",
    (
        ("sbuf", "sbuf", nisa.unknown_engine),
        ("sbuf", "psum", nisa.vector_engine),
        ("psum", "psum", nisa.unknown_engine),
    ),
)
def test_tensor_copy_uses_fp32_intermediate_cast(
    patched_scope, src_buffer, dst_buffer, engine
):
    """tensor_copy should cast through float32 when src/dst dtypes differ."""
    del patched_scope
    src_data = np.array([[0.1, 0.9, 1.24], [1.51, 2.9, -5.2]], dtype=np.float32)
    src = _nd(src_data, buffer=src_buffer)
    dst = b2.NDArray(shape=src.shape, dtype=nl.bfloat16, buffer=dst_buffer)
    nisa.tensor_copy(dst, src, engine=engine)
    expected = np.asarray(np.asarray(src_data, dtype=np.float32), dtype=dst.data.dtype)
    _assert_tensor_equal(dst.data, expected)


def test_tensor_copy_scalar_engine_psum_to_sbuf_requires_nc_v3_or_newer(patched_scope):
    """tensor_copy should reject default nc-v2 scalar-engine psum->sbuf copies."""
    del patched_scope
    src_data = np.array([[0.1, 0.9, 1.24], [1.51, 2.9, -5.2]], dtype=np.float32)
    src = _nd(src_data, buffer="psum")
    dst = b2.NDArray(shape=src.shape, dtype=nl.bfloat16, buffer="sbuf")
    with pytest.raises(
        ValueError,
        match="Scalar Engine tensor_copy is unsupported on NeuronCore-v2",
    ):
        nisa.tensor_copy(dst, src, engine=nisa.scalar_engine)


@pytest.mark.parametrize(
    "src_buffer,dst_buffer",
    (("sbuf", "psum"), ("psum", "sbuf"), ("psum", "psum")),
)
def test_tensor_copy_gpsimd_rejects_psum_tiles(patched_scope, src_buffer, dst_buffer):
    """tensor_copy should reject gpsimd copies that touch psum."""
    del patched_scope
    src = _nd(np.arange(8, dtype=np.float32).reshape(2, 4), buffer=src_buffer)
    dst = b2.NDArray(shape=src.shape, dtype=nl.float32, buffer=dst_buffer)
    with pytest.raises(ValueError, match="GpSimd tensor_copy cannot access PSUM"):
        nisa.tensor_copy(dst, src, engine=nisa.gpsimd_engine)


def test_tensor_copy_scalar_engine_requires_nc_v3_or_newer(patched_scope):
    """tensor_copy should reject scalar-engine copies on simulated nc-v2."""
    del patched_scope
    prev = b2.nki_builder.nc_version
    b2.nki_builder.nc_version = nisa.nc_version.gen2
    try:
        src = _nd(np.arange(8, dtype=np.float32).reshape(2, 4), buffer="sbuf")
        dst = b2.NDArray(shape=src.shape, dtype=nl.float32, buffer="sbuf")
        with pytest.raises(
            ValueError,
            match="Scalar Engine tensor_copy is unsupported on NeuronCore-v2",
        ):
            nisa.tensor_copy(dst, src, engine=nisa.scalar_engine)
    finally:
        b2.nki_builder.nc_version = prev


def test_tensor_tensor(patched_scope):
    """tensor_tensor should apply the provided binary op."""
    del patched_scope
    a, b, out = _core_inputs()
    a.buffer = b.buffer = "sbuf"
    nisa.tensor_tensor(out, a, b, nl.subtract)
    assert np.array_equal(out.data, a.data - b.data)


def test_tensor_scalar(patched_scope):
    """tensor_scalar should support chained ops with operand1/op1."""
    del patched_scope
    a, _, out = _core_inputs()
    nisa.tensor_scalar(out, a, nl.multiply, 2.0, op1=nl.add, operand1=1.0)
    assert np.array_equal(out.data, a.data * 2.0 + 1.0)


def test_tensor_scalar_broadcasts_free_dim_only(patched_scope):
    """tensor_scalar should broadcast only along free dimensions."""
    del patched_scope
    data = _nd(np.arange(6, dtype=np.float32).reshape(2, 3))
    dst = _zeros((2, 3))
    operand_free = _nd(np.array([[10.0], [20.0]], dtype=np.float32))
    nisa.tensor_scalar(dst, data, nl.add, operand_free)
    expected = data.data + np.array([[10.0], [20.0]], dtype=np.float32)
    assert np.array_equal(dst.data, expected)

    operand_partition = _nd(np.array([[10.0, 20.0, 30.0]], dtype=np.float32))
    with pytest.raises(
        ValueError, match="tensor_scalar only broadcasts free dimensions"
    ):
        nisa.tensor_scalar(dst, data, nl.add, operand_partition)


@pytest.mark.parametrize(
    "data_shape,operand_shape,dst_shape",
    (
        ((16, 8), (16, 1), (16, 2, 4)),
        ((16, 8), (16, 1), (16, 4, 2)),
        ((16, 2, 4), (16, 1), (16, 8)),
        ((16, 2, 4), (16, 1), (16, 2, 4)),
        ((16, 2, 4), (16, 1), (16, 4, 2)),
        ((16, 4), (16, 1, 1), (16, 4)),
    ),
)
def test_tensor_scalar_accepts_mismatch3_row_broadcast_reshapes(
    patched_scope, data_shape, operand_shape, dst_shape
):
    """tensor_scalar should accept mismatch3 row-broadcast reshape cases."""
    del patched_scope
    data = _nd(np.arange(np.prod(data_shape), dtype=np.float32).reshape(data_shape))
    operand = _nd(
        np.linspace(0.5, 1.5, num=np.prod(operand_shape), dtype=np.float32).reshape(
            operand_shape
        )
    )
    dst = _zeros(dst_shape)
    nisa.tensor_scalar(dst, data, nl.multiply, operand)
    expected = data.data * operand.data.reshape(
        (data_shape[0],) + (1,) * (len(data_shape) - 1)
    )
    assert np.array_equal(dst.data, expected.reshape(dst_shape))


def test_tensor_reduce_shape(patched_scope):
    """tensor_reduce should reduce across the selected axis."""
    del patched_scope
    src = _nd(np.array([[1.0, 2.0, 3.0], [5.0, 4.0, 0.0]]))
    dst = _zeros((2, 1))
    nisa.tensor_reduce(dst, nl.maximum, src, axis=1, keepdims=True)
    assert dst.data.shape == (2, 1)
    assert np.array_equal(dst.data.reshape(-1), np.array([3.0, 5.0]))


def test_reciprocal(patched_scope):
    """reciprocal should compute elementwise multiplicative inverse."""
    del patched_scope
    rec = _zeros((2, 4))
    src = _nd(np.array([[1.0, 2.0, 4.0, 8.0], [1.0, 2.0, 4.0, 8.0]]))
    nisa.reciprocal(rec, src)
    assert np.allclose(rec.data[0, :2], np.array([1.0, 0.5]))


def test_nc_transpose(patched_scope):
    """nc_transpose should transpose partition and flattened free axes."""
    del patched_scope
    src = _nd(np.arange(24, dtype=float).reshape(2, 3, 4))
    dst = _zeros((12, 2), buffer="psum")
    nisa.nc_transpose(dst, src)
    assert np.array_equal(dst.data, src.data.reshape(2, -1).T)


def test_nc_matmul(patched_scope):
    """nc_matmul should match transposed stationary matmul semantics."""
    del patched_scope
    stationary = b2.NDArray(
        value=np.arange(12, dtype=np.float32).reshape(4, 3),
        dtype=nl.float32,
        buffer="sbuf",
    )
    moving = b2.NDArray(
        value=np.arange(8, dtype=np.float32).reshape(4, 2),
        dtype=nl.float32,
        buffer="sbuf",
    )
    dst = b2.NDArray(shape=(3, 2), dtype=nl.float32, buffer="psum")
    nisa.nc_matmul(dst, stationary, moving)
    assert np.allclose(dst.data, stationary.data.T @ moving.data)


def test_nc_matmul_requires_psum_dst_and_sbuf_inputs(patched_scope):
    """nc_matmul should enforce psum dst and sbuf inputs."""
    del patched_scope
    stationary = _nd(np.arange(12, dtype=float).reshape(4, 3), buffer="sbuf")
    moving = _nd(np.arange(8, dtype=float).reshape(4, 2), buffer="sbuf")
    bad_dst = _zeros((3, 2), buffer="sbuf")
    with pytest.raises(ValueError, match="nc_matmul requires dst in PSUM"):
        nisa.nc_matmul(bad_dst, stationary, moving)

    dst = _zeros((3, 2), buffer="psum")
    bad_stationary = _nd(np.arange(12, dtype=float).reshape(4, 3), buffer="hbm")
    with pytest.raises(ValueError, match="nc_matmul requires stationary in SBUF"):
        nisa.nc_matmul(dst, bad_stationary, moving)

    bad_moving = _nd(np.arange(8, dtype=float).reshape(4, 2), buffer="hbm")
    with pytest.raises(ValueError, match="nc_matmul requires moving in SBUF"):
        nisa.nc_matmul(dst, stationary, bad_moving)


def test_activation(patched_scope):
    """activation should write outputs into destination."""
    del patched_scope
    src = _act_input()
    dst = _zeros((2, 3))
    bias = _nd(np.ones((2, 1), dtype=np.float32))
    nisa.activation(dst, nl.reciprocal, src, bias=bias, scale=1.0)
    assert dst.data.shape == (2, 3)


def test_activation_accepts_scalar_bias(patched_scope):
    """activation should accept documented scalar bias arguments."""
    del patched_scope
    src = _act_input()
    dst = _zeros((2, 3))
    nisa.activation(dst, nl.reciprocal, src, bias=1.0, scale=1.0)
    assert dst.data.shape == (2, 3)


@pytest.mark.parametrize(
    "engine,src_buffer,dst_buffer",
    (
        (nisa.vector_engine, "sbuf", "sbuf"),
        (nisa.vector_engine, "psum", "sbuf"),
        (nisa.vector_engine, "sbuf", "psum"),
        (nisa.vector_engine, "psum", "psum"),
        (nisa.tensor_engine, "sbuf", "psum"),
    ),
)
def test_nc_transpose_documented_engine_buffer_success_cases(
    patched_scope, engine, src_buffer, dst_buffer
):
    """nc_transpose should accept the documented engine/buffer success cases."""
    del patched_scope
    src = _typed_ndarray((4, 3), "float32", buffer=src_buffer)
    dst = b2.NDArray(shape=(3, 4), dtype=nl.float32, buffer=dst_buffer)
    nisa.nc_transpose(dst, src, engine=engine)
    assert np.array_equal(dst.data, src.data.T)


@pytest.mark.parametrize(
    "engine,src_buffer,dst_buffer",
    (
        (nisa.vector_engine, "hbm", "sbuf"),
        (nisa.vector_engine, "sbuf", "hbm"),
        (nisa.tensor_engine, "sbuf", "sbuf"),
        (nisa.tensor_engine, "psum", "psum"),
        (nisa.tensor_engine, "psum", "sbuf"),
    ),
)
def test_nc_transpose_rejects_documented_engine_buffer_failures(
    patched_scope, engine, src_buffer, dst_buffer
):
    """nc_transpose should reject engine/buffer combinations outside the docs."""
    del patched_scope
    src = _typed_ndarray((4, 3), "float32", buffer=src_buffer)
    dst = b2.NDArray(shape=(3, 4), dtype=nl.float32, buffer=dst_buffer)
    with pytest.raises(ValueError, match="buffer|Engine|SBUF|PSUM|HBM"):
        nisa.nc_transpose(dst, src, engine=engine)


def test_nc_transpose_rejects_documented_dtype_mismatch(patched_scope):
    """nc_transpose should reject documented dst/data dtype mismatches."""
    del patched_scope
    src = _typed_ndarray((4, 3), "bfloat16", buffer="sbuf")
    dst = b2.NDArray(shape=(3, 4), dtype=nl.float32, buffer="sbuf")
    with pytest.raises(ValueError, match="dtype"):
        nisa.nc_transpose(dst, src, engine=nisa.vector_engine)


@pytest.mark.parametrize("dst_shape", ((3, 2, 4), (3, 4, 2)))
def test_nc_transpose_mismatch3_engine_split(dst_shape, patched_scope):
    """nc_transpose should allow reshape dst on vector engine but not tensor engine."""
    del patched_scope
    src = _typed_ndarray((8, 3), "float32", buffer="sbuf")
    vec_dst = b2.NDArray(shape=dst_shape, dtype=nl.float32, buffer="sbuf")
    ten_dst = b2.NDArray(shape=dst_shape, dtype=nl.float32, buffer="psum")
    nisa.nc_transpose(vec_dst, src, engine=nisa.vector_engine)
    assert np.array_equal(vec_dst.data, src.data.T.reshape(dst_shape))
    with pytest.raises(
        ValueError, match="Tensor Engine nc_transpose requires dst shape"
    ):
        nisa.nc_transpose(ten_dst, src, engine=nisa.tensor_engine)


@pytest.mark.parametrize(
    "stationary_shape,moving_shape",
    (
        ((256, 128), (256, 512)),
        ((128, 256), (128, 512)),
        ((128, 128), (128, 1024)),
    ),
)
def test_nc_matmul_rejects_documented_shape_limits(
    patched_scope, stationary_shape, moving_shape
):
    """nc_matmul should reject documented stationary/moving tile size violations."""
    del patched_scope
    stationary = _typed_ndarray(stationary_shape, "float32", buffer="sbuf")
    moving = _typed_ndarray(moving_shape, "float32", buffer="sbuf", seed=1)
    dst = b2.NDArray(
        shape=(stationary_shape[1], moving_shape[1]),
        dtype=nl.float32,
        buffer="psum",
    )
    with pytest.raises(ValueError, match="partition|free dim|128|512"):
        nisa.nc_matmul(dst, stationary, moving)


@pytest.mark.parametrize(
    "dst_dtype,stationary_dtype,moving_dtype",
    (
        ("float32", "float32", "float16"),
        ("float32", "bfloat16", "float32"),
        ("float16", "float32", "float16"),
    ),
)
def test_nc_matmul_rejects_documented_dtype_rules(
    patched_scope, dst_dtype, stationary_dtype, moving_dtype
):
    """nc_matmul should reject documented dtype combinations."""
    del patched_scope
    stationary = _typed_ndarray((32, 16), stationary_dtype, buffer="sbuf")
    moving = _typed_ndarray((32, 8), moving_dtype, buffer="sbuf", seed=1)
    dst = b2.NDArray(shape=(16, 8), dtype=_dtype_token(dst_dtype), buffer="psum")
    with pytest.raises(ValueError, match="float32|dtype|dst"):
        nisa.nc_matmul(dst, stationary, moving)


@pytest.mark.parametrize(
    "tile_position,tile_size",
    (
        ((0, 0), (32, 32)),
        ((0, 0), (128, 128)),
        ((0, 0), (64, 128)),
        ((64, 0), (64, 128)),
        ((0, 0), (128, 64)),
        ((0, 64), (128, 64)),
        ((0, 0), (64, 64)),
        ((64, 64), (64, 64)),
    ),
)
def test_nc_matmul_documented_tiling_success_cases(
    patched_scope, tile_position, tile_size
):
    """nc_matmul should accept the documented tiling success cases."""
    del patched_scope
    stationary_shape = (
        tile_size[0] if tile_size[0] != 64 else 32,
        min(tile_size[1], 128),
    )
    moving_shape = (stationary_shape[0], 256)
    stationary = _typed_ndarray(stationary_shape, "float32", buffer="sbuf")
    moving = _typed_ndarray(moving_shape, "float32", buffer="sbuf", seed=1)
    dst = b2.NDArray(
        shape=(stationary_shape[1], moving_shape[1]),
        dtype=nl.float32,
        buffer="psum",
    )
    nisa.nc_matmul(
        dst,
        stationary,
        moving,
        tile_position=tile_position,
        tile_size=tile_size,
    )
    assert dst.data.shape == (stationary_shape[1], moving_shape[1])


@pytest.mark.parametrize(
    "stationary_shape,moving_shape,tile_position,tile_size",
    (
        ((128, 128), (128, 256), (0, 0), (256, 256)),
        ((128, 128), (128, 256), (0, 0), (256, 128)),
        ((64, 128), (64, 256), (0, 0), (64, 64)),
        ((128, 64), (128, 256), (0, 0), (64, 64)),
        ((128, 128), (128, 256), (0, 0), (64, 128)),
        ((64, 128), (64, 256), (32, 0), (64, 64)),
        ((64, 128), (64, 256), (0, 32), (64, 64)),
        ((64, 128), (64, 256), (256, 0), (64, 64)),
        ((64, 128), (64, 256), (0, 256), (64, 64)),
        ((48, 48), (48, 48), (0, 0), (48, 48)),
        ((96, 96), (96, 128), (96, 96), (96, 96)),
    ),
)
def test_nc_matmul_rejects_documented_tiling_failures(
    patched_scope, stationary_shape, moving_shape, tile_position, tile_size
):
    """nc_matmul should reject the documented invalid tiling configurations."""
    del patched_scope
    stationary = _typed_ndarray(stationary_shape, "float32", buffer="sbuf")
    moving = _typed_ndarray(moving_shape, "float32", buffer="sbuf", seed=1)
    dst = b2.NDArray(
        shape=(stationary_shape[1], moving_shape[1]),
        dtype=nl.float32,
        buffer="psum",
    )
    with pytest.raises(ValueError, match="tile|row|col|128|both"):
        nisa.nc_matmul(
            dst,
            stationary,
            moving,
            tile_position=tile_position,
            tile_size=tile_size,
        )


@pytest.mark.parametrize(
    "stationary_shape,dst_shape",
    (
        ((32, 6), (2, 3, 4, 5)),
        ((32, 2, 3), (2, 3, 4, 5)),
        ((32, 2, 3), (6, 4, 5)),
        ((32, 2, 3), (2, 12, 5)),
        ((32, 2, 3), (2, 3, 20)),
        ((32, 2, 3), (2, 60)),
    ),
)
def test_nc_matmul_rejects_mismatch3_unsupported_reshapes(
    patched_scope, stationary_shape, dst_shape
):
    """nc_matmul should reject mismatch3 shape cases that do not compile."""
    del patched_scope
    stationary = _typed_ndarray(stationary_shape, "float32", buffer="sbuf")
    moving = _typed_ndarray((32, 4, 5), "float32", buffer="sbuf", seed=1)
    dst = b2.NDArray(shape=dst_shape, dtype=nl.float32, buffer="psum")
    with pytest.raises(
        ValueError, match="rank-2|preserve stationary free dim and moving free dims"
    ):
        nisa.nc_matmul(dst, stationary, moving)


@pytest.mark.parametrize("tile_position,tile_size", (((), (64, 64)), ((64, 64), ())))
def test_nc_matmul_requires_both_tile_position_and_tile_size(
    patched_scope, tile_position, tile_size
):
    """nc_matmul should require tile_position and tile_size together."""
    del patched_scope
    stationary = _typed_ndarray((64, 128), "float32", buffer="sbuf")
    moving = _typed_ndarray((64, 256), "float32", buffer="sbuf", seed=1)
    dst = b2.NDArray(shape=(128, 256), dtype=nl.float32, buffer="psum")
    with pytest.raises(ValueError, match="both|tile_position|tile_size"):
        nisa.nc_matmul(
            dst,
            stationary,
            moving,
            tile_position=tile_position,
            tile_size=tile_size,
        )


@pytest.mark.parametrize(
    "op,expected",
    (
        ("add", np.array([[6.0, 8.0], [10.0, 12.0]], dtype=np.float32)),
        ("minimum", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)),
        ("power", np.array([[1.0, 64.0], [2187.0, 65536.0]], dtype=np.float32)),
    ),
)
def test_tensor_tensor_documented_operator_success_cases(patched_scope, op, expected):
    """tensor_tensor should cover documented operator categories."""
    del patched_scope
    lhs = _nd(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), buffer="sbuf")
    rhs = _nd(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32), buffer="sbuf")
    dst = b2.NDArray(shape=(2, 2), dtype=nl.float32, buffer="sbuf")
    nisa.tensor_tensor(dst, lhs, rhs, getattr(nl, op), engine=nisa.unknown_engine)
    _assert_tensor_equal(dst.data, expected)


@pytest.mark.parametrize(
    "lhs_dtype,rhs_dtype,dst_dtype,op",
    (
        ("float32", "float32", "float32", "bitwise_and"),
        ("float32", "int8", "float32", "bitwise_or"),
        ("int32", "float16", "int32", "bitwise_xor"),
    ),
)
def test_tensor_tensor_rejects_documented_bitvec_dtype_failures(
    patched_scope, lhs_dtype, rhs_dtype, dst_dtype, op
):
    """tensor_tensor should reject documented bitvec dtype mismatches."""
    del patched_scope
    lhs = _typed_ndarray((2, 4), lhs_dtype, buffer="sbuf")
    rhs = _typed_ndarray((2, 4), rhs_dtype, buffer="sbuf", seed=1)
    dst = b2.NDArray(shape=(2, 4), dtype=_dtype_token(dst_dtype), buffer="sbuf")
    with pytest.raises(ValueError, match="integer|dtype|bitvec"):
        nisa.tensor_tensor(dst, lhs, rhs, getattr(nl, op), engine=nisa.vector_engine)


def test_tensor_tensor_rejects_documented_gpsimd_psum_case(patched_scope):
    """tensor_tensor should reject gpsimd reads or writes involving psum."""
    del patched_scope
    lhs = _typed_ndarray((2, 4), "float32", buffer="psum")
    rhs = _typed_ndarray((2, 4), "float32", buffer="sbuf", seed=1)
    dst = b2.NDArray(shape=(2, 4), dtype=nl.float32, buffer="sbuf")
    with pytest.raises(ValueError, match="GpSimd|PSUM"):
        nisa.tensor_tensor(dst, lhs, rhs, nl.power, engine=nisa.gpsimd_engine)


def test_tensor_tensor_rejects_documented_psum_power_dst(patched_scope):
    """tensor_tensor should reject nl.power outputs to psum."""
    del patched_scope
    lhs = _typed_ndarray((2, 4), "float32", buffer="sbuf")
    rhs = _typed_ndarray((2, 4), "float32", buffer="sbuf", seed=1)
    dst = b2.NDArray(shape=(2, 4), dtype=nl.float32, buffer="psum")
    with pytest.raises(ValueError, match="GpSimd|PSUM|power"):
        nisa.tensor_tensor(dst, lhs, rhs, nl.power, engine=nisa.gpsimd_engine)


@pytest.mark.parametrize(
    "dst_shape,data_shape,axis,keepdims",
    (
        ((128, 1), (128, 32), 1, False),
        ((128, 1), (128, 32), [1], False),
        ((128, 1), (128, 32), (1,), False),
        ((32, 6), (32, 5, 6), [1], False),
        ((32, 1, 6), (32, 5, 6), [1], True),
    ),
)
def test_tensor_reduce_documented_shape_axis_success_cases(
    patched_scope, dst_shape, data_shape, axis, keepdims
):
    """tensor_reduce should cover the documented legal shape/axis cases."""
    del patched_scope
    src = _typed_ndarray(data_shape, "float32", buffer="sbuf")
    dst = b2.NDArray(shape=dst_shape, dtype=nl.float32, buffer="sbuf")
    nisa.tensor_reduce(dst, nl.add, src, axis=axis, keepdims=keepdims)
    axis_tuple = (axis,) if isinstance(axis, int) else tuple(axis)
    expected = np.sum(src.data, axis=axis_tuple, keepdims=keepdims)
    _assert_tensor_equal(dst.data, expected.reshape(dst.shape))


@pytest.mark.parametrize(
    "op",
    (
        "invert",
        "left_shift",
        "right_shift",
        "equal",
        "not_equal",
        "greater_equal",
        "greater",
        "less_equal",
        "less",
        "rsqrt",
        "reciprocal",
        "abs",
        "power",
    ),
)
def test_tensor_reduce_rejects_documented_illegal_ops(patched_scope, op):
    """tensor_reduce should reject non-reduction operators from the docs."""
    del patched_scope
    src = _typed_ndarray((8, 4), "float32", buffer="sbuf")
    dst = b2.NDArray(shape=(8, 1), dtype=nl.float32, buffer="sbuf")
    with pytest.raises(
        (ValueError, RuntimeError), match="legal reduction|Unsupported|op"
    ):
        nisa.tensor_reduce(dst, getattr(nl, op), src, axis=[1])


@pytest.mark.parametrize(
    "axis",
    ([0], [0, 1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [2, 3], [1, 3]),
)
def test_tensor_reduce_rejects_compiler_invalid_dims(patched_scope, axis):
    """tensor_reduce should match compiler parity for invalid reduction dims."""
    del patched_scope
    src = _typed_ndarray((128, 2, 3, 4, 5), "float32", buffer="sbuf")
    dst = b2.NDArray(shape=(128, 1), dtype=nl.float32, buffer="sbuf")
    with pytest.raises(ValueError, match="not a valid dim"):
        nisa.tensor_reduce(dst, nl.add, src, axis=axis)


@pytest.mark.parametrize(
    "dst_shape,keepdims",
    (
        ((16, 4, 2), True),
        ((16, 4, 2), False),
        ((16, 2, 4), True),
        ((16, 2, 4), False),
        ((16, 1, 2, 4), True),
        ((16, 1, 2, 4), False),
    ),
)
def test_tensor_reduce_rejects_mismatch3_reshaped_destinations(
    patched_scope, dst_shape, keepdims
):
    """tensor_reduce should reject mismatch3 reshaped destinations."""
    del patched_scope
    src = _typed_ndarray((16, 3, 8), "float32", buffer="sbuf")
    dst = b2.NDArray(shape=dst_shape, dtype=nl.float32, buffer="sbuf")
    with pytest.raises(
        ValueError, match="cannot reduce free dim of data into free dim of dst"
    ):
        nisa.tensor_reduce(dst, nl.add, src, axis=1, keepdims=keepdims)


@pytest.mark.parametrize(
    "dst_shape,data_shape",
    (
        ((128, 1), (256, 32)),
        ((256, 1), (256, 32)),
        ((128, 2), (128, 32)),
        ((128, 1), (128, 2, 4)),
    ),
)
def test_tensor_reduce_rejects_documented_axis_and_shape_failures(
    patched_scope, dst_shape, data_shape
):
    """tensor_reduce should reject documented illegal axis/shape combinations."""
    del patched_scope
    src = _typed_ndarray(data_shape, "float32", buffer="sbuf")
    dst = b2.NDArray(shape=dst_shape, dtype=nl.float32, buffer="sbuf")
    with pytest.raises(ValueError, match="partition|reduce|shape|128"):
        nisa.tensor_reduce(dst, nl.add, src, axis=[1])


@pytest.mark.parametrize(
    # "op", (nl.add, nl.subtract, nl.multiply, nl.maximum, nl.minimum)
    "op",
    ("add", "subtract", "multiply", "maximum", "minimum"),
)
def test_tensor_reduce_documented_negate_success_cases(patched_scope, op):
    """tensor_reduce should support negate for documented arithmetic ops."""
    del patched_scope
    src = _typed_ndarray((8, 4), "float32", buffer="sbuf")
    dst = b2.NDArray(shape=(8, 1), dtype=nl.float32, buffer="sbuf")
    nisa.tensor_reduce(dst, getattr(nl, op), src, axis=[1], negate=True)
    assert dst.data.shape == (8, 1)


@pytest.mark.parametrize(
    "op",
    (
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "logical_and",
        "logical_or",
        "logical_xor",
    ),
)
def test_tensor_reduce_rejects_documented_negate_failures(patched_scope, op):
    """tensor_reduce should reject negate for non-arithmetic reduction ops."""
    del patched_scope
    src = _typed_ndarray((8, 4), "int32", buffer="sbuf")
    dst = b2.NDArray(shape=(8, 1), dtype=nl.int32, buffer="sbuf")
    with pytest.raises(ValueError, match="negate|arithmetic"):
        nisa.tensor_reduce(dst, getattr(nl, op), src, axis=[1], negate=True)


@pytest.mark.parametrize(
    "reverse0,op1,reverse1,expected",
    (
        (True, None, False, np.array([[9.0, 8.0], [7.0, 6.0]], dtype=np.float32)),
        (
            True,
            "subtract",
            False,
            np.array([[4.0, 3.0], [2.0, 1.0]], dtype=np.float32),
        ),
        (
            False,
            "subtract",
            True,
            np.array([[14.0, 13.0], [12.0, 11.0]], dtype=np.float32),
        ),
        (
            True,
            "subtract",
            True,
            np.array([[-4.0, -3.0], [-2.0, -1.0]], dtype=np.float32),
        ),
    ),
)
def test_tensor_scalar_documented_reverse_cases(
    patched_scope, reverse0, op1, reverse1, expected
):
    """tensor_scalar should cover the documented reverse ordering cases."""
    del patched_scope
    data = _nd(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    dst = b2.NDArray(shape=(2, 2), dtype=nl.float32, buffer="sbuf")
    nisa.tensor_scalar(
        dst,
        data,
        nl.subtract,
        10.0,
        reverse0=reverse0,
        op1=getattr(nl, op1) if op1 else None,
        operand1=5.0 if op1 is not None else None,
        reverse1=reverse1,
    )
    _assert_tensor_equal(dst.data, expected)


@pytest.mark.parametrize(
    "op0,op1",
    (("logical_or", "add"), ("multiply", "logical_and")),
)
def test_tensor_scalar_rejects_documented_mixed_op_types(patched_scope, op0, op1):
    """tensor_scalar should reject mixed bitvec and arithmetic operator chains."""
    del patched_scope
    data = _typed_ndarray((2, 4), "int32", buffer="sbuf")
    dst = b2.NDArray(shape=(2, 4), dtype=nl.int32, buffer="sbuf")
    with pytest.raises(ValueError, match="bitvec|arithmetic|mixed"):
        nisa.tensor_scalar(
            dst, data, getattr(nl, op0), 1, op1=getattr(nl, op1), operand1=1
        )


@pytest.mark.parametrize(
    "op0,engine",
    (
        ("logical_or", "scalar_engine"),
        ("logical_or", "gpsimd_engine"),
        ("rsqrt", "vector_engine"),
    ),
)
def test_tensor_scalar_rejects_documented_engine_failures(patched_scope, op0, engine):
    """tensor_scalar should reject engine/op combinations outside the docs."""
    del patched_scope
    data = _typed_ndarray((2, 4), "float32", buffer="sbuf")
    dst = b2.NDArray(shape=(2, 4), dtype=nl.float32, buffer="sbuf")
    with pytest.raises(
        ValueError, match="Vector Engine|Scalar Engine|GpSimd|bitvec|rsqrt"
    ):
        nisa.tensor_scalar(
            dst, data, getattr(nl, op0), 1.0, engine=getattr(nisa, engine)
        )


@pytest.mark.parametrize(
    "data_dtype,operand,op0",
    (
        ("float32", np.array([[1.0], [2.0]], dtype=np.float32), "logical_or"),
        ("float32", np.array([[1.0], [2.0]], dtype=np.float16), "logical_or"),
    ),
)
def test_tensor_scalar_rejects_documented_dtype_failures(
    patched_scope, data_dtype, operand, op0
):
    """tensor_scalar should reject the documented dtype failures."""
    del patched_scope
    data = _typed_ndarray((2, 4), data_dtype, buffer="sbuf")
    dst = b2.NDArray(shape=(2, 4), dtype=nl.float32, buffer="sbuf")
    with pytest.raises(ValueError, match="dtype|float32|integer|bitvec"):
        nisa.tensor_scalar(dst, data, getattr(nl, op0), _nd(operand))


def test_activation_documented_scalar_success_case(patched_scope):
    """activation should accept documented scalar scale/bias and reduce settings."""
    del patched_scope
    data = _typed_ndarray((8, 8), "float32", buffer="sbuf")
    dst = b2.NDArray(shape=(8, 8), dtype=nl.float32, buffer="sbuf")
    reduce_res = b2.NDArray(shape=(8, 1), dtype=nl.float32, buffer="sbuf")
    nisa.activation(
        dst,
        nl.copy,
        data,
        bias=0.0,
        scale=1.0,
        reduce_op=None,
        reduce_res=reduce_res,
        reduce_cmd=nisa.reduce_cmd.idle,
    )
    assert dst.data.shape == (8, 8)


def test_activation_documented_vector_success_case(patched_scope):
    """activation should accept documented vector scale/bias layout."""
    del patched_scope
    data = _typed_ndarray((3, 5, 7), "float32", buffer="psum")
    dst = b2.NDArray(shape=(3, 35), dtype=nl.float32, buffer="psum")
    scale = _nd(np.ones((3, 1), dtype=np.float32), buffer="psum")
    bias = _nd(np.zeros((3, 1), dtype=np.float32), buffer="psum")
    reduce_res = b2.NDArray(shape=(3, 1), dtype=nl.float32, buffer="psum")
    nisa.activation(
        dst,
        nl.copy,
        data,
        bias=bias,
        scale=scale,
        reduce_op=nl.add,
        reduce_res=reduce_res,
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
    )
    assert dst.data.shape == (3, 35)
    assert np.array_equal(
        reduce_res.data,
        np.sum(dst.data.reshape(dst.shape[0], -1), axis=1, keepdims=True),
    )


def test_activation_accepts_mismatch3_reduce_res_p_1(patched_scope):
    """activation should materialize the canonical mismatch3 reduce_res tile."""
    del patched_scope
    data = _typed_ndarray((8, 4), "float32", buffer="sbuf")
    dst = b2.NDArray(shape=(8, 4), dtype=nl.float32, buffer="sbuf")
    reduce_res = b2.NDArray(shape=(8, 1), dtype=nl.float32, buffer="sbuf")
    nisa.activation(
        dst,
        nl.copy,
        data,
        reduce_op=nl.add,
        reduce_res=reduce_res,
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
    )
    assert np.array_equal(
        reduce_res.data,
        np.sum(dst.data.reshape(dst.shape[0], -1), axis=1, keepdims=True),
    )


@pytest.mark.parametrize("reduce_shape", ((8,), (8, 1, 1)))
def test_activation_rejects_noncanonical_reduce_res_shapes(patched_scope, reduce_shape):
    """activation should reject reduce_res shapes outside the canonical `(P, 1)`."""
    del patched_scope
    data = _typed_ndarray((8, 4), "float32", buffer="sbuf")
    dst = b2.NDArray(shape=(8, 4), dtype=nl.float32, buffer="sbuf")
    if len(reduce_shape) < 2:
        with pytest.raises(
            ValueError, match="SBUF/PSUM tensors must have at least 2 dims"
        ):
            b2.NDArray(shape=reduce_shape, dtype=nl.float32, buffer="sbuf")
        return
    reduce_res = b2.NDArray(shape=reduce_shape, dtype=nl.float32, buffer="sbuf")
    with pytest.raises(ValueError, match="reduce_res must have shape"):
        nisa.activation(
            dst,
            nl.copy,
            data,
            reduce_op=nl.add,
            reduce_res=reduce_res,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
        )


def test_activation_rejects_documented_reduce_op_failure(patched_scope):
    """activation should reject reduce_op values other than None or nl.add."""
    del patched_scope
    data = _typed_ndarray((4, 8), "float32", buffer="sbuf")
    dst = b2.NDArray(shape=(4, 8), dtype=nl.float32, buffer="sbuf")
    reduce_res = b2.NDArray(shape=(4, 1), dtype=nl.float32, buffer="sbuf")
    with pytest.raises(ValueError, match="reduce_op|nl.add"):
        nisa.activation(
            dst,
            nl.copy,
            data,
            reduce_op=nl.maximum,
            reduce_res=reduce_res,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
        )


@pytest.mark.parametrize(
    "dst_shape,data_shape",
    (((4, 128), (2, 256)), ((4, 128), (4, 256)), ((129, 128), (129, 128))),
)
def test_activation_rejects_documented_shape_failures(
    patched_scope, dst_shape, data_shape
):
    """activation should reject documented dst/data shape failures."""
    del patched_scope
    data = _typed_ndarray(data_shape, "float32", buffer="sbuf")
    dst = b2.NDArray(shape=dst_shape, dtype=nl.float32, buffer="sbuf")
    with pytest.raises(ValueError, match="partition|free dim|128|shape"):
        nisa.activation(dst, nl.copy, data)


def test_program_id_invalid_axis_raises(patched_scope):
    """program_id should reject out-of-range launch axes."""
    del patched_scope
    b2.nki_builder.grid_idx = [0, 1]
    with pytest.raises(ValueError, match="Invalid axis"):
        nl.program_id(2)


def test_ds_non_scalar_start_raises(patched_scope):
    """ds should reject non-scalar start inputs."""
    del patched_scope
    with pytest.raises(ValueError, match="Expected scalar value"):
        nl.ds(np.array([1, 2]), 3)


def test_interpreted_function_execution(patched_scope):
    """NKIBeta2InterpretedFunction should execute kernels over launch grids."""
    del patched_scope

    class _ClientManager:
        """Minimal client manager used by interpreted function tests."""

        def grid_callback(self, grid):
            """Record the launch grid."""
            self.grid = grid

        def grid_idx_callback(self, idx):
            """Record the active grid index."""
            self.last_idx = idx

        def arg_callback(self, name, arg, ret):
            """Record the latest argument callback payload."""
            self.last_arg = (name, arg, ret)

        def pre_run_callback(self, fn):
            """Allow execution to proceed."""
            self.last_fn = fn
            return True

        def post_run_callback(self, fn):
            """Allow execution to proceed after one program."""
            self.last_post_fn = fn
            return True

    def copy_kernel(src, dst):
        """Copy one source tile into destination."""
        nisa.dma_copy(dst=dst, src=src)

    fn = b2.NKIBeta2InterpretedFunction(copy_kernel)
    src = np.arange(16, dtype=float).reshape(4, 4)
    dst = np.empty_like(src)
    fn.run(src, dst, grid=(1,), client_manager=_ClientManager())
    assert np.array_equal(dst, src)


def test_ndarray_shape_dtype_buffer_matrix(patched_scope):
    """ndarray should match documented shape/dtype/buffer behavior."""
    del patched_scope
    for dtype_name in NL_DTYPE_CASES:
        if not hasattr(nl, dtype_name):
            continue
        dtype = _dtype_token(dtype_name)
        for shape in SHAPE_CASES:
            for buffer in (nl.hbm, nl.sbuf, nl.psum):
                if dtype in STORAGE_DTYPES:
                    if (
                        _buffer_token_name(buffer) in ("sbuf", "psum")
                        and len(shape) < 2
                    ):
                        with pytest.raises(
                            ValueError,
                            match="SBUF/PSUM tensors must have at least 2 dims",
                        ):
                            nl.ndarray(shape, dtype=dtype, buffer=buffer)
                    else:
                        out = nl.ndarray(shape, dtype=dtype, buffer=buffer)
                        assert out.shape == shape
                        assert out.buffer == _buffer_token_name(buffer)
                        assert out.data.dtype == STORAGE_DTYPES[dtype]
                else:
                    with pytest.raises(TypeError, match="Unsupported dtype"):
                        nl.ndarray(shape, dtype=dtype, buffer=buffer)
    with pytest.raises(TypeError, match="Unsupported buffer type"):
        nl.ndarray((1,), dtype=_dtype_token("float32"), buffer="private_hbm")


def test_zeros_shape_dtype_matrix(patched_scope):
    """zeros should match shape and dtype behavior."""
    del patched_scope
    for dtype_name in NL_DTYPE_CASES:
        if not hasattr(nl, dtype_name):
            continue
        dtype = _dtype_token(dtype_name)
        for shape in SHAPE_CASES:
            if dtype in STORAGE_DTYPES:
                if len(shape) < 2:
                    with pytest.raises(
                        ValueError,
                        match="SBUF/PSUM tensors must have at least 2 dims",
                    ):
                        nl.zeros(shape, dtype=dtype)
                else:
                    out = nl.zeros(shape, dtype=dtype)
                    assert out.shape == shape
                    assert out.buffer == _buffer_token_name(nl.sbuf)
                    assert out.data.dtype == STORAGE_DTYPES[dtype]
                    assert np.all(out.data == 0)
            else:
                with pytest.raises(TypeError, match="Unsupported dtype"):
                    nl.zeros(shape, dtype=dtype)


def test_zeros_rejects_buffer_kwarg(patched_scope):
    """zeros should reject explicit buffer kwarg."""
    del patched_scope
    with pytest.raises(
        TypeError,
        match="unexpected keyword argument 'buffer' in builtinfunction 'builtin_lang_zeros'",
    ):
        nl.zeros((2, 2), dtype=nl.float32, buffer=nl.sbuf)


@pytest.mark.parametrize(
    "dst_shape,data_shape,dst_buffer",
    (((4, 8), (8, 4), "sbuf"), ((35, 3), (3, 5, 7), "psum")),
)
@pytest.mark.parametrize(
    "dst_dtype_name,data_dtype_name",
    (("float32", "float32"), ("bfloat16", "bfloat16"), ("bfloat16", "float32")),
)
def test_nc_transpose_shape_dtype_cases(
    patched_scope, dst_shape, data_shape, dst_buffer, dst_dtype_name, data_dtype_name
):
    """nc_transpose should match documented shape and dtype combinations."""
    del patched_scope
    src_data = (
        np.arange(np.prod(data_shape), dtype=np.float32).reshape(data_shape) + 0.25
    )
    data = _nd(src_data, dtype=STORAGE_DTYPES[data_dtype_name], buffer="sbuf")
    dst = b2.NDArray(
        shape=dst_shape,
        dtype=_dtype_token(dst_dtype_name),
        buffer=dst_buffer,
    )
    nisa.nc_transpose(dst, data)
    expected = np.asarray(data.data).reshape(data.shape[0], -1).T
    expected = np.asarray(expected, dtype=dst.data.dtype)
    _assert_tensor_equal(dst.data, expected)


def test_nc_transpose_infers_vector_engine_for_small_tiles(patched_scope):
    """nc_transpose should infer vector engine for <= (32, 32) tiles."""
    del patched_scope
    src = _typed_ndarray((8, 4), "float32", buffer="psum")
    dst = b2.NDArray(shape=(4, 8), dtype=nl.float32, buffer="psum")
    nisa.nc_transpose(dst, src)
    assert np.array_equal(dst.data, src.data.T)


def test_nc_transpose_infers_tensor_engine_for_medium_tiles(patched_scope):
    """nc_transpose should infer tensor engine for <= (128, 128) tiles."""
    del patched_scope
    src = _typed_ndarray((3, 5, 7), "float32", buffer="sbuf")
    dst = b2.NDArray(shape=(35, 3), dtype=nl.float32, buffer="psum")
    nisa.nc_transpose(dst, src)
    assert np.array_equal(dst.data, src.data.reshape(3, -1).T)


def test_nc_transpose_inferred_engine_rejects_psum_to_psum_large_tiles(patched_scope):
    """nc_transpose should reject psum->psum when inference selects tensor engine."""
    del patched_scope
    src = _typed_ndarray((33, 1), "float32", buffer="psum")
    dst = b2.NDArray(shape=(1, 33), dtype=nl.float32, buffer="psum")
    with pytest.raises(
        ValueError, match="Tensor Engine nc_transpose requires SBUF -> PSUM"
    ):
        nisa.nc_transpose(dst, src)


def test_nc_transpose_rejects_tiles_larger_than_tensor_engine(patched_scope):
    """nc_transpose should reject tiles that exceed both inferred engine limits."""
    del patched_scope
    src = _typed_ndarray((129, 1), "float32", buffer="sbuf")
    dst = b2.NDArray(shape=(1, 129), dtype=nl.float32, buffer="psum")
    with pytest.raises(ValueError, match="tile too large for nc_transpose"):
        nisa.nc_transpose(dst, src)


def test_nc_matmul_matches_numpy_matrix(patched_scope):
    """nc_matmul should match numpy across documented shapes/dtypes/distributions."""
    del patched_scope
    shape_cases = (((5, 3), (5, 7)), ((128, 128), (128, 128)))
    dtype_names = ("float32", "bfloat16")
    distributions = ("linspace", "randn")
    seed = 17

    for stationary_shape, moving_shape in shape_cases:
        for dtype_name in dtype_names:
            for dist in distributions:
                lhs_np = _matrix(stationary_shape, dist, dtype_name, seed)
                rhs_np = _matrix(moving_shape, dist, dtype_name, seed + 1)
                stationary = b2.NDArray(
                    value=lhs_np, dtype=_dtype_token(dtype_name), buffer="sbuf"
                )
                moving = b2.NDArray(
                    value=rhs_np, dtype=_dtype_token(dtype_name), buffer="sbuf"
                )
                dst = b2.NDArray(
                    shape=(stationary_shape[1], moving_shape[1]),
                    dtype=nl.float32,
                    buffer="psum",
                )
                dst.data.fill(0)
                nisa.nc_matmul(dst, stationary, moving)
                expected = np.zeros(dst.data.shape, dtype=dst.data.dtype)
                expected += np.asarray(
                    np.asarray(stationary.data).T @ np.asarray(moving.data),
                    dtype=dst.data.dtype,
                )
                _assert_tensor_equal(dst.data, expected)
                seed += 2


def test_nc_matmul_accumulates_across_calls(patched_scope):
    """nc_matmul should accumulate into dst across repeated calls."""
    del patched_scope
    shape_cases = (((5, 3), (5, 7)), ((128, 128), (128, 128)))
    dtype_names = ("float32", "bfloat16")
    distributions = ("linspace", "randn")
    seed = 101

    for stationary_shape, moving_shape in shape_cases:
        for dtype_name in dtype_names:
            for dist in distributions:
                lhs0 = b2.NDArray(
                    value=_matrix(stationary_shape, dist, dtype_name, seed),
                    dtype=_dtype_token(dtype_name),
                    buffer="sbuf",
                )
                rhs0 = b2.NDArray(
                    value=_matrix(moving_shape, dist, dtype_name, seed + 1),
                    dtype=_dtype_token(dtype_name),
                    buffer="sbuf",
                )
                lhs1 = b2.NDArray(
                    value=_matrix(stationary_shape, dist, dtype_name, seed + 2),
                    dtype=_dtype_token(dtype_name),
                    buffer="sbuf",
                )
                rhs1 = b2.NDArray(
                    value=_matrix(moving_shape, dist, dtype_name, seed + 3),
                    dtype=_dtype_token(dtype_name),
                    buffer="sbuf",
                )
                dst = b2.NDArray(
                    shape=(stationary_shape[1], moving_shape[1]),
                    dtype=nl.float32,
                    buffer="psum",
                )
                dst.data.fill(0)
                nisa.nc_matmul(dst, lhs0, rhs0)
                nisa.nc_matmul(dst, lhs1, rhs1)
                expected = np.zeros(dst.data.shape, dtype=dst.data.dtype)
                expected += np.asarray(
                    np.asarray(lhs0.data).T @ np.asarray(rhs0.data),
                    dtype=dst.data.dtype,
                )
                expected += np.asarray(
                    np.asarray(lhs1.data).T @ np.asarray(rhs1.data),
                    dtype=dst.data.dtype,
                )
                _assert_tensor_equal(dst.data, expected)
                seed += 4


def test_nc_matmul_flattens_rank3_moving_operand(patched_scope):
    """nc_matmul should flatten moving free dims before the matrix multiply."""
    del patched_scope
    stationary = _typed_ndarray((128, 32), "float32", buffer="sbuf")
    moving = _typed_ndarray((128, 2, 16), "float32", buffer="sbuf", seed=1)
    dst = b2.NDArray(shape=(32, 2, 16), dtype=nl.float32, buffer="psum")
    nisa.nc_matmul(dst, stationary, moving)
    expected = stationary.data.reshape(128, 32).T @ moving.data.reshape(128, 32)
    _assert_tensor_equal(dst.data, expected.reshape(dst.shape))


@pytest.mark.parametrize("dtype_name", ("float64", "float32", "bfloat16", "int16"))
def test_reciprocal_value_dtype_cases(patched_scope, dtype_name):
    """reciprocal should match numpy and overwrite non-zero destination."""
    del patched_scope
    dtype = _dtype_token(dtype_name)
    src = _nd(np.array([[0.0, 23.0, 4.0 / 7.0]], dtype=STORAGE_DTYPES[dtype_name]))
    dst = b2.NDArray(shape=src.shape, dtype=dtype, buffer="sbuf")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        nisa.reciprocal(dst, src)
        expected = np.reciprocal(np.asarray(src.data, dtype=np.float32)).astype(
            dst.data.dtype
        )
    _assert_tensor_equal(dst.data, expected)

    dst_overwrite = _nd(np.full(src.shape, 9, dtype=dst.data.dtype))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        nisa.reciprocal(dst_overwrite, src)
    _assert_tensor_equal(dst_overwrite.data, expected)


@pytest.mark.parametrize("dtype_name", ("float64", "float32", "bfloat16", "int16"))
@pytest.mark.parametrize(
    "op_name,np_op",
    (
        ("add", np.add),
        ("subtract", np.subtract),
        ("multiply", np.multiply),
        ("maximum", np.maximum),
    ),
)
def test_binary_ops_match_numpy_with_overflow_cases(
    patched_scope, dtype_name, op_name, np_op
):
    """tensor_tensor/tensor_scalar should match numpy for binary op cases."""
    del patched_scope
    lhs = _matrix((19, 23), "linspace", dtype_name, 31)
    rhs = _matrix((19, 23), "randn", dtype_name, 47)
    op = getattr(nl, op_name)
    lhs_nd = b2.NDArray(value=lhs, dtype=_dtype_token(dtype_name), buffer="sbuf")
    rhs_nd = b2.NDArray(value=rhs, dtype=_dtype_token(dtype_name), buffer="sbuf")

    tensor_dst = b2.NDArray(
        shape=lhs.shape, dtype=_dtype_token(dtype_name), buffer="sbuf"
    )
    nisa.tensor_tensor(tensor_dst, lhs_nd, rhs_nd, op)
    _assert_tensor_equal(tensor_dst.data, np_op(lhs, rhs))

    scalar = rhs[0, 0].item()
    scalar_dst = b2.NDArray(
        shape=lhs.shape, dtype=_dtype_token(dtype_name), buffer="sbuf"
    )
    nisa.tensor_scalar(scalar_dst, lhs_nd, op, scalar)
    _assert_tensor_equal(scalar_dst.data, np_op(lhs, scalar))

    if dtype_name == "int16":
        lhs_overflow = np.array([32767, -32768, 30000, -30000], dtype=np.int16)
        rhs_overflow = np.array([1, 1, 10000, -10000], dtype=np.int16)
        lhs_overflow_nd = b2.NDArray(
            value=lhs_overflow.reshape(1, 4), dtype=nl.int16, buffer="sbuf"
        )
        rhs_overflow_nd = b2.NDArray(
            value=rhs_overflow.reshape(1, 4), dtype=nl.int16, buffer="sbuf"
        )

        overflow_dst = b2.NDArray(shape=(1, 4), dtype=nl.int16, buffer="sbuf")
        nisa.tensor_tensor(overflow_dst, lhs_overflow_nd, rhs_overflow_nd, op)
        assert np.array_equal(
            overflow_dst.data.reshape(-1), np_op(lhs_overflow, rhs_overflow)
        )

        overflow_scalar_dst = b2.NDArray(shape=(1, 4), dtype=nl.int16, buffer="sbuf")
        nisa.tensor_scalar(
            overflow_scalar_dst, lhs_overflow_nd, op, int(rhs_overflow[0])
        )
        assert np.array_equal(
            overflow_scalar_dst.data.reshape(-1),
            np_op(lhs_overflow, int(rhs_overflow[0])),
        )


@pytest.mark.parametrize("dtype_name", ("float64", "float32", "bfloat16", "int16"))
def test_sqrt_rsqrt_match_numpy_with_edge_values(patched_scope, dtype_name):
    """sqrt/rsqrt should match numpy for regular and edge-value inputs."""
    del patched_scope
    if dtype_name == "int16":
        values = np.array([0, 1, 4, 9, 256, 1024, 32767], dtype=np.int16)
    else:
        values = np.asarray(
            [0.0, 1.0, 4.0, 9.0, 256.0, 1e-12, 1e12], dtype=STORAGE_DTYPES[dtype_name]
        )
    values_nd = _nd(values.reshape(1, -1))
    dst = _zeros((1, values.shape[0]))
    nisa.activation(dst, nl.sqrt, values_nd)
    sqrt_actual = dst.data.reshape(-1)
    sqrt_expected = np.sqrt(values)
    _assert_tensor_equal(sqrt_actual, sqrt_expected)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        dst = _zeros((1, values.shape[0]))
        nisa.activation(dst, nl.rsqrt, values_nd)
        rsqrt_actual = dst.data.reshape(-1)
        rsqrt_expected = 1.0 / np.sqrt(values)
    _assert_tensor_equal(rsqrt_actual, rsqrt_expected)


def test_dma_copy_buffer_dtype_rmw_matrix(patched_scope):
    """dma_copy should match documented buffer/dtype matrix and rmw behavior."""
    del patched_scope
    shape = (19, 23)
    dtype_pairs = (
        ("float32", "float32"),
        ("bfloat16", "bfloat16"),
        ("bfloat16", "float32"),
    )
    for dst_buffer in ("hbm", "sbuf"):
        for src_buffer in ("hbm", "sbuf"):
            for dst_dtype_name, src_dtype_name in dtype_pairs:
                src_np = _matrix(shape, "arange", src_dtype_name, 61)
                src = b2.NDArray(
                    value=src_np,
                    dtype=_dtype_token(src_dtype_name),
                    buffer=src_buffer,
                )
                dst = b2.NDArray(
                    shape=shape,
                    dtype=_dtype_token(dst_dtype_name),
                    buffer=dst_buffer,
                )

                nisa.dma_copy(dst, src, dst_rmw_op=None)
                expected = np.asarray(src.data, dtype=dst.data.dtype)
                _assert_tensor_equal(dst.data, expected)

                dst_init = _matrix(shape, "linspace", dst_dtype_name, 79)
                dst.data[...] = np.asarray(dst_init, dtype=dst.data.dtype)
                nisa.dma_copy(dst, src, dst_rmw_op=nl.add)
                expected_rmw = np.asarray(
                    np.add(np.asarray(dst_init), np.asarray(src.data)),
                    dtype=dst.data.dtype,
                )
                _assert_tensor_equal(dst.data, expected_rmw)

    with pytest.raises(ValueError, match="dma_copy only supports HBM/SBUF"):
        nisa.dma_copy(
            b2.NDArray(shape=(4, 4), dtype=_dtype_token("float32"), buffer="sbuf"),
            b2.NDArray(shape=(4, 4), dtype=_dtype_token("float32"), buffer="psum"),
        )


def test_ndarray_has_no_broadcast_to_method():
    """NDArray should not expose broadcast_to helper."""
    assert not hasattr(b2.NDArray(value=np.array([1])), "broadcast_to")


def test_tensor_tensor_rejects_scalar_vector_broadcast(patched_scope):
    """tensor_tensor should reject scalar/vector broadcasts."""
    del patched_scope
    dst = _zeros((2, 3))
    lhs = _nd(np.arange(6, dtype=np.float32).reshape(2, 3))
    rhs_vector = b2.NDArray(value=np.array([1.0, 2.0, 3.0], dtype=np.float32))
    with pytest.raises(
        ValueError,
        match="tensor_tensor doesn't broadcast scalars/vectors; use tensor_scalar",
    ):
        nisa.tensor_tensor(dst, lhs, rhs_vector, nl.add)


def test_tensor_tensor_views_second_last_dims_to_dst(patched_scope):
    """tensor_tensor should reshape src free dims to match dst free dims."""
    del patched_scope
    dst = _zeros((2, 3, 2))
    lhs = _nd(np.arange(12, dtype=np.float32).reshape(2, 6))
    rhs = _nd(np.ones((2, 6), dtype=np.float32))
    nisa.tensor_tensor(dst, lhs, rhs, nl.add)
    expected = lhs.data.reshape(dst.shape) + rhs.data.reshape(dst.shape)
    assert np.array_equal(dst.data, expected)


@pytest.mark.parametrize(
    "lhs_shape,rhs_shape,dst_shape",
    (
        ((32, 6), (32, 2, 3), (32, 6)),
        ((32, 6), (32, 2, 3), (32, 3, 2)),
        ((32, 2, 6), (32, 3, 4), (32, 3, 4)),
    ),
)
def test_tensor_tensor_accepts_mismatch3_ap_compatible_reshapes(
    patched_scope, lhs_shape, rhs_shape, dst_shape
):
    """tensor_tensor should accept mismatch3 AP-compatible reshape cases."""
    del patched_scope
    lhs = _nd(np.arange(np.prod(lhs_shape), dtype=np.float32).reshape(lhs_shape))
    rhs = _nd(np.arange(np.prod(rhs_shape), dtype=np.float32).reshape(rhs_shape) + 1.0)
    dst = _zeros(dst_shape)
    nisa.tensor_tensor(dst, lhs, rhs, nl.add)
    expected = lhs.data.reshape(lhs_shape[0], -1) + rhs.data.reshape(rhs_shape[0], -1)
    assert np.array_equal(dst.data, expected.reshape(dst_shape))


def test_exponential_requires_nc_v4_or_newer(patched_scope):
    """exponential should reject simulated pre-v4 targets."""
    del patched_scope
    prev = b2.nki_builder.nc_version
    b2.nki_builder.nc_version = nisa.nc_version.gen3
    try:
        with pytest.raises(
            ValueError, match="exponential only supports >= NeuronCore-v4"
        ):
            getattr(nisa, "exponential")(_zeros((1, 2)), _nd([[0.0, 1.0]]))
    finally:
        b2.nki_builder.nc_version = prev


def test_ndarray_mutation_is_not_supported(patched_scope):
    """In-place mutation on tensor objects should raise."""
    del patched_scope
    tensor = nl.ndarray((2, 2), dtype=nl.float32, buffer=nl.sbuf)
    with pytest.raises(RuntimeError, match="mutation not supported"):
        tensor[0, 0] = 1


def test_ndarray_is_zero_initialized(patched_scope):
    """ndarray values should be initialized to zero."""
    del patched_scope
    src = nl.ndarray((2, 2), dtype=nl.float32, buffer=nl.sbuf)
    assert np.array_equal(src.data, np.zeros((2, 2), dtype=src.data.dtype))


def test_ndarray_use_before_write_is_rejected(patched_scope):
    """Using an allocated tensor before a write should raise undefined-use."""
    del patched_scope
    src = nl.ndarray((2, 2), dtype=nl.float32, buffer=nl.sbuf)
    dst = _zeros((2, 2), dtype=np.float32)
    with pytest.raises(RuntimeError, match="Illegal IR, encountered undefined use"):
        nisa.tensor_copy(dst, src)


def test_slice_writes_mark_parent_tensor_defined(patched_scope):
    """Slice writes should allow a later full-tensor read from the parent."""
    del patched_scope
    src = _nd(np.arange(128 * 32, dtype=np.float32).reshape(128, 32))
    src_tile = nl.ndarray(src.shape, dtype=nl.float32, buffer=nl.sbuf)
    assembled = nl.ndarray(src.shape, dtype=nl.float32, buffer=nl.sbuf)
    out = b2.NDArray(shape=src.shape, dtype=nl.float32, buffer="sbuf")

    nisa.dma_copy(dst=src_tile, src=src)
    nisa.tensor_copy(dst=assembled[:, :16], src=src_tile[:, :16])
    nisa.tensor_copy(dst=assembled[:, 16:], src=src_tile[:, 16:])
    nisa.tensor_copy(dst=out, src=assembled)

    assert np.array_equal(out.data, src.data)


def test_ndarray_int_index_drops_axis():
    """HBM integer indexing should match NumPy rank-reducing semantics."""
    tensor = _nd(np.arange(24, dtype=np.float32).reshape(2, 3, 4), buffer="hbm")
    slice0 = tensor[0]
    slice1 = tensor[:, 0]
    assert slice0.shape == (3, 4)
    assert slice1.shape == (2, 4)
    assert np.array_equal(slice0.data, tensor.data[0])
    assert np.array_equal(slice1.data, tensor.data[:, 0])


@pytest.mark.parametrize("buffer", ("sbuf", "psum"))
def test_ndarray_sbuf_psum_partition_index_is_rejected(buffer):
    """SBUF/PSUM indexing should preserve the partition dimension."""
    tensor = _nd(np.arange(24, dtype=np.float32).reshape(2, 3, 4), buffer=buffer)
    with pytest.raises(ValueError, match="preserve the partition dim"):
        _ = tensor[0]
    assert np.array_equal(tensor[:, 0].data, tensor.data[:, 0])


def test_dma_copy_ap_mismatch_error_message(patched_scope):
    """dma_copy should reject non-view reshapes that hardware does not compile."""
    del patched_scope
    rows, cols = 2, 8
    src = _nd(np.arange(rows * cols, dtype=np.float32).reshape(rows, cols))
    dst = _nd(np.zeros((rows, cols // 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="Expect AP same number of elements"):
        nisa.dma_copy(dst=dst, src=src)


def test_tensor_negation_is_not_supported(patched_scope):
    """Unary negation on tensor objects should raise."""
    del patched_scope
    row_max = nl.ndarray((2, 1), dtype=nl.float32, buffer=nl.sbuf)
    with pytest.raises(TypeError, match="cannot negate values of this type"):
        _ = -row_max


def test_ndarray_public_surface_and_value_backed_paths(patched_scope):
    """NDArray surface methods should behave consistently on public allocations."""
    del patched_scope
    tensor = b2.ndarray(
        (2, 2),
        dtype=nl.float32,
        buffer=nl.sbuf,
        value=np.arange(4, dtype=np.float32),
    )
    assert tensor.shape == (2, 2)
    assert tensor.address == tensor.data.ctypes.data
    assert tensor.data_ptr() == tensor.address
    assert tensor.stride() == (2, 1)
    assert tensor.element_size() == tensor.data.dtype.itemsize
    assert tensor.cpu() is tensor
    assert tensor.detach() is tensor
    assert tensor.numpy() is tensor.data
    assert repr(tensor) == "NDArray(shape=(2, 2), dtype=float32)"

    reshaped = tensor.reshape(1, 4)
    assert reshaped.shape == (1, 4)
    assert np.array_equal(reshaped.data, np.arange(4, dtype=np.float32).reshape(1, 4))


def test_ndarray_index_assignment_and_binary_surface(patched_scope):
    """NDArray indexing and assignment should preserve beta2 tensor semantics."""
    del patched_scope
    tensor = _nd(np.arange(6, dtype=np.float32).reshape(2, 3), buffer="hbm")
    index = 1

    assert np.array_equal(tensor[index].data, tensor.data[1])
    assert np.array_equal(tensor[-1].data, tensor.data[1])

    tensor[:, :] = _nd(np.full((2, 3), 7.0, dtype=np.float32))
    assert np.array_equal(tensor.data, np.full((2, 3), 7.0, dtype=np.float32))

    tensor[0] = _nd(np.full((1, 3), 5.0, dtype=np.float32), buffer="hbm")
    assert np.array_equal(
        tensor.data,
        np.array([[5.0, 5.0, 5.0], [7.0, 7.0, 7.0]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="Expect AP same number of elements"):
        tensor[:, :] = _nd(np.arange(6, dtype=np.float32), buffer="hbm")

    with pytest.raises(TypeError, match="binary operators on tensors not supported"):
        _ = tensor & tensor


def test_buffer_views_and_named_buffer_tokens(patched_scope):
    """Buffer-backed views should match public buffer token semantics."""
    del patched_scope

    class _BufferToken:
        """Minimal buffer token that mimics the NKI public token surface."""

        def __init__(self, name):
            self.name = name

    raw = b2.Buffer("sbuf", shape=(16,))
    assert raw.data is not None
    assert raw.data.shape == (16,)
    assert raw.data.dtype == np.uint8

    src = np.arange(4, dtype=np.float32).reshape(2, 2)
    byte_backed = b2.Buffer("sbuf", data=src.view(np.uint8).reshape(-1))
    byte_view = byte_backed.view(nl.float32, (2, 2))
    assert np.array_equal(byte_view.data, src)

    typed_backed = b2.Buffer("sbuf", data=src.copy())
    typed_view = typed_backed.view(nl.float32, (2, 2))
    assert np.array_equal(typed_view.data, src)

    with pytest.raises(ValueError, match="Buffer shape mismatch"):
        byte_backed.view(nl.float32, (3, 2))

    for name in ("hbm", "sbuf", "psum"):
        shape = (1,) if name == "hbm" else (1, 1)
        out = nl.ndarray(shape, dtype=nl.float32, buffer=_BufferToken(name))
        assert out.buffer == name


def test_exponential_reduce_res_records_row_sums(patched_scope):
    """exponential should optionally write the documented reduction output."""
    del patched_scope
    src = _nd(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
    dst = _zeros((2, 2), dtype=np.float32)
    reduce_res = _zeros((2, 1), dtype=np.float32)
    max_value = _nd(np.array([[0.0], [2.0]], dtype=np.float32))
    prev = b2.nki_builder.nc_version
    b2.nki_builder.nc_version = nisa.nc_version.gen4
    try:
        getattr(nisa, "exponential")(
            dst, src, max_value=max_value, reduce_res=reduce_res
        )
        expected = np.exp(src.data - max_value.data)
        assert np.array_equal(dst.data, expected)
        assert np.array_equal(reduce_res.data, expected.sum(axis=1, keepdims=True))
    finally:
        b2.nki_builder.nc_version = prev


def test_tensor_scalar_and_tensor_tensor_public_validation_edges(patched_scope):
    """Public tensor ops should cover remaining unsupported validation edges."""
    del patched_scope
    data = _typed_ndarray((2, 4), "int32", buffer="sbuf")
    dst = b2.NDArray(shape=(2, 4), dtype=nl.int32, buffer="sbuf")
    operand = _nd(np.full((2, 4), 2, dtype=np.int32))
    with pytest.raises(
        ValueError,
        match="1st Immediate pointer's number of elements per partition must be 1",
    ):
        nisa.tensor_scalar(dst, data, nl.add, operand)

    with pytest.raises(ValueError, match="operand0/1 dtype must match bitvec integer"):
        nisa.tensor_scalar(
            dst,
            data,
            nl.logical_or,
            _nd(np.ones((2, 4), dtype=np.float32)),
        )

    with pytest.raises(ValueError, match="Unsupported op"):
        nisa.tensor_scalar(dst, data, np.add, 1)

    with pytest.raises(ValueError, match="Unsupported op"):
        nisa.tensor_scalar(dst, data, nl.add, 1, op1=np.add, operand1=1)

    logic_data = _typed_ndarray((2, 2), "int32", buffer="sbuf")
    logic_dst = b2.NDArray(shape=(2, 2), dtype=nl.int32, buffer="sbuf")
    nisa.tensor_tensor(logic_dst, logic_data, logic_data, nl.logical_or)
    assert np.array_equal(
        logic_dst.data,
        np.logical_or(logic_data.data, logic_data.data),
    )

    with pytest.raises(ValueError, match="Unsupported op"):
        nisa.tensor_tensor(logic_dst, logic_data, logic_data, np.add)

    reduce_dst = b2.NDArray(shape=(2, 1), dtype=nl.float32, buffer="sbuf")
    with pytest.raises(ValueError, match="integer dtypes for bitvec operators"):
        nisa.tensor_reduce(
            reduce_dst,
            nl.logical_or,
            _typed_ndarray((2, 4), "float32"),
            axis=[1],
        )


def test_misc_public_surface_branches(patched_scope):
    """Public beta2 helpers should expose stable string and unpatch behavior."""
    del patched_scope
    assert str(nl.add) == "SubOp(add)"
    assert b2.nki_unpatch_lang() is None

    data = _typed_ndarray((2, 4), "float32", buffer="sbuf")
    dst = b2.NDArray(shape=(2, 4), dtype=nl.float32, buffer="sbuf")
    with pytest.raises(ValueError, match="Unsupported op"):
        nisa.activation(dst, np.exp, data)


def test_builder_public_setters_update_grid_state():
    """Builder setters should update grid dimensions and visible program ids."""
    b2.nki_builder.set_grid_dim(8)
    assert b2.nki_builder.grid_dims == (8,)
    assert b2.nki_builder.grid_idx == [
        0,
    ]

    b2.nki_builder.set_grid_idx(6)
    assert b2.nki_builder.grid_idx == [
        6,
    ]

    b2.nki_builder.set_grid_idx(3, 2, 7)
    assert b2.nki_builder.grid_idx == [
        3,
    ]


def test_interpreted_function_public_grid_edge_cases(patched_scope):
    """Interpreted kernels should cover the remaining public grid control flow."""
    del patched_scope

    def copy_kernel(src, dst):
        """Copy one tile into the destination."""
        nisa.dma_copy(dst=dst, src=src)

    fn = b2.NKIBeta2InterpretedFunction(copy_kernel)
    src = np.arange(4, dtype=np.float32).reshape(2, 2)
    dst = np.zeros_like(src)
    fn.run(src, dst, grid=2)
    assert np.array_equal(dst, src)

    with pytest.raises(ValueError, match="Grid must have at least one dimension"):
        fn.run(src, dst, grid=())

    class _PreStop:
        """Stop execution before the kernel body runs."""

        def grid_callback(self, grid):
            self.grid = grid

        def grid_idx_callback(self, idx):
            self.idx = idx

        def arg_callback(self, name, arg, ret):
            self.arg = (name, arg, ret)

        def pre_run_callback(self, fn):
            self.fn = fn
            return False

        def post_run_callback(self, fn):
            raise AssertionError(f"post_run_callback should not run for {fn}")

    dst_pre = np.zeros_like(src)
    fn.run(src, dst_pre, grid=(1,), client_manager=_PreStop())
    assert np.array_equal(dst_pre, np.zeros_like(src))

    class _PostStop:
        """Stop execution immediately after the first kernel invocation."""

        def grid_callback(self, grid):
            self.grid = grid

        def grid_idx_callback(self, idx):
            self.idx = idx

        def arg_callback(self, name, arg, ret):
            self.arg = (name, arg, ret)

        def pre_run_callback(self, fn):
            self.fn = fn
            return True

        def post_run_callback(self, fn):
            self.post_fn = fn
            return False

    dst_post = np.zeros_like(src)
    fn.run(src, dst_post, grid=(2,), client_manager=_PostStop())
    assert np.array_equal(dst_post, src)

    class _RankMismatch:
        """Mutate the tracked grid rank before execution to trigger validation."""

        def grid_callback(self, grid):
            del grid
            b2.nki_builder.set_grid_dim(1, 1)

        def grid_idx_callback(self, idx):
            self.idx = idx

        def arg_callback(self, name, arg, ret):
            self.arg = (name, arg, ret)

        def pre_run_callback(self, fn):
            self.fn = fn
            return True

        def post_run_callback(self, fn):
            self.post_fn = fn
            return True

    with pytest.raises(ValueError, match="1D SPMD"):
        fn.run(src, np.zeros_like(src), grid=(1,), client_manager=_RankMismatch())


def test_dma_copy_contiguous_hbm_block_into_2d_tile(patched_scope):
    """An HBM row block reshaped into a 2-D SBUF tile (two rows per partition) copies in row-major order."""
    del patched_scope
    src = _nd(np.arange(8 * 4, dtype=np.float32).reshape(8, 4), buffer="hbm")
    dst = _nd(np.zeros((4, 8), dtype=np.float32))
    nisa.dma_copy(dst=dst, src=src)
    assert np.array_equal(dst.data, np.arange(32, dtype=np.float32).reshape(4, 8))
    back = _nd(np.zeros((8, 4), dtype=np.float32), buffer="hbm")
    nisa.dma_copy(dst=back, src=dst)
    assert np.array_equal(back.data, src.data)

