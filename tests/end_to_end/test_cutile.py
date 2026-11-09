# ruff: noqa: E402

import importlib.util
from io import BytesIO
import math
from pathlib import Path

import numpy as np
import pytest

ct = pytest.importorskip("cuda.tile")

import tilelens
from tilelens.clients import Tracer
from tilelens.core.data import Dot, Grid, Load, ReduceSum, Store
from tilelens.core.frontend.cutile import frontend, interpreted_launches
from tilelens.core.simulation.cutile import (
    CuTileInterpretedFunction,
    DTYPES,
    cutile_builder,
)
from tilelens.core.trace import launches
from tilelens.visualizer import interface


@pytest.fixture(autouse=True)
def clear_traces():
    tilelens.clear()
    yield
    tilelens.clear()


@ct.kernel
def copy_kernel(src, dst):
    tile = ct.load(src, (0,), (4,))
    ct.store(dst, (0,), tile)


@ct.kernel
def matmul_kernel(lhs, rhs, dst):
    x = ct.load(lhs, (0, 0), (2, 4))
    y = ct.load(rhs, (0, 0), (4, 2))
    ct.store(dst, (0, 0), ct.matmul(x, y))


@ct.kernel
def reduction_kernel(src, total, scan):
    tile = ct.load(src, (0,), (1024,))
    ct.store(total, (0,), ct.sum(tile))
    ct.store(scan, (0,), ct.cumsum(tile, axis=0))


@ct.kernel
def boolean_kernel(src, total, dst):
    tile = ct.load(src, (0,), (4,))
    ct.store(total, (0,), ct.sum(tile))
    ct.store(dst, (0,), tile + 2)


@ct.kernel
def indexed_kernel(src, dst):
    view = src.slice(0, 1, 4)
    indices = ct.arange(4, dtype=ct.int32) - 1
    values = ct.gather(view, indices, padding_value=0)
    product = ct.mma(
        ct.reshape(values, (2, 2)),
        ct.ones((2, 2), ct.float32),
        ct.zeros((2, 2), ct.float32),
    )
    ct.scatter(dst, ct.arange(4, dtype=ct.int32), ct.reshape(product, (4,)))


def test_indexed_kernel():
    src = np.array([10, 20, 30, 40], np.float32)
    dst = np.zeros(4, np.float32)
    tilelens.trace("tracer", frontend="cutile")(indexed_kernel)[(1,)](src, dst)
    np.testing.assert_array_equal(dst, [20, 20, 70, 70])
    records = launches[-1].records
    loads = [r for r in records if isinstance(r, Load)]
    stores = [r for r in records if isinstance(r, Store)]
    assert len(loads) == len(stores) == 1
    np.testing.assert_array_equal(loads[0].masks, [False, True, True, True])
    np.testing.assert_array_equal(loads[0].offsets[loads[0].masks], [4, 8, 12])
    assert sum(isinstance(r, Dot) for r in records) == 1
    check_visualizer()


def load_example():
    path = Path(__file__).resolve().parents[2] / "examples" / "cutile" / "matmul.py"
    spec = importlib.util.spec_from_file_location("cutile_matmul_e2e", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_visualizer():
    interface.update_global_data(force=True)
    client = interface.app.test_client()
    response = client.get("/api/data")
    assert response.status_code == 200
    assert response.get_json()["ops"]["visualization_data"]
    for key, record in interface.raw_tensor_data.items():
        if "global_tensor" in record:
            response = client.post("/api/getLoadTensor", json={"uuid": key})
            assert response.status_code == 200
            payload = response.get_json()
            assert payload["shape"] == list(record["global_tensor"].shape)
            assert np.asarray(payload["values"]).size > 0
            np.testing.assert_array_equal(payload["values"], record["global_tensor"])
            response = client.post(
                "/api/getLoadValue", json={"uuid": key, "x": 0, "y": 0, "z": 0}
            )
            assert response.status_code == 200
            assert response.get_json()["value"] == float(
                record["global_tensor"].flat[0]
            )
        elif "input_data" in record:
            response = client.post("/api/getMatmulC", json={"uuid": key})
            assert response.status_code == 200
            np.testing.assert_allclose(
                response.get_json()["values"],
                record["input_data"] @ record["other_data"],
            )


@pytest.mark.parametrize("dtype", [ct.bfloat16, ct.float8_e4m3fn, ct.float8_e5m2])
@pytest.mark.parametrize("matmul", [False, True])
def test_low_precision_trace(dtype, matmul, tmp_path):
    storage = DTYPES[dtype]
    if matmul:
        lhs = np.ones((2, 4), dtype=storage)
        rhs = np.full((4, 2), 2, dtype=storage)
        dst = np.zeros((2, 2), dtype=storage)
        tilelens.trace("tracer", frontend="cutile")(matmul_kernel)[(1,)](lhs, rhs, dst)
        np.testing.assert_array_equal(dst.astype(np.float32), np.full((2, 2), 8))
    else:
        src = np.arange(4).astype(storage)
        dst = np.zeros_like(src)
        tilelens.trace("tracer", frontend="cutile")(copy_kernel)[(1,)](src, dst)
        np.testing.assert_array_equal(dst, src)
    assert dst.dtype == storage
    check_visualizer()
    path = tmp_path / "low_precision.tvz"
    tilelens.save(path)
    tilelens.clear()
    tilelens.load(path)
    check_visualizer()
    assert any(t.element_size() == storage.itemsize for t in launches[-1].tensors)
    assert any(t.dtype == str(dtype) for t in launches[-1].tensors)


def test_reduction_kernels():
    src = np.ones(1024, dtype=DTYPES[ct.bfloat16])
    total, scan = np.zeros(1, dtype=src.dtype), np.zeros_like(src)
    tilelens.trace("tracer", frontend="cutile")(reduction_kernel)[(1,)](
        src, total, scan
    )
    np.testing.assert_array_equal(total.astype(np.float32), [1024])
    np.testing.assert_array_equal(scan, np.arange(1, 1025).astype(src.dtype))
    tilelens.clear()
    mask = np.array([True, False, True, True])
    count, output = np.zeros(1, np.int32), np.zeros(4, np.int32)
    tilelens.trace("tracer", frontend="cutile")(boolean_kernel)[(1,)](
        mask, count, output
    )
    np.testing.assert_array_equal(count, [3])
    np.testing.assert_array_equal(output, [3, 2, 3, 3])


@pytest.mark.parametrize("shape", [(4, 8, 16), (3, 5, 9)])
def test_example_kernel(shape, tmp_path):
    example = load_example()
    m, k, n = shape
    lhs = np.arange(m * k, dtype=np.float32).reshape(m, k)
    rhs = np.arange(k * n, dtype=np.float32).reshape(k, n)
    dst = np.zeros((m, n), np.float32)
    grid = (math.ceil(m / example.TILE_M), math.ceil(n / example.TILE_N))
    tilelens.trace("tracer", frontend="cutile")(example.matmul_kernel)[grid](
        lhs, rhs, dst
    )
    np.testing.assert_allclose(dst, lhs @ rhs)

    records = launches[-1].records
    k_tiles = math.ceil(k / example.TILE_K)
    assert sum(isinstance(r, Grid) for r in records) == math.prod(grid)
    assert sum(isinstance(r, Dot) for r in records) == math.prod(grid) * k_tiles
    assert sum(isinstance(r, Store) for r in records) == math.prod(grid)
    loads = [r for r in records if isinstance(r, Load)]
    assert len(loads) == math.prod(grid) * k_tiles * 2
    assert all(r.call_path for r in loads)
    if m == 3:
        assert any(not r.masks.all() for r in loads)

    check_visualizer()
    assert interface.app.test_client().get("/").status_code == 200
    path = tmp_path / "matmul.tvz"
    tilelens.save(path)
    tilelens.clear()
    tilelens.load(path)
    check_visualizer()


def test_float64_reduction_kernel():
    src = np.zeros(1024, np.float64)
    src[0] = 1e300
    total, scan = np.zeros(1, np.float64), np.zeros_like(src)
    tilelens.trace("tracer", frontend="cutile")(reduction_kernel)[(1,)](
        src, total, scan
    )
    np.testing.assert_array_equal(total, [1e300])
    np.testing.assert_array_equal(scan, np.full(1024, 1e300))


def test_kernels_compile():
    compilation = pytest.importorskip("cuda.tile.compilation")
    cases = [
        (
            load_example().matmul_kernel,
            [(ct.float32, (3, 5)), (ct.float32, (5, 9)), (ct.float32, (3, 9))],
        ),
        (indexed_kernel, [(ct.float32, (4,))] * 2),
        (boolean_kernel, [(ct.bool_, (4,)), (ct.int32, (1,)), (ct.int32, (4,))]),
        (
            reduction_kernel,
            [(ct.bfloat16, (1024,)), (ct.bfloat16, (1,)), (ct.bfloat16, (1024,))],
        ),
        (
            matmul_kernel,
            [(ct.bfloat16, (2, 4)), (ct.bfloat16, (4, 2)), (ct.bfloat16, (2, 2))],
        ),
        (
            reduction_kernel,
            [(ct.float64, (1024,)), (ct.float64, (1,)), (ct.float64, (1024,))],
        ),
    ]
    for kernel, arrays in cases:
        parameters = [
            compilation.ArrayConstraint(
                dtype,
                len(shape),
                index_dtype=ct.int32,
                stride_lower_bound_incl=0,
                alias_groups=(),
                may_alias_internally=False,
                shape_constant=shape,
            )
            for dtype, shape in arrays
        ]
        signature = compilation.KernelSignature(
            parameters, compilation.CallingConvention.cutile_python_v2()
        )
        buffer = BytesIO()
        compilation.export_kernel(
            kernel,
            [signature],
            buffer,
            gpu_code="sm_100",
            output_format="tileir_bytecode",
            bytecode_version="13.1",
        )
        assert buffer.getvalue()


def test_interpreted_kernel():
    @ct.kernel
    def add_kernel(a, b, out):
        block = ct.bid(0)
        x = ct.load(a, (block,), (4,), padding_mode=ct.PaddingMode.ZERO)
        y = ct.load(b, (block,), (4,), padding_mode=ct.PaddingMode.ZERO)
        ct.store(out, (block,), x + y)

    a = np.arange(10, dtype=np.int32)
    b = np.arange(10, dtype=np.int32) * 2
    output = np.empty_like(a)
    original_load = ct.load

    CuTileInterpretedFunction(add_kernel).run(a, b, output, grid=(3,))

    np.testing.assert_array_equal(output, a + b)
    assert ct.load is original_load


def test_softmax_kernel_and_builtin_max_restore():
    @ct.kernel
    def softmax(src, dst):
        x = ct.load(src, (ct.bid(0), 0), (1, 8), padding_mode=ct.PaddingMode.NEG_INF)
        m = max(ct.full((), -float("inf"), ct.float32), ct.max(x))
        probabilities = ct.exp(x - m)
        ct.store(dst, (ct.bid(0), 0), probabilities / ct.sum(probabilities))

    data = np.array([[100, 101, 102, 103, 104], [-3, -2, -1, 0, 1]], np.float32)
    result = np.zeros_like(data)
    original_exp = ct.exp
    CuTileInterpretedFunction(softmax).run(data, result, grid=(2,))
    expected = np.exp(data - data.max(1, keepdims=True))
    expected /= expected.sum(1, keepdims=True)
    np.testing.assert_allclose(result, expected, rtol=1e-6)
    assert ct.exp is original_exp


def test_kernel_preserves_shadowed_builtins_and_keyword_defaults(monkeypatch):
    calls = []

    def helper(x, y):
        calls.append(True)
        return x + y

    def kernel(src, dst, *, increment=2):
        values = ct.load(src, (0,), (4,))
        ct.store(dst, (0,), max(values, increment))

    monkeypatch.setitem(kernel.__globals__, "max", helper)
    src = np.arange(4, dtype=np.float32)
    dst = np.zeros_like(src)
    CuTileInterpretedFunction(kernel).run(src, dst)
    np.testing.assert_array_equal(dst, src + 2)
    assert calls == [True]


def test_interpreted_launches_keep_records_per_launch():
    src = np.arange(4, dtype=np.float32)
    dst = np.zeros(4, dtype=np.float32)
    with interpreted_launches(Tracer()):
        ct.launch(None, (1,), copy_kernel, (src, dst))
        ct.launch(None, (1,), copy_kernel, (src, dst))

    # A reused tracer does not repeat the first launch's records.
    assert [len(launch.records) for launch in launches] == [3, 3]
    np.testing.assert_array_equal(dst, src)


def block_copy_kernel(src, dst):
    block = ct.bid(0)
    tile = ct.load(src, index=(block,), shape=(4,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(dst, index=(block,), tile=tile)


def test_atomic_updates_trace_memory_and_return_old_values():
    @ct.kernel
    def atomic_kernel(values, old):
        previous = ct.atomic_add(values, ct.full((2,), 0, ct.int32), 1)
        ct.store(old, (0,), previous)

    values = np.array([10], np.int32)
    old = np.empty(2, np.int32)
    tilelens.trace("tracer", frontend="cutile")(atomic_kernel)[(1,)](values, old)
    np.testing.assert_array_equal(values, [12])
    np.testing.assert_array_equal(old, [10, 11])
    records = launches[-1].records
    assert sum(isinstance(r, Load) for r in records) == 1
    assert sum(isinstance(r, Store) for r in records) == 2


def test_sampling_and_strided_offsets():
    src = np.arange(24, dtype=np.float32)[::2]
    dst = np.zeros(24, dtype=np.float32)[::2]
    traced = tilelens.trace(Tracer(grid_idx=1), frontend="cutile")(block_copy_kernel)
    traced.run(src=src, dst=dst, grid=(3,))
    np.testing.assert_array_equal(dst, src)
    records = launches[-1].records
    assert [type(r) for r in records] == [Grid, Grid, Load, Store, Grid]
    np.testing.assert_array_equal(records[2].offsets, np.arange(4, 8) * 8)
    assert launches[-1].grid == (3, 1, 1)


@pytest.mark.parametrize("traced", [False, True])
def test_failure_restores_patches(traced):
    original_load = ct.load
    original_builder_load = cutile_builder.load

    @ct.kernel
    def failing_kernel(src):
        ct.load(src, (0,), (4,))
        raise RuntimeError("intentional failure")

    runner = (
        tilelens.trace("tracer", frontend="cutile")(failing_kernel)
        if traced
        else CuTileInterpretedFunction(failing_kernel)
    )
    with pytest.raises(RuntimeError, match="intentional failure"):
        runner.run(np.arange(4, dtype=np.float32), grid=(1,))
    assert ct.load is original_load
    assert cutile_builder.load == original_builder_load
    assert not launches


@pytest.mark.parametrize("traced", [False, True])
def test_frontend_owns_language_patch(monkeypatch, traced):
    original_load = ct.load
    patch_lang = frontend.patch_lang
    calls = []

    def record_patch(fn, client_manager=None):
        calls.append(client_manager)
        return patch_lang(fn, client_manager)

    monkeypatch.setattr(frontend, "patch_lang", record_patch)
    runner = (
        tilelens.trace(frontend="cutile")(block_copy_kernel)
        if traced
        else CuTileInterpretedFunction(block_copy_kernel)
    )
    src = np.arange(6, dtype=np.float32)
    dst = np.zeros_like(src)
    runner.run(src, dst, grid=(2,))
    np.testing.assert_array_equal(dst, src)
    assert len(calls) == 1
    assert (calls[0] is not None) == traced
    assert ct.load is original_load
    if traced:
        assert [type(r) for r in launches[-1].records] == [Grid, Load, Store] * 2


def test_reduction_and_interpreter_wrapper():
    def reduce_kernel(src, dst):
        tile = ct.load(src, (0,), (4,))
        ct.store(dst, (0,), ct.sum(tile, axis=0))

    traced = tilelens.trace(frontend="cutile")(CuTileInterpretedFunction(reduce_kernel))
    dst = np.zeros(1, dtype=np.float32)
    traced(np.arange(4, dtype=np.float32), dst)
    np.testing.assert_array_equal(dst, [6])
    assert [type(r) for r in launches[-1].records] == [Grid, Load, ReduceSum, Store]


@pytest.mark.parametrize("grid", [2, (2,), (2, 2), (2, 2, 2)])
@pytest.mark.parametrize("trace", [False, True])
def test_grid_axes(grid, trace):
    def kernel(dst):
        x, y, z = ct.bid(0), ct.bid(1), ct.bid(2)
        value = ct.full((1, 1, 1), 100 * x + 10 * y + z, ct.int32)
        ct.store(dst, (x, y, z), value)

    shape = (grid,) if isinstance(grid, int) else grid
    shape += (1,) * (3 - len(shape))
    dst = np.full(shape, -1, dtype=np.int32)
    runner = (
        tilelens.trace(frontend="cutile")(kernel)
        if trace
        else CuTileInterpretedFunction(kernel)
    )
    runner.run(dst, grid=grid)
    x, y, z = np.indices(shape)
    np.testing.assert_array_equal(dst, 100 * x + 10 * y + z)


def test_unsupported_client_is_rejected():
    with pytest.raises(ValueError, match="only the tracer"):
        tilelens.trace("profiler", frontend="cutile")(block_copy_kernel)


def test_trace_tiled_views(tmp_path):
    tilelens.clear()

    @tilelens.trace(frontend="cutile")
    def kernel(src, dst):
        src_view = src.slice(0, 1, 4).slice(1, 2, 7)
        dst_view = dst.slice(0, 1, 4).slice(1, 2, 7)
        tiled = src_view.tiled_view((2, 4), padding_mode=ct.PaddingMode.ZERO)
        output = dst_view.tiled_view((2, 4))
        for i in range(tiled.num_tiles(0)):
            for j in range(tiled.num_tiles(1)):
                output.store((i, j), tiled.load((i, j)) + 1)

    src = np.arange(32, dtype=np.float32).reshape(4, 8)
    dst = np.zeros_like(src)
    try:
        kernel(src, dst)
        np.testing.assert_array_equal(dst[1:4, 2:7], src[1:4, 2:7] + 1)
        records = [r for r in launches[-1].records if isinstance(r, (Load, Store))]
        assert len(records) == 8
        assert {r.ptr for r in records} == {src.ctypes.data, dst.ctypes.data}
        np.testing.assert_array_equal(
            records[0].offsets, np.array([[10, 11, 12, 13], [18, 19, 20, 21]]) * 4
        )
        assert records[-1].masks.sum() == 1
        tilelens.save(tmp_path / "views.tvz")
        tilelens.load(tmp_path / "views.tvz")
        interface.update_global_data(force=True)
        assert interface.app.test_client().get("/api/data").status_code == 200
    finally:
        tilelens.clear()
