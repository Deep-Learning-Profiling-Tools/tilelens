"""Tests for cuTile tracing and visualization."""
# ruff: noqa: E402

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

ct = pytest.importorskip("cuda.tile")

import tilelens
from tilelens.clients import Tracer
from tilelens.core.data import Dot, Grid, Load, Store
from tilelens.core.simulation.cutile import cutile_builder
from tilelens.core.trace import launches

example_path = Path(__file__).resolve().parents[2] / "examples" / "cutile" / "matmul.py"
spec = importlib.util.spec_from_file_location("cutile_matmul_example", example_path)
assert spec is not None
assert spec.loader is not None
matmul_example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matmul_example)
matmul_kernel = matmul_example.matmul_kernel
TILE_M, TILE_N = matmul_example.TILE_M, matmul_example.TILE_N


@pytest.fixture(autouse=True)
def clear_traces():
    tilelens.clear()
    yield
    tilelens.clear()


@pytest.mark.parametrize("shape", [(4, 8, 16), (3, 5, 9)])
def test_matmul_records_and_visualization(shape, tmp_path):
    m, k, n = shape
    lhs = np.arange(m * k, dtype=np.float32).reshape(m, k)
    rhs = np.arange(k * n, dtype=np.float32).reshape(k, n)
    out = np.zeros((m, n), dtype=np.float32)
    grid = (math.ceil(m / TILE_M), math.ceil(n / TILE_N))

    traced = tilelens.trace("tracer", frontend="cutile")(matmul_kernel)
    traced[grid](lhs, rhs, out)

    np.testing.assert_allclose(out, lhs @ rhs)
    records = launches[-1].records
    assert sum(isinstance(r, Grid) for r in records) == math.prod(grid)
    assert sum(isinstance(r, Dot) for r in records) == math.prod(grid) * math.ceil(
        k / 4
    )
    assert sum(isinstance(r, Store) for r in records) == math.prod(grid)
    loads = [r for r in records if isinstance(r, Load)]
    assert len(loads) == math.prod(grid) * math.ceil(k / 4) * 2
    assert all(r.call_path for r in loads)
    if m == 3:
        assert any(not r.masks.all() for r in loads)

    from tilelens.visualizer import interface

    def check_visualization():
        interface.update_global_data(force=True)
        response = interface.app.test_client().get("/api/data")
        assert response.status_code == 200
        assert response.get_json()["ops"]["visualization_data"]
        dots = {
            key: v for key, v in interface.raw_tensor_data.items() if "input_data" in v
        }
        assert len(dots) == math.prod(grid) * math.ceil(k / 4)
        for key, dot in dots.items():
            response = interface.app.test_client().post(
                "/api/getMatmulC", json={"uuid": key}
            )
            assert response.status_code == 200
            np.testing.assert_allclose(
                response.get_json()["values"], dot["input_data"] @ dot["other_data"]
            )
        assert interface.app.test_client().get("/").status_code == 200

    check_visualization()
    path = tmp_path / "matmul.tvz"
    tilelens.save(path)
    tilelens.clear()
    tilelens.load(path)
    check_visualization()


def copy_kernel(src, dst):
    block = ct.bid(0)
    tile = ct.load(src, index=(block,), shape=(4,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(dst, index=(block,), tile=tile)


def test_indexed_views_and_mma_use_shared_records():
    @ct.kernel
    def indexed_kernel(src, dst):
        view = src.slice(0, 1, 4)
        indices = ct.arange(4) - 1
        values = ct.gather(view, indices, padding_value=0)
        product = ct.mma(
            ct.reshape(values, (2, 2)),
            ct.ones((2, 2), ct.float32),
            ct.zeros((2, 2), ct.float32),
        )
        ct.scatter(dst, ct.arange(4), ct.reshape(product, (4,)))

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
    assert sum(isinstance(r, Load) for r in records) == 2
    assert sum(isinstance(r, Store) for r in records) == 3


def test_sampling_and_strided_offsets():
    src = np.arange(24, dtype=np.float32)[::2]
    dst = np.zeros(24, dtype=np.float32)[::2]
    traced = tilelens.trace(Tracer(grid_idx=1), frontend="cutile")(copy_kernel)
    traced.run(src=src, dst=dst, grid=(3,))
    np.testing.assert_array_equal(dst, src)
    records = launches[-1].records
    assert [type(r) for r in records] == [Grid, Grid, Load, Store, Grid]
    np.testing.assert_array_equal(records[2].offsets, np.arange(4, 8) * 8)
    assert launches[-1].grid == (3, 1, 1)


@pytest.mark.parametrize("traced", [False, True])
def test_failure_restores_patches(traced):
    from tilelens.core.simulation.cutile import CuTileInterpretedFunction

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
    from tilelens.core.frontend.cutile import frontend
    from tilelens.core.simulation.cutile import CuTileInterpretedFunction

    original_load = ct.load
    patch_lang = frontend.patch_lang
    calls = []

    def record_patch(fn, client_manager=None):
        calls.append(client_manager)
        return patch_lang(fn, client_manager)

    monkeypatch.setattr(frontend, "patch_lang", record_patch)
    runner = (
        tilelens.trace(frontend="cutile")(copy_kernel)
        if traced
        else CuTileInterpretedFunction(copy_kernel)
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
    from tilelens.core.data import ReduceSum
    from tilelens.core.simulation.cutile import CuTileInterpretedFunction

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
    from tilelens.core.simulation.cutile import CuTileInterpretedFunction

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
        tilelens.trace("profiler", frontend="cutile")(copy_kernel)


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
        from tilelens.visualizer.interface import app, update_global_data

        update_global_data(force=True)
        assert app.test_client().get("/api/data").status_code == 200
    finally:
        tilelens.clear()
