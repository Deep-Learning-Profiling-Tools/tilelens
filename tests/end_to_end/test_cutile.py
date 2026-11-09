# ruff: noqa: E402

import importlib.util
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

ct = pytest.importorskip("cuda.tile")

import tilelens
from tilelens.core.simulation.cutile import CuTileInterpretedFunction, DTYPES
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


def test_example_kernel(tmp_path):
    example = load_example()
    lhs = np.arange(15, dtype=np.float32).reshape(3, 5)
    rhs = np.arange(45, dtype=np.float32).reshape(5, 9)
    dst = np.zeros((3, 9), np.float32)
    tilelens.trace("tracer", frontend="cutile")(example.matmul_kernel)[(2, 2)](
        lhs, rhs, dst
    )
    np.testing.assert_array_equal(dst, lhs @ rhs)
    check_visualizer()
    tilelens.save(tmp_path / "matmul.tvz")


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
