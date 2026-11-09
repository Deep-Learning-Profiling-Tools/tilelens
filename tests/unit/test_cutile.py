"""Tests for cuTile values and operations."""

import operator

import numpy as np
import pytest

ct = pytest.importorskip("cuda.tile")

from tilelens.core.simulation.cutile import (  # noqa: E402
    Array,
    Tile,
    cutile_builder,
)


def test_tile_creation():
    data = np.array([[1, 2], [3, 4]], dtype=np.int32)
    tile = Tile(data)

    assert tile.shape == (2, 2)
    assert tile.dtype == ct.int32
    np.testing.assert_array_equal(tile.data, data)

    # Test data ownership and immutability.
    data[0, 0] = 99
    assert tile.data[0, 0] == 1
    with pytest.raises(ValueError):
        tile.data[0, 0] = 99

    assert Tile(3).dtype == ct.int32
    assert Tile(3.0).dtype == ct.float32


def test_arithmetic():
    x = Tile(np.array([8, 4], dtype=np.int32))
    y = Tile(np.array([2, 2], dtype=np.int32))

    np.testing.assert_array_equal((x + y).data, [10, 6])
    np.testing.assert_array_equal((x - y).data, [6, 2])
    np.testing.assert_array_equal((10 - x).data, [2, 6])
    np.testing.assert_array_equal((x * y).data, [16, 8])
    np.testing.assert_array_equal((x / y).data, [4.0, 2.0])
    np.testing.assert_array_equal((16 / x).data, [2.0, 4.0])
    np.testing.assert_array_equal((x < y).data, [False, False])
    np.testing.assert_array_equal((x > y).data, [True, True])
    np.testing.assert_array_equal((x <= y).data, [False, False])
    np.testing.assert_array_equal((x >= y).data, [True, True])
    np.testing.assert_array_equal((x & y).data, [0, 0])
    np.testing.assert_array_equal((x | y).data, [10, 6])
    assert (x / y).dtype == ct.float32
    assert (x < y).dtype == ct.bool_

    # Test mixed int32/float32 promotion.
    mixed = Tile(np.array([1, 2], np.int32)) + Tile(np.array([0.5, 1.5], np.float32))
    assert mixed.dtype == ct.float32


def test_tile_operations():
    full = cutile_builder.full((2, 2), 3, ct.int32)
    reshaped = cutile_builder.reshape(full, (4,))
    converted = cutile_builder.astype(reshaped, ct.float32)
    total = cutile_builder.sum(full)

    lhs = Tile(np.array([[1, 2], [3, 4]], dtype=np.float32))
    rhs = Tile(np.array([[2, 0], [1, 2]], dtype=np.float32))
    product = cutile_builder.matmul(lhs, rhs)

    np.testing.assert_array_equal(full.data, [[3, 3], [3, 3]])
    np.testing.assert_array_equal(reshaped.data, [3, 3, 3, 3])
    assert converted.dtype == ct.float32
    assert total.data == 12
    np.testing.assert_array_equal(product.data, [[4, 4], [10, 8]])


def test_load_and_store():
    source = Array(np.arange(10, dtype=np.int32))
    first = cutile_builder.load(source, (0,), (4,))
    tail = cutile_builder.load(source, (2,), (4,), padding_mode=ct.PaddingMode.ZERO)

    output = np.full(10, -1, dtype=np.int32)
    cutile_builder.store(Array(output), (0,), first)
    cutile_builder.store(Array(output), (2,), tail)

    np.testing.assert_array_equal(first.data, [0, 1, 2, 3])
    np.testing.assert_array_equal(tail.data, [8, 9, 0, 0])
    np.testing.assert_array_equal(output, [0, 1, 2, 3, -1, -1, -1, -1, 8, 9])


def test_tile_shape_operations():
    tile = cutile_builder.arange(16, dtype=ct.int32).reshape((4, -1))
    np.testing.assert_array_equal(tile.transpose().data, tile.data.T)
    np.testing.assert_array_equal(tile.permute((1, 0)).data, tile.data.T)
    np.testing.assert_array_equal(tile.extract((1, 0), (2, 2)).data, [[8, 9], [12, 13]])
    assert tile[:, None, ...].shape == (4, 1, 4)
    assert cutile_builder.expand_dims(tile, -1).shape == (4, 4, 1)
    assert cutile_builder.broadcast_to(
        cutile_builder.arange(4, dtype=ct.int32), (2, 4)
    ).shape == (2, 4)
    assert tile.astype(ct.float32).dtype == ct.float32
    assert operator.index(cutile_builder.full((1,), 3, ct.int32).item()) == 3
    with pytest.raises(TypeError, match="extract"):
        tile[1:3]
    with pytest.raises(ValueError):
        cutile_builder.full((3,), 0, ct.float32)
    with pytest.raises(ValueError):
        tile.extract((2, 0), (2, 2))


@pytest.mark.parametrize(
    "view",
    [
        lambda a: a.T,
        lambda a: a[:, ::2],
        lambda a: a[::-1, ::-1],
        lambda a: np.broadcast_to(a[:1, :], (4, 8)),
    ],
)
def test_host_strides_and_order(view):
    data = view(np.arange(32, dtype=np.float32).reshape(4, 8))
    array = Array(data)
    assert array.strides == tuple(s // data.itemsize for s in data.strides)
    for order in ("C", "F", (1, 0)):
        expected = data if order == "C" else data.T
        tile = cutile_builder.load(array, (0, 0), (2, 4), order=order)
        np.testing.assert_array_equal(tile.data, expected[:2, :4])


def test_views_alias_storage_and_traversal():
    storage = np.arange(32, dtype=np.float32).reshape(4, 8)
    array = Array(storage)
    view = array.slice(0, 1, 4).slice(-1, Tile(2), Tile(7))
    assert view.shape == (3, 5)
    assert np.shares_memory(view.data, storage)
    tiled = view.tiled_view(
        (2, 4), padding_mode=ct.PaddingMode.ZERO, traversal_steps=(1, 2)
    )
    assert (tiled.num_tiles(0), tiled.num_tiles(1)) == (3, 3)
    np.testing.assert_array_equal(
        tiled.load((1, 1)).data, [[20, 21, 22, 0], [28, 29, 30, 0]]
    )
    tiled.store((2, 2), 99.0)
    assert storage[3, 6] == 99
    assert storage[3, 7] == 31
    with pytest.raises(ValueError):
        array.slice(0, -1, 2)
    with pytest.raises(ValueError):
        array.tiled_view((2,))
    with pytest.raises(ValueError):
        array.tiled_view((2, 4), traversal_steps=(0, 4))


def test_permuted_store_and_scalar_access():
    data = np.zeros((4, 8), np.float32)
    cutile_builder.store(
        Array(data), (1, 0), cutile_builder.full((4, 2), 7, ct.float32), order="F"
    )
    np.testing.assert_array_equal(data[:2, 4:], 7)
    assert not data[2:, :].any()
    cutile_builder.store(Array(data), (3, 7), Tile(11))
    assert cutile_builder.load(Array(data), (3, 7), ()).data == 11
    with pytest.raises(ValueError):
        cutile_builder.load(Array(data), (0,), (4,))


@pytest.mark.parametrize(
    "mode,expected",
    [
        (ct.PaddingMode.ZERO, 0),
        (ct.PaddingMode.NEG_ZERO, -0.0),
        (ct.PaddingMode.NAN, np.nan),
        (ct.PaddingMode.POS_INF, np.inf),
        (ct.PaddingMode.NEG_INF, -np.inf),
    ],
)
def test_padding_modes(mode, expected):
    value = cutile_builder.load(
        Array(np.arange(3, dtype=np.float32)), 0, 4, padding_mode=mode
    )
    np.testing.assert_equal(value.data[-1], expected)
    if mode == ct.PaddingMode.NEG_ZERO:
        assert np.signbit(value.data[-1])


def test_indexed_memory_bounds_masks_and_broadcast():
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    rows = Tile(np.array([[-1], [2]], np.int32))
    cols = Tile(np.array([[0, 4, 1, 3]], np.int32))
    gathered = cutile_builder.gather(Array(data), (rows, cols), padding_value=-9)
    np.testing.assert_array_equal(gathered.data, [[-9, -9, -9, -9], [8, -9, 9, 11]])
    out = np.zeros_like(data)
    cutile_builder.scatter(
        Array(out), (rows, cols), 7, mask=Tile(np.array([[True, True, False, True]]))
    )
    np.testing.assert_array_equal(out, [[0, 0, 0, 0], [0, 0, 0, 0], [7, 0, 0, 7]])
    with pytest.raises(IndexError, match="check_bounds=False"):
        cutile_builder.gather(Array(data), (rows, cols), check_bounds=False)
    # The mask excludes invalid indices when bounds checking is disabled.
    np.testing.assert_array_equal(
        cutile_builder.gather(
            Array(data), (rows, cols), mask=False, check_bounds=False
        ).data,
        np.zeros((2, 4)),
    )
    with pytest.raises(TypeError, match="integer"):
        cutile_builder.gather(Array(data), (Tile(1.0), 0))


def test_atomic_add_repeated_indices_and_old_values():
    data = np.array([10, 20], np.int32)
    indices = Tile(np.array([0, 0, 1, -1], np.int32))
    old = cutile_builder.atomic_add(
        Array(data), indices, Tile(np.array([1, 2, 3, 99], np.int32))
    )
    np.testing.assert_array_equal(data, [13, 23])
    np.testing.assert_array_equal(old.data[:3], [10, 11, 20])


@pytest.mark.parametrize(
    "dtype",
    [
        ct.int8,
        ct.uint8,
        ct.int16,
        ct.int32,
        ct.int64,
        ct.float16,
        ct.bfloat16,
        ct.float32,
    ],
)
def test_dtype_storage_roundtrip_and_scalar_arithmetic(dtype):
    tile = cutile_builder.full((4,), 2, dtype)
    assert tile.dtype == dtype
    assert (tile + 1).dtype == dtype
    np.testing.assert_array_equal((tile * 2).data.astype(np.float32), [4] * 4)
    assert cutile_builder.astype(tile, ct.float32).dtype == ct.float32


def test_mma_accumulation_widens_int8_and_preserves_accumulator():
    x = Tile(np.full((2, 4), 100, np.int8))
    y = Tile(np.full((4, 2), 2, np.int8))
    result = cutile_builder.mma(x, y, cutile_builder.full((2, 2), 10, ct.int32))
    assert result.dtype == ct.int32
    np.testing.assert_array_equal(result.data, np.full((2, 2), 810))
    float_result = cutile_builder.mma(
        cutile_builder.astype(x, ct.float16),
        cutile_builder.astype(y, ct.float16),
        cutile_builder.zeros((2, 2), ct.float32),
    )
    assert float_result.dtype == ct.float32
    np.testing.assert_array_equal(float_result.data, np.full((2, 2), 800))


def test_reductions_scan_and_concat():
    x = Tile(np.array([[1, 3, 3, -1], [4, 2, 1, 0]], np.float32))
    np.testing.assert_array_equal(
        cutile_builder.max(x, 1, keepdims=True).data, [[3], [4]]
    )
    np.testing.assert_array_equal(cutile_builder.argmax(x, 1).data, [1, 0])
    assert cutile_builder.argmax(x).dtype == ct.int32
    np.testing.assert_array_equal(
        cutile_builder.cumsum(x, 1, reverse=True).data, [[6, 5, 2, -1], [7, 3, 1, 0]]
    )
    assert cutile_builder.cat((x, x), 0).shape == (4, 4)
    with pytest.raises(ValueError, match="same shape"):
        cutile_builder.cat((x, x.reshape((8,))), 0)


def test_integer_operators_and_scalar_control_flow():
    x = Tile(np.array([1, 2, 3, 4], np.int64))
    np.testing.assert_array_equal((x // 2).data, [0, 1, 1, 2])
    np.testing.assert_array_equal((x % 2).data, [1, 0, 1, 0])
    np.testing.assert_array_equal((x ^ 1).data, [0, 3, 2, 5])
    np.testing.assert_array_equal((x << 2).data, [4, 8, 12, 16])
    np.testing.assert_array_equal((x >> 1).data, [0, 1, 1, 2])
    np.testing.assert_array_equal((x != 2).data, [True, False, True, True])
    np.testing.assert_array_equal((-x).data, [-1, -2, -3, -4])
    assert bool(Tile(1))
    with pytest.raises(TypeError, match="scalar"):
        bool(x)


def test_transcendentals_selection_and_fp16_rounding():
    x = Tile(np.array([1, 2, 4, 8], np.float32))
    np.testing.assert_allclose(cutile_builder.rsqrt(x).data, 1 / np.sqrt(x.data))
    np.testing.assert_allclose(
        cutile_builder.exp2(cutile_builder.log(x)).data,
        np.exp2(np.log(x.data)),
        rtol=1e-6,
    )
    np.testing.assert_array_equal(
        cutile_builder.where(x > 2, x, -1).data, [-1, -1, 4, 8]
    )
    assert cutile_builder.exp(cutile_builder.astype(x, ct.float16)).dtype == ct.float16


def test_mixed_float_tiles_do_not_narrow_float64():
    precise = Tile(np.array([1.0 + 2**-40], np.float64))
    zero = Tile(np.zeros(1, np.float32))
    result = precise + zero
    assert result.dtype == ct.float64
    np.testing.assert_array_equal(result.data, precise.data)
    np.testing.assert_array_equal((precise > Tile(np.ones(1, np.float32))).data, [True])


@pytest.mark.parametrize("fill", [1.0, 0.1])
def test_bfloat16_reductions(fill):
    tile = cutile_builder.full((1024,), fill, ct.bfloat16)
    expected = tile.data.astype(np.float32)
    total = cutile_builder.sum(tile)
    scan = cutile_builder.cumsum(tile)
    reverse = cutile_builder.cumsum(tile, reverse=True)
    assert total.dtype == scan.dtype == reverse.dtype == ct.bfloat16
    np.testing.assert_array_equal(
        total.data, np.asarray(expected.sum(), dtype=tile.data.dtype)
    )
    np.testing.assert_array_equal(scan.data, expected.cumsum().astype(tile.data.dtype))
    np.testing.assert_array_equal(
        reverse.data, expected[::-1].cumsum()[::-1].astype(tile.data.dtype)
    )


def test_boolean_results():
    tile = Tile(np.array([True, False, True, True]))
    assert cutile_builder.sum(tile).dtype == ct.int32
    assert cutile_builder.sum(tile).data == 3
    scan = cutile_builder.cumsum(tile)
    assert scan.dtype == ct.int32
    np.testing.assert_array_equal(scan.data, [1, 1, 2, 3])
    assert cutile_builder.max(tile).dtype == ct.int32
    assert (tile + 2).dtype == (2 + tile).dtype == ct.int32
    np.testing.assert_array_equal((tile + 2).data, [3, 2, 3, 3])
    np.testing.assert_array_equal(
        (tile + Tile(np.full(4, 2, np.int32))).data, [3, 2, 3, 3]
    )
    with pytest.raises(TypeError, match="Boolean arithmetic"):
        tile + tile


def test_scalar_float64_results():
    tile = Tile(np.array([1e300, 1, 1, 1], np.float64))
    for result in (
        cutile_builder.sum(tile),
        cutile_builder.max(tile),
        Tile(np.float64(1e300)),
    ):
        assert result.dtype == ct.float64
        assert result.data == 1e300
    scalar = Tile(np.array(1e-300, np.float64))
    assert (scalar + scalar).dtype == ct.float64
    assert (scalar + scalar).data == 2e-300


def test_arange_requires_dtype():
    with pytest.raises(TypeError, match="dtype"):
        cutile_builder.arange(4)
