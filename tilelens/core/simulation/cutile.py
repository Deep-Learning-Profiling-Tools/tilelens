"""NumPy-backed values and grid execution for the cuTile interpreter."""

from __future__ import annotations

import builtins
import inspect
import itertools
import operator
import types
from typing import Any

import numpy as np

from tilelens.utils.dtypes import STORAGE_DTYPES

try:
    import cuda.tile as ct
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Install cuda-tile to use the cuTile interpreter."
    ) from exc

from ..frontend.base import get_frontend


DTYPES = {
    ct.bool_: np.dtype(np.bool_),
    ct.int8: STORAGE_DTYPES["int8"],
    ct.uint8: STORAGE_DTYPES["uint8"],
    ct.int16: STORAGE_DTYPES["int16"],
    ct.int32: STORAGE_DTYPES["int32"],
    ct.int64: STORAGE_DTYPES["int64"],
    ct.float16: STORAGE_DTYPES["float16"],
    ct.bfloat16: STORAGE_DTYPES["bfloat16"],
    ct.float32: STORAGE_DTYPES["float32"],
    ct.float64: STORAGE_DTYPES["float64"],
    ct.float8_e4m3fn: STORAGE_DTYPES["float8_e4m3fn"],
    ct.float8_e5m2: STORAGE_DTYPES["float8_e5m2"],
}
NUMPY_DTYPES = {dtype: token for token, dtype in DTYPES.items()}


def _is_float(dtype):
    return dtype.kind == "f" or dtype in (
        DTYPES[ct.bfloat16],
        DTYPES[ct.float8_e4m3fn],
        DTYPES[ct.float8_e5m2],
    )


def _binary_values(x, y, force_float=False):
    """Choose operand dtypes without NumPy's integer/float widening.

    This implements only part of cuTile's dtype promotion rules.
    """
    x_value, y_value = _tile_value(x), _tile_value(y)
    if isinstance(x, Tile) and isinstance(y, (bool, int, float)):
        dtype = x_value.dtype
        if isinstance(y, float) and not _is_float(dtype):
            dtype = np.dtype(np.float32)
    elif isinstance(y, Tile) and isinstance(x, (bool, int, float)):
        dtype = y_value.dtype
        if isinstance(x, float) and not _is_float(dtype):
            dtype = np.dtype(np.float32)
    elif x_value.dtype == y_value.dtype:
        dtype = x_value.dtype
    elif _is_float(x_value.dtype) or _is_float(y_value.dtype):
        floats = [v.dtype for v in (x_value, y_value) if _is_float(v.dtype)]
        dtype = max(floats, key=lambda d: d.itemsize)
        if len(floats) == 2 and floats[0] != floats[1]:
            dtype = np.dtype(
                np.float64 if np.dtype(np.float64) in floats else np.float32
            )
    else:
        dtype = np.result_type(x_value.dtype, y_value.dtype)
    if force_float and not _is_float(dtype):
        dtype = np.dtype(np.float32)
    # Use float32 arithmetic for NumPy void storage dtypes.
    compute_dtype = dtype if dtype.kind != "V" else np.dtype(np.float32)
    return x_value.astype(compute_dtype), y_value.astype(compute_dtype), dtype


def _normalize_shape(shape: Any) -> tuple:
    """Normalize cuTile's scalar-or-tuple shape and index arguments."""
    return (shape,) if isinstance(shape, (int, np.integer, Tile)) else tuple(shape)


def _numpy_dtype(dtype):
    # TF32 is a compute format. CPU interpretation uses float32 precision.
    if dtype == ct.tfloat32:
        return np.dtype(np.float32)
    return DTYPES[dtype]


def _validate_tile_shape(shape: Any) -> tuple[int, ...]:
    shape = tuple(operator.index(d) for d in _normalize_shape(shape))
    if any(d <= 0 or d & (d - 1) for d in shape):
        raise ValueError("Tile dimensions must be positive powers of two")
    return shape


def _normalize_axis(axis: int, ndim: int) -> int:
    axis = operator.index(axis)
    if not -ndim <= axis < ndim:
        raise ValueError(f"Axis {axis} is invalid for rank {ndim}")
    return axis % ndim


def _normalize_order(order: Any, ndim: int) -> tuple[int, ...]:
    if order == "C":
        axes = tuple(range(ndim))
    elif order == "F":
        axes = tuple(reversed(range(ndim)))
    else:
        axes = tuple(order)
    if sorted(axes) != list(range(ndim)):
        raise ValueError("order must be a permutation of the array axes")
    return axes


def _tile_access(array, index, shape, order="C", traversal_steps=None):
    """Shared tile geometry for numerical execution and tracing."""
    shape = _validate_tile_shape(shape)
    axes = _normalize_order(order, array.ndim)
    access_shape = shape or (1,) * array.ndim
    index = tuple(operator.index(i) for i in _normalize_shape(index))
    steps = access_shape if traversal_steps is None else tuple(traversal_steps)
    if (
        len(index) != array.ndim
        or len(access_shape) != array.ndim
        or len(steps) != array.ndim
    ):
        raise ValueError("Tile shape, index, and traversal steps must match array rank")
    if any(operator.index(s) <= 0 for s in steps):
        raise ValueError("Traversal steps must be positive integers")
    starts = tuple(i * s for i, s in zip(index, steps))
    return shape, axes, access_shape, starts


def _tile_value(value):
    """Return the NumPy value stored in a tile or Python scalar."""
    if isinstance(value, Tile):
        return value.data
    if isinstance(value, bool):
        return np.asarray(value, dtype=np.bool_)
    if isinstance(value, int):
        return np.asarray(value, dtype=np.int32)
    if isinstance(value, float):
        return np.asarray(value, dtype=np.float32)
    return np.asarray(value)


def _indexed_access(array, indices, mask=None, check_bounds=True):
    """Broadcast element indices and validity for gather/scatter and tracing."""
    indices = indices if isinstance(indices, tuple) else (indices,)
    if len(indices) != array.ndim:
        raise ValueError("Index tuple must match array rank")
    keys = np.broadcast_arrays(*(_tile_value(i) for i in indices))
    if any(key.dtype.kind not in "iu" for key in keys):
        raise TypeError("Array indices must be integer tiles or scalars")
    valid = np.ones(keys[0].shape, dtype=bool)
    if mask is not None:
        valid &= np.broadcast_to(_tile_value(mask), valid.shape)
    bounds = np.ones_like(valid)
    for key, size in zip(keys, array.shape):
        bounds &= (key >= 0) & (key < size)
    if check_bounds:
        valid &= bounds
    elif np.any(valid & ~bounds):
        raise IndexError("Out-of-bounds access with check_bounds=False")
    return tuple(keys), valid


class Tile:
    """Immutable local tile value backed by an owned NumPy array."""

    def __init__(self, value, dtype=None):
        if isinstance(value, Tile):
            value = value.data
        if dtype is not None:
            value = np.asarray(value, dtype=_numpy_dtype(dtype))
        elif isinstance(value, int):
            value = np.asarray(value, dtype=np.int32)
        elif isinstance(value, float):
            value = np.asarray(value, dtype=np.float32)

        # Keep tiles immutable and independent of the source array.
        self.data = np.array(value, copy=True)
        self.data.setflags(write=False)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return NUMPY_DTYPES[self.data.dtype]

    def __index__(self):
        if self.shape != () or self.data.dtype.kind not in "iu":
            raise TypeError("Only scalar integer tiles can be used as indices")
        return int(self.data)

    def item(self):
        if self.data.size != 1:
            raise ValueError("item requires a single-element tile")
        return Tile(self.data.reshape(()))

    def __getitem__(self, index):
        keys = index if isinstance(index, tuple) else (index,)
        if any(k is not None and k is not Ellipsis and k != slice(None) for k in keys):
            raise TypeError(
                "Tile indexing supports only None, ellipsis, and full slices; use extract"
            )
        return Tile(self.data[keys])

    def reshape(self, shape):
        return cutile_builder.reshape(self, shape)

    def permute(self, axes):
        return cutile_builder.permute(self, axes)

    def transpose(self, axis0=None, axis1=None):
        return cutile_builder.transpose(self, axis0, axis1)

    def astype(self, dtype):
        return cutile_builder.astype(self, dtype)

    def extract(self, index, shape):
        return cutile_builder.extract(self, index, shape)

    # Python operators
    def __add__(self, other):
        return cutile_builder.add(self, other)

    def __radd__(self, other):
        return cutile_builder.add(other, self)

    def __sub__(self, other):
        return cutile_builder.sub(self, other)

    def __rsub__(self, other):
        return cutile_builder.sub(other, self)

    def __mul__(self, other):
        return cutile_builder.mul(self, other)

    def __rmul__(self, other):
        return cutile_builder.mul(other, self)

    def __truediv__(self, other):
        return cutile_builder.truediv(self, other)

    def __rtruediv__(self, other):
        return cutile_builder.truediv(other, self)

    def __lt__(self, other):
        return cutile_builder.less(self, other)

    def __gt__(self, other):
        return cutile_builder.greater(self, other)

    def __le__(self, other):
        return cutile_builder.less_equal(self, other)

    def __ge__(self, other):
        return cutile_builder.greater_equal(self, other)

    def __and__(self, other):
        return cutile_builder.bitwise_and(self, other)

    def __or__(self, other):
        return cutile_builder.bitwise_or(self, other)

    def __matmul__(self, other):
        return cutile_builder.matmul(self, other)

    def __bool__(self):
        if self.shape != ():
            raise TypeError("Only scalar tiles can control Python branches")
        return bool(self.data)

    def __neg__(self):
        return Tile(np.negative(self.data))

    def __eq__(self, other):
        return cutile_builder.equal(self, other)

    def __ne__(self, other):
        return cutile_builder.not_equal(self, other)

    def __floordiv__(self, other):
        return cutile_builder.floordiv(self, other)

    def __rfloordiv__(self, other):
        return cutile_builder.floordiv(other, self)

    def __mod__(self, other):
        return cutile_builder.mod(self, other)

    def __rmod__(self, other):
        return cutile_builder.mod(other, self)

    def __xor__(self, other):
        return cutile_builder.bitwise_xor(self, other)

    def __lshift__(self, other):
        return cutile_builder.bitwise_lshift(self, other)

    def __rshift__(self, other):
        return cutile_builder.bitwise_rshift(self, other)

    def __rand__(self, other):
        return cutile_builder.bitwise_and(other, self)

    def __ror__(self, other):
        return cutile_builder.bitwise_or(other, self)

    def __rxor__(self, other):
        return cutile_builder.bitwise_xor(other, self)


class Array:
    """Mutable global storage or a slice sharing its parent's NumPy memory."""

    def __init__(self, value, parent=None, origin=None):
        self.data = value
        self._parent = parent
        self._origin = (0,) * value.ndim if origin is None else origin

    @property
    def strides(self):
        return self.stride()

    def is_contiguous(self):
        return self.data.flags.c_contiguous

    def slice(self, axis, start, stop):
        axis = _normalize_axis(axis, self.ndim)
        start, stop = operator.index(start), operator.index(stop)
        if not 0 <= start < self.shape[axis] or not start <= stop <= self.shape[axis]:
            raise ValueError(
                "Array.slice bounds must satisfy 0 <= start < N and start <= stop <= N"
            )
        keys = [slice(None)] * self.ndim
        keys[axis] = slice(start, stop)
        origin = list(self._origin)
        origin[axis] += start
        return Array(self.data[tuple(keys)], parent=self, origin=tuple(origin))

    def tiled_view(
        self,
        tile_shape,
        *,
        padding_mode=ct.PaddingMode.UNDETERMINED,
        traversal_steps=None,
    ):
        return TiledView(self, tile_shape, padding_mode, traversal_steps)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return NUMPY_DTYPES[self.data.dtype]

    def data_ptr(self):
        return self.data.ctypes.data

    def stride(self):
        return tuple(s // self.element_size() for s in self.data.strides)

    def element_size(self):
        return self.data.itemsize

    def cpu(self):
        return self

    def detach(self):
        return self

    def numpy(self):
        return self.data

    def get_offsets(self):
        offsets = np.int64(0)
        for size, stride in zip(self.shape, self.data.strides):
            offsets = np.expand_dims(offsets, -1) + np.arange(size) * stride
        return Array(offsets)


class TiledView:
    """Partition an Array into tiles without copying its underlying storage."""

    def __init__(self, array, tile_shape, padding_mode, traversal_steps):
        self.array = array
        self.tile_shape = _validate_tile_shape(tile_shape)
        self.padding_mode = padding_mode
        self.traversal_steps = (
            (self.tile_shape or (1,) * array.ndim)
            if traversal_steps is None
            else _normalize_shape(traversal_steps)
        )
        _tile_access(
            array, (0,) * array.ndim, tile_shape, traversal_steps=self.traversal_steps
        )

    @property
    def dtype(self):
        return self.array.dtype

    def num_tiles(self, axis):
        axis = _normalize_axis(axis, self.array.ndim)
        return (
            self.array.shape[axis] + self.traversal_steps[axis] - 1
        ) // self.traversal_steps[axis]

    def load(self, index, **kwargs):
        return cutile_builder.load(
            self.array,
            index,
            self.tile_shape,
            padding_mode=self.padding_mode,
            traversal_steps=self.traversal_steps,
            **kwargs,
        )

    def store(self, index, tile, **kwargs):
        tile = cutile_builder.broadcast_to(tile, self.tile_shape)
        return cutile_builder.store(
            self.array, index, tile, traversal_steps=self.traversal_steps, **kwargs
        )


class Builder:
    """Implement cuTile operations used by the interpreter and frontend hooks."""

    def __init__(self):
        self.grid_dims = (1,)
        self.grid_idx = (0,)

    def set_grid_dim(self, *grid_dims):
        self.grid_dims = grid_dims

    def set_grid_idx(self, *grid_idx):
        self.grid_idx = grid_idx

    def bid(self, axis):
        return self.grid_idx[axis]

    def num_tiles(self, array, /, axis, shape, order="C"):
        _, axes, shape, _ = _tile_access(array, (0,) * array.ndim, shape, order)
        axis = _normalize_axis(axis, array.ndim)
        return (array.shape[axes[axis]] + shape[axis] - 1) // shape[axis]

    @staticmethod
    def _region(array_shape, starts, shape):
        """Create array and tile slices for one tile-space index."""
        array_slices, tile_slices = [], []
        for extent, start, size in zip(array_shape, starts, shape):
            array_start = max(0, min(extent, start))
            array_stop = max(0, min(extent, start + size))
            length = max(0, array_stop - array_start)
            tile_start = max(0, min(size, -start))
            array_slices.append(slice(array_start, array_start + length))
            tile_slices.append(slice(tile_start, tile_start + length))
        return tuple(array_slices), tuple(tile_slices)

    def load(
        self,
        array,
        /,
        index,
        shape,
        *,
        order="C",
        padding_mode=ct.PaddingMode.UNDETERMINED,
        traversal_steps=None,
        **kwargs,
    ):
        """Load a tile from a global array using a tile-space index."""
        if shape is None:
            keys, valid = _indexed_access(
                array, index, kwargs.get("mask"), kwargs.get("check_bounds", True)
            )
            padding = _tile_value(kwargs.get("padding_value", 0))
            value = np.broadcast_to(padding, valid.shape)
            value = value.astype(array.data.dtype).copy()
            value[valid] = array.data[tuple(key[valid] for key in keys)]
            return Tile(value)
        shape, axes, access_shape, starts = _tile_access(
            array, index, shape, order, traversal_steps
        )
        data = array.data.transpose(axes)
        array_slices, tile_slices = self._region(data.shape, starts, access_shape)

        # UNDETERMINED padding leaves out-of-bounds elements uninitialized.
        if padding_mode == ct.PaddingMode.UNDETERMINED:
            value = np.empty(access_shape, dtype=data.dtype)
        else:
            padding = {
                ct.PaddingMode.ZERO: 0,
                ct.PaddingMode.NEG_ZERO: -0.0,
                ct.PaddingMode.NAN: np.nan,
                ct.PaddingMode.POS_INF: np.inf,
                ct.PaddingMode.NEG_INF: -np.inf,
            }
            if padding_mode not in padding:
                raise ValueError("Unsupported padding mode")
            if padding_mode != ct.PaddingMode.ZERO and not _is_float(data.dtype):
                raise TypeError("Nonzero padding modes require a floating-point array")
            value = np.full(access_shape, padding[padding_mode], dtype=data.dtype)
        value[tile_slices] = data[array_slices]
        return Tile(value.reshape(shape))

    def store(
        self, array, /, index, tile, *, order="C", traversal_steps=None, **kwargs
    ):
        """Store the in-bounds part of a tile into a global array."""
        if kwargs.get("_indexed", False):
            keys, valid = _indexed_access(
                array, index, kwargs.get("mask"), kwargs.get("check_bounds", True)
            )
            value = np.broadcast_to(_tile_value(tile), valid.shape)
            array.data[tuple(key[valid] for key in keys)] = value[valid]
            return
        value = _tile_value(tile)
        _, axes, shape, starts = _tile_access(
            array, index, value.shape, order, traversal_steps
        )
        data = array.data.transpose(axes)
        array_slices, tile_slices = self._region(data.shape, starts, shape)
        data[array_slices] = value.reshape(shape)[tile_slices]

    def full(self, shape, fill_value, dtype):
        return Tile(
            np.full(
                _validate_tile_shape(shape),
                _tile_value(fill_value),
                dtype=_numpy_dtype(dtype),
            )
        )

    def zeros(self, shape, dtype):
        return self.full(shape, 0, dtype)

    def ones(self, shape, dtype):
        return self.full(shape, 1, dtype)

    def arange(self, size, *, dtype=ct.int32, start=0, step=1):
        (size,) = _validate_tile_shape(size)
        return Tile((start + np.arange(size) * step).astype(_numpy_dtype(dtype)))

    @staticmethod
    def _binary(x, y, operation, force_float=False):
        x, y, dtype = _binary_values(x, y, force_float)
        return Tile(operation(x, y).astype(dtype))

    @staticmethod
    def _compare(x, y, operation):
        x, y, _ = _binary_values(x, y)
        return Tile(operation(x, y))

    def add(self, x, y, /, **kwargs):
        return self._binary(x, y, np.add)

    def sub(self, x, y, /, **kwargs):
        return self._binary(x, y, np.subtract)

    def mul(self, x, y, /, **kwargs):
        return self._binary(x, y, np.multiply)

    def truediv(self, x, y, /, **kwargs):
        return self._binary(x, y, np.true_divide, force_float=True)

    def less(self, x, y, /):
        return self._compare(x, y, np.less)

    def greater(self, x, y, /):
        return self._compare(x, y, np.greater)

    def less_equal(self, x, y, /):
        return self._compare(x, y, np.less_equal)

    def greater_equal(self, x, y, /):
        return self._compare(x, y, np.greater_equal)

    def bitwise_and(self, x, y, /):
        return Tile(np.bitwise_and(_tile_value(x), _tile_value(y)))

    def bitwise_or(self, x, y, /):
        return Tile(np.bitwise_or(_tile_value(x), _tile_value(y)))

    def reshape(self, x, /, shape):
        value = np.reshape(_tile_value(x), _normalize_shape(shape))
        _validate_tile_shape(value.shape)
        return Tile(value)

    def broadcast_to(self, x, /, shape):
        return Tile(np.broadcast_to(_tile_value(x), _validate_tile_shape(shape)))

    def permute(self, x, /, axes):
        value = _tile_value(x)
        return Tile(value.transpose(_normalize_order(axes, value.ndim)))

    def transpose(self, x, /, axis0=None, axis1=None):
        value = _tile_value(x)
        if axis0 is None and axis1 is None and value.ndim == 2:
            axis0, axis1 = 0, 1
        if axis0 is None or axis1 is None:
            raise ValueError("transpose requires two axes except for 2D tiles")
        return Tile(
            np.swapaxes(
                value,
                _normalize_axis(axis0, value.ndim),
                _normalize_axis(axis1, value.ndim),
            )
        )

    def expand_dims(self, x, /, axis):
        return Tile(np.expand_dims(_tile_value(x), operator.index(axis)))

    def extract(self, x, /, index, shape):
        value = _tile_value(x)
        shape = _validate_tile_shape(shape)
        index = tuple(operator.index(i) for i in _normalize_shape(index))
        if len(shape) != value.ndim or len(index) != value.ndim:
            raise ValueError("extract shape and index must match tile rank")
        if any(
            n % s or not 0 <= i < n // s for n, s, i in zip(value.shape, shape, index)
        ):
            raise ValueError(
                "extract requires divisible shapes and in-bounds tile indices"
            )
        return Tile(
            value[tuple(slice(i * s, (i + 1) * s) for i, s in zip(index, shape))]
        )

    def astype(self, x, dtype, /):
        return Tile(_tile_value(x).astype(_numpy_dtype(dtype)))

    def sum(self, x, /, axis=None, *, keepdims=False, **kwargs):
        value = _tile_value(x)
        return Tile(np.sum(value, axis=axis, keepdims=keepdims, dtype=value.dtype))

    def matmul(self, x, y, /):
        return Tile(np.matmul(_tile_value(x), _tile_value(y)))

    def floordiv(self, x, y, /):
        return self._binary(x, y, np.floor_divide)

    def mod(self, x, y, /):
        return self._binary(x, y, np.remainder)

    def equal(self, x, y, /):
        return self._compare(x, y, np.equal)

    def not_equal(self, x, y, /):
        return self._compare(x, y, np.not_equal)

    def bitwise_xor(self, x, y, /):
        return self._binary(x, y, np.bitwise_xor)

    def bitwise_lshift(self, x, y, /):
        return self._binary(x, y, np.left_shift)

    def bitwise_rshift(self, x, y, /):
        return self._binary(x, y, np.right_shift)

    def where(self, condition, x, y, /):
        x, y, dtype = _binary_values(x, y)
        return Tile(np.where(_tile_value(condition), x, y).astype(dtype))

    def maximum(self, x, y, /, **kwargs):
        return self._binary(x, y, np.maximum)

    def minimum(self, x, y, /, **kwargs):
        return self._binary(x, y, np.minimum)

    @staticmethod
    def _unary(x, operation):
        value = _tile_value(x)
        if not _is_float(value.dtype):
            raise TypeError("Transcendental operations require floating-point tiles")
        return Tile(
            operation(
                value.astype(np.float64 if value.dtype == np.float64 else np.float32)
            ).astype(value.dtype)
        )

    def exp(self, x, /, **kwargs):
        return self._unary(x, np.exp)

    def exp2(self, x, /, **kwargs):
        return self._unary(x, np.exp2)

    def log(self, x, /, **kwargs):
        return self._unary(x, np.log)

    def rsqrt(self, x, /, **kwargs):
        return self._unary(x, lambda value: 1 / np.sqrt(value))

    def max(self, x, /, axis=None, *, keepdims=False, **kwargs):
        return Tile(np.max(_tile_value(x), axis=axis, keepdims=keepdims))

    def argmax(self, x, /, axis=None, *, keepdims=False):
        return Tile(
            np.asarray(
                np.argmax(_tile_value(x), axis=axis, keepdims=keepdims), dtype=np.int32
            )
        )

    def cumsum(self, x, /, axis=0, *, reverse=False, **kwargs):
        value = _tile_value(x)
        if reverse:
            value = np.flip(value, axis)
        result = np.cumsum(value, axis=axis, dtype=value.dtype)
        return Tile(np.flip(result, axis) if reverse else result)

    def cat(self, tiles, /, axis):
        if len(tiles) != 2 or tiles[0].shape != tiles[1].shape:
            raise ValueError("cat requires two tiles of the same shape")
        return Tile(np.concatenate([_tile_value(tile) for tile in tiles], axis=axis))

    def mma(self, x, y, /, acc, **kwargs):
        accumulator = _tile_value(acc)
        # Accumulate narrow integer/float inputs in the accumulator's format.
        lhs = _tile_value(x).astype(accumulator.dtype)
        rhs = _tile_value(y).astype(accumulator.dtype)
        result = np.matmul(lhs, rhs) + accumulator
        return Tile(result.astype(accumulator.dtype))

    def gather(
        self,
        array,
        indices,
        /,
        *,
        mask=None,
        padding_value=0,
        check_bounds=True,
        **kwargs,
    ):
        # Use load so gathers trigger the same tracing callbacks as tiled loads.
        return self.load(
            array,
            indices,
            None,
            mask=mask,
            padding_value=padding_value,
            check_bounds=check_bounds,
            **kwargs,
        )

    def scatter(
        self, array, indices, value, /, *, mask=None, check_bounds=True, **kwargs
    ):
        return self.store(
            array,
            indices,
            value,
            _indexed=True,
            mask=mask,
            check_bounds=check_bounds,
            **kwargs,
        )

    def atomic_add(self, array, indices, update, /, *, check_bounds=True, **kwargs):
        """Sequential element updates, returning old values for repeated indices."""
        keys, valid = _indexed_access(array, indices, check_bounds=check_bounds)
        values = np.broadcast_to(_tile_value(update), valid.shape)
        positions = tuple(key[valid] for key in keys)
        offsets = np.ravel_multi_index(positions, array.shape)
        if np.unique(offsets).size == offsets.size:
            previous = self.gather(array, indices, check_bounds=check_bounds)
            self.scatter(
                array, indices, previous.data + values, check_bounds=check_bounds
            )
            return previous
        old = np.zeros(valid.shape, dtype=array.data.dtype)
        for position in np.ndindex(valid.shape):
            if valid[position]:
                key = tuple(int(index[position]) for index in keys)
                previous = self.gather(array, key, check_bounds=False)
                old[position] = previous.data
                self.scatter(
                    array, key, previous.data + values[position], check_bounds=False
                )
        return Tile(old)


cutile_builder = Builder()


def cutile_patch_lang(scope=None):
    """Bind cuda.tile operations to the current interpreter builder methods."""

    def _set_attr(obj, name, value):
        if scope is None:
            setattr(obj, name, value)
        else:
            scope.set_attr(obj, name, value)

    for name in (
        "bid",
        "load",
        "store",
        "full",
        "add",
        "sub",
        "mul",
        "truediv",
        "less",
        "greater",
        "less_equal",
        "greater_equal",
        "bitwise_and",
        "bitwise_or",
        "reshape",
        "astype",
        "sum",
        "matmul",
        "zeros",
        "ones",
        "arange",
        "num_tiles",
        "broadcast_to",
        "permute",
        "transpose",
        "expand_dims",
        "extract",
        "floordiv",
        "mod",
        "equal",
        "not_equal",
        "bitwise_xor",
        "bitwise_lshift",
        "bitwise_rshift",
        "where",
        "maximum",
        "minimum",
        "exp",
        "exp2",
        "log",
        "rsqrt",
        "max",
        "argmax",
        "cumsum",
        "cat",
        "mma",
        "gather",
        "scatter",
        "atomic_add",
    ):
        _set_attr(ct, name, getattr(cutile_builder, name))
    _set_attr(ct, "static_iter", iter)


def cutile_unpatch_lang(scope=None):
    """Restore the language functions saved by a patch scope."""
    if scope is not None:
        scope.restore()


class CuTileInterpretedFunction:
    """Execute a Python kernel over a grid with optional tracing callbacks."""

    def __init__(self, fn: Any) -> None:
        # @ct.kernel stores the original Python function in _pyfunc.
        self.fn = fn._pyfunc if isinstance(fn, ct.kernel) else fn

    def _execution_function(self):
        """Handle elementwise min/max in the kernel's globals."""
        namespace = self.fn.__globals__.copy()

        def tile_minmax(name, operation):
            original = namespace.get(name, getattr(builtins, name))
            # A kernel may intentionally shadow a builtin with a helper.
            if original is not getattr(builtins, name):
                return original

            def wrapped(*args, **kwargs):
                if len(args) == 2 and any(isinstance(arg, Tile) for arg in args):
                    return operation(*args, **kwargs)
                return original(*args, **kwargs)

            return wrapped

        namespace["max"] = tile_minmax("max", cutile_builder.maximum)
        namespace["min"] = tile_minmax("min", cutile_builder.minimum)
        return types.FunctionType(
            self.fn.__code__,
            namespace,
            self.fn.__name__,
            self.fn.__defaults__,
            self.fn.__closure__,
        )

    def run(self, *args, grid=(1,), client_manager=None, **kwargs) -> None:
        grid_dims = (grid,) if isinstance(grid, int) else tuple(grid)
        if not 1 <= len(grid_dims) <= 3 or any(
            not isinstance(dim, int) or dim < 0 for dim in grid_dims
        ):
            raise ValueError(
                "cuTile grid must contain one to three nonnegative integers"
            )
        grid_dims += (1,) * (3 - len(grid_dims))
        cutile_builder.set_grid_dim(*grid_dims)

        # Wrap NumPy arguments without copying their data.
        args = tuple(Array(arg) if isinstance(arg, np.ndarray) else arg for arg in args)
        kwargs = {
            name: Array(arg) if isinstance(arg, np.ndarray) else arg
            for name, arg in kwargs.items()
        }

        if client_manager is not None:
            bound = inspect.signature(self.fn).bind(*args, **kwargs)
            bound.apply_defaults()
            for name, arg in bound.arguments.items():
                client_manager.arg_callback(name, arg, arg)
            client_manager.grid_callback(grid_dims)

        # Traced execution is already patched by ClientManager.patch_run().
        # Standalone execution uses the same frontend hook without clients.
        scope = (
            get_frontend("cutile").patch_lang(self.fn)
            if client_manager is None
            else None
        )
        try:
            execution_fn = self._execution_function()
            execution_fn.__kwdefaults__ = self.fn.__kwdefaults__
            for grid_idx in itertools.product(*(range(dim) for dim in grid_dims)):
                cutile_builder.set_grid_idx(*grid_idx)
                if client_manager is not None:
                    client_manager.grid_idx_callback(grid_idx)
                    if not client_manager.pre_run_callback(self.fn):
                        return
                execution_fn(*args, **kwargs)
                if client_manager is not None:
                    if not client_manager.post_run_callback(self.fn):
                        return
        finally:
            cutile_unpatch_lang(scope)
