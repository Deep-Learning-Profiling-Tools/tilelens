"""NumPy-backed values and grid execution for the cuTile interpreter."""

from __future__ import annotations

import builtins
from contextlib import ExitStack, contextmanager
from functools import wraps
import inspect
import itertools
import operator
import threading
import types
import warnings
from typing import Annotated, Any, get_args, get_origin, get_type_hints

import numpy as np

from tilelens.utils.dtypes import STORAGE_DTYPES

try:
    import cuda.tile as ct
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Install cuda-tile to use the cuTile interpreter."
    ) from exc

from ..frontend.base import get_frontend


cutile_execution_lock = threading.RLock()


DTYPES = {
    ct.bool_: np.dtype(np.bool_),
    ct.int8: STORAGE_DTYPES["int8"],
    ct.uint8: STORAGE_DTYPES["uint8"],
    ct.uint16: STORAGE_DTYPES["uint16"],
    ct.uint32: STORAGE_DTYPES["uint32"],
    ct.uint64: STORAGE_DTYPES["uint64"],
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
if hasattr(ct, "float8_e8m0fnu"):
    DTYPES[ct.float8_e8m0fnu] = STORAGE_DTYPES["float8_e8m0fnu"]
    NUMPY_DTYPES[DTYPES[ct.float8_e8m0fnu]] = ct.float8_e8m0fnu

# Save signatures before patching the language module.
_LANG_FUNCTIONS = {
    name: getattr(ct, name)
    for name in ct.__all__
    if inspect.isfunction(getattr(ct, name))
}
if hasattr(ct, "Slice"):
    _LANG_FUNCTIONS["Slice"] = ct.Slice
_LANG_SIGNATURES = {name: inspect.signature(fn) for name, fn in _LANG_FUNCTIONS.items()}
_TILE_FUNCTION_CODE = ct.function(lambda: None).__code__


def _is_tile_function(fn):
    # cuTile 1.0 did not mark its tile-only wrappers with an attribute.
    return getattr(fn, "_cutile_function_wrapper", False) or (
        isinstance(fn, types.FunctionType) and fn.__code__ is _TILE_FUNCTION_CODE
    )


def _is_float(dtype):
    return dtype.kind == "f" or dtype in (
        DTYPES[ct.bfloat16],
        DTYPES[ct.float8_e4m3fn],
        DTYPES[ct.float8_e5m2],
        STORAGE_DTYPES["float8_e8m0fnu"],
    )


def _is_low_precision(dtype):
    return dtype in (
        DTYPES[ct.bfloat16],
        DTYPES[ct.float8_e4m3fn],
        DTYPES[ct.float8_e5m2],
        STORAGE_DTYPES["float8_e8m0fnu"],
    )


def _binary_values(x, y, force_float=False):
    """Use cuTile's promotion rules, including weak literals and range checks."""
    from cuda.tile._ir.ops_utils import promote_dtypes

    token = promote_dtypes(_promotion_type(x), _promotion_type(y), force_float)
    dtype = _numpy_dtype(token)
    compute_dtype = np.dtype(np.float32) if _is_low_precision(dtype) else dtype
    return (
        _tile_value(x).astype(compute_dtype),
        _tile_value(y).astype(compute_dtype),
        dtype,
    )


def _promotion_type(value):
    from cuda.tile._ir.type import LooselyTypedScalar

    if isinstance(value, (bool, int, float)) and not isinstance(value, np.generic):
        return LooselyTypedScalar(value)
    return (
        value.dtype
        if isinstance(value, Tile)
        else NUMPY_DTYPES[_tile_value(value).dtype]
    )


def _implicit_value(value, dtype):
    from cuda.tile._ir.ops_utils import promote_dtypes
    from cuda.tile._ir.type import LooselyTypedScalar

    dtype = _numpy_dtype(dtype)
    target = NUMPY_DTYPES[dtype]
    source = _promotion_type(value)
    if isinstance(source, LooselyTypedScalar):
        try:
            from cuda.tile._ir.cast_ops import check_implicit_cast
        except ImportError:
            from cuda.tile._ir.ops_utils import check_implicit_cast
        check_implicit_cast(source, target)
    elif promote_dtypes(source, target) != target:
        raise TypeError(f"Cannot implicitly cast to {target}")
    return _tile_value(value).astype(dtype)


def _normalize_shape(shape: Any) -> tuple:
    """Normalize cuTile's scalar-or-tuple shape and index arguments."""
    return (shape,) if isinstance(shape, (int, np.integer, Tile)) else tuple(shape)


def _numpy_dtype(dtype):
    # TF32 is a compute format. CPU interpretation uses float32 precision.
    if dtype == ct.tfloat32:
        return np.dtype(np.float32)
    if dtype in DTYPES:
        return DTYPES[dtype]
    from cuda.tile._ir.typing_support import to_dtype

    try:
        return DTYPES[to_dtype(dtype)]
    except (KeyError, TypeError):
        try:
            normalized = np.dtype(dtype)
            if normalized in NUMPY_DTYPES:
                return normalized
        except TypeError:
            pass
        raise TypeError(f"Unsupported dtype: {dtype}") from None


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
    valid = all(
        0 <= i < (array.shape[axis] + s - 1) // s
        for i, s, axis in zip(index, steps, axes)
    )
    starts = (
        tuple(i * s for i, s in zip(index, steps))
        if valid
        else tuple(array.shape[axis] for axis in axes)
    )
    return shape, axes, access_shape, starts


def _tile_value(value):
    """Return the NumPy value stored in a tile or Python scalar."""
    return value.data if isinstance(value, Tile) else np.asarray(value)


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


def _unary_value(value):
    if isinstance(value, (bool, int, float)) and not isinstance(value, np.generic):
        return Tile(value).data
    return _tile_value(value)


class Tile:
    """Immutable local tile value backed by an owned NumPy array."""

    def __init__(self, value, dtype=None):
        if isinstance(value, Tile):
            if dtype is None:
                dtype = value.dtype
            value = value.data
        if dtype is not None:
            value = np.asarray(value, dtype=_numpy_dtype(dtype))
        elif isinstance(value, np.generic):
            value = np.asarray(value)
        elif isinstance(value, bool):
            value = np.asarray(value, dtype=np.bool_)
        elif isinstance(value, int):
            try:
                from cuda.tile._ir.typing_support import dtype_of_constant_scalar
            except ImportError:
                from cuda.tile._ir.typing_support import (
                    typeof_pyval as dtype_of_constant_scalar,
                )

            value = np.asarray(
                value, dtype=_numpy_dtype(dtype_of_constant_scalar(value))
            )
        elif isinstance(value, float):
            value = np.asarray(value, dtype=np.float32)

        # Keep tiles immutable and independent of the source array.
        self.data = np.array(value, copy=True)
        self.data.setflags(write=False)
        self._dtype = (
            ct.tfloat32 if dtype == ct.tfloat32 else NUMPY_DTYPES[self.data.dtype]
        )

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self._dtype

    def __index__(self):
        if self.shape != () or self.data.dtype.kind not in "iu":
            raise TypeError("Only scalar integer tiles can be used as indices")
        return int(self.data)

    def item(self):
        if self.data.size != 1:
            raise ValueError("item requires a single-element tile")
        return Tile(self.data.reshape(()), self.dtype)

    def __getitem__(self, index):
        keys = index if isinstance(index, tuple) else (index,)
        if any(k is not None and k is not Ellipsis and k != slice(None) for k in keys):
            raise TypeError(
                "Tile indexing supports only None, ellipsis, and full slices; use extract"
            )
        return Tile(self.data[keys], self.dtype)

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

    def insert(self, index, value):
        return cutile_builder.insert(self, index, value)

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

    def __divmod__(self, other):
        return cutile_builder.divmod(self, other)

    def __rdivmod__(self, other):
        return cutile_builder.divmod(other, self)

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

    def __pow__(self, other, modulo=None):
        if modulo is not None:
            raise TypeError("cuTile does not support three-argument pow")
        return cutile_builder.pow(self, other)

    def __rpow__(self, other, modulo=None):
        if modulo is not None:
            raise TypeError("cuTile does not support three-argument pow")
        return cutile_builder.pow(other, self)

    def __rlshift__(self, other):
        return cutile_builder.bitwise_lshift(other, self)

    def __rrshift__(self, other):
        return cutile_builder.bitwise_rshift(other, self)

    def __abs__(self):
        return cutile_builder.abs(self)

    def __invert__(self):
        return cutile_builder.bitwise_not(self)

    def __pos__(self):
        if self.data.dtype == np.dtype(np.bool_):
            return cutile_builder.astype(self, ct.int32)
        return self

    def __neg__(self):
        return cutile_builder.negative(self)

    def __rmatmul__(self, other):
        return cutile_builder.matmul(other, self)

    # Needed before an eager Python f-string reaches Builder.print.
    def __str__(self):
        return cutile_builder._format_tile(self.data, lambda item: str(item.item()))

    __repr__ = __str__

    def __format__(self, spec):
        if not spec:
            return self.__str__()
        return cutile_builder._format_tile(
            self.data, lambda item: builtins.format(item.item(), spec)
        )


class Array:
    """Mutable global storage or a slice sharing its parent's NumPy memory."""

    def __init__(self, value, parent=None, origin=None):
        if any(stride % value.itemsize for stride in value.strides):
            raise ValueError("Array byte strides must be multiples of element size")
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
        if _is_low_precision(self.data.dtype):
            return self.data.astype(np.float32)
        return self.data

    def get_offsets(self):
        offsets = np.int64(0)
        for size, stride in zip(self.shape, self.data.strides):
            offsets = np.expand_dims(offsets, -1) + np.arange(size) * stride
        return Array(offsets)


class _TraceArray:
    """Numeric trace data with the original array's memory metadata."""

    def __init__(self, array):
        self.array = array

    @property
    def data(self):
        return self.array.numpy()

    def __getattr__(self, name):
        return getattr(self.array, name)


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
        return Tile(
            (self.array.shape[axis] + self.traversal_steps[axis] - 1)
            // self.traversal_steps[axis],
            ct.int32,
        )

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
        tile = Tile(_implicit_value(tile, self.array.dtype))
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
        axis = operator.index(axis)
        if axis not in (0, 1, 2):
            raise ValueError(f"Invalid axis {axis} for 3D grid")
        return Tile(self.grid_idx[axis], ct.int32)

    def num_tiles(self, array, /, axis, shape, order="C"):
        _, axes, shape, _ = _tile_access(array, (0,) * array.ndim, shape, order)
        axis = _normalize_axis(axis, array.ndim)
        return Tile(
            (array.shape[axes[axis]] + shape[axis] - 1) // shape[axis], ct.int32
        )

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
            padding = _implicit_value(kwargs.get("padding_value", 0), array.dtype)
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
            value = np.broadcast_to(_implicit_value(tile, array.dtype), valid.shape)
            array.data[tuple(key[valid] for key in keys)] = value[valid]
            return
        value = _implicit_value(tile, array.dtype)
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
                _implicit_value(fill_value, dtype),
                dtype=_numpy_dtype(dtype),
            ),
            dtype=dtype,
        )

    def zeros(self, shape, dtype):
        return self.full(shape, 0, dtype)

    def ones(self, shape, dtype):
        return self.full(shape, 1, dtype)

    def arange(self, size, /, *, dtype, start=0, step=1):
        (size,) = _validate_tile_shape(size)
        return Tile((start + np.arange(size) * step).astype(_numpy_dtype(dtype)))

    @staticmethod
    def _binary(x, y, operation, force_float=False):
        x, y, dtype = _binary_values(x, y, force_float)
        if dtype == np.dtype(np.bool_) and operation in (
            np.add,
            np.subtract,
            np.multiply,
            np.floor_divide,
            np.remainder,
        ):
            raise TypeError("Boolean arithmetic requires an integer cast")
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
        return self._binary(x, y, np.bitwise_and)

    def bitwise_or(self, x, y, /):
        return self._binary(x, y, np.bitwise_or)

    def reshape(self, x, /, shape):
        value = np.reshape(_tile_value(x), _normalize_shape(shape))
        _validate_tile_shape(value.shape)
        return Tile(value, x.dtype if isinstance(x, Tile) else None)

    def broadcast_to(self, x, /, shape):
        return Tile(
            np.broadcast_to(_tile_value(x), _validate_tile_shape(shape)),
            x.dtype if isinstance(x, Tile) else None,
        )

    def permute(self, x, /, axes):
        value = _tile_value(x)
        return Tile(
            value.transpose(_normalize_order(axes, value.ndim)),
            x.dtype if isinstance(x, Tile) else None,
        )

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
            ),
            x.dtype if isinstance(x, Tile) else None,
        )

    def expand_dims(self, x, /, axis):
        return Tile(
            np.expand_dims(_tile_value(x), operator.index(axis)),
            x.dtype if isinstance(x, Tile) else None,
        )

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
            value[tuple(slice(i * s, (i + 1) * s) for i, s in zip(index, shape))],
            x.dtype if isinstance(x, Tile) else None,
        )

    def astype(self, x, dtype, /):
        value, target = _tile_value(x), _numpy_dtype(dtype)
        if _is_float(value.dtype) and target.kind in "iu":
            bounds = np.iinfo(target)
            result = np.zeros(value.shape, dtype=target)
            finite = np.isfinite(value) & (value > bounds.min) & (value < bounds.max)
            result[finite] = value[finite].astype(target)
            result[value <= bounds.min] = bounds.min
            result[value >= bounds.max] = bounds.max
            return Tile(result)
        return Tile(
            value.astype(target), dtype=ct.tfloat32 if dtype == ct.tfloat32 else None
        )

    def insert(self, x, /, index, value):
        if (
            not isinstance(x, Tile)
            or not isinstance(value, Tile)
            or x.dtype != value.dtype
        ):
            raise TypeError("insert requires tiles with the same dtype")
        shape = value.shape or (1,) * x.ndim
        self.extract(x, index, shape)
        indices = tuple(operator.index(i) for i in _normalize_shape(index))
        result = x.data.copy()
        result[
            tuple(slice(i * s, (i + 1) * s) for i, s in zip(indices, shape))
        ] = value.data.reshape(shape)
        return Tile(result, x.dtype)

    def sum(self, x, /, axis=None, *, keepdims=False, **kwargs):
        value = _tile_value(x)
        dtype = np.dtype(np.int32) if value.dtype == np.dtype(np.bool_) else value.dtype
        compute_dtype = np.float32 if _is_low_precision(dtype) else dtype
        return Tile(
            np.asarray(
                np.sum(value, axis=axis, keepdims=keepdims, dtype=compute_dtype),
                dtype=dtype,
            )
        )

    def matmul(self, x, y, /):
        from cuda.tile._datatype import _resolve_mma_supported_dtype
        from cuda.tile._ir.ops_utils import promote_dtypes

        lhs, rhs = _tile_value(x), _tile_value(y)
        common = promote_dtypes(_promotion_type(x), _promotion_type(y))
        accumulator = _resolve_mma_supported_dtype(common, common)
        dtype = _numpy_dtype(common)
        compute = _numpy_dtype(accumulator)
        if compute == np.float16:
            compute = np.dtype(np.float32)
        return Tile(
            np.asarray(
                np.matmul(lhs.astype(compute), rhs.astype(compute)), dtype=dtype
            ),
            common,
        )

    def floordiv(self, x, y, /):
        lhs, rhs, dtype = _binary_values(x, y)
        result = np.floor(lhs / rhs) if _is_float(dtype) else np.floor_divide(lhs, rhs)
        return Tile(np.asarray(result, dtype=dtype))

    def mod(self, x, y, /):
        return self._binary(x, y, np.remainder)

    def divmod(self, x, y, /):
        _, _, dtype = _binary_values(x, y)
        if dtype.kind not in "iu":
            raise NotImplementedError("divmod requires integer operands")
        return self.floordiv(x, y), self.mod(x, y)

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
        return self._binary(x, y, np.fmax)

    def minimum(self, x, y, /, **kwargs):
        return self._binary(x, y, np.fmin)

    @staticmethod
    def _unary(x, operation):
        value = _unary_value(x)
        if not _is_float(value.dtype):
            value = value.astype(np.float32)
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
        value = _tile_value(x)
        if value.dtype == np.dtype(np.bool_):
            value = value.astype(np.int32)
        return Tile(np.fmax.reduce(value, axis=axis, keepdims=keepdims))

    def argmax(self, x, /, axis=None, *, keepdims=False):
        return self._arg_reduce(x, axis, keepdims, maximum=True)

    @staticmethod
    def _arg_reduce(x, axis, keepdims, *, maximum):
        value = _tile_value(x)
        if _is_low_precision(value.dtype):
            value = value.astype(np.float32)
        operation = np.fmax if maximum else np.fmin
        extremum = operation.reduce(value, axis=axis, keepdims=True)
        matches = value == extremum
        if axis is None:
            result = np.argmax(matches.reshape(-1))
            if keepdims:
                result = np.asarray(result).reshape((1,) * value.ndim)
        else:
            result = np.argmax(matches, axis=axis, keepdims=keepdims)
        return Tile(np.asarray(result, dtype=np.int32))

    def cumsum(self, x, /, axis=0, *, reverse=False, **kwargs):
        value = _tile_value(x)
        if reverse:
            value = np.flip(value, axis)
        dtype = np.dtype(np.int32) if value.dtype == np.dtype(np.bool_) else value.dtype
        compute_dtype = np.float32 if _is_low_precision(dtype) else dtype
        result = np.cumsum(value, axis=axis, dtype=compute_dtype).astype(dtype)
        return Tile(np.flip(result, axis) if reverse else result)

    def cat(self, tiles, /, axis):
        if len(tiles) != 2 or tiles[0].shape != tiles[1].shape:
            raise ValueError("cat requires two tiles of the same shape")
        return Tile(np.concatenate([_tile_value(tile) for tile in tiles], axis=axis))

    def mma(self, x, y, /, acc, *, use_fast_acc=False):
        from cuda.tile._datatype import _resolve_mma_supported_dtype

        _resolve_mma_supported_dtype(
            _promotion_type(x), _promotion_type(y), _promotion_type(acc)
        )
        if use_fast_acc:
            raise NotImplementedError("fast accumulation is unsupported")
        accumulator = _tile_value(acc)
        compute = np.float32 if accumulator.dtype == np.float16 else accumulator.dtype
        lhs = _tile_value(x).astype(compute)
        rhs = _tile_value(y).astype(compute)
        if lhs.ndim < 2 or rhs.ndim < 2:
            raise ValueError("mma requires matrix operands")
        output_shape = np.broadcast_shapes(lhs.shape[:-2], rhs.shape[:-2]) + (
            lhs.shape[-2],
            rhs.shape[-1],
        )
        if accumulator.shape != output_shape:
            raise ValueError(f"Invalid mma accumulator shape: {accumulator.shape}")
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

    def _atomic(
        self, array, indices, update, operation, *, check_bounds=True, expected=None
    ):
        keys, valid = _indexed_access(array, indices, check_bounds=check_bounds)
        updates = np.broadcast_to(_implicit_value(update, array.dtype), valid.shape)
        previous = self.gather(array, indices, check_bounds=check_bounds)
        old = previous.data.copy()
        final = previous.data.copy()
        positions = tuple(key[valid] for key in keys)
        offsets = sum(
            (key * stride for key, stride in zip(positions, array.data.strides)),
            np.zeros(np.count_nonzero(valid), dtype=np.int64),
        )
        order = np.argsort(offsets, kind="stable")
        sorted_offsets = offsets[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_offsets)) + 1, len(order)]
        old_values, new_values = old[valid], final[valid]
        valid_updates = updates[valid]
        comparisons = (
            None
            if expected is None
            else np.broadcast_to(_implicit_value(expected, array.dtype), valid.shape)
        )
        if comparisons is not None:
            old[~valid] = comparisons[~valid]
            comparisons = comparisons[valid]
        for start, stop in zip(starts[:-1], starts[1:]):
            group = order[start:stop]
            if not group.size:
                continue
            initial = old_values[group[0]]
            if operation in ("exchange", "cas"):
                current = initial
                for lane in group:
                    old_values[lane] = current
                    if operation == "exchange" or current == comparisons[lane]:
                        current = valid_updates[lane]
            else:
                sequence = np.concatenate(
                    (np.asarray(initial).reshape(1), valid_updates[group])
                )
                accumulated = operation.accumulate(sequence, dtype=array.data.dtype)
                old_values[group] = accumulated[:-1]
                current = accumulated[-1]
            new_values[group] = current
        old[valid], final[valid] = old_values, new_values
        # One vector access per operation, even when addresses repeat.
        self.scatter(
            array, indices, Tile(final), mask=Tile(valid), check_bounds=check_bounds
        )
        return Tile(old)

    def atomic_add(self, array, indices, update, /, *, check_bounds=True, **kwargs):
        return self._atomic(array, indices, update, np.add, check_bounds=check_bounds)

    def atomic_max(self, array, indices, update, /, *, check_bounds=True, **kwargs):
        return self._atomic(array, indices, update, np.fmax, check_bounds=check_bounds)

    def atomic_min(self, array, indices, update, /, *, check_bounds=True, **kwargs):
        return self._atomic(array, indices, update, np.fmin, check_bounds=check_bounds)

    def atomic_and(self, array, indices, update, /, *, check_bounds=True, **kwargs):
        return self._atomic(
            array, indices, update, np.bitwise_and, check_bounds=check_bounds
        )

    def atomic_or(self, array, indices, update, /, *, check_bounds=True, **kwargs):
        return self._atomic(
            array, indices, update, np.bitwise_or, check_bounds=check_bounds
        )

    def atomic_xor(self, array, indices, update, /, *, check_bounds=True, **kwargs):
        return self._atomic(
            array, indices, update, np.bitwise_xor, check_bounds=check_bounds
        )

    def atomic_xchg(self, array, indices, update, /, *, check_bounds=True, **kwargs):
        return self._atomic(
            array, indices, update, "exchange", check_bounds=check_bounds
        )

    def atomic_cas(
        self, array, indices, expected, desired, /, *, check_bounds=True, **kwargs
    ):
        return self._atomic(
            array, indices, desired, "cas", expected=expected, check_bounds=check_bounds
        )

    @staticmethod
    def _check_cpu_math_options(
        rounding_mode=None, flush_to_zero=False, *, rounding_modes=(ct.RoundingMode.RN,)
    ):
        if not isinstance(flush_to_zero, bool):
            raise TypeError("flush_to_zero must be a constant bool")
        if rounding_mode is not None and not isinstance(rounding_mode, ct.RoundingMode):
            raise TypeError("rounding_mode must be a RoundingMode or None")
        if flush_to_zero:
            raise NotImplementedError("flush_to_zero is unsupported")
        if rounding_mode is not None and rounding_mode not in rounding_modes:
            raise NotImplementedError(
                "CPU simulation does not implement this rounding mode"
            )

    def abs(self, x, /):
        value = _unary_value(x)
        if value.dtype == np.dtype(np.bool_) or value.dtype.kind == "u":
            return Tile(value)
        if value.dtype.kind not in "if" and not _is_float(value.dtype):
            raise TypeError("abs requires a numeric tile")
        compute = value.astype(np.float32) if _is_low_precision(value.dtype) else value
        return Tile(np.abs(compute).astype(value.dtype))

    def negative(self, x, /):
        value = _unary_value(x)
        if value.dtype == np.dtype(np.bool_):
            value = value.astype(np.int32)
        if value.dtype.kind not in "iuf" and not _is_float(value.dtype):
            raise TypeError("negative requires a numeric tile")
        compute = value.astype(np.float32) if _is_low_precision(value.dtype) else value
        return Tile(np.negative(compute).astype(value.dtype))

    def bitwise_not(self, x, /):
        value = _tile_value(x)
        if value.dtype.kind not in "biu":
            raise TypeError("bitwise_not requires an integer or boolean tile")
        return Tile(np.bitwise_not(value))

    def sqrt(self, x, /, *, rounding_mode=None, flush_to_zero=False):
        self._check_cpu_math_options(rounding_mode, flush_to_zero)
        return self._unary(x, np.sqrt)

    def sin(self, x, /):
        return self._unary(x, np.sin)

    def cos(self, x, /):
        return self._unary(x, np.cos)

    def tan(self, x, /):
        return self._unary(x, np.tan)

    def tanh(self, x, /, *, rounding_mode=None):
        self._check_cpu_math_options(
            rounding_mode, rounding_modes=(ct.RoundingMode.FULL,)
        )
        value = _tile_value(x)
        if (
            rounding_mode is not None
            and _is_float(value.dtype)
            and value.dtype != np.float32
        ):
            raise TypeError("Explicit FULL tanh rounding requires float32")
        return self._unary(x, np.tanh)

    def sinh(self, x, /):
        return self._unary(x, np.sinh)

    def cosh(self, x, /):
        return self._unary(x, np.cosh)

    def log2(self, x, /):
        return self._unary(x, np.log2)

    def floor(self, x, /):
        if not _is_float(_tile_value(x).dtype):
            raise TypeError("floor requires a floating-point tile")
        return self._unary(x, np.floor)

    def ceil(self, x, /):
        if not _is_float(_tile_value(x).dtype):
            raise TypeError("ceil requires a floating-point tile")
        return self._unary(x, np.ceil)

    def isnan(self, x, /):
        value = _tile_value(x)
        if not _is_float(value.dtype):
            # A Python scalar is loosely typed in the installed compiler.
            if isinstance(x, (bool, int, float)):
                return Tile(np.asarray(False, dtype=np.bool_))
            raise TypeError("isnan requires a floating-point tile")
        compute = value.astype(np.float32) if _is_low_precision(value.dtype) else value
        return Tile(np.isnan(compute))

    def pow(self, x, y, /):
        if all(
            isinstance(value, (bool, int, float)) and not isinstance(value, np.generic)
            for value in (x, y)
        ):
            return builtins.pow(x, y)
        lhs, rhs, dtype = _binary_values(x, y)
        if not _is_float(dtype):
            # 1.5.0 emits FPowF only for dynamic operands; there is no integer
            # power lowering. Do not silently adopt NumPy's integer power.
            raise NotImplementedError(
                "Dynamic integer/bool pow has no cuTile 1.5.0 lowering"
            )
        return Tile(np.power(lhs, rhs).astype(dtype))

    def atan2(self, x1, x2, /):
        lhs, rhs, dtype = _binary_values(x1, x2)
        if not _is_float(dtype):
            raise NotImplementedError(
                "Dynamic integer/bool atan2 has no cuTile 1.5.0 lowering"
            )
        return Tile(np.arctan2(lhs, rhs).astype(dtype))

    def cdiv(self, x, y, /):
        lhs, rhs, dtype = _binary_values(x, y)
        if dtype.kind not in "iu":
            raise TypeError("cdiv requires integer operands")
        if np.any(rhs == 0):
            raise ZeroDivisionError("cdiv by zero")
        # Exact ceil-division without fixed-width intermediate overflow.
        result = -np.floor_divide(-lhs.astype(object), rhs.astype(object))
        limits = np.iinfo(dtype)
        if np.any(result < limits.min) or np.any(result > limits.max):
            raise NotImplementedError(
                "Integer division overflow has no defined CPU result"
            )
        return Tile(np.asarray(result, dtype=dtype))

    @staticmethod
    def _reduction_axes(axis, ndim):
        if axis is None:
            return tuple(range(ndim))
        axes = axis if isinstance(axis, tuple) else (axis,)
        if any(isinstance(a, (Tile, bool, np.bool_)) for a in axes):
            raise TypeError("Reduction axes must be constant integers")
        axes = tuple(_normalize_axis(a, ndim) for a in axes)
        if len(set(axes)) != len(axes):
            raise ValueError("Repeated reduction axis")
        return axes

    def min(self, x, /, axis=None, *, keepdims=False, flush_to_zero=False):
        self._check_cpu_math_options(flush_to_zero=flush_to_zero)
        if not isinstance(keepdims, bool):
            raise TypeError("keepdims must be a constant bool")
        value = _tile_value(x)
        if value.dtype == np.dtype(np.bool_):
            value = value.astype(np.int32)
        if value.dtype.kind not in "iuf" and not _is_float(value.dtype):
            raise TypeError("min requires a numeric tile")
        dtype = value.dtype
        compute = value.astype(np.float32) if _is_low_precision(dtype) else value
        # cuTile MinF uses propagate_nan=False, unlike np.min.
        result = np.fmin.reduce(
            compute, axis=self._reduction_axes(axis, value.ndim), keepdims=keepdims
        )
        return Tile(np.asarray(result, dtype=dtype))

    def prod(
        self,
        x,
        /,
        axis=None,
        *,
        keepdims=False,
        rounding_mode=None,
        flush_to_zero=False,
    ):
        self._check_cpu_math_options(rounding_mode, flush_to_zero)
        if not isinstance(keepdims, bool):
            raise TypeError("keepdims must be a constant bool")
        value = _tile_value(x)
        if value.dtype == np.dtype(np.bool_):
            value = value.astype(np.int32)
        axes = sorted(self._reduction_axes(axis, value.ndim), reverse=True)
        result = Tile(value)
        # Cast after each scalar multiply; do not use NumPy's int64 accumulator
        # or accumulate an entire float16 reduction in float32.
        for a in axes:
            result = self.reduce(result, a, operator.mul, 1, keepdims=keepdims)
        return result

    def argmin(self, x, /, axis=None, *, keepdims=False):
        return self._arg_reduce(x, axis, keepdims, maximum=False)

    def cumprod(
        self, x, /, axis=0, *, reverse=False, rounding_mode=None, flush_to_zero=False
    ):
        self._check_cpu_math_options(rounding_mode, flush_to_zero)
        value = _tile_value(x)
        if value.dtype == np.dtype(np.bool_):
            value = value.astype(np.int32)
        return self.scan(Tile(value), axis, operator.mul, 1, reverse=reverse)

    def _fold(self, x, axis, func, identity, *, scan, reverse=False, keepdims=False):
        """Sequential CPU reference for a pure associative scalar callback."""
        if not isinstance(reverse, bool) or not isinstance(keepdims, bool):
            raise TypeError("reverse and keepdims must be constant bools")
        if isinstance(axis, (Tile, tuple, bool, np.bool_)) or axis is None:
            raise TypeError("Custom reduce/scan accepts one constant integer axis")
        tuple_mode = isinstance(x, tuple)
        inputs = x if tuple_mode else (x,)
        if not inputs or any(not isinstance(t, Tile) for t in inputs):
            raise TypeError(
                "Custom reduce/scan requires a Tile or nonempty tuple of Tiles"
            )
        ids = identity if tuple_mode else (identity,)
        if not isinstance(ids, tuple) or len(ids) != len(inputs):
            raise TypeError("Identity tuple must match the number of input tiles")
        if any(not isinstance(i, (bool, int, float)) for i in ids):
            raise TypeError("Identities must be constant Python scalar values")
        # Unwrap cuTile's host-only guard before calling the reduction body.
        while _is_tile_function(func):
            func = func.__wrapped__
        if not callable(func):
            raise TypeError("func must be callable")
        values = np.broadcast_arrays(*(_tile_value(t) for t in inputs))
        shape = values[0].shape
        a = _normalize_axis(axis, len(shape))
        row_shape = shape[:a] + shape[a + 1 :]
        output_shape = shape if scan else row_shape
        outputs = [np.empty(output_shape, dtype=v.dtype) for v in values]
        initial = tuple(Tile(i, t.dtype) for i, t in zip(ids, inputs))
        positions = range(shape[a] - 1, -1, -1) if reverse else range(shape[a])
        for row in np.ndindex(row_shape):
            acc = initial
            for position in positions:
                key = row[:a] + (position,) + row[a:]
                current = tuple(Tile(v[key]) for v in values)
                # reverse=True is a prefix over the reversed sequence.
                combined = func(*acc, *current)
                results = combined if tuple_mode else (combined,)
                if not isinstance(results, tuple) or len(results) != len(inputs):
                    raise TypeError("Callback result structure must match its inputs")
                checked = []
                for result, template in zip(results, inputs):
                    if _tile_value(result).shape != ():
                        raise TypeError("Callback must return scalar Tiles")
                    checked.append(Tile(_implicit_value(result, template.dtype)))
                acc = tuple(checked)
                if scan:
                    for out, result in zip(outputs, acc):
                        out[key] = result.data
            if not scan:
                for out, result in zip(outputs, acc):
                    out[row] = result.data
        if not scan and keepdims:
            outputs = [np.expand_dims(out, a) for out in outputs]
        tiles = tuple(Tile(out) for out in outputs)
        return tiles if tuple_mode else tiles[0]

    def reduce(self, x, /, axis, func, identity, *, keepdims=False):
        return self._fold(x, axis, func, identity, scan=False, keepdims=keepdims)

    def scan(self, x, /, axis, func, identity, *, reverse=False):
        return self._fold(x, axis, func, identity, scan=True, reverse=reverse)

    @staticmethod
    def _byte_dtype(dtype):
        if dtype == ct.bool_:
            raise TypeError("Byte reinterpretation to/from bool is not supported")
        if dtype in (ct.tfloat32, getattr(ct, "float4_e2m1fn", None)):
            raise NotImplementedError(
                "TF32/sub-byte byte reinterpretation is unsupported"
            )
        try:
            storage = _numpy_dtype(dtype)
        except (KeyError, TypeError) as exc:
            raise NotImplementedError(f"No registered CPU storage for {dtype}") from exc
        return storage

    def bitcast(self, x, /, dtype):
        value = _unary_value(x)
        if value.dtype == np.dtype(np.bool_):
            raise TypeError("bitcast from bool is not supported")
        if isinstance(x, Tile) and x.dtype == ct.tfloat32:
            raise NotImplementedError(
                "Raw TF32 representation is not modeled by float32 storage"
            )
        target = self._byte_dtype(dtype)
        if value.dtype.itemsize != target.itemsize:
            raise TypeError("bitcast requires equal element bit widths")
        # Flatten first: NumPy's contiguity and 0d view restrictions must not
        # change the logical tile shape or reject a scalar bitcast.
        return Tile(
            np.array(value, copy=True, order="C")
            .reshape(-1)
            .view(target)
            .reshape(value.shape)
        )

    def pack_to_bytes(self, x, /):
        value = _unary_value(x)
        if value.dtype == np.dtype(np.bool_):
            raise TypeError("pack_to_bytes from bool is not supported")
        if isinstance(x, Tile) and x.dtype == ct.tfloat32:
            raise NotImplementedError(
                "Raw TF32 representation is not modeled by float32 storage"
            )
        if value.dtype.hasobject or value.dtype.kind in "SUcMm":
            raise TypeError("pack_to_bytes requires numeric tile storage")
        if not np.little_endian:
            raise NotImplementedError(
                "Byte packing is implemented for little-endian hosts only"
            )
        # Current Tile cannot represent a logical 4-bit element. Do not
        # accept an ml_dtypes float4 array as an unpacked byte-per-value tile.
        if value.dtype.name == "float4_e2m1fn":
            raise NotImplementedError("Sub-byte packing needs logical dtype metadata")
        return Tile(np.array(value, copy=True, order="C").reshape(-1).view(np.uint8))

    def unpack_from_bytes(self, x, /, dtype):
        value = _tile_value(x)
        if value.ndim != 1 or value.dtype != np.dtype(np.uint8):
            raise TypeError("unpack_from_bytes requires a 1D uint8 tile")
        target = self._byte_dtype(dtype)
        if value.size % target.itemsize:
            raise ValueError("Byte count must be divisible by the target element size")
        if not np.little_endian:
            raise NotImplementedError(
                "Byte unpacking is implemented for little-endian hosts only"
            )
        return Tile(np.array(value, copy=True, order="C").view(target))

    def astile(self, value, /, *, dtype):
        def build(item):
            if isinstance(item, tuple):
                _validate_tile_shape((len(item),))
                children = [build(child) for child in item]
                if any(child.shape != children[0].shape for child in children):
                    raise ValueError("astile requires uniform nested tuple shapes")
                return np.stack(children)
            if not isinstance(item, (Tile, bool, int, float, np.generic)):
                raise TypeError("astile accepts a scalar or nested tuple of scalars")
            scalar = _tile_value(item)
            if scalar.shape != ():
                raise TypeError("astile tuple leaves must be scalar values")
            return self.astype(Tile(scalar), dtype).data

        return Tile(build(value))

    def assert_(self, cond, /, message=None):
        if not isinstance(cond, Tile) or cond.data.dtype != np.dtype(np.bool_):
            raise TypeError("assert_ requires a boolean Tile")
        if message is not None and not isinstance(message, str):
            raise TypeError("assert_ message must be a string or None")
        if not np.all(cond.data):
            raise AssertionError("" if message is None else message)

    def static_assert(self, condition, message=None, /):
        if isinstance(condition, (Tile, Array)):
            raise TypeError("static_assert requires a constant condition")
        if not condition:
            raise AssertionError(message or "static assertion failed")

    def static_eval(self, expr, /):
        if isinstance(expr, (Tile, Array)):
            raise TypeError("static_eval requires a constant expression")
        return expr

    def ensure_constant(self, value, /):
        return self.static_eval(value)

    def grid_dependency_control_wait(self):
        raise NotImplementedError(
            "Programmatic launch dependencies require GPU execution"
        )

    def grid_dependency_control_launch_dependents(self):
        raise NotImplementedError(
            "Programmatic launch dependencies require GPU execution"
        )

    def static_iter(self, iterable, /):
        # Snapshot the iterable before unrolling, as the compiler does.
        if isinstance(iterable, (Tile, Array)):
            raise TypeError("static_iter requires a Python iterable")
        return iter(tuple(iterable))

    def assume_divisible_by(self, x, divisor):
        value = _tile_value(x)
        if value.shape != () or value.dtype.kind not in "iu":
            raise TypeError("assume_divisible_by requires an integer scalar")
        if not isinstance(divisor, int) or isinstance(divisor, bool) or divisor <= 0:
            raise ValueError("divisor must be a positive constant integer")
        if int(value) % divisor:
            raise ValueError("assume_divisible_by claim is false")
        return x

    def num_blocks(self, axis):
        if not isinstance(axis, int) or isinstance(axis, bool) or not 0 <= axis <= 2:
            raise ValueError("num_blocks axis must be a constant integer in [0, 2]")
        return Tile(
            np.asarray((tuple(self.grid_dims) + (1, 1, 1))[axis], dtype=np.int32)
        )

    class _AdvancedSlice:
        """CPU value for ct.Slice; retain start and length without executing a stub."""

        def __init__(self, start, length):
            value = _tile_value(start)
            if value.shape != () or value.dtype.kind != "i":
                raise TypeError("Slice start must be a signed integer scalar")
            if not isinstance(length, int) or isinstance(length, bool):
                raise TypeError("Slice length must be a constant integer")
            _validate_tile_shape((length,))
            self.start = int(value)
            self.length = length

    def Slice(self, start, length, /):
        return self._AdvancedSlice(start, length)

    def _advanced_index_keys(self, array, indices):
        if (
            array.ndim < 2
            or not isinstance(indices, tuple)
            or len(indices) != array.ndim
        ):
            raise TypeError(
                "Advanced indexing requires rank >= 2 and one index per axis"
            )
        vectors = []
        sparse = 0
        sparse_vector = None
        for item in indices:
            if isinstance(item, self._AdvancedSlice):
                vector = np.arange(item.length, dtype=np.int64) + item.start
            elif (
                isinstance(item, Tile)
                and item.ndim == 1
                and item.data.dtype.kind in "iu"
            ):
                sparse += 1
                vector = item.data
                sparse_vector = vector
            else:
                raise TypeError("Indices must be 1D integer Tiles or ct.Slice values")
            vectors.append(vector)
        if sparse != 1:
            raise ValueError("Advanced indexing requires exactly one sparse index tile")
        _validate_tile_shape(tuple(len(v) for v in vectors))
        # Duplicate sparse indices would introduce unordered conflicting stores.
        # Loads are still well-defined; the store method checks this flag.
        repeated = np.unique(sparse_vector).size != sparse_vector.size
        keys = []
        for a, vector in enumerate(vectors):
            shape = [1] * array.ndim
            shape[a] = vector.size
            keys.append(Tile(vector.reshape(shape)))
        return tuple(keys), repeated

    @staticmethod
    def _check_memory_hints(latency, allow_tma):
        if latency is not None and (
            not isinstance(latency, int) or not 1 <= latency <= 10
        ):
            raise ValueError("latency must be None or an integer in [1, 10]")
        if allow_tma is not None and not isinstance(allow_tma, bool):
            raise TypeError("allow_tma must be None or a constant bool")
        # Valid hints affect performance only, not CPU memory semantics.

    def load_advanced_indexing(
        self,
        array,
        indices,
        /,
        *,
        padding_mode=ct.PaddingMode.UNDETERMINED,
        latency=None,
        allow_tma=None,
    ):
        self._check_memory_hints(latency, allow_tma)
        keys, _ = self._advanced_index_keys(array, indices)
        _, valid = _indexed_access(array, keys)
        if not np.any(valid):
            raise NotImplementedError(
                "Entirely out-of-bounds advanced loads are undefined"
            )
        if padding_mode == ct.PaddingMode.UNDETERMINED:
            if not np.all(valid):
                raise NotImplementedError(
                    "Cannot supply a defined value for undetermined padding"
                )
            padding = 0
        else:
            modes = {
                ct.PaddingMode.ZERO: 0,
                ct.PaddingMode.NEG_ZERO: -0.0,
                ct.PaddingMode.NAN: np.nan,
                ct.PaddingMode.POS_INF: np.inf,
                ct.PaddingMode.NEG_INF: -np.inf,
            }
            if padding_mode not in modes:
                raise ValueError("Unsupported padding mode")
            if padding_mode != ct.PaddingMode.ZERO and not _is_float(array.data.dtype):
                raise TypeError("Nonzero padding requires a floating-point array")
            padding = modes[padding_mode]
        return self.gather(array, keys, padding_value=padding, check_bounds=True)

    def store_advanced_indexing(
        self, array, indices, tile, /, *, latency=None, allow_tma=None
    ):
        self._check_memory_hints(latency, allow_tma)
        keys, repeated = self._advanced_index_keys(array, indices)
        _, valid = _indexed_access(array, keys)
        if not np.any(valid):
            raise NotImplementedError(
                "Entirely out-of-bounds advanced stores are undefined"
            )
        if repeated:
            raise NotImplementedError(
                "Duplicate sparse store indices have unordered write semantics"
            )
        if not isinstance(tile, Tile) or tile.shape != valid.shape:
            raise ValueError(
                "Stored Tile shape must exactly match the index tile shape"
            )
        value = Tile(_implicit_value(tile, array.dtype))
        return self.scatter(array, keys, value, check_bounds=True)

    def printf(self, format, *args):
        """Format tile values for CPU diagnostics."""
        import re

        if not isinstance(format, str):
            raise TypeError("printf format must be a string")
        if any(not isinstance(arg, Tile) for arg in args):
            raise TypeError("printf arguments must be Tiles")
        values = iter(args)

        def render(match):
            if match[0] == "%%":
                return "%"
            options, specifier = match.groups()
            if specifier is None:
                raise ValueError("Unsupported or invalid printf format")
            if specifier in "aA":
                raise NotImplementedError("Hexadecimal float formatting is unsupported")
            try:
                value = next(values).data
            except StopIteration:
                raise ValueError("Not enough printf arguments") from None
            integer = value.dtype.kind in "biu"
            if (integer and specifier not in "diuoxX") or (
                not integer
                and (not _is_float(value.dtype) or specifier not in "eEfFgG")
            ):
                raise TypeError("printf specifier does not match tile dtype")

            def scalar_format(item):
                number = int(item) if integer else float(item)
                if integer and specifier in "uoxX":
                    bits = 1 if value.dtype.kind == "b" else value.dtype.itemsize * 8
                    number &= (1 << bits) - 1
                return ("%" + options + specifier) % number

            return self._format_tile(value, scalar_format)

        # The Tile dtype supplies the width, so C length modifiers are skipped.
        pattern = r"%%|%([ #+0-]*\d*(?:\.\d+)?)(?:hh|h|ll|l)?([diuoxXeEfFgGaA])|%"
        result = re.sub(pattern, render, format)
        if next(values, None) is not None:
            raise ValueError("Too many printf arguments")
        builtins.print(result, end="")

    @staticmethod
    def _format_tile(value, scalar_format):
        if value.ndim == 0:
            return scalar_format(value[()])
        return (
            "["
            + ", ".join(Builder._format_tile(child, scalar_format) for child in value)
            + "]"
        )

    def print(self, *args, sep=" ", end="\n"):
        if not isinstance(sep, str) or not isinstance(end, str):
            raise TypeError("print sep and end must be strings")

        def render(value, nested=False):
            if isinstance(value, Tile):
                return self._format_tile(value.data, lambda item: str(item.item()))
            if isinstance(value, tuple):
                contents = ", ".join(render(item, True) for item in value)
                return "(" + contents + ("," if len(value) == 1 else "") + ")"
            if isinstance(value, str):
                return repr(value) if nested else value
            if isinstance(value, (bool, int, float, ct.DType)):
                return str(value)
            raise NotImplementedError(
                "CPU print does not support this cuTile value type"
            )

        builtins.print(*(render(arg) for arg in args), sep=sep, end=end)

    @staticmethod
    def _scaled_operands(x, x_scale, y, y_scale):
        from cuda.tile._datatype import _resolve_mma_scaled_supported_dtype

        _resolve_mma_scaled_supported_dtype(
            x.dtype, x_scale.dtype, y.dtype, y_scale.dtype, ct.float32
        )
        if x.dtype not in (ct.float8_e4m3fn, ct.float8_e5m2):
            raise NotImplementedError("Sub-byte scaled matmul is unsupported")

        def expand(value, scale, axis):
            if value.ndim != scale.ndim or value.ndim not in (2, 3):
                raise ValueError("Scaled matmul requires matching matrix ranks")
            if any(
                a != b
                for i, (a, b) in enumerate(zip(value.shape, scale.shape))
                if i != axis % value.ndim
            ):
                raise ValueError("Invalid scale shape")
            size, remainder = divmod(value.shape[axis], scale.shape[axis])
            if remainder or size != 32:
                raise ValueError("FP8 scaling blocks must have 32 elements")
            # Avoid overflowing one scaled operand before reciprocal scales cancel.
            return value.data.astype(np.float64) * np.repeat(
                scale.data.astype(np.float64), size, axis=axis
            )

        return expand(x, x_scale, -1), expand(y, y_scale, -2)

    def mma_scaled(self, x, x_scale, y, y_scale, /, acc):
        if acc.dtype != ct.float32:
            raise TypeError("Scaled matmul requires a float32 accumulator")
        lhs, rhs = self._scaled_operands(x, x_scale, y, y_scale)
        result = np.matmul(lhs, rhs)
        if result.shape != acc.shape:
            raise ValueError("Invalid scaled matmul accumulator shape")
        return Tile(np.asarray(result + acc.data, dtype=np.float32))


cutile_builder = Builder()


def _language_operation(name):
    original = _LANG_FUNCTIONS[name]

    @wraps(original)
    def execute(*args, **kwargs):
        bound = _LANG_SIGNATURES[name].bind(*args, **kwargs)
        bound.apply_defaults()
        for key, enum in (
            ("memory_order", ct.MemoryOrder),
            ("memory_scope", ct.MemoryScope),
        ):
            if key in bound.arguments and not isinstance(bound.arguments[key], enum):
                raise TypeError(f"Invalid {key}: {bound.arguments[key]}")
        cutile_builder._check_cpu_math_options(
            bound.arguments.get("rounding_mode"),
            bound.arguments.get("flush_to_zero", False),
            rounding_modes=(ct.RoundingMode.RN, ct.RoundingMode.FULL),
        )
        method = getattr(cutile_builder, name, None)
        if method is None:
            raise NotImplementedError(f"ct.{name} is unsupported")
        return method(*args, **kwargs)

    return execute


def cutile_patch_lang(scope=None):
    """Bind cuTile operations to the interpreter while preserving signatures."""
    for name in _LANG_FUNCTIONS:
        if name in ("function", "compiler_timeout"):
            continue
        value = _language_operation(name)
        if scope is None:
            setattr(ct, name, value)
        else:
            scope.set_attr(ct, name, value)


@contextmanager
def _execution_scope(fn):
    """Patch referenced aliases and helper globals, then restore them."""
    scope, visited = ExitStack(), set()
    originals = {
        id(value): name
        for name, value in _LANG_FUNCTIONS.items()
        if name not in ("function", "compiler_timeout")
    }
    missing = object()

    def set_global(namespace, name, value):
        previous = namespace.get(name, missing)
        if previous is missing:
            scope.callback(namespace.pop, name, None)
        else:
            scope.callback(operator.setitem, namespace, name, previous)
        namespace[name] = value

    def patched_value(value):
        operation = originals.get(id(value))
        if operation is not None:
            wrapper = _language_operation(operation)
            visited.add(id(wrapper))
            return wrapper
        return inspect.unwrap(value) if _is_tile_function(value) else value

    def tile_minmax(original, operation):
        def execute(*args, **kwargs):
            if len(args) == 2 and any(isinstance(arg, Tile) for arg in args):
                return operation(*args, **kwargs)
            return original(*args, **kwargs)

        return execute

    def patch_function(function):
        if not isinstance(function, types.FunctionType) or id(function) in visited:
            return
        visited.add(id(function))
        namespace = function.__globals__
        for name, operation in (
            ("max", cutile_builder.maximum),
            ("min", cutile_builder.minimum),
        ):
            original = getattr(builtins, name)
            if namespace.get(name, original) is original:
                set_global(namespace, name, tile_minmax(original, operation))
        for name in function.__code__.co_names:
            value = namespace.get(name)
            replacement = patched_value(value)
            if replacement is not value:
                set_global(namespace, name, replacement)
                value = replacement
            if isinstance(value, types.FunctionType):
                patch_function(value)
            elif isinstance(value, types.ModuleType) and not value.__name__.startswith(
                ("cuda", "numpy", "tilelens")
            ):
                for attribute in function.__code__.co_names:
                    helper = getattr(value, attribute, None)
                    replacement = patched_value(helper)
                    if replacement is not helper:
                        set_global(value.__dict__, attribute, replacement)
                    patch_function(replacement)
        for cell in function.__closure__ or ():
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            replacement = patched_value(value)
            if replacement is not value:
                scope.callback(setattr, cell, "cell_contents", value)
                cell.cell_contents = replacement
            patch_function(replacement)

    with scope:
        patch_function(fn)
        yield fn


def _parameter_annotations(fn):
    """Resolve constant and integer-scalar annotations for both execution paths."""
    constant_type = type(get_args(ct.Constant[int])[1])
    scalar_alias = getattr(ct, "ScalarInt64", None)
    scalar_type: Any = (
        type(get_args(scalar_alias)[1]) if scalar_alias is not None else ()
    )
    result = {}
    for name, annotation in get_type_hints(fn, include_extras=True).items():
        if get_origin(annotation) is not Annotated:
            continue
        metadata = get_args(annotation)[1:]
        is_constant = any(isinstance(item, constant_type) for item in metadata)
        scalar_dtype = next(
            (item.dtype for item in metadata if isinstance(item, scalar_type)), None
        )
        result[name] = (is_constant, scalar_dtype)
    return result


def _check_integer_scalar(value, dtype):
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise TypeError("ScalarAnnotation requires an integer scalar")
    bounds = np.iinfo(_numpy_dtype(dtype))
    if not bounds.min <= value <= bounds.max:
        raise ValueError(f"Integer argument is out of range of {dtype}")


def _wrap_argument(arg, scalar_dtype=None):
    if scalar_dtype is not None:
        _check_integer_scalar(arg, scalar_dtype)
        return Tile(arg, scalar_dtype)
    if isinstance(arg, (Array, Tile)):
        return arg
    if isinstance(arg, np.ndarray):
        return Array(arg)
    if isinstance(arg, (bool, int, float, np.generic)):
        return Tile(arg)
    if arg is None or isinstance(arg, (str, tuple)):
        return arg
    import torch

    if isinstance(arg, torch.Tensor):
        if arg.device.type != "cpu":
            raise TypeError("cuTile interpretation requires CPU tensors")
        arg = arg.detach()
        storage = {
            torch.bfloat16: torch.int16,
            torch.float8_e4m3fn: torch.uint8,
            torch.float8_e5m2: torch.uint8,
        }
        if hasattr(torch, "float8_e8m0fnu"):
            storage[torch.float8_e8m0fnu] = torch.uint8
        if arg.dtype in storage:
            dtype = _numpy_dtype(arg.dtype)
            return Array(arg.view(storage[arg.dtype]).numpy().view(dtype))
        return Array(arg.numpy())
    raise TypeError(f"Unsupported cuTile argument: {type(arg).__name__}")


def cutile_unpatch_lang(scope=None):
    """Restore the language functions saved by a patch scope."""
    if scope is not None:
        scope.restore()


class CuTileInterpretedFunction:
    """Execute a Python kernel over a grid with optional tracing callbacks."""

    def __init__(self, fn: Any) -> None:
        # @ct.kernel stores the original Python function in _pyfunc.
        self.fn = fn._pyfunc if isinstance(fn, ct.kernel) else fn

    def run(self, *args, grid=(1,), client_manager=None, **kwargs) -> None:
        with cutile_execution_lock:
            return self._run(*args, grid=grid, client_manager=client_manager, **kwargs)

    def _run(self, *args, grid=(1,), client_manager=None, **kwargs) -> None:
        grid_dims = (grid,) if isinstance(grid, int) else tuple(grid)
        if not 1 <= len(grid_dims) <= 3 or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0
            for dim in grid_dims
        ):
            raise ValueError("cuTile grid must contain one to three positive integers")
        grid_dims += (1,) * (3 - len(grid_dims))
        cutile_builder.set_grid_dim(*grid_dims)

        bound = inspect.signature(self.fn).bind(*args, **kwargs)
        bound.apply_defaults()
        annotations = _parameter_annotations(self.fn)
        for name, arg in bound.arguments.items():
            is_constant, scalar_dtype = annotations.get(name, (False, None))
            if not is_constant:
                bound.arguments[name] = _wrap_argument(arg, scalar_dtype)
        args, kwargs = bound.args, bound.kwargs

        if client_manager is not None:
            for name, arg in bound.arguments.items():
                if isinstance(arg, Array) and not arg.is_contiguous():
                    warnings.warn(
                        f"cuTile argument {name!r} is noncontiguous; visualization "
                        "and trace saving require contiguous arrays",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                trace_arg = (
                    _TraceArray(arg)
                    if isinstance(arg, Array) and _is_low_precision(arg.data.dtype)
                    else arg
                )
                client_manager.arg_callback(name, trace_arg, arg)
            client_manager.grid_callback(grid_dims)

        # Traced execution is already patched by ClientManager.patch_run().
        # Standalone execution uses the same frontend hook without clients.
        scope = (
            get_frontend("cutile").patch_lang(self.fn)
            if client_manager is None
            else None
        )
        try:
            with _execution_scope(self.fn) as execution_fn:
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
