from tilelens.utils.traceback_utils import CODE_KEYS, get_code_key
import numpy as np

try:
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.compiler as ncc
except (
    ModuleNotFoundError
) as exc:  # pragma: no cover - only hit when optional deps missing
    raise ModuleNotFoundError(
        "NeuronX dependencies are missing. Install tilelens[nki] to enable the NKI interpreter."
    ) from exc
import inspect
import os
import textwrap
from ..frontend.nki_transform import transform_code
from ..masked_load_store import masked_load, masked_store, tile_mask, tile_shape


_PACKAGE_DIR = (
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    + os.sep
)


def _value(x):
    """The numpy value of an NDArray operand; other operands (scalars, numpy arrays) as is."""
    return x.data if isinstance(x, NDArray) else x


def _is_hbm(buffer):
    return getattr(buffer, "name", None) in ("hbm", "shared_hbm", "private_hbm")


def _tile_keys(data, keys):
    """NKI treats a 1-D tile (a partition vector) as [P, 1]: `t[:, 0]` indexes it too
    (`t[:, None]` still adds an axis)."""
    if (
        np.ndim(data) == 1
        and len(keys) == 2
        and isinstance(keys[1], (int, np.integer, slice))
    ):
        return keys[:1]
    return keys


def _axis(axis):
    """NKI accepts a list of axes; numpy needs an int or a tuple."""
    return tuple(axis) if isinstance(axis, list) else axis


_UFUNCS = {
    "add": np.add,
    "subtract": np.subtract,
    "multiply": np.multiply,
    "divide": np.divide,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "max": np.maximum,
    "sum": np.add,
    "prod": np.multiply,
    "min": np.minimum,
    "power": np.power,
    "pow": np.power,
    "equal": np.equal,
    "not_equal": np.not_equal,
    "less": np.less,
    "less_equal": np.less_equal,
    "greater": np.greater,
    "greater_equal": np.greater_equal,
    "bitwise_and": np.bitwise_and,
    "bitwise_or": np.bitwise_or,
    "bitwise_xor": np.bitwise_xor,
    "logical_and": np.logical_and,
    "logical_or": np.logical_or,
}


def _ufunc(op):
    """The numpy ufunc of an NKI arithmetic operator: a numpy ufunc itself, or an
    `nl` function such as `nl.add` (patched or not), looked up by name."""
    if isinstance(op, np.ufunc):
        return op
    name = getattr(op, "__name__", None)
    if name in _UFUNCS:
        return _UFUNCS[name]
    raise NotImplementedError(
        f"NKI operator {op!r} is not supported by the interpreter"
    )


def _int(x):
    """A Python int from an index value (an NDArray scalar, a numpy scalar, or an int)."""
    return x if x is None or isinstance(x, int) else int(np.asarray(_value(x)))


def _free_axes(ndim):
    return tuple(range(1, ndim))


def _extreme(dtype, lowest):
    """Identity of a masked max (lowest) or min (highest) reduction over `dtype`."""
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return info.min if lowest else info.max
    return -np.inf if lowest else np.inf


def _shape(shape):
    """A shape with `par_dim(n)` entries (imported directly from NKI) as plain ints."""
    if isinstance(shape, (tuple, list)):
        return tuple(int(getattr(d, "value", d)) for d in shape)
    return int(getattr(shape, "value", shape))


def _partition_aligned(operand, ndim):
    """NKI broadcasts a 1-D tile, which has only the partition dimension, along the free
    dimensions of an N-D tile: as an array it is [P, 1, ...], not [P] aligned to the last axis."""
    value = _value(operand)
    if np.ndim(value) == 1 and ndim >= 2:
        return np.reshape(value, (np.shape(value)[0],) + (1,) * (ndim - 1))
    return value


def _aligned_pair(a, b):
    va, vb = _value(a), _value(b)
    return _partition_aligned(va, np.ndim(vb)), _partition_aligned(vb, np.ndim(va))


def _expand_oob_slices(keys, shape):
    """NKI tiles keep their size when a slice runs past the tensor; numpy truncates the slice.
    If a key of ints and slices has a slice out of range, index with one array per slice
    instead (laid out like the slices' outer product), so the tile keeps its shape and the
    out-of-bounds lanes reach masked_load's checks."""
    keys = keys if isinstance(keys, tuple) else (keys,)
    keys = tuple(
        int(k) if isinstance(k, np.ndarray) and k.ndim == 0 else k for k in keys
    )
    if not all(isinstance(k, (slice, int, np.integer)) for k in keys):
        return keys
    ranges = []
    for dim, k in enumerate(keys):
        if isinstance(k, slice) and dim < len(shape):
            start = 0 if k.start is None else k.start
            stop = shape[dim] if k.stop is None else k.stop
            ranges.append(
                (dim, np.arange(start, stop, 1 if k.step is None else k.step))
            )
    if not any(r.size and (r.min() < 0 or r.max() >= shape[d]) for d, r in ranges):
        return keys
    out = list(keys)
    for i, (dim, r) in enumerate(ranges):
        out[dim] = r.reshape((1,) * i + (-1,) + (1,) * (len(ranges) - i - 1))
    return tuple(out)


def _cast(value, dtype):
    return np.asarray(value) if dtype is None else np.asarray(value).astype(dtype)


def _bool_tile(pred, shape, reverse=False):
    """A predicate or mask operand as a boolean array of `shape` (a 1-D predicate is a
    partition vector)."""
    value = np.asarray(_partition_aligned(pred, len(shape)), dtype=bool)
    value = np.broadcast_to(value, shape)
    return ~value if reverse else value


def _per_partition(value):
    """A [P, ...] array as [P, N]: ISA instructions match operands by partition and
    elements per partition, not by the shape of the free dimensions."""
    value = np.asarray(value)
    return value.reshape(value.shape[0], -1) if value.ndim else value.reshape(1, 1)


def _write_into(dst, value):
    """Write `value` into the tile `dst` in place, as ISA instructions with a destination
    operand do, through to the tensor `dst` was sliced from when the slice is a copy
    (numpy copies on indexing with arrays)."""
    target = dst.data
    value = np.asarray(value)
    if value.shape != target.shape and value.size == target.size:
        value = value.reshape(target.shape)
    target[...] = np.broadcast_to(value, target.shape)
    while getattr(dst, "_parent", None) is not None:
        base, keys = dst._parent
        if not np.may_share_memory(base.data, dst.data):
            base.data[keys] = dst.data.reshape(base.data[keys].shape)
        dst = base


class NDArray:
    def __init__(self, buffer=None, name="", **kwargs):
        self.buffer = buffer
        self.name = name
        self.kwargs = kwargs
        val = None
        if "shape" in kwargs and "dtype" in kwargs:
            shape = kwargs.pop("shape")
            dtype = kwargs.pop("dtype")
            val = np.ndarray(shape, dtype=dtype)
        if "value" in kwargs:
            # Normalize scalars to 0-d numpy arrays so downstream code (e.g. visualizer)
            # can rely on ndarray attributes like `.ctypes`, `.shape`, `.dtype`, `.strides`.
            assert val is None or val.shape == kwargs["value"].shape
            val = kwargs["value"]
        self._data_ptr = None
        self.data = val

    @property
    def shape(self):
        return self.data.shape if self.data is not None else None

    @property
    def dtype(self):
        return self.data.dtype if self.data is not None else None

    def data_ptr(self):
        if self._data_ptr is None:
            self._data_ptr = self.data.ctypes.data
        return self._data_ptr

    def stride(self):
        return self.data.strides

    def element_size(self):
        return self.dtype.itemsize

    def cpu(self):
        return self

    def detach(self):
        return self

    def numpy(self):
        return self.data

    def get_offsets(self):
        """
        Generate offset arrays for each dimension based on shape and stride.
        Given array with shape (A, ..., Z) and strides (a, ..., z), return offsets:
        a * arange(A)[:, None, ..., None] + ... + z * arange(Z)[None, None, ..., :]
        """
        offsets = 0
        for dim_size, stride in zip(self.shape, self.stride()):
            offsets = np.expand_dims(offsets, -1) + np.arange(dim_size) * stride
        return NDArray(value=offsets, name=self.name)

    def __repr__(self):
        return f"NDArray(shape={self.shape}, dtype={self.dtype}, name={self.name})"

    def __getitem__(self, keys):
        """Implement slicing operations for NDArray"""
        if self.data is None:
            raise AttributeError("NDArray has no value to slice")
        if not isinstance(keys, tuple):
            keys = (keys,)

        if (
            len(keys) == 1
            and np.asarray(_value(keys[0])).dtype == bool
            and np.ndim(_value(keys[0])) == self.data.ndim
        ):
            # `t[mask]` attaches a predicate to the tile in NKI (it is not numpy's boolean
            # selection); the interpreter does not model predicates of computations
            return self
        if _is_hbm(self.buffer) and not isinstance(self, HBMAccess):
            # an access pattern, read by the load it is passed to (with that load's mask)
            return HBMAccess(self, tuple(_value(k) for k in keys))

        # Apply the slicing to the underlying numpy array
        new_keys = _tile_keys(
            self.data, [k.data if isinstance(k, NDArray) else k for k in keys]
        )
        sliced_value = self.data[tuple(new_keys)]

        # Create a new NDArray with the sliced data
        sliced = NDArray(value=sliced_value, name=f"{self.name}_slice")
        sliced._parent = (
            self,
            tuple(new_keys),
        )  # for ISA instructions that write into it
        return sliced

    def __setitem__(self, keys, value):
        if not isinstance(keys, tuple):
            keys = (keys,)

        # Apply the slicing to the underlying numpy array
        new_keys = _tile_keys(
            self.data, tuple(k.data if isinstance(k, NDArray) else k for k in keys)
        )
        target_shape = self.data[new_keys].shape
        reduce = value if isinstance(value, LoopReduced) else None
        value = np.asarray(value.data if isinstance(value, NDArray) else value)
        if value.shape != target_shape and value.size == int(np.prod(target_shape)):
            value = value.reshape(target_shape)  # e.g. a [P, 1] result into a [P] slot
        if reduce is not None:
            # `dst[...] = nl.loop_reduce(x, op, ...)`: the destination is op's other input
            current = self.data[new_keys]
            value = reduce.op(current, np.broadcast_to(value, target_shape))
            if reduce.mask is not None:
                value = np.where(reduce.mask.reshape(target_shape), value, current)
        self.data[new_keys] = value

        return self

    def _binary_op(self, other, op_func, op_name, op_symbol):
        if isinstance(other, NDArray):
            return NDArray(
                value=op_func(*_aligned_pair(self.data, other.data)),
                name=f"{self.name}_{op_name}_{other.name}",
            )
        elif np.isscalar(other):
            return NDArray(
                value=op_func(self.data, other), name=f"{self.name}_{op_name}_scalar"
            )
        raise TypeError(
            f"Unsupported operand type(s) for {op_symbol}: 'NDArray' and '{type(other).__name__}'"
        )

    def _rbinary_op(self, other, op_func, op_name, op_symbol):
        if isinstance(other, NDArray):
            return NDArray(
                value=op_func(*_aligned_pair(other.data, self.data)),
                name=f"{other.name}_{op_name}_{self.name}",
            )
        elif np.isscalar(other):
            return NDArray(
                value=op_func(other, self.data), name=f"scalar_{op_name}_{self.name}"
            )
        raise TypeError(
            f"Unsupported operand type(s) for {op_symbol}: '{type(other).__name__}' and 'NDArray'"
        )

    # Define operator +/-/*//
    def __add__(self, other):
        return self._binary_op(other, lambda a, b: a + b, "add", "+")

    def __radd__(self, other):
        return self._rbinary_op(other, lambda a, b: a + b, "add", "+")

    def __sub__(self, other):
        return self._binary_op(other, lambda a, b: a - b, "sub", "-")

    def __rsub__(self, other):
        return self._rbinary_op(other, lambda a, b: a - b, "sub", "-")

    def __mul__(self, other):
        return self._binary_op(other, lambda a, b: a * b, "mul", "*")

    def __rmul__(self, other):
        return self._rbinary_op(other, lambda a, b: a * b, "mul", "*")

    def __truediv__(self, other):
        return self._binary_op(other, lambda a, b: a / b, "div", "/")

    def __rtruediv__(self, other):
        return self._rbinary_op(other, lambda a, b: a / b, "div", "/")

    def __lt__(self, other):
        return self._binary_op(other, lambda a, b: a < b, "lt", "<")

    def __gt__(self, other):
        return self._binary_op(other, lambda a, b: a > b, "gt", ">")

    def __le__(self, other):
        return self._binary_op(other, lambda a, b: a <= b, "le", "<=")

    def __ge__(self, other):
        return self._binary_op(other, lambda a, b: a >= b, "ge", ">=")

    def __and__(self, other):
        return self._binary_op(other, lambda a, b: a & b, "and", "&")

    def __or__(self, other):
        return self._binary_op(other, lambda a, b: a | b, "or", "|")

    def reshape(self, *args, shape=None, **kwargs):
        # `t.reshape(shape)`, `t.reshape(*shape)`, or `t.reshape(shape=...)`; a view of the
        # same memory, as reshaping an HBM tensor is in NKI.
        if shape is None:
            shape = args[0] if len(args) == 1 else args
        return NDArray(
            buffer=self.buffer,
            value=self.data.reshape(_shape(shape)),
            name=f"{self.name}_reshape",
            **kwargs,
        )

    def view(self, dtype):
        # the same memory reinterpreted as another dtype (e.g. fp32 values as 4 fp8 bytes)
        return NDArray(
            buffer=self.buffer, value=self.data.view(dtype), name=f"{self.name}_view"
        )

    def broadcast_to(self, *args, **kwargs):
        return NDArray(
            value=np.broadcast_to(self.data, *args),
            name=f"{self.name}_broadcast_to",
            **kwargs,
        )


class HBMAccess(NDArray):
    """`t[keys]` of an HBM tensor outside a load or store, as in `a = t[i, j]` followed by
    `nl.load(a, mask=m)`: in NKI an access pattern that the load transfers, applying its
    mask. Load and store take `tensor` and `keys` from it, so clients see the access to
    `tensor`; its own value is read only when used otherwise (an unmasked read)."""

    def __init__(self, tensor, keys):
        self._value = None
        super().__init__(buffer=tensor.buffer, name=f"{tensor.name}_access")
        self.tensor, self.keys = tensor, keys
        self._parent = (tensor, nki_builder._convert_keys_to_numpy(keys))

    @property
    def data(self):
        if self._value is None and getattr(self, "tensor", None) is not None:
            self._value = self.tensor.data[self._parent[1]]
        return self._value

    @data.setter
    def data(self, value):
        self._value = value


class LoopReduced(NDArray):
    """The result of `nl.loop_reduce(x, op, ...)`. Assigned into a tensor slice, it combines
    with the slice's current value: `dst[...] = op(dst[...], x)`, as in NKI's simulator.
    Bound to a plain name it is just `x`."""

    def __init__(self, value, op, mask=None, **kwargs):
        super().__init__(value=value, **kwargs)
        self.op = op
        self.mask = (
            None
            if mask is None
            else np.broadcast_to(np.asarray(mask, bool), value.shape)
        )


class GridIndices(NDArray):
    """The result of `nl.mgrid[...]`: unpacks into one index array per axis, and, as in NKI,
    also names them `.p` (the partition axis), `.x`, `.y`, and `.z` (the free axes)."""

    def _axis(self, i):
        return NDArray(value=self.data[i], name=f"{self.name}_{i}")

    def __iter__(self):
        return (self._axis(i) for i in range(self.data.shape[0]))

    p = property(lambda self: self._axis(0))
    x = property(lambda self: self._axis(1))
    y = property(lambda self: self._axis(2))
    z = property(lambda self: self._axis(3))


class MGrid:
    def __getitem__(self, keys):
        keys = keys if isinstance(keys, tuple) else (keys,)
        keys = tuple(
            slice(_int(k.start), _int(k.stop), _int(k.step))
            if isinstance(k, slice)
            else k
            for k in keys
        )
        return GridIndices(value=np.mgrid[keys], name="mgrid")


class Builder:
    def __init__(self, grid_dims=None):
        # TODO: infinite grid dims for NKI
        self.grid_dims = grid_dims if grid_dims is not None else (1, 1, 1)
        self.grid_ndim = 1
        self.grid_x = None
        self.grid_y = None
        self.grid_z = None
        self.fn = None
        self.shared_hbm_arrays = {}
        self.rng = np.random.default_rng(0)  # nl.dropout; reseeded by nl.random_seed
        self.accumulator = None  # the Vector Engine's per-partition reduce registers

    def set_grid_dim(self, *grid_dims):
        self.grid_dims = grid_dims

    def set_grid_idx(self, x, y, z):
        self.grid_x = x
        self.grid_y = y
        self.grid_z = z

    def ndarray(self, shape, dtype, *, buffer=None, name=None, **kwargs):
        shape = _shape(shape)
        if buffer == nl.shared_hbm:
            if name is None:
                # file name + function name + line number of the allocation in the kernel,
                # so every program of the grid gets the same array. Skip TileLens's own
                # frames (the op patching wraps this method), or every shared HBM tensor
                # of a kernel would share one name, and one array.
                frame = inspect.currentframe().f_back
                while frame is not None and frame.f_code.co_filename.startswith(
                    _PACKAGE_DIR
                ):
                    frame = frame.f_back
                file_name = frame.f_code.co_filename
                function_name = frame.f_code.co_name
                line_number = frame.f_lineno
                name = f"{file_name}_{function_name}_{line_number}"
            if name in self.shared_hbm_arrays:
                # Return the existing shared HBM array
                ret = self.shared_hbm_arrays[name]
            else:
                # Create a new shared HBM array and store it
                ret = NDArray(
                    buffer=buffer, name=name, shape=shape, dtype=dtype, **kwargs
                )
                self.shared_hbm_arrays[name] = ret
        else:
            ret = NDArray(buffer=buffer, name=name, shape=shape, dtype=dtype, **kwargs)
        return ret

    def zeros(self, shape, dtype, *, buffer=None, name=None, **kwargs):
        shape = _shape(shape)
        value = np.zeros(shape, dtype=dtype)
        return self.ndarray(
            shape, dtype, buffer=buffer, name=name, value=value, **kwargs
        )

    def arange(self, *args):
        return NDArray(value=np.arange(*args))

    def program_id(self, axis: int):
        if axis == 0:
            return self.grid_x
        elif axis == 1:
            return self.grid_y
        elif axis == 2:
            return self.grid_z
        else:
            raise ValueError(f"Invalid axis: {axis}. Must be 0, 1, or 2.")

    def _convert_keys_to_numpy(self, keys):
        """Convert any NDArrays in keys to numpy arrays."""
        if isinstance(keys, (tuple, list)):
            return tuple(self._convert_keys_to_numpy(k) for k in keys)
        elif isinstance(keys, NDArray):
            return keys.data
        elif isinstance(keys, slice):  # e.g. nl.ds(start, size) with a computed start
            return slice(_int(keys.start), _int(keys.stop), _int(keys.step))
        else:
            return keys

    def load(self, src: NDArray, keys=None, *, mask=None, **kwargs):
        """Load array elements with masking for out-of-bounds errors."""
        src, keys = access(src, keys)
        ndarray = src.data
        mask_value = _value(mask)
        dtype = kwargs.pop("dtype", None)

        # Convert any NDArrays in keys to numpy arrays
        numpy_keys = self._convert_keys_to_numpy(keys)

        # Call the actual masked_load function
        numpy_keys = _expand_oob_slices(numpy_keys, ndarray.shape)
        mask_value = tile_mask(ndarray, numpy_keys, mask_value)
        result = masked_load(ndarray, numpy_keys, mask=mask_value)

        # Convert result back to NDArray
        return NDArray(
            value=_cast(result, dtype), name=f"{src.name}_masked_load", **kwargs
        )

    def store(
        self,
        dst: NDArray,
        keys=None,
        value: NDArray | None = None,
        *,
        mask=None,
        **kwargs,
    ):
        """Store array elements with masking for out-of-bounds errors."""
        dst, keys = access(dst, keys)
        ndarray = dst.data
        mask_value = _value(mask)

        # Convert any NDArrays in keys to numpy arrays
        numpy_keys = self._convert_keys_to_numpy(keys)

        # NKI broadcasts the stored value and the mask to the tile shape
        numpy_keys = _expand_oob_slices(numpy_keys, ndarray.shape)
        tile_shape_ = tile_shape(ndarray, numpy_keys)
        value_array = np.asarray(_value(value))
        same_size = value_array.size == int(np.prod(tile_shape_))
        if value_array.shape != tile_shape_ and same_size:
            # e.g. a [P, 1] reduction stored into a 1-D slice
            value_array = value_array.reshape(tile_shape_)
        value_array = np.broadcast_to(value_array, tile_shape_).astype(ndarray.dtype)
        mask_value = tile_mask(ndarray, numpy_keys, mask_value)

        # Call the actual masked_store function
        masked_store(ndarray, numpy_keys, value_array, mask=mask_value)

        return dst

    def load_transpose2d(
        self, src: NDArray, keys=None, *, mask=None, dtype=None, **kwargs
    ):
        loaded = self.load(src, keys, mask=mask, **kwargs)
        value = loaded.data.astype(dtype) if dtype is not None else loaded.data
        return NDArray(value=value.T, name=f"{src.name}_load_transpose2d", **kwargs)

    def _unary_op(
        self, x: NDArray, np_func, op_name, *, dtype=None, mask=None, **kwargs
    ):
        return NDArray(
            value=_cast(np_func(x.data), dtype), name=f"{x.name}_{op_name}", **kwargs
        )

    def _binary_op(self, x, y, np_func, op_name, *, dtype=None, mask=None, **kwargs):
        return NDArray(
            value=_cast(np_func(*_aligned_pair(x, y)), dtype),
            name=f"{getattr(x, 'name', 'scalar')}_{op_name}_{getattr(y, 'name', 'scalar')}",
            **kwargs,
        )

    def _reduce(
        self,
        x,
        axis,
        np_func,
        op_name,
        *,
        dtype=None,
        mask=None,
        keepdims=False,
        initial=None,
        **kwargs,
    ):
        reduce_kwargs = {}
        if mask is not None:
            reduce_kwargs["where"] = np.asarray(_value(mask), dtype=bool)
            if initial is not None:
                reduce_kwargs["initial"] = initial
        value = np_func(x.data, axis=_axis(axis), keepdims=keepdims, **reduce_kwargs)
        if not keepdims and np.ndim(value) == 1 and x.data.ndim >= 2:
            # NKI keeps a free dimension: reducing all free axes of a [P, ...] tile gives [P, 1]
            value = value[:, None]
        return NDArray(value=_cast(value, dtype), name=f"{x.name}_{op_name}", **kwargs)

    # Elementwise operator implementations
    def exp(self, x: NDArray, **kwargs):
        return self._unary_op(x, np.exp, "exp", **kwargs)

    def relu(self, x: NDArray, **kwargs):
        return self._unary_op(x, lambda v: np.maximum(v, 0), "relu", **kwargs)

    def sigmoid(self, x: NDArray, **kwargs):
        return self._unary_op(x, lambda v: 1 / (1 + np.exp(-v)), "sigmoid", **kwargs)

    def tanh(self, x: NDArray, **kwargs):
        return self._unary_op(x, np.tanh, "tanh", **kwargs)

    def silu(self, x: NDArray, **kwargs):
        # SiLU(x) = x * sigmoid(x)
        sigmoid_x = 1 / (1 + np.exp(-x.data))
        return NDArray(value=x.data * sigmoid_x, name=f"{x.name}_silu", **kwargs)

    def gelu(self, x: NDArray, **kwargs):
        # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_pi = np.sqrt(2 / np.pi)
        inner = sqrt_2_pi * (x.data + 0.044715 * np.power(x.data, 3))
        return NDArray(
            value=0.5 * x.data * (1 + np.tanh(inner)), name=f"{x.name}_gelu", **kwargs
        )

    def sqrt(self, x: NDArray, **kwargs):
        return self._unary_op(x, np.sqrt, "sqrt", **kwargs)

    def abs(self, x: NDArray, **kwargs):
        return self._unary_op(x, np.abs, "abs", **kwargs)

    def log(self, x: NDArray, **kwargs):
        return self._unary_op(x, np.log, "log", **kwargs)

    def pow(self, x: NDArray, exponent, **kwargs):
        if isinstance(exponent, NDArray):
            return NDArray(
                value=np.power(x.data, exponent.data),
                name=f"{x.name}_pow_{exponent.name}",
                **kwargs,
            )
        elif np.isscalar(exponent):
            return NDArray(
                value=np.power(x.data, exponent),
                name=f"{x.name}_pow_{exponent}",
                **kwargs,
            )
        else:
            raise TypeError(f"Unsupported exponent type: {type(exponent)}")

    def reciprocal(self, x: NDArray, **kwargs):
        return self._unary_op(x, lambda v: 1 / v, "reciprocal", **kwargs)

    def matmul(self, x: NDArray, y: NDArray, transpose_x=False, mask=None, **kwargs):
        x_value = x.data
        if transpose_x:
            x_value = x_value.T
        y_value = y.data
        return NDArray(
            value=(x_value @ y_value), name=f"{x.name}_{y.name}_matmul", **kwargs
        )

    def copy(self, x: NDArray, **kwargs):
        return self._unary_op(x, np.copy, "copy", **kwargs)

    def sum(self, x: NDArray, *args, **kwargs):
        axis = args[0] if args else kwargs.pop("axis", None)
        kwargs.setdefault("keepdims", kwargs.pop("keep_dims", False))
        return self._reduce(x, axis, np.sum, "sum", **kwargs)

    def max(self, x: NDArray, axis, **kwargs):
        lowest = _extreme(x.data.dtype, lowest=True)
        return self._reduce(x, axis, np.max, "max", initial=lowest, **kwargs)

    def min(self, x: NDArray, axis, **kwargs):
        highest = _extreme(x.data.dtype, lowest=False)
        return self._reduce(x, axis, np.min, "min", initial=highest, **kwargs)

    def mean(self, x: NDArray, axis, **kwargs):
        return self._reduce(x, axis, np.mean, "mean", **kwargs)

    def add(self, x, y, **kwargs):
        return self._binary_op(x, y, np.add, "add", **kwargs)

    def subtract(self, x, y, **kwargs):
        return self._binary_op(x, y, np.subtract, "subtract", **kwargs)

    def divide(self, x, y, **kwargs):
        return self._binary_op(x, y, np.divide, "divide", **kwargs)

    def maximum(self, x, y, **kwargs):
        return self._binary_op(x, y, np.maximum, "maximum", **kwargs)

    def minimum(self, x, y, **kwargs):
        return self._binary_op(x, y, np.minimum, "minimum", **kwargs)

    def static_cast(self, x, dtype):
        return NDArray(
            value=_cast(_value(x), dtype), name=f"{getattr(x, 'name', 'scalar')}_cast"
        )

    def par_dim(self, value):
        # Marks a partition dimension for the compiler; the size is all the simulator needs.
        return value

    def square(self, x: NDArray, **kwargs):
        return self._unary_op(x, np.square, "square", **kwargs)

    def rsqrt(self, x: NDArray, **kwargs):
        return self._unary_op(x, lambda v: 1 / np.sqrt(v), "rsqrt", **kwargs)

    def multiply(self, x, y, **kwargs):
        return self._binary_op(x, y, np.multiply, "multiply", **kwargs)

    # ---- nl: tensor creation, slicing, grid ------------------------------
    def full(self, shape, fill_value, dtype, *, buffer=None, name=None, **kwargs):
        shape = _shape(shape)
        value = np.full(shape, fill_value, dtype=dtype)
        return self.ndarray(
            shape, dtype, buffer=buffer, name=name, value=value, **kwargs
        )

    def ones(self, shape, dtype, *, buffer=None, name=None, **kwargs):
        return self.full(shape, 1, dtype, buffer=buffer, name=name, **kwargs)

    def shared_constant(self, constant, dtype=None, **kwargs):
        value = (
            np.asarray(constant) if dtype is None else np.asarray(constant, dtype=dtype)
        )
        return NDArray(value=value, buffer=nl.shared_hbm, name="shared_constant")

    def transpose(self, x, *, dtype=None, mask=None, **kwargs):
        return NDArray(
            value=_cast(np.transpose(x.data), dtype), name=f"{x.name}_transpose"
        )

    def ds(self, start, size):
        # A slice that keeps a computed start as is, so clients can see where it came from.
        return slice(start, start + size)

    def program_ndim(self):
        return self.grid_ndim

    def shared_identity_matrix(self, n, dtype=np.uint8, **kwargs):
        return NDArray(
            value=np.eye(n, dtype=dtype),
            buffer=nl.shared_hbm,
            name="shared_identity_matrix",
        )

    def floor(self, x, **kwargs):
        return self._unary_op(x, np.floor, "floor", **kwargs)

    def mod(self, x, y, **kwargs):
        return self._binary_op(x, y, np.mod, "mod", **kwargs)

    def num_programs(self, axes=None):
        dims = [d for d in self.grid_dims]
        if axes is None:
            return int(np.prod(dims))
        if isinstance(axes, (list, tuple)):
            return int(np.prod([dims[_int(a)] for a in axes]))
        return dims[_int(axes)]

    # ---- nisa: instructions -------------------------------------------
    def _activate(self, op, value):
        if isinstance(op, np.ufunc):
            return op(value)
        name = getattr(op, "__name__", None)
        impl = getattr(self, name, None) if name else None
        if impl is None or not callable(impl):
            raise NotImplementedError(
                f"NKI activation {op!r} is not supported by the interpreter"
            )
        return impl(NDArray(value=np.asarray(value))).data

    def _reduce_into(self, reduce_op, value, reduce_res):
        # Reduce along the free axes into `reduce_res`, a [P, 1] tile.
        if reduce_op is None or reduce_res is None:
            return
        reduced = _ufunc(reduce_op).reduce(
            value, axis=_free_axes(value.ndim), keepdims=True
        )
        _write_into(
            reduce_res,
            np.broadcast_to(
                reduced.reshape(
                    reduce_res.data.shape[:1] + (1,) * (reduce_res.data.ndim - 1)
                ),
                reduce_res.data.shape,
            ),
        )

    def nc_matmul(self, stationary, moving, *, mask=None, **kwargs):
        # stationary.T @ moving on the Tensor Engine, accumulated in fp32 (PSUM).
        value = np.matmul(
            stationary.data.astype(np.float32).T, moving.data.astype(np.float32)
        )
        return NDArray(value=value, name=f"{stationary.name}_{moving.name}_nc_matmul")

    def nc_transpose(self, data, *, mask=None, dtype=None, **kwargs):
        return NDArray(
            value=_cast(np.transpose(data.data), dtype),
            name=f"{data.name}_nc_transpose",
        )

    def activation(
        self,
        op,
        data,
        *,
        bias=None,
        scale=1.0,
        reduce_op=None,
        reduce_res=None,
        mask=None,
        dtype=None,
        **kwargs,
    ):
        ndim = data.data.ndim
        value = data.data.astype(np.float32) * np.asarray(
            _partition_aligned(scale, ndim), dtype=np.float32
        )
        if bias is not None:
            value, bias_value = _aligned_pair(value, bias)  # either may be a [P] vector
            value = value + bias_value
        value = self._activate(op, value)
        self._reduce_into(reduce_op, value, reduce_res)
        return NDArray(
            value=_cast(value, dtype if dtype is not None else data.data.dtype),
            name=f"{data.name}_activation",
        )

    def activation_reduce(
        self,
        op,
        data,
        *,
        reduce_op,
        reduce_res,
        bias=None,
        scale=1.0,
        mask=None,
        dtype=None,
        **kwargs,
    ):
        return self.activation(
            op,
            data,
            bias=bias,
            scale=scale,
            reduce_op=reduce_op,
            reduce_res=reduce_res,
            mask=mask,
            dtype=dtype,
        )

    def tensor_tensor(self, data1, data2, op, *, dtype=None, mask=None, **kwargs):
        value = _ufunc(op)(*_aligned_pair(data1, data2))
        return NDArray(value=_cast(value, dtype), name="tensor_tensor")

    def tensor_scalar(
        self,
        data,
        op0,
        operand0,
        reverse0=False,
        op1=None,
        operand1=None,
        reverse1=False,
        *,
        dtype=None,
        mask=None,
        **kwargs,
    ):
        def apply(op, x, operand, reverse):
            if getattr(op, "__name__", None) in ("abs", "absolute"):
                return np.abs(x)  # NKI requires its operand to be zero
            u = _ufunc(op)
            operand = _partition_aligned(operand, np.ndim(x))
            return u(operand, x) if reverse else u(x, operand)

        value = apply(op0, data.data, operand0, reverse0)
        if op1 is not None:
            value = apply(op1, value, operand1, reverse1)
        return NDArray(
            value=_cast(value, dtype if dtype is not None else data.data.dtype),
            name=f"{data.name}_tensor_scalar",
        )

    def tensor_scalar_reduce(
        self,
        *,
        data,
        op0,
        operand0,
        reduce_op,
        reduce_res,
        reverse0=False,
        dtype=None,
        mask=None,
        **kwargs,
    ):
        out = self.tensor_scalar(data, op0, operand0, reverse0, dtype=dtype)
        self._reduce_into(reduce_op, out.data, reduce_res)
        return out

    def tensor_reduce(
        self,
        op,
        data,
        axis,
        *,
        mask=None,
        dtype=None,
        negate=False,
        keepdims=False,
        **kwargs,
    ):
        value = _ufunc(op).reduce(data.data, axis=_axis(axis), keepdims=keepdims)
        if not keepdims and np.ndim(value) == 1 and data.data.ndim >= 2:
            value = value[:, None]  # NKI keeps a free dimension, as in nl reductions
        if negate:
            value = -value
        return NDArray(value=_cast(value, dtype), name=f"{data.name}_tensor_reduce")

    def tensor_copy(self, src, *, mask=None, dtype=None, **kwargs):
        return NDArray(
            value=_cast(np.copy(src.data), dtype), name=f"{src.name}_tensor_copy"
        )

    def memset(self, shape, value, dtype, *, mask=None, **kwargs):
        return NDArray(value=np.full(_shape(shape), value, dtype=dtype), name="memset")

    def affine_select(
        self, pred, on_true_tile, on_false_value, *, mask=None, dtype=None, **kwargs
    ):
        value = np.where(
            np.asarray(_value(pred), dtype=bool), _value(on_true_tile), on_false_value
        )
        return NDArray(
            value=_cast(value, dtype if dtype is not None else on_true_tile.data.dtype),
            name="affine_select",
        )

    def dma_copy(
        self,
        *,
        dst,
        src,
        dst_keys=None,
        src_keys=None,
        mask=None,
        dst_rmw_op=None,
        **kwargs,
    ):
        # A DMA transfer reads `src[src_keys]` and writes `dst[dst_keys]`; both go through
        # load/store, so analysis clients see them as a load and a store.
        value = self.load(src, src_keys, mask=mask)
        if dst_rmw_op is not None:
            current = self.load(dst, dst_keys, mask=mask)
            value = NDArray(value=_ufunc(dst_rmw_op)(current.data, value.data))
        self.store(dst, dst_keys, value, mask=mask)

    def scalar_tensor_tensor(
        self,
        *,
        data,
        op0,
        operand0,
        op1,
        operand1,
        reverse0=False,
        reverse1=False,
        dtype=None,
        mask=None,
        **kwargs,
    ):
        # (data <op0> operand0) <op1> operand1
        first = self.tensor_scalar(data, op0, operand0, reverse0)
        u = _ufunc(op1)
        a, b = _aligned_pair(first.data, operand1)
        value = u(b, a) if reverse1 else u(a, b)
        return NDArray(
            value=_cast(value, dtype if dtype is not None else data.data.dtype),
            name=f"{data.name}_scalar_tensor_tensor",
        )

    def nisa_reciprocal(self, data, *, dtype=None, mask=None, **kwargs):
        return NDArray(
            value=_cast(
                1 / np.asarray(data.data, dtype=np.float32), dtype or data.data.dtype
            ),
            name=f"{data.name}_reciprocal",
        )

    def tensor_tensor_scan(
        self,
        data0,
        data1,
        initial,
        op0,
        op1,
        reverse0=False,
        reverse1=False,
        *,
        dtype=None,
        mask=None,
        **kwargs,
    ):
        # Per partition, along the free axis:
        #   result[:, i] = op1(op0(data0[:, i], result[:, i-1]), data1[:, i]), result[:, -1] = initial
        u0, u1 = _ufunc(op0), _ufunc(op1)
        d0 = data0.data.reshape(data0.data.shape[0], -1).astype(np.float32)
        d1 = data1.data.reshape(data1.data.shape[0], -1).astype(np.float32)
        prev = np.broadcast_to(
            np.asarray(_partition_aligned(initial, 2), dtype=np.float32),
            (d0.shape[0], 1),
        )[:, 0]
        result = np.empty_like(d0)
        for i in range(d0.shape[1]):
            first = u0(prev, d0[:, i]) if reverse0 else u0(d0[:, i], prev)
            prev = u1(d1[:, i], first) if reverse1 else u1(first, d1[:, i])
            result[:, i] = prev
        out_dtype = dtype if dtype is not None else data0.data.dtype
        return NDArray(
            value=_cast(result.reshape(data0.data.shape), out_dtype),
            name=f"{data0.name}_scan",
        )

    def iota(self, expr, dtype, *, mask=None, **kwargs):
        # A constant tile built from an index expression.
        return NDArray(value=_cast(_value(expr), dtype), name="iota")

    def dma_transpose(
        self, src, *, src_keys=None, axes=None, mask=None, dtype=None, **kwargs
    ):
        # A DMA read of `src[src_keys]` (seen by clients as a load), transposed.
        value = self.load(src, src_keys, mask=mask).data
        value = np.transpose(value) if axes is None else np.transpose(value, axes)
        return NDArray(value=_cast(value, dtype), name=f"{src.name}_dma_transpose")

    def get_nc_version(self):
        # The interpreter models a Trn1 NeuronCore (gen2), or a Trn2 one (gen3) when NKI's
        # platform target is overridden to trn2, as for NKI's own simulator.
        if os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE", "").startswith("trn2"):
            return nisa.nc_version.gen3
        return nisa.nc_version.gen2

    # ---- nl: selection, loop and program reductions, random ---------------
    def where(self, condition, x, y, *, dtype=None, mask=None, **kwargs):
        xv, yv, cv = _value(x), _value(y), _value(condition)
        ndim = max(np.ndim(xv), np.ndim(yv), np.ndim(cv))
        value = np.where(
            np.asarray(_partition_aligned(cv, ndim), dtype=bool),
            _partition_aligned(xv, ndim),
            _partition_aligned(yv, ndim),
        )
        return NDArray(value=_cast(value, dtype), name="where")

    def equal(self, x, y, *, dtype=bool, mask=None, **kwargs):
        return self._binary_op(x, y, np.equal, "equal", dtype=dtype)

    def loop_reduce(self, x, op, loop_indices, *, dtype=None, mask=None, **kwargs):
        value = _cast(_value(x), dtype)
        mask_value = None if mask is None else _value(mask)
        return LoopReduced(value, _ufunc(op), mask_value, name="loop_reduce")

    def all_reduce(self, x, op, program_axes, *, dtype=None, mask=None, **kwargs):
        # A reduction across the programs of the grid. The interpreter runs the programs one
        # after another, so it can only reduce over a grid of one program.
        if int(np.prod(self.grid_dims)) != 1:
            raise NotImplementedError(
                "nl.all_reduce across several programs is not supported by the interpreter"
            )
        return NDArray(value=_cast(np.copy(_value(x)), dtype), name="all_reduce")

    def random_seed(self, seed, *, mask=None, **kwargs):
        self.rng = np.random.default_rng(int(np.asarray(_value(seed)).ravel()[0]))

    def dropout(self, x, rate, *, dtype=None, mask=None, **kwargs):
        # Zeroes each element with probability `rate` (no rescaling, as in NKI). Like NKI's
        # simulator, the random numbers do not match the hardware's.
        rate = _partition_aligned(_value(rate), x.data.ndim)
        keep = self.rng.random(x.data.shape) >= np.asarray(rate, dtype=np.float32)
        return NDArray(
            value=_cast(np.where(keep, x.data, 0).astype(x.data.dtype), dtype),
            name=f"{x.name}_dropout",
        )

    # ---- nisa: statistics, partition reduction, predicated copies -----------
    def tensor_partition_reduce(self, op, data, *, mask=None, dtype=None, **kwargs):
        value = _ufunc(op).reduce(data.data, axis=0, keepdims=True)
        return NDArray(
            value=_cast(value, dtype if dtype is not None else data.data.dtype),
            name=f"{data.name}_partition_reduce",
        )

    def bn_stats(self, data, *, mask=None, dtype=None, **kwargs):
        # Per partition: (count, mean, variance * count) of the elements at even positions,
        # then of those at odd positions, over the elements the mask keeps; in fp32.
        values = _per_partition(data.data).astype(np.float32)
        keep = (
            np.ones(values.shape, dtype=bool)
            if mask is None
            else _per_partition(_bool_tile(_value(mask), data.data.shape))
        )
        stats = []
        for parity in (0, 1):
            v, k = values[:, parity::2], keep[:, parity::2]
            count = k.sum(axis=1, keepdims=True).astype(np.float32)
            mean = np.where(k, v, 0).sum(axis=1, keepdims=True) / np.maximum(count, 1)
            m2 = np.square(np.where(k, v - mean, 0)).sum(axis=1, keepdims=True)
            stats += [count, mean, m2]
        return NDArray(
            value=_cast(
                np.concatenate(stats, axis=1),
                dtype if dtype is not None else data.data.dtype,
            ),
            name=f"{data.name}_bn_stats",
        )

    def bn_aggr(self, data, *, mask=None, dtype=None, **kwargs):
        # Combines (count, mean, variance * count) triples per partition into (mean, variance).
        triples = _per_partition(data.data).astype(np.float32)
        triples = triples.reshape(triples.shape[0], -1, 3)
        count, mean, m2 = triples[..., 0], triples[..., 1], triples[..., 2]
        total = count.sum(axis=1, keepdims=True)
        overall = (count * mean).sum(axis=1, keepdims=True) / np.maximum(total, 1)
        m2_total = m2.sum(axis=1, keepdims=True) + (
            count * np.square(mean - overall)
        ).sum(axis=1, keepdims=True)
        value = np.concatenate([overall, m2_total / np.maximum(total, 1)], axis=1)
        return NDArray(
            value=_cast(value, dtype if dtype is not None else data.data.dtype),
            name=f"{data.name}_bn_aggr",
        )

    def select_reduce(
        self,
        *,
        dst,
        predicate,
        on_true,
        on_false,
        reduce_res=None,
        reduce_cmd=None,
        reduce_op=None,
        reverse_pred=False,
        mask=None,
        dtype=None,
        **kwargs,
    ):
        # dst = where(predicate, on_true, on_false); with a reduce command, the per-partition
        # max of the result also goes through the accumulator registers into `reduce_res`.
        true_value = _per_partition(_value(on_true))
        pred = _per_partition(_bool_tile(_value(predicate), np.shape(_value(on_true))))
        if reverse_pred:
            pred = ~pred
        false_value = _partition_aligned(_value(on_false), 2)
        result = np.where(pred, true_value, false_value)
        _write_into(dst, _cast(result, dtype if dtype is not None else dst.data.dtype))
        command = getattr(reduce_cmd, "name", "idle")
        if command == "idle":
            return
        partial = np.max(result.astype(np.float32), axis=1)
        if command == "reset" or self.accumulator is None:
            self.accumulator = np.full(partial.shape, -np.inf, dtype=np.float32)
        if command == "reset_reduce":
            self.accumulator = partial
        elif command in ("reduce", "load_reduce"):
            self.accumulator = np.maximum(self.accumulator, partial)
        if reduce_res is not None:
            _write_into(reduce_res, self.accumulator.astype(reduce_res.data.dtype))

    def tensor_copy_predicated(
        self,
        *,
        src,
        dst,
        predicate,
        reverse_pred=False,
        mask=None,
        dtype=None,
        **kwargs,
    ):
        # dst = where(predicate, src, dst), matching operands by partition.
        current = _per_partition(dst.data)
        pred = _per_partition(
            _bool_tile(_value(predicate), np.shape(_value(predicate)))
        )
        pred = np.broadcast_to(pred, current.shape)
        if reverse_pred:
            pred = ~pred
        src_value = _value(src)
        source = (
            _per_partition(src_value)
            if np.ndim(src_value) == dst.data.ndim or np.size(src_value) == current.size
            else _partition_aligned(src_value, 2)
        )
        _write_into(dst, np.where(pred, source, current).astype(dst.data.dtype))

    def tiled_offloaded_memcpy(
        self, *, dst, src, dst_keys=None, src_keys=None, dtype=None, **kwargs
    ):
        # A DMA copy from one HBM tile to another, seen by clients as a load and a store.
        self.store(dst, dst_keys, self.load(src, src_keys))

    def tiled_offloaded_fma(
        self,
        *src,
        dst,
        scales,
        src_keys=None,
        dst_keys=None,
        dtype=None,
        mask=None,
        **kwargs,
    ):
        # dst = src[0] * scales[0] + src[1] * scales[1] + ..., by the DMA engine, in HBM.
        keys = src_keys if src_keys is not None else [None] * len(src)
        acc = 0
        for tensor, key, scale in zip(src, keys, scales):
            acc = acc + self.load(tensor, key).data.astype(np.float32) * scale
        self.store(dst, dst_keys, NDArray(value=np.asarray(acc)))

    def range(self, *args, **kwargs):
        # affine_range, sequential_range, and static_range all iterate like range()
        return range(*args)

    def multi_buffer(self, *args, **kwargs):
        # a scheduling directive for a range (`directives=`), which ranges ignore
        return None


nki_builder = Builder()


def nki_patch_lang(scope=None):
    def _set_attr(obj, name, value):
        if scope is None:
            setattr(obj, name, value)
        else:
            scope.set_attr(obj, name, value)

    _set_attr(nl, "ndarray", nki_builder.ndarray)
    _set_attr(nl, "program_id", nki_builder.program_id)
    _set_attr(nl, "arange", nki_builder.arange)

    _set_attr(nl, "load", nki_builder.load)
    _set_attr(nl, "store", nki_builder.store)
    # see https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/nki.language.html

    # TODO: implement
    # matmul-specific
    # nl.shared_hbm
    # nl.psum
    _set_attr(nl, "affine_range", nki_builder.range)
    _set_attr(nl, "sequential_range", nki_builder.range)
    _set_attr(nl, "static_range", nki_builder.range)
    _set_attr(nl, "par_dim", nki_builder.par_dim)
    _set_attr(nl, "zeros", nki_builder.zeros)
    _set_attr(nl, "mgrid", MGrid())
    _set_attr(nl, "matmul", nki_builder.matmul)
    _set_attr(nl, "copy", nki_builder.copy)
    _set_attr(nl, "sum", nki_builder.sum)
    _set_attr(nl, "square", nki_builder.square)
    _set_attr(nl, "rsqrt", nki_builder.rsqrt)
    _set_attr(nl, "multiply", nki_builder.multiply)
    for name in (
        "full",
        "ones",
        "shared_constant",
        "shared_identity_matrix",
        "transpose",
        "ds",
        "num_programs",
        "program_ndim",
        "floor",
        "mod",
        "where",
        "equal",
        "loop_reduce",
        "all_reduce",
        "random_seed",
        "dropout",
    ):
        _set_attr(nl, name, getattr(nki_builder, name))
    for name in (
        "nc_matmul",
        "nc_transpose",
        "activation",
        "activation_reduce",
        "tensor_tensor",
        "tensor_scalar",
        "tensor_scalar_reduce",
        "tensor_reduce",
        "tensor_copy",
        "memset",
        "affine_select",
        "dma_copy",
        "dma_transpose",
        "scalar_tensor_tensor",
        "iota",
        "tensor_tensor_scan",
        "get_nc_version",
        "tensor_partition_reduce",
        "bn_stats",
        "bn_aggr",
        "select_reduce",
        "tensor_copy_predicated",
    ):
        _set_attr(nisa, name, getattr(nki_builder, name))
    _set_attr(nisa, "reciprocal", nki_builder.nisa_reciprocal)
    _set_attr(ncc, "multi_buffer", nki_builder.multi_buffer)
    _set_attr(nisa, "_tiled_offloaded_memcpy", nki_builder.tiled_offloaded_memcpy)
    _set_attr(nisa, "_tiled_offloaded_fma", nki_builder.tiled_offloaded_fma)
    for name in (
        "max",
        "min",
        "mean",
        "add",
        "subtract",
        "divide",
        "maximum",
        "minimum",
        "static_cast",
    ):
        _set_attr(nl, name, getattr(nki_builder, name))

    # attention-specific
    _set_attr(nl, "load_transpose2d", nki_builder.load_transpose2d)
    # nisa.affine_select
    # nl.tensor_reduce
    # nisa.activation
    nl.broadcast_to
    # nisa.nc_transpose

    # Elementwise operators
    _set_attr(nl, "exp", nki_builder.exp)
    _set_attr(nl, "relu", nki_builder.relu)
    _set_attr(nl, "sigmoid", nki_builder.sigmoid)
    _set_attr(nl, "tanh", nki_builder.tanh)
    _set_attr(nl, "silu", nki_builder.silu)
    _set_attr(nl, "gelu", nki_builder.gelu)
    _set_attr(nl, "sqrt", nki_builder.sqrt)
    _set_attr(nl, "abs", nki_builder.abs)
    _set_attr(nl, "log", nki_builder.log)
    _set_attr(nl, "pow", nki_builder.pow)
    _set_attr(nl, "reciprocal", nki_builder.reciprocal)

    _set_attr(nl, "device_print", print)


def nki_unpatch_lang(scope=None):
    if scope is not None and hasattr(scope, "restore"):
        scope.restore()


def _compile_rewritten(fn, namespace):
    """Compile `fn` with the NKI frontend AST transformer applied (decorators dropped, loads
    and stores given keys) into `namespace`, and return the new function."""
    # source of the function, stripped of leading indents in case it was defined in scope
    transformed_code = transform_code(textwrap.dedent(inspect.getsource(fn)))
    import random
    import string

    rand_str = "".join(random.choices(string.ascii_letters + string.digits, k=16))
    os.makedirs("/tmp/tilelens", exist_ok=True)
    filename = f"/tmp/tilelens/{rand_str}.py"
    with open(filename, "w") as f:
        f.write(transformed_code)
    exec(compile(transformed_code, filename=filename, mode="exec"), namespace)
    new_fn = namespace[fn.__name__]
    CODE_KEYS.add(get_code_key(new_fn))  # trace rewritten function for clients
    return new_fn


def _rewritten_namespace(module_globals, namespaces):
    """A copy of a module's globals in which its NKI kernels (`@nki.jit` functions, which
    kernels call as helpers) are rewritten, and NKI kernels it imports from other modules
    are their rewrites there. One namespace per module, so kernels that call each other
    find the rewrites. `namespaces` maps id(module globals) to the namespaces made so far."""
    from ..trace import nki_kernel_function

    key = id(module_globals)
    if key in namespaces:
        return namespaces[key]
    namespace = namespaces[key] = module_globals.copy()
    for name, value in module_globals.items():
        func = nki_kernel_function(value)
        if func is None:
            continue
        if func.__globals__ is module_globals:
            namespace[name] = _compile_rewritten(func, namespace)
        else:
            other = _rewritten_namespace(func.__globals__, namespaces)
            namespace[name] = other.get(func.__name__, value)
    return namespace


def _rewrite_function(fn):
    """`fn` rewritten by the NKI frontend AST transformer, in the rewritten namespace of its
    module, so the NKI kernels it calls by name are rewritten too."""
    return _compile_rewritten(fn, _rewritten_namespace(fn.__globals__, {}))


def full_keys(tensor):
    """Keys of a whole-tensor access such as `nl.load(t)` or `nl.store(t, value=v)`."""
    return tuple(slice(None) for _ in range(tensor.data.ndim))


def access(tensor, keys):
    """The tensor and keys a load or store accesses: `nl.load(t[k])` arrives as (t, k),
    `nl.load(t)` as (t, None), and `a = t[k]; nl.load(a)` as (a, None) with `a` an
    HBMAccess."""
    if keys is None and isinstance(tensor, HBMAccess):
        return tensor.tensor, tensor.keys
    return tensor, (full_keys(tensor) if keys is None else keys)


def _to_numpy(ret):
    if isinstance(ret, NDArray):
        return ret.data
    if isinstance(ret, (tuple, list)):
        return type(ret)(_to_numpy(r) for r in ret)
    return ret


class NKIInterpretedFunction:
    def __init__(self, fn):
        self.fn = fn

    def run(self, *args, **kwargs):
        grid_dims = kwargs.pop(
            "grid", (1, 1, 1)
        )  # Remove grid from kwargs to avoid passing it to the function
        nki_builder.grid_ndim = len(grid_dims)
        # make it 3d if not
        if len(grid_dims) == 1:
            grid_dims = (grid_dims[0], 1, 1)
        elif len(grid_dims) == 2:
            grid_dims = (grid_dims[0], grid_dims[1], 1)
        elif len(grid_dims) != 3:
            raise ValueError(
                f"Grid must be 1, 2, or 3 dimensions, got {len(grid_dims)}"
            )
        nki_builder.set_grid_dim(*grid_dims)
        nki_builder.shared_hbm_arrays = {}
        nki_builder.rng = np.random.default_rng(0)
        nki_builder.accumulator = None
        nki_builder.fn = self.fn

        kwargs.pop("warmup", None)  # Remove warmup from kwargs if it exists
        client_manager = kwargs.pop(
            "client_manager", None
        )  # Remove client_manager from kwargs if it exists

        # Call grid_callback once before grid execution (similar to Triton)
        if client_manager is not None:
            client_manager.grid_callback(grid_dims)

        # Apply the NKI frontend AST transformer, to the kernel and to the NKI kernels it
        # calls (`@nki.jit` helpers), once.
        if not getattr(self, "_rewritten", False):
            from ..trace import nki_kernel_function

            fn = nki_kernel_function(self.fn) or self.fn
            if hasattr(fn, "__code__"):
                self.fn = _rewrite_function(fn)
            self._rewritten = True

        # convert args to NDArray if they are not already, also those passed by keyword
        args = [
            NDArray(value=arg, buffer=nl.hbm) if isinstance(arg, np.ndarray) else arg
            for arg in args
        ]
        kwargs = {
            name: NDArray(value=arg, buffer=nl.hbm)
            if isinstance(arg, np.ndarray)
            else arg
            for name, arg in kwargs.items()
        }

        name_args = inspect.getcallargs(self.fn, *args, **kwargs)
        call_args = {}
        for name, arg in name_args.items():
            call_args[name] = arg
            ret = arg
            client_manager.arg_callback(name, arg, ret)

        ret = None
        for x in range(grid_dims[0]):
            for y in range(grid_dims[1]):
                for z in range(grid_dims[2]):
                    nki_builder.set_grid_idx(x, y, z)

                    # Call grid_idx_callback for each grid iteration (similar to Triton)
                    if client_manager is not None:
                        client_manager.grid_idx_callback((x, y, z))

                    if not client_manager.pre_run_callback(self.fn):
                        # a client that needs only some programs (the profiler's block
                        # sampling) skips this one, as the Triton frontend does
                        continue
                    ret = self.fn(*args, **kwargs)
                    if not client_manager.post_run_callback(self.fn):
                        # a client that needs no more programs stops the grid early
                        return _to_numpy(ret)
        # Kernels that allocate their outputs in shared HBM return them; every program
        # returns the same arrays, so return the last program's, as numpy arrays.
        return _to_numpy(ret)
