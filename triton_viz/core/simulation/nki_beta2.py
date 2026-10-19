from __future__ import annotations

import inspect
import itertools
from collections.abc import Callable
from typing import Any, Optional, TypeAlias, cast

import numpy as np

from triton_viz.utils.dtypes import DTypeLike, STORAGE_DTYPES
from triton_viz.utils.traceback_utils import CODE_KEYS, get_code_key

try:
    import nki.isa as nisa
    import nki.language as nl
except (
    ModuleNotFoundError
) as exc:  # pragma: no cover - only hit when optional deps missing
    raise ModuleNotFoundError(
        "NKI dependencies are missing. Install triton-viz[nki] to enable the "
        "NKI Beta 2 interpreter."
    ) from exc


_ERR_BINARY_TENSOR_OP = (
    "binary operators on tensors not supported. Use nki.isa directly."
)
_ERR_TENSOR_SUBSCRIPT = "subscript not supported, for 'tensor'"
_ERR_TENSOR_MUTATION = "mutation not supported"
_ERR_UNDEFINED_USE = "Illegal IR, encountered undefined use"
_ERR_AP_MISMATCH = "Expect AP same number of elements"

ScalarLike: TypeAlias = bool | int | float
ArrayLike: TypeAlias = np.ndarray[Any, Any]
TensorLike: TypeAlias = "NDArray | ArrayLike"
TensorOrScalar: TypeAlias = "NDArray | ArrayLike | ScalarLike"
BufferLike: TypeAlias = Any


class ParDim:
    """Partition-dimension shape marker compatible with ``nl.par_dim``.

    The public ``nki.language`` compatibility package used by Triton-Viz does
    not currently expose ``par_dim``.  The real Neuron token stores an integer
    ``value`` but is not directly accepted by NumPy as a shape.  This marker
    preserves the annotation long enough for ``ndarray`` to record which shape
    axis is the SBUF partition axis, while ``int(marker)`` provides the concrete
    extent used by the interpreter.
    """

    def __init__(self, value: int) -> None:
        value = int(value)
        _require(value > 0, f"par_dim must be positive, received {value}")
        self.value = value

    def __int__(self) -> int:
        return self.value

    def __index__(self) -> int:
        return self.value

    def __repr__(self) -> str:
        return f"par_dim({self.value})"


def par_dim(value: int) -> ParDim:
    """Mark one tensor shape axis as the NKI partition dimension."""
    return ParDim(value)


def _normalize_shape(shape: Any) -> tuple[tuple[int, ...], int | None]:
    """Return concrete shape integers and the explicitly marked partition axis."""
    dimensions = tuple(shape)
    axes = [index for index, dim in enumerate(dimensions) if isinstance(dim, ParDim)]
    _require(
        len(axes) <= 1,
        f"shape may contain at most one par_dim, received {dimensions}",
    )
    concrete = tuple(int(dim) for dim in dimensions)
    _require(
        all(dim > 0 for dim in concrete),
        f"tensor shape dimensions must be positive, received {concrete}",
    )
    return concrete, axes[0] if axes else None


def _storage_dtype(dtype: DTypeLike) -> np.dtype[Any] | None:
    """Resolve a logical dtype to concrete NumPy storage dtype."""
    try:
        return STORAGE_DTYPES[dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported dtype: {dtype}") from exc


def _cast_value(value: TensorOrScalar, dtype: DTypeLike) -> ArrayLike:
    """Cast value to destination dtype storage using shared ml_dtypes mapping."""
    return np.asarray(value).astype(_storage_dtype(dtype))


def _mark_defined(value: Any) -> None:
    """Mark NDArray values as initialized after writes."""
    while isinstance(value, NDArray):
        value._defined = True
        value = value._parent


def _store(dst: NDArray, value: TensorOrScalar) -> NDArray:
    """Write a computed value into a destination tensor."""
    dst.data[...] = _cast_value(value, dst.dtype)
    _mark_defined(dst)
    return dst


def _tensor_value(value: TensorOrScalar | None) -> TensorOrScalar | None:
    """Unwrap NDArray inputs and validate they are initialized."""
    if isinstance(value, NDArray) and not value._defined:
        raise RuntimeError(_ERR_UNDEFINED_USE)
    return value.data if isinstance(value, NDArray) else value


def _array(value: Any, dtype: Any = None) -> ArrayLike:
    return np.asarray(_tensor_value(value), dtype=dtype)


def _as_scalar(value: ScalarLike | NDArray) -> int:
    """Convert scalar-like value into int."""
    array = _array(value)
    _require(
        array.size == 1,
        f"Expected scalar value, received shape {array.shape} with {array.size} elements",
    )
    return int(array.reshape(-1)[0])


def _dtype_str(dtype: DTypeLike) -> str:
    """Return a stable string name for logical or storage dtypes."""
    return np.dtype(_storage_dtype(dtype)).name


def _buffer_name(value: BufferLike) -> str:
    """Return the interpreter buffer name for values with storage."""
    return getattr(value, "buffer", None) or "sbuf"


def _shape_of(value: Any) -> tuple[int, ...]:
    """Return a stable shape tuple for tensors or array-like values."""
    return tuple(getattr(value, "shape", np.asarray(value).shape))


def _dtype_name(value: Any) -> str:
    """Return a stable dtype name for tensor or dtype-like inputs."""
    dtype = getattr(value, "dtype", value)
    try:
        return _dtype_str(dtype)
    except Exception:
        return np.dtype(dtype).name


def _value_desc(value: Any) -> str:
    """Format shape/dtype/buffer details for error messages."""
    parts = [f"shape={_shape_of(value)}", f"dtype={_dtype_name(value)}"]
    if hasattr(value, "buffer"):
        parts.append(f"buffer={_buffer_name(value)}")
    return ", ".join(parts)


def _require(condition: Any, message: str) -> None:
    """Raise ValueError with a stable message when condition is false."""
    if not condition:
        raise ValueError(message)


def _is_integer_dtype(dtype: DTypeLike) -> bool:
    """Return whether dtype is integer/bool-like."""
    return np.issubdtype(np.dtype(_storage_dtype(dtype)), np.integer)


def _partition_and_free(value: TensorLike) -> tuple[int, int]:
    """Return partition size and flattened free size."""
    if isinstance(value, NDArray):
        shape = cast(tuple[int, ...], value.data.shape)
        axis = 0 if value.partition_axis is None else value.partition_axis
    else:
        shape = cast(tuple[int, ...], value.shape)
        axis = 0
    partitions = shape[axis]
    free = int(
        np.prod(
            tuple(dim for index, dim in enumerate(shape) if index != axis),
            dtype=np.int64,
        )
    )
    return partitions, free


def _nc_version_value(version: Any) -> int:
    """Normalize NC version tokens into plain integers for comparisons."""
    raw = getattr(version, "value", version)
    return raw if isinstance(raw, int) else int(raw)


def _activation_operand(
    value: TensorOrScalar | None, data_shape: tuple[int, ...], label: str
) -> float | ArrayLike:
    """Normalize activation scale/bias into broadcastable arrays."""
    if value is None:
        return 0.0
    arr = _array(value, dtype=np.float32)
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    _require(
        arr.shape == (data_shape[0], 1),
        f"{label} must be a scalar or shape ({data_shape[0]}, 1), received shape {arr.shape}",
    )
    return np.broadcast_to(
        arr.reshape(data_shape[0], 1), (data_shape[0], int(np.prod(data_shape[1:])))
    ).reshape(data_shape)


class NDArray:
    """Lightweight NumPy-backed tensor used by the NKI beta2 interpreter."""

    def __init__(
        self,
        buffer: str | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: DTypeLike = None,
        value: Any = None,
        origin: str = "raw",
        parent_ndim: int | None = None,
        defined: bool | None = None,
        parent: NDArray | None = None,
        partition_axis: int | None = None,
    ) -> None:
        """Initialize an interpreter tensor backed by NumPy storage.

        Args:
            buffer: Logical buffer name such as ``"sbuf"``, ``"hbm"``, or
                ``"psum"``.
            shape: Requested storage shape when allocating a new tensor.
            dtype: Logical dtype token used to choose storage dtype.
            value: Optional initial value. When provided, the tensor uses this
                payload instead of allocating fresh storage.
            origin: Interpreter provenance tag used for mutation rules.
            parent_ndim: Rank of the parent tensor when this value is a view.
            defined: Explicit initialized-state override for undefined-use
                tracking.
            parent: Parent tensor when this value is a sliced view.
        """
        self.buffer = buffer
        self._origin = origin
        self._parent = parent
        storage_shape, marked_axis = (
            _normalize_shape(shape) if shape is not None else (None, None)
        )
        if partition_axis is None:
            partition_axis = (
                marked_axis
                if marked_axis is not None
                else (0 if buffer in ("sbuf", "psum") else None)
            )
        storage_dtype = _storage_dtype(dtype)
        assert value is not None or storage_shape is not None
        if value is None:
            assert storage_shape is not None
            self.data = np.zeros(storage_shape, dtype=storage_dtype)
        else:
            self.data = (
                _cast_value(value, dtype) if dtype is not None else np.asarray(value)
            )
        if storage_shape is not None and self.data.shape != storage_shape:
            self.data = self.data.reshape(storage_shape)
        if self.buffer in ("sbuf", "psum") and origin != "access":
            _require(
                self.data.ndim >= 2,
                f"SBUF/PSUM tensors must have at least 2 dims, received shape {self.data.shape}",
            )
        self._defined = (
            value is not None or origin != "tensor" if defined is None else defined
        )
        self._parent_ndim = self.data.ndim if parent_ndim is None else parent_ndim
        self.partition_axis = partition_axis
        self.dtype = self.data.dtype if dtype is None else dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Return tensor shape."""
        return self.data.shape

    @property
    def address(self) -> int:
        """Return host pointer integer for underlying storage."""
        return self.data.ctypes.data

    def data_ptr(self) -> int:
        """Return pointer address alias for frontend compatibility."""
        return self.address

    def stride(self) -> tuple[int, ...]:
        """Return element strides."""
        itemsize = self.element_size()
        return tuple(int(stride // itemsize) for stride in self.data.strides)

    def element_size(self) -> int:
        """Return item size in bytes."""
        return getattr(self.dtype, "itemsize", self.data.dtype.itemsize)

    def cpu(self) -> NDArray:
        """Return self to mimic framework tensor api."""
        return self

    def detach(self) -> NDArray:
        """Return self to mimic framework tensor api."""
        return self

    def numpy(self) -> ArrayLike:
        """Return underlying NumPy array."""
        return self.data

    def __repr__(self) -> str:
        return f"NDArray(shape={self.shape}, dtype={self.dtype})"

    def __getitem__(self, keys: Any) -> NDArray:
        """Implement slicing operations for NDArray."""
        if not isinstance(keys, tuple):
            keys = (keys,)
        expanded_keys = list(keys) + [slice(None)] * (self.data.ndim - len(keys))
        if (
            self.partition_axis is not None
            and isinstance(expanded_keys[self.partition_axis], int)
            and self.buffer in ("sbuf", "psum")
        ):
            raise ValueError("SBUF/PSUM indexing must preserve the partition dim")
        sliced_value = self.data[keys]
        new_partition_axis = self.partition_axis
        if new_partition_axis is not None:
            if isinstance(expanded_keys[new_partition_axis], int):
                new_partition_axis = None
            else:
                new_partition_axis -= sum(
                    isinstance(key, int) for key in expanded_keys[:new_partition_axis]
                )
        return NDArray(
            value=sliced_value,
            buffer=self.buffer,
            origin="access",
            parent_ndim=self._parent_ndim,
            defined=self._defined,
            parent=self,
            partition_axis=new_partition_axis,
        )

    def __setitem__(self, keys: Any, value: Any) -> NDArray:
        """Assign values into a sliced region."""
        if self._origin == "tensor":
            raise RuntimeError(_ERR_TENSOR_MUTATION)
        if not isinstance(keys, tuple):
            keys = (keys,)
        expanded_keys = list(keys) + [slice(None)] * (self.data.ndim - len(keys))
        if (
            self.partition_axis is not None
            and isinstance(expanded_keys[self.partition_axis], int)
            and self.buffer in ("sbuf", "psum")
        ):
            raise ValueError("SBUF/PSUM indexing must preserve the partition dim")
        target = self.data[keys]
        if isinstance(value, NDArray):
            _require(
                value.data.size == target.size
                and value._parent_ndim == self._parent_ndim,
                _ERR_AP_MISMATCH,
            )
        stored_value = _tensor_value(value)
        assert stored_value is not None
        target[...] = _cast_value(stored_value, self.dtype)
        _mark_defined(self)
        return self

    def reshape(self, *args: Any, **kwargs: Any) -> NDArray:
        """Return a reshaped tensor view."""
        return NDArray(
            value=self.data.reshape(*args),
            buffer=self.buffer,
            partition_axis=kwargs.pop("partition_axis", None),
            **kwargs,
        )

    def __neg__(self) -> "NDArray":
        raise TypeError("cannot negate values of this type")

    def _raise_binary_op(self, *_args: Any, **_kwargs: Any) -> NDArray:
        raise TypeError(_ERR_BINARY_TENSOR_OP)

    __and__ = __or__ = __xor__ = __lshift__ = __rshift__ = _raise_binary_op
    __rand__ = __ror__ = __rxor__ = __rlshift__ = __rrshift__ = _raise_binary_op
    __iand__ = __ior__ = __ixor__ = __ilshift__ = __irshift__ = _raise_binary_op


class Buffer:
    """NKI memory region abstraction used by sbuf/hbm/psum."""

    def __init__(
        self,
        buffer: str,
        shape: tuple[int, ...] | None = None,
        data: ArrayLike | None = None,
    ) -> None:
        """Initialize a raw buffer region.

        Args:
            buffer: Logical buffer name.
            shape: Optional byte-storage shape used when allocating a raw backing
                buffer.
            data: Optional existing raw or typed backing array.
        """
        self.buffer = buffer
        self.data = data
        if shape is not None and data is None:
            self.data = np.empty(shape=shape, dtype=np.uint8)

    def view(
        self, dtype: DTypeLike, shape: tuple[int, ...], *, origin: str = "raw"
    ) -> NDArray:
        """Materialize a typed tensor view over this buffer region.

        Args:
            dtype: Logical dtype token for the returned view.
            shape: Tensor shape for the returned view.
            origin: Interpreter provenance tag for the returned tensor.
        """
        probe = NDArray(
            buffer=self.buffer,
            dtype=dtype,
            shape=shape,
            origin=origin,
            parent_ndim=len(shape),
        )
        if self.data is not None:
            _require(
                self.data.nbytes == probe.data.nbytes,
                "Buffer shape mismatch: have "
                f"{self.data.nbytes} bytes, expected {probe.data.nbytes}",
            )
            probe.data = (
                self.data.view(probe.data.dtype).reshape(probe.data.shape)
                if self.data.dtype == np.uint8
                else np.asarray(self.data, dtype=probe.data.dtype).reshape(
                    probe.data.shape
                )
            )
        return probe


sbuf, hbm, psum = map(Buffer, ("sbuf", "hbm", "psum"))


def ndarray(
    shape: tuple[int | ParDim, ...],
    dtype: DTypeLike,
    *,
    buffer: BufferLike = None,
    value: Any = None,
) -> NDArray:
    """Create an interpreter tensor on the requested NKI buffer.

    Args:
        shape: Requested tensor shape.
        dtype: Logical dtype token for the returned tensor.
        buffer: Buffer token or buffer-like object naming ``sbuf``, ``hbm``, or
            ``psum``.
        value: Optional tensor construction override for a preexisting payload.
    """
    if not isinstance(buffer, Buffer):
        buffer = globals().get(getattr(buffer, "name", buffer or "sbuf"))
        if not isinstance(buffer, Buffer):
            raise TypeError(f"Unsupported buffer type: {type(buffer).__name__}")
    if value is not None:
        concrete_shape, partition_axis = _normalize_shape(shape)
        return NDArray(
            buffer=buffer.buffer,
            shape=concrete_shape,
            dtype=dtype,
            value=value,
            origin="tensor",
            parent_ndim=len(concrete_shape),
            partition_axis=partition_axis,
        )
    concrete_shape, partition_axis = _normalize_shape(shape)
    tensor = buffer.view(dtype, concrete_shape, origin="tensor")
    tensor.partition_axis = partition_axis
    return tensor


def zeros(shape: tuple[int, ...], dtype: DTypeLike, *, buffer: Any = None) -> NDArray:
    """Create a zero-initialized tensor.

    Args:
        shape: Requested tensor shape.
        dtype: Logical dtype token for the returned tensor.
        buffer: Unsupported beta2 argument kept for signature parity.
    """
    if buffer is not None:
        raise TypeError(
            "unexpected keyword argument 'buffer' in builtinfunction 'builtin_lang_zeros'"
        )
    return _store(ndarray(shape, dtype), 0)


def nc_transpose(
    dst: NDArray,
    data: NDArray,
    engine: Any = nisa.engine.unknown,
    name: str | None = None,
) -> NDArray:
    """Compute a transpose between partition and flattened free axes.

    Args:
        dst: Destination tensor receiving the transpose result.
        data: Source tensor to transpose.
        engine: Requested execution engine token.
        name: Optional kernel op name retained for API parity.
    """
    value = _array(data)
    data_par, data_free = _partition_and_free(value)
    _require(
        max(data_par, data_free) <= 128,
        f"tile too large for nc_transpose, received shape {value.shape}",
    )
    src_buffer, dst_buffer = _buffer_name(data), _buffer_name(dst)
    if engine == nisa.engine.unknown:
        engine = (
            nisa.vector_engine if max(data_par, data_free) <= 32 else nisa.tensor_engine
        )
    if engine == nisa.vector_engine:
        _require(
            src_buffer in ("sbuf", "psum") and dst_buffer in ("sbuf", "psum"),
            "Vector Engine nc_transpose only supports SBUF/PSUM, received "
            f"src buffer {src_buffer} and dst buffer {dst_buffer}",
        )
        _require(
            max(data_par, data_free) <= 32,
            f"Operand must be no larger than (32, 32) for a Vector Engine transpose, received: {value.shape}",
        )
    elif engine == nisa.tensor_engine:
        _require(
            src_buffer == "sbuf" and dst_buffer == "psum",
            "Tensor Engine nc_transpose requires SBUF -> PSUM, received "
            f"src buffer {src_buffer} and dst buffer {dst_buffer}",
        )
        _require(
            dst.shape == (data_free, data_par),
            "Tensor Engine nc_transpose requires dst shape (free, partition), "
            f"expected {(data_free, data_par)}, received {dst.shape}",
        )
    _require(
        _dtype_str(dst.dtype) == _dtype_str(data.dtype)
        or (
            _dtype_str(dst.dtype) == "bfloat16" and _dtype_str(data.dtype) == "float32"
        ),
        f"dst dtype must match data dtype, received dst={_dtype_name(dst.dtype)} and data={_dtype_name(data.dtype)}",
    )
    transposed = value if value.ndim < 2 else value.reshape(value.shape[0], -1).T
    return _store(dst, transposed.reshape(dst.shape))


def nc_matmul(
    dst: NDArray,
    stationary: NDArray,
    moving: NDArray,
    is_stationary_onezero: bool = False,
    is_moving_onezero: bool = False,
    is_transpose: bool = False,
    tile_position: tuple[int, int] | tuple[()] = (),
    tile_size: tuple[int, int] | tuple[()] = (),
    perf_mode: Any = nisa.matmul_perf_mode.none,
    name: str | None = None,
) -> NDArray:
    """Compute ``dst += stationary.T @ moving``.

    Args:
        dst: PSUM destination tensor to accumulate into.
        stationary: Stationary SBUF operand.
        moving: Moving SBUF operand.
        is_stationary_onezero: Unused beta2 compatibility flag.
        is_moving_onezero: Unused beta2 compatibility flag.
        is_transpose: Unused beta2 compatibility flag.
        tile_position: Optional tile origin for documented matmul tiling.
        tile_size: Optional tile shape paired with ``tile_position``.
        perf_mode: Unused beta2 performance-mode token.
        name: Optional kernel op name retained for API parity.
    """
    _require(
        _buffer_name(dst) == "psum",
        f"nc_matmul requires dst in PSUM, received dst buffer {_buffer_name(dst)}",
    )
    _require(
        _buffer_name(stationary) == "sbuf",
        f"nc_matmul requires stationary in SBUF, received stationary buffer {_buffer_name(stationary)}",
    )
    _require(
        _buffer_name(moving) == "sbuf",
        f"nc_matmul requires moving in SBUF, received moving buffer {_buffer_name(moving)}",
    )
    stationary_value, moving_value = _array(stationary), _array(moving)
    _require(
        stationary_value.ndim == 2,
        f"nc_matmul stationary must be rank-2, received shape {stationary_value.shape}",
    )
    stationary_par, stationary_free = _partition_and_free(stationary_value)
    moving_par, moving_free = _partition_and_free(moving_value)
    _require(
        stationary_par == moving_par,
        "partition dims of stationary and moving must match, received "
        f"stationary={stationary_par} and moving={moving_par}",
    )
    _require(
        stationary_par <= 128,
        f"partition dim of stationary > 128, received {stationary_par}",
    )
    _require(
        stationary_free <= 128,
        f"free dim of stationary > 128, received {stationary_free}",
    )
    _require(
        moving_free <= 512,
        f"free dim of moving > 512, received {moving_free}",
    )
    _require(
        (_dtype_str(stationary.dtype) == "float32")
        == (_dtype_str(moving.dtype) == "float32"),
        "if one matmul operand is float32, both must be float32, received "
        f"stationary={_dtype_name(stationary.dtype)} and moving={_dtype_name(moving.dtype)}",
    )
    _require(
        _dtype_str(dst.dtype) in {"float32", "bfloat16"},
        f"dst must be float32 or bfloat16, received {_dtype_name(dst.dtype)}",
    )
    _require(
        dst.shape == (stationary_free, *moving_value.shape[1:]),
        "nc_matmul dst must preserve stationary free dim and moving free dims, "
        f"expected {(stationary_free, *moving_value.shape[1:])}, received {dst.shape}",
    )
    _require(
        bool(tile_position) == bool(tile_size),
        f"both tile_position and tile_size must be supplied, received tile_position={tile_position}, tile_size={tile_size}",
    )
    if tile_size:
        for size in tile_size:
            _require(
                size in (32, 64, 128),
                f"tile_size dims must be 32, 64, or 128, received {tile_size}",
            )
        _require(
            stationary_par <= tile_size[0] and stationary_free <= tile_size[1],
            "stationary tile size exceeds tile_size, received "
            f"stationary shape {(stationary_par, stationary_free)} and tile_size {tile_size}",
        )
        for start, size in zip(tile_position, tile_size):
            _require(
                start % size == 0,
                f"tile start must align to tile size, received start={start}, size={size}",
            )
            _require(
                start + size <= 128,
                f"tile extent must not exceed 128, received start={start}, size={size}",
            )
    result = stationary_value.T @ moving_value.reshape(moving_par, moving_free)
    return _store(
        dst,
        np.asarray(dst.data) + _cast_value(result.reshape(dst.shape), dst.dtype),
    )


def reciprocal(dst: NDArray, data: NDArray, name: str | None = None) -> NDArray:
    """Compute elementwise reciprocal into the destination tensor.

    Args:
        dst: Destination tensor receiving reciprocal values.
        data: Source tensor providing reciprocal inputs.
        name: Optional kernel op name retained for API parity.
    """
    result = _store(dst, np.reciprocal(_array(data, dtype=np.float32)))
    result._nki_api = "reciprocal"
    result._nki_engine = "scalar"
    result._nki_inputs = (data,)
    return result


def exponential(
    dst: NDArray,
    src: NDArray,
    max_value: TensorOrScalar = 0.0,
    reduce_res: NDArray | None = None,
    reduce_cmd: Any = nisa.reduce_cmd.idle,
    reduce_init: float = 0.0,
) -> NDArray:
    """Compute ``exp(src - max_value)`` into the destination tensor.

    Args:
        dst: Destination tensor receiving the exponentiated output.
        src: Source tensor to exponentiate.
        max_value: Scalar or tensor broadcast operand subtracted before ``exp``.
        reduce_res: Optional destination tensor for row-wise output sums.
        reduce_cmd: Unused beta2 reduction token kept for API parity.
        reduce_init: Unused beta2 reduction init value kept for API parity.
    """
    _require(
        _nc_version_value(nki_builder.nc_version)
        >= _nc_version_value(nisa.nc_version.gen4),
        f"exponential only supports >= NeuronCore-v4, received nc_version={nki_builder.nc_version}",
    )
    output = np.exp(_array(src, dtype=np.float32) - _array(max_value, dtype=np.float32))
    _store(dst, output)
    if reduce_res is not None:
        reduced = np.sum(
            np.asarray(output).reshape(output.shape[0], -1), axis=1, keepdims=True
        )
        _store(reduce_res, reduced)
    return dst


def affine_range(start: Any, stop: Any = None, step: Any = 1) -> range:
    """Create a Python range from static or register-backed bounds.

    Args:
        start: Range start or stop when ``stop`` is omitted.
        stop: Optional range stop.
        step: Optional range step.
    """
    start = _as_scalar(start)
    args = (start,) if stop is None else (start, _as_scalar(stop), _as_scalar(step))
    return range(*args)


def ds(start: int | NDArray, size: int | NDArray) -> slice:
    """Return an NKI-style dynamic slice using start/extent semantics.

    Args:
        start: Slice start offset.
        size: Slice extent.
    """
    begin = _as_scalar(start)
    return slice(begin, begin + _as_scalar(size))


class TileSize:
    """Minimal tile size constants used by beta2 example kernels."""

    pmax = 128
    gemm_stationary_fmax = 128
    gemm_moving_fmax = 512


tile_size = TileSize()


def _broadcast_tensor_scalar_operand(
    data: TensorLike, operand: TensorOrScalar
) -> ArrayLike:
    """Broadcast tensor_scalar operand over free dimensions only."""
    data_arr = np.asarray(data)
    operand_arr = np.asarray(operand)
    if operand_arr.size == 1:
        return operand_arr
    _require(
        operand_arr.shape[0] == data_arr.shape[0]
        and operand_arr.shape[1:] == (1,) * (operand_arr.ndim - 1),
        "tensor_scalar only broadcasts free dimensions, received "
        f"data shape {data_arr.shape} and operand shape {operand_arr.shape}",
    )
    target_shape = (data_arr.shape[0],) + (1,) * (data_arr.ndim - 1)
    return np.broadcast_to(operand_arr.reshape(target_shape), data_arr.shape)


class SubOp:
    """
    NKI ISA ops like tensor_tensor have parameters to modify which "sub-operation" each element does (e.g. nl.add).
    There are a couple of NKI quirks with these ops (e.g. unary ops can be used in nisa ops that pass in two args)
    so we use this instead of numpy ops.
    """

    def __init__(
        self,
        np_op: Callable[..., Any],
        bitwise: bool,
        reducible: bool,
        name: str | None = None,
    ) -> None:
        """Initialize a patched NKI operator token.

        Args:
            np_op: Underlying NumPy callable implementing the op.
            bitwise: Whether the op follows integer bitwise rules.
            reducible: Whether the op can be used in ``tensor_reduce``.
            name: Optional stable display name override.
        """
        self._op = np_op  # NOTE: Only NKI ISA ops should use self._op/self.run
        self.bitwise = bitwise
        self.reducible = reducible
        self.num_args = len(inspect.signature(self._op).parameters)
        self.name = name or getattr(np_op, "__name__", type(np_op).__name__)

    def run(self, *args: Any) -> Any:
        return self._op(*args[: self.num_args])

    def __call__(self, *args: Any, **kwargs: Any) -> NDArray:
        raise TypeError(_ERR_BINARY_TENSOR_OP)

    def __repr__(self) -> str:
        return f"SubOp({self.name})"


def _contiguous_block_copy(dst: NDArray, src: NDArray) -> bool:
    """A reshaping DMA the compiler accepts: one side is a contiguous HBM block, the other a 2-D SBUF
    tile, with equal element counts; data moves in row-major (partition-major) order. Verified on
    device for agent kernels (e.g. an HBM (256, 128) block into a (128, 256) tile). Other reshapes,
    such as splitting a tile's free dimension, stay rejected."""
    if int(np.prod(dst.shape)) != int(np.prod(src.shape)):
        return False
    pairs = ((src, dst), (dst, src))
    for hbm, sbuf in pairs:
        if _buffer_name(hbm) == "hbm" and _buffer_name(sbuf) == "sbuf" and len(sbuf.shape) == 2:
            data = getattr(hbm, "data", None)
            if data is not None and getattr(data, "flags", None) is not None and data.flags["C_CONTIGUOUS"]:
                return True
    return False


def dma_copy(
    dst: NDArray,
    src: NDArray,
    dst_rmw_op: SubOp | None = None,
    oob_mode: Any = nisa.oob_mode.error,
    dge_mode: Any = nisa.dge_mode.unknown,
    unique_indices: bool = True,
    name: str | None = None,
) -> NDArray:
    """Copy data from a source tensor into a destination tensor.

    Args:
        dst: Destination tensor in HBM or SBUF.
        src: Source tensor in HBM or SBUF.
        dst_rmw_op: Optional read-modify-write op. Only ``nl.add`` is supported.
        oob_mode: Unused beta2 out-of-bounds mode token kept for API parity.
        dge_mode: Unused beta2 DMA gather engine token kept for API parity.
        unique_indices: Unused beta2 uniqueness flag kept for API parity.
        name: Optional kernel op name retained for API parity.
    """
    _require(
        _buffer_name(dst) in ("hbm", "sbuf") and _buffer_name(src) in ("hbm", "sbuf"),
        "dma_copy only supports HBM/SBUF source and destination, received "
        f"src buffer {_buffer_name(src)} and dst buffer {_buffer_name(dst)}",
    )
    src_value = _array(src)
    _require(
        dst.shape == src.shape or _contiguous_block_copy(dst, src),
        f"{_ERR_AP_MISMATCH}: dst shape {dst.shape}, src shape {src.shape}",
    )
    if dst_rmw_op is None:
        return _store(dst, src_value.reshape(dst.shape))
    _require(
        dst_rmw_op == nl.add,
        f"only nl.add is supported for dma_copy dst_rmw_op, received {dst_rmw_op}",
    )
    return _store(dst, _array(dst_rmw_op.run(dst.data, src_value)))


def dma_transpose(
    dst: NDArray,
    src: NDArray,
    axes: Any = None,
    priority: Any = None,
    dge_mode: Any = nisa.dge_mode.unknown,
    oob_mode: Any = nisa.oob_mode.error,
    name: str | None = None,
) -> NDArray:
    """Transpose one HBM tile into SBUF using the DMA-transpose path."""
    del priority, dge_mode, oob_mode, name
    _require(
        _buffer_name(src) == "hbm" and _buffer_name(dst) == "sbuf",
        "dma_transpose requires HBM -> SBUF, received "
        f"src buffer {_buffer_name(src)} and dst buffer {_buffer_name(dst)}",
    )
    source = _array(src)
    permutation = tuple(reversed(range(source.ndim))) if axes is None else tuple(axes)
    _require(
        sorted(permutation) == list(range(source.ndim)),
        f"invalid dma_transpose axes {permutation} for rank {source.ndim}",
    )
    transposed = np.transpose(source, permutation)
    _require(
        transposed.shape == dst.shape,
        f"{_ERR_AP_MISMATCH}: transposed src shape {transposed.shape}, dst shape {dst.shape}",
    )
    return _store(dst, transposed)


def tensor_copy(
    dst: NDArray,
    src: NDArray,
    engine: Any = nisa.engine.unknown,
    name: str | None = None,
) -> NDArray:
    """Copy tensor data within on-chip memory.

    Args:
        dst: Destination tensor in SBUF or PSUM.
        src: Source tensor in SBUF or PSUM.
        engine: Requested execution engine token.
        name: Optional kernel op name retained for API parity.
    """
    dst_buffer, src_buffer = _buffer_name(dst), _buffer_name(src)
    _require(
        dst_buffer in ("sbuf", "psum") and src_buffer in ("sbuf", "psum"),
        "tensor_copy is on-chip only, received "
        f"src buffer {src_buffer} and dst buffer {dst_buffer}",
    )
    if engine == nisa.scalar_engine and _nc_version_value(
        nki_builder.nc_version
    ) <= _nc_version_value(nisa.nc_version.gen2):
        raise ValueError("Scalar Engine tensor_copy is unsupported on NeuronCore-v2")
    if engine == nisa.gpsimd_engine and "psum" in (dst_buffer, src_buffer):
        raise ValueError("GpSimd tensor_copy cannot access PSUM")
    src_value = _array(src)
    _require(
        _partition_and_free(src_value) == _partition_and_free(dst),
        f"{_ERR_AP_MISMATCH}: src shape {src.shape}, dst shape {dst.shape}",
    )
    return _store(dst, src_value.reshape(dst.shape))


def tensor_tensor(
    dst: NDArray,
    data1: NDArray,
    data2: NDArray,
    op: SubOp,
    engine: Optional[Any] = nisa.engine.unknown,
    name: str | None = None,
) -> NDArray:
    """Apply a binary tensor op and store the result in ``dst``.

    Args:
        dst: Destination tensor for the computed output.
        data1: Left-hand tensor operand.
        data2: Right-hand tensor operand.
        op: Patched beta2 operator token such as ``nl.add``.
        engine: Requested execution engine token.
        name: Optional kernel op name retained for API parity.
    """
    _require(isinstance(op, SubOp), f"Unsupported op: {op}")
    dst_buffer, lhs_buffer, rhs_buffer = map(_buffer_name, (dst, data1, data2))
    _require(
        not (lhs_buffer == "psum" and rhs_buffer == "psum"),
        "tensor_tensor cannot read both inputs from PSUM, received "
        f"lhs buffer {lhs_buffer} and rhs buffer {rhs_buffer}",
    )
    if op.name == "power":
        engine = nisa.gpsimd_engine
    if engine == nisa.gpsimd_engine and "psum" in (dst_buffer, lhs_buffer, rhs_buffer):
        raise ValueError(
            "attempted to access PSUM with explicit (e.g. non-default engine arg)/implicit (e.g. op == nl.power) use of GpSimd Engine"
        )
    _require(
        (lhs_buffer, rhs_buffer)
        in (("sbuf", "sbuf"), ("sbuf", "psum"), ("psum", "sbuf")),
        "buffer combination for data1/data2 must be in sbuf/sbuf, sbuf/psum, or psum/sbuf, "
        f"received ({lhs_buffer}, {rhs_buffer})",
    )
    lhs, rhs = _array(data1), _array(data2)
    _require(
        lhs.ndim >= 2 and rhs.ndim >= 2,
        "tensor_tensor doesn't broadcast scalars/vectors; use tensor_scalar, "
        f"received lhs shape {lhs.shape} and rhs shape {rhs.shape}",
    )
    lhs_pf_size, rhs_pf_size, dst_pf_size = map(
        _partition_and_free, (data1, data2, dst)
    )
    _require(
        lhs_pf_size == rhs_pf_size == dst_pf_size,
        "tensor_tensor doesn't broadcast scalars/vectors; use tensor_scalar, "
        f"received lhs pf {lhs_pf_size}, rhs pf {rhs_pf_size}, dst pf {dst_pf_size}",
    )
    lhs = lhs.reshape(*lhs_pf_size)
    rhs = rhs.reshape(*rhs_pf_size)
    if op.bitwise:
        dtypes = (dst.dtype, data1.dtype, data2.dtype)
        _require(
            all(_is_integer_dtype(dtype) for dtype in dtypes),
            "data and dst must be integer dtypes for bitvec operators, received "
            f"dst={_dtype_name(dst.dtype)}, data1={_dtype_name(data1.dtype)}, data2={_dtype_name(data2.dtype)}",
        )
    return _store(dst, _array(op.run(lhs, rhs)).reshape(dst.shape))


def tensor_reduce(
    dst: NDArray,
    op: SubOp,
    data: NDArray,
    axis: int | list[int] | tuple[int, ...],
    negate: bool = False,
    keepdims: bool = False,
    name: str | None = None,
) -> NDArray:
    """Reduce a tensor along axes using add/max/min-like ops.

    Args:
        dst: Destination tensor receiving the reduced output.
        op: Patched reduction operator token.
        data: Source tensor to reduce.
        axis: Reduction axis or axes.
        negate: Whether to negate the reduced result.
        keepdims: Whether to preserve reduced dimensions as size-1 axes.
        name: Optional kernel op name retained for API parity.
    """
    _require(isinstance(op, SubOp), f"Unsupported op: {op}")
    source = _array(data)
    axis_tuple = (axis,) if isinstance(axis, int) else tuple(axis)
    _require(
        axis_tuple == (1,),
        f"not a valid dim, received axis={axis_tuple}",
    )
    data_par, _ = _partition_and_free(source)
    dst_par, _ = _partition_and_free(dst)
    _require(
        data_par == dst_par,
        f"tensor_reduce partition dim mismatch, received data={data_par} and dst={dst_par}",
    )
    _require(
        data_par <= 128,
        f"tensor_reduce partition dim > 128, received {data_par}",
    )
    _require(
        _partition_and_free(np.sum(source, axis=axis_tuple))
        == _partition_and_free(dst),
        "cannot reduce free dim of data into free dim of dst, received "
        f"reduced pf {_partition_and_free(np.sum(source, axis=axis_tuple))} and dst pf {_partition_and_free(dst)}",
    )
    _require(op.reducible, f"Op is not a legal reduction op, received {op}")
    if op.bitwise:
        _require(
            _is_integer_dtype(data.dtype) and _is_integer_dtype(dst.dtype),
            "data and dst must be integer dtypes for bitvec operators, received "
            f"data={_dtype_name(data.dtype)} and dst={_dtype_name(dst.dtype)}",
        )
    if negate and op.bitwise:
        raise ValueError("can only negate when reducing with an arithmetic operator")
    reduced = cast(Any, op._op).reduce(source, axis=axis_tuple, keepdims=keepdims)
    if negate:
        reduced = -np.asarray(reduced)
    reduced = np.asarray(reduced)
    if reduced.shape != dst.shape:
        _require(
            reduced.ndim == 1 and dst.shape == (reduced.shape[0], 1),
            "cannot reduce free dim of data into free dim of dst, received "
            f"reduced shape {reduced.shape} and dst shape {dst.shape}",
        )
        reduced = reduced.reshape(dst.shape)
    result = _store(dst, reduced)
    return result


def tensor_scalar(
    dst: NDArray,
    data: NDArray,
    op0: SubOp,
    operand0: TensorOrScalar,
    reverse0: bool = False,
    op1: SubOp | None = None,
    operand1: TensorOrScalar | None = None,
    reverse1: bool = False,
    engine: Any = nisa.engine.unknown,
    name: str | None = None,
) -> NDArray:
    """Apply one or two tensor-scalar operations and write into ``dst``.

    Args:
        dst: Destination tensor for the computed output.
        data: Source tensor operand.
        op0: First patched beta2 operator token.
        operand0: Scalar or tensor operand paired with ``op0``.
        reverse0: Whether to reverse the argument order for ``op0``.
        op1: Optional second patched beta2 operator token.
        operand1: Optional scalar or tensor operand paired with ``op1``.
        reverse1: Whether to reverse the argument order for ``op1``.
        engine: Requested execution engine token.
        name: Optional kernel op name retained for API parity.
    """
    if not isinstance(op0, SubOp):
        raise ValueError(f"Unsupported op: {op0}")
    if not (op1 is None or isinstance(op1, SubOp)):
        raise ValueError(f"Unsupported op: {op1}")
    _require(
        op1 is None or op0.bitwise == op1.bitwise,
        f"bitvec and arithmetic ops can't be mixed, received op0={op0} and op1={op1}",
    )
    _require(
        not (op0.bitwise and engine in (nisa.scalar_engine, nisa.gpsimd_engine)),
        f"bitvec ops must run on Vector Engine, received engine={engine}",
    )
    _require(
        not (op0.name == "rsqrt" and engine == nisa.vector_engine),
        f"rsqrt can only run on GpSimd/Scalar Engine, received engine={engine}",
    )
    if op0.bitwise:
        _require(
            _is_integer_dtype(data.dtype) and _is_integer_dtype(dst.dtype),
            f"data must be int dtype for bitvec ops, received data={_dtype_name(data.dtype)} and dst={_dtype_name(dst.dtype)}",
        )
    result = _array(data)
    _require(
        _partition_and_free(result) == _partition_and_free(dst),
        "tensor_scalar dst must preserve partition/free shape, received "
        f"data pf {_partition_and_free(result)} and dst pf {_partition_and_free(dst)}",
    )
    for idx, (op, operand, reverse) in enumerate(
        (
            (op0, operand0, reverse0),
            (op1, operand1, reverse1),
        ),
    ):
        if op is None:
            continue
        if op0.bitwise and isinstance(operand, NDArray):
            _require(
                _is_integer_dtype(operand.dtype),
                f"operand0/1 dtype must match bitvec integer requirements, received {_dtype_name(operand.dtype)}",
            )
        assert operand is not None
        if isinstance(operand, NDArray):
            operand = _array(operand)
            if operand.ndim > 1 and operand.shape[0] == result.shape[0]:
                operand_free = int(np.prod(operand.shape[1:], dtype=np.int64))
                idx_str = "1st" if idx == 0 else "2nd"
                _require(
                    operand_free == 1,
                    f"{idx_str} Immediate pointer's number of elements per partition must be 1, received operand shape {operand.shape}",
                )
        operand_value = _broadcast_tensor_scalar_operand(result, operand)
        result = _array(
            op.run(*(operand_value, result) if reverse else (result, operand_value))
        )
    tagged = _store(dst, result.reshape(dst.shape))
    tagged._nki_api = (
        str(op0.name)
        if op1 is None
        else f"{op0.name}_{op1.name}"
    )
    tagged._nki_engine = "vector"
    tagged._nki_inputs = tuple(
        item for item in (data, operand0, operand1)
        if isinstance(item, NDArray)
    )
    return tagged


def activation(
    dst: NDArray,
    op: SubOp,
    data: NDArray,
    bias: TensorOrScalar | None = None,
    scale: TensorOrScalar = 1.0,
    reduce_op: SubOp | None = None,
    reduce_res: NDArray | None = None,
    reduce_cmd: Any = nisa.reduce_cmd.idle,
    name: str | None = None,
) -> NDArray:
    """Apply an activation epilogue into ``dst``.

    Args:
        dst: Destination tensor receiving the activation output.
        op: Patched unary activation operator token.
        data: Source tensor to transform.
        bias: Optional scalar or per-partition bias operand.
        scale: Optional scalar or per-partition scale operand.
        reduce_op: Optional reduction operator token. Only ``nl.add`` is legal.
        reduce_res: Optional destination tensor for a companion reduction result.
        reduce_cmd: Unused beta2 reduction command token kept for API parity.
        name: Optional kernel op name retained for API parity.
    """
    _require(isinstance(op, SubOp), f"Unsupported op: {op}")
    _require(
        _buffer_name(dst) in ("sbuf", "psum")
        and _buffer_name(data) in ("sbuf", "psum"),
        "activation only supports SBUF/PSUM, received "
        f"data buffer {_buffer_name(data)} and dst buffer {_buffer_name(dst)}",
    )
    data_value = _array(data, dtype=np.float32)
    data_par, data_free = _partition_and_free(data_value)
    dst_par, dst_free = _partition_and_free(dst)
    _require(
        dst_par == data_par,
        f"dst/data partition dim mismatch, received dst={dst_par} and data={data_par}",
    )
    _require(
        dst_free == data_free,
        f"dst/data free dim mismatch, received dst={dst_free} and data={data_free}",
    )
    _require(
        data_par <= 128,
        f"data partition dim > 128, received {data_par}",
    )
    _require(
        reduce_op is None or reduce_op.name in ("add", "maximum"),
        f"activation reduce_op must be None, nl.add or nl.maximum, received {reduce_op}",
    )
    scale_value = _activation_operand(scale, data_value.shape, "scale")
    bias_value = _activation_operand(bias, data_value.shape, "bias")
    output_value = _array(op.run(data_value * scale_value + bias_value))
    tagged = _store(dst, output_value.reshape(dst.shape))
    tagged._nki_api = str(op.name)
    tagged._nki_engine = "scalar"
    # Keep bias/scale in the recorded inputs so cross-engine dependencies
    # resolve correctly. The cost model treats an activation as one data
    # stream (see ``_input_stream_count``), matching its Level-B calibration.
    tagged._nki_inputs = tuple(
        item for item in (data, bias, scale) if isinstance(item, NDArray)
    )
    if reduce_res is not None and reduce_op is not None:
        _require(
            reduce_res.shape == (data_par, 1),
            f"reduce_res must have shape (P, 1), expected {(data_par, 1)}, received {reduce_res.shape}",
        )
        rows = np.asarray(output_value, dtype=np.float32).reshape(data_par, -1)
        # nl.maximum compiles on device for activation accumulation (verified on agent kernels).
        reduced = (np.max(rows, axis=1, keepdims=True) if reduce_op.name == "maximum"
                   else np.sum(rows, axis=1, keepdims=True))
        _store(reduce_res, reduced)
    return tagged


class Builder:
    """Tracks grid dimensions/index for the active interpreted kernel."""

    def __init__(self, grid_dims: tuple[int, ...] | None = None) -> None:
        """Initialize grid-tracking state for interpreted kernel launches.

        Args:
            grid_dims: Optional initial launch grid dimensions.
        """
        self.grid_dims = tuple(grid_dims) if grid_dims is not None else (1,)
        self.grid_idx = [0] * len(self.grid_dims)
        self.nc_version = nisa.nc_version.gen2
        self.fn = None

    def set_grid_dim(self, *grid_dims: int) -> None:
        """Update the active launch grid dimensions.

        Args:
            *grid_dims: New launch grid dimensions.
        """
        _require(
            len(grid_dims) == 1,
            f"NKI Beta 2 currently only supports 1D SPMD grids, received {grid_dims}",
        )
        self.grid_dims = tuple(grid_dims)
        self.grid_idx = [0] * len(self.grid_dims)

    def set_grid_idx(self, x: int, y: int = 0, z: int = 0) -> None:
        """Update the current program-id coordinates.

        Args:
            x: Grid index for axis 0.
            y: Grid index for axis 1.
            z: Grid index for axis 2.
        """
        self.grid_idx = [x]


nki_builder = Builder()


def program_id(axis: int) -> int:
    """Return the current program-id component for the requested axis.

    Args:
        axis: Program-id axis to query.
    """
    # if 0 <= (axis := int(axis)) < len(nki_builder.grid_idx):
    if 0 <= axis < len(nki_builder.grid_idx):
        return nki_builder.grid_idx[axis]
    raise ValueError(f"Invalid axis {axis} for {len(nki_builder.grid_idx)}D grid")


def nki_patch_lang(scope: Any = None) -> None:
    """Patch ``nl`` and ``nisa`` to use beta2 interpreter implementations.

    Args:
        scope: Optional patch scope that exposes ``set_attr`` for reversible
            patching.
    """
    set_attr = setattr if scope is None else scope.set_attr
    set_attr(nl, "ndarray", ndarray)
    set_attr(nl, "zeros", zeros)
    set_attr(nl, "copy", SubOp(np.copy, False, False))
    set_attr(nl, "add", SubOp(np.add, False, True))
    set_attr(nl, "subtract", SubOp(np.subtract, False, True))
    set_attr(nl, "multiply", SubOp(np.multiply, False, True))
    set_attr(nl, "maximum", SubOp(np.maximum, False, True))
    set_attr(nl, "minimum", SubOp(np.minimum, False, True))
    set_attr(nl, "power", SubOp(np.power, False, False))
    set_attr(nl, "logical_and", SubOp(np.logical_and, True, True))
    set_attr(nl, "logical_or", SubOp(np.logical_or, True, True))
    set_attr(nl, "logical_xor", SubOp(np.logical_xor, True, True))
    set_attr(nl, "bitwise_and", SubOp(np.bitwise_and, True, True))
    set_attr(nl, "bitwise_or", SubOp(np.bitwise_or, True, True))
    set_attr(nl, "bitwise_xor", SubOp(np.bitwise_xor, True, True))
    set_attr(nl, "sqrt", SubOp(np.sqrt, False, False))
    set_attr(nl, "rsqrt", SubOp(lambda v: 1.0 / np.sqrt(v), False, False, name="rsqrt"))
    set_attr(nl, "exp", SubOp(np.exp, False, False))
    set_attr(nl, "reciprocal", SubOp(np.reciprocal, False, False))
    set_attr(nl, "tile_size", tile_size)
    set_attr(nl, "par_dim", par_dim)
    set_attr(nl, "sbuf", sbuf)
    set_attr(nl, "hbm", hbm)
    set_attr(nl, "psum", psum)
    set_attr(nl, "program_id", program_id)
    set_attr(nl, "affine_range", affine_range)
    set_attr(nl, "ds", ds)

    set_attr(nisa, "nc_matmul", nc_matmul)
    set_attr(nisa, "nc_transpose", nc_transpose)
    set_attr(nisa, "activation", activation)
    set_attr(nisa, "exponential", exponential)
    set_attr(nisa, "reciprocal", reciprocal)
    set_attr(nisa, "dma_copy", dma_copy)
    set_attr(nisa, "dma_transpose", dma_transpose)
    set_attr(nisa, "tensor_copy", tensor_copy)
    set_attr(nisa, "tensor_tensor", tensor_tensor)
    set_attr(nisa, "tensor_reduce", tensor_reduce)
    set_attr(nisa, "tensor_scalar", tensor_scalar)


def nki_unpatch_lang(scope: Any = None) -> None:
    """Restore beta2 language patches when the scope supports it.

    Args:
        scope: Optional patch scope that exposes ``restore``.
    """
    if scope is not None and hasattr(scope, "restore"):
        scope.restore()


class NKIBeta2InterpretedFunction:
    """Callable wrapper that executes NKI Beta 2 kernels with interpreter semantics."""

    def __init__(self, fn: Any) -> None:
        """Store the original kernel function.

        Args:
            fn: Kernel function to execute under beta2 interpreter semantics.
        """
        self.fn = fn

    def run(self, *args: Any, **kwargs: Any) -> None:
        """Run the wrapped kernel over a launch grid with tracing callbacks.

        Args:
            *args: Positional kernel arguments.
            **kwargs: Kernel keyword arguments, including optional ``grid``,
                ``warmup``, and ``client_manager`` controls.
        """
        grid_dims = kwargs.pop("grid", (1,))
        grid_dims = (grid_dims,) if isinstance(grid_dims, int) else tuple(grid_dims)
        _require(
            grid_dims,
            f"Grid must have at least one dimension, received {grid_dims}",
        )
        nki_builder.grid_dims = tuple(int(dim) for dim in grid_dims)
        nki_builder.grid_idx = [0] * len(nki_builder.grid_dims)
        nki_builder.fn = self.fn
        kwargs.pop("warmup", None)
        client_manager = kwargs.pop("client_manager", None)
        if client_manager is not None:
            client_manager.grid_callback(grid_dims)
        exec_globals = self.fn.__globals__
        exec_globals.update(sbuf=sbuf, hbm=hbm, psum=psum)
        CODE_KEYS.add(get_code_key(self.fn))
        kernel_args = tuple(
            arg
            if isinstance(arg, (NDArray, bool, int, float, str)) or arg is None
            else NDArray(value=arg, buffer="hbm")
            for arg in args
        )
        bound = inspect.signature(self.fn).bind(*kernel_args, **kwargs)
        bound.apply_defaults()
        for name, arg in bound.arguments.items():
            assert arg is None or isinstance(arg, (bool, int, float, str, NDArray))
            if client_manager is not None:
                client_manager.arg_callback(name, arg, arg)
        for grid_idx in itertools.product(*(range(dim) for dim in grid_dims)):
            if len(grid_idx) != len(nki_builder.grid_dims):
                raise ValueError(
                    "Grid index rank mismatch: got "
                    f"{len(grid_idx)}, expected {len(nki_builder.grid_dims)}"
                )
            nki_builder.grid_idx = [int(idx) for idx in grid_idx]
            if client_manager is not None:
                client_manager.grid_idx_callback(
                    tuple(int(idx) for idx in grid_idx) + (0,) * (3 - len(grid_idx))
                )
                if not client_manager.pre_run_callback(self.fn):
                    return
            self.fn(*kernel_args, **kwargs)
            if client_manager is not None and not client_manager.post_run_callback(
                self.fn
            ):
                return
