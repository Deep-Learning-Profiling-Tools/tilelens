from triton_viz.utils.traceback_utils import CODE_KEYS, get_code_key
import numpy as np

try:
    import neuronxcc.nki.language as nl
except (
    ModuleNotFoundError
) as exc:  # pragma: no cover - only hit when optional deps missing
    raise ModuleNotFoundError(
        "NeuronX dependencies are missing. Install triton-viz[nki] to enable the NKI interpreter."
    ) from exc
try:
    import neuronxcc.nki.isa as nisa
except (  # pragma: no cover - only hit when optional deps missing
    ModuleNotFoundError
) as exc:
    raise ModuleNotFoundError(
        "NeuronX dependencies are missing. Install triton-viz[nki] to enable the NKI interpreter."
    ) from exc
import inspect
import textwrap
from dataclasses import dataclass
from typing import Any

from ..frontend.nki_transform import transform_code
from ..masked_load_store import masked_load, masked_store


@dataclass
class _StorageState:
    identity: int
    version: int = 0


class ParDim:
    """Partition-dimension shape marker compatible with ``nl.par_dim``."""

    def __init__(self, value: int) -> None:
        self.value = int(value)
        if self.value <= 0:
            raise ValueError(f"par_dim must be positive, received {value}")

    def __int__(self) -> int:
        return self.value

    def __index__(self) -> int:
        return self.value

    def __repr__(self) -> str:
        return f"par_dim({self.value})"


def par_dim(value: int) -> ParDim:
    """Return a partition-dimension shape marker."""
    return ParDim(value)


def ds(start: int, size: int) -> slice:
    """Return a dynamic-slice equivalent accepted by NumPy indexing."""
    return slice(int(start), int(start) + int(size))



def array_byte_runs(a: np.ndarray, origin: int, max_runs: int = 512):
    """Exact merged half-open byte runs of view ``a`` that starts ``origin`` bytes into its root.

    A bounding interval treats a strided slice such as ``t[:, j, :]`` of a partition-major tile as
    covering every other slice; its exact footprint is one run per partition. Returns None when the
    view has more than ``max_runs`` runs or a negative stride (callers keep the bounding interval).
    """
    if not a.size:
        return ()
    item = int(a.dtype.itemsize)
    dims = [(int(n), int(st)) for n, st in zip(a.shape, a.strides) if int(n) > 1]
    if any(st < 0 for _, st in dims):
        return None
    run = item
    while dims and dims[-1][1] == run:
        run *= dims[-1][0]
        dims.pop()
    count = 1
    for n, _ in dims:
        count *= n
    if count > max_runs:
        return None
    offsets = np.zeros(1, dtype=np.int64)
    for n, st in dims:
        offsets = (offsets[:, None] + np.arange(n, dtype=np.int64)[None, :] * st).reshape(-1)
    runs: list[list[int]] = []
    for start in np.sort(origin + offsets).tolist():
        if runs and start <= runs[-1][1]:
            runs[-1][1] = max(runs[-1][1], start + run)
        else:
            runs.append([start, start + run])
    return tuple((lo, hi) for lo, hi in runs)

class MGridResult:
    """Small compatibility object exposing ``.p`` and ``.x`` index arrays."""

    def __init__(self, data: np.ndarray, p: "NDArray", x: "NDArray") -> None:
        self.data = data
        self.p = p
        self.x = x


class MGrid:
    """Compatibility stand-in for ``nl.mgrid`` used by Tilebench kernels."""

    def __getitem__(self, key: Any) -> MGridResult:
        mesh = np.mgrid[key]
        if getattr(mesh, "ndim", 0) < 2 or mesh.shape[0] != 2:
            raise ValueError("NKI mgrid simulation expects exactly two index axes")
        return MGridResult(
            mesh,
            NDArray(value=np.asarray(mesh[0]), name="mgrid_p"),
            NDArray(value=np.asarray(mesh[1]), name="mgrid_x"),
        )


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
        storage_state = kwargs.pop("_storage_state", None)
        self._data_ptr = None
        self.data = val
        self._storage_state = storage_state or _StorageState(self._root_data_ptr())

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

    def _root_data_ptr(self):
        if self.data is None or not hasattr(self.data, "ctypes"):
            return id(self)
        root = self.data
        while isinstance(getattr(root, "base", None), np.ndarray):
            root = root.base
        return int(root.ctypes.data)

    def storage_id(self):
        return self._storage_state.identity

    def tensor_version(self):
        return self._storage_state.version

    def mark_write(self):
        self._storage_state.version += 1
        return self._storage_state.version

    def byte_range(self):
        """Inclusive-exclusive byte range relative to the root allocation."""
        if self.data is None or not self.data.size:
            return (0, 0)
        origin = int(self.data.ctypes.data) - self.storage_id()
        low = high = origin
        for size, stride in zip(self.data.shape, self.data.strides):
            extent = (int(size) - 1) * int(stride)
            low += min(0, extent)
            high += max(0, extent)
        return (low, high + int(self.data.dtype.itemsize))

    def byte_runs(self, max_runs: int = 512):
        """Exact merged byte runs of this view relative to its root allocation (see ``array_byte_runs``)."""
        if self.data is None or not self.data.size:
            return ()
        return array_byte_runs(self.data, int(self.data.ctypes.data) - self.storage_id(), max_runs)

    def _view_or_copy(self, value, name):
        state = self._storage_state if np.shares_memory(value, self.data) else None
        return NDArray(value=value, name=name, _storage_state=state)

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

        # Apply the slicing to the underlying numpy array
        new_keys = [k.data if isinstance(k, NDArray) else k for k in keys]
        sliced_value = self.data[tuple(new_keys)]

        # Create a new NDArray with the sliced data
        return self._view_or_copy(sliced_value, f"{self.name}_slice")

    def __setitem__(self, keys, value):
        if not isinstance(keys, tuple):
            keys = (keys,)

        # Apply the slicing to the underlying numpy array
        new_keys = [k.data if isinstance(k, NDArray) else k for k in keys]
        self.data[tuple(new_keys)] = value.data
        version = self.mark_write()
        # The compute callback fires before Python executes ``dst[...] = value``.
        # Retarget that already-emitted mutable record to the physical SBUF
        # allocation/version written by the assignment, preserving SSA value
        # identity without inventing a separate zero-time copy event.
        record = getattr(value, "_trace_record", None)
        if record is not None:
            byte_range = self.byte_range()
            if hasattr(record, "output_storages"):
                record.output_ptrs = (self.data_ptr(),)
                record.output_storages = (self.storage_id(),)
                record.output_ranges = (byte_range,)
                record.output_versions = (version,)
            elif hasattr(record, "output_storage"):
                record.output_ptr = self.data_ptr()
                record.output_storage = self.storage_id()
                record.output_range = byte_range
                record.output_version = version

        return self

    def _binary_op(self, other, op_func, op_name, op_symbol):
        if isinstance(other, NDArray):
            return NDArray(
                value=op_func(self.data, other.data),
                name=f"{self.name}_{op_name}_{other.name}",
            )
        elif np.isscalar(other):
            return NDArray(
                value=op_func(self.data, other), name=f"{self.name}_{op_name}_scalar"
            )
        raise TypeError(
            f"Unsupported operand type(s) for {op_symbol}: 'NDArray' and '{type(other).__name__}'"
        )

    def __iadd__(self, other):
        """Mutating add, matching TensorE/PSUM accumulation semantics."""
        if isinstance(other, NDArray):
            np.add(self.data, other.data, out=self.data, casting="unsafe")
        elif np.isscalar(other):
            np.add(self.data, other, out=self.data, casting="unsafe")
        else:
            return NotImplemented
        version = self.mark_write()
        # The frontend emits the Dot record before Python performs the in-place
        # accumulation. Retarget the already-emitted record to the physical
        # destination storage/version written by ``+=``.
        record = getattr(other, "_trace_record", None)
        if record is not None and hasattr(record, "output_storage"):
            record.output_ptr = self.data_ptr()
            record.output_storage = self.storage_id()
            record.output_range = self.byte_range()
            record.output_version = version
        return self

    def _rbinary_op(self, other, op_func, op_name, op_symbol):
        if isinstance(other, NDArray):
            return NDArray(
                value=op_func(other.data, self.data),
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

    def reshape(self, *args, **kwargs):
        return self._view_or_copy(self.data.reshape(*args), f"{self.name}_reshape")

    def broadcast_to(self, *args, **kwargs):
        return self._view_or_copy(
            np.broadcast_to(self.data, *args), f"{self.name}_broadcast_to"
        )


class Builder:
    def __init__(self, grid_dims=None):
        # TODO: infinite grid dims for NKI
        self.grid_dims = grid_dims if grid_dims is not None else (1, 1, 1)
        self.grid_x = None
        self.grid_y = None
        self.grid_z = None
        self.fn = None
        self.shared_hbm_arrays = {}

    def set_grid_dim(self, *grid_dims):
        self.grid_dims = grid_dims

    def set_grid_idx(self, x, y, z):
        self.grid_x = x
        self.grid_y = y
        self.grid_z = z

    def ndarray(self, shape, dtype, *, buffer=None, name=None, **kwargs):
        if buffer == nl.shared_hbm:
            if name is None:
                # file name + function name + line number
                frame = inspect.currentframe().f_back
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
        else:
            return keys

    def load(self, src: NDArray, keys, *, mask=None, **kwargs):
        """Load array elements with masking for out-of-bounds errors."""
        # Convert NDArray to numpy array
        ndarray = src.data
        mask_value = getattr(mask, "data", mask) if mask is not None else None

        # Convert any NDArrays in keys to numpy arrays
        numpy_keys = self._convert_keys_to_numpy(keys)

        # Call the actual masked_load function
        result = masked_load(ndarray, numpy_keys, mask=mask_value)

        # Convert result back to NDArray
        return NDArray(
            value=result,
            name=f"{src.name}_masked_load",
            buffer=nl.sbuf,
            **kwargs,
        )

    def store(self, dst: NDArray, keys, value: NDArray, *, mask=None, **kwargs):
        """Store array elements with masking for out-of-bounds errors."""
        # Convert NDArrays to numpy arrays
        ndarray = dst.data
        value_array = value.data
        mask_value = getattr(mask, "data", mask) if mask is not None else None

        # Convert any NDArrays in keys to numpy arrays
        numpy_keys = self._convert_keys_to_numpy(keys)

        # Call the actual masked_store function
        masked_store(ndarray, numpy_keys, value_array, mask=mask_value)
        dst.mark_write()

        return dst

    def load_transpose2d(
        self, src: NDArray, keys=None, *, mask=None, dtype=None, **kwargs
    ):
        if keys is None:
            keys = tuple(slice(None) for _ in range(src.data.ndim))
        loaded = self.load(src, keys, mask=mask, **kwargs)
        value = loaded.data.astype(dtype) if dtype is not None else loaded.data
        return NDArray(value=value.T, name=f"{src.name}_load_transpose2d", **kwargs)

    @staticmethod
    def _tag(
        nd: "NDArray", api: str, engine: str, inputs, *, compute_mask_provided=False, compute_mask=None
    ) -> "NDArray":
        """Annotate a result NDArray so the tracer can emit an NkiCompute event."""
        nd._nki_api = api
        nd._nki_engine = engine
        nd._nki_inputs = tuple(i for i in inputs if isinstance(i, NDArray))
        nd._nki_compute_mask_provided = bool(compute_mask_provided or compute_mask is not None)
        nd._nki_compute_mask = compute_mask
        return nd

    def _unary_op(self, x: NDArray, np_func, op_name, **kwargs):
        # Activation-class unary ops run on the ScalarE (activation) engine.
        nd = NDArray(value=np_func(x.data), name=f"{x.name}_{op_name}", **kwargs)
        return self._tag(
            nd, op_name, "scalar", (x,),
            compute_mask=kwargs.get("mask"),
        )

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
        nd = NDArray(value=x.data * sigmoid_x, name=f"{x.name}_silu", **kwargs)
        return self._tag(nd, "silu", "scalar", (x,))

    def gelu(self, x: NDArray, **kwargs):
        # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_pi = np.sqrt(2 / np.pi)
        inner = sqrt_2_pi * (x.data + 0.044715 * np.power(x.data, 3))
        nd = NDArray(
            value=0.5 * x.data * (1 + np.tanh(inner)), name=f"{x.name}_gelu", **kwargs
        )
        return self._tag(nd, "gelu", "scalar", (x,))

    def sqrt(self, x: NDArray, **kwargs):
        return self._unary_op(x, np.sqrt, "sqrt", **kwargs)

    def abs(self, x: NDArray, **kwargs):
        return self._unary_op(x, np.abs, "abs", **kwargs)

    def log(self, x: NDArray, **kwargs):
        return self._unary_op(x, np.log, "log", **kwargs)

    def pow(self, x: NDArray, exponent, **kwargs):
        if isinstance(exponent, NDArray):
            nd = NDArray(
                value=np.power(x.data, exponent.data),
                name=f"{x.name}_pow_{exponent.name}",
                **kwargs,
            )
            return self._tag(nd, "pow", "scalar", (x, exponent))
        elif np.isscalar(exponent):
            nd = NDArray(
                value=np.power(x.data, exponent),
                name=f"{x.name}_pow_{exponent}",
                **kwargs,
            )
            return self._tag(nd, "pow", "scalar", (x,))
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
        dtype = kwargs.pop("dtype", None)
        value = x.data.astype(dtype) if dtype is not None else np.copy(x.data)
        nd = NDArray(value=value, name=f"{x.name}_copy", **kwargs)
        return self._tag(nd, "copy", "vector", (x,))

    def nc_transpose(self, src: NDArray, **kwargs):
        """Functional TensorE nc_transpose for the legacy Tilebench interpreter.

        ``nc_transpose`` operates on an on-chip tile and is frequently used as
        ``dst[...] = nisa.nc_transpose(src[...])``. The returned view shares the
        source storage, so later ``Dot``/load dependencies still resolve through
        the same allocation identity.
        """
        engine = kwargs.pop("engine", "tensor")
        nd = NDArray(
            value=src.data.T,
            name=f"{src.name}_nc_transpose",
            buffer=kwargs.pop("buffer", nl.psum),
            **kwargs,
        )
        nd._nki_transpose_engine = engine
        return nd

    def nc_matmul(
        self,
        stationary: NDArray,
        moving: NDArray,
        *,
        dst: NDArray | None = None,
        **kwargs,
    ) -> NDArray:
        """Functional ``nisa.nc_matmul``: computes ``stationary.T @ moving``."""
        lhs = np.asarray(stationary.data).T
        rhs = np.asarray(moving.data)
        value = lhs @ rhs
        if dst is not None:
            dst.data[...] = np.asarray(dst.data) + value
            return dst
        return NDArray(
            value=value,
            name=f"{stationary.name}_{moving.name}_nc_matmul",
            buffer=kwargs.pop("buffer", nl.psum),
        )

    def sum(self, x: NDArray, *args, mask=None, **kwargs):
        axis = args[0] if args else kwargs.pop("axis", None)
        keepdims = kwargs.pop("keepdims", kwargs.pop("keep_dims", False))
        return self._reduce(
            x,
            np.sum,
            "reduce_sum",
            axis=axis,
            keepdims=keepdims,
            dtype=kwargs.pop("dtype", None),
            mask=mask,
        )

    def square(self, x: NDArray, **kwargs):
        return self._unary_op(x, np.square, "square", **kwargs)

    def rsqrt(self, x: NDArray, **kwargs):
        return self._unary_op(x, lambda v: 1 / np.sqrt(v), "rsqrt", **kwargs)

    def multiply(self, x: NDArray, y: NDArray, **kwargs):
        if isinstance(y, NDArray):
            nd = NDArray(
                value=np.multiply(x.data, y.data),
                name=f"{x.name}_multiply_{y.name}",
                **kwargs,
            )
            return self._tag(
                nd, "multiply", "vector", (x, y),
                compute_mask=kwargs.get("mask"),
            )
        elif np.isscalar(y):
            nd = NDArray(
                value=np.multiply(x.data, y),
                name=f"{x.name}_multiply_scalar",
                **kwargs,
            )
            return self._tag(
                nd, "multiply", "vector", (x,),
                compute_mask=kwargs.get("mask"),
            )
        else:
            raise TypeError(f"Unsupported type for multiply: {type(y)}")

    # ------------------------------------------------------------------
    # Additional nl.* ops needed by real Tilebench kernels (softmax,
    # rmsnorm, layernorm, ...). These give correct NumPy semantics so the
    # exact same @nki.jit kernel that runs on hardware can also be traced.
    # ------------------------------------------------------------------
    @staticmethod
    def _as_np(value):
        """Return the raw NumPy payload for an NDArray or a passthrough scalar."""
        return value.data if isinstance(value, NDArray) else value

    def _binary(self, x, y, np_func, op_name, *, dtype=None, mask=None, **_kwargs):
        result = np_func(self._as_np(x), self._as_np(y))
        if dtype is not None:
            result = result.astype(dtype)
        xn = x.name if isinstance(x, NDArray) else "scalar"
        yn = y.name if isinstance(y, NDArray) else "scalar"
        nd = NDArray(value=result, name=f"{xn}_{op_name}_{yn}")
        return self._tag(
            nd, op_name, "vector", (x, y),
            compute_mask=mask,
        )

    def add(self, x, y, *, dtype=None, mask=None, **kwargs):
        return self._binary(x, y, np.add, "add", dtype=dtype, mask=mask)

    def subtract(self, x, y, *, dtype=None, mask=None, **kwargs):
        return self._binary(x, y, np.subtract, "subtract", dtype=dtype, mask=mask)

    def divide(self, x, y, *, dtype=None, mask=None, **kwargs):
        return self._binary(x, y, np.divide, "divide", dtype=dtype, mask=mask)

    def maximum(self, x, y, *, dtype=None, mask=None, **kwargs):
        return self._binary(x, y, np.maximum, "maximum", dtype=dtype, mask=mask)

    def minimum(self, x, y, *, dtype=None, mask=None, **kwargs):
        return self._binary(x, y, np.minimum, "minimum", dtype=dtype, mask=mask)

    def greater(self, x, y, *, dtype=None, mask=None, **kwargs):
        return self._binary(x, y, np.greater, "greater", dtype=dtype, mask=mask)

    @staticmethod
    def _reduction_identity(dtype: np.dtype, op_name: str):
        """Return the value used by a masked-out reduction lane."""
        dtype = np.dtype(dtype)
        if op_name in {"reduce_sum", "mean"}:
            return dtype.type(0)
        if np.issubdtype(dtype, np.bool_):
            return False if op_name == "max" else True
        if np.issubdtype(dtype, np.floating):
            return -np.inf if op_name == "max" else np.inf
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            return info.min if op_name == "max" else info.max
        # NumPy does not classify ml_dtypes.bfloat16 as np.floating, but it
        # supports infinities and the same reduction identities.
        if dtype.name == "bfloat16":
            return dtype.type(-np.inf if op_name == "max" else np.inf)
        raise TypeError(f"Unsupported masked reduction dtype: {dtype}")

    def _reduce(
        self,
        x: NDArray,
        np_func,
        op_name,
        *,
        axis=None,
        keepdims=False,
        dtype=None,
        mask=None,
        **kwargs,
    ):
        data = x.data
        active_mask = None
        if mask is not None:
            active_mask = np.broadcast_to(
                np.asarray(self._as_np(mask), dtype=bool), data.shape
            )
        if op_name == "mean" and active_mask is not None:
            summed = np.sum(
                np.where(active_mask, data, self._reduction_identity(data.dtype, op_name)),
                axis=axis,
                keepdims=keepdims,
            )
            count = np.sum(active_mask, axis=axis, keepdims=keepdims)
            result = np.divide(
                summed,
                count,
                out=np.full(np.shape(summed), np.nan, dtype=np.result_type(summed, float)),
                where=count != 0,
            )
        else:
            if active_mask is not None:
                data = np.where(
                    active_mask, data, self._reduction_identity(data.dtype, op_name)
                )
            result = np_func(data, axis=axis, keepdims=keepdims)
        if dtype is not None:
            result = result.astype(dtype)
        nd = NDArray(value=result, name=f"{x.name}_{op_name}")
        inputs = (x, mask) if isinstance(mask, NDArray) else (x,)
        return self._tag(
            nd, op_name, "vector", inputs,
            compute_mask=mask,
        )

    def max(self, x: NDArray, *, axis=None, keepdims=False, dtype=None, mask=None, **kwargs):
        return self._reduce(x, np.max, "max", axis=axis, keepdims=keepdims, dtype=dtype, mask=mask)

    def min(self, x: NDArray, *, axis=None, keepdims=False, dtype=None, mask=None, **kwargs):
        return self._reduce(x, np.min, "min", axis=axis, keepdims=keepdims, dtype=dtype, mask=mask)

    def mean(self, x: NDArray, *, axis=None, keepdims=False, dtype=None, mask=None, **kwargs):
        return self._reduce(x, np.mean, "mean", axis=axis, keepdims=keepdims, dtype=dtype, mask=mask)

    def where(self, condition, x, y, *, dtype=None, **kwargs):
        result = np.where(self._as_np(condition), self._as_np(x), self._as_np(y))
        if dtype is not None:
            result = result.astype(dtype)
        nd = NDArray(value=result, name="where")
        return self._tag(nd, "where", "vector", (condition, x, y))

    def full(self, shape, fill_value, dtype=None, *, buffer=None, name=None, **kwargs):
        value = np.full(shape, self._as_np(fill_value), dtype=dtype)
        return self.ndarray(shape, value.dtype, buffer=buffer, name=name, value=value)

    def broadcast_to(self, x: NDArray, shape, **kwargs):
        nd = x.broadcast_to(shape)
        return self._tag(nd, "broadcast_to", "vector", (x,))

    def static_cast(self, x: NDArray, dtype, **kwargs):
        nd = NDArray(value=x.data.astype(dtype), name=f"{x.name}_static_cast")
        return self._tag(nd, "static_cast", "vector", (x,))

    def range(self, stop):
        return range(stop)


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
    _set_attr(nl, "static_range", nki_builder.range)
    _set_attr(nl, "sequential_range", nki_builder.range)
    _set_attr(nl, "par_dim", par_dim)
    _set_attr(nl, "ds", ds)
    _set_attr(nl, "zeros", nki_builder.zeros)
    _set_attr(nl, "mgrid", MGrid())
    _set_attr(nl, "matmul", nki_builder.matmul)
    _set_attr(nl, "copy", nki_builder.copy)
    _set_attr(nl, "sum", nki_builder.sum)
    _set_attr(nl, "square", nki_builder.square)
    _set_attr(nl, "rsqrt", nki_builder.rsqrt)
    _set_attr(nl, "multiply", nki_builder.multiply)

    # Reductions and elementwise binaries used by real Tilebench kernels.
    _set_attr(nl, "max", nki_builder.max)
    _set_attr(nl, "min", nki_builder.min)
    _set_attr(nl, "mean", nki_builder.mean)
    _set_attr(nl, "add", nki_builder.add)
    _set_attr(nl, "subtract", nki_builder.subtract)
    _set_attr(nl, "divide", nki_builder.divide)
    _set_attr(nl, "maximum", nki_builder.maximum)
    _set_attr(nl, "minimum", nki_builder.minimum)
    _set_attr(nl, "greater", nki_builder.greater)
    _set_attr(nl, "where", nki_builder.where)
    _set_attr(nl, "full", nki_builder.full)
    _set_attr(nl, "broadcast_to", nki_builder.broadcast_to)
    _set_attr(nl, "static_cast", nki_builder.static_cast)

    # attention-specific
    _set_attr(nl, "load_transpose2d", nki_builder.load_transpose2d)
    _set_attr(nisa, "nc_matmul", nki_builder.nc_matmul)
    _set_attr(nisa, "nc_transpose", nki_builder.nc_transpose)
    # nisa.affine_select
    # nl.tensor_reduce
    # nisa.activation
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


class NKIInterpretedFunction:
    def __init__(self, fn):
        self.fn = fn

    def run(self, *args, **kwargs):
        grid_dims = kwargs.pop(
            "grid", (1, 1, 1)
        )  # Remove grid from kwargs to avoid passing it to the function
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
        nki_builder.fn = self.fn

        kwargs.pop("warmup", None)  # Remove warmup from kwargs if it exists
        client_manager = kwargs.pop(
            "client_manager", None
        )  # Remove client_manager from kwargs if it exists

        # Call grid_callback once before grid execution (similar to Triton)
        if client_manager is not None:
            client_manager.grid_callback(grid_dims)

        # Apply the NKI frontend AST transformer.
        if hasattr(self.fn, "__code__"):
            # Get the source code of the function (stripped of leading indents in case it was defined in scope)
            source_code = textwrap.dedent(inspect.getsource(self.fn))
            # Transform the source code using the AST transformer
            transformed_code = transform_code(source_code)
            # Create a new function from the transformed code
            exec_globals = self.fn.__globals__.copy()
            import random
            import string
            import os

            rand_str = "".join(
                random.choices(string.ascii_letters + string.digits, k=16)
            )
            os.makedirs("/tmp/triton-viz", exist_ok=True)
            filename = f"/tmp/triton-viz/{rand_str}.py"
            with open(filename, "w") as f:
                f.write(transformed_code)
            code_obj = compile(transformed_code, filename=filename, mode="exec")
            exec(code_obj, exec_globals)
            self.fn = exec_globals[self.fn.__name__]
            CODE_KEYS.add(get_code_key(self.fn))  # trace rewritten function for clients

        # convert args to NDArray if they are not already
        args = [
            NDArray(value=arg) if isinstance(arg, np.ndarray) else arg for arg in args
        ]

        name_args = inspect.getcallargs(self.fn, *args)
        call_args = {}
        for name, arg in name_args.items():
            call_args[name] = arg
            ret = arg
            client_manager.arg_callback(name, arg, ret)

        ret_val = None
        for x in range(grid_dims[0]):
            for y in range(grid_dims[1]):
                for z in range(grid_dims[2]):
                    nki_builder.set_grid_idx(x, y, z)

                    # Call grid_idx_callback for each grid iteration (similar to Triton)
                    if client_manager is not None:
                        client_manager.grid_idx_callback((x, y, z))

                    if not client_manager.pre_run_callback(self.fn):
                        return ret_val
                    ret_val = self.fn(*args, **kwargs)
                    if not client_manager.post_run_callback(self.fn):
                        return ret_val
        return ret_val
