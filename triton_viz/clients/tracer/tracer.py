from collections.abc import Callable

from ...core.client import Client
from ...core.callbacks import OpCallbacks, ForLoopCallbacks
from ...core.data import (
    Op,
    Load,
    Store,
    Transfer,
    DmaTranspose,
    BinaryOp,
    ReduceSum,
    Dot,
    Grid,
    Allocate,
    NkiCompute,
    TensorTranspose,
)
from ...utils.traceback_utils import extract_user_frames
from triton_viz.core.masked_load_store import masked_load
import numpy as np


def _convert_grid_idx(grid_idx) -> tuple[int, int, int] | None:
    if grid_idx is None:
        return grid_idx

    grid_idx = (grid_idx, 0, 0) if isinstance(grid_idx, int) else grid_idx
    if len(grid_idx) == 1:
        grid_idx = (grid_idx[0], 0, 0)
    elif len(grid_idx) == 2:
        grid_idx = (grid_idx[0], grid_idx[1], 0)
    return grid_idx


def _moved_bytes(mask: np.ndarray, tensor) -> int:
    """Return logical bytes transferred by a masked load or store.

    This matches the profiler's byte-counting semantics: every active mask
    element contributes one tensor element. It is a logical traffic count, not
    a model of cache-line fetches, coalescing, or repeated-address deduplication.
    """
    return int(np.count_nonzero(mask)) * int(tensor.element_size())


class Tracer(Client):
    NAME = "tracer"

    def __init__(
        self,
        callpath: bool = True,
        grid_idx: tuple[int] | int | None = None,
        compute_masks: bool = False,
        allocation_metadata: bool = False,
        multi_output_metadata: bool = False,
        scalar_state_metadata: bool = False,
        hardware_engine_metadata: bool = False,
        sdk_setup_metadata: bool = False,
    ):
        super().__init__()  # Initialize parent class
        self.callpath = callpath
        self.compute_masks = compute_masks
        self.allocation_metadata = allocation_metadata
        self.multi_output_metadata = multi_output_metadata or scalar_state_metadata
        self.scalar_state_metadata = scalar_state_metadata
        self.hardware_engine_metadata = hardware_engine_metadata
        self.sdk_setup_metadata = sdk_setup_metadata
        self.grid_idx = _convert_grid_idx(grid_idx)
        self.records: list = []
        self.tensors: list = []
        self.sample = True

    def _get_tensor(self, data_ptr):
        # From a give ptr, get where the original tensor is stored
        # Tensors have been sorted by ptr
        ret_idx = 0
        for i in range(len(self.tensors)):
            if data_ptr < self.tensors[i].data_ptr():
                break
            ret_idx = i
        return self.tensors[ret_idx]

    def pre_run_callback(self, fn: Callable) -> bool:
        return True

    def post_run_callback(self, fn: Callable) -> bool:
        return True

    def pre_warmup_callback(self, jit_fn, *args, **kwargs) -> bool:
        return False

    def post_warmup_callback(self, jit_fn, ret) -> None:
        pass

    def arg_callback(self, name, arg, arg_cvt):
        if hasattr(arg, "data_ptr"):
            self.tensors.append(arg)

    def grid_idx_callback(self, grid_idx: tuple[int, ...]):
        if self.grid_idx is not None and grid_idx != self.grid_idx:
            self.sample = False
        else:
            self.sample = True

        # Create a Grid record for this grid index
        with self._lock_context():
            self.records.append(Grid(idx=grid_idx))

    def grid_callback(self, grid: tuple[int, ...]):
        self.tensors = sorted(self.tensors, key=lambda x: x.data_ptr())

    def register_op_callback(self, op_type: type[Op]) -> OpCallbacks:
        def post_allocate_callback(ret):
            assert hasattr(ret, "data")
            self.tensors.append(ret)

        def _convert_keys_to_numpy(keys):
            """Convert any NDArrays in keys to numpy arrays."""
            if isinstance(keys, (tuple, list)):
                return tuple(_convert_keys_to_numpy(k) for k in keys)
            return keys.data if hasattr(keys, "data") else keys

        def _storage(value):
            return (
                int(value.storage_id())
                if hasattr(value, "storage_id")
                else int(value.data_ptr())
            )

        def _range(value):
            if hasattr(value, "byte_range"):
                return tuple(int(item) for item in value.byte_range())
            size = int(np.prod(value.shape)) * int(value.element_size())
            return (0, size)

        def _version(value):
            return int(value.tensor_version()) if hasattr(value, "tensor_version") else 0

        def _runs(value):
            """(root allocation id, exact byte runs) of an operand view, or None.

            The beta2 NDArray has no storage identity: its recorded storage is the view's own
            pointer and its range (0, nbytes). The root of the NumPy base chain gives the
            allocation that DMA transfers are keyed by.
            """
            if hasattr(value, "byte_runs") and hasattr(value, "storage_id"):
                return (int(value.storage_id()), value.byte_runs())
            data = getattr(value, "data", None)
            if not isinstance(data, np.ndarray):
                return None
            from triton_viz.core.simulation.nki import array_byte_runs
            root = data
            while isinstance(getattr(root, "base", None), np.ndarray):
                root = root.base
            origin = int(data.ctypes.data) - int(root.ctypes.data)
            return (int(root.ctypes.data), array_byte_runs(data, origin))

        def _allocation(value):
            if not hasattr(value, "data_ptr") or not hasattr(value, "shape"):
                return None
            root = value
            while getattr(root, "_parent", None) is not None:
                root = root._parent
            memory = getattr(root, "buffer", "unknown")
            memory = str(getattr(memory, "buffer", memory))
            axis = getattr(root, "partition_axis", None)
            if axis is None and memory in ("sbuf", "psum") and len(root.shape) == 2:
                axis = 0
            return {
                "storage": _storage(root),
                "view_storage": _storage(value),
                "shape": list(root.shape),
                "partition_axis": axis,
                "dtype": str(getattr(root, "dtype", "unknown")),
                "memory": memory,
            }

        def _attach_runs(rec, inputs, outputs):
            # Exact byte runs of each operand view, parallel to input_ranges / output ranges
            # (used only by CostModel.exact_tile_ranges).
            rec.input_runs = tuple(_runs(value) for value in inputs)
            rec.output_runs = tuple(_runs(value) for value in outputs)
            if self.allocation_metadata:
                rec.input_allocations = tuple(_allocation(value) for value in inputs)
                rec.output_allocations = tuple(_allocation(value) for value in outputs)

        @self.lock_fn
        def post_load_callback(ret, ptr, mask, keys):
            if not self.sample:
                return

            if keys is None:  # i.e. for triton, ptr = pointer + offsets
                first_ptr = np.reshape(ptr.data, (-1))[0]
                tensor = self._get_tensor(first_ptr)
                offsets = ptr.data - tensor.data_ptr()
                mask_data = (
                    mask.data
                    if mask is not None
                    else np.ones(np.asarray(offsets).shape, dtype=bool)
                )
            else:
                keys = _convert_keys_to_numpy(keys)
                if mask is None:
                    offsets = masked_load(ptr.get_offsets().data, keys)
                    mask_data = np.ones(np.asarray(offsets).shape, dtype=bool)
                else:
                    mask_data = mask.data
                    offsets = masked_load(
                        ptr.get_offsets().data, keys, mask=mask_data
                    )
                tensor = ptr

            rec = Load(
                tensor.data_ptr(),
                offsets,
                mask_data,
                bytes=_moved_bytes(mask_data, tensor),
                src_storage=_storage(tensor),
                src_version=_version(tensor),
                src_dtype=str(tensor.dtype),
                dst_ptr=int(ret.data_ptr()),
                dst_storage=_storage(ret),
                dst_range=_range(ret),
                dst_version=_version(ret),
                mask_provided=mask is not None,
            )
            rec.call_path = extract_user_frames(num_frames=1)
            self.records.append(rec)

        @self.lock_fn
        def pre_store_callback(ptr, mask, keys, value):
            if not self.sample:
                return

            if keys is None:  # i.e. for triton, ptr = pointer + offsets, so keys=None
                first_ptr = np.reshape(ptr.data, (-1))[0]
                tensor = self._get_tensor(first_ptr)
                offsets = ptr.data - tensor.data_ptr()
                mask_data = mask.data
            else:
                keys = _convert_keys_to_numpy(keys)
                if mask is None:
                    offsets = masked_load(ptr.get_offsets().data, keys)
                    mask_data = np.ones_like(offsets).astype(bool)
                else:
                    mask_data = mask.data
                    offsets = masked_load(ptr.get_offsets().data, keys, mask=mask_data)
                tensor = ptr

            rec = Store(
                tensor.data_ptr(),
                offsets,
                mask_data,
                bytes=_moved_bytes(mask_data, tensor),
                src_ptr=int(value.data_ptr()),
                src_storage=_storage(value),
                src_range=_range(value),
                src_version=_version(value),
                src_dtype=str(value.dtype),
                dst_storage=_storage(tensor),
                dst_version=_version(tensor) + 1,
                mask_provided=mask is not None,
            )
            rec.call_path = extract_user_frames(num_frames=1)
            self.records.append(rec)

        @self.lock_fn
        def pre_transfer_callback(src, dst, mem_src, mem_dst, dma_pattern="copy"):
            # TODO: currently only works with NKI Beta 2. Make DSL-agnostic by
            # making tensor interface so we can safely call data_ptr/data/...
            if not self.sample:
                return

            def _get_offsets(ptr):
                strides = tuple(
                    int(stride) * int(ptr.element_size()) for stride in ptr.stride()
                )
                offsets = np.int64(0)
                for dim_size, stride in zip(ptr.shape, strides):
                    offsets = np.expand_dims(offsets, -1) + (
                        np.arange(dim_size, dtype=np.int64) * stride
                    )
                return offsets

            def _base_tensor(ptr):
                base = ptr
                while getattr(base, "_parent", None) is not None:
                    base = base._parent
                return (
                    base
                    if hasattr(base, "data_ptr")
                    else self._get_tensor(ptr.data_ptr())
                )

            src_tensor = _base_tensor(src)
            dst_tensor = _base_tensor(dst)
            src_offsets = _get_offsets(src) + (src.data_ptr() - src_tensor.data_ptr())
            dst_offsets = _get_offsets(dst) + (dst.data_ptr() - dst_tensor.data_ptr())
            rec = Transfer(
                src_ptr=src_tensor.data_ptr(),
                dst_ptr=dst_tensor.data_ptr(),
                src_offsets=src_offsets,
                dst_offsets=dst_offsets,
                mem_src=mem_src,
                mem_dst=mem_dst,
                bytes=np.prod(dst.shape) * dst.element_size(),
                src_partition_axis=getattr(src, "partition_axis", None),
                dst_partition_axis=getattr(dst, "partition_axis", None),
                dma_pattern=str(dma_pattern),
            )
            if self.allocation_metadata:
                _attach_runs(rec, (src,), (dst,))
            if src.element_size() != dst.element_size():
                rec.src_bytes = int(np.prod(src.shape)) * src.element_size()
                rec.dst_bytes = int(np.prod(dst.shape)) * dst.element_size()
            rec.call_path = extract_user_frames(num_frames=1)
            self.records.append(rec)

        @self.lock_fn
        def post_reduce_sum_callback(
            ret, input, axis=None, keep_dims=False, api_op="reduce_sum"
        ):
            if not self.sample:
                return
            input_data = getattr(getattr(input, "handle", None), "data", None)
            if input_data is None:
                input_data = getattr(input, "data", None)
            output_data = getattr(getattr(ret, "handle", None), "data", None)
            if output_data is None:
                output_data = getattr(ret, "data", None)
            input_shape = input_data.shape if input_data is not None else ()
            output_shape = output_data.shape if output_data is not None else ()
            input_ptrs = (
                (int(input.data_ptr()),) if hasattr(input, "data_ptr") else ()
            )
            output_ptr = int(ret.data_ptr()) if hasattr(ret, "data_ptr") else None
            rec = ReduceSum(
                input_shape,
                axis,
                keep_dims,
                output_shape,
                input_ptrs=input_ptrs,
                output_ptr=output_ptr,
                input_storages=(_storage(input),),
                input_ranges=(_range(input),),
                input_versions=(_version(input),),
                output_storage=_storage(ret),
                output_range=_range(ret),
                output_version=_version(ret),
                api_op=str(api_op or "reduce_sum"),
                input_dtypes=(
                    (str(input.dtype),) if hasattr(input, "dtype") else ()
                ),
                output_dtype=str(ret.dtype) if hasattr(ret, "dtype") else None,
            )
            rec.call_path = extract_user_frames(num_frames=1)
            ret._trace_record = rec
            _attach_runs(rec, (input,), (ret,))
            self.records.append(rec)

        @self.lock_fn
        def post_dot_callback(ret, input, other, transpose_input=False):
            if not self.sample:
                return
            # ``transpose_input`` means the matmul consumes ``input`` transposed
            # (stationary.T @ moving). We transpose only the recorded shape/value
            # for rendering; ``input.data_ptr()`` is kept as the original SBUF
            # address so the dependency on the producing transfer still matches.
            input_data = input.data.T if transpose_input else input.data
            input_shape = input_data.shape
            other_shape = other.data.shape
            ret_shape = ret.data.shape
            # Pass input/other raw arrays so draw.py can render MatMul
            # Capture operand/destination base pointers when the frontend exposes
            # them so the cost model can resolve TensorE's DMA dependencies by
            # exact pointer matching (see Dot.input_ptrs/output_ptr).
            input_ptrs = tuple(
                int(value.data_ptr())
                for value in (input, other)
                if hasattr(value, "data_ptr")
            )
            output_ptr = int(ret.data_ptr()) if hasattr(ret, "data_ptr") else None
            rec = Dot(
                input_shape,
                other_shape,
                ret_shape,
                input_data,
                other.data,
                input_ptrs=input_ptrs,
                output_ptr=output_ptr,
                input_storages=tuple(_storage(value) for value in (input, other)),
                input_ranges=tuple(_range(value) for value in (input, other)),
                input_versions=tuple(_version(value) for value in (input, other)),
                output_storage=_storage(ret),
                output_range=_range(ret),
                output_version=_version(ret),
                input_dtypes=tuple(
                    str(value.dtype)
                    for value in (input, other)
                    if hasattr(value, "dtype")
                ),
                output_dtype=str(ret.dtype) if hasattr(ret, "dtype") else None,
            )
            rec.call_path = extract_user_frames(num_frames=1)
            ret._trace_record = rec
            _attach_runs(rec, (input, other), (ret,))
            self.records.append(rec)

        @self.lock_fn
        def post_binary_callback(ret, input, other, op, dst):
            if not self.sample:
                return
            input_shape = tuple(getattr(input, "shape", ()))
            other_shape = tuple(getattr(other, "shape", ()))
            output_shape = tuple(getattr(ret, "shape", getattr(dst, "shape", ())))
            op_name = getattr(op, "name", getattr(op, "__name__", str(op)))
            rec = BinaryOp(
                op=str(op_name),
                input_shape=input_shape,
                output_shape=output_shape,
                other_shape=other_shape,
                input_ptrs=tuple(
                    int(value.data_ptr())
                    for value in (input, other)
                    if hasattr(value, "data_ptr")
                ),
                output_ptr=(
                    int(dst.data_ptr()) if hasattr(dst, "data_ptr") else None
                ),
                input_storages=tuple(
                    _storage(value)
                    for value in (input, other)
                    if hasattr(value, "data_ptr")
                ),
                input_ranges=tuple(
                    _range(value)
                    for value in (input, other)
                    if hasattr(value, "data_ptr")
                ),
                input_versions=tuple(
                    _version(value)
                    for value in (input, other)
                    if hasattr(value, "data_ptr")
                ),
                output_storage=(
                    _storage(dst) if hasattr(dst, "data_ptr") else None
                ),
                output_range=(_range(dst) if hasattr(dst, "data_ptr") else None),
                output_version=(
                    _version(dst) if hasattr(dst, "data_ptr") else None
                ),
            )
            rec.call_path = extract_user_frames(num_frames=1)
            ret._trace_record = rec
            _attach_runs(rec, [v for v in (input, other) if hasattr(v, "data_ptr")], [dst] if hasattr(dst, "data_ptr") else [])
            self.records.append(rec)

        @self.lock_fn
        def post_nki_compute_callback(ret, *args, **kwargs):
            if not self.sample:
                return
            # The frontend builder _tag method attaches the true op info to the
            # returned NDArray. The adapter passes ret to the post_callback.
            if not hasattr(ret, "_nki_api"):
                return
            inputs = getattr(ret, "_nki_inputs", ())
            input_ptrs = tuple(int(x.data_ptr()) for x in inputs if hasattr(x, "data_ptr"))
            outputs = (ret,) + (tuple(getattr(ret, "_nki_extra_outputs", ())) if self.multi_output_metadata else ())
            output_ptrs = tuple(int(x.data_ptr()) for x in outputs if hasattr(x, "data_ptr"))
            input_shapes = tuple(tuple(x.shape) for x in inputs if hasattr(x, "shape"))
            output_shape = tuple(ret.shape) if hasattr(ret, "shape") else ()
            output_shapes = tuple(tuple(x.shape) for x in outputs if hasattr(x, "shape"))
            input_dtypes = tuple(str(x.dtype) for x in inputs if hasattr(x, "dtype"))
            output_dtype = str(ret.dtype) if hasattr(ret, "dtype") else ""
            compute_attrs = {
                "compute_mask_provided": bool(getattr(ret, "_nki_compute_mask_provided", False))
            }
            if self.sdk_setup_metadata and hasattr(ret, "_nki_sdk_setup"):
                compute_attrs["sdk_setup"] = ret._nki_sdk_setup
            if self.scalar_state_metadata and hasattr(ret, "_nki_scalar_state"):
                compute_attrs["scalar_state"] = ret._nki_scalar_state
            if len(outputs) > 1:
                compute_attrs.update(
                    output_dtypes=[str(x.dtype) for x in outputs],
                    output_roles=["primary"] + ["reduction"] * (len(outputs) - 1),
                    reduction_op=getattr(ret, "_nki_reduction_op", None),
                )
            if ret._nki_api == "memset" and getattr(ret, "_nki_memory_initializer", False):
                memory = getattr(ret, "buffer", "unknown")
                allocation = ret
                while getattr(allocation, "_parent", None) is not None:
                    allocation = allocation._parent
                output_axis = getattr(ret, "partition_axis", None)
                allocation_axis = getattr(allocation, "partition_axis", None)
                if output_axis is None and len(ret.shape) == 2:
                    output_axis = 0
                if allocation_axis is None and len(allocation.shape) == 2:
                    allocation_axis = 0
                compute_attrs.update(
                    allocation_storage=_storage(allocation),
                    allocation_shape=list(allocation.shape),
                    allocation_partition_axis=allocation_axis,
                    memory_initializer=True,
                    output_memory=str(getattr(memory, "buffer", memory)),
                    output_partition_axis=output_axis,
                    fill_value=ret._nki_fill_value,
                )
            mask = getattr(ret, "_nki_compute_mask", None)
            if self.compute_masks and mask is not None:
                from ...core.compute_mask import compute_mask_geometry
                try:
                    domain = np.broadcast_shapes(*input_shapes) if input_shapes else output_shape
                except ValueError:
                    compute_attrs["compute_mask_geometry"] = {"status": "unsupported_input_domain"}
                else:
                    value = mask.data if hasattr(mask, "data_ptr") else mask
                    compute_attrs["compute_mask_geometry"] = compute_mask_geometry(value, domain)
            rec = NkiCompute(
                api_op=str(ret._nki_api),
                engine=str(getattr(ret, "_nki_hardware_engine", getattr(ret, "_nki_engine", "vector"))
                           if self.hardware_engine_metadata else getattr(ret, "_nki_engine", "vector")),
                input_ptrs=input_ptrs,
                output_ptrs=output_ptrs,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                input_dtypes=input_dtypes,
                output_dtype=output_dtype,
                attrs=compute_attrs,
                input_storages=tuple(_storage(x) for x in inputs),
                input_ranges=tuple(_range(x) for x in inputs),
                input_versions=tuple(_version(x) for x in inputs),
                output_storages=tuple(_storage(x) for x in outputs),
                output_ranges=tuple(_range(x) for x in outputs),
                output_versions=tuple(_version(x) for x in outputs),
            )
            rec.call_path = extract_user_frames(num_frames=1)
            for output in outputs:
                output._trace_record = rec
            _attach_runs(rec, inputs, outputs)
            self.records.append(rec)

        @self.lock_fn
        def post_tensor_transpose_callback(ret, data, engine="tensor"):
            if not self.sample:
                return
            rec = TensorTranspose(
                input_shape=tuple(data.shape),
                output_shape=tuple(ret.shape),
                input_ptrs=(int(data.data_ptr()),),
                output_ptr=int(ret.data_ptr()) if hasattr(ret, "data_ptr") else None,
                input_storages=(_storage(data),),
                input_ranges=(_range(data),),
                input_versions=(_version(data),),
                output_storage=_storage(ret),
                output_range=_range(ret),
                output_version=_version(ret),
                input_dtypes=(str(data.dtype),),
                output_dtype=str(ret.dtype),
                engine=str(getattr(engine, "name", engine) or "tensor"),
            )
            rec.call_path = extract_user_frames(num_frames=1)
            ret._trace_record = rec
            _attach_runs(rec, (data,), (ret,))
            self.records.append(rec)

        callbacks = {
            Allocate: OpCallbacks(after_callback=post_allocate_callback),
            Load: OpCallbacks(after_callback=post_load_callback),
            Store: OpCallbacks(before_callback=pre_store_callback),
            Transfer: OpCallbacks(before_callback=pre_transfer_callback),
            DmaTranspose: OpCallbacks(before_callback=pre_transfer_callback),
            ReduceSum: OpCallbacks(after_callback=post_reduce_sum_callback),
            Dot: OpCallbacks(after_callback=post_dot_callback),
            BinaryOp: OpCallbacks(after_callback=post_binary_callback),
            NkiCompute: OpCallbacks(after_callback=post_nki_compute_callback),
            TensorTranspose: OpCallbacks(
                after_callback=post_tensor_transpose_callback
            ),
        }
        return callbacks.get(op_type, OpCallbacks())

    def register_for_loop_callback(self):
        return ForLoopCallbacks()

    @property
    def sample(self) -> bool:
        return self._get_thread_local("sample", True)

    @sample.setter
    def sample(self, value: bool) -> None:
        self._set_thread_local("sample", value)

    def finalize(self) -> list:
        with self._lock_context():
            self.tensors.clear()
            return self.records
