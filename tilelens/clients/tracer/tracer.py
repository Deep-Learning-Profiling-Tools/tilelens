from collections.abc import Callable

from ...core.client import Client
from ...core.callbacks import OpCallbacks, ForLoopCallbacks
from ...core.data import (
    Op,
    Load,
    Store,
    Transfer,
    ReduceSum,
    Dot,
    Grid,
    Allocate,
    AtomicCas,
    AtomicRMW,
    IntToPtr,
)
from ...utils.traceback_utils import extract_user_frames
from tilelens.core.masked_load_store import masked_load
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


def _storage_ranges(name, arg):
    """Yield (name, tensor, start, end) for each torch storage behind a kernel arg."""
    if isinstance(arg, (tuple, list)):
        for i, item in enumerate(arg):
            yield from _storage_ranges(f"{name}[{i}]", item)
        return
    # triton.reinterpret wrappers and host TensorDescriptors keep the tensor in `base`
    tensor = arg if hasattr(arg, "untyped_storage") else getattr(arg, "base", None)
    if hasattr(tensor, "untyped_storage"):
        storage = tensor.untyped_storage()
        if storage.nbytes():
            start = storage.data_ptr()
            yield name, tensor, start, start + storage.nbytes()


class Tracer(Client):
    NAME = "tracer"

    def __init__(
        self,
        callpath: bool = True,
        grid_idx: tuple[int] | int | None = None,
    ):
        super().__init__()  # Initialize parent class
        self.callpath = callpath
        self.grid_idx = _convert_grid_idx(grid_idx)
        self.records: list = []
        self.tensors: list = []
        # (arg name, tensor, start, end) of the storages behind the kernel args
        self.storages: list = []
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

    def _check_in_bounds(self, op_name: str, ptr, mask) -> None:
        """
        Raise IndexError before an access runs off the storage of a tensor arg.

        Triton's interpreter performs loads, stores and atomics on raw host
        memory, so an active lane past the end of an allocation reads or
        corrupts whatever follows it (glibc heap metadata, for instance, which
        later crashes the process at exit). The tracer only judges memory it
        was given: when any lane of the pointer block, masked-off lanes
        included, touches a tensor arg's storage, every active lane must lie
        fully inside some known storage. Accesses that never touch a known
        storage, and every access of a program after it casts an integer to a
        pointer (pointer tables), are left alone, so this is a best-effort
        guard; the Sanitizer is the complete out-of-bounds check.
        """
        element_ty = getattr(getattr(ptr, "dtype", None), "element_ty", None)
        data = getattr(ptr, "data", None)
        if element_ty is None or not isinstance(data, np.ndarray) or not self.storages:
            return
        if self._get_thread_local("int_to_ptr", False):
            return  # pointers built from integers cannot be attributed to an arg

        lanes = data.reshape(-1)
        mask_data = True if mask is None else getattr(mask, "data", mask)
        try:
            active = np.broadcast_to(
                np.asarray(mask_data, dtype=bool), data.shape
            ).reshape(-1)
        except ValueError:
            return  # the frontend adapter did not pair mask lanes with pointer lanes
        active_lanes = lanes[active]
        if active_lanes.size == 0:
            return
        # pointer elements (pointer tables) have no primitive_bitwidth
        itemsize = max(1, getattr(element_ty, "primitive_bitwidth", 64) // 8)
        lowest, highest = int(active_lanes.min()), int(active_lanes.max())
        if any(
            start <= lowest and highest + itemsize <= end
            for _, _, start, end in self.storages
        ):
            return  # common case: every active lane is inside one storage

        inside = np.zeros(active_lanes.shape, dtype=bool)
        for _, _, start, end in self.storages:
            inside |= (active_lanes >= start) & (active_lanes <= end - itemsize)
        if inside.all():
            return
        # Masked-off lanes help attribute the access: Triton splits a float
        # atomic_max/min into two calls by sign, and the call holding only the
        # out-of-bounds lanes still points at the tensor through its other lanes.
        touched = [
            storage
            for storage in self.storages
            if ((lanes > storage[2] - itemsize) & (lanes < storage[3])).any()
        ]
        if not touched:
            return

        outside = active_lanes[~inside]
        first_bad = int(outside[0])
        name, tensor, start, end = min(
            touched, key=lambda s: max(s[2] - first_bad, first_bad - s[3] + 1, 0)
        )
        base = tensor.data_ptr()
        ranges = " and ".join(
            f"[{int(part.min()) - base}, {int(part.max()) - base + itemsize})"
            for part in (outside[outside < start], outside[outside >= start])
            if part.size
        )
        frames = extract_user_frames(num_frames=1)
        where = (
            f"{frames[0].filename}:{frames[0].lineno} in {frames[0].func_name}\n"
            f"    {frames[0].line_of_code.strip()}"
            if frames
            else "an unknown kernel location"
        )
        raise IndexError(
            f"Tracer refused an out-of-bounds {op_name} in program "
            f"{self._get_thread_local('program_id')} at {where}\n"
            f"  {outside.size} of {active_lanes.size} active lanes fall outside "
            f"`{name}` (dtype={tensor.dtype}, shape={tuple(tensor.shape)}): they "
            f"touch bytes {ranges} from its data_ptr, but its storage spans "
            f"[{start - base}, {end - base}).\n"
            "  Triton's interpreter would run this access on raw host memory, "
            "reading or writing memory outside the tensor. Mask the out-of-range "
            'lanes, or run the kernel under the Sanitizer (tilelens.trace("sanitizer")) '
            "for a full out-of-bounds report."
        )

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
        self.storages.extend(_storage_ranges(name, arg))

    def grid_idx_callback(self, grid_idx: tuple[int, ...]):
        self._set_thread_local("program_id", grid_idx)
        self._set_thread_local("int_to_ptr", False)
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

        @self.lock_fn
        def pre_load_callback(ptr, mask, keys):
            if keys is None:
                self._check_in_bounds("load", ptr, mask)
            if not self.sample:
                return

            if keys is None:  # i.e. for triton, ptr = pointer + offsets
                first_ptr = np.reshape(ptr.data, (-1))[0]
                tensor = self._get_tensor(first_ptr)
                offsets = ptr.data - tensor.data_ptr()
            else:
                keys = _convert_keys_to_numpy(keys)
                offsets = masked_load(ptr.get_offsets().data, keys, mask=mask.data)
                tensor = ptr

            rec = Load(tensor.data_ptr(), offsets, mask.data)
            rec.call_path = extract_user_frames(num_frames=1)
            self.records.append(rec)

        @self.lock_fn
        def pre_store_callback(ptr, mask, keys):
            if keys is None:
                self._check_in_bounds("store", ptr, mask)
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

            rec = Store(tensor.data_ptr(), offsets, mask_data)
            rec.call_path = extract_user_frames(num_frames=1)
            self.records.append(rec)

        @self.lock_fn
        def pre_atomic_rmw_callback(ptr, mask):
            self._check_in_bounds("atomic", ptr, mask)

        @self.lock_fn
        def pre_atomic_cas_callback(ptr):
            self._check_in_bounds("atomic_cas", ptr, None)

        def post_int_to_ptr_callback(ret, *args):
            self._set_thread_local("int_to_ptr", True)

        @self.lock_fn
        def pre_transfer_callback(src, dst, mem_src, mem_dst):
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
            )
            rec.call_path = extract_user_frames(num_frames=1)
            self.records.append(rec)

        @self.lock_fn
        def post_reduce_sum_callback(ret, input, axis=None, keep_dims=False):
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
            self.records.append(ReduceSum(input_shape, axis, keep_dims, output_shape))

        @self.lock_fn
        def post_dot_callback(ret, input, other):
            if not self.sample:
                return
            input_shape = input.data.shape
            other_shape = other.data.shape
            ret_shape = ret.data.shape
            # Pass input/other raw arrays so draw.py can render MatMul
            rec = Dot(input_shape, other_shape, ret_shape, input.data, other.data)
            rec.call_path = extract_user_frames(num_frames=1)
            self.records.append(rec)

        callbacks = {
            Allocate: OpCallbacks(after_callback=post_allocate_callback),
            Load: OpCallbacks(before_callback=pre_load_callback),
            Store: OpCallbacks(before_callback=pre_store_callback),
            AtomicRMW: OpCallbacks(before_callback=pre_atomic_rmw_callback),
            AtomicCas: OpCallbacks(before_callback=pre_atomic_cas_callback),
            IntToPtr: OpCallbacks(after_callback=post_int_to_ptr_callback),
            Transfer: OpCallbacks(before_callback=pre_transfer_callback),
            ReduceSum: OpCallbacks(after_callback=post_reduce_sum_callback),
            Dot: OpCallbacks(after_callback=post_dot_callback),
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
            self.storages.clear()
            return self.records
