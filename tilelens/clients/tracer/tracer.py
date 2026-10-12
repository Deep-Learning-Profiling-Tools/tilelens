from collections.abc import Callable
import warnings

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
    InterpreterDivergence,
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
        self.sample = True
        self._divergence_noticed = False

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
        self._divergence_noticed = False

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

        @self.lock_fn
        def pre_interpreter_divergence_callback(kind, source_location):
            if self._divergence_noticed:
                return
            self._divergence_noticed = True
            # Not a registry-keyed warnings.warn, which the default filter
            # would show only once per process for the same message.
            filename, lineno, _ = source_location
            warnings.warn_explicit(
                "[Tracer] "
                + InterpreterDivergence.describe(kind, source_location)
                + ". This trace may not show what the compiled kernel does.",
                RuntimeWarning,
                filename,
                lineno,
            )

        callbacks = {
            Allocate: OpCallbacks(after_callback=post_allocate_callback),
            Load: OpCallbacks(before_callback=pre_load_callback),
            Store: OpCallbacks(before_callback=pre_store_callback),
            Transfer: OpCallbacks(before_callback=pre_transfer_callback),
            ReduceSum: OpCallbacks(after_callback=post_reduce_sum_callback),
            Dot: OpCallbacks(after_callback=post_dot_callback),
            InterpreterDivergence: OpCallbacks(
                before_callback=pre_interpreter_divergence_callback
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
