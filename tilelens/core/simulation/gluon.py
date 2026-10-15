"""NumPy-backed execution for Gluon kernels.

The simulator reuses Triton's interpreter tensors, memory operations, and
host/device argument copyback machinery, while adapting Gluon's language
signatures to Triton's interpreter semantic layer.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
import inspect
import threading
from typing import Any

import numpy as np
import triton
import triton.language as tl
from triton.experimental.gluon import language as gl  # type: ignore
from triton.experimental.gluon.language import _core as gluon_core  # type: ignore
from triton.experimental.gluon.language import _layouts as gluon_layouts  # type: ignore
from triton.experimental.gluon.language import _math as gluon_math  # type: ignore
from triton.experimental.gluon.language import _semantic as gluon_semantic_module  # type: ignore
from triton.experimental.gluon.language import _standard as gluon_standard  # type: ignore
from triton.experimental.gluon.language import amd as gluon_amd  # type: ignore
from triton.experimental.gluon.language.amd import cdna3 as gluon_amd_cdna3  # type: ignore
from triton.experimental.gluon.language.amd import cdna4 as gluon_amd_cdna4  # type: ignore
from triton.experimental.gluon.language.amd import gfx1250 as gluon_amd_gfx1250  # type: ignore
from triton.experimental.gluon.language.amd import rdna3 as gluon_amd_rdna3  # type: ignore
from triton.experimental.gluon.language.amd import rdna4 as gluon_amd_rdna4  # type: ignore
from triton.experimental.gluon.language.amd.cdna4 import (  # type: ignore
    async_copy as gluon_amd_cdna4_async_copy,
)
from triton.experimental.gluon.language.amd.gfx1250 import (  # type: ignore
    async_copy as gluon_amd_async_copy,
)
from triton.experimental.gluon.language.amd.gfx1250 import (  # type: ignore
    cluster as gluon_amd_cluster,
)
from triton.experimental.gluon.language.amd.gfx1250 import (  # type: ignore
    mbarrier as gluon_amd_mbarrier,
)
from triton.experimental.gluon.language.amd.gfx1250 import (  # type: ignore
    tdm as gluon_amd_tdm,
)
from triton.experimental.gluon.language.nvidia.ampere import (  # type: ignore
    async_copy as gluon_ampere_async_copy,
)
from triton.experimental.gluon.language.nvidia.ampere import (  # type: ignore
    mbarrier as gluon_ampere_mbarrier,
)
from triton.experimental.gluon.language.nvidia import blackwell as gluon_blackwell  # type: ignore
from triton.experimental.gluon.language.nvidia import hopper as gluon_hopper  # type: ignore
from triton.experimental.gluon.language.nvidia.blackwell import clc as gluon_clc  # type: ignore
from triton.experimental.gluon.language.nvidia.blackwell import (  # type: ignore
    tma as gluon_blackwell_tma,
)
from triton.experimental.gluon.language.nvidia.hopper import (  # type: ignore
    mbarrier as gluon_hopper_mbarrier,
)
from triton.experimental.gluon.language.nvidia.hopper import (  # type: ignore
    tma as gluon_hopper_tma,
)
from triton.runtime.interpreter import (  # type: ignore
    GridExecutor,
    InterpreterError,
    TensorHandle,
    _convert_float,
    _get_np_dtype,
    _implicit_cvt,
    _mxfp_value_handle_to_float32,
    _patch_lang_core,
    _patch_lang_tensor,
    _unpack_e2m1,
    interpreter_builder,
)

from ..frontend.base import _LangPatchScope

_MISSING = object()
_HOST_TENSOR_DESCRIPTOR_TYPES = {"TensorDescriptor", "TensorDescriptorIm2Col"}


class _MBarrierState:
    def __init__(self, ready: bool = True, phase: int = 1) -> None:
        self.ready = ready
        self.phase = phase
        self.pending_bytes = 0
        self.condition = threading.Condition()


class _WarpSpecializeScheduler:
    """Round-robin scheduler for CPU threads simulating Gluon partitions."""

    def __init__(self, count: int) -> None:
        self._active = list(range(count))
        self._turn_index = 0
        self._condition = threading.Condition()
        self._local = threading.local()

    def bind(self, partition_id: int) -> None:
        self._local.partition_id = partition_id

    def yield_point(self) -> None:
        partition_id = getattr(self._local, "partition_id", None)
        if partition_id is None:
            return
        with self._condition:
            # Patched Gluon builtins call this before executing. Only the
            # partition whose turn is active may proceed; the rest park until a
            # later builtin advances the simulated warp-specialized region.
            while (
                partition_id in self._active
                and self._active
                and self._active[self._turn_index] != partition_id
            ):
                self._condition.wait()
            if partition_id not in self._active or not self._active:
                return
            self._turn_index = (self._turn_index + 1) % len(self._active)
            self._condition.notify_all()

    def finish(self, partition_id: int) -> None:
        with self._condition:
            if partition_id in self._active:
                # Finished partitions no longer participate in the round-robin.
                # Keep the current turn pointing at the same logical successor
                # after removing an earlier list element.
                removed_index = self._active.index(partition_id)
                self._active.pop(removed_index)
                if self._active:
                    if removed_index < self._turn_index:
                        self._turn_index -= 1
                    self._turn_index %= len(self._active)
                else:
                    self._turn_index = 0
            self._condition.notify_all()


class GluonSemantic(gluon_semantic_module.GluonSemantic):
    """Triton interpreter semantic with Gluon-compatible builtin signatures."""

    def num_warps(self, _generator: Any = None):
        return 1

    def num_ctas(self):
        return 1

    def convert_layout(self, value: Any, layout: Any, assert_trivial: bool = False):
        ty = value.type
        if not isinstance(ty, gluon_core.distributed_type):
            return value
        # Triton's interpreter only changes the handle metadata here; Gluon still
        # expects a distributed_type wrapper carrying the requested layout.
        ret_ty = gluon_core.distributed_type(ty.element_ty, ty.shape, layout)
        handle = self.builder.create_convert_layout(
            ret_ty.to_ir(self.builder), value.handle
        )
        return gluon_core.tensor(handle, ret_ty)

    def _with_auto_layout(self, value: Any) -> Any:
        ty = getattr(value, "type", None)
        if not isinstance(ty, gluon_core.distributed_type):
            return value
        ret_ty = gluon_core.distributed_type(
            ty.element_ty,
            ty.shape,
            gluon_layouts.AutoLayout(),
        )
        return gluon_core.tensor(value.handle, ret_ty)

    def store(
        self,
        pointer: Any,
        value: Any,
        mask: Any,
        boundary_check: Any,
        cache_modifier: Any,
        eviction_policy: Any,
    ):
        return super().store(
            self._with_auto_layout(pointer),
            value,
            self._with_auto_layout(mask),
            boundary_check,
            cache_modifier,
            eviction_policy,
        )

    def warp_specialize(
        self,
        functions_and_args: Any,
        worker_num_warps: Any,
        worker_num_regs: Any = None,
        generator: Any = None,
    ):
        functions_and_args = list(functions_and_args)
        if len(functions_and_args) <= 1:
            result = None
            for fn, args in functions_and_args:
                result = fn(*tuple(args))
            return result

        from tilelens.core.frontend import gluon as gluon_frontend

        scheduler = _WarpSpecializeScheduler(len(functions_and_args))
        gluon_frontend._set_warp_specialize_scheduler(scheduler)
        grid_dim = self.builder.grid_dim
        grid_idx = self.builder.grid_idx
        results = [None] * len(functions_and_args)

        def run_partition(partition_id: int, fn: Callable, args: Any) -> None:
            if grid_dim is not None:
                self.builder.set_grid_dim(*grid_dim)
            if grid_idx is not None:
                self.builder.set_grid_idx(*grid_idx)
            scheduler.bind(partition_id)
            try:
                results[partition_id] = fn(*tuple(args))
            finally:
                scheduler.finish(partition_id)

        try:
            with ThreadPoolExecutor(max_workers=len(functions_and_args)) as executor:
                futures = [
                    executor.submit(run_partition, partition_id, fn, args)
                    for partition_id, (fn, args) in enumerate(functions_and_args)
                ]
                for future in futures:
                    future.result()
        finally:
            gluon_frontend._set_warp_specialize_scheduler(None)
        return results[0]


gluon_semantic: GluonSemantic


def _unwrap_constexpr(value: Any) -> Any:
    if isinstance(value, tl.constexpr):
        return value.value
    return value


def _to_python_scalar(value: Any) -> Any:
    value = _unwrap_constexpr(value)
    if isinstance(value, tl.tensor):
        data = value.handle.data
        if np.asarray(data).size == 1:
            return np.asarray(data).reshape(-1)[0].item()
        return data
    if isinstance(value, TensorHandle):
        data = value.data
        if np.asarray(data).size == 1:
            return np.asarray(data).reshape(-1)[0].item()
        return data
    return value


def _to_int(value: Any) -> int:
    return int(_to_python_scalar(value))


def _host_array(base: Any) -> np.ndarray:
    detach = getattr(base, "detach", None)
    cpu = getattr(base, "cpu", None)
    if callable(detach) and callable(cpu):
        return detach().cpu().numpy()
    numpy = getattr(base, "numpy", None)
    if callable(numpy):
        return numpy()
    return np.asarray(base)


class BlockType:
    def __init__(self, element_ty: tl.dtype, shape: list[int]) -> None:
        self.element_ty = element_ty
        self.shape = [int(dim) for dim in shape]

    @property
    def nbytes(self) -> int:
        bitwidth = getattr(self.element_ty, "primitive_bitwidth", 8)
        return int(np.prod(self.shape, dtype=np.int64)) * max(1, int(bitwidth) // 8)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, BlockType)
            and self.element_ty == other.element_ty
            and self.shape == list(other.shape)
        )


class TensorDescriptorType:
    def __init__(self, block_type: BlockType, layout: Any = None) -> None:
        self.block_type = block_type
        self.layout = layout


class TensorDescriptorLayoutType:
    def __init__(self, block_type: tl.block_type, is_signed: bool, layout: Any) -> None:
        self.block_type = block_type
        self.is_signed = is_signed
        self.layout = layout


class SharedMemoryDescriptorType:
    def __init__(
        self,
        element_ty: tl.dtype,
        shape: list[int],
        layout: Any,
        alloc_shape: list[int] | None = None,
    ) -> None:
        self.element_ty = element_ty
        self.shape = tuple(int(dim) for dim in shape)
        self.layout = layout
        self.alloc_shape = (
            tuple(int(dim) for dim in alloc_shape)
            if alloc_shape is not None
            else self.shape
        )

    @property
    def nbytes_per_cta(self) -> int:
        bitwidth = getattr(self.element_ty, "primitive_bitwidth", 8)
        return int(np.prod(self.shape, dtype=np.int64)) * max(1, int(bitwidth) // 8)


class TensorDescriptor(gluon_hopper_tma.tensor_descriptor):
    def __init__(
        self,
        base: Any,
        shape: list[int],
        strides: list[int],
        block_shape: list[int],
        layout: Any,
        padding: str = "zero",
        element_strides: list[int] | None = None,
        pixel_box_lower_corner: list[int] | None = None,
        pixel_box_upper_corner: list[int] | None = None,
        origin: list[int] | None = None,
    ) -> None:
        self.base = base
        self.shape = tuple(int(dim) for dim in shape)
        self.strides = tuple(int(stride) for stride in strides)
        self.block_shape = tuple(int(dim) for dim in block_shape)
        self.layout = layout
        self.padding = padding
        self.element_strides = (
            tuple(int(stride) for stride in element_strides)
            if element_strides is not None
            else tuple(1 for _ in self.shape)
        )
        self.pixel_box_lower_corner = (
            tuple(int(corner) for corner in pixel_box_lower_corner)
            if pixel_box_lower_corner is not None
            else None
        )
        self.pixel_box_upper_corner = (
            tuple(int(corner) for corner in pixel_box_upper_corner)
            if pixel_box_upper_corner is not None
            else None
        )
        self.origin = (
            tuple(int(offset) for offset in origin)
            if origin is not None
            else tuple(0 for _ in self.shape)
        )
        # Host descriptors may be built from interpreter handles, Gluon tensors,
        # or torch tensors. Infer the element dtype before normalizing pointers.
        if isinstance(base, tl.tensor):
            dtype = base.dtype
        elif isinstance(base, TensorHandle):
            dtype = base.dtype
        elif getattr(base, "dtype", _MISSING) is not _MISSING:
            dtype = tl.str_to_ty(triton.runtime.jit.mangle_type(base), None)
        else:
            raise TypeError(f"Cannot infer descriptor dtype from {type(base)}")
        self.dtype = dtype.element_ty if isinstance(dtype, tl.pointer_type) else dtype
        self.block_type = BlockType(self.dtype, list(self.block_shape))
        self.type = TensorDescriptorType(self.block_type)
        self.nbytes_per_cta = self.block_type.nbytes
        self.mode = "im2col" if self.pixel_box_lower_corner is not None else "tiled"

    @property
    def block_shape(self):
        return self.__dict__["_block_shape"]

    @block_shape.setter
    def block_shape(self, value: Any) -> None:
        self.__dict__["_block_shape"] = tuple(int(dim) for dim in value)

    @property
    def layout(self):
        return self.__dict__["_layout"]

    @layout.setter
    def layout(self, value: Any) -> None:
        self.__dict__["_layout"] = value

    @property
    def dtype(self):
        return self.__dict__["_dtype"]

    @dtype.setter
    def dtype(self, value: Any) -> None:
        self.__dict__["_dtype"] = value

    @property
    def block_type(self):
        return self.__dict__["_block_type"]

    @block_type.setter
    def block_type(self, value: Any) -> None:
        self.__dict__["_block_type"] = value

    @property
    def nbytes_per_cta(self) -> int:
        return self.__dict__["_nbytes_per_cta"]

    @nbytes_per_cta.setter
    def nbytes_per_cta(self, value: Any) -> None:
        self.__dict__["_nbytes_per_cta"] = int(value)

    @property
    def _array(self) -> np.ndarray:
        # Descriptors created inside a kernel hold an interpreter pointer handle.
        # Resolve it back to the CPU tensor copied by GridExecutor before loads.
        if isinstance(self.base, TensorHandle):
            host_tensor = getattr(self.base, "attr", {}).get("gluon.host_tensor")
            if host_tensor is not None:
                return _host_array(host_tensor)
            return np.asarray(self.base.data)
        return _host_array(self.base)

    @property
    def data(self) -> np.ndarray:
        return self._array

    @property
    def handle(self):
        return self

    def data_ptr(self) -> int:
        data_ptr = getattr(self.base, "data_ptr", None)
        if callable(data_ptr):
            return int(data_ptr())
        return int(self.data.__array_interface__["data"][0])

    def load_block(self, coord: Any) -> np.ndarray:
        coord = _normalize_coord(coord, len(self.shape))
        result = np.zeros(self.block_shape, dtype=_get_np_dtype(self.dtype))
        src = self._array
        for block_idx in np.ndindex(self.block_shape):
            src_idx = tuple(
                self.origin[dim] + coord[dim] + block_idx[dim]
                for dim in range(len(coord))
            )
            if all(0 <= src_idx[dim] < self.shape[dim] for dim in range(len(coord))):
                result[block_idx] = src[src_idx]
        return result

    def store_block(self, coord: Any, values: np.ndarray) -> None:
        coord = _normalize_coord(coord, len(self.shape))
        dst = self._array
        for block_idx in np.ndindex(self.block_shape):
            dst_idx = tuple(
                self.origin[dim] + coord[dim] + block_idx[dim]
                for dim in range(len(coord))
            )
            if all(0 <= dst_idx[dim] < self.shape[dim] for dim in range(len(coord))):
                dst[dst_idx] = values[block_idx]

    def reduce_block(self, coord: Any, values: np.ndarray, kind: str) -> None:
        coord = _normalize_coord(coord, len(self.shape))
        dst = self._array
        for block_idx in np.ndindex(self.block_shape):
            dst_idx = tuple(
                self.origin[dim] + coord[dim] + block_idx[dim]
                for dim in range(len(coord))
            )
            if not all(
                0 <= dst_idx[dim] < self.shape[dim] for dim in range(len(coord))
            ):
                continue
            value = values[block_idx]
            if kind == "add":
                dst[dst_idx] += value
            elif kind == "min":
                dst[dst_idx] = np.minimum(dst[dst_idx], value)
            elif kind == "max":
                dst[dst_idx] = np.maximum(dst[dst_idx], value)
            elif kind == "and":
                dst[dst_idx] = np.bitwise_and(dst[dst_idx], value)
            elif kind == "or":
                dst[dst_idx] = np.bitwise_or(dst[dst_idx], value)
            elif kind == "xor":
                dst[dst_idx] = np.bitwise_xor(dst[dst_idx], value)
            else:
                raise ValueError(f"Unsupported TMA atomic reduction: {kind}")

    def load_im2col_block(self, coord: Any, offsets: Any) -> np.ndarray:
        if len(self.shape) != 4:
            raise NotImplementedError("TMA im2col currently supports rank-4 NHWC")
        if self.pixel_box_lower_corner is None or self.pixel_box_upper_corner is None:
            raise TypeError("TMA im2col descriptor requires pixel box corners")
        coord = _normalize_coord(coord, len(self.shape))
        offsets = _normalize_coord(offsets, len(self.shape) - 2)
        result = np.zeros(self.block_shape, dtype=_get_np_dtype(self.dtype))
        if self.padding == "nan":
            result.fill(np.nan)
        src = self._array
        n_dim, h_dim, w_dim, c_dim = self.shape
        lower_h = self.pixel_box_lower_corner[0] + offsets[0]
        lower_w = self.pixel_box_lower_corner[1] + offsets[1]
        upper_h = h_dim - 1 + self.pixel_box_upper_corner[0] + offsets[0]
        upper_w = w_dim - 1 + self.pixel_box_upper_corner[1] + offsets[1]
        box_h = upper_h - lower_h + 1
        box_w = upper_w - lower_w + 1
        # Gluon im2col flattens the sliding image box into the first block axis;
        # reconstruct NHWC coordinates so OOB pixels keep the padding value.
        start_h = self.origin[1] + coord[1] + offsets[0]
        start_w = self.origin[2] + coord[2] + offsets[1]
        start_linear = (start_h - lower_h) * box_w + (start_w - lower_w)
        start_batch = self.origin[0] + coord[0]
        coord_c = self.origin[3] + coord[3]
        stride_c = self.element_strides[3]
        for pixel in range(self.block_shape[0]):
            linear = start_linear + pixel
            batch = start_batch + linear // (box_h * box_w)
            within_batch = linear % (box_h * box_w)
            h = lower_h + within_batch // box_w
            w = lower_w + within_batch % box_w
            for channel in range(self.block_shape[1]):
                c = coord_c + channel * stride_c
                if (
                    0 <= batch < n_dim
                    and 0 <= h < h_dim
                    and 0 <= w < w_dim
                    and 0 <= c < c_dim
                ):
                    result[pixel, channel] = src[batch, h, w, c]
        return result


class CLCResult:
    def is_canceled(self, _semantic: Any = None):
        return gluon_semantic.to_tensor(False)

    def program_id(self, dim: Any, _semantic: Any = None):
        return gluon_semantic.to_tensor(0)


def _normalize_coord(coord: Any, ndim: int) -> tuple[int, ...]:
    coord = _unwrap_constexpr(coord)
    if isinstance(coord, (list, tuple)):
        values = tuple(_to_int(item) for item in coord)
    else:
        values = (_to_int(coord),)
    if len(values) != ndim:
        raise ValueError(f"Expected {ndim} TMA coordinates, got {len(values)}")
    return values


def _descriptor_from_host(value: Any) -> TensorDescriptor:
    if isinstance(value, TensorDescriptor):
        return value
    if type(value).__name__ in _HOST_TENSOR_DESCRIPTOR_TYPES:
        return TensorDescriptor(
            value.base,
            list(value.shape),
            list(value.strides),
            list(value.block_shape),
            value.layout,
            getattr(value, "padding", "zero"),
            getattr(value, "element_strides", None),
            getattr(value, "pixel_box_lower_corner", None),
            getattr(value, "pixel_box_upper_corner", None),
        )
    raise TypeError(f"Unsupported TMA descriptor type: {type(value)}")


def _implicit_gluon_cvt(name: str, value: Any, constexprs: set[str]) -> Any:
    if name in constexprs:
        return value
    if isinstance(value, TensorDescriptor) or (
        type(value).__name__ in _HOST_TENSOR_DESCRIPTOR_TYPES
    ):
        return _descriptor_from_host(value)
    converted = _implicit_cvt(value)
    data_ptr = getattr(value, "data_ptr", None)
    if callable(data_ptr) and isinstance(converted, tl.tensor):
        converted.handle.set_attr("gluon.host_tensor", value)
    return converted


class SharedMemoryHandle(TensorHandle, gluon_core.shared_memory_descriptor):
    def __init__(
        self,
        element_ty: tl.dtype,
        shape: list[int],
        layout: Any,
        data: np.ndarray | None = None,
    ) -> None:
        self.element_ty = element_ty
        self.shape = tuple(int(dim) for dim in shape)
        self.layout = layout
        data = (
            np.zeros(self.shape, dtype=_get_np_dtype(element_ty))
            if data is None
            else data
        )
        TensorHandle.__init__(self, data, element_ty)
        self.type = SharedMemoryDescriptorType(
            self.element_ty,
            list(self.shape),
            self.layout,
        )

    @property
    def shape(self):
        if "type" not in self.__dict__:
            return self.__dict__["_shape"]
        return self.type.shape

    @shape.setter
    def shape(self, value: Any) -> None:
        self.__dict__["_shape"] = tuple(int(dim) for dim in value)

    @property
    def dtype(self):
        return self.__dict__["dtype"]

    @dtype.setter
    def dtype(self, value: Any) -> None:
        self.__dict__["dtype"] = value
        self.element_ty = value

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def layout(self):
        if "type" not in self.__dict__:
            return self.__dict__["_layout"]
        return self.type.layout

    @layout.setter
    def layout(self, value: Any) -> None:
        self.__dict__["_layout"] = value

    @property
    def nbytes_per_cta(self) -> int:
        return self.type.nbytes_per_cta

    def data_ptr(self) -> int:
        return int(self.data.__array_interface__["data"][0])

    def load(self, layout: Any = None, _semantic: Any = None) -> tl.tensor:
        layout = self.layout if layout is None else layout
        return tl.tensor(
            _tensor_result(self.data.copy(), self.element_ty, layout),
            tl.block_type(self.element_ty, list(self.shape)),
        )

    def store(self, value: Any, _semantic: Any = None) -> None:
        value = gluon_semantic.to_tensor(value)
        self.data[...] = np.asarray(value.handle.data, dtype=self.data.dtype)
        return None

    def index(self, idx: Any) -> "SharedMemoryHandle":
        idx_value = _to_int(idx)
        return SharedMemoryHandle(
            self.element_ty,
            list(self.shape[1:]),
            self.layout,
            self.data[idx_value],
        )

    def slice(
        self,
        start: Any,
        length: Any,
        dim: Any = 0,
        _semantic: Any = None,
    ) -> "SharedMemoryHandle":
        start_value = _to_int(start)
        length_value = _to_int(length)
        dim_value = _to_int(dim)
        slices = [slice(None)] * len(self.shape)
        slices[dim_value] = slice(start_value, start_value + length_value)
        shape = list(self.shape)
        shape[dim_value] = length_value
        return SharedMemoryHandle(
            self.element_ty,
            shape,
            self.layout,
            self.data[tuple(slices)],
        )

    def reshape(
        self,
        shape: Any,
        _semantic: Any = None,
    ) -> "SharedMemoryHandle":
        shape = tuple(int(_unwrap_constexpr(dim)) for dim in _unwrap_constexpr(shape))
        return SharedMemoryHandle(
            self.element_ty,
            list(shape),
            self.layout,
            self.data.reshape(shape),
        )

    def _reinterpret(
        self,
        shape: Any,
        layout: Any,
        _semantic: Any = None,
        _generator: Any = None,
    ) -> "SharedMemoryHandle":
        shape = tuple(int(_unwrap_constexpr(dim)) for dim in _unwrap_constexpr(shape))
        return SharedMemoryHandle(
            self.element_ty,
            list(shape),
            _unwrap_constexpr(layout),
            self.data.reshape(shape),
        )

    def permute(self, order: Any, _semantic: Any = None) -> "SharedMemoryHandle":
        order = tuple(int(_unwrap_constexpr(item)) for item in _unwrap_constexpr(order))
        return SharedMemoryHandle(
            self.element_ty,
            [self.shape[dim] for dim in order],
            self.layout,
            np.transpose(self.data, order),
        )


class TensorMemoryDescriptorType:
    def __init__(
        self,
        element_ty: tl.dtype,
        shape: list[int],
        layout: Any,
        alloc_shape: list[int] | None = None,
    ) -> None:
        self.element_ty = element_ty
        self.shape = [int(dim) for dim in shape]
        self.layout = layout
        self.alloc_shape = (
            [int(dim) for dim in alloc_shape] if alloc_shape is not None else self.shape
        )

    def get_reg_layout(self, num_warps: Any = None, instr_variant: str = "32x32b"):
        return self.layout


class TensorMemoryHandle(TensorHandle, gluon_blackwell.tensor_memory_descriptor):
    def __init__(
        self,
        element_ty: tl.dtype,
        shape: list[int],
        layout: Any,
        data: np.ndarray | None = None,
    ) -> None:
        self.element_ty = element_ty
        self.shape = [int(dim) for dim in shape]
        self.layout = layout
        data = (
            np.zeros(self.shape, dtype=_get_np_dtype(element_ty))
            if data is None
            else data
        )
        TensorHandle.__init__(self, data, element_ty)
        self.type = TensorMemoryDescriptorType(
            self.element_ty,
            self.shape,
            self.layout,
        )

    @property
    def shape(self):
        return self.__dict__["_shape"]

    @shape.setter
    def shape(self, value: Any) -> None:
        self.__dict__["_shape"] = [int(dim) for dim in value]

    @property
    def layout(self):
        return self.__dict__["_layout"]

    @layout.setter
    def layout(self, value: Any) -> None:
        self.__dict__["_layout"] = value

    @property
    def dtype(self):
        return self.__dict__["dtype"]

    @dtype.setter
    def dtype(self, value: Any) -> None:
        self.__dict__["dtype"] = value
        self.element_ty = value

    @property
    def rank(self) -> int:
        return len(self.shape)

    def data_ptr(self) -> int:
        return int(self.data.__array_interface__["data"][0])

    def get_reg_layout(
        self,
        num_warps: Any = None,
        instr_variant: str = "32x32b",
        _semantic: Any = None,
        _generator: Any = None,
    ):
        return self.type.get_reg_layout(num_warps, instr_variant)

    def load(
        self,
        layout: Any = None,
        _semantic: Any = None,
        _generator: Any = None,
    ) -> tl.tensor:
        layout = self.layout if layout is None else layout
        return tl.tensor(
            _tensor_result(self.data.copy(), self.element_ty, layout),
            tl.block_type(self.element_ty, list(self.shape)),
        )

    def _load_reduction(
        self,
        np_func: Callable,
        layout: Any = None,
        abs: Any = False,
        propagate_nan: Any = None,
        _semantic: Any = None,
        _generator: Any = None,
    ) -> tuple[tl.tensor, tl.tensor]:
        values = self.load(layout, _semantic, _generator)
        data = np.asarray(self.data)
        if bool(_unwrap_constexpr(abs)):
            data = np.abs(data)
        reduced_data = np_func(data, axis=1)
        reduced_data = np.asarray(reduced_data, dtype=data.dtype)
        layout = self.layout if layout is None else layout
        reduced = tl.tensor(
            _tensor_result(reduced_data, self.element_ty, layout),
            tl.block_type(self.element_ty, list(reduced_data.shape)),
        )
        return values, reduced

    def load_min(
        self,
        layout: Any = None,
        abs: Any = False,
        propagate_nan: Any = None,
        _semantic: Any = None,
        _generator: Any = None,
    ) -> tuple[tl.tensor, tl.tensor]:
        return self._load_reduction(
            np.min,
            layout,
            abs,
            propagate_nan,
            _semantic,
            _generator,
        )

    def load_max(
        self,
        layout: Any = None,
        abs: Any = False,
        propagate_nan: Any = None,
        _semantic: Any = None,
        _generator: Any = None,
    ) -> tuple[tl.tensor, tl.tensor]:
        return self._load_reduction(
            np.max,
            layout,
            abs,
            propagate_nan,
            _semantic,
            _generator,
        )

    def store(self, value: Any, pred: Any = True, _semantic: Any = None) -> None:
        if not bool(_to_python_scalar(pred)):
            return None
        value = gluon_semantic.to_tensor(value)
        self.data[...] = np.asarray(value.handle.data, dtype=self.data.dtype)
        return None

    def index(self, idx: Any, _semantic: Any = None) -> "TensorMemoryHandle":
        idx_value = _to_int(idx)
        return TensorMemoryHandle(
            self.element_ty,
            self.shape[1:],
            self.layout,
            self.data[idx_value],
        )

    def slice(
        self,
        start: Any,
        length: Any,
        _semantic: Any = None,
    ) -> "TensorMemoryHandle":
        start_value = _to_int(start)
        length_value = _to_int(length)
        return TensorMemoryHandle(
            self.element_ty,
            [self.shape[0], length_value],
            self.layout,
            self.data[:, start_value : start_value + length_value],
        )

    def reshape(
        self,
        shape: Any,
        _semantic: Any = None,
    ) -> "TensorMemoryHandle":
        shape = [int(_unwrap_constexpr(dim)) for dim in _unwrap_constexpr(shape)]
        return TensorMemoryHandle(
            self.element_ty,
            shape,
            self.layout,
            self.data.reshape(shape),
        )


def _get_handle_layout(handle: TensorHandle) -> Any:
    return handle.attr.get("gluon.layout", gluon_layouts.AutoLayout())


def _tensor_result(
    data: np.ndarray, dtype: tl.dtype, layout: Any = None
) -> TensorHandle:
    handle = TensorHandle(data, dtype)
    handle.set_attr("gluon.layout", layout or gluon_layouts.AutoLayout())
    return handle


def _local_indices(
    indices: np.ndarray, axis: int
) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
    for output_index in np.ndindex(indices.shape):
        memory_index = list(output_index)
        memory_index[axis] = int(indices[output_index])
        yield output_index, tuple(memory_index)


def _rmw_kind(rmw_op: Any) -> str:
    name = getattr(rmw_op, "name", str(rmw_op)).lower()
    if "fadd" in name or name.endswith("add") or ".add" in name:
        return "add"
    if "umin" in name or "min" in name:
        return "min"
    if "umax" in name or "max" in name:
        return "max"
    if "and" in name:
        return "and"
    if "or" in name:
        return "or"
    if "xor" in name:
        return "xor"
    if "xchg" in name:
        return "xchg"
    raise ValueError(f"Unsupported local atomic scatter RMW op: {rmw_op}")


def _apply_rmw(kind: str, old: Any, value: Any) -> Any:
    if kind == "add":
        return old + value
    if kind == "min":
        return np.minimum(old, value)
    if kind == "max":
        return np.maximum(old, value)
    if kind == "and":
        return np.bitwise_and(old, value)
    if kind == "or":
        return np.bitwise_or(old, value)
    if kind == "xor":
        return np.bitwise_xor(old, value)
    if kind == "xchg":
        return value
    raise ValueError(f"Unsupported local atomic scatter RMW kind: {kind}")


def _scale_values(scale: np.ndarray) -> np.ndarray:
    if np.issubdtype(scale.dtype, np.integer):
        raw = scale.astype(np.uint8, copy=False)
        values = np.exp2(raw.astype(np.int16) - 127).astype(np.float32)
        return np.where(raw == 255, np.nan, values)
    return scale.astype(np.float32)


def _expand_scale_groups(scale: np.ndarray, rows: int, k: int) -> np.ndarray:
    scale = _scale_values(np.asarray(scale))
    if scale.ndim == 0:
        return np.full((rows, k), scale.item(), dtype=np.float32)
    if scale.ndim == 1:
        scale = scale.reshape(1, -1)
    scale = np.broadcast_to(scale, (rows, scale.shape[1]))
    repeats = max(1, int(np.ceil(k / scale.shape[1])))
    return np.repeat(scale, repeats, axis=1)[:, :k]


def _convert_float_result(data: np.ndarray, dtype: tl.dtype) -> np.ndarray:
    if dtype == tl.bfloat16:
        return _convert_float(data, tl.float32, tl.bfloat16, None).view(
            _get_np_dtype(tl.bfloat16)
        )
    return data.astype(_get_np_dtype(dtype), copy=False)


class Builder(interpreter_builder.__class__):
    """Interpreter builder with the Gluon-specific methods used by semantics."""

    def __init__(self) -> None:
        super().__init__()
        if not hasattr(self.options, "enable_iisan"):
            object.__setattr__(self.options, "enable_iisan", False)
        # Barrier state is builder-scoped so repeated patch scopes do not share
        # stale ids from previous simulated launches.
        self._barrier_states: dict[int, _MBarrierState] = {}
        self._pending_clc_barriers: set[int] = set()

    def _barrier_key(self, barrier: Any) -> int:
        handle = getattr(barrier, "handle", None)
        if isinstance(handle, SharedMemoryHandle):
            barrier = handle
        if isinstance(barrier, SharedMemoryHandle):
            # Shared memory descriptors can be rewrapped; the backing buffer is
            # the stable identity for matching expect/arrive/wait calls.
            return int(barrier.data.__array_interface__["data"][0])
        return id(barrier)

    def _barrier_state(self, barrier: Any) -> _MBarrierState:
        key = self._barrier_key(barrier)
        state = self._barrier_states.get(key)
        if state is None:
            state = _MBarrierState()
            self._barrier_states[key] = state
        return state

    def _signal_barrier(self, barrier: Any) -> None:
        state = self._barrier_state(barrier)
        with state.condition:
            state.pending_bytes = 0
            state.phase ^= 1
            state.ready = True
            state.condition.notify_all()

    _WRAPPED_LAYOUT_OPS: dict[str, tuple[str, int | None]] = {
        "binary_op": ("arg", 0),
        "cast_impl": ("arg", 0),
        "create_make_range": ("ret_ty", 0),
        "create_get_program_id": ("auto", None),
        "create_get_num_programs": ("auto", None),
        "create_addptr": ("auto", None),
        "create_masked_load": ("arg", 0),
        "create_splat": ("ret_ty", 0),
        "create_reshape": ("arg", 0),
        "create_trans": ("arg", 0),
        "create_cat": ("arg", 0),
        "create_join": ("arg", 0),
        "create_split": ("arg", 0),
        "create_unsplat": ("arg", 0),
        "create_gather": ("arg", 0),
        "create_fabs": ("arg", 0),
        "create_erf": ("arg", 0),
        "create_rsqrt": ("arg", 0),
        "create_fma": ("arg", 2),
        "create_idiv": ("arg", 0),
        "create_umulhi": ("arg", 0),
        "create_ashr": ("arg", 0),
        "create_fp_to_fp": ("arg", 0),
        "create_bitcast": ("arg", 0),
        "create_atomic_cas": ("arg", 0),
        "create_atomic_rmw": ("arg", 1),
        "create_dot": ("arg", 2),
        "create_dot_scaled": ("arg", 9),
        "create_descriptor_gather": ("arg", 1),
    }

    def __getattribute__(self, name: str):
        attr = super().__getattribute__(name)
        layout_source = type(self)._WRAPPED_LAYOUT_OPS.get(name)
        if layout_source is None or not callable(attr):
            return attr

        def wrapped(*args: Any, **kwargs: Any):
            result = attr(*args, **kwargs)
            layout = self._layout_from_source(layout_source, args)
            return self._attach_layout(result, layout)

        return wrapped

    def _ret_layout(self, ret_ty: Any, fallback: Any = None) -> Any:
        return getattr(
            ret_ty,
            "gluon_layout",
            getattr(ret_ty, "layout", fallback or gluon_layouts.AutoLayout()),
        )

    def _layout_from_source(
        self,
        source: tuple[str, int | None],
        args: tuple[Any, ...],
    ) -> Any:
        kind, index = source
        if kind == "auto":
            return gluon_layouts.AutoLayout()
        if index is None or index >= len(args):
            return gluon_layouts.AutoLayout()
        value = args[index]
        if kind == "ret_ty":
            return self._ret_layout(value)
        if isinstance(value, TensorHandle):
            return _get_handle_layout(value)
        return gluon_layouts.AutoLayout()

    def _attach_layout(self, value: Any, layout: Any) -> Any:
        if isinstance(value, TensorHandle):
            return _tensor_result(value.data, value.dtype, layout)
        if isinstance(value, tuple):
            return tuple(self._attach_layout(item, layout) for item in value)
        return value

    def get_auto_layout(self):
        return gluon_layouts.AutoLayout()

    def get_coalesced_layout(self):
        return gluon_layouts.CoalescedLayout()

    def get_blocked_layout(
        self,
        size_per_thread: Any,
        threads_per_warp: Any,
        warps_per_cta: Any,
        order: Any,
        cga_layout: Any = None,
    ):
        return gluon_layouts.BlockedLayout(
            list(size_per_thread),
            list(threads_per_warp),
            list(warps_per_cta),
            list(order),
            [] if cga_layout is None else list(cga_layout),
        )

    def get_slice_layout(self, dim: int, parent: Any):
        return gluon_layouts.SliceLayout(int(dim), parent)

    def get_distributed_linear_layout(
        self,
        reg_bases: Any,
        lane_bases: Any,
        warp_bases: Any,
        block_bases: Any,
        shape: Any,
    ):
        return gluon_layouts.DistributedLinearLayout(
            list(reg_bases),
            list(lane_bases),
            list(warp_bases),
            list(block_bases),
            list(shape),
        )

    def get_dot_operand_layout(self, operand_index: int, parent: Any, k_width: int):
        return gluon_layouts.DotOperandLayout(int(operand_index), parent, int(k_width))

    def get_mma_layout(
        self,
        version: Any,
        warps_per_cta: Any,
        cga_layout: Any,
        instr_shape: Any,
    ):
        return gluon_layouts.NVMMADistributedLayout(
            list(version),
            list(warps_per_cta),
            list(instr_shape),
            [] if cga_layout is None else list(cga_layout),
        )

    def get_nvmma_shared_layout(
        self,
        swizzle_byte_width: int,
        element_bitwidth: int,
        transposed: bool,
        fp4_padded: bool,
        cga_layout: Any,
        rank: int,
    ):
        return gluon_layouts.NVMMASharedLayout(
            int(swizzle_byte_width),
            int(element_bitwidth),
            int(rank),
            bool(transposed),
            bool(fp4_padded),
            [] if cga_layout is None else list(cga_layout),
        )

    def get_swizzled_shared_layout(
        self,
        vec: int,
        per_phase: int,
        max_phase: int,
        order: Any,
        cga_layout: Any,
    ):
        return gluon_layouts.SwizzledSharedLayout(
            int(vec),
            int(per_phase),
            int(max_phase),
            list(order),
            [] if cga_layout is None else list(cga_layout),
        )

    def get_padded_shared_layout(
        self,
        intervals: Any,
        paddings: Any,
        offset_bases: Any,
        cga_layout: Any,
        shape: Any,
    ):
        return gluon_layouts.PaddedSharedLayout(
            [list(pair) for pair in zip(intervals, paddings)],
            list(offset_bases),
            [] if cga_layout is None else list(cga_layout),
            list(shape),
        )

    def get_partitioned_shared_layout(
        self,
        num_partitions: int,
        num_groups: int,
        partition_dim: int,
        partition_layout: Any,
    ):
        return gluon_amd_tdm.PartitionedSharedLayout(
            int(num_partitions),
            int(num_groups),
            int(partition_dim),
            partition_layout,
        )

    def get_shared_linear_layout(
        self,
        offset_bases: Any,
        block_bases: Any = None,
        alignment: int = 16,
    ):
        return gluon_layouts.SharedLinearLayout(
            list(offset_bases),
            [] if block_bases is None else list(block_bases),
            int(alignment),
        )

    def get_amd_mfma_layout(
        self,
        version: Any,
        warps_per_cta: Any,
        instr_shape: Any,
        transposed: Any,
        cga_layout: Any,
        tiles_per_warp: Any,
        element_bitwidth: Any,
    ):
        return gluon_amd.AMDMFMALayout(
            int(version),
            list(instr_shape),
            bool(transposed),
            list(warps_per_cta),
            None if element_bitwidth is None else int(element_bitwidth),
            None if tiles_per_warp is None else list(tiles_per_warp),
            [] if cga_layout is None else list(cga_layout),
        )

    def get_amd_wmma_layout(
        self,
        version: Any,
        transposed: Any,
        warp_bases: Any,
        reg_bases: Any,
        cga_layout: Any,
        instr_shape: Any,
        rank: Any,
    ):
        return gluon_amd.AMDWMMALayout(
            int(version),
            bool(transposed),
            [list(basis) for basis in warp_bases],
            None if reg_bases is None else [list(basis) for basis in reg_bases],
            None if instr_shape is None else list(instr_shape),
            [] if cga_layout is None else [list(basis) for basis in cga_layout],
            None if rank is None else int(rank),
        )

    def get_tensor_memory_layout(
        self,
        block: Any,
        col_stride: Any,
        cga_layout: Any = None,
        two_ctas: Any = False,
        fp4_padded: Any = False,
    ):
        cga_bases = [] if cga_layout is None else [list(basis) for basis in cga_layout]
        layout_fields = getattr(
            gluon_blackwell.TensorMemoryLayout,
            "__dataclass_fields__",
            {},
        )
        # Older Gluon builds used cta_split_num and had no FP4 padding field.
        if "fp4_padded" in layout_fields:
            return gluon_blackwell.TensorMemoryLayout(
                tuple(block),
                int(col_stride),
                cga_bases,
                bool(two_ctas),
                bool(fp4_padded),
            )
        if "cta_split_num" in layout_fields:
            return gluon_blackwell.TensorMemoryLayout(
                tuple(block),
                int(col_stride),
                cga_bases or None,
                bool(two_ctas),
            )
        return gluon_blackwell.TensorMemoryLayout(
            tuple(block),
            int(col_stride),
            cga_bases,
            bool(two_ctas),
        )

    def get_tensor_memory_scales_layout(self, cga_layout: Any):
        return gluon_blackwell.TensorMemoryScalesLayout(
            [] if cga_layout is None else [list(basis) for basis in cga_layout]
        )

    def get_distributed_ty(self, element_ty: tl.dtype, shape: Any, layout: Any):
        block_ty = tl.block_type(element_ty, list(shape))
        block_ty.gluon_layout = layout
        return block_ty

    def get_shared_mem_desc_ty(
        self,
        element_ty: tl.dtype,
        shape: Any,
        layout: Any,
        alloc_shape: Any,
    ):
        return SharedMemoryDescriptorType(
            element_ty,
            list(shape),
            layout,
            list(alloc_shape),
        )

    def get_tensor_mem_desc_ty(
        self,
        element_ty: tl.dtype,
        shape: Any,
        layout: Any,
        alloc_shape: Any,
    ):
        return TensorMemoryDescriptorType(
            element_ty,
            list(shape),
            layout,
            list(alloc_shape),
        )

    def get_tensor_descriptor_layout_type(
        self,
        block_type: tl.block_type,
        is_signed: bool,
        layout: Any,
    ):
        return TensorDescriptorLayoutType(block_type, bool(is_signed), layout)

    def get_tensor_descriptor_im2col_layout_type(
        self,
        block_type: tl.block_type,
        is_signed: bool,
        layout: Any,
    ):
        return TensorDescriptorLayoutType(block_type, bool(is_signed), layout)

    def get_gluon_layout_from_tensor(self, handle: TensorHandle):
        return _get_handle_layout(handle)

    def get_gluon_layout_from_memdesc(self, handle: Any):
        return getattr(handle, "layout", gluon_layouts.AutoLayout())

    def is_convert_layout_trivial(self, _ret_ty: Any, _value: Any) -> bool:
        return True

    def create_broadcast(self, arg: TensorHandle, ret_ty: Any):
        result = super().create_broadcast(arg, getattr(ret_ty, "shape", ret_ty))
        return _tensor_result(
            result.data,
            result.dtype,
            self._ret_layout(ret_ty, _get_handle_layout(arg)),
        )

    def create_expand_dims(self, arg: TensorHandle, axis: int):
        result = super().create_expand_dims(arg, axis)
        layout = _get_handle_layout(arg)
        if isinstance(layout, gluon_layouts.SliceLayout):
            layout = layout.parent
        return _tensor_result(result.data, result.dtype, layout)

    def create_convert_layout(self, ret_ty: Any, value: TensorHandle):
        layout = getattr(ret_ty, "layout", gluon_layouts.AutoLayout())
        return _tensor_result(np.asarray(value.data).copy(), value.dtype, layout)

    def create_extract_slice(
        self,
        ret_ty: Any,
        source: TensorHandle,
        offsets: Any,
    ):
        shape = list(getattr(ret_ty, "shape", np.asarray(source.data).shape))
        starts = [int(_unwrap_constexpr(offset)) for offset in offsets]
        slices = tuple(slice(start, start + dim) for start, dim in zip(starts, shape))
        return _tensor_result(
            np.asarray(source.data)[slices].copy(),
            source.dtype,
            getattr(ret_ty, "layout", _get_handle_layout(source)),
        )

    def create_local_alloc(
        self,
        desc_ty: SharedMemoryDescriptorType,
        value: TensorHandle | None = None,
    ):
        data = np.zeros(desc_ty.alloc_shape, dtype=_get_np_dtype(desc_ty.element_ty))
        if value is not None:
            data[...] = np.asarray(value.data, dtype=data.dtype)
        return SharedMemoryHandle(
            desc_ty.element_ty, list(desc_ty.shape), desc_ty.layout, data
        )

    def create_local_load(self, _ret_ty: Any, mem_desc: SharedMemoryHandle):
        return _tensor_result(
            np.asarray(mem_desc.data).copy(),
            mem_desc.element_ty,
            mem_desc.layout,
        )

    def create_local_store(self, mem_desc: SharedMemoryHandle, value: TensorHandle):
        mem_desc.data[...] = np.asarray(value.data, dtype=mem_desc.data.dtype)
        return None

    def create_local_gather(
        self,
        _ret_ty: Any,
        mem_desc: SharedMemoryHandle,
        indices: TensorHandle,
        axis: int,
    ):
        indices_data = np.asarray(indices.data)
        result = np.empty(indices_data.shape, dtype=mem_desc.data.dtype)
        # Gluon scatter/gather indices describe one logical axis while the
        # remaining coordinates come from the output tensor position.
        for output_index, memory_index in _local_indices(indices_data, int(axis)):
            result[output_index] = mem_desc.data[memory_index]
        return _tensor_result(result, mem_desc.element_ty, _get_handle_layout(indices))

    def create_local_scatter(
        self,
        mem_desc: SharedMemoryHandle,
        values: TensorHandle,
        indices: TensorHandle,
        axis: int,
    ):
        indices_data = np.asarray(indices.data)
        values_data = np.asarray(values.data)
        for output_index, memory_index in _local_indices(indices_data, int(axis)):
            mem_desc.data[memory_index] = values_data[output_index]
        return None

    def create_local_atomic_scatter_rmw(
        self,
        rmw_op: Any,
        mem_desc: SharedMemoryHandle,
        values: TensorHandle,
        indices: TensorHandle,
        mask: TensorHandle | None,
        axis: int,
    ):
        kind = _rmw_kind(rmw_op)
        indices_data = np.asarray(indices.data)
        values_data = np.asarray(values.data)
        mask_data = None if mask is None else np.asarray(mask.data, dtype=bool)
        old_values = np.empty(indices_data.shape, dtype=mem_desc.data.dtype)
        for output_index, memory_index in _local_indices(indices_data, int(axis)):
            old = mem_desc.data[memory_index]
            old_values[output_index] = old
            if mask_data is not None and not mask_data[output_index]:
                continue
            mem_desc.data[memory_index] = _apply_rmw(
                kind,
                old,
                values_data[output_index],
            )
        return _tensor_result(
            old_values, mem_desc.element_ty, _get_handle_layout(values)
        )

    def create_local_dealloc(self, _mem_desc: SharedMemoryHandle):
        return None

    def create_memdesc_subslice(
        self,
        ret_ty: SharedMemoryDescriptorType,
        mem_desc: SharedMemoryHandle,
        offsets: Any,
    ):
        slices = []
        for dim, offset in enumerate(offsets):
            start = int(offset)
            stop = start + int(ret_ty.shape[dim])
            slices.append(slice(start, stop))
        return SharedMemoryHandle(
            ret_ty.element_ty,
            list(ret_ty.shape),
            ret_ty.layout,
            mem_desc.data[tuple(slices)],
        )

    def create_memdesc_index(
        self,
        ret_ty: SharedMemoryDescriptorType,
        mem_desc: SharedMemoryHandle,
        index: TensorHandle,
    ):
        index_value = int(np.asarray(index.data).reshape(-1)[0])
        return SharedMemoryHandle(
            ret_ty.element_ty,
            list(ret_ty.shape),
            ret_ty.layout,
            mem_desc.data[index_value],
        )

    def create_memdesc_trans(self, mem_desc: SharedMemoryHandle, order: Any):
        order = tuple(int(item) for item in order)
        return SharedMemoryHandle(
            mem_desc.element_ty,
            [mem_desc.shape[dim] for dim in order],
            mem_desc.layout,
            np.transpose(mem_desc.data, order),
        )

    def create_memdesc_reshape(self, mem_desc: SharedMemoryHandle, shape: Any):
        shape = [int(_unwrap_constexpr(dim)) for dim in _unwrap_constexpr(shape)]
        return SharedMemoryHandle(
            mem_desc.element_ty,
            shape,
            mem_desc.layout,
            mem_desc.data.reshape(shape),
        )

    def create_memdesc_reinterpret(
        self,
        ret_ty: SharedMemoryDescriptorType,
        mem_desc: SharedMemoryHandle,
    ):
        data = mem_desc.data.view(_get_np_dtype(ret_ty.element_ty)).reshape(
            ret_ty.shape
        )
        return SharedMemoryHandle(
            ret_ty.element_ty, list(ret_ty.shape), ret_ty.layout, data
        )

    def create_tmem_alloc(
        self,
        desc_ty: TensorMemoryDescriptorType,
        value: TensorHandle | None = None,
    ):
        data = np.zeros(desc_ty.alloc_shape, dtype=_get_np_dtype(desc_ty.element_ty))
        if value is not None:
            data[...] = np.asarray(value.data, dtype=data.dtype)
        return TensorMemoryHandle(
            desc_ty.element_ty, list(desc_ty.shape), desc_ty.layout, data
        )

    def create_tmem_load(self, _ret_ty: Any, desc: TensorMemoryHandle, *args: Any):
        if not args:
            return _tensor_result(
                np.asarray(desc.data).copy(), desc.element_ty, desc.layout
            )
        red_op = args[0]
        abs_flag = args[1] if len(args) > 1 else False
        data = np.asarray(desc.data)
        if bool(_to_python_scalar(abs_flag)):
            data = np.abs(data)
        name = getattr(red_op, "name", str(red_op)).lower()
        reduced_data = np.min(data, axis=1) if "min" in name else np.max(data, axis=1)
        return (
            _tensor_result(np.asarray(desc.data).copy(), desc.element_ty, desc.layout),
            _tensor_result(
                np.asarray(reduced_data, dtype=data.dtype), desc.element_ty, desc.layout
            ),
            desc.layout,
        )

    def create_tmem_store(
        self,
        desc: TensorMemoryHandle,
        value: TensorHandle,
        pred: TensorHandle,
    ):
        if bool(np.asarray(pred.data).reshape(-1)[0]):
            desc.data[...] = np.asarray(value.data, dtype=desc.data.dtype)
        return None

    def create_tmem_subslice(
        self,
        ret_ty: TensorMemoryDescriptorType,
        desc: TensorMemoryHandle,
        start: Any,
    ):
        start_value = _to_int(start)
        return TensorMemoryHandle(
            ret_ty.element_ty,
            ret_ty.shape,
            ret_ty.layout,
            desc.data[:, start_value : start_value + ret_ty.shape[-1]],
        )

    def create_make_tensor_descriptor(
        self,
        desc_ty: TensorDescriptorLayoutType,
        base_handle: TensorHandle,
        shape_handles: list[TensorHandle],
        stride_handles: list[TensorHandle],
        padding: Any = "zero",
    ):
        shape = [_to_int(handle) for handle in shape_handles]
        strides = [_to_int(handle) for handle in stride_handles]
        block_shape = list(desc_ty.block_type.shape)
        return TensorDescriptor(
            base_handle,
            shape,
            strides,
            block_shape,
            desc_ty.layout,
            str(padding).split(".")[-1].lower(),
        )

    def create_update_tensor_descriptor(
        self,
        desc: TensorDescriptor,
        add_offsets: list[TensorHandle] | None = None,
        set_bounds: list[TensorHandle] | None = None,
        pred: Any = None,
        clamp_bounds: bool = False,
    ):
        origin = list(desc.origin)
        shape = list(desc.shape)
        if add_offsets:
            offsets = _normalize_coord(add_offsets, len(desc.shape))
            origin = [origin[dim] + offsets[dim] for dim in range(len(origin))]
        if set_bounds:
            bounds = _normalize_coord(set_bounds, len(desc.shape))
            shape = [origin[dim] + bounds[dim] for dim in range(len(origin))]
        return TensorDescriptor(
            desc.base,
            shape,
            list(desc.strides),
            list(desc.block_shape),
            desc.layout,
            desc.padding,
            list(desc.element_strides),
            (
                list(desc.pixel_box_lower_corner)
                if desc.pixel_box_lower_corner is not None
                else None
            ),
            (
                list(desc.pixel_box_upper_corner)
                if desc.pixel_box_upper_corner is not None
                else None
            ),
            origin,
        )

    def create_async_tma_copy_global_to_local(
        self,
        tensor_desc: TensorDescriptor,
        coord: Any,
        barrier: Any,
        result: SharedMemoryHandle,
        pred: TensorHandle,
        multicast: Any = False,
        offsets: Any = None,
    ):
        if not bool(np.asarray(pred.data).reshape(-1)[0]):
            return None
        if offsets is None:
            result.data[...] = tensor_desc.load_block(coord)
        else:
            result.data[...] = tensor_desc.load_im2col_block(coord, offsets)
        state = self._barrier_state(barrier)
        nbytes = tensor_desc.block_type.nbytes
        should_signal = True
        with state.condition:
            if state.pending_bytes > nbytes:
                state.pending_bytes -= nbytes
                should_signal = False
        if should_signal:
            self._signal_barrier(barrier)
        return None

    def create_async_tma_copy_local_to_global(
        self,
        tensor_desc: TensorDescriptor,
        coord: Any,
        src: SharedMemoryHandle,
    ):
        tensor_desc.store_block(coord, np.asarray(src.data))
        return None

    def create_async_tma_reduce(
        self,
        kind: Any,
        tensor_desc: TensorDescriptor,
        coord: Any,
        src: SharedMemoryHandle,
    ):
        kind_name = str(kind).split(".")[-1].lower()
        tensor_desc.reduce_block(coord, np.asarray(src.data), kind_name)
        return None

    def create_async_tma_store_wait(self, pendings: Any = 0, read_only: Any = True):
        return None

    def create_async_tma_gather(
        self,
        tensor_desc: TensorDescriptor,
        x_offsets: TensorHandle,
        y_offset: TensorHandle,
        barrier: Any,
        result: SharedMemoryHandle,
        pred: TensorHandle,
        multicast: Any = False,
    ):
        if not bool(np.asarray(pred.data).reshape(-1)[0]):
            return None
        rows = np.asarray(x_offsets.data).astype(np.int64).reshape(-1)
        col_offset = _to_int(y_offset)
        result.data.fill(0)
        src = tensor_desc._array
        for out_row, src_row in enumerate(rows):
            for out_col in range(result.data.shape[1]):
                src_col = col_offset + out_col
                if (
                    0 <= src_row < tensor_desc.shape[0]
                    and 0 <= src_col < tensor_desc.shape[1]
                ):
                    result.data[out_row, out_col] = src[int(src_row), src_col]
        self._signal_barrier(barrier)
        return None

    def create_async_tma_scatter(
        self,
        tensor_desc: TensorDescriptor,
        x_offsets: TensorHandle,
        y_offset: TensorHandle,
        src: SharedMemoryHandle,
    ):
        rows = np.asarray(x_offsets.data).astype(np.int64).reshape(-1)
        col_offset = _to_int(y_offset)
        if col_offset < 0 or np.any(rows < 0):
            raise ValueError("async_scatter offsets must be non-negative")
        dst = tensor_desc._array
        values = np.asarray(src.data, dtype=_get_np_dtype(tensor_desc.dtype))
        for in_row, dst_row in enumerate(rows):
            if dst_row >= tensor_desc.shape[0]:
                continue
            for in_col in range(values.shape[1]):
                dst_col = col_offset + in_col
                if 0 <= dst_col < tensor_desc.shape[1]:
                    dst[int(dst_row), dst_col] = values[in_row, in_col]
        return None

    def create_clc_try_cancel(self, result: Any, barrier: Any):
        self._pending_clc_barriers.add(self._barrier_key(barrier))
        return None

    def create_clc_load_result(self, _src: Any):
        return CLCResult()

    def create_clc_is_canceled(self, result: CLCResult):
        return result.is_canceled().handle

    def create_clc_get_program_id(self, result: CLCResult, dim: Any):
        return result.program_id(dim).handle

    def get_int128_ty(self):
        return tl.int64

    def create_set_auto_layout(self, layout: Any, value: TensorHandle):
        return _tensor_result(np.asarray(value.data).copy(), value.dtype, layout)

    def create_histogram(
        self,
        data: TensorHandle,
        bins: int,
        mask: TensorHandle | None,
        _layout: Any = None,
    ):
        result = super().create_histogram(data, bins, mask)
        return _tensor_result(
            result.data,
            result.dtype,
            _layout or _get_handle_layout(data),
        )

    def create_fp4_to_fp(self, src: TensorHandle, elem_type: tl.dtype, axis: int):
        return _tensor_result(
            _convert_float_result(_unpack_e2m1(src.data, int(axis)), elem_type),
            elem_type,
            _get_handle_layout(src),
        )

    def create_scaled_upcast_fp8(
        self,
        _ret_ty: Any,
        src: TensorHandle,
        scale: TensorHandle,
    ):
        if src.dtype in (tl.float8e4nv, tl.float8e5):
            values = _mxfp_value_handle_to_float32(src)
        else:
            values = np.asarray(src.data, dtype=np.float32)
        return _tensor_result(
            values * np.broadcast_to(_scale_values(scale.data), values.shape),
            tl.float32,
            _get_handle_layout(src),
        )

    def create_scaled_upcast_fp4(
        self,
        src: TensorHandle,
        scale: TensorHandle,
        elem_type: tl.dtype,
        axis: int,
    ):
        values = _unpack_e2m1(src.data, int(axis)).astype(np.float32)
        return _tensor_result(
            _convert_float_result(
                values * np.broadcast_to(_scale_values(scale.data), values.shape),
                elem_type,
            ),
            elem_type,
            _get_handle_layout(scale),
        )

    def _buffer_ptrs(self, ptr: TensorHandle, offsets: TensorHandle):
        return self.create_addptr(ptr, offsets)

    def create_buffer_load(
        self,
        _ret_ty: Any,
        ptr: TensorHandle,
        offsets: TensorHandle,
        mask: TensorHandle | None,
        other: TensorHandle | None,
        cache_modifier: Any = None,
    ):
        ptrs = self._buffer_ptrs(ptr, offsets)
        if mask is None:
            mask = _tensor_result(
                np.ones_like(ptrs.data, dtype=bool),
                tl.int1,
                _get_handle_layout(ptrs),
            )
        return self.create_masked_load(ptrs, mask, other, None, None, False)

    def create_buffer_store(
        self,
        stored_value: TensorHandle,
        ptr: TensorHandle,
        offsets: TensorHandle,
        mask: TensorHandle | None,
        cache_modifier: Any = None,
    ):
        ptrs = self._buffer_ptrs(ptr, offsets)
        if mask is None:
            mask = _tensor_result(
                np.ones_like(ptrs.data, dtype=bool),
                tl.int1,
                _get_handle_layout(ptrs),
            )
        return self.create_masked_store(ptrs, stored_value, mask, None, None)

    def create_buffer_atomic_rmw(
        self,
        rmw_op: Any,
        ptr: TensorHandle,
        offsets: TensorHandle,
        value: TensorHandle,
        sem: Any,
        scope: Any,
        mask: TensorHandle | None,
    ):
        ptrs = self._buffer_ptrs(ptr, offsets)
        if mask is None:
            mask = _tensor_result(
                np.ones_like(ptrs.data, dtype=bool),
                tl.int1,
                _get_handle_layout(ptrs),
            )
        old = self.create_masked_load(ptrs, mask, None, None, None, False)
        new_data = _apply_rmw(_rmw_kind(rmw_op), old.data, value.data)
        new_value = _tensor_result(
            new_data.astype(old.data.dtype),
            old.dtype,
            _get_handle_layout(old),
        )
        self.create_masked_store(ptrs, new_value, mask, None, None)
        return old

    def create_buffer_load_to_local(
        self,
        dest: SharedMemoryHandle,
        ptr: TensorHandle,
        offsets: TensorHandle,
        mask: TensorHandle | None,
        other: TensorHandle | None,
        *args: Any,
    ):
        value = self.create_buffer_load(None, ptr, offsets, mask, other)
        dest.data[...] = np.asarray(value.data, dtype=dest.data.dtype)
        return None

    def create_async_copy_global_to_local(
        self,
        dest: SharedMemoryHandle,
        ptr: TensorHandle,
        mask: TensorHandle | None,
        other: TensorHandle | None = None,
        cache_modifier: Any = None,
        eviction_policy: Any = None,
        is_volatile: Any = False,
    ):
        if not isinstance(mask, TensorHandle):
            mask = _tensor_result(
                np.ones_like(ptr.data, dtype=bool),
                tl.int1,
                _get_handle_layout(ptr),
            )
        value = self.create_masked_load(
            ptr,
            mask,
            other if isinstance(other, TensorHandle) else None,
            cache_modifier,
            eviction_policy,
            bool(_unwrap_constexpr(is_volatile)),
        )
        dest.data[...] = np.asarray(value.data, dtype=dest.data.dtype)
        return None

    def create_async_copy_local_to_global(
        self,
        src: SharedMemoryHandle,
        ptr: TensorHandle,
        mask: TensorHandle | None,
        *args: Any,
    ):
        value = _tensor_result(np.asarray(src.data), src.element_ty, src.layout)
        if mask is None:
            mask = _tensor_result(
                np.ones_like(value.data, dtype=bool),
                tl.int1,
                _get_handle_layout(value),
            )
        return self.create_masked_store(ptr, value, mask, None, None)

    def create_async_copy_mbarrier_arrive(self, mbarrier: Any, *args: Any):
        self._signal_barrier(mbarrier)
        return None

    def create_async_copy_lds_barrier_arrive(self, mbarrier: Any, *args: Any):
        self.create_async_copy_mbarrier_arrive(mbarrier, *args)
        return None

    def create_async_commit_group(self):
        return None

    def create_async_wait_group(self, num_outstanding: Any = 0):
        return None

    def create_mbarrier_init(self, barrier: Any, *args: Any):
        state = self._barrier_state(barrier)
        with state.condition:
            state.pending_bytes = 0
            state.phase = 1
            state.ready = True
            state.condition.notify_all()
        return None

    def create_mbarrier_inval(self, barrier: Any):
        self._signal_barrier(barrier)
        return None

    def create_mbarrier_expect(self, barrier: Any, *args: Any):
        expected = args[0] if args else 0
        key = self._barrier_key(barrier)
        state = self._barrier_state(barrier)
        with state.condition:
            state.pending_bytes = max(0, _to_int(expected))
            state.ready = False
        if key in self._pending_clc_barriers:
            # The CLC try_cancel path records the barrier before expect(); once
            # expect arrives, complete the simulated transaction immediately.
            self._pending_clc_barriers.remove(key)
            self._signal_barrier(barrier)
        return None

    def create_mbarrier_arrive(self, barrier: Any, *args: Any):
        pred = args[0] if args else True
        if bool(_to_python_scalar(pred)):
            self._signal_barrier(barrier)
        return None

    def create_mbarrier_wait(
        self, barrier: Any, phase: Any = None, pred: Any = True, *args: Any
    ):
        if not bool(_to_python_scalar(pred)):
            return None
        phase_value = None if phase is None else (_to_int(phase) & 1)
        state = self._barrier_state(barrier)
        while True:
            with state.condition:
                if state.ready and (phase_value is None or state.phase == phase_value):
                    return None
                # There is no frontend warp-specialization yield hook for Gluon;
                # use a short timed wait so another simulated partition can run.
                state.condition.wait(timeout=0.001)
        return None

    def create_lds_barrier_init(self, barrier: Any, *args: Any):
        return self.create_mbarrier_init(barrier, *args)

    def create_lds_barrier_wait(self, barrier: Any, phase: Any = None, *args: Any):
        return self.create_mbarrier_wait(barrier, phase, *args)

    def create_lds_barrier_arrive(self, barrier: Any, *args: Any):
        pred = args[0] if args else True
        if bool(_to_python_scalar(pred)):
            self._signal_barrier(barrier)
        return _tensor_result(np.array([0], dtype=np.int32), tl.int32)

    def create_fence_mbarrier_init_release_cluster(self):
        return None

    def create_fence_async_shared(self, cluster: Any = False):
        return None

    def create_async_shared_store(
        self,
        dst: SharedMemoryHandle,
        value: TensorHandle,
        mbarrier: Any,
    ):
        dst.data[...] = np.asarray(value.data, dtype=dst.data.dtype)
        self._signal_barrier(mbarrier)
        return None

    def create_cluster_arrive(self, *args: Any):
        return None

    def create_cluster_wait(self):
        return None

    def create_cluster_barrier(self, *args: Any):
        return None

    def create_amd_cluster_arrive(self):
        return None

    def create_amd_cluster_wait(self):
        return None

    def create_async_tdm_copy_global_to_local(
        self,
        src: TensorDescriptor,
        offsets_or_dest: Any,
        dest_or_mbarrier: Any = None,
        pred_or_cache_modifier: Any = None,
        mbarrier_or_warp_used_hint: Any = None,
        *args: Any,
    ):
        if isinstance(offsets_or_dest, SharedMemoryHandle):
            offsets = None
            dest = offsets_or_dest
            pred = True
            mbarrier = dest_or_mbarrier
        else:
            offsets = offsets_or_dest
            dest = dest_or_mbarrier
            pred = pred_or_cache_modifier
            mbarrier = mbarrier_or_warp_used_hint
        if pred is not None and not bool(_to_python_scalar(pred)):
            return None
        coord = [0] * len(src.shape) if offsets is None else offsets
        dest.data[...] = src.load_block(coord)
        if mbarrier is not None:
            self._signal_barrier(mbarrier)
        return None

    def create_async_tdm_copy_local_to_global(
        self,
        dest: TensorDescriptor,
        offsets_or_src: Any,
        src_or_mbarrier: Any = None,
        mbarrier_or_cache_modifier: Any = None,
        *args: Any,
    ):
        if isinstance(offsets_or_src, SharedMemoryHandle):
            offsets = None
            src = offsets_or_src
            mbarrier = src_or_mbarrier
        else:
            offsets = offsets_or_src
            src = src_or_mbarrier
            mbarrier = mbarrier_or_cache_modifier
        coord = [0] * len(dest.shape) if offsets is None else offsets
        dest.store_block(coord, np.asarray(src.data))
        if mbarrier is not None:
            self._signal_barrier(mbarrier)
        return None

    def create_async_tdm_wait(self, num_outstanding: Any = 0):
        return None

    def create_async_tdm_scatter(
        self,
        desc: TensorDescriptor,
        row_indices: TensorHandle,
        col_offset: TensorHandle,
        src: SharedMemoryHandle,
        mbarrier: Any = None,
    ):
        rows = np.asarray(row_indices.data).astype(np.int64).reshape(-1)
        offset = _to_int(col_offset)
        if offset < 0 or np.any(rows < 0):
            raise ValueError("async_scatter offsets must be non-negative")
        dst = desc._array
        values = np.asarray(src.data, dtype=_get_np_dtype(desc.dtype))
        for in_row, dst_row in enumerate(rows):
            if dst_row >= desc.shape[0]:
                continue
            for in_col in range(values.shape[1]):
                dst_col = offset + in_col
                if 0 <= dst_col < desc.shape[1]:
                    dst[int(dst_row), dst_col] = values[in_row, in_col]
        if mbarrier is not None:
            self._signal_barrier(mbarrier)
        return None

    def create_async_tdm_gather(
        self,
        desc: TensorDescriptor,
        row_indices: TensorHandle,
        col_offset: TensorHandle,
        dst: SharedMemoryHandle,
        pred: TensorHandle | bool | None = True,
        mbarrier: Any = None,
    ):
        if isinstance(pred, SharedMemoryHandle) and mbarrier is None:
            mbarrier = pred
            pred = True
        if pred is not None and not bool(_to_python_scalar(pred)):
            return None
        rows = np.asarray(row_indices.data).astype(np.int64).reshape(-1)
        offset = _to_int(col_offset)
        dst.data.fill(0)
        src = desc._array
        for out_row, src_row in enumerate(rows):
            for out_col in range(dst.data.shape[1]):
                src_col = offset + out_col
                if 0 <= src_row < desc.shape[0] and 0 <= src_col < desc.shape[1]:
                    dst.data[out_row, out_col] = src[int(src_row), src_col]
        if mbarrier is not None:
            self._signal_barrier(mbarrier)
        return None

    def create_tdm_prefetch(self, *args: Any):
        return CLCResult()

    def create_tmem_copy(self, src: SharedMemoryHandle, dst: TensorMemoryHandle):
        dst.data[...] = np.asarray(src.data, dtype=dst.data.dtype)
        return None

    def _signal_mbarriers(self, mbarriers: Any, preds: Any = None) -> None:
        if mbarriers is None:
            return
        barriers = mbarriers if isinstance(mbarriers, (list, tuple)) else (mbarriers,)
        if preds is None:
            pred_values = (True,) * len(barriers)
        elif isinstance(preds, (list, tuple)):
            pred_values = tuple(preds)
        else:
            pred_values = (preds,)
        for index, barrier in enumerate(barriers):
            pred = pred_values[index] if index < len(pred_values) else True
            if bool(_to_python_scalar(pred)):
                self._signal_barrier(barrier)

    def create_tcgen05_mma(
        self,
        a: Any,
        b: Any,
        acc: TensorMemoryHandle,
        use_acc: Any = True,
        pred: Any = True,
        mbarriers: Any = None,
        mbarrier_preds: Any = None,
        two_ctas: Any = False,
        multicast: Any = False,
    ):
        if not bool(_to_python_scalar(pred)):
            return None
        result = np.matmul(
            np.asarray(a.data).astype(np.float32),
            np.asarray(b.data).astype(np.float32),
        )
        if bool(_to_python_scalar(use_acc)):
            result = result + acc.data.astype(np.float32)
        acc.data[...] = result.astype(acc.data.dtype, copy=False)
        self._signal_mbarriers(mbarriers, mbarrier_preds)
        return None

    def create_tcgen05_mma_scaled(
        self,
        a: Any,
        b: Any,
        acc: TensorMemoryHandle,
        a_scale: Any,
        b_scale: Any,
        a_type: Any,
        b_type: Any,
        use_acc: Any = True,
        pred: Any = True,
        mbarriers: Any = None,
        mbarrier_preds: Any = None,
        two_ctas: Any = False,
        multicast: Any = False,
    ):
        if not bool(_to_python_scalar(pred)):
            return None
        a_data = np.asarray(a.data).astype(np.float32)
        b_data = np.asarray(b.data).astype(np.float32)
        k_dim = a_data.shape[1]
        lhs = a_data * _expand_scale_groups(
            np.asarray(a_scale.data), a_data.shape[0], k_dim
        )
        if b_data.shape[0] == k_dim:
            rhs_scales = _expand_scale_groups(
                np.asarray(b_scale.data), b_data.shape[1], k_dim
            )
            rhs = b_data * rhs_scales.T
        else:
            rhs_scales = _expand_scale_groups(
                np.asarray(b_scale.data), b_data.shape[0], k_dim
            )
            rhs = (b_data * rhs_scales).T
        result = np.matmul(lhs, rhs)
        if bool(_to_python_scalar(use_acc)):
            result = result + acc.data.astype(np.float32)
        acc.data[...] = result.astype(acc.data.dtype, copy=False)
        self._signal_mbarriers(mbarriers, mbarrier_preds)
        return None

    def create_tcgen05_commit(self, barrier: Any, pred: TensorHandle, *args: Any):
        if bool(np.asarray(pred.data).reshape(-1)[0]):
            self._signal_barrier(barrier)
        return None

    def create_warpgroup_mma(
        self,
        a: Any,
        b: Any,
        acc: TensorHandle,
        use_acc: Any = True,
        precision: Any = None,
        max_num_imprecise_acc: Any = None,
        is_async: Any = False,
    ):
        result = np.matmul(
            np.asarray(a.data).astype(np.float32),
            np.asarray(b.data).astype(np.float32),
        )
        if bool(_to_python_scalar(use_acc)):
            result = result + np.asarray(acc.data, dtype=np.float32)
        return _tensor_result(
            result.astype(_get_np_dtype(acc.dtype), copy=False),
            acc.dtype,
            _get_handle_layout(acc),
        )

    def create_warpgroup_mma_wait(self, deps: list[TensorHandle], num_outstanding: Any):
        return deps

    def create_warp_pipeline_border(self, marker: Any, priority: Any):
        return None

    def create_warp_yield(self, values: list[Any]):
        return values

    def create_warp_return(self):
        return None


gluon_builder = Builder()
gluon_semantic = GluonSemantic(gluon_builder)


_GLUON_BUILTIN_MODULES: tuple[Any, ...] = (
    tl.core,
    gluon_core,
    gl,
    gluon_math,
    gluon_standard,
    gluon_amd,
    gluon_amd_cdna3,
    gluon_amd_cdna4,
    gluon_amd_cdna4_async_copy,
    gluon_amd_gfx1250,
    gluon_amd_async_copy,
    gluon_amd_cluster,
    gluon_amd_mbarrier,
    gluon_amd_tdm,
    gluon_amd_rdna3,
    gluon_amd_rdna4,
    gluon_ampere_async_copy,
    gluon_ampere_mbarrier,
    gluon_blackwell,
    gluon_blackwell_tma,
    gluon_clc,
    gluon_hopper,
    gluon_hopper_mbarrier,
    gluon_hopper_tma,
)


_GLUON_BUILTIN_CLASSES: tuple[Any, ...] = (
    gluon_core.tensor,
    gluon_core.shared_memory_descriptor,
    gluon_blackwell.tensor_memory_descriptor,
    gluon_clc.clc_result,
)


_GLUON_NON_SEMANTIC_BUILTINS: tuple[tuple[Any, str, Callable], ...] = tuple(
    (module, name, getattr(module, name))
    for module, name in (
        (gluon_amd_cdna4, "_compute_efficient_padded_shared_layout_impl"),
        (gluon_amd_cdna4, "_get_mfma_scale_layout_impl"),
        (gluon_amd_gfx1250, "_get_wmma_scale_layout_impl"),
        (gluon_blackwell, "_compute_tmem_reg_layout"),
    )
)


def _make_tma_atomic_reduce(kind: Any) -> Callable:
    @gluon_core.builtin
    def async_atomic_reduce(tensor_desc, coord, src, _semantic=None):
        semantic = gluon_semantic if _semantic is None else _semantic
        coord = semantic._convert_to_ir_values(coord, require_i64=False)
        semantic.builder.create_async_tma_reduce(
            kind,
            tensor_desc.handle,
            coord,
            src.handle,
        )

    return async_atomic_reduce


def _patch_tma_compat_aliases(module: Any, scope: _LangPatchScope) -> None:
    if module is None:
        return

    for new_name, old_name in (
        ("async_load", "async_copy_global_to_shared"),
        ("async_load_im2col", "async_copy_global_to_shared_im2col"),
        ("async_store", "async_copy_shared_to_global"),
    ):
        if not hasattr(module, new_name) and hasattr(module, old_name):
            scope.set_attr(module, new_name, getattr(module, old_name))

    reduce_kind = getattr(gluon_core.ir, "DESCRIPTOR_REDUCE_KIND", None)
    if reduce_kind is None:
        return
    for name, kind in (
        ("async_atomic_add", reduce_kind.ADD),
        ("async_atomic_min", reduce_kind.MIN),
        ("async_atomic_max", reduce_kind.MAX),
        ("async_atomic_and", reduce_kind.AND),
        ("async_atomic_or", reduce_kind.OR),
        ("async_atomic_xor", reduce_kind.XOR),
    ):
        if not hasattr(module, name):
            scope.set_attr(module, name, _make_tma_atomic_reduce(kind))


def _yield_warp_specialize() -> None:
    from tilelens.core.frontend import gluon as gluon_frontend

    gluon_frontend._maybe_yield_warp_specialize()


def _patch_gluon_non_semantic_builtins(scope: _LangPatchScope) -> None:
    for module, name, member in _GLUON_NON_SEMANTIC_BUILTINS:

        def new_member(
            *args: Any,
            member: Callable = member,
            **kwargs: Any,
        ):
            _yield_warp_specialize()
            kwargs.pop("_semantic", None)
            return member(*args, **kwargs)

        scope.set_attr(module, name, new_member)


def _patch_gluon_builtins(pkg: Any, scope: _LangPatchScope) -> None:
    for name, member in inspect.getmembers(pkg):
        if not callable(member):
            continue
        if not (
            getattr(member, gluon_core.GLUON_BUILTIN, False)
            or tl.core.is_builtin(member)
        ):
            continue

        def new_member(
            *args: Any,
            member: Callable = member,
            **kwargs: Any,
        ):
            _yield_warp_specialize()
            # Gluon builtins may pass a compile-time semantic object; replace it
            # with this builder-backed semantic while preserving user arguments.
            kwargs = {key: value for key, value in kwargs.items() if key != "_semantic"}
            return member(*args, **kwargs, _semantic=gluon_semantic)

        scope.set_attr(pkg, name, new_member)


def patch_lang(fn: Callable) -> _LangPatchScope:
    """Patch Gluon builtins to execute through the NumPy interpreter builder."""

    scope = _LangPatchScope()
    _patch_tma_compat_aliases(gluon_hopper_tma, scope)
    _patch_tma_compat_aliases(gluon_blackwell_tma, scope)
    for module in _GLUON_BUILTIN_MODULES:
        _patch_gluon_builtins(module, scope)
    for cls in _GLUON_BUILTIN_CLASSES:
        _patch_gluon_builtins(cls, scope)
    _patch_gluon_non_semantic_builtins(scope)

    def allocate_mbarrier(*_args: Any, **_kwargs: Any):
        return _tensor_result(np.array([0], dtype=np.int32), tl.int32)

    def init_mbarrier(mbarrier: Any, count: Any = 1, **_kwargs: Any):
        return gluon_builder.create_mbarrier_init(mbarrier, count)

    def expect_mbarrier(
        mbarrier: Any,
        bytes_per_cta: Any = None,
        pred: Any = True,
        **_kwargs: Any,
    ):
        if bool(_to_python_scalar(pred)):
            return gluon_builder.create_mbarrier_expect(mbarrier, bytes_per_cta)
        return None

    def arrive_mbarrier(
        mbarrier: Any,
        *,
        count: Any = 1,
        pred: Any = True,
        **_kwargs: Any,
    ):
        return gluon_builder.create_mbarrier_arrive(mbarrier, pred, count)

    def wait_mbarrier(
        mbarrier: Any,
        phase: Any,
        pred: Any = True,
        deps: Any = (),
        **_kwargs: Any,
    ):
        return gluon_builder.create_mbarrier_wait(mbarrier, phase, pred, deps)

    def invalidate_mbarrier(mbarrier: Any, **_kwargs: Any):
        return gluon_builder.create_mbarrier_inval(mbarrier)

    if gluon_ampere_mbarrier is not None:
        scope.set_attr(gluon_ampere_mbarrier, "allocate_mbarrier", allocate_mbarrier)
        scope.set_attr(gluon_ampere_mbarrier, "init", init_mbarrier)
        scope.set_attr(gluon_ampere_mbarrier, "expect", expect_mbarrier)
        scope.set_attr(gluon_ampere_mbarrier, "arrive", arrive_mbarrier)
        scope.set_attr(gluon_ampere_mbarrier, "wait", wait_mbarrier)
        scope.set_attr(gluon_ampere_mbarrier, "invalidate", invalidate_mbarrier)

    if gluon_hopper_mbarrier is not None:
        scope.set_attr(gluon_hopper_mbarrier, "allocate_mbarrier", allocate_mbarrier)
        scope.set_attr(gluon_hopper_mbarrier, "init", init_mbarrier)
        scope.set_attr(gluon_hopper_mbarrier, "expect", expect_mbarrier)
        scope.set_attr(gluon_hopper_mbarrier, "arrive", arrive_mbarrier)
        scope.set_attr(gluon_hopper_mbarrier, "wait", wait_mbarrier)
        scope.set_attr(gluon_hopper_mbarrier, "invalidate", invalidate_mbarrier)

    _patch_lang_tensor(gluon_core.tensor, scope)
    _patch_lang_core(gluon_core, scope)
    return scope


class GluonInterpretedFunction:
    """Callable wrapper that executes a Gluon kernel on CPU through NumPy."""

    def __init__(self, fn: Callable, arg_names: list[str] | None = None) -> None:
        self.fn = fn
        signature = inspect.signature(fn)
        self.arg_names = arg_names or [v.name for v in signature.parameters.values()]
        annotations = fn.__annotations__
        self.constexprs = {
            name
            for name in self.arg_names
            if annotations.get(name) in ("constexpr", tl.constexpr)
        }

    def _call_args(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        argspec = inspect.getfullargspec(self.fn)
        filtered_kwargs = {
            key: value for key, value in kwargs.items() if key in argspec.args
        }
        return inspect.getcallargs(self.fn, *args, **filtered_kwargs)

    @staticmethod
    def _canonical_grid(grid: Any, call_args: dict[str, Any]) -> tuple[int, int, int]:
        if callable(grid):
            grid = grid(call_args)
        if isinstance(grid, int):
            grid = (grid,)
        grid = tuple(int(dim) for dim in grid)
        if len(grid) > 3:
            raise ValueError(f"grid must have at most 3 dimensions, got {grid}")
        return grid + (1,) * (3 - len(grid))

    def run(self, *args_dev: Any, grid: Any, warmup: bool = False, **kwargs: Any):
        if warmup:
            return None

        client_manager = kwargs.pop("client_manager", None)
        should_patch_lang = kwargs.pop("_patch_lang", True)
        if "num_warps" not in self.arg_names:
            kwargs.pop("num_warps", None)
        if "num_ctas" not in self.arg_names:
            kwargs.pop("num_ctas", None)

        grid_helper = GridExecutor(self.fn, self.arg_names, grid)
        args_hst, kwargs_hst = grid_helper._init_args_hst(args_dev, kwargs)
        raw_call_args = self._call_args(tuple(args_hst), kwargs_hst)
        call_args = {
            name: _implicit_gluon_cvt(name, value, self.constexprs)
            for name, value in raw_call_args.items()
        }
        canonical_grid = self._canonical_grid(grid, raw_call_args)
        interpreter_builder.set_grid_dim(*canonical_grid)
        gluon_builder.set_grid_dim(*canonical_grid)

        if client_manager is not None:
            for name, arg in raw_call_args.items():
                client_manager.arg_callback(name, arg, call_args[name])
                base = getattr(arg, "base", None)
                data_ptr = getattr(base, "data_ptr", None)
                if callable(data_ptr):
                    client_manager.arg_callback(f"{name}.base", base, base)
            client_manager.grid_callback(canonical_grid)

        patch_scope = patch_lang(self.fn) if should_patch_lang else _LangPatchScope()
        try:

            def run_grid() -> None:
                for x in range(canonical_grid[0]):
                    for y in range(canonical_grid[1]):
                        for z in range(canonical_grid[2]):
                            interpreter_builder.set_grid_idx(x, y, z)
                            gluon_builder.set_grid_idx(x, y, z)
                            if client_manager is not None:
                                client_manager.grid_idx_callback((x, y, z))
                            self.fn(**call_args)

            run_grid()
        except Exception as exc:
            if triton.knobs.compilation.front_end_debugging:
                raise
            raise InterpreterError(repr(exc)) from exc
        finally:
            patch_scope.restore()

        grid_helper._restore_args_dev(args_dev, args_hst, kwargs, kwargs_hst)
        return None
