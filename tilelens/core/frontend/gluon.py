from collections.abc import Callable
from typing import Any

import numpy as np
import triton.language as tl
import triton.experimental.gluon.language as gluon_lang  # type: ignore
from triton.experimental.gluon.language import _core as gluon_core  # type: ignore
from triton.experimental.gluon.language import _math as gluon_math  # type: ignore
from triton.experimental.gluon.language import _semantic as gluon_semantic  # type: ignore
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
    tdm as gluon_amd_tdm,
)
from triton.experimental.gluon.language.nvidia.blackwell import (  # type: ignore
    tma as gluon_blackwell_tma,
)
from triton.experimental.gluon.language.nvidia.hopper import (  # type: ignore
    tma as gluon_hopper_tma,
)
from triton.runtime.interpreter import TensorHandle

from ..data import (
    AddPtr,
    Allocate,
    BinaryOp,
    Broadcast,
    Dot,
    ExpandDims,
    Fma,
    Join,
    Load,
    MakeRange,
    Op,
    ProgramId,
    Reshape,
    Rsqrt,
    Splat,
    Store,
    TernaryOp,
    UnaryOp,
)
from ..symbolic_metadata import (
    INT32,
    SymbolicTensorValue,
    SymbolicTypeSpec,
    TensorDescriptorAccess,
)

from .base import AdapterResult, Frontend, _LangPatchScope, register_frontend
from .triton import TritonFrontend

_WARP_SPECIALIZE_SCHEDULER: Any = None
_MISSING = object()


class GluonAsyncCopyLoad(Load):
    pass


class GluonAsyncCopyStore(Store):
    pass


class GluonBufferLoadToShared(Load):
    pass


def _set_warp_specialize_scheduler(scheduler: Any) -> Any:
    global _WARP_SPECIALIZE_SCHEDULER
    previous = _WARP_SPECIALIZE_SCHEDULER
    _WARP_SPECIALIZE_SCHEDULER = scheduler
    return previous


def _maybe_yield_warp_specialize() -> None:
    if _WARP_SPECIALIZE_SCHEDULER is not None:
        _WARP_SPECIALIZE_SCHEDULER.yield_point()


def _gluon_pointer_load_adapter(
    ptr: Any,
    *_args: Any,
    mask: Any = None,
    other: Any = None,
    **_kwargs: Any,
) -> AdapterResult:
    if mask is None and _args:
        mask = _args[0]
    ptr = _gluon_tensor_handle(ptr)
    mask = _gluon_tensor_handle(mask)
    if mask is None and isinstance(ptr, TensorHandle):
        mask = TensorHandle(np.ones_like(ptr.data, dtype=bool), tl.int1)
    return AdapterResult(ptr, mask, None)


_gluon_load_adapter = _gluon_pointer_load_adapter


def _gluon_descriptor_load_adapter(
    pred_arg_index: int,
    *,
    descriptor_kwarg: str,
    coords_kwarg: str,
) -> Callable[..., AdapterResult]:
    def adapter(
        *args: Any,
        mask: Any = None,
        pred: Any = None,
        **kwargs: Any,
    ) -> AdapterResult:
        if mask is None:
            mask = pred
        load_args = args[1:] if args else ()
        if mask is None and len(load_args) > pred_arg_index:
            mask = load_args[pred_arg_index]
        descriptor = args[0] if args else kwargs[descriptor_kwarg]
        coords = load_args[0] if load_args else kwargs[coords_kwarg]
        return AdapterResult(
            TensorDescriptorAccess(descriptor, coords, mask), mask, None
        )

    return adapter


def _gluon_async_copy_load_adapter(
    smem: Any,
    pointer: Any,
    mask: Any = None,
    *_args: Any,
    **_kwargs: Any,
) -> AdapterResult:
    # Async copy load traces source global memory, not the destination SMEM.
    return _gluon_pointer_load_adapter(pointer, mask=mask)


def _gluon_async_copy_store_adapter(
    pointer: Any,
    smem: Any,
    mask: Any = None,
    *_args: Any,
    **_kwargs: Any,
) -> AdapterResult:
    # Async copy store traces destination global memory, not the source SMEM.
    return _gluon_pointer_store_adapter(pointer, mask=mask)


def _gluon_buffer_load_to_shared_adapter(
    smem: Any,
    ptr: Any,
    offsets: Any,
    *args: Any,
    **kwargs: Any,
) -> AdapterResult:
    # CDNA4 buffer_load_to_shared receives base pointer plus vector offsets.
    # Rebuild the effective pointer so clients see a normal masked load.
    return _gluon_pointer_load_adapter(
        ptr + offsets,
        mask=args[0] if args else kwargs.get("mask"),
    )


def _gluon_pointer_store_adapter(
    ptr: Any,
    value: Any = None,
    *_args: Any,
    mask: Any = None,
    **_kwargs: Any,
) -> AdapterResult:
    if _is_global_tensor_descriptor_like(ptr) and value is not None:
        return AdapterResult(TensorDescriptorAccess(ptr, value, mask), mask, None)
    if mask is None and _args:
        mask = _args[0]
    ptr = _gluon_tensor_handle(ptr)
    mask = _gluon_tensor_handle(mask)
    if mask is None and isinstance(ptr, TensorHandle):
        mask = TensorHandle(np.ones_like(ptr.data, dtype=bool), tl.int1)
    return AdapterResult(ptr, mask, None)


_gluon_store_adapter = _gluon_pointer_store_adapter


def _is_global_tensor_descriptor_like(value: Any) -> bool:
    type_name = type(value).__name__
    return type_name in {
        "tensor_descriptor",
        "tensor_descriptor_im2col",
        "TensorDescriptor",
        "TensorDescriptorIm2Col",
    }


def _gluon_tensor_handle(value: Any) -> Any:
    if isinstance(value, gluon_core.tensor):
        return value.handle
    return value


def _gluon_binary_adapter(
    lhs: Any,
    rhs: Any,
    *_args: Any,
    **_kwargs: Any,
) -> AdapterResult:
    return AdapterResult(lhs, rhs)


def _gluon_semantic_binary_adapter(
    _semantic: Any,
    lhs: Any,
    rhs: Any,
    *_args: Any,
    **_kwargs: Any,
) -> AdapterResult:
    return AdapterResult(lhs, rhs)


def _gluon_unary_adapter(arg: Any, *_args: Any, **_kwargs: Any) -> AdapterResult:
    return AdapterResult(arg)


def _gluon_binary_op_adapter(op: Callable) -> Callable[..., AdapterResult]:
    def adapter(lhs: Any, rhs: Any, *_args: Any, **_kwargs: Any) -> AdapterResult:
        return AdapterResult(lhs, rhs, op)

    return adapter


def _gluon_semantic_binary_op_adapter(op: Callable) -> Callable[..., AdapterResult]:
    def adapter(
        _semantic: Any,
        lhs: Any,
        rhs: Any,
        *_args: Any,
        **_kwargs: Any,
    ) -> AdapterResult:
        return AdapterResult(lhs, rhs, op)

    return adapter


def _gluon_unary_op_adapter(op: Callable) -> Callable[..., AdapterResult]:
    def adapter(arg: Any, *_args: Any, **_kwargs: Any) -> AdapterResult:
        return AdapterResult(arg, op)

    return adapter


def _gluon_where_adapter(
    condition: Any,
    x: Any,
    y: Any,
    *_args: Any,
    **_kwargs: Any,
) -> AdapterResult:
    return AdapterResult(condition, x, y, np.where)


def _gluon_dot_adapter(a: Any, b: Any, *_args: Any, **_kwargs: Any) -> AdapterResult:
    return AdapterResult(_gluon_tensor_handle(a), _gluon_tensor_handle(b))


def _gluon_dot_overrider_adapter(
    a: Any,
    b: Any,
    acc: Any,
    *_args: Any,
    **_kwargs: Any,
) -> AdapterResult:
    return AdapterResult(
        _gluon_tensor_handle(a),
        _gluon_tensor_handle(b),
        _gluon_tensor_handle(acc),
        None,
        None,
    )


def _gluon_allocate_adapter(*args: Any, **_kwargs: Any) -> AdapterResult:
    return AdapterResult(*args[:4])


def _gluon_make_range_adapter(start: Any, end: Any, *_args: Any, **_kwargs: Any):
    start_value = int(start)
    end_value = int(end)
    return AdapterResult(
        SymbolicTypeSpec(INT32, (end_value - start_value,)), start_value, end_value
    )


def _existing_ops(namespace: Any, attrs: dict[str, type[Op]]) -> dict[str, type[Op]]:
    namespace_attrs = vars(namespace)
    return {attr: op_type for attr, op_type in attrs.items() if attr in namespace_attrs}


GLUON_CORE_OPS: dict[str, type[Op]] = {
    "program_id": ProgramId,
    "load": Load,
    "store": Store,
    "arange": MakeRange,
    "full": Splat,
    "dot_fma": Dot,
    "expand_dims": ExpandDims,
    "reshape": Reshape,
    "broadcast": Broadcast,
    "join": Join,
    "add": BinaryOp,
    "sub": BinaryOp,
    "mul": BinaryOp,
    "where": TernaryOp,
}


GLUON_MATH_OPS: dict[str, type[Op]] = {
    "exp": UnaryOp,
    "exp2": UnaryOp,
    "fma": Fma,
    "log": UnaryOp,
    "log2": UnaryOp,
    "cos": UnaryOp,
    "rsqrt": Rsqrt,
    "sin": UnaryOp,
    "sqrt": UnaryOp,
    "abs": UnaryOp,
    "floor": UnaryOp,
    "ceil": UnaryOp,
}


GLUON_NAMESPACES: dict[Any, dict[str, type[Op]]] = {
    gluon_core: dict(GLUON_CORE_OPS),
    gluon_lang: dict(GLUON_CORE_OPS),
    gluon_math: GLUON_MATH_OPS,
    gluon_semantic.GluonSemantic: {
        "add": AddPtr,
        "less_than": BinaryOp,
        "less_equal": BinaryOp,
        "greater_than": BinaryOp,
        "greater_equal": BinaryOp,
    },
}
_TMA_NAMESPACES: tuple[tuple[Any, dict[str, type[Op]]], ...] = (
    (
        gluon_hopper_tma,
        {
            "make_tensor_descriptor": Allocate,
            "async_load": Load,
            "async_load_im2col": Load,
            "async_store": Store,
            "async_copy_global_to_shared": Load,
            "async_copy_global_to_shared_im2col": Load,
            "async_copy_shared_to_global": Store,
            "async_atomic_add": Store,
            "async_atomic_min": Store,
            "async_atomic_max": Store,
            "async_atomic_and": Store,
            "async_atomic_or": Store,
            "async_atomic_xor": Store,
        },
    ),
    (
        gluon_blackwell_tma,
        {
            "make_tensor_descriptor": Allocate,
            "async_load": Load,
            "async_load_im2col": Load,
            "async_store": Store,
            "async_copy_global_to_shared": Load,
            "async_copy_global_to_shared_im2col": Load,
            "async_copy_shared_to_global": Store,
            "async_atomic_add": Store,
            "async_atomic_min": Store,
            "async_atomic_max": Store,
            "async_atomic_and": Store,
            "async_atomic_or": Store,
            "async_atomic_xor": Store,
        },
    ),
    (
        gluon_amd_cdna3,
        {
            "mfma": Dot,
            "buffer_load": Load,
            "buffer_store": Store,
            "buffer_atomic_add": Store,
            "buffer_atomic_max": Store,
            "buffer_atomic_min": Store,
            "buffer_atomic_and": Store,
            "buffer_atomic_or": Store,
            "buffer_atomic_xor": Store,
            "buffer_atomic_xchg": Store,
        },
    ),
    (
        gluon_amd_cdna4,
        {
            "mfma": Dot,
            "buffer_load": Load,
            "buffer_store": Store,
            "buffer_atomic_add": Store,
            "buffer_atomic_max": Store,
            "buffer_atomic_min": Store,
            "buffer_atomic_and": Store,
            "buffer_atomic_or": Store,
            "buffer_atomic_xor": Store,
            "buffer_atomic_xchg": Store,
        },
    ),
    (
        gluon_amd_rdna3,
        {
            "wmma": Dot,
        },
    ),
    (
        gluon_amd_rdna4,
        {
            "wmma": Dot,
        },
    ),
    (
        gluon_amd_cdna4_async_copy,
        {
            "global_load_to_shared": GluonAsyncCopyLoad,
            "buffer_load_to_shared": GluonBufferLoadToShared,
        },
    ),
    (
        gluon_amd_gfx1250,
        {
            "wmma": Dot,
            "buffer_load": Load,
            "buffer_store": Store,
        },
    ),
    (
        gluon_amd_tdm,
        {
            "make_tensor_descriptor": Allocate,
            "update_tensor_descriptor": Allocate,
            "async_load": Load,
            "async_store": Store,
        },
    ),
    (
        gluon_amd_async_copy,
        {
            "global_to_shared": GluonAsyncCopyLoad,
            "shared_to_global": GluonAsyncCopyStore,
        },
    ),
)
for namespace, attrs in _TMA_NAMESPACES:
    existing = _existing_ops(namespace, attrs)
    if existing:
        GLUON_NAMESPACES[namespace] = existing

GLUON_ADAPTERS: dict[type[Op], Callable[..., AdapterResult]] = {
    ProgramId: lambda axis, *_args, **_kwargs: AdapterResult(axis),
    Load: _gluon_pointer_load_adapter,
    Store: _gluon_pointer_store_adapter,
    GluonAsyncCopyLoad: _gluon_async_copy_load_adapter,
    GluonAsyncCopyStore: _gluon_async_copy_store_adapter,
    GluonBufferLoadToShared: _gluon_buffer_load_to_shared_adapter,
    MakeRange: _gluon_make_range_adapter,
    Splat: lambda shape, value, *_args, **_kwargs: AdapterResult(shape, value),
    Allocate: _gluon_allocate_adapter,
    Dot: _gluon_dot_adapter,
    BinaryOp: _gluon_binary_adapter,
    AddPtr: _gluon_binary_adapter,
    UnaryOp: _gluon_unary_adapter,
    TernaryOp: _gluon_where_adapter,
    Fma: lambda x, y, z, *_args, **_kwargs: AdapterResult(x, y, z),
    Rsqrt: lambda arg, *_args, **_kwargs: AdapterResult(arg),
}
_GLUON_BINARY_NUMPY_OPS: dict[str, Callable] = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
}
_GLUON_SEMANTIC_BINARY_NUMPY_OPS: dict[str, Callable] = {
    "less_than": np.less,
    "less_equal": np.less_equal,
    "greater_than": np.greater,
    "greater_equal": np.greater_equal,
}
_GLUON_UNARY_NUMPY_OPS: dict[str, Callable] = {
    "abs": np.abs,
    "ceil": np.ceil,
    "cos": np.cos,
    "exp": np.exp,
    "exp2": np.exp2,
    "floor": np.floor,
    "log": np.log,
    "log2": np.log2,
    "sin": np.sin,
    "sqrt": np.sqrt,
}
_TMA_LOAD_PRED_ARG_INDICES: dict[str, int] = {
    "async_load": 3,
    "async_load_im2col": 4,
    "async_copy_global_to_shared": 3,
    "async_copy_global_to_shared_im2col": 4,
}
GLUON_CALLABLE_ADAPTERS: dict[Callable, Callable[..., AdapterResult]] = {}
for namespace, attrs in GLUON_NAMESPACES.items():
    for attr, op_type in attrs.items():
        original = getattr(namespace, attr)
        if namespace is gluon_semantic.GluonSemantic and op_type is AddPtr:
            GLUON_CALLABLE_ADAPTERS[original] = _gluon_semantic_binary_adapter
        elif (
            namespace is gluon_semantic.GluonSemantic
            and op_type is BinaryOp
            and attr in _GLUON_SEMANTIC_BINARY_NUMPY_OPS
        ):
            GLUON_CALLABLE_ADAPTERS[original] = _gluon_semantic_binary_op_adapter(
                _GLUON_SEMANTIC_BINARY_NUMPY_OPS[attr]
            )
        elif op_type is BinaryOp and attr in _GLUON_BINARY_NUMPY_OPS:
            GLUON_CALLABLE_ADAPTERS[original] = _gluon_binary_op_adapter(
                _GLUON_BINARY_NUMPY_OPS[attr]
            )
        elif op_type is Dot:
            GLUON_CALLABLE_ADAPTERS[original] = _gluon_dot_overrider_adapter
        elif op_type is UnaryOp and attr in _GLUON_UNARY_NUMPY_OPS:
            GLUON_CALLABLE_ADAPTERS[original] = _gluon_unary_op_adapter(
                _GLUON_UNARY_NUMPY_OPS[attr]
            )
        if namespace is gluon_amd_tdm and attr == "async_load":
            GLUON_CALLABLE_ADAPTERS[original] = _gluon_descriptor_load_adapter(
                2,
                descriptor_kwarg="src",
                coords_kwarg="offsets",
            )
        elif attr in _TMA_LOAD_PRED_ARG_INDICES:
            GLUON_CALLABLE_ADAPTERS[original] = _gluon_descriptor_load_adapter(
                _TMA_LOAD_PRED_ARG_INDICES[attr],
                descriptor_kwarg="tensor_desc",
                coords_kwarg="coord",
            )


class GluonFrontend(Frontend):
    def __init__(self):
        definition = Frontend.from_namespaces(
            name="gluon",
            builder=None,
            namespaces=GLUON_NAMESPACES,
            adapters=GLUON_ADAPTERS,
        )
        super().__init__(
            name=definition.name,
            builder=definition.builder,
            original_ops=definition.original_ops,
            adapters=definition.adapters,
            namespaces=definition.namespaces,
        )

    @staticmethod
    def symbolic_ops_for_op_type(op_type: type[Op]) -> tuple[str, ...]:
        return TritonFrontend.symbolic_ops_for_op_type(op_type)

    def maybe_yield_for_multism(self) -> None:
        _maybe_yield_warp_specialize()

    def run_op_overrider(
        self,
        op: Callable,
        op_type: type[Op],
        op_overrider: Callable,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ):
        adapter = GLUON_CALLABLE_ADAPTERS.get(op, self.adapters[op_type])
        adapter_result = adapter(*args, **kwargs)
        return op_overrider(*adapter_result.args, **adapter_result.kwargs)

    def normalize_symbolic_value(self, value: Any) -> Any:
        if type(value).__name__ == "constexpr":
            constexpr_value = getattr(value, "value", _MISSING)
            if constexpr_value is not _MISSING:
                return constexpr_value
        if isinstance(value, gluon_core.tensor):
            value = value.handle
        if isinstance(value, TensorHandle):
            dtype_or_spec = TritonFrontend._normalize_triton_type(value.dtype)
            dtype = (
                dtype_or_spec.dtype
                if isinstance(dtype_or_spec, SymbolicTypeSpec)
                else dtype_or_spec
            )
            return SymbolicTensorValue(
                value.data,
                dtype,
                dict(getattr(value, "attr", {})),
            )
        if isinstance(value, (tl.block_type, tl.pointer_type, tl.core.dtype)):
            return TritonFrontend._normalize_triton_type(value)
        if isinstance(value, SymbolicTensorValue):
            return value
        return super().normalize_symbolic_value(value)

    def to_frontend_symbolic_value(self, value: Any) -> Any:
        if isinstance(value, tuple):
            return tuple(self.to_frontend_symbolic_value(item) for item in value)
        if isinstance(value, list):
            return [self.to_frontend_symbolic_value(item) for item in value]
        return value

    def from_frontend_symbolic_value(self, value: Any) -> Any:
        if isinstance(value, gluon_core.tensor):
            value = value.handle
        if isinstance(value, tuple):
            return tuple(self.from_frontend_symbolic_value(item) for item in value)
        if isinstance(value, list):
            return [self.from_frontend_symbolic_value(item) for item in value]
        return value

    def wrap_symbolic_concrete_fn(self, concrete_fn: Callable) -> Callable:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            frontend_args = tuple(self.to_frontend_symbolic_value(arg) for arg in args)
            frontend_kwargs = {
                key: self.to_frontend_symbolic_value(value)
                for key, value in kwargs.items()
            }
            ret = concrete_fn(*frontend_args, **frontend_kwargs)
            return self.from_frontend_symbolic_value(ret)

        return wrapped

    def patch_lang(self, fn, client_manager: Any = None) -> _LangPatchScope:
        from tilelens.core.simulation.gluon import patch_lang as gluon_patch_lang

        return gluon_patch_lang(fn)


frontend = register_frontend(GluonFrontend())
