from collections.abc import Callable
from typing import Any

from tilelens.core.data import (
    Allocate,
    Dot,
    Load,
    MakeRange,
    Op,
    ProgramId,
    ReduceSum,
    Store,
    UnaryOp,
)

from .base import AdapterResult, Frontend, _LangPatchScope, register_frontend


HAS_NKI = False
nki_builder = None
try:
    from tilelens.core.simulation.nki import NDArray, nki_builder  # type: ignore

    HAS_NKI = True
except ModuleNotFoundError:
    pass


def _nki_dot_adapter(x: Any, y: Any, *_args: Any, **_kwargs: Any) -> AdapterResult:
    assert HAS_NKI
    if _kwargs.get("transpose_x", False):
        x = NDArray(value=x.data.T, name=x.name)
    return AdapterResult(x, y)


def _nki_reduce_sum_adapter(
    input_tensor: Any, *args: Any, mask: Any = None, **kwargs: Any
) -> AdapterResult:
    axis = args[0] if args else kwargs.get("axis")
    keep_dims = kwargs.get("keep_dims", kwargs.get("keepdims", False))
    return AdapterResult(input_tensor, axis, keep_dims)


NKI_ADAPTERS: dict[type[Op], Callable[..., AdapterResult]] = {}
NKI_NAMESPACES: dict[Any, dict[str, type[Op]]] = {}
if HAS_NKI:
    assert nki_builder is not None

    NKI_NAMESPACES = {
        nki_builder: {
            "program_id": ProgramId,
            "ndarray": Allocate,
            "load": Load,
            "store": Store,
            "matmul": Dot,
            "_unary_op": UnaryOp,
            "sum": ReduceSum,
            "arange": MakeRange,
        }
    }

    NKI_ADAPTERS = {
        ProgramId: lambda axis, *_args, **_kwargs: AdapterResult(axis),
        Allocate: lambda *_args, **_kwargs: AdapterResult(),
        Load: lambda src, keys, *, mask=None, **_kwargs: AdapterResult(
            src,
            mask,
            keys,
        ),
        Store: lambda dst, keys, value, *, mask=None, **_kwargs: AdapterResult(
            dst,
            mask,
            keys,
        ),
        Dot: _nki_dot_adapter,
        ReduceSum: _nki_reduce_sum_adapter,
    }


class NKIFrontend(Frontend):
    def __init__(self):
        definition = Frontend.from_namespaces(
            name="nki",
            builder=nki_builder,
            namespaces=NKI_NAMESPACES,
            adapters=NKI_ADAPTERS,
        )
        super().__init__(
            name=definition.name,
            builder=definition.builder,
            original_ops=definition.original_ops,
            adapters=definition.adapters,
            namespaces=definition.namespaces,
        )

    def wrap_kernel(self, kernel, client):
        from ..trace import NKITrace

        return NKITrace(kernel, client, beta2=False)

    def patch_lang(self, fn, client_manager: Any = None) -> _LangPatchScope:
        from tilelens.core.simulation.nki import nki_patch_lang

        scope = _LangPatchScope()
        nki_patch_lang(scope)
        return scope


frontend = register_frontend(NKIFrontend())
