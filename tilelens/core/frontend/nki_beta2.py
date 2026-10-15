from collections.abc import Callable
from typing import Any

from tilelens.core.data import Allocate, Dot, Op, ProgramId, Transfer

from .base import AdapterResult, Frontend, _LangPatchScope, register_frontend


HAS_NKI_BETA2 = False
nisa = None
nl = None
NDArray = None
nki_builder = None
try:
    from tilelens.core.simulation import nki_beta2 as _nki_beta2  # type: ignore

    NDArray = _nki_beta2.NDArray
    nki_builder = _nki_beta2.nki_builder
    nisa = _nki_beta2.nisa
    nl = _nki_beta2.nl

    HAS_NKI_BETA2 = True
except ModuleNotFoundError:
    pass


def _nki_beta2_dot_adapter(
    dst: Any, stationary: Any, moving: Any, *_args: Any, **kwargs: Any
) -> AdapterResult:
    assert NDArray is not None
    x = NDArray(value=stationary.data.T)
    return AdapterResult(x, moving)


NKI_BETA2_ADAPTERS: dict[type[Op], Callable[..., AdapterResult]] = {}
NKI_BETA2_NAMESPACES: dict[Any, dict[str, type[Op]]] = {}
if HAS_NKI_BETA2:
    assert nki_builder is not None

    NKI_BETA2_NAMESPACES = {
        _nki_beta2: {
            "program_id": ProgramId,
            "ndarray": Allocate,
            "nc_matmul": Dot,
            "dma_copy": Transfer,
            "tensor_copy": Transfer,
        },
    }

    NKI_BETA2_ADAPTERS = {
        ProgramId: lambda axis, *_args, **_kwargs: AdapterResult(axis),
        Allocate: lambda *_args, **_kwargs: AdapterResult(),
        Dot: _nki_beta2_dot_adapter,
        Transfer: lambda dst, src, *_args, **_kwargs: AdapterResult(
            src,
            dst,
            src.buffer,
            dst.buffer,
        ),
    }


class NKIBeta2Frontend(Frontend):
    def __init__(self):
        definition = Frontend.from_namespaces(
            name="nki_beta2",
            builder=nki_builder,
            namespaces=NKI_BETA2_NAMESPACES,
            adapters=NKI_BETA2_ADAPTERS,
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

        return NKITrace(kernel, client, beta2=True)

    def patch_lang(self, fn, client_manager: Any = None) -> _LangPatchScope:
        from tilelens.core.simulation.nki_beta2 import nki_patch_lang

        scope = _LangPatchScope()
        nki_patch_lang(scope)
        return scope


frontend = register_frontend(NKIBeta2Frontend())
