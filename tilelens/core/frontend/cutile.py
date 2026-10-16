"""Adapt cuTile interpreter operations to the shared tracing callbacks."""

from collections.abc import Callable
from contextlib import contextmanager
from threading import Lock
from typing import Any

import numpy as np

from tilelens.core.data import Dot, Load, Op, ProgramId, ReduceSum, Store

from .base import AdapterResult, Frontend, _LangPatchScope, register_frontend


HAS_CUTILE = False
cutile_builder: Any = None
try:
    from tilelens.core.simulation.cutile import (
        Array,
        Tile,
        _indexed_access,
        _tile_access,
        _tile_value,
        cutile_builder,
    )

    HAS_CUTILE = True
except ModuleNotFoundError:
    pass


def _cutile_memory_adapter(
    array: Any,
    index: Any,
    shape: Any,
    order: Any = "C",
    traversal_steps: Any = None,
) -> AdapterResult:
    assert HAS_CUTILE
    _, axes, access_shape, starts = _tile_access(
        array, index, shape, order, traversal_steps
    )
    root = array
    while root._parent is not None:
        root = root._parent
    keys: list[Any] = [None] * array.ndim
    mask = np.ones(access_shape, dtype=bool)
    for axis, (start, size, array_axis) in enumerate(zip(starts, access_shape, axes)):
        coordinates = start + np.arange(size)
        broadcast_shape = [1] * array.ndim
        broadcast_shape[axis] = size
        coordinates = coordinates.reshape(broadcast_shape)
        keys[array_axis] = Tile(coordinates + array._origin[array_axis])
        mask &= (coordinates >= 0) & (coordinates < array.shape[array_axis])
    return AdapterResult(root, Tile(mask), tuple(keys))


def _cutile_load_adapter(
    array: Any,
    index: Any,
    shape: Any = None,
    *,
    order: Any = "C",
    traversal_steps: Any = None,
    **kwargs: Any,
) -> AdapterResult:
    if shape is None:
        return _cutile_indexed_adapter(
            array, index, kwargs.get("mask"), kwargs.get("check_bounds", True)
        )
    return _cutile_memory_adapter(array, index, shape, order, traversal_steps)


def _cutile_store_adapter(
    array: Any,
    index: Any,
    tile: Any = None,
    *,
    order: Any = "C",
    traversal_steps: Any = None,
    **kwargs: Any,
) -> AdapterResult:
    assert HAS_CUTILE
    # Scatter uses the Store callback with indexed coordinates.
    if kwargs.get("_indexed", False):
        return _cutile_indexed_adapter(
            array, index, kwargs.get("mask"), kwargs.get("check_bounds", True)
        )
    return _cutile_memory_adapter(
        array, index, _tile_value(tile).shape, order, traversal_steps
    )


def _cutile_indexed_adapter(
    array: Any,
    indices: Any,
    mask: Any = None,
    check_bounds: bool = True,
) -> AdapterResult:
    keys, valid = _indexed_access(array, indices, mask, check_bounds)
    # Memory callbacks expect arrays, even for scalar indices.
    if valid.ndim == 0:
        keys = tuple(key.reshape(1) for key in keys)
        valid = valid.reshape(1)
    root = array
    while root._parent is not None:
        root = root._parent
    return AdapterResult(
        root,
        Tile(valid),
        tuple(Tile(key + origin) for key, origin in zip(keys, array._origin)),
    )


def _cutile_dot_adapter(x: Any, y: Any, *_args: Any, **_kwargs: Any) -> AdapterResult:
    assert HAS_CUTILE
    # Copy read-only tiles before the visualizer converts them to Torch tensors.
    if len(_args) == 3:
        lhs_data, rhs_data = cutile_builder._scaled_operands(x, y, _args[0], _args[1])
        return AdapterResult(Array(lhs_data), Array(rhs_data))
    lhs, rhs = Array(x.data), Array(y.data)
    if lhs.data.dtype.kind in "iu":
        lhs, rhs = (
            Array(lhs.data.astype(np.float32)),
            Array(rhs.data.astype(np.float32)),
        )
    return AdapterResult(Array(lhs.numpy().copy()), Array(rhs.numpy().copy()))


def _cutile_reduce_sum_adapter(
    input_tensor: Any,
    axis: Any = None,
    *,
    keepdims: bool = False,
    **kwargs: Any,
) -> AdapterResult:
    return AdapterResult(input_tensor, axis, keepdims)


CUTILE_ADAPTERS: dict[type[Op], Callable[..., AdapterResult]] = {}
CUTILE_NAMESPACES: dict[Any, dict[str, type[Op]]] = {}
if HAS_CUTILE:
    assert cutile_builder is not None

    CUTILE_NAMESPACES = {
        cutile_builder: {
            "bid": ProgramId,
            "load": Load,
            "store": Store,
            "matmul": Dot,
            "mma": Dot,
            "mma_scaled": Dot,
            "sum": ReduceSum,
        }
    }

    CUTILE_ADAPTERS = {
        ProgramId: lambda axis, *_args, **_kwargs: AdapterResult(axis),
        Load: _cutile_load_adapter,
        Store: _cutile_store_adapter,
        Dot: _cutile_dot_adapter,
        ReduceSum: _cutile_reduce_sum_adapter,
    }


class CuTileFrontend(Frontend):
    def __init__(self) -> None:
        definition = Frontend.from_namespaces(
            name="cutile",
            builder=cutile_builder,
            namespaces=CUTILE_NAMESPACES,
            adapters=CUTILE_ADAPTERS,
        )
        super().__init__(
            name=definition.name,
            builder=definition.builder,
            original_ops=definition.original_ops,
            adapters=definition.adapters,
            namespaces=definition.namespaces,
        )

    def patch_lang(self, fn, client_manager: Any = None) -> _LangPatchScope:
        from tilelens.core.simulation.cutile import cutile_patch_lang

        scope = _LangPatchScope()
        cutile_patch_lang(scope)
        return scope


frontend = register_frontend(CuTileFrontend())


_launch_scope_lock = Lock()
_launch_scope = None
_launch_options = None
_launch_users = 0


@contextmanager
def interpreted_launches(client="tracer"):
    """Run ct.launch on CPU; overlapping scopes must use identical settings."""
    global _launch_scope, _launch_options, _launch_users

    import cuda.tile as ct
    import tilelens
    from tilelens.core.trace import CuTileTrace

    def launch(stream, grid, kernel, args):
        runner = (
            kernel
            if isinstance(kernel, CuTileTrace)
            else tilelens.trace(client, frontend="cutile")(kernel)
        )
        return runner.run(*args, grid=grid)

    # The global entry point can be shared by worker threads. Hold this guard
    # only while changing ownership; each runner serializes its own execution.
    with _launch_scope_lock:
        options = client
        if _launch_users and options != _launch_options:
            raise RuntimeError(
                "Overlapping interpreted_launches contexts must use the same "
                "client settings"
            )
        if not _launch_users:
            _launch_scope = _LangPatchScope()
            _launch_scope.set_attr(ct, "launch", launch)
            _launch_options = options
        _launch_users += 1
    try:
        yield
    finally:
        with _launch_scope_lock:
            _launch_users -= 1
            if not _launch_users:
                _launch_scope.restore()
                _launch_scope = None
                _launch_options = None
