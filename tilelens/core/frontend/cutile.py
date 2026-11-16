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


def _root(array: Any) -> Any:
    """The array that owns the memory of a sliced view."""
    while array._parent is not None:
        array = array._parent
    return array


class _AccessArray:
    """The root array, with byte offsets for only the elements one access touches.

    The tracer reads offsets through ``get_offsets()`` and indexes them with the
    adapter's keys, so tracing costs O(tile) per access instead of O(array).
    """

    def __init__(self, root: Any, keys: tuple, mask: np.ndarray) -> None:
        keys = tuple(np.asarray(_tile_value(key)) for key in keys)
        shape = np.broadcast_shapes(*(key.shape for key in keys), mask.shape)
        offsets = np.zeros(shape, dtype=np.int64)
        for key, extent, stride in zip(keys, root.shape, root.data.strides):
            # Masked-off lanes may lie outside the array; clamp them.
            offsets = offsets + np.clip(key, 0, max(extent - 1, 0)) * stride
        self._root = root
        self._offsets = offsets
        # Identity keys into the offsets array above.
        self.keys = tuple(Tile(key) for key in np.indices(shape))

    def get_offsets(self) -> Any:
        return Array(self._offsets)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._root, name)


def _access_result(array: Any, keys: tuple, mask: np.ndarray) -> AdapterResult:
    access = _AccessArray(_root(array), keys, mask)
    return AdapterResult(access, Tile(mask), access.keys)


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
    keys: list[Any] = [None] * array.ndim
    mask = np.ones(access_shape, dtype=bool)
    for axis, (start, size, array_axis) in enumerate(zip(starts, access_shape, axes)):
        coordinates = start + np.arange(size)
        broadcast_shape = [1] * array.ndim
        broadcast_shape[axis] = size
        coordinates = coordinates.reshape(broadcast_shape)
        keys[array_axis] = coordinates + array._origin[array_axis]
        mask &= (coordinates >= 0) & (coordinates < array.shape[array_axis])
    return _access_result(array, tuple(keys), mask)


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
    return _access_result(
        array, tuple(key + origin for key, origin in zip(keys, array._origin)), valid
    )


def _cutile_dot_adapter(x: Any, y: Any, *_args: Any, **_kwargs: Any) -> AdapterResult:
    assert HAS_CUTILE
    # Copy read-only tiles before the visualizer converts them to Torch tensors.
    # mma takes (acc,); mma_scaled takes (x_scale, y, y_scale[, acc]) with acc= allowed.
    if len(_args) >= 2:
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
    from tilelens.core.simulation.cutile import cutile_execution_lock
    from tilelens.core.trace import CuTileTrace

    def launch(stream, grid, kernel, args, /, *, programmatic_dependent_launch=False):
        # The CPU interpreter runs launches in order, so dependent launch is a no-op.
        if isinstance(kernel, CuTileTrace):
            return kernel.run(*args, grid=grid)
        # Reset under the execution lock so concurrent launches sharing this
        # tracer cannot clear or append to each other's records.
        with cutile_execution_lock:
            records = getattr(client, "records", None)
            if isinstance(records, list):
                records.clear()
            return tilelens.trace(client, frontend="cutile")(kernel).run(
                *args, grid=grid
            )

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
