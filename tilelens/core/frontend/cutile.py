"""Adapt cuTile interpreter operations to the shared tracing callbacks."""

from collections.abc import Callable
from contextlib import contextmanager
import sys
from threading import Lock
from typing import Any

import numpy as np

from tilelens.core.data import Dot, Load, Op, ProgramId, ReduceSum, Store

from ..patch import LoopSite, loop_file_token
from .base import AdapterResult, Frontend, _LangPatchScope, register_frontend


HAS_CUTILE = False
cutile_builder: Any = None
try:
    from tilelens.core.simulation.cutile import (
        Array,
        Tile,
        _indexed_access,
        _numpy_dtype,
        _promotion_type,
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


def cutile_zero_load(
    array: Any,
    index: Any,
    shape: Any = None,
    *,
    order: Any = "C",
    traversal_steps: Any = None,
    **kwargs: Any,
) -> Any:
    """Return zeros shaped like ct.load or ct.gather without reading memory."""
    assert HAS_CUTILE
    if shape is None:
        _, valid = _indexed_access(
            array, index, kwargs.get("mask"), kwargs.get("check_bounds", True)
        )
        return Tile(np.zeros(valid.shape, dtype=array.data.dtype))
    shape, *_ = _tile_access(array, index, shape, order, traversal_steps)
    return Tile(np.zeros(shape, dtype=array.data.dtype))


def cutile_skip_store(*_args: Any, **_kwargs: Any) -> None:
    """Drop ct.store and ct.scatter writes."""
    return None


def cutile_zero_dot(x: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
    """Return zeros shaped like matmul, mma, or mma_scaled without computing."""
    assert HAS_CUTILE
    # mma takes (acc,); mma_scaled takes (y, y_scale[, acc]) with acc= allowed.
    acc = kwargs.get("acc", args[-1] if args else None)
    if acc is not None:
        return Tile(np.zeros(acc.shape, dtype=_tile_value(acc).dtype))

    from cuda.tile._ir.ops_utils import promote_dtypes

    common = promote_dtypes(_promotion_type(x), _promotion_type(y))
    lhs, rhs = _tile_value(x), _tile_value(y)
    # An empty contraction axis yields matmul's output shape with no arithmetic.
    lhs = np.zeros(lhs.shape[:-1] + (0,))
    rhs = np.zeros(rhs.shape[:-2] + (0,) + rhs.shape[-1:] if rhs.ndim > 1 else (0,))
    return Tile(np.matmul(lhs, rhs).astype(_numpy_dtype(common)), common)


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
        self._loop_patched = False
        self._current_client_manager: Any = None

    def run_op_overrider(
        self,
        op: Callable,
        op_type: type[Op],
        op_overrider: Callable,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        # Atomics access memory through load and store. Triton keeps atomics as
        # separate ops that overriders never replace, so run them unchanged.
        if kwargs.get("_atomic", False):
            return op(*args, **kwargs)
        return op_overrider(*args, **kwargs)

    def patch_for_loop(self) -> None:
        self._loop_patched = True

    def unpatch_for_loop(self) -> None:
        self._loop_patched = False

    def loop_iter_wrapper(
        self,
        iterable_callable: Callable,
        iter_args: Any,
        iter_kwargs: Any,
        lineno: int,
        range_type: str,
    ) -> Any:
        """Report a rewritten for-loop to clients, like Triton's loop wrapper."""
        client_manager = self._current_client_manager
        if client_manager is None:
            # A rewritten helper can outlive its traced launch.
            return iterable_callable(*iter_args, **iter_kwargs)
        caller_file = sys._getframe(1).f_code.co_filename
        loop_site = LoopSite(lineno, loop_file_token(caller_file))
        return client_manager.loop_iter_wrapper(
            iterable_callable, iter_args, iter_kwargs, loop_site, range_type
        )

    def rewrite_loops(self, fn: Any) -> Any:
        """Return fn with its for-loops routed through loop_iter_wrapper."""
        from .cutile_transform import rewrite_loops

        return rewrite_loops(fn, self.loop_iter_wrapper)

    def patch_lang(self, fn, client_manager: Any = None) -> _LangPatchScope:
        from tilelens.core.simulation import cutile as cutile_simulation

        scope = _LangPatchScope()
        cutile_simulation.cutile_patch_lang(scope)
        # Like Triton and NKI, traced runs execute kernels and helpers with
        # their for-loops rewritten; standalone runs execute the originals.
        if self._loop_patched and client_manager is not None:
            scope.set_attr(self, "_current_client_manager", client_manager)
            scope.set_attr(cutile_simulation, "_rewrite_loops", self.rewrite_loops)
        return scope


frontend = register_frontend(CuTileFrontend())


_launch_scope_lock = Lock()
_launch_scope = None
_launch_options = None
_launch_users = 0
_launch_cli = False


@contextmanager
def interpreted_launches(client="tracer", *, cli=False):
    """Run ct.launch on CPU; overlapping scopes must use identical settings.

    CLI wrappers pass cli=True. Like the Triton CLI, that scope traces each
    kernel once so client state accumulates across launches, and it rejects
    user-traced kernels and nested scopes.
    """
    global _launch_scope, _launch_options, _launch_users, _launch_cli

    import cuda.tile as ct
    import tilelens
    from tilelens.core.trace import CuTileTrace

    traces: dict[Any, CuTileTrace] = {}

    def launch(stream, grid, kernel, args):
        if cli:
            # tilelens.trace() rejects already-traced kernels under the CLI.
            if kernel not in traces:
                traces[kernel] = tilelens.trace(client, frontend="cutile")(kernel)
            runner = traces[kernel]
        elif isinstance(kernel, CuTileTrace):
            runner = kernel
        else:
            runner = tilelens.trace(client, frontend="cutile")(kernel)
        return runner.run(*args, grid=grid)

    # The global entry point can be shared by worker threads. Hold this guard
    # only while changing ownership; each runner serializes its own execution.
    with _launch_scope_lock:
        options = client
        if _launch_users and (cli or _launch_cli):
            raise RuntimeError(
                "interpreted_launches() cannot be used together with CLI wrapper "
                "(e.g., tile-profiler). Please remove interpreted_launches() from "
                "your code when using CLI tools."
            )
        if _launch_users and options != _launch_options:
            raise RuntimeError(
                "Overlapping interpreted_launches contexts must use the same "
                "client settings"
            )
        if not _launch_users:
            _launch_scope = _LangPatchScope()
            _launch_scope.set_attr(ct, "launch", launch)
            _launch_options = options
            _launch_cli = cli
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
                _launch_cli = False
