"""Sanitize cuTile accesses whose bounds the user's mask is responsible for.

cuTile bounds-checks tiled ``ct.load``/``ct.store`` and, by default, gather,
scatter, and atomics itself. Only ``check_bounds=False`` leaves an indexed
access to the user's ``mask=``, like Triton's ``tl.load(ptr, mask)``, so only
those accesses are checked.

The kernel runs normally. Tiles built from ``ct.bid``, ``ct.arange``, loop
variables, constants, and arithmetic carry a symbolic shadow, and Z3 checks
each access for every block at once. Indices derived from loaded data have no
shadow; those accesses are checked concretely in every block instead.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from z3 import And, BoolVal, Int, Or, sat
from z3.z3 import BoolRef, IntNumRef

from ...core.callbacks import ForLoopCallbacks, OpCallbacks
from ...core.data import (
    AtomicCas,
    AtomicRMW,
    BinaryOp,
    Broadcast,
    CastImpl,
    ExpandDims,
    Load,
    MakeRange,
    Op,
    ProgramId,
    Reshape,
    Splat,
    Store,
    TernaryOp,
    UnaryOp,
    Unsplat,
)
from ...core.patch import LoopSite
from ...core.symbolic_metadata import INT32
from ..symbolic_engine import LoopContext, SymbolicExpr, _range_to_iterator_constraint
from .sanitizer import SymbolicSanitizer

_SHADOW = "_tilelens_symbolic"

_CUTILE_OP_TYPES = (
    ProgramId,
    MakeRange,
    BinaryOp,
    UnaryOp,
    TernaryOp,
    Reshape,
    Broadcast,
    ExpandDims,
    CastImpl,
    Splat,
    Unsplat,
    Load,
    Store,
    AtomicRMW,
    AtomicCas,
)

_BINARY_OPS = {
    "add": "add",
    "sub": "sub",
    "mul": "mul",
    "floordiv": "idiv",
    "mod": "mod",
    "less": "less",
    "less_equal": "less_equal",
    "greater": "greater",
    "greater_equal": "greater_equal",
    "equal": "equal",
    "not_equal": "not_equal",
    "maximum": "maximum",
    "minimum": "minimum",
    "bitwise_and": "bitwise_and",
    "bitwise_or": "bitwise_or",
    "bitwise_xor": "bitwise_xor",
    "bitwise_lshift": "left_shift",
    "bitwise_rshift": "right_shift",
}

# Shape-only ops keep their input's symbolic value.
_SHAPE_OPS = ("reshape", "broadcast_to", "expand_dims", "__getitem__")

_LOW_PRECISION_TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}


def _tile_type() -> Any:
    from ...core.simulation.cutile import Tile

    return Tile


def _const(value: Any) -> SymbolicExpr | None:
    """Symbolic constant for a launch-invariant Python value or Tile."""
    if isinstance(value, _tile_type()):
        data = value.data
        if data.dtype.kind not in "biu":
            return None
        if data.size == 1:
            return SymbolicExpr.create("const", int(data.item()), INT32)
        return SymbolicExpr.create("const", data.astype(np.int64), INT32)
    if isinstance(value, (bool, int, np.integer)):
        return SymbolicExpr.create("const", int(value), INT32)
    return None


def _shadow(value: Any) -> SymbolicExpr | None:
    """The symbolic value of an index or mask, or None if it is unknown."""
    if isinstance(value, _tile_type()):
        return getattr(value, _SHADOW, None)
    return _const(value)


def _arg(args: tuple, kwargs: dict, index: int, name: str, default: Any = None):
    if len(args) > index:
        return args[index]
    return kwargs.get(name, default)


def _torch_view(array: Any) -> torch.Tensor | None:
    """A Torch tensor sharing the array's memory, for reports."""
    data = array.data
    try:
        return torch.from_numpy(data)
    except (TypeError, ValueError):
        pass
    storage = {1: np.uint8, 2: np.int16, 4: np.int32, 8: np.int64}.get(
        data.dtype.itemsize
    )
    if storage is None:
        return None
    try:
        tensor = torch.from_numpy(data.view(storage))
    except (TypeError, ValueError):
        return None
    torch_dtype = _LOW_PRECISION_TORCH_DTYPES.get(data.dtype.name)
    return tensor.view(torch_dtype) if torch_dtype is not None else tensor


def _lanes(value: Any, count: int) -> list[Any]:
    return value if isinstance(value, list) else [value] * count


def _as_bool(value: Any) -> BoolRef:
    return value if isinstance(value, BoolRef) else value != 0


class CuTileSanitizer(SymbolicSanitizer):
    """Check cuTile gather, scatter, and atomics run with check_bounds=False."""

    def __init__(self, abort_on_error: bool = True):
        super().__init__(abort_on_error=abort_on_error)
        self.array_tensors: dict[int, tuple[Any, torch.Tensor]] = {}
        # Access signature -> whether it can go out of bounds.
        self.checked_accesses: dict[int, bool] = {}

    # ── Launch setup ──────────────────────────────────────────────

    def arg_callback(self, name: str, arg: Any, arg_cvt: Any) -> None:
        from ...core.simulation.cutile import Array

        value = arg_cvt if arg_cvt is not None else arg
        if isinstance(value, Array):
            tensor = _torch_view(value)
            if tensor is not None:
                self.array_tensors[id(value)] = (value, tensor)
                SymbolicSanitizer.arg_callback(self, name, tensor, None)
            return
        if isinstance(value, _tile_type()):
            # Scalar arguments are the same in every block.
            shadow = _const(value)
            if shadow is not None:
                setattr(value, _SHADOW, shadow)
            self._cache_non_tensor_arg(name, value.data.item())
            return
        self._cache_non_tensor_arg(name, value)

    def grid_callback(self, grid: tuple[int, ...]) -> None:
        super().grid_callback(grid)
        self.checked_accesses.clear()

    def _clear_symbolic_launch_state(self) -> None:
        super()._clear_symbolic_launch_state()
        self.array_tensors.clear()
        self.checked_accesses.clear()

    # ── Op callbacks ──────────────────────────────────────────────

    def register_op_callback(self, op_type: type[Op]) -> OpCallbacks:
        if op_type not in _CUTILE_OP_TYPES:
            return OpCallbacks()
        cached = self.op_callback_cache.get(op_type)
        if cached is None:
            overrider = self.lock_fn(self._run_op)
            setattr(overrider, "receives_op", True)
            cached = OpCallbacks(op_overrider=overrider)
            self.op_callback_cache[op_type] = cached
        return cached

    def _run_op(self, op: Callable, *args: Any, **kwargs: Any) -> Any:
        name = op.__name__
        if name == "__bool__":
            # Python control flow on a block-dependent Tile can take different
            # paths in other blocks, so every block must run.
            shadow = _shadow(args[0])
            if shadow is None or shadow.has_op("pid"):
                self._on_data_dependent_value()
            return op(*args, **kwargs)
        if name in ("load", "store") or name.startswith("atomic_"):
            return self._run_access(op, name, args, kwargs)

        ret = op(*args, **kwargs)
        if isinstance(ret, _tile_type()):
            shadow = self._op_shadow(name, args, kwargs, ret)
            if shadow is not None:
                setattr(ret, _SHADOW, shadow)
        return ret

    def _op_shadow(
        self, name: str, args: tuple, kwargs: dict, ret: Any
    ) -> SymbolicExpr | None:
        if name == "bid":
            return SymbolicExpr.create("pid", int(_arg(args, kwargs, 0, "axis")))
        if name == "arange":
            size = _const(_arg(args, kwargs, 0, "size"))
            if size is None:
                return None
            return SymbolicExpr.create(
                "arange",
                SymbolicExpr.from_value(INT32),
                SymbolicExpr.create("const", 0, INT32),
                size,
            )
        if name in _BINARY_OPS:
            lhs = _shadow(_arg(args, kwargs, 0, "x"))
            rhs = _shadow(_arg(args, kwargs, 1, "y"))
            if lhs is None or rhs is None:
                return None
            return SymbolicExpr.create(_BINARY_OPS[name], lhs, rhs)
        if name == "negative":
            arg = _shadow(_arg(args, kwargs, 0, "x"))
            if arg is None:
                return None
            return SymbolicExpr.create(
                "sub", SymbolicExpr.create("const", 0, INT32), arg
            )
        if name == "where":
            parts = [
                _shadow(_arg(args, kwargs, i, key))
                for i, key in enumerate(("cond", "x", "y"))
            ]
            if any(part is None for part in parts):
                return None
            return SymbolicExpr.create("where", *parts)
        if name in _SHAPE_OPS:
            return _shadow(args[0] if args else None)
        if name == "astype":
            # Integer and boolean casts keep index values.
            if ret.data.dtype.kind not in "biu":
                return None
            return _shadow(_arg(args, kwargs, 0, "x"))
        if name == "zeros":
            return SymbolicExpr.create("const", 0, INT32)
        if name == "ones":
            return SymbolicExpr.create("const", 1, INT32)
        if name == "full":
            if ret.data.dtype.kind not in "biu":
                return None
            return _shadow(_arg(args, kwargs, 1, "fill_value"))
        return None

    # ── Memory accesses ───────────────────────────────────────────

    def _run_access(self, op: Callable, name: str, args: tuple, kwargs: dict):
        array, indices = args[0], _arg(args, kwargs, 1, "index")
        if name == "load":
            indexed = _arg(args, kwargs, 2, "shape") is None
            op_type: type[Load] | type[Store] = Load
        elif name == "store":
            indexed = bool(kwargs.get("_indexed", False))
            op_type = Store
        else:
            # Atomics read and write; report them as stores.
            indexed = True
            op_type = Store
        # cuTile bounds-checks everything else itself.
        if not indexed or kwargs.get("check_bounds", True):
            return op(*args, **kwargs)

        mask = None if name.startswith("atomic_") else kwargs.get("mask")
        if self._check_indexed(array, indices, mask, op_type):
            # Keep going after a report: out-of-bounds lanes are skipped.
            kwargs = {**kwargs, "check_bounds": True}
        return op(*args, **kwargs)

    def _check_indexed(
        self,
        array: Any,
        indices: Any,
        mask: Any,
        op_type: type[Load] | type[Store],
    ) -> bool:
        """Report an out-of-bounds index under the mask; return True if found."""
        indices = indices if isinstance(indices, tuple) else (indices,)
        shadows = [_shadow(index) for index in indices]
        mask_shadow = None if mask is None else _shadow(mask)
        known = [shadow for shadow in shadows if shadow is not None]
        if len(known) == len(shadows) and (mask is None or mask_shadow is not None):
            found = self._check_symbolic(array, known, mask_shadow, op_type)
            if found is not None:
                return found
        # Indices from loaded data differ per block: check this block's values.
        self._on_data_dependent_value()
        return self._check_concrete(array, indices, mask, op_type, shadows)

    def _check_symbolic(
        self,
        array: Any,
        shadows: list[SymbolicExpr],
        mask_shadow: SymbolicExpr | None,
        op_type: type[Load] | type[Store],
    ) -> bool | None:
        """Z3 check over all blocks; None means it must be checked concretely."""
        loop_constraints = [
            ctx.iterator_constraint
            for ctx in self.loop_stack
            if ctx.iterator_constraint is not None
        ]
        signature = hash(
            (
                op_type,
                id(array),
                array.shape,
                tuple(shadow.signature() for shadow in shadows),
                None if mask_shadow is None else mask_shadow.signature(),
                tuple(hash(constraint) for constraint in loop_constraints),
            )
        )
        if signature in self.checked_accesses:
            # Already reported once; keep skipping out-of-bounds lanes.
            return self.checked_accesses[signature]

        exprs, constraints = [], []
        for shadow in shadows:
            expr, expr_constraints = shadow.eval()
            exprs.append(expr)
            constraints.append(expr_constraints)
        if mask_shadow is not None:
            mask_expr, mask_constraints = mask_shadow.eval()
            constraints.append(mask_constraints)
        else:
            mask_expr = BoolVal(True)
        sizes = {len(e) for e in [*exprs, mask_expr] if isinstance(e, list)}
        if len(sizes) > 1:
            return None
        count = sizes.pop() if sizes else 1
        index_lanes = [_lanes(expr, count) for expr in exprs]
        mask_lanes = [_as_bool(m) for m in _lanes(mask_expr, count)]
        violations = [
            And(
                mask_lanes[lane],
                Or(
                    *(
                        Or(index[lane] < 0, index[lane] >= extent)
                        for index, extent in zip(index_lanes, array.shape)
                    )
                ),
            )
            for lane in range(count)
        ]

        solver = self.solver
        assert solver is not None
        solver.push()
        try:
            solver.add(Or(*violations))
            for constraint in [*constraints, *loop_constraints]:
                if constraint is not None:
                    solver.add(constraint)
            if solver.check() != sat:
                self.checked_accesses[signature] = False
                return False
            model = solver.model()
            lane = next(
                lane
                for lane, violation in enumerate(violations)
                if model.evaluate(violation, model_completion=True)
            )
            position = [
                model.evaluate(index[lane], model_completion=True)
                for index in index_lanes
            ]
        finally:
            solver.pop()
        if not all(isinstance(value, IntNumRef) for value in position):
            return None
        self.checked_accesses[signature] = True
        self._report_index(
            array,
            [value.as_long() for value in position],
            op_type,
            shadows[0],
        )
        return True

    def _check_concrete(
        self,
        array: Any,
        indices: tuple,
        mask: Any,
        op_type: type[Load] | type[Store],
        shadows: list[SymbolicExpr | None],
    ) -> bool:
        Tile = _tile_type()
        keys = np.broadcast_arrays(
            *(
                index.data if isinstance(index, Tile) else np.asarray(index)
                for index in indices
            )
        )
        active = np.ones(keys[0].shape, dtype=bool)
        if mask is not None:
            mask_value = mask.data if isinstance(mask, Tile) else np.asarray(mask)
            active &= np.broadcast_to(mask_value.astype(bool), active.shape)
        inside = np.ones_like(active)
        for key, extent in zip(keys, array.shape):
            inside &= (key >= 0) & (key < extent)
        bad = np.argwhere(active & ~inside)
        if not len(bad):
            return False
        first = tuple(bad[0])
        shadow = next((s for s in shadows if s is not None), None)
        self._report_index(array, [int(key[first]) for key in keys], op_type, shadow)
        return True

    def _report_index(
        self,
        array: Any,
        position: list[int],
        op_type: type[Load] | type[Store],
        symbolic_expr: SymbolicExpr | None,
    ) -> None:
        root = array
        while root._parent is not None:
            root = root._parent
        entry = self.array_tensors.get(id(root))
        tensor = entry[1] if entry is not None else _torch_view(root)
        address = array.data.ctypes.data + sum(
            index * stride for index, stride in zip(position, array.data.strides)
        )
        self._report(op_type, tensor, address, symbolic_expr)

    # ── Loops ─────────────────────────────────────────────────────

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        return ForLoopCallbacks(
            range_wrapper_factory=self.lock_fn(self._cutile_loop_bounds),
            before_loop_callback=self.lock_fn(self._cutile_loop_before),
            loop_iter_overrider=self.lock_fn(self._cutile_loop_iter),
            after_loop_callback=self.lock_fn(self._cutile_loop_after),
        )

    def _cutile_loop_bounds(
        self,
        _iterable: Any,
        _loop_site: LoopSite,
        _range_type: str,
        iter_args: tuple,
        iter_kwargs: dict,
        _iter_callable: Callable,
    ) -> None:
        # A block-dependent loop length differs in other blocks.
        for value in (*iter_args, *iter_kwargs.values()):
            if isinstance(value, _tile_type()):
                shadow = _shadow(value)
                if shadow is None or shadow.has_op("pid"):
                    self._on_data_dependent_value()
        return None

    def _cutile_loop_before(self, loop_site: LoopSite, iterable: Any) -> None:
        if not isinstance(iterable, range):
            return
        idx_z3 = Int(f"loop_i_{loop_site}")
        idx = SymbolicExpr.create("const", idx_z3, INT32)
        constraint = _range_to_iterator_constraint(
            idx_z3, start=iterable.start, stop=iterable.stop, step=iterable.step
        )
        ctx = LoopContext(
            loop_site,
            len(iterable),
            idx,
            idx_z3,
            constraint,
            start=iterable.start,
            stop=iterable.stop,
            step=iterable.step,
        )
        idx.loop_ctx = ctx
        self.loop_stack.append(ctx)

    def _cutile_loop_iter(self, loop_site: LoopSite, idx: Any) -> Any:
        if not self.loop_stack or self.loop_stack[-1].loop_site != loop_site:
            return None
        ctx = self.loop_stack[-1]
        ctx.current_value = int(idx)
        # The kernel sees the real iteration value; checks see the loop variable.
        tile = _tile_type()(int(idx))
        setattr(tile, _SHADOW, ctx.idx)
        return tile

    def _cutile_loop_after(self, loop_site: LoopSite) -> None:
        if self.loop_stack and self.loop_stack[-1].loop_site == loop_site:
            self.loop_stack.pop()
