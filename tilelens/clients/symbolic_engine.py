from __future__ import annotations

import math
import warnings
import weakref
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from functools import reduce
from typing import (
    Any,
    ClassVar,
    Literal,
    NoReturn,
    TypeAlias,
    cast,
)

import numpy as np
from anytree import Node, RenderTree
from torch import Tensor
from z3 import (
    Int,
    IntVal,
    If,
    Sum,
    And,
    Or,
    Solver,
    simplify,
    Int2BV,
    BV2Int,
    BitVecRef,
    LShR,
    BoolVal,
    Not,
)
from z3.z3 import BoolRef, ArithRef, IntNumRef, ExprRef, Tactic, Probe

from ..core.client import Client
from ..core.callbacks import OpCallbacks, ForLoopCallbacks
from ..core.config import config as cfg
from ..core.patch import LoopSite
from ..core.data import (
    Op,
    UnaryOp,
    BinaryOp,
    TernaryOp,
    Fma,
    ProgramId,
    Dot,
    MakeRange,
    AddPtr,
    ExpandDims,
    Broadcast,
    ReduceSum,
    ReduceXor,
    ReduceOr,
    ReduceMax,
    ReduceMin,
    Sort,
    Splat,
    Unsplat,
    Idiv,
    Rsqrt,
    CastImpl,
    Reshape,
    Trans,
    Join,
    Split,
    Fabs,
    Ashr,
    Advance,
    FpToFp,
    Umulhi,
    CumSum,
    Bitcast,
    PtrToInt,
    IntToPtr,
    AtomicCas,
    AtomicRMW,
    RawLoad,
    RawStore,
    Load,
    Store,
    MakeBlockPointer,
    TensorPointerLoad,
    TensorPointerStore,
)
from ..core.symbolic_metadata import (
    FLOAT32,
    INT1,
    INT32,
    DTYPE_BY_NAME,
    SymbolicDType,
    SymbolicPointerDType,
    SymbolicScalarDType,
    SymbolicTensorValue,
    SymbolicTypeSpec,
    TensorDescriptorAccess,
    dtype_to_numpy,
    element_bytewidth,
    is_pointer_dtype,
    normalize_symbolic_value,
    pointee_dtype,
    pointer_type,
    type_spec,
    unpack_type_spec,
)
from .utils import (
    check_storage_contiguous,
    get_physical_addr_from_tensor_slice,
    get_physical_addr_per_element,
    check_inner_stride_equal_to_one,
)


AccessMode: TypeAlias = Literal["read", "write"]


Z3Expr: TypeAlias = ExprRef | list[ExprRef] | Tactic | Probe
ConstraintExpr: TypeAlias = ExprRef | bool | int | float
ConstraintConjunction: TypeAlias = BoolRef | None
SymbolicChild: TypeAlias = "SymbolicExpr | tuple[SymbolicExpr, ...]"


def _constraint_to_bool(expr: ConstraintExpr) -> BoolRef:
    if isinstance(expr, BoolRef):
        return expr
    if isinstance(expr, bool):
        return BoolVal(expr)
    if isinstance(expr, IntNumRef):
        return BoolVal(expr.as_long() != 0)
    if isinstance(expr, (int, float)):
        return BoolVal(int(expr) != 0)
    return cast(BoolRef, expr != 0)


def _iter_constraints_to_bool(
    expr_constraints: Sequence[ConstraintExpr],
) -> Iterator[BoolRef]:
    for constraint in expr_constraints:
        yield _constraint_to_bool(constraint)


def _and_constraints(
    *constraints: ConstraintExpr | Sequence[ConstraintExpr] | None,
) -> ConstraintConjunction:
    parts: list[BoolRef] = []
    for constraint in constraints:
        if constraint is None:
            continue
        if isinstance(constraint, (list, tuple)):
            parts.extend(_iter_constraints_to_bool(constraint))
            continue
        parts.append(_constraint_to_bool(constraint))

    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return And(*parts)


def _range_to_iterator_constraint(
    var: ArithRef, *, start: int, stop: int, step: int
) -> BoolRef:
    """Return a Z3 constraint describing values produced by ``range(start, stop, step)``."""
    if step == 0:
        raise ValueError("range() step cannot be 0")

    if step > 0:
        bounds = And(var >= start, var < stop)
    else:
        # Python range with negative step iterates while value > stop.
        bounds = And(var <= start, var > stop)

    abs_step = abs(step)
    if abs_step == 1:
        return bounds

    # `i in range(start, stop, step)` iff bounds hold and (i-start) is a multiple
    # of abs(step). Use a positive modulus to avoid negative-divisor semantics.
    return And(bounds, (var - start) % abs_step == 0)


def _literal_int(value: Any) -> int | None:
    if isinstance(value, SymbolicExpr):
        if value.op != "const":
            return None
        return _literal_int(value.to_py())
    if isinstance(value, (bool, int)):
        return int(value)
    return None


def _shape_to_tuple(shape: Any) -> tuple[int, ...]:
    if hasattr(shape, "handle"):
        shape = shape.handle
    if hasattr(shape, "to_py"):
        shape = shape.to_py()
    if isinstance(shape, np.ndarray):
        shape = shape.tolist()
    if isinstance(shape, (int, np.integer)):
        return (int(shape),)
    return tuple(cast(int, _literal_int(x)) for x in shape)


@dataclass
class PendingCheck:
    symbolic_expr: SymbolicExpr
    addr_expr: Z3Expr
    constraints: ConstraintConjunction
    # Lightweight source location: (filename, lineno, func_name)
    # Captured immediately to preserve accurate line info for deferred checks
    source_location: tuple[str, int, str] | None = None


@dataclass
class LoopContext:
    loop_site: LoopSite
    length: int
    idx: Any
    idx_z3: ArithRef
    iterator_constraint: BoolRef | None = None
    start: int = 0
    stop: int = 0
    step: int = 1
    current_value: int | None = None
    signature_cache: dict[int, int] = field(default_factory=dict)
    pending_checks: list[PendingCheck] = field(default_factory=list)


class SymbolicExprDataWrapper:
    """
    This wrapper is used as a workaround for frontend tensor truthiness code.
    Some runtimes inspect:
        "data = self.handle.data
        return bool(data) if data.size == 1 else True"
    Symbolic expressions therefore expose a data-like object with a size.
    """

    def __init__(self, symbolic_expr: SymbolicExpr, value: str | None = None):
        self._value = value
        self.symbolic_expr = symbolic_expr

    def invalidate(self) -> None:
        self._value = None

    def _ensure_value(self) -> str:
        if self._value is None:
            self._value = str(self.symbolic_expr)
        return self._value

    @property
    def value(self) -> str:
        return self._ensure_value()

    @value.setter
    def value(self, v: str) -> None:
        self._value = v

    @property
    def size(self) -> int:
        shape = self.symbolic_expr.shape
        if len(shape) == 0:
            return 1
        return math.prod(shape)

    def _scalar_data(self) -> np.ndarray:
        if self.size != 1:
            raise ValueError(
                f"Expected scalar symbolic data, got shape {self.symbolic_expr.shape}"
            )
        concrete = self.symbolic_expr.concretize()
        if not isinstance(concrete, SymbolicTensorValue):
            raise TypeError(f"Expected symbolic tensor value, got {type(concrete)}")
        return concrete.data

    def __getitem__(self, item: Any) -> Any:
        return self._scalar_data()[item]

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> np.ndarray:
        return self._scalar_data().squeeze(axis)

    @staticmethod
    def coerce_int(val: Any) -> int:
        if isinstance(val, IntNumRef):
            return val.as_long()
        if isinstance(val, (int, np.integer, bool)):
            return int(val)
        if isinstance(val, float):
            return int(val)
        if isinstance(val, SymbolicTensorValue):
            if val.data.size != 1:
                raise ValueError(
                    "Expected scalar symbolic tensor value, got size "
                    f"{val.data.size}"
                )
            return int(val.data.item())
        if isinstance(val, np.ndarray):
            if val.size != 1:
                raise ValueError(f"Expected scalar ndarray, got size {val.size}")
            return int(val.item())
        raise ValueError(
            f"SymbolicExprDataWrapper cannot coerce type {type(val)} to int"
        )

    def __int__(self) -> int:
        int_val, _ = self.symbolic_expr.eval()
        if isinstance(int_val, list):
            int_val = int_val[0]
        return self.coerce_int(int_val)

    def __bool__(self) -> bool:
        return bool(self._scalar_data().item())

    def __str__(self) -> str:
        return self._ensure_value()

    def __repr__(self) -> str:
        return self._ensure_value()


@dataclass(frozen=True)
class SymbolicTensorDescriptorValue:
    base: Any
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    block_shape: tuple[int, ...]
    base_offset: int = 0


# Gluon descriptor objects are created during frontend codegen and may be
# short-lived. Keep their symbolic metadata in a weak map so this module-level
# registry does not extend descriptor lifetime after the launch/codegen state is
# released.
_SYMBOLIC_TENSOR_DESCRIPTORS: weakref.WeakKeyDictionary[
    Any, SymbolicTensorDescriptorValue
] = weakref.WeakKeyDictionary()


class SymbolicExpr:
    BASIC_OPS: ClassVar[tuple[str, ...]] = ("const", "pid", "arange")
    INDIRECT_OPS: ClassVar[tuple[str, ...]] = (
        "load",
        "store",
        "tensor_pointer_load",
        "tensor_pointer_store",
    )
    UNARY_OPS: ClassVar[tuple[str, ...]] = (
        "cos",
        "exp",
        "exp2",
        "abs",
        "fabs",
        "floor",
        "ceil",
        "log",
        "log2",
        "sqrt",
        "sin",
        "rsqrt",
        "negative",
    )
    UNARY_OP_SET: ClassVar[frozenset[str]] = frozenset(UNARY_OPS)
    BINARY_OP_SYMBOL_TABLE: ClassVar[dict[str, str]] = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "div": "/",
        "idiv": "//",
        "mod": "%",
        "less": "<",
        "less_equal": "<=",
        "greater": ">",
        "greater_equal": ">=",
        "not_equal": "!=",
        "equal": "==",
        "maximum": "max",
        "minimum": "min",
        "bitwise_and": "&",
        "bitwise_or": "|",
        "bitwise_xor": "^",
        "right_shift": ">>",
        "left_shift": "<<",
        "ashr": ">>>",
        "umulhi": "umulhi",
    }
    BINARY_OPS: ClassVar[tuple[str, ...]] = tuple(BINARY_OP_SYMBOL_TABLE.keys())
    BINARY_OP_SET: ClassVar[frozenset[str]] = frozenset(BINARY_OPS)
    TERNARY_OPS: ClassVar[tuple[str, ...]] = ("where", "fma")
    REDUCE_OPS: ClassVar[tuple[str, ...]] = (
        "sum",
        "max",
        "min",
        "argmax",
        "argmin",
        "xor_sum",
        "reduce_or",
        "dot",
    )
    SCAN_OPS: ClassVar[tuple[str, ...]] = ("cumsum",)
    SORT_OPS: ClassVar[tuple[str, ...]] = ("sort",)
    POINTER_OPS: ClassVar[tuple[str, ...]] = (
        "make_block_ptr",
        "addptr",
        "advance",
        "descriptor_access",
    )
    RESHAPE_OPS: ClassVar[tuple[str, ...]] = (
        "splat",
        "unsplat",
        "expand_dims",
        "broadcast",
        "reshape",
        "join",
        "split",
        "trans",
    )
    CAST_OPS: ClassVar[tuple[str, ...]] = ("cast_impl", "bitcast", "fp_to_fp")
    ATOMIC_OPS: ClassVar[tuple[str, ...]] = ("atomic_cas", "atomic_rmw")
    SUPPORTED_OPS: ClassVar[tuple[str, ...]] = (
        BASIC_OPS
        + INDIRECT_OPS
        + UNARY_OPS
        + BINARY_OPS
        + TERNARY_OPS
        + REDUCE_OPS
        + POINTER_OPS
        + RESHAPE_OPS
        + CAST_OPS
        + SCAN_OPS
        + SORT_OPS
        + ATOMIC_OPS
    )
    SUPPORTED_OP_SET: ClassVar[frozenset[str]] = frozenset(SUPPORTED_OPS)

    PID0: ClassVar[ArithRef] = Int("pid_0")
    PID1: ClassVar[ArithRef] = Int("pid_1")
    PID2: ClassVar[ArithRef] = Int("pid_2")

    ARANGE_DICT: ClassVar[
        dict[tuple[int, int], tuple[ArithRef, ConstraintConjunction]]
    ] = {}
    _OP_CLASS_MAP: ClassVar[dict[str, type[SymbolicExpr]]] = {}
    _CONCRETE_FNS: ClassVar[dict[str, Callable[..., Any]]] = {}

    @classmethod
    def register_op_class(
        cls, op_cls: type[SymbolicExpr], op_types: tuple[str, ...]
    ) -> None:
        for op_type in op_types:
            cls._OP_CLASS_MAP[op_type] = op_cls

    @classmethod
    def set_concrete_fn(cls, op: str, concrete_fn: Callable[..., Any]) -> None:
        cls._CONCRETE_FNS[op] = concrete_fn

    @classmethod
    def create(cls, op: str, *args: Any) -> SymbolicExpr:
        op_cls = cls._OP_CLASS_MAP.get(op)
        if op_cls is None:
            raise NotImplementedError(f"No operator class registered for {op}")
        return op_cls(op, *args)

    def __init__(self, op: str):
        """
        :param op: Operation type, e.g. "const", "add", "sub", "mul", "div", "pid", "arange"
        """
        assert op in self.SUPPORTED_OP_SET, f"Unsupported op: {op}"
        self.op = op
        # Tensor-like attributes used by frontend runtimes.
        self.attr: dict[str, Any] = {}
        self.dtype: SymbolicDType | None = None
        self.shape: tuple[int, ...] = ()

        # Functions and arguments for concretization
        self.concrete_fn: Callable[..., Any] | None = self._CONCRETE_FNS.get(op)

        # deal with args
        self.children: dict[str, SymbolicChild | None] = {}

        # for-loop iterator association
        self.loop_ctx: LoopContext | None = None

        # z3
        self.z3: Z3Expr | None = None

        self.constraints: ConstraintConjunction = None
        self._simplified_z3: Z3Expr | None = None
        self._simplified_constraints: ConstraintConjunction | None = None
        self._has_op_cache: dict[str, bool] = {}
        self.expr_signature: int | None = None
        self._data_wrapper: SymbolicExprDataWrapper | None = None

    @staticmethod
    def _unpack_dtype(
        dtype: SymbolicDType | SymbolicTypeSpec,
        fallback_shape: Sequence[int] = (),
    ) -> tuple[SymbolicDType, tuple[int, ...]]:
        return unpack_type_spec(dtype, fallback_shape)

    def add_child(self, name: str, value: Any) -> None:
        child = SymbolicExpr.from_value(value) if value is not None else None
        self.children[name] = child
        setattr(self, name, child)
        self._has_op_cache.clear()
        self.expr_signature = None
        self._simplified_z3 = None
        self._simplified_constraints = None
        if self._data_wrapper is not None:
            self._data_wrapper.invalidate()

    def signature(self) -> int:
        """Return a structural hash for deduplicating equivalent expr trees."""
        if self.expr_signature is not None:
            return self.expr_signature

        child_parts: list[tuple[str, Any]] = []
        for name, child in self.children.items():
            if child is None:
                child_parts.append((name, None))
            elif isinstance(child, tuple):
                child_parts.append((name, tuple(item.signature() for item in child)))
            else:
                child_parts.append((name, child.signature()))

        value_part: Any = None
        if self.op == "const":
            value_part = self._value_signature(getattr(self, "value", None))

        self.expr_signature = hash(
            (
                self.op,
                self.dtype,
                self.shape,
                value_part,
                tuple(child_parts),
            )
        )
        return self.expr_signature

    @classmethod
    def _value_signature(cls, value: Any) -> Any:
        if isinstance(value, SymbolicTensorValue):
            return (
                "tensor",
                value.dtype,
                value.data.shape,
                value.data.dtype.str,
                hash(value.data.tobytes()),
            )
        if isinstance(value, np.ndarray):
            return (
                "ndarray",
                value.shape,
                value.dtype.str,
                hash(value.tobytes()),
            )
        if isinstance(value, cls.tuple_types):
            seq = cast(Sequence[Any], value)
            return tuple(cls._value_signature(item) for item in seq)
        if isinstance(
            value, (int, np.integer, bool, float, np.floating, str, type(None))
        ):
            return value
        return (type(value).__name__, repr(value))

    def __add__(self, other: SymbolicExpr) -> SymbolicExpr:
        return SymbolicExpr.create("add", self, other)

    def __sub__(self, other: SymbolicExpr) -> SymbolicExpr:
        return SymbolicExpr.create("sub", self, other)

    def __mul__(self, other: SymbolicExpr) -> SymbolicExpr:
        return SymbolicExpr.create("mul", self, other)

    def __truediv__(self, other: SymbolicExpr) -> SymbolicExpr:
        return SymbolicExpr.create("div", self, other)

    def __floordiv__(self, other: SymbolicExpr) -> SymbolicExpr:
        return SymbolicExpr.create("idiv", self, other)

    def __mod__(self, other: SymbolicExpr) -> SymbolicExpr:
        return SymbolicExpr.create("mod", self, other)

    def __lt__(self, other: object) -> Any:
        return SymbolicExpr.create("less", self, other)

    def __le__(self, other: object) -> Any:
        return SymbolicExpr.create("less_equal", self, other)

    def __ne__(self, other: object) -> Any:
        return SymbolicExpr.create("not_equal", self, other)

    def __eq__(self, other: object) -> Any:
        return SymbolicExpr.create("equal", self, other)

    def _to_anytree(self) -> Node:
        """Convert this SymbolicExpr into an anytree Node."""
        # Generate the node label
        label = self._node_label()
        root = Node(label)
        # Recursively add child nodes
        for child_key, child_symbolic_expr in self.children.items():
            if child_symbolic_expr is None:
                Node(f"{child_key}: None", parent=root)
                continue
            if isinstance(child_symbolic_expr, tuple):
                tuple_node = Node(f"{child_key}: tuple", parent=root)
                for idx, item in enumerate(child_symbolic_expr):
                    if isinstance(item, SymbolicExpr):
                        child_node = item._to_anytree()
                        child_node.name = f"{idx}: {child_node.name}"
                        child_node.parent = tuple_node
                    else:
                        Node(f"{idx}: {item!r}", parent=tuple_node)
                continue
            child_node = child_symbolic_expr._to_anytree()
            child_node.name = f"{child_key}: {child_node.name}"
            child_node.parent = root

        return root

    def _node_label(self) -> str:
        """Generate a short label for this node."""
        label = self._node_label_core()
        if self.shape:
            label = f"{label} [dtype={self.dtype}, shape={self.shape}]"
        else:
            label = f"{label} [dtype={self.dtype}]"

        return label

    def _node_label_core(self) -> str:
        return self.op

    tuple_types: ClassVar[tuple[type, ...]] = (tuple, list)
    _NORMALIZE_VALUE: ClassVar[Callable[[Any], Any]] = staticmethod(
        normalize_symbolic_value
    )
    _WRAP_LOOP_INDEX: ClassVar[Callable[[Any, SymbolicDType], Any]] = staticmethod(
        lambda expr, _dtype: expr
    )

    @staticmethod
    def _infer_literal_dtype(var: Any) -> SymbolicDType | SymbolicTypeSpec:
        if isinstance(var, SymbolicTensorValue):
            if var.data.size != 1:
                raise ValueError(
                    f"Unsupported var.data: {var.data} with length more than one!"
                )
            return var.dtype
        if isinstance(var, SymbolicTypeSpec):
            return var
        if isinstance(var, (SymbolicScalarDType, SymbolicPointerDType)):
            return var
        if isinstance(var, SymbolicExpr.tuple_types):
            seq = cast(Sequence[Any], var)
            if len(seq) == 0:
                raise ValueError("Cannot infer dtype from an empty tuple/list.")
            first_dtype = SymbolicExpr._infer_literal_dtype(seq[0])
            for v in seq[1:]:  # assume only one consistent dtype in the tuple
                dtype = SymbolicExpr._infer_literal_dtype(v)
                if dtype != first_dtype:
                    raise ValueError(
                        f"All elements in the tuple must have the same dtype, but found {first_dtype} and {dtype}"
                    )
            return first_dtype
        if isinstance(var, (int, np.integer, bool)):
            return INT32
        if isinstance(var, (float, np.floating)):
            return FLOAT32
        raise ValueError(f"Unsupported type: {type(var)}")

    @classmethod
    def set_frontend_hooks(
        cls,
        normalize_value: Callable[[Any], Any],
        wrap_loop_index: Callable[[Any, SymbolicDType], Any],
    ) -> None:
        cls._NORMALIZE_VALUE = staticmethod(normalize_value)
        cls._WRAP_LOOP_INDEX = staticmethod(wrap_loop_index)

    @classmethod
    def wrap_loop_index(cls, expr: Any, dtype: SymbolicDType) -> Any:
        return cls._WRAP_LOOP_INDEX(expr, dtype)

    # Stored on the class and may be accessed through either the class or an instance;
    # therefore the callable must tolerate an extra bound argument (self/cls).
    loop_ctx_provider: ClassVar[Callable[..., LoopContext | None] | None] = None

    @classmethod
    def set_loop_ctx_provider(cls, fn: Callable[..., LoopContext | None]) -> None:
        """Register a function that, on each call,
        returns the current active `LoopContext` (or `None` if none exists)."""
        cls.loop_ctx_provider = fn

    @classmethod
    def from_value(cls, var: Any) -> SymbolicExpr | tuple[SymbolicExpr, ...]:
        """Create a SymbolicExpr from a Python value."""
        var = cls._NORMALIZE_VALUE(var)

        if isinstance(var, cls):  # if already SymbolicExpr
            return var

        if isinstance(var, SymbolicExpr.tuple_types):  # if a tuple
            seq = cast(Sequence[Any], var)
            children: list[SymbolicExpr] = []
            for item in seq:
                child = cls.from_value(item)
                if isinstance(child, tuple):
                    raise ValueError("Nested symbolic tuples are not supported.")
                children.append(child)
            return tuple(children)

        dtype = SymbolicExpr._infer_literal_dtype(var)

        if isinstance(var, SymbolicTensorValue):
            if var.data.size != 1:
                raise ValueError(
                    "SymbolicExpr.from_value only supports scalar tensor values!"
                )
            return cls.create("const", var.data.item(), dtype)
        if isinstance(var, (int, np.integer, bool, float, np.floating)):
            return cls.create("const", var, dtype)
        if isinstance(
            var, (SymbolicScalarDType, SymbolicPointerDType, SymbolicTypeSpec)
        ):
            return cls.create("const", var, dtype)

        raise ValueError("Unknown type:", type(var))

    def eval(
        self, simplify_constraints: bool = True
    ) -> tuple[Z3Expr, ConstraintConjunction]:
        """
        Returns a tuple (expr, constraints):
        - expr: Z3 expression (or list of expressions) corresponding to the root node
        - constraints: conjunction of constraint expressions (Z3 Bool or scalars),
          or None when there are no constraints,
          recording all range constraints created by program_id and arange
        """
        expr, constraints = self._to_z3()

        if self._simplified_z3 is None:
            if isinstance(expr, list):
                self._simplified_z3 = [simplify(e) for e in expr]
            else:
                self._simplified_z3 = simplify(expr)

        if constraints is not None and simplify_constraints:
            if self._simplified_constraints is None:
                # Z3 stubs type simplify(...) as ExprRef even when input is BoolRef.
                self._simplified_constraints = cast(BoolRef, simplify(constraints))
            return self._simplified_z3, self._simplified_constraints

        return self._simplified_z3, constraints

    def to_py(self) -> Any:
        """Return a Python value for this expression."""
        raise NotImplementedError("to_py must be implemented by subclasses")

    def concretize(self) -> Any:
        """Return a concrete tensor value for this expression."""
        raise NotImplementedError(f"Concretize for op {self.op} is not implemented")

    def replace_subtree(self, anchor_op: str | None = None) -> SymbolicExpr:
        """
        Post-order traversal that replaces *all* sub-trees with constant nodes
        produced by `concretize`.
        """
        for name, child in list(self.children.items()):
            if child is None:
                continue
            if isinstance(child, tuple):
                new_child_tuple = tuple(
                    item.replace_subtree(anchor_op) for item in child
                )
                self.add_child(name, new_child_tuple)
                continue
            new_child_expr = child.replace_subtree(anchor_op)
            self.add_child(name, new_child_expr)

        # inplace replace to "const" node
        if anchor_op is None or (
            self.op == anchor_op
            and all(
                (child is None)
                or (
                    all(not item.has_op(anchor_op) for item in child)
                    if isinstance(child, tuple)
                    else not child.has_op(anchor_op)
                )
                for child in self.children.values()
            )
        ):
            if self.op == "const" and isinstance(
                getattr(self, "value", None),
                (SymbolicScalarDType, SymbolicPointerDType, SymbolicTypeSpec),
            ):
                return self
            concrete = self.concretize()
            if not isinstance(concrete, SymbolicTensorValue):
                raise TypeError(f"Unexpected dtype: {type(concrete)}!")

            self = SymbolicExpr.create(
                "const",
                concrete,
                type_spec(concrete.dtype, concrete.shape),
            )

        return self

    def has_op(self, op_name: str) -> bool:
        """Return True when the subtree contains an op with the given name."""
        cached = self._has_op_cache.get(op_name)
        if cached is not None:
            return cached
        if self.op == op_name:
            self._has_op_cache[op_name] = True
            return True
        for _, child_symbolic_expr in self.children.items():
            if child_symbolic_expr is None:
                continue
            if isinstance(child_symbolic_expr, tuple):
                if any(child.has_op(op_name) for child in child_symbolic_expr):
                    self._has_op_cache[op_name] = True
                    return True
                continue
            if child_symbolic_expr.has_op(op_name):
                self._has_op_cache[op_name] = True
                return True
        self._has_op_cache[op_name] = False
        return False

    def has_vector_const(self) -> bool:
        value = getattr(self, "value", None)
        if (
            self.op == "const"
            and isinstance(value, SymbolicTensorValue)
            and value.data.size != 1
        ):
            return True
        for child_symbolic_expr in self.children.values():
            if child_symbolic_expr is None:
                continue
            if isinstance(child_symbolic_expr, tuple):
                if any(child.has_vector_const() for child in child_symbolic_expr):
                    return True
                continue
            if child_symbolic_expr.has_vector_const():
                return True
        return False

    def to_tree_str(self) -> str:
        """
        Render the AST as an ASCII tree using anytree.RenderTree.
        """
        root = self._to_anytree()
        lines = []
        for prefix, _, node in RenderTree(root):
            lines.append(f"{prefix}{node.name}")
        return "\n" + "\n".join(lines)

    # Tensor-like methods, not used in sanitizer.
    def set_attr(self, key, value):
        self.attr[key] = value

    def __str__(self) -> str:
        return self.to_tree_str()

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def data(self) -> SymbolicExprDataWrapper:
        """Return a wrapper suitable for external consumers."""
        if self._data_wrapper is None:
            self._data_wrapper = SymbolicExprDataWrapper(self)
        return self._data_wrapper

    def _to_z3(self) -> tuple[Z3Expr, ConstraintConjunction]:
        if self.z3 is not None:
            return self.z3, self.constraints

        self.z3, self.constraints = self._to_z3_impl()
        if self.z3 is None:
            raise NotImplementedError(f"Eval for op {self.op} is not implemented")
        return self.z3, self.constraints

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        raise NotImplementedError(f"Eval for op {self.op} is not implemented")


class ConstSymbolicExpr(SymbolicExpr):
    value: Any

    def __init__(
        self,
        op: str,
        value: Any,
        dtype: SymbolicDType | SymbolicTypeSpec,
    ):
        super().__init__(op)
        value = SymbolicExpr._NORMALIZE_VALUE(value)
        dtype = SymbolicExpr._NORMALIZE_VALUE(dtype)
        self.value = value
        fallback_shape = value.shape if isinstance(value, SymbolicTensorValue) else ()
        self.dtype, self.shape = self._unpack_dtype(dtype, fallback_shape)

    def _node_label_core(self) -> str:
        return f"const={self.value}"

    def to_py(self) -> Any:
        """
        Valid only for nodes with op == 'const':
        - If `value` is a tensor value:
            • Scalar  -> return int/float
            • Multi-element -> return a Python list
        - Otherwise, return the original Python object
          (e.g., int, float, tuple, list, etc.).
        """
        v = self.value
        if isinstance(v, SymbolicTensorValue):
            if len(v.data) == 1:
                return v.data.item()  # scalar case
            else:
                return v.data.tolist()  # multi-element case
        return v

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        value = self.value
        if isinstance(value, SymbolicTensorValue):
            value = self.value.data

        if self.loop_ctx:  # if the self is a loop iterator
            z3_expr: Z3Expr = self.loop_ctx.idx_z3
        elif isinstance(
            value, np.ndarray
        ):  # only const nodes can be created with ndarray
            z3_expr = [IntVal(int(v)) for v in value.flat]
        elif isinstance(value, tuple):
            z3_expr = [IntVal(int(v)) for v in value]
        elif isinstance(value, (int, np.integer, bool, float, np.floating)):
            # Convert to int for Z3 - Z3's IntVal cannot parse float strings
            z3_expr = IntVal(int(value))
        else:
            # For other values, use 0 as a placeholder (e.g., for optional mask/other)
            z3_expr = IntVal(0)
        return z3_expr, None

    def concretize(self) -> Any:
        dtype = self.dtype
        if dtype is None:
            raise RuntimeError("const node is missing dtype information")

        if self.loop_ctx is not None and self.loop_ctx.current_value is not None:
            return SymbolicTensorValue(
                np.array([self.loop_ctx.current_value], dtype=dtype_to_numpy(dtype)),
                dtype,
            )
        if isinstance(self.value, (int, np.integer, bool, float, np.floating)):
            return SymbolicTensorValue(
                np.array([self.value], dtype=dtype_to_numpy(dtype)),
                dtype,
            )
        elif isinstance(self.value, SymbolicExpr.tuple_types):
            seq = cast(Sequence[Any], self.value)
            np_array = np.array(seq, dtype=dtype_to_numpy(dtype))
            return SymbolicTensorValue(np_array, dtype)
        elif isinstance(self.value, np.ndarray):
            return SymbolicTensorValue(
                np.asarray(self.value, dtype=dtype_to_numpy(dtype)),
                dtype,
            )
        elif isinstance(self.value, SymbolicTensorValue):
            return self.value

        raise RuntimeError(f"Unsupported const value type: {type(self.value)}")


class PidSymbolicExpr(SymbolicExpr):
    axis: SymbolicExpr

    def __init__(self, op: str, axis: Any):
        super().__init__(op)
        self.add_child("axis", axis)
        self.dtype = INT32

    def _node_label_core(self) -> str:
        axis_val = self.axis.to_py()
        return f"pid_{axis_val}"

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        axis_val = self.axis.to_py()
        if axis_val == 0:
            return SymbolicExpr.PID0, None
        if axis_val == 1:
            return SymbolicExpr.PID1, None
        return SymbolicExpr.PID2, None

    def concretize(self) -> Any:
        return self.concrete_fn(self.axis.to_py())  # type: ignore


class ArangeSymbolicExpr(SymbolicExpr):
    ret_ty: ConstSymbolicExpr
    start: ConstSymbolicExpr
    end: ConstSymbolicExpr

    def __init__(self, op: str, ret_ty: Any, start: Any, end: Any):
        super().__init__(op)
        self.add_child("ret_ty", ret_ty)
        self.add_child("start", start)
        self.add_child("end", end)
        # Program ID / arange are always int32
        start_const = cast(ConstSymbolicExpr, self.start)
        end_const = cast(ConstSymbolicExpr, self.end)
        self.dtype = INT32
        self.shape = (end_const.value - start_const.value,)

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        start = self.start.to_py()
        end = self.end.to_py()
        key = (start, end)
        if key in SymbolicExpr.ARANGE_DICT:
            return SymbolicExpr.ARANGE_DICT[key]
        name = f"arange_{start}_{end}"
        v = Int(name)
        constraints = _and_constraints(v >= start, v < end)
        SymbolicExpr.ARANGE_DICT[key] = (v, constraints)
        return v, constraints

    def concretize(self) -> Any:
        return self.concrete_fn(
            self.ret_ty.to_py(), self.start.to_py(), self.end.to_py()
        )  # type: ignore


class IndirectSymbolicExprBase(SymbolicExpr):
    ptr: SymbolicExpr
    mask: SymbolicExpr | None
    other: SymbolicExpr | None

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        ptr_expr = self.ptr
        ptr, constraints_ptr = ptr_expr._to_z3()
        mask_expr = self.mask
        mask_constraint: ConstraintExpr | Sequence[ConstraintExpr] | None = None
        if mask_expr is not None:
            mask, _ = mask_expr._to_z3()
            mask_constraint = mask
        return ptr, _and_constraints(constraints_ptr, mask_constraint)

    def concretize(self) -> Any:
        ptr_concrete = self.ptr.concretize()
        if self.mask is None:
            mask_concrete = SymbolicTensorValue(
                np.ones_like(ptr_concrete.data, dtype=bool), INT1
            )
        else:
            mask_concrete = self.mask.concretize()

        if self.other is None:
            other_concrete = None
        else:
            other_concrete = self.other.concretize()

        return self.concrete_fn(
            ptr_concrete,
            mask_concrete,
            other_concrete,
            None,  # cache_modifier
            None,  # eviction_policy
            None,  # is_volatile
        )  # type: ignore


class LoadSymbolicExpr(IndirectSymbolicExprBase):
    def __init__(self, op: str, ptr: Any, mask: Any = None, other: Any = None):
        super().__init__(op)
        self.add_child("ptr", ptr)
        self.add_child("mask", mask)
        self.add_child("other", other)
        ptr_dtype = self.ptr.dtype
        self.dtype = pointee_dtype(ptr_dtype)
        self.shape = self.ptr.shape


class StoreSymbolicExpr(IndirectSymbolicExprBase):
    value: SymbolicExpr

    def __init__(
        self,
        op: str,
        ptr: Any,
        value: Any,
        mask: Any = None,
        other: Any = None,
    ):
        super().__init__(op)
        self.add_child("ptr", ptr)
        self.add_child("value", value)
        self.add_child("mask", mask)
        self.add_child("other", other)


class UnarySymbolicExpr(SymbolicExpr):
    arg: SymbolicExpr

    def __init__(self, op: str, arg: Any):
        if op not in SymbolicExpr.UNARY_OP_SET:
            raise NotImplementedError(f"Unsupported unary op: {op}")
        super().__init__(op)
        self.add_child("arg", arg)
        self.dtype = self.arg.dtype
        self.shape = self.arg.shape

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        val, constraints = self.arg._to_z3()
        handler = self._Z3_BUILDERS.get(self.op)
        if handler is None:
            raise NotImplementedError(f"Unary op {self.op} is not implemented")
        return handler(val), constraints

    def concretize(self) -> Any:
        concrete = self.arg.concretize()
        handler = self._NUMPY_OPS.get(self.op)
        if handler is None:
            raise NotImplementedError(
                f"Concretize for unary op '{self.op}' is not implemented"
            )
        dtype = self.dtype
        if dtype is None:
            raise RuntimeError(f"{self.op} node is missing dtype information")
        return SymbolicTensorValue(
            np.asarray(handler(concrete.data), dtype=dtype_to_numpy(dtype)),
            dtype,
        )

    @staticmethod
    def _abs(val) -> Z3Expr:
        return If(val >= 0, val, -val)

    @staticmethod
    def _negative(val) -> Z3Expr:
        if isinstance(val, list):
            return [-cast(Any, item) for item in val]
        return cast(Z3Expr, -cast(Any, val))

    _Z3_BUILDERS: ClassVar[dict[str, Callable[[Z3Expr], Z3Expr]]] = {
        "abs": _abs,
        "fabs": _abs,
        "negative": _negative,
    }

    _NUMPY_OPS: ClassVar[dict[str, Callable[[Any], Any]]] = {
        "cos": np.cos,
        "exp": np.exp,
        "exp2": np.exp2,
        "abs": np.abs,
        "fabs": np.abs,
        "floor": np.floor,
        "ceil": np.ceil,
        "log": np.log,
        "log2": np.log2,
        "sqrt": np.sqrt,
        "sin": np.sin,
        "rsqrt": lambda x: 1 / np.sqrt(x),
        "negative": np.negative,
    }


class BinarySymbolicExpr(SymbolicExpr):
    lhs: SymbolicExpr
    rhs: SymbolicExpr

    def __init__(self, op: str, lhs: Any, rhs: Any):
        if op not in SymbolicExpr.BINARY_OP_SET:
            raise NotImplementedError(f"Unsupported binary op: {op}")
        super().__init__(op)
        self.add_child("lhs", lhs)
        self.add_child("rhs", rhs)
        self.dtype = self.lhs.dtype
        self.shape = self.lhs.shape

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        lhs, constraints_lhs = self.lhs._to_z3()
        rhs, constraints_rhs = self.rhs._to_z3()
        constraints = _and_constraints(constraints_lhs, constraints_rhs)

        handler = self._Z3_BUILDERS.get(self.op)
        if handler is None:
            raise NotImplementedError(f"Eval for op {self.op} is not implemented")
        return handler(self, lhs, rhs), constraints

    def _to_tensor_value(self, value: Any) -> SymbolicTensorValue:
        if isinstance(value, SymbolicTensorValue):
            return value
        dtype = self.dtype
        if dtype is None:
            raise RuntimeError(f"{self.op} node is missing dtype information")
        return SymbolicTensorValue(
            np.asarray(value, dtype=dtype_to_numpy(dtype)), dtype
        )

    def concretize(self) -> Any:
        lhs_concrete = self.lhs.concretize()
        rhs_concrete = self.rhs.concretize()
        np_op = self._NUMPY_OPS.get(self.op, None)
        # Most ops (add, sub, mul, ...) have a NumPy mapping and are called
        # with concrete_fn(lhs, rhs, np_op). Some ops like "idiv" and Triton's
        # signed right shift have concrete functions that handle the computation
        # internally and only take (lhs, rhs). Fall through to the 2-arg call
        # for those.
        if np_op is not None:
            if self.concrete_fn is not None:
                return self._to_tensor_value(
                    self.concrete_fn(lhs_concrete, rhs_concrete, np_op)  # type: ignore
                )
            return self._to_tensor_value(np_op(lhs_concrete.data, rhs_concrete.data))
        if self.concrete_fn is None:
            if self.op == "idiv":
                return self._to_tensor_value(
                    np.floor_divide(lhs_concrete.data, rhs_concrete.data)
                )
            if self.op == "ashr":
                return self._to_tensor_value(
                    np.right_shift(
                        lhs_concrete.data.astype(np.int64),
                        rhs_concrete.data,
                    )
                )
            raise NotImplementedError(
                f"Concretize for binary op '{self.op}' is not implemented"
            )
        return self._to_tensor_value(self.concrete_fn(lhs_concrete, rhs_concrete))  # type: ignore

    @staticmethod
    def _apply_binop(op_func, left, right):
        lhs_is_list = isinstance(left, list)
        rhs_is_list = isinstance(right, list)
        if lhs_is_list and rhs_is_list:
            if len(left) != len(right):
                raise ValueError(
                    f"List operands must have same length: {len(left)} vs {len(right)}"
                )
            return [op_func(li, ri) for li, ri in zip(left, right)]
        if lhs_is_list:
            return [op_func(li, right) for li in left]
        if rhs_is_list:
            return [op_func(left, ri) for ri in right]
        return op_func(left, right)

    @staticmethod
    def _infer_bitwidth(expr):
        dtype = getattr(expr, "dtype", None)
        if dtype is None:
            return None
        if hasattr(dtype, "primitive_bitwidth"):
            return dtype.primitive_bitwidth
        if hasattr(dtype, "element_ty") and hasattr(
            dtype.element_ty, "primitive_bitwidth"
        ):
            return dtype.element_ty.primitive_bitwidth
        return None

    @classmethod
    def _to_bv(cls, v, bitwidth):
        if isinstance(v, list):
            return [cls._to_bv(x, bitwidth) for x in v]
        if isinstance(v, BoolRef):
            v = If(v, IntVal(1), IntVal(0))
        if isinstance(v, BitVecRef):
            return v
        return Int2BV(v, bitwidth)

    @classmethod
    def _from_bv(cls, v):
        if isinstance(v, list):
            return [cls._from_bv(x) for x in v]
        return BV2Int(v, is_signed=False)

    def _op_add(self, lhs, rhs):
        return self._apply_binop(lambda a, b: a + b, lhs, rhs)

    def _op_sub(self, lhs, rhs):
        return self._apply_binop(lambda a, b: a - b, lhs, rhs)

    def _op_mul(self, lhs, rhs):
        return self._apply_binop(lambda a, b: a * b, lhs, rhs)

    def _op_idiv(self, lhs, rhs):
        return self._apply_binop(lambda a, b: a / b, lhs, rhs)

    def _op_mod(self, lhs, rhs):
        return self._apply_binop(lambda a, b: a % b, lhs, rhs)

    def _op_less(self, lhs, rhs):
        return self._apply_binop(lambda a, b: a < b, lhs, rhs)

    def _op_less_equal(self, lhs, rhs):
        return self._apply_binop(lambda a, b: a <= b, lhs, rhs)

    def _op_greater(self, lhs, rhs):
        return self._apply_binop(lambda a, b: a > b, lhs, rhs)

    def _op_greater_equal(self, lhs, rhs):
        return self._apply_binop(lambda a, b: a >= b, lhs, rhs)

    def _op_equal(self, lhs, rhs):
        return self._apply_binop(lambda a, b: a == b, lhs, rhs)

    def _op_not_equal(self, lhs, rhs):
        return self._apply_binop(lambda a, b: a != b, lhs, rhs)

    def _op_maximum(self, lhs, rhs):
        return self._apply_binop(lambda a, b: If(a >= b, a, b), lhs, rhs)

    def _op_minimum(self, lhs, rhs):
        return self._apply_binop(lambda a, b: If(a <= b, a, b), lhs, rhs)

    def _op_bitwise_and(self, lhs, rhs):
        lhs_expr = self.lhs
        rhs_expr = self.rhs
        bitwidth = (
            self._infer_bitwidth(self)
            or self._infer_bitwidth(lhs_expr)
            or self._infer_bitwidth(rhs_expr)
            or 64
        )

        def _bit_and(a, b):
            if isinstance(a, BoolRef) and isinstance(b, BoolRef):
                return And(a, b)
            return self._from_bv(self._to_bv(a, bitwidth) & self._to_bv(b, bitwidth))

        return self._apply_binop(_bit_and, lhs, rhs)

    def _op_bitwise_or(self, lhs, rhs):
        lhs_expr = self.lhs
        rhs_expr = self.rhs
        bitwidth = (
            self._infer_bitwidth(self)
            or self._infer_bitwidth(lhs_expr)
            or self._infer_bitwidth(rhs_expr)
            or 64
        )

        def _bit_or(a, b):
            if isinstance(a, BoolRef) and isinstance(b, BoolRef):
                return Or(a, b)
            return self._from_bv(self._to_bv(a, bitwidth) | self._to_bv(b, bitwidth))

        return self._apply_binop(_bit_or, lhs, rhs)

    def _op_bitwise_xor(self, lhs, rhs):
        lhs_expr = self.lhs
        rhs_expr = self.rhs
        bitwidth = (
            self._infer_bitwidth(self)
            or self._infer_bitwidth(lhs_expr)
            or self._infer_bitwidth(rhs_expr)
            or 64
        )

        def _bit_xor(a, b):
            return self._from_bv(self._to_bv(a, bitwidth) ^ self._to_bv(b, bitwidth))

        return self._apply_binop(_bit_xor, lhs, rhs)

    def _op_right_shift(self, lhs, rhs):
        lhs_expr = self.lhs
        rhs_expr = self.rhs
        bitwidth = (
            self._infer_bitwidth(self)
            or self._infer_bitwidth(lhs_expr)
            or self._infer_bitwidth(rhs_expr)
            or 64
        )

        def _right_shift(a, b):
            shifted = LShR(self._to_bv(a, bitwidth), self._to_bv(b, bitwidth))
            return BV2Int(shifted, is_signed=False)

        return self._apply_binop(_right_shift, lhs, rhs)

    def _op_left_shift(self, lhs, rhs):
        lhs_expr = self.lhs
        rhs_expr = self.rhs
        bitwidth = (
            self._infer_bitwidth(self)
            or self._infer_bitwidth(lhs_expr)
            or self._infer_bitwidth(rhs_expr)
            or 64
        )

        def _left_shift(a, b):
            shifted = self._to_bv(a, bitwidth) << self._to_bv(b, bitwidth)
            return BV2Int(shifted, is_signed=False)

        return self._apply_binop(_left_shift, lhs, rhs)

    def _op_ashr(self, lhs, rhs):
        bitwidth = (
            self._infer_bitwidth(self)
            or self._infer_bitwidth(self.lhs)
            or self._infer_bitwidth(self.rhs)
            or 64
        )

        def _ashr(a, b):
            shifted = self._to_bv(a, bitwidth) >> self._to_bv(b, bitwidth)
            return BV2Int(shifted, is_signed=True)

        return self._apply_binop(_ashr, lhs, rhs)

    _NUMPY_OPS: ClassVar[dict[str, Callable[[Any, Any], Any]]] = {
        "add": np.add,
        "sub": np.subtract,
        "mul": np.multiply,
        "div": np.divide,
        "mod": np.fmod,
        "less": np.less,
        "less_equal": np.less_equal,
        "greater": np.greater,
        "greater_equal": np.greater_equal,
        "equal": np.equal,
        "not_equal": np.not_equal,
        "maximum": np.maximum,
        "minimum": np.minimum,
        "bitwise_and": np.bitwise_and,
        "bitwise_or": np.bitwise_or,
        "bitwise_xor": np.bitwise_xor,
        "right_shift": np.right_shift,
        "left_shift": np.left_shift,
    }

    _Z3_BUILDERS: ClassVar[
        dict[str, Callable[[BinarySymbolicExpr, Z3Expr, Z3Expr], Z3Expr]]
    ] = {
        "add": _op_add,
        "sub": _op_sub,
        "mul": _op_mul,
        "idiv": _op_idiv,
        "mod": _op_mod,
        "less": _op_less,
        "less_equal": _op_less_equal,
        "greater": _op_greater,
        "greater_equal": _op_greater_equal,
        "equal": _op_equal,
        "not_equal": _op_not_equal,
        "maximum": _op_maximum,
        "minimum": _op_minimum,
        "bitwise_and": _op_bitwise_and,
        "bitwise_or": _op_bitwise_or,
        "bitwise_xor": _op_bitwise_xor,
        "right_shift": _op_right_shift,
        "left_shift": _op_left_shift,
        "ashr": _op_ashr,
    }


class WhereSymbolicExpr(SymbolicExpr):
    cond: SymbolicExpr
    lhs: SymbolicExpr
    rhs: SymbolicExpr

    def __init__(self, op: str, cond: Any, lhs: Any, rhs: Any):
        super().__init__(op)
        self.add_child("cond", cond)
        self.add_child("lhs", lhs)
        self.add_child("rhs", rhs)
        self.dtype = self.lhs.dtype
        self.shape = self.lhs.shape

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        handler = self._Z3_BUILDERS.get(self.op)
        if handler is None:
            raise NotImplementedError(f"Eval for op {self.op} is not implemented")
        return handler(self)

    def concretize(self) -> Any:
        cond = self.cond.concretize()
        lhs = self.lhs.concretize()
        rhs = self.rhs.concretize()
        if isinstance(cond, SymbolicExpr):
            cond = cond.concretize()
        if isinstance(lhs, SymbolicExpr):
            lhs = lhs.concretize()
        if isinstance(rhs, SymbolicExpr):
            rhs = rhs.concretize()
        if not (
            isinstance(cond, SymbolicTensorValue)
            and isinstance(lhs, SymbolicTensorValue)
            and isinstance(rhs, SymbolicTensorValue)
        ):
            raise TypeError(
                "where concretization expects tensor value condition and values"
            )
        dtype = self.dtype
        if dtype is None:
            raise RuntimeError("where node is missing dtype information")
        data = np.where(cond.data.astype(bool), lhs.data, rhs.data)
        return SymbolicTensorValue(np.asarray(data, dtype=dtype_to_numpy(dtype)), dtype)

    def _where(self) -> tuple[Z3Expr, ConstraintConjunction]:
        def _normalize(expr):
            if not isinstance(expr, list):
                return [expr]
            return expr

        def _broadcast(*lists):
            max_len = max(len(lst) for lst in lists)
            broadcasted = []
            for lst in lists:
                if len(lst) == 1:
                    broadcasted.append(lst * max_len)
                else:
                    assert len(lst) == max_len, "Incompatible lengths for broadcasting"
                    broadcasted.append(lst)
            return tuple(broadcasted)

        cond_expr = self.cond
        lhs_expr = self.lhs
        rhs_expr = self.rhs
        cond, constraints_cond = cond_expr._to_z3()
        lhs, constraints_lhs = lhs_expr._to_z3()
        rhs, constraints_rhs = rhs_expr._to_z3()

        cond = _normalize(cond)
        lhs = _normalize(lhs)
        rhs = _normalize(rhs)
        cond, lhs, rhs = _broadcast(cond, lhs, rhs)

        z3_expr = [If(cond[i], lhs[i], rhs[i]) for i in range(len(cond))]
        constraints = _and_constraints(
            constraints_cond, constraints_lhs, constraints_rhs
        )
        return z3_expr, constraints

    _Z3_BUILDERS: ClassVar[
        dict[str, Callable[[WhereSymbolicExpr], tuple[Z3Expr, ConstraintConjunction]]]
    ] = {
        "where": _where,
    }


class FmaSymbolicExpr(SymbolicExpr):
    x: SymbolicExpr
    y: SymbolicExpr
    z: SymbolicExpr

    def __init__(self, op: str, x: Any, y: Any, z: Any):
        super().__init__(op)
        self.add_child("x", x)
        self.add_child("y", y)
        self.add_child("z", z)
        self.dtype = self.z.dtype
        self.shape = self.z.shape

    @staticmethod
    def _normalize(expr):
        return expr if isinstance(expr, list) else [expr]

    @staticmethod
    def _broadcast(*lists):
        max_len = max(len(lst) for lst in lists)
        broadcasted = []
        for lst in lists:
            if len(lst) == 1:
                broadcasted.append(lst * max_len)
            else:
                assert len(lst) == max_len, "Incompatible lengths for broadcasting"
                broadcasted.append(lst)
        return tuple(broadcasted)

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        x, constraints_x = self.x._to_z3()
        y, constraints_y = self.y._to_z3()
        z, constraints_z = self.z._to_z3()
        x, y, z = self._broadcast(
            self._normalize(x), self._normalize(y), self._normalize(z)
        )
        return (
            [x_i * y_i + z_i for x_i, y_i, z_i in zip(x, y, z)],
            _and_constraints(constraints_x, constraints_y, constraints_z),
        )

    def concretize(self) -> Any:
        x = self.x.concretize()
        y = self.y.concretize()
        z = self.z.concretize()
        data = x.data * y.data + z.data
        dtype = self.dtype
        if dtype is None:
            raise RuntimeError("fma node is missing dtype information")
        return SymbolicTensorValue(np.asarray(data, dtype=dtype_to_numpy(dtype)), dtype)


class ReduceSymbolicExpr(SymbolicExpr):
    _SUPPORTED_OPS: ClassVar[tuple[str, ...]] = (
        "sum",
        "max",
        "min",
        "xor_sum",
        "reduce_or",
        "argmax",
        "argmin",
    )
    input: SymbolicExpr
    keepdims: SymbolicExpr
    axis: SymbolicExpr | None

    def __init__(self, op: str, input: Any, axis: Any = None, keepdims: bool = False):
        if op not in self._SUPPORTED_OPS:
            raise NotImplementedError(f"Unsupported reduce op: {op}")
        super().__init__(op)
        self.add_child("input", input)
        self.add_child("keepdims", keepdims)
        self.add_child("axis", axis)

        # Compute output dtype/shape from input dtype/shape, axis, and keepdims
        input_shape = list(self.input.shape)
        assert (
            input_shape
        ), "ReduceSymbolicExpr expects block input with non-empty shape"
        # Resolve axis/keepdims from the normalized children. Negative axes are
        # valid Triton axes and must be interpreted relative to input rank.
        axis_val = self.axis.to_py() if self.axis is not None else None
        if axis_val is not None:
            axis_val = int(axis_val)
            if axis_val < 0:
                axis_val += len(input_shape)
            if axis_val < 0 or axis_val >= len(input_shape):
                raise ValueError(
                    f"Reduction axis {axis_val} is out of bounds for shape {self.input.shape}"
                )
        keepdims_val = bool(self.keepdims.to_py())
        if axis_val is not None:
            if keepdims_val:
                output_shape = (
                    input_shape[:axis_val] + [1] + input_shape[axis_val + 1 :]
                )
            else:
                output_shape = input_shape[:axis_val] + input_shape[axis_val + 1 :]
        else:
            output_shape = [1] * len(input_shape) if keepdims_val else []
        scalar_ty: SymbolicDType
        if op in ("argmax", "argmin"):
            scalar_ty = INT32
        else:
            assert self.input.dtype is not None
            scalar_ty = self.input.dtype
        self.dtype = scalar_ty
        self.shape = tuple(output_shape)

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        handler = self._Z3_BUILDERS.get(self.op)
        if handler is None:
            raise NotImplementedError(f"Eval for op {self.op} is not implemented")
        return handler(self)

    def _reduce_sum(self) -> tuple[Z3Expr, ConstraintConjunction]:
        arr, constraints = self.input._to_z3()
        return Sum(arr), constraints

    def _reduce_max(self) -> tuple[Z3Expr, ConstraintConjunction]:
        arr, constraints = self.input._to_z3()
        return reduce(lambda a, b: If(a >= b, a, b), arr), constraints

    def _reduce_min(self) -> tuple[Z3Expr, ConstraintConjunction]:
        arr, constraints = self.input._to_z3()
        return reduce(lambda a, b: If(a <= b, a, b), arr), constraints

    def _reduce_xor_sum(self) -> tuple[Z3Expr, ConstraintConjunction]:
        arr, constraints = self.input._to_z3()
        bitwidth = BinarySymbolicExpr._infer_bitwidth(self.input) or 64
        # Z3 integer expressions do not have bitwise operators; convert each
        # operand to the Triton integer width before folding the reduction.
        return (
            reduce(
                lambda a, b: BinarySymbolicExpr._from_bv(
                    BinarySymbolicExpr._to_bv(a, bitwidth)
                    ^ BinarySymbolicExpr._to_bv(b, bitwidth)
                ),
                arr,
            ),
            constraints,
        )

    def _reduce_or(self) -> tuple[Z3Expr, ConstraintConjunction]:
        arr, constraints = self.input._to_z3()
        bitwidth = BinarySymbolicExpr._infer_bitwidth(self.input) or 64
        # This uses the same bit-vector lowering as other bitwise symbolic ops,
        # but folds across one reduction axis.
        return (
            reduce(
                lambda a, b: BinarySymbolicExpr._from_bv(
                    BinarySymbolicExpr._to_bv(a, bitwidth)
                    | BinarySymbolicExpr._to_bv(b, bitwidth)
                ),
                arr,
            ),
            constraints,
        )

    _Z3_BUILDERS: ClassVar[
        dict[str, Callable[[ReduceSymbolicExpr], tuple[Z3Expr, ConstraintConjunction]]]
    ] = {
        "sum": _reduce_sum,
        "max": _reduce_max,
        "min": _reduce_min,
        "xor_sum": _reduce_xor_sum,
        "reduce_or": _reduce_or,
    }


class DotSymbolicExpr(SymbolicExpr):
    a: SymbolicExpr
    b: SymbolicExpr
    d: SymbolicExpr | None

    def __init__(self, op: str, a: Any, b: Any, d: Any = None):
        super().__init__(op)
        self.add_child("a", a)
        self.add_child("b", b)
        self.add_child("d", d)

        # dot(a, b): 2D (M,K)x(K,N)->(M,N) or 3D batched (B,M,K)x(B,K,N)->(B,M,N)
        a_shape = self.a.shape
        b_shape = self.b.shape
        if len(a_shape) == 2 and len(b_shape) == 2:
            self.shape = (a_shape[0], b_shape[1])
        elif len(a_shape) == 3 and len(b_shape) == 3:
            self.shape = (a_shape[0], a_shape[1], b_shape[2])

        # Triton always passes an accumulator d with the correct output dtype
        # (determined by out_dtype param or input types: int8->int32, fp64->fp64, etc.)
        if self.d is not None and self.d.dtype is not None:
            self.dtype = self.d.dtype
        else:
            raise ValueError(
                "DotSymbolicExpr requires accumulator (d) with a valid dtype"
            )

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        raise NotImplementedError(f"Eval for op {self.op} is not implemented")


class CumsumSymbolicExpr(SymbolicExpr):
    input: SymbolicExpr
    axis: SymbolicExpr
    reverse: SymbolicExpr

    def __init__(self, op: str, input: Any, axis: Any, reverse: Any, dtype: Any):
        super().__init__(op)
        self.add_child("input", input)
        self.add_child("axis", axis)
        self.add_child("reverse", reverse)
        dtype = self.input.dtype if dtype is None else dtype
        dtype = SymbolicExpr._NORMALIZE_VALUE(dtype)
        self.dtype, self.shape = self._unpack_dtype(dtype, self.input.shape)

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        raise NotImplementedError(f"Eval for op {self.op} is not implemented")

    def concretize(self) -> Any:
        input_concrete = self.input.concretize()
        axis = int(self.axis.to_py())
        reverse = bool(self.reverse.to_py())
        data = input_concrete.data
        if reverse:
            data = np.flip(data, axis=axis)
        out = np.cumsum(data, axis=axis)
        if reverse:
            out = np.flip(out, axis=axis)
        return SymbolicTensorValue(
            out.astype(dtype_to_numpy(input_concrete.dtype)),
            input_concrete.dtype,
        )


class MakeBlockPtrSymbolicExpr(SymbolicExpr):
    base: SymbolicExpr
    ndim: int
    block_shape_values: list[int]
    order_values: list[int]
    shape_keys: list[str]
    stride_keys: list[str]
    offset_keys: list[str]

    def __init__(
        self,
        op: str,
        base: Any,
        shape_list: Sequence[Any],
        stride_list: Sequence[Any],
        offset_list: Sequence[Any],
        block_shape_vals: Sequence[int],
        order_vals: Sequence[int],
    ):
        super().__init__(op)
        self.add_child("base", base)
        self.ndim = len(block_shape_vals)
        self.block_shape_values = list(block_shape_vals)
        self.order_values = list(order_vals)
        self.shape_keys: list[str] = []
        self.stride_keys: list[str] = []
        self.offset_keys: list[str] = []
        for i in range(self.ndim):
            shape_key = f"shape_{i}"
            stride_key = f"stride_{i}"
            offset_key = f"offset_{i}"
            self.add_child(shape_key, shape_list[i])
            self.add_child(stride_key, stride_list[i])
            self.add_child(offset_key, offset_list[i])
            self.shape_keys.append(shape_key)
            self.stride_keys.append(stride_key)
            self.offset_keys.append(offset_key)
        self.dtype = base.dtype
        self.shape = base.shape

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        raise NotImplementedError(
            "Use TensorPointerLoad/Store to access block pointers"
        )


def _offset_pointer_to_z3(
    ptr_expr: SymbolicExpr,
    offset_expr: SymbolicExpr,
) -> tuple[Z3Expr, ConstraintConjunction]:
    ptr_z3, constraints_ptr = ptr_expr._to_z3()
    offset_z3, constraints_offset = offset_expr._to_z3()
    constraints = _and_constraints(constraints_ptr, constraints_offset)
    element_size = element_bytewidth(ptr_expr.dtype)
    if not isinstance(ptr_z3, list) and not isinstance(offset_z3, list):  # hot path
        z3_expr = ptr_z3 + offset_z3 * element_size
    elif isinstance(ptr_z3, list) and isinstance(offset_z3, list):
        if len(ptr_z3) != len(offset_z3):
            raise ValueError(
                f"ptr {ptr_z3} and offset {offset_z3} don't have the same length!"
            )
        z3_expr = [p + o * element_size for p, o in zip(ptr_z3, offset_z3)]
    elif isinstance(ptr_z3, list):
        z3_expr = [p + offset_z3 * element_size for p in ptr_z3]
    else:  # isinstance(offset_z3, list):
        z3_expr = [ptr_z3 + o * element_size for o in offset_z3]
    return z3_expr, constraints


class AddPtrSymbolicExpr(SymbolicExpr):
    ptr: SymbolicExpr
    offset: SymbolicExpr

    def __init__(self, op: str, ptr: Any, offset: Any):
        super().__init__(op)
        self.add_child("ptr", ptr)
        self.add_child("offset", offset)
        self.dtype = self.ptr.dtype
        self.shape = self.ptr.shape

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        return _offset_pointer_to_z3(self.ptr, self.offset)

    def concretize(self) -> Any:
        return self.concrete_fn(self.ptr.concretize(), self.offset.concretize())  # type: ignore


class DescriptorAccessSymbolicExpr(SymbolicExpr):
    base: SymbolicExpr
    offset: SymbolicExpr
    coords: tuple[SymbolicExpr, ...]
    extents: tuple[SymbolicExpr, ...]
    block_extents: tuple[SymbolicExpr, ...]
    pred: SymbolicExpr | None

    def __init__(
        self,
        op: str,
        base: Any,
        offset: Any,
        coords: Any,
        extents: Any,
        block_extents: Any,
        pred: Any = None,
    ):
        super().__init__(op)
        self.add_child("base", base)
        self.add_child("offset", offset)
        self.add_child("coords", coords)
        self.add_child("extents", extents)
        self.add_child("block_extents", block_extents)
        self.add_child("pred", pred)
        self.dtype = self.base.dtype
        self.shape = self.base.shape
        self._addr_ok_cache: BoolRef | None = None

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        return _offset_pointer_to_z3(self.base, self.offset)

    def concretize(self) -> Any:
        return self.concrete_fn(self.base.concretize(), self.offset.concretize())  # type: ignore

    @property
    def addr_ok(self) -> BoolRef | None:
        if self._addr_ok_cache is not None:
            return self._addr_ok_cache
        if _descriptor_predicate_is_false(self.pred):
            self._addr_ok_cache = BoolVal(True)
            return self._addr_ok_cache

        constraints: list[ConstraintExpr] = []
        bounds: list[BoolRef] = []
        for coord, extent, block_extent in zip(
            self.coords,
            self.extents,
            self.block_extents,
        ):
            coord_z3, coord_constraints = coord.eval()
            if isinstance(coord_z3, list):
                return None
            extent_value = _literal_int(extent)
            block_extent_value = _literal_int(block_extent)
            if extent_value is None or block_extent_value is None:
                return None
            bounds.append(cast(BoolRef, coord_z3 >= 0))
            bounds.append(cast(BoolRef, coord_z3 + block_extent_value <= extent_value))
            if coord_constraints is not None:
                constraints.append(coord_constraints)

        ok = And(*bounds) if len(bounds) > 1 else bounds[0]
        pred_bool, pred_constraints = _predicate_bool(self.pred)
        if pred_bool is not None:
            ok = Or(Not(pred_bool), ok)
        if pred_constraints is not None:
            constraints.append(pred_constraints)
        addr_ok = _and_constraints(*constraints, ok)
        self._addr_ok_cache = cast(
            BoolRef,
            addr_ok if addr_ok is not None else BoolVal(True),
        )
        return self._addr_ok_cache


class AdvanceSymbolicExpr(SymbolicExpr):
    ptr: SymbolicExpr
    ndim: int
    delta_keys: list[str]

    def __init__(self, op: str, ptr: Any, offset_list: Sequence[Any]):
        super().__init__(op)
        self.add_child("ptr", ptr)
        self.ndim = len(offset_list)
        self.delta_keys: list[str] = []
        for i in range(self.ndim):
            dk = f"delta_{i}"
            self.add_child(dk, offset_list[i])
            self.delta_keys.append(dk)
        self.dtype = ptr.dtype
        self.shape = ptr.shape

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        raise NotImplementedError(
            "Use TensorPointerLoad/Store to access block pointers"
        )


class SplatSymbolicExpr(SymbolicExpr):
    block_type: SymbolicExpr
    arg: SymbolicExpr

    def __init__(self, op: str, block_type: Any, arg: Any):
        super().__init__(op)
        self.add_child("block_type", block_type)
        self.add_child("arg", arg)
        self.dtype = self.block_type.dtype
        self.shape = self.block_type.shape

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        return self.arg._to_z3()

    def concretize(self) -> Any:
        return self.concrete_fn(self.block_type.to_py(), self.arg.concretize())  # type: ignore


class UnsplatSymbolicExpr(SymbolicExpr):
    arg: SymbolicExpr

    def __init__(self, op: str, arg: Any):
        super().__init__(op)
        self.add_child("arg", arg)
        self.dtype = self.arg.dtype
        self.shape = ()

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        value, constraints = self.arg._to_z3()
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError(
                    f"Unsplat expects a single-element tensor, got {len(value)}"
                )
            return value[0], constraints
        return value, constraints

    def concretize(self) -> Any:
        concrete = self.arg.concretize()
        if self.concrete_fn is not None:
            return self.concrete_fn(concrete)  # type: ignore
        if not isinstance(concrete, SymbolicTensorValue):
            raise TypeError(f"Expected symbolic tensor value, got {type(concrete)}")
        if concrete.data.size != 1:
            raise ValueError(
                f"Unsplat expects a single-element tensor, got {concrete.shape}"
            )
        dtype = self.dtype
        if dtype is None:
            raise RuntimeError("unsplat node is missing dtype information")
        data = np.array([concrete.data.reshape(-1)[0]], dtype=dtype_to_numpy(dtype))
        return SymbolicTensorValue(data, dtype)


class SortSymbolicExpr(SymbolicExpr):
    input: SymbolicExpr
    dim: SymbolicExpr | None
    descending: SymbolicExpr
    stable: SymbolicExpr | None

    def __init__(self, op: str, input: Any, dim: Any, descending: Any, stable: Any):
        super().__init__(op)
        self.add_child("input", input)
        self.add_child("dim", dim)
        self.add_child("descending", descending)
        self.add_child("stable", stable)
        self.dtype = self.input.dtype
        self.shape = self.input.shape

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        return self.input._to_z3()

    def concretize(self) -> Any:
        concrete = self.input.concretize()
        axis = self.dim.to_py() if self.dim is not None else 0
        descending = bool(self.descending.to_py())
        stable = bool(self.stable.to_py()) if self.stable is not None else False
        sort_kwargs = {"kind": "stable"} if stable else {}
        data = np.sort(concrete.data, axis=axis, **sort_kwargs)
        if descending:
            data = np.flip(data, axis=axis)
        return SymbolicTensorValue(data.astype(concrete.data.dtype), concrete.dtype)


class ExpandDimsSymbolicExpr(SymbolicExpr):
    arg: SymbolicExpr
    axis: SymbolicExpr

    def __init__(self, op: str, arg: Any, axis: Any):
        super().__init__(op)
        self.add_child("arg", arg)
        self.add_child("axis", axis)
        # Update shape to reflect the new shape with an inserted dimension of size 1
        arg_shape = list(self.arg.shape) if self.arg.shape else []
        axis_val = axis if isinstance(axis, int) else axis.to_py()
        # Handle negative axis
        if axis_val < 0:
            axis_val = len(arg_shape) + 1 + axis_val
        # Insert dimension of size 1 at the specified axis
        new_shape = arg_shape[:axis_val] + [1] + arg_shape[axis_val:]
        self.dtype = self.arg.dtype
        self.shape = tuple(new_shape)

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        return self.arg._to_z3()

    def concretize(self) -> Any:
        concrete = self.arg.concretize()
        axis = self.axis.to_py()
        return SymbolicTensorValue(
            np.expand_dims(concrete.data, axis=axis).astype(concrete.data.dtype),
            concrete.dtype,
        )


class BroadcastSymbolicExpr(SymbolicExpr):
    arg: SymbolicExpr

    def __init__(self, op: str, arg: Any, shape: Any):
        super().__init__(op)
        self.add_child("arg", arg)
        shape = _shape_to_tuple(shape) if shape else ()
        self.dtype = self.arg.dtype if self.arg.dtype else INT32
        self.shape = shape if shape else self.arg.shape

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        return self.arg._to_z3()

    def concretize(self) -> Any:
        concrete = self.arg.concretize()
        return SymbolicTensorValue(
            np.broadcast_to(concrete.data, self.shape).astype(concrete.data.dtype),
            concrete.dtype,
        )


class ReshapeSymbolicExpr(SymbolicExpr):
    arg: SymbolicExpr

    def __init__(self, op: str, arg: Any, shape: Any):
        super().__init__(op)
        self.add_child("arg", arg)
        self.dtype = self.arg.dtype
        self.shape = _shape_to_tuple(shape)

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        return self.arg._to_z3()

    def concretize(self) -> Any:
        concrete = self.arg.concretize()
        return SymbolicTensorValue(
            np.reshape(concrete.data, self.shape).astype(concrete.data.dtype),
            concrete.dtype,
        )


class TransSymbolicExpr(SymbolicExpr):
    arg: SymbolicExpr
    permutation: SymbolicExpr

    def __init__(self, op: str, arg: Any, permutation: Any):
        super().__init__(op)
        self.add_child("arg", arg)
        self.add_child("permutation", permutation)
        axes = _shape_to_tuple(permutation)
        self.dtype = self.arg.dtype
        self.shape = tuple(self.arg.shape[i] for i in axes) if self.arg.shape else ()

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        return self.arg._to_z3()

    def concretize(self) -> Any:
        concrete = self.arg.concretize()
        return SymbolicTensorValue(
            np.transpose(concrete.data, axes=_shape_to_tuple(self.permutation)).astype(
                concrete.data.dtype
            ),
            concrete.dtype,
        )


class JoinSymbolicExpr(SymbolicExpr):
    lhs: SymbolicExpr
    rhs: SymbolicExpr

    def __init__(self, op: str, lhs: Any, rhs: Any):
        super().__init__(op)
        self.add_child("lhs", lhs)
        self.add_child("rhs", rhs)
        self.dtype = self.lhs.dtype
        self.shape = self.lhs.shape

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        raise NotImplementedError(
            "Join operation is not implemented in Z3 evaluation yet"
        )


class SplitSymbolicExpr(SymbolicExpr):
    arg: SymbolicExpr
    index: SymbolicExpr

    def __init__(self, op: str, arg: Any, index: Any):
        super().__init__(op)
        self.add_child("arg", arg)
        self.add_child("index", index)
        self.dtype = self.arg.dtype
        self.shape = self.arg.shape[:-1]

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        raise NotImplementedError(
            "Split operation is not implemented in Z3 evaluation yet"
        )

    def concretize(self) -> Any:
        concrete = self.arg.concretize()
        index = int(self.index.to_py())
        return SymbolicTensorValue(
            concrete.data[..., index].astype(concrete.data.dtype),
            concrete.dtype,
        )


class CastSymbolicExpr(SymbolicExpr):
    _SUPPORTED_OPS: ClassVar[tuple[str, ...]] = ("cast_impl", "bitcast")
    src: SymbolicExpr
    dst_type: SymbolicExpr

    def __init__(self, op: str, src: Any, dst_type: Any):
        if op not in self._SUPPORTED_OPS:
            raise NotImplementedError(f"Unsupported cast op: {op}")
        super().__init__(op)
        self.add_child("src", src)
        self.add_child("dst_type", dst_type)
        self.dtype, self.shape = self._unpack_dtype(
            self.dst_type.to_py(), self.src.shape
        )

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        return self.src._to_z3()

    def concretize(self) -> Any:
        src_concrete = self.src.concretize()
        if self.dtype is None:
            raise RuntimeError("cast node is missing dtype information")
        dst = type_spec(self.dtype, self.shape) if self.shape else self.dtype
        if self.concrete_fn is not None:
            return self.concrete_fn(src_concrete, dst)  # type: ignore
        return SymbolicTensorValue(
            src_concrete.data.astype(dtype_to_numpy(self.dtype)),
            self.dtype,
        )


class FpToFpSymbolicExpr(SymbolicExpr):
    src: SymbolicExpr
    dst_type: SymbolicExpr
    rounding_mode: SymbolicExpr

    def __init__(self, op: str, src: Any, dst_type: Any, rounding_mode: Any):
        super().__init__(op)
        self.add_child("src", src)
        self.add_child("dst_type", dst_type)
        self.add_child("rounding_mode", rounding_mode)
        self.dtype, self.shape = self._unpack_dtype(
            self.dst_type.to_py(), self.src.shape
        )

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        raise NotImplementedError(f"Eval for op {self.op} is not implemented")


class AtomicCasSymbolicExpr(SymbolicExpr):
    ptr: SymbolicExpr
    cmp: SymbolicExpr
    val: SymbolicExpr

    def __init__(self, op: str, ptr: Any, cmp: Any, val: Any):
        super().__init__(op)
        self.add_child("ptr", ptr)
        self.add_child("cmp", cmp)
        self.add_child("val", val)
        self.dtype = self.val.dtype
        self.shape = self.val.shape

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        raise NotImplementedError("atomic_cas operation is not implemented yet")


class AtomicRmwSymbolicExpr(SymbolicExpr):
    ptr: SymbolicExpr
    val: SymbolicExpr
    mask: SymbolicExpr | None

    def __init__(self, op: str, ptr: Any, val: Any, mask: Any = None):
        super().__init__(op)
        self.add_child("ptr", ptr)
        self.add_child("val", val)
        self.add_child("mask", mask)
        self.dtype = self.val.dtype
        self.shape = self.val.shape

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        raise NotImplementedError(f"Eval for op {self.op} is not implemented")


class TensorPointerSymbolicExpr(SymbolicExpr):
    """Common base for block-pointer load/store expressions."""

    ptr: SymbolicExpr
    boundary_check: tuple[int, ...]

    @staticmethod
    def _resolve_element_dtype(
        ptr: SymbolicExpr,
    ) -> SymbolicDType | None:
        return pointee_dtype(ptr.dtype)

    @staticmethod
    def _resolve_block_shape(ptr: SymbolicExpr) -> tuple[int, ...]:
        """Walk the block pointer chain to find the block_shape."""
        if isinstance(ptr, MakeBlockPtrSymbolicExpr):
            return tuple(ptr.block_shape_values)
        if isinstance(ptr, AdvanceSymbolicExpr):
            return TensorPointerSymbolicExpr._resolve_block_shape(ptr.ptr)
        return ()

    def _resolve_block_ptr_components(
        self, ptr: SymbolicExpr
    ) -> tuple[
        SymbolicExpr,
        list[SymbolicExpr],
        list[SymbolicExpr],
        list[SymbolicExpr],
        list[int],
    ]:
        """Walk advance chain -> (base, shapes[], strides[], offsets[], block_shape[])"""
        if isinstance(ptr, MakeBlockPtrSymbolicExpr):
            shapes = [getattr(ptr, k) for k in ptr.shape_keys]
            strides = [getattr(ptr, k) for k in ptr.stride_keys]
            offsets = [getattr(ptr, k) for k in ptr.offset_keys]
            return ptr.base, shapes, strides, offsets, ptr.block_shape_values
        elif isinstance(ptr, AdvanceSymbolicExpr):
            base, shapes, strides, offsets, bs = self._resolve_block_ptr_components(
                ptr.ptr
            )
            deltas = [getattr(ptr, k) for k in ptr.delta_keys]
            new_offsets = [
                SymbolicExpr.create("add", off, d) for off, d in zip(offsets, deltas)
            ]
            return base, shapes, strides, new_offsets, bs
        raise TypeError(f"Expected block pointer, got {type(ptr)}")

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        (
            base,
            shapes,
            strides,
            offsets,
            block_shape,
        ) = self._resolve_block_ptr_components(self.ptr)

        elem_size = element_bytewidth(base.dtype)

        base_z3, c_base = base._to_z3()
        addr = base_z3
        parts: list[ConstraintExpr | Sequence[ConstraintExpr]] = []
        if c_base:
            parts.append(c_base)

        for d in range(len(block_shape)):
            k_d = Int(f"blk_k_{d}")
            off_z3, c_off = offsets[d]._to_z3()
            stride_z3, c_stride = strides[d]._to_z3()

            addr = addr + (off_z3 + k_d) * stride_z3 * elem_size
            parts.append(And(k_d >= 0, k_d < block_shape[d]))
            if c_off:
                parts.append(c_off)
            if c_stride:
                parts.append(c_stride)

            if d in self.boundary_check:
                shape_z3, c_shape = shapes[d]._to_z3()
                parts.append(And(off_z3 + k_d >= 0, off_z3 + k_d < shape_z3))
                if c_shape:
                    parts.append(c_shape)

        return addr, _and_constraints(*parts)


class TensorPointerLoadSymbolicExpr(TensorPointerSymbolicExpr):
    def __init__(self, op: str, ptr: Any, boundary_check: Any):
        super().__init__(op)
        self.add_child("ptr", ptr)
        self.boundary_check = tuple(boundary_check) if boundary_check else ()
        self.dtype = self._resolve_element_dtype(ptr)
        self.shape = self._resolve_block_shape(ptr)


class TensorPointerStoreSymbolicExpr(TensorPointerSymbolicExpr):
    value: SymbolicExpr

    def __init__(self, op: str, ptr: Any, value: Any, boundary_check: Any):
        super().__init__(op)
        self.add_child("ptr", ptr)
        self.add_child("value", value)
        self.boundary_check = tuple(boundary_check) if boundary_check else ()
        self.dtype = self._resolve_element_dtype(ptr)
        self.shape = ()


SymbolicExpr.register_op_class(ConstSymbolicExpr, ("const",))
SymbolicExpr.register_op_class(PidSymbolicExpr, ("pid",))
SymbolicExpr.register_op_class(ArangeSymbolicExpr, ("arange",))
SymbolicExpr.register_op_class(LoadSymbolicExpr, ("load",))
SymbolicExpr.register_op_class(StoreSymbolicExpr, ("store",))
SymbolicExpr.register_op_class(UnarySymbolicExpr, SymbolicExpr.UNARY_OPS)
SymbolicExpr.register_op_class(BinarySymbolicExpr, SymbolicExpr.BINARY_OPS)
SymbolicExpr.register_op_class(WhereSymbolicExpr, ("where",))
SymbolicExpr.register_op_class(FmaSymbolicExpr, ("fma",))
SymbolicExpr.register_op_class(
    ReduceSymbolicExpr,
    ("sum", "max", "min", "xor_sum", "reduce_or", "argmax", "argmin"),
)
SymbolicExpr.register_op_class(DotSymbolicExpr, ("dot",))
SymbolicExpr.register_op_class(CumsumSymbolicExpr, ("cumsum",))
SymbolicExpr.register_op_class(MakeBlockPtrSymbolicExpr, ("make_block_ptr",))
SymbolicExpr.register_op_class(AddPtrSymbolicExpr, ("addptr",))
SymbolicExpr.register_op_class(DescriptorAccessSymbolicExpr, ("descriptor_access",))
SymbolicExpr.register_op_class(AdvanceSymbolicExpr, ("advance",))
SymbolicExpr.register_op_class(SplatSymbolicExpr, ("splat",))
SymbolicExpr.register_op_class(UnsplatSymbolicExpr, ("unsplat",))
SymbolicExpr.register_op_class(SortSymbolicExpr, ("sort",))
SymbolicExpr.register_op_class(ExpandDimsSymbolicExpr, ("expand_dims",))
SymbolicExpr.register_op_class(BroadcastSymbolicExpr, ("broadcast",))
SymbolicExpr.register_op_class(ReshapeSymbolicExpr, ("reshape",))
SymbolicExpr.register_op_class(TransSymbolicExpr, ("trans",))
SymbolicExpr.register_op_class(JoinSymbolicExpr, ("join",))
SymbolicExpr.register_op_class(SplitSymbolicExpr, ("split",))
SymbolicExpr.register_op_class(CastSymbolicExpr, ("cast_impl", "bitcast"))
SymbolicExpr.register_op_class(FpToFpSymbolicExpr, ("fp_to_fp",))
SymbolicExpr.register_op_class(AtomicCasSymbolicExpr, ("atomic_cas",))
SymbolicExpr.register_op_class(AtomicRmwSymbolicExpr, ("atomic_rmw",))
SymbolicExpr.register_op_class(TensorPointerLoadSymbolicExpr, ("tensor_pointer_load",))
SymbolicExpr.register_op_class(
    TensorPointerStoreSymbolicExpr, ("tensor_pointer_store",)
)


# ── Shared constants and utilities for symbolic clients ──────────

_UNARY_NUMPY_TO_SYM_OP: dict[Callable[..., Any], str] = {
    np.cos: "cos",
    np.exp: "exp",
    np.exp2: "exp2",
    np.abs: "abs",
    np.floor: "floor",
    np.ceil: "ceil",
    np.log: "log",
    np.log2: "log2",
    np.sqrt: "sqrt",
    np.sin: "sin",
    np.negative: "negative",
}

_BINARY_NUMPY_TO_SYM_OP: dict[Callable[..., Any], str] = {
    np.add: "add",
    np.subtract: "sub",
    np.multiply: "mul",
    np.divide: "div",
    np.less: "less",
    np.less_equal: "less_equal",
    np.greater: "greater",
    np.greater_equal: "greater_equal",
    np.not_equal: "not_equal",
    np.equal: "equal",
    np.fmod: "mod",
    np.maximum: "maximum",
    np.fmax: "maximum",
    np.minimum: "minimum",
    np.fmin: "minimum",
    np.bitwise_and: "bitwise_and",
    np.bitwise_or: "bitwise_or",
    np.bitwise_xor: "bitwise_xor",
    np.right_shift: "right_shift",
    np.left_shift: "left_shift",
}


def symbolic_tensor_descriptor_value(
    descriptor: Any,
    *,
    base: Any = None,
    shape: Any = None,
    strides: Any = None,
    block_shape: Any = None,
) -> SymbolicTensorDescriptorValue:
    registered = _SYMBOLIC_TENSOR_DESCRIPTORS.get(descriptor)
    if registered is not None:
        return registered

    if base is None:
        base = descriptor.base
        shape = descriptor.shape
        strides = descriptor.strides
        block_shape = descriptor.block_shape

    value = SymbolicTensorDescriptorValue(
        base=base,
        shape=_shape_to_tuple(shape),
        strides=_shape_to_tuple(strides),
        block_shape=_shape_to_tuple(block_shape),
        base_offset=int(getattr(descriptor, "base_offset", 0)),
    )
    # The weak-key registry follows the frontend descriptor object's lifetime.
    _SYMBOLIC_TENSOR_DESCRIPTORS[descriptor] = value
    return value


def _descriptor_base_ptr(base: Any) -> SymbolicExpr:
    if isinstance(base, Tensor):
        dtype_name = str(base.dtype).removeprefix("torch.")
        return SymbolicExpr.create(
            "const",
            int(base.data_ptr()),
            pointer_type(DTYPE_BY_NAME.get(dtype_name, FLOAT32)),
        )
    return cast(SymbolicExpr, SymbolicExpr.from_value(base))


def _descriptor_coords(coords: Any) -> tuple[Any, ...]:
    if isinstance(coords, (tuple, list)) or _is_triton_tuple(coords):
        return tuple(coords)
    return (coords,)


def _is_triton_tuple(value: Any) -> bool:
    # Keep symbolic_engine import-light; importing triton.language here breaks
    # clients that only need the shared symbolic model.
    value_type = type(value)
    return (
        value_type.__module__ == "triton.language.core"
        and value_type.__name__ == "tuple"
    )


def _descriptor_predicate_is_false(pred: Any) -> bool:
    if pred is None:
        return False
    value = _literal_int(pred)
    return value == 0


def _predicate_bool(pred: Any) -> tuple[BoolRef | None, ConstraintConjunction]:
    if pred is None:
        return None, None
    pred_expr = cast(SymbolicExpr, SymbolicExpr.from_value(pred))
    pred_z3, pred_constraints = pred_expr.eval()
    if isinstance(pred_z3, list):
        pred_bool = Or(*(_constraint_to_bool(item) for item in pred_z3))
    else:
        pred_bool = _constraint_to_bool(pred_z3)
    return pred_bool, pred_constraints


def _descriptor_offset(
    descriptor: SymbolicTensorDescriptorValue, coords: tuple[Any, ...]
) -> Any:
    offset: Any = SymbolicExpr.create("const", int(descriptor.base_offset), INT32)
    for coord, stride in zip(coords, descriptor.strides):
        term = coord
        if stride != 1:
            term = SymbolicExpr.create(
                "mul", coord, SymbolicExpr.create("const", int(stride), INT32)
            )
        offset = SymbolicExpr.create("add", offset, term)
    return offset


def symbolic_tensor_descriptor_access(
    descriptor: Any,
    coords: Any,
    *,
    pred: Any = None,
) -> SymbolicExpr:
    descriptor_value = symbolic_tensor_descriptor_value(descriptor)
    coord_values = _descriptor_coords(coords)

    offset = _descriptor_offset(descriptor_value, coord_values)
    base_ptr = _descriptor_base_ptr(descriptor_value.base)
    return SymbolicExpr.create(
        "descriptor_access",
        base_ptr,
        offset,
        coord_values,
        descriptor_value.shape,
        descriptor_value.block_shape,
        pred,
    )


@dataclass
class RangeWrapper:
    iterable: Any
    length: int
    start: int
    stop: int
    step: int

    def __iter__(self) -> Iterator[Any]:
        return iter(self.iterable)

    def __len__(self) -> int:
        return self.length


class SymbolicClient(Client):
    """Base class for clients that use the symbolic engine.

    Carries the shared launch-scoped state (tensor registry, Z3 solver, address
    and pid premises), the pure-math + memory-access operation overriders,
    for-loop infrastructure, and the pending-check loop-flush template.
    Subclasses only provide their domain-specific ``_handle_access_check`` and
    ``_process_pending_check`` hooks (plus optional overrides like
    ``_addr_ok_premise`` and the cache hooks).
    """

    LOG_TAG: ClassVar[str] = "SymbolicClient"
    # Verb printed inside ``_loop_hook_after`` when each pending check is
    # flushed. Subclasses override (e.g. "recording" for race detector,
    # "checking" for sanitizer) so the verbose log reads correctly.
    LOG_VERB: ClassVar[str] = "processing"

    def __init__(self) -> None:
        super().__init__()
        self.loop_stack: list[LoopContext] = []
        self.grid: tuple[int, ...] | None = None
        self.tensors: list[Tensor] = []
        self.tensor_addrs: list[tuple[int, int, Tensor]] = []
        self.tensor_names: dict[int, set[str]] = {}
        self.need_full_grid: bool | None = None
        self.last_grid: tuple[int, int, int] | None = None
        self._active_blocks: int = 0
        self._launch_should_stop: bool = False
        self._pending_launch_clear: bool = False
        self.addr_ok: BoolRef | None = None
        self.pid_ok: BoolRef | None = None
        self.solver: Solver | None = Solver()
        self.addr_sym: ArithRef | None = Int("addr")
        self.addr_ok_cache: dict[int, BoolRef] = {}
        self.loop_iterator_constraint_cache: dict[
            tuple[LoopSite, int, int, int], BoolRef
        ] = {}
        self.access_check_cache: set[int] = set()
        self.op_overrider_map = self._build_op_overrider_map()
        self.op_callback_cache: dict[type[Op], OpCallbacks] = {}
        self.for_loop_callbacks = ForLoopCallbacks(
            range_wrapper_factory=self.lock_fn(self._wrap_range),
            before_loop_callback=self.lock_fn(self._loop_hook_before),
            loop_iter_overrider=self.lock_fn(self._loop_hook_iter_overrider),
            after_loop_callback=self.lock_fn(self._loop_hook_after),
        )
        SymbolicExpr.set_loop_ctx_provider(
            lambda *_args, **_kwargs: (self.loop_stack[-1] if self.loop_stack else None)
        )

    # ── Shared operation overriders ───────────────────────────────

    def _op_program_id_overrider(self, axis):
        return SymbolicExpr.create("pid", axis)

    def _op_unary_op_overrider(self, arg, op):
        arg_sym = SymbolicExpr.from_value(arg)
        name = _UNARY_NUMPY_TO_SYM_OP.get(op)
        if name is None:
            raise NotImplementedError(f"Unsupported unary operation: {op}")
        return SymbolicExpr.create(name, arg_sym)

    def _op_binary_op_overrider(self, lhs, rhs, op):
        lhs_sym = SymbolicExpr.from_value(lhs)
        rhs_sym = SymbolicExpr.from_value(rhs)
        op_name = _BINARY_NUMPY_TO_SYM_OP.get(op)
        if op_name is None:
            raise NotImplementedError(f"Unsupported binary operation: {op}")
        return SymbolicExpr.create(op_name, lhs_sym, rhs_sym)

    def _op_ternary_op_overrider(self, lhs, rhs, other, op):
        lhs_sym = SymbolicExpr.from_value(lhs)
        rhs_sym = SymbolicExpr.from_value(rhs)
        other_sym = SymbolicExpr.from_value(other)
        if op is np.where:
            return SymbolicExpr.create("where", lhs_sym, rhs_sym, other_sym)
        raise NotImplementedError(f"Unsupported ternary operation: {op}")

    def _op_fma_overrider(self, x, y, z):
        return SymbolicExpr.create(
            "fma",
            SymbolicExpr.from_value(x),
            SymbolicExpr.from_value(y),
            SymbolicExpr.from_value(z),
        )

    def _op_addptr_overrider(self, ptr, offset):
        return SymbolicExpr.create(
            "addptr",
            SymbolicExpr.from_value(ptr),
            SymbolicExpr.from_value(offset),
        )

    def _op_dot_overrider(self, a, b, d, input_precision, max_num_imprecise_acc):
        a_sym = SymbolicExpr.from_value(a)
        b_sym = SymbolicExpr.from_value(b)
        d_sym = SymbolicExpr.from_value(d) if d is not None else None
        return SymbolicExpr.create("dot", a_sym, b_sym, d_sym)

    def _op_make_range_overrider(self, ret_ty, start, end):
        return SymbolicExpr.create(
            "arange",
            SymbolicExpr.from_value(ret_ty),
            SymbolicExpr.from_value(start),
            SymbolicExpr.from_value(end),
        )

    def _op_expand_dims_overrider(self, arg, axis):
        return SymbolicExpr.create("expand_dims", SymbolicExpr.from_value(arg), axis)

    def _op_broadcast_overrider(self, arg, shape):
        return SymbolicExpr.create("broadcast", SymbolicExpr.from_value(arg), shape)

    def _op_reduce_sum_overrider(self, input, axis=None, keep_dims=False, **kwargs):
        return SymbolicExpr.create(
            "sum", SymbolicExpr.from_value(input), axis, keep_dims
        )

    def _op_reduce_xor_overrider(self, input, axis=None, keep_dims=False, **kwargs):
        return SymbolicExpr.create(
            "xor_sum", SymbolicExpr.from_value(input), axis, keep_dims
        )

    def _op_reduce_or_overrider(self, input, axis=None, keep_dims=False, **kwargs):
        return SymbolicExpr.create(
            "reduce_or", SymbolicExpr.from_value(input), axis, keep_dims
        )

    def _op_reduce_max_overrider(self, input, axis=None, keep_dims=False, **kwargs):
        input_sym = SymbolicExpr.from_value(input)
        val = SymbolicExpr.create("max", input_sym, axis, keep_dims)
        if kwargs.get("return_indices", False):
            idx = SymbolicExpr.create("argmax", input_sym, axis, keep_dims)
            return (val, idx)
        return val

    def _op_reduce_min_overrider(self, input, axis=None, keep_dims=False, **kwargs):
        input_sym = SymbolicExpr.from_value(input)
        val = SymbolicExpr.create("min", input_sym, axis, keep_dims)
        if kwargs.get("return_indices", False):
            idx = SymbolicExpr.create("argmin", input_sym, axis, keep_dims)
            return (val, idx)
        return val

    def _op_sort_overrider(self, input, dim=None, descending=False, stable=None):
        return SymbolicExpr.create(
            "sort",
            SymbolicExpr.from_value(input),
            dim,
            descending,
            stable,
        )

    def _op_splat_overrider(self, shape, arg):
        return SymbolicExpr.create("splat", shape, SymbolicExpr.from_value(arg))

    def _op_unsplat_overrider(self, arg):
        return SymbolicExpr.create("unsplat", SymbolicExpr.from_value(arg))

    def _op_idiv_overrider(self, lhs, rhs):
        return SymbolicExpr.from_value(lhs) // SymbolicExpr.from_value(rhs)

    def _op_rsqrt_overrider(self, arg):
        return SymbolicExpr.create("rsqrt", SymbolicExpr.from_value(arg))

    def _op_cast_impl_overrider(self, src, dst_type):
        return SymbolicExpr.create("cast_impl", src, dst_type)

    def _op_reshape_overrider(self, arg, shape, allow_reorder):
        return SymbolicExpr.create(
            "reshape",
            SymbolicExpr.from_value(arg),
            SymbolicExpr.from_value(shape),
        )

    def _op_trans_overrider(self, arg, perm=(1, 0)):
        return SymbolicExpr.create("trans", SymbolicExpr.from_value(arg), perm)

    def _op_join_overrider(self, lhs, rhs):
        return SymbolicExpr.create(
            "join",
            SymbolicExpr.from_value(lhs),
            SymbolicExpr.from_value(rhs),
        )

    def _op_split_overrider(self, arg):
        arg_sym = SymbolicExpr.from_value(arg)
        return (
            SymbolicExpr.create("split", arg_sym, 0),
            SymbolicExpr.create("split", arg_sym, 1),
        )

    def _op_fabs_overrider(self, arg):
        return SymbolicExpr.create("fabs", SymbolicExpr.from_value(arg))

    def _op_ashr_overrider(self, lhs, rhs):
        return SymbolicExpr.create(
            "ashr",
            SymbolicExpr.from_value(lhs),
            SymbolicExpr.from_value(rhs),
        )

    def _op_advance_overrider(self, ptr, offsets):
        ptr_sym = SymbolicExpr.from_value(ptr)
        offset_syms = [SymbolicExpr.from_value(o) for o in offsets]
        return SymbolicExpr.create("advance", ptr_sym, offset_syms)

    def _op_fp_to_fp_overrider(self, src, dst_type, rounding_mode):
        return SymbolicExpr.create(
            "fp_to_fp",
            SymbolicExpr.from_value(src),
            dst_type,
            rounding_mode,
        )

    def _op_umulhi_overrider(self, lhs, rhs):
        return SymbolicExpr.create(
            "umulhi",
            SymbolicExpr.from_value(lhs),
            SymbolicExpr.from_value(rhs),
        )

    def _op_cumsum_overrider(self, input, axis, reverse=False, dtype=None):
        return SymbolicExpr.create(
            "cumsum",
            SymbolicExpr.from_value(input),
            axis,
            reverse,
            dtype,
        )

    def _op_bitcast_overrider(self, src, dst_type):
        return SymbolicExpr.create("bitcast", SymbolicExpr.from_value(src), dst_type)

    def _op_raw_load_overrider(self, ptr, cache_modifier, eviction_policy, is_volatile):
        return self._op_load_overrider(
            ptr, None, None, cache_modifier, eviction_policy, is_volatile
        )

    def _op_raw_store_overrider(self, ptr, value, cache_modifier, eviction_policy):
        return self._op_store_overrider(
            ptr, value, None, cache_modifier, eviction_policy
        )

    def _op_atomic_cas_overrider(self, ptr, cmp, val, sem, scope):
        ptr_sym = SymbolicExpr.from_value(ptr)
        cmp_sym = SymbolicExpr.from_value(cmp)
        val_sym = SymbolicExpr.from_value(val)
        return SymbolicExpr.create("atomic_cas", ptr_sym, cmp_sym, val_sym)

    def _op_atomic_rmw_overrider(self, rmwOp, ptr, val, mask, sem, scope):
        ptr_sym = SymbolicExpr.from_value(ptr)
        val_sym = SymbolicExpr.from_value(val)
        mask_sym = SymbolicExpr.from_value(mask)
        return SymbolicExpr.create("atomic_rmw", ptr_sym, val_sym, mask_sym)

    def _build_op_overrider_map(self) -> dict[type[Op], Callable]:
        """Return a mapping of shared Op types to their overrider methods."""
        return {
            ProgramId: self._op_program_id_overrider,
            UnaryOp: self._op_unary_op_overrider,
            BinaryOp: self._op_binary_op_overrider,
            TernaryOp: self._op_ternary_op_overrider,
            Fma: self._op_fma_overrider,
            AddPtr: self._op_addptr_overrider,
            Dot: self._op_dot_overrider,
            MakeRange: self._op_make_range_overrider,
            ExpandDims: self._op_expand_dims_overrider,
            Broadcast: self._op_broadcast_overrider,
            ReduceSum: self._op_reduce_sum_overrider,
            ReduceXor: self._op_reduce_xor_overrider,
            ReduceOr: self._op_reduce_or_overrider,
            ReduceMax: self._op_reduce_max_overrider,
            ReduceMin: self._op_reduce_min_overrider,
            Sort: self._op_sort_overrider,
            Splat: self._op_splat_overrider,
            Unsplat: self._op_unsplat_overrider,
            Idiv: self._op_idiv_overrider,
            Rsqrt: self._op_rsqrt_overrider,
            CastImpl: self._op_cast_impl_overrider,
            Reshape: self._op_reshape_overrider,
            Trans: self._op_trans_overrider,
            Join: self._op_join_overrider,
            Split: self._op_split_overrider,
            Fabs: self._op_fabs_overrider,
            Ashr: self._op_ashr_overrider,
            Advance: self._op_advance_overrider,
            FpToFp: self._op_fp_to_fp_overrider,
            Umulhi: self._op_umulhi_overrider,
            CumSum: self._op_cumsum_overrider,
            Bitcast: self._op_bitcast_overrider,
            PtrToInt: self._op_bitcast_overrider,
            IntToPtr: self._op_bitcast_overrider,
            AtomicCas: self._op_atomic_cas_overrider,
            AtomicRMW: self._op_atomic_rmw_overrider,
            RawLoad: self._op_raw_load_overrider,
            RawStore: self._op_raw_store_overrider,
            Load: self._op_load_overrider,
            Store: self._op_store_overrider,
            MakeBlockPointer: self._op_make_block_ptr_overrider,
            TensorPointerLoad: self._op_tensor_pointer_load_overrider,
            TensorPointerStore: self._op_tensor_pointer_store_overrider,
        }

    def register_op_callback(self, op_type: type[Op], *args, **kwargs) -> OpCallbacks:
        cached = self.op_callback_cache.get(op_type)
        if cached is not None:
            return cached
        overrider = self.op_overrider_map.get(op_type)
        if overrider is not None:
            callbacks = OpCallbacks(op_overrider=self.lock_fn(overrider))
        else:
            callbacks = OpCallbacks()
        self.op_callback_cache[op_type] = callbacks
        return callbacks

    # ── For-loop infrastructure ───────────────────────────────────

    def _on_data_dependent_value(self) -> None:
        """Hook called when a data-dependent value forces concretization."""
        self.need_full_grid = True

    def _materialize_memory_operand(self, expr: Any) -> Any:
        if not isinstance(expr, SymbolicExpr):
            return expr

        materialized = False
        for anchor_op in ("sort", "cumsum", "load"):
            if expr.has_op(anchor_op):
                materialized = True
                expr = expr.replace_subtree(anchor_op)

        if materialized:
            self._on_data_dependent_value()
            if expr.has_vector_const():
                expr = expr.replace_subtree()
            return expr

        return expr

    def _symbolic_memory_ptr(self, ptr: Any) -> SymbolicExpr:
        if isinstance(ptr, TensorDescriptorAccess):
            return symbolic_tensor_descriptor_access(
                ptr.descriptor,
                ptr.coords,
                pred=ptr.pred,
            )
        return cast(SymbolicExpr, SymbolicExpr.from_value(ptr))

    def _should_skip_loop_hooks(self) -> bool:
        """Return True to skip loop hook processing."""
        return False

    def _materialize_loop_value(self, expr: Any) -> int:
        if isinstance(expr, SymbolicExpr):
            if expr.op == "const":
                return SymbolicExprDataWrapper.coerce_int(expr.to_py())
            elif expr.has_op("load"):
                self._on_data_dependent_value()
                expr = expr.replace_subtree("load")
                # replace_subtree("load") only concretizes load nodes, so the
                # result may still be a compound op
                # (e.g. div(load, load) -> div(const, const)).
                # Calling replace_subtree() with no anchor_op unconditionally
                # concretizes every node in the tree, collapsing the whole
                # expression into a single const value.
                if expr.op != "const":
                    expr = expr.replace_subtree()
                return SymbolicExprDataWrapper.coerce_int(expr.to_py())
            else:
                z3_expr, _ = expr.eval()
                if isinstance(z3_expr, IntNumRef):
                    return z3_expr.as_long()
                self._on_data_dependent_value()
                expr = expr.replace_subtree()
                return SymbolicExprDataWrapper.coerce_int(expr.to_py())
        return int(expr)

    def _wrap_range(
        self,
        iterable,
        _loop_site,
        _range_type,
        iter_args=None,
        iter_kwargs=None,
        _iter_callable=None,
    ):
        if self._should_skip_loop_hooks():
            return None
        iter_args = tuple(iter_args or ())
        iter_kwargs = iter_kwargs or {}

        if isinstance(iterable, RangeWrapper):
            return iterable

        args = tuple(SymbolicExpr.from_value(v) for v in iter_args)
        if not args and iter_kwargs:
            start_expr = SymbolicExpr.from_value(iter_kwargs.get("start", 0))
            stop_expr = SymbolicExpr.from_value(
                iter_kwargs.get("stop", iter_kwargs.get("end"))
            )
            step_expr = SymbolicExpr.from_value(iter_kwargs.get("step", 1))
            if stop_expr is not None:
                args = (start_expr, stop_expr, step_expr)

        if not args and (
            isinstance(iterable, range)
            or (
                hasattr(iterable, "start")
                and hasattr(iterable, "stop")
                and hasattr(iterable, "step")
            )
        ):
            args = tuple(
                SymbolicExpr.from_value(getattr(iterable, attr, default))
                for attr, default in (("start", 0), ("stop", 0), ("step", 1))
            )

        if not args:
            return None

        start_expr, stop_expr, step_expr = 0, None, 1
        if len(args) == 1:
            stop_expr = args[0]
        elif len(args) == 2:
            start_expr, stop_expr = args[0], args[1]
        else:
            start_expr, stop_expr, step_expr = args[0], args[1], args[2]

        step = self._materialize_loop_value(step_expr)
        start = self._materialize_loop_value(start_expr)
        stop = self._materialize_loop_value(stop_expr)

        concrete_range = range(start, stop, step)
        length = len(concrete_range)

        return RangeWrapper(
            concrete_range, length=length, start=start, stop=stop, step=step
        )

    def _loop_hook_before(self, loop_site: LoopSite, iterable):
        if self._should_skip_loop_hooks():
            return
        if not isinstance(iterable, RangeWrapper):
            if cfg.verbose:
                print("not a range wrapper, skipping for-loop iterator association.")
            return

        # LoopSite's string form embeds both the function-relative line and
        # file token, keeping symbolic iterator vars distinct for loops at the
        # same relative line in different files.
        idx_z3 = Int(f"loop_i_{loop_site}")
        sym = SymbolicExpr.create("const", idx_z3, INT32)
        idx = SymbolicExpr.wrap_loop_index(sym, INT32)
        constraint_key = (loop_site, iterable.start, iterable.stop, iterable.step)
        iterator_constraint = self.loop_iterator_constraint_cache.get(constraint_key)
        if iterator_constraint is None:
            iterator_constraint = _range_to_iterator_constraint(
                idx_z3,
                start=iterable.start,
                stop=iterable.stop,
                step=iterable.step,
            )
            self.loop_iterator_constraint_cache[constraint_key] = iterator_constraint
        ctx = LoopContext(
            loop_site,
            iterable.length,
            idx,
            idx_z3,
            iterator_constraint,
            start=iterable.start,
            stop=iterable.stop,
            step=iterable.step,
        )
        sym.loop_ctx = ctx
        self.loop_stack.append(ctx)

        if cfg.verbose:
            print(f"[{self.LOG_TAG}] ▶ enter loop@{loop_site}, len={iterable.length}")

    def _loop_hook_iter_overrider(self, loop_site: LoopSite, idx):
        if self._should_skip_loop_hooks():
            return idx
        if self.loop_stack and self.loop_stack[-1].loop_site == loop_site:
            self.loop_stack[-1].current_value = int(idx)
            return self.loop_stack[-1].idx
        return idx

    def _loop_hook_after(self, loop_site: LoopSite) -> None:
        if self._should_skip_loop_hooks():
            return
        if not self.loop_stack or self.loop_stack[-1].loop_site != loop_site:
            return
        ctx = self.loop_stack.pop()

        solver = self.solver
        addr_sym = self.addr_sym
        assert solver is not None
        assert addr_sym is not None

        all_iterator_constraints: list[BoolRef] = []
        if ctx.pending_checks:
            solver.push()
            iterator_constraint = ctx.iterator_constraint
            if iterator_constraint is not None:
                solver.add(iterator_constraint)
                all_iterator_constraints.append(iterator_constraint)

            # Also add constraints for all outer loops that are still active.
            for outer_ctx in self.loop_stack:
                outer_constraint = outer_ctx.iterator_constraint
                if outer_constraint is not None:
                    solver.add(outer_constraint)
                    all_iterator_constraints.append(outer_constraint)

        for pending in ctx.pending_checks:
            if cfg.verbose:
                print(
                    f"[{self.LOG_TAG}] ▶ {self.LOG_VERB}:",
                    pending.addr_expr,
                    f" with iterator constraints: {all_iterator_constraints} ",
                    f" and expression-related constraints: {pending.constraints} ",
                )
            self._process_pending_check(ctx, pending, all_iterator_constraints)

        if ctx.pending_checks:
            solver.pop()

        if cfg.verbose:
            print(
                f"[{self.LOG_TAG}] ▶ leave loop@{loop_site} end. "
                f"(processed {len(ctx.pending_checks)} unique addr patterns)"
            )

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        return self.for_loop_callbacks

    # ── Tensor resolution helpers ─────────────────────────────────────

    def _collect_tensor_base(self, expr: SymbolicExpr) -> Any | None:
        def walk(node: SymbolicExpr) -> Any | None:
            if (
                node.op == "const"
                and is_pointer_dtype(node.dtype)
                and not isinstance(
                    getattr(node, "value", None),
                    (SymbolicScalarDType, SymbolicPointerDType, SymbolicTypeSpec),
                )
            ):
                return node.to_py()
            for child in node.children.values():
                if child is None:
                    continue
                if isinstance(child, tuple):
                    for item in child:
                        base = walk(item)
                        if base is not None:
                            return base
                else:
                    base = walk(child)
                    if base is not None:
                        return base
            return None

        return walk(expr)

    def _record_tensor_name(self, tensor: Tensor, name: str) -> None:
        if not name:
            return
        names = self.tensor_names.setdefault(id(tensor), set())
        names.add(name)

    def _get_tensor_name(self, tensor: Tensor) -> str | None:
        names = self.tensor_names.get(id(tensor))
        if not names:
            return None
        return ", ".join(sorted(names))

    def _resolve_tensor(self, symbolic_expr: SymbolicExpr) -> Tensor | None:
        """Best-effort base-tensor lookup. Returns None on failure.

        Deliberately does NOT use a nearest-segment fallback (that's only
        meaningful with a concrete witness address from the Z3 model, which
        doesn't exist at access-capture time).
        """
        base = self._collect_tensor_base(symbolic_expr)
        if base is None:
            return None
        base_candidates = base if isinstance(base, list) else [base]
        for candidate in base_candidates:
            candidate = int(candidate)
            for tensor in self.tensors:
                if tensor.data_ptr() == candidate:
                    return tensor
            for start, end, tensor in self.tensor_addrs:
                if start <= candidate <= end:
                    return tensor
        return None

    # ── Memory-op overriders ──────────────────────────────────────────

    def _op_make_block_ptr_overrider(
        self, base, shape, strides, offsets, tensor_shape, order
    ):
        base_sym = SymbolicExpr.from_value(base)
        shape_syms = [SymbolicExpr.from_value(s) for s in shape]
        stride_syms = [SymbolicExpr.from_value(s) for s in strides]
        offset_syms = [SymbolicExpr.from_value(o) for o in offsets]
        block_shape_vals = [int(b) for b in tensor_shape]
        order_vals = [int(o) for o in order]
        return SymbolicExpr.create(
            "make_block_ptr",
            base_sym,
            shape_syms,
            stride_syms,
            offset_syms,
            block_shape_vals,
            order_vals,
        )

    def _op_load_overrider(self, ptr, mask, other, *args):
        ptr = self._materialize_memory_operand(ptr)
        mask = self._materialize_memory_operand(mask)
        ptr_sym = self._symbolic_memory_ptr(ptr)
        mask_sym = SymbolicExpr.from_value(mask) if mask is not None else None
        other_sym = SymbolicExpr.from_value(other) if other is not None else None
        ret = SymbolicExpr.create("load", ptr_sym, mask_sym, other_sym)

        self._handle_access_check(ret, Load, "read")
        return ret

    def _op_store_overrider(self, ptr, value, mask, *args):
        ptr = self._materialize_memory_operand(ptr)
        mask = self._materialize_memory_operand(mask)
        ptr_sym = self._symbolic_memory_ptr(ptr)
        value_sym = SymbolicExpr.from_value(value)
        mask_sym = SymbolicExpr.from_value(mask) if mask is not None else None
        ret = SymbolicExpr.create("store", ptr_sym, value_sym, mask_sym)

        self._handle_access_check(ret, Store, "write")
        return ret

    def _op_tensor_pointer_load_overrider(
        self,
        ptr,
        boundary_check,
        padding_option,
        cache_modifier,
        eviction_policy,
        is_volatile,
    ):
        ptr = self._materialize_memory_operand(ptr)
        ptr_sym = SymbolicExpr.from_value(ptr)
        ret = SymbolicExpr.create("tensor_pointer_load", ptr_sym, boundary_check)
        self._handle_access_check(ret, TensorPointerLoad, "read")
        return ret

    def _op_tensor_pointer_store_overrider(
        self, ptr, value, boundary_check, cache_modifier, eviction_policy
    ):
        ptr = self._materialize_memory_operand(ptr)
        ptr_sym = SymbolicExpr.from_value(ptr)
        value_sym = SymbolicExpr.from_value(value)
        ret = SymbolicExpr.create(
            "tensor_pointer_store", ptr_sym, value_sym, boundary_check
        )
        self._handle_access_check(ret, TensorPointerStore, "write")
        return ret

    # ── Abstract hooks subclasses MUST implement ──────────────────────

    def _handle_access_check(
        self, expr: SymbolicExpr, op_type: type[Op], access_mode: AccessMode
    ) -> None:
        """Per-access domain logic: invoked from each memory-op overrider.

        Typical impls either record the access (race detector) or run a Z3
        satisfiability check (sanitizer). Inside a loop the impl usually
        defers via the ``pending_checks`` queue and lets ``_loop_hook_after``
        call ``_process_pending_check`` at the flush point.
        """
        raise NotImplementedError

    def _process_pending_check(
        self,
        ctx: LoopContext,
        pending: PendingCheck,
        iter_constraints: list[BoolRef],
    ) -> None:
        """Handle a single pending check when a loop is flushed.

        ``ctx`` is passed in even though the two current consumers don't need
        it — reserving it keeps the base signature stable for Step 2
        loop-aware reasoning (outer-loop introspection, nested-loop
        diagnostics, etc.).
        """
        raise NotImplementedError

    # ── Overridable hooks with safe defaults ──────────────────────────

    def _addr_ok_expr(self) -> BoolRef:
        addr = self.addr_sym
        assert addr is not None
        if self.addr_ok is None:
            addr_ok_expr = (
                Or(*[And(addr >= s, addr <= e) for s, e, _ in self.tensor_addrs])
                if self.tensor_addrs
                else BoolVal(False)
            )
            self.addr_ok = cast(BoolRef, addr_ok_expr)
        return self.addr_ok

    def _addr_ok_premise(self) -> BoolRef:
        """Constraint added to the solver under ``addr_ok``.

        Race detector keeps the positive premise (every captured access sits
        inside a registered tensor). Sanitizer can defer address-range
        premises until each access check when it needs access-specific ranges.
        """
        return self._addr_ok_expr()

    def _cache_non_tensor_arg(self, name: str, arg: Any) -> None:
        """Called from ``arg_callback`` for args without a ``data_ptr``."""
        pass

    def _cache_tensor_arg(self, arg: Tensor) -> None:
        """Called from ``arg_callback`` after a tensor is registered."""
        pass

    def _tensor_physical_addresses(
        self, name: str, arg: Tensor
    ) -> list[tuple[int, int, Tensor]]:
        if arg.is_contiguous() or check_storage_contiguous(arg):
            start = arg.data_ptr()
            end = arg.data_ptr() + (arg.numel() - 1) * arg.element_size()
            return [(start, end, arg)]

        if check_inner_stride_equal_to_one(arg):
            return [
                (start, end, arg)
                for start, end in get_physical_addr_from_tensor_slice(arg)
            ]

        threshold = cfg.symbolic_per_element_warn_threshold
        if threshold > 0 and arg.numel() > threshold:
            warnings.warn(
                f"Tensor {name!r} has {arg.numel()} elements with "
                "non-unit inner stride; per-element enumeration may "
                "slow the symbolic solver significantly.",
                stacklevel=2,
            )
        return [(start, end, arg) for start, end in get_physical_addr_per_element(arg)]

    def _clear_cache(self) -> None:
        """Extra cache state cleared at the end of a launch, by
        ``post_run_callback`` or by ``finalize``."""
        pass

    def _clear_symbolic_launch_state(self) -> None:
        """Launch-scoped state cleared under the shared guard."""
        self.tensors.clear()
        self.tensor_addrs.clear()
        self.tensor_names.clear()
        self.addr_ok_cache.clear()
        self.access_check_cache.clear()

    # ── Client callbacks with shared defaults ─────────────────────────

    def grid_idx_callback(self, grid_idx: tuple[int, ...]) -> None:
        self.grid_idx = grid_idx

    def finalize(self) -> list:
        # A launch whose last block never reaches post_run_callback (e.g.
        # every block was vetoed) would otherwise keep its launch state.
        self._clear_cache()
        self._clear_symbolic_launch_state()
        return []

    def arg_callback(self, name: str, arg: Any, arg_cvt: Any) -> None:
        if isinstance(arg, (tuple, list)):
            for idx, item in enumerate(arg):
                self.arg_callback(f"{name}[{idx}]", item, None)
            self._cache_non_tensor_arg(name, tuple(type(item).__name__ for item in arg))
            return
        if hasattr(arg, "base") and hasattr(arg.base, "data_ptr"):
            arg = arg.base
        if not hasattr(arg, "data_ptr"):
            self._cache_non_tensor_arg(name, arg)
            return
        tensor_physical_addresses = self._tensor_physical_addresses(name, arg)
        self._record_tensor_name(arg, name)
        self._cache_tensor_arg(arg)
        self.tensors.append(arg)
        self.tensor_addrs.extend(tensor_physical_addresses)

    def grid_callback(self, grid: tuple[int, ...]) -> None:
        grid = tuple(int(g) for g in grid)
        self.last_grid = (grid[0] - 1, grid[1] - 1, grid[2] - 1)
        self.grid = grid
        self._active_blocks = 0
        self._launch_should_stop = False
        self._pending_launch_clear = False
        self.addr_ok = None
        self.pid_ok = cast(
            BoolRef,
            And(
                SymbolicExpr.PID0 < grid[0],
                SymbolicExpr.PID1 < grid[1],
                SymbolicExpr.PID2 < grid[2],
                SymbolicExpr.PID0 >= 0,
                SymbolicExpr.PID1 >= 0,
                SymbolicExpr.PID2 >= 0,
            ),
        )

        if self.solver is None:
            self.solver = Solver()
        else:
            self.solver.reset()
        self.solver.add(self._addr_ok_premise())
        self.solver.add(self.pid_ok)

    def pre_run_callback(self, fn: Callable) -> bool:
        if self._launch_should_stop:
            return False
        return True if self.need_full_grid is None else self.need_full_grid

    def block_start_callback(self, fn: Callable) -> None:
        self._active_blocks += 1

    def post_run_callback(self, fn: Callable) -> bool:
        if self.need_full_grid is None:
            self.need_full_grid = False
        if self.grid_idx == self.last_grid or not self.need_full_grid:
            # Under concurrent block execution, another block may already be
            # running in this launch. Defer clearing tensor/solver caches until
            # every block that started has reached post_run.
            self._launch_should_stop = True
            self._pending_launch_clear = True
        self._active_blocks = max(0, self._active_blocks - 1)
        if self._pending_launch_clear and self._active_blocks == 0:
            self._clear_cache()
            self._clear_symbolic_launch_state()
            self._pending_launch_clear = False
        ret = self.need_full_grid
        self.need_full_grid = None
        return ret


class NullSymbolicClient:
    """Shared no-op body for ``Null*`` variants of symbolic clients.

    Used by ``NullRaceDetector`` / ``NullSanitizer`` (mixed in ahead of the
    factory class) to satisfy Client's ``@abstractmethod`` contract without
    providing any real behavior. Every callback raises loudly so misuse is
    obvious; ``__getattr__`` catches anything we forget to list explicitly.

    Not a ``Client`` subclass on its own — always combined with a factory
    (``NullRaceDetector(NullSymbolicClient, RaceDetector)``), so the MRO
    puts the no-op impls ahead of the factory's abstract stubs.
    """

    def _disabled(self, method: str) -> NoReturn:
        raise RuntimeError(
            f"[{type(self).__name__}] '{method}' was called, "
            "but the feature is off; no functionality is available."
        )

    def arg_callback(self, name: str, arg: Any, arg_cvt: Any) -> None:
        self._disabled("arg_callback")

    def finalize(self) -> list:
        self._disabled("finalize")

    def grid_callback(self, grid: tuple[int, ...]) -> None:
        self._disabled("grid_callback")

    def grid_idx_callback(self, grid_idx: tuple[int, ...]) -> None:
        self._disabled("grid_idx_callback")

    def register_op_callback(
        self, op_type: type[Op], *args: Any, **kwargs: Any
    ) -> OpCallbacks:
        self._disabled("register_op_callback")

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        self._disabled("register_for_loop_callback")

    def __getattr__(self, name: str) -> Any:
        self._disabled(name)
