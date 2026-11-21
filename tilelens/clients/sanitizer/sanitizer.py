from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import (
    Any,
    ClassVar,
    TypeVar,
    cast,
)
import sys

import numpy as np
from torch import Tensor
from triton.tools.tensor_descriptor import TensorDescriptor
from z3 import And, BoolVal, Not, Or, sat, unknown
from z3.z3 import BoolRef, IntNumRef

from ...core.client import Client
from ...core.callbacks import OpCallbacks, ForLoopCallbacks
from ...core.data import (
    Op,
    Allocate,
    Load,
    Store,
)
from ..symbolic_engine import (
    SymbolicExpr,
    SymbolicClient,
    NullSymbolicClient,
    PendingCheck,
    LoopContext,
    Z3Expr,
    ConstraintConjunction,
    AccessMode,
    symbolic_tensor_descriptor_value,
)
from .data import OutOfBoundsRecordZ3, UndecidedAccessRecordZ3
from ...utils.traceback_utils import (
    extract_user_frames,
    capture_current_source_location,
    location_to_traceback_info,
)
from .report import (
    print_oob_record,
    print_oob_record_pdb_style,
    print_undecided_record,
)
from .range_summary import access_range_proves_in_bounds
from ...core.config import config as cfg
from ...core.symbolic_metadata import (
    ConcreteAccessRoute,
    ConcreteRangeInfo,
    SymbolicTensorValue,
)

SanitizerT = TypeVar("SanitizerT", bound="Sanitizer")


class Sanitizer(Client):
    """
    Factory class that returns the concrete sanitizer implementation
    based on the value of ``cfg.enable_sanitizer``.
    """

    NAME = "sanitizer"
    LOG_TAG: ClassVar[str] = "Sanitizer"
    LOG_VERB: ClassVar[str] = "checking"

    def __new__(cls: type[SanitizerT], *args: Any, **kwargs: Any) -> SanitizerT:
        if cls is Sanitizer:
            target_cls = cast(
                type["Sanitizer"],
                SymbolicSanitizer if cfg.enable_sanitizer else NullSanitizer,
            )
            obj = object.__new__(target_cls)
            cast(Any, target_cls).__init__(obj, *args, **kwargs)
            return cast(SanitizerT, obj)
        return cast(SanitizerT, object.__new__(cls))

    def __init__(self, abort_on_error: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.abort_on_error: bool = abort_on_error

    def pre_run_callback(self, fn: Callable) -> bool:
        return True

    def post_run_callback(self, fn: Callable) -> bool:
        return True

    def pre_warmup_callback(self, jit_fn: Callable, *args: Any, **kwargs: Any) -> bool:
        return False

    def post_warmup_callback(self, jit_fn: Callable, ret: Any) -> None:
        pass

    def arg_callback(self, name: str, arg: Any, arg_cvt: Any) -> None:
        raise NotImplementedError

    def finalize(self) -> list:
        raise NotImplementedError

    def grid_callback(self, grid: tuple[int, ...]) -> None:
        raise NotImplementedError

    def grid_idx_callback(self, grid_idx: tuple[int, ...]) -> None:
        raise NotImplementedError

    def register_op_callback(
        self, op_type: type[Op], *args: Any, **kwargs: Any
    ) -> OpCallbacks:
        raise NotImplementedError

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        raise NotImplementedError


def _make_signature(
    addr_expr: Z3Expr,
    constraints: ConstraintConjunction,
) -> int:
    """
    Convert (addr, constraints) into a stable string signature.
    • addr_expr can be a single z3 expr or list[expr]
    • constraints is a conjunction of expr
    """
    if isinstance(addr_expr, list):
        if len(addr_expr) == 1:
            addr_hash = hash(addr_expr[0])
        else:
            # Order-stable hashing avoids O(n log n) sorting in hot paths.
            addr_hash = hash(tuple(hash(e) for e in addr_expr))
    else:
        addr_hash = hash(addr_expr)

    constr_hash = 0 if constraints is None else hash(constraints)

    return hash((addr_hash, constr_hash))


@dataclass(frozen=True)
class _FnSymbolicCache:
    fn: Callable
    grid: tuple[int, ...]
    args: tuple

    @cached_property
    def hash_value(self) -> int:
        return hash((self.fn, self.grid, self.args))

    def __hash__(self) -> int:
        return self.hash_value


_fn_symbolic_cache_set: set[_FnSymbolicCache] = set()


class SymbolicSanitizer(Sanitizer, SymbolicClient):
    def __init__(self, abort_on_error: bool = True):
        super().__init__(abort_on_error=abort_on_error)
        self.records: list[OutOfBoundsRecordZ3 | UndecidedAccessRecordZ3] = []
        self.cache_args: list[Any] = []
        self.cache_grid: tuple[int, ...] | None = None

    # Explicit forwarders to SymbolicClient: the Sanitizer factory
    # carries concrete stubs (NotImplementedError or ``return True``) to
    # satisfy Client's @abstractmethod contract, and those stubs would
    # otherwise shadow SymbolicClient's impls in the subclass MRO.
    def grid_idx_callback(self, grid_idx: tuple[int, ...]) -> None:
        SymbolicClient.grid_idx_callback(self, grid_idx)

    def finalize(self) -> list:
        return SymbolicClient.finalize(self)

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        return SymbolicClient.register_for_loop_callback(self)

    def arg_callback(self, name: str, arg: Any, arg_cvt: Any) -> None:
        SymbolicClient.arg_callback(self, name, arg, arg_cvt)

    def register_op_callback(self, op_type: type[Op]) -> OpCallbacks:
        if op_type is Allocate:
            cached = self.op_callback_cache.get(op_type)
            if cached is None:
                cached = OpCallbacks(
                    after_callback=self.lock_fn(self._op_allocate_after_callback)
                )
                self.op_callback_cache[op_type] = cached
            return cached
        return SymbolicClient.register_op_callback(self, op_type)

    def post_run_callback(self, fn: Callable) -> bool:
        return SymbolicClient.post_run_callback(self, fn)

    # ── Sanitizer-specific hook overrides ─────────────────────────

    def _op_allocate_after_callback(
        self,
        ret: Any,
        base: Any,
        shape: Any,
        strides: Any,
        block_shape: Any,
    ) -> None:
        symbolic_tensor_descriptor_value(
            ret,
            base=base,
            shape=shape,
            strides=strides,
            block_shape=block_shape,
        )

    def _addr_ok_premise(self) -> BoolRef:
        # Sanitizer adds the actual invalid-address premise under the
        # per-access push/pop scope, because it may need tensor-specific
        # ranges instead of the global union of all registered tensors.
        return BoolVal(True)

    def _cache_non_tensor_arg(self, name: str, arg: Any) -> None:
        # TODO: init a reserved_args field per frontend to filter out these args
        if name not in ["num_warps", "num_stages", "maxnreg", "num_ctas"]:
            if isinstance(arg, TensorDescriptor):
                self.cache_args.append(
                    (
                        "TensorDescriptor",
                        repr(arg.base),
                        tuple(arg.shape),
                        tuple(arg.strides),
                        tuple(arg.block_shape),
                        arg.padding,
                    )
                )
                return
            self.cache_args.append(arg)

    def _cache_tensor_arg(self, arg: Tensor) -> None:
        self.cache_args.append((arg.shape, arg.stride(), arg.dtype))

    def _clear_cache(self) -> None:
        self.cache_args.clear()
        self.cache_grid = None

    def grid_callback(self, grid: tuple[int, ...]) -> None:
        # Sanitizer needs ``cache_grid`` tracked alongside the shared setup so
        # ``pre_run_callback`` can consult it before deciding whether to skip
        # the launch.
        self.cache_grid = tuple(int(g) for g in grid)
        SymbolicClient.grid_callback(self, grid)

    def _op_load_overrider(self, ptr, mask, other, *_args):
        ptr = self._materialize_memory_operand(ptr)
        mask = self._materialize_memory_operand(mask)
        ptr_sym = self._symbolic_memory_ptr(ptr)
        mask_sym = SymbolicExpr.from_value(mask) if mask is not None else None
        other_sym = SymbolicExpr.from_value(other) if other is not None else None
        ret = SymbolicExpr.create("load", ptr_sym, mask_sym, other_sym)

        self._route_pointer_access(ret, ptr_sym, mask_sym, Load, "read")
        return ret

    def _op_store_overrider(self, ptr, _value, mask, *_args):
        ptr = self._materialize_memory_operand(ptr)
        mask = self._materialize_memory_operand(mask)
        ptr_sym = self._symbolic_memory_ptr(ptr)
        mask_sym = SymbolicExpr.from_value(mask) if mask is not None else None
        # Store values do not affect address validity. Avoiding the value
        # subtree keeps sanitizer store checks focused on pointer + mask.
        ret = SymbolicExpr.create("store", ptr_sym, None, mask_sym)

        self._route_pointer_access(ret, ptr_sym, mask_sym, Store, "write")
        return ret

    def _route_pointer_access(
        self,
        access_expr: SymbolicExpr,
        ptr_sym: SymbolicExpr,
        mask_sym: SymbolicExpr | None,
        op_type: type[Op],
        access_mode: AccessMode,
    ) -> None:
        concrete_route = self._concrete_ptr_access_route(access_expr, ptr_sym, mask_sym)
        if concrete_route is not None:
            self._check_concrete_addr_values(
                concrete_route.addrs,
                access_expr,
                None,
                concrete_route.range_info,
            )
            return

        self._handle_access_check(access_expr, op_type, access_mode)

    def _concrete_ptr_access_route(
        self,
        access_expr: SymbolicExpr,
        ptr_sym: SymbolicExpr,
        mask_sym: SymbolicExpr | None,
    ) -> ConcreteAccessRoute | None:
        if self.loop_stack:
            return None

        concrete_addrs = self._concrete_ptr_addrs(ptr_sym, mask_sym)
        if concrete_addrs is None:
            return None

        range_info = self._concrete_addr_range_info(access_expr)
        if range_info is None:
            return None
        return ConcreteAccessRoute(concrete_addrs, range_info)

    def _concrete_ptr_addrs(
        self, ptr_sym: SymbolicExpr, mask_sym: SymbolicExpr | None
    ) -> list[int] | None:
        if ptr_sym.op != "const":
            return None
        ptr_value = getattr(ptr_sym, "value", None)
        if not isinstance(ptr_value, SymbolicTensorValue):
            return None

        addrs = np.asarray(ptr_value.data, dtype=np.uint64).reshape(-1)
        if mask_sym is None:
            return [int(addr) for addr in addrs]

        if mask_sym.op != "const":
            return None
        mask_value = getattr(mask_sym, "value", None)
        if isinstance(mask_value, SymbolicTensorValue):
            mask = np.asarray(mask_value.data, dtype=bool)
        elif isinstance(mask_value, (bool, int, np.integer)):
            mask = np.asarray(mask_value, dtype=bool)
        else:
            return None

        try:
            mask = np.broadcast_to(mask, ptr_value.data.shape).reshape(-1)
        except ValueError:
            return None
        return [int(addr) for addr, active in zip(addrs, mask) if active]

    def _concrete_addr_range_info(
        self, symbolic_expr: SymbolicExpr
    ) -> ConcreteRangeInfo | None:
        tensor = self._resolve_tensor(symbolic_expr)
        if tensor is None:
            ranges = [(start, end, tensor) for start, end, tensor in self.tensor_addrs]
        else:
            ranges = [
                (start, end, range_tensor)
                for start, end, range_tensor in self.tensor_addrs
                if range_tensor is tensor
            ]
        if not ranges:
            return None
        return ConcreteRangeInfo(tensor, ranges)

    def pre_run_callback(self, fn: Callable) -> bool:
        if self.cache_grid:
            fn_cache = _FnSymbolicCache(fn, self.cache_grid, tuple(self.cache_args))
            self._clear_cache()
            if fn_cache not in _fn_symbolic_cache_set:
                _fn_symbolic_cache_set.add(fn_cache)
                return SymbolicClient.pre_run_callback(self, fn)
            return False
        return SymbolicClient.pre_run_callback(self, fn)

    def _find_tensor_for_expr(
        self, symbolic_expr: SymbolicExpr, violation_addr: int
    ) -> Tensor:
        # Prefer mapping from pointer base addresses present in the expression.
        resolved = self._resolve_tensor(symbolic_expr)
        if resolved is not None:
            return resolved

        # Fall back to the closest registered segment.
        if self.tensor_addrs:

            def _distance(seg: tuple[int, int, Tensor]) -> int:
                start, end, _tensor = seg
                if violation_addr < start:
                    return start - violation_addr
                if violation_addr > end:
                    return violation_addr - end
                return 0

            return min(self.tensor_addrs, key=_distance)[2]

        # Fall back to the first registered tensor.
        if self.tensors:
            return self.tensors[0]

        raise RuntimeError("No tensor registered in SymbolicSanitizer!")

    def _addr_ok_for_expr(self, symbolic_expr: SymbolicExpr) -> BoolRef:
        """Return the valid-address predicate for one memory access.

        Prefer the ranges belonging to the tensor that the pointer expression
        originates from. The global ``addr_ok`` union is only a fallback for
        expressions whose base tensor cannot be resolved.
        """
        ptr = cast(Any, symbolic_expr).ptr
        if ptr.op == "descriptor_access":
            # Descriptor/TMA accesses need descriptor-coordinate bounds, not
            # just the final pointer address range inside the backing tensor.
            return ptr.addr_ok

        resolved_tensor = self._resolve_tensor(symbolic_expr)
        if resolved_tensor is None:
            return self._addr_ok_expr()

        cache_key = id(resolved_tensor)
        cached = self.addr_ok_cache.get(cache_key)
        if cached is not None:
            return cached

        addr_sym = self.addr_sym
        assert addr_sym is not None
        ranges = [
            And(addr_sym >= start, addr_sym <= end)
            for start, end, tensor in self.tensor_addrs
            if tensor is resolved_tensor
        ]
        if not ranges:
            return BoolVal(False)
        addr_ok = ranges[0] if len(ranges) == 1 else cast(BoolRef, Or(*ranges))
        self.addr_ok_cache[cache_key] = addr_ok
        return addr_ok

    def _check_range_satisfiable(
        self,
        access_addr: Z3Expr,
        expr_constraints: ConstraintConjunction,
        symbolic_expr: SymbolicExpr,
        source_location: tuple[str, int, str] | None = None,
    ) -> None:
        # Use push/pop on persistent solver
        solver = self.solver
        addr_sym = self.addr_sym
        assert solver is not None
        assert addr_sym is not None
        addr_ok = self._addr_ok_for_expr(symbolic_expr)

        def _report_if_sat() -> None:
            status = solver.check()
            if status == unknown:
                # Not a proof: the access went unchecked.
                self._report_undecided(
                    symbolic_expr, solver.reason_unknown(), source_location
                )
                return
            if status != sat:
                return

            model = solver.model()
            violation_val = model.evaluate(addr_sym, model_completion=True)
            if isinstance(violation_val, IntNumRef):
                violation_addr = violation_val.as_long()
            else:
                raise RuntimeError("Unexpected violation address type from Z3 model!")

            tensor = self._find_tensor_for_expr(symbolic_expr, violation_addr)
            if symbolic_expr.op in ("store", "tensor_pointer_store"):
                op_type: type[Load] | type[Store] = Store
            else:
                op_type = Load

            self._report(
                op_type, tensor, violation_addr, symbolic_expr, source_location
            )

        solver.push()
        try:
            solver.add(Not(addr_ok))
            if expr_constraints is not None:
                solver.add(expr_constraints)

            if isinstance(access_addr, list):
                if not access_addr:
                    return
                solver.add(Or(*(addr_sym == addr for addr in access_addr)))
                _report_if_sat()
                return

            solver.add(addr_sym == access_addr)
            _report_if_sat()
        finally:
            solver.pop()

    def _check_concrete_addr_values(
        self,
        concrete_addrs: list[int],
        symbolic_expr: SymbolicExpr,
        source_location: tuple[str, int, str] | None,
        range_info: ConcreteRangeInfo,
    ) -> bool:
        if not concrete_addrs:
            return True

        min_addr = min(concrete_addrs)
        max_addr = max(concrete_addrs)
        tensor = range_info.tensor
        ranges = range_info.ranges

        if len(ranges) == 1:
            start, end, _ = ranges[0]
            if min_addr >= start and max_addr <= end:
                return True
            violation_addr = next(
                addr for addr in concrete_addrs if addr < start or addr > end
            )
            report_tensor = (
                tensor
                if tensor is not None
                else self._find_tensor_for_expr(symbolic_expr, violation_addr)
            )
            op_type: type[Load] | type[Store]
            if symbolic_expr.op in ("store", "tensor_pointer_store"):
                op_type = Store
            else:
                op_type = Load
            self._report(
                op_type,
                report_tensor,
                violation_addr,
                symbolic_expr,
                source_location,
            )
            return True

        for addr in concrete_addrs:
            if any(start <= addr <= end for start, end, _ in ranges):
                continue
            report_tensor = (
                tensor
                if tensor is not None
                else self._find_tensor_for_expr(symbolic_expr, addr)
            )
            if symbolic_expr.op in ("store", "tensor_pointer_store"):
                op_type = Store
            else:
                op_type = Load
            self._report(op_type, report_tensor, addr, symbolic_expr, source_location)
            return True

        return True

    def _range_summary_proves_safe(self, symbolic_expr: SymbolicExpr) -> bool:
        resolved_tensor = self._resolve_tensor(symbolic_expr)
        if resolved_tensor is None:
            ranges = self.tensor_addrs
        else:
            ranges = [
                (start, end, tensor)
                for start, end, tensor in self.tensor_addrs
                if tensor is resolved_tensor
            ]
        if not ranges:
            return False
        return access_range_proves_in_bounds(symbolic_expr, ranges, self.grid)

    def _handle_access_check(
        self,
        expr: SymbolicExpr,
        op_type: type[Op],
        access_mode: AccessMode,
    ) -> None:
        """Evaluate a memory access and either check now or defer inside a loop.

        ``op_type`` and ``access_mode`` are accepted to match the base-class
        signature; sanitizer derives op_type from ``expr.op`` inside
        ``_check_range_satisfiable`` for historical compatibility, and it
        doesn't distinguish reads vs writes at the Z3 level.
        """
        if not self.loop_stack:
            expr_signature = hash(("expr", expr.signature()))
            if expr_signature in self.access_check_cache:
                return
            z3_addr, z3_constraints = expr.eval()
            z3_signature = hash(("z3", _make_signature(z3_addr, z3_constraints)))
            if z3_signature in self.access_check_cache:
                self.access_check_cache.add(expr_signature)
                return
            self.access_check_cache.add(expr_signature)
            self.access_check_cache.add(z3_signature)
            self._check_range_satisfiable(z3_addr, z3_constraints, expr)
            return

        z3_addr, z3_constraints = expr.eval()
        ctx = self.loop_stack[-1]
        signature = _make_signature(z3_addr, z3_constraints)
        pending_idx = ctx.signature_cache.get(signature)
        if pending_idx is None:
            # Capture source location now while we're still in the user's tl.load/tl.store call.
            # This is a lightweight operation that only traverses frame objects.
            # The actual source line will be read later only if an error is detected.
            source_location = capture_current_source_location()
            ctx.signature_cache[signature] = len(ctx.pending_checks)
            pending_check = PendingCheck(
                symbolic_expr=expr,
                addr_expr=z3_addr,
                constraints=z3_constraints,
                source_location=source_location,
            )
            ctx.pending_checks.append(pending_check)
        else:
            if cfg.verbose:
                print(f"[{self.LOG_TAG}]  ↪ skip duplicated addr in loop")

    def _process_pending_check(
        self,
        ctx: LoopContext,
        pending: PendingCheck,
        iter_constraints: list[BoolRef],
    ) -> None:
        if self._range_summary_proves_safe(pending.symbolic_expr):
            return
        self._check_range_satisfiable(
            pending.addr_expr,
            pending.constraints,
            pending.symbolic_expr,
            pending.source_location,
        )

    def _report(
        self,
        op_type: type[Load] | type[Store],
        tensor: Tensor,
        violation_address: int,
        symbolic_expr: SymbolicExpr | None = None,
        source_location: tuple[str, int, str] | None = None,
    ) -> None:
        # Use pre-captured location if available (for deferred checks in loops),
        # otherwise capture it now (for immediate checks outside loops)
        if source_location is not None:
            traceback_info = [location_to_traceback_info(source_location)]
        else:
            traceback_info = extract_user_frames()

        tensor_name = self._get_tensor_name(tensor)
        oob_record = OutOfBoundsRecordZ3(
            op_type=op_type,
            user_code_tracebacks=traceback_info,
            tensor=tensor,
            violation_address=violation_address,
            constraints=None,
            symbolic_expr=symbolic_expr,
            tensor_name=tensor_name,
        )
        if self.abort_on_error:
            # Use the new PDB-style print function if available
            if symbolic_expr is not None or cfg.verbose:
                print_oob_record_pdb_style(oob_record, symbolic_expr)
            else:
                print_oob_record(oob_record)
            sys.exit(1)
        else:
            self.records.append(oob_record)

    def _report_undecided(
        self,
        symbolic_expr: SymbolicExpr,
        reason: str,
        source_location: tuple[str, int, str] | None = None,
    ) -> None:
        # Z3 swallows a Ctrl+C during check() and answers unknown.
        if reason == "interrupted from keyboard":
            raise KeyboardInterrupt(f"Z3 query cancelled ({reason})")

        if source_location is not None:
            traceback_info = [location_to_traceback_info(source_location)]
        else:
            traceback_info = extract_user_frames()

        tensor = self._resolve_tensor(symbolic_expr)
        if symbolic_expr.op in ("store", "tensor_pointer_store"):
            op_type: type[Load] | type[Store] = Store
        else:
            op_type = Load
        record = UndecidedAccessRecordZ3(
            op_type=op_type,
            tensor=tensor,
            user_code_tracebacks=traceback_info,
            reason=reason,
            symbolic_expr=symbolic_expr,
            tensor_name=None if tensor is None else self._get_tensor_name(tensor),
        )
        # Not a finding, so the program runs on even with abort_on_error.
        if self.abort_on_error:
            print_undecided_record(record)
        else:
            self.records.append(record)


class NullSanitizer(NullSymbolicClient, Sanitizer):
    """
    A do-nothing object returned when the sanitizer is off.
    Every callback raises via ``NullSymbolicClient`` so misuse is obvious.
    """

    def __init__(self, abort_on_error: bool = True, *args: Any, **kwargs: Any):
        super().__init__(abort_on_error=abort_on_error)
