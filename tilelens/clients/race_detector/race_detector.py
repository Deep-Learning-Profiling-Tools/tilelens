from bisect import bisect_left
import threading
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    Any,
    ClassVar,
    TypeVar,
    cast,
)

import torch
from z3 import (
    And,
    If,
    Implies,
    Int,
    IntSort,
    IntVal,
    K,
    Or,
    Select,
    Store as Z3ArrayStore,
)
from z3.z3 import BoolRef

from ...core.client import Client
from ...core.callbacks import OpCallbacks, ForLoopCallbacks
from ...core.data import (
    Op,
    AtomicCas,
    AtomicRMW,
    Barrier,
    InlineAsm,
    InterpreterDivergence,
    Load,
    Store,
)
from ...ir.verdict import IRVerdict, Refusal, SourceLocation
from ..symbolic_engine import (
    SymbolicExpr,
    ArangeSymbolicExpr,
    BroadcastSymbolicExpr,
    ExpandDimsSymbolicExpr,
    LoadSymbolicExpr,
    AtomicCasSymbolicExpr,
    AtomicRmwSymbolicExpr,
    SymbolicClient,
    NullSymbolicClient,
    PendingCheck,
    LoopContext,
    Z3Expr,
    ConstraintConjunction,
    AccessMode,
    _and_constraints,
    _constraint_to_bool,
)
from .data import (
    AccessEventRecord,
    MemorySem,
    RaceAccess,
    RaceRecord,
    RaceReport,
    RaceType,
)
from .hb_common import (
    UnsupportedSymbolicRaceQuery,
    apply_sub,
    normalize_copy_local_vars,
)
from .two_copy_symbolic_hb_solver import TwoCopySymbolicHBSolver
from ...utils.traceback_utils import (
    capture_current_source_location,
    location_to_traceback_info,
    scalar_truthiness_from_user_code,
)
from ...core.config import config as cfg

RaceDetectorT = TypeVar("RaceDetectorT", bound="RaceDetector")


def _hash_signature_part(x: Any) -> int:
    """Hash one component of an event signature, tolerating Z3 expressions,
    nested lists/tuples, and unhashable values."""
    if isinstance(x, (list, tuple)):
        return hash(tuple(_hash_signature_part(v) for v in x))
    if x is None:
        return 0
    try:
        return hash(x)
    except TypeError:
        return hash(repr(x))


def _make_event_signature(
    access_mode: AccessMode,
    source_location: tuple[str, int, str] | None,
    addr_expr: Z3Expr,
    local_constraints: ConstraintConjunction,
    active_expr: Any = True,
) -> int:
    """Signature used to dedupe repeated access events within a single loop.

    Distinct from sanitizer's ``_make_signature``: ``access_mode`` and
    ``source_location`` are part of the key so a ``load`` and a ``store`` at
    the same address inside the same loop stay as separate events (different
    program-order nodes for the HB analysis). ``active_expr`` (the mask) is
    part of the key so two accesses with the same address but different
    masks stay apart.
    """
    return hash(
        (
            access_mode,
            source_location,
            _hash_signature_part(addr_expr),
            _hash_signature_part(local_constraints),
            _hash_signature_part(active_expr),
        )
    )


@dataclass
class PendingEvent(PendingCheck):
    """Extends ``PendingCheck`` with access-mode and op_type metadata.

    ``LoopContext.pending_checks`` is typed as ``list[PendingCheck]`` but is
    fine accepting subclass instances at runtime. Keeping the extension local
    to this module avoids widening the shared schema used by sanitizer.
    """

    access_mode: AccessMode = "read"
    op_type: type[Op] = Load
    active: Any = True
    # Ordering reserved when the access is captured, alongside barriers:
    # the loop flush can happen much later (and inner loops flush first).
    program_seq: int | None = None
    # Every capture of this access, deduplicated ones included: (its
    # program_seq, its _loop_key()).
    hits: list[tuple[int, tuple[tuple[Any, int | None], ...]]] = field(
        default_factory=list
    )
    lane_position: tuple[Any, ...] | None = None
    # Which execution of its signature within one loop iteration this is.
    occurrence: int = 0


class RaceDetector(Client):
    """Factory class that returns the concrete race-detector implementation:
    :class:`NullRaceDetector` when ``cfg.enable_race_detector`` is off,
    otherwise :class:`SymbolicRaceDetector`, the eager detector, which
    reasons about global memory across program instances from one
    interpreter-driven symbolic capture.
    """

    NAME = "race_detector"
    LOG_TAG: ClassVar[str] = "RaceDetector"
    LOG_VERB: ClassVar[str] = "recording"

    def __new__(cls: type[RaceDetectorT], *args: Any, **kwargs: Any) -> RaceDetectorT:
        if cls is RaceDetector:
            # The disable flag wins over the mode, as for the Sanitizer:
            # RaceDetector(compile=True) is a NullRaceDetector when it is off.
            target_cls = cast(
                type["RaceDetector"],
                SymbolicRaceDetector if cfg.enable_race_detector else NullRaceDetector,
            )
            # A RaceDetector subclass: Python calls its __init__ once this
            # returns, with the call's own arguments, ``compile`` included.
            return cast(RaceDetectorT, object.__new__(target_cls))
        return cast(RaceDetectorT, object.__new__(cls))

    # The public detector interface, populated by the concrete subclasses.
    #
    # ``last_status`` values:
    #   "ok"          — the solver ran (last_reports holds its races)
    #   "unsupported" — the launch uses something the detector does not model
    #                   (see unsupported_reason / unsupported_kind)
    #   "vacuous"     — no race query is satisfiable, but neither is the
    #                   launch's base system: no proof is claimed
    #   "aborted"     — no verdict exists: set at construction and at every
    #                   launch's begin_launch/grid_callback; only finalize()
    #                   upgrades it, so a launch that dies before finalize
    #                   can never be read as a clean "ok"
    #   "disabled"    — race detector backend is off (NullRaceDetector)
    last_reports: list[RaceRecord]
    last_status: str
    unsupported_reason: str | None

    def __init__(self, abort_on_error: bool = False, *args: Any, **kwargs: Any) -> None:
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


class _UnsupportedRMWReturn(SymbolicExpr):
    """Sentinel SymbolicExpr returned by ``SymbolicRaceDetector`` for a
    FLOAT-TYPED (or unknown-dtype) atomic-RMW result — integer RMW returns
    are value-modeled and return the event expression instead. Outside the
    integer model, the return's symbolic semantics are not modeled; if a
    kernel consumes it downstream (e.g. ``mask = old == 0.0``), the eventual
    ``_to_z3_impl`` call raises :class:`UnsupportedSymbolicRaceQuery`, which
    the wrapping ``_safe_eval`` in ``_handle_*_check`` converts into a clean
    ``_mark_unsupported`` so the launch finishes without raising.

    The op string MUST be ``"atomic_rmw"`` because
    ``SymbolicExpr.__init__`` asserts ``op in self.SUPPORTED_OPS``;
    coining a new op here would assert-fail at construction time.
    """

    def __init__(self, *, dtype: Any = None, shape: Any = ()) -> None:
        super().__init__("atomic_rmw")
        self.dtype = dtype
        self.shape = shape

    def _to_z3_impl(self) -> tuple[Any, Any]:
        raise UnsupportedSymbolicRaceQuery(
            "atomic_rmw return value used downstream is not modeled"
        )


def _atomic_old_value(
    expr: SymbolicExpr,
) -> tuple[Z3Expr, ConstraintConjunction]:
    """The race detector's atomic-value provider: the old value an
    atomic_cas / atomic_rmw returns is per-program-instance nondeterminism,
    so it lowers to a fresh Int per lane. Var names derive from id(expr), so
    repeated evaluation yields the SAME Z3 vars (Z3 interns by name) and a
    record's old_value stays identical to every downstream use; the two-copy
    solver renames them per copy (copy_local_vars). Whether an RMW's
    observation is value-modeled at all (integer dtype) is decided by the
    overrider, which returns the sentinel for float-typed RMWs."""
    operands: tuple[SymbolicExpr | None, ...]
    if isinstance(expr, AtomicCasSymbolicExpr):
        operands = (expr.ptr, expr.cmp, expr.val)
    else:
        rmw = cast(AtomicRmwSymbolicExpr, expr)
        operands = (rmw.ptr, rmw.val, rmw.mask)
    ptr_z3, _ = cast(SymbolicExpr, operands[0])._to_z3()
    constraints = _and_constraints(
        *(operand._to_z3()[1] for operand in operands if operand is not None)
    )
    name = f"{expr.op}_old_{id(expr)}"
    if isinstance(ptr_z3, list):
        return [Int(f"{name}_{idx}") for idx in range(len(ptr_z3))], constraints
    return Int(name), constraints


def _witness_element(record: AccessEventRecord, addr: int) -> int | None:
    if record.tensor is None:
        return None
    return (addr - int(record.tensor.data_ptr())) // max(1, record.elem_size)


def _race_record(report: RaceReport) -> RaceRecord:
    race_type = report.race_type
    writes = (
        race_type in (RaceType.WAW, RaceType.RAW),
        race_type in (RaceType.WAW, RaceType.WAR),
    )
    assert report.witness_addrs is not None
    first, second = (
        RaceAccess(
            op=event.record.op_type.name,
            writes=event_writes,
            tensor_name=event.record.tensor_name,
            source_location=event.record.source_location,
            element=_witness_element(event.record, addr),
        )
        for event, event_writes, addr in zip(
            (report.first, report.second), writes, report.witness_addrs
        )
    )
    assert report.witness_grid_a is not None and report.witness_grid_b is not None
    return RaceRecord(
        race_type=race_type,
        first=first,
        second=second,
        programs=(report.witness_grid_a, report.witness_grid_b),
        reason=report.reason,
    )


def print_race_record(record: RaceRecord) -> None:
    """Print one race the detector found."""
    print("============================================================")
    print("                      Data Race Detected                    ")
    print("============================================================")
    print(f"Race Type: {record.race_type}")
    for label, access, program in (
        ("First", record.first, record.programs[0]),
        ("Second", record.second, record.programs[1]),
    ):
        mode = "write" if access.writes else "read"
        target = ""
        if access.tensor_name is not None:
            target = f" of {access.tensor_name}"
            if access.element is not None:
                target += f"[{access.element}]"
        print(f"{label}: {access.op} ({mode}){target}, program {program}")
        if access.source_location is not None:
            info = location_to_traceback_info(access.source_location)
            print(f"  File: {info.filename}, Line: {info.lineno}, in {info.func_name}")
            print(f"  Code: {info.line_of_code}")
    if record.reason:
        print(f"Reason: {record.reason}")
    print("------------------------------------------------------------")


class SymbolicRaceDetector(RaceDetector, SymbolicClient):
    # Upper bound on numel for an input tensor used as a tl.load value source.
    # Mirrors _MAX_INITIAL_ATOMIC_ELEMENTS in two_copy_symbolic_hb_solver.py;
    # larger sources are marked unsupported rather than blowing up the Z3
    # array snapshot.
    _MAX_LOAD_SOURCE_ELEMENTS: ClassVar[int] = 1024

    # ``compile`` is the factory's mode switch: RaceDetector(compile=False)
    # constructs this class with it.
    def __init__(self, abort_on_error: bool = False, *, compile: bool = False):
        if compile:
            raise NotImplementedError(
                "RaceDetector(compile=True): the compiled race detector is not "
                "available yet; RaceDetector() is the eager one."
            )
        super().__init__(abort_on_error=abort_on_error)
        self.records: list[AccessEventRecord] = []
        # Tile-level fences (tl.debug_barrier) seen by the capture: the
        # program_seq of each one executed, in the sequence the access
        # records use. Every capture of each record flushed from a loop, by
        # event_id: (program_seq, _loop_key()). _fence_between is the fence
        # rule over them.
        self.fences: list[int] = []
        self._record_hits: dict[
            int, list[tuple[int, tuple[tuple[Any, int | None], ...]]]
        ] = {}
        # Every range each loop site ran over (_fence_epoch).
        self._loop_ranges: dict[Any, set[tuple[int, int, int]]] = {}
        # id(load/atomic SymbolicExpr node) -> event_id, for dependency
        # order (data.py AccessEventRecord.dep_loads).
        self._dep_anchor_ids: dict[int, int] = {}
        self.last_reports: list[RaceRecord] = []
        # Status of the most recent finalize(); see RaceDetector. Initialized
        # pessimistically and re-armed to "aborted" at every launch's
        # begin_launch/grid_callback; only finalize() upgrades it, so neither
        # a never-launched detector nor a launch that dies mid-kernel can
        # ever be read as clean. last_reports being empty does NOT imply
        # "no race" unless last_status == "ok".
        self.last_status: str = "aborted"
        self._program_seq: int = 0
        self._event_seq: int = 0
        self._launch_grid: tuple[int, int, int] = (1, 1, 1)
        self._captured_symbolic_template: bool = False
        # One-shot capture slot: claimed atomically by the first admitted
        # block in pre_run_callback, sealed by that block's
        # post_run_callback. Blocks running outside the slot (need_full_grid
        # reruns for a co-attached client, sibling workers under
        # TILELENS_NUM_SMS >= 2) execute the kernel but must not mutate
        # the shared per-launch record state.
        self._capture_claimed: bool = False
        self._capture_thread_id: int | None = None
        self._unsupported_capture: bool = False
        self.unsupported_reason: str | None = None
        # The kind of the last launch's refusal ("unsupported", "timeout",
        # "unknown", "vacuous", "block-truthiness", "inline-asm"), None while
        # supported, and where in user code it arose when known.
        self.unsupported_kind: str | None = None
        self._unsupported_loc: tuple[str, int, str] | None = None
        self._arange_dict_snapshot: dict[Any, Any] = {}
        # Load-value modelling: per-launch Z3 array cache and bidirectional
        # region tracking. Keys are (base, elem_size, numel, str(dtype));
        # values are (z3_array_const, [IntVal(base+i*es) for i in range(n)]).
        self._load_array_cache: dict[
            tuple[int, int, int, str], tuple[Any, list[Any]]
        ] = {}
        self._load_value_regions: list[tuple[int, int, Any]] = []
        self._written_regions: list[tuple[int, int, Any]] = []
        # Set when a write event's target tensor cannot be resolved; the
        # load-value provider raises unsupported on subsequent loads because
        # the unknown write may alias a snapshotted load source.
        self._unknown_written_region_seen: bool = False
        # Finished-loop iterator substitutions, keyed by LoopSite:
        # (idx_z3, IntVal(final iteration value)). See _apply_finished_iter_subs
        # for why leftover iterator references must be concretized at record time.
        self._finished_loop_iter_subs: dict[Any, tuple[Any, Any]] = {}
        # Stash of the substitution entry popped when a loop re-enters,
        # restored on a zero-iteration exit (a zero-trip loop leaves the
        # leftover Python variable — and thus its final value — unchanged).
        self._suspended_iter_subs: list[tuple[Any, tuple[Any, Any] | None]] = []
        # Every iterator var created this launch, plus the ones whose final
        # value varied across activations under an active outer loop
        # (substituting any single constant for those would be wrong).
        self._known_iter_var_keys: set[tuple[int, str, str]] = set()
        self._unstable_iter_var_keys: set[tuple[int, str, str]] = set()
        # Launch-level signatures of records flushed from loops. A loop
        # nested under an outer loop re-enters once per outer iteration with
        # a fresh LoopContext, so its per-activation signature_cache cannot
        # dedupe across activations; without this cache every outer iteration
        # appends another structurally identical record. Maps to the
        # record's event_id, which collects the duplicates' hits.
        self._loop_flush_signatures: dict[int, int] = {}
        # Executions so far of each (access signature, _loop_key()).
        self._iteration_occurrences: dict[tuple[int, tuple], int] = {}

    # ── Unsupported-launch plumbing ──────────────────────────────────────

    def _mark_unsupported(self, reason: str, kind: str = "unsupported") -> None:
        """Mark the current launch as unsupported by the two-copy solver.
        The first refusal of a launch is the one reported.

        Discards any partial records so finalize() can't accidentally feed
        them to the solver. Callers MUST return after invoking this — kernel
        tracing may still execute, and the early-return guards in
        ``_handle_*_check`` / ``_record_*_event`` keep further events from
        leaking back into ``self.records``.
        """
        if self._unsupported_capture:
            return
        self._unsupported_loc = capture_current_source_location()
        self._unsupported_capture = True
        self.unsupported_reason = reason
        self.unsupported_kind = kind
        self.last_status = "unsupported"
        self.records = []

    @contextmanager
    def _load_value_semantics(self):
        """Install the ``tl.load`` value provider around one detector-
        triggered evaluation.

        The provider slot is class-global on ``SymbolicExpr``, so it is held
        only while the detector evaluates; engine-side evaluations outside
        that keep the default load lowering. Save/restore keeps nested
        detector evaluations re-entrant and exception-safe.
        """
        prev = SymbolicExpr._load_value_provider
        prev_owner = SymbolicExpr._load_value_provider_owner
        SymbolicExpr._load_value_provider = self._load_value_provider_impl
        SymbolicExpr._load_value_provider_owner = id(self)
        try:
            yield
        finally:
            SymbolicExpr._load_value_provider = prev
            SymbolicExpr._load_value_provider_owner = prev_owner

    def _safe_eval(self, expr: "SymbolicExpr", reason: str) -> tuple[Any, Any] | None:
        """Eval a SymbolicExpr under the detector's load-value semantics,
        marking the launch unsupported on
        :class:`UnsupportedSymbolicRaceQuery`. Returns ``None`` when
        unsupported so callers can ``if result is None: return``.

        ``NotImplementedError`` is how the symbolic engine signals a missing
        Z3 lowering (cumsum, dot, ...); it gets the same unsupported
        treatment so a lowering gap surfaces as a verdict instead of
        crashing the launch.
        """
        try:
            with self._load_value_semantics():
                return expr.eval()
        except UnsupportedSymbolicRaceQuery as exc:
            self._mark_unsupported(str(exc) or reason)
            return None
        except NotImplementedError as exc:
            self._mark_unsupported(f"{reason}: {exc}" if str(exc) else reason)
            return None

    @staticmethod
    def _combine_constraints(*constraints: Any) -> tuple[Any, ...]:
        """Flat tuple of non-None constraints; the two-copy solver's
        ``iter_constraints`` recursively flattens nested tuples/lists.
        """
        return tuple(c for c in constraints if c is not None)

    # Ops whose RESULT may not appear in an event ADDRESS. Plain tl.load is
    # deliberately absent: a plain load of a READ-ONLY tensor lowers to a
    # Select over the launch snapshot with its domain facts, which is exact
    # for this launch's tensor contents. Atomic returns are interleaving-
    # dependent, not snapshot-stable; sort/cumsum have no snapshot semantics.
    # Reduce results are gated because ReduceSymbolicExpr folds over a
    # SINGLE symbolic lane (an arange is one symbolic variable), so
    # tl.max(...) in an address degenerates to a solver-chosen element and
    # fabricates races nondeterministically; lift only with a true per-lane
    # fold.
    _VALUE_DEPENDENT_ADDRESS_OPS: ClassVar[tuple[str, ...]] = (
        "atomic_cas",
        "atomic_rmw",
        "sort",
        "cumsum",
        "sum",
        "max",
        "min",
        "xor_sum",
        "reduce_or",
        "argmax",
        "argmin",
    )

    @classmethod
    def _find_value_dependent_op(cls, expr: SymbolicExpr | None) -> str | None:
        """Name of the first value-dependent op embedded in ``expr`` (a
        pointer expression), or None when the address is value-independent.
        """
        if expr is None:
            return None
        for op in cls._VALUE_DEPENDENT_ADDRESS_OPS:
            try:
                if expr.has_op(op):
                    return op
            except Exception:
                continue
        return None

    def _reject_data_dependent_address(self, ptr_expr: SymbolicExpr | None) -> bool:
        """If ``ptr_expr`` depends on a runtime value, mark the launch
        unsupported and return True; callers should ``return`` immediately
        on True.
        """
        op = self._find_value_dependent_op(ptr_expr)
        if op is None:
            return False
        self._mark_unsupported(
            f"data-dependent memory address through {op} is unsupported "
            "by the current symbolic race detector"
        )
        return True

    # Ops whose value differs across program instances (pid, arange lanes)
    # or depends on runtime memory contents (loads, sorts/scans over loaded
    # tensors, atomic returns). A scalar built only from constants — or from
    # enclosing loop iterators, which concretize per iteration — takes the
    # same value in every block, so host-side use of it stays sound.
    _PER_INSTANCE_OPS: ClassVar[tuple[str, ...]] = (
        "pid",
        "arange",
        "load",
        "atomic_cas",
        "atomic_rmw",
        "sort",
        "cumsum",
    )

    @classmethod
    def _expr_varies_per_instance(cls, expr: SymbolicExpr | None) -> bool:
        if expr is None:
            return False
        try:
            return any(expr.has_op(op) for op in cls._PER_INSTANCE_OPS)
        except Exception:
            return True  # fail closed: unknown counts as per-instance

    def _scalar_concretize_observer_impl(self, expr: SymbolicExpr) -> None:
        """Policy for the engine's scalar-concretization hook.

        Host-side control flow (``if pid == 0:``, ``while flag:``) forces a
        scalar symbolic value to the capture block's concrete value. Under
        one-shot capture that bakes block (0,0,0)'s branch decisions into
        the template for every PID: events inside a pid-guarded branch
        become unconditional for all pids (false positives) and branches
        the capture block doesn't take record nothing (false negatives) —
        with last_status still "ok". No path condition is modeled, so the
        only sound verdict is unsupported.
        """
        if self._unsupported_capture:
            return
        if not self._expr_varies_per_instance(expr):
            return
        if not scalar_truthiness_from_user_code():
            return
        self._mark_unsupported(
            "host-side control flow on a value that varies per program "
            "instance (program id, tl.arange, or loaded data) is "
            "unsupported by one-shot symbolic capture"
        )

    def _on_data_dependent_value(self, expr: Any = None) -> None:
        """Loop bounds / materialized operands that depend on loads or pids
        are concretized to the capture block's values (the sanitizer
        compensates by running the full grid via need_full_grid; one-shot
        capture cannot), so the launch verdict would be silently wrong.

        A bound built only from enclosing loop iterators is exempt: it
        concretizes to the right value on every iteration, and the
        finished-iterator machinery models the leftovers.
        """
        SymbolicClient._on_data_dependent_value(self, expr)
        if self._unsupported_capture:
            return
        if isinstance(expr, SymbolicExpr) and not self._expr_varies_per_instance(expr):
            return
        self._mark_unsupported(
            "loop bound or operand depending on per-instance or loaded "
            "data was concretized; unsupported by one-shot symbolic "
            "capture"
        )

    @staticmethod
    def _tensor_region(tensor: Any) -> tuple[int, int, Any]:
        """Byte-region (base, end_exclusive, tensor). ``end`` is exclusive.

        Using regions instead of ``data_ptr()`` equality so views/slices that
        share storage but differ in offset are still detected as overlapping.
        """
        base = int(tensor.data_ptr())
        elem_size = int(tensor.element_size()) if hasattr(tensor, "element_size") else 1
        end = base + int(tensor.numel()) * elem_size
        return (base, end, tensor)

    @staticmethod
    def _regions_overlap(a: tuple[int, int, Any], b: tuple[int, int, Any]) -> bool:
        return max(a[0], b[0]) < min(a[1], b[1])

    def _note_written_tensor(self, tensor: Any) -> bool:
        """Register a write target. Returns True if the caller may proceed
        with recording the event, False if the launch was marked unsupported.

        - Unknown target: set ``_unknown_written_region_seen``. If a load
          snapshot already exists, also mark unsupported (the unknown write
          may alias it). Otherwise allow the event through; the flag gates
          any *subsequent* load via the provider's first check.
        - Known target overlapping an existing load source: mark unsupported.
        """
        if tensor is None:
            self._unknown_written_region_seen = True
            if self._load_value_regions:
                self._mark_unsupported(
                    "write to unknown region after tl.load value snapshot "
                    "is unsupported"
                )
                return False
            return True

        region = self._tensor_region(tensor)
        for snap in self._load_value_regions:
            if self._regions_overlap(region, snap):
                self._mark_unsupported(
                    "tl.store/atomic into a tensor previously read as a "
                    "tl.load value source is unsupported"
                )
                return False
        # Dedup writes to the same region — no need to track multiple
        # equal entries.
        for existing in self._written_regions:
            if existing[0] == region[0] and existing[1] == region[1]:
                return True
        self._written_regions.append(region)
        return True

    def _note_load_source_or_raise(self, tensor: Any) -> None:
        """Register a tensor as a load-value source. Raises
        :class:`UnsupportedSymbolicRaceQuery` when the source overlaps a
        region this kernel has already written to.
        """
        region = self._tensor_region(tensor)
        for written in self._written_regions:
            if self._regions_overlap(region, written):
                raise UnsupportedSymbolicRaceQuery(
                    "tl.load value from a tensor written by this kernel is "
                    "unsupported"
                )
        for existing in self._load_value_regions:
            if existing[0] == region[0] and existing[1] == region[1]:
                return
        self._load_value_regions.append(region)

    # ── Load-value provider (tl.load value semantics in Z3) ────────────────

    @staticmethod
    def _is_modelable_dtype(dtype: Any) -> bool:
        """Only integer-valued or bool input tensors are modeled. Floats and
        complex dtypes raise unsupported because the Z3 model is integer-
        only and silently downcasting would mask real value-dependent
        behaviour.
        """
        if dtype is None:
            return False
        try:
            if dtype == torch.bool:
                return True
            if hasattr(dtype, "is_floating_point") and dtype.is_floating_point:
                return False
            if hasattr(dtype, "is_complex") and dtype.is_complex:
                return False
            # Treat anything else with a finite integer representation as
            # modelable. Triton's int8/int16/int32/int64/uint8 all qualify.
            return getattr(dtype, "is_signed", None) is not None or hasattr(
                dtype, "itemsize"
            )
        except Exception:
            return False

    def _snapshot_array_for_tensor(self, tensor: Any) -> tuple[Any, list[Any]]:
        """Build (or fetch from cache) a Z3 Array representing the tensor's
        current contents, plus the list of known IntVal addresses for every
        element. Raises :class:`UnsupportedSymbolicRaceQuery` on any input
        the model can't faithfully represent.
        """
        if not hasattr(tensor, "numel") or not hasattr(tensor, "data_ptr"):
            raise UnsupportedSymbolicRaceQuery(
                "tl.load value modelling requires a torch tensor source"
            )
        if hasattr(tensor, "is_contiguous") and not bool(tensor.is_contiguous()):
            raise UnsupportedSymbolicRaceQuery(
                "tl.load value from a non-contiguous tensor is unsupported"
            )
        numel = int(tensor.numel())
        if numel <= 0:
            raise UnsupportedSymbolicRaceQuery(
                "tl.load value from an empty tensor is unsupported"
            )
        if numel > self._MAX_LOAD_SOURCE_ELEMENTS:
            raise UnsupportedSymbolicRaceQuery(
                f"tl.load source tensor exceeds the "
                f"{self._MAX_LOAD_SOURCE_ELEMENTS}-element snapshot cap "
                f"({numel} elements)"
            )
        if not self._is_modelable_dtype(getattr(tensor, "dtype", None)):
            raise UnsupportedSymbolicRaceQuery(
                f"tl.load value with dtype {getattr(tensor, 'dtype', '?')} "
                "is unsupported (only integer/bool values are modeled)"
            )

        base = int(tensor.data_ptr())
        elem_size = int(tensor.element_size()) if hasattr(tensor, "element_size") else 1
        elem_size = max(1, elem_size)
        cache_key = (base, elem_size, numel, str(tensor.dtype))
        cached = self._load_array_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            host = tensor.detach() if hasattr(tensor, "detach") else tensor
            host = host.cpu() if hasattr(host, "cpu") else host
            flat = host.reshape(-1).tolist()
            int_values = [int(v) for v in flat]
        except Exception as exc:
            raise UnsupportedSymbolicRaceQuery(
                f"tl.load value tensor snapshot failed: {exc}"
            ) from exc

        arr = K(IntSort(), IntVal(0))
        known_addrs: list[Any] = []
        for i, value in enumerate(int_values):
            addr_i = IntVal(base + i * elem_size)
            arr = Z3ArrayStore(arr, addr_i, IntVal(value))
            known_addrs.append(addr_i)

        self._load_array_cache[cache_key] = (arr, known_addrs)
        return arr, known_addrs

    @staticmethod
    def _to_lane_list(value: Any) -> list[Any]:
        if isinstance(value, list):
            return list(value)
        return [value]

    @staticmethod
    def _broadcast_lanes(lanes: list[Any], n: int) -> list[Any]:
        if len(lanes) == n:
            return lanes
        if len(lanes) == 1:
            return [lanes[0]] * n
        raise UnsupportedSymbolicRaceQuery(
            f"tl.load lane count mismatch (expected {n}, got {len(lanes)})"
        )

    def _load_value_provider_impl(
        self, load_expr: LoadSymbolicExpr
    ) -> tuple[Z3Expr, ConstraintConjunction]:
        """Compute Z3 value semantics for ``tl.load``.

        Address-of-event recording stays with ``expr.ptr._to_z3()`` (see
        ``_handle_access_check``); this provider only kicks in when the
        loaded value is consumed as part of a downstream expression (e.g.
        ``(v == 0)`` in a store mask). The returned Z3 expression evaluates
        to the actual loaded value via ``Select(arr, addr)`` over a
        per-launch tensor snapshot.

        Boundaries that raise :class:`UnsupportedSymbolicRaceQuery`:
          - launch has already seen a write to an unresolved tensor region;
          - source pointer cannot be resolved to a known tensor;
          - source overlaps a tensor this kernel writes to;
          - source is non-contiguous, too large, or has unsupported dtype;
          - masked load without an explicit ``other`` (no sound default for
            inactive lanes).
        """
        if self._unknown_written_region_seen:
            raise UnsupportedSymbolicRaceQuery(
                "tl.load value snapshot is unsupported after a write to an "
                "unknown kernel region"
            )

        ptr_z3, ptr_constraints = load_expr.ptr._to_z3()

        tensor = self._resolve_tensor(load_expr.ptr)
        if tensor is None:
            raise UnsupportedSymbolicRaceQuery(
                "tl.load value from unknown tensor is unsupported"
            )

        self._note_load_source_or_raise(tensor)
        arr, known_addrs = self._snapshot_array_for_tensor(tensor)

        addr_lanes = self._to_lane_list(ptr_z3)
        lane_count = len(addr_lanes)

        if load_expr.mask is None:
            values = [Select(arr, a) for a in addr_lanes]
            domain_terms = [Or(*(a == k for k in known_addrs)) for a in addr_lanes]
            extra_constraints: tuple[Any, ...] = (_and_constraints(*domain_terms),)
        else:
            if load_expr.other is None:
                raise UnsupportedSymbolicRaceQuery(
                    "masked tl.load without explicit `other` is unsupported"
                )
            mask_z3, mask_constraints = load_expr.mask._to_z3()
            other_z3, other_constraints = load_expr.other._to_z3()
            mask_lanes = self._broadcast_lanes(self._to_lane_list(mask_z3), lane_count)
            other_lanes = self._broadcast_lanes(
                self._to_lane_list(other_z3), lane_count
            )

            values = []
            domain_terms = []
            for a, m, o in zip(addr_lanes, mask_lanes, other_lanes):
                m_bool = _constraint_to_bool(m)
                values.append(If(m_bool, Select(arr, a), o))
                domain_terms.append(Implies(m_bool, Or(*(a == k for k in known_addrs))))
            extra_constraints = (
                mask_constraints,
                other_constraints,
                _and_constraints(*domain_terms),
            )

        result: Z3Expr = values[0] if lane_count == 1 else values
        constraints = _and_constraints(ptr_constraints, *extra_constraints)
        return result, constraints

    # Explicit forwarders to SymbolicClient: the RaceDetector factory
    # carries concrete stubs (NotImplementedError or ``return True``) to
    # satisfy Client's @abstractmethod contract, and those stubs would
    # otherwise shadow SymbolicClient's impls in the subclass MRO.
    def grid_idx_callback(self, grid_idx: tuple[int, ...]) -> None:
        SymbolicClient.grid_idx_callback(self, grid_idx)
        # Capture is one-shot: program_seq spans a single symbolic pass over
        # all records, so we only reset it on grid_callback, not per block.

    def _fence_between(
        self, first: AccessEventRecord, second: AccessEventRecord
    ) -> bool:
        """The solver's fence rule: two accesses of one program instance are
        ordered when the capture saw a tl.debug_barrier between every
        execution of one and every execution of the other, and, in each
        iteration of their common loops, both executed and ``first`` before
        ``second``.

        Every program instance runs the captured block's host-side control
        flow (a per-instance branch or loop bound is unsupported), so the
        captured executions are every instance's. A loop record stands for
        all its iterations; the barriers must separate them all: in a loop
        doing store, barrier, load, the load of one iteration and the store
        of the next are unordered unless a barrier also runs on the back
        edge.
        """
        if not self.fences:
            return False
        hits_a, hits_b = self._hits_of(first), self._hits_of(second)
        if {bisect_left(self.fences, seq) for seq, _ in hits_a} & {
            bisect_left(self.fences, seq) for seq, _ in hits_b
        }:
            return False
        stack_a = [site for site, _ in hits_a[0][1]]
        stack_b = [site for site, _ in hits_b[0][1]]
        shared = [site for site in stack_a if site in stack_b]

        def spans(hits, stack) -> dict[tuple, tuple[int, int]] | None:
            out: dict[tuple, tuple[int, int]] = {}
            for seq, key in hits:
                if [site for site, _ in key] != stack:
                    return None
                it = tuple(value for site, value in key if site in shared)
                lo, hi = out.get(it, (seq, seq))
                out[it] = (min(lo, seq), max(hi, seq))
            return out

        spans_a, spans_b = spans(hits_a, stack_a), spans(hits_b, stack_b)
        if spans_a is None or spans_b is None or spans_a.keys() != spans_b.keys():
            return False
        return all(spans_a[it][1] < spans_b[it][0] for it in spans_a)

    def _fence_epoch(self, record: AccessEventRecord) -> Any:
        """The solver's fence epoch: the number of barriers one program
        instance has executed before an execution of ``record``. An int
        when every captured execution agrees, else a term over the
        record's loop iterators (the ``loop_i_*`` variables of its
        iterations): affine in a single loop's iterator when that fits
        every iteration of the loop, else an If-chain over the captured
        iterations. An iteration that was not captured, or was captured
        with two epochs, maps to an unconstrained value."""
        by_key: dict[tuple, set[int]] = {}
        for seq, key in self._hits_of(record):
            by_key.setdefault(key, set()).add(bisect_left(self.fences, seq))
        epochs = set().union(*by_key.values())
        if len(epochs) == 1:
            return next(iter(epochs))
        iterators = {v.decl().name(): v for v in record.copy_local_vars}
        affine = self._affine_epoch(by_key, iterators)
        if affine is not None:
            return affine
        term: Any = Int(f"fence_epoch_unknown_{record.event_id}")
        for key, found in by_key.items():
            if len(found) != 1:
                continue
            conditions = [
                iterators[f"loop_i_{site}"] == value
                for site, value in key
                if value is not None and f"loop_i_{site}" in iterators
            ]
            if not conditions or len(conditions) != len(key):
                continue
            term = If(And(*conditions), IntVal(next(iter(found))), term)
        return term

    def _affine_epoch(self, by_key: dict, iterators: dict) -> Any:
        """``e0 + k * (i - i0) / d`` for a record in one loop whose epoch
        grows by k per iteration over EVERY iteration of the loop's range
        (so no iteration the solver allows gets an extrapolated epoch);
        else None. Keeps the solver's terms linear where the If-chain
        would have one branch per iteration."""
        points = []
        for key, found in by_key.items():
            if len(found) != 1 or len(key) != 1 or key[0][1] is None:
                return None
            points.append((key[0][1], next(iter(found))))
        sites = {key[0][0] for key in by_key}
        if len(sites) != 1 or len(points) < 2:
            return None
        site = next(iter(sites))
        var = iterators.get(f"loop_i_{site}")
        ranges = self._loop_ranges.get(site, set())
        if var is None or len(ranges) != 1:
            return None
        if {v for v, _ in points} != set(range(*next(iter(ranges)))):
            return None
        points.sort()
        (v0, e0), (v1, e1) = points[0], points[1]
        d, k = v1 - v0, e1 - e0
        if any((v - v0) % d or e != e0 + k * ((v - v0) // d) for v, e in points):
            return None
        offset = var - v0 if d == 1 else (var - v0) / d
        return IntVal(e0) + k * offset

    def _hits_of(
        self, record: AccessEventRecord
    ) -> list[tuple[int, tuple[tuple[Any, int | None], ...]]]:
        return self._record_hits.get(record.event_id) or [(record.program_seq, ())]

    def _loop_key(self) -> tuple[tuple[Any, int | None], ...]:
        """The current iteration of every enclosing loop."""
        for ctx in self.loop_stack:
            self._loop_ranges.setdefault(ctx.loop_site, set()).add(
                (ctx.start, ctx.stop, ctx.step)
            )
        return tuple((ctx.loop_site, ctx.current_value) for ctx in self.loop_stack)

    def finalize(self) -> list:
        """Run the two-copy symbolic HB solver and return this launch's
        records: a RaceRecord per race (each also printed), or, for a launch
        with no proof (``last_status`` "unsupported", "vacuous" or
        "aborted"), an IRVerdict saying why.

        Limitations carried by the underlying ``TwoCopySymbolicHBSolver``:
          - Initial atomic source covers scalar tensors and small contiguous
            flag arrays (``numel <= 1024``); larger or non-contiguous
            tensors are conservatively reported as races.
          - A third program instance is an atomic's rf source, but
            synchronization passed on through it is not modeled (a lock
            handed from one instance to another through a third is
            over-reported).
          - Float atomic-RMW return values are not modeled — downstream use
            of the return marks the launch unsupported.
          - Atomic CAS/RMW inside loops are not modeled — the launch is
            marked unsupported instead of recording phantom events.
        """
        try:
            self.last_reports = []
            if not self._captured_symbolic_template and not self._unsupported_capture:
                # The capture was never sealed: no block ran to completion,
                # so no analysis ran and the launch-start "aborted" stands.
                self.last_status = "aborted"
            elif not self._unsupported_capture:
                self._solve()
            if self.last_status == "ok":
                for record in self.last_reports:
                    print_race_record(record)
                return list(self.last_reports)
            return [self._verdict()]
        finally:
            # Unconditional: an exception escaping finalize (a z3 error)
            # must still release the launch runtime — the class-level hook
            # slots would otherwise leak into later launches.
            self._clear_launch_runtime()

    def _solve(self) -> None:
        try:
            solver = TwoCopySymbolicHBSolver(
                self.records,
                grid=self._launch_grid,
                arange_dict=self._arange_dict_snapshot,
                fence_between=self._fence_between,
                fence_epoch=self._fence_epoch,
            )
            reports = solver.find_races()
            if not reports and not solver.check_feasibility():
                # Every pair query was vacuously UNSAT because the premises
                # admit no execution at all (e.g. a poll for a value no
                # modeled source can supply): a race-freedom proof needs a
                # feasible base system, so none is claimed. (A non-empty
                # report list needs no check: a SAT race query is itself a
                # feasibility witness.)
                self.last_status = "vacuous"
                self.unsupported_kind = "vacuous"
                self.unsupported_reason = (
                    "premises unsatisfiable: the base system admits no "
                    "execution, so no race-freedom proof is claimed"
                )
                return
        except UnsupportedSymbolicRaceQuery as exc:
            if exc.reason_unknown in ("timeout", "canceled"):
                kind = "timeout"
            elif exc.reason_unknown is not None:
                kind = "unknown"
            else:
                kind = "unsupported"
            self._mark_unsupported(str(exc), kind)
            return
        self.last_status = "ok"
        self.last_reports = [_race_record(report) for report in reports]

    def _verdict(self) -> IRVerdict:
        """The IRVerdict of a launch with no proof; also warns."""
        kind = self.unsupported_kind or self.last_status
        reason = self.unsupported_reason or "the launch ended before its analysis"
        loc = self._unsupported_loc
        message = f"[{self.LOG_TAG}] launch {self.last_status} ({kind}): {reason}"
        if loc is None:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        else:
            # warn_explicit with no registry, so Python's default filter does
            # not collapse later launches' notices.
            warnings.warn_explicit(message, RuntimeWarning, loc[0], loc[1])
        return IRVerdict(
            client=self.NAME,
            status=self.last_status,
            refusal=Refusal(
                kind,
                reason,
                loc=None if loc is None else SourceLocation(loc[0], loc[1]),
            ),
        )

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        return SymbolicClient.register_for_loop_callback(self)

    def begin_launch(self, call: Any) -> None:
        # Arm the pessimistic verdict before anything of the launch runs:
        # arg conversion and the user's grid lambda run before grid_callback,
        # and a crash there must not leave the previous launch's "ok"
        # readable as this launch's verdict, nor the previous launch's
        # refusal be kept by this launch's abort_launch.
        self.last_status = "aborted"
        self.last_reports = []
        self._unsupported_capture = False
        self.unsupported_reason = self.unsupported_kind = None
        self._unsupported_loc = None

    def arg_callback(self, name: str, arg: Any, arg_cvt: Any) -> None:
        SymbolicClient.arg_callback(self, name, arg, arg_cvt)

    def grid_callback(self, grid: tuple[int, ...]) -> None:
        self.records = []
        self.fences = []
        self._record_hits = {}
        self._loop_ranges = {}
        self._dep_anchor_ids = {}
        self.last_reports = []
        # Pessimistic until finalize() proves otherwise; every finalize()
        # path overwrites this with the real outcome.
        self.last_status = "aborted"
        self._program_seq = 0
        self._event_seq = 0
        normalized = tuple(int(dim) for dim in grid)
        while len(normalized) < 3:
            normalized = normalized + (1,)
        self._launch_grid = cast(tuple[int, int, int], normalized[:3])
        self._captured_symbolic_template = False
        self._capture_claimed = False
        self._capture_thread_id = None
        self._unsupported_capture = False
        self.unsupported_reason = None
        self.unsupported_kind = None
        self._unsupported_loc = None
        self._arange_dict_snapshot = {}
        self._load_array_cache = {}
        self._load_value_regions = []
        self._written_regions = []
        self._unknown_written_region_seen = False
        self._finished_loop_iter_subs = {}
        self._suspended_iter_subs = []
        self._known_iter_var_keys = set()
        self._unstable_iter_var_keys = set()
        self._loop_flush_signatures = {}
        self._iteration_occurrences = {}
        SymbolicExpr.ARANGE_DICT.clear()
        # SymbolicClient.grid_callback also clears loop_stack and empties
        # every hook slot, so a launch that aborted mid-loop cannot poison
        # this one.
        SymbolicClient.grid_callback(self, grid)
        # The scalar-concretize observer must span the whole kernel run
        # (host-side truthiness fires from interpreter code, not from
        # detector-triggered evals), and so must the atomic-value provider
        # (the engine also lowers atomic results itself, e.g. in a loop
        # bound). Both are installed launch-wide with an owner token;
        # _clear_launch_runtime only uninstalls while we still own the slot.
        # The load-value provider is NOT installed here — it is scoped to
        # the detector's own evaluations via _load_value_semantics so other
        # clients' expr.eval() never sees it.
        SymbolicExpr._scalar_concretize_observer = self._scalar_concretize_observer_impl
        SymbolicExpr._scalar_concretize_observer_owner = id(self)
        SymbolicExpr._atomic_value_provider = _atomic_old_value
        SymbolicExpr._atomic_value_provider_owner = id(self)

    def abort_launch(self, exc: BaseException) -> None:
        SymbolicClient.abort_launch(self, exc)
        # A launch already marked unsupported keeps that verdict: the first
        # refusal is the one reported, even if interpretation raised later.
        if not self._unsupported_capture:
            self.last_status = "aborted"
            self.unsupported_reason = self.unsupported_kind = None
            self._unsupported_loc = None
        self.last_reports = []
        self._clear_launch_runtime()

    def register_op_callback(self, op_type: type[Op]) -> OpCallbacks:
        if op_type is Barrier:
            return OpCallbacks(before_callback=self.lock_fn(self._on_barrier))
        if op_type is InterpreterDivergence:
            return OpCallbacks(
                before_callback=self.lock_fn(self._on_interpreter_divergence)
            )
        if op_type is InlineAsm:
            return OpCallbacks(before_callback=self.lock_fn(self._on_inline_asm))
        return SymbolicClient.register_op_callback(self, op_type)

    def _on_barrier(self, *args: Any, **kwargs: Any) -> None:
        """``tl.debug_barrier``: record a tile-level fence at the current
        program_seq position. The fence carries no accesses; _fence_between
        reads ``fences`` to decide which same-instance access pairs are
        fence-ordered."""
        if self._unsupported_capture or not self._capture_active():
            return
        self.fences.append(self._next_program_seq())

    def _on_interpreter_divergence(
        self, kind: str, source_location: tuple[str, int, str]
    ) -> None:
        # The interpreted run is not what the compiled kernel does, so the
        # capture says nothing reliable about it: never a proof.
        if self._unsupported_capture:
            return
        self._mark_unsupported(
            "interpreter divergence: "
            + InterpreterDivergence.describe(kind, source_location)
            + ". The race detector cannot analyse this launch.",
            kind,
        )
        self._unsupported_loc = source_location

    def _on_inline_asm(self) -> None:
        # The approximated result is not the assembly's, and an access or
        # mask built from it would be analysed on wrong values.
        if self._unsupported_capture:
            return
        self._mark_unsupported(
            "tl.inline_asm_elementwise cannot be executed by the interpreter "
            "(its result is approximated); the race detector cannot analyse "
            "this launch",
            "inline-asm",
        )

    # ── Race-detector-only overrides for load/store overriders ────────────
    # The shared SymbolicClient versions concretise nested loads in ``ptr``
    # and ``mask`` via ``replace_subtree("load")`` and flip
    # ``need_full_grid``. Under one-shot symbolic capture that turns the
    # first block's concrete mask value into a template for every PID — the
    # source of the load-dependent-mask false negative. The race detector
    # has its own ``LoadSymbolicExpr`` value semantics (per-PID Z3
    # ``Select(arr, addr)``) plus a pointer-side ``_reject_data_dependent_
    # address`` gate, so it doesn't need (and must not use) the
    # concretisation path. Sanitizer keeps the shared behaviour.

    def _op_load_overrider(self, ptr, mask=None, other=None, *args, **kwargs):
        ptr_sym = SymbolicExpr.from_value(ptr)
        mask_sym = SymbolicExpr.from_value(mask) if mask is not None else None
        other_sym = SymbolicExpr.from_value(other) if other is not None else None
        ret = SymbolicExpr.create("load", ptr_sym, mask_sym, other_sym)
        self._handle_access_check(ret, Load, "read")
        return ret

    def _op_store_overrider(self, ptr, value, mask=None, *args, **kwargs):
        ptr_sym = SymbolicExpr.from_value(ptr)
        value_sym = SymbolicExpr.from_value(value)
        mask_sym = SymbolicExpr.from_value(mask) if mask is not None else None
        ret = SymbolicExpr.create("store", ptr_sym, value_sym, mask_sym)
        self._handle_access_check(ret, Store, "write")
        return ret

    def pre_run_callback(self, fn: Callable) -> bool:
        # Block scheduling stays with the shared SymbolicClient machinery:
        # it honors need_full_grid (so a co-attached client is not starved
        # of blocks when the engine concretized per-block values) and counts
        # _active_blocks for the deferred launch-state clear. One-shot
        # capture is layered on top: the first admitted block claims the
        # capture slot atomically, so sibling workers under
        # TILELENS_NUM_SMS >= 2 never interleave captures into the shared
        # per-launch record state.
        with self._lock_context():
            should_run = SymbolicClient.pre_run_callback(self, fn)
            if should_run and not self._capture_claimed:
                self._capture_claimed = True
                self._capture_thread_id = threading.get_ident()
            return should_run

    def post_run_callback(self, fn: Callable) -> bool:
        with self._lock_context():
            if self._capture_thread_id == threading.get_ident():
                try:
                    self._seal_capture()
                finally:
                    self._capture_thread_id = None
            return SymbolicClient.post_run_callback(self, fn)

    def _capture_active(self) -> bool:
        """True when events observed on the current thread belong to the
        one-shot capture.

        Blocks executed outside the capture slot (need_full_grid reruns for
        a co-attached client, sibling workers under TILELENS_NUM_SMS >= 2)
        still run the kernel, but their events must not reach the shared
        per-launch record/loop state — the two-copy solver already reasons
        over all blocks from the single captured template. Direct handler
        calls with no claimed slot (unit-level use) count as capturing.
        """
        if self._captured_symbolic_template:
            return False
        tid = self._capture_thread_id
        return tid is None or tid == threading.get_ident()

    def _seal_capture(self) -> None:
        """Seal the one-shot capture once the capturing block finishes.

        Runs exactly once per launch, on the thread that claimed the capture
        slot in pre_run_callback. The seal flag is set even on the
        unsupported paths so finalize() can tell a completed capture from a
        launch aborted mid-block.
        """
        try:
            # If a handler already marked the launch unsupported, don't try
            # to force-eval half-baked record state — short-circuit cleanly.
            if self._unsupported_capture:
                return
            # Defense in depth: a LoopContext that survives to launch end
            # was never flushed (its teardown hook did not fire), so its
            # deferred accesses are missing from the records — the capture
            # is incomplete and an "ok" verdict would be silently wrong.
            if self.loop_stack:
                self._mark_unsupported(
                    "loop context left open at launch end; loop-deferred "
                    "accesses were never recorded"
                )
                return
            try:
                self._force_eval_record_templates()
            except UnsupportedSymbolicRaceQuery as exc:
                self._mark_unsupported(str(exc))
                return
            # Defensive sweep: load-side / write-side checks at record time
            # should already catch overlaps, but loop-deferred events can
            # re-order tensors through `_process_pending_check`. Cross-
            # product check the two region lists once before sealing.
            for src in self._load_value_regions:
                for dst in self._written_regions:
                    if self._regions_overlap(src, dst):
                        self._mark_unsupported(
                            "tl.load value source overlaps a tensor written "
                            "by this kernel"
                        )
                        return
            # Snapshot ARANGE_DICT after templates are evaluated so the
            # two-copy solver's arange substitutions are independent of
            # subsequent launches.
            self._arange_dict_snapshot = dict(SymbolicExpr.ARANGE_DICT)
        finally:
            self._captured_symbolic_template = True

    # ── Event recording ───────────────────────────────────────────────────

    def _clear_launch_runtime(self) -> None:
        self._clear_cache()
        self._clear_symbolic_launch_state()
        self.need_full_grid = None
        self.solver = None
        self.addr_ok = None
        self.pid_ok = None
        # addr_sym is instance-lifetime, not launch-scoped: it is created once
        # in SymbolicClient.__init__ and never recreated, and the next launch's
        # grid_callback dereferences it via _addr_ok_premise(). Clearing it
        # here would crash the second launch of any traced kernel.
        self.grid = None
        self.grid_idx = None
        self.last_grid = None
        self._capture_claimed = False
        self._capture_thread_id = None
        self._program_seq = 0
        self._event_seq = 0
        self._load_array_cache = {}
        self._load_value_regions = []
        self._written_regions = []
        self._unknown_written_region_seen = False
        self._finished_loop_iter_subs = {}
        self._suspended_iter_subs = []
        self._known_iter_var_keys = set()
        self._unstable_iter_var_keys = set()
        self._loop_flush_signatures = {}
        self._iteration_occurrences = {}
        self._reclaim_hook_slots(only_own=True)

    @staticmethod
    def _normalize_constraints(
        constraints: ConstraintConjunction,
    ) -> tuple[Any, ...]:
        if constraints is None:
            return ()
        if isinstance(constraints, (list, tuple)):
            return tuple(constraints)
        return (constraints,)

    @staticmethod
    def _debug_name(
        op_type: type[Op],
        source_location: tuple[str, int, str] | None,
    ) -> str:
        base = getattr(op_type, "name", op_type.__name__.lower())
        if source_location is None:
            return base
        _, lineno, func_name = source_location
        if func_name:
            return f"{func_name}:{lineno}:{base}"
        return f"{base}:{lineno}"

    def _next_program_seq(self) -> int:
        seq = self._program_seq
        self._program_seq += 1
        return seq

    def _next_event_id(self) -> int:
        seq = self._event_seq
        self._event_seq += 1
        return seq

    @staticmethod
    def _infer_elem_size(expr: SymbolicExpr | None) -> int:
        """Best-effort byte-width of an access's element type.

        Prefer the pointer element type, so ``elem_size`` cannot silently
        degrade to 1 (and the solver's byte-overlap predicate to
        ``addr ==``). Fallback chain:
        ``expr.ptr.dtype`` → ``expr.dtype`` → ``expr.value.dtype`` → 1.
        """
        if expr is None:
            return 1

        def dtype_to_size(dtype: Any) -> int | None:
            if dtype is None:
                return None
            elem_ty = getattr(dtype, "element_ty", dtype)
            bw = getattr(elem_ty, "primitive_bitwidth", None)
            if bw is None:
                return None
            try:
                return max(1, int(bw) // 8)
            except Exception:
                return None

        try:
            ptr = getattr(expr, "ptr", None)
            size = dtype_to_size(getattr(ptr, "dtype", None))
            if size is not None:
                return size
            size = dtype_to_size(getattr(expr, "dtype", None))
            if size is not None:
                return size
            value = getattr(expr, "value", None)
            size = dtype_to_size(getattr(value, "dtype", None))
            if size is not None:
                return size
        except Exception:
            pass
        return 1

    def _current_loop_iter_vars(self) -> tuple[Any, ...]:
        return tuple(c.idx_z3 for c in self.loop_stack)

    # ── Finished-loop iterator concretization ─────────────────────────────
    # After a range loop exits, the leftover Python loop variable (and any
    # value derived from it) still lowers to the loop's symbolic loop_i_*
    # var, yet every program instance concretely holds the same final
    # iteration value. Keeping the symbolic var in post-loop records is
    # unsound either way: un-renamed it pins both solver copies together,
    # and renamed (which happens launch-wide once any in-loop record lists
    # the var in copy_local_vars) it roams unbounded because post-loop
    # records carry no range premise for it. Substituting the concrete
    # final value at record time models the actual semantics — but only
    # when that value is well defined:
    #   * a zero-trip re-activation leaves the leftover variable (and so
    #     its substitution) unchanged, hence the stash/restore;
    #   * a loop whose final value varies across activations under a still-
    #     active outer loop has no single correct constant (deferred records
    #     dedupe across those activations), so its var is marked unstable
    #     and any record still referencing it is rejected as unsupported by
    #     _refs_unresolved_iter_var.

    # The loop hooks fire for every executed block, but the loop bookkeeping
    # (loop_stack, suspended/finished iterator substitutions) belongs to the
    # one-shot capture: blocks running outside the capture slot iterate
    # concretely and must not touch it — a sibling worker popping the
    # capture thread's stash is exactly the kind of corruption the slot
    # exists to prevent. _wrap_range stays ungated: every running block
    # still needs its loop bounds materialized. LoopSite is a hashable
    # NamedTuple, so it serves directly as the _finished_loop_iter_subs key.

    def _loop_hook_iter_overrider(self, loop_site: Any, idx: Any) -> Any:
        if not self._capture_active():
            return idx
        return SymbolicClient._loop_hook_iter_overrider(self, loop_site, idx)

    def _loop_hook_before(self, loop_site: Any, iterable: Any) -> None:
        if not self._capture_active():
            return
        SymbolicClient._loop_hook_before(self, loop_site, iterable)
        if self.loop_stack and self.loop_stack[-1].loop_site == loop_site:
            ctx = self.loop_stack[-1]
            self._known_iter_var_keys.add(self._iter_var_key(ctx.idx_z3))
            # Reactivation: suspend the finished-value substitution so this
            # loop's own deferred records keep the symbolic var for per-copy
            # renaming. Restored on a zero-iteration exit.
            self._suspended_iter_subs.append(
                (loop_site, self._finished_loop_iter_subs.pop(loop_site, None))
            )

    def _loop_hook_after(self, loop_site: Any) -> None:
        if not self._capture_active():
            return
        ctx = (
            self.loop_stack[-1]
            if self.loop_stack and self.loop_stack[-1].loop_site == loop_site
            else None
        )
        SymbolicClient._loop_hook_after(self, loop_site)
        if ctx is None:
            return
        stashed_site, stashed = self._suspended_iter_subs.pop()
        assert stashed_site == loop_site
        var_key = self._iter_var_key(ctx.idx_z3)
        if var_key in self._unstable_iter_var_keys:
            return
        if ctx.current_value is None:
            # Zero-trip activation: the leftover variable still holds the
            # previous activation's final value — restore its substitution.
            if stashed is not None:
                self._finished_loop_iter_subs[loop_site] = stashed
            return
        # Register AFTER the super() flush so the loop's own records (which
        # are recorded during the flush) keep their symbolic iterator; only
        # records created after this point see the concrete final value.
        final = IntVal(int(ctx.current_value))
        if stashed is not None and not stashed[1].eq(final) and self.loop_stack:
            # The final value varies across activations while an outer loop
            # is active: pendings deferred in that outer loop dedupe across
            # the differing activations, so no constant is correct.
            self._unstable_iter_var_keys.add(var_key)
            return
        self._finished_loop_iter_subs[loop_site] = (ctx.idx_z3, final)

    def _apply_finished_iter_subs(self, value: Any) -> Any:
        if not self._finished_loop_iter_subs:
            return value
        return apply_sub(value, tuple(self._finished_loop_iter_subs.values()))

    @staticmethod
    def _iter_var_key(v: Any) -> tuple[int, str, str]:
        # Mirrors the dedup key used by hb_common.normalize_copy_local_vars.
        return (v.hash(), str(v.sort()), v.decl().name())

    def _refs_unresolved_iter_var(
        self, values: tuple[Any, ...], allowed_vars: tuple[Any, ...]
    ) -> bool:
        """True if any Z3 expr in ``values`` references a loop iterator var
        that is neither substituted away nor legitimately symbolic here.

        ``allowed_vars`` are the iterators the record is allowed to keep
        symbolic: the still-active outer loops' plus (for deferred records)
        the flushed loop's own — exactly the ones the two-copy solver
        alpha-renames via copy_local_vars. Anything else is a finished
        iterator whose substitution was skipped (unstable final value or a
        lifecycle corner); recording it would be silently wrong, so the
        caller marks the launch unsupported instead.
        """
        disallowed = self._known_iter_var_keys - {
            self._iter_var_key(v) for v in allowed_vars
        }
        if not disallowed:
            return False
        stack: list[Any] = list(values)
        while stack:
            v = stack.pop()
            if v is None or isinstance(v, (bool, int, float, str)):
                continue
            if isinstance(v, (list, tuple)):
                stack.extend(v)
                continue
            if not hasattr(v, "num_args"):
                continue
            if v.num_args() == 0:
                if self._iter_var_key(v) in disallowed:
                    return True
                continue
            stack.extend(v.children())
        return False

    def _force_eval_record_templates(self) -> None:
        """Ensure record template fields are Z3-ish, not unevaluated SymbolicExpr.

        Triggers ``.eval()`` on captured ``SymbolicExpr`` fields so the
        snapshotted ``ARANGE_DICT`` is complete before the solver consumes
        the records. Does NOT re-eval ``record.old_value`` /
        ``record.symbolic_expr`` themselves: re-evaluating an
        ``AtomicCasSymbolicExpr`` would not change anything thanks to caching,
        but we keep the rule explicit so downstream maintainers don't
        accidentally invalidate launch-level CAS-return identity.
        """

        def force(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, (bool, int, str)):
                return value
            if isinstance(value, list):
                return [force(v) for v in value]
            if isinstance(value, tuple):
                return tuple(force(v) for v in value)
            if isinstance(value, SymbolicExpr):
                z3_value, _ = value.eval(simplify_constraints=False)
                return z3_value
            return value

        with self._load_value_semantics():
            for record in self.records:
                try:
                    record.addr_expr = force(record.addr_expr)
                    record.local_constraints = self._normalize_constraints(
                        force(record.local_constraints)
                    )
                    record.premises = self._normalize_constraints(
                        force(record.premises)
                    )
                    if record.cas_cmp_value is not None:
                        record.cas_cmp_value = force(record.cas_cmp_value)
                    if record.cas_new_value is not None:
                        record.cas_new_value = force(record.cas_new_value)
                except Exception as exc:  # pragma: no cover - defensive
                    raise UnsupportedSymbolicRaceQuery(
                        f"failed to normalize record templates: {exc}"
                    ) from exc

    # ── Dependency order ──────────────────────────────────────────────────
    # Ops that keep every element at its position: a chain of these from a
    # load's result to a later operation's value/mask means position lambda
    # of the later operation depends on position lambda of the load.
    _ELEMENTWISE_OPS: frozenset[str] = frozenset(
        (
            "where",
            "fma",
            "cast_impl",
            "bitcast",
            "fp_to_fp",
            "addptr",
            "const",
            "pid",
            "arange",
        )
    )
    # Memory ops whose RESULT anchors a dependency (the chain stops here).
    _DEP_ANCHOR_OPS: frozenset[str] = frozenset(("load", "atomic_rmw", "atomic_cas"))

    def _append_record(self, record: AccessEventRecord) -> None:
        self.records.append(record)
        expr = record.symbolic_expr
        if expr is None or getattr(expr, "op", None) not in self._DEP_ANCHOR_OPS:
            return
        # A block anchor without a lane_position (e.g. B lanes on one
        # address, z + tl.zeros([B])) has no position to pair a dependent
        # lane with: it anchors no dependency, rather than one that orders
        # every lane of it before the dependent.
        if tuple(getattr(expr, "shape", None) or ()) and record.lane_position is None:
            return
        self._dep_anchor_ids[id(expr)] = record.event_id

    def _dep_loads_of(self, expr: Any) -> tuple[int, ...]:
        """event_ids of the anchors this record's value / mask / compare
        operands reach through element-wise ops only (the positional rule).
        The record's own pointer chain is excluded: a loaded value in an
        address is the indirect-address refusal, not a dependency."""
        if expr is None or not hasattr(expr, "children"):
            return ()
        roots = [
            child
            for name, child in expr.children.items()
            if name in ("value", "val", "mask", "cmp") and child is not None
        ]
        elementwise = (
            self._ELEMENTWISE_OPS
            | set(SymbolicExpr.UNARY_OPS)
            | set(SymbolicExpr.BINARY_OPS)
        )
        # Evaluate sets bottom-up rather than walking a union of all leaves:
        # only one `where` value arm contributes at a given position. Its
        # condition is unconditional, while an arm-derived anchor must be
        # present in BOTH alternatives (the static arith.select contract).
        # An explicit stack preserves support for deep, shared expression DAGs.
        dependencies: dict[int, frozenset[int]] = {}
        empty: frozenset[int] = frozenset()
        stack = [(root, False) for root in roots]
        while stack:
            node, expanded = stack.pop()
            key = id(node)
            if key in dependencies:
                continue
            op = getattr(node, "op", None)
            if op in self._DEP_ANCHOR_OPS:
                eid = self._dep_anchor_ids.get(key)
                dependencies[key] = frozenset((eid,)) if eid is not None else empty
                continue  # stop at the anchor, excluding its own operands
            if op not in elementwise:
                dependencies[key] = empty
                continue  # position-changing or unknown operation
            children = node.children
            if op == "where" and set(children) != {"cond", "lhs", "rhs"}:
                dependencies[key] = empty
                continue  # do not guess the arms of an unfamiliar selection
            if not expanded:
                stack.append((node, True))
                stack.extend((child, False) for child in children.values())
                continue
            if op == "where":
                condition = dependencies[id(children["cond"])]
                lhs = dependencies[id(children["lhs"])]
                rhs = dependencies[id(children["rhs"])]
                dependencies[key] = condition | (lhs & rhs)
            else:
                dependencies[key] = empty.union(
                    *(dependencies[id(child)] for child in children.values())
                )
        return tuple(sorted(empty.union(*(dependencies[id(root)] for root in roots))))

    def _record_access_event(
        self,
        access_mode: AccessMode,
        op_type: type[Op],
        access_addr: Z3Expr,
        expr_constraints: ConstraintConjunction,
        symbolic_expr: SymbolicExpr,
        source_location: tuple[str, int, str] | None = None,
        *,
        semantic_constraints: tuple[Any, ...] = (),
        copy_local_vars: tuple[Any, ...] = (),
        active: Any = True,
        loop_flush: bool = False,
        hits: list[tuple[int, tuple[tuple[Any, int | None], ...]]] | None = None,
        program_seq: int | None = None,
        lane_position: tuple[Any, ...] | None = None,
        occurrence: int = 0,
    ) -> None:
        if self._unsupported_capture:
            return
        tensor = self._resolve_tensor(symbolic_expr)
        tensor_name = self._get_tensor_name(tensor) if tensor is not None else None

        # Bidirectional self-write guard. For writes (and atomics), the
        # target region must not overlap a tensor we've already snapshotted
        # as a load-value source — that would mean the snapshot represents
        # stale memory. Likewise, an unresolved write target may alias an
        # existing snapshot, so we set a flag the load-value provider checks
        # before snapshotting any further tensors.
        if access_mode == "write" or op_type in (AtomicCas, AtomicRMW):
            if not self._note_written_tensor(tensor):
                return

        # Two-copy capture: keep raw symbolic templates (PID0/1/2 preserved)
        # rather than snapshotting solver assertions, which can carry sampled
        # PID equalities that pin pid_a == pid_b == sampled_pid after alpha-
        # renaming and break the two-copy alias query.
        access_addr = self._apply_finished_iter_subs(access_addr)
        active = self._apply_finished_iter_subs(active)
        local = self._normalize_constraints(
            self._apply_finished_iter_subs(expr_constraints)
        )
        premises = self._normalize_constraints(
            self._apply_finished_iter_subs(semantic_constraints)
        )
        allowed_iter_vars = copy_local_vars + self._current_loop_iter_vars()
        if self._refs_unresolved_iter_var(
            (access_addr, active, local, premises), allowed_iter_vars
        ):
            self._mark_unsupported(
                "access references a finished loop iterator with no stable "
                "final value"
            )
            return

        normalized_copy_vars = normalize_copy_local_vars(copy_local_vars)
        if loop_flush:
            # A loop nested under another loop re-enters once per outer
            # iteration, re-flushing structurally identical pending events
            # each time (fresh LoopContext, fresh signature_cache). One
            # record already covers every iteration pair through its symbolic
            # iterators, so dedupe at launch scope — AFTER the finished-
            # iterator substitutions and the unresolved-var rejection above,
            # which are the parts that can legitimately differ (or become
            # unsupported) across activations.
            signature = hash(
                (
                    _make_event_signature(
                        access_mode, source_location, access_addr, local, active
                    ),
                    op_type.__name__,
                    occurrence,
                    _hash_signature_part(premises),
                    tuple(self._iter_var_key(v) for v in normalized_copy_vars),
                )
            )
            existing = self._loop_flush_signatures.get(signature)
            if existing is not None:
                self._record_hits[existing].extend(hits or ())
                return

        event_id = self._next_event_id()
        if loop_flush:
            self._loop_flush_signatures[signature] = event_id
            self._record_hits[event_id] = list(hits or ())
        self._append_record(
            AccessEventRecord(
                op_type=op_type,
                access_mode=access_mode,
                tensor=tensor,
                tensor_name=tensor_name,
                symbolic_expr=symbolic_expr,
                dep_loads=self._dep_loads_of(symbolic_expr),
                lane_position=lane_position,
                addr_expr=access_addr,
                premises=premises,
                local_constraints=local,
                source_location=source_location,
                grid_idx=None,
                program_seq=self._next_program_seq()
                if program_seq is None
                else program_seq,
                debug_name=self._debug_name(op_type, source_location),
                active=active,
                reads=access_mode == "read",
                writes=access_mode == "write",
                event_id=event_id,
                elem_size=self._infer_elem_size(symbolic_expr),
                copy_local_vars=normalized_copy_vars,
            )
        )

    @staticmethod
    def _normalize_sem(sem: str | None) -> MemorySem:
        if sem is None:
            return "acq_rel"
        name = getattr(sem, "name", sem)
        normalized = str(name).lower()
        if normalized == "relaxed":
            return "relaxed"
        if normalized == "acquire":
            return "acquire"
        if normalized == "release":
            return "release"
        if normalized in ("acquire_release", "acq_rel"):
            return "acq_rel"
        if normalized == "plain":
            return "plain"
        return cast(MemorySem, normalized)

    @staticmethod
    def _normalize_scope(scope: str | None) -> str:
        if scope is None:
            return "gpu"
        name = getattr(scope, "name", scope)
        normalized = str(name).lower()
        normalized = {"system": "sys"}.get(normalized, normalized)
        if normalized not in ("cta", "gpu", "sys"):
            # An unrecognized scope must not fall open to device scope:
            # a narrower future scope (e.g. cluster) would silently be
            # judged morally strong device-wide. Unsupported is the
            # honest verdict.
            raise UnsupportedSymbolicRaceQuery(
                f"unsupported memory scope {normalized!r}"
            )
        return normalized

    def _record_atomic_cas_event(
        self,
        symbolic_expr: SymbolicExpr,
        addr_expr: Z3Expr,
        expr_constraints: ConstraintConjunction,
        cmp_value: Any,
        value: Any,
        old_value: Any,
        sem: str | None,
        scope: str | None,
        source_location: tuple[str, int, str] | None = None,
        *,
        semantic_constraints: tuple[Any, ...] = (),
        lane_position: tuple[Any, ...] | None = None,
    ) -> None:
        if self._unsupported_capture:
            return
        tensor = self._resolve_tensor(symbolic_expr)
        tensor_name = self._get_tensor_name(tensor) if tensor is not None else None
        if not self._note_written_tensor(tensor):
            return

        addr_expr = self._apply_finished_iter_subs(addr_expr)
        cmp_value = self._apply_finished_iter_subs(cmp_value)
        value = self._apply_finished_iter_subs(value)
        local = self._normalize_constraints(
            self._apply_finished_iter_subs(expr_constraints)
        )
        premises = self._normalize_constraints(
            self._apply_finished_iter_subs(semantic_constraints)
        )
        loop_vars = self._current_loop_iter_vars()
        if self._refs_unresolved_iter_var(
            (addr_expr, cmp_value, value, local, premises), loop_vars
        ):
            self._mark_unsupported(
                "atomic_cas references a finished loop iterator with no "
                "stable final value"
            )
            return

        # Raw symbolic templates: writes / written_value are recomputed by the
        # two-copy solver per copy from cas_cmp_value / cas_new_value /
        # old_value. Storing them here as None keeps the per-copy CAS return
        # rename in lockstep with the substitution applied to old_value.
        self._append_record(
            AccessEventRecord(
                op_type=AtomicCas,
                access_mode="read",
                tensor=tensor,
                tensor_name=tensor_name,
                symbolic_expr=symbolic_expr,
                dep_loads=self._dep_loads_of(symbolic_expr),
                lane_position=lane_position,
                addr_expr=addr_expr,
                premises=premises,
                local_constraints=local,
                source_location=source_location,
                grid_idx=None,
                program_seq=self._next_program_seq(),
                debug_name=self._debug_name(AtomicCas, source_location),
                active=True,
                reads=True,
                writes=None,
                is_atomic=True,
                atomic_kind="cas",
                sem=self._normalize_sem(sem),
                scope=self._normalize_scope(scope),
                old_value=old_value,
                written_value=None,
                event_id=self._next_event_id(),
                elem_size=self._infer_elem_size(symbolic_expr),
                cas_cmp_value=cmp_value,
                cas_new_value=value,
                copy_local_vars=normalize_copy_local_vars((old_value,) + loop_vars),
            )
        )

    def _record_atomic_rmw_event(
        self,
        symbolic_expr: SymbolicExpr,
        addr_expr: Z3Expr,
        expr_constraints: ConstraintConjunction,
        sem: str | None,
        scope: str | None,
        source_location: tuple[str, int, str] | None = None,
        *,
        semantic_constraints: tuple[Any, ...] = (),
        active: Any = True,
        old_value: Any = None,
        rmw_op: str | None = None,
        rmw_operand: Any = None,
        lane_position: tuple[Any, ...] | None = None,
    ) -> None:
        if self._unsupported_capture:
            return
        tensor = self._resolve_tensor(symbolic_expr)
        tensor_name = self._get_tensor_name(tensor) if tensor is not None else None
        if not self._note_written_tensor(tensor):
            return

        addr_expr = self._apply_finished_iter_subs(addr_expr)
        active = self._apply_finished_iter_subs(active)
        rmw_operand = self._apply_finished_iter_subs(rmw_operand)
        local = self._normalize_constraints(
            self._apply_finished_iter_subs(expr_constraints)
        )
        premises = self._normalize_constraints(
            self._apply_finished_iter_subs(semantic_constraints)
        )
        loop_vars = self._current_loop_iter_vars()
        if self._refs_unresolved_iter_var(
            (addr_expr, active, rmw_operand, local, premises), loop_vars
        ):
            self._mark_unsupported(
                "atomic_rmw references a finished loop iterator with no "
                "stable final value"
            )
            return

        # Raw symbolic template, mirroring the CAS record: old_value is the
        # fresh observation var (returned to the kernel by the overrider, so
        # downstream masks reference the SAME var), listed in
        # copy_local_vars for the per-copy alpha-rename. None for an
        # unmodeled (float-typed) RMW — the solver then keeps this write in
        # the unmodeled-writer set.
        self._append_record(
            AccessEventRecord(
                op_type=AtomicRMW,
                access_mode="read",
                tensor=tensor,
                tensor_name=tensor_name,
                symbolic_expr=symbolic_expr,
                dep_loads=self._dep_loads_of(symbolic_expr),
                lane_position=lane_position,
                addr_expr=addr_expr,
                premises=premises,
                local_constraints=local,
                source_location=source_location,
                grid_idx=None,
                program_seq=self._next_program_seq(),
                debug_name=self._debug_name(AtomicRMW, source_location),
                active=active,
                reads=True,
                writes=True,  # RMW always writes when active
                is_atomic=True,
                atomic_kind="rmw",
                sem=self._normalize_sem(sem),
                scope=self._normalize_scope(scope),
                old_value=old_value,
                written_value=None,
                rmw_op=rmw_op,
                rmw_operand=rmw_operand,
                event_id=self._next_event_id(),
                elem_size=self._infer_elem_size(symbolic_expr),
                cas_cmp_value=None,
                cas_new_value=None,
                copy_local_vars=normalize_copy_local_vars(
                    ((old_value,) if old_value is not None else ()) + loop_vars
                ),
            )
        )

    def _handle_access_check(
        self,
        expr: SymbolicExpr,
        op_type: type[Op],
        access_mode: AccessMode,
    ) -> None:
        """Capture a memory access expression.

        Outside any loop: recorded immediately. Inside a loop: deferred to the
        enclosing loop's flush point, with ``_make_event_signature`` used to
        dedupe events that repeat across iterations of the same loop.

        The pointer and mask expressions are evaluated separately, BOTH
        under the load-value provider: an embedded plain ``tl.load`` in the
        pointer chain lowers to a ``Select`` over the read-only launch
        snapshot (its domain facts ride the pointer's constraint conjunction
        into ``active``), and the mask becomes the event's ``active``
        condition so ``_lower_record`` can take per-lane lane-values rather
        than ``And``-collapsing a vector mask into a scalar local constraint.
        CAS/RMW returns and sort/cumsum results in a pointer still reject
        (snapshot-unstable).
        """
        if self._unsupported_capture or not self._capture_active():
            return
        # Reject addressing through snapshot-UNSTABLE runtime values
        # (atomic returns, sort/cumsum); plain-load indirection is admitted
        # and lowers to snapshot selects under the provider.
        ptr_attr = getattr(expr, "ptr", None)
        if self._reject_data_dependent_address(ptr_attr):
            return
        if ptr_attr is None:
            return
        ptr_result = self._safe_eval(ptr_attr, f"{op_type.__name__} ptr eval")
        if ptr_result is None:
            return
        z3_addr, ptr_constraints = ptr_result

        active_expr: Any = True
        mask_constraints: ConstraintConjunction = None
        mask_attr = getattr(expr, "mask", None)
        if mask_attr is not None:
            mask_result = self._safe_eval(mask_attr, f"{op_type.__name__} mask eval")
            if mask_result is None:
                return
            active_expr, mask_constraints = mask_result

        z3_constraints = _and_constraints(
            ptr_constraints,
            mask_constraints,
            *self._one_dimensional_lane_constraints(ptr_attr, mask_attr),
        )
        source_location = capture_current_source_location()
        lane_position = self._lane_position(ptr_attr, mask_attr)

        if not self.loop_stack:
            self._record_access_event(
                access_mode,
                op_type,
                z3_addr,
                z3_constraints,
                expr,
                source_location,
                active=active_expr,
                lane_position=lane_position,
            )
            return

        ctx = self.loop_stack[-1]
        loop_key = self._loop_key()
        signature = _make_event_signature(
            access_mode, source_location, z3_addr, z3_constraints, active_expr
        )
        # The k-th execution of a signature in one iteration is its own
        # operation (an unrolled tl.static_range body, a helper called
        # twice): it dedupes only with the k-th of the other iterations, so
        # each record keeps one position in the loop body.
        occurrence = self._iteration_occurrences.get((signature, loop_key), 0)
        self._iteration_occurrences[(signature, loop_key)] = occurrence + 1
        signature = hash((signature, occurrence))
        hit = (self._next_program_seq(), loop_key)
        pending_idx = ctx.signature_cache.get(signature)
        if pending_idx is None:
            ctx.signature_cache[signature] = len(ctx.pending_checks)
            ctx.pending_checks.append(
                PendingEvent(
                    symbolic_expr=expr,
                    addr_expr=z3_addr,
                    constraints=z3_constraints,
                    source_location=source_location,
                    access_mode=access_mode,
                    op_type=op_type,
                    active=active_expr,
                    program_seq=hit[0],
                    hits=[hit],
                    lane_position=lane_position,
                    occurrence=occurrence,
                )
            )
        else:
            cast(PendingEvent, ctx.pending_checks[pending_idx]).hits.append(hit)
            if cfg.verbose:
                print(f"[{self.LOG_TAG}]  ↪ skip duplicated addr in loop")

    @staticmethod
    def _lane_position(*roots: SymbolicExpr | None) -> tuple[Any, ...] | None:
        """The block position of the access's lanes, for
        AccessEventRecord.lane_position: per dimension, an arange feeding
        that dimension through splat, expand_dims, broadcast and
        element-wise ops, as ``var - start``. Any arange feeding a dimension
        is that dimension's index, so the first one found is kept; a later
        root (the mask after the pointer) only fills dimensions left open.
        None when a dimension has no arange or an op moves elements
        (reshape, trans, reduce, ...)."""
        elementwise = frozenset(
            (
                *SymbolicExpr.BINARY_OPS,
                *SymbolicExpr.UNARY_OPS,
                *SymbolicExpr.TERNARY_OPS,
                *SymbolicExpr.CAST_OPS,
                "addptr",
                "load",
            )
        )
        walked = elementwise | {"expand_dims", "broadcast"}

        def operands(node: SymbolicExpr) -> list[SymbolicExpr]:
            out = []
            for child in node.children.values():
                for c in child if isinstance(child, tuple) else (child,):
                    if isinstance(c, SymbolicExpr):
                        out.append(c)
            return out

        def merge(parts: list, rank: int) -> tuple[Any, ...] | None:
            out: tuple[Any, ...] = (None,) * rank
            for part in parts:
                if part is None:
                    return None
                out = tuple(x if x is not None else y for x, y in zip(out, part))
            return out

        def position(node: SymbolicExpr) -> tuple[Any, ...] | None:
            shape = tuple(node.shape or ())
            if isinstance(node, ArangeSymbolicExpr):
                return (node._to_z3()[0] - node.start.to_py(),)
            if not shape or node.op in ("const", "pid", "splat"):
                return (None,) * len(shape)
            if isinstance(node, ExpandDimsSymbolicExpr):
                inner = positions[id(node.arg)]
                axis = node.axis.to_py() % len(shape)
                return None if inner is None else inner[:axis] + (None,) + inner[axis:]
            if isinstance(node, BroadcastSymbolicExpr):
                inner = positions[id(node.arg)]
                arg_shape = tuple(node.arg.shape or ())
                if inner is None or len(arg_shape) != len(shape):
                    return None
                return tuple(
                    None if arg_dim == 1 != dim else coord
                    for coord, arg_dim, dim in zip(inner, arg_shape, shape)
                )
            if node.op not in elementwise:
                return None
            parts = []
            for child in operands(node):
                if tuple(child.shape or ()) == shape:
                    parts.append(positions[id(child)])
                elif child.shape:
                    return None
            return merge(parts, len(shape))

        positions: dict[int, tuple[Any, ...] | None] = {}
        stack = [(root, False) for root in roots if root is not None]
        while stack:
            node, expanded = stack.pop()
            if id(node) in positions:
                continue
            if not expanded and node.op in walked:
                stack.append((node, True))
                stack.extend((child, False) for child in operands(node))
                continue
            positions[id(node)] = position(node)
        if roots[0] is None or not roots[0].shape:
            return None
        shape = tuple(roots[0].shape)
        result = merge(
            [
                positions[id(root)]
                for root in roots
                if root is not None
                and tuple(root.shape or ()) == shape
                and positions[id(root)] is not None
            ],
            len(shape),
        )
        if result is None or any(coord is None for coord in result):
            return None
        return result

    @staticmethod
    def _one_dimensional_lane_constraints(
        *roots: SymbolicExpr | None,
    ) -> tuple[Any, ...]:
        """Couple independent arange calls denoting the same 1-D position.

        These are access-local facts, never launch-wide equalities between
        creation sites: a source arange can participate in different tile
        axes at other accesses. Restrict this normalization to uniformly
        1-D, elementwise pointer/mask DAGs; do not guess through a reshape,
        reduction, transpose, or a multidimensional intermediate.
        """
        allowed = (
            set(SymbolicExpr.BINARY_OPS)
            | set(SymbolicExpr.UNARY_OPS)
            | set(SymbolicExpr.CAST_OPS)
            | {
                "const",
                "pid",
                "arange",
                "load",
                "addptr",
                "where",
                "splat",
                "broadcast",
            }
        )
        seen: set[int] = set()
        ranges: list[ArangeSymbolicExpr] = []
        extents: set[int] = set()
        stack = [root for root in roots if root is not None]
        while stack:
            node = stack.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            if node.op not in allowed or len(node.shape) > 1:
                return ()
            if node.shape and node.shape[0] != 1:
                extents.add(node.shape[0])
            if isinstance(node, ArangeSymbolicExpr):
                ranges.append(node)
            for child in node.children.values():
                if isinstance(child, tuple):
                    stack.extend(child)
                elif child is not None:
                    stack.append(child)
        if len(ranges) < 2 or len(extents) != 1:
            return ()
        extent = next(iter(extents))
        if any(node.shape != (extent,) for node in ranges):
            return ()
        coordinates = [node._to_z3()[0] - node.start.to_py() for node in ranges]
        return tuple(coord == coordinates[0] for coord in coordinates[1:])

    def _handle_atomic_cas_check(
        self,
        expr: SymbolicExpr,
        sem: str | None,
        scope: str | None,
    ) -> None:
        if self._unsupported_capture or not self._capture_active():
            return
        # Loop check FIRST — before any .eval() can produce side effects
        # (ARANGE_DICT entries, fresh CAS-old vars, downstream sentinels).
        if self.loop_stack:
            self._mark_unsupported(
                "atomic_cas inside loop is unsupported by the two-copy solver"
            )
            return

        expr_atomic = cast(AtomicCasSymbolicExpr, expr)
        if self._reject_data_dependent_address(expr_atomic.ptr):
            return
        result = self._safe_eval(expr, "atomic_cas eval")
        if result is None:
            return
        old_value, expr_constraints = result
        result = self._safe_eval(expr_atomic.ptr, "atomic_cas ptr eval")
        if result is None:
            return
        # Keep the sub-eval constraint conjunctions explicitly (matching
        # the RMW site) — the full-CAS eval above already folded them via
        # the per-node cache, but that is a coincidence of caching, not a
        # contract. Snapshot domain facts for a lifted pointer MUST reach
        # the query.
        addr_expr, addr_constraints = result
        result = self._safe_eval(expr_atomic.cmp, "atomic_cas cmp eval")
        if result is None:
            return
        cmp_value, cmp_constraints = result
        result = self._safe_eval(expr_atomic.val, "atomic_cas val eval")
        if result is None:
            return
        value, val_constraints = result
        expr_constraints = self._combine_constraints(
            expr_constraints, addr_constraints, cmp_constraints, val_constraints
        )

        source_location = capture_current_source_location()
        try:
            self._record_atomic_cas_event(
                symbolic_expr=expr,
                addr_expr=addr_expr,
                expr_constraints=expr_constraints,
                cmp_value=cmp_value,
                value=value,
                old_value=old_value,
                sem=sem,
                scope=scope,
                source_location=source_location,
                lane_position=self._lane_position(expr_atomic.ptr),
            )
        except UnsupportedSymbolicRaceQuery as exc:  # an unknown memory scope
            self._mark_unsupported(str(exc))

    def _op_atomic_cas_overrider(
        self,
        ptr: Any,
        cmp: Any,
        val: Any,
        sem: str | None = None,
        scope: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> SymbolicExpr:
        ptr_sym = SymbolicExpr.from_value(ptr)
        cmp_sym = SymbolicExpr.from_value(cmp)
        val_sym = SymbolicExpr.from_value(val)
        ret = SymbolicExpr.create("atomic_cas", ptr_sym, cmp_sym, val_sym)
        self._handle_atomic_cas_check(ret, sem=sem, scope=scope)
        return ret

    def _handle_atomic_rmw_check(
        self,
        expr: SymbolicExpr,
        sem: str | None,
        scope: str | None,
        rmw_op: str | None = None,
    ) -> None:
        """``rmw_op`` non-None means the return value is MODELED: the
        expression itself is evaluated to the fresh observation var(s) —
        exactly like the CAS old value — and the operand is kept so the
        solver can model the write part f_op(old, v). ``rmw_op=None`` keeps
        the footprint-only record."""
        if self._unsupported_capture or not self._capture_active():
            return
        # Loop check FIRST — see _handle_atomic_cas_check for rationale.
        if self.loop_stack:
            self._mark_unsupported(
                "atomic_rmw inside loop is unsupported by the two-copy solver"
            )
            return

        expr_rmw = cast(AtomicRmwSymbolicExpr, expr)
        if self._reject_data_dependent_address(expr_rmw.ptr):
            return
        ptr_result = self._safe_eval(expr_rmw.ptr, "atomic_rmw ptr eval")
        if ptr_result is None:
            return
        addr_expr, addr_constraints = ptr_result

        if expr_rmw.mask is not None:
            mask_result = self._safe_eval(expr_rmw.mask, "atomic_rmw mask eval")
            if mask_result is None:
                return
            mask_z3, mask_constraints = mask_result
        else:
            mask_z3, mask_constraints = None, None

        old_value: Any = None
        rmw_operand: Any = None
        operand_constraints: ConstraintConjunction = None
        if rmw_op is not None:
            old_result = self._safe_eval(expr, "atomic_rmw eval")
            if old_result is None:
                return
            old_value, _ = old_result
            operand_result = self._safe_eval(expr_rmw.val, "atomic_rmw val eval")
            if operand_result is None:
                return
            rmw_operand, operand_constraints = operand_result

        expr_constraints = self._combine_constraints(
            addr_constraints, mask_constraints, operand_constraints
        )
        active = mask_z3 if mask_z3 is not None else True
        source_location = capture_current_source_location()

        try:
            self._record_atomic_rmw_event(
                symbolic_expr=expr,
                addr_expr=addr_expr,
                expr_constraints=expr_constraints,
                sem=sem,
                scope=scope,
                source_location=source_location,
                active=active,
                old_value=old_value,
                rmw_op=rmw_op,
                rmw_operand=rmw_operand,
                lane_position=self._lane_position(expr_rmw.ptr, expr_rmw.mask),
            )
        except UnsupportedSymbolicRaceQuery as exc:  # an unknown memory scope
            self._mark_unsupported(str(exc))

    @staticmethod
    def _atomic_rmw_return_dtype(
        ptr_sym: SymbolicExpr | tuple[SymbolicExpr, ...],
        val_sym: SymbolicExpr | tuple[SymbolicExpr, ...],
    ) -> Any:
        ptr_dtype = getattr(ptr_sym, "dtype", None)
        elem_ty = getattr(ptr_dtype, "element_ty", None)
        if elem_ty is not None:
            return elem_ty
        return getattr(val_sym, "dtype", None)

    @staticmethod
    def _normalize_rmw_op(rmw_op: Any) -> str | None:
        name = getattr(rmw_op, "name", None)
        if name is None:
            name = str(rmw_op) if rmw_op is not None else None
        if not name:
            return None
        return name.lower()

    @staticmethod
    def _is_integer_triton_dtype(dtype: Any) -> bool:
        """True for element types the Int-sort observation model can carry.
        Float-typed RMW returns stay on the sentinel; UNSIGNED types stay
        unmodeled too — modular wraparound is defined behavior there, and the
        unbounded-Int model would silently diverge from it. The symbolic
        engine hands SymbolicScalarDType (np_dtype-backed); raw triton dtypes
        (is_floating()/is_int()) are accepted too. Unknown dtypes fail
        closed."""
        if dtype is None:
            return False
        try:
            np_dtype = getattr(dtype, "np_dtype", None)
            if np_dtype is not None:
                # numpy kinds: i=int, b=bool; u (uint: wraps), f/c fail.
                return np_dtype.kind in ("i", "b")
            is_floating = getattr(dtype, "is_floating", None)
            if callable(is_floating) and is_floating():
                return False
            if str(getattr(dtype, "name", "") or "").startswith("uint"):
                return False
            is_int = getattr(dtype, "is_int", None)
            if callable(is_int) and is_int():
                return True
            is_bool = getattr(dtype, "is_bool", None)
            if callable(is_bool) and is_bool():
                return True
        except Exception:
            return False
        return False

    def _op_atomic_rmw_overrider(
        self,
        rmwOp: Any,
        ptr: Any,
        val: Any,
        mask: Any,
        sem: str | None = None,
        scope: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> SymbolicExpr:
        ptr_sym = SymbolicExpr.from_value(ptr)
        val_sym = SymbolicExpr.from_value(val)
        mask_sym = None if mask is None else SymbolicExpr.from_value(mask)
        event_expr = SymbolicExpr.create("atomic_rmw", ptr_sym, val_sym, mask_sym)
        # An INTEGER-typed RMW return is modeled — the record carries a fresh
        # observation var and the kernel gets the event expression back, so
        # downstream uses (mask = old == 0) reference the same var. The op
        # name is recorded even when only the observation is modelable (the
        # solver models the write part for add/max/min/xchg and keeps the
        # rest in the unmodeled-writer set). Addresses derived from the
        # return still fail-stop via _reject_data_dependent_address.
        op_name = self._normalize_rmw_op(rmwOp)
        dtype = self._atomic_rmw_return_dtype(ptr_sym, val_sym)
        modeled = op_name is not None and self._is_integer_triton_dtype(dtype)
        self._handle_atomic_rmw_check(
            event_expr, sem=sem, scope=scope, rmw_op=op_name if modeled else None
        )
        if modeled:
            return event_expr
        # Float-typed (or unknown-dtype) RMW: the return value's symbolic
        # semantics are NOT modeled. Downstream use triggers
        # UnsupportedSymbolicRaceQuery via the sentinel's _to_z3_impl, which
        # the wrapping _safe_eval converts into _mark_unsupported.
        return _UnsupportedRMWReturn(
            dtype=dtype,
            shape=getattr(ptr_sym, "shape", ()),
        )

    # ── Per-pending handler invoked from SymbolicClient's loop template

    def _process_pending_check(
        self,
        ctx: LoopContext,
        pending: PendingCheck,
        iter_constraints: list[BoolRef],
    ) -> None:
        if self._unsupported_capture:
            return
        # Items enqueued by _handle_access_check are PendingEvent instances
        # (subclass of PendingCheck) — narrow so attribute accesses are
        # type-safe under Literal["read", "write"].
        assert isinstance(pending, PendingEvent)
        # _loop_hook_after pops the flushed loop off loop_stack BEFORE calling
        # this, so _current_loop_iter_vars() only sees still-active outer
        # loops — the flushed loop's own iterator must come from ctx. Without
        # it the two-copy solver would not alpha-rename loop_i_* per copy,
        # pinning both program instances to the same iteration and missing
        # every cross-iteration cross-block race. The iterator's range
        # constraint travels alongside in iter_constraints -> premises, so
        # the renamed var stays bounded in each copy.
        self._record_access_event(
            pending.access_mode,
            pending.op_type,
            pending.addr_expr,
            pending.constraints,
            pending.symbolic_expr,
            pending.source_location,
            semantic_constraints=tuple(iter_constraints),
            copy_local_vars=(*self._current_loop_iter_vars(), ctx.idx_z3),
            active=pending.active,
            loop_flush=True,
            hits=pending.hits,
            program_seq=pending.program_seq,
            lane_position=pending.lane_position,
            occurrence=pending.occurrence,
        )


class NullRaceDetector(NullSymbolicClient, RaceDetector):
    """A do-nothing object returned when the race detector is off.
    Every callback raises via ``NullSymbolicClient`` so misuse is obvious.
    """

    def __init__(self, abort_on_error: bool = False, *args: Any, **kwargs: Any):
        super().__init__(abort_on_error=abort_on_error)
        # Distinguish "no race" from "race detector wasn't running": a Null
        # detector reports last_status == "disabled" so callers don't read
        # last_reports == [] as a clean pass.
        self.last_reports = []
        self.last_status = "disabled"
        self.unsupported_reason = "race detector disabled"
