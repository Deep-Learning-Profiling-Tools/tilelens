from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Literal

import torch

from ...core.data import Op


MemorySem = Literal["plain", "relaxed", "acquire", "release", "acq_rel"]
AtomicKind = Literal["none", "cas", "rmw"]


class RaceType(Enum):
    RAW = auto()  # Read-After-Write
    WAR = auto()  # Write-After-Read
    WAW = auto()  # Write-After-Write


@dataclass
class AccessEventRecord:
    op_type: type[Op]
    access_mode: Literal["read", "write"]
    tensor: torch.Tensor | None = None
    tensor_name: str | None = None
    symbolic_expr: Any = None
    addr_expr: Any = None
    premises: tuple[Any, ...] = field(default_factory=tuple)
    local_constraints: tuple[Any, ...] = field(default_factory=tuple)
    source_location: tuple[str, int, str] | None = None
    grid_idx: tuple[int, ...] | None = None
    program_seq: int = -1
    debug_name: str | None = None
    active: Any = True
    reads: Any = None
    writes: Any = None
    is_atomic: bool = False
    atomic_kind: AtomicKind = "none"
    sem: MemorySem = "plain"
    scope: str | None = None
    old_value: Any = None
    written_value: Any = None

    # Two-copy solver fields.
    event_id: int = -1  # stable dedup key (per launch)
    elem_size: int = 1  # for byte-overlap when > 1

    # Pre-exit representative (the await abstraction): True for
    # the value-model-free twin of an awaited poll that stands for ALL
    # FAILED iterations of the spin. It mirrors the poll's footprint,
    # activity and program_seq (equal seq = mutually po-unordered with the
    # poll) under a fresh event_id, and carries NO value model — so no
    # rf/sw edge can enter or leave it, preserving the failed iterations'
    # unorderedness. The solver excludes it from the unmodeled-
    # overlapping-writer escape and skips the (rep, own poll)
    # intra-instance pair.
    pre_exit: bool = False

    # CAS-specific raw symbolic pieces. None for non-CAS records.
    cas_cmp_value: Any = None
    cas_new_value: Any = None

    # RMW value modeling. ``rmw_op`` is the canonical lowercase
    # op name ("add", "max", "min", "xchg", ...; "exch" is normalized to
    # "xchg" at capture). ``rmw_operand`` is the symbolic operand v so the
    # write part can be modeled as f_op(old, v). Both stay None for a
    # non-RMW record or an RMW whose value semantics are not modeled (float
    # ops, bitwise and/or/xor, unsigned umax/umin) — the solver then keeps
    # the record's write in the UNMODELED-writer set (rf_unknown escape),
    # which is the over-report direction. When the return value is modeled,
    # ``old_value`` holds the fresh observation var o_r (alpha-renamed per
    # copy via ``copy_local_vars``, exactly like the CAS return).
    rmw_op: str | None = None
    rmw_operand: Any = None

    # Z3 vars representing per-program-instance nondeterminism for THIS record
    # (the fresh CAS return var, this record's loop iterator vars). The two-copy
    # solver collects these across all records and alpha-renames each ORIGINAL
    # var exactly once per copy (launch-level), so downstream records that
    # reference the same var get the same _a/_b rename. Tensor base pointers,
    # kernel scalar args, and global constants are explicitly NOT included.
    copy_local_vars: tuple[Any, ...] = field(default_factory=tuple)

    # Dependency order: event_ids of the load / atomic records whose
    # RESULT this record's value, mask, or compare operand consumes through
    # an element-wise chain (no reduce, scan, broadcast, reshape, transpose,
    # join/split, splat in between). At the same position such a pair is
    # ordered by causality (the value must exist before it is written), so
    # the intra-instance query ranges over DISTINCT positions only for
    # these pairs.
    dep_loads: tuple[int, ...] = ()

    # Operational control conditions used for value-causality dependencies,
    # before execution-domain premises are conjoined into event activity.
    # None uses local_constraints/premises instead, which can contain
    # real masks and loop control. Frontends with separate domain facts must
    # supply only actual control conditions (for example loop existence).
    # active, addresses and atomic operands are tracked separately.
    causal_constraints: tuple[Any, ...] | None = None

    # Earlier await records that must exit before this operation when both
    # execute in one program copy. These are conditional control edges:
    # the solver must conjoin both endpoints' activity/path predicates,
    # rather than treating source order alone as an unconditional dependency.
    await_dependencies: tuple[int, ...] = ()
    # Observation-producing records consumed by this await's exit predicate,
    # including the poll and any separately read expected/threshold value.
    await_observations: tuple[int, ...] = ()


@dataclass(frozen=True)
class RaceReport:
    """Race detected between two memory events.

    ``first`` and ``second`` are the two events of the witness; each carries
    its ``AccessEventRecord`` as ``.record``.
    """

    first: Any
    second: Any
    model: dict[str, str] = field(default_factory=dict)
    reason: str = ""
    race_type_value: RaceType | None = None
    witness_addr: int | None = None
    witness_grid_a: tuple[int, int, int] | None = None
    witness_grid_b: tuple[int, int, int] | None = None

    @property
    def first_record(self) -> AccessEventRecord:
        return self.first.record

    @property
    def second_record(self) -> AccessEventRecord:
        return self.second.record

    @property
    def race_type(self) -> RaceType:
        # TwoCopySymbolicHBSolver, the only producer, always sets it.
        assert self.race_type_value is not None
        return self.race_type_value


__all__ = [
    "AccessEventRecord",
    "AtomicKind",
    "MemorySem",
    "RaceReport",
    "RaceType",
]
