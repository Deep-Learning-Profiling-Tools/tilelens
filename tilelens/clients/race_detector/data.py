from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import torch

from ...core.data import Op
from ...ir.verdict import ConfigVerdict, IRVerdict


MemorySem = Literal["plain", "relaxed", "acquire", "release", "acq_rel"]
AtomicKind = Literal["none", "cas", "rmw"]


class RaceType(str, Enum):
    # str-valued, so a RaceRecord holds (and a saved trace keeps) the name.
    RAW = "RAW"  # Read-After-Write
    WAR = "WAR"  # Write-After-Read
    WAW = "WAW"  # Write-After-Write


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
    # these pairs. A block anchor is listed only when it has a
    # lane_position.
    dep_loads: tuple[int, ...] = ()
    # The block position of a lane, one coordinate per dimension of the
    # access's block (the arange feeding that dimension, minus its start).
    # When both records of a dependency-ordered pair carry one, the
    # dependency edge and the DISTINCT-position query compare positions
    # through it. None: not derived.
    lane_position: tuple[Any, ...] | None = None

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
    # Each access's own address in the witness (they differ when the two
    # element sizes do).
    witness_addrs: tuple[int, int] | None = None
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


@dataclass(frozen=True)
class RaceAccess:
    """One access of a race witness."""

    op: str  # the Op's name: "load", "store", "atomic_cas" or "atomic_rmw"
    writes: bool  # whether it writes in the witness
    tensor_name: str | None
    source_location: tuple[str, int, str] | None  # (filename, lineno, func)
    # The element it touches in the witness, counted in elements from the
    # tensor's data pointer (the flat index for a contiguous tensor); None
    # when the tensor is unknown.
    element: int | None


@dataclass(frozen=True)
class RaceRecord:
    """A race the race detector found, as ``Launch.records`` holds it.

    Plain data, so ``tilelens.save()`` keeps it: no Z3 terms, no tensors.
    ``programs`` are the program ids of the two accesses in the witness
    (equal for a race inside one program instance).
    """

    race_type: str  # a RaceType value
    first: RaceAccess
    second: RaceAccess
    programs: tuple[tuple[int, int, int], tuple[int, int, int]]
    reason: str

    def __post_init__(self) -> None:
        # A RaceType member is held as its plain string, as a load returns it.
        object.__setattr__(self, "race_type", RaceType(self.race_type).value)


@dataclass(frozen=True)
class CompiledRaceRecord(RaceRecord):
    """A race the compiled race detector found in one config of a launch
    (``config``: its config kwargs, as RaceConfigVerdict.config). An
    access's ``element`` is counted from the launch tensor's data pointer."""

    config: Mapping[str, Any] = field(default_factory=dict)

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "config", dict(self.config))


@dataclass(frozen=True)
class RaceConfigVerdict(ConfigVerdict):
    """The compiled race detector's verdict on one config of a launch.

    ``status`` is "ok" (proved race-free), "races", "unsupported" or
    "vacuous" (no execution satisfies the premises, so no proof is
    claimed). For "ok", ``rung`` is the proof's: "T0" (any scalar
    parameters and any grid), "T1" (this launch's parameters, any grid)
    or "T1-launch" (this launch's parameters and grid). "Any grid" is any
    extent along the grid axes the kernel reads (a program id or the
    program count) and this launch's extent along the others (a kernel
    with atomics: any extent along those too, except at T1 with a launch
    grid). ``qualifiers`` name what a proof or a race assumes or read:
    "assumes no i32 overflow" (the widths whose obligations can fail in
    the proof's domain; they hold at this launch), "assumes termination"
    (the kernel spins on a flag), "reads tensor contents" (it holds for
    the launch's pre-launch contents of the tensors it read). ``grid_fragile`` are the races the
    query found at some other grid behind a "T1-launch" proof, or behind an
    "unsupported" config whose query at the launch's grid was not decided:
    evidence only, never races of this launch."""

    rung: str | None = None
    qualifiers: tuple[str, ...] = ()
    grid_fragile: tuple[CompiledRaceRecord, ...] = ()

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "qualifiers", tuple(self.qualifiers))
        object.__setattr__(self, "grid_fragile", tuple(self.grid_fragile))


@dataclass(frozen=True)
class RaceVerdict(IRVerdict):
    """The compiled race detector's verdict on one traced launch: ``status``
    "races" if a config races, else "unsupported" if a config was not
    decided, else "vacuous" if a config's proof would be vacuous, else
    "ok". For "ok", ``scope`` says what the weakest config's proof holds
    for ("any-params-any-grid", "this-params-any-grid" or
    "this-params-this-grid"; "any grid" as RaceConfigVerdict.rung says),
    and ``qualifiers`` gathers the configs'."""

    qualifiers: tuple[str, ...] = ()

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "qualifiers", tuple(self.qualifiers))


__all__ = [
    "AccessEventRecord",
    "AtomicKind",
    "CompiledRaceRecord",
    "MemorySem",
    "RaceAccess",
    "RaceConfigVerdict",
    "RaceRecord",
    "RaceReport",
    "RaceType",
    "RaceVerdict",
]
