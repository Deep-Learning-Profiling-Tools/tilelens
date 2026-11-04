"""Two-copy symbolic HB solver.

The race detector's race-finder. The solver duplicates each
recorded access into two symbolic program-instances ``A`` and ``B`` (alpha-
renaming PIDs, ``tl.arange`` lane vars, and per-record copy-local vars such as
the CAS return), then asks Z3 whether any pair of cross-copy events is
unordered, in conflict, and aliasing.

Model boundary (atomic read-from sources):
  A value-modeled atomic reader takes its value from the initial contents
  (when identifiable), from a value-modeled atomic writer event of the two
  copies, or from the open world: any thread of the grid (another lane of
  either modeled instance, or a third instance) running a CAS or a
  value-modeled xchg / max / min, lowered under fresh pid, lane and
  copy-local variables. Within one copy the events of one arange variable
  are on one lane, so without the open world a write by another lane of the
  same instance, or by a third instance, could never be the source. An
  open-world writer synchronizes only with what its own instance orders
  before it (a fence, or a dependency at its position); synchronization
  that reaches it through a third instance is not modeled (over-reported).
  When no initial source is identifiable, or an unmodeled write (plain
  store, wrap-capable or unmodeled atomic RMW) can overlap the location,
  ``rf_unknown_R`` admits any value but does NOT enable a synchronizes-with
  edge, so the reader's old value is never over-constrained. Identity
  writes and counted records need no open-world source (see
  ``_open_world_writer_records``). The closed world remains only for the
  compiled frontends' pre-exit and await-gated records.

  Thread identity: within one copy, two records' events run on one thread
  only at a known same position of two blocks of one shape
  (``_same_thread``). A record whose lane variables do not name its lanes
  (no ``lane_position``: a block on one address, a broadcast, transposed or
  reshaped pointer) is never identified with one thread by its variables:
  it has distinct lanes at any assignment, its two copies in one instance
  may or may not be one operation, it gets no open-world exclusion, and
  coherence and rf candidacy give it no program order.

Address-domain invariant:
  ``record.addr_expr`` consumed by this solver MUST be a byte address matching
  ``tensor.data_ptr()``. ``byte_overlap`` and ``initial_atomic_source`` rely on
  this. Capture-side normalisation must convert element / tensor-relative
  offsets to byte addresses BEFORE the records reach the solver.

Z3 context:
  Every term lives in Z3's main context. Records must be lowered there (a
  term from another context fails with Z3's "Context mismatch"), and since
  that context is shared and not thread-safe, the caller serializes solves.

Intra-instance duplicate lanes:
  Cross-copy queries assert ``different_blocks``, so they can never witness
  two lanes of one store colliding inside a single program instance — and a
  grid=(1,1,1) launch makes ``different_blocks`` UNSAT outright. A separate
  same-instance query pins ``pid_a == pid_b``; loop iterators stay free, and
  an atomic's observation is shared only at one lane of its record.
  Requiring a lane-identity difference then asks whether two DISTINCT lanes
  of the same access conflict, in any two iterations. Distinct operations
  of one instance are concurrently in flight unless the caller's
  ``fence_between`` hook or a dependency (at one position and iteration)
  orders them, so unordered pairs of distinct operations enter the same
  query, also across iterations. The caller's ``fence_epoch`` hook keeps
  only the executions no fence separates.

Z3 ``unknown`` policy:
  Every query runs on its own Solver with a per-Solver timeout
  (``timeout_ms``); no global Z3 parameter is changed. ``unknown`` (a
  timeout included) is never treated as unsat. A race query that comes back
  undecided raises :class:`UnsupportedSymbolicRaceQuery` (the launch reports
  ``unsupported`` instead of a silent clean verdict), and an undecided
  overlap in the unmodeled-writer check opens the ``rf_unknown`` escape.

RMW value modeling:
  An integer AtomicRMW with a modeled op (``add``/``max``/``min``/``xchg``)
  gets a fresh observation symbol ``o_r`` (``record.old_value``, alpha-
  renamed per copy exactly like the CAS return) and a modeled write part
  ``f_op(o_r, v)`` (``written_value``). ``o_r`` is justified like a CAS
  observation: rf candidates are the value-modeled atomic writers, the
  initial-value source, and the unknown source when an unmodeled writer can
  overlap. Value-modeled atomics join the per-location atomic order, which
  makes RMW **atomicity/immediacy** structural: one order position covers
  the read AND the write part, and the existing "no successful same-address
  writer strictly between rf source and reader" constraint is exactly the
  immediacy axiom. Two extensions build on it:

  - **Reads-through (release sequences over RMW chains)**: sw(w, r) also
    holds when r reads-through w via a bounded chain of modeled RMW write
    parts (each link an rf edge). Chains through UNMODELED grid instances
    are covered only by the counting axiom below.
  - **Counting axiom** (guarded): for a location L touched by EXACTLY ONE
    always-active scalar constant-increment ``add`` RMW record with a known
    initial value and no other possible writer, per-copy rank variables
    r ∈ [0, |G|) satisfy ``o = init + c·rank``, rank equality iff same
    block, and ``co ⇔ rank<``. Under those guards every write to L between
    two instances is a link of the same RMW chain, so reads-through is
    equivalent to coherence order — giving last-block-done and work-queue
    disjointness. A counted reader also gets an ``rf_chain`` source choice
    (an unmodeled instance at rank-1) so real races at non-adjacent ranks
    are not squeezed out of the closed world; ``rf_chain`` itself never
    enables synchronizes-with. If ANY guard fails the axiom is omitted
    entirely (over-report direction), and an observation variable used in
    an ADDRESS position without counting support raises
    ``UnsupportedSymbolicRaceQuery`` instead of widening silently.

Limitations (current):
  - **Initial atomic source covers scalar tensors and small contiguous flag
    arrays** (``numel <= _MAX_INITIAL_ATOMIC_ELEMENTS = 1024``). Larger or
    non-contiguous tensors fall through to ``rf_unknown_R``, which
    deliberately does NOT enable synchronizes-with; guarded acq/rel CAS over
    them is reported as races conservatively.
  - **Two program instances, plus open-world writers** — a third block is
    an rf source, but synchronization that travels through it
    (writer-via-third-block CAS chains, a lock passed on) is not modeled;
    the counting axiom is the one guarded exception.
  - **One symbolic lane for a record whose lanes are not named** (``z +
    tl.zeros([B])``, a broadcast or transposed pointer): its lanes are
    taken to be distinct and unordered, which over-reports (two columns of
    one broadcast row writing one value, say).
  - **Float-typed RMW returns and bitwise/unsigned RMW ops are not value-
    modeled** (``fadd``, float max, ``and``/``or``/``xor``, ``umax``/
    ``umin``): they get no modeled write part, so the record's write stays
    in the unmodeled-writer set.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from .guarded_division import GuardedDivisionApplicabilityCache
    from .snapshot_lemmas import SnapshotLemmaCache

from z3 import (
    And,
    AtMost,
    Bool,
    BoolVal,
    Const,
    ExprRef,
    If,
    Implies,
    Int,
    IntVal,
    Not,
    Or,
    Solver,
    is_false,
    is_true,
    sat,
    simplify,
    unsat,
)
from z3.z3 import BoolRef, IntNumRef, ModelRef

from .conflict_simplification import (
    _PureSelectExpressionCache,
    conflict_impossible,
    conflict_precheck_features,
)
from .data import AccessEventRecord, RaceReport, RaceType
from .hb_common import (
    CountingScopeUnsupported,
    UnsupportedSymbolicRaceQuery,
    apply_sub,
    as_bool,
    build_transitive_hb,
    conflicting_access_modes,
    is_acquire_sem,
    is_release_sem,
    iter_constraints,
    lane_value,
    modeled_atomic_read_from,
    normalize_copy_local_vars,
    to_lanes,
)
from .report_diagnostics import append_missing_fence_diagnostic


@dataclass(frozen=True)
class CopyContext:
    label: str  # "a" or "b"
    pid: tuple[Any, Any, Any]
    pid_substitutions: tuple[tuple[Any, Any], ...]
    arange_substitutions: tuple[tuple[Any, Any], ...]
    arange_constraints: tuple[Any, ...]
    copy_local_substitutions: tuple[tuple[Any, Any], ...]  # launch-level


def _immutable_precheck_source(value: Any) -> bool:
    """Admit only immutable leaves and containers for identity-based reuse."""
    if type(value) in (bool, int, float):
        return True
    if isinstance(value, ExprRef):
        # User-defined subclasses can override coercion with mutable state.
        # Standard z3py expression classes represent immutable ASTs.
        return type(value).__module__ == ExprRef.__module__
    if type(value) is tuple:
        return all(_immutable_precheck_source(item) for item in value)
    if type(value) is CopyContext:
        # Frozen records can still contain mutable fields supplied by callers.
        # These are exactly the fields read by the common precheck below.
        return all(
            _immutable_precheck_source(field)
            for field in (
                value.pid,
                value.copy_local_substitutions,
                value.arange_substitutions,
            )
        )
    return False


@dataclass
class _ConflictPrecheckCommon:
    # Retaining the source objects makes identity checks safe against id reuse.
    sources: tuple[Any, ...]
    expression: BoolRef
    shared: tuple[tuple[Any, Any], ...]
    correspondence: tuple[tuple[Any, Any], ...]
    has_snapshot: bool | None = None
    intra_expression: BoolRef | None = None


@dataclass(frozen=True)
class _BaseConstraintCache:
    # Keep the objects alive: comparisons use Python identity, never Z3 equality.
    snapshot: tuple[tuple[Any, ...], ...]
    expression: BoolRef


@dataclass(frozen=True)
class SymbolicMemoryEvent:
    idx: int
    copy: str
    record: AccessEventRecord
    name: str
    lane: int
    event_id: int
    program_seq: int
    pid: tuple[Any, Any, Any]
    addr: Any
    elem_size: int
    active: BoolRef
    reads: BoolRef
    writes: BoolRef
    is_atomic: bool
    atomic_kind: str
    sem: str
    scope: str | None
    old_value: Any = None
    written_value: Any = None
    # The substituted RMW operand v (None for non-RMW / unmodeled RMW);
    # the counting axiom reads the constant increment off it.
    rmw_operand: Any = None
    # Substituted operational conditions, excluding proof-domain premises.
    # Reads/writes retain those premises for all execution constraints;
    # causality must not mistake them for observation dependencies.
    causal_inputs: tuple[Any, ...] = ()


def _import_symbolic_expr_pids():
    # Local import: SymbolicExpr is a heavy module; keep tests cheap.
    from ..symbolic_engine import SymbolicExpr

    return (SymbolicExpr.PID0, SymbolicExpr.PID1, SymbolicExpr.PID2)


def _is_symbolic_dim(d: Any) -> bool:
    """A grid dim that is a Z3 expression rather than a Python int.

    Must be an isinstance check: duck-typing on ``sort`` misfires on numpy
    scalars (ndarray.sort) and would leave them un-coerced where the old
    ``int(d)`` handled them."""
    return isinstance(d, ExprRef)


@dataclass(frozen=True)
class _CountingInfo:
    """The counting axiom fired for one RMW record: per-copy
    rank variables tied to the observation values and the atomic order."""

    idx_a: int
    idx_b: int
    rank_a: Any
    rank_b: Any
    init: int
    inc: int
    loc: int


def _rmw_written_value(op: str | None, old: Any, v: Any) -> Any:
    """The modeled write part f_op(old, v) of an RMW, or ``None`` when the
    op has no Int-sort model: non-identity bitwise and/or/xor need
    bitvectors, unsigned umax/umin diverge from the signed Int order, and
    float ops are outside the integer model. The one bitwise case with an
    exact Int model is the identity or/xor of a provably-zero operand
    (f(old, 0) = old), the write-back shape of an identity-RMW await poll;
    a symbolic or nonzero operand keeps ``None``. ``None`` keeps the
    record's write in the unmodeled-writer set (rf_unknown escape) — the
    over-report direction."""
    if op is None or old is None or v is None:
        return None
    if op == "add":
        return old + v
    if op == "max":
        return If(old >= v, old, v)
    if op == "min":
        return If(old <= v, old, v)
    if op == "xchg":
        return v
    if op in ("or", "xor") and _as_numeral(v) == 0:
        return old
    return None


def _as_numeral(v: Any) -> int | None:
    """``v`` as a concrete int if it simplifies to one, else None."""
    if v is None:
        return None
    if isinstance(v, (bool, int)):
        return int(v)
    try:
        s = simplify(v)
    except Exception:
        return None
    if isinstance(s, IntNumRef):
        return s.as_long()
    return None


def _z3_var_key(v: Any) -> tuple[int, str, str]:
    # Mirrors the dedup key used by hb_common.normalize_copy_local_vars.
    return (v.hash(), str(v.sort()), v.decl().name())


def _collect_z3_var_keys(values: tuple[Any, ...]) -> set[tuple[int, str, str]]:
    """Keys of every 0-ary leaf in ``values``. Numeral leaves are included
    but can never collide with a variable's key."""
    seen: set[tuple[int, str, str]] = set()
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
            try:
                seen.add(_z3_var_key(v))
            except Exception:
                pass
            continue
        stack.extend(v.children())
    return seen


class TwoCopySymbolicHBSolver:
    """Two-copy symbolic happens-before solver.

    Construct it over one launch's records, then call :meth:`find_races`
    (and :meth:`check_feasibility` before claiming race freedom). Both the
    constructor and the queries raise :class:`UnsupportedSymbolicRaceQuery`
    (or its subclass :class:`CountingScopeUnsupported`) when a record cannot
    be lowered or a query is undecided. See the module docstring for the
    model boundary and address-domain invariants.
    """

    # Per-Solver Z3 budget of every query this solver creates (milliseconds).
    # The enumeration's per-case and retry limits below only ever shorten
    # it. Never a global Z3 parameter: a solve must not change Z3's
    # defaults for other clients.
    TIMEOUT_MS: ClassVar[int] = 180_000
    ENUM_MAX_CASES: ClassVar[int] = 1024
    ENUM_CASE_TIMEOUT_MS: ClassVar[int] = 5000
    ENUM_TOTAL_BUDGET_S: ClassVar[float] = 60.0
    # Once one query has been decided by enumeration the launch's claim
    # is already scoped to its extent, so a LATER query's long symbolic
    # attempt can no longer buy a stronger claim: it gets this short
    # budget instead and falls through to the (near-free) enumeration.
    # Applied only when the cross case split fits ENUM_MAX_CASES, so a
    # query whose split would overflow keeps its full symbolic budget.
    ENUM_RETRY_TIMEOUT_MS: ClassVar[int] = 10_000

    def __init__(
        self,
        records: list[AccessEventRecord],
        *,
        grid: tuple[Any, ...],
        arange_dict: dict[Any, Any] | None = None,
        extra_assumptions: tuple[Any, ...] = (),
        only_pairs: frozenset[tuple[int, int]] | None = None,
        enum_fallback_grid: tuple[int, int, int] | None = None,
        launch_ceiling: bool = False,
        fence_between: Callable[[AccessEventRecord, AccessEventRecord], bool]
        | None = None,
        fence_epoch: Callable[[AccessEventRecord], Any] | None = None,
        timeout_ms: int | None = None,
    ) -> None:
        self.records = list(records)
        # ``fence_epoch(record)``: the number of fences one instance has
        # executed before an execution of ``record``, an int or a term over
        # the record's loop iterators. Two executions of one instance with
        # equal epochs have no fence between them; with different epochs a
        # fence orders them. None: no epoch constraint.
        self.fence_epoch = fence_epoch
        # Fence-ordered intra-instance semantics: program order between two
        # records of one instance is a happens-before edge only when a
        # fence orders them, and same-instance pairs of distinct operations
        # that no fence orders enter the intra-instance race query. Which fence orders which pair is the
        # caller's rule: ``fence_between(first, second)`` is asked for two
        # records of one instance with ``first.program_seq <
        # second.program_seq``. None: no fence orders any pair.
        self.fence_between = fence_between
        self._fence_cache: dict[tuple[int, int], bool] = {}
        self._uses_arange_cache: dict[int, bool] = {}
        self.timeout_ms = self.TIMEOUT_MS if timeout_ms is None else int(timeout_ms)
        if self.timeout_ms <= 0:
            # Z3 reads timeout=0 as "no limit".
            raise ValueError(f"timeout_ms must be positive, got {timeout_ms}")
        # Dependency order: unordered record pairs (load, dependent) whose
        # same-position accesses are ordered by causality.
        self.dep_pairs: frozenset[tuple[int, int]] = frozenset(
            (int(d), int(r.event_id))
            for r in self.records
            for d in getattr(r, "dep_loads", ())
        )
        self.grid = self._normalize_grid(grid)
        self.arange_dict = dict(arange_dict or {})
        self.extra_assumptions = tuple(extra_assumptions)
        # Enumeration fallback: when a race query returns Z3-unknown and
        # a concrete launch grid was provided, the query is re-asked as
        # an exhaustive case split over concrete pid assignments at that
        # extent. All-UNSAT decides the query AT THE LAUNCH EXTENT ONLY
        # (the caller must degrade the claim to launch scope: enum_used
        # is set); a SAT case is a normal witness with in-extent pids.
        # Refused (the original unknown propagates) when the case
        # count exceeds ENUM_MAX_CASES, a case is itself undecided,
        # or ENUM_TOTAL_BUDGET_S is exhausted: fail-closed to the
        # pre-fallback behavior.
        self.enum_fallback_grid = enum_fallback_grid
        self.enum_used = False
        self._enum_deadline: float | None = None
        # True when this solver's strongest possible claim is ALREADY
        # launch-scoped (the pinned requery): then a long symbolic
        # attempt can never buy more than the enumeration fallback
        # delivers, so the short budget applies from the first query,
        # not only after the first enumeration.
        self.launch_ceiling = launch_ceiling
        # Requery restriction (sound by UNSAT monotonicity): when set,
        # only event pairs whose UNORDERED record-id pair is listed are
        # queried. The caller may use this ONLY when every omitted pair
        # is already known UNSAT under a WEAKER system (fewer
        # assumptions): adding constraints (here: grid pins in
        # extra_assumptions) cannot turn an UNSAT query SAT, so the
        # omitted pairs contribute no reports either way. Keyed by
        # record event_id and unordered, so every lane and both
        # orientations of a listed record pair are still queried.
        self.only_pairs = only_pairs
        # The launch's premises after per-copy substitution (loop-iterator
        # ranges and, decisively, the awaited exit predicates): the
        # feasibility query asserts them, because activity gating folds
        # them into every event, so the bare base system is always
        # satisfiable through the all-inactive (non-terminating)
        # valuation, which does not satisfy the premises. Filled during
        # record lowering, before any constraint family is built.
        self.launch_premises: list[BoolRef] = []

        # 1. PID vars + substitutions for both copies.
        pid_a, pid_b = self._make_pid_vars()
        pid_subs_a, pid_subs_b = self._make_pid_subs(pid_a, pid_b)

        # 2. Grid bounds + different-block constraints.
        (
            self.grid_constraints,
            self.different_blocks,
        ) = self._make_grid_and_diff_block_constraints(pid_a, pid_b)

        # 3. Arange substitutions + range constraints from the snapshot.
        (
            arange_subs_a,
            arange_subs_b,
            arange_consts_a,
            arange_consts_b,
        ) = self._make_arange_subs_and_constraints()
        self.arange_constraints_a = tuple(arange_consts_a)
        self.arange_constraints_b = tuple(arange_consts_b)

        # 4. LAUNCH-LEVEL copy-local substitutions (union over all records).
        copy_local_subs_a, copy_local_subs_b = self._make_launch_copy_local_subs()

        # 5. Build the two CopyContexts (frozen).
        self.ctx_a = CopyContext(
            label="a",
            pid=tuple(pid_a),
            pid_substitutions=tuple(pid_subs_a),
            arange_substitutions=tuple(arange_subs_a),
            arange_constraints=self.arange_constraints_a,
            copy_local_substitutions=tuple(copy_local_subs_a),
        )
        self.ctx_b = CopyContext(
            label="b",
            pid=tuple(pid_b),
            pid_substitutions=tuple(pid_subs_b),
            arange_substitutions=tuple(arange_subs_b),
            arange_constraints=self.arange_constraints_b,
            copy_local_substitutions=tuple(copy_local_subs_b),
        )

        # 6. Lower every record under both contexts.
        self.events: list[SymbolicMemoryEvent] = self._lower_two_copies()
        self._event_of: dict[tuple[int, str, int], SymbolicMemoryEvent] = {
            (id(e.record), e.copy, e.lane): e for e in self.events
        }
        # Observation vars (CAS / RMW returns) among the copy-local vars; the
        # other copy-local vars are loop iterators.
        copy_local_keys = {
            _z3_var_key(orig) for orig, _ in self.ctx_a.copy_local_substitutions
        }
        self._observation_keys: set[tuple[int, str, str]] = set()
        for record in self.records:
            self._observation_keys |= (
                _collect_z3_var_keys((record.old_value,)) & copy_local_keys
            )
        self._record_by_id = {record.event_id: record for record in self.records}
        self._record_cache: dict[tuple[str, int], Any] = {}
        self._anchor_pin_cache: dict[int, tuple[BoolRef, ...]] = {}
        self._same_thread_cache: dict[tuple[int, int], BoolRef | None] = {}
        self._epoch_cache: dict[int, tuple[Any, Any]] = {}

        # 7. Atomic-order vars, counting axioms, RF source booleans, and the
        # reads-through relation, BEFORE building the HB closure (HB reads
        # self.reads_through for synchronizes_with). Order matters: the
        # counting axiom's applicability is consulted while building rf
        # choices (it adds the rf_chain source), and the reads-through
        # closure folds in both the direct rf edges and the counting pairs.
        self.atomic_order: dict[int, Any] = self._make_atomic_order_vars()
        self.rf_source: dict[tuple[int, int], BoolRef] = {}
        self.rf_init_source: dict[int, BoolRef] = {}
        self.rf_unknown_source: dict[int, BoolRef] = {}
        self.rf_chain_source: dict[int, BoolRef] = {}
        # Open-world rf choices: (selector, reader, writer record, the
        # writer's event in a fresh context, that context, the condition
        # the selector implies).
        self.rf_open_world: list[
            tuple[
                BoolRef,
                SymbolicMemoryEvent,
                AccessEventRecord,
                SymbolicMemoryEvent,
                CopyContext,
                BoolRef,
            ]
        ] = []
        self._open_world_edges: dict[tuple[int, int], list[BoolRef]] = {}
        self.rf_constraints: list[BoolRef] = []
        self.atomic_coherence_constraints: list[BoolRef] = []
        self.counting_constraints: list[BoolRef] = []
        self._counting_grid_restricted: set[int] = set()
        self._counting: dict[int, _CountingInfo] = self._build_counting_axioms()
        self._build_read_from_choices()
        self._build_open_world_edges()
        self._build_atomic_coherence_constraints()
        # wf-vc (value causality / no out-of-thin-air): its own family —
        # coherence is per-location order, value causality is a
        # cross-location well-formedness axiom of the value model itself.
        self.vc_read_rank: dict[int, Any] = {}
        self.vc_write_rank: dict[int, Any] = {}
        self.value_causality_constraints: list[BoolRef] = []
        self._build_value_causality_constraints()
        self._assert_no_uncounted_observation_addresses()
        self.reads_through: dict[tuple[int, int], BoolRef] = self._build_reads_through()

        # 8. Build HB transitive closure (synchronizes_with reads
        # reads_through).
        self.hb = build_transitive_hb(self.events, self._edge)

        # 9. Coherence hb-consistency axiom (co-hb) — needs self.hb, hence
        # after step 8; appended to atomic_coherence_constraints, which is
        # consumed only at query time in _base_solver.
        self._build_coherence_hb_constraints()

    # ──────────────────────── Public API ────────────────────────

    _CROSS_INSTANCE_REASON: str = (
        "unordered conflicting memory accesses across two symbolic "
        "program instances under the current symbolic assumptions"
    )
    _INTRA_INSTANCE_REASON: str = (
        "conflicting lanes of a single program instance touch the same "
        "bytes with no defined intra-instance order"
    )

    def find_races(self) -> list[RaceReport]:
        """Race witnesses across two instances and within one, one report
        per unordered (record, lane) pair whose race query is SAT."""
        import time as _time

        events_a = [e for e in self.events if e.copy == "a"]
        events_b = [e for e in self.events if e.copy == "b"]
        # (kind, seconds, sat) per executed query.
        self.query_stats: list[tuple[str, float, bool]] = []
        # Only full symbolic UNSAT results are reusable. Keep their ASTs
        # alive, and limit the cache to this race-finding invocation.
        self._unsat_race_queries: dict[tuple[int, int], BoolRef] = {}
        self._guarded_query_cache: dict[BoolRef, BoolRef] = {}
        # Structural summaries retain immutable ASTs only for this invocation.
        self._guarded_division_applicability_cache: GuardedDivisionApplicabilityCache | None = None
        self._snapshot_lemma_cache: SnapshotLemmaCache | None = None

        candidates: list[tuple[SymbolicMemoryEvent, SymbolicMemoryEvent, ModelRef, str]]
        candidates = []
        for a in events_a:
            for b in events_b:
                if self._pair_excluded(a, b):
                    continue
                precheck_start = _time.perf_counter()
                if self._conflict_precheck(a, b):
                    self.query_stats.append(
                        ("cross", _time.perf_counter() - precheck_start, False)
                    )
                    continue
                precheck_s = _time.perf_counter() - precheck_start
                race_expression = self._race_expr(a, b)
                normalization_start = _time.perf_counter()
                normalized = self._normalized_race_query(
                    race_expression, (self.different_blocks,)
                )
                known_unsat = self._known_unsat_race_query(normalized)
                precheck_s += _time.perf_counter() - normalization_start
                if known_unsat:
                    self.query_stats.append(("cross", precheck_s, False))
                    continue

                def _build(a=a, b=b, normalized=normalized) -> Solver:
                    if normalized is None:
                        s = self._new_solver()
                        s.add(self._race_expr(a, b))
                    else:
                        s = self._solver()
                        s.add(normalized)
                    return s

                solver = _build()
                self._cap_symbolic_retry(solver)
                t0 = _time.perf_counter()
                model: ModelRef | None = None
                try:
                    is_sat = self._race_query_is_sat(solver, a, b)
                    if is_sat:
                        model = solver.model()
                    else:
                        self._remember_unsat_race_query(normalized)
                except UnsupportedSymbolicRaceQuery as exc:
                    is_sat, model = self._enumerate_pair(_build, False, exc)
                self.query_stats.append(
                    ("cross", precheck_s + _time.perf_counter() - t0, is_sat)
                )
                if is_sat:
                    candidates.append((a, b, model, self._CROSS_INSTANCE_REASON))

        candidates.extend(self._find_intra_instance_candidates(events_a, events_b))
        return self._dedupe_reports(candidates)

    def check_feasibility(self, extra: tuple[Any, ...] = ()) -> bool:
        """Feasibility half of the race-freedom certificate: does the base
        system admit any execution at all?

        The m^2 per-pair race queries are all UNSAT even when the
        premises themselves are unsatisfiable (an await's termination
        premise no execution can meet), and a proof claimed over an
        empty set of executions would be vacuous. The certificate is
        feasibility AND race freedom, and this one satisfiability query
        of the base constraints WITH the launch's premises asserted
        discharges the first conjunct. The premises
        must be asserted explicitly: activity gating folds them into
        every event, so the bare base system always admits the
        all-inactive valuation (the non-terminating execution), and a bare
        check would be trivially satisfiable.
        Deliberately WITHOUT ``different_blocks``: a
        single-instance launch is feasible even though the
        cross-instance constraint is unsatisfiable on its grid.

        Z3 ``unknown`` must not certify feasibility — escalate to
        :class:`UnsupportedSymbolicRaceQuery` exactly like a race
        query, so the caller reports the launch as unsupported rather
        than certified.
        """
        solver = self._base_solver()
        for p in self.launch_premises:
            solver.add(p)
        for c in extra:
            solver.add(as_bool(c))
        result = solver.check()
        if result == sat:
            return True
        if result == unsat:
            return False
        detail = solver.reason_unknown()
        raise UnsupportedSymbolicRaceQuery(
            "Z3 could not decide the feasibility query"
            + (f" ({detail})" if detail else ""),
            reason_unknown=detail,
        )

    def _enum_pid_cases(
        self, same_instance: bool
    ) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]] | None:
        """Concrete pid assignments covering the fallback grid, or None
        when no fallback grid is set or the case count exceeds
        ENUM_MAX_CASES."""
        g = self.enum_fallback_grid
        if g is None:
            return None
        n = g[0] * g[1] * g[2]
        count = n if same_instance else n * (n - 1)
        if count <= 0 or count > self.ENUM_MAX_CASES:
            return None
        pids = [
            (x, y, z) for z in range(g[2]) for y in range(g[1]) for x in range(g[0])
        ]
        if same_instance:
            return [(p, p) for p in pids]
        return [(pa, pb) for pa in pids for pb in pids if pa != pb]

    def _enumerate_pair(
        self,
        build_solver: Any,
        same_instance: bool,
        original: Exception,
    ) -> tuple[bool, ModelRef | None]:
        """Decide one Z3-undecided pair by exhaustive concrete-pid case
        split at the fallback grid's extent.

        ``build_solver`` rebuilds the pair's solver exactly as the
        querying loop did (base system + pair constraints + race
        expression); each case then pins both copies' pid triples to
        concrete values, so the disjunction of the cases is exactly the
        original query WITH the grid bounded to the fallback extent.
        All cases UNSAT therefore decides the query AT THAT EXTENT (the
        caller must scope the claim accordingly); any SAT case yields a
        normal model whose witness pids are in-extent by construction.
        ``original`` (the symbolic attempt's unknown) is re-raised
        whenever the split cannot be completed: too many cases, a case
        itself undecided, or the total budget exhausted — fail-closed
        to the pre-fallback behavior.
        """
        import time as _time

        cases = self._enum_pid_cases(same_instance)
        if cases is None:
            raise original
        self.enum_used = True
        if self._enum_deadline is None:
            self._enum_deadline = _time.monotonic() + self.ENUM_TOTAL_BUDGET_S
        deadline = self._enum_deadline
        pid_a = [Int(f"pid_a_{i}") for i in range(3)]
        pid_b = [Int(f"pid_b_{i}") for i in range(3)]
        for pa, pb in cases:
            if _time.monotonic() > deadline:
                raise original
            solver = build_solver()
            assignments = tuple(
                (pid, IntVal(value, ctx=pid.ctx))
                for pids, values in ((pid_a, pa), (pid_b, pb))
                for pid, value in zip(pids, values)
            )
            # Substitute into EVERY original assertion, including activity,
            # HB/rf, array and domain premises. Constant PIDs can eliminate
            # nested address div/mod before the arithmetic solver sees them.
            # Rebuilding first also observes any changed source constraints.
            expression = simplify(apply_sub(And(*solver.assertions()), assignments))
            solver.reset()
            solver.add(expression)
            # Keep the original PID symbols in SAT models. Together these
            # pins and the substituted assertions are equivalent to the old
            # assertions plus pins, including all non-PID witness variables.
            solver.add(*(pid == value for pid, value in assignments))
            solver.set(timeout=min(self.timeout_ms, self.ENUM_CASE_TIMEOUT_MS))
            result = solver.check()
            if result == sat:
                return True, solver.model()
            if result != unsat:
                raise original
        return False, None

    def _cap_symbolic_retry(self, solver: Solver) -> None:
        """Cap this symbolic attempt's budget (ENUM_RETRY_TIMEOUT_MS)
        once a long attempt can no longer buy a stronger claim: after
        the first enumeration (the claim is degraded to launch scope),
        or from the start when the solver's claim ceiling is launch
        scope by construction (the pinned requery). Gated on the CROSS
        split fitting ENUM_MAX_CASES: then every possible unknown in
        this solver is enumerable, so the cap cannot lose a decision."""
        if (self.enum_used or self.launch_ceiling) and self._enum_pid_cases(
            False
        ) is not None:
            solver.set(timeout=min(self.timeout_ms, self.ENUM_RETRY_TIMEOUT_MS))

    def _pair_excluded(self, a: SymbolicMemoryEvent, b: SymbolicMemoryEvent) -> bool:
        """True when an ``only_pairs`` restriction is active and this
        record pair is not in it (see ``__init__``); the pair's queries
        are then skipped as already-known UNSAT."""
        if self.only_pairs is None:
            return False
        lo, hi = sorted((a.event_id, b.event_id))
        return (lo, hi) not in self.only_pairs

    def _normalized_race_query(
        self, race_expression: BoolRef, pair_constraints: tuple[Any, ...]
    ) -> BoolRef | None:
        """Normalize the complete query before constructing a fresh solver.

        All base premises, the cross/same-instance conditions, lane identity,
        and the original race expression are retained. Unsupported container
        or coercion inputs use the original construction path instead.
        """
        if not all(
            isinstance(c, BoolRef) or type(c) is bool  # noqa: E721 (only immutable builtin scalars)
            for c in (race_expression, *pair_constraints)
        ):
            return None
        base = self._base_constraint_conjunction()
        if base is None:
            return None
        from .guarded_division import (
            GuardedDivisionApplicabilityCache,
            guarded_division_normal_form,
        )

        original = simplify(And(base, *pair_constraints, race_expression))
        cache = getattr(self, "_guarded_query_cache", None)
        if cache is None:
            self._guarded_query_cache = cache = {}
        if original not in cache:
            applicability_cache = getattr(
                self, "_guarded_division_applicability_cache", None
            )
            if applicability_cache is None:
                applicability_cache = (
                    self._guarded_division_applicability_cache
                ) = GuardedDivisionApplicabilityCache()
            cache[original] = guarded_division_normal_form(
                original, applicability_cache=applicability_cache
            )
        return cache[original]

    @staticmethod
    def _race_query_cache_key(expression: BoolRef) -> tuple[int, int]:
        # AST ids are unique only within a context. The cache retains the
        # expression (and its context), preventing either id from recycling.
        return id(expression.ctx), expression.get_id()

    def _known_unsat_race_query(self, expression: BoolRef | None) -> bool:
        if expression is None:
            return False
        cached = getattr(self, "_unsat_race_queries", None)
        return cached is not None and self._race_query_cache_key(expression) in cached

    def _remember_unsat_race_query(self, expression: BoolRef | None) -> None:
        if expression is None:
            return
        cached = getattr(self, "_unsat_race_queries", None)
        if cached is None:
            self._unsat_race_queries = cached = {}
        cached[self._race_query_cache_key(expression)] = expression

    @staticmethod
    def _race_query_is_sat(
        solver: Solver, a: SymbolicMemoryEvent, b: SymbolicMemoryEvent
    ) -> bool:
        """``solver.check()`` with Z3 ``unknown`` made conservative.

        ``unknown`` (timeout, nonlinear give-up) must not collapse into
        unsat: dropping the pair would turn an undecided query into a
        silent clean "ok" verdict. There is no witness model to report
        either, so escalate to :class:`UnsupportedSymbolicRaceQuery`; the
        caller reports the launch as unsupported instead of race-free.
        """
        result = solver.check()
        if result == sat:
            return True
        if result == unsat:
            return False
        detail = solver.reason_unknown()
        raise UnsupportedSymbolicRaceQuery(
            f"Z3 could not decide the race query for {a.name} vs {b.name}"
            + (f" ({detail})" if detail else ""),
            reason_unknown=detail,
        )

    def _find_intra_instance_candidates(
        self,
        events_a: list[SymbolicMemoryEvent],
        events_b: list[SymbolicMemoryEvent],
    ) -> list[tuple[SymbolicMemoryEvent, SymbolicMemoryEvent, ModelRef, str]]:
        """Duplicate-lane conflicts inside a single program instance.

        See the module docstring: cross-copy queries assert
        ``different_blocks`` and therefore cannot witness these (under
        grid=(1,1,1) they are vacuously unsat). An atomic op's lanes
        serialize, so the intra-instance hazards are duplicate addresses
        across the lanes of a single non-atomic store, plus pairs of
        distinct operations that neither the caller's fence rule
        (``fence_between``) nor a dependency orders, including pairs with
        equal or unset sequence numbers.
        """
        import time as _time

        same_instance = self._same_instance_constraints()
        out: list[tuple[SymbolicMemoryEvent, SymbolicMemoryEvent, ModelRef, str]]
        out = []
        for a in events_a:
            for b in events_b:
                if self._pair_excluded(a, b):
                    continue
                lane_cond = self._intra_pair_lane_condition(a, b)
                if lane_cond is None:
                    continue
                precheck_start = _time.perf_counter()
                if self._conflict_precheck(
                    a, b, same_instance=True, lane_cond=lane_cond
                ):
                    if hasattr(self, "query_stats"):
                        self.query_stats.append(
                            ("intra", _time.perf_counter() - precheck_start, False)
                        )
                    continue
                precheck_s = _time.perf_counter() - precheck_start
                race_expression = self._race_expr(a, b)
                normalization_start = _time.perf_counter()
                normalized = self._normalized_race_query(
                    race_expression, (*same_instance, lane_cond)
                )
                known_unsat = self._known_unsat_race_query(normalized)
                precheck_s += _time.perf_counter() - normalization_start
                if known_unsat:
                    if hasattr(self, "query_stats"):
                        self.query_stats.append(("intra", precheck_s, False))
                    continue

                def _build(
                    a=a, b=b, lane_cond=lane_cond, normalized=normalized
                ) -> Solver:
                    if normalized is None:
                        s = self._base_solver()
                        for c in same_instance:
                            s.add(c)
                        s.add(lane_cond)
                        s.add(self._race_expr(a, b))
                    else:
                        s = self._solver()
                        s.add(normalized)
                    return s

                solver = _build()
                self._cap_symbolic_retry(solver)
                t0 = _time.perf_counter()
                model: ModelRef | None = None
                try:
                    is_sat = self._race_query_is_sat(solver, a, b)
                    if is_sat:
                        model = solver.model()
                    else:
                        self._remember_unsat_race_query(normalized)
                except UnsupportedSymbolicRaceQuery as exc:
                    is_sat, model = self._enumerate_pair(_build, True, exc)
                if hasattr(self, "query_stats"):
                    self.query_stats.append(
                        ("intra", precheck_s + _time.perf_counter() - t0, is_sat)
                    )
                if is_sat:
                    out.append((a, b, model, self._INTRA_INSTANCE_REASON))
        return out

    def _conflict_precheck(
        self,
        a: SymbolicMemoryEvent,
        b: SymbolicMemoryEvent,
        *,
        same_instance: bool = False,
        lane_cond: BoolRef | None = None,
    ) -> bool:
        """Prove a necessary conflict condition UNSAT without changing HB.

        The single private entry point of the precheck. It never changes
        enumeration state or participates in feasibility certification.
        Actual byte overlap and activity stay in the weaker condition;
        only the full solver supplies SAT witnesses.
        """
        if is_false(a.writes) and is_false(b.writes):
            return True
        # Read/read pairs (and structurally exempt atomic pairs) have no
        # conflict independently of their potentially large address DAGs.
        if (
            self._ordinary_access_mode(a) is None
            or self._ordinary_access_mode(b) is None
        ) and is_true(simplify(Not(conflicting_access_modes(a, b)))):
            return True
        features = getattr(self, "_conflict_precheck_events", None)
        if features is None:
            features = self._conflict_precheck_events = {
                event.idx: conflict_precheck_features(event.addr)
                for event in self.events
            }
        common = self._conflict_precheck_common()
        # Even linear addresses otherwise enter array solving when shared
        # snapshot premises contain Selects. The same necessary-condition
        # precheck can prove them disjoint without the full array/HB query.
        if not any(features[a.idx]) and not any(features[b.idx]):
            if common.has_snapshot is None:
                common.has_snapshot = conflict_precheck_features(common.expression)[1]
            if not common.has_snapshot:
                return False
        simplify_first = not (features[a.idx][0] or features[b.idx][0])
        conflict = self._conflict(a, b)
        if same_instance:
            assert lane_cond is not None
            # In one instance, pids agree. Lane variables, iterators and
            # observations MUST remain independent.
            if common.intra_expression is None:
                common.intra_expression = apply_sub(common.expression, common.shared)
            conditions = [
                common.intra_expression,
                apply_sub(conflict, common.shared),
                apply_sub(lane_cond, common.shared),
            ]
        else:
            conditions = [common.expression, conflict, self.different_blocks]
        expression_cache = None
        if simplify_first:
            expression_cache = getattr(self, "_conflict_expression_cache", None)
            if expression_cache is None:
                expression_cache = (
                    self._conflict_expression_cache
                ) = _PureSelectExpressionCache()
        from .snapshot_lemmas import SnapshotLemmaCache

        snapshot_cache = getattr(self, "_snapshot_lemma_cache", None)
        if snapshot_cache is None:
            snapshot_cache = self._snapshot_lemma_cache = SnapshotLemmaCache()
        return conflict_impossible(
            conditions,
            common.correspondence,
            same_instance=same_instance,
            simplify_first=simplify_first,
            expression_cache=expression_cache,
            snapshot_cache=snapshot_cache,
        )

    def _conflict_precheck_common(self) -> _ConflictPrecheckCommon:
        """Reuse the immutable common conjunction, never a query decision.

        The constructor stores constraint sequences as tuples and contexts
        as frozen records. Attribute replacement invalidates the cached
        conjunction, including removal of launch pins. Cache admission also
        checks their contents: mutable or custom coercible conditions must
        be read on every call, even inside an unchanged tuple. Admission is
        checked only when building a new entry, preserving cheap cache hits.
        """
        sources = (
            self.grid_constraints,
            self.arange_constraints_a,
            self.arange_constraints_b,
            self.extra_assumptions,
            self.ctx_a,
            self.ctx_b,
        )
        cached = getattr(self, "_conflict_common_cache", None)
        if cached is not None and all(
            current is old for current, old in zip(sources, cached.sources)
        ):
            return cached
        self._conflict_common_cache = None
        # One instance shares its pids only: its two lane roles may be at
        # different loop iterations and see different atomic results.
        shared = tuple(zip(self.ctx_b.pid, self.ctx_a.pid))
        correspondence = shared + tuple(
            (bvar, avar)
            for (_, avar), (_, bvar) in zip(
                self.ctx_a.copy_local_substitutions + self.ctx_a.arange_substitutions,
                self.ctx_b.copy_local_substitutions + self.ctx_b.arange_substitutions,
            )
        )
        expression = And(
            self.grid_constraints,
            *self.arange_constraints_a,
            *self.arange_constraints_b,
            *[as_bool(condition) for condition in self.extra_assumptions],
        )
        cached = _ConflictPrecheckCommon(sources, expression, shared, correspondence)
        if all(_immutable_precheck_source(source) for source in sources):
            self._conflict_common_cache = cached
        return cached

    def _intra_pair_lane_condition(
        self, a: SymbolicMemoryEvent, b: SymbolicMemoryEvent
    ) -> BoolRef | None:
        """Lane-identity constraint for an intra-instance pair, or ``None``
        when the pair cannot race within one instance (fence-ordered,
        serialized, never writes, or the symmetric duplicate of an
        already-queried pair).

        The two copies of one instance may be at different iterations of
        a loop, so the condition also requires equal fence epochs: no fence
        executed between the two executions.
        """
        if a.record is b.record:
            # Lanes of one atomic op serialize against each other; a
            # load's duplicate lanes read-read and cannot conflict.
            if a.record.is_atomic or a.record.access_mode != "write":
                return None
            if a.lane > b.lane:
                return None  # symmetric duplicate
            if a.lane < b.lane:
                return self._with_epochs(a, b, BoolVal(True))  # distinct lanes
            # One lane rewriting at another iteration is program order.
            differ = self._lane_identity_differs(a)
            return None if differ is None else self._with_epochs(a, b, differ)
        # Distinct ops of one instance are concurrently in flight unless
        # fence_between or a dependency orders them.
        if a.event_id > b.event_id:
            return None  # symmetric duplicate
        if (a.record.pre_exit or b.record.pre_exit) and a.program_seq == b.program_seq:
            # The pre-exit representative vs its OWN awaited record (they
            # share the poll's seq): a thread's poll does not race its own
            # failed iterations. Do NOT rely on the conflict exemption —
            # for a cta-scoped poll it is void and the equal-seq pair
            # would surface as a same-instance race against itself; for
            # device-scoped polls the pair is exempt-UNSAT anyway, so
            # skipping is uniform and cheaper.
            return None
        if a.program_seq >= 0 and b.program_seq >= 0 and a.program_seq != b.program_seq:
            if self._fence_orders(a, b):
                return None
            if (a.event_id, b.event_id) in self.dep_pairs or (
                b.event_id,
                a.event_id,
            ) in self.dep_pairs:
                # Dependency order: the same position IN ONE ITERATION is
                # ordered by causality, so only distinct positions or
                # distinct iterations can race.
                parts = [self._lanes_differ(a, b)]
                shared = {
                    _z3_var_key(orig) for orig, _, _ in self._record_iterators(a.record)
                }
                parts += [
                    var_a != var_b
                    for orig, var_a, var_b in self._record_iterators(b.record)
                    if _z3_var_key(orig) in shared
                ]
                parts = [p for p in parts if p is not None]
                if not parts:
                    return None
                return self._with_epochs(
                    a, b, parts[0] if len(parts) == 1 else Or(*parts)
                )
            # Fence-ordered semantics: two distinct operations of one
            # instance with no fence between them are concurrently in
            # flight, so the pair is queried like any conflicting pair.
        return self._with_epochs(a, b, BoolVal(True))

    def _with_epochs(
        self, a: SymbolicMemoryEvent, b: SymbolicMemoryEvent, cond: BoolRef
    ) -> BoolRef | None:
        """``cond`` restricted to executions of one instance that no fence
        separates (equal fence epochs), or ``None`` when every pair of
        executions is separated."""
        if self.fence_epoch is None:
            return cond
        epoch_a = self._epoch(a.record)[0]
        epoch_b = self._epoch(b.record)[1]
        if isinstance(epoch_a, int) and isinstance(epoch_b, int):
            return cond if epoch_a == epoch_b else None
        same = epoch_a == epoch_b
        return same if is_true(cond) else And(cond, same)

    def _epoch(self, record: AccessEventRecord) -> tuple[Any, Any]:
        """The record's fence epoch in copy a and in copy b (an int in
        both when it is constant). An epoch the caller leaves unknown,
        ``fence_epoch_unknown_*``, is no copy-local variable, so it is the
        same in both copies: harmless, it constrains nothing."""
        cached = self._epoch_cache.get(id(record))
        if cached is None:
            assert self.fence_epoch is not None
            epoch = self.fence_epoch(record)
            if isinstance(epoch, int):
                cached = (epoch, epoch)
            else:
                cached = (
                    apply_sub(epoch, self.ctx_a.copy_local_substitutions),
                    apply_sub(epoch, self.ctx_b.copy_local_substitutions),
                )
            self._epoch_cache[id(record)] = cached
        return cached

    @staticmethod
    def _positions_equal(
        rec1: AccessEventRecord,
        sub1: tuple[tuple[Any, Any], ...],
        rec2: AccessEventRecord,
        sub2: tuple[tuple[Any, Any], ...],
    ) -> BoolRef | None:
        """A lane of ``rec1`` (lane variables renamed by ``sub1``) and one
        of ``rec2`` (by ``sub2``) are at the same block position, compared
        dimension by dimension; ``None`` unless both records carry a
        ``lane_position`` of one rank. The single position comparison of
        the solver: one copy (dependency edge), two copies (distinct
        positions), or a fresh context (open-world sources)."""
        pos1, pos2 = rec1.lane_position, rec2.lane_position
        if not (pos1 and pos2 and len(pos1) == len(pos2)):
            return None
        dims = [apply_sub(p, sub1) == apply_sub(q, sub2) for p, q in zip(pos1, pos2)]
        return dims[0] if len(dims) == 1 else And(*dims)

    def _lanes_differ(
        self, a: SymbolicMemoryEvent, b: SymbolicMemoryEvent
    ) -> BoolRef | None:
        """Constraint that the a-copy lane of ``a`` and the b-copy lane of
        ``b`` are DIFFERENT positions, over the arange vars either event
        uses; ``None`` when neither has a lane (scalar accesses: one
        position, dependency-ordered, never queried).

        When both records carry a ``lane_position`` of one rank, the
        positions are compared dimension by dimension instead: each access
        may index its block with its own aranges (every block-pointer
        access does), and then some variable always differs, whatever the
        two lanes' positions."""
        same = self._positions_equal(
            a.record,
            self.ctx_a.arange_substitutions,
            b.record,
            self.ctx_b.arange_substitutions,
        )
        if same is not None:
            return Not(same)
        occurring = _collect_z3_var_keys((a.addr, a.active, b.addr, b.active))
        diffs = [
            var_a != var_b
            for (_, var_a), (_, var_b) in zip(
                self.ctx_a.arange_substitutions, self.ctx_b.arange_substitutions
            )
            if _z3_var_key(var_a) in occurring or _z3_var_key(var_b) in occurring
        ]
        if not diffs:
            return None
        return diffs[0] if len(diffs) == 1 else Or(*diffs)

    def _lane_identity_differs(self, e: SymbolicMemoryEvent) -> BoolRef | None:
        """Constraint that the a/b copies of ``e`` denote two DIFFERENT
        lanes of its record, or ``None`` for a true scalar access (no
        second lane exists).

        Each arange summary var is injective in the lane index, so any one
        of the record's arange vars differing across the copies witnesses
        two distinct lanes. Vars in the activity condition count too: a
        store whose address ignores the lane still has its lanes
        distinguished by the mask. A record whose lane variables do not
        name its lanes (_lanes_not_named) has distinct lanes at any
        assignment: the B lanes of ``z + tl.zeros([B])``, or the columns of
        a broadcast row pointer that share its row variable.
        """
        if self._lanes_not_named(e.record):
            return BoolVal(True)
        occurring = _collect_z3_var_keys((e.addr, e.active, e.writes))
        diffs = [
            var_a != var_b
            for (_, var_a), (_, var_b) in zip(
                self.ctx_a.arange_substitutions, self.ctx_b.arange_substitutions
            )
            if _z3_var_key(var_a) in occurring or _z3_var_key(var_b) in occurring
        ]
        if not diffs:
            return None
        return diffs[0] if len(diffs) == 1 else Or(*diffs)

    def _same_instance_constraints(self) -> tuple[BoolRef, ...]:
        """Pin the b copy onto the a copy's program instance.

        ``pid_a == pid_b`` makes the two copies the same block. Loop
        iterators stay free: the two lane roles may be at different
        iterations, and the pair's fence epochs and dependency condition
        decide which iteration pairs race. An atomic's observation is one
        dynamic operation's result only at one lane and iteration of its
        record, so it is shared exactly there.
        """
        cons: list[BoolRef] = [self.ctx_a.pid[i] == self.ctx_b.pid[i] for i in range(3)]
        renamed = {
            _z3_var_key(orig): (var_a, var_b)
            for (orig, var_a), (_, var_b) in zip(
                self.ctx_a.copy_local_substitutions, self.ctx_b.copy_local_substitutions
            )
        }
        seen: set[tuple[int, str, str]] = set()
        for record in self.records:
            keys = _collect_z3_var_keys((record.old_value,)) & self._observation_keys
            keys -= seen
            if not keys:
                continue
            seen |= keys
            same_op = self._same_op_condition(record)
            for key in sorted(keys):
                var_a, var_b = renamed[key]
                cons.append(
                    var_a == var_b
                    if is_true(same_op)
                    else Implies(same_op, var_a == var_b)
                )
        return tuple(cons)

    def _record_lane_vars(self, record: AccessEventRecord) -> tuple[Any, ...]:
        """The original arange variables a record's fields use: together
        with the pid, they name one lane of the record."""
        key = ("lane_vars", id(record))
        cached = self._record_cache.get(key)
        if cached is None:
            occurring = _collect_z3_var_keys(
                (
                    record.addr_expr,
                    record.active,
                    record.reads,
                    record.writes,
                    record.local_constraints,
                    record.old_value,
                    record.rmw_operand,
                    record.cas_cmp_value,
                    record.cas_new_value,
                    record.lane_position,
                )
            )
            cached = self._record_cache[key] = tuple(
                orig
                for orig, _ in self.ctx_a.arange_substitutions
                if _z3_var_key(orig) in occurring
            )
        return cached

    def _record_iterators(
        self, record: AccessEventRecord
    ) -> list[tuple[Any, Any, Any]]:
        """(original, copy-a, copy-b) of the record's loop iterators."""
        mine = {_z3_var_key(v) for v in record.copy_local_vars}
        return [
            (orig, var_a, var_b)
            for (orig, var_a), (_, var_b) in zip(
                self.ctx_a.copy_local_substitutions, self.ctx_b.copy_local_substitutions
            )
            if _z3_var_key(orig) in mine
            and _z3_var_key(orig) not in self._observation_keys
        ]

    def _lane_equal(
        self,
        record: AccessEventRecord,
        sub_x: tuple[tuple[Any, Any], ...],
        sub_y: tuple[tuple[Any, Any], ...],
    ) -> BoolRef:
        """Two renamings of ``record`` are at the same lane."""
        eqs = [
            apply_sub(v, sub_x) == apply_sub(v, sub_y)
            for v in self._record_lane_vars(record)
        ]
        if not eqs:
            return BoolVal(True)
        return eqs[0] if len(eqs) == 1 else And(*eqs)

    def _same_op_condition(self, record: AccessEventRecord) -> BoolRef:
        """Within one instance, the copy-a and copy-b events of one record
        lane are one dynamic operation exactly when they are at the same
        lane and the same loop iteration. Equal lane variables name one
        lane only when they name the record's lanes; otherwise
        (_lanes_not_named) a free choice keeps both worlds: one operation,
        or two lanes at equal variables."""
        key = ("same_op", id(record))
        cached = self._record_cache.get(key)
        if cached is None:
            lane = self._lane_equal(
                record, self.ctx_a.arange_substitutions, self.ctx_b.arange_substitutions
            )
            eqs = [] if is_true(lane) else [lane]
            eqs += [
                var_a == var_b for _, var_a, var_b in self._record_iterators(record)
            ]
            if self._lanes_not_named(record):
                eqs.append(Bool(f"same_op_choice_{record.event_id}"))
            if not eqs:
                cached = BoolVal(True)
            else:
                cached = eqs[0] if len(eqs) == 1 else And(*eqs)
            self._record_cache[key] = cached
        return cached

    # ──────────────────────── Construction ────────────────────────

    @staticmethod
    def _normalize_grid(grid: tuple[Any, ...]) -> tuple[Any, Any, Any]:
        """Concrete launches pass ints; a static front-end passes Z3
        Ints so the verdict covers EVERY grid (each symbolic dim gets a
        ``>= 1`` bound in the grid constraints)."""
        dims = [d if _is_symbolic_dim(d) else int(d) for d in grid]
        while len(dims) < 3:
            dims.append(1)
        return (dims[0], dims[1], dims[2])

    @staticmethod
    def _make_pid_vars():
        pid_a = [Int(f"pid_a_{i}") for i in range(3)]
        pid_b = [Int(f"pid_b_{i}") for i in range(3)]
        return pid_a, pid_b

    @staticmethod
    def _make_pid_subs(pid_a, pid_b):
        orig = _import_symbolic_expr_pids()
        sub_a = tuple((orig[i], pid_a[i]) for i in range(3))
        sub_b = tuple((orig[i], pid_b[i]) for i in range(3))
        return sub_a, sub_b

    def _make_grid_and_diff_block_constraints(self, pid_a, pid_b):
        grid_constraints = And(
            *[And(pid_a[i] >= 0, pid_a[i] < self.grid[i]) for i in range(3)],
            *[And(pid_b[i] >= 0, pid_b[i] < self.grid[i]) for i in range(3)],
            # A symbolic dim needs its own lower bound or a zero/negative
            # grid would make every pid constraint vacuously unsat and turn
            # any query into a false proof.
            *[d >= 1 for d in self.grid if _is_symbolic_dim(d)],
        )
        different_blocks = Or(
            pid_a[0] != pid_b[0],
            pid_a[1] != pid_b[1],
            pid_a[2] != pid_b[2],
        )
        return grid_constraints, different_blocks

    def _make_arange_subs_and_constraints(self):
        self._arange_ranges: dict[tuple[int, str, str], tuple[Any, Any]] = {}
        sub_a, sub_b = [], []
        cons_a, cons_b = [], []
        for key, value in self.arange_dict.items():
            # ARANGE_DICT entry shape: key=(start, end) or
            # (start, end, filename, lineno) — the engine keys interned vars
            # by creation site so independent same-range arange instances stay
            # distinct. value=(orig_var, _). Per-copy names derive from the
            # original var's name so every dict entry renames uniquely.
            try:
                start, end = key[0], key[1]
                orig_var = value[0] if isinstance(value, (list, tuple)) else value
                base_name = orig_var.decl().name()
            except Exception:
                continue
            var_a = Int(f"{base_name}__a")
            var_b = Int(f"{base_name}__b")
            self._arange_ranges[_z3_var_key(orig_var)] = (start, end)
            sub_a.append((orig_var, var_a))
            sub_b.append((orig_var, var_b))
            cons_a.append(And(var_a >= start, var_a < end))
            cons_b.append(And(var_b >= start, var_b < end))
        return sub_a, sub_b, cons_a, cons_b

    def _make_launch_copy_local_subs(self):
        all_vars = normalize_copy_local_vars(
            v for r in self.records for v in r.copy_local_vars
        )
        subs_a, subs_b = [], []
        for i, v in enumerate(all_vars):
            base = f"{v.decl().name()}__{i}__{v.hash()}"
            subs_a.append((v, Const(f"{base}__a", v.sort())))
            subs_b.append((v, Const(f"{base}__b", v.sort())))
        return tuple(subs_a), tuple(subs_b)

    # ──────────────────────── Lowering ────────────────────────

    def _lower_two_copies(self) -> list[SymbolicMemoryEvent]:
        events: list[SymbolicMemoryEvent] = []
        for ctx in (self.ctx_a, self.ctx_b):
            for record in self.records:
                events.extend(self._lower_record(record, ctx, len(events)))
        return events

    @staticmethod
    def _active_access_mode(active: BoolRef, condition: BoolRef) -> BoolRef:
        """Fold literal mode conditions while retaining the full activity guard."""
        if is_true(condition):
            return active
        if is_false(condition):
            return condition
        return And(active, condition)

    def _lower_record(
        self,
        record: AccessEventRecord,
        ctx: CopyContext,
        start_idx: int,
        *,
        launch_premises: bool = True,
    ) -> list[SymbolicMemoryEvent]:
        if record.addr_expr is None:
            raise UnsupportedSymbolicRaceQuery(
                "AccessEventRecord.addr_expr is required for two-copy lowering"
            )

        sub = (
            ctx.pid_substitutions
            + ctx.arange_substitutions
            + ctx.copy_local_substitutions
        )

        addr_all = apply_sub(record.addr_expr, sub)
        active_all = apply_sub(record.active, sub)
        local_all = apply_sub(record.local_constraints, sub)
        prem_all = apply_sub(record.premises, sub)
        causal_all = (
            (local_all, prem_all)
            if record.causal_constraints is None
            else apply_sub(record.causal_constraints, sub)
        )

        addr_lanes = to_lanes(addr_all)
        n_lanes = len(addr_lanes) or 1

        # Per-record CAS substitutions are needed in cas_cmp/new/old.
        cas_old_all: Any = None
        cas_cmp_all: Any = None
        cas_new_all: Any = None
        if record.atomic_kind == "cas":
            cas_old_all = apply_sub(record.old_value, sub)
            cas_cmp_all = apply_sub(record.cas_cmp_value, sub)
            cas_new_all = apply_sub(record.cas_new_value, sub)

        # local_constraints / premises are FLAT lists of globally applicable
        # constraints (mask conditions, address-validity, loop iterators); they
        # are NOT lane-indexed. Per-lane variation lives in record.active /
        # record.reads / record.writes.
        local_terms = tuple(as_bool(c) for c in iter_constraints(local_all))
        prem_terms = tuple(as_bool(c) for c in iter_constraints(prem_all))
        if launch_premises:
            self.launch_premises.extend(prem_terms)
        causal_terms = tuple(as_bool(c) for c in iter_constraints(causal_all))

        out: list[SymbolicMemoryEvent] = []
        for lane, addr in enumerate(addr_lanes):
            causal_inputs = (
                as_bool(lane_value(active_all, lane, n_lanes)),
                *causal_terms,
            )
            active = And(
                as_bool(lane_value(active_all, lane, n_lanes)),
                *local_terms,
                *prem_terms,
            )

            rmw_operand: Any = None
            if record.atomic_kind == "cas":
                old = lane_value(cas_old_all, lane, n_lanes)
                cmp_ = lane_value(cas_cmp_all, lane, n_lanes)
                new = lane_value(cas_new_all, lane, n_lanes)
                if old is None or cmp_ is None or new is None:
                    raise UnsupportedSymbolicRaceQuery(
                        "CAS record missing old_value / cas_cmp_value / cas_new_value"
                    )
                success = old == cmp_
                # Preserve the compare dependency in the write condition,
                # without importing execution-domain premises into vdep.
                causal_inputs += (success,)
                reads = active
                writes = And(active, success)
                written = If(success, new, old)
                old_value: Any = old
                written_value: Any = written
            elif record.is_atomic:
                # AtomicRMW: always reads and always writes when active.
                # Exception: the pre-exit representative of a CAS poll
                # carries record-level statically-False writes (a FAILED
                # CAS writes nothing) — honor it.
                reads = active
                writes = (
                    BoolVal(False)
                    if (record.pre_exit and record.writes is False)
                    else active
                )
                old_value = None
                written_value = None
                if record.old_value is not None:
                    # Value-modeled RMW: the observation symbol
                    # o_r lives in copy_local_vars, so `sub` already alpha-
                    # renamed it per copy; the write part is f_op(o_r, v).
                    old_value = lane_value(
                        apply_sub(record.old_value, sub), lane, n_lanes
                    )
                    if record.rmw_operand is not None:
                        rmw_operand = lane_value(
                            apply_sub(record.rmw_operand, sub), lane, n_lanes
                        )
                    written_value = _rmw_written_value(
                        record.rmw_op, old_value, rmw_operand
                    )
            else:
                if record.reads is None:
                    read_cond: Any = record.access_mode == "read"
                else:
                    read_cond = lane_value(apply_sub(record.reads, sub), lane, n_lanes)
                if record.writes is None:
                    write_cond: Any = record.access_mode == "write"
                else:
                    write_cond = lane_value(
                        apply_sub(record.writes, sub), lane, n_lanes
                    )
                reads = self._active_access_mode(active, as_bool(read_cond))
                writes = self._active_access_mode(active, as_bool(write_cond))
                old_value = None
                written_value = None

            name = record.debug_name or f"e{record.event_id}"
            if n_lanes > 1:
                name = f"{name}.lane{lane}"
            name = f"{name}.{ctx.label}"

            out.append(
                SymbolicMemoryEvent(
                    idx=start_idx + len(out),
                    copy=ctx.label,
                    record=record,
                    name=name,
                    lane=lane,
                    event_id=record.event_id,
                    program_seq=record.program_seq,
                    pid=ctx.pid,
                    addr=addr,
                    elem_size=max(1, int(record.elem_size)),
                    active=active,
                    reads=reads,
                    writes=writes,
                    is_atomic=record.is_atomic,
                    atomic_kind=record.atomic_kind,
                    sem=record.sem,
                    scope=record.scope,
                    old_value=old_value,
                    written_value=written_value,
                    rmw_operand=rmw_operand,
                    causal_inputs=causal_inputs,
                )
            )
        return out

    # ──────────────────────── Edges & HB closure ────────────────────────

    def _fence_orders(self, e1: SymbolicMemoryEvent, e2: SymbolicMemoryEvent) -> bool:
        """The caller's fence rule for two operations of one instance at
        distinct known positions, asked in program order."""
        first, second = (e1, e2) if e1.program_seq < e2.program_seq else (e2, e1)
        return self._fence_orders_records(first.record, second.record)

    def _fence_orders_records(
        self, first: AccessEventRecord, second: AccessEventRecord
    ) -> bool:
        """``fence_between(first, second)``, cached; ``first`` comes first
        in program order."""
        if self.fence_between is None:
            return False
        key = (id(first), id(second))
        cached = self._fence_cache.get(key)
        if cached is None:
            cached = self._fence_cache[key] = bool(self.fence_between(first, second))
        return cached

    def _program_order(
        self, e1: SymbolicMemoryEvent, e2: SymbolicMemoryEvent
    ) -> BoolRef:
        if e1.copy != e2.copy:
            return BoolVal(False)
        if e1.program_seq < 0 or e2.program_seq < 0:
            return BoolVal(False)
        if e1.program_seq >= e2.program_seq:
            return BoolVal(False)
        if not self._fence_orders(e1, e2):
            if (e1.event_id, e2.event_id) in self.dep_pairs:
                # Dependency order: a lane of the dependent is ordered after
                # the anchor's lane at its OWN block position only
                # (cross-position pairs are the intra-instance query's
                # business).
                same = self._same_position(e1, e2)
                if same is None:
                    return BoolVal(False)
                return And(e1.active, e2.active, same)
            # Fence-ordered semantics: source order alone carries no
            # happens-before between distinct operations of one instance.
            return BoolVal(False)
        return And(e1.active, e2.active)  # active-gated

    def _same_position(
        self, e1: SymbolicMemoryEvent, e2: SymbolicMemoryEvent
    ) -> BoolRef | None:
        """Constraint that two events of one copy are lanes at the SAME
        block position, or ``None`` when that cannot be told (then the
        dependency gives no edge, which is conservative).

        Both records must carry a ``lane_position`` of one rank, compared
        dimension by dimension: the two accesses may index their blocks
        with separate aranges (every block-pointer access does), whose
        variables are independent within a copy, and a shared arange
        variable is no position either once an op moves elements (a
        transposed pointer keeps the variables of the untransposed one).
        Without positions, two events on the same lane that use no arange
        variable are taken to be at the same position. That holds for a
        rank-0 anchor (one position) but not for a block on one address
        (``z + tl.zeros([B])``: B positions, no arange), so the caller must
        not list such a block anchor in ``dep_pairs``; SymbolicRaceDetector
        anchors no dependency on a block record without a position."""
        sub = (self.ctx_a if e1.copy == "a" else self.ctx_b).arange_substitutions
        same = self._positions_equal(e1.record, sub, e2.record, sub)
        if same is not None:
            return same
        if e1.lane != e2.lane or self._uses_arange(e1) or self._uses_arange(e2):
            return None
        return BoolVal(True)

    def _same_thread(
        self, e1: SymbolicMemoryEvent, e2: SymbolicMemoryEvent
    ) -> BoolRef | None:
        """Constraint that two events of one copy run on one thread: the
        same position (_same_position) of two blocks of one shape,
        simplified, so that a position shared through one arange variable
        is ``True``. ``None`` when that cannot be told: a transposed or
        broadcast pointer keeps the variables of another position, and two
        positionless blocks of different shapes have no common lane."""
        key = (e1.idx, e2.idx)
        if key in self._same_thread_cache:
            return self._same_thread_cache[key]
        shape_1 = getattr(getattr(e1.record, "symbolic_expr", None), "shape", None)
        shape_2 = getattr(getattr(e2.record, "symbolic_expr", None), "shape", None)
        same = None
        if tuple(shape_1 or ()) == tuple(shape_2 or ()):
            same = self._same_position(e1, e2)
            if same is not None:
                same = simplify(same)
        self._same_thread_cache[key] = same
        return same

    def _uses_arange(self, e: SymbolicMemoryEvent) -> bool:
        """Whether an arange variable of the event's copy occurs in it."""
        used = self._uses_arange_cache.get(e.idx)
        if used is None:
            ctx = self.ctx_a if e.copy == "a" else self.ctx_b
            occurring = _collect_z3_var_keys((e.addr, e.active))
            used = self._uses_arange_cache[e.idx] = any(
                _z3_var_key(var) in occurring for _, var in ctx.arange_substitutions
            )
        return used

    @staticmethod
    def _exact_atomic_addr(w: SymbolicMemoryEvent, r: SymbolicMemoryEvent) -> BoolRef:
        """Same-location predicate for the VALUE-MODELED atomic machinery
        (rf, coherence order). Membership is presence of an observation
        symbol: CAS always has one, an RMW exactly when its value semantics
        are modeled)."""
        if not (w.is_atomic and r.is_atomic):
            return BoolVal(False)
        if w.old_value is None or r.old_value is None:
            return BoolVal(False)
        if w.elem_size != r.elem_size:
            return BoolVal(False)
        return w.addr == r.addr

    def _same_dynamic_op(
        self, e: SymbolicMemoryEvent, f: SymbolicMemoryEvent
    ) -> BoolRef:
        """Predicate: the two events denote ONE dynamic operation — the
        cross-copy alpha-twins of a record (same record, same lane) with
        both copies pinned to the same block, at the same lane position
        and loop iteration. The same-instance query pins
        pids equal, so without this identification the coherence machinery
        would demand two distinct order positions and two distinct rf
        sources for what is a single real operation, making e.g. the
        rank-0 winner's world structurally UNSAT (counterexample:
        single-winner duplicate-lane WAW silently proved clean)."""
        if e.record is not f.record or e.lane != f.lane or e.copy == f.copy:
            return BoolVal(False)
        pid_a, pid_b = self.ctx_a.pid, self.ctx_b.pid
        same = self._same_op_condition(e.record)
        return And(
            *[pid_a[i] == pid_b[i] for i in range(3)],
            *([] if is_true(same) else [same]),
        )

    @staticmethod
    def _scope_ok(w: SymbolicMemoryEvent, r: SymbolicMemoryEvent) -> BoolRef:
        for scope in (w.scope, r.scope):
            if scope not in (None, "cta", "gpu", "sys"):
                # Never widen an unknown scope to device-wide moral
                # strength; the capture normalizes and refuses first,
                # this guards events arriving by other routes.
                raise UnsupportedSymbolicRaceQuery(
                    f"unsupported memory scope {scope!r}"
                )
        if w.scope == "cta" or r.scope == "cta":
            return And(
                w.pid[0] == r.pid[0],
                w.pid[1] == r.pid[1],
                w.pid[2] == r.pid[2],
            )
        return BoolVal(True)

    def _synchronizes_with(
        self, w: SymbolicMemoryEvent, r: SymbolicMemoryEvent
    ) -> BoolRef:
        """sw goes through READS-THROUGH, not bare rf: a release
        write also synchronizes with an acquire read that reads the tail of
        an RMW chain rooted at it. ``rf_unknown`` / ``rf_chain`` sources are
        deliberately absent from reads_through — an unmodeled source must
        never manufacture an hb edge."""
        rt = self.reads_through.get((w.idx, r.idx))
        if rt is None:
            return BoolVal(False)
        return And(
            BoolVal(is_release_sem(w.sem)),
            BoolVal(is_acquire_sem(r.sem)),
            self._scope_ok(w, r),
            rt,
        )

    def _edge(self, e1: SymbolicMemoryEvent, e2: SymbolicMemoryEvent) -> BoolRef:
        # Synchronization through an open-world source (_build_open_world_edges),
        # a self-edge included: it makes reading such a write cyclic.
        extra = self._open_world_edges.get((e1.idx, e2.idx), ())
        if e1.idx == e2.idx:
            return Or(*extra) if extra else BoolVal(False)
        # Without a reads-through source, sw is identically False. Avoid
        # constructing Or(po, False) for every pair in a large capture.
        if (e1.idx, e2.idx) not in self.reads_through:
            base = self._program_order(e1, e2)
            return Or(base, *extra) if extra else base
        return Or(
            self._program_order(e1, e2),
            self._synchronizes_with(e1, e2),
            *extra,
        )

    # ──────────────────────── Read-from / RF source choices ────────────────

    def _can_be_rf_candidate(
        self, w: SymbolicMemoryEvent, r: SymbolicMemoryEvent
    ) -> bool:
        if w.idx == r.idx:
            return False
        # A writer of the reader's own operation, or a later one of its
        # thread, is no rf source. A later writer that may be another
        # thread (another position, or one that cannot be told) may run
        # first; co-po excludes it at a known same position.
        if w.copy == r.copy and w.program_seq >= r.program_seq:
            if w.program_seq == r.program_seq:
                return False
            same = self._same_thread(w, r)
            return not (same is not None and is_true(same))
        return True

    # Cap on initial-source disjunction size. Above this, the solver falls
    # back to rf_unknown (no synchronizes-with) — keeps formulas tractable.
    _MAX_INITIAL_ATOMIC_ELEMENTS: int = 1024

    @classmethod
    def _initial_atomic_source(cls, r: SymbolicMemoryEvent) -> Any:
        """Predicate that ``r`` reads the launch-time initial value.

        Supports scalar tensors and small contiguous flag arrays. For large or
        non-contiguous tensors, returns ``None`` so the solver falls back to
        ``rf_unknown`` without synchronizes-with.
        """
        t = r.record.tensor
        if t is None or r.old_value is None:
            return None
        # The value model is integer-only, so a float-valued flag must
        # fall back to rf_unknown rather than be silently truncated —
        # int(0.7) == 0 would let the modeled CAS succeed where the real one fails (or
        # mask a real race behind a fabricated single-winner lock).
        dtype = getattr(t, "dtype", None)
        if dtype is not None and (
            bool(getattr(dtype, "is_floating_point", False))
            or bool(getattr(dtype, "is_complex", False))
        ):
            return None
        try:
            numel = int(t.numel())
            if numel <= 0 or numel > cls._MAX_INITIAL_ATOMIC_ELEMENTS:
                return None
            if hasattr(t, "is_contiguous") and not bool(t.is_contiguous()):
                return None
            base = int(t.data_ptr())
            elem_size = (
                int(t.element_size()) if hasattr(t, "element_size") else r.elem_size
            )
            elem_size = max(1, elem_size)
            tensor_for_read = t.detach() if hasattr(t, "detach") else t
            tensor_for_read = (
                tensor_for_read.cpu()
                if hasattr(tensor_for_read, "cpu")
                else tensor_for_read
            )
            values = tensor_for_read.reshape(-1).tolist()
        except Exception:
            return None

        clauses = []
        for i, value in enumerate(values):
            # bool is an int subclass; anything else (a duck-typed tensor
            # without a dtype attribute yielding floats) must not be
            # truncated — fall back to rf_unknown.
            if not isinstance(value, int):
                return None
            clauses.append(
                And(
                    r.addr == IntVal(base + i * elem_size),
                    r.old_value == IntVal(int(value)),
                )
            )

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return Or(*clauses)

    def _template_solver(self) -> Solver:
        """A solver holding only the template bounds (grid + arange) — used
        for guard checks that must stay CONSERVATIVE (fewer assumptions →
        more guard failures → axioms omitted, never fabricated)."""
        solver = self._solver()
        solver.add(self.grid_constraints)
        for c in self.arange_constraints_a:
            solver.add(c)
        for c in self.arange_constraints_b:
            solver.add(c)
        return solver

    def _has_unmodeled_overlapping_writer(self, r: SymbolicMemoryEvent) -> bool:
        """True when a write the rf model does not include — a plain store,
        or an atomic whose WRITTEN VALUE is not modeled (float RMW,
        non-identity bitwise RMW, unsigned umax/umin) — can overlap the
        location ``r`` reads.

        Such a writer can publish a value no modeled rf choice contains; without an escape hatch the reader's ``old_value`` would be
        over-constrained and every conflict gated on it silently vanishes
        (e.g. a guard flag set via an unmodeled ``tl.atomic_xchg``). Overlap
        is decided by Z3 on the symbolic addresses under grid/arange bounds
        AND both events' activity (``e.writes`` / ``r.reads`` — masks, path
        conditions, in-bounds premises): a writer only publishes a value
        when it actually writes, and only reads that actually happen need a
        source. Without the activity gate, an address expression drifting
        arithmetically past its tensor under an unbounded symbolic grid
        would open the escape for writers of a DIFFERENT tensor. Writers to
        other tensors (distinct concrete bases) never weaken the closed
        world. A VALUE-MODELED atomic is excluded from this set only to the
        extent its overlap is COVERED by rf candidacy — exact address at
        equal width (_exact_atomic_addr). A torn overlap (different widths,
        or same width at a misaligned unequal address) can publish bytes
        the rf model never delivers, so it must open the escape like any
        unmodeled write (counterexample: an 8-byte modeled xchg over a
        4-byte reader was excluded from BOTH channels and pinned the reader
        to the initial value — a false proof).
        """
        # Pre-exit representatives are excluded (load-bearing: the
        # identity-RMW rep's write half sits AT the awaited location and
        # would otherwise open rf_unknown for its own poll, flipping clean
        # producer-consumer baselines to false data races). Soundness of
        # the exclusion: an identity write-back republishes only a value
        # some other write already put there; if that value's writer is
        # weak, THAT writer opens the escape itself. The CAS-poll and
        # plain-load reps are excluded by the feasibility check anyway
        # (their writes are statically False).
        candidates = [
            e
            for e in self.events
            if ((not e.is_atomic and e.record.access_mode == "write") or e.is_atomic)
            and e.idx != r.idx
            and not e.record.pre_exit
            and self._can_be_rf_candidate(e, r)
        ]
        if not candidates:
            return False
        solver = self._template_solver()
        solver.add(r.reads)
        for e in candidates:
            solver.push()
            solver.add(e.writes)
            solver.add(self._byte_overlap(e, r))
            if e.is_atomic and e.written_value is not None:
                # An ADD write's modeled value old+v is unbounded-Int
                # arithmetic, but the machine wraps: unless the record
                # carries the counting certificate (guard (f) bounded every
                # reachable value inside the dtype range) or provably adds
                # zero, the published value may be one the model cannot
                # express — treat the write as unmodeled (counterexample:
                # an INT32_MAX counter's wrap-gated store was proved dead
                # through the closed-world value chain).
                # max/min never leave the operands' range and xchg/CAS
                # write user values, so only `add` is wrap-capable.
                wrap_capable = (
                    (e.record.rmw_op or "") == "add"
                    and _as_numeral(e.rmw_operand) != 0
                    and id(e.record) not in self._counting
                )
                if not wrap_capable and e.elem_size == r.elem_size:
                    # rf covers exactly the equal-width same-address
                    # overlap; only the residual TORN overlap counts as
                    # unmodeled. (Width mismatch: no rf edge exists at all
                    # — any overlap is uncovered.)
                    solver.add(e.addr != r.addr)
            # Z3 ``unknown`` must open the escape: keeping the closed world
            # on an undecided overlap would over-constrain the reader's old
            # value and silently hide every conflict gated on it.
            feasible = solver.check() != unsat
            solver.pop()
            if feasible:
                return True
        return False

    def _build_read_from_choices(self) -> None:
        # Atomic source model.
        # If the initial scalar source is identifiable, source choices are
        # (initial source) + (value-modeled atomic writers of the two copies
        # — CAS and modeled RMW write parts incl. identity or/xor-of-zero) +
        # (open-world writers: _open_world_sources).
        # If the initial source is not identifiable — or an UNMODELED write
        # (plain store, float RMW, non-identity bitwise RMW) can overlap the
        # location, publishing a value no modeled write contains —
        # rf_unknown is introduced and does NOT enable synchronizes-with.
        # Coherence orders the two copies' writer events only; an
        # open-world writer joins no order. For a COUNTED record the source
        # set is widened by rf_chain: the reader may take its value from an
        # unmodeled grid instance at rank-1 (the counting equations already
        # pin the value), so real races at non-adjacent ranks are not
        # squeezed out. rf_chain never enables synchronizes-with either —
        # the counted sw edge rides on coherence order via reads_through.
        modeled_writers = [
            e for e in self.events if e.is_atomic and e.written_value is not None
        ]
        open_world_writers = self._open_world_writer_records()
        for r in self.events:
            if not r.is_atomic or r.old_value is None:
                continue
            choices: list[BoolRef] = []
            init_pred = self._initial_atomic_source(r)

            if init_pred is not None:
                rf_init = Bool(f"rf_init_{r.idx}")
                self.rf_init_source[r.idx] = rf_init
                choices.append(rf_init)
                self.rf_constraints.append(Implies(rf_init, And(r.reads, init_pred)))
            if init_pred is None or self._has_unmodeled_overlapping_writer(r):
                rf_unknown = Bool(f"rf_unknown_{r.idx}")
                self.rf_unknown_source[r.idx] = rf_unknown
                choices.append(rf_unknown)
                self.rf_constraints.append(Implies(rf_unknown, r.reads))

            counting = self._counting.get(id(r.record))
            if counting is not None:
                rank_r = counting.rank_a if r.copy == "a" else counting.rank_b
                rf_chain = Bool(f"rf_chain_{r.idx}")
                self.rf_chain_source[r.idx] = rf_chain
                choices.append(rf_chain)
                # Source = the unmodeled instance at rank-1; rank >= 1 or
                # the only source with the matching value is the initial
                # one, which rf_init already covers.
                self.rf_constraints.append(Implies(rf_chain, And(r.reads, rank_r >= 1)))

            for w in modeled_writers:
                if not self._can_be_rf_candidate(w, r):
                    continue
                rf = Bool(f"rf_{w.idx}_to_{r.idx}")
                choices.append(rf)
                self.rf_source[(w.idx, r.idx)] = rf
                self.rf_constraints.append(
                    Implies(
                        rf,
                        modeled_atomic_read_from(
                            w, r, same_atomic_addr_fn=self._exact_atomic_addr
                        ),
                    )
                )

            if r.idx not in self.rf_unknown_source:
                # rf_unknown already admits any value for r.
                choices.extend(self._open_world_sources(r, open_world_writers))

            if choices:
                self.rf_constraints.append(Implies(r.reads, Or(*choices)))
                self.rf_constraints.append(
                    Implies(Not(r.reads), And(*(Not(c) for c in choices)))
                )
                if len(choices) > 1:
                    self.rf_constraints.append(AtMost(*choices, 1))

    # ──────────────────── open-world rf sources ────────────────────

    def _open_world_writer_records(self) -> list[AccessEventRecord]:
        """Value-modeled atomic writers that any thread of the grid may run:
        every CAS, and every value-modeled xchg / max / min RMW. The rest
        need no open-world source:

          * ``add`` with a nonzero operand is wrap-capable, so it is an
            unmodeled writer and opens rf_unknown for every reader it can
            overlap;
          * an identity write (``add`` / ``or`` / ``xor`` of zero, the
            no-change branch of ``max`` / ``min``) writes back the value it
            read: its reader can read that value's writer instead, through
            the same release sequence;
          * a counted record is covered by rf_chain and the counting
            equations;
          * a pre-exit or await-gated record (compiled frontends only)
            keeps the closed world.
        """
        return [
            record
            for record in self.records
            if record.is_atomic
            and not record.pre_exit
            and not record.await_dependencies
            and not record.await_observations
            and id(record) not in self._counting
            and (
                record.atomic_kind == "cas"
                or (
                    record.old_value is not None
                    and record.rmw_op in ("xchg", "max", "min")
                    and record.rmw_operand is not None
                )
            )
        ]

    def _fresh_context(self, tag: str) -> tuple[CopyContext, BoolRef]:
        """A copy context for one more thread of the grid: fresh pid,
        lane and copy-local variables, and their bounds."""
        pid = tuple(Int(f"pid_{tag}_{i}") for i in range(3))
        arange_subs = []
        bounds = [And(pid[i] >= 0, pid[i] < self.grid[i]) for i in range(3)]
        for orig, _ in self.ctx_a.arange_substitutions:
            start, end = self._arange_ranges[_z3_var_key(orig)]
            var = Int(f"{orig.decl().name()}__{tag}")
            arange_subs.append((orig, var))
            bounds.append(And(var >= start, var < end))
        copy_local = tuple(
            (orig, Const(f"{var_a.decl().name()}__{tag}", orig.sort()))
            for orig, var_a in self.ctx_a.copy_local_substitutions
        )
        ctx = CopyContext(
            label=tag,
            pid=pid,
            pid_substitutions=tuple(zip(_import_symbolic_expr_pids(), pid)),
            arange_substitutions=tuple(arange_subs),
            arange_constraints=(),
            copy_local_substitutions=copy_local,
        )
        return ctx, And(*bounds)

    @staticmethod
    def _lanes_not_named(record: AccessEventRecord) -> bool:
        """A block record without a lane_position: its lane variables, if
        any, do not name its lanes. On a block on one address (``z +
        tl.zeros([B])``) one symbolic lane stands for all of them; on a
        broadcast pointer one assignment of its variables stands for
        several lanes; on a transposed or reshaped pointer an assignment
        names a lane at another position than an untransposed record with
        the same variables."""
        shape = getattr(getattr(record, "symbolic_expr", None), "shape", None)
        return bool(shape) and record.lane_position is None

    def _value_needs_no_other_observation(self, record: AccessEventRecord) -> bool:
        """The record's write uses no copy-local value but its own
        observation and those of same-thread anchors, so its event in a
        copy and a fresh lowering at the same pid and lane write the same
        value. A same-thread anchor X is outside any loop, names its lanes,
        and uses only lane variables of the record: at the record's pid and
        lane in a copy, X's event is the one dynamic X whose result that
        thread uses (a scalar CAS's result, say)."""
        key = ("same_value", id(record))
        cached = self._record_cache.get(key)
        if cached is not None:
            return cached
        own = _collect_z3_var_keys((record.old_value,))
        used = _collect_z3_var_keys(
            (
                record.addr_expr,
                record.active,
                record.local_constraints,
                record.premises,
                record.rmw_operand,
                record.cas_cmp_value,
                record.cas_new_value,
            )
        )
        copy_local = {
            _z3_var_key(orig) for orig, _ in self.ctx_a.copy_local_substitutions
        }
        rest = (used & copy_local) - own
        mine = {_z3_var_key(v) for v in self._record_lane_vars(record)}
        for other in self.records:
            if not rest:
                break
            keys = _collect_z3_var_keys((other.old_value,)) & rest
            if not keys:
                continue
            if (
                self._record_iterators(other)
                or self._lanes_not_named(other)
                or not {_z3_var_key(v) for v in self._record_lane_vars(other)} <= mine
            ):
                break
            rest -= keys
        cached = self._record_cache[key] = not rest
        return cached

    def _open_world_sources(
        self, r: SymbolicMemoryEvent, writers: list[AccessEventRecord]
    ) -> list[BoolRef]:
        """rf choices for ``r`` from any thread that runs an open-world
        writer record W: another lane of one of the two modeled instances,
        or another instance of the grid. Each (reader, W) gets W lowered
        under its own fresh context. The lowered writer is no event of the
        solver: it joins no race pair, coherence order, value causality,
        counting or reads-through relation, and it synchronizes only
        through _build_open_world_edges.

        Exclusions apply only when W's lane variables name its lanes
        (not _lanes_not_named). A fresh writer at the pid and lane of W's
        event in a copy is that event itself (atomics run in no loop). It
        is excluded where the copy's event is already an rf choice of ``r``
        with the same value, so that the coherence of the copy's event is
        not bypassed, and where the copy's event is a later operation of
        ``r``'s own thread, which ``r`` cannot read. A writer at ``r``'s
        own pid and lane is ``r`` itself, which never reads its own
        write. Every other thread running W stays a source."""
        choices: list[BoolRef] = []
        template: Solver | None = None
        ctx_r = self.ctx_a if r.copy == "a" else self.ctx_b
        for record in writers:
            if max(1, int(record.elem_size)) != r.elem_size:
                continue
            tag = f"ow{r.idx}w{record.event_id}"
            ctx, bounds = self._fresh_context(tag)
            identifiable = not self._lanes_not_named(record)
            same_value = self._value_needs_no_other_observation(record)
            for w in self._lower_record(record, ctx, -1, launch_premises=False):
                value = (
                    w.written_value if record.atomic_kind == "cas" else w.rmw_operand
                )
                if value is None:
                    continue
                if template is None:
                    template = self._template_solver()
                template.push()
                template.add(bounds, w.writes, r.reads, w.addr == r.addr)
                feasible = template.check() != unsat  # unknown keeps it
                template.pop()
                if not feasible:
                    continue
                cond = [
                    bounds,
                    w.writes,
                    r.reads,
                    w.addr == r.addr,
                    value == r.old_value,
                ]
                for c in (self.ctx_a, self.ctx_b) if identifiable else ():
                    same_thread = And(
                        *[ctx.pid[i] == c.pid[i] for i in range(3)],
                        self._lane_equal(
                            record, ctx.arange_substitutions, c.arange_substitutions
                        ),
                    )
                    if record is r.record and c is ctx_r and w.lane == r.lane:
                        cond.append(Not(same_thread))
                        continue
                    twin = self._event_of.get((id(record), c.label, w.lane))
                    if twin is None:
                        continue
                    if same_value and (twin.idx, r.idx) in self.rf_source:
                        cond.append(Not(same_thread))
                    elif twin.copy == r.copy and twin.program_seq > r.program_seq:
                        later = self._same_thread(twin, r)
                        if later is not None and is_true(later):
                            cond.append(Not(same_thread))
                rf = Bool(f"rf_any_{record.event_id}_{w.lane}_to_{r.idx}")
                condition = And(*cond)
                self.rf_constraints.append(Implies(rf, condition))
                self.rf_open_world.append((rf, r, record, w, ctx, condition))
                choices.append(rf)
        return choices

    def _build_open_world_edges(self) -> None:
        """Synchronizes-with through an open-world source: when ``r``
        acquires what a release W writes, every event of W's instance that
        a fence, or a dependency at W's own position, orders before W
        happens before ``r``. The edges join the HB matrix as direct
        (e, r) edges; nothing else of W's instance is modeled, so
        synchronization that reaches W through a third instance is lost
        (over-reported)."""
        template: Solver | None = None
        for rf, r, record, w, ctx, condition in self.rf_open_world:
            if not (is_release_sem(record.sem) and is_acquire_sem(r.sem)):
                continue
            if record.program_seq < 0:
                continue
            scope = self._scope_ok(w, r)
            # An edge from a copy's event needs the fresh writer in that
            # copy's instance. Where the choice rules that out (a twin
            # exclusion, an address only another instance writes), the edge
            # is always false: skip it, which keeps the HB closure small.
            alive: dict[str, bool] = {}
            for c in (self.ctx_a, self.ctx_b):
                if template is None:
                    template = self._template_solver()
                template.push()
                template.add(condition, *[w.pid[i] == c.pid[i] for i in range(3)])
                alive[c.label] = template.check() != unsat  # unknown keeps it
                template.pop()
            for e in self.events:
                if e.program_seq < 0 or e.program_seq >= record.program_seq:
                    continue
                if not alive[e.copy]:
                    continue
                if self._fence_orders_records(e.record, record):
                    ordered: BoolRef = BoolVal(True)
                elif (e.event_id, record.event_id) in self.dep_pairs:
                    sub_e = (
                        self.ctx_a if e.copy == "a" else self.ctx_b
                    ).arange_substitutions
                    same = self._positions_equal(
                        e.record, sub_e, record, ctx.arange_substitutions
                    )
                    if same is None:
                        if (
                            e.lane != w.lane
                            or self._uses_arange(e)
                            or self._record_lane_vars(record)
                        ):
                            continue
                        same = BoolVal(True)
                    ordered = same
                else:
                    continue
                term = And(
                    rf,
                    scope,
                    *[w.pid[i] == e.pid[i] for i in range(3)],
                    e.active,
                    ordered,
                )
                self._open_world_edges.setdefault((e.idx, r.idx), []).append(term)

    # ──────────────────────── Conflict / race query ────────────────────────

    @staticmethod
    def _byte_overlap(a: SymbolicMemoryEvent, b: SymbolicMemoryEvent) -> BoolRef:
        if a.elem_size == 1 and b.elem_size == 1:
            return a.addr == b.addr
        return And(
            a.addr < b.addr + b.elem_size,
            b.addr < a.addr + a.elem_size,
        )

    def _conflict(self, a: SymbolicMemoryEvent, b: SymbolicMemoryEvent) -> BoolRef:
        mode_a = self._ordinary_access_mode(a)
        mode_b = self._ordinary_access_mode(b)
        if mode_a is not None and mode_b is not None:
            if not (mode_a or mode_b):
                return BoolVal(False)
            # Lowering already includes activity in the ordinary read/write
            # mode. Under the two activity guards the mode predicate is True.
            return And(a.active, b.active, self._byte_overlap(a, b))
        return And(
            a.active,
            b.active,
            self._byte_overlap(a, b),
            conflicting_access_modes(a, b),
        )

    @staticmethod
    def _ordinary_access_mode(event: SymbolicMemoryEvent) -> bool | None:
        """Recognize exact lowered plain modes, without assuming an op's kind.

        Identity with the immutable activity expression is sufficient; unknown
        or conditional modes and atomics retain the general conflict formula.
        """
        if event.is_atomic:
            return None
        if event.writes is event.active and is_false(event.reads):
            return True
        if event.reads is event.active and is_false(event.writes):
            return False
        return None

    def _faithful_position(self, record: AccessEventRecord) -> bool:
        """The record's lane_position names every lane: one distinct arange
        variable per dimension. (Two same-range aranges on one source line
        share a variable; a position over them covers the diagonal only.)"""
        key = ("faithful", id(record))
        cached = self._record_cache.get(key)
        if cached is None:
            cached = bool(record.lane_position)
            seen: set[tuple[int, str, str]] = set()
            for coord in record.lane_position or ():
                keys = _collect_z3_var_keys((coord,)) & set(self._arange_ranges)
                if len(keys) != 1 or keys & seen:
                    cached = False
                    break
                seen |= keys
            self._record_cache[key] = cached
        return cached

    def _position_extents(self, record: AccessEventRecord) -> tuple[Any, ...]:
        """Per dimension, the size of the arange feeding it (faithful
        positions only)."""
        out = []
        for coord in record.lane_position or ():
            (key,) = _collect_z3_var_keys((coord,)) & set(self._arange_ranges)
            start, end = self._arange_ranges[key]
            out.append(end - start)
        return tuple(out)

    def _anchor_pins(self, e: SymbolicMemoryEvent) -> tuple[BoolRef, ...]:
        """Pin the lanes of a race endpoint's dependency anchors to its own
        position, in its copy.

        A dependent consumes its anchors' results element-wise, so the lane
        of ``e`` at position q uses each anchor's lane at q (transitively).
        Within a copy, an anchor with its own aranges is otherwise on an
        independent lane, and the endpoint's mask or value would read
        another lane's result. Pinning only chooses which lanes of the
        anchors the copy stands for; every other lane of a value-modeled
        writer stays an rf source through the open world. So a pin is
        added only when it is satisfiable for every lane of ``e``: both
        positions faithful with equal extents, and the anchor's position
        variables disjoint from those already pinned (or the two positions
        identical). A pin is never needed for soundness; skipping one only
        loses precision."""
        cached = self._anchor_pin_cache.get(e.idx)
        if cached is not None:
            return cached
        pins: list[BoolRef] = []
        record = e.record
        if record.dep_loads and self._faithful_position(record):
            sub = (self.ctx_a if e.copy == "a" else self.ctx_b).arange_substitutions
            lane_vars = {_z3_var_key(var) for _, var in sub}
            target = tuple(apply_sub(p, sub) for p in record.lane_position or ())
            extents = self._position_extents(record)
            pinned = _collect_z3_var_keys(target) & lane_vars
            positions = [target]
            seen: set[int] = set()
            queue = list(record.dep_loads)
            while queue:
                eid = int(queue.pop(0))
                if eid in seen:
                    continue
                seen.add(eid)
                anchor = self._record_by_id.get(eid)
                if anchor is None:
                    continue
                queue.extend(anchor.dep_loads)
                coords = anchor.lane_position or ()
                if (
                    not self._faithful_position(anchor)
                    or len(coords) != len(target)
                    or self._position_extents(anchor) != extents
                ):
                    continue
                position = tuple(apply_sub(p, sub) for p in coords)
                if any(
                    all(x.eq(y) for x, y in zip(position, other)) for other in positions
                ):
                    continue  # already the endpoint's position
                keys = _collect_z3_var_keys(position) & lane_vars
                if keys & pinned:
                    continue
                pinned |= keys
                positions.append(position)
                pins.extend(x == y for x, y in zip(position, target))
        cached = self._anchor_pin_cache[e.idx] = tuple(pins)
        return cached

    def _race_expr(self, a: SymbolicMemoryEvent, b: SymbolicMemoryEvent) -> BoolRef:
        forward, reverse = self.hb[a.idx][b.idx], self.hb[b.idx][a.idx]
        if is_true(forward) or is_true(reverse):
            return BoolVal(False)
        terms = [self._conflict(a, b), *self._anchor_pins(a), *self._anchor_pins(b)]
        if not is_false(forward):
            terms.append(Not(forward))
        if not is_false(reverse):
            terms.append(Not(reverse))
        return terms[0] if len(terms) == 1 else And(*terms)

    # ──────────────────── modeled-atomic coherence ────────────────────

    def _modeled_atomic_events(self) -> list[SymbolicMemoryEvent]:
        """Events in the per-location atomic order: every VALUE-MODELED
        atomic — CAS always, RMW exactly when its observation is modeled."""
        return [e for e in self.events if e.is_atomic and e.old_value is not None]

    def _make_atomic_order_vars(self) -> dict[int, Any]:
        """One symbolic atomic-order position per value-modeled atomic.

        The variable denotes the position of the WHOLE operation in the
        per-location atomic order — the read and the write part share it,
        which is what makes RMW atomicity/immediacy
        structural: no write can sit between an RMW's read and its write
        because they occupy one position, and the rf constraints below
        forbid a writer strictly between the source and the reader.
        """
        return {
            e.idx: Int(f"atomic_order_{e.idx}")
            for e in self.events
            if e.is_atomic and e.old_value is not None
        }

    def _build_atomic_coherence_constraints(self) -> None:
        """Closed-world atomic coherence for the two modeled program copies.

        Without these constraints, two CAS try-locks at the same flag could
        both read the initial value and both succeed — producing a false WAW
        on guarded stores; likewise two modeled RMWs could both observe the
        initial counter. The coherence model (over value-modeled atomics):

          * Active actions get bounded atomic-order positions.
          * Same-address active actions are distinct in the per-location
            order.
          * Same-copy program order is preserved for same-address actions.
          * If a reader reads the initial source, no modeled successful
            writer at the same address may precede it.
          * If r reads from modeled writer w, w must be before r in the order
            and no modeled same-address successful writer may sit between
            them — with the shared read/write position this is exactly the
            RMW immediacy axiom: what makes two successful lock
            acquisitions of the same "0" unsatisfiable.

        This is not a full GPU memory model, but it suffices to suppress the
        most obvious unsoundness around atomic synchronization patterns.
        """
        atomic_events = self._modeled_atomic_events()
        if not atomic_events:
            return

        n_orders = max(1, len(atomic_events))
        cons = self.atomic_coherence_constraints

        for e in atomic_events:
            ord_e = self.atomic_order[e.idx]
            cons.append(Implies(e.reads, And(ord_e >= 0, ord_e < n_orders)))

        for i, e in enumerate(atomic_events):
            for f in atomic_events[i + 1 :]:
                same_addr = self._exact_atomic_addr(e, f)
                both_active_same_addr = And(e.reads, f.reads, same_addr)
                ord_e = self.atomic_order[e.idx]
                ord_f = self.atomic_order[f.idx]

                # Distinct positions apply to distinct dynamic operations;
                # a record's cross-copy twins under pinned-equal pids are
                # ONE operation and share the position instead.
                same_op = self._same_dynamic_op(e, f)
                cons.append(
                    Implies(And(both_active_same_addr, Not(same_op)), ord_e != ord_f)
                )
                cons.append(Implies(same_op, ord_e == ord_f))

                # co-po: per-location order follows program order on one
                # thread (it holds on hardware, fence or not; the race
                # query's fence-ordered HB is a separate over-approximation).
                # Only at a known same thread: a transposed pointer's lane
                # with the same variables is another thread.
                same_thread = self._same_thread(e, f) if e.copy == f.copy else None
                if (
                    same_thread is not None
                    and e.program_seq >= 0
                    and f.program_seq >= 0
                    and e.program_seq != f.program_seq
                ):
                    guard = (
                        both_active_same_addr
                        if is_true(same_thread)
                        else And(both_active_same_addr, same_thread)
                    )
                    if e.program_seq < f.program_seq:
                        cons.append(Implies(guard, ord_e < ord_f))
                    else:
                        cons.append(Implies(guard, ord_f < ord_e))

        # rf_init: no modeled successful writer at the same address may
        # precede the reader in the per-location order. The reader's own
        # cross-copy twin (same dynamic op when pids coincide) is not a
        # DISTINCT writer and must not block rf_init.
        for r in atomic_events:
            rf_init = self.rf_init_source.get(r.idx)
            if rf_init is None:
                continue
            ord_r = self.atomic_order[r.idx]
            for w in atomic_events:
                if w.idx == r.idx:
                    continue
                ord_w = self.atomic_order[w.idx]
                cons.append(
                    Implies(
                        And(
                            rf_init,
                            w.writes,
                            self._exact_atomic_addr(w, r),
                            Not(self._same_dynamic_op(w, r)),
                        ),
                        ord_r < ord_w,
                    )
                )

        # rf from modeled writer w to reader r: w precedes r and no modeled
        # same-address successful writer sits strictly between w and r
        # (immediacy — the reader's write part shares its position).
        # Twins of w or r (same dynamic op) share the endpoints' positions
        # and are not "between".
        for r in atomic_events:
            ord_r = self.atomic_order[r.idx]
            for w in atomic_events:
                rf = self.rf_source.get((w.idx, r.idx))
                if rf is None:
                    continue
                ord_w = self.atomic_order[w.idx]
                cons.append(Implies(rf, ord_w < ord_r))
                # An operation never reads-from itself: forbid the rf edge
                # between cross-copy twins when they denote one op.
                cons.append(Implies(rf, Not(self._same_dynamic_op(w, r))))

                for v in atomic_events:
                    if v.idx in (w.idx, r.idx):
                        continue
                    ord_v = self.atomic_order[v.idx]
                    cons.append(
                        Implies(
                            And(
                                rf,
                                v.writes,
                                self._exact_atomic_addr(v, r),
                                Not(self._same_dynamic_op(v, r)),
                                Not(self._same_dynamic_op(v, w)),
                            ),
                            Or(ord_v < ord_w, ord_r < ord_v),
                        )
                    )

    def _build_coherence_hb_constraints(self) -> None:
        """Coherence hb-consistency axiom (co-hb): hb-ordered distinct
        same-location value-modeled atomics take increasing per-location
        positions. This extends co-po's antecedent from same-copy program
        order to the full hb closure; co-po stays asserted on its own,
        since program order between unfenced operations is no hb edge.
        Without it, a relaxed atomic read that
        is hb-after a same-location write can still source the initial
        value (the message-passing counterexample). Together with the
        no-intervening-writer clauses of rf/rf_init, the four C++
        coherence shapes (write-write, read-write, write-read, read-read)
        follow. Gating mirrors co-po (activity via ``reads``, location via
        the exact-atomic-address predicate); cross-copy twins under
        pinned-equal pids denote ONE operation sharing a position and are
        exempt, as in the uniqueness clause."""
        atomic_events = self._modeled_atomic_events()
        cons = self.atomic_coherence_constraints
        for e in atomic_events:
            ord_e = self.atomic_order[e.idx]
            for f in atomic_events:
                if f.idx == e.idx:
                    continue
                if e.copy == f.copy and 0 <= e.program_seq < f.program_seq:
                    same_thread = self._same_thread(e, f)
                    if same_thread is not None and is_true(same_thread):
                        continue  # co-po asserts this pair without the hb antecedent
                cons.append(
                    Implies(
                        And(
                            self.hb[e.idx][f.idx],
                            e.reads,
                            f.reads,
                            self._exact_atomic_addr(e, f),
                            Not(self._same_dynamic_op(e, f)),
                        ),
                        ord_e < self.atomic_order[f.idx],
                    )
                )

    # ──────────────────── value causality (wf-vc) ────────────────────

    def _build_value_causality_constraints(self) -> None:
        """wf-vc: value-causality axiom (no out-of-thin-air values).

        Rationale (PTX causality / Tile-IR no-thin-air): the hardware
        memory model never lets a relaxed atomic load observe a value whose
        only justification is a write that itself depends on that very
        observation — two relaxed reads each observing the value the other
        thread's dependent write produced is forbidden even without any
        acquire/release ordering. Formally the model requires rf ∪ vdep to
        be acyclic, where vdep(r, e) holds when read r's observation
        reaches event e's written value, operand, activity, or address.

        Encoding: every value-modeled atomic event e (the per-location
        atomic-order membership set) gets a PAIR of fresh Int causality
        ranks vc_r(e) <= vc_w(e) — value flows read-to-write within one
        operation, which also orders an identity write-back after its own
        observation. Two edge families then rule the cycles out:

          * rf edge: for each MODELED-writer selector rf(w, r), assert
            rf(w, r) -> vc_w(w) < vc_r(r): a read's value exists only
            after its source write produced it. rf_init / rf_unknown /
            rf_chain sources get NO edge — a reader sourced there has an
            unconstrained vc_r and roots the graph, so cycles through the
            initial value, the open world, or the counted chain are NOT
            excluded (conservative: those values may be justified outside
            the two modeled copies).
          * static vdep edge: if r's observation symbol occurs in another
            value-modeled event e's written value or RMW operand (a CAS's
            cmp/new terms live inside its written_value If-term), assert
            vc_r(r) < vc_w(e); if it occurs in e's activity (reads/writes
            conditions) or address, assert vc_r(r) < vc_r(e) — the value
            is needed before e's read part can even issue.

        Operational conditions are preserved separately from the lowered
        reads/writes predicates. Those predicates also contain global
        execution premises (including every await's exit condition) and
        source-domain restrictions, which are not value dependencies.
        Scanning them would invent reciprocal dependencies between two
        independent awaits and make even the bare base inconsistent.
        Frontends distinguish actual loop/mask control from domain facts;
        records without causal_constraints use local_constraints/premises
        as operational conditions, including interpreter loop-control
        premises.

        Gating: genuine static vdep edges remain UNCONDITIONAL. Captured
        operational expressions refer to earlier observations within one
        copy, and alpha-renaming separates the copies. Global termination
        conditions do not meet that ordering argument and are excluded
        from dependency extraction, while remaining in event activity and
        the independent feasibility check. Cycle-closing rf edges remain
        activity-gated through their selectors. A preceding await's exit
        is a genuine control dependency of later operations: these explicit
        edges require both same-copy endpoints to execute, so mutually
        exclusive branches do not constrain each other's causality ranks.

        Occurrence is checked on GENUINE observation symbols only: the keys
        of r.old_value intersected with the copy-local rename targets (the
        observation vars live in copy_local_vars, like the CAS return).
        Anything that leaks other vars into old_value yields an empty key
        set and simply drops the vdep edge — the over-report direction.

        Plain stores and non-value-modeled atomics carry no ranks: a
        causality cycle through them would need an rf edge FROM them, which
        the closed world never provides (they open the rf_unknown escape
        instead), so their omission is conservative.
        """
        atomic_events = self._modeled_atomic_events()
        if not atomic_events:
            return
        cons = self.value_causality_constraints
        for e in atomic_events:
            self.vc_read_rank[e.idx] = Int(f"vc_r_{e.idx}")
            self.vc_write_rank[e.idx] = Int(f"vc_w_{e.idx}")
            cons.append(self.vc_read_rank[e.idx] <= self.vc_write_rank[e.idx])

        # rf edges: modeled-writer selectors only (never rf_init /
        # rf_unknown / rf_chain — those readers root the graph).
        for (w_idx, r_idx), rf in self.rf_source.items():
            cons.append(
                Implies(rf, self.vc_write_rank[w_idx] < self.vc_read_rank[r_idx])
            )

        # Static vdep edges, by observation-symbol occurrence.
        copy_local_targets = {
            _z3_var_key(var)
            for subs in (
                self.ctx_a.copy_local_substitutions,
                self.ctx_b.copy_local_substitutions,
            )
            for (_, var) in subs
        }
        for r in atomic_events:
            obs_keys = _collect_z3_var_keys((r.old_value,)) & copy_local_targets
            if not obs_keys:
                continue
            exit_sources = (
                [
                    source
                    for source in atomic_events
                    if source.copy == r.copy
                    and source.event_id in r.record.await_observations
                ]
                if r.record.await_observations
                else [r]
            )
            for e in atomic_events:
                if e.idx == r.idx:
                    continue
                if obs_keys & _collect_z3_var_keys((e.written_value, e.rmw_operand)):
                    cons.append(self.vc_read_rank[r.idx] < self.vc_write_rank[e.idx])
                if obs_keys & _collect_z3_var_keys((e.causal_inputs, e.addr)):
                    cons.append(self.vc_read_rank[r.idx] < self.vc_read_rank[e.idx])
                # Exiting a preceding await controls later operations, but a
                # poll in another branch does not. The encoder supplies
                # source-earlier candidates; both same-copy path/activity
                # predicates decide whether that control edge actually exists.
                if r.copy == e.copy and r.event_id in e.record.await_dependencies:
                    # An exit may compare the poll with another read's value.
                    # Both are control inputs; retaining only the poll would
                    # allow an expected value to come from a future publication.
                    for source in exit_sources:
                        cons.append(
                            Implies(
                                And(r.reads, e.reads, source.reads),
                                self.vc_read_rank[source.idx]
                                < self.vc_read_rank[e.idx],
                            )
                        )

    # ──────────────────── counting axiom ────────────────────

    def _build_counting_axioms(self) -> dict[int, _CountingInfo]:
        """Per-copy rank variables for each RMW record whose guards hold.

        Guards, all checked on the TEMPLATE (Z3 ``unknown`` fails a guard —
        the axiom is then OMITTED, never approximated):
          (a) the record is a value-modeled ``add`` RMW on a true scalar
              (rank 0, no lane variable: a block of lanes on one address
              increments it once per lane), and its address simplifies to
              ONE numeral in both copies (a fixed scalar location L);
          (b) its activity is provably true for every instance;
          (c) the increment is a constant c > 0;
          (d) L's initial value is known via the rf-init machinery's own
              side conditions;
          (e) no OTHER event's write part can overlap L (which also keeps
              the rf_unknown escape closed for the counted readers).
        The grid-size product must stay linear: at most one symbolic dim.

        Justification of the emitted constraints: under (a)-(e) every write
        to L is an instance of this one always-active RMW, so the
        per-location coherence order is exactly the arrival order; the k-th
        arriver observes init + c·k (induction over the chain, using
        atomicity/immediacy), distinct instances have distinct ranks, and
        coherence order agrees with rank order. Consequences the solver
        derives for free: distinct observations across blocks (work-queue
        disjointness) and "o = init + c·(|G|-1) forces coherence-after-
        everyone" (last-block-done).
        """
        out: dict[int, _CountingInfo] = {}
        by_record: dict[int, list[SymbolicMemoryEvent]] = {}
        for e in self.events:
            if e.is_atomic and e.atomic_kind == "rmw" and e.old_value is not None:
                by_record.setdefault(id(e.record), []).append(e)
        if not by_record:
            return out
        # |G| must stay linear (rank < g0·g1·g2): >1 symbolic dim is
        # Z3-unknown bait, so the axiom is omitted for such grids.
        if sum(1 for d in self.grid if _is_symbolic_dim(d)) > 1:
            return out
        grid_size = self.grid[0] * self.grid[1] * self.grid[2]

        for rec_key, evts in by_record.items():
            if len(evts) != 2:
                continue  # multi-lane record (or a missing copy)
            e_a = next((e for e in evts if e.copy == "a"), None)
            e_b = next((e for e in evts if e.copy == "b"), None)
            if e_a is None or e_b is None:
                continue
            if (e_a.record.rmw_op or "") != "add":
                continue
            info = self._counting_guards(e_a, e_b, grid_size)
            if info is not None:
                out[rec_key] = info
        return out

    def _counting_guards(
        self,
        e_a: SymbolicMemoryEvent,
        e_b: SymbolicMemoryEvent,
        grid_size: Any,
    ) -> _CountingInfo | None:
        # (a) fixed scalar location: both copies' addresses are the SAME
        # numeral (no pid/arange/copy-local dependence survives simplify).
        loc = _as_numeral(e_a.addr)
        if loc is None or _as_numeral(e_b.addr) != loc:
            return None
        # One increment per instance: a block of lanes on one address
        # (cnt + 0 * offs, cnt + tl.zeros([B])) increments it once per lane.
        record = e_a.record
        if (
            record.lane_position is not None
            or self._record_lane_vars(record)
            or getattr(getattr(record, "symbolic_expr", None), "shape", None)
        ):
            return None
        # (c) constant increment > 0, identical in both copies.
        inc = _as_numeral(e_a.rmw_operand)
        if inc is None or inc <= 0 or _as_numeral(e_b.rmw_operand) != inc:
            return None
        # (b) activity provably true for every instance of the grid.
        solver = self._template_solver()
        for e in (e_a, e_b):
            solver.push()
            solver.add(Not(e.active))
            always_active = solver.check() == unsat
            solver.pop()
            if not always_active:
                return None
        # (d) known initial value, via the rf-init machinery's own side
        # conditions (so rf_init is guaranteed to exist as a choice).
        if self._initial_atomic_source(e_a) is None:
            return None
        init = self._initial_value_at(e_a, loc)
        if init is None:
            return None
        # (f) no machine-integer wraparound on any LAUNCHABLE grid: the
        # value model is unbounded-Int, so pinning o = init + c·rank is
        # sound only while the largest reachable observation stays inside
        # the element type's signed range (counterexample: an
        # INT32_MAX-initialized counter really wraps on hardware while the
        # model proved its wrap-gated store dead). Symbolic dims are
        # bounded by the CUDA launch caps (2^31-1 on x, 65535 on y/z) —
        # no real launch exceeds them, so certifying up to the cap
        # certifies every execution the claim covers.
        elem = max(1, e_a.elem_size)
        max_g = 1
        caps = (2**31 - 1, 65535, 65535)
        for axis, d in enumerate(self.grid):
            max_g *= caps[axis] if _is_symbolic_dim(d) else max(1, int(d))
        signed_max = (1 << (8 * elem - 1)) - 1
        # The largest reachable value is the LAST WRITE, init + inc·|G|
        # (the last observation plus one increment) — the certificate must
        # cover it, since it is what the escape hatch trusts to stay
        # representable.
        if init + inc * max_g > signed_max:
            if any(_is_symbolic_dim(d) for d in self.grid):
                self._counting_grid_restricted.add(id(e_a.record))
            return None
        # (e) no other event's write part can overlap [L, L+elem). The
        # check runs under the axiom's own PROVISIONAL observation bounds
        # (init <= o <= init + c·(|G|-1)) — sound by first-violation
        # induction over L's coherence order: before the first foreign
        # write, every observation is a bounded chain value, so a foreign
        # write whose address needs an out-of-bound o cannot be first
        # (work-queue stores addressed by the observation would otherwise
        # alias everything and kill the axiom that pins them).
        own = {e_a.idx, e_b.idx}
        lo = IntVal(init)
        hi = IntVal(init) + IntVal(inc) * (grid_size - 1)
        for e in self.events:
            if e.idx in own:
                continue
            if not e.is_atomic and e.record.access_mode != "write":
                continue
            solver.push()
            for o in (e_a.old_value, e_b.old_value):
                solver.add(o >= lo, o <= hi)
            solver.add(e.writes)
            solver.add(e.addr < IntVal(loc + elem))
            solver.add(IntVal(loc) < e.addr + e.elem_size)
            overlap_possible = solver.check() != unsat
            solver.pop()
            if overlap_possible:
                return None

        rank_a = Int(f"rmw_rank_{e_a.idx}_a")
        rank_b = Int(f"rmw_rank_{e_b.idx}_b")
        pid_a, pid_b = self.ctx_a.pid, self.ctx_b.pid
        same_block = And(*[pid_a[i] == pid_b[i] for i in range(3)])
        ord_a = self.atomic_order[e_a.idx]
        ord_b = self.atomic_order[e_b.idx]
        cons = self.counting_constraints
        cons.append(And(rank_a >= 0, rank_a < grid_size))
        cons.append(And(rank_b >= 0, rank_b < grid_size))
        # rank is the instance's position in L's coherence order: equal iff
        # the two copies denote the same block.
        cons.append(same_block == (rank_a == rank_b))
        cons.append(e_a.old_value == IntVal(init) + IntVal(inc) * rank_a)
        cons.append(e_b.old_value == IntVal(init) + IntVal(inc) * rank_b)
        cons.append((ord_b < ord_a) == (rank_b < rank_a))
        return _CountingInfo(
            idx_a=e_a.idx,
            idx_b=e_b.idx,
            rank_a=rank_a,
            rank_b=rank_b,
            init=init,
            inc=inc,
            loc=loc,
        )

    @classmethod
    def _initial_value_at(cls, r: SymbolicMemoryEvent, loc: int) -> int | None:
        """The launch-time integer value stored at byte address ``loc`` in
        the reader's tensor, under the same side conditions as
        ``_initial_atomic_source`` (integer contents, aligned, in range)."""
        t = r.record.tensor
        if t is None:
            return None
        try:
            base = int(t.data_ptr())
            elem = max(
                1, int(t.element_size()) if hasattr(t, "element_size") else r.elem_size
            )
            off = loc - base
            if off < 0 or off % elem != 0:
                return None
            idx = off // elem
            if idx >= int(t.numel()):
                return None
            tensor_for_read = t.detach() if hasattr(t, "detach") else t
            tensor_for_read = (
                tensor_for_read.cpu()
                if hasattr(tensor_for_read, "cpu")
                else tensor_for_read
            )
            value = tensor_for_read.reshape(-1).tolist()[idx]
        except Exception:
            return None
        if not isinstance(value, int):  # bool is an int subclass; floats fail
            return None
        return int(value)

    # ──────────────────── reads-through ────────────────────

    def _build_reads_through(self) -> dict[tuple[int, int], BoolRef]:
        """rt(w, r): r reads-through w — directly (rf), via a bounded chain
        of value-modeled RMW write parts (each link an rf edge, so each
        intermediate's write part is implied active), or — for a counted
        record — via coherence order between its two copies (justified in
        ``_build_counting_axioms``: every write to the counted location is a
        link of the same RMW chain). Chain length over MODELED events is
        bounded by their count; chains through unmodeled grid instances are
        exactly what the counting pairs cover.

        Every rf hop of a chain must be morally strong (PTX observation
        order recurses over ``morally_strong ∩ rf``; scoped-RC11 release
        sequences use ``incl ∩ rf`` at every relay), so each link — the
        seed rf included — conjoins ``_scope_ok`` of ITS OWN endpoints.
        Ordering stays endpoint-only: interior relays may be relaxed. A
        scope-incompatible interior hop breaks the chain even when the
        chain's two endpoints are mutually inclusive (counterexample:
        a cta-scoped relay CAS in another block let a release publish
        through it, certifying a PTX-racy launch race-free). A counted
        record's chain runs through unmodeled ranks of its one textual
        operation, so a ``cta``-scoped counted record cannot guarantee
        per-hop inclusion across ranks and contributes no chain edge.
        """
        by_idx = {e.idx: e for e in self.events}
        rt: dict[tuple[int, int], BoolRef] = {
            (w_idx, r_idx): And(rf, self._scope_ok(by_idx[w_idx], by_idx[r_idx]))
            for (w_idx, r_idx), rf in self.rf_source.items()
        }
        n_modeled = len(self._modeled_atomic_events())
        frontier: dict[tuple[int, int], BoolRef] = dict(rt)
        for _ in range(max(0, n_modeled - 1)):
            grown: dict[tuple[int, int], BoolRef] = {}
            for (w_idx, m_idx), pred in frontier.items():
                for (m2_idx, r_idx), rf2 in self.rf_source.items():
                    if m2_idx != m_idx or r_idx == w_idx:
                        continue
                    step = And(
                        pred,
                        rf2,
                        self._scope_ok(by_idx[m2_idx], by_idx[r_idx]),
                    )
                    key = (w_idx, r_idx)
                    grown[key] = Or(grown[key], step) if key in grown else step
            if not grown:
                break
            for key, pred in grown.items():
                rt[key] = Or(rt[key], pred) if key in rt else pred
            frontier = grown
        for info in self._counting.values():
            if by_idx[info.idx_a].scope == "cta":
                continue
            ord_a = self.atomic_order[info.idx_a]
            ord_b = self.atomic_order[info.idx_b]
            for w_idx, r_idx, before in (
                (info.idx_b, info.idx_a, ord_b < ord_a),
                (info.idx_a, info.idx_b, ord_a < ord_b),
            ):
                key = (w_idx, r_idx)
                rt[key] = Or(rt[key], before) if key in rt else before
        return rt

    def _assert_no_uncounted_observation_addresses(self) -> None:
        """An observation variable may feed an ADDRESS only when its
        record's counting axiom fired (the address is then affine in the
        pinned rank — the work-queue pattern). Anything else raises: an unconstrained
        observation in an address would alias everything, and silently
        widening is the forbidden failure mode — unsupported is the honest
        verdict."""
        counted: set[int] = set()
        for info in self._counting.values():
            counted.update((info.idx_a, info.idx_b))
        obs_keys: dict[tuple[int, str, str], SymbolicMemoryEvent] = {}
        for e in self._modeled_atomic_events():
            if e.idx in counted:
                continue
            for key in _collect_z3_var_keys((e.old_value,)):
                obs_keys[key] = e
        if not obs_keys:
            return
        for e in self.events:
            hit = _collect_z3_var_keys((e.addr,)) & set(obs_keys)
            if hit:
                src = obs_keys[next(iter(hit))]
                error = (
                    CountingScopeUnsupported
                    if id(src.record) in self._counting_grid_restricted
                    else UnsupportedSymbolicRaceQuery
                )
                raise error(
                    f"the observation of atomic {src.name} feeds the address "
                    f"of {e.name}, and the counting axiom's guards do not "
                    "hold for it — an atomic return in address position is "
                    "only modeled under the counting axiom"
                )

    def _base_constraint_conjunction(self) -> BoolRef | None:
        """Reuse immutable assertions, with snapshots of every mutable source.

        Constraint lists and the HB matrix are public and may be changed in
        place. Checking their actual elements costs only Python work on a
        cache hit; no Boolean coercion or Z3 construction is repeated. The
        cache is first populated at query time, after co-hb has been built.
        """
        containers = (
            self.arange_constraints_a,
            self.arange_constraints_b,
            self.rf_constraints,
            self.atomic_coherence_constraints,
            self.counting_constraints,
            self.value_causality_constraints,
            self.extra_assumptions,
        )
        # Do not consume a one-shot iterable before the original add path.
        if not all(isinstance(source, (list, tuple)) for source in containers):
            self._base_constraint_cache = None
            return None
        families = (
            (self.grid_constraints,),
            tuple(self.arange_constraints_a),
            tuple(self.arange_constraints_b),
            tuple(self.rf_constraints),
            tuple(self.atomic_coherence_constraints),
            tuple(self.counting_constraints),
            tuple(self.value_causality_constraints),
            tuple(self.extra_assumptions),
        )
        rows = tuple(self.hb[i] for i in range(len(self.events)))
        diagonal = tuple(row[i] for i, row in enumerate(rows))
        sources = (
            self.grid_constraints,
            self.arange_constraints_a,
            self.arange_constraints_b,
            self.rf_constraints,
            self.atomic_coherence_constraints,
            self.counting_constraints,
            self.value_causality_constraints,
            self.extra_assumptions,
            self.events,
            self.hb,
            *rows,
        )
        snapshot = (sources, *families, diagonal)
        cached = getattr(self, "_base_constraint_cache", None)
        if cached is not None and all(
            len(current) == len(previous)
            and all(a is b for a, b in zip(current, previous))
            for current, previous in zip(snapshot, cached.snapshot)
        ):
            return cached.expression

        self._base_constraint_cache = None
        # Solver.add also accepts nested containers and custom coercible
        # values. Their contents may change without changing object identity;
        # retain the original add path for those nonstandard inputs.
        if (
            not all(
                isinstance(c, BoolRef) or type(c) is bool  # noqa: E721 (only immutable builtin scalars)
                for family in families[:-1]
                for c in family
            )
            or not all(
                isinstance(c, ExprRef) or type(c) in (bool, int, float)
                for c in families[-1]
            )
            or not all(
                isinstance(c, BoolRef) or type(c) is bool  # noqa: E721 (immutable builtin scalars)
                for c in diagonal
            )
        ):
            return None

        constraints = [c for family in families[:-1] for c in family]
        constraints.extend(as_bool(c) for c in families[-1])
        # Only syntactic False diagonals can be omitted: a True or
        # conditional cycle must still exclude its valuation from executions.
        constraints.extend(Not(c) for c in diagonal if not is_false(c))
        expression = And(*constraints)
        self._base_constraint_cache = _BaseConstraintCache(snapshot, expression)
        return expression

    def _base_solver(self) -> Solver:
        """Assertions shared by every race query; the caller adds the
        cross-instance (``different_blocks``) or same-instance constraints.
        """
        solver = self._solver()
        conjunction = self._base_constraint_conjunction()
        if conjunction is not None:
            solver.add(conjunction)
            return solver
        solver.add(self.grid_constraints)
        for c in self.arange_constraints_a:
            solver.add(c)
        for c in self.arange_constraints_b:
            solver.add(c)
        for c in self.rf_constraints:
            solver.add(c)
        for c in self.atomic_coherence_constraints:
            solver.add(c)
        for c in self.counting_constraints:
            solver.add(c)
        for c in self.value_causality_constraints:
            solver.add(c)
        for c in self.extra_assumptions:
            solver.add(as_bool(c))
        # HB irreflexivity: H = TC(po ∪ sw) must be acyclic in any valuation
        # that models an execution; without Not(H[i][i]) the solver may
        # witness race queries with cyclic po ∪ sw valuations.
        for i in range(len(self.events)):
            solver.add(Not(self.hb[i][i]))
        return solver

    def _solver(self) -> Solver:
        """A fresh Z3 solver with this solver's per-query budget."""
        solver = Solver()
        solver.set(timeout=self.timeout_ms)
        return solver

    def _new_solver(self) -> Solver:
        solver = self._base_solver()
        solver.add(self.different_blocks)
        return solver

    # ──────────────────────── Reports ────────────────────────

    @staticmethod
    def _canonical_pair(
        a: SymbolicMemoryEvent, b: SymbolicMemoryEvent
    ) -> tuple[SymbolicMemoryEvent, SymbolicMemoryEvent]:
        if (a.event_id, a.lane) <= (b.event_id, b.lane):
            return a, b
        return b, a

    def _dedupe_reports(
        self,
        candidates: list[
            tuple[SymbolicMemoryEvent, SymbolicMemoryEvent, ModelRef, str]
        ],
    ) -> list[RaceReport]:
        seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        reports: list[RaceReport] = []
        for a, b, model, reason in candidates:
            first, second = self._canonical_pair(a, b)
            key = (
                (first.event_id, first.lane),
                (second.event_id, second.lane),
            )
            if key in seen:
                continue
            seen.add(key)
            reports.append(self._make_report(first, second, model, reason))
        return reports

    def _make_report(
        self,
        first: SymbolicMemoryEvent,
        second: SymbolicMemoryEvent,
        model: ModelRef,
        reason: str,
    ) -> RaceReport:
        fw = bool(is_true(model.evaluate(first.writes, model_completion=True)))
        sw = bool(is_true(model.evaluate(second.writes, model_completion=True)))
        if fw and sw:
            race_type: RaceType = RaceType.WAW
        elif fw:
            race_type = RaceType.RAW
        else:
            race_type = RaceType.WAR

        addr_a_val = model.evaluate(first.addr, model_completion=True).as_long()
        addr_b_val = model.evaluate(second.addr, model_completion=True).as_long()
        if first.elem_size > 1 or second.elem_size > 1:
            witness_addr = max(addr_a_val, addr_b_val)
        else:
            witness_addr = addr_a_val

        witness_grid_a = tuple(
            model.evaluate(first.pid[i], model_completion=True).as_long()
            for i in range(3)
        )
        witness_grid_b = tuple(
            model.evaluate(second.pid[i], model_completion=True).as_long()
            for i in range(3)
        )

        if reason == self._INTRA_INSTANCE_REASON:
            positioned = (
                first.program_seq >= 0
                and second.program_seq >= 0
                and first.program_seq != second.program_seq
            )
            reason = append_missing_fence_diagnostic(
                reason,
                same_instance=witness_grid_a == witness_grid_b,
                distinct_operations=first.record is not second.record,
                first_seq=first.program_seq,
                second_seq=second.program_seq,
                fenced=positioned and self._fence_orders(first, second),
                first_site=first.record.source_location,
                second_site=second.record.source_location,
                first_mode="write" if fw else "read",
                second_mode="write" if sw else "read",
                pre_exit=first.record.pre_exit or second.record.pre_exit,
            )

        assert race_type is not None, "race_type_value must always be populated"
        return RaceReport(
            first=first,
            second=second,
            model=self._model_to_dict(model),
            reason=reason,
            race_type_value=race_type,
            witness_addr=int(witness_addr),
            witness_addrs=(addr_a_val, addr_b_val),
            witness_grid_a=witness_grid_a,
            witness_grid_b=witness_grid_b,
        )

    @staticmethod
    def _model_to_dict(model: ModelRef) -> dict[str, str]:
        return {decl.name(): str(model[decl]) for decl in model.decls()}


__all__ = [
    "CopyContext",
    "SymbolicMemoryEvent",
    "TwoCopySymbolicHBSolver",
]
