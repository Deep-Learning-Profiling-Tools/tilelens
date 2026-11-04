"""AccessGraph -> the two-copy solver's records (global memory).

``Encoder`` lowers every access of the TTIR reader's ``AccessGraph`` into
the ``AccessEventRecord`` shape the eager race detector gives the
``TwoCopySymbolicHBSolver``, through ``RaceLeaves``:

* ``addr_expr`` is a byte address: ``data_ptr + offset * elem_size`` for a
  launch (the T1 rungs), the byte offset from the tensor's own base at the
  any-parameters rung (T0, where accesses of different tensors never
  conflict: one encoding per tensor);
* ``mask`` and ``path`` make the record's ``active``; each enclosing loop's
  iteration premise, the in-bounds premise, the loaded values' domains and
  the width premises are local constraints (they gate the record's
  activity, and stay out of the solver's feasibility base);
* an integer RMW outside loops and a CAS are value-modeled: the access's
  observation is the record's ``old_value``;
* an awaited poll (``awaits``) gets its exit predicate as a premise of every
  record (where the predicate's own width obligations hold), and a pre-exit
  representative;
* a tile access (``AccessEvent.shape``) carries its shape and, when a lane
  variable names each of its dimensions, its ``lane_position``; otherwise
  the solver reads its lanes as not named (distinct lanes at any
  assignment) and it anchors no dependency.

The in-bounds premise: a record's address stays inside its tensor's
allocation. Without it an unbounded grid would stray into other tensors'
addresses; a race reachable only through an out-of-bounds access is the
compiled sanitizer's finding, not this model's.

The width premises: the reader's integer terms are exact while their
``width_obligations`` hold. A record is active only where its obligations
hold, each one only where a Select arm reads its term (``GuardDag``), so a
proof at an any-grid rung assumes no overflow, which ``violation`` lets the
caller check (and discharge exactly at the launch).

Over-approximated records (a dropped mask, an unmodeled branch, a free
observation or loaded value in a mask or path, a loop bound on a free
loaded value) are listed in ``uncertain``: an UNSAT over them still proves,
a SAT touching one is no witness.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from z3 import And, ArithRef, BoolRef, BoolVal, ExprRef, Implies, Int, IntVal, Not
from z3 import simplify

from ....core.data import AtomicCas, AtomicRMW, Load, Store
from ....ir.lowering import GuardDag, children, fold
from ....ir.ttir_reader import (
    AccessEvent,
    AccessGraph,
    Arange,
    Bin,
    BoolBin,
    Cmp,
    Const,
    IntCast,
    IterArgOffset,
    Loaded,
    LoopInfo,
    LoopVar,
    NumPrograms,
    Observed,
    Param,
    Pid,
    Select,
    Term,
    WidthObligation,
    width_obligations,
)
from ....ir.ttir_reader import Not as TNot
from ... import pids
from ..data import AccessEventRecord
from .fences import FenceRule
from .leaves import (
    LaunchTensor,
    RaceKind,
    RaceLeaves,
    RaceRefusal,
    loaded_leaves,
    observed_leaves,
    signed,
    terms_below,
)

_KNOWN_SEMS = ("relaxed", "acquire", "release", "acq_rel")
# The printer's RMW spellings -> the solver's op names.
_RMW_OPS = {"exch": "xchg"}
# Pre-launch values the solver's atomic read-from and counting read
# (its own initial-value cap).
INIT_VALUES_MAX_ELEMENTS = 1024


@dataclass
class Encoding:
    """The solver's input for one encoding (or one T0 tensor group)."""

    records: list[AccessEventRecord]
    arange_dict: dict[Any, Any]
    # Event ids of over-approximated records (see the module docstring).
    uncertain: set[int]
    # The snapshot equalities (``extra_assumptions``).
    assumptions: tuple[BoolRef, ...]
    # Some record read a snapshot: a verdict holds for these contents.
    content_qualified: bool
    pid_axes: frozenset[int]
    has_atomics: bool
    fences: FenceRule


@dataclass(frozen=True)
class _Tile:
    """A tile record's ``symbolic_expr``: the solver reads only its shape."""

    shape: tuple[int, ...]


@dataclass(frozen=True)
class _Loop:
    var: ArithRef  # the iteration index k
    premise: BoolRef  # k is an iteration that runs
    trips: int | None  # the trip count, when a constant
    domain: tuple[BoolRef, ...]  # its bounds' loaded values' domains


class _InitValues:
    """The tensor surface the solver's initial-value source reads, over the
    pre-launch values (the solver never touches a real tensor here)."""

    def __init__(self, facts: Any, values: tuple[int, ...]) -> None:
        self._facts = facts
        self._values = values

    def data_ptr(self) -> int:
        return self._facts.data_ptr

    def element_size(self) -> int:
        return self._facts.elem_size

    def numel(self) -> int:
        return self._facts.numel

    def is_contiguous(self) -> bool:
        return self._facts.contiguous

    def reshape(self, *_shape: Any) -> _InitValues:
        return self

    def tolist(self) -> list[int]:
        return list(self._values)


# ─────────────────────────── term predicates ───────────────────────────


def _loop_names(graph: AccessGraph, access: AccessEvent) -> tuple[str, ...]:
    if access.loops:
        return access.loops
    return (graph.loop.loop_ssa,) if access.in_loop and graph.loop else ()


def access_terms(access: AccessEvent) -> tuple[Term, ...]:
    return tuple(
        t
        for t in (
            access.offset,
            access.mask,
            access.path,
            access.atomic_val,
            access.atomic_cmp,
        )
        if t is not None
    )


def graph_terms(graph: AccessGraph) -> list[object]:
    terms: list[object] = []
    for a in graph.accesses:
        terms += access_terms(a)
    for lp in graph.loops:
        terms += [lp.lower, lp.upper, lp.step]
    for info in graph.iter_args:
        terms += [info.offset0, info.delta]
    return terms


def mentions_loaded(term: object, graph: AccessGraph) -> bool:
    return any(isinstance(t, Loaded) for t in terms_below((term,), graph))


def _conjuncts(term: Term) -> list[Term]:
    out, stack = [], [term]
    while stack:
        t = stack.pop()
        if isinstance(t, BoolBin) and t.op == "and":
            stack += [t.b, t.a]
        else:
            out.append(t)
    return out


def _same(a: object, b: object) -> bool:
    if a is b:
        return True
    try:
        return a == b
    except RecursionError:
        return False


def _runs_wherever(d: AccessEvent, a: AccessEvent) -> bool:
    """``d`` runs at every lane and iteration where ``a`` does: no dropped
    mask or unmodeled branch, its loops enclose ``a``, and its mask and path
    conjuncts are among ``a``'s. (Where ``d`` is masked off, a value ``a``
    reads from it does not come from ``d``'s sources.)"""
    if d.mask_dropped or d.guarded or a.loops[: len(d.loops)] != d.loops:
        return False
    have = [c for t in (a.mask, a.path) if t is not None for c in _conjuncts(t)]
    return all(
        (isinstance(c, Const) and bool(c.value)) or any(_same(c, h) for h in have)
        for t in (d.mask, d.path)
        if t is not None
        for c in _conjuncts(t)
    )


def _var_ids(terms: Sequence[Any]) -> set[int]:
    """The ids of every node of ``terms`` (Z3 interns them: a variable has
    one id wherever it occurs)."""
    seen: set[int] = set()
    stack = [t for t in terms if isinstance(t, ExprRef)]
    while stack:
        t = stack.pop()
        i = t.get_id()
        if i not in seen:
            seen.add(i)
            stack.extend(t.children())
    return seen


# Symbolic leaves: at T0 every leaf but constants; at T1 the params are
# concrete.
_T0_SYMBOLIC: tuple[type, ...] = (Pid, Param, Arange, LoopVar, IterArgOffset)
_T0_SYMBOLIC += (Observed, NumPrograms, Loaded)
_T1_SYMBOLIC = tuple(c for c in _T0_SYMBOLIC if c is not Param)


def linear(term: object, graph: AccessGraph, symbolic: tuple[type, ...]) -> bool:
    """No product of two symbolic terms and no symbolic divisor (``symbolic``
    names the leaves a rung leaves symbolic), the shape Z3 decides."""

    def apply(t: object, kids: Sequence[tuple[bool, bool]]) -> tuple[bool, bool]:
        sym = isinstance(t, symbolic) or any(s for s, _ in kids)
        lin = all(ok for _, ok in kids)
        if isinstance(t, Bin) and t.op == "*":
            lin = lin and not (kids[0][0] and kids[1][0])
        elif isinstance(t, Bin) and t.op in ("//", "%", "u//", "u%"):
            lin = lin and not kids[1][0]
        elif isinstance(t, (LoopVar, IterArgOffset)):
            # lower + k * step, offset0 + k * delta
            lin = kids[0][1] and not kids[1][0]
        return sym, lin

    return fold(term, graph, apply, {})[1]


def _cas_operand_ok(term: object, graph: AccessGraph) -> bool:
    """A CAS operand with no arithmetic and no width change: a literal, an
    observation, a comparison of atoms, its i1 zero extension, or a Select
    of those (an observed i32 + 1 must not become unbounded addition where
    it decides the CAS)."""
    atoms = (Const, Pid, Arange, Observed)

    def apply(t: object, kids: Sequence[tuple[bool, bool]]) -> tuple[bool, bool]:
        # (a condition, a value)
        if isinstance(t, Const):
            return True, True
        if isinstance(t, Observed):
            return False, True
        if isinstance(t, Cmp):
            ok = isinstance(t.a, atoms) and isinstance(t.b, atoms)
            return ok, ok
        if isinstance(t, TNot):
            return kids[0][0], kids[0][0]
        if isinstance(t, BoolBin):
            ok = kids[0][0] and kids[1][0]
            return ok, ok
        if isinstance(t, IntCast) and t.kind == "extui" and t.src_bits == 1:
            return False, kids[0][0]
        if isinstance(t, Select):
            return False, kids[0][0] and kids[1][1] and kids[2][1]
        return False, False

    return fold(term, graph, apply, {})[1]


def _widens(term: object, graph: AccessGraph) -> bool:
    """``term`` changes an integer's width (an IntCast other than an i1's
    zero extension)."""
    return any(
        isinstance(t, IntCast) and not (t.kind == "extui" and t.src_bits == 1)
        for t in terms_below((term,), graph)
    )


def content_free_view(access: AccessEvent, graph: AccessGraph) -> AccessEvent:
    """The access with every loaded value free: a mask or path conjunct
    that reads one is dropped (the solver could pick its value either way,
    and it may be nonlinear), and the access is flagged over-approximated.
    An address on a loaded value is left alone (it refuses)."""

    def strip(t: Term | None) -> tuple[Term | None, bool]:
        if t is None:
            return None, False
        conj = _conjuncts(t)
        kept = [c for c in conj if not mentions_loaded(c, graph)]
        if len(kept) == len(conj):
            return t, False
        if not kept:
            return None, True
        out = kept[0]
        for c in kept[1:]:
            out = BoolBin("and", out, c)
        return out, True

    mask, m_dropped = strip(access.mask)
    path, p_dropped = strip(access.path)
    if not (m_dropped or p_dropped):
        return access
    return replace(
        access,
        mask=mask,
        path=path,
        mask_dropped=access.mask_dropped or m_dropped,
        guarded=access.guarded or p_dropped,
    )


def t0_linearity_gate(graph: AccessGraph, awaits: Mapping[int, Term]) -> bool:
    """The any-parameters rung is tried only when every address, mask, path
    and exit predicate is linear with the params symbolic."""
    for i, access in enumerate(graph.accesses):
        view = content_free_view(access, graph)
        terms = [view.offset, view.mask, view.path, awaits.get(i)]
        if not all(linear(t, graph, _T0_SYMBOLIC) for t in terms if t is not None):
            return False
    return True


def written_sources(
    graph: AccessGraph, tensors: Mapping[str, LaunchTensor]
) -> dict[str, str]:
    """The tensors loaded values are read from that the kernel may write:
    their snapshot does not stand for what the load reads (the static model
    cannot order the instances), so their values stay free."""
    sources = {
        graph.accesses[leaf.access_index].base_param
        for leaf in loaded_leaves(graph_terms(graph), graph)
    }
    written = {a.base_param for a in graph.accesses if a.kind != "load"}

    def interval(name: str) -> tuple[int, int] | None:
        t = tensors.get(name)
        return None if t is None else t.facts.allocation_interval()

    out: dict[str, str] = {}
    for src in sorted(sources):
        si = interval(src)
        for w in sorted(written):
            wi = interval(w)
            if src == w or (si and wi and max(si[0], wi[0]) < min(si[1], wi[1])):
                out[src] = f"the kernel writes {w!r}, which overlaps the source"
                break
    return out


def symbolic_grid(
    enc: Encoding, launch_grid: tuple[int, int, int] | None, t0: bool = False
) -> tuple[Any, Any, Any]:
    """The grid of an any-grid rung: symbolic along the axes the kernel
    reads; along an unread axis the launch's extent (blocks that differ
    only there behave alike, but a launch along it is still checked), or
    symbolic for a kernel with atomics without a launch extent or at T0
    (their blocks need not behave alike)."""

    def dim(i: int) -> Any:
        if i in enc.pid_axes or (enc.has_atomics and (t0 or launch_grid is None)):
            return pids.grid_var(i)
        return int(launch_grid[i]) if launch_grid is not None else 1

    return (dim(0), dim(1), dim(2))


def grid_pins(grid: tuple[Any, Any, Any], launch: tuple[int, int, int]) -> tuple:
    """Equalities pinning the symbolic dims of ``grid`` to the launch."""
    return tuple(
        d == IntVal(launch[i]) for i, d in enumerate(grid) if not isinstance(d, int)
    )


def _fits(value: ArithRef, bits: int, is_signed: bool) -> BoolRef:
    if is_signed:
        half = 1 << (bits - 1)
        return And(value >= -half, value < half)
    return And(value >= 0, value < (1 << bits))


def _is_increment(ob: WidthObligation, loop: LoopInfo) -> bool:
    """The loop's increment obligation (``upper - 1 + step``), which holds
    only where the loop runs."""
    t = ob.term
    return (
        isinstance(t, Bin)
        and t.op == "+"
        and t.b is loop.step
        and isinstance(t.a, Bin)
        and t.a.op == "-"
        and t.a.a is loop.upper
    )


def _guard_operands(t: object, graph: AccessGraph) -> tuple:
    if isinstance(t, Loaded):
        return tuple(k for k in (t.offset, t.mask, t.other) if k is not None)
    return children(t, graph)


# ─────────────────────────── the encoder ───────────────────────────


class Encoder:
    """One encoding of ``graph``: ``params`` None is the any-parameters
    rung (T0); otherwise the launch's params and tensors (T1), and
    ``blind`` reads no snapshot (the content-free attempt)."""

    def __init__(
        self,
        graph: AccessGraph,
        params: Mapping[str, int] | None,
        tensors: Mapping[str, LaunchTensor],
        awaits: Mapping[int, Term],
        *,
        blind: bool = False,
    ) -> None:
        self.graph = graph
        self.t0 = params is None
        self.tensors = tensors
        self.awaits = awaits
        unusable = {} if self.t0 else written_sources(graph, tensors)
        self.leaves = RaceLeaves(graph, params, tensors, blind=blind, unusable=unusable)
        self.modeled = {
            i
            for i, a in enumerate(graph.accesses)
            if a.kind in ("atomic_rmw", "atomic_cas")
            and not a.elem_float
            and not a.in_loop
        }
        # Dependency order is causal, so it composes: an access that uses
        # an atomic's result at a lane comes after the loads that atomic
        # used at that lane, where the atomic runs (the reader lists direct
        # sources only).
        self.deps: list[tuple[int, ...]] = []
        for a in graph.accesses:
            found = set(a.deps)
            for d in a.deps:
                if _runs_wherever(graph.accesses[d], a):
                    found.update(self.deps[d])
            self.deps.append(tuple(sorted(found)))
        # access index -> its lane position (see lane_position)
        self._positions: dict[int, tuple[Any, ...] | None] = {}
        self.loops: dict[str, _Loop] = {}
        self.free_bound_loops: set[str] = set()
        for index, lp in enumerate(graph.loops):
            self._bind_loop(lp, index)
        self.fences = FenceRule(
            graph, {name: (b.var, b.trips) for name, b in self.loops.items()}
        )
        # (width, where one obligation of an encoded record or an await's
        # exit predicate fails)
        self.violations: list[tuple[int, BoolRef]] = []
        self.await_premises, self.await_obs = self._await_premises()

    # ── loops ──

    def _bind_loop(self, lp: LoopInfo, index: int) -> None:
        leaves = self.leaves
        try:
            lower = leaves.value(lp.lower)
            upper = leaves.value(lp.upper)
            step = leaves.value(lp.step)
        except RaceRefusal as r:
            raise r.at(lp.line_no, lp.loc)
        bound_leaves = loaded_leaves((lp.lower, lp.upper), self.graph)
        if any(
            self._free(leaf) or self._pad_reaches(leaf, ()) for leaf in bound_leaves
        ):
            # The iteration range is over-approximated.
            self.free_bound_loops.add(lp.loop_ssa)
        domain = leaves.domain_premises(bound_leaves)
        lower_c, upper_c, step_c = (_as_long(v) for v in (lower, upper, step))

        def refuse(why: str) -> RaceRefusal:
            return RaceRefusal(RaceKind.LOOP_BOUND, why, line_no=lp.line_no, loc=lp.loc)

        # A symbolic k * step is nonlinear; scf.for needs a positive step.
        if step_c is None:
            raise refuse("the loop step is not a compile-time constant")
        if step_c <= 0:
            raise refuse(f"the loop step is {step_c}; only positive steps are modeled")
        k = Int("ttir_loop_k" if index == 0 else f"ttir_loop_k{index}")
        trips: int | None = None
        if lower_c is not None and upper_c is not None:
            trips = max(0, -(-(upper_c - lower_c) // step_c))
            premise = And(k >= 0, k < trips)
        else:
            # A bound that stays symbolic: at T1 only one linear in the
            # pids, lanes and iterations (a grid-stride or triangular loop).
            if not self.t0 and not all(
                linear(t, self.graph, _T1_SYMBOLIC) for t in (lp.lower, lp.upper)
            ):
                raise refuse(
                    "the loop bounds are neither constant at the launch nor "
                    "linear in the program ids and iterations"
                )
            premise = And(k >= 0, lower + k * IntVal(step_c) < upper)
        self.loops[lp.loop_ssa] = _Loop(k, premise, trips, domain)
        # A single iteration reads its bounds' lower value directly.
        leaves.iterations[lp.loop_ssa] = IntVal(0) if trips == 1 else k

    def iteration_premises(self, access: AccessEvent) -> tuple[BoolRef, ...]:
        out: list[BoolRef] = []
        for name in _loop_names(self.graph, access):
            b = self.loops[name]
            out += [b.premise, *b.domain]
        return tuple(out)

    def zero_trip(self, access: AccessEvent) -> bool:
        return any(
            self.loops[name].trips == 0 for name in _loop_names(self.graph, access)
        )

    # ── loaded values ──

    def _free(self, leaf: Loaded) -> str | None:
        return self.leaves.snapshot_problem(self.graph.accesses[leaf.access_index])

    def _pad_reaches(self, leaf: Loaded, guards: Sequence[Term]) -> bool:
        """The load's masked-off lanes have no value (a mask without
        ``other``) and the consumer's own mask and path do not repeat the
        load's mask (which would keep those lanes inactive)."""
        if leaf.mask is None or leaf.other is not None or self._free(leaf):
            return False
        have = [c for g in guards for c in _conjuncts(g)]
        return not all(any(_same(c, h) for h in have) for c in _conjuncts(leaf.mask))

    def widened(self, access: AccessEvent, exit_pred: Term | None) -> bool:
        """The access's activity or address is over-approximated."""
        if access.mask_dropped or access.guarded:
            return True
        for t in (access.mask, access.path):
            if t is not None and observed_leaves((t,), self.graph) - self.modeled:
                return True  # a free observation
        guards = [t for t in (access.mask, access.path) if t is not None]
        for t in (access.offset, access.mask, access.path, exit_pred):
            if t is None:
                continue
            for leaf in loaded_leaves((t,), self.graph):
                if t is not access.offset and self._free(leaf):
                    return True
                if self._pad_reaches(leaf, guards):
                    return True
        return any(
            name in self.free_bound_loops for name in _loop_names(self.graph, access)
        )

    # ── awaits ──

    def _await_premises(self) -> tuple[tuple[BoolRef, ...], tuple[ArithRef, ...]]:
        """For each await, ``reach(poll) -> exit``: every terminating run
        satisfies it, so it holds in every record's activity (asserting it
        on the poll alone would let a model leave the poll inactive and drop
        its synchronization). The exit predicate is exact only where its
        width obligations hold: the premise asks them too, and where the
        poll is reached they join ``violations``. A guarded await gives
        none."""
        graph, leaves = self.graph, self.leaves
        premises, obs = [], []
        for index, exit_pred in self.awaits.items():
            access = graph.accesses[index]
            if access.guarded:
                continue
            try:
                exit_b = leaves.cond(exit_pred)
                guard = [
                    leaves.cond(t) for t in (access.mask, access.path) if t is not None
                ]
                # The predicate's own obligations, as a mask's.
                probe = replace(
                    access,
                    loops=(),
                    in_loop=False,
                    path=None,
                    mask=exit_pred,
                    offset=Const(0),
                )
                dag = GuardDag([exit_pred], graph, leaves.cond, _guard_operands)
                reach = And(
                    *guard,
                    *self.iteration_premises(access),
                    *leaves.domain_premises(loaded_leaves((exit_pred,), graph)),
                )
                held: list[BoolRef] = []
                for ob in width_obligations(graph, probe):
                    premise = self._held(dag, ob)
                    held.append(premise)
                    self.violations.append((ob.bits, And(reach, Not(premise))))
            except RaceRefusal as r:
                raise r.at(access.line_no, access.loc)
            if guard or held:
                premises.append(Implies(And(*guard, *held), exit_b))
            else:
                premises.append(exit_b)
            obs.append(leaves.observation(index))
        return tuple(premises), tuple(obs)

    def _await_dependencies(self, seq: int, access: AccessEvent) -> tuple[int, ...]:
        if access.guarded or access.mask_dropped:
            return ()
        return tuple(
            i
            for i in self.awaits
            if i < seq
            and not self.graph.accesses[i].guarded
            and not self.graph.accesses[i].mask_dropped
        )

    # ── width premises ──

    def _held(self, dag: GuardDag, ob: WidthObligation) -> BoolRef:
        """Where obligation ``ob`` holds or no Select arm reads its term."""
        fits = _fits(self.leaves.value(ob.term), ob.bits, ob.signed)
        # A signed quotient that does not fit is undefined in either arm.
        undefined = isinstance(ob.term, Bin) and ob.term.op == "//"
        return fits if undefined else dag.premise(ob.term, fits)

    def _width(
        self, access: AccessEvent
    ) -> tuple[BoolRef | None, list[tuple[int, BoolRef]]]:
        """The access's width premise (each obligation where a Select arm
        reads its term) and where it fails while it matters: a loop's
        obligations where the loop is reached (its increment where it
        runs), the path's at every iteration, the mask's where the path
        holds, the offset's where the access runs."""
        graph, leaves = self.graph, self.leaves
        names = _loop_names(graph, access)
        loops: list[LoopInfo] = []
        for n in names:
            lp = graph.loop_info(n) or graph.loop
            assert lp is not None
            loops.append(lp)
        roots: list[object] = []
        for lp in loops:
            roots += [lp.lower, lp.upper, lp.step]
        roots += [access.path, access.mask, access.offset]
        dag = GuardDag(roots, graph, leaves.cond, _guard_operands)
        path = leaves.cond(access.path) if access.path is not None else BoolVal(True)
        mask = leaves.cond(access.mask) if access.mask is not None else BoolVal(True)
        iters = [self.loops[n].premise for n in names]
        held: list[BoolRef] = []
        failing: list[tuple[int, BoolRef]] = []

        def need(ob: WidthObligation, where: BoolRef) -> None:
            premise = self._held(dag, ob)
            held.append(premise)
            failing.append((ob.bits, And(where, Not(premise))))

        for j, (name, lp) in enumerate(zip(names, loops)):
            probe = replace(
                access,
                loops=(name,),
                in_loop=True,
                path=None,
                mask=None,
                offset=Const(0),
            )
            reach = And(path, *iters[:j])
            for ob in width_obligations(graph, probe):
                if ob.role == "loop":
                    runs = _is_increment(ob, lp)
                    need(ob, And(reach, iters[j]) if runs else reach)
        where = {
            "path": And(*iters) if iters else BoolVal(True),
            "mask": And(*iters, path),
            "offset": And(*iters, path, mask),
        }
        flat = replace(access, loops=(), in_loop=False)
        for ob in width_obligations(graph, flat):
            need(ob, where[ob.role])
        return (And(*held) if held else None), failing

    # ── lane positions ──

    def lane_position(self, seq: int) -> tuple[Any, ...] | None:
        """The block position of access ``seq``'s lanes: per dimension of
        its tile, the lane variable of that ``(dim, extent)`` (0 along an
        extent-1 dimension). None when a dimension is named by no lane
        variable its address and activity read (a broadcast pointer, a block
        on one address), or when they do not lower."""
        if seq in self._positions:
            return self._positions[seq]
        access = self.graph.accesses[seq]
        leaves = self.leaves
        position: tuple[Any, ...] | None = None
        try:
            terms = [leaves.value(access.offset)]
            terms += [
                leaves.cond(t) for t in (access.mask, access.path) if t is not None
            ]
        except RaceRefusal:
            terms = []
        if terms:
            ids = _var_ids(terms)
            rank = len(access.shape)
            coords: list[Any] = []
            for d, n in enumerate(access.shape):
                if n == 1:
                    coords.append(IntVal(0))
                    continue
                # a rank-1 tile's aranges are unplaced (dim -1)
                keys = [(-1, n), (0, n)] if rank == 1 else [(d, n)]
                found = [
                    leaves.lanes[k]
                    for k in keys
                    if k in leaves.lanes and leaves.lanes[k].get_id() in ids
                ]
                if not found:
                    coords = []
                    break
                coords.append(found[0])
            position = tuple(coords) if coords else None
        self._positions[seq] = position
        return position

    def _tile(self, seq: int) -> bool:
        return any(n > 1 for n in self.graph.accesses[seq].shape)

    def _anchors(self, seq: int) -> tuple[int, ...]:
        """The dependency anchors of access ``seq``: a tile anchor only
        with a lane position (the solver compares positions through it;
        a block without one has no position to be ordered at)."""
        return tuple(
            d
            for d in self.deps[seq]
            if not self._tile(d) or self.lane_position(d) is not None
        )

    # ── one record ──

    def record(self, seq: int, access: AccessEvent) -> AccessEventRecord:
        try:
            return self._record(seq, access)
        except RaceRefusal as r:
            raise r.at(access.line_no, access.loc)

    def _record(self, seq: int, access: AccessEvent) -> AccessEventRecord:
        graph, leaves = self.graph, self.leaves
        where = f"line {access.line_no}"
        meta = None if self.t0 else self.tensors.get(access.base_param)
        if not self.t0:
            if meta is None:
                raise RaceRefusal(
                    RaceKind.MISSING_BINDING,
                    f"pointer argument {access.base_param!r} has no tensor binding",
                )
            if meta.facts.allocation_interval() is None:
                layout = "" if meta.facts.contiguous else "non-contiguous "
                raise RaceRefusal(
                    RaceKind.MISSING_BINDING,
                    f"{layout}tensor {access.base_param!r}: the in-bounds "
                    "premise needs a verified allocation extent",
                )
        if access.elem_bits <= 0:
            raise RaceRefusal(
                RaceKind.OTHER, f"unknown element width for {access.base_param!r}"
            )
        elem = (access.elem_bits + 7) // 8  # an i1 is a byte
        old_value: Any = None
        rmw_op: str | None = None
        rmw_operand: Any = None
        cas_cmp: Any = None
        cas_new: Any = None
        exit_pred = self.awaits.get(seq)
        if access.kind == "atomic_rmw" and seq in self.modeled:
            assert access.atomic is not None
            old_value = leaves.observation(seq)
            op = access.atomic.rmw_op
            rmw_op = _RMW_OPS.get(op, op) if op else None
            if access.atomic_val is not None:
                try:
                    rmw_operand = leaves.value(access.atomic_val)
                except RaceRefusal:
                    rmw_operand = None  # an unmodeled operand: the write stays open
        elif access.kind == "atomic_cas":
            cas_cmp, cas_new = self._cas(seq, access, exit_pred is not None)
            old_value = leaves.observation(seq)
        if observed_leaves((access.offset,), graph) - self.modeled:
            raise RaceRefusal(
                RaceKind.INDIRECT_ADDRESS,
                f"{where}: the address depends on an atomic observation that is "
                "not value-modeled (a float or loop-carried atomic)",
            )
        # (a free loop bound only widens the loop's accesses)
        for leaf in loaded_leaves((access.offset,), graph, loops=False):
            why = self._free(leaf)
            if why is not None:
                raise RaceRefusal(
                    RaceKind.SNAPSHOT_BOUND
                    if why.startswith("too large")
                    else RaceKind.INDIRECT_ADDRESS,
                    f"{where}: the address depends on a loaded value with no "
                    f"usable snapshot ({why})",
                )
        offset = leaves.value(access.offset)
        bounds: tuple[BoolRef, ...] = ()
        if meta is None:
            addr = offset * IntVal(elem)
        else:
            facts = meta.facts
            if facts.elem_size != elem:
                raise RaceRefusal(
                    RaceKind.OTHER,
                    f"element width mismatch for {access.base_param!r}: the TTIR "
                    f"reads {elem} bytes, the launch tensor has {facts.elem_size}",
                )
            addr = IntVal(facts.data_ptr) + offset * IntVal(elem)
            start, end = facts.allocation_interval()  # type: ignore[misc]
            bounds = (addr >= IntVal(start), addr + IntVal(elem) <= IntVal(end))
        active: Any = True
        conds = [leaves.cond(t) for t in (access.mask, access.path) if t is not None]
        if conds:
            active = conds[0] if len(conds) == 1 else And(*conds)

        atomic = access.atomic
        scope: str | None = None
        reads: Any = None
        writes: Any = None
        if access.kind in ("atomic_rmw", "atomic_cas"):
            assert atomic is not None
            sem: Any = atomic.sem if atomic.sem in _KNOWN_SEMS else "relaxed"
            scope = atomic.scope
            is_atomic = True
            access_mode: Any = "read"
            if access.kind == "atomic_rmw":
                op_type: type = AtomicRMW
                atomic_kind: Any = "rmw"
                reads, writes = True, True
            else:
                # The solver's CAS lowering writes only on success.
                op_type, atomic_kind = AtomicCas, "cas"
                reads = True
        else:
            sem, is_atomic, atomic_kind = "plain", False, "none"
            op_type = Store if access.kind == "store" else Load
            access_mode = "write" if access.kind == "store" else "read"

        value_terms = [access.mask, access.path, exit_pred]
        if access.kind == "atomic_cas":
            value_terms += [access.atomic_cmp, access.atomic_val]
        roots = [access.offset, *(t for t in value_terms if t is not None)]
        loaded = loaded_leaves(roots, graph)
        iteration = self.iteration_premises(access)
        width, failing = self._width(access)
        domain = leaves.domain_premises(loaded)
        self.violations += [(bits, And(*domain, f)) for bits, f in failing]
        # Every observation a record reads is renamed per copy: the solver
        # renames only the copy-local variables of the records it is given.
        # Of the loops, only the record's own: the solver reads them as the
        # record's iterators, and one of a loop the access is not in would
        # make one execution two. The loops of an awaited poll too, whose
        # premise rides every record.
        names = dict.fromkeys(_loop_names(graph, access))
        for index in self.awaits:
            names.update(dict.fromkeys(_loop_names(graph, graph.accesses[index])))
        copy_local = tuple(self.loops[n].var for n in names)
        copy_local += tuple(
            leaves.observation(i) for i in sorted(observed_leaves(roots, graph))
        )
        if old_value is not None:
            copy_local += (old_value,)
        copy_local += self.await_obs + leaves.pads
        tensor = None
        if old_value is not None and meta is not None:
            tensor = _init_values(meta)
        source = (
            (access.loc.file, access.loc.line, graph.kernel_name)
            if access.loc is not None
            else None
        )
        tile = _Tile(tuple(access.shape)) if self._tile(seq) else None
        return AccessEventRecord(
            op_type=op_type,
            access_mode=access_mode,
            tensor=tensor,  # type: ignore[arg-type]
            tensor_name=access.base_param,
            symbolic_expr=tile,
            addr_expr=addr,
            # The await premises hold in every execution; the iteration
            # ranges gate only this record (several loops may run for
            # disjoint instances).
            premises=self.await_premises,
            local_constraints=bounds
            + iteration
            + domain
            + ((width,) if width is not None else ()),
            source_location=source,
            program_seq=seq,
            dep_loads=self._anchors(seq),
            lane_position=self.lane_position(seq) if tile else None,
            causal_constraints=iteration,
            await_observations=(
                tuple(sorted(observed_leaves((exit_pred,), graph)))
                if exit_pred is not None
                else ()
            ),
            await_dependencies=self._await_dependencies(seq, access),
            debug_name=f"{graph.kernel_name}:ttir{access.line_no}:{access.kind}",
            active=active,
            reads=reads,
            writes=writes,
            is_atomic=is_atomic,
            atomic_kind=atomic_kind,
            sem=sem,
            scope=scope,
            old_value=old_value,
            rmw_op=rmw_op,
            rmw_operand=rmw_operand,
            cas_cmp_value=cas_cmp,
            cas_new_value=cas_new,
            event_id=seq,
            elem_size=elem,
            copy_local_vars=copy_local,
        )

    def _cas(self, seq: int, access: AccessEvent, awaited: bool) -> tuple[Any, Any]:
        """The CAS's compare and new values. Neither may be dropped (they
        decide success and the read-from), so an operand outside the exact
        integer fragment refuses."""
        graph = self.graph
        where = f"line {access.line_no}"

        def refuse(why: str) -> RaceRefusal:
            return RaceRefusal(RaceKind.CAS_VALUE, f"{where}: {why}")

        if access.in_loop:
            raise RaceRefusal(
                RaceKind.CONTROL_FLOW,
                f"{where}: CAS inside scf.for (one observation cannot stand "
                "for one per iteration)",
            )
        if access.elem_float:
            raise refuse("a float-typed CAS is outside the integer value model")
        if access.atomic_cmp is None or access.atomic_val is None:
            raise refuse("the CAS compare or new value is not modelable")
        if not awaited:
            for operand in (access.atomic_cmp, access.atomic_val):
                if loaded_leaves((operand,), graph):
                    raise refuse("a CAS operand depends on a plain loaded value")
                observed = observed_leaves((operand,), graph)
                if observed - self.modeled:
                    raise refuse(
                        "a CAS operand depends on an atomic observation without "
                        "an integer value model"
                    )
                if _widens(operand, graph):
                    raise refuse("a CAS operand changes an integer's width")
                if not _cas_operand_ok(operand, graph):
                    raise refuse(
                        "CAS operand arithmetic is outside the modeled integer "
                        "fragment"
                    )
                if any(
                    graph.accesses[i].elem_bits != access.elem_bits for i in observed
                ):
                    raise refuse("a CAS operand changes an observation's width")
        return self.leaves.value(access.atomic_cmp), self.leaves.value(
            access.atomic_val
        )

    # ── encodings ──

    def domain(self) -> tuple[BoolRef, ...]:
        """The ranges every query assumes, for the obligation check."""
        out: list[BoolRef] = []
        for axis in pids.PID_AXES:
            g = pids.grid_var(axis)
            out += [pids.pid_var(axis) >= 0, pids.pid_var(axis) < g, g >= 1]
        for (start, end, *_), (var, _) in self.leaves.arange_dict.items():
            out += [var >= start, var < end]
        return tuple(out) + tuple(self.leaves.assumptions) + self.await_premises

    def encode(
        self, accesses: Sequence[tuple[int, AccessEvent]] | None = None
    ) -> Encoding:
        """The records of ``accesses`` (by default every access of the
        graph; an access of a zero-trip loop runs never and is left out)."""
        graph = self.graph
        items = list(enumerate(graph.accesses)) if accesses is None else list(accesses)
        records: list[AccessEventRecord] = []
        uncertain: set[int] = set()
        # Pre-exit representatives get ids above every access's.
        next_rep = len(graph.accesses)
        for seq, access in items:
            if self.zero_trip(access):
                continue
            rec = self.record(seq, access)
            records.append(rec)
            widened = self.widened(access, self.awaits.get(seq))
            if widened:
                uncertain.add(seq)
            if seq in self.awaits:
                records.append(pre_exit_representative(rec, access, next_rep))
                if widened:
                    uncertain.add(next_rep)
                next_rep += 1
        return Encoding(
            records=records,
            arange_dict=dict(self.leaves.arange_dict),
            uncertain=uncertain,
            assumptions=tuple(self.leaves.assumptions),
            content_qualified=self.leaves.used_snapshot,
            pid_axes=frozenset(graph.pid_axes),
            has_atomics=any(a.atomic is not None for a in graph.accesses),
            fences=self.fences,
        )


def _as_long(v: ArithRef) -> int | None:
    try:
        return simplify(v).as_long()
    except Exception:
        return None


def _init_values(meta: LaunchTensor) -> _InitValues | None:
    """The pre-launch values an atomic's read-from may start from: a small
    signed integer tensor's (an unsigned one wraps by definition)."""
    facts = meta.facts
    if (
        meta.values is None
        or not facts.contiguous
        or facts.numel > INIT_VALUES_MAX_ELEMENTS
        or facts.dtype.startswith("torch.uint")
    ):
        return None
    bits = facts.elem_size * 8
    return _InitValues(facts, tuple(signed(int(v), bits) for v in meta.values))


def pre_exit_representative(
    poll: AccessEventRecord, access: AccessEvent, event_id: int
) -> AccessEventRecord:
    """One record for every failed iteration of an awaited poll: the poll's
    footprint, activity, sem, scope and program_seq (unordered with the
    poll, ordered like it against everything else) under a fresh id, with
    no value model, so no read-from or synchronization reaches it."""
    if access.kind == "atomic_cas":
        # A failed CAS reads and writes nothing: a read-only RMW record.
        extra: dict[str, Any] = dict(
            op_type=AtomicRMW, atomic_kind="rmw", reads=True, writes=False
        )
    elif access.kind == "atomic_rmw":
        # An identity RMW writes back what it read on every iteration.
        extra = dict(reads=True, writes=True)
    else:
        extra = dict(reads=None, writes=None)
    return replace(
        poll,
        event_id=event_id,
        debug_name=f"{poll.debug_name}:pre-exit",
        pre_exit=True,
        old_value=None,
        rmw_op=None,
        rmw_operand=None,
        cas_cmp_value=None,
        cas_new_value=None,
        **extra,
    )


def graph_mentions_loaded(graph: AccessGraph) -> bool:
    return bool(loaded_leaves(graph_terms(graph), graph))


__all__ = [
    "Encoder",
    "Encoding",
    "content_free_view",
    "graph_mentions_loaded",
    "grid_pins",
    "linear",
    "pre_exit_representative",
    "symbolic_grid",
    "t0_linearity_gate",
    "written_sources",
]
