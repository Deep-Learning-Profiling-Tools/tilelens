"""TTIR reader shared by the compiled-mode clients.

Reads the pre-optimization Triton IR (TTIR) of one kernel specialization
into an ``AccessGraph``: the kernel's function arguments, every global
memory access (``tt.load`` / ``tt.store`` / ``tt.atomic_rmw`` /
``tt.atomic_cas``) as an *element offset* expression relative to a base
pointer argument, the mask guarding it, and the loop structure. Scalar
arguments (``n_elements``, ``M``, strides, ...) stay symbolic (``Param``
nodes) and are substituted with concrete launch values later;
``tl.constexpr`` values are already folded into TTIR constants.

Why TTIR (not TTGIR): element addressing is cleanest here, before
layouts/pipelining add noise, and TTIR has no indirect loads unless the
kernel itself gathers — the data-dependent case, marked with ``DataDep``.

This module is mechanism-only: it reads and flags (``DataDep`` markers,
``guarded`` accesses, width obligations, ``UnsupportedTTIR``); what to do
about a flagged or unsupported kernel is the policy of each client that
consumes the graph. It either represents the IR faithfully or raises an
``UnsupportedTTIR`` whose ``kind`` says what it cannot represent.

Structure comes from ``_mlir_walk``: Triton's MLIR bindings parse the text
and give the op tree, values, types and the attributes the reader reads; a
line scan of the same text adds each op's line and the locs the bindings do
not expose. The reader walks that op tree over regions and blocks, and its
environment is keyed by the walk's value indices, never by printed SSA
names.

Address model: ``tt.addptr(base, off)`` accumulates an ELEMENT offset; the
byte address is ``base.data_ptr() + offset * elem_size``. An access is OOB
iff, for some program id / arange lane / loop iteration with its mask true,
the element offset escapes ``[0, numel)`` of its base tensor.

Integer model: every integer term denotes the IR value's SIGNED
reading as an unbounded integer (i1 terms are booleans, 0/1). That reading
is exact while the ``width_obligations`` of the access hold (they include
the loop's own increment); a consumer that evaluates terms with unbounded
integers must discharge (or report) them. ``Param`` values are the IR's
signed reading of the argument too. Division or remainder by zero is
undefined in the IR rather than a width condition, and so is an scf.for
step that is not positive: no obligation excludes them, that is the
consumer's call.

Terms are frozen dataclasses with the generated ``==`` / ``hash`` /
``repr``, which recurse: on a term deeper than Python's recursion limit
(``kernel_deep_chain`` has more than 1000 levels) they, and
``copy.deepcopy``, raise ``RecursionError``; only ``pickle`` and this
module's own walkers are iterative. Consumers of such graphs key memo
tables by identity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Iterable, Iterator, NoReturn, Sequence

from ._mlir_walk import (
    MisalignedModule,
    Module,
    ModuleParseError,
    Op,
    SourceLoc,
    parse_type,
    walk_module,
)


class TTIRKind(str, Enum):
    """What the reader cannot represent. Only representational limits live
    here; refusals a client makes about a graph it did receive (a
    data-dependent mask, a CAS value, ...) are that client's own kinds."""

    INDIRECT_ADDRESS = "indirect-address"
    DATA_DEPENDENT_BOUND = "data-dependent-bound"
    NESTED_LOOP = "nested-loop"
    CONTROL_FLOW = "control-flow"
    OUT_OF_VOCABULARY = "out-of-vocabulary"
    CALL = "call"
    LOOP_VARIANT_ADVANCE = "loop-variant-advance"
    INLINE_ASM = "inline-asm"
    READER_MISALIGNMENT = "reader-misalignment"
    UNPARSABLE = "unparsable"
    OTHER = "other"

    def __str__(self) -> str:
        return self.value

    def __format__(self, spec: str) -> str:
        return format(self.value, spec)


class UnsupportedTTIR(Exception):
    """Raised for constructs outside the compiled-mode model (indirect or
    data-dependent addressing, block pointers, nested loops, calls, ...).

    ``kind`` (a :class:`TTIRKind`) is the machine-readable class of the
    limitation, ``message`` the human-readable detail, ``line_no`` the
    refused op's line in the TTIR text and ``loc`` its user-source location
    (None when unknown). Clients read these fields; ``str(exc)`` is the
    message alone.
    """

    def __init__(
        self,
        kind: TTIRKind | str,
        message: str,
        *,
        line_no: int | None = None,
        loc: SourceLoc | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = TTIRKind(kind)
        self.message = message
        self.line_no = line_no
        self.loc = loc

    def __reduce__(self):
        return (
            type(self),
            (self.kind, self.message),
            {"line_no": self.line_no, "loc": self.loc},
        )


# ─────────────────────────── address-expression terms ───────────────────────────
# A small lazily-evaluated tree. Leaves that are only known at launch time
# (scalar kernel args) are Param nodes; pid / arange / loop variables become
# free variables with range constraints in a client's query.
#
# Bin, Cmp and IntCast also record the op they came from (``line_no``,
# ``loc``) for width obligations; those two fields take no part in
# equality, hashing or repr, so equal expressions compare equal wherever
# they were computed.


@dataclass(frozen=True)
class Const:
    value: int


@dataclass(frozen=True)
class Pid:
    axis: int  # 0=x, 1=y, 2=z


@dataclass(frozen=True)
class NumPrograms:
    """``tt.get_num_programs axis`` — the launch grid size along ``axis``.
    Uniform across program instances, but it PARAMETERIZES the kernel's
    behavior by the grid."""

    axis: int


@dataclass(frozen=True)
class Arange:
    ssa: str  # unique per make_range site (the walk's result value index)
    start: int
    end: int
    # Which tensor dimension this lane index varies along. -1 = 1D / not yet
    # placed; set and kept current by expand_dims. Consumers key a lane
    # variable by (dim, end - start), not by make_range site: every tensor
    # one access combines has the access's shape, so all aranges along one
    # dim with one extent index the SAME position there (tl.arange(0, 16) +
    # tl.arange(16, 32) is 2i + 16, not i + j + 16), while a single
    # make_range reused for several dimensions of a tile (triton does this)
    # is one independent variable per dimension, or the modeled footprint
    # would collapse to the diagonal.
    dim: int = -1


@dataclass(frozen=True)
class Param:
    name: str  # scalar kernel argument, substituted per launch


@dataclass(frozen=True)
class IterArgOffset:
    """The element-offset contribution of a loop-carried pointer at the
    current iteration: ``offset0 + k * delta`` (resolved from
    ``graph.iter_args[arg_id]`` at eval time)."""

    arg_id: int


@dataclass(frozen=True)
class LoopVar:
    """The scf.for induction variable; a free variable over the iterations
    that run (e.g. it appears in masks like ``K - k*BLOCK_K``)."""

    loop_ssa: str


# Integer ops (Bin.op): the arith op each spelling reads, signed first.
# "//" and "%" truncate toward zero (divsi / remsi); the "u"-prefixed ops
# read their operands unsigned (divui / remui / minui / maxui).
_BIN_OPS = {
    "arith.addi": "+",
    "arith.subi": "-",
    "arith.muli": "*",
    "arith.divsi": "//",
    "arith.remsi": "%",
    "arith.minsi": "min",
    "arith.maxsi": "max",
    "arith.divui": "u//",
    "arith.remui": "u%",
    "arith.minui": "umin",
    "arith.maxui": "umax",
}
UNSIGNED_BIN_OPS = frozenset({"u//", "u%", "umin", "umax"})
UNSIGNED_PREDICATES = frozenset({"ult", "ule", "ugt", "uge"})
_SIGNED_PREDICATES = frozenset({"slt", "sle", "sgt", "sge"})


@dataclass(frozen=True)
class Bin:
    op: str  # + - * // % min max, or u// u% umin umax (see _BIN_OPS)
    a: "Term"
    b: "Term"
    # Width of the integer result (the IR type). None for the element-offset
    # sum a ``tt.addptr`` accumulates, which is address arithmetic, not an
    # IR integer.
    bits: int | None = None
    line_no: int | None = field(default=None, compare=False, repr=False)
    loc: SourceLoc | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class Cmp:
    pred: str  # eq/ne, slt/sle/sgt/sge, ult/ule/ugt/uge (unsigned reads)
    a: "Term"
    b: "Term"
    bits: int | None = None  # operand width
    line_no: int | None = field(default=None, compare=False, repr=False)
    loc: SourceLoc | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class BoolBin:
    op: str  # and / or
    a: "Term"
    b: "Term"


@dataclass(frozen=True)
class Select:
    cond: "Term"
    t: "Term"
    f: "Term"


@dataclass(frozen=True)
class Not:
    """Boolean negation — the path condition of an scf.if else-region."""

    a: "Term"


@dataclass(frozen=True)
class IntCast:
    """``arith.trunci`` / ``extsi`` / ``extui`` of ``x`` from ``src_bits`` to
    ``dst_bits`` (a cast is never a value passthrough). Its value is
    ``x`` exactly when the cast's width obligation holds (trunci: ``x``
    fits the destination; extui: ``x`` is non-negative); extsi always
    preserves the signed reading. ``extsi`` from i1 (true -> -1) is read as
    ``0 - extui(x)`` and never appears as an IntCast."""

    kind: str  # "trunci" | "extsi" | "extui"
    src_bits: int
    dst_bits: int
    x: "Term"
    line_no: int | None = field(default=None, compare=False, repr=False)
    loc: SourceLoc | None = field(default=None, compare=False, repr=False)


# Sentinel for a value loaded from memory (tt.load result) or computed from
# loaded data (arith.*f, tt.dot, ...). If one ever reaches an address it
# means data-dependent addressing -> unsupported; in a mask it is dropped
# (``mask_dropped``), under an scf.if it leaves the branch ``guarded``.
@dataclass(frozen=True)
class DataDep:
    why: str = "value derived from loaded data"


@dataclass(frozen=True)
class Observed:
    """The value the access at ``graph.accesses[access_index]`` observed (an
    atomic's OLD value, or a while condition's load):
    a fresh per-program-instance symbol, NOT a function of other leaves. The
    reader binds an INTEGER-typed ``tt.atomic_rmw`` / ``tt.atomic_cas``
    result to this instead of ``DataDep`` so downstream masks and branch
    conditions stay modelable, and an integer ``tt.load`` result of an
    ``scf.while`` condition region too (while_regions=True); float-typed
    results keep the DataDep fallback. What an observation means (a free
    variable, a modeled value, a refusal in an address) is each consumer's
    policy; find them with the graph-aware :func:`mentions_observed` and
    :func:`observed_indices`.

    A tensor atomic observes one old value per lane, and the symbol stands
    for the value at the lane the surrounding term is read at. It has no
    lane placement of its own, so the reader never lets two lanes of one
    tensor observation meet: an ``expand_dims`` of a term holding one
    degrades to DataDep."""

    access_index: int


Term = (
    Const
    | Pid
    | NumPrograms
    | Arange
    | Param
    | IterArgOffset
    | LoopVar
    | Bin
    | Cmp
    | BoolBin
    | Select
    | Not
    | IntCast
    | DataDep
    | Observed
)


def _children(t: object) -> tuple:
    if isinstance(t, (Bin, Cmp, BoolBin)):
        return (t.a, t.b)
    if isinstance(t, Select):
        return (t.cond, t.t, t.f)
    if isinstance(t, (Not,)):
        return (t.a,)
    if isinstance(t, IntCast):
        return (t.x,)
    return ()


def _nodes(
    roots: Iterable[object], iter_args: "Sequence[IterArgInfo] | None"
) -> Iterator[object]:
    """Every node reachable from ``roots`` in pre-order, each once (by
    identity); with ``iter_args``, an IterArgOffset also reaches its
    IterArgInfo's ``offset0`` and ``delta``. Iterative: terms can be deeper
    than Python's recursion limit."""
    seen: set[int] = set()
    stack = [r for r in reversed(list(roots)) if r is not None]
    while stack:
        t = stack.pop()
        if id(t) in seen:
            continue
        seen.add(id(t))
        yield t
        if isinstance(t, IterArgOffset):
            if iter_args is not None:
                info = iter_args[t.arg_id]
                stack += [info.delta, info.offset0]
            continue
        stack.extend(reversed(_children(t)))


def mentions_observed(term: object, graph: "AccessGraph") -> bool:
    """True when ``term`` reaches an :class:`Observed` leaf, through the
    graph's loop-carried pointers (an IterArgOffset's ``offset0`` and
    ``delta``) included."""
    return any(isinstance(n, Observed) for n in _nodes((term,), graph.iter_args))


def observed_indices(term: object, graph: "AccessGraph") -> frozenset[int]:
    """The access indices of every :class:`Observed` leaf ``term`` reaches
    (see :func:`mentions_observed`)."""
    return frozenset(
        n.access_index
        for n in _nodes((term,), graph.iter_args)
        if isinstance(n, Observed)
    )


# DataDep is also the generic unknown-value top (loop accumulators,
# unmodeled ops, ...). Only these ``why`` prefixes mean the value truly
# derives from MEMORY CONTENTS — refusals classify just those as
# indirection; the rest are modeling gaps and keep the default kind.
_MEMORY_WHYS = (
    "loaded value",
    "atomic result",
    "arith over loaded data",
    "cmpi over loaded data",
    "select over loaded data",
    "bool op over loaded data",
)


def _from_memory(v: object) -> bool:
    return isinstance(v, DataDep) and v.why.startswith(_MEMORY_WHYS)


@dataclass(frozen=True)
class PtrValue:
    """A pointer-typed value: base argument + accumulated element offset (a
    single lane's offset; arange/loop free vars cover all lanes and
    iterations in the query)."""

    base_param: str
    offset: Term


# ─────────────────────────── graph structures ───────────────────────────


@dataclass(frozen=True)
class FuncArg:
    name: str  # the Python parameter name (NameLoc), else "arg<index>"
    is_ptr: bool
    elem_bits: int  # for ptr args: pointee width; 0 for scalars
    # For integer scalars: the IR width (i32 -> 32, i1 -> 1); 0 otherwise.
    int_bits: int = 0


@dataclass(frozen=True)
class AtomicInfo:
    """An atomic's attributes, in the keywords the printer prints."""

    # tt.atomic_rmw's op: "and", "or", "xor", "add", "fadd", "max", "min",
    # "umax", "umin", "exch"; None for tt.atomic_cas
    rmw_op: str | None
    sem: str  # "relaxed", "acquire", "release", "acq_rel"
    scope: str  # "gpu", "cta", "sys"


@dataclass(frozen=True)
class AccessEvent:
    kind: str  # "load" | "store" | "atomic_rmw" | "atomic_cas"
    base_param: str
    offset: Term
    mask: Term | None  # None = unconditional access
    elem_bits: int
    loc: SourceLoc | None
    line_no: int
    # True when some enclosing scf.if condition could NOT be modeled (it
    # derives from loaded data). The access is then checked as if
    # unconditional: UNSAT stays a sound proof, but a SAT model may sit in a
    # branch the launch never takes. Modeled conditions ride in ``path``
    # instead and do not set this flag.
    guarded: bool = False
    # Conjunction of the MODELED enclosing branch conditions, with
    # else-regions negated (Not). The access executes iff path ∧ mask.
    path: Term | None = None
    # True when the access sits inside the scf.for body: it executes once
    # per iteration — and NOT AT ALL when the launch's trip count is zero,
    # which consumers must model (a zero-trip loop has no footprint).
    in_loop: bool = False
    # True when the printed mask operand derived from loaded data and was
    # over-approximated as FREE (mask=None): dropping a constraint only
    # widens the modeled footprint, so UNSAT stays a sound proof — but a SAT
    # model may pick a lane the real mask disables.
    mask_dropped: bool = False
    # Set iff the access is an atomic.
    atomic: AtomicInfo | None = None
    # An atomic's value operand (tt.atomic_rmw val, tt.atomic_cas val) and a
    # CAS's compare operand as Terms; None when the operand is not modelable
    # (loaded data, a float, a pointer) or absent.
    atomic_val: Term | None = None
    atomic_cmp: Term | None = None
    # The accessed pointer's pointee is neither an integer nor a pointer.
    elem_float: bool = False
    # The earlier load and atomic accesses (indices into graph.accesses)
    # whose result reaches this access's value operands lane for lane: a
    # store's value and mask, an atomic_rmw's val and mask, an atomic_cas's
    # cmp and val, united. Element-wise ops keep their operands' lanes,
    # tt.dot its accumulator's, arith.select its condition's and those both
    # arms share; any other op, a region result or a block argument keeps
    # none (see _Builder.positional). Exact or fewer, never more; () for a
    # load.
    deps: tuple[int, ...] = ()
    # Op.path of the access op: the block indices from the module body down
    # to the access's own block (one block per region the access sits in).
    # () only in a hand-built graph, where it reads as the function body.
    block_path: tuple[int, ...] = ()


@dataclass(frozen=True)
class Fence:
    """A ``ttg.barrier`` (what ``tl.debug_barrier()`` prints), at its place
    among the accesses: after ``position + 0.5`` of them. ``path``,
    ``guarded``, ``in_loop`` and ``block_path`` read as an access's do."""

    position: float
    op: str
    # the barrier's addrSpace bits: local 1, global_read 2, global_write 4,
    # tensor_read 8, tensor_write 16 (`all` is 31)
    addr_space: int
    path: Term | None
    guarded: bool
    in_loop: bool
    block_path: tuple[int, ...]
    line_no: int
    loc: SourceLoc | None


@dataclass(frozen=True)
class WhileRegion:
    """One ``scf.while``, read under ``parse_ttir(while_regions=True)``:
    its condition ("before") region's accesses and ``scf.condition``, its
    body ("after") region's ops and accesses. The accesses are entries of
    ``graph.accesses`` like any other (``in_loop`` stays the scf.for's);
    an integer ``tt.load`` result of the before region reads as Observed,
    like an atomic's. Nested regions' accesses count in their enclosing
    region's too."""

    line_no: int
    loc: SourceLoc | None
    before: tuple[int, ...]  # access indices
    cond: Term | None  # None when not modelable (loaded data, ...)
    after_ops: tuple[str, ...]  # op names of the body block, its scf.yield included
    after: tuple[int, ...]  # access indices
    carries: bool  # the loop has iter args or results
    parent: int | None  # index of the enclosing WhileRegion
    block_path: tuple[int, ...]  # Op.path of the scf.while
    # its before and after blocks: an op inside it has a path that extends
    # block_path with one of them
    blocks: tuple[int, int]


@dataclass(frozen=True)
class IterArgInfo:
    """A loop-carried pointer: ``offset0 + k * delta`` at iteration k. A
    tile of one expanded (``tt.expand_dims``) inside the loop is an entry of
    its own, the same pointer with ``offset0`` and ``delta`` expanded alike,
    so its lanes sit at their positions in the expanded shape."""

    arg_id: int
    base_param: str
    offset0: Term
    delta: Term  # per-iteration element advance (loop-invariant)
    loop_ssa: str = ""  # the scf.for this iter_arg belongs to (LoopInfo.loop_ssa)


@dataclass(frozen=True)
class LoopInfo:
    loop_ssa: str
    lower: Term
    upper: Term
    step: Term
    # Width of the induction variable, and whether the loop compares its
    # bounds unsigned (``scf.for unsigned``: the bounds' width obligations
    # then require them non-negative).
    bits: int | None = None
    unsigned: bool = False
    line_no: int | None = None
    loc: SourceLoc | None = None


@dataclass(frozen=True)
class AccessGraph:
    kernel_name: str
    func_args: tuple[FuncArg, ...]
    accesses: tuple[AccessEvent, ...]
    loop: LoopInfo | None
    # Loop-carried pointers, indexed by arg_id (``iter_args[k].arg_id == k``),
    # expanded tiles of them included (see IterArgInfo).
    iter_args: tuple[IterArgInfo, ...] = ()
    # Every program-id axis a tt.get_program_id / tt.get_num_programs reads,
    # whether or not its value reaches a modeled term (in a combine region
    # too).
    pid_axes: frozenset[int] = frozenset()
    fences: tuple[Fence, ...] = ()
    whiles: tuple[WhileRegion, ...] = ()  # only under while_regions=True

    def __post_init__(self) -> None:
        object.__setattr__(self, "pid_axes", frozenset(self.pid_axes))
        for name in ("func_args", "accesses", "iter_args", "fences", "whiles"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def arg(self, name: str) -> FuncArg | None:
        for a in self.func_args:
            if a.name == name:
                return a
        return None


# ─────────────────────────── width obligations ───────────────────────────


@dataclass(frozen=True)
class WidthObligation:
    """``term`` must fit the width it is read at: with ``signed``,
    ``-2**(bits-1) <= term < 2**(bits-1)``; otherwise ``0 <= term < 2**bits``.
    ``line_no`` / ``loc`` are the op that imposes it; ``role`` is the part
    of the access it comes from (see :func:`width_obligations`)."""

    term: Term
    bits: int
    signed: bool
    line_no: int | None
    loc: SourceLoc | None
    role: str  # "loop" | "path" | "mask" | "offset"


def width_obligations(
    graph: AccessGraph, access: AccessEvent
) -> tuple[WidthObligation, ...]:
    """The conditions under which the unbounded-integer reading of
    ``access`` (its offset, mask and path, loop-carried pointers resolved,
    and the loop of an access in the loop) equals the IR's fixed-width
    arithmetic:

    * every integer ``Bin`` result fits its width, signed;
    * the quotient of a signed remainder fits too (``INT_MIN % -1`` is
      undefined in the IR);
    * the operands of an unsigned op or predicate are non-negative;
    * a ``trunci`` operand fits the destination width (to i1: is 0 or 1);
    * an ``extui`` operand is non-negative;
    * the bounds of an ``unsigned`` loop are non-negative;
    * the loop's increment does not wrap: ``upper - 1 + step``, which
      bounds the last iterate plus ``step``, fits the induction variable
      (sufficient, not necessary).

    ``role`` says where an obligation comes from: ``"loop"`` (the loop's
    bounds and increment), ``"path"``, ``"mask"`` or ``"offset"``. A term
    node several parts share is listed once, under the first role in that
    order, and the order is also the discipline for discharging them: an
    ``offset`` obligation only matters where the access executes (path and
    mask hold), a ``mask`` one only where the path holds, and ``path`` and
    ``loop`` ones hold unconditionally (the increment's only when the loop
    runs at least once). Both arms of a ``Select`` are listed, without its
    condition, so checking an arm unconditionally may report an overflow in
    the arm the launch does not take.

    Mechanism only: listed role by role in walk order, each (term node,
    width, signedness) once; nothing is evaluated."""
    loop = graph.loop if access.in_loop else None
    out: list[WidthObligation] = []
    seen: set[tuple[int, int, bool]] = set()

    def need(term: object, bits: int, signed: bool, site: object, role: str) -> None:
        key = (id(term), bits, signed)
        if key not in seen:
            seen.add(key)
            out.append(
                WidthObligation(
                    term,  # type: ignore[arg-type]
                    bits,
                    signed,
                    getattr(site, "line_no", None),
                    getattr(site, "loc", None),
                    role,
                )
            )

    groups: list[tuple[str, tuple[object, ...]]] = []
    if loop is not None:
        if loop.bits is not None:
            if loop.unsigned:
                for bound in (loop.lower, loop.upper, loop.step):
                    need(bound, loop.bits, False, loop, "loop")
            latch = Bin(
                "+",
                Bin("-", loop.upper, Const(1), loop.bits, loop.line_no, loop.loc),
                loop.step,
                loop.bits,
                loop.line_no,
                loop.loc,
            )
            need(latch, loop.bits, not loop.unsigned, loop, "loop")
        groups.append(("loop", (loop.lower, loop.upper, loop.step)))
    groups += [
        ("path", (access.path,)),
        ("mask", (access.mask,)),
        ("offset", (access.offset,)),
    ]
    for role, roots in groups:
        for n in _nodes(roots, graph.iter_args):
            if isinstance(n, Bin) and n.bits is not None:
                need(n, n.bits, True, n, role)
                if n.op in UNSIGNED_BIN_OPS:
                    need(n.a, n.bits, False, n, role)
                    need(n.b, n.bits, False, n, role)
                elif n.op == "%":
                    quotient = Bin("//", n.a, n.b, n.bits, n.line_no, n.loc)
                    need(quotient, n.bits, True, n, role)
            elif isinstance(n, Cmp) and n.pred in UNSIGNED_PREDICATES and n.bits:
                need(n.a, n.bits, False, n, role)
                need(n.b, n.bits, False, n, role)
            elif isinstance(n, IntCast):
                if n.kind == "trunci":
                    need(n.x, n.dst_bits, n.dst_bits > 1, n, role)
                elif n.kind == "extui" and n.src_bits > 1:
                    need(n.x, n.src_bits, False, n, role)
    return tuple(out)


# ─────────────────────────── the reader ───────────────────────────


def parse_ttir(text: str, *, while_regions: bool = False) -> AccessGraph:
    """Read one TTIR module into an AccessGraph (single-path model: at most
    one scf.for, structured scf.if only).

    ``while_regions=True`` reads each ``scf.while`` too (a ``WhileRegion`` in
    ``graph.whiles``) instead of refusing it as control flow; a kernel
    without one reads the same either way.

    Raises :class:`UnsupportedTTIR` for anything the graph cannot represent:
    a text the walk layer cannot align (``READER_MISALIGNMENT``) or does not
    parse (``UNPARSABLE``: the MLIR parser rejects it, it holds a
    block-pointer type, or the installed Triton is not 3.8), indirect
    addressing,
    pointers outside global memory, nested/while loops, unstructured
    control flow, calls, loop-variant addresses, inline asm that is impure
    or handed an address, or any op outside the address vocabulary that
    could touch memory.
    """
    try:
        module = walk_module(text)
    except MisalignedModule as e:
        raise UnsupportedTTIR(
            TTIRKind.READER_MISALIGNMENT, "; ".join(e.problems), line_no=e.line_no
        ) from e
    except ModuleParseError as e:
        raise UnsupportedTTIR(
            TTIRKind.UNPARSABLE, e.diagnostic, line_no=e.line_no
        ) from e
    return _Builder(module, while_regions).build()


# Memory ops outside the modeled vocabulary (TMA descriptors, ...: targets
# without native TMA lower descriptors to pointer math before TTIR is
# printed; sm90+ and hip TDM targets such as gfx1250 keep them).
_MEMORY_PREFIXES = ("tt.descriptor_", "tt.experimental_")
_ACCESS_OPS = frozenset({"tt.load", "tt.store", "tt.atomic_rmw", "tt.atomic_cas"})
# Dialects whose ops the reader reads as data or control; from any other
# dialect only the inert ops (below) are accepted.
_DIALECTS = frozenset({"tt", "arith", "math", "scf", "cf", "ub"})


# Result-free ops without memory effects the reader skips (a barrier orders
# memory, it addresses none: it becomes a Fence, except inside a combine
# region). tl.debug_barrier() is `ttg.barrier all`: the one ttg op accepted,
# not the dialect. gpu.barrier, which the frontend never emits, is not (the
# bindings still parse it).
_INERT = frozenset(
    {
        "tt.return",
        "scf.yield",
        "tt.reduce.return",
        "tt.scan.return",
        "tt.print",
        "tt.assert",
        "llvm.intr.assume",
        "ttg.barrier",
    }
)
# Ops that hand the reader values it cannot see into (their operands must
# carry no address, not even as an integer).
_OPAQUE = frozenset({"tt.elementwise_inline_asm", "tt.extern_elementwise"})
# Element-wise ops: lane i of the result reads lane i of each operand
# (AccessEvent.deps). A packed inline asm reads its whole pack, which one
# thread holds, so a lane stays in the thread that holds it.
_ELEMENTWISE_PREFIXES = ("arith.", "math.", "tt.precise_")
_ELEMENTWISE = _OPAQUE | {
    "tt.addptr",
    "tt.bitcast",
    "tt.fp_to_fp",
    "tt.int_to_ptr",
    "tt.ptr_to_int",
    "tt.clampf",
    "tt.mulhiui",
}
# Structured-region nesting the reader follows (deeper refuses, instead of
# exhausting Python's recursion limit).
_MAX_DEPTH = 200
_LOOP_SSA = "%loop"  # the single-path model has at most one loop
# The DataDep reason of a value carried by the loop that is not a pointer:
# in an address it makes the address loop-variant.
_LOOP_CARRIED = "loop accumulator"
_RE_ADDR_SPACE = re.compile(r", (\d+)>$")


def _addr_space(type_text: str) -> int:
    """Address space of a (tensor of) ``!tt.ptr`` type; 1 (global memory)
    when unprinted, as the printer elides it."""
    m = _RE_ADDR_SPACE.search(parse_type(type_text).elem)
    return int(m.group(1)) if m else 1


@dataclass
class _IfFrame:
    """Walker state for one scf.if region being read."""

    cond: "Term | None"  # modeled condition; None -> accesses stay `guarded`
    branch: str = "then"


class _ForFrame:
    """Marks the scf.for body being read."""


@dataclass
class _WhileFrame:
    """Walker state for one scf.while being read (while_regions=True)."""

    index: int  # into _Builder.whiles
    stage: str = "before"  # "after" once its scf.condition is read
    cond: object = None  # the scf.condition's value
    split: int = 0  # len(accesses) at the scf.condition


def _positional_operands(op: Op) -> tuple[int, ...]:
    """The operands whose lane i reaches lane i of ``op``'s results."""
    if op.name == "tt.dot":  # (a, b, c): only the accumulator keeps its lanes
        return op.operands[2:3]
    if op.name.startswith(_ELEMENTWISE_PREFIXES) or op.name in _ELEMENTWISE:
        return op.operands
    return ()


def _pointee_bits(pointee: str | None) -> int:
    if pointee is None:
        return 0
    if pointee.startswith("!tt.ptr<"):
        return 64  # a pointer to pointers
    return parse_type(pointee).int_bits or parse_type(pointee).float_bits or 0


class _Builder:
    def __init__(self, module: Module, while_regions: bool = False) -> None:
        self.m = module
        self.while_regions = while_regions
        # value index -> Term (int/bool), PtrValue, or DataDep
        self.env: dict[int, object] = {}
        self.func_args: list[FuncArg] = []
        self.accesses: list[AccessEvent] = []
        self.iter_args: list[IterArgInfo] = []
        self.loop: LoopInfo | None = None
        self.loops_opened = 0
        self.fences: list[Fence] = []
        self.whiles: list[WhileRegion | None] = []
        self.frames: list[_IfFrame | _ForFrame | _WhileFrame] = []
        self.depth = 0
        # (arg_id, axis) -> the arg_id of that iter_arg's tile expanded at
        # axis inside the loop (its delta is filled in when the loop closes)
        self.expanded: dict[tuple[int, int], int] = {}
        # access indices of the tensor observations (one per lane)
        self.lane_observed: set[int] = set()
        # access op index -> access index, and value index -> the accesses
        # it reads lane for lane (see positional)
        self.access_op: dict[int, int] = {}
        self.prov: dict[int, frozenset[int]] = {}

    # ── helpers ──
    def refuse(self, kind: TTIRKind, message: str, op: Op | None) -> NoReturn:
        raise UnsupportedTTIR(
            kind,
            message,
            line_no=op.line_no if op is not None else None,
            loc=op.loc if op is not None else None,
        )

    def val(self, v: int) -> object:
        got = self.env.get(v)
        if got is None:
            # A value the reader never bound (defined in a region it does not
            # read): be conservative.
            return DataDep(f"unresolved value {v}")
        return got

    def bind(self, op: Op, value: object) -> None:
        for r in op.results:
            self.env[r] = value

    def as_term(self, v: object, ctx: str, op: Op) -> Term:
        if isinstance(v, DataDep):
            self.refuse(TTIRKind.OTHER, f"{ctx}: data-dependent ({v.why})", op)
        if isinstance(v, PtrValue):
            self.refuse(TTIRKind.OTHER, f"{ctx}: pointer used as integer", op)
        return v  # type: ignore[return-value]

    def int_bits(self, v: int) -> int | None:
        return parse_type(parse_type(self.m.values[v].type).elem).int_bits

    def branch_state(self) -> tuple[bool, Term | None, bool]:
        """(guarded, path, in_loop) for an access under the open frames:
        ``guarded`` if any enclosing condition is unmodeled; ``path`` is the
        conjunction of the modeled ones (else-regions negated), outermost
        first; ``in_loop`` when the scf.for body encloses the access."""
        guarded = False
        path: Term | None = None
        in_loop = False
        for f in self.frames:
            if isinstance(f, _ForFrame):
                in_loop = True
                continue
            if isinstance(f, _WhileFrame):
                continue
            if f.cond is None:
                guarded = True
                continue
            c: Term = f.cond if f.branch == "then" else Not(f.cond)
            path = c if path is None else BoolBin("and", path, c)
        return guarded, path, in_loop

    def arg(self, name: str) -> FuncArg | None:
        return next((a for a in self.func_args if a.name == name), None)

    def global_pointers(self, types: Iterable[str], op: Op) -> None:
        """Element offsets are into global memory: refuse a pointer into any
        other address space (shared memory, ...)."""
        for t in types:
            if "!tt.ptr<" in t and _addr_space(t) != 1:
                self.refuse(
                    TTIRKind.OUT_OF_VOCABULARY,
                    f"pointer type {t} is not in global memory (address space 1)",
                    op,
                )

    # ── the module ──
    def build(self) -> AccessGraph:
        m = self.m
        if not m.funcs:
            self.refuse(TTIRKind.OTHER, "no tt.func found (not TTIR?)", None)
        for op in m.ops:
            if len(op.path) == 1 and op.name != "tt.func":
                self.refuse(
                    TTIRKind.OUT_OF_VOCABULARY,
                    f"{op.name} at module level is not TTIR",
                    op,
                )
        # A call's callee runs in its own frame with its own arguments; the
        # graph has no call model, so any call, and any second function,
        # refuses (the callee name comes from the symbol, quoted or not).
        calls = [op for op in m.ops if op.name == "tt.call"]
        if calls:
            first = calls[0]
            self.refuse(
                TTIRKind.CALL,
                f"tt.call to {first.attrs.get('callee')!r}: calls are not modeled",
                first,
            )
        if len(m.funcs) > 1:
            extra = m.funcs[1]
            self.refuse(
                TTIRKind.CALL,
                f"{len(m.funcs)} functions in the module (tt.func "
                f"{extra.sym_name!r}): calls are not modeled",
                m.ops[extra.op],
            )
        func = m.funcs[0]
        fop = m.ops[func.op]
        if not fop.regions or not fop.regions[0]:
            self.refuse(TTIRKind.OTHER, f"tt.func {func.sym_name!r} has no body", fop)
        for fa in func.args:
            self.bind_arg(fa, fop)
        body = fop.regions[0]
        self.block(body[0])
        if len(body) > 1:
            self.refuse(
                TTIRKind.CONTROL_FLOW,
                f"tt.func {func.sym_name!r} has {len(body)} blocks",
                fop,
            )
        return AccessGraph(
            kernel_name=func.sym_name,
            func_args=tuple(self.func_args),
            accesses=tuple(self.accesses),
            loop=self.loop,
            iter_args=tuple(self.iter_args),
            pid_axes=frozenset(
                op.attrs["axis"]
                for op in self.m.ops
                if op.name in ("tt.get_program_id", "tt.get_num_programs")
            ),
            fences=tuple(self.fences),
            whiles=tuple(self.whiles),  # type: ignore[arg-type]
        )

    def bind_arg(self, fa, fop: Op) -> None:
        ti = parse_type(fa.type)
        name = fa.name if fa.name is not None else f"arg{fa.index}"
        if self.arg(name) is not None:
            self.refuse(TTIRKind.OTHER, f"two parameters named {name!r}", fop)
        self.global_pointers((fa.type,), fop)
        is_ptr = ti.pointee is not None and not ti.shape
        int_bits = ti.int_bits if not is_ptr and not ti.shape else None
        self.func_args.append(
            FuncArg(
                name=name,
                is_ptr=is_ptr,
                elem_bits=_pointee_bits(ti.pointee) if is_ptr else 0,
                int_bits=int_bits or 0,
            )
        )
        if is_ptr:
            self.env[fa.value] = PtrValue(name, Const(0))
        elif int_bits is not None:
            self.env[fa.value] = Param(name)
        else:
            self.env[fa.value] = DataDep(f"{fa.type} argument")

    def block(self, bidx: int) -> None:
        self.depth += 1
        try:
            for oi in self.m.blocks[bidx].ops:
                self.visit(self.m.ops[oi])
        finally:
            self.depth -= 1

    def visit(self, op: Op) -> None:
        if self.depth > _MAX_DEPTH:
            self.refuse(
                TTIRKind.OTHER,
                f"structured regions nested deeper than {_MAX_DEPTH}",
                op,
            )
        name = op.name
        self.global_pointers(op.operand_types + op.result_types, op)
        handler = _HANDLERS.get(name)
        if handler is not None:
            handler(self, op)
        elif name not in _INERT:
            self.op_other(op)

    # ── value ops ──
    def op_program_id(self, op: Op) -> None:
        self.bind(op, Pid(op.attrs["axis"]))

    def op_num_programs(self, op: Op) -> None:
        self.bind(op, NumPrograms(op.attrs["axis"]))

    def op_make_range(self, op: Op) -> None:
        self.bind(op, Arange(f"%v{op.results[0]}", op.attrs["start"], op.attrs["end"]))

    def op_constant(self, op: Op) -> None:
        v = op.attrs["value"]
        if isinstance(v, bool):
            # i1 constants (e.g. the dense<true> mask of an unmasked atomic).
            # Const(0/1) in a boolean position is coerced by the evaluator.
            self.bind(op, Const(1 if v else 0))
        elif isinstance(v, int):
            self.bind(op, Const(v))
        else:
            self.bind(op, DataDep("float/array constant"))

    def op_passthrough(self, op: Op) -> None:
        # tt.splat: replicate a scalar / seed a pointer tile;
        # tt.broadcast: a shape change, value passthrough
        self.bind(op, self.val(op.operands[0]))

    def op_expand_dims(self, op: Op) -> None:
        v = self.val(op.operands[0])
        axis = op.attrs["axis"]
        root = v.offset if isinstance(v, PtrValue) else v
        if any(
            isinstance(n, Observed) and n.access_index in self.lane_observed
            for n in _nodes((root,), self.iter_args)
        ):
            # Observed has no lane placement: two lanes of one tensor
            # observation must not meet in a term (see Observed).
            self.bind(op, DataDep("atomic result expanded across lanes"))
            return
        self.bind(op, _expand_dims(v, axis, lambda t: self.expanded_iter_arg(t, axis)))

    def expanded_iter_arg(self, t: IterArgOffset, axis: int) -> IterArgOffset:
        """The loop-carried pointer ``t`` with its lanes re-placed by an
        expand_dims at ``axis``: an iter_args entry of its own (IterArgInfo),
        one per (pointer, axis), whose delta the loop fills in on closing."""
        key = (t.arg_id, axis)
        aid = self.expanded.get(key)
        if aid is None:
            src = self.iter_args[t.arg_id]
            aid = len(self.iter_args)
            offset0 = _expand_dims(src.offset0, axis)
            self.iter_args.append(
                IterArgInfo(aid, src.base_param, offset0, Const(0), _LOOP_SSA)  # type: ignore[arg-type]
            )
            self.expanded[key] = aid
        return IterArgOffset(aid)

    def op_cast(self, op: Op) -> None:
        x = self.val(op.operands[0])
        if isinstance(x, (DataDep, PtrValue)):
            self.bind(op, x)
            return
        kind = op.name.split(".", 1)[1]
        src = self.int_bits(op.operands[0])
        dst = self.int_bits(op.results[0])
        assert src is not None and dst is not None
        term: Term = x  # type: ignore[assignment]
        if kind == "extsi" and src == 1:
            # sign-extending an i1 maps true to -1
            ext = IntCast("extui", 1, dst, term, op.line_no, op.loc)
            self.bind(op, Bin("-", Const(0), ext, dst, op.line_no, op.loc))
            return
        self.bind(op, IntCast(kind, src, dst, term, op.line_no, op.loc))

    def op_addptr(self, op: Op) -> None:
        base, off = self.val(op.operands[0]), self.val(op.operands[1])
        if not isinstance(base, PtrValue):
            self.refuse(
                _address_kind(base),
                f"addptr base is not a pointer{_why(base)}",
                op,
            )
        if isinstance(off, DataDep):
            # A value in an address chain that cannot be modeled: a free
            # address makes the query meaningless. Only offsets truly
            # derived from MEMORY CONTENTS classify as indirection.
            kind = _address_kind(off)
            what = (
                "carried by the loop"
                if kind is TTIRKind.LOOP_VARIANT_ADVANCE
                else "data-dependent"
            )
            self.refuse(kind, f"addptr offset: {what} ({off.why})", op)
        off_t = self.as_term(off, "addptr offset", op)
        self.bind(
            op,
            PtrValue(
                base.base_param,  # type: ignore[union-attr]
                Bin("+", base.offset, off_t, None, op.line_no, op.loc),  # type: ignore[union-attr]
            ),
        )

    def op_bin(self, op: Op) -> None:
        a, b = self.val(op.operands[0]), self.val(op.operands[1])
        bits = self.int_bits(op.results[0])
        if isinstance(a, DataDep) or isinstance(b, DataDep):
            self.bind(op, _data_dep((a, b), "arith over loaded data"))
        elif bits == 1:
            self.bind(op, DataDep(f"{op.name} on i1"))
        else:
            self.bind(
                op,
                Bin(
                    _BIN_OPS[op.name],
                    self.as_term(a, "arith", op),
                    self.as_term(b, "arith", op),
                    bits,
                    op.line_no,
                    op.loc,
                ),
            )

    def op_cmpi(self, op: Op) -> None:
        a, b = self.val(op.operands[0]), self.val(op.operands[1])
        pred = op.attrs["predicate"]
        bits = self.int_bits(op.operands[0])
        if isinstance(a, DataDep) or isinstance(b, DataDep):
            self.bind(op, _data_dep((a, b), "cmpi over loaded data"))
        elif bits == 1 and pred in _SIGNED_PREDICATES:
            # a signed i1 reads true as -1; the boolean model reads it as 1
            self.bind(op, DataDep("signed comparison of i1 values"))
        else:
            self.bind(
                op,
                Cmp(
                    pred,
                    self.as_term(a, "cmpi", op),
                    self.as_term(b, "cmpi", op),
                    bits,
                    op.line_no,
                    op.loc,
                ),
            )

    def op_boolbin(self, op: Op) -> None:
        if self.int_bits(op.results[0]) != 1:
            # Wide-int andi/ori is BITWISE arithmetic, not boolean logic;
            # modeling it as And/Or would silently corrupt address math.
            # Degrade to DataDep so an address use fails closed.
            self.bind(
                op,
                DataDep(f"bitwise {op.name} on non-i1 type {op.result_types[0]}"),
            )
            return
        a, b = self.val(op.operands[0]), self.val(op.operands[1])
        is_and = op.name == "arith.andi"
        if isinstance(a, DataDep) or isinstance(b, DataDep):
            why = _data_dep((a, b), "bool op over loaded data").why
            self.bind(op, DataDep(why))
            return
        self.bind(
            op,
            BoolBin(
                "and" if is_and else "or",
                self.as_term(a, "bool", op),
                self.as_term(b, "bool", op),
            ),
        )

    def op_select(self, op: Op) -> None:
        c, t, f = (self.val(v) for v in op.operands)
        self.bind(op, self.merge(c, t, f, "select"))

    def merge(self, c: object, t: object, f: object, what: str) -> object:
        """``c ? t : f`` for an arith.select or an scf.if result: pointers
        of one base select their offsets; anything unmodelable is DataDep."""
        if (
            isinstance(c, (DataDep, PtrValue))
            or isinstance(t, DataDep)
            or (isinstance(f, DataDep))
        ):
            return _data_dep((c, t, f), "select over loaded data")
        if isinstance(t, PtrValue) and isinstance(f, PtrValue):
            if t.base_param != f.base_param:
                return DataDep(f"{what} of pointers with different bases")
            return PtrValue(t.base_param, Select(c, t.offset, f.offset))  # type: ignore[arg-type]
        if isinstance(t, PtrValue) or isinstance(f, PtrValue):
            return DataDep(f"{what} of a pointer and an integer")
        return Select(c, t, f)  # type: ignore[arg-type]

    def op_bitcast(self, op: Op) -> None:
        # A pointer cast that keeps the element width keeps element offsets
        # (atomic_max on floats casts f32 -> i32 pointers); any other
        # bitcast reinterprets data.
        src = parse_type(op.operand_types[0])
        dst = parse_type(op.result_types[0])
        x = self.val(op.operands[0])
        if src.pointee is not None and dst.pointee is not None:
            if _pointee_bits(src.pointee) == _pointee_bits(dst.pointee):
                self.bind(op, x)
            else:
                self.bind(op, DataDep("pointer bitcast changes the element width"))
            return
        self.bind(op, _data_dep((x,), f"unmodeled op {op.name} at line {op.line_no}"))

    # ── accesses ──
    def access(
        self,
        op: Op,
        kind: str,
        mask_v: int | None,
        atomic: AtomicInfo | None = None,
        atomic_val: Term | None = None,
        atomic_cmp: Term | None = None,
    ) -> None:
        ptr_v = op.operands[0]
        ptr = self.val(ptr_v)
        if not isinstance(ptr, PtrValue):
            self.refuse(
                _address_kind(ptr),
                f"{kind} of a non-pointer value{_why(ptr)}",
                op,
            )
        pointee = parse_type(parse_type(self.m.values[ptr_v].type).elem).pointee
        elem_bits = _pointee_bits(pointee)
        base = self.arg(ptr.base_param)  # type: ignore[union-attr]
        if base is None or base.elem_bits != elem_bits:
            self.refuse(
                TTIRKind.OTHER,
                f"{kind} element width {elem_bits} differs from its base "
                f"{ptr.base_param!r}",  # type: ignore[union-attr]
                op,
            )
        mask: Term | None = None
        mask_dropped = False
        if mask_v is not None:
            mv = self.val(mask_v)
            if isinstance(mv, DataDep):
                # Mask derived from loaded data: over-approximate it as free
                # (any lane may be active) instead of failing the kernel.
                # See AccessEvent.mask_dropped for the soundness discipline.
                mask_dropped = True
            elif isinstance(mv, PtrValue):
                self.refuse(TTIRKind.OTHER, "pointer as mask", op)
            else:
                mask = mv  # type: ignore[assignment]
        guarded, path, in_loop = self.branch_state()
        assert op.line_no is not None
        # the value operands: a store's value and mask, an atomic_rmw's val
        # and mask, an atomic_cas's cmp and val (ODS order, after the pointer)
        deps = self.positional(op.operands[1:]) if kind != "load" else ()
        self.access_op[op.index] = len(self.accesses)
        self.accesses.append(
            AccessEvent(
                kind=kind,
                base_param=ptr.base_param,  # type: ignore[union-attr]
                offset=ptr.offset,  # type: ignore[union-attr]
                mask=mask,
                elem_bits=elem_bits,
                loc=op.loc,
                line_no=op.line_no,
                guarded=guarded,
                path=path,
                in_loop=in_loop,
                mask_dropped=mask_dropped,
                atomic=atomic,
                atomic_val=atomic_val,
                atomic_cmp=atomic_cmp,
                elem_float=pointee is not None
                and parse_type(pointee).int_bits is None
                and parse_type(pointee).pointee is None,
                deps=deps,
                block_path=op.path,
            )
        )

    def positional(self, values: Iterable[int]) -> tuple[int, ...]:
        """The accesses whose result reaches ``values`` lane for lane: a
        load's or atomic's result is its own access's; an element-wise op
        (_ELEMENTWISE) keeps its operands' and tt.dot its accumulator's; an
        arith.select keeps its condition's and those both arms share; any
        other op (a shape change, a reduction, a region's result) and a
        block argument keep none. ``values`` unite: the access reads each of
        them at the lane. Memoized per value, iterative."""
        memo = self.prov
        roots = list(values)
        stack = [(v, False) for v in roots]
        while stack:
            v, ready = stack.pop()
            if v in memo:
                continue
            src = self.m.values[v].op
            if src is None:
                memo[v] = frozenset()
                continue
            if src in self.access_op:
                memo[v] = frozenset((self.access_op[src],))
                continue
            op = self.m.ops[src]
            ins = _positional_operands(op)
            if not ready and any(x not in memo for x in ins):
                stack.append((v, True))
                stack.extend((x, False) for x in ins if x not in memo)
            elif op.name == "arith.select":
                c, t, f = (memo[x] for x in ins)
                memo[v] = c | (t & f)
            else:
                memo[v] = frozenset().union(*(memo[x] for x in ins))
        return tuple(sorted(frozenset().union(*(memo[v] for v in roots))))

    def mask_operand(self, op: Op, index: int) -> int | None:
        if len(op.operands) <= index:
            return None
        v = op.operands[index]
        if self.int_bits(v) != 1:
            self.refuse(TTIRKind.OTHER, f"{op.name} operand {index} is not a mask", op)
        return v

    def observed_binding(self, op: Op, why: str = "atomic result") -> object:
        """The value of the just-recorded access's result: Observed for an
        integer-typed result, DataDep(``why``) otherwise (floats stay
        outside the Int model)."""
        if op.results and self.int_bits(op.results[0]) is not None:
            index = len(self.accesses) - 1
            if parse_type(self.m.values[op.results[0]].type).shape:
                self.lane_observed.add(index)
            return Observed(index)
        return DataDep(why)

    def op_load(self, op: Op) -> None:
        # ODS operands (ptr, mask?, other?). Both are optional, so the
        # generic form spells which are present in operandSegmentSizes (the
        # custom form prints a lone second operand as the mask).
        segments = op.attrs.get("operandSegmentSizes")
        if segments is None:
            mask_v = self.mask_operand(op, 1)
        elif (
            len(segments) != 3 or segments[0] != 1 or sum(segments) != len(op.operands)
        ):
            self.refuse(TTIRKind.OTHER, f"tt.load operand segments {segments}", op)
        else:
            mask_v = self.mask_operand(op, 1) if segments[1] else None
        self.access(op, "load", mask_v)
        if any(isinstance(f, _WhileFrame) and f.stage == "before" for f in self.frames):
            # a poll: what the loop's condition reads
            self.bind(op, self.observed_binding(op, "loaded value"))
        else:
            self.bind(op, DataDep("loaded value"))

    def op_store(self, op: Op) -> None:
        # ODS operands (ptr, value, mask?)
        self.access(op, "store", self.mask_operand(op, 2))

    def op_atomic_rmw(self, op: Op) -> None:
        # ODS operands (ptr, val, mask?)
        self.access(
            op,
            "atomic_rmw",
            self.mask_operand(op, 2),
            atomic=AtomicInfo(op.attrs["rmw_op"], op.attrs["sem"], op.attrs["scope"]),
            atomic_val=_operand_term(self.val(op.operands[1])),
        )
        self.bind(op, self.observed_binding(op))

    def op_atomic_cas(self, op: Op) -> None:
        # ODS operands (ptr, cmp, val); CAS has no mask: unconditional footprint
        self.access(
            op,
            "atomic_cas",
            None,
            atomic=AtomicInfo(None, op.attrs["sem"], op.attrs["scope"]),
            atomic_val=_operand_term(self.val(op.operands[2])),
            atomic_cmp=_operand_term(self.val(op.operands[1])),
        )
        self.bind(op, self.observed_binding(op))

    # ── structured control flow ──
    def op_for(self, op: Op) -> None:
        # The single-path model has one induction variable: a second loop
        # (sequential or nested) cannot be represented, and a loop under an
        # scf.if runs a branch-dependent iteration count — a control-flow
        # limitation, not one more induction variable.
        if self.loops_opened or self.frames:
            self.refuse(
                TTIRKind.CONTROL_FLOW
                if any(isinstance(f, _IfFrame) for f in self.frames)
                else TTIRKind.NESTED_LOOP,
                "multiple/nested loops",
                op,
            )
        self.loops_opened += 1
        bounds: dict[str, Term] = {}
        for label, v in zip(("lower", "upper", "step"), op.operands[:3]):
            bv = self.val(v)
            if isinstance(bv, DataDep):
                # The CSR shape: for k in range(loaded_start, loaded_end).
                self.refuse(
                    TTIRKind.DATA_DEPENDENT_BOUND
                    if _from_memory(bv)
                    else TTIRKind.OTHER,
                    f"loop {label} bound: data-dependent ({bv.why})",
                    op,
                )
            if any(isinstance(n, Observed) for n in _nodes((bv,), None)):
                # A trip count driven by an atomic observation is a dynamic
                # work-fetch loop.
                self.refuse(
                    TTIRKind.DATA_DEPENDENT_BOUND,
                    f"loop {label} bound depends on an atomic observation",
                    op,
                )
            bounds[label] = self.as_term(bv, f"loop {label}", op)
        body = self.m.blocks[op.regions[0][0]]
        iv, carried = body.args[0], body.args[1:]
        self.env[iv] = LoopVar(_LOOP_SSA)
        ptr_args: list[tuple[int, int]] = []  # (iter_arg position, arg_id)
        for k, (arg_v, init_v) in enumerate(zip(carried, op.operands[3:])):
            init = self.val(init_v)
            if isinstance(init, PtrValue):
                aid = len(self.iter_args)
                self.iter_args.append(
                    IterArgInfo(aid, init.base_param, init.offset, Const(0), _LOOP_SSA)
                )
                self.env[arg_v] = PtrValue(init.base_param, IterArgOffset(aid))
                ptr_args.append((k, aid))
            elif isinstance(init, DataDep):
                self.env[arg_v] = DataDep(init.why)
            else:
                self.env[arg_v] = DataDep(_LOOP_CARRIED)
        self.frames.append(_ForFrame())
        self.block(op.regions[0][0])
        self.frames.pop()
        yield_op = self.m.ops[body.ops[-1]]
        for k, aid in ptr_args:
            delta = self.loop_delta(self.val(yield_op.operands[k]), aid, yield_op)
            self.iter_args[aid] = replace(self.iter_args[aid], delta=delta)
        # Expanded tiles advance by their source's delta, expanded alike (a
        # source precedes the tiles expanded from it).
        for (src, axis), aid in self.expanded.items():
            source = self.iter_args[src].delta
            if any(
                isinstance(n, Observed) and n.access_index in self.lane_observed
                for n in _nodes((source,), None)
            ):
                self.refuse(
                    TTIRKind.OTHER,
                    f"loop-carried pointer {src} is expanded but advances by a "
                    "per-lane atomic result",
                    yield_op,
                )
            expanded = _expand_dims(source, axis)
            self.iter_args[aid] = replace(self.iter_args[aid], delta=expanded)  # type: ignore[arg-type]
        self.loop = LoopInfo(
            loop_ssa=_LOOP_SSA,
            lower=bounds["lower"],
            upper=bounds["upper"],
            step=bounds["step"],
            bits=self.int_bits(iv),
            unsigned=bool(op.attrs["unsignedCmp"]),
            line_no=op.line_no,
            loc=op.loc,
        )
        self.bind(op, DataDep("loop result"))

    def loop_delta(self, y: object, aid: int, op: Op) -> Term:
        """The loop-invariant advance of loop-carried pointer ``aid`` from
        its yielded value, which must be ``IterArgOffset(aid) + delta`` on the
        same base: the model reads iteration k's offset as
        ``offset0 + k * delta``."""
        info = self.iter_args[aid]
        if not isinstance(y, PtrValue) or y.base_param != info.base_param:
            self.refuse(
                TTIRKind.LOOP_VARIANT_ADVANCE,
                f"loop-carried pointer {aid} (base {info.base_param!r}) is not "
                "advanced from itself",
                op,
            )
        delta = _loop_delta(y.offset, aid)  # type: ignore[union-attr]
        if delta is None:
            self.refuse(
                TTIRKind.LOOP_VARIANT_ADVANCE,
                f"loop-carried pointer {aid} is not advanced by addptr from "
                "its own previous value",
                op,
            )
        for n in _nodes((delta,), None):
            if (
                isinstance(n, (LoopVar, IterArgOffset))
                or isinstance(n, Observed)
                and self.accesses[n.access_index].in_loop
            ):
                self.refuse(
                    TTIRKind.LOOP_VARIANT_ADVANCE,
                    f"loop-carried pointer {aid} advances by a loop-variant amount",
                    op,
                )
        return delta  # type: ignore[return-value]

    def op_if(self, op: Op) -> None:
        cv = self.val(op.operands[0])
        # A pointer can't be a condition; loaded data (DataDep) can't be
        # modeled -> the region stays pessimistically ``guarded``.
        cond = None if isinstance(cv, (DataDep, PtrValue)) else cv
        frame = _IfFrame(cond)  # type: ignore[arg-type]
        yields: list[list[object]] = []
        for ri, region in enumerate(op.regions):
            if not region:
                continue
            if len(region) != 1:
                self.op_control_flow(op)
            frame.branch = "then" if ri == 0 else "else"
            self.frames.append(frame)
            self.block(region[0])
            self.frames.pop()
            y = self.m.ops[self.m.blocks[region[0]].ops[-1]]
            yields.append([self.val(v) for v in y.operands])
        if op.results and len(yields) != 2:
            self.refuse(TTIRKind.OTHER, "scf.if with results but no else region", op)
        for k, r in enumerate(op.results):
            if cond is None:
                self.env[r] = _data_dep((cv,), "select over loaded data")
            else:
                self.env[r] = self.merge(cond, yields[0][k], yields[1][k], "scf.if")

    def op_while(self, op: Op) -> None:
        """An scf.while: a WhileRegion under while_regions=True, else
        refused. Its region arguments and results are loop-carried values
        (DataDep)."""
        if not self.while_regions or [len(r) for r in op.regions] != [1, 1]:
            self.op_control_flow(op)
        before, after = (self.m.blocks[r[0]] for r in op.regions)
        for v in before.args + after.args:
            self.env[v] = DataDep(_LOOP_CARRIED)
        parent = next(
            (f.index for f in reversed(self.frames) if isinstance(f, _WhileFrame)),
            None,
        )
        frame = _WhileFrame(len(self.whiles))
        self.whiles.append(None)  # filled in below, after the nested ones
        first = len(self.accesses)
        self.frames.append(frame)
        self.block(before.index)  # ends with the scf.condition (op_condition)
        self.block(after.index)
        self.frames.pop()
        cond = frame.cond
        assert op.line_no is not None
        self.whiles[frame.index] = WhileRegion(
            line_no=op.line_no,
            loc=op.loc,
            before=tuple(range(first, frame.split)),
            cond=None if isinstance(cond, (DataDep, PtrValue)) else cond,  # type: ignore[arg-type]
            after_ops=tuple(self.m.ops[oi].name for oi in after.ops),
            after=tuple(range(frame.split, len(self.accesses))),
            carries=bool(op.operands or op.results),
            parent=parent,
            block_path=op.path,
            blocks=(before.index, after.index),
        )
        self.bind(op, DataDep("loop result"))

    def op_condition(self, op: Op) -> None:
        # the terminator of the before region of the scf.while being read
        frame = self.frames[-1] if self.frames else None
        if not (isinstance(frame, _WhileFrame) and frame.stage == "before"):
            self.op_control_flow(op)
        frame.cond = self.val(op.operands[0])
        frame.stage, frame.split = "after", len(self.accesses)

    def op_barrier(self, op: Op) -> None:
        guarded, path, in_loop = self.branch_state()
        assert op.line_no is not None
        self.fences.append(
            Fence(
                position=len(self.accesses) - 0.5,
                op=op.name,
                addr_space=op.attrs["addr_space"],
                path=path,
                guarded=guarded,
                in_loop=in_loop,
                block_path=op.path,
                line_no=op.line_no,
                loc=op.loc,
            )
        )

    # ── everything else ──
    def op_control_flow(self, op: Op) -> NoReturn:
        self.refuse(TTIRKind.CONTROL_FLOW, f"control flow {op.name} is unsupported", op)

    def op_call(self, op: Op) -> None:
        self.refuse(TTIRKind.CALL, "calls are not modeled", op)

    def opaque_hazard(self, op: Op) -> str | None:
        """Why an opaque op (inline asm, an extern function) may touch
        memory the graph cannot see, or None: it is impure, or an operand
        hands it an address, as a pointer or as an integer made from one."""
        if op.attrs.get("pure") is not True:
            return "side effects"
        if any(parse_type(t).pointee is not None for t in op.operand_types):
            return "a pointer operand"
        if self.from_pointer(op.operands):
            return "an operand computed from a pointer (tt.ptr_to_int)"
        return None

    def from_pointer(self, values: Iterable[int]) -> bool:
        """True when a ``tt.ptr_to_int`` result flows into one of ``values``
        (through any op, region results and loop-carried values included;
        loaded data carries no address)."""
        m = self.m
        stack = list(values)
        seen: set[int] = set()
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            value = m.values[v]
            if value.op is not None:
                src = m.ops[value.op]
                if src.name == "tt.ptr_to_int":
                    return True
                if src.name in _ACCESS_OPS:
                    continue
            else:
                assert value.block is not None
                src = m.ops[m.blocks[value.block].op]
                if src.name == "tt.func":
                    continue
            stack += src.operands
            for region in src.regions:
                for b in region:
                    ops = m.blocks[b].ops
                    if ops:
                        stack += m.ops[ops[-1]].operands
        return False

    def op_inline_asm(self, op: Op) -> None:
        hazard = self.opaque_hazard(op)
        if hazard is not None:
            self.refuse(TTIRKind.INLINE_ASM, f"inline asm with {hazard}", op)
        self.bind(op, _data_dep(map(self.val, op.operands), "unmodeled inline asm"))

    def op_extern(self, op: Op) -> None:
        hazard = self.opaque_hazard(op)
        if hazard is not None:
            self.refuse(
                TTIRKind.OUT_OF_VOCABULARY,
                f"{op.name} {op.attrs.get('symbol')!r} with {hazard}",
                op,
            )
        self.op_other(op)

    def op_other(self, op: Op) -> None:
        name = op.name
        if name.startswith(_MEMORY_PREFIXES):
            self.refuse(TTIRKind.OUT_OF_VOCABULARY, f"unsupported memory op {name}", op)
        if name.split(".", 1)[0] not in _DIALECTS:
            self.refuse(TTIRKind.OUT_OF_VOCABULARY, f"op {name} is not TTIR", op)
        if name.startswith(("scf.", "cf.")):
            self.op_control_flow(op)
        if op.regions:
            self.pure_regions(op)
        elif not op.results:
            # a result-free op the reader does not know may write memory
            self.refuse(TTIRKind.OUT_OF_VOCABULARY, f"unmodeled op {name}", op)
        # Plain data (floats, dots, reductions, ...): memory-derived exactly
        # when an operand is.
        self.bind(
            op,
            _data_dep(
                map(self.val, op.operands), f"unmodeled op {name} at line {op.line_no}"
            ),
        )

    def pure_regions(self, op: Op) -> None:
        """The regions of an op the reader does not follow (tt.reduce /
        tt.scan combine bodies) compute values only: refuse any memory,
        call or opaque op inside."""
        stack = [b for region in op.regions for b in region]
        while stack:
            for oi in self.m.blocks[stack.pop()].ops:
                inner = self.m.ops[oi]
                name = inner.name
                if (
                    name in _ACCESS_OPS
                    or name == "tt.call"
                    or name.startswith(_MEMORY_PREFIXES)
                    or (name.split(".", 1)[0] not in _DIALECTS and name not in _INERT)
                    or (name in _OPAQUE and self.opaque_hazard(inner) is not None)
                    or (
                        not inner.results
                        and not inner.regions
                        and name not in _INERT
                        and name != "scf.condition"
                    )
                ):
                    self.refuse(
                        TTIRKind.OUT_OF_VOCABULARY,
                        f"{name} inside the region of {op.name}",
                        inner,
                    )
                stack.extend(b for region in inner.regions for b in region)


def _data_dep(operands: Iterable[object], why: str) -> DataDep:
    """The DataDep of an op the reader cannot model over ``operands``: a
    memory-derived reason when any operand derives from memory (``why``
    itself when it is one), else the first unmodelable operand's own reason
    (a modeling gap stays a modeling gap), else ``why``."""
    first: DataDep | None = None
    for x in operands:
        if _from_memory(x):
            return DataDep(
                why
                if why.startswith(_MEMORY_WHYS)
                else f"arith over loaded data ({why})"
            )
        if first is None and isinstance(x, DataDep):
            first = x
    return DataDep(first.why) if first is not None else DataDep(why)


def _why(v: object) -> str:
    return f" ({v.why})" if isinstance(v, DataDep) else ""


def _address_kind(v: object) -> TTIRKind:
    """The refusal kind of an unmodelable value in an address: loaded data
    is indirection, a loop-carried integer a loop-variant address."""
    if _from_memory(v):
        return TTIRKind.INDIRECT_ADDRESS
    if isinstance(v, DataDep) and v.why == _LOOP_CARRIED:
        return TTIRKind.LOOP_VARIANT_ADVANCE
    return TTIRKind.OTHER


def _operand_term(v: object) -> Term | None:
    """An atomic's cmp / val operand as a Term, or None when unmodelable."""
    return None if isinstance(v, (DataDep, PtrValue)) else v  # type: ignore[return-value]


def _with_children(t: object, kids: tuple) -> object:
    if isinstance(t, (Bin, Cmp, BoolBin)):
        return replace(t, a=kids[0], b=kids[1])
    if isinstance(t, Select):
        return replace(t, cond=kids[0], t=kids[1], f=kids[2])
    if isinstance(t, Not):
        return replace(t, a=kids[0])
    if isinstance(t, IntCast):
        return replace(t, x=kids[0])
    raise TypeError(type(t).__name__)


def _expand_dims(
    v: object,
    axis: int,
    iter_arg: Callable[[IterArgOffset], object] | None = None,
) -> object:
    """``tt.expand_dims`` inserts a size-1 dimension at ``axis``: every
    Arange lane index moves to its position in the new shape (a 1D range
    sits at 0; a dimension at or after ``axis`` shifts up by one). Pointer
    tiles follow like integer tiles, so an address and a mask expanded the
    same way keep sharing their lane variables; ``iter_arg`` maps a
    loop-carried pointer's IterArgOffset to its expanded tile's. Iterative,
    sharing-preserving (terms can be deeper than the recursion limit)."""
    if isinstance(v, PtrValue):
        return PtrValue(v.base_param, _expand_dims(v.offset, axis, iter_arg))  # type: ignore[arg-type]
    memo: dict[int, object] = {}
    stack: list[tuple[object, bool]] = [(v, False)]
    while stack:
        t, ready = stack.pop()
        if id(t) in memo:
            continue
        kids = _children(t)
        if isinstance(t, Arange):
            pos = 0 if t.dim < 0 else t.dim
            memo[id(t)] = replace(t, dim=pos + 1 if pos >= axis else pos)
        elif isinstance(t, IterArgOffset) and iter_arg is not None:
            memo[id(t)] = iter_arg(t)
        elif not kids:
            memo[id(t)] = t
        elif not ready:
            stack.append((t, True))
            stack.extend((k, False) for k in kids if id(k) not in memo)
        else:
            new = tuple(memo[id(k)] for k in kids)
            memo[id(t)] = (
                t if all(n is k for n, k in zip(new, kids)) else _with_children(t, new)
            )
    return memo[id(v)]


def _loop_delta(offset: Term, arg_id: int) -> Term | None:
    """From a yielded pointer offset ``IterArgOffset(arg_id) + d1 + ... + dn``
    (a chain of addptr sums, any association), pull out the delta
    ``d1 + ... + dn`` (``Const(0)`` for none); None for any other shape."""
    found = 0
    rest: list[Term] = []
    stack: list[Term] = [offset]
    while stack:
        t = stack.pop()
        if isinstance(t, Bin) and t.op == "+" and t.bits is None:
            stack += [t.b, t.a]
        elif isinstance(t, IterArgOffset) and t.arg_id == arg_id:
            found += 1
        else:
            rest.append(t)
    if found != 1:
        return None
    if not rest:
        return Const(0)
    delta = rest[0]
    for t in rest[1:]:
        delta = Bin("+", delta, t)
    return delta


_HANDLERS = {
    "tt.get_program_id": _Builder.op_program_id,
    "tt.get_num_programs": _Builder.op_num_programs,
    "tt.make_range": _Builder.op_make_range,
    "arith.constant": _Builder.op_constant,
    "tt.splat": _Builder.op_passthrough,
    "tt.broadcast": _Builder.op_passthrough,
    "tt.expand_dims": _Builder.op_expand_dims,
    "arith.extsi": _Builder.op_cast,
    "arith.extui": _Builder.op_cast,
    "arith.trunci": _Builder.op_cast,
    "tt.addptr": _Builder.op_addptr,
    **{name: _Builder.op_bin for name in _BIN_OPS},
    "arith.cmpi": _Builder.op_cmpi,
    "arith.andi": _Builder.op_boolbin,
    "arith.ori": _Builder.op_boolbin,
    "arith.select": _Builder.op_select,
    "tt.bitcast": _Builder.op_bitcast,
    "tt.load": _Builder.op_load,
    "tt.store": _Builder.op_store,
    "tt.atomic_rmw": _Builder.op_atomic_rmw,
    "tt.atomic_cas": _Builder.op_atomic_cas,
    "scf.for": _Builder.op_for,
    "scf.if": _Builder.op_if,
    "scf.while": _Builder.op_while,
    "scf.condition": _Builder.op_condition,
    "ttg.barrier": _Builder.op_barrier,
    "tt.call": _Builder.op_call,
    "tt.elementwise_inline_asm": _Builder.op_inline_asm,
    "tt.extern_elementwise": _Builder.op_extern,
}
