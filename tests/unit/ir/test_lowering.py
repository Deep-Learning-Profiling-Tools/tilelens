"""tilelens.ir.lowering: the Term -> Z3 lowering shared by the compiled-mode
clients.

CPU only: terms are built by hand over small AccessGraphs and lowered with
a test TermLeaves whose leaves are free Z3 variables (or constants), then
checked by Z3 equivalence or by evaluation.
The compiled sanitizer's own results on top of the lowering are pinned in
tests/unit/sanitizer_compiled/.
"""

from __future__ import annotations

import sys

import pytest
import z3

from tilelens.ir.lowering import Lowerer, children, fold
from tilelens.ir.ttir_reader import (
    AccessGraph,
    Arange,
    Bin,
    BoolBin,
    Cmp,
    Const,
    DataDep,
    IntCast,
    IterArgInfo,
    IterArgOffset,
    LoopInfo,
    LoopVar,
    Not,
    NumPrograms,
    Observed,
    Param,
    Pid,
    PtrValue,
    Select,
)

LOOP = LoopInfo("%loop", lower=Param("lo"), upper=Param("hi"), step=Param("st"))


class _Refusal(Exception):
    """A client's own refusal, raised from a leaf."""


class _Leaves:
    """Every leaf a free Int of ``ctx`` named after it; records the names
    it made."""

    def __init__(self, ctx: z3.Context | None = None):
        self.ctx = ctx
        self.made: list[str] = []

    def _var(self, name: str) -> z3.ArithRef:
        self.made.append(name)
        return z3.Int(name, self.ctx)

    def param(self, t):
        return self._var(t.name)

    def pid(self, t):
        return self._var(f"pid_{t.axis}")

    def num_programs(self, t):
        return self._var(f"grid_{t.axis}")

    def arange(self, t):
        return self._var(f"arange_{t.start}_{t.end}_d{t.dim}")

    def iteration(self, loop_ssa):
        return self._var(f"k{loop_ssa}")

    def observed(self, t):
        return self._var(f"observed_{t.access_index}")

    def data_dep(self, t):
        raise _Refusal(t.why)


def _graph(loop: LoopInfo | None = None, iter_args=()) -> AccessGraph:
    return AccessGraph("k", (), (), loop, iter_args)


def _lower(term, graph: AccessGraph | None = None, leaves: _Leaves | None = None):
    return Lowerer(graph or _graph(LOOP), leaves or _Leaves()).lower(term)


def _v(name: str, ctx: z3.Context | None = None) -> z3.ArithRef:
    return z3.Int(name, ctx)


def _proved(claim) -> bool:
    solver = z3.Solver(ctx=claim.ctx)
    solver.add(z3.Not(claim))
    return solver.check() == z3.unsat


def _eval(term) -> int | bool:
    e = z3.simplify(_lower(term))
    if z3.is_bool(e):
        assert z3.is_true(e) or z3.is_false(e), e
        return z3.is_true(e)
    return e.as_long()


X, Y = Param("x"), Param("y")


# ─────────────────────────── the Z3 context ───────────────────────────


def _every_kind() -> tuple[object, AccessGraph]:
    """One term reaching every kind of the algebra but DataDep."""
    graph = _graph(LOOP, (IterArgInfo(0, "p", Param("o0"), Const(4), "%loop"),))
    lane = Bin("+", Arange("%r", 0, 16, dim=0), IntCast("extsi", 32, 64, Pid(1)))
    moved = Bin("*", Bin("//", IterArgOffset(0), NumPrograms(0)), LoopVar("%loop"))
    cond = BoolBin("or", Cmp("ult", X, Const(8)), Not(Cmp("eq", Observed(2), Y)))
    return Select(cond, Bin("umin", lane, moved), Bin("%", X, Const(3))), graph


@pytest.mark.parametrize("own_context", [False, True])
def test_terms_are_made_in_the_leaves_context(own_context, monkeypatch):
    """ctx=None lowers into Z3's main context, a given Context into that
    Context (a constant included), and the lowering creates none."""
    ctx = z3.Context() if own_context else None
    expected = ctx if own_context else z3.main_ctx()
    term, graph = _every_kind()

    def no_context(*args, **kwargs):
        raise AssertionError("the lowering created a z3.Context")

    monkeypatch.setattr(z3.Context, "__init__", no_context)
    lowerer = Lowerer(graph, _Leaves(ctx))
    lowered = [lowerer.lower(term), lowerer.value(term), lowerer.cond(term)]
    lowered += [lowerer.lower(Const(7)), lowerer.cond(Const(1))]
    lowered += [lowerer.value(Cmp("slt", X, Y)), lowerer.cond(Not(Const(0)))]
    monkeypatch.undo()
    assert all(e.ctx is expected for e in lowered)
    assert z3.is_int(lowered[0]) and z3.is_bool(lowered[2])


# ─────────────────────────── iteration and the memo ───────────────────────────


def test_a_chain_deeper_than_the_recursion_limit_lowers():
    graph = _graph()
    chain = Pid(0)
    for _ in range(5000):
        chain = Bin("+", chain, Const(1), 32)
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(1000)
    try:
        lowered = Lowerer(graph, _Leaves()).value(chain)
    finally:
        sys.setrecursionlimit(limit)
    assert _proved(lowered == _v("pid_0") + 5000)


def test_fold_applies_children_first_and_each_term_once():
    graph = _graph(LOOP, (IterArgInfo(0, "p", Const(2), Const(3)),))
    two = Const(2)
    root = Bin("+", Bin("*", two, two), IterArgOffset(0))
    order: list[object] = []

    def apply(t, values):
        order.append(t)
        if isinstance(t, Const):
            return t.value
        if isinstance(t, IterArgOffset):
            return values[0] + 10 * values[1]  # iteration 10
        return values[0] * values[1] if t.op == "*" else values[0] + values[1]

    memo: dict = {}
    assert fold(root, graph, apply, memo) == 2 * 2 + (2 + 10 * 3)
    ids = [id(t) for t in order]
    assert len(ids) == 6 and ids.count(id(two)) == 1
    assert ids.index(id(two)) < ids.index(id(root.a)) < ids.index(id(root))
    # the memo holds every term: a second fold applies nothing
    assert fold(root, graph, apply, memo) == 36 and len(order) == 6


def test_children_is_the_value_dependency_relation():
    info = IterArgInfo(0, "p", Param("o0"), Param("d"))
    graph = _graph(LOOP, (info,))
    less = Cmp("slt", X, Y)

    def kids(t) -> list[int]:
        return [id(k) for k in children(t, graph)]

    assert (
        kids(Bin("+", X, Y))
        == kids(less)
        == kids(BoolBin("or", X, Y))
        == [id(X), id(Y)]
    )
    assert kids(Select(less, X, Y)) == [id(less), id(X), id(Y)]
    assert kids(Not(less)) == kids(IntCast("trunci", 64, 32, less)) == [id(less)]
    assert kids(IterArgOffset(0)) == [id(info.offset0), id(info.delta)]
    assert kids(LoopVar("%loop")) == [id(LOOP.lower), id(LOOP.step)]
    for leaf in (Const(1), Pid(0), NumPrograms(1), Arange("%r", 0, 4), X, Observed(0)):
        assert kids(leaf) == []
    assert kids(DataDep()) == []


# ─────────────────────────── operator semantics ───────────────────────────


@pytest.mark.parametrize(
    "unsigned, signed", [("u//", "//"), ("u%", "%"), ("umin", "min"), ("umax", "max")]
)
def test_unsigned_ops_read_as_their_signed_twins(unsigned, signed):
    lowerer = Lowerer(_graph(), _Leaves())
    assert _proved(
        lowerer.lower(Bin(unsigned, X, Y)) == lowerer.lower(Bin(signed, X, Y))
    )


@pytest.mark.parametrize(
    "unsigned, signed", [("ult", "slt"), ("ule", "sle"), ("ugt", "sgt"), ("uge", "sge")]
)
def test_unsigned_predicates_read_as_their_signed_twins(unsigned, signed):
    lowerer = Lowerer(_graph(), _Leaves())
    got, twin = lowerer.lower(Cmp(unsigned, X, Y)), lowerer.lower(Cmp(signed, X, Y))
    assert z3.is_bool(got) and _proved(got == twin)


@pytest.mark.parametrize(
    "pred, expected",
    [
        ("slt", lambda x, y: x < y),
        ("sle", lambda x, y: x <= y),
        ("sgt", lambda x, y: x > y),
        ("sge", lambda x, y: x >= y),
        ("eq", lambda x, y: x == y),
        ("ne", lambda x, y: x != y),
    ],
)
def test_predicates(pred, expected):
    assert _proved(_lower(Cmp(pred, X, Y)) == expected(_v("x"), _v("y")))


@pytest.mark.parametrize("kind", ["trunci", "extsi", "extui"])
def test_an_int_cast_reads_as_its_operand(kind):
    assert z3.eq(_lower(IntCast(kind, 64, 32, X)), _v("x"))
    # of an i1: the compare's 0/1
    cast = _lower(IntCast(kind, 1, 32, Cmp("slt", X, Y)))
    assert z3.is_int(cast) and _proved(cast == z3.If(_v("x") < _v("y"), 1, 0))


def test_i1_values_are_bool_or_0_1_by_position():
    x, y = _v("x"), _v("y")
    lowerer = Lowerer(_graph(), _Leaves())
    less = Cmp("slt", X, Y)
    # an integer position: 0/1
    assert _proved(lowerer.lower(Bin("+", less, Const(1))) == z3.If(x < y, 1, 0) + 1)
    assert _proved(lowerer.value(less) == z3.If(x < y, 1, 0))
    # a boolean position: an i1 constant (dense<true>) and an Int are != 0
    assert _proved(lowerer.lower(BoolBin("and", Const(1), less)) == (x < y))
    assert _proved(lowerer.lower(BoolBin("or", X, Const(0))) == (x != 0))
    assert _proved(lowerer.lower(Not(Const(0))))
    assert _proved(lowerer.cond(X) == (x != 0))
    # a compare of an i1 with an integer reads the i1 as 0/1
    assert _proved(lowerer.lower(Cmp("eq", less, Const(1))) == (x < y))
    # a Select: its condition is boolean; Bool arms stay Bool
    both = lowerer.lower(Select(X, less, Cmp("eq", X, Y)))
    assert z3.is_bool(both) and _proved(both == z3.If(x != 0, x < y, x == y))
    # arms of different sorts are Int
    mixed = lowerer.lower(Select(less, Cmp("eq", X, Const(0)), Const(5)))
    assert z3.is_int(mixed)
    assert _proved(mixed == z3.If(x < y, z3.If(x == 0, 1, 0), 5))


@pytest.mark.parametrize(
    "op, a, b, expected",
    [
        ("min", 3, -2, -2),
        ("min", -2, 3, -2),
        ("max", 3, -2, 3),
        ("max", -2, 3, 3),
        ("+", 7, -9, -2),
        ("-", 7, -9, 16),
        ("*", 7, -9, -63),
    ],
)
def test_arithmetic(op, a, b, expected):
    assert _eval(Bin(op, Const(a), Const(b), 32)) == expected


@pytest.mark.parametrize(
    "a, b, quotient, remainder",
    [
        (7, 2, 3, 1),
        (-7, 2, -3, -1),
        (7, -2, -3, 1),
        (-7, -2, 3, -1),
        (6, 3, 2, 0),
        (-6, 3, -2, 0),
        (0, -5, 0, 0),
        (-1, 5, 0, -1),
    ],
)
def test_division_truncates_toward_zero(a, b, quotient, remainder):
    """arith.divsi / remsi: the remainder has the dividend's sign."""
    assert _eval(Bin("//", Const(a), Const(b), 32)) == quotient
    assert _eval(Bin("%", Const(a), Const(b), 32)) == remainder


def test_loop_terms_read_the_clients_iteration():
    """LoopVar is lower + k * step, IterArgOffset offset0 + k * delta, with
    k the leaves' iteration of that loop: an IterArgInfo without a loop_ssa
    is the graph's loop's."""
    graph = _graph(
        LOOP,
        (
            IterArgInfo(0, "p", Param("o0"), Param("d0")),
            IterArgInfo(1, "p", Param("o1"), Param("d1"), "%other"),
        ),
    )
    leaves = _Leaves()
    lowerer = Lowerer(graph, leaves)
    k, other = _v("k%loop"), _v("k%other")
    assert _proved(lowerer.lower(LoopVar("%loop")) == _v("lo") + k * _v("st"))
    assert _proved(lowerer.lower(IterArgOffset(0)) == _v("o0") + k * _v("d0"))
    assert _proved(lowerer.lower(IterArgOffset(1)) == _v("o1") + other * _v("d1"))
    assert "hi" not in leaves.made  # the upper bound is not the variable's value


# ─────────────────────────── leaves and bugs ───────────────────────────


def test_a_leaf_refusal_propagates_unchanged():
    with pytest.raises(_Refusal, match="loaded value"):
        Lowerer(_graph(), _Leaves()).lower(BoolBin("and", X, DataDep("loaded value")))


@pytest.mark.parametrize(
    "term, error, match",
    [
        (PtrValue("p", Const(0)), TypeError, "unknown term PtrValue"),
        (Bin("**", X, Y), ValueError, "unknown integer op"),
        (Cmp("oeq", X, Y), ValueError, "unknown cmpi predicate"),
        (BoolBin("xor", X, Y), ValueError, "unknown boolean op"),
    ],
)
def test_a_term_outside_the_algebra_is_a_bug(term, error, match):
    with pytest.raises(error, match=match):
        _lower(term)


@pytest.mark.parametrize("term", [LoopVar("%loop"), IterArgOffset(0)])
def test_a_loop_term_without_a_loop_is_a_bug(term):
    graph = _graph(None, (IterArgInfo(0, "p", Const(0), Const(1)),))
    with pytest.raises(ValueError, match="without a loop"):
        Lowerer(graph, _Leaves()).lower(term)
