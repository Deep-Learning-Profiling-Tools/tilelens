"""Happens-before edges and their transitive closure keep every conditional
edge, cycle obligation and scope check."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest
import z3

from tilelens.clients.race_detector import hb_common
from tilelens.clients.race_detector.hb_common import UnsupportedSymbolicRaceQuery

from .solver_records import every_fence, fences_at, load, solve, store


def _dense_closure(edges):
    """The plain layered Floyd-Warshall recurrence, as the oracle."""
    reach = [list(row) for row in edges]
    for k in range(len(reach)):
        reach = [
            [
                z3.Or(reach[i][j], z3.And(reach[i][k], reach[k][j]))
                for j in range(len(reach))
            ]
            for i in range(len(reach))
        ]
    return reach


def _closure(edges):
    return hb_common.build_transitive_hb(
        list(range(len(edges))), lambda i, j: edges[i][j]
    )


def _assert_equivalent(actual, expected):
    differences = [
        a != b
        for actual_row, expected_row in zip(actual, expected)
        for a, b in zip(actual_row, expected_row)
    ]
    assert len(actual) == len(expected)
    if differences:
        solver = z3.Solver(ctx=differences[0].ctx)
        solver.add(z3.Or(*differences))
        assert solver.check() == z3.unsat


def test_empty_graph_and_constant_self_edges():
    def unreachable(*args):
        raise AssertionError("an empty graph has no edges")

    assert hb_common.build_transitive_hb([], unreachable) == []
    for self_edge in (False, True):
        assert z3.is_true(_closure([[z3.BoolVal(self_edge)]])[0][0]) == self_edge


def test_acyclicity_rejects_a_forced_cycle_and_keeps_a_conditional_cut():
    p, q = z3.Bools("acyclic_p acyclic_q")
    f = z3.BoolVal(False)
    result = _closure([[f, p], [q, f]])
    acyclic = z3.And(*(z3.Not(result[i][i]) for i in range(2)))
    _assert_equivalent([[acyclic]], [[z3.Not(z3.And(p, q))]])


def test_arithmetic_array_and_custom_context_edges_keep_full_conditions():
    ctx = z3.Context()
    index, limit = z3.Ints("index limit", ctx=ctx)
    snapshot = z3.Array("snapshot", z3.IntSort(ctx), z3.IntSort(ctx))
    f = z3.BoolVal(False, ctx=ctx)
    edges = [
        [f, z3.And(index >= 0, index < limit), f],
        [f, f, z3.Select(snapshot, index) == 7],
        [index == limit, f, z3.BoolVal(True, ctx=ctx)],
    ]
    result = _closure(edges)
    assert all(entry.ctx == ctx for row in result for entry in row)
    _assert_equivalent(result, _dense_closure(edges))


@pytest.mark.parametrize("seed,size", [(0, 2), (1, 3), (2, 4), (3, 5), (4, 6)])
def test_small_symbolic_graphs_match_the_layered_recurrence(seed, size):
    choose = random.Random(seed).choice
    p, q, r = z3.Bools(f"random_p_{seed} random_q_{seed} random_r_{seed}")
    choices = [
        z3.BoolVal(False),
        z3.BoolVal(True),
        p,
        q,
        r,
        z3.Not(p),
        z3.And(p, q),
        z3.Or(q, r),
    ]
    edges = [[choose(choices) for _ in range(size)] for _ in range(size)]
    _assert_equivalent(_closure(edges), _dense_closure(edges))


def _edge_solver(*, fence_between=None, dependency=False):
    active_load, active_store = z3.Bools("edge_active_load edge_active_store")
    first = load(z3.IntVal(100), event_id=0, program_seq=0, mask=active_load)
    second = store(z3.IntVal(100), event_id=1, program_seq=2, mask=active_store)
    if dependency:
        second.dep_loads = (first.event_id,)
    return solve([first, second], fence_between=fence_between)


@pytest.mark.parametrize(
    "fence_between,dependency,ordered",
    [
        (every_fence, False, True),
        (None, False, False),
        (fences_at(1), False, True),
        (None, True, True),
        (fences_at(0, 2), False, False),
    ],
)
def test_program_order_is_an_edge_only_across_a_fence_or_a_dependency(
    fence_between, dependency, ordered
):
    core = _edge_solver(fence_between=fence_between, dependency=dependency)
    assert not core.reads_through
    differences = []
    for first in core.events:
        for second in core.events:
            expected = (
                z3.And(first.active, second.active)
                if ordered
                and first.copy == second.copy
                and first.program_seq < second.program_seq
                else z3.BoolVal(False)
            )
            differences.append(core._edge(first, second) != expected)
    _assert_equivalent([differences], [[z3.BoolVal(False)] * len(differences)])


def test_reads_through_keeps_the_cross_copy_cta_scope_condition():
    core = _edge_solver()
    first = replace(core.events[0], sem="release", scope="cta")
    second = replace(core.events[-1], sem="acquire", scope="gpu")
    rt = z3.Bool("edge_reads_through")
    core.reads_through[(first.idx, second.idx)] = rt
    expected = z3.And(rt, *(a == b for a, b in zip(first.pid, second.pid)))
    _assert_equivalent([[core._edge(first, second)]], [[expected]])


@pytest.mark.parametrize("invalid_endpoint", ["writer", "reader"])
def test_reads_through_rejects_an_unknown_scope(invalid_endpoint):
    core = _edge_solver()
    bad = "unsupported-scope"
    first = replace(
        core.events[0],
        sem="release",
        scope=bad if invalid_endpoint == "writer" else "gpu",
    )
    second = replace(
        core.events[-1],
        sem="acquire",
        scope=bad if invalid_endpoint == "reader" else "gpu",
    )
    # Presence matters even when the stored reads-through formula is False.
    core.reads_through[(first.idx, second.idx)] = z3.BoolVal(False)
    with pytest.raises(UnsupportedSymbolicRaceQuery, match="unsupported memory scope"):
        core._edge(first, second)
