"""The UNSAT-only conflict precheck, the snapshot lemmas it derives, and the
query caches: each may discard a pair only when the full query is UNSAT."""

from __future__ import annotations

import pytest
import z3

from tilelens.clients.race_detector import conflict_simplification as cs
from tilelens.clients.race_detector import two_copy_symbolic_hb_solver as tc
from tilelens.clients.race_detector.snapshot_lemmas import (
    SnapshotLemmaCache,
    snapshot_table_lemmas,
)
from tilelens.clients.symbolic_engine import SymbolicExpr

from .solver_records import every_fence, rmw, solve, store

PID0, PID2 = SymbolicExpr.PID0, SymbolicExpr.PID2


def _check(*conditions):
    solver = z3.Solver(ctx=conditions[0].ctx)
    solver.set(timeout=2000)
    solver.add(*conditions)
    answer = solver.check()
    assert answer != z3.unknown
    return answer


# ──────────────────────── mixed-radix relaxation ────────────────────────


def _radix_formula(*, missing=None, lane_extent=64, bytes_per_block=256):
    """Two flattened block indices q*r + d with 0 <= d < r, r >= 1."""
    qa, qb, da, db, la, lb, r = z3.Ints("qa qb da db la lb radix")
    aa = bytes_per_block * (qa * r + da) + 4 * la
    ab = bytes_per_block * (qb * r + db) + 4 * lb
    bounds = {"positive": r >= 1, "lower": da >= 0, "upper": da < r}
    conditions = [value for name, value in bounds.items() if name != missing]
    conditions += [
        db >= 0,
        db < r,
        la >= 0,
        la < lane_extent,
        lb >= 0,
        lb < lane_extent,
        z3.Or(qa != qb, da != db),
        aa < ab + 4,
        ab < aa + 4,
    ]
    correspondence = ((qb, qa), (db, da), (lb, la))
    variables = dict(qa=qa, qb=qb, da=da, db=db, la=la, lb=lb, r=r)
    return conditions, correspondence, variables


def test_mixed_radix_precheck_proves_disjoint_blocks_without_pinning_radix():
    conditions, correspondence, _ = _radix_formula()
    assert cs.conflict_impossible(conditions, correspondence)


@pytest.mark.parametrize(
    "kwargs,witness",
    [
        ({"missing": "positive"}, None),  # implied, but not explicit
        ({"missing": "lower"}, dict(r=2, qa=1, da=-1, qb=0, db=1, la=0, lb=0)),
        ({"missing": "upper"}, dict(r=2, qa=0, da=2, qb=1, db=0, la=0, lb=0)),
        # a lane carries into the next block
        ({"lane_extent": 65}, dict(r=2, qa=0, qb=0, da=0, db=1, la=64, lb=0)),
        # partial byte overlap of unequal starts
        (
            {"lane_extent": 1, "bytes_per_block": 2},
            dict(r=2, qa=0, qb=0, da=0, db=1, la=0, lb=0),
        ),
    ],
)
def test_radix_relaxation_needs_every_original_premise(kwargs, witness):
    conditions, correspondence, variables = _radix_formula(**kwargs)
    assert not cs.conflict_impossible(conditions, correspondence)
    if witness is not None:
        pins = [variables[name] == value for name, value in witness.items()]
        assert _check(*conditions, *pins) == z3.sat


@pytest.mark.parametrize("factor", ["radix", "snapshot"])
def test_private_switch_disables_only_its_optional_path(monkeypatch, factor):
    if factor == "radix":
        conditions, correspondence, _ = _radix_formula()
        switch = "_ENABLE_RADIX_PRECHECK"
    else:
        value = z3.Select(
            z3.Array("switch_table", z3.IntSort(), z3.IntSort()), z3.Int("i")
        )
        conditions, correspondence = [value < 0, value >= 0], ()
        switch = "_ENABLE_SNAPSHOT_PRECHECK"
    assert cs.conflict_impossible(conditions, correspondence)
    monkeypatch.setattr(cs, switch, False)
    assert not cs.conflict_impossible(conditions, correspondence)


def test_unknown_precheck_leaves_the_pair_to_the_full_query(monkeypatch):
    settings = []

    class UnknownPrecheck:
        def set(self, **options):
            settings.append(options)

        def add(self, *conditions):
            pass

        def check(self):
            return z3.unknown

    monkeypatch.setattr(cs.z3, "SolverFor", lambda logic: UnknownPrecheck())
    g2 = z3.Int("grid_2")
    rec = store(2 * (PID0 * g2 + PID2), event_id=0)
    solver = solve([rec], grid=(1, 1, g2), extra_assumptions=(g2 == 2,))
    assert solver.find_races()
    assert settings and all(options == {"timeout": 500} for options in settings)


def test_precheck_decides_disjoint_blocks_without_the_full_query(monkeypatch):
    g0, g1, g2 = z3.Ints("grid_0 grid_1 grid_2")
    row, col = z3.Ints("row col")
    addr = 4 * (4096 * (PID0 * g2 + PID2) + 4096 * SymbolicExpr.PID1 + 64 * row + col)
    mask = z3.And(SymbolicExpr.PID1 * 64 + row < 64, SymbolicExpr.PID1 * 64 + row >= 0)
    solver = solve(
        [store(1_000_000 + addr, event_id=0, mask=mask)],
        grid=(g0, g1, g2),
        arange_dict={(0, 64, "row"): (row, None), (0, 64, "col"): (col, None)},
    )

    def full_query_must_not_run(*args):
        raise AssertionError("the necessary conflict condition is already UNSAT")

    monkeypatch.setattr(solver, "_race_query_is_sat", full_query_must_not_run)
    assert solver.find_races() == []
    assert [(kind, sat) for kind, _, sat in solver.query_stats] == [
        ("cross", False),
        ("intra", False),
    ]
    # Feasibility stays an independent obligation of the full base system.
    monkeypatch.setattr(solver, "_conflict_precheck", full_query_must_not_run)
    assert solver.check_feasibility()


@pytest.mark.parametrize("atomic", [False, True])
def test_partial_byte_overlap_survives_the_precheck(atomic):
    g2 = z3.Int("grid_2")
    addr = 2 * (PID0 * g2 + PID2)
    rec = rmw(addr, event_id=0) if atomic else store(addr, event_id=0)
    reports = solve([rec], grid=(1, 1, g2), extra_assumptions=(g2 == 2,)).find_races()
    assert reports and reports[0].witness_grid_a != reports[0].witness_grid_b


# ──────────────────────── snapshot lemmas ────────────────────────


def _snapshot(values=(9, 4, 7), *, start=0):
    array = z3.Array("snapshot", z3.IntSort(), z3.IntSort())
    a, b, va, vb = z3.Ints("a b va vb")
    cells = [array[start + i] == value for i, value in enumerate(values)]
    return z3.And(*cells, array[a] == va, array[b] == vb), array, a, b


@pytest.mark.parametrize("start", [-8, 0, 13])
def test_complete_table_entails_range_and_injectivity(start):
    expression, array, a, b = _snapshot(start=start)
    lemmas = snapshot_table_lemmas(expression)
    assert len(lemmas) == 3
    assert _check(expression, z3.Not(z3.And(*lemmas))) == z3.unsat
    inside = [a >= start, a <= start + 2, b >= start, b <= start + 2]
    collide = [a != b, array[a] == array[b]]
    assert _check(expression, *lemmas, *inside, *collide) == z3.unsat
    # Outside the certified interval every value stays possible.
    assert (
        _check(expression, *lemmas, a == start, b == 10**6, array[a] == array[b])
        == z3.sat
    )


def test_duplicate_values_or_a_missing_cell_keep_the_collision():
    expression, array, a, b = _snapshot((9, 9, 7))
    lemmas = snapshot_table_lemmas(expression)
    assert len(lemmas) == 2  # range facts only
    assert _check(expression, *lemmas, a == 0, b == 1, array[a] == array[b]) == z3.sat
    expression, array, a, b = _snapshot()
    holes = z3.And(*[c for c in expression.children() if not z3.eq(c, array[1] == 4)])
    assert snapshot_table_lemmas(holes) == ()
    # The solver's path: the cache must refuse the hole too.
    assert SnapshotLemmaCache().lemmas([*holes.children()]) == ()


@pytest.mark.parametrize("connective", ["or", "implies", "ite"])
def test_conditional_cells_never_become_facts(connective):
    expression, array, a, b = _snapshot()
    cells, reads = expression.children()[:3], expression.children()[3:]
    flag = z3.Bool("flag")
    conditional = {
        "or": z3.Or(flag, z3.And(*cells)),
        "implies": z3.Implies(flag, z3.And(*cells)),
        "ite": z3.If(flag, z3.And(*cells), z3.BoolVal(True)),
    }[connective]
    assert snapshot_table_lemmas(z3.And(conditional, *reads)) == ()
    assert SnapshotLemmaCache().lemmas([conditional, *reads]) == ()


def test_cached_lemmas_match_the_standalone_helper_and_follow_changes():
    expression, array, a, b = _snapshot((2, 0, 1))
    cells, reads = list(expression.children()[:3]), expression.children()[3:]
    cache = SnapshotLemmaCache()
    lemmas = cache.lemmas([z3.And(*cells), *reads])
    expected = snapshot_table_lemmas(expression)
    assert [x.sexpr() for x in lemmas] == [x.sexpr() for x in expected]
    cells[1] = array[1] == 2  # a duplicate value drops the injectivity fact
    assert len(cache.lemmas([*cells, *reads])) == 2


def _destination_conflict(values=(2, 0, 1), *, launch=True):
    table = z3.Array("destination", z3.IntSort(), z3.IntSort())
    pa, pb, la, lb, grid = z3.Ints("pa pb la lb grid")
    address_a, address_b = 16 * table[pa] + 4 * la, 16 * table[pb] + 4 * lb
    conditions = [
        *(table[index] == value for index, value in enumerate(values)),
        grid >= 1,
        pa >= 0,
        pb >= 0,
        pa < grid,
        pb < grid,
        pa != pb,
        la >= 0,
        lb >= 0,
        la < 4,
        lb < 4,
        address_a < address_b + 4,
        address_b < address_a + 4,
    ]
    if launch:
        conditions.append(grid == len(values))
    return conditions


@pytest.mark.parametrize(
    "values,launch,impossible",
    [((2, 0, 1), True, True), ((2, 0, 1), False, False), ((2, 2, 1), True, False)],
)
def test_snapshot_precheck_keeps_scope_and_duplicate_destinations(
    values, launch, impossible
):
    """A permutation table rules out the collision at the launch extent only;
    an any-grid query or a duplicate destination keeps it."""
    conditions = _destination_conflict(values, launch=launch)
    assert (_check(*conditions) == z3.unsat) is impossible
    answer = cs.conflict_impossible(
        conditions,
        simplify_first=True,
        expression_cache=cs._PureSelectExpressionCache(),
        snapshot_cache=SnapshotLemmaCache(),
    )
    assert answer is impossible


def test_common_snapshot_premises_enable_the_linear_precheck(monkeypatch):
    lane = z3.Int("common_lane")
    table = z3.Array("common_table", z3.IntSort(), z3.IntSort())
    rec = store(
        4096 + 256 * PID0 + 4 * lane, event_id=0, mask=z3.And(lane >= 0, lane < 64)
    )
    solver = solve(
        [rec],
        grid=(z3.Int("common_grid"), 1, 1),
        arange_dict={(0, 64, "lane"): (lane, None)},
        extra_assumptions=(z3.Select(table, 0) == 7, z3.Select(table, 1) == 11),
    )

    def full_query_must_not_run(*args):
        raise AssertionError("the linear necessary condition proves disjointness")

    monkeypatch.setattr(solver, "_race_query_is_sat", full_query_must_not_run)
    assert solver.find_races() == []


# ──────────────────────── query caches ────────────────────────


def _repeated_stores(monkeypatch, *, grid=(2, 1, 1), **kwargs):
    records = [store(4096 + 8 * PID0, event_id=i, program_seq=i) for i in range(2)]
    core = solve(records, grid=grid, fence_between=every_fence, **kwargs)
    # Exercise full-query reuse independently of the optional precheck.
    monkeypatch.setattr(core, "_conflict_precheck", lambda *a, **kw: False)
    return core


def test_identical_unsat_queries_are_decided_once_per_invocation(monkeypatch):
    core = _repeated_stores(monkeypatch)
    created = []
    original = tc.Solver

    def counted(*args, **kwargs):
        created.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(tc, "Solver", counted)
    assert core.find_races() == []
    assert len(created) == 1
    assert [(kind, sat) for kind, _, sat in core.query_stats] == [("cross", False)] * 4
    assert core.find_races() == [] and len(created) == 2


def test_launch_extent_unsat_cannot_hide_an_any_grid_race(monkeypatch):
    """The address ignores pid1: the fallback launch is disjoint, but a
    second pid1 row aliases in the symbolic grid."""
    grid1 = z3.Int("any_grid_1")
    core = _repeated_stores(
        monkeypatch, grid=(2, grid1, 1), enum_fallback_grid=(2, 1, 1)
    )
    original = core._race_query_is_sat
    calls = []

    def first_unknown(query, first, second):
        calls.append(True)
        if len(calls) == 1:
            raise tc.UnsupportedSymbolicRaceQuery("force launch-only fallback")
        return original(query, first, second)

    monkeypatch.setattr(core, "_race_query_is_sat", first_unknown)
    reports = core.find_races()
    assert core.enum_used and len(calls) == 4
    assert any(r.witness_grid_a[1] != r.witness_grid_b[1] for r in reports)


@pytest.mark.parametrize("family", ["rf_constraints", "extra_assumptions"])
def test_base_constraint_cache_follows_mutated_families(family):
    solver = solve([store(z3.IntVal(4096), event_id=0)], grid=(1, 1, 1))
    conditions = []
    setattr(solver, family, conditions)
    assert solver._base_solver().check() == z3.sat
    conditions.append(z3.BoolVal(False))
    assert solver._base_solver().check() == z3.unsat
    conditions[0] = z3.BoolVal(True)
    assert solver._base_solver().check() == z3.sat
    solver.hb[0][0] = z3.BoolVal(True)  # a forced HB cycle admits no execution
    assert solver._base_solver().check() == z3.unsat
