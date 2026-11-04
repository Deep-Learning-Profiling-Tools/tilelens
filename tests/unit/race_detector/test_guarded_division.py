"""Guarded division folding: only an original mask certifies a variable
divisor's value, and every failure keeps the original query."""

from __future__ import annotations

import pytest
import z3

from tilelens.clients.race_detector import guarded_division as gd


def _grouped(*, masked=True):
    pid, lane, grid = z3.Ints("pid lane grid")
    quotient = pid / 4
    size = z3.If(quotient >= 0, 2 - 2 * quotient, 2)
    remainder = z3.If(size == 0, pid % 4, (pid % 4) % size)
    row = 64 * quotient + 32 * remainder + lane
    conditions = [pid >= 0, pid < grid, grid >= 1, lane >= 0, lane < 32]
    if masked:
        conditions.append(row < 64)
    return z3.And(*conditions), (pid, lane, grid), quotient, size, row


def _check(formula, *extra):
    solver = z3.Solver()
    solver.set(timeout=3000)
    solver.add(formula, *extra)
    result = solver.check()
    assert result != z3.unknown
    return result, solver


def test_access_mask_certifies_constants_without_a_grid_pin():
    original, (pid, lane, grid), quotient, size, _ = _grouped()
    rewritten = gd.guarded_division_normal_form(original)
    assert not z3.eq(original, rewritten)
    assert _check(z3.Xor(original, rewritten))[0] == z3.unsat
    for extent in (4, 1000000):
        answer, solver = _check(rewritten, grid == extent, pid == 1, lane == 0)
        assert answer == z3.sat
        model = solver.model()
        assert z3.is_true(model.eval(original, model_completion=True))
        assert model.eval(quotient).as_long() == 0
        assert model.eval(size).as_long() == 2


@pytest.mark.parametrize("guard", [None, "or", "implies"])
def test_absent_or_conditional_mask_keeps_zero_and_negative_divisors(guard):
    base, (pid, lane, grid), _, size, row = _grouped(masked=False)
    flag = z3.Bool("flag")
    original = {
        None: base,
        "or": z3.And(base, z3.Or(flag, row < 64)),
        "implies": z3.And(base, z3.Implies(flag, row < 64)),
    }[guard]
    rewritten = gd.guarded_division_normal_form(original)
    extra = () if guard is None else (flag if guard == "or" else z3.Not(flag),)
    for instance, divisor in ((4, 0), (8, -2)):
        result, solver = _check(
            rewritten, grid == 16, pid == instance, lane == 0, *extra
        )
        assert result == z3.sat
        assert solver.model().eval(size).as_long() == divisor
        assert z3.is_true(solver.model().eval(original, model_completion=True))


@pytest.mark.parametrize(
    "failure", ["unknown", "sat", "exception", "budget", "disabled"]
)
def test_failed_certificate_keeps_the_original_formula(monkeypatch, failure):
    original, *_ = _grouped()
    check = z3.Solver.check

    def refuse_uniqueness(self, *args, **kwargs):
        # Allow discovery of a candidate value, then refuse its uniqueness.
        if any(z3.is_not(a) and z3.is_eq(a.arg(0)) for a in self.assertions()):
            if failure == "exception":
                raise z3.Z3Exception("injected certificate failure")
            return z3.unknown if failure == "unknown" else z3.sat
        return check(self, *args, **kwargs)

    if failure == "budget":
        monkeypatch.setattr(gd, "_CERTIFY_TIMEOUT_MS", 0)
    elif failure == "disabled":
        monkeypatch.setattr(gd, "_ENABLE_GUARDED_DIVISION", False)
    else:
        monkeypatch.setattr(z3.Solver, "check", refuse_uniqueness)
    assert z3.eq(gd.guarded_division_normal_form(original), original)


def test_quantified_input_is_unchanged():
    original, (pid, _, _), *_ = _grouped()
    quantified = z3.Exists(pid, original)
    assert z3.eq(gd.guarded_division_normal_form(quantified), quantified)


def test_applicability_cache_skips_known_negative_dags(monkeypatch):
    cache = gd.GuardedDivisionApplicabilityCache()
    x = z3.Int("cache_x")
    original = x / 4 >= 0  # constant divisor: nothing to certify

    def unexpected_scan(*args, **kwargs):
        raise AssertionError("negative applicability must skip normalization")

    monkeypatch.setattr(gd, "_guarded_division_normal_form", unexpected_scan)
    for _ in range(2):
        assert z3.eq(
            gd.guarded_division_normal_form(original, applicability_cache=cache),
            original,
        )
    # A positive answer still runs the query-local certificate every time.
    monkeypatch.undo()
    masked, *_ = _grouped()
    base, *_ = _grouped(masked=False)
    assert not z3.eq(
        gd.guarded_division_normal_form(masked, applicability_cache=cache), masked
    )
    rewritten = gd.guarded_division_normal_form(base, applicability_cache=cache)
    assert _check(z3.Xor(base, rewritten))[0] == z3.unsat
