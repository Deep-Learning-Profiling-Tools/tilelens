"""Hand-built AccessEventRecords for the two-copy solver's unit tests."""

from __future__ import annotations

import torch

from tilelens.clients.race_detector.data import AccessEventRecord
from tilelens.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)
from tilelens.core.data import AtomicCas, AtomicRMW, Load, Store


def store(addr, *, event_id, program_seq=0, elem_size=4, mask=None, active=True):
    return AccessEventRecord(
        op_type=Store,
        access_mode="write",
        addr_expr=addr,
        local_constraints=() if mask is None else (mask,),
        active=active,
        reads=False,
        writes=True,
        event_id=event_id,
        program_seq=program_seq,
        elem_size=elem_size,
    )


def load(addr, *, event_id, program_seq=0, elem_size=4, mask=None, active=True):
    return AccessEventRecord(
        op_type=Load,
        access_mode="read",
        addr_expr=addr,
        local_constraints=() if mask is None else (mask,),
        active=active,
        reads=True,
        writes=False,
        event_id=event_id,
        program_seq=program_seq,
        elem_size=elem_size,
    )


def cas(
    addr,
    cmp_value,
    new_value,
    old_value,
    *,
    event_id,
    program_seq=0,
    sem="acq_rel",
    scope="gpu",
    elem_size=4,
    tensor=None,
    active=True,
):
    return AccessEventRecord(
        op_type=AtomicCas,
        access_mode="read",
        tensor=tensor,
        addr_expr=addr,
        active=active,
        reads=True,
        writes=None,
        is_atomic=True,
        atomic_kind="cas",
        sem=sem,
        scope=scope,
        old_value=old_value,
        cas_cmp_value=cmp_value,
        cas_new_value=new_value,
        event_id=event_id,
        program_seq=program_seq,
        elem_size=elem_size,
        copy_local_vars=(old_value,),
    )


def rmw(
    addr,
    old=None,
    *,
    op=None,
    operand=None,
    event_id,
    program_seq=0,
    sem="acq_rel",
    scope="gpu",
    elem_size=4,
    tensor=None,
    active=True,
):
    """An atomic RMW; value-modeled when ``old`` (its observation var) is
    given, exactly like the CAS return."""
    return AccessEventRecord(
        op_type=AtomicRMW,
        access_mode="read",
        tensor=tensor,
        addr_expr=addr,
        active=active,
        reads=True,
        writes=True,
        is_atomic=True,
        atomic_kind="rmw",
        sem=sem,
        scope=scope,
        old_value=old,
        rmw_op=op,
        rmw_operand=operand,
        event_id=event_id,
        program_seq=program_seq,
        elem_size=elem_size,
        copy_local_vars=() if old is None else (old,),
    )


def counter(value=0):
    return torch.full((1,), value, dtype=torch.int32)


def fences_at(*positions):
    """The eager capture's fence rule: a fence recorded at program_seq
    position ``f`` orders the two records exactly when ``f`` lies strictly
    between their positions."""

    def fence_between(first, second):
        return any(first.program_seq < f < second.program_seq for f in positions)

    return fence_between


def every_fence(first, second):
    """A fence between every two operations: source order is HB order."""
    return True


def solve(records, *, grid=(2, 1, 1), arange_dict=None, **kwargs):
    return TwoCopySymbolicHBSolver(
        records, grid=grid, arange_dict=arange_dict or {}, **kwargs
    )


def report_pairs(reports):
    return {(r.first.event_id, r.second.event_id) for r in reports}
