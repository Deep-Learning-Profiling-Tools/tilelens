import pytest
import torch
import triton
import triton.language as tl
from z3 import substitute

import tilelens
from tilelens.clients import RaceDetector
from tilelens.clients.race_detector.race_detector import (
    SymbolicRaceDetector,
    NullRaceDetector,
)
from tilelens.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)
from tilelens.core.config import config as cfg


# ======== Factory Test ========


def test_race_detector_factory_toggle():
    saved = cfg.enable_race_detector
    try:
        cfg.enable_race_detector = True
        assert isinstance(RaceDetector(), SymbolicRaceDetector)

        cfg.enable_race_detector = False
        assert isinstance(RaceDetector(), NullRaceDetector)
    finally:
        cfg.enable_race_detector = saved


def test_compile_mode_is_not_available_yet(monkeypatch):
    monkeypatch.setattr(cfg, "enable_race_detector", True)
    for make in (RaceDetector, SymbolicRaceDetector):
        with pytest.raises(NotImplementedError, match="compiled race detector"):
            make(compile=True)


@triton.jit
def _dispatch_kernel(x_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(x_ptr + offs, tl.load(x_ptr + offs))


# ======== Flag-off escape hatch ========


def test_flag_off_returns_raw_kernel():
    """With ENABLE_RACE_DETECTOR=0 a string-dispatched trace decorator must
    leave the kernel uninstrumented so opting in has literally zero impact
    when the flag is off."""
    saved = cfg.enable_race_detector
    try:
        cfg.enable_race_detector = False
        traced = tilelens.trace("race_detector")(_dispatch_kernel)

        # Should be the raw JIT kernel, not a TritonTrace wrapper.
        from tilelens.core.trace import TritonTrace

        assert not isinstance(traced, TritonTrace)
        assert traced is _dispatch_kernel
    finally:
        cfg.enable_race_detector = saved


def test_flag_off_returns_raw_kernel_for_factory_instance():
    """ENABLE_RACE_DETECTOR=0 + trace(client=RaceDetector()) must take the
    flag-off fast path. The factory's ``__new__`` already returned a
    NullRaceDetector; the trace decorator must recognize that and leave the
    kernel untraced, otherwise it would wrap the kernel and then crash at
    callback-registration time with NullSymbolicClient's raising methods.

    Identity-check alone is sufficient to prove the fix: without it, the
    predicate would miss the factory-returned instance and ``traced`` would
    be a ``TritonTrace`` wrapper.
    """
    saved = cfg.enable_race_detector
    try:
        cfg.enable_race_detector = False
        # The factory call happens after the flag flip, so __new__ dispatches
        # to NullRaceDetector.
        traced = tilelens.trace(client=RaceDetector())(_dispatch_kernel)

        from tilelens.core.trace import TritonTrace

        assert not isinstance(traced, TritonTrace)
        assert traced is _dispatch_kernel
    finally:
        cfg.enable_race_detector = saved


# ======== Solver Contract ========


@triton.jit
def _pid_store_kernel(x_ptr):
    tl.store(x_ptr + tl.program_id(0), 1)


def test_solver_renames_the_captured_pid_per_copy():
    """The two-copy solver renames SymbolicExpr's pid vars per copy; a
    captured pid-dependent address must change under that rename, or both
    copies would be one program and every cross-program race would be
    missed."""
    detector = SymbolicRaceDetector()
    tilelens.trace(client=detector)(_pid_store_kernel)[(2,)](torch.zeros(2))
    (record,) = detector.records
    solver = TwoCopySymbolicHBSolver(detector.records, grid=(2, 1, 1))
    for ctx in (solver.ctx_a, solver.ctx_b):
        renamed = substitute(record.addr_expr, *ctx.pid_substitutions)
        assert not renamed.eq(record.addr_expr)
        assert f"pid_{ctx.label}_0" in str(renamed)
