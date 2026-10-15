"""CPU-only tests of the traced-launch lifecycle."""

import pytest
import torch
import triton
import triton.language as tl

import tilelens
from tilelens.clients.profiler.profiler import Profiler
from tilelens.core.trace import launches


@pytest.fixture(autouse=True)
def _real_jit():
    # tests/unit/test_multithreading.py sets TRITON_INTERPRET=1 at import
    # time; these tests need @triton.jit to build real JITFunctions.
    from triton import knobs

    with knobs.runtime.scope():
        knobs.runtime.interpret = False
        yield


def test_autotune_over_heuristics_runs_on_copies_of_the_users_chain():
    @triton.autotune(
        configs=[triton.Config({"BLOCK": 4}), triton.Config({"BLOCK": 8})],
        key=["n"],
        cache_results=True,
    )
    @triton.heuristics({"EVEN": lambda args: args["n"] % args["BLOCK"] == 0})
    @triton.jit
    def user(x_ptr, out_ptr, n, BLOCK: tl.constexpr, EVEN: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1, mask=mask)

    # A benchmarker the user's autotuner cached (do_bench is a cached_property).
    user.__dict__["do_bench"] = None
    before = dict(vars(user))
    x, out = torch.arange(8, dtype=torch.float32), torch.zeros(8)

    tilelens.trace("tracer")(user)[(2,)](x, out, 8)

    torch.testing.assert_close(out, x + 1)
    assert vars(user) == before and user.cache == {}


def test_a_launch_does_not_warm_up(monkeypatch):
    @triton.jit
    def add_one(x_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(x_ptr + offs, tl.load(x_ptr + offs) + 1)

    traced = tilelens.trace("tracer")(add_one)
    # Only the profiler compiles each launch.
    monkeypatch.setattr(traced, "warmup", lambda *a, **k: pytest.fail("warmed up"))

    traced[(1,)](torch.zeros(8), BLOCK=8)


def test_a_profiler_compiles_each_launch(monkeypatch):
    @triton.heuristics({"BLOCK": lambda args: args["x_ptr"].numel()})
    @triton.jit
    def heur_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, tl.load(x_ptr + offs) + 1)

    traced = tilelens.trace(Profiler())(heur_kernel)
    compiles = []

    def run(*args, grid, warmup, BLOCK, **kwargs):
        compiles.append((warmup, BLOCK))

    monkeypatch.setattr(traced.jit_fn, "run", run)

    traced[(1,)](torch.arange(8, dtype=torch.float32), torch.zeros(8))

    assert compiles == [(True, 8)]


def test_each_traced_launch_is_recorded_separately():
    @triton.jit
    def add_one(x_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(x_ptr + offs, tl.load(x_ptr + offs) + 1, mask=offs < n)

    traced = tilelens.trace("tracer")(add_one)
    before = len(launches)
    x = torch.zeros(8)

    traced[(1,)](x, 8, BLOCK=8)
    traced[(2,)](x, 8, BLOCK=4)

    assert [launch.grid for launch in launches[before:]] == [(1, 1, 1), (2, 1, 1)]
