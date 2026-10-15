"""CPU-only tests of the traced-launch lifecycle."""

import pytest
import torch
import triton
import triton.language as tl

import tilelens
from tilelens.core.callbacks import ForLoopCallbacks, OpCallbacks
from tilelens.core.client import Client
from tilelens.core.trace import launches


@pytest.fixture(autouse=True)
def _real_jit():
    # tests/unit/test_multithreading.py sets TRITON_INTERPRET=1 at import
    # time; these tests need @triton.jit to build real JITFunctions.
    from triton import knobs

    with knobs.runtime.scope():
        knobs.runtime.interpret = False
        yield


class _EagerClient(Client):
    """Interpreting client that records its warmup callbacks."""

    NAME = "eager"

    def __init__(self, *, warmup_vote=False):
        super().__init__()
        self.calls: list = []
        self.warmup_vote = warmup_vote

    def pre_run_callback(self, fn):
        return True

    def post_run_callback(self, fn):
        return True

    def arg_callback(self, name, arg, arg_cvt):
        pass

    def grid_callback(self, grid):
        pass

    def grid_idx_callback(self, grid_idx):
        pass

    def register_op_callback(self, op_type, *args, **kwargs):
        return OpCallbacks()

    def register_for_loop_callback(self):
        return ForLoopCallbacks()

    def finalize(self):
        return []

    def pre_warmup_callback(self, jit_fn, *args, **kwargs):
        self.calls.append("pre_warmup")
        return self.warmup_vote

    def post_warmup_callback(self, jit_fn, ret):
        self.calls.append(("post_warmup", ret))


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

    tilelens.trace(_EagerClient())(user)[(2,)](x, out, 8)

    torch.testing.assert_close(out, x + 1)
    assert vars(user) == before and user.cache == {}


def test_a_launch_does_not_warm_up():
    @triton.heuristics({"BLOCK": lambda args: 8})
    @triton.jit
    def heur_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, tl.load(x_ptr + offs) + 1)

    # Even with a client that would vote for a real compile.
    client = _EagerClient(warmup_vote=True)
    traced = tilelens.trace(client)(heur_kernel)
    x, out = torch.arange(8, dtype=torch.float32), torch.zeros(8)

    traced[(1,)](x, out)

    torch.testing.assert_close(out, x + 1)
    assert client.calls == []


def test_each_traced_launch_is_recorded_separately():
    @triton.jit
    def add_one(x_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(x_ptr + offs, tl.load(x_ptr + offs) + 1, mask=offs < n)

    traced = tilelens.trace(_EagerClient())(add_one)
    before = len(launches)
    x = torch.zeros(8)

    traced[(1,)](x, 8, BLOCK=8)
    traced[(2,)](x, 8, BLOCK=4)

    assert [launch.grid for launch in launches[before:]] == [(1, 1, 1), (2, 1, 1)]


def _add_one(x):
    return x + 1


def test_a_traced_function_called_outside_a_kernel_leaves_the_language_alone():
    helper = tilelens.trace(_EagerClient())(triton.jit(_add_one))
    program_id = tl.program_id

    # E.g. a real compile reaching the trace as a plain Python callee.
    with pytest.raises(RuntimeError, match="outside of the scope of a kernel"):
        helper(1)

    assert tl.program_id is program_id
