"""CPU-only tests of the traced-launch lifecycle."""

import types

import pytest
import torch
import triton
import triton.language as tl

import tilelens
from tilelens.core.callbacks import ForLoopCallbacks, OpCallbacks
from tilelens.core.client import Client
from tilelens.core.trace import _unwrapped_trace_globals, launches


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


class _OtherEagerClient(_EagerClient):
    NAME = "other_eager"


@pytest.mark.parametrize("user_post_hook", [False, True])
def test_autotune_over_heuristics_runs_on_copies_of_the_users_chain(user_post_hook):
    posts = []

    @triton.autotune(
        configs=[triton.Config({"BLOCK": 4}), triton.Config({"BLOCK": 8})],
        key=["n"],
        restore_value=["out_ptr"],
        # Triton's own restore_value post_hook, or a user's that closes over
        # just `posts` and must stay bound to it.
        post_hook=(lambda nargs, exception: posts.append(exception))
        if user_post_hook
        else None,
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
    assert posts == ([None, None] if user_post_hook else [])


def test_a_heuristics_kernel_launches_through_its_voted_warmup(monkeypatch):
    @triton.heuristics({"BLOCK": lambda args: 8})
    @triton.jit
    def heur_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, tl.load(x_ptr + offs) + 1)

    client = _EagerClient(warmup_vote=True)
    traced = tilelens.trace(client)(heur_kernel)
    compiles = []

    def run(*args, grid, warmup, BLOCK, **kwargs):
        compiles.append((warmup, BLOCK))

    monkeypatch.setattr(traced.jit_fn, "run", run)
    x, out = torch.arange(8, dtype=torch.float32), torch.zeros(8)

    traced[(1,)](x, out)

    torch.testing.assert_close(out, x + 1)
    # One voted compile, with the heuristic's constexpr filled in.
    assert compiles == [(True, 8)]
    assert client.calls == ["pre_warmup", ("post_warmup", None)]


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


def _unwrap_leaf(x):
    return x + 1


def _unwrap_helper(x):
    return _unwrap_traced_leaf(x)  # noqa: F821


def test_unwrapped_trace_globals_swaps_only_what_the_kernel_reaches(monkeypatch):
    leaf, helper, unrelated = (
        tilelens.trace(_OtherEagerClient())(triton.jit(fn))
        for fn in (_unwrap_leaf, _unwrap_helper, _unwrap_leaf)
    )
    monkeypatch.setitem(globals(), "_unwrap_traced_leaf", leaf)
    pkg = types.ModuleType("tilelens_test_pkg")
    pkg.api = types.ModuleType("tilelens_test_pkg.api")
    pkg.api.helper = helper
    kernel_globals = {"helper": helper, "pkg": pkg, "unrelated": unrelated}
    exec("def kernel_fn():\n    return helper, pkg.api.helper\n", kernel_globals)

    with pytest.raises(KeyError):
        with _unwrapped_trace_globals(kernel_globals["kernel_fn"]):
            # Direct, through a two-level module path, and transitively
            # through the traced helper's own globals.
            assert kernel_globals["helper"] is helper.jit_fn
            assert pkg.api.helper is helper.jit_fn
            assert globals()["_unwrap_traced_leaf"] is leaf.jit_fn
            # Not reachable from the kernel's code: left alone.
            assert kernel_globals["unrelated"] is unrelated
            raise KeyError("restore on error")

    assert kernel_globals["helper"] is helper and pkg.api.helper is helper
    assert globals()["_unwrap_traced_leaf"] is leaf


def test_a_traced_function_called_outside_a_kernel_leaves_the_language_alone():
    helper = tilelens.trace(_OtherEagerClient())(triton.jit(_unwrap_leaf))
    program_id = tl.program_id

    # E.g. a real compile reaching the trace as a plain Python callee.
    with pytest.raises(RuntimeError, match="outside of the scope of a kernel"):
        helper(1)

    assert tl.program_id is program_id


def test_a_voted_compile_sees_traced_helpers_as_jit_functions(monkeypatch):
    # As under the CLI wrappers, the device function is itself traced.
    helper = tilelens.trace(_EagerClient())(triton.jit(_unwrap_leaf))
    monkeypatch.setitem(globals(), "_codegen_helper", helper)

    @triton.jit
    def kernel(x_ptr, FN: tl.constexpr, FNS: tl.constexpr):
        tl.store(x_ptr, FN(_codegen_helper(tl.load(x_ptr))))  # noqa: F821

    traced = tilelens.trace(_EagerClient(warmup_vote=True))(kernel)
    seen = []

    def run(x, FN, *, grid, warmup, FNS):
        # What the compile resolves: the constexpr arguments and the global.
        seen.append((FN, FNS, globals()["_codegen_helper"]))

    monkeypatch.setattr(traced.jit_fn, "run", run)

    # FN by position, FNS by keyword inside a tuple.
    traced.warmup(torch.zeros(1), helper, FNS=(helper,), grid=(1,))

    assert seen == [(helper.jit_fn, (helper.jit_fn,), helper.jit_fn)]
