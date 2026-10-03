"""CPU-only tests of the traced-launch lifecycle."""

import types
from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl
from triton.runtime import Autotuner
from triton.runtime.autotuner import Heuristics
from triton.runtime.interpreter import InterpretedFunction

import tilelens
from tilelens.core.callbacks import ForLoopCallbacks, OpCallbacks
from tilelens.core.client import Client, ClientManager
from tilelens.core.frontend.base import LANG_PATCH_SCOPES, get_frontend
from tilelens.core.trace import (
    KernelTraceSupport,
    TritonTrace,
    _untraced_call_args,
    _unwrapped_trace_globals,
    launches,
)


@pytest.fixture(autouse=True)
def _real_jit(monkeypatch):
    # tests/unit/test_multithreading.py sets TRITON_INTERPRET=1 at import time,
    # and a traced launch's patch scope restores knobs.runtime.interpret as an
    # explicit override. These tests need @triton.jit to build real
    # JITFunctions, so pin the knob off and put back exactly what was there.
    from triton import knobs

    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    missing = object()
    previous = knobs.runtime.__dict__.get("interpret", missing)
    knobs.runtime.__dict__["interpret"] = False
    yield
    if previous is missing:
        knobs.runtime.__dict__.pop("interpret", None)
    else:
        knobs.runtime.__dict__["interpret"] = previous


# ======== Fake clients =========


class _EagerClient(Client):
    """Interpreting client that records its warmup and finalize callbacks."""

    NAME = "eager"

    def __init__(self, *, warmup_vote=False, loop_overrider=None, records=()):
        super().__init__()
        self.calls: list = []
        self.warmup_vote = warmup_vote
        self.loop_overrider = loop_overrider
        self.records = list(records)

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
        return ForLoopCallbacks(loop_iter_overrider=self.loop_overrider)

    def finalize(self):
        self.calls.append("finalize")
        return list(self.records)

    def pre_warmup_callback(self, jit_fn, *args, **kwargs):
        self.calls.append("pre_warmup")
        return self.warmup_vote

    def post_warmup_callback(self, jit_fn, ret):
        self.calls.append(("post_warmup", ret))


class _OtherEagerClient(_EagerClient):
    NAME = "other_eager"


# ======== Fake compile =========


class _FakeKernel:
    """What a fake warmup compile returns."""


def _install_fake_run(monkeypatch, jit_fn, run):
    """Install ``run(*args, grid, warmup, **kwargs)`` on a real JITFunction,
    so its warmup compiles (warmup=True) never reach a device."""
    monkeypatch.setattr(jit_fn, "run", run, raising=False)


@pytest.fixture
def fake_compile(monkeypatch):
    """Record a real JITFunction's warmup compiles instead of running them."""

    def install(jit_fn):
        calls: list[SimpleNamespace] = []

        def run(*args, grid, warmup, **kwargs):
            calls.append(
                SimpleNamespace(args=args, grid=grid, warmup=warmup, kwargs=kwargs)
            )
            return _FakeKernel()

        _install_fake_run(monkeypatch, jit_fn, run)
        return calls

    return install


def _make_plain_kernel():
    @triton.jit
    def add_one(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1, mask=mask)

    return add_one


def _make_autotuned_kernel(**autotune_kwargs):
    @triton.autotune(
        configs=[triton.Config({"BLOCK": 4}), triton.Config({"BLOCK": 8})],
        key=["n"],
        **autotune_kwargs,
    )
    @triton.heuristics({"EVEN": lambda args: args["n"] % args["BLOCK"] == 0})
    @triton.jit
    def add_one_tuned(x_ptr, out_ptr, n, BLOCK: tl.constexpr, EVEN: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1, mask=mask)

    return add_one_tuned


def _grid8(meta):
    # The interpreter hands grid callables tensor-converted runtime args, so
    # interpreted launches may only read constexprs here.
    return (triton.cdiv(8, meta["BLOCK"]),)


def _dummy_lang_fn():
    """Provides tl globals for patch_lang in patch_run tests."""
    return tl.arange(0, 1)


# ======== patch_warmup =========


class _FakeWarmupJit:
    def __init__(self):
        self.warmups: list[dict] = []

    def warmup(self, *args, **kwargs):
        self.warmups.append(kwargs)
        return "compiled"


def test_patch_warmup_polls_every_client_and_compiles_on_any_vote():
    voter, abstainer = _EagerClient(warmup_vote=True), _OtherEagerClient()
    manager = ClientManager([voter, abstainer])
    jit_fn = _FakeWarmupJit()

    with manager.patch_warmup(jit_fn):
        ret = jit_fn.warmup(1, grid=(1,), warmup=False)

    assert ret == "compiled"
    assert jit_fn.warmups == [{"grid": (1,)}]
    assert voter.calls == ["pre_warmup", ("post_warmup", "compiled")]
    assert abstainer.calls == ["pre_warmup", ("post_warmup", "compiled")]
    assert "warmup" not in vars(jit_fn)


def test_patch_warmup_scopes_closing_out_of_order_leave_nothing_behind():
    # Overlapping scopes on one jit_fn, the first opened closing first, as
    # on two host threads.
    jit_fn = _FakeWarmupJit()
    first = ClientManager([_EagerClient()]).patch_warmup(jit_fn)
    second = ClientManager([_EagerClient(warmup_vote=True)]).patch_warmup(jit_fn)
    first.__enter__()
    second.__enter__()
    first.__exit__(None, None, None)

    # The scope still open votes alone, over the original warmup.
    assert jit_fn.warmup(1, grid=(1,)) == "compiled"

    second.__exit__(None, None, None)
    assert "warmup" not in vars(jit_fn)


# ======== patch_run =========


def _first_op():
    frontend = get_frontend("triton")
    namespace, attrs = next(iter(frontend.namespaces.items()))
    attr = next(iter(attrs))
    return frontend, namespace, attr


def test_patch_run_loop_hook_conflict_leaves_nothing_patched():
    # The sanitizer and the race detector each install a loop-iteration
    # overrider, so tracing a kernel with both refuses here.
    manager = ClientManager(
        [
            _EagerClient(loop_overrider=lambda site, idx: idx),
            _OtherEagerClient(loop_overrider=lambda site, idx: idx),
        ]
    )
    frontend, namespace, attr = _first_op()
    original = frontend.original_ops[namespace][attr]
    scopes_before = len(LANG_PATCH_SCOPES.get("triton", []))

    with pytest.raises(RuntimeError, match="Only one loop_iter overrider"):
        with manager.patch_run(_dummy_lang_fn, frontend_name="triton"):
            pass

    assert getattr(namespace, attr) is original
    assert frontend._patch_calls_scope == 0
    assert not frontend._loop_ast_patched
    assert len(LANG_PATCH_SCOPES.get("triton", [])) == scopes_before
    assert manager._iter_overrider is None


# ======== finalize =========


def test_finalize_runs_every_client_and_reraises_first_exception():
    class _Exiting(_EagerClient):
        NAME = "exiting"

        def finalize(self):
            super().finalize()
            raise SystemExit(3)

    class _Failing(_OtherEagerClient):
        NAME = "failing"

        def finalize(self):
            self.finalized = True
            raise ValueError("second failure")

    record = object()
    exiting, healthy, failing = (
        _Exiting(),
        _OtherEagerClient(records=[record]),
        _Failing(),
    )
    manager = ClientManager([exiting, healthy, failing])

    with pytest.raises(SystemExit) as info:
        manager.finalize()

    assert info.value.code == 3
    assert "finalize" in healthy.calls
    assert failing.finalized
    assert manager.launch.records == [record]


# ======== runner chain rebuild =========


def test_trace_does_not_mutate_the_users_autotuner_chain():
    user = _make_autotuned_kernel(restore_value=["out_ptr"])
    heuristics, jit_fn = user.fn, user.fn.fn
    before = dict(vars(user))
    before_heuristics = dict(vars(heuristics))

    traced = tilelens.trace(_EagerClient())(user)

    assert vars(user).keys() == before.keys()
    assert all(vars(user)[k] is v for k, v in before.items())
    assert all(vars(heuristics)[k] is v for k, v in before_heuristics.items())
    assert vars(heuristics).keys() == before_heuristics.keys()

    # Interpreter chain: copies of both layers over the InterpretedFunction.
    runner = traced.runner
    assert isinstance(runner, Autotuner) and runner is not user
    assert isinstance(runner.fn, Heuristics) and runner.fn is not heuristics
    assert isinstance(runner.fn.fn, InterpretedFunction)
    assert runner.fn.fn is traced.interpreted_fn
    assert runner._do_bench is KernelTraceSupport.dummy_benchmarker

    # Real chain: copies of both layers over the user's JITFunction.
    real = traced.warmup_runner
    assert isinstance(real, Autotuner) and real is not user and real is not runner
    assert isinstance(real.fn, Heuristics) and real.fn is not heuristics
    assert real.fn.fn is jit_fn is traced.jit_fn
    assert real._do_bench is user._do_bench

    # Per-run state is private to each copy.
    assert len({id(user.cache), id(runner.cache), id(real.cache)}) == 3
    assert runner.cache_results is False and real.cache_results is False


def test_rebuilt_autotuner_restore_hooks_bind_to_the_copy():
    user = _make_autotuned_kernel(restore_value=["out_ptr"])
    traced = tilelens.trace(_EagerClient())(user)
    out = torch.ones(2)
    nargs = {"out_ptr": out}

    traced.runner.pre_hook(nargs)
    out.zero_()
    traced.runner.post_hook(nargs, exception=None)

    assert torch.equal(out, torch.ones(2))
    assert "restore_copies" in vars(traced.runner)
    assert "restore_copies" not in vars(user)


def test_interpreter_copy_drops_a_benchmarker_cached_on_the_user_autotuner():
    user = _make_autotuned_kernel()
    sentinel = object()
    user.__dict__["do_bench"] = sentinel

    traced = tilelens.trace(_EagerClient())(user)

    assert traced.runner.do_bench is KernelTraceSupport.dummy_benchmarker
    assert user.do_bench is sentinel


def test_autotune_over_heuristics_interprets_every_layer():
    user = _make_autotuned_kernel()
    traced = tilelens.trace(_EagerClient())(user)
    x = torch.arange(8, dtype=torch.float32)
    out = torch.zeros(8)

    traced[_grid8](x, out, 8)

    torch.testing.assert_close(out, x + 1)
    assert user.cache == {}


def test_a_heuristics_kernel_launches_through_its_voted_warmup(fake_compile):
    @triton.heuristics({"BLOCK": lambda args: 8})
    @triton.jit
    def heur_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, tl.load(x_ptr + offs) + 1)

    client = _EagerClient(warmup_vote=True)
    traced = tilelens.trace(client)(heur_kernel)
    calls = fake_compile(traced.jit_fn)
    x, out = torch.arange(8, dtype=torch.float32), torch.zeros(8)

    traced[(1,)](x, out, 8)

    torch.testing.assert_close(out, x + 1)
    # One voted compile, with the heuristic's constexpr filled in.
    assert [(c.warmup, c.kwargs["BLOCK"]) for c in calls] == [(True, 8)]
    assert client.calls[0] == "pre_warmup"
    assert client.calls[1][0] == "post_warmup"
    assert isinstance(client.calls[1][1], _FakeKernel)


# ======== launch lifecycle =========


def test_each_traced_launch_is_recorded_separately():
    traced = tilelens.trace(_EagerClient(records=["record"]))(_make_plain_kernel())
    before = len(launches)
    a, b = torch.zeros(8), torch.zeros(16)

    traced[(2,)](a, a, 8, BLOCK=4)
    traced[(4,)](b, b, 16, BLOCK=4)

    first, second = launches[before:]
    assert first is not second
    assert (first.grid, second.grid) == ((2, 1, 1), (4, 1, 1))


# ======== unwrapped trace globals =========


def _unwrap_leaf(x):
    return x + 1


def _unwrap_helper(x):
    return _unwrap_traced_leaf(x)  # noqa: F821


def test_unwrapped_trace_globals_swaps_only_what_the_kernel_reaches():
    module_globals = globals()
    leaf = tilelens.trace(_OtherEagerClient())(triton.jit(_unwrap_leaf))
    helper = tilelens.trace(_OtherEagerClient())(triton.jit(_unwrap_helper))
    unrelated = tilelens.trace(_OtherEagerClient())(_make_plain_kernel())
    # A package whose `api` re-exports `impl`'s binding.
    pkg = types.ModuleType("tilelens_test_pkg")
    pkg.api = types.ModuleType("tilelens_test_pkg.api")
    pkg.impl = types.ModuleType("tilelens_test_pkg.impl")
    pkg.api.helper = pkg.impl.helper = helper
    kernel_globals = {"helper": helper, "pkg": pkg, "unrelated": unrelated, "keep": 1}
    exec("def kernel_fn():\n    return helper, pkg.api.helper\n", kernel_globals)
    module_globals["_unwrap_traced_leaf"] = leaf
    module_globals["_unwrap_traced_unrelated"] = unrelated

    try:
        with pytest.raises(KeyError):
            with _unwrapped_trace_globals(kernel_globals["kernel_fn"]):
                # Direct, through a two-level module path, and transitively
                # through the traced helper's own globals.
                assert kernel_globals["helper"] is helper.jit_fn
                assert pkg.api.helper is helper.jit_fn
                assert module_globals["_unwrap_traced_leaf"] is leaf.jit_fn
                # Not reachable from the kernel's code: left alone.
                assert pkg.impl.helper is helper
                assert kernel_globals["unrelated"] is unrelated
                assert module_globals["_unwrap_traced_unrelated"] is unrelated
                assert kernel_globals["keep"] == 1
                raise KeyError("restore on error")

        assert kernel_globals["helper"] is helper
        assert pkg.api.helper is helper
        assert module_globals["_unwrap_traced_leaf"] is leaf
    finally:
        module_globals.pop("_unwrap_traced_leaf", None)
        module_globals.pop("_unwrap_traced_unrelated", None)


def test_unwrapped_trace_globals_covers_names_bound_to_traced_defaults():
    # Triton's dependency walker resolves a parameter default expression
    # (``FN=helper``) in the kernel's globals; the name is not in co_names.
    helper = tilelens.trace(_OtherEagerClient())(triton.jit(_unwrap_leaf))
    kernel_globals = {"helper": helper, "alias": helper, "other": helper.jit_fn}
    exec("def kernel_fn(FN=helper):\n    return FN\n", kernel_globals)

    with _unwrapped_trace_globals(kernel_globals["kernel_fn"]):
        assert kernel_globals["helper"] is helper.jit_fn
        assert kernel_globals["alias"] is helper.jit_fn
    assert kernel_globals["helper"] is kernel_globals["alias"] is helper


def test_untraced_call_args_unwraps_arguments_tuples_and_defaults():
    helper = tilelens.trace(_OtherEagerClient())(triton.jit(_unwrap_leaf))
    raw = triton.jit(_unwrap_leaf)
    # A trace without a JITFunction has nothing to unwrap to.
    no_jit = TritonTrace(InterpretedFunction(_unwrap_leaf), _OtherEagerClient())

    @triton.jit
    def kernel(
        x_ptr,
        FN: tl.constexpr,
        FNS: tl.constexpr,
        ACT: tl.constexpr = helper,
        N: tl.constexpr = 1,
    ):
        pass

    x = torch.zeros(1)
    args, kwargs = _untraced_call_args(
        kernel, (x, helper), {"FNS": (raw, helper), "num_warps": 4}
    )
    assert args[0] is x and args[1] is helper.jit_fn
    assert kwargs["FNS"][0] is raw and kwargs["FNS"][1] is helper.jit_fn
    # The traced default is passed explicitly; plain defaults are left alone.
    assert kwargs["ACT"] is helper.jit_fn
    assert "N" not in kwargs and kwargs["num_warps"] == 4

    fns = (raw, no_jit)
    args, kwargs = _untraced_call_args(kernel, (x, no_jit, fns, raw), {})
    assert args[1] is no_jit and args[2] is fns and args[3] is raw
    assert kwargs == {}


def test_a_traced_function_refuses_to_run_outside_an_interpreted_launch():
    helper = tilelens.trace(_OtherEagerClient())(triton.jit(_unwrap_leaf))
    program_id = tl.program_id

    # E.g. a real compile reaching the trace as a plain Python callee.
    with pytest.raises(TypeError, match="outside a traced launch's interpreter"):
        helper(1)

    # The interpreter never ran, so triton.language was never patched.
    assert tl.program_id is program_id


def test_tritons_interpreter_runs_a_traced_helper_untraced(monkeypatch):
    from triton import knobs

    program_id = tl.program_id
    x, out = torch.arange(8, dtype=torch.float32), torch.zeros(8)
    with knobs.runtime.scope():
        knobs.runtime.interpret = True
        helper = tilelens.trace(_OtherEagerClient())(triton.jit(_unwrap_leaf))
        monkeypatch.setitem(globals(), "_interp_helper", helper)

        @triton.jit
        def kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
            offs = tl.arange(0, BLOCK)
            tl.store(out_ptr + offs, _interp_helper(tl.load(x_ptr + offs)))  # noqa: F821

        assert isinstance(kernel, InterpretedFunction)
        kernel[(1,)](x, out, BLOCK=8)

    torch.testing.assert_close(out, x + 1)
    assert tl.program_id is program_id


def _codegen(jit_fn, bound):
    """Run the part of a real compile that resolves callees on ``jit_fn``,
    for a fixed target and with no device: the cache key's dependency walk
    (triton.compile keys the kernel first), then the code generator."""
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.compiler.compiler import make_backend

    signature, constexprs = {}, {}
    for index, param in enumerate(jit_fn.params):
        if param.is_constexpr:
            signature[param.name] = "constexpr"
            constexprs[(index,)] = bound[param.name]
        else:
            signature[param.name] = "*fp32"
    src = ASTSource(fn=jit_fn, signature=signature, constexprs=constexprs)
    src.hash()
    target = GPUTarget("cuda", 80, 32)
    backend = make_backend(target)
    options = backend.parse_options({})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    codegen = backend.get_codegen_implementation(options)
    return src.make_ir(target, options, codegen, backend.get_module_map(), context)


@pytest.mark.parametrize("reach", ["global", "argument"])
def test_a_voted_compile_sees_traced_helpers_as_jit_functions(monkeypatch, reach):
    # As under the CLI wrappers, the device function is itself traced.
    helper = tilelens.trace(_EagerClient())(triton.jit(_unwrap_leaf))
    monkeypatch.setitem(globals(), "_codegen_helper", helper)

    @triton.jit
    def by_name(x_ptr, out_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, _codegen_helper(tl.load(x_ptr + offs)))  # noqa: F821

    @triton.jit
    def by_argument(x_ptr, out_ptr, FN: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, FN(tl.load(x_ptr + offs)))

    if reach == "global":
        kernel, extra = by_name, {}
    else:
        kernel, extra = by_argument, {"FN": helper}
    traced = tilelens.trace(_EagerClient(warmup_vote=True))(kernel)
    compiled = []

    def run(*args, grid, warmup, **kwargs):
        assert warmup
        bound = {**dict(zip(traced.jit_fn.arg_names, args)), **kwargs}
        compiled.append(str(_codegen(traced.jit_fn, bound)))

    _install_fake_run(monkeypatch, traced.jit_fn, run)
    program_id = tl.program_id
    x, out = torch.arange(8, dtype=torch.float32), torch.zeros(8)

    traced[(1,)](x, out, BLOCK=8, **extra)

    assert len(compiled) == 1 and "_unwrap_leaf" in compiled[0]
    torch.testing.assert_close(out, x + 1)
    assert tl.program_id is program_id
    assert globals()["_codegen_helper"] is helper
