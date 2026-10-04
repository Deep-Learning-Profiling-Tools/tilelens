"""CPU-only tests of the core IR lifecycle: client declarations, ClientManager
dispatch rules, the ``ir_capture`` run wrapper and TritonTrace's runner handling.

The core compiles IR kernels on the host (tilelens.core.host_compile).
These tests pin call sequences, so a fake stands in for that compile: a
``_FakeJit`` compiles through its ``fake_compile``, and a real JITFunction
gets one from ``fake_compile(jit_fn)`` (``_install_fake_run``); a host
compile nothing faked fails the test. The real host compile is tested in
tests/unit/ir/test_host_compile.py, and end to end under a toy IR client in
tests/end_to_end/test_ir_client.py.
"""
import ast
import gc
import importlib
import inspect
import re
import threading
import weakref
from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl
from triton.compiler.errors import CompileTimeAssertionFailure
from triton.runtime.interpreter import InterpretedFunction

import tilelens
from tilelens.core.callbacks import ForLoopCallbacks, OpCallbacks
from tilelens.core.client import (
    Client,
    ClientManager,
    LanguagePatchedError,
    LaunchCall,
    LaunchEvent,
    _resolve_grid,
)
from tilelens.core.config import DEFAULT_IR_TARGET, config as tilelens_config
from tilelens.core.data import Store
from tilelens.core.frontend.base import LANG_PATCH_SCOPES, get_frontend
from tilelens.core.host_compile import HostCompiler, default_ir_target
from tilelens.core.trace import (
    GluonTrace,
    NKITrace,
    TraceInterface,
    TritonTrace,
)

# `tilelens.core.trace` the attribute is the trace() decorator; the module
# holds the `launches` list.
trace_module = importlib.import_module("tilelens.core.trace")


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


@pytest.fixture(autouse=True)
def _default_ir_target(monkeypatch):
    # Whatever TILELENS_IR_TARGET the caller has set.
    monkeypatch.setattr(tilelens_config, "ir_target", DEFAULT_IR_TARGET)


@pytest.fixture(autouse=True)
def _fake_host_compile(monkeypatch):
    """Route the core's host compile to the jit_fn's ``fake_compile`` (see
    the module docstring); every compile is recorded in ``compiles`` as
    (jit_fn, target)."""
    compiles: list[tuple] = []

    def compile(self, jit_fn, args, kwargs, *, target):
        fake = getattr(jit_fn, "fake_compile", None)
        assert fake is not None, f"unexpected host compile of {jit_fn!r}"
        compiles.append((jit_fn, target))
        return fake(*args, **kwargs)

    monkeypatch.setattr(HostCompiler, "compile", compile)
    return compiles


# ======== Fake clients =========


class _EagerClient(Client):
    """Interpreting client that records every callback it receives."""

    NAME = "eager"

    def __init__(self, *, warmup_vote=False, loop_overrider=None, records=()):
        super().__init__()
        self.calls: list = []
        self.stores = 0
        self.warmup_vote = warmup_vote
        self.loop_overrider = loop_overrider
        self.records = list(records)
        self.on_store = self._on_store

    def _on_store(self, *args, **kwargs):
        self.stores += 1

    def pre_run_callback(self, fn):
        self.calls.append("pre_run")
        return True

    def post_run_callback(self, fn):
        self.calls.append("post_run")
        return True

    def arg_callback(self, name, arg, arg_cvt):
        self.calls.append(("arg", name))

    def grid_callback(self, grid):
        self.calls.append(("grid", grid))

    def grid_idx_callback(self, grid_idx):
        self.calls.append("grid_idx")

    def register_op_callback(self, op_type, *args, **kwargs):
        if op_type is Store:
            return OpCallbacks(before_callback=self.on_store)
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

    def begin_launch(self, call):
        self.calls.append("begin")

    def abort_launch(self, exc):
        self.calls.append(("abort", type(exc)))

    def before_launch(self, event):
        self.calls.append("before_launch")


class _SiblingEagerClient(Client):
    """A second interpreting client class, unrelated to _EagerClient."""

    NAME = "sibling_eager"

    def __init__(self, loop_overrider=None):
        super().__init__()
        self.loop_overrider = loop_overrider

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
        return []

    def pre_warmup_callback(self, jit_fn, *args, **kwargs):
        return False

    def post_warmup_callback(self, jit_fn, ret):
        pass


class _IRClient(Client):
    """IR client: records lifecycle hooks; interpreter hooks must never fire."""

    NEEDS_INTERPRETER = False
    IR_STAGES = frozenset({"ttir"})
    LAUNCH = "skip"

    def __init__(self, log=None, *, raise_in_before=None, records=()):
        super().__init__()
        self.log = [] if log is None else log
        self.events: list[LaunchEvent] = []
        self.failures: list[LaunchEvent] = []
        self.finalized: list[list[LaunchEvent]] = []
        self.launch_calls: list[LaunchCall] = []
        self.raise_in_before = raise_in_before
        self.records = list(records)

    def begin_launch(self, call):
        self.log.append("begin")
        self.launch_calls.append(call)
        self.events = []
        self.failures = []

    def abort_launch(self, exc):
        self.log.append(("abort", type(exc)))

    def before_launch(self, event):
        self.log.append("before")
        if self.raise_in_before is not None:
            raise self.raise_in_before
        self.events.append(event)

    def after_launch(self, event):
        self.log.append("after")

    def compile_failed(self, event):
        self.log.append(("compile_failed", type(event.error)))
        self.failures.append(event)

    def finalize(self):
        self.log.append("finalize")
        self.finalized.append(list(self.events))
        return list(self.records)

    def pre_warmup_callback(self, jit_fn, *args, **kwargs):
        self.log.append("pre_warmup")
        return False

    def post_warmup_callback(self, jit_fn, ret):
        self.log.append("post_warmup")

    def _unreachable(self, *args, **kwargs):
        raise AssertionError(f"interpreter hook reached IR client {self.NAME}")

    pre_run_callback = _unreachable
    post_run_callback = _unreachable
    arg_callback = _unreachable
    grid_callback = _unreachable
    grid_idx_callback = _unreachable
    register_op_callback = _unreachable
    register_for_loop_callback = _unreachable


class _SkipIRClient(_IRClient):
    NAME = "ir_skip"


class _PeerIRClient(_IRClient):
    """A second IR client class, unrelated to _SkipIRClient."""

    NAME = "ir_peer"


# ======== Fake compile =========


class _FakeKernel:
    def __init__(self, key):
        self.hash = f"hash-{key}"
        self.asm = {"ttir": f"// ttir {key}"}

    def _init_handles(self):
        # A CompiledKernel loads its binary here; IR mode never does.
        raise AssertionError("IR mode loaded a kernel")


def _fake_kernel_signature(x_ptr, n, BLOCK=4):
    pass


class _FakeJit:
    """Stands in for a JITFunction: the host compile calls fake_compile; its
    ``run``, the real launch, must never be reached."""

    signature = inspect.signature(_fake_kernel_signature)

    def __init__(self, log=None, *, compile_error=None):
        self.log = [] if log is None else log
        self.compile_error = compile_error

    def fake_compile(self, *args, **kwargs):
        self.log.append("compile")
        if self.compile_error is not None:
            raise self.compile_error
        return _FakeKernel(kwargs.get("BLOCK", 4))

    def run(self, *args, grid, warmup, **kwargs):
        raise AssertionError("a traced launch with IR clients ran the kernel")


def _install_fake_run(monkeypatch, jit_fn, run):
    """Install ``run(*args, grid, warmup, **kwargs)`` on a real JITFunction
    as both its ``run`` (what a voted warmup compiles through) and its fake
    host compile (warmup=True, grid=None: a host compile needs no grid)."""
    monkeypatch.setattr(jit_fn, "run", run, raising=False)
    monkeypatch.setattr(
        jit_fn,
        "fake_compile",
        lambda *args, **kwargs: run(*args, grid=None, warmup=True, **kwargs),
        raising=False,
    )


@pytest.fixture
def fake_compile(monkeypatch):
    """Record a real JITFunction's host compiles and runs, in order, instead
    of performing them.

    ``compile_error(kwargs)`` may return an exception for a compile to
    raise, e.g. per config.
    """

    def install(jit_fn, *, fail_first=False, compile_error=None):
        calls: list[SimpleNamespace] = []

        def run(*args, grid, warmup, **kwargs):
            calls.append(
                SimpleNamespace(args=args, grid=grid, warmup=warmup, kwargs=kwargs)
            )
            if fail_first and len(calls) == 1:
                raise RuntimeError("compile failed")
            if warmup and compile_error is not None:
                error = compile_error(kwargs)
                if error is not None:
                    raise error
            return _FakeKernel(tuple(sorted(kwargs.items())))

        _install_fake_run(monkeypatch, jit_fn, run)
        return calls

    return install


def _fake_bench(kernel_call, quantiles):
    # An Autotuner do_bench that needs no GPU: every config ties.
    kernel_call()
    return [1.0, 1.0, 1.0]


def _static_assert_failure():
    return CompileTimeAssertionFailure(None, ast.Pass(), "static_assert failed")


def _call(**overrides):
    fields: dict = dict(jit_fn=None, args=(), kwargs={}, grid=(1,), capture=False)
    fields.update(overrides)
    return LaunchCall(**fields)


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


def _grid(meta):
    return (triton.cdiv(meta["n"], meta["BLOCK"]),)


def _dummy_lang_fn():
    """Provides tl globals for patch_lang in patch_run tests."""
    return tl.arange(0, 1)


def _in_thread(fn, *args):
    """Run ``fn(*args)`` on another host thread; its result or exception."""
    outcome: dict = {}

    def target():
        try:
            outcome["result"] = fn(*args)
        except BaseException as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=target)
    worker.start()
    worker.join(30)
    assert not worker.is_alive()
    return outcome


# ======== declarations and composition =========


class _NotSkippingIRClient(_IRClient):
    NAME = "ir_not_skipping"
    LAUNCH = "indifferent"


def test_add_clients_refuses_an_ir_client_that_does_not_skip_the_launch():
    """A traced launch with IR clients never runs the real kernel, so an IR
    client must declare LAUNCH="skip"; the refused batch inserts nothing."""
    manager = ClientManager([_SkipIRClient(), _EagerClient()])

    with pytest.raises(ValueError, match="must declare LAUNCH = 'skip'"):
        manager.add_clients([_PeerIRClient(), _NotSkippingIRClient()])

    assert list(manager.clients) == ["ir_skip", "eager"]

    # An interpreting client's LAUNCH is not read.
    class _EagerSkip(_EagerClient):
        NAME = "eager_skip"
        LAUNCH = "skip"

    manager.add_clients([_EagerSkip()])
    assert list(manager.clients) == ["ir_skip", "eager", "eager_skip"]


def test_add_clients_rejects_unknown_launch_value():
    class _BadIRClient(_IRClient):
        NAME = "ir_bad"
        LAUNCH = "maybe"

    with pytest.raises(ValueError, match="LAUNCH must be one of"):
        ClientManager([_BadIRClient()])


@pytest.mark.parametrize(
    "stages, named", [({"TTIR"}, "['TTIR']"), ({"ttir", "ttgir"}, "['ttgir']")]
)
def test_add_clients_refuses_stages_a_host_compile_does_not_produce(stages, named):
    """An IR_STAGES name other than "ttir" (a later stage, or a misspelled
    one) is the client's bug: a ValueError when the client is added, never
    a compile failure."""

    class _Stages(_IRClient):
        NAME = "ir_stages"
        IR_STAGES = frozenset(stages)

    with pytest.raises(ValueError, match=re.escape(f"_Stages.IR_STAGES: {named}")):
        ClientManager([_Stages()])

    # Declaring none is fine (the client reads no stage).
    class _NoStages(_IRClient):
        NAME = "ir_no_stages"
        IR_STAGES = frozenset()

    ClientManager([_NoStages()])


# ======== patch_run =========


def _first_op():
    frontend = get_frontend("triton")
    namespace, attrs = next(iter(frontend.namespaces.items()))
    attr = next(iter(attrs))
    return frontend, namespace, attr


def test_patch_run_registers_ops_only_for_interpreting_clients():
    eager = _EagerClient()
    # The IR client would raise if asked for op or loop callbacks.
    manager = ClientManager([eager, _PeerIRClient()])
    frontend, namespace, attr = _first_op()
    original = frontend.original_ops[namespace][attr]
    store_patches = [
        (ns, name)
        for ns, attrs in frontend.namespaces.items()
        for name, op_type in attrs.items()
        if op_type is Store
    ]
    assert store_patches

    with manager.patch_run(_dummy_lang_fn, frontend_name="triton"):
        for ns, name in store_patches:
            assert getattr(ns, name).before_callback is eager.on_store

    assert getattr(namespace, attr) is original


# ======== interpreter callbacks =========


def test_interpreter_callbacks_reach_only_interpreting_clients():
    eager = _EagerClient()
    manager = ClientManager([eager, _PeerIRClient()])
    tensor = torch.zeros(1)

    assert manager.pre_run_callback(_dummy_lang_fn) is True
    assert manager.post_run_callback(_dummy_lang_fn) is True
    manager.arg_callback("x_ptr", tensor, tensor)
    manager.grid_callback((2, 1, 1))
    manager.grid_idx_callback((0, 0, 0))

    assert eager.calls == [
        "pre_run",
        "post_run",
        ("arg", "x_ptr"),
        ("grid", (2, 1, 1)),
        "grid_idx",
    ]
    assert tensor in manager.launch.tensors
    assert manager.launch.grid == (2, 1, 1)


def test_run_votes_without_interpreting_clients_keep_the_grid_running():
    manager = ClientManager([_PeerIRClient()])

    assert manager.pre_run_callback(_dummy_lang_fn) is True
    assert manager.post_run_callback(_dummy_lang_fn) is True


# ======== finalize, begin/abort =========


def test_abort_hook_failure_never_masks_the_launch_exception():
    class _BrokenAbort(_EagerClient):
        NAME = "broken_abort"

        def abort_launch(self, exc):
            raise RuntimeError("abort hook failed")

    log: list = []
    manager = ClientManager([_BrokenAbort(), _PeerIRClient(log)])
    launch_exc = ValueError("launch failed")

    if hasattr(launch_exc, "add_note"):
        manager.abort_launch(launch_exc)
        assert any("abort hook failed" in n for n in launch_exc.__notes__)
    else:
        with pytest.warns(RuntimeWarning, match="abort hook failed"):
            manager.abort_launch(launch_exc)
    # Every client still got the abort.
    assert log == [("abort", ValueError)]


def test_abort_hook_interrupt_propagates_after_every_client():
    class _Interrupting(_EagerClient):
        NAME = "interrupting"

        def abort_launch(self, exc):
            raise KeyboardInterrupt

    log: list = []
    manager = ClientManager([_Interrupting(), _PeerIRClient(log)])
    launch_exc = ValueError("launch failed")

    with pytest.raises(KeyboardInterrupt) as info:
        manager.abort_launch(launch_exc)

    assert info.value.__cause__ is launch_exc
    assert log == [("abort", ValueError)]


def test_begin_failure_aborts_exactly_the_clients_that_began():
    class _BrokenBegin(_PeerIRClient):
        fail = True

        def begin_launch(self, call):
            super().begin_launch(call)
            if self.fail:
                raise KeyError("begin failed")

    log: list = []
    first, broken, last = _SkipIRClient(log), _BrokenBegin(log), _EagerClient()
    manager = ClientManager([first, broken, last])

    with pytest.raises(KeyError):
        manager.begin_launch(_call())

    # The failing client began (and may hold partial state); `last` never did.
    assert log == ["begin", "begin", ("abort", KeyError), ("abort", KeyError)]
    assert last.calls == []
    # No launch was left open, so another host thread may begin one.
    broken.fail = False
    assert "error" not in _in_thread(manager.begin_launch, _call())


def test_begin_launch_refuses_another_threads_launch_without_touching_it():
    log: list = []
    manager = ClientManager([_SkipIRClient(log)])
    manager.begin_launch(_call())
    launch = manager.launch

    refused = _in_thread(manager.begin_launch, _call())["error"]
    # A stray abort from that thread does not reach this launch either.
    _in_thread(manager.abort_launch, refused)

    assert isinstance(refused, RuntimeError)
    assert "another host thread" in str(refused)
    assert manager.launch is launch
    assert log == ["begin"]

    # Once this launch ends, the other thread may begin the next one.
    manager.finalize()
    assert "error" not in _in_thread(manager.begin_launch, _call())
    assert log == ["begin", "finalize", "begin"]


# ======== ir_capture =========


def test_ir_capture_compiles_without_launching(_fake_host_compile):
    log: list = []
    ir, eager = _SkipIRClient(log), _EagerClient()
    manager = ClientManager([ir, eager])
    jit_fn = _FakeJit(log)
    x = torch.zeros(10)

    with manager.ir_capture(jit_fn):
        assert "run" in vars(jit_fn)
        ret = jit_fn.run(x, 10, grid=_grid, warmup=False, BLOCK=4, num_warps=2)

    assert "run" not in vars(jit_fn)
    # A launching call compiles too; nothing launches (_FakeJit.run raises).
    assert log == ["compile", "before", "after"]
    target = default_ir_target()
    assert _fake_host_compile == [(jit_fn, target)]
    (event,) = ir.events
    assert event.target == target
    assert ret is event.kernel
    assert event.jit_fn is jit_fn
    assert event.args == (x, 10)
    assert dict(event.kwargs) == {"BLOCK": 4, "num_warps": 2}
    assert dict(event.bound_args) == {"x_ptr": x, "n": 10, "BLOCK": 4}
    assert event.grid is _grid
    assert event.resolved_grid == (3, 1, 1)
    assert event.specialization == "hash-4"
    assert "ttir" in event.kernel.asm
    # Launch.grid from the binding, and no IR event for the interpreting
    # peer. The binding never adds tensors (none may outlive the launch):
    # an interpreted run's arg_callback records its own.
    assert not manager.launch.tensors
    assert manager.launch.grid == (3, 1, 1)
    assert "before_launch" not in eager.calls


def test_ir_capture_restores_on_error_and_does_not_double_wrap():
    log: list = []
    manager = ClientManager([_SkipIRClient(log, raise_in_before=ValueError("stop"))])
    jit_fn = _FakeJit(log)

    with pytest.raises(ValueError, match="stop"):
        with manager.ir_capture(jit_fn):
            wrapper = jit_fn.run
            with manager.ir_capture(jit_fn):
                assert jit_fn.run is wrapper
            assert jit_fn.run is wrapper
            jit_fn.run(torch.zeros(4), 4, grid=(1,), warmup=True)

    # before_launch raised: no after_launch, wrapper removed.
    assert log == ["compile", "before"]
    assert "run" not in vars(jit_fn)


def test_ir_capture_refuses_another_owner_and_ignores_other_threads():
    class _LaunchingJit(_FakeJit):
        def run(self, *args, grid, warmup, **kwargs):
            self.log.append("launch")
            return "launched"

    log: list = []
    first = ClientManager([_SkipIRClient(log)])
    second = ClientManager([_PeerIRClient()])
    jit_fn = _LaunchingJit(log)

    with first.ir_capture(jit_fn):
        with pytest.raises(RuntimeError, match="already being captured"):
            with second.ir_capture(jit_fn):
                pass
        outcome = _in_thread(
            lambda: jit_fn.run(torch.zeros(4), 4, grid=(1,), warmup=False)
        )

    # The other thread's call went straight to the original run.
    assert outcome == {"result": "launched"}
    assert log == ["launch"]


def test_ir_capture_delivers_each_specialization_once_per_launch():
    log: list = []
    ir = _SkipIRClient(log)
    manager = ClientManager([ir])
    jit_fn = _FakeJit(log)
    x = torch.zeros(4)

    with manager.ir_capture(jit_fn):
        jit_fn.run(x, 4, grid=(1,), warmup=True, BLOCK=4)
    with manager.ir_capture(jit_fn):
        # e.g. the configs of an autotuned kernel, compiled again.
        for block in (4, 4, 8, 4):
            jit_fn.run(x, 4, grid=(1,), warmup=True, BLOCK=block)

    assert [e.specialization for e in ir.events] == ["hash-4", "hash-8"]
    assert log.count("compile") == 5  # every call still compiled

    # The next traced launch reports its specializations again.
    manager.begin_launch(_call(capture=True))
    with manager.ir_capture(jit_fn):
        jit_fn.run(x, 4, grid=(1,), warmup=True, BLOCK=4)
    assert [e.specialization for e in ir.events] == ["hash-4"]


class _Opaque:
    """A value the binding fingerprint knows nothing about (weakref-able)."""


def test_ir_capture_delivers_each_binding_of_a_specialization():
    """Calls compiling to one kernel ("hash-4") are told apart by their
    binding fingerprint, never by tensor data."""
    ir = _SkipIRClient()
    manager = ClientManager([ir])
    jit_fn = _FakeJit()
    x, y = torch.zeros(8), torch.zeros(8)
    opaque, items = _Opaque(), [1]

    def delivered(*args, grid=(1,), **kwargs) -> bool:
        before = len(ir.events)
        jit_fn.run(*args, grid=grid, warmup=True, **kwargs)
        return len(ir.events) > before

    with manager.ir_capture(jit_fn):
        assert delivered(x, 4)
        assert not delivered(x, 4)  # the same call again
        assert not delivered(x, 4, grid=_grid)  # a callable grid: (1, 1, 1)
        assert delivered(x, 5)  # a scalar's value
        assert delivered(x, 4.0)  # ... and type
        assert delivered(x, 4, grid=(2,))  # the grid
        assert delivered(x, 4, num_warps=8)  # a kwarg (compile option)
        assert delivered(y, 4)  # a tensor's data_ptr
        assert delivered(x[:4], 4)  # ... shape
        assert delivered(x[::2], 4)  # ... strides
        assert delivered(x.view(torch.int32), 4)  # ... dtype
        x.add_(1)
        assert not delivered(x, 4)  # never its data
        assert delivered(x, (4, 5))  # a tuple, item by item
        assert not delivered(x, (4, 5))
        assert delivered(x, opaque)  # anything else by identity
        assert not delivered(x, opaque)
        assert delivered(x, items)
        assert delivered(x, [1])  # equal, but another object

    assert {e.specialization for e in ir.events} == {"hash-4"}
    assert [e.resolved_grid for e in ir.events][:4] == [(1, 1, 1)] * 3 + [(2, 1, 1)]


class _ConstexprJit(_FakeJit):
    """A _FakeJit whose BLOCK is a tl.constexpr parameter."""

    params = [
        SimpleNamespace(name=name, is_constexpr=name == "BLOCK")
        for name in _FakeJit.signature.parameters
    ]


class _FreshDType:
    """Equal to every other instance in all but identity, as a tl.dtype a
    heuristic builds per call is (the fake kernel's hash holds the repr)."""

    def __repr__(self):
        return "fp32"


def test_a_constexpr_argument_counts_only_through_the_specialization():
    """Triton hashes a constexpr argument into the kernel, so the binding
    fingerprint leaves it out: an equal but fresh constexpr object per call
    adds no binding, passed by keyword or positionally; a non-constexpr
    argument still counts by identity."""
    ir = _SkipIRClient()
    manager = ClientManager([ir])
    jit_fn = _ConstexprJit()
    x = torch.zeros(8)

    def delivered(*args, **kwargs) -> bool:
        before = len(ir.events)
        jit_fn.run(*args, grid=(1,), warmup=True, **kwargs)
        return len(ir.events) > before

    with manager.ir_capture(jit_fn):
        assert delivered(x, 4, BLOCK=_FreshDType())  # compiles "hash-fp32"
        assert not delivered(x, 4, BLOCK=_FreshDType())
        assert delivered(x, 5, BLOCK=_FreshDType())  # a runtime argument
        assert delivered(x, 4, _FreshDType())  # "hash-4": BLOCK is no kwarg
        assert not delivered(x, 4, _FreshDType())
        opaque = _Opaque()
        assert delivered(x, opaque, BLOCK=_FreshDType())
        assert delivered(x, _Opaque(), BLOCK=_FreshDType())

    specializations = [e.specialization for e in ir.events]
    assert specializations == ["hash-fp32"] * 2 + ["hash-4"] + ["hash-fp32"] * 2
    # Each event still carries the call's own constexpr object.
    assert all(isinstance(e.bound_args["BLOCK"], _FreshDType) for e in ir.events)


def test_a_pinned_value_outlives_its_call_only_until_the_launch_ends():
    """An unknown value's identity token keeps the object alive for the
    launch, so a fresh object per call never reuses a delivered id; once
    the launch ends (finalize or abort), nothing holds it."""

    class _Forgetful(_SkipIRClient):
        def before_launch(self, event):
            self.log.append(event.bound_args["n"].__class__.__name__)

    for end in ("finalize", "abort"):
        ir = _Forgetful()
        manager = ClientManager([ir])
        jit_fn = _FakeJit()
        manager.begin_launch(_call(capture=True))
        refs = []
        with manager.ir_capture(jit_fn):
            for _ in range(3):
                value = _Opaque()
                refs.append(weakref.ref(value))
                jit_fn.run(torch.zeros(4), value, grid=(1,), warmup=True)
                del value
        # Three distinct objects, three events: none was freed mid-launch.
        assert ir.log.count("_Opaque") == 3
        assert all(ref() is not None for ref in refs)
        if end == "finalize":
            manager.finalize()
        else:
            manager.abort_launch(RuntimeError("launch failed"))
        gc.collect()
        assert all(ref() is None for ref in refs)


def test_a_failing_compile_is_reported_as_data():
    log: list = []
    ir = _SkipIRClient(log)
    manager = ClientManager([ir])
    x = torch.zeros(4)
    error = RuntimeError("the front end failed")
    broken = _FakeJit(log, compile_error=error)

    with manager.ir_capture(broken) as window:
        assert broken.run(x, 4, grid=(1,), warmup=True) is None
    assert (window.compiled, window.failures) == (0, [error])

    assert ir.events == []
    (failure,) = ir.failures
    assert failure.error is error and failure.kernel is None
    assert failure.specialization is None
    assert failure.target == default_ir_target()
    assert dict(failure.bound_args) == {"x_ptr": x, "n": 4, "BLOCK": 4}


def test_a_failing_config_is_reported_once_per_launch():
    """The same failing call compiled again in a launch is no news; another
    call, or the next launch, is."""
    log: list = []
    ir = _SkipIRClient(log)
    manager = ClientManager([ir])
    broken = _FakeJit(log, compile_error=_static_assert_failure())
    x = torch.zeros(4)

    with manager.ir_capture(broken) as window:
        for block in (8, 8, 16):
            assert broken.run(x, 4, grid=(1,), warmup=True, BLOCK=block) is None

    assert [dict(f.kwargs)["BLOCK"] for f in ir.failures] == [8, 16]
    assert len(window.failures) == 3
    manager.begin_launch(_call(capture=True))
    with manager.ir_capture(broken):
        broken.run(x, 4, grid=(1,), warmup=True, BLOCK=8)
    assert [dict(f.kwargs)["BLOCK"] for f in ir.failures] == [8]


class _HipIRClient(_PeerIRClient):
    def __init__(self, log=None):
        super().__init__(log)
        self.ir_target = "hip:gfx942"


def test_one_trace_compiles_for_one_target(_fake_host_compile):
    """IR clients that name different targets are refused before anything
    compiles; a target set explicitly to the default's value is the
    default."""
    from triton.backends.compiler import GPUTarget

    cuda89 = GPUTarget("cuda", 89, 32)
    ir, hip = _SkipIRClient(), _HipIRClient()
    manager = ClientManager([ir, hip])
    jit_fn = _FakeJit()

    with pytest.raises(
        ValueError,
        match=re.escape("these name several (cuda:89: ir_skip; hip:gfx942: ir_peer)"),
    ):
        with manager.ir_capture(jit_fn):
            pass
    assert _fake_host_compile == []

    hip.ir_target = cuda89
    with manager.ir_capture(jit_fn):
        jit_fn.run(torch.zeros(4), 4, grid=(1,), warmup=True)
    assert _fake_host_compile == [(jit_fn, cuda89)]
    assert ir.events[0] is hip.events[0]
    assert ir.events[0].target == cuda89


def test_an_invalid_configured_target_fails_the_launch(fake_compile, monkeypatch):
    monkeypatch.setattr(tilelens_config, "ir_target", "sm80")
    log: list = []
    traced = tilelens.trace(_SkipIRClient(log))(_make_plain_kernel())
    calls = fake_compile(traced.jit_fn)

    with pytest.raises(ValueError, match="no IR target"):
        traced[(1,)](torch.zeros(4), torch.zeros(4), 4, BLOCK=4)
    assert calls == [] and log == ["begin", ("abort", ValueError)]


@pytest.fixture
def patched_language():
    """An interpreted traced launch's language patch, as if active on
    another host thread."""
    scopes = LANG_PATCH_SCOPES.setdefault("triton", [])
    scope = object()
    scopes.append(scope)
    yield
    scopes.remove(scope)


def test_an_ir_only_launch_raises_the_patched_language_refusal(
    fake_compile, patched_language
):
    """No compile's outcome, so not a compile failure that lets a launch
    survive: the IR-only launch fails (concurrent traced launches that
    mix interpretation and real compiles are unsupported). The IR client
    saw it as data first."""
    log: list = []
    ir = _SkipIRClient(log)
    traced = tilelens.trace(ir)(_make_plain_kernel())
    calls = fake_compile(traced.jit_fn)

    with pytest.raises(LanguagePatchedError, match="language patched"):
        traced[(2,)](torch.zeros(8), torch.zeros(8), 8, BLOCK=4)
    assert log == [
        "begin",
        ("compile_failed", LanguagePatchedError),
        ("abort", LanguagePatchedError),
    ]
    assert calls == []


def test_resolve_grid():
    assert _resolve_grid((2,), {}) == (2, 1, 1)
    assert _resolve_grid((2, 3, 4), {}) == (2, 3, 4)
    assert _resolve_grid(lambda meta: (meta["n"], 2), {"n": 5}) == (5, 2, 1)
    assert _resolve_grid(None, {}) is None
    assert _resolve_grid(lambda meta: (meta["missing"],), {}) is None
    assert _resolve_grid((1, 1, 1, 1), {}) is None


# ======== TritonTrace.run lifecycle =========


def test_ir_only_launch_compiles_every_config_without_running(fake_compile):
    log: list = []
    ir = _SkipIRClient(log)
    traced = tilelens.trace(ir)(_make_autotuned_kernel())
    calls = fake_compile(traced.jit_fn)
    x = torch.arange(8, dtype=torch.float32)
    out = torch.zeros(8)

    traced[_grid](x, out, 8)

    assert torch.equal(out, torch.zeros(8))
    assert [c.warmup for c in calls] == [True, True]
    assert log == ["begin", "before", "after", "before", "after", "finalize"]
    (events,) = ir.finalized
    assert [e.kwargs["BLOCK"] for e in events] == [4, 8]
    assert [e.kwargs["EVEN"] for e in events] == [True, True]
    assert [e.resolved_grid for e in events] == [(2, 1, 1), (1, 1, 1)]
    assert len({e.specialization for e in events}) == 2
    (call,) = ir.launch_calls
    assert call.jit_fn is traced.jit_fn and call.capture is True
    assert call.args == (x, out, 8) and dict(call.kwargs) == {} and call.grid is _grid
    # The fake compile is back in place; the capture wrapper is gone.
    assert "run" in vars(traced.jit_fn)
    assert not getattr(traced.jit_fn.run, "_tilelens_ir_capture", False)

    # The next launch compiles every config again, whatever the autotune
    # cache holds.
    traced[_grid](x, out, 8)
    assert [e.kwargs["BLOCK"] for e in ir.finalized[1]] == [4, 8]


@pytest.mark.parametrize(
    "failing",
    [
        # A config Autotuner._bench drops when it fails like this,
        {8: _static_assert_failure},
        # every config,
        {4: _static_assert_failure, 8: _static_assert_failure},
        # an error the autotuner does not tolerate.
        {8: lambda: ValueError("bad config")},
    ],
    ids=["one-config", "every-config", "untolerated-error"],
)
def test_ir_only_compile_failures_never_fail_the_launch(fake_compile, failing):
    """A config that fails to compile for the IR target is data for
    the IR clients, whatever the error and even when no config compiled; the
    skipped launch returns as it does when every config compiles (None for
    an autotuned kernel)."""
    log: list = []
    ir = _SkipIRClient(log)
    traced = tilelens.trace(ir)(_make_autotuned_kernel())
    fake_compile(
        traced.jit_fn,
        compile_error=lambda kw: failing[kw["BLOCK"]]()
        if kw["BLOCK"] in failing
        else None,
    )
    x, out = torch.zeros(8), torch.zeros(8)

    assert traced[_grid](x, out, 8) is None

    assert log[0] == "begin" and log[-1] == "finalize"
    assert not [e for e in log if isinstance(e, tuple) and e[0] == "abort"]
    (events,) = ir.finalized
    assert [e.kwargs["BLOCK"] for e in events] == [
        b for b in (4, 8) if b not in failing
    ]
    # Every failing config reached the IR client as data, once.
    assert sorted(f.kwargs["BLOCK"] for f in ir.failures) == sorted(failing)


def test_a_failed_host_compile_ends_the_launch_normally(fake_compile):
    """A plain kernel's only config fails to compile: the skipped launch
    returns None (the host-compiled kernel it returns otherwise), and the
    next launch compiles as if nothing had happened."""
    log: list = []
    ir = _SkipIRClient(log)
    traced = tilelens.trace(ir)(_make_plain_kernel())
    calls = fake_compile(traced.jit_fn, fail_first=True)
    args = (torch.zeros(8), torch.zeros(8), 8)

    assert traced[(2,)](*args, BLOCK=4) is None
    assert log == ["begin", ("compile_failed", RuntimeError), "finalize"]
    assert not getattr(traced.jit_fn.run, "_tilelens_ir_capture", False)

    log.clear()
    assert isinstance(traced[(2,)](*args, BLOCK=4), _FakeKernel)

    assert log == ["begin", "before", "after", "finalize"]
    assert [len(events) for events in ir.finalized] == [0, 1]
    assert len(calls) == 2


def test_mixed_trace_compiles_for_ir_and_interprets_for_eager(fake_compile):
    log: list = []
    ir, eager = _SkipIRClient(log), _EagerClient()
    traced = tilelens.trace(ir)(_make_plain_kernel())
    traced = tilelens.trace(eager)(traced)
    calls = fake_compile(traced.jit_fn)
    x = torch.arange(8, dtype=torch.float32)
    out = torch.zeros(8)

    traced[(2,)](x, out, 8, BLOCK=4)

    # IR client: one host compile, no real launch.
    assert [c.warmup for c in calls] == [True]
    (events,) = ir.finalized
    assert len(events) == 1
    # Interpreting client: the full interpreted run, which wrote the output.
    assert eager.stores == 2
    assert eager.calls.count("pre_run") == 2
    assert "finalize" in eager.calls
    torch.testing.assert_close(out, x + 1)
    # Both clients vote on the legacy warmup.
    assert "pre_warmup" in eager.calls and "pre_warmup" in log


def test_mixed_trace_survives_ir_compile_failures(fake_compile):
    # E.g. a kernel the host compile rejects, which the interpreter runs.
    log: list = []
    ir, eager = _SkipIRClient(log), _EagerClient()
    traced = tilelens.trace(eager)(tilelens.trace(ir)(_make_plain_kernel()))
    fake_compile(
        traced.jit_fn,
        compile_error=lambda kw: RuntimeError("the host compile failed"),
    )
    x = torch.arange(8, dtype=torch.float32)
    out = torch.zeros(8)

    traced[(2,)](x, out, 8, BLOCK=4)

    torch.testing.assert_close(out, x + 1)
    assert eager.stores == 2 and "finalize" in eager.calls
    assert ir.finalized == [[]]
    assert [type(f.error) for f in ir.failures] == [RuntimeError]
    assert not any(isinstance(e, tuple) and e[0] == "abort" for e in log)


# ======== a call that does not bind raises, as untraced =========


def _bind_failure():
    """The TypeError the host compile raises for a call missing ``n``,
    marked as a bind failure (tilelens.core.host_compile.bind_failed)."""
    from tilelens.core import host_compile

    exc = TypeError("dynamic_func() missing 1 required positional argument: 'n'")
    host_compile._mark_bind_failed(exc)
    return exc


@pytest.mark.parametrize(
    "make, launch",
    [
        (_make_plain_kernel, lambda k, x, out: k[(2,)](x, out, 8, BLOCK=4)),
        (
            lambda: _make_autotuned_kernel(do_bench=_fake_bench),
            lambda k, x, out: k[_grid](x, out, 8),
        ),
    ],
    ids=["plain", "autotuned"],
)
def test_a_traced_call_that_does_not_bind_raises_as_untraced(
    fake_compile, make, launch
):
    """An IR-only launch raises the bind failure ("the program goes on" is
    for kernel compile failures only), from the first compile
    of the compile-only pass: the IR client's launch is aborted, never
    finalized, nothing launches or is recorded, and the next launch runs as
    if nothing had happened."""
    log: list = []
    ir = _SkipIRClient(log)
    traced = tilelens.trace(ir)(make())
    unbound = _bind_failure()
    failing = [unbound]
    calls = fake_compile(
        traced.jit_fn, compile_error=lambda kw: failing.pop() if failing else None
    )
    x, out = torch.zeros(8), torch.zeros(8)
    launches = len(trace_module.launches)

    with pytest.raises(TypeError) as raised:
        launch(traced, x, out)

    assert raised.value is unbound
    assert log == ["begin", ("abort", TypeError)]
    assert [c.warmup for c in calls] == [True]
    assert len(trace_module.launches) == launches
    assert not getattr(traced.jit_fn.run, "_tilelens_ir_capture", False)

    log.clear()
    launch(traced, x, out)
    assert log[0] == "begin" and log[-1] == "finalize"
    assert len(trace_module.launches) == launches + 1


def test_a_mixed_trace_raises_a_call_that_does_not_bind_before_interpreting(
    fake_compile,
):
    """A mixed trace (IR and eager clients): the IR clients' compile pass
    raises the bind failure before the interpreter runs (which would fail on
    the same call too), so the untraced JIT's error is the one raised; every
    client's launch is aborted."""
    log: list = []
    ir, eager = _SkipIRClient(log), _EagerClient()
    traced = tilelens.trace(eager)(tilelens.trace(ir)(_make_plain_kernel()))
    unbound = _bind_failure()
    fake_compile(traced.jit_fn, compile_error=lambda kw: unbound)
    x, out = torch.arange(8, dtype=torch.float32), torch.zeros(8)

    with pytest.raises(TypeError) as raised:
        traced[(2,)](x, out, 8, BLOCK=4)

    assert raised.value is unbound
    assert log == ["begin", ("abort", TypeError)]
    assert eager.calls == ["begin", ("abort", TypeError)]
    assert eager.stores == 0 and torch.equal(out, torch.zeros(8))


def test_a_launch_failing_in_finalize_is_not_aborted(fake_compile):
    class _ExitingIR(_SkipIRClient):
        def finalize(self):
            super().finalize()
            raise SystemExit(3)

    log: list = []
    traced = TritonTrace(_make_plain_kernel(), _ExitingIR(log))
    traced.add_client(_PeerIRClient(log))
    fake_compile(traced.jit_fn)

    with pytest.raises(SystemExit):
        traced[(1,)](torch.zeros(4), torch.zeros(4), 4, BLOCK=4)

    # Each launch ends in finalize or abort, never both.
    assert log.count("finalize") == 2
    assert not any(isinstance(e, tuple) and e[0] == "abort" for e in log)


def test_each_traced_launch_is_recorded_separately(fake_compile):
    class _VerdictIR(_SkipIRClient):
        def finalize(self):
            super().finalize()
            return [f"verdict-{len(self.finalized)}"]

    traced = tilelens.trace(_VerdictIR())(_make_plain_kernel())
    fake_compile(traced.jit_fn)
    before = len(trace_module.launches)
    a, b = torch.zeros(8), torch.zeros(16)

    traced[(2,)](a, a, 8, BLOCK=4)
    traced[(4,)](b, b, 16, BLOCK=4)

    first, second = trace_module.launches[before:]
    assert first is not second
    assert first.records == ["verdict-1"] and second.records == ["verdict-2"]
    # Launch.grid from the IR binding, per launch; an IR-only launch
    # records no tensors.
    assert not first.tensors and not second.tensors
    assert (first.grid, second.grid) == ((2, 1, 1), (4, 1, 1))


def test_cli_shape_traces_the_autotuner_over_an_inner_trace(fake_compile):
    # The CLI wrappers turn every @triton.jit into a TritonTrace and wrap the
    # Autotuner built on it again: TritonTrace(Autotuner(TritonTrace(JIT))).
    inner_ir, outer_ir = _SkipIRClient(), _SkipIRClient()
    jit_fn = _make_plain_kernel()
    inner = TritonTrace(jit_fn, inner_ir)
    user = triton.autotune(
        configs=[triton.Config({"BLOCK": 4}), triton.Config({"BLOCK": 8})],
        key=["n"],
    )(inner)
    before = dict(vars(user))
    outer = TritonTrace(user, outer_ir)
    fake_compile(outer.jit_fn)
    x, out = torch.zeros(8), torch.zeros(8)

    outer[_grid](x, out, 8)

    assert outer.jit_fn is inner.jit_fn is jit_fn
    (events,) = outer_ir.finalized
    assert [e.kwargs["BLOCK"] for e in events] == [4, 8]
    assert inner_ir.log == []
    assert torch.equal(out, torch.zeros(8))
    assert vars(user).keys() == before.keys()
    assert all(vars(user)[k] is v for k, v in before.items())


def test_a_skipped_launch_returns_the_kernel_only_without_an_autotuner(fake_compile):
    @triton.heuristics({"BLOCK": lambda args: 4})
    @triton.jit
    def heur_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, tl.load(x_ptr + offs))

    args = (torch.zeros(8), torch.zeros(8), 8)
    plain = tilelens.trace(_SkipIRClient())(_make_plain_kernel())
    fake_compile(plain.jit_fn)
    heur = tilelens.trace(_SkipIRClient())(heur_kernel)
    fake_compile(heur.jit_fn)
    tuned = tilelens.trace(_SkipIRClient())(_make_autotuned_kernel())
    fake_compile(tuned.jit_fn)

    # One config: the kernel the untraced launch would return.
    assert isinstance(plain[(2,)](*args, BLOCK=4), _FakeKernel)
    assert isinstance(heur[(2,)](*args), _FakeKernel)
    # Autotuned: no config was picked.
    assert tuned[_grid](*args) is None


def test_an_autotuned_call_passing_a_tuned_parameter_raises_as_untraced(
    fake_compile,
):
    """The untraced launch's autotuning refuses a call that passes one of a
    config's meta-parameters itself (Autotuner._bench's ValueError). The IR
    compiles go through Autotuner.warmup instead, and refuse it the same
    way, not with a TypeError naming a tilelens internal."""
    log: list = []
    traced = tilelens.trace(_SkipIRClient(log))(_make_autotuned_kernel())
    calls = fake_compile(traced.jit_fn)
    args = (torch.zeros(8), torch.zeros(8), 8)

    with pytest.raises(ValueError, match="Conflicting meta-parameters: BLOCK"):
        traced[_grid](*args, BLOCK=4)

    assert calls == [] and log == ["begin", ("abort", ValueError)]


class _ForgetfulIRClient(_IRClient):
    """An IR client that keeps nothing of a launch's arguments."""

    NAME = "ir_forgetful"

    def begin_launch(self, call):
        self.log.append("begin")

    def before_launch(self, event):
        self.log.append("before")


def _raise_in_pre_hook(nargs):
    raise RuntimeError("pre_hook failed")


@pytest.mark.parametrize("path", ["ir", "interpreted"])
def test_a_launch_that_raises_keeps_no_caller_tensor(path):
    """Autotuner.run and .warmup keep the call's arguments in ``nargs``
    until they return. Once a launch has raised, none of the trace's
    autotuner copies keeps the caller's tensors."""
    if path == "ir":
        # The IR compiles refuse a call passing a tuned meta-parameter.
        traced = tilelens.trace(_ForgetfulIRClient())(_make_autotuned_kernel())
        kwargs, error = {"BLOCK": 4}, "Conflicting meta-parameters"
    else:
        # The interpreted autotuning raises in a config's pre_hook.
        kernel = triton.autotune(
            configs=[triton.Config({"BLOCK": 4}, pre_hook=_raise_in_pre_hook)],
            key=["n"],
        )(_make_plain_kernel())
        traced = tilelens.trace(_EagerClient())(kernel)
        kwargs, error = {}, "pre_hook failed"

    def launch() -> list[weakref.ref]:
        # The caller's references end with this frame.
        x, out = torch.zeros(8), torch.zeros(8)
        with pytest.raises((ValueError, RuntimeError), match=error):
            traced[_grid](x, out, 8, **kwargs)
        return [weakref.ref(x), weakref.ref(out)]

    refs = launch()
    gc.collect()

    for runner in (traced.runner, traced.warmup_runner, traced.ir_runner):
        assert getattr(runner, "nargs", None) is None
    assert [ref() for ref in refs] == [None, None]


def test_launch_grid_is_the_grid_the_compiled_configs_share(fake_compile):
    # Nothing launches: configs that disagree on the grid leave it open, a
    # grid they share is the launch's.
    args = (torch.zeros(8), torch.zeros(8), 8)
    traced = tilelens.trace(_SkipIRClient())(_make_autotuned_kernel())
    fake_compile(traced.jit_fn)
    traced[_grid](*args)
    assert trace_module.launches[-1].grid is None
    traced[(3,)](*args)
    assert trace_module.launches[-1].grid == (3, 1, 1)


def test_launch_tensors_have_one_representation_per_launch(fake_compile):
    x = torch.arange(8, dtype=torch.float32)
    out = torch.zeros(8)

    ir_only = tilelens.trace(_SkipIRClient())(_make_plain_kernel())
    fake_compile(ir_only.jit_fn)
    ir_only[(2,)](x, out, 8, BLOCK=4)
    # None: holding the caller's device tensors would keep them alive
    # after the launch; the IR clients' records hold the facts they need.
    assert not trace_module.launches[-1].tensors
    assert trace_module.launches[-1].grid == (2, 1, 1)

    mixed = tilelens.trace(_EagerClient())(
        tilelens.trace(_SkipIRClient())(_make_plain_kernel())
    )
    fake_compile(mixed.jit_fn)
    mixed[(2,)](x, out, 8, BLOCK=4)
    # Only the interpreter's host copies, whose addresses the eager
    # clients' records use; not the caller's tensors on top.
    tensors = trace_module.launches[-1].tensors
    assert len(tensors) == 2
    assert not {id(t) for t in tensors} & {id(x), id(out)}


def test_ir_compiles_are_not_gated_by_an_instance_warmup_patch(
    fake_compile, monkeypatch
):
    # E.g. a vote gate someone left on the JITFunction, declining every
    # compile: IR compiles go through JITFunction's own warmup.
    def declining_warmup(*args, **kwargs):
        return None

    args = (torch.zeros(8), torch.zeros(8), 8)
    for traced, grid, kwargs in (
        (tilelens.trace(_SkipIRClient())(_make_plain_kernel()), (2,), {"BLOCK": 4}),
        (tilelens.trace(_SkipIRClient())(_make_autotuned_kernel()), _grid, {}),
    ):
        calls = fake_compile(traced.jit_fn)
        monkeypatch.setattr(traced.jit_fn, "warmup", declining_warmup, raising=False)
        traced[grid](*args, **kwargs)
        (ir,) = traced.client_manager.ir_clients()
        assert ir.finalized[-1] and calls


def test_ir_launch_compiles_on_untraced_arguments(fake_compile):
    helper = tilelens.trace(_SiblingEagerClient())(triton.jit(_unwrap_leaf))

    @triton.jit
    def apply(x_ptr, out_ptr, n, FN: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, tl.load(x_ptr + offs))

    ir = _SkipIRClient()
    traced = tilelens.trace(ir)(apply)
    calls = fake_compile(traced.jit_fn)
    traced[(1,)](torch.zeros(4), torch.zeros(4), 4, FN=helper, BLOCK=4)

    # The host compile sees the JITFunction; events describe the call as
    # made.
    assert [c.kwargs["FN"] for c in calls] == [helper.jit_fn]
    assert all(e.kwargs["FN"] is helper for e in ir.finalized[0])


def test_ir_clients_without_a_jit_function():
    ir = _SkipIRClient()
    traced = TritonTrace(InterpretedFunction(_make_plain_kernel().fn), ir)
    assert traced.jit_fn is None
    x = torch.arange(8, dtype=torch.float32)
    out = torch.zeros(8)

    assert traced[(2,)](x, out, 8, BLOCK=4) is None

    # No compiled kernel, so no events, and nothing runs.
    assert ir.finalized == [[]]
    (call,) = ir.launch_calls
    assert call.jit_fn is None and call.capture is False
    assert torch.equal(out, torch.zeros(8))


@pytest.mark.parametrize("trace_cls", [GluonTrace, NKITrace])
def test_gluon_and_nki_traces_run_the_launch_lifecycle(trace_cls):
    # Built without __init__: the Gluon simulation and NKI are not importable
    # everywhere, and an IR-only trace never reaches them.
    class _BrokenBegin(_SkipIRClient):
        def begin_launch(self, call):
            super().begin_launch(call)
            raise KeyError("begin failed")

    log: list = []
    traced = trace_cls.__new__(trace_cls)
    TraceInterface.__init__(traced, _SkipIRClient(log))

    # Only IR clients: nothing is interpreted.
    assert traced[(2,)](torch.zeros(4)) is None
    assert log == ["begin", "finalize"]
    (call,) = traced.client_manager.clients["ir_skip"].launch_calls
    assert call.jit_fn is None and call.capture is False and call.grid == (2,)

    log = []
    traced = trace_cls.__new__(trace_cls)
    TraceInterface.__init__(traced, _BrokenBegin(log))
    with pytest.raises(KeyError):
        traced[(2,)](torch.zeros(4))
    assert log == ["begin", ("abort", KeyError)]


# ======== nested traced calls =========


def _unwrap_leaf(x):
    return x + 1


def _kernel_calling_nested_leaf(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, _nested_traced_leaf(tl.load(x_ptr + offs)))  # noqa: F821


def test_nested_traced_calls_compare_only_interpreting_clients(fake_compile):
    # The CLI shape: the helper is traced with the eager client only, the
    # kernel with it and an IR client, which takes no part in the
    # interpreted run.
    module_globals = globals()
    module_globals["_nested_traced_leaf"] = tilelens.trace(_EagerClient())(
        triton.jit(_unwrap_leaf)
    )
    try:
        traced = tilelens.trace(_EagerClient())(
            tilelens.trace(_SkipIRClient())(triton.jit(_kernel_calling_nested_leaf))
        )
        fake_compile(traced.jit_fn)
        x = torch.arange(4, dtype=torch.float32)
        out = torch.zeros(4)

        traced[(1,)](x, out, BLOCK=4)

        torch.testing.assert_close(out, x + 1)
    finally:
        module_globals.pop("_nested_traced_leaf", None)
