"""The IR client layer (tilelens.ir.{launch,capture,verdict,client}) on fake
LaunchEvents: launch binding, the per-launch artifact log, the parse cache,
verdict records and the IRClient finalize template. No GPU; the real-kernel
counterparts live in tests/end_to_end/test_ir_client.py.
"""

from __future__ import annotations

import dataclasses
import enum
import gc
import importlib
import pickle
import subprocess
import sys
import types
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

import tilelens
from tilelens.core.client import ClientManager, LaunchCall
from tilelens.core.data import Launch
from tilelens.ir import (
    ArtifactLog,
    CompileFailure,
    ConfigVerdict,
    IRClient,
    IRVerdict,
    ParseCache,
    ParseOutcome,
    Refusal,
    SourceLocation,
    TensorFacts,
    bind_launch,
)

trace_module = importlib.import_module("tilelens.core.trace")
REPO = Path(__file__).resolve().parents[3]


# ======== fakes =========


class _Refusal(Exception):
    """Stands in for the TTIR reader's UnsupportedTTIR."""

    def __init__(self, message, kind, line_no=None, loc=None):
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.line_no = line_no
        self.loc = loc


class _FakeKernel:
    def __init__(self, key, *, asm=None, metadata=True):
        self.hash = f"hash-{key}"
        self.asm = (
            {"ttir": f"// ttir {key}", "ttgir": f"// ttgir {key}", "cubin": b"\x7fELF"}
            if asm is None
            else asm
        )
        if metadata:
            self.metadata = SimpleNamespace(
                target=SimpleNamespace(backend="cuda", arch=89, warp_size=32),
                num_warps=4,
                num_stages=3,
                shared=512,
                name=f"kernel_{key}",
                hash=self.hash,
            )


@triton.jit
def _kernel(x_ptr, out_ptr, n, flag, scale, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pass


@triton.jit
def _tuple_kernel(ptrs, n):
    pass


class _Descriptor:
    """A descriptor-style argument: the kernel addresses its .base tensor."""

    def __init__(self, base):
        self.base = base


def _grid(meta):
    return (triton.cdiv(meta["n"], meta["BLOCK"]),)


def _event(args, kwargs, *, grid=(1,), kernel=None, error=None, target=None):
    # The core's own event builder: bound_args and resolved_grid as
    # ir_capture computes them.
    return ClientManager._launch_event(
        _kernel, args, kwargs, grid, kernel, error=error, target=target
    )


def _call(**kwargs):
    return LaunchCall(jit_fn=_kernel, args=(), kwargs=kwargs, grid=None, capture=True)


def _tensors():
    x = torch.arange(64, dtype=torch.float32)
    out = torch.zeros(64, dtype=torch.float32)
    return x, out


class _ToyIR(IRClient):
    NAME = "toy_ir"
    LAUNCH = "skip"
    IR_STAGES = frozenset({"ttir"})

    def __init__(self, analyze=None):
        super().__init__()
        self.calls: list = []
        self._analyze = analyze

    def analyze_launch(self, log):
        self.calls.append(("analyze", log.call, log.specializations, log.failures))
        if self._analyze is not None:
            return self._analyze(log)
        per_config = [
            ConfigVerdict(spec.specialization, spec.config, "seen")
            for spec in log.specializations
        ]
        return ["report"], IRVerdict(self.NAME, "ok", per_config=per_config)

    def on_analysis_error(self, exc):
        self.calls.append(("error", exc))
        return IRVerdict(self.NAME, "error", notes=[repr(exc)])


# ======== launch binding =========


def test_tensor_facts_read_the_view_and_its_storage():
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    view = base[1:, 1:]
    binding = bind_launch(
        _event((view, base, 1, False, 0.0), {"BLOCK": 1, "EVEN": True})
    )
    facts = binding.tensors["x_ptr"]

    assert facts == TensorFacts(
        data_ptr=base.data_ptr() + 5 * 4,
        elem_size=4,
        numel=6,
        shape=(2, 3),
        strides=(4, 1),
        dtype="torch.float32",
        contiguous=False,
        storage_data_ptr=base.data_ptr(),
        storage_nbytes=48,
    )
    # A strided view's allocation is its storage, not numel * elem_size.
    assert facts.allocation_interval() == (base.data_ptr(), base.data_ptr() + 48)


_FACTS = dict(
    data_ptr=1024,
    elem_size=4,
    numel=8,
    shape=(8,),
    strides=(1,),
    dtype="torch.float32",
    contiguous=True,
)


@pytest.mark.parametrize(
    "overrides, interval",
    [
        # Without storage metadata only a contiguous view's extent is known.
        ({}, (1024, 1056)),
        ({"contiguous": False}, None),
        # Partial or inconsistent storage metadata never falls back to numel.
        ({"storage_data_ptr": 1024}, None),
        ({"storage_data_ptr": 2048, "storage_nbytes": 64}, None),
        ({"storage_data_ptr": 1024, "storage_nbytes": 16}, None),
        ({"storage_data_ptr": 1000, "storage_nbytes": 100}, (1000, 1100)),
        ({"elem_size": 0}, None),
    ],
)
def test_allocation_interval_refuses_unknown_extents(overrides, interval):
    assert TensorFacts(**{**_FACTS, **overrides}).allocation_interval() == interval


def test_bind_launch_splits_arguments_by_kind():
    x, out = _tensors()
    # The caller passed BLOCK; a Heuristics layer added EVEN and an
    # Autotuner config num_warps.
    event = _event(
        (x, _Descriptor(out), 64, True, 0.5),
        {"BLOCK": 16, "EVEN": True, "num_warps": 4},
        grid=_grid,
    )
    binding = bind_launch(event, _call(BLOCK=16))

    assert binding.error is None
    assert dict(binding.params) == {"n": 64, "flag": 1}
    assert not isinstance(binding.params["flag"], bool)
    assert binding.tensors.keys() == {"x_ptr", "out_ptr"}
    assert binding.tensors["out_ptr"].data_ptr == out.data_ptr()
    assert binding.tensors["x_ptr"].numel == 64
    # Floats are no binding fact; constexprs keep their values.
    assert "scale" not in binding.params
    assert dict(binding.constexprs) == {"BLOCK": 16, "EVEN": True}
    assert dict(binding.config) == {"EVEN": True, "num_warps": 4}
    assert binding.raw_grid is _grid
    assert binding.grid == (4, 1, 1)
    with pytest.raises(TypeError):
        binding.params["n"] = 1  # type: ignore[index]


def test_bind_launch_without_the_call_counts_every_kwarg_as_config():
    x, out = _tensors()
    binding = bind_launch(
        _event((x, out, 64, False, 0.5), {"BLOCK": 16, "EVEN": False})
    )
    assert dict(binding.config) == {"BLOCK": 16, "EVEN": False}
    # A heuristic overriding a caller kwarg with another value is config too.
    heuristic = bind_launch(
        _event((x, out, 64, False, 0.5), {"BLOCK": 32, "EVEN": False}),
        _call(BLOCK=16, EVEN=False),
    )
    assert dict(heuristic.config) == {"BLOCK": 32}


def test_config_kwargs_tell_the_callers_scalars_by_value():
    x, out = _tensors()
    big = 10**6
    recomputed = int(str(big))  # equal, but another object
    assert recomputed is not big
    step = torch.tensor(1)
    binding = bind_launch(
        _event(
            (x, out, 64, False, 0.5),
            {"BLOCK": recomputed, "EVEN": True, "step": torch.tensor(1), "mode": 1},
        ),
        _call(BLOCK=big, EVEN=True, step=step, mode=True),
    )
    # An equal plain scalar of the same type is the caller's; another object
    # of any other kind, or a value of another type, is config.
    assert binding.config.keys() == {"step", "mode"}


def test_bind_launch_leaves_out_tuple_arguments():
    x, out = _tensors()
    event = ClientManager._launch_event(_tuple_kernel, ((x, out), 64), {}, (1,), None)
    binding = bind_launch(event)
    # Two TTIR pointer arguments, but no binding fact and no error: a
    # consumer must treat them as unknown (see LaunchBinding).
    assert event.bound_args["ptrs"] == (x, out)
    assert dict(binding.tensors) == {}
    assert dict(binding.params) == {"n": 64}
    assert binding.error is None


def test_bind_launch_never_raises():
    x, out = _tensors()

    class _Unreadable:
        def data_ptr(self):
            return 0

        def element_size(self):
            return 4

        def numel(self):
            raise RuntimeError("no numel")

    binding = bind_launch(
        _event((_Unreadable(), out, 64, False, 0.5), {"BLOCK": 4, "EVEN": True})
    )
    assert binding.error == "argument 'x_ptr': RuntimeError: no numel"
    assert binding.tensors.keys() == {"out_ptr"}
    assert dict(binding.params) == {"n": 64, "flag": 0}

    # An unresolvable grid is no error; a non-integer resolved grid is.
    unresolved = bind_launch(
        _event((x, out, 64, False, 0.5), {"BLOCK": 4, "EVEN": True}, grid=None)
    )
    assert unresolved.grid is None and unresolved.error is None
    broken = dataclasses.replace(
        _event((x, out, 64, False, 0.5), {"BLOCK": 4, "EVEN": True}),
        resolved_grid=("wide", 1, 1),
    )
    assert bind_launch(broken).grid is None
    assert "grid ('wide', 1, 1): TypeError" in bind_launch(broken).error

    # Not an event at all: everything unreadable, still a binding.
    nothing = bind_launch(SimpleNamespace(jit_fn=None))  # type: ignore[arg-type]
    assert nothing.error is not None and not nothing.tensors


@pytest.mark.parametrize(
    "resolved, grid",
    [
        ((np.int64(3), torch.tensor(2), 1), (3, 2, 1)),
        # The untraced launch rejects a float grid; it is not truncated.
        ((2.7, 1, 1), None),
    ],
)
def test_bind_launch_converts_grid_dims_as_the_launcher_does(resolved, grid):
    x, out = _tensors()
    event = dataclasses.replace(
        _event((x, out, 64, False, 0.5), {"BLOCK": 4, "EVEN": True}),
        resolved_grid=resolved,
    )
    binding = bind_launch(event)
    assert binding.grid == grid
    assert (binding.error is None) == (grid is not None)
    if grid is None:
        assert "TypeError: 'float' object cannot be interpreted" in binding.error


# ======== artifact log =========


def test_artifact_log_keeps_declared_stages_meta_and_bindings():
    x, out = _tensors()
    log = ArtifactLog({"ttir", "ptx"})
    call = _call()
    log.reset(call)
    kernel_a, kernel_b = _FakeKernel("a"), _FakeKernel("b")
    # Config A, config B, then A again with another grid (another binding).
    log.record(
        _event((x, out, 64, False, 0.5), {"BLOCK": 4, "EVEN": True}, kernel=kernel_a)
    )
    log.record(
        _event((x, out, 64, False, 0.5), {"BLOCK": 8, "EVEN": True}, kernel=kernel_b)
    )
    log.record(
        _event(
            (x, out, 64, False, 0.5),
            {"BLOCK": 4, "EVEN": True},
            grid=(2,),
            kernel=kernel_a,
        )
    )

    assert log.call is call
    spec_a, spec_b = log.specializations
    assert (spec_a.specialization, spec_b.specialization) == ("hash-a", "hash-b")
    assert len(spec_a.bindings) == 2 and len(spec_b.bindings) == 1
    # Only the declared stages the kernel has: no ttgir, no ptx.
    assert dict(spec_a.artifacts.stages) == {"ttir": "// ttir a"}
    assert dict(spec_a.artifacts.meta) == {
        "backend": "cuda",
        "arch": 89,
        "num_warps": 4,
        "num_stages": 3,
        "shared": 512,
        "name": "kernel_a",
        "config": {"BLOCK": 4, "EVEN": True},
    }
    assert spec_a.config == {"BLOCK": 4, "EVEN": True}
    assert spec_b.config == {"BLOCK": 8, "EVEN": True}
    assert spec_a.artifacts.error is None
    assert log.failures == ()


def test_artifact_log_records_compile_failures_with_their_config():
    from triton.backends.compiler import GPUTarget

    x, out = _tensors()
    log = ArtifactLog({"ttir"})
    log.reset(_call())
    compile_error = RuntimeError("static_assert failed")
    option_error = ValueError("num_ctas > 1 requires NVIDIA SM90+")
    target = GPUTarget("cuda", 80, 32)
    args = (x, out, 64, False, 0.5)
    log.record_failure(
        _event(args, {"BLOCK": 64, "EVEN": True}, error=compile_error, target=target)
    )
    # An event built outside the core names no target.
    log.record_failure(_event(args, {"BLOCK": 8, "EVEN": True}, error=option_error))

    assert log.failures == (
        CompileFailure(compile_error, {"BLOCK": 64, "EVEN": True}, target, _kernel),
        CompileFailure(option_error, {"BLOCK": 8, "EVEN": True}, None, _kernel),
    )
    assert log.specializations == ()


def test_artifact_log_contains_unreadable_kernels():
    x, out = _tensors()

    class _BrokenAsm(_FakeKernel):
        @property
        def asm(self):
            raise RuntimeError("asm gone")

        @asm.setter
        def asm(self, value):
            pass

    log = ArtifactLog({"ttir", "sass"})
    log.reset(_call())
    args = (x, out, 64, False, 0.5)
    log.record(_event(args, {"BLOCK": 4, "EVEN": True}, kernel=_BrokenAsm("a")))
    log.record(
        _event(
            args,
            {"BLOCK": 8, "EVEN": True},
            kernel=_FakeKernel("b", asm={"cubin": b""}, metadata=False),
        )
    )

    broken, bare = log.specializations
    assert broken.artifacts.error == "asm: RuntimeError: asm gone"
    assert broken.artifacts.meta["name"] == "kernel_a"
    # A declared stage the kernel lacks is simply absent; no metadata at all
    # is an error.
    assert dict(bare.artifacts.stages) == {}
    assert bare.artifacts.error.startswith("metadata: AttributeError")
    assert bare.artifacts.meta["num_warps"] is None


# ======== parse cache =========


class _StubReader:
    def __init__(self, result=None):
        self.calls: list = []
        self.result = result

    def __call__(self, text, **options):
        self.calls.append((text, options))
        if isinstance(self.result, BaseException):
            raise self.result
        return ("graph", text, tuple(sorted(options.items())))


def test_parse_cache_parses_each_text_once_per_options():
    reader = _StubReader()
    cache = ParseCache(reader, refusal=_Refusal)

    first = cache.get("module a")
    assert first == ParseOutcome(graph=("graph", "module a", ()))
    assert cache.get("module a") is first
    cache.get("module b")
    cache.get("module a", keep_going=True)
    cache.get("module a", keep_going=True)
    assert reader.calls == [
        ("module a", {}),
        ("module b", {}),
        ("module a", {"keep_going": True}),
    ]


def test_parse_cache_keeps_refusals_with_their_kind():
    def refuse():
        raise _Refusal(
            "scf.while", "control-flow", line_no=7, loc=SourceLocation("k.py", 3, 1)
        )

    try:
        refuse()
    except _Refusal as exc:
        refusal = exc
    assert refusal.__traceback__ is not None
    reader = _StubReader(refusal)
    cache = ParseCache(reader, refusal=_Refusal)

    outcome = cache.get("module")
    assert outcome.graph is None and outcome.error is None
    assert outcome.refusal is refusal and refusal.kind == "control-flow"
    # A cached refusal keeps no frames alive.
    assert refusal.__traceback__ is None
    assert cache.get("module") is outcome
    assert len(reader.calls) == 1
    assert Refusal.from_exception(outcome.refusal) == Refusal(
        "control-flow", "scf.while", 7, SourceLocation("k.py", 3, 1)
    )


class _Held:
    """A reader-frame local whose lifetime a test watches."""


def test_a_cached_refusal_keeps_no_frame_of_its_chain_alive():
    held = []

    def reader(text):
        local = _Held()
        held.append(weakref.ref(local))
        try:
            raise KeyError("walk")
        except KeyError:
            # A cause next to the implicit context: both chain links hold
            # a traceback into this frame.
            raise _Refusal("scf.while", "control-flow") from ValueError("cause")

    cache = ParseCache(reader, refusal=_Refusal)
    try:
        raise LookupError("the caller's")
    except LookupError as caller:
        refusal = cache.get("module").refusal
        # The exception the caller was handling is the caller's: untouched,
        # and no longer chained to the cached refusal.
        assert caller.__traceback__ is not None
    assert isinstance(refusal.__cause__, ValueError)
    walk = refusal.__context__
    assert isinstance(walk, KeyError) and walk.__context__ is None
    assert all(e.__traceback__ is None for e in (refusal, refusal.__cause__, walk))
    gc.collect()
    assert held[0]() is None


def test_parse_cache_reports_other_errors_without_caching_them():
    reader = _StubReader(RecursionError("too deep"))
    cache = ParseCache(reader, refusal=_Refusal)

    assert cache.get("module") == ParseOutcome(error="RecursionError: too deep")
    assert cache.get("module").error == "RecursionError: too deep"
    assert len(reader.calls) == 2
    # Without a refusal type, a refusal-shaped exception is just an error.
    plain = ParseCache(_StubReader(_Refusal("x", "control-flow")), refusal=KeyError)
    assert plain.get("module").error == "_Refusal: x"


def test_parse_cache_never_raises():
    reader = _StubReader()
    cache = ParseCache(reader)
    # A lone surrogate hashes (as "?") and parses.
    assert cache.get("module \ud800").graph == ("graph", "module \ud800", ())
    assert cache.get(None).error.startswith("AttributeError")  # type: ignore[arg-type]
    assert cache.get("module", layout=[1]).error.startswith("TypeError")
    # Every option name reaches the reader, "text" and "self" included.
    assert cache.get("module", text=1).error.startswith("TypeError: _StubReader")
    options = ParseCache(lambda text, /, **options: options, refusal=_Refusal)
    assert options.get("module", text=1, self=2).graph == {"text": 1, "self": 2}


@pytest.fixture
def default_reader(monkeypatch):
    """The module ParseCache resolves its default reader from: the real
    tilelens.ir.ttir_reader when it exists (its parse_ttir replaced), else a
    stand-in with the same two names."""
    try:
        module = importlib.import_module("tilelens.ir.ttir_reader")
    except ModuleNotFoundError as exc:
        if exc.name != "tilelens.ir.ttir_reader":
            raise
        module = types.ModuleType("tilelens.ir.ttir_reader")
        module.UnsupportedTTIR = _Refusal  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "tilelens.ir.ttir_reader", module)
    readers = []

    def install(result=None):
        reader = _StubReader(result)
        readers.append(reader)
        monkeypatch.setattr(module, "parse_ttir", reader, raising=False)
        return reader

    install.module = module  # type: ignore[attr-defined]
    return install


def test_parse_cache_without_a_reader_module_is_an_error_outcome(monkeypatch):
    """While tilelens.ir.ttir_reader cannot be imported, a lookup with the
    default reader reports that as an error outcome, never raises, and
    caches nothing."""
    monkeypatch.setitem(sys.modules, "tilelens.ir.ttir_reader", None)
    cache = ParseCache()

    outcome = cache.get("module")

    assert outcome.graph is None and outcome.refusal is None
    assert outcome.error.startswith("ModuleNotFoundError: ")
    assert "tilelens.ir.ttir_reader" in outcome.error
    assert cache.get("module") is not outcome  # not cached: retried


def test_parse_cache_resolves_its_default_reader_at_each_lookup(default_reader):
    cache = ParseCache()
    first = default_reader()
    cache.get("module")
    cache.get("module")
    # A replaced reader is another reader: parsed again, keyed apart.
    second = default_reader()
    cache.get("module")
    assert (len(first.calls), len(second.calls)) == (1, 1)

    # The default refusal type is the reader module's UnsupportedTTIR.
    unsupported = default_reader.module.UnsupportedTTIR
    refusal = unsupported(kind="control-flow", message="no")
    default_reader(refusal)
    assert cache.get("refused").refusal is refusal


# ======== verdict records =========


def test_verdicts_are_plain_frozen_picklable_records():
    config = {"BLOCK": 16}
    per_config = [ConfigVerdict("hash-a", config, "proved", n_reports=0)]
    verdict = IRVerdict(
        "toy_ir",
        "ok",
        scope="launch",
        refusal=Refusal("control-flow", "scf.while", 3, SourceLocation("k.py", 1, 1)),
        per_config=per_config,
        notes=["note"],
    )

    assert verdict.per_config == (ConfigVerdict("hash-a", {"BLOCK": 16}, "proved"),)
    assert verdict.notes == ("note",)
    config["BLOCK"] = 32  # the verdict holds its own copy
    assert verdict.per_config[0].config == {"BLOCK": 16}
    with pytest.raises(dataclasses.FrozenInstanceError):
        verdict.status = "races"  # type: ignore[misc]
    assert pickle.loads(pickle.dumps(verdict)) == verdict


def test_verdicts_are_not_hashable_and_take_no_bare_strings():
    # Frozen, but a config dict has no hash: no verdict claims one.
    with pytest.raises(TypeError, match="unhashable"):
        hash(ConfigVerdict("hash-a", {"BLOCK": 1}, "ok"))
    with pytest.raises(TypeError, match="unhashable"):
        hash(IRVerdict("toy_ir", "ok"))
    # A str is a sequence, but never the notes or configs meant.
    with pytest.raises(TypeError, match="notes takes a sequence"):
        IRVerdict("toy_ir", "ok", notes="solver timed out")
    with pytest.raises(TypeError, match="per_config takes a sequence"):
        IRVerdict("toy_ir", "ok", per_config="hash-a")  # type: ignore[arg-type]


class _Kind(str, enum.Enum):
    CONTROL_FLOW = "control-flow"


def test_verdicts_round_trip_through_a_saved_trace(tmp_path, monkeypatch):
    # Every verdict field is a value a trace can hold (trace_io
    # registers the tilelens.ir.verdict records).
    refusal = Refusal(_Kind.CONTROL_FLOW, "scf.while", 3, SourceLocation("k.py", 1, 1))
    verdict = IRVerdict(
        "toy_ir",
        "unsupported",
        scope="launch",
        refusal=refusal,
        per_config=[
            ConfigVerdict("hash-a", {"BLOCK": 16, "num_warps": 4}, "proved"),
            ConfigVerdict(None, {"BLOCK": 64}, "refused", refusal, n_reports=2),
        ],
        notes=["note"],
    )
    saved = [Launch(grid=(4, 1, 1), records=["report", verdict])]
    monkeypatch.setattr(trace_module, "launches", saved)

    tilelens.save(tmp_path / "trace.tvz")
    (launch,) = tilelens.load(tmp_path / "trace.tvz")

    assert launch.records == ["report", verdict]
    # A str-valued kind enum is saved as its string.
    kind = launch.records[1].refusal.kind
    assert isinstance(kind, str) and not isinstance(kind, enum.Enum)


def test_refusal_from_exception_reads_the_structured_fields():
    loc = SourceLocation("k.py", 2)
    assert Refusal.from_exception(_Refusal("m", "call", 4, loc)) == Refusal(
        "call", "m", 4, loc
    )

    class _Bare(Exception):
        kind = "inline-asm"

    assert Refusal.from_exception(_Bare("impure")) == Refusal("inline-asm", "impure")


# ======== IRClient =========


def test_ir_client_is_abstract_and_inert():
    class _Partial(IRClient):
        NAME = "partial"

        def analyze_launch(self, log):
            return [], IRVerdict(self.NAME, "ok")

    with pytest.raises(TypeError, match="abstract"):
        _Partial()  # type: ignore[abstract]

    ir = _ToyIR()
    # The warmup vote declines.
    assert ir.pre_warmup_callback(_kernel) is False


def _run_launch(manager, ir, *, events=(), failures=()):
    call = _call()
    manager.begin_launch(call)
    for event in events:
        ir.before_launch(event)
    for event in failures:
        ir.compile_failed(event)
    manager.finalize()
    return call


def test_finalize_returns_the_reports_then_the_verdict():
    x, out = _tensors()
    ir = _ToyIR()
    manager = ClientManager([ir])
    args = (x, out, 64, False, 0.5)
    events = [
        _event(args, {"BLOCK": 4, "EVEN": True}, kernel=_FakeKernel("a")),
        _event(args, {"BLOCK": 8, "EVEN": True}, kernel=_FakeKernel("b")),
    ]
    failure = _event(args, {"BLOCK": 64, "EVEN": True}, error=RuntimeError("bad"))
    call = _run_launch(manager, ir, events=events, failures=[failure])

    ((_, seen_call, specs, failures),) = ir.calls
    assert seen_call is call
    assert [s.specialization for s in specs] == ["hash-a", "hash-b"]
    assert [f.config for f in failures] == [{"BLOCK": 64, "EVEN": True}]
    verdict = manager.launch.records[-1]
    assert manager.launch.records == ["report", verdict]
    assert verdict is ir.last_verdict
    assert [c.config for c in verdict.per_config] == [
        {"BLOCK": 4, "EVEN": True},
        {"BLOCK": 8, "EVEN": True},
    ]
    # The log is released once the launch is finalized.
    assert (ir.artifacts.call, ir.artifacts.specializations) == (None, ())


def test_a_launch_with_nothing_captured_is_the_subclass_call():
    # TRITON_INTERPRET / an InterpretedFunction runner / Gluon / NKI: no
    # JITFunction, so the log stays empty, and only log.call says why.
    def analyze(log):
        if not log.call.capture:
            refusal = Refusal("no-capture", "no compiled kernel to read")
            return [], IRVerdict(_ToyIR.NAME, "unsupported", refusal=refusal)
        return [], IRVerdict(_ToyIR.NAME, "ok")

    ir = _ToyIR(analyze)
    manager = ClientManager([ir])
    call = LaunchCall(jit_fn=None, args=(), kwargs={}, grid=(4,), capture=False)
    manager.begin_launch(call)
    manager.finalize()

    assert ir.calls == [("analyze", call, (), ())]
    assert manager.launch.records == [ir.last_verdict]
    assert ir.last_verdict.refusal.kind == "no-capture"


def test_analysis_exceptions_go_to_the_client_handler():
    boom = RuntimeError("solver crashed")

    def analyze(log):
        raise boom

    ir = _ToyIR(analyze)
    manager = ClientManager([ir])
    _run_launch(manager, ir)

    assert ir.calls[-1] == ("error", boom)
    assert manager.launch.records == [ir.last_verdict]
    assert ir.last_verdict.status == "error"


def test_an_exiting_analysis_propagates_and_releases_the_log():
    x, out = _tensors()

    def analyze(log):
        raise SystemExit(1)  # e.g. abort_on_error

    ir = _ToyIR(analyze)
    manager = ClientManager([ir])
    manager.begin_launch(_call())
    ir.before_launch(
        _event(
            (x, out, 64, False, 0.5),
            {"BLOCK": 4, "EVEN": True},
            kernel=_FakeKernel("a"),
        )
    )
    with pytest.raises(SystemExit):
        manager.finalize()
    assert ir.last_verdict is None
    assert ir.artifacts.specializations == ()


def test_each_launch_starts_from_a_clean_log_and_no_verdict():
    x, out = _tensors()
    ir = _ToyIR()
    manager = ClientManager([ir])
    args = (x, out, 64, False, 0.5)
    _run_launch(
        manager,
        ir,
        events=[_event(args, {"BLOCK": 4, "EVEN": True}, kernel=_FakeKernel("a"))],
    )
    assert ir.last_verdict is not None

    # An aborted launch leaves no verdict and nothing recorded behind.
    call = _call()
    manager.begin_launch(call)
    assert ir.last_verdict is None and ir.artifacts.call is call
    ir.before_launch(_event(args, {"BLOCK": 8, "EVEN": True}, kernel=_FakeKernel("b")))
    manager.abort_launch(RuntimeError("launch failed"))
    assert ir.artifacts.specializations == ()

    _run_launch(manager, ir)
    assert ir.calls[-1][2] == ()


def test_importing_the_ir_layer_imports_no_triton():
    code = (
        "import sys\n"
        "import tilelens.ir as ir\n"
        "import tilelens.ir.capture, tilelens.ir.launch, tilelens.ir.verdict\n"
        "for name in ir.__all__:\n"
        "    getattr(ir, name)\n"
        "assert 'triton' not in sys.modules, sorted(m for m in sys.modules if 'triton' in m)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO)
