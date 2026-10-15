"""End-to-end tests of the IR client layer: a toy IRClient under tilelens.trace
on real kernels, compiled on the host for the default target (CPU tensors,
no GPU), gets an ArtifactLog per launch and puts its IRVerdict into
Launch.records. Counterparts on fake events live in
tests/unit/ir/test_ir_capture.py.
"""

from __future__ import annotations

import importlib

import pytest
import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler.errors import CompileTimeAssertionFailure

import tilelens
from tilelens.core.config import DEFAULT_IR_TARGET
from tilelens.ir import ConfigVerdict, IRClient, IRVerdict, ParseCache, Refusal

trace_module = importlib.import_module("tilelens.core.trace")
config_module = importlib.import_module("tilelens.core.config")


def _real_compiles_available() -> bool:
    # Triton imported under TRITON_INTERPRET=1 builds its own standard library
    # as InterpretedFunctions, so nothing can compile for real in-process. No
    # GPU is needed: IR mode compiles on the host.
    import triton.language.standard as tl_standard
    from triton.runtime.jit import JITFunction

    return isinstance(tl_standard.cdiv, JITFunction)


pytestmark = pytest.mark.skipif(
    not _real_compiles_available(),
    reason="Triton was imported under TRITON_INTERPRET=1: nothing compiles in-process",
)


@pytest.fixture(autouse=True)
def _no_driver(unreachable_driver):
    """IR mode needs no GPU: Triton's driver is unreachable here, as on
    a machine without one (where it raises "0 active drivers")."""
    unreachable_driver("IR mode queried Triton's driver")


@pytest.fixture(autouse=True)
def _default_ir_target(monkeypatch):
    """The default IR target, whatever TILELENS_IR_TARGET the caller
    set: in the process config, and in any Config read from the environment."""
    for name in ("TILELENS_IR_TARGET", "TRITON_VIZ_IR_TARGET"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config_module.config, "ir_target", DEFAULT_IR_TARGET)


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


class _StubRefusal(Exception):
    def __init__(self, message, kind):
        super().__init__(message)
        self.kind = kind


class _StubReader:
    """Counts the texts it is asked to parse; its graph is the text."""

    def __init__(self):
        self.texts: list[str] = []

    def __call__(self, text):
        self.texts.append(text)
        return text


class _ToyIR(IRClient):
    """Parses each specialization's TTIR through a ParseCache; one
    ConfigVerdict per compiled or failed config."""

    NAME = "toy_ir"
    LAUNCH = "skip"
    IR_STAGES = frozenset({"ttir"})

    def __init__(self, reader=None):
        super().__init__()
        self.parses = (
            ParseCache() if reader is None else ParseCache(reader, refusal=_StubRefusal)
        )
        self.logs: list[tuple] = []
        self.outcomes: list = []

    def analyze_launch(self, log):
        self.logs.append((log.specializations, log.failures))
        per_config = []
        for spec in log.specializations:
            outcome = self.parses.get(spec.artifacts.stages["ttir"])
            self.outcomes.append(outcome)
            if outcome.refusal is not None:
                refusal = Refusal.from_exception(outcome.refusal)
                per_config.append(
                    ConfigVerdict(spec.specialization, spec.config, "refused", refusal)
                )
            else:
                status = "parsed" if outcome.error is None else "error"
                per_config.append(
                    ConfigVerdict(spec.specialization, spec.config, status)
                )
        for failure in log.failures:
            per_config.append(ConfigVerdict(None, failure.config, "compile-failed"))
        return [], IRVerdict(self.NAME, "ok", per_config=per_config)

    def on_analysis_error(self, exc):
        return IRVerdict(self.NAME, "error", notes=[f"{type(exc).__name__}: {exc}"])


def _make_add_one():
    @triton.jit
    def add_one(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1, mask=mask)

    return add_one


def _make_autotuned(*blocks):
    @triton.autotune(
        configs=[triton.Config({"BLOCK": b}, num_warps=1) for b in blocks],
        key=["n"],
    )
    @triton.heuristics({"EVEN": lambda args: args["n"] % args["BLOCK"] == 0})
    @triton.jit
    def add_one_tuned(x_ptr, out_ptr, n, BLOCK: tl.constexpr, EVEN: tl.constexpr):
        tl.static_assert(BLOCK <= 32)
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1, mask=mask)

    return add_one_tuned


def _grid(meta):
    return (triton.cdiv(meta["n"], meta["BLOCK"]),)


def _inputs(n=64):
    x = torch.arange(n, dtype=torch.float32)
    return x, torch.zeros_like(x)


def test_launch_records_carry_the_ir_verdict():
    reader = _StubReader()
    ir = _ToyIR(reader)
    traced = tilelens.trace(ir)(_make_add_one())
    x, out = _inputs()

    kernel = traced[(4,)](x, out, 64, BLOCK=16)

    # LAUNCH="skip": compiled and analyzed, never launched.
    assert torch.equal(out, torch.zeros_like(x))
    launch = trace_module.launches[-1]
    assert launch.records == [ir.last_verdict]
    verdict = ir.last_verdict
    assert verdict == IRVerdict(
        "toy_ir",
        "ok",
        per_config=(ConfigVerdict(kernel.hash, {}, "parsed"),),
    )
    (text,) = reader.texts
    assert text == kernel.asm["ttir"] and "tt.func" in text

    ((spec,), failures) = ir.logs[0]
    assert failures == ()
    assert spec.artifacts.stages.keys() == {"ttir"}
    meta = spec.artifacts.meta
    assert (meta["backend"], meta["name"], meta["num_warps"]) == ("cuda", "add_one", 4)
    # Compiled for the default target, through TTIR only: no shared-memory
    # size yet.
    assert (meta["arch"], meta["shared"]) == (89, None)
    (binding,) = spec.bindings
    assert binding.error is None
    assert binding.tensors.keys() == {"x_ptr", "out_ptr"}
    facts = binding.tensors["x_ptr"]
    assert (facts.data_ptr, facts.numel, facts.elem_size) == (x.data_ptr(), 64, 4)
    assert (facts.shape, facts.strides, facts.dtype) == ((64,), (1,), "torch.float32")
    assert facts.allocation_interval() == (x.data_ptr(), x.data_ptr() + 256)
    assert dict(binding.params) == {"n": 64}
    assert dict(binding.constexprs) == {"BLOCK": 16}
    assert binding.grid == (4, 1, 1)


def test_autotune_gives_one_config_verdict_per_config():
    reader = _StubReader()
    ir = _ToyIR(reader)
    traced = tilelens.trace(ir)(_make_autotuned(16, 32))
    x, out = _inputs()

    traced[_grid](x, out, 64)
    first = ir.last_verdict
    traced[_grid](x, out, 64)

    for verdict in (first, ir.last_verdict):
        assert [c.status for c in verdict.per_config] == ["parsed", "parsed"]
        configs = [c.config for c in verdict.per_config]
        assert [(c["BLOCK"], c["num_warps"], c["EVEN"]) for c in configs] == [
            (16, 1, True),
            (32, 1, True),
        ]
        assert len({c.specialization for c in verdict.per_config}) == 2
    # The second launch finds both TTIR texts in the parse cache.
    assert len(reader.texts) == 2
    assert [launch.records for launch in trace_module.launches[-2:]] == [
        [first],
        [ir.last_verdict],
    ]


def test_a_config_that_fails_to_compile_is_recorded():
    ir = _ToyIR(_StubReader())
    # BLOCK=64 trips the kernel's static_assert.
    traced = tilelens.trace(ir)(_make_autotuned(16, 64))
    x, out = _inputs()

    traced[_grid](x, out, 64)

    parsed, failed = ir.last_verdict.per_config
    assert (parsed.status, parsed.config["BLOCK"]) == ("parsed", 16)
    assert (failed.status, failed.specialization) == ("compile-failed", None)
    assert (failed.config["BLOCK"], failed.config["num_warps"]) == (64, 1)
    ((spec,), (failure,)) = ir.logs[0]
    assert spec.config["BLOCK"] == 16
    assert isinstance(failure.error, CompileTimeAssertionFailure)
    assert failure.target == GPUTarget("cuda", 89, 32)  # the default


def test_another_triton_release_compiles_nothing_and_goes_on(monkeypatch):
    """IR mode runs on Triton 3.8 only: on any other release the host
    compile is unavailable, so the launch's config fails to compile, saying
    why, and the program goes on."""
    from tilelens.core.host_compile import host_compile_unavailable, triton_api

    reader = _StubReader()
    ir = _ToyIR(reader)
    traced = tilelens.trace(ir)(_make_add_one())
    x, out = _inputs()
    triton_api.cache_clear()  # a refusal is not cached
    monkeypatch.setattr(triton, "__version__", "3.6.0")

    assert traced[(4,)](x, out, 64, BLOCK=16) is None

    (config,) = ir.last_verdict.per_config
    assert config.status == "compile-failed" and reader.texts == []
    ((), (failure,)) = ir.logs[0]
    unavailable = host_compile_unavailable(failure.error)
    assert "supports Triton 3.8.x only" in str(unavailable)
