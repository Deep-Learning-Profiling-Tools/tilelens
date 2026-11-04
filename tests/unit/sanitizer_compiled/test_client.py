"""tilelens.clients.sanitizer.compiled.client: the CompiledSanitizer, its
factory and its reports, on fake launches.

CPU only: fake compiled kernels hold the IR tests' TTIR (tests/unit/ir/
ttir_corpus.py, compiled at test time) or small
TTIR texts whose locs point into a kernel source the test writes, and the
launch events are the core's own, built from CPU tensors. The real-kernel
counterparts (host-compiled, CPU too) live in
tests/end_to_end/test_compiled_sanitizer.py.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import triton.language as tl

import tilelens
import tilelens.ir
from tilelens.clients import CompiledSanitizer as ExportedCompiledSanitizer
from tilelens.clients import CompiledSanitizerRecord as ExportedRecord
from tilelens.clients import Sanitizer
from tilelens.clients.sanitizer.compiled import CompiledSanitizer, SanitizerKind
from tilelens.clients.sanitizer.data import CompiledSanitizerRecord
from tilelens.clients.sanitizer.sanitizer import NullSanitizer, SymbolicSanitizer
from tilelens.core.client import ClientManager, LaunchCall
from tilelens.core.config import config as cfg
from tilelens.core.data import Load, Store
from tilelens.ir import ConfigVerdict, IRClient, IRVerdict, ParseCache
from tilelens.ir.capture import CompiledArtifacts, CompiledSpecialization
from tilelens.ir.launch import tensor_facts
from tilelens.ir.ttir_reader import TTIRKind, UnsupportedTTIR, parse_ttir
from tilelens.ir.verdict import SourceLocation

from ..ir import ttir_corpus

client_module = importlib.import_module("tilelens.clients.sanitizer.compiled.client")
trace_module = importlib.import_module("tilelens.core.trace")

ADD_TTIR = ttir_corpus.text("ttir/golden_add_sm80.ttir")
GATHER_TTIR = ttir_corpus.text("ttir/golden_gather_sm80.ttir")


# ======== fakes =========


class _FakeJit:
    """What the core reads of a JITFunction to bind a launch."""

    def __init__(self, fn, constexprs=()):
        self.signature = inspect.signature(fn)
        self.params = [
            SimpleNamespace(name=name, is_constexpr=name in constexprs)
            for name in self.signature.parameters
        ]


def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE):
    pass


def _gather_kernel(idx_ptr, src_ptr, out_ptr, n_elements, BLOCK_SIZE):
    pass


def _div_kernel(p_ptr, d):
    pass


ADD = _FakeJit(_add_kernel, {"BLOCK_SIZE"})
GATHER = _FakeJit(_gather_kernel, {"BLOCK_SIZE"})
DIV = _FakeJit(_div_kernel)


class _FakeKernel:
    """A CompiledKernel as the artifact log reads it."""

    def __init__(self, key, asm):
        self.hash = f"hash-{key}"
        self.asm = asm
        self.metadata = SimpleNamespace(
            target=SimpleNamespace(backend="cuda", arch=89),
            num_warps=4,
            num_stages=3,
            shared=0,
            name=f"kernel_{key}",
        )


def _compiled(jit, args, kwargs=None, *, ttir, key="a", grid=(1,)):
    """A before_launch event: ``jit`` compiled (to ``ttir``) for this call."""
    asm = {} if ttir is None else {"ttir": ttir}
    return ClientManager._launch_event(
        jit, args, dict(kwargs or {}), grid, _FakeKernel(key, asm)
    )


def _failed(jit, args, kwargs, error, *, target=None):
    """A compile_failed event."""
    return ClientManager._launch_event(
        jit, args, dict(kwargs), (1,), None, error=error, target=target
    )


def _call(jit, *, capture=True):
    return LaunchCall(jit_fn=jit, args=(), kwargs={}, grid=None, capture=capture)


def _launch(san, jit=ADD, *, compiled=(), failures=(), capture=True, peers=()):
    """One traced launch through a ClientManager, as the core drives it."""
    manager = ClientManager([san, *peers])
    manager.begin_launch(_call(jit, capture=capture))
    for event in compiled:
        manager._dispatch_ir("before_launch", event, manager.ir_clients())
    for event in failures:
        manager._dispatch_ir("compile_failed", event, manager.ir_clients())
    manager.finalize()
    return manager.launch


def _add_args(numel=4096, n=4096):
    x, y, out = (torch.zeros(numel) for _ in range(3))
    return (x, y, out, n)


def _add_event(numel=4096, n=4096, *, blocks=4, key="a", kwargs=None):
    """golden add (BLOCK_SIZE 1024 folded, masked by n) over ``blocks``
    programs."""
    kwargs = {"BLOCK_SIZE": 1024} if kwargs is None else kwargs
    args = _add_args(numel, n)
    return _compiled(ADD, args, kwargs, ttir=ADD_TTIR, key=key, grid=(blocks,))


def _oob_event(**kwargs):
    """golden add with every lane of 5 * 1024 active over 4096 elements."""
    return _add_event(n=10**6, blocks=5, **kwargs)


def _gather_event(key="g"):
    args = (torch.zeros(64, dtype=torch.int32), torch.zeros(64), torch.zeros(64), 64)
    return _compiled(GATHER, args, {"BLOCK_SIZE": 1024}, ttir=GATHER_TTIR, key=key)


# The divide kernel: a source file whose lines the TTIR locs point at.
DIV_SOURCE = """\
@triton.jit
def div_kernel(p_ptr, d):
    pid = tl.program_id(0)
    q = pid // d
    tl.load(p_ptr + q)
"""


def _div_ttir(path: Path) -> str:
    path.write_text(DIV_SOURCE, encoding="utf-8")
    f = str(path)
    return f"""module {{
  tt.func public @div_kernel(%p: !tt.ptr<i32> loc("p_ptr"("{f}":2:0)), %d: i32 loc("d"("{f}":2:0))) attributes {{noinline = false}} {{
    %pid = tt.get_program_id x : i32 loc("{f}":3:10)
    %q = arith.divsi %pid, %d : i32 loc("{f}":4:8)
    %a = tt.addptr %p, %q : !tt.ptr<i32>, i32 loc("{f}":5:12)
    %v = tt.load %a : !tt.ptr<i32> loc("{f}":5:4)
    tt.return loc("{f}":5:4)
  }} loc("{f}":2:0)
}} loc("{f}":2:0)
"""


def _div_event(ttir, d, *, numel=4, blocks=4):
    return _compiled(
        DIV, (torch.zeros(numel, dtype=torch.int32), d), ttir=ttir, grid=(blocks,)
    )


class _Peer(IRClient):
    """Another IR client in the same trace."""

    NAME = "peer"
    IR_STAGES = frozenset()
    LAUNCH = "skip"

    def __init__(self):
        super().__init__()
        self.finalized = 0

    def analyze_launch(self, log):
        self.finalized += 1
        return [], IRVerdict(self.NAME, "ok")

    def on_analysis_error(self, exc):
        raise AssertionError(exc)


@pytest.fixture
def _isolate_sanitizer_cfg():
    saved = cfg.enable_sanitizer
    yield
    cfg.enable_sanitizer = saved


@pytest.fixture
def quiet():
    return CompiledSanitizer(abort_on_error=False)


# ======== the factory and the declarations =========


def test_factory_dispatches_on_compile(_isolate_sanitizer_cfg):
    cfg.enable_sanitizer = True
    compiled = Sanitizer(compile=True)
    assert type(compiled) is CompiledSanitizer
    # A virtual subclass: a sanitizer mode, not an eager one.
    assert isinstance(compiled, Sanitizer)
    assert not issubclass(CompiledSanitizer, SymbolicSanitizer)
    assert (compiled.abort_on_error, compiled.timeout_ms) == (True, 10_000)
    tuned = Sanitizer(compile=True, abort_on_error=False, timeout_ms=50)
    assert (tuned.abort_on_error, tuned.timeout_ms) == (False, 50)

    for eager in (Sanitizer(), Sanitizer(compile=False, abort_on_error=False)):
        assert type(eager) is SymbolicSanitizer
    assert Sanitizer(compile=False, abort_on_error=False).abort_on_error is False
    # ``compile`` is a keyword, and the eager class is never the compiled one.
    with pytest.raises(TypeError):
        Sanitizer(True, True)
    with pytest.raises(TypeError, match="Sanitizer\\(compile=True\\)"):
        SymbolicSanitizer(compile=True)
    for timeout_ms in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive int"):
            CompiledSanitizer(timeout_ms=timeout_ms)


def test_factory_initializes_the_eager_sanitizer_once(
    _isolate_sanitizer_cfg, monkeypatch
):
    cfg.enable_sanitizer = True
    calls = []
    original = SymbolicSanitizer.__init__

    def counting(self, *args, **kwargs):
        calls.append(kwargs)
        original(self, *args, **kwargs)

    monkeypatch.setattr(SymbolicSanitizer, "__init__", counting)
    Sanitizer(compile=False, abort_on_error=False)
    assert calls == [{"compile": False, "abort_on_error": False}]


def test_the_disable_flag_wins_over_compile(_isolate_sanitizer_cfg):
    cfg.enable_sanitizer = False
    off = Sanitizer(compile=True, abort_on_error=False)
    assert type(off) is NullSanitizer
    # trace() leaves a kernel traced with it untraced.
    kernel = MagicMock()
    assert tilelens.trace(off)(kernel) is kernel
    # So it does with a CompiledSanitizer built directly, like an explicit
    # SymbolicSanitizer(): the kernel then runs, and is not silently skipped.
    assert tilelens.trace(CompiledSanitizer(abort_on_error=False))(kernel) is kernel
    assert tilelens.trace(SymbolicSanitizer(abort_on_error=False))(kernel) is kernel


def test_declarations_and_composition():
    san = CompiledSanitizer()
    assert san.NAME == "compiled_sanitizer" != SymbolicSanitizer.NAME
    assert (san.IR_STAGES, san.LAUNCH, san.NEEDS_INTERPRETER) == (
        frozenset({"ttir"}),
        "skip",
        False,
    )
    assert ExportedCompiledSanitizer is CompiledSanitizer
    assert ExportedRecord is CompiledSanitizerRecord
    assert tilelens.ir.SourceLocation is SourceLocation
    # The eager sanitizer and another IR client can share its trace.
    manager = ClientManager([san, SymbolicSanitizer(abort_on_error=False), _Peer()])
    assert set(manager.clients) == {"compiled_sanitizer", "sanitizer", "peer"}


def test_the_status_is_none_until_a_launch_is_finalized(quiet):
    assert (quiet.last_status, quiet.last_verdict, quiet.records) == (None, None, [])
    _launch(quiet, compiled=[_oob_event()])
    assert quiet.last_status == "violations"
    manager = ClientManager([quiet])
    manager.begin_launch(_call(ADD))
    assert (quiet.last_status, quiet.last_verdict, quiet.records) == (None, None, [])


# ======== proofs and findings =========


def test_an_in_bounds_launch_is_ok(capsys):
    san = CompiledSanitizer()  # abort_on_error: nothing to report
    launch = _launch(san, compiled=[_add_event()])

    verdict = san.last_verdict
    assert san.last_status == "ok" and san.records == []
    assert launch.records == [verdict]
    assert verdict == IRVerdict(
        "compiled_sanitizer",
        "ok",
        # the proof holds for this launch's arguments, grid and tensors
        scope="launch",
        per_config=(ConfigVerdict("hash-a", {"BLOCK_SIZE": 1024}, "ok"),),
    )
    assert capsys.readouterr().out == ""


def test_findings_become_records_after_which_the_verdict_follows(quiet, capsys):
    event = _oob_event()
    launch = _launch(quiet, compiled=[event])

    records = quiet.records
    assert launch.records == [*records, quiet.last_verdict]
    assert [(r.kind, r.op_type, r.tensor_name) for r in records] == [
        ("out-of-bounds", Load, "x_ptr"),
        ("out-of-bounds", Load, "y_ptr"),
        ("out-of-bounds", Store, "out_ptr"),
    ]
    accesses = parse_ttir(ADD_TTIR).accesses
    for record, access, tensor in zip(records, accesses, event.bound_args.values()):
        assert record.tensor_facts == tensor_facts(tensor)
        assert record.config == {"BLOCK_SIZE": 1024}
        offset = record.violation_offset
        assert 4096 <= offset < 5120
        assert record.violation_address == tensor.data_ptr() + offset * 4
        assert record.witness["pid_0"] == 4
        (lane,) = [v for k, v in record.witness.items() if k.startswith("arange_")]
        assert 4 * 1024 + lane == offset
        (tb,) = record.user_code_tracebacks
        assert (tb.filename, tb.lineno, tb.func_name) == (
            access.loc.file,
            access.loc.line,
            "add_kernel",
        )
        assert f"{offset}" in record.detail
    (config,) = quiet.last_verdict.per_config
    assert (config.status, config.n_reports, config.refusal) == ("violations", 3, None)
    # Printed only with abort_on_error or TILELENS_VERBOSE.
    assert capsys.readouterr().out == ""


def test_a_division_by_zero_record_points_at_the_division(quiet, tmp_path):
    ttir = _div_ttir(tmp_path / "k.py")
    _launch(quiet, DIV, compiled=[_div_event(ttir, 0)])

    (record,) = quiet.records
    assert (record.kind, record.op_type, record.tensor_name) == (
        "division-by-zero",
        Load,
        "p_ptr",
    )
    assert (record.violation_offset, record.violation_address) == (None, None)
    assert set(record.witness) == {"pid_0", "pid_1", "pid_2"}
    (tb,) = record.user_code_tracebacks
    assert (tb.lineno, tb.func_name, tb.line_of_code.strip()) == (
        4,
        "div_kernel",
        "q = pid // d",
    )
    assert "divisor" in record.detail

    # d = 1: no division by zero, but pid 1..3 read past the one element.
    _launch(quiet, DIV, compiled=[_div_event(ttir, 1, numel=1)])
    (record,) = quiet.records
    assert (record.kind, record.violation_offset) == ("out-of-bounds", 1)
    assert record.user_code_tracebacks[0].line_of_code.strip() == "tl.load(p_ptr + q)"


def test_a_finding_without_a_source_location_names_its_ttir_line(quiet):
    text = (
        "module {\n  tt.func public @k(%p: !tt.ptr<i32>, %d: i32) "
        "attributes {noinline = false} {\n"
        "    %pid = tt.get_program_id x : i32\n"
        "    %q = arith.divsi %pid, %d : i32\n"
        "    %a = tt.addptr %p, %q : !tt.ptr<i32>, i32\n"
        "    %v = tt.load %a : !tt.ptr<i32>\n"
        "    tt.return\n  }\n}\n"
    )
    nameless = _FakeJit(lambda arg0, arg1: None)
    event = _compiled(nameless, (torch.zeros(4, dtype=torch.int32), 0), ttir=text)
    _launch(quiet, nameless, compiled=[event])

    (record,) = quiet.records
    assert record.user_code_tracebacks == []
    assert record.detail.endswith("(TTIR line 4)")


def test_a_loop_finding_on_an_unbound_pointer_has_no_tensor_facts(quiet):
    # The loop's bound divides by zero before its store reads anything; the
    # store's pointer is bound to no tensor (e.g. a tuple argument's part).
    text = (
        "module {\n  tt.func public @k(%p: !tt.ptr<i32>, %d: i32) "
        "attributes {noinline = false} {\n"
        "    %c0 = arith.constant 0 : i32\n"
        "    %c1 = arith.constant 1 : i32\n"
        "    %c8 = arith.constant 8 : i32\n"
        "    %u = arith.divsi %c8, %d : i32\n"
        "    scf.for %i = %c0 to %u step %c1 : i32 {\n"
        "      %a = tt.addptr %p, %i : !tt.ptr<i32>, i32\n"
        "      tt.store %a, %c0 : !tt.ptr<i32>\n"
        "    }\n"
        "    tt.return\n  }\n}\n"
    )
    nameless = _FakeJit(lambda arg0, arg1: None)
    _launch(quiet, nameless, compiled=[_compiled(nameless, (None, 0), ttir=text)])

    (record,) = quiet.records
    assert (record.kind, record.op_type, record.tensor_name) == (
        "division-by-zero",
        Store,
        "arg0",
    )
    assert record.tensor_facts is None
    # The store itself could not be checked.
    assert quiet.last_verdict.refusal.kind == "missing-binding"
    assert quiet.last_status == "violations"


# ======== configs, refusals and the union =========


def test_the_status_is_the_union_over_configs(quiet):
    _launch(
        quiet,
        compiled=[
            _add_event(key="a", kwargs={"BLOCK_SIZE": 1024, "num_warps": 4}),
            _oob_event(key="b", kwargs={"BLOCK_SIZE": 1024, "num_warps": 8}),
        ],
    )
    verdict = quiet.last_verdict
    assert verdict.status == "violations" and verdict.refusal is None
    assert [(c.specialization, c.status, c.n_reports) for c in verdict.per_config] == [
        ("hash-a", "ok", 0),
        ("hash-b", "violations", 3),
    ]
    assert {r.config["num_warps"] for r in quiet.records} == {8}

    # An unsupported config makes an otherwise clean launch unsupported, and
    # its refusal is the verdict's; a finding elsewhere still wins.
    _launch(quiet, compiled=[_add_event(), _gather_event()])
    assert quiet.last_status == "unsupported"
    assert quiet.last_verdict.refusal == quiet.last_verdict.per_config[1].refusal
    _launch(quiet, compiled=[_gather_event(), _oob_event()])
    verdict = quiet.last_verdict
    assert [c.status for c in verdict.per_config] == ["unsupported", "violations"]
    assert verdict.status == "violations"
    # Not all there is: part of the launch was not checked.
    assert verdict.refusal.kind == "indirect-address"


def test_configs_sharing_one_kernel_are_checked_and_reported_apart(quiet):
    """Two configs that compile to one kernel ("hash-a") but set a
    runtime int kwarg differently each get their own verdict, checked
    against their own binding; findings carry their config."""
    x, y, out, _ = _add_args()

    def config_event(n, blocks):
        kwargs = {"BLOCK_SIZE": 1024, "n_elements": n}
        return _compiled(ADD, (x, y, out), kwargs, ttir=ADD_TTIR, grid=(blocks,))

    _launch(quiet, compiled=[config_event(4096, 4), config_event(10**6, 5)])

    verdict = quiet.last_verdict
    assert verdict.status == "violations"
    assert [
        (c.specialization, c.config["n_elements"], c.status, c.n_reports)
        for c in verdict.per_config
    ] == [("hash-a", 4096, "ok", 0), ("hash-a", 10**6, "violations", 3)]
    assert {r.config["n_elements"] for r in quiet.records} == {10**6}

    # A kernel the reader refuses leaves every config sharing it unchecked.
    args = (torch.zeros(64, dtype=torch.int32), torch.zeros(64), torch.zeros(64))
    events = [
        _compiled(GATHER, args, {"BLOCK_SIZE": 1024, "n_elements": n}, ttir=GATHER_TTIR)
        for n in (32, 64)
    ]
    _launch(quiet, GATHER, compiled=events)
    first, second = quiet.last_verdict.per_config
    assert [c.config["n_elements"] for c in (first, second)] == [32, 64]
    assert first.status == second.status == "unsupported"
    assert first.refusal == second.refusal and first.refusal.kind == "indirect-address"


def test_configs_are_told_apart_by_their_values_not_their_reprs(quiet):
    """Bindings of one kernel group into configs by value: a tensor config
    kwarg (e.g. a heuristic's view) by its data_ptr, shape, strides and
    dtype, and reported by those facts, never its data; other values that
    print alike stay apart unless they are equal."""

    def event(value, *, oob):
        args = _add_args(n=10**6 if oob else 4096)
        kwargs = {"BLOCK_SIZE": 1024, "V": value}
        return _compiled(ADD, args, kwargs, ttir=ADD_TTIR, grid=(5 if oob else 4,))

    def per_config(first, second):
        _launch(quiet, compiled=[event(first, oob=False), event(second, oob=True)])
        return [(c.status, c.n_reports) for c in quiet.last_verdict.per_config]

    apart = [("ok", 0), ("violations", 3)]
    together = [("violations", 3)]
    view = torch.zeros(4)
    assert per_config(view, torch.zeros(4)) == apart  # one repr, two tensors
    assert per_config(_Opaque(), _Opaque()) == apart  # one repr, two objects
    assert per_config(view, view) == together
    assert per_config(tl.dtype("fp32"), tl.dtype("fp32")) == together  # equal

    per_config(view, view)
    (config,) = quiet.last_verdict.per_config
    assert config.config["V"] == (
        f"<tensor data_ptr={view.data_ptr():#x} shape=(4,) strides=(1,) "
        "dtype=torch.float32>"
    )
    assert {r.config["V"] for r in quiet.records} == {config.config["V"]}


def test_a_kernel_delivered_without_a_binding_is_never_ok(quiet):
    """Nothing was checked, so nothing is "ok": ArtifactLog delivers every
    kernel with a binding, but another producer might not."""
    artifacts = CompiledArtifacts(stages={"ttir": ADD_TTIR}, meta={"config": {}})
    spec = CompiledSpecialization("hash-x", artifacts, bindings=())
    ((records, verdict),) = quiet._check_specialization(spec)
    assert records == [] and verdict.n_reports == 0
    assert (verdict.status, verdict.refusal.kind) == ("unsupported", "internal-error")


def test_a_reader_refusal_keeps_its_kind_and_loc_on_cache_hits(quiet):
    texts = []

    def reader(text):
        texts.append(text)
        return parse_ttir(text)

    quiet.parses = ParseCache(reader, refusal=UnsupportedTTIR)
    _launch(quiet, GATHER, compiled=[_gather_event()])
    first = quiet.last_verdict
    _launch(quiet, GATHER, compiled=[_gather_event()])

    assert texts == [GATHER_TTIR]  # read once, across launches
    assert quiet.last_verdict == first
    assert (first.status, quiet.records) == ("unsupported", [])
    refusal = first.refusal
    assert refusal == first.per_config[0].refusal
    # The reader's enum kind, held as its plain string.
    assert refusal.kind == "indirect-address" and not isinstance(refusal.kind, TTIRKind)
    assert isinstance(refusal.loc, SourceLocation) and refusal.line_no is not None
    assert refusal.message.startswith(f"{refusal.loc.file}:{refusal.loc.line}: ")


def test_an_abstention_is_unsupported_with_the_sanitizers_kind(quiet):
    # A float n_elements has no binding: the mask reading it is unknown.
    x, y, out, _ = _add_args()
    event = _compiled(ADD, (x, y, out, 4096.0), {"BLOCK_SIZE": 1024}, ttir=ADD_TTIR)
    _launch(quiet, compiled=[event])

    verdict = quiet.last_verdict
    assert verdict.status == "unsupported" and quiet.records == []
    assert verdict.refusal.kind == SanitizerKind.MISSING_BINDING == "missing-binding"
    assert "n_elements" in verdict.refusal.message


def _static_assert_failure():
    from triton.compiler.errors import CompileTimeAssertionFailure

    return CompileTimeAssertionFailure(None, ast.Pass(), "BLOCK_SIZE <= 1024")


def _wrapped(cause):
    """``cause`` as Triton's code generator re-raises an error of a called
    @jit helper or builtin: a CompilationError raised from it."""
    from triton.compiler.errors import CompilationError

    wrapped = CompilationError(None, ast.Pass(), None)
    wrapped.__cause__ = cause
    return wrapped


def test_a_compile_failure_the_target_may_explain_is_unsupported(quiet):
    """A config that failed to compile for the IR target may compile for
    (and launch on) the user's GPU, unchecked: unsupported, kind
    compile-failed, naming the target and how to name another; never "ok",
    never a note."""
    from triton.backends.compiler import GPUTarget

    args = _add_args()
    cuda89 = GPUTarget("cuda", 89, 32)
    failures = [
        _failed(
            ADD,
            args,
            {"BLOCK_SIZE": 2048},
            ValueError("num_ctas > 1 requires NVIDIA SM90+ (Hopper)"),
            target=cuda89,
        ),
        # An fp8 type the target lacks: the code generator wraps it.
        _failed(
            ADD,
            args,
            {"BLOCK_SIZE": 4096},
            _wrapped(ValueError("type fp8e4nv not supported in this architecture")),
            target=cuda89,
        ),
        # An event built outside the core names no target.
        _failed(ADD, args, {"BLOCK_SIZE": 512}, RuntimeError("bad option")),
    ]
    _launch(quiet, compiled=[_add_event()], failures=failures)

    verdict = quiet.last_verdict
    assert (verdict.status, verdict.scope, verdict.notes) == ("unsupported", None, ())
    ok, *failed = verdict.per_config
    assert ok.status == "ok"
    assert [(c.specialization, c.config, c.status, c.refusal.kind) for c in failed] == [
        (None, {"BLOCK_SIZE": 2048}, "unsupported", "compile-failed"),
        (None, {"BLOCK_SIZE": 4096}, "unsupported", "compile-failed"),
        (None, {"BLOCK_SIZE": 512}, "unsupported", "compile-failed"),
    ]
    assert verdict.refusal == failed[0].refusal
    assert failed[0].refusal.message == (
        "it failed to compile for cuda:89 (ValueError: num_ctas > 1 requires "
        "NVIDIA SM90+ (Hopper)), so it was not checked; a kernel can compile for "
        "one target and fail for another, so it may launch on a GPU of another "
        "kind: to check it, name a target it compiles for "
        "(Sanitizer(compile=True, target=...), or TILELENS_IR_TARGET)"
    )
    # The innermost error, not the wrapper's source excerpt.
    assert failed[1].refusal.message.startswith(
        "it failed to compile for cuda:89 (ValueError: type fp8e4nv not supported "
        "in this architecture), so it was not checked;"
    )
    assert failed[2].refusal.message.startswith(
        "it failed to compile (RuntimeError: bad option), so it was not checked;"
    )


def test_a_failure_no_target_compiles_past_is_a_note(quiet):
    """A failing tl.static_assert (also in a called helper) or a construct
    Triton never compiles fails for every target: the config never launches
    anywhere, so it is only noted, and the launch can be "ok". Not so once
    the compile had asked for its target (e.g. tl.target_info), or for any
    other error."""
    from triton.backends.compiler import GPUTarget
    from triton.compiler.errors import UnsupportedLanguageConstruct

    from tilelens.core import host_compile

    cuda89 = GPUTarget("cuda", 89, 32)
    args = _add_args()
    nowhere = [
        _static_assert_failure(),
        _wrapped(_static_assert_failure()),
        UnsupportedLanguageConstruct(None, ast.Pass(), "nested function"),
    ]
    failures = [
        _failed(ADD, args, {"BLOCK_SIZE": 2048 * (i + 1)}, error, target=cuda89)
        for i, error in enumerate(nowhere)
    ]
    _launch(quiet, compiled=[_add_event()], failures=failures)

    verdict = quiet.last_verdict
    assert verdict.status == "ok" and len(verdict.per_config) == 1
    assert len(verdict.notes) == 3
    assert verdict.notes[0] == (
        "config {'BLOCK_SIZE': 2048} was not checked: it failed to compile for "
        "cuda:89 (CompileTimeAssertionFailure: BLOCK_SIZE <= 1024), an error of "
        "its own code whatever the target, so it never launches"
    )
    # Through the helper's wrapper: the assertion's own message.
    assert "(CompileTimeAssertionFailure: BLOCK_SIZE <= 1024)" in verdict.notes[1]
    assert "(UnsupportedLanguageConstruct: nested function)" in verdict.notes[2]

    # The same errors after a target query, or wrapped around another error,
    # may be the target's.
    queried = _static_assert_failure()
    host_compile._mark_target_queried(queried)
    maybe = [queried, _wrapped(ValueError("x")), None]
    failures = [
        _failed(ADD, args, {"BLOCK_SIZE": 2048 * (i + 1)}, error, target=cuda89)
        for i, error in enumerate(maybe)
    ]
    _launch(quiet, compiled=[_add_event()], failures=failures)
    verdict = quiet.last_verdict
    assert verdict.status == "unsupported" and verdict.notes == ()
    assert [c.refusal.kind for c in verdict.per_config[1:]] == ["compile-failed"] * 3


def test_a_host_compile_that_could_not_run_is_unsupported_not_a_note(quiet):
    """HostCompileUnavailable (Triton's compile API, or a compile that asked
    for a device) says nothing about the kernel: the config may launch, so
    it is unsupported, never a note that it cannot launch there."""
    from triton.backends.compiler import GPUTarget
    from triton.compiler.errors import CompilationError

    from tilelens.core.host_compile import HostCompileUnavailable

    cuda80 = GPUTarget("cuda", 80, 32)
    unavailable = HostCompileUnavailable("Triton asked its driver for 'utils'")
    # Triton's code generator re-raises what the kernel's code raised.
    wrapped = CompilationError("def add(...)", None, repr(unavailable))
    wrapped.__cause__ = unavailable
    failures = [
        _failed(ADD, _add_args(), {"BLOCK_SIZE": 2048}, unavailable, target=cuda80),
        _failed(ADD, _add_args(), {"BLOCK_SIZE": 4096}, wrapped, target=cuda80),
    ]
    _launch(quiet, compiled=[_add_event()], failures=failures)
    verdict = quiet.last_verdict
    assert verdict.status == "unsupported" and verdict.notes == ()
    ok, *refused = verdict.per_config
    assert ok.status == "ok"
    assert [(c.config, c.status, c.refusal.kind) for c in refused] == [
        ({"BLOCK_SIZE": 2048}, "unsupported", "host-compile-unavailable"),
        ({"BLOCK_SIZE": 4096}, "unsupported", "host-compile-unavailable"),
    ]
    assert all("asked its driver for 'utils'" in c.refusal.message for c in refused)

    # Nothing compiled: the launch's refusal says why.
    _launch(quiet, failures=failures[:1])
    verdict = quiet.last_verdict
    assert (verdict.status, verdict.refusal.kind) == (
        "unsupported",
        SanitizerKind.HOST_COMPILE_UNAVAILABLE,
    )
    assert len(verdict.per_config) == 1 and verdict.notes == ()


def test_a_launch_that_compiled_nothing_is_unsupported(quiet):
    # No JITFunction (TRITON_INTERPRET, Gluon, NKI): nothing was captured.
    _launch(quiet, capture=False)
    verdict = quiet.last_verdict
    assert (verdict.status, verdict.refusal.kind) == (
        "unsupported",
        "no-compiled-kernel",
    )
    assert "TRITON_INTERPRET" in verdict.refusal.message

    # Every config failed to compile for the target (a mixed trace goes
    # on to interpret the launch): the first one's refusal.
    from triton.backends.compiler import GPUTarget

    cuda89 = GPUTarget("cuda", 89, 32)
    failure = _failed(
        ADD, _add_args(), {"BLOCK_SIZE": 2048}, RuntimeError("bad"), target=cuda89
    )
    _launch(quiet, failures=[failure])
    verdict = quiet.last_verdict
    assert (verdict.status, verdict.refusal.kind) == ("unsupported", "compile-failed")
    (config,) = verdict.per_config
    assert config.refusal == verdict.refusal and verdict.notes == ()
    assert "for cuda:89 (RuntimeError: bad)" in verdict.refusal.message
    assert "TILELENS_IR_TARGET" in verdict.refusal.message

    # Every config failed with an error no target compiles past: compile-failed
    # too, the notes saying why.
    failures = [
        _failed(ADD, _add_args(), {"BLOCK_SIZE": b}, _static_assert_failure(), target=t)
        for b, t in ((2048, cuda89), (4096, None))
    ]
    _launch(quiet, failures=failures)
    verdict = quiet.last_verdict
    assert (verdict.status, verdict.refusal.kind) == ("unsupported", "compile-failed")
    assert verdict.per_config == () and len(verdict.notes) == 2
    assert verdict.refusal.message == (
        "no config of the launch compiled for cuda:89, so nothing was checked: "
        "each failed with an error of its own code whatever the target (see the "
        "notes)"
    )


def test_errors_no_target_decides_name_no_target(quiet):
    """A call that does not bind the kernel's parameters (the host compile
    marks it, see bind_failed) and a compile refused while the language is
    patched (no compile ran) are unsupported, never notes, but their
    refusals send nobody to another target: the untraced call raises the
    bind error on any GPU, and the refused compile says nothing about the
    kernel. (A traced launch raises a bind failure instead; only an
    event built outside the core, as here, delivers one.)"""
    from triton.backends.compiler import GPUTarget

    from tilelens.core import host_compile
    from tilelens.core.client import LanguagePatchedError

    cuda89 = GPUTarget("cuda", 89, 32)
    unbound = TypeError("dynamic_func() missing 1 required positional argument: 'n'")
    host_compile._mark_bind_failed(unbound)
    patched = LanguagePatchedError("a Triton compile cannot run while ...")
    failures = [
        _failed(ADD, _add_args(), {"BLOCK_SIZE": b}, error, target=cuda89)
        for b, error in ((2048, unbound), (4096, patched))
    ]
    _launch(quiet, compiled=[_add_event()], failures=failures)

    verdict = quiet.last_verdict
    assert verdict.status == "unsupported" and verdict.notes == ()
    ok, bind, refused = verdict.per_config
    assert ok.status == "ok"
    assert (bind.status, bind.refusal.kind) == ("unsupported", "compile-failed")
    assert bind.refusal.message == (
        "the call does not bind to the kernel's parameters (TypeError: "
        "dynamic_func() missing 1 required positional argument: 'n'), so it was "
        "not checked; Triton raises this error for the call whatever the GPU"
    )
    assert (refused.status, refused.refusal.kind) == (
        "unsupported",
        "host-compile-unavailable",
    )
    assert refused.refusal.message == "a Triton compile cannot run while ..."
    for config in (bind, refused):
        assert "TILELENS_IR_TARGET" not in config.refusal.message


def test_a_launch_no_config_of_which_compiled_prints_its_notes(capsys):
    """The refusal of a launch every config of which failed only as a note
    says "see the notes": they are printed with it, each once."""
    from triton.backends.compiler import GPUTarget

    cuda89 = GPUTarget("cuda", 89, 32)
    det = CompiledSanitizer()  # abort_on_error: prints what was not checked
    failures = [
        _failed(
            ADD, _add_args(), {"BLOCK_SIZE": b}, _static_assert_failure(), target=cuda89
        )
        for b in (2048, 4096)
    ]
    for _ in range(2):
        _launch(det, failures=failures)
    assert det.last_verdict.per_config == () and len(det.last_verdict.notes) == 2

    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "[CompiledSanitizer] not checked: compile-failed: no config of the launch "
        "compiled for cuda:89, so nothing was checked: each failed with an error "
        "of its own code whatever the target (see the notes)",
        *(f"[CompiledSanitizer] note: {note}" for note in det.last_verdict.notes),
    ]
    assert "(CompileTimeAssertionFailure: BLOCK_SIZE <= 1024)" in lines[1]


def test_a_kernel_without_ttir_is_unsupported(quiet):
    _launch(quiet, compiled=[_compiled(ADD, _add_args(), ttir=None)])
    (config,) = quiet.last_verdict.per_config
    assert (config.status, config.refusal.kind) == ("unsupported", "no-ttir")


def test_analysis_bugs_are_contained_per_config(quiet, monkeypatch):
    def reader(text):
        if "gather_kernel" in text:
            raise KeyError("reader bug")
        return parse_ttir(text)

    quiet.parses = ParseCache(reader, refusal=UnsupportedTTIR)
    real_check = client_module.check_graph

    def check_graph(graph, binding, **kw):
        if binding.grid == (2, 1, 1):
            raise ZeroDivisionError("evaluator bug")
        return real_check(graph, binding, **kw)

    monkeypatch.setattr(client_module, "check_graph", check_graph)
    _launch(
        quiet,
        compiled=[_gather_event(), _add_event(blocks=2, key="b"), _oob_event()],
    )
    reader_bug, evaluator_bug, checked = quiet.last_verdict.per_config
    assert (reader_bug.refusal.kind, evaluator_bug.refusal.kind) == (
        "internal-error",
        "internal-error",
    )
    assert "KeyError" in reader_bug.refusal.message
    assert evaluator_bug.refusal.message == "ZeroDivisionError: evaluator bug"
    # The other config's findings still count.
    assert checked.status == "violations" and len(quiet.records) == 3
    assert quiet.last_status == "violations"


def test_a_bug_outside_the_configs_is_an_internal_error(quiet, monkeypatch):
    def broken(failure):
        raise RuntimeError("note bug")

    monkeypatch.setattr(client_module, "_failure_refusal", broken)
    failure = _failed(ADD, _add_args(), {"BLOCK_SIZE": 2048}, RuntimeError("bad"))
    launch = _launch(quiet, compiled=[_oob_event()], failures=[failure])

    verdict = quiet.last_verdict
    assert (verdict.status, verdict.refusal.kind) == ("unsupported", "internal-error")
    assert verdict.refusal.message == "RuntimeError: note bug"
    assert quiet.records == [] and launch.records == [verdict]


def test_a_repeated_launch_reuses_its_check(quiet, monkeypatch):
    """A training loop: fresh tensors of the same shapes each launch. The
    check runs once; each launch's records hold its own tensors' addresses."""
    calls = []
    real = client_module.check_graph

    def counting(graph, binding, **kwargs):
        calls.append(binding)
        return real(graph, binding, **kwargs)

    monkeypatch.setattr(client_module, "check_graph", counting)
    addresses, events = [], []  # the events hold the tensors: fresh addresses
    for _ in range(3):
        event = _oob_event()
        events.append(event)
        _launch(quiet, compiled=[event])
        x = event.bound_args["x_ptr"]
        (record,) = [r for r in quiet.records if r.tensor_name == "x_ptr"]
        assert record.violation_address == x.data_ptr() + record.violation_offset * 4
        assert record.tensor_facts == tensor_facts(x)
        addresses.append(record.violation_address)
    assert len(calls) == 1 and len(set(addresses)) == 3
    assert quiet.last_status == "violations"
    # Another argument (or shape) is another check.
    _launch(quiet, compiled=[_add_event()])
    _launch(quiet, compiled=[_add_event(numel=8192, n=8192, blocks=8)])
    assert len(calls) == 3 and quiet.last_status == "ok"


def test_an_interrupt_during_the_check_is_not_contained(quiet, monkeypatch):
    """Ctrl+C is the user's: never an internal-error verdict."""

    def interrupted(graph, binding, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(client_module, "check_graph", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _launch(quiet, compiled=[_oob_event()])


# ======== reporting and abort_on_error =========


def test_abort_on_error_reports_then_exits_once_every_client_finalized(capsys):
    san, peer = CompiledSanitizer(), _Peer()
    with pytest.raises(SystemExit) as exc_info:
        _launch(san, compiled=[_oob_event()], peers=[peer])

    assert exc_info.value.code == 1
    assert peer.finalized == 1
    assert san.last_status == "violations" and len(san.records) == 3
    out = capsys.readouterr().out
    assert out.count("Out-Of-Bounds Access Detected") == 3
    assert "Tensor Arg: x_ptr" in out and "Operation: Store" in out
    assert "Witness: pid_0=4" in out and "Config: {'BLOCK_SIZE': 1024}" in out
    assert f"{san.records[0].violation_address:#x}" in out


def test_abort_on_error_reports_but_never_exits_for_unchecked_parts(capsys):
    san = CompiledSanitizer()
    _launch(san, GATHER, compiled=[_gather_event()])
    assert san.last_status == "unsupported"
    (line,) = capsys.readouterr().out.splitlines()
    assert line.startswith(
        "[CompiledSanitizer] not checked (config {'BLOCK_SIZE': 1024}): "
        "indirect-address: "
    )
    # Once per client: a kernel launched in a loop does not repeat it.
    _launch(san, GATHER, compiled=[_gather_event()])
    assert san.last_status == "unsupported"
    assert capsys.readouterr().out == ""
    # A launch-level refusal is printed too.
    _launch(san, capture=False)
    assert "not checked: no-compiled-kernel: " in capsys.readouterr().out


def test_every_config_that_failed_to_compile_is_printed_once(capsys):
    """Failed configs have no kernel: each is printed once per config and
    message, and the launch never exits for them."""
    san = CompiledSanitizer()
    failures = [
        _failed(ADD, _add_args(), {"BLOCK_SIZE": b}, ValueError("not here"))
        for b in (2048, 4096)
    ]
    _launch(san, compiled=[_add_event()], failures=failures)
    assert san.last_status == "unsupported"
    lines = capsys.readouterr().out.splitlines()
    assert [line.split(": compile-failed: ")[0] for line in lines] == [
        "[CompiledSanitizer] not checked (config {'BLOCK_SIZE': 2048})",
        "[CompiledSanitizer] not checked (config {'BLOCK_SIZE': 4096})",
    ]
    _launch(san, compiled=[_add_event()], failures=failures)
    assert capsys.readouterr().out == ""


def test_an_unchecked_op_prints_once_whatever_its_message(capsys):
    """Once per compiled kernel, kind and op: a withheld finding's message
    names an element offset that changes with the tensors' sizes."""
    printed: set = set()
    loc = SourceLocation("k.py", 7)

    def verdict(message, kind="unmodelable-condition", line_no=3):
        refusal = client_module.Refusal(kind, message, line_no, loc)
        return IRVerdict(
            "compiled_sanitizer",
            "unsupported",
            refusal=refusal,
            per_config=(ConfigVerdict("hash-a", {}, "unsupported", refusal),),
        )

    client_module.print_unchecked(verdict("at element offset 4096"), printed)
    client_module.print_unchecked(verdict("at element offset 8192"), printed)
    assert len(capsys.readouterr().out.splitlines()) == 1
    client_module.print_unchecked(verdict("offset 1", line_no=4), printed)
    client_module.print_unchecked(verdict("offset 1", "data-dependent-mask"), printed)
    assert len(capsys.readouterr().out.splitlines()) == 2


def test_verbose_prints_without_aborting(quiet, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cfg, "verbose", True)
    _launch(quiet, DIV, compiled=[_div_event(_div_ttir(tmp_path / "k.py"), 0)])
    out = capsys.readouterr().out
    assert "Division By Zero Detected" in out
    assert "Code: q = pid // d" in out
    assert "Invalid access detected" not in out


# ======== persistence =========


class _Opaque:
    def __repr__(self):
        return "<opaque>"


def test_a_launch_round_trips_through_a_saved_trace(quiet, tmp_path):
    kwargs = {"BLOCK_SIZE": 1024, "DTYPE": _Opaque()}
    launch = _launch(quiet, compiled=[_oob_event(kwargs=kwargs), _gather_event()])
    # A config value a trace cannot hold is kept as its repr.
    assert quiet.records[0].config == {"BLOCK_SIZE": 1024, "DTYPE": "<opaque>"}
    assert quiet.last_verdict.per_config[0].config["DTYPE"] == "<opaque>"

    saved = list(trace_module.launches)
    trace_module.launches[:] = [launch]
    try:
        path = tilelens.save(tmp_path / "trace.zip")
        (loaded,) = tilelens.load(path)
    finally:
        trace_module.launches[:] = saved
    assert loaded.records == launch.records
    assert all(isinstance(r, CompiledSanitizerRecord) for r in loaded.records[:-1])
    assert loaded.records[-1].per_config[1].refusal.loc == (
        launch.records[-1].per_config[1].refusal.loc
    )
