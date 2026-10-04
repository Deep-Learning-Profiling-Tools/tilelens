"""Host compile vs the JIT's own compile, for one target, on the CPU.

IR mode compiles on the host for its target instead of through
``JITFunction.run``. The JIT compiles a launch for whatever Triton's active
driver says the device is, and a stand-in driver (a device id, a stream and
a target, nothing more) lets it compile without a GPU, for any target: the
oracle for what HostCompiler lifts from ``JITFunction.run`` and
``triton.compile`` (the binder call, the options the JIT adds, the
truncated compile's hash key), wherever these tests run.

For every launch below and each of cuda:80, cuda:90 and hip:gfx942, the
JIT's kernel and the host's must be one kernel: the same hash (Triton's
name for the specialization), the same TTIR text, and the same
compiled-sanitizer verdict, down to each finding's witness. The launches cover the JIT's integer typing at the
i32/i64 boundary (2**31 - 1, 2**31, -2**31, 2**32), the equal-to-1
specialization, bool and float arguments, a tensor descriptor and the
front end's target queries (tl.target_info).
"""

from __future__ import annotations

import pytest
import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget

import tilelens
from tilelens.clients import Sanitizer
from tilelens.core.client import ClientManager, LaunchCall
from tilelens.core.host_compile import HostCompiler, parse_ir_target
from tilelens.ir.ttir_reader import UnsupportedTTIR, parse_ttir


def _real_compiles_available() -> bool:
    # Triton imported under TRITON_INTERPRET=1 builds its own standard library
    # as InterpretedFunctions, so nothing can compile for real in-process.
    import triton.language.standard as tl_standard
    from triton.runtime.jit import JITFunction

    return isinstance(tl_standard.cdiv, JITFunction)


pytestmark = pytest.mark.skipif(
    not _real_compiles_available(),
    reason="Triton was imported under TRITON_INTERPRET=1: nothing compiles in-process",
)

CUDA80 = GPUTarget("cuda", 80, 32)
# IR target -> the stand-in's device id: JITFunction.device_caches keeps a
# target per device, so every target gets a device of its own.
TARGETS = {"cuda:80": 0, "cuda:90": 1, "hip:gfx942": 2}


@pytest.fixture(autouse=True)
def _real_jit(monkeypatch, tmp_path_factory):
    # tests/unit/test_multithreading.py sets TRITON_INTERPRET=1 at import
    # time; pin the knob off so @triton.jit builds real JITFunctions. The
    # JIT's compiles go to a cache of this module's own: its kernels are
    # compiled here, never read back from another checkout's cache.
    from triton import knobs

    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    monkeypatch.setenv(
        "TRITON_CACHE_DIR", str(tmp_path_factory.getbasetemp() / "jit-cache")
    )
    missing = object()
    previous = knobs.runtime.__dict__.get("interpret", missing)
    knobs.runtime.__dict__["interpret"] = False
    yield
    if previous is missing:
        knobs.runtime.__dict__.pop("interpret", None)
    else:
        knobs.runtime.__dict__["interpret"] = previous


class _StandInDriver:
    """What JITFunction.run asks Triton's active driver for, on a machine
    whose device ``device`` is a GPU of ``target``."""

    def __init__(self, target, device):
        self.target, self.device = target, device

    def get_current_device(self):
        return self.device

    def get_current_stream(self, device=None):
        return 0

    def get_current_target(self):
        return self.target


@pytest.fixture(params=list(TARGETS))
def target(request, monkeypatch):
    """The IR target, and a stand-in driver for a device of it."""
    from triton.runtime.driver import driver

    target = parse_ir_target(request.param)
    stand_in = _StandInDriver(target, TARGETS[request.param])
    monkeypatch.setattr(type(driver), "active", property(lambda self: stand_in))
    return target


def _masked_copy():
    @triton.jit
    def masked_copy(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)

    return masked_copy


def _strided_store():
    @triton.jit
    def strided_store(x_ptr, S, BLOCK: tl.constexpr):
        off = tl.program_id(0) * S  # an i32 product for an i32 S
        tl.store(x_ptr + off + tl.arange(0, BLOCK), 1.0)

    return strided_store


def _flagged_scale():
    @triton.jit
    def flagged_scale(x_ptr, out_ptr, flag, scale, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        if flag:
            tl.store(out_ptr + offs, tl.load(x_ptr + offs) * scale)  # unmasked

    return flagged_scale


def _target_branches():
    @triton.jit
    def target_branches(x_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        if tl.target_info.cuda_capability_geq(9, 0):
            tl.store(x_ptr + offs, 1.0)  # unmasked from sm90
        elif tl.target_info.is_hip():
            tl.store(x_ptr + offs, 2.0)  # unmasked on HIP
        else:
            tl.store(x_ptr + offs, 3.0, mask=offs < n)

    return target_branches


def _descriptor_bump():
    @triton.jit
    def descriptor_bump(desc, BLOCK: tl.constexpr):
        desc.store([0, 0], desc.load([0, 0]) + 1)

    return descriptor_bump


def _descriptor():
    from triton.tools.tensor_descriptor import TensorDescriptor

    return (TensorDescriptor.from_tensor(torch.zeros(64, 64), [16, 16]),)


BOUNDARY = [2**31 - 1, 2**31, -(2**31), 2**32, 1, 0, 64]


def _cases():
    """(id, kernel factory, args builder, kwargs, grid)."""
    cases = []
    for n in BOUNDARY:
        cases.append(
            (
                f"masked_copy-n={n}",
                _masked_copy,
                lambda n=n: (torch.zeros(64), torch.zeros(64), n),
                {"BLOCK": 16},
                (8,),  # 128 lanes over 64 elements: in bounds iff n <= 64
            )
        )
        cases.append(
            (
                f"strided_store-S={n}",
                _strided_store,
                lambda n=n: (torch.zeros(64), n),
                {"BLOCK": 16},
                (3,),
            )
        )
    for flag in (True, False):
        for scale in (1.5, -0.0):
            cases.append(
                (
                    f"flagged_scale-flag={flag}-scale={scale}",
                    _flagged_scale,
                    lambda flag=flag, scale=scale: (
                        torch.zeros(64),
                        torch.zeros(64),
                        flag,
                        scale,
                    ),
                    {"BLOCK": 16},
                    (8,),  # 128 lanes over 64 elements: out of bounds if flag
                )
            )
    cases.append(
        (
            "target_branches",
            _target_branches,
            lambda: (torch.zeros(64), 64),
            {"BLOCK": 16},
            (8,),  # out of bounds on sm90+ and HIP only
        )
    )
    cases.append(
        ("descriptor_bump", _descriptor_bump, _descriptor, {"BLOCK": 16}, (1,))
    )
    return cases


CASES = _cases()


def _reading(text: str):
    try:
        return parse_ttir(text)
    except UnsupportedTTIR as e:
        return ("refused", e.kind, str(e.loc))


def _verdict_from(kernel, jit_fn, args, kwargs, grid):
    """The compiled sanitizer's verdict and findings for one launch whose
    compiled kernel is ``kernel``, delivered as the core delivers it."""
    san = Sanitizer(compile=True, abort_on_error=False)
    manager = ClientManager([san])
    manager.begin_launch(
        LaunchCall(jit_fn=jit_fn, args=args, kwargs=kwargs, grid=grid, capture=True)
    )
    event = ClientManager._launch_event(
        jit_fn, args, kwargs, grid, kernel, target=kernel.metadata.target
    )
    manager._dispatch_ir("before_launch", event, manager.ir_clients())
    manager.finalize()
    return san.last_verdict, san.records


def _summary(verdict, records):
    return (
        verdict.status,
        verdict.scope,
        None if verdict.refusal is None else verdict.refusal.kind,
        tuple((c.specialization, c.status, c.n_reports) for c in verdict.per_config),
        sorted(
            (
                r.kind,
                r.op_type.__name__,
                r.tensor_name,
                r.violation_offset,
                tuple(sorted(r.witness.items())),
            )
            for r in records
        ),
    )


@pytest.mark.parametrize(
    "make, build, kwargs, grid", [c[1:] for c in CASES], ids=[c[0] for c in CASES]
)
def test_the_jit_and_the_host_compile_the_same_kernel(
    target, make, build, kwargs, grid
):
    kernel = make()
    args = build()
    jit = kernel.warmup(*args, grid=grid, **kwargs)
    host = HostCompiler().compile(kernel, args, kwargs, target=target)

    assert jit.metadata.target == host.target == target
    assert host.hash == jit.hash
    assert host.asm["ttir"] == jit.asm["ttir"]
    from_host = _verdict_from(host, kernel, args, kwargs, grid)
    assert _summary(*from_host) == _summary(
        *_verdict_from(jit, kernel, args, kwargs, grid)
    )

    # A traced launch (the host path end to end, which never asks the
    # driver) reaches that verdict too.
    san = Sanitizer(compile=True, abort_on_error=False, target=target)
    tilelens.trace(san)(kernel)[grid](*args, **kwargs)
    assert _summary(san.last_verdict, san.records) == _summary(*from_host)


def test_the_cases_are_not_vacuous():
    """The launches exercise both verdicts, every finding kind they can,
    both integer widths, and branches the target decides."""
    statuses, kinds, widths = set(), set(), set()
    for _, make, build, kwargs, grid in CASES:
        kernel = make()
        args = build()
        host = HostCompiler().compile(kernel, args, kwargs, target=CUDA80)
        graph = _reading(host.asm["ttir"])
        if not isinstance(graph, tuple):
            widths |= {a.int_bits for a in graph.func_args if a.int_bits}
        verdict, records = _verdict_from(host, kernel, args, kwargs, grid)
        statuses.add(verdict.status)
        kinds |= {r.kind for r in records}
    assert {"ok", "violations"} <= statuses
    assert kinds == {"out-of-bounds", "integer-overflow"}
    assert {32, 64} <= widths

    by_target = {
        spec: HostCompiler()
        .compile(
            _target_branches(),
            (torch.zeros(64), 64),
            {"BLOCK": 16},
            target=parse_ir_target(spec),
        )
        .asm["ttir"]
        for spec in TARGETS
    }
    assert len(set(by_target.values())) == len(TARGETS)
