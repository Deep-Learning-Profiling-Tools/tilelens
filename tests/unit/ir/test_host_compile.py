"""tilelens.core.host_compile: IR targets and the host compile.

CPU only, and no driver: every compile here runs with Triton's driver made
unreachable (``no_driver``), as on a machine without a GPU. Kernels are
built inside the tests so their JITFunctions are real even when
TRITON_INTERPRET was set during collection.
"""

from __future__ import annotations

import importlib
import re

import pytest
import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler.errors import CompileTimeAssertionFailure

from tilelens.core.config import DEFAULT_IR_TARGET, Config
from tilelens.core.host_compile import (
    HostCompileUnavailable,
    HostCompiler,
    HostKernel,
    format_ir_target,
    parse_ir_target,
    resolve_ir_target,
    target_queried,
)

config_module = importlib.import_module("tilelens.core.config")


def _real_compiles_available() -> bool:
    # Triton imported under TRITON_INTERPRET=1 builds its own standard library
    # as InterpretedFunctions, so nothing can compile for real in-process.
    import triton.language.standard as tl_standard
    from triton.runtime.jit import JITFunction

    return isinstance(tl_standard.cdiv, JITFunction)


needs_compiles = pytest.mark.skipif(
    not _real_compiles_available(),
    reason="Triton was imported under TRITON_INTERPRET=1: nothing compiles in-process",
)


@pytest.fixture(autouse=True)
def _real_jit(monkeypatch):
    # tests/unit/test_multithreading.py sets TRITON_INTERPRET=1 at import
    # time; pin the knob off so @triton.jit builds real JITFunctions.
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


@pytest.fixture
def no_driver(unreachable_driver):
    """Make Triton's active driver unreachable, as without a GPU (where it
    raises "0 active drivers"): any driver query fails the test."""
    unreachable_driver("the host compile queried Triton's driver")


CUDA80 = GPUTarget("cuda", 80, 32)
# The default IR target (sm89: the first that compiles fp8e4nv).
CUDA89 = GPUTarget("cuda", 89, 32)


# ======== targets =========


@pytest.mark.parametrize(
    "spec, target",
    [
        ("cuda:80", CUDA80),
        (" CUDA:120 ", GPUTarget("cuda", 120, 32)),
        ("cuda:80:64", GPUTarget("cuda", 80, 64)),
        ("hip:gfx942", GPUTarget("hip", "gfx942", 64)),
        ("hip:gfx90a", GPUTarget("hip", "gfx90a", 64)),
        ("hip:gfx1100", GPUTarget("hip", "gfx1100", 32)),
        ("hip:gfx1100:64", GPUTarget("hip", "gfx1100", 64)),
        (GPUTarget("hip", "gfx950", 64), GPUTarget("hip", "gfx950", 64)),
    ],
)
def test_parse_ir_target_reads_the_documented_forms(spec, target):
    assert parse_ir_target(spec) == target
    assert parse_ir_target(format_ir_target(target)) == target


@pytest.mark.parametrize(
    "spec",
    [
        "",
        "cuda",
        "cuda:",
        "cuda:sm80",
        "sm80",
        "80",
        "cuda:80:",
        "rocm:gfx942",
        "hip:942",
        "hip:gfx942:x",
        80,
        ("cuda", 80, 32),
        GPUTarget("cuda", "80", 32),
        GPUTarget("hip", 942, 64),
        GPUTarget("cpu", "x86", 1),
        GPUTarget("cuda", 80, 0),
        # A warp size is positive in either form, a capability an int >= 70
        # (a bool is no int here), a gfx arch gfx<major><minor><stepping>.
        "cuda:80:0",
        "hip:gfx942:0",
        "cuda:60",
        "hip:gfx9",
        GPUTarget("cuda", True, 32),
        GPUTarget("cuda", 80, True),
        GPUTarget("cuda", 60, 32),
        GPUTarget("hip", "gfx9", 64),
    ],
)
def test_parse_ir_target_rejects_what_names_no_target(spec):
    with pytest.raises(ValueError, match="invalid IR target .*expected 'cuda:"):
        parse_ir_target(spec)


def test_resolve_takes_the_clients_target_else_the_configured_one(monkeypatch):
    monkeypatch.setattr(config_module.config, "ir_target", DEFAULT_IR_TARGET)
    assert resolve_ir_target("cuda:90") == GPUTarget("cuda", 90, 32)
    assert resolve_ir_target(None) == CUDA89
    monkeypatch.setattr(config_module.config, "ir_target", "hip:gfx942")
    assert resolve_ir_target(None) == GPUTarget("hip", "gfx942", 64)
    # A client's own target wins over the configured one.
    assert resolve_ir_target(CUDA80) == CUDA80
    monkeypatch.setattr(config_module.config, "ir_target", "gfx942")
    with pytest.raises(ValueError, match=r"TILELENS_IR_TARGET\) is 'gfx942'"):
        resolve_ir_target(None)
    with pytest.raises(ValueError, match="invalid IR target 'gfx942'"):
        resolve_ir_target("gfx942")


@pytest.mark.parametrize(
    "env, expected",
    [
        ({}, "cuda:89"),
        ({"TILELENS_IR_TARGET": "cuda:90"}, "cuda:90"),
    ],
)
def test_the_configured_target_comes_from_the_environment(monkeypatch, env, expected):
    monkeypatch.delenv("TILELENS_IR_TARGET", raising=False)
    monkeypatch.delenv("TRITON_VIZ_IR_TARGET", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert Config().ir_target == expected


# ======== the host compile =========


def _make_scalars():
    @triton.jit
    def scalars(x_ptr, n, flag, scale, none_arg, pair, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK) + pair[0] + n
        vals = tl.load(x_ptr + offs, mask=offs < pair[1]) * scale
        if flag:
            tl.store(x_ptr + offs, vals, mask=offs < pair[1])

    return scalars


def _make_copy():
    @triton.jit
    def copy(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)

    return copy


def _signature(ttir: str) -> dict[str, str]:
    """The TTIR entry function's arguments: name -> type."""
    header = re.search(r"tt\.func public @\w+\((.*?)\) attributes", ttir, re.S)
    assert header is not None, ttir
    return dict(re.findall(r"%([\w.]+): ([^\s{,)]+)", header.group(1)))


@needs_compiles
@pytest.mark.parametrize(
    "value, ttir_type",
    [
        (2**31 - 1, "i32"),
        (2**31, "i64"),
        (-(2**31), "i32"),
        (-(2**31) - 1, "i64"),
        (2**32, "i64"),
        (2**63, "i64"),  # u64 in the JIT's signature; TTIR integers are signless
        (16, "i32"),
        (1, None),  # the equal-to-1 specialization: a constexpr, no argument
    ],
)
def test_integers_are_typed_by_value_as_the_jit_types_them(no_driver, value, ttir_type):
    kernel = HostCompiler().compile(
        _make_copy(),
        (torch.zeros(64), torch.zeros(64), value),
        {"BLOCK": 16},
        target=CUDA80,
    )
    assert _signature(kernel.asm["ttir"]).get("n") == ttir_type


@needs_compiles
def test_a_host_compile_stops_after_ttir(no_driver):
    x = torch.zeros(64)
    kernel = HostCompiler().compile(
        _make_scalars(),
        (x, 5, True, 1.5, None, (3, 4)),
        {"BLOCK": 16, "num_warps": 2},
        target=CUDA80,
    )
    assert isinstance(kernel, HostKernel)
    assert list(kernel.asm) == ["ttir"]
    assert kernel.name == kernel.metadata.name == "scalars"
    assert kernel.target == kernel.metadata.target == CUDA80
    assert (kernel.metadata.num_warps, kernel.metadata.hash) == (2, kernel.hash)
    # Nothing after TTIR ran: no shared-memory size, no binary.
    assert not hasattr(kernel.metadata, "shared")
    # bool i1, float f32, a tuple one argument per item; None is a constexpr.
    # the tuple's items are named by their path in it
    first, second = "pair.0", "pair.1"
    assert _signature(kernel.asm["ttir"]) == {
        "x_ptr": "!tt.ptr<f32>",
        "n": "i32",
        "flag": "i1",
        "scale": "f32",
        first: "i32",
        second: "i32",
    }
    # Divisibility by 16 is specialized as the JIT does.
    assert "tt.divisibility = 16" in kernel.asm["ttir"].split("attributes")[0]


@needs_compiles
def test_compiles_are_cached_per_call_and_target(no_driver):
    compiler = HostCompiler()
    copy = _make_copy()
    x, out = torch.zeros(64), torch.zeros(64)

    def compile(n=64, target=CUDA80, **kwargs):
        return compiler.compile(
            copy, (x, out, n), {"BLOCK": 16, **kwargs}, target=target
        )

    first = compile()
    assert compile() is first
    # Another tensor with the same specialization is the same kernel.
    assert compiler.compile(copy, (torch.zeros(8), out, 64), {"BLOCK": 16}, target=CUDA80) is first  # fmt: skip
    assert compile(n=80) is first  # 80 % 16 == 0: specialized alike
    assert compile(n=65) is not first
    assert compile(num_warps=8) is not first
    hip = compile(target=GPUTarget("hip", "gfx942", 64))
    assert hip is not first and hip.hash != first.hash
    assert hip.target == GPUTarget("hip", "gfx942", 64)


@needs_compiles
def test_targets_compile_their_own_ttir(no_driver):
    """A tensor descriptor is rewritten to pointers below sm90 only, as the
    target's backend does (the target decides, not the machine)."""
    tensor_descriptor = pytest.importorskip("triton.tools.tensor_descriptor")

    @triton.jit
    def bump(desc, BLOCK: tl.constexpr):
        desc.store([0, 0], desc.load([0, 0]) + 1)

    desc = tensor_descriptor.TensorDescriptor.from_tensor(torch.zeros(64, 64), [16, 16])
    compiler = HostCompiler()
    sm80, sm90 = (
        compiler.compile(bump, (desc,), {"BLOCK": 16}, target=parse_ir_target(t))
        for t in ("cuda:80", "cuda:90")
    )
    assert "tt.descriptor_load" not in sm80.asm["ttir"]
    assert "tt.descriptor_load" in sm90.asm["ttir"]


_SCALE = tl.constexpr(2)


@needs_compiles
def test_a_changed_global_is_refused_like_the_jit_refuses_it(no_driver, monkeypatch):
    @triton.jit
    def scaled(x_ptr, BLOCK: tl.constexpr):
        tl.store(x_ptr + tl.arange(0, BLOCK) * _SCALE, 1.0)

    compiler = HostCompiler()
    call = ((torch.zeros(64),), {"BLOCK": 16})
    compiler.compile(scaled, *call, target=CUDA80)
    monkeypatch.setitem(globals(), "_SCALE", tl.constexpr(3))
    # The cached kernel read the old value: stale, not handed out.
    with pytest.raises(RuntimeError, match="_SCALE has changed since we compiled"):
        compiler.compile(scaled, *call, target=CUDA80)


def test_what_is_no_jit_function_cannot_be_host_compiled():
    with pytest.raises(HostCompileUnavailable, match="has no 'signature'"):
        HostCompiler().compile(object(), (), {}, target=CUDA80)


@needs_compiles
@pytest.mark.parametrize(
    "knob, named",
    [
        ("override", "TRITON_KERNEL_OVERRIDE"),
        ("use_ir_loc", "USE_IR_LOC"),
        ("ir_override", "'ir_override' compile option"),
        ("custom_pipeline", "a custom pipeline"),
    ],
)
def test_what_only_the_whole_pipeline_applies_is_refused(
    no_driver, monkeypatch, knob, named
):
    """A knob under which triton.compile changes the TTIR, or keys the
    kernel apart, in stages a host compile does not run: refused as
    HostCompileUnavailable, never a TTIR the device would not run."""
    from triton import knobs

    call = (_make_copy(), (torch.zeros(64), torch.zeros(64), 64))
    options = {"BLOCK": 16}
    with monkeypatch.context() as patch:
        if knob == "override":
            patch.setattr(knobs.compilation, "override", True)
        elif knob == "use_ir_loc":
            patch.setattr(knobs.compilation, "use_ir_loc", "ttir")
        elif knob == "ir_override":
            options["ir_override"] = "kernel.ttir"
        else:
            patch.setattr(
                knobs.runtime, "add_stages_inspection_hook", lambda *args: None
            )
        with pytest.raises(HostCompileUnavailable, match=re.escape(named)):
            HostCompiler().compile(*call, options, target=CUDA80)
    # Without it, the same call compiles.
    assert HostCompiler().compile(*call, {"BLOCK": 16}, target=CUDA80)


# ======== the target the front end sees =========


class _Machine:
    """A stand-in for Triton's active driver on a machine with a GPU of
    ``target``, counting the target queries it answers."""

    def __init__(self, target):
        self.target = target
        self.queries = 0

    def get_current_target(self):
        self.queries += 1
        return self.target

    def get_current_device(self):
        return 0

    def get_current_stream(self, device=None):
        return 0


def _on_machine(monkeypatch, machine):
    """Make ``machine`` Triton's active driver; None: no GPU (Triton then
    raises "0 active drivers", which tl.target_info reads as no target)."""
    from triton.runtime.driver import driver

    def active(self):
        if machine is None:
            raise RuntimeError("0 active drivers ([]). There should only be one.")
        return machine

    monkeypatch.setattr(type(driver), "active", property(active))


def _make_target_branches():
    @triton.jit
    def branches(x_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        if tl.target_info.is_cuda():
            tl.store(x_ptr + offs, 1.0)
        if tl.target_info.cuda_capability_geq(8, 9):
            tl.store(x_ptr + offs, 2.0)
        if tl.target_info.is_hip():
            tl.store(x_ptr + offs, 3.0)

    return branches


def _stored(ttir: str) -> set[float]:
    return {
        float(v)
        for v in re.findall(
            r"arith\.constant dense<([-+.e0-9]+)> : tensor<16xf32>", ttir
        )
    }


@needs_compiles
@pytest.mark.parametrize(
    "spec, stored",
    [
        ("cuda:80", {1.0}),
        ("cuda:89", {1.0, 2.0}),
        ("hip:gfx942", {3.0}),
    ],
)
def test_the_front_end_asks_the_compiles_target(no_driver, spec, stored):
    """tl.target_info reads Triton's driver; a host compile answers it with
    the compile's target (never the machine's, and here there is none)."""
    kernel = HostCompiler().compile(
        _make_target_branches(),
        (torch.zeros(16),),
        {"BLOCK": 16},
        target=parse_ir_target(spec),
    )
    assert _stored(kernel.asm["ttir"]) == stored


@needs_compiles
def test_the_ttir_does_not_depend_on_the_machine(monkeypatch):
    """Without a GPU, or on any GPU, a target's TTIR (and hash) is the same:
    the machine's driver is never asked for its target."""
    machines = [
        None,
        _Machine(GPUTarget("cuda", 89, 32)),
        _Machine(GPUTarget("cuda", 120, 32)),
        _Machine(GPUTarget("hip", "gfx942", 64)),
    ]
    targets = [parse_ir_target(t) for t in ("cuda:80", "cuda:90", "hip:gfx942")]
    seen: dict = {}
    for machine in machines:
        _on_machine(monkeypatch, machine)
        kernel = _make_target_branches()
        for target in targets:
            compiled = HostCompiler().compile(
                kernel,
                (torch.zeros(16),),
                {"BLOCK": 16},
                target=target,
            )
            seen.setdefault(target, set()).add((compiled.hash, compiled.asm["ttir"]))
        assert machine is None or machine.queries == 0
    assert all(len(compiles) == 1 for compiles in seen.values()), seen


@needs_compiles
def test_native_tma_is_the_targets(no_driver):
    """The semantic's native-TMA check (a 16-bit descriptor atomic_min)
    reads the compile's target too: fine for cuda:90, refused for cuda:80
    as on an sm80 device."""
    from triton.compiler.errors import CompilationError

    tensor_descriptor = pytest.importorskip("triton.tools.tensor_descriptor")

    @triton.jit
    def shrink(desc, BLOCK: tl.constexpr):
        desc.atomic_min([0, 0], desc.load([0, 0]))

    desc = tensor_descriptor.TensorDescriptor.from_tensor(
        torch.zeros(64, 64, dtype=torch.float16), [16, 16]
    )
    compiler = HostCompiler()
    sm90 = compiler.compile(
        shrink, (desc,), {"BLOCK": 16}, target=parse_ir_target("cuda:90")
    )
    assert "tt.descriptor_reduce" in sm90.asm["ttir"]
    with pytest.raises(CompilationError, match="native tma") as raised:
        compiler.compile(shrink, (desc,), {"BLOCK": 16}, target=CUDA80)
    # The front end asked for the target, and the answer is what failed.
    assert target_queried(raised.value)


@needs_compiles
def test_a_compile_error_says_whether_the_front_end_asked_for_the_target(no_driver):
    """target_queried marks a kernel's compile error when Triton's front end
    had asked for the compile's target before it (here tl.target_info in a
    static_assert). A failure no target query decides is not marked, even
    one the target's compile options decide (num_ctas > 1 below sm90)."""

    @triton.jit
    def bounded(x_ptr, BLOCK: tl.constexpr):
        tl.static_assert(BLOCK <= 32)
        tl.store(x_ptr + tl.arange(0, BLOCK), 1.0)

    @triton.jit
    def hopper_only(x_ptr, BLOCK: tl.constexpr):
        tl.static_assert(tl.target_info.cuda_capability_geq(9, 0))
        tl.store(x_ptr + tl.arange(0, BLOCK), 1.0)

    def error(kernel, block, target, **options):
        with pytest.raises(Exception) as raised:
            HostCompiler().compile(
                kernel,
                (torch.zeros(64),),
                {"BLOCK": block, **options},
                target=target,
            )
        return raised.value

    too_big = error(bounded, 64, CUDA89)
    assert isinstance(too_big, CompileTimeAssertionFailure)
    assert not target_queried(too_big)
    for_hopper = error(hopper_only, 16, CUDA89)
    assert isinstance(for_hopper, CompileTimeAssertionFailure)
    assert target_queried(for_hopper)
    two_ctas = error(bounded, 16, CUDA89, num_ctas=2)
    assert isinstance(two_ctas, ValueError) and "num_ctas > 1" in str(two_ctas)
    assert not target_queried(two_ctas)
    # Each compile answers for itself: the same kernel compiles for sm90.
    HostCompiler().compile(
        hopper_only,
        (torch.zeros(64),),
        {"BLOCK": 16},
        target=parse_ir_target("cuda:90"),
    )
    # No host compile raised these.
    assert not target_queried(ValueError("x")) and not target_queried(None)


@needs_compiles
def test_a_device_query_the_kernel_catches_still_counts(no_driver):
    """The host compile refuses a device query; the kernel's code may catch
    that and fall back to an answer of its own ("no big shared memory"),
    which a GPU might not give: a compile error after it is marked as one
    that asked, like one after the target query."""
    from triton.runtime.jit import constexpr_function

    @constexpr_function
    def has_big_smem():
        from triton.runtime import driver

        try:
            properties = driver.active.utils.get_device_properties(0)
        except Exception:
            return False
        return properties["max_shared_mem"] >= 200_000

    @triton.jit
    def big_smem_only(x_ptr, BLOCK: tl.constexpr):
        tl.static_assert(has_big_smem())
        tl.store(x_ptr + tl.arange(0, BLOCK), 1.0)

    with pytest.raises(CompileTimeAssertionFailure) as raised:
        HostCompiler().compile(
            big_smem_only,
            (torch.zeros(16),),
            {"BLOCK": 16},
            target=CUDA89,
        )
    assert target_queried(raised.value)


@needs_compiles
def test_a_kernel_that_keeps_its_target_answer_is_marked_after_it_asked(no_driver):
    """A kernel's code may keep the target answer (a memo) and ask only on
    its first compile: a later compile of the same kernel for the same
    target, failing on the kept answer, is marked too. Another kernel that
    never asked is not."""
    from triton.runtime.jit import constexpr_function

    @constexpr_function
    def is_hopper(_memo={}):  # noqa: B006  the kernel's own memo
        if "arch" not in _memo:
            from triton.runtime import driver

            _memo["arch"] = driver.active.get_current_target().arch
        return _memo["arch"] >= 90

    @triton.jit
    def hopper_only(x_ptr, HOPPER: tl.constexpr, BLOCK: tl.constexpr):
        if HOPPER:
            tl.static_assert(is_hopper())
        else:
            tl.static_assert(is_hopper() or True)
        tl.store(x_ptr + tl.arange(0, BLOCK), 1.0)

    @triton.jit
    def bounded(x_ptr, BLOCK: tl.constexpr):
        tl.static_assert(BLOCK <= 32)
        tl.store(x_ptr + tl.arange(0, BLOCK), 1.0)

    compiler = HostCompiler()
    x = torch.zeros(64)
    # Asks for the target, and keeps the answer.
    compiler.compile(
        hopper_only,
        (x,),
        {"HOPPER": False, "BLOCK": 16},
        target=CUDA89,
    )
    assert is_hopper.fn.__defaults__[0] == {"arch": 89}
    with pytest.raises(CompileTimeAssertionFailure) as raised:
        compiler.compile(
            hopper_only,
            (x,),
            {"HOPPER": True, "BLOCK": 16},
            target=CUDA89,
        )
    assert target_queried(raised.value)
    with pytest.raises(CompileTimeAssertionFailure) as raised:
        compiler.compile(bounded, (x,), {"BLOCK": 64}, target=CUDA89)
    assert not target_queried(raised.value)


@needs_compiles
def test_a_call_that_does_not_bind_is_marked_as_such(no_driver):
    """A call the JIT's binder rejects (a missing argument) fails on any
    device, whatever the target answers: bind_failed marks it, and it is
    never marked as having asked for the target, even for a kernel whose
    earlier compile asked. An option the target's backend does not know
    fails later, in the compile, and is no bind failure (another backend
    may know it)."""
    from tilelens.core.host_compile import bind_failed

    @triton.jit
    def on_cuda(x_ptr, n, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        if tl.target_info.is_cuda():
            tl.store(x_ptr + offs, 1.0, mask=offs < n)

    compiler = HostCompiler()
    x = torch.zeros(16)
    compiler.compile(on_cuda, (x, 16), {"BLOCK": 16}, target=CUDA89)
    with pytest.raises(TypeError, match="required positional argument: 'n'") as raised:
        compiler.compile(on_cuda, (x,), {"BLOCK": 16}, target=CUDA89)
    assert bind_failed(raised.value) and not target_queried(raised.value)
    with pytest.raises(KeyError, match="waves_per_eu") as raised:
        compiler.compile(
            on_cuda, (x, 16), {"BLOCK": 16, "waves_per_eu": 2}, target=CUDA89
        )
    assert not bind_failed(raised.value)
    assert not bind_failed(ValueError("x")) and not bind_failed(None)


@needs_compiles
def test_a_call_the_jit_cannot_key_is_marked_as_a_bind_failure(no_driver):
    """JITFunction.run keys the call right after binding it
    (compute_cache_key: the bound specialization and the call's options),
    on any device: an unhashable constexpr value fails there, whatever the
    target, so it is the call's own error too, raised as untraced."""
    from tilelens.core.host_compile import bind_failed, unknown_options

    kernel = _make_copy()
    x = torch.zeros(16)
    for target in (CUDA89, GPUTarget("hip", "gfx942", 64)):
        with pytest.raises(TypeError, match="unhashable type: 'list'") as raised:
            HostCompiler().compile(kernel, (x, x, 16), {"BLOCK": [16]}, target=target)
        assert bind_failed(raised.value) and not unknown_options(raised.value)


@needs_compiles
def test_an_option_no_backend_of_the_target_knows_is_named(no_driver):
    """The JIT's KeyError for a keyword that is neither a parameter nor an
    option of the target's backend says which keywords (unknown_options);
    it stays no bind failure: another backend may know them."""
    from tilelens.core.host_compile import bind_failed, unknown_options

    kernel = _make_copy()
    x = torch.zeros(16)
    hip = GPUTarget("hip", "gfx942", 64)
    cases = [
        (CUDA89, {"bogus": 1}, ("bogus",)),
        (CUDA89, {"waves_per_eu": 2}, ("waves_per_eu",)),
        (hip, {"maxnreg": 64}, ("maxnreg",)),
        (hip, {"bogus": 1, "maxnreg": 64}, ("bogus", "maxnreg")),
    ]
    for target, options, names in cases:
        with pytest.raises(KeyError, match="unrecognised") as raised:
            HostCompiler().compile(
                kernel, (x, x, 16), {"BLOCK": 16, **options}, target=target
            )
        assert unknown_options(raised.value) == names
        assert not bind_failed(raised.value)
    HostCompiler().compile(
        kernel, (x, x, 16), {"BLOCK": 16, "waves_per_eu": 2}, target=hip
    )
    assert unknown_options(KeyError("x")) == () and unknown_options(None) == ()


def _device_is_zero():
    # A host function a constexpr function may call from a kernel (marked
    # like tl.target_info.current_target) that asks Triton's driver for the
    # device, as a compile never should on the host.
    return triton.runtime.driver.active.get_current_device() == 0


_device_is_zero.__triton_builtin__ = True  # type: ignore[attr-defined]


@needs_compiles
def test_a_device_query_while_compiling_is_refused(no_driver):
    from triton.compiler.errors import CompilationError
    from triton.runtime.jit import constexpr_function

    from tilelens.core.host_compile import host_compile_unavailable

    @constexpr_function
    def on_device_zero():
        return _device_is_zero()

    @triton.jit
    def device_dependent(x_ptr, BLOCK: tl.constexpr):
        if on_device_zero():
            tl.store(x_ptr + tl.arange(0, BLOCK), 1.0)

    # Triton's code generator re-raises it as the kernel's CompilationError.
    with pytest.raises(CompilationError) as raised:
        HostCompiler().compile(
            device_dependent, (torch.zeros(16),), {"BLOCK": 16}, target=CUDA80
        )
    unavailable = host_compile_unavailable(raised.value)
    assert isinstance(unavailable, HostCompileUnavailable)
    assert "'get_current_device'" in str(unavailable)
    # A kernel's own compile error has none behind it, nor has one raised
    # "from None" while handling it.
    assert host_compile_unavailable(CompilationError("src", None, "bad")) is None
    try:
        try:
            raise HostCompileUnavailable("no device")
        except HostCompileUnavailable:
            raise KeyError("the kernel's") from None
    except KeyError as exc:
        assert host_compile_unavailable(exc) is None


@needs_compiles
def test_override_arch_does_not_reach_a_host_compile(no_driver, monkeypatch):
    """TRITON_OVERRIDE_ARCH retargets the JIT; a host compile stays for the
    target it was asked for, a hip one included."""
    from triton.compiler.errors import CompilationError

    monkeypatch.setenv("TRITON_OVERRIDE_ARCH", "sm90")

    @triton.jit
    def to_fp8(x_ptr, out_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, tl.load(x_ptr + offs).to(tl.float8e4nv).to(tl.float32))

    compiler = HostCompiler()
    copy = _make_copy()
    call = ((torch.zeros(64), torch.zeros(64), 64), {"BLOCK": 16})
    assert compiler.compile(copy, *call, target=CUDA80).metadata.arch == "sm80"
    # sm80's rules: no num_ctas > 1, no fp8e4nv.
    with pytest.raises(ValueError, match="num_ctas > 1 requires NVIDIA SM90"):
        compiler.compile(copy, call[0], {**call[1], "num_ctas": 2}, target=CUDA80)
    with pytest.raises(CompilationError, match="fp8e4nv not supported"):
        compiler.compile(to_fp8, call[0][:2], {"BLOCK": 16}, target=CUDA80)
    hip = compiler.compile(copy, *call, target=parse_ir_target("hip:gfx942"))
    assert hip.metadata.arch == "gfx942"

    # Where "arch" is the kernel's own parameter the arch cannot be pinned,
    # and a compile for another arch is refused, not mislabeled.
    @triton.jit
    def with_arch(x_ptr, arch, BLOCK: tl.constexpr):
        tl.store(x_ptr + tl.arange(0, BLOCK), arch)

    with pytest.raises(HostCompileUnavailable, match="arch 'sm90', not 'sm80'"):
        compiler.compile(
            with_arch, (torch.zeros(16), 1.0), {"BLOCK": 16}, target=CUDA80
        )
