"""Host compile: one JITFunction call compiled for a GPUTarget without a GPU.

IR mode reads the kernels Triton compiles, but the analysis is CPU work, so
the compile is too: :class:`HostCompiler` binds a call with the JIT's own
binder (``create_function_from_signature`` for the target's backend: the
signature types, i32 / i64 / u64 integers by value, the equal-to-1 and
divisibility specializations, tuples, tensor descriptors, constexprs,
``do_not_specialize``), packs it with ``JITFunction._pack_args`` and
compiles an ``ASTSource`` for the target, exactly as ``JITFunction.run``
would on a device of that target. No driver is queried and nothing is
loaded or launched: no ``get_current_device``, no stream, no
``_init_handles``. The JIT runtime's own hooks are not called either (the
function's ``pre_run_hooks``, ``knobs.runtime.jit_cache_hook`` /
``jit_post_compile_hook``, async compile mode): the host compile is no JIT
run.

The target is the caller's, whatever the machine has. While a thread host
compiles, Triton's ``driver.active`` answers that thread's target query
(``get_current_target``: what ``tl.target_info.is_cuda()`` /
``cuda_capability_geq()`` / ``is_hip()`` and the front end's own target
checks read) with the compile's target, and refuses any device query with
:class:`HostCompileUnavailable`; other threads, and this one outside its
compile, see Triton's own driver. The compile options name the target's
arch, so ``TRITON_OVERRIDE_ARCH`` does not reach a host compile either. A
compile that raises after its front end asked the driver anything (the
target, or a device query it refused and the kernel's code may have caught),
or after an earlier compile of the same kernel for the target did, says so
(:func:`target_queried`): what failed may be the target's answer. A call
that does not bind the kernel's parameters says that instead
(:func:`bind_failed`): the JIT raises the same for it on any device.

The pipeline stops after the backend's ``ttir`` passes (the same passes
``triton.compile`` runs): the front end, then TTIR, never ``ttgir`` /
``llir`` / the binary. Such a truncated compile is kept in memory only (a
``HostKernel``), never in Triton's on-disk cache, whose entries must hold
the whole pipeline. What only ``triton.compile``'s whole pipeline applies
to the TTIR is refused as :class:`HostCompileUnavailable` rather than
silently left out: ``TRITON_KERNEL_OVERRIDE``, ``USE_IR_LOC``, an
``ir_override`` compile option, and a custom pipeline
(``knobs.runtime.add_stages_inspection_hook``, which also keys the
kernel). ``TRITON_KERNEL_DUMP`` changes no stage; a host compile is just
not dumped.

The APIs used are private to Triton. :func:`triton_api` checks that they
exist and that the front end's target queries can be scoped, and the first
compile for each target host-compiles a small built-in kernel first, so a
changed API fails as :class:`HostCompileUnavailable` naming it rather than
as an error blamed on the user's kernel. They are Triton 3.8's: on any
other release triton_api refuses (tilelens.core.config.IR_TRITON_RELEASE).

Importing this module does not import Triton.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import linecache
import re
import threading
from collections import namedtuple
from collections.abc import Hashable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType, SimpleNamespace
from typing import Any

from . import config as config_module
from .config import DEFAULT_IR_TARGET


class HostCompileUnavailable(RuntimeError):
    """The host compile cannot run: the installed Triton lacks (or changed)
    an API it uses, or the compile asked for something only a device has.
    Never a kernel's own compile error."""


# The attributes a host compile's exception carries when the front end had
# asked the driver before it was raised (see target_queried), and when the
# call did not bind the kernel's parameters (see bind_failed).
_TARGET_QUERIED = "_tilelens_target_queried"
_BIND_FAILED = "_tilelens_bind_failed"
# The keyword arguments no option of the target's backend names (see
# unknown_options).
_UNKNOWN_OPTIONS = "_tilelens_unknown_options"


def target_queried(exc: BaseException | None) -> bool:
    """Whether ``exc`` was raised by a host compile whose front end had
    asked Triton's driver anything before it failed, so the failure may
    follow from the target's answer: the target
    (``driver.active.get_current_target()``: ``tl.target_info``, the
    tensor-descriptor lowering's native-TMA check, a constexpr function
    asking the driver), or anything else, which the host compile refuses
    (a device query the kernel's code caught, falling back to an answer of
    its own, is still a question about the device). Also true when an
    earlier compile of the same kernel for the same target by the same
    HostCompiler had asked: the kernel's code may keep the answer (a memo)
    and not ask again. False for any other exception, a bind failure
    (bind_failed) included. What the compile options derive from the target
    (e.g. its fp8 types, or whether ``num_ctas > 1`` is allowed) is no
    query. An answer the kernel's code keeps from a compile this
    HostCompiler did not run (another trace's, another target's, the
    untraced program's), and never asks for again, cannot be seen."""
    return exc is not None and getattr(exc, _TARGET_QUERIED, False) is True


def bind_failed(exc: BaseException | None) -> bool:
    """Whether ``exc`` was raised by a host compile while binding the call
    to the kernel's parameters (the JIT's binder: a missing or unexpected
    argument, an argument of a type Triton cannot pass) or keying it
    (``compute_cache_key``: e.g. an unhashable constexpr value): no target,
    and no compile, decides it, so ``JITFunction.run`` raises the same for
    the call on any device, and a traced launch raises it as is (see
    tilelens.core.client.ClientManager.ir_capture). False for any other
    exception, e.g. the KeyError for a keyword that names neither a
    parameter nor an option of the target's backend (another backend may
    know it, see unknown_options), and for a HostCompileUnavailable."""
    return exc is not None and getattr(exc, _BIND_FAILED, False) is True


def _mark(exc: BaseException, attr: str) -> None:
    try:
        setattr(exc, attr, True)
    except Exception:  # an exception type that takes no attribute
        pass


def _mark_target_queried(exc: BaseException) -> None:
    _mark(exc, _TARGET_QUERIED)


def _mark_bind_failed(exc: BaseException) -> None:
    _mark(exc, _BIND_FAILED)


def unknown_options(exc: BaseException | None) -> tuple[str, ...]:
    """The call's keyword arguments that name neither a parameter of the
    kernel nor a compile option of the target's backend, when ``exc`` is
    the KeyError the JIT raises for them (``JITFunction._pack_args``); ()
    for any other exception. Such a call fails on every device whose
    backend does not know them (a misspelled option: on every GPU), yet
    another backend may know them (e.g. HIP's ``waves_per_eu``), so the
    compile failure is the target's, not the call's (not bind_failed)."""
    names = getattr(exc, _UNKNOWN_OPTIONS, ()) if exc is not None else ()
    return names if isinstance(names, tuple) else ()


def _mark_unknown_options(
    exc: BaseException, jit_fn: Any, backend: Any, kwargs: Mapping[str, Any]
) -> None:
    # JITFunction._pack_args's own check: a keyword in neither the parsed
    # options nor the signature. Parsing again is how the JIT reads the
    # options; if parsing is what failed, nothing is marked.
    try:
        known = vars(backend.parse_options(dict(kwargs)))
    except Exception:
        return
    params = {param.name for param in jit_fn.params}
    names = tuple(k for k in kwargs if k not in known and k not in params)
    if names:
        try:
            setattr(exc, _UNKNOWN_OPTIONS, names)
        except Exception:
            pass


def _mark_call_error(exc: BaseException) -> None:
    """Mark ``exc``, raised while binding or keying the call, as the call's
    own error (bind_failed), unless the host compile could not run."""
    if host_compile_unavailable(exc) is None:
        _mark_bind_failed(exc)


def host_compile_unavailable(exc: BaseException) -> HostCompileUnavailable | None:
    """The HostCompileUnavailable behind ``exc``: ``exc`` itself, or one it
    was raised from or while handling (Triton's code generator re-raises
    what a kernel's code raised as a CompilationError from it); None if
    there is none, i.e. ``exc`` is the kernel's own compile error."""
    seen: set[int] = set()
    link: BaseException | None = exc
    while link is not None and id(link) not in seen:
        if isinstance(link, HostCompileUnavailable):
            return link
        seen.add(id(link))
        # The chain a traceback shows: the cause, else the unsuppressed context.
        if link.__cause__ is not None:
            link = link.__cause__
        else:
            link = None if link.__suppress_context__ else link.__context__
    return None


# ─────────────────────────── targets ───────────────────────────

_TARGET_FORMS = (
    "'cuda:<compute capability>' (e.g. 'cuda:80', 'cuda:90'), "
    "'hip:<gfx arch>' (e.g. 'hip:gfx942'), either optionally followed by "
    "':<warp size>', or a triton.backends.compiler.GPUTarget"
)
_RE_CUDA = re.compile(r"cuda:(\d+)(?::(\d+))?")
# gfx<major><minor><stepping>: gfx90a, gfx942, gfx1100, ...
_RE_GFX = r"gfx\d{1,2}[0-9a-z]{2}"
_RE_HIP = re.compile(rf"hip:({_RE_GFX})(?::(\d+))?")
# Volta: no Triton release targets an older NVIDIA GPU.
_MIN_CUDA_CAPABILITY = 70


def _is_int(value: Any) -> bool:
    # A bool is an int, but no capability or warp size.
    return isinstance(value, int) and not isinstance(value, bool)


def _checked_target(target: Any, spec: Any) -> Any:
    backend, arch, warp_size = target.backend, target.arch, target.warp_size
    valid = (
        backend == "cuda"
        and _is_int(arch)
        and arch >= _MIN_CUDA_CAPABILITY
        or backend == "hip"
        and isinstance(arch, str)
        and re.fullmatch(_RE_GFX, arch) is not None
    )
    if not valid or not _is_int(warp_size) or warp_size <= 0:
        raise ValueError(
            f"invalid IR target {spec!r}: expected {_TARGET_FORMS}; a CUDA "
            f"compute capability is at least {_MIN_CUDA_CAPABILITY}, a warp "
            "size positive"
        )
    return target


@functools.lru_cache(maxsize=64)
def _parse_target_spec(spec: str) -> Any:
    from triton.backends.compiler import GPUTarget

    text = spec.strip().lower()
    if match := _RE_CUDA.fullmatch(text):
        capability, warp_size = match.group(1), match.group(2)
        target = GPUTarget("cuda", int(capability), int(warp_size) if warp_size else 32)
    elif match := _RE_HIP.fullmatch(text):
        gfx, warp_size = match.group(1), match.group(2)
        # CDNA (gfx9*) runs 64-wide wavefronts, RDNA 32-wide.
        default = 64 if gfx.startswith("gfx9") else 32
        target = GPUTarget("hip", gfx, int(warp_size) if warp_size else default)
    else:
        raise ValueError(f"invalid IR target {spec!r}: expected {_TARGET_FORMS}")
    return _checked_target(target, spec)


def parse_ir_target(spec: Any) -> Any:
    """The ``GPUTarget`` an IR target spec names: a ``GPUTarget`` itself, or
    a string such as ``"cuda:89"``, ``"cuda:90"``, ``"hip:gfx942"`` or
    ``"hip:gfx1100:32"``. Raises ValueError for anything else, a CUDA
    compute capability below 70 or a warp size that is not positive
    included."""
    from triton.backends.compiler import GPUTarget

    if isinstance(spec, GPUTarget):
        return _checked_target(spec, spec)
    if isinstance(spec, str):
        return _parse_target_spec(spec)
    raise ValueError(f"invalid IR target {spec!r}: expected {_TARGET_FORMS}")


def format_ir_target(target: Any) -> str:
    """A GPUTarget as the spec parse_ir_target reads back, e.g.
    ``"cuda:89"``; the warp size only where it is not the default."""
    backend = getattr(target, "backend", None)
    arch = getattr(target, "arch", None)
    warp_size = getattr(target, "warp_size", None)
    if backend not in ("cuda", "hip"):
        return repr(target)
    spec = f"{backend}:{arch}"
    try:
        default = _parse_target_spec(spec).warp_size
    except ValueError:
        return repr(target)
    return spec if warp_size == default else f"{spec}:{warp_size}"


def resolve_ir_target(requested: Any = None) -> Any:
    """The ``GPUTarget`` for a client's ``ir_target``: ``requested`` when it
    is set, else the configured default (``tilelens.config.ir_target``, from
    ``TILELENS_IR_TARGET``, else ``"cuda:89"``)."""
    if requested is not None:
        return parse_ir_target(requested)
    spec = config_module.config.ir_target
    try:
        return parse_ir_target(spec)
    except ValueError as exc:
        raise ValueError(
            f"tilelens.config.ir_target (TILELENS_IR_TARGET) is {spec!r}, which "
            f"is no IR target: {exc}"
        ) from None


def default_ir_target() -> Any:
    """``GPUTarget("cuda", 89, 32)`` (sm89 is the first capability Triton
    compiles fp8e4nv for)."""
    return parse_ir_target(DEFAULT_IR_TARGET)


def _target_arch(target: Any) -> str | None:
    """``target``'s ``arch`` compile option, as its backend's
    parse_options derives it unless TRITON_OVERRIDE_ARCH says otherwise;
    None for a backend this module does not know."""
    if target.backend == "cuda":
        return f"sm{target.arch}"
    if target.backend == "hip":
        return str(target.arch)
    return None


# ─────────────────── the target Triton's front end sees ───────────────────

_MISSING = object()


class _TargetDriver:
    """``triton.runtime.driver.active`` on a thread while it host-compiles:
    it answers the target query with the compile's target, so Triton's
    front end (``tl.target_info``, its own target checks, a user's
    constexpr function) sees the target the kernel is compiled for, never
    the machine's device. The host has no device, stream or device
    property to give, so anything else is refused, with one exception:
    ``CompiledKernel.__del__`` unloads a module through the driver, see
    _UnloadOnlyUtils."""

    def __init__(self, target: Any, set_aside: Any) -> None:
        self._target = target
        # A context manager factory that sets this thread's scope aside
        # (_ScopedActiveDriver.set_aside), for unloading (_UnloadOnlyUtils).
        self._set_aside = set_aside
        # Whether the driver was asked anything (see target_queried).
        self.queried = False

    def get_current_target(self) -> Any:
        self.queried = True
        return self._target

    @property
    def utils(self) -> Any:
        return _UnloadOnlyUtils(self, self._set_aside)

    def _refuse(self, name: str) -> Any:
        self.queried = True
        raise HostCompileUnavailable(
            f"compiling for {format_ir_target(self._target)} on the host, "
            f"Triton asked its driver for {name!r}: a host compile has no "
            "device to ask and answers only the target query"
        )

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return self._refuse(name)


class _UnloadOnlyUtils:
    """``driver.active.utils`` on a thread while it host-compiles.
    ``CompiledKernel.__del__`` unloads a loaded module through it: a kernel
    a real launch loaded can be collected on any thread, in the middle of a
    host compile too. ``unload_module`` releases the module through the
    driver the thread has outside its compile, which loaded it; it asks
    nothing about the device, so the compile is not marked as having asked
    (see target_queried). Anything else is refused as any device query."""

    def __init__(self, scoped: _TargetDriver, set_aside: Any) -> None:
        self._scoped = scoped
        self._set_aside = set_aside

    def unload_module(self, module: Any) -> Any:
        from triton.runtime.driver import driver

        with self._set_aside():
            return driver.active.utils.unload_module(module)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return self._scoped._refuse(f"utils.{name}")


class _ScopedActiveDriver:
    """Thread-scoped ``driver.active`` (see _TargetDriver).

    While any thread host-compiles, the DriverConfig class's ``active``
    property is wrapped: a compiling thread gets its _TargetDriver, every
    other thread (and the compiling one outside its compile) whatever
    ``active`` was before, i.e. Triton's own driver or a test's stand-in.
    The last compile to end puts the class attribute back, unless someone
    replaced the wrapper in the meantime. Replacing the process-wide active
    driver instead would hand the target driver to another thread's real
    launch.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._local = threading.local()
        self._depth = 0
        self._owner: Any = None
        self._previous: Any = _MISSING
        self._wrapper: Any = None

    @contextmanager
    def targeting(self, config_cls: type, target: Any) -> Iterator[_TargetDriver]:
        """Scope ``driver.active`` on this thread to a _TargetDriver for
        ``target``, which is yielded."""
        with self._lock:
            if self._depth == 0:
                self._install(config_cls)
            self._depth += 1
        saved = getattr(self._local, "driver", None)
        scoped = self._local.driver = _TargetDriver(target, self.set_aside)
        try:
            yield scoped
        finally:
            self._local.driver = saved
            with self._lock:
                self._depth -= 1
                if self._depth == 0:
                    self._uninstall()

    @contextmanager
    def set_aside(self) -> Iterator[None]:
        """This thread's scope set aside: ``driver.active`` is what it is
        outside every host compile of the thread."""
        saved = getattr(self._local, "driver", None)
        self._local.driver = None
        try:
            yield
        finally:
            self._local.driver = saved

    def _install(self, config_cls: type) -> None:
        fallback = inspect.getattr_static(config_cls, "active")
        local = self._local

        def active(config: Any) -> Any:
            scoped = getattr(local, "driver", None)
            if scoped is not None:
                return scoped
            return fallback.__get__(config, type(config))

        self._owner = config_cls
        self._previous = config_cls.__dict__.get("active", _MISSING)
        self._wrapper = property(active)
        setattr(config_cls, "active", self._wrapper)

    def _uninstall(self) -> None:
        owner, wrapper = self._owner, self._wrapper
        if owner is not None and owner.__dict__.get("active") is wrapper:
            if self._previous is _MISSING:
                delattr(owner, "active")
            else:
                setattr(owner, "active", self._previous)
        self._owner, self._previous, self._wrapper = None, _MISSING, None


_SCOPED_DRIVER = _ScopedActiveDriver()


# ─────────────────────────── Triton's API ───────────────────────────


def _unavailable(version: str, what: str) -> HostCompileUnavailable:
    return HostCompileUnavailable(
        f"IR mode compiles kernels on the host with Triton's private compile "
        f"API, and on Triton {version} {what}"
    )


@functools.lru_cache(maxsize=1)
def triton_api() -> SimpleNamespace:
    """The Triton internals the host compile uses, checked for presence,
    and the front end's target queries checked to answer a scoped target
    (see _TargetDriver). Raises HostCompileUnavailable naming what is
    missing or does not behave so; a failure is not cached."""
    import triton

    unsupported = config_module.ir_triton_unsupported()
    if unsupported is not None:
        raise HostCompileUnavailable(unsupported)

    def missing(what: str) -> HostCompileUnavailable:
        return _unavailable(triton.__version__, f"it lacks {what}")

    try:
        from triton import knobs
        from triton._C.libtriton import get_cache_invalidating_env_vars, ir
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource, get_cache_key, make_backend
        from triton.compiler.compiler import filter_traceback
        from triton.runtime.driver import driver
        from triton.runtime.jit import (
            JITFunction,
            compute_cache_key,
            create_function_from_signature,
        )
    except ImportError as exc:
        raise missing(str(exc)) from exc
    for owner, name, attr in (
        (ir, "triton._C.libtriton.ir", "context"),
        (ir, "triton._C.libtriton.ir", "load_dialects"),
        (ASTSource, "ASTSource", "make_ir"),
        (knobs.runtime, "knobs.runtime", "debug"),
        (knobs.compilation, "knobs.compilation", "instrumentation_mode"),
    ):
        if not hasattr(owner, attr):
            raise missing(f"{name}.{attr}")
    if not isinstance(inspect.getattr_static(type(driver), "active", None), property):
        raise missing(
            "triton.runtime.driver.driver.active as a property of its class, "
            "which the host compile scopes to answer the target query"
        )
    for target in (GPUTarget("cuda", 80, 32), GPUTarget("cuda", 90, 32)):
        try:
            with _SCOPED_DRIVER.targeting(type(driver), target):
                wrong = _unscoped_target_queries(target)
        except Exception as exc:
            raise _unavailable(
                triton.__version__,
                "its front end's target queries could not be asked "
                f"({type(exc).__name__}: {exc})",
            ) from exc
        if wrong:
            raise _unavailable(
                triton.__version__,
                f"its front end's target queries answer {wrong} while compiling "
                f"for {format_ir_target(target)} on the host",
            )
    return SimpleNamespace(
        version=triton.__version__,
        knobs=knobs,
        ir=ir,
        get_cache_invalidating_env_vars=get_cache_invalidating_env_vars,
        GPUTarget=GPUTarget,
        ASTSource=ASTSource,
        get_cache_key=get_cache_key,
        make_backend=make_backend,
        compute_cache_key=compute_cache_key,
        create_function_from_signature=create_function_from_signature,
        filter_traceback=filter_traceback,
        driver_config=type(driver),
        JITFunction=JITFunction,
    )


def _unscoped_target_queries(target: Any) -> dict[str, Any]:
    """The front end's target queries that do not answer ``target`` under
    its scope, with what they answer."""
    from triton.language import target_info
    from triton.language.semantic import TritonSemantic

    wrong: dict[str, Any] = {}
    if (got := target_info.current_target()) != target:
        wrong["tl.target_info.current_target()"] = got
    # It reads nothing of the semantic, only the driver's target.
    native = TritonSemantic._has_native_tma(None)
    if native != (target.backend == "cuda" and target.arch >= 90):
        wrong["TritonSemantic._has_native_tma()"] = native
    return wrong


# Host-compiled before the first compile for each target: scalars only (it
# needs no tensor, and no name from triton.language); ``one`` takes the
# equal-to-1 constexpr specialization. Its source is registered with
# linecache under a name of its own, so the JIT reads it from there and not
# from this file, which may have changed on disk since it was imported.
_SELF_TEST_SOURCE = """\
def _self_test_kernel(n, flag, one):
    if flag:
        n = n * one
"""
_SELF_TEST_FILE = "<tilelens host-compile self-test>"


def _self_test_jit_function(api: SimpleNamespace) -> Any:
    lines = _SELF_TEST_SOURCE.splitlines(keepends=True)
    linecache.cache[_SELF_TEST_FILE] = (
        len(_SELF_TEST_SOURCE),
        None,
        lines,
        _SELF_TEST_FILE,
    )
    namespace: dict[str, Any] = {"__name__": __name__}
    exec(compile(_SELF_TEST_SOURCE, _SELF_TEST_FILE, "exec"), namespace)
    return api.JITFunction(namespace["_self_test_kernel"])


@functools.lru_cache(maxsize=None)
def _self_test_target(target: Any) -> None:
    """Host-compile the built-in _self_test_kernel for ``target`` through its TTIR;
    raise HostCompileUnavailable if that fails. Only a success is cached."""
    api = triton_api()
    try:
        kernel = HostCompiler().compile(
            _self_test_jit_function(api),
            (5, True, 1),
            {},
            target=target,
            _self_test=True,
        )
        text = kernel.asm["ttir"]
    except Exception as exc:
        raise _unavailable(
            api.version,
            f"a built-in test kernel failed to host-compile for "
            f"{format_ir_target(target)} ({type(exc).__name__}: {exc})",
        ) from exc
    if "tt.func" not in text or "_self_test_kernel" not in text:
        raise _unavailable(
            api.version,
            f"a built-in test kernel host-compiled for {format_ir_target(target)} "
            "to no TTIR function",
        )


# ─────────────────────────── the artifact ───────────────────────────


@dataclass(frozen=True, eq=False)
class HostKernel:
    """A kernel compiled on the host through its TTIR (see the module
    docstring): what a ``CompiledKernel`` holds of it, never loaded."""

    # The specialization: what triton.compile names this kernel.
    hash: str
    name: str
    # Stage -> text, every stage compiled, in pipeline order: "ttir", and
    # any stage a backend runs before it.
    asm: Mapping[str, str | bytes] = field(repr=False)
    # A namedtuple, as CompiledKernel.metadata: "target", "name", "hash",
    # the compile options and whatever the compiled stages added.
    metadata: Any = field(repr=False)

    @property
    def target(self) -> Any:
        return self.metadata.target


# The stage a host compile stops after.
_TTIR = "ttir"


def _unsupported_pipeline(
    api: SimpleNamespace, kwargs: Mapping[str, Any]
) -> str | None:
    """What makes ``triton.compile`` change the TTIR (or the kernel's hash)
    in a way only its whole pipeline applies, which the host compile does
    not run; None when nothing does."""
    compilation = api.knobs.compilation
    if getattr(compilation, "override", False):
        return "TRITON_KERNEL_OVERRIDE (knobs.compilation.override)"
    if getattr(compilation, "use_ir_loc", None):
        return "USE_IR_LOC (knobs.compilation.use_ir_loc)"
    if kwargs.get("ir_override"):
        return "the 'ir_override' compile option"
    if getattr(api.knobs.runtime, "add_stages_inspection_hook", None) is not None:
        return "a custom pipeline (knobs.runtime.add_stages_inspection_hook)"
    return None


def _check_used_globals(jit_fn: Any) -> None:
    # JITFunction.run's check, for every kernel handed out: a kernel
    # compiled before a global it reads changed is stale.
    not_present = object()
    for (name, _), (value, globals_dict) in jit_fn.used_global_vals.items():
        if (new := globals_dict.get(name, not_present)) != value:
            raise RuntimeError(
                f"Global variable {name} has changed since we compiled this "
                f"kernel, from {value} to {new}"
            )


class HostCompiler:
    """Host compiles with an in-process cache (one per trace: the
    ClientManager's ``compiler``), keyed by the JIT's own specialization
    key (``compute_cache_key``: the bound specialization and the call's
    compile options) and the target."""

    def __init__(self) -> None:
        # (id(jit_fn), target) -> (jit_fn, backend, binder, key cache).
        self._binders: dict[Hashable, tuple[Any, Any, Any, dict]] = {}
        # (id(jit_fn), target) -> jit_fn, for each kernel a compile of which
        # for the target asked the driver (see target_queried).
        self._asked: dict[Hashable, Any] = {}
        # (id(jit_fn), specialization key, target) -> (jit_fn, kernel).
        self._kernels: dict[Hashable, tuple[Any, Any]] = {}

    def compile(
        self,
        jit_fn: Any,
        args: tuple,
        kwargs: Mapping[str, Any],
        *,
        target: Any,
        _self_test: bool = False,
    ) -> HostKernel:
        """Compile the call ``jit_fn.run(*args, **kwargs)`` would compile on
        a device of ``target`` (a GPUTarget), through its TTIR. Raises what
        the JIT's bind, pack or compile raises (a bind failure marked as
        such, see bind_failed; any other error marked when this compile, or
        an earlier one of ``jit_fn`` for ``target``, had asked the driver,
        see target_queried), or HostCompileUnavailable. (``_self_test``: the
        built-in test compile, which skips the checks it is part of.)"""
        api = triton_api()
        for attr in ("signature", "params", "_pack_args", "used_global_vals"):
            if not hasattr(jit_fn, attr):
                raise HostCompileUnavailable(
                    f"cannot host-compile {jit_fn!r}: it has no {attr!r} "
                    f"(a JITFunction of Triton {api.version} has)"
                )
        if not _self_test:
            _self_test_target(target)
        asked_key = (id(jit_fn), target)
        with _SCOPED_DRIVER.targeting(api.driver_config, target) as scoped:
            try:
                return self._compile(api, jit_fn, args, kwargs, target, _self_test)
            except Exception as exc:
                asked_before = self._asked.get(asked_key) is jit_fn
                if not bind_failed(exc) and (scoped.queried or asked_before):
                    _mark_target_queried(exc)
                raise
            finally:
                if scoped.queried:
                    self._asked[asked_key] = jit_fn

    def _compile(
        self,
        api: SimpleNamespace,
        jit_fn: Any,
        args: tuple,
        kwargs: Mapping[str, Any],
        target: Any,
        self_test: bool,
    ) -> HostKernel:
        backend, binder, key_cache = self._binder(api, jit_fn, target)
        # What JITFunction.run adds to every call's options.
        kwargs = dict(kwargs)
        kwargs["debug"] = (
            kwargs.get("debug", getattr(jit_fn, "debug", None))
            or api.knobs.runtime.debug
        )
        kwargs["instrumentation_mode"] = api.knobs.compilation.instrumentation_mode
        # The target's arch as a compile option, which the backend's
        # parse_options takes over TRITON_OVERRIDE_ARCH: a host compile is
        # for the target it was asked for. Not where "arch" is the
        # call's own (a launch option, or a kernel parameter).
        arch = _target_arch(target)
        if (
            arch is not None
            and "arch" not in kwargs
            and all(param.name != "arch" for param in jit_fn.params)
        ):
            kwargs["arch"] = arch
        try:
            bound_args, specialization, options = binder(*args, **kwargs)
        except Exception as exc:
            # The call's own error (see bind_failed); the backend only adds
            # its tensor-alignment flags to the specialization.
            _mark_call_error(exc)
            raise
        try:
            cache_key = api.compute_cache_key(key_cache, specialization, options)
        except Exception as exc:
            # The call's own error too (e.g. an unhashable constexpr value):
            # the key is the bound specialization and the call's options,
            # which JITFunction.run keys the call by right after its binder,
            # on any device.
            _mark_call_error(exc)
            raise
        unsupported = None if self_test else _unsupported_pipeline(api, kwargs)
        if unsupported is not None:
            raise HostCompileUnavailable(
                f"under {unsupported}, triton.compile changes the kernel in its "
                "whole pipeline, and a host compile runs the front end and the "
                "TTIR passes only"
            )
        key = (id(jit_fn), cache_key, target)
        cached = self._kernels.get(key)
        if cached is not None and cached[0] is jit_fn:
            kernel = cached[1]
        else:
            try:
                options, signature, constexprs, attrs = jit_fn._pack_args(
                    backend, kwargs, bound_args, specialization, options
                )
            except KeyError as exc:
                _mark_unknown_options(exc, jit_fn, backend, kwargs)
                raise
            compiled_arch = getattr(options, "arch", arch)
            if compiled_arch != arch:
                raise HostCompileUnavailable(
                    f"the call's compile options name arch {compiled_arch!r}, "
                    f"not {arch!r} of the IR target {format_ir_target(target)} "
                    "(an 'arch' launch option, or TRITON_OVERRIDE_ARCH with a "
                    "kernel parameter named 'arch'), so its host compile would "
                    "not be for the target"
                )
            source = api.ASTSource(jit_fn, signature, constexprs, attrs)
            kernel = self._compile_source(api, source, backend, target, options)
            self._kernels[key] = (jit_fn, kernel)
        _check_used_globals(jit_fn)
        return kernel

    def _binder(self, api: SimpleNamespace, jit_fn: Any, target: Any) -> tuple:
        entry = self._binders.get((id(jit_fn), target))
        if entry is None or entry[0] is not jit_fn:
            backend = api.make_backend(target)
            binder = api.create_function_from_signature(
                jit_fn.signature, jit_fn.params, backend
            )
            entry = self._binders[(id(jit_fn), target)] = (jit_fn, backend, binder, {})
        return entry[1:]

    @staticmethod
    def _compile_source(
        api: SimpleNamespace,
        source: Any,
        backend: Any,
        target: Any,
        options: Any,
    ) -> HostKernel:
        """triton.compile's front half: the front end, then the backend's
        stages through "ttir"."""
        pipeline: dict[str, Any] = {}
        backend.add_stages(pipeline, options, source.language)
        names = list(pipeline)
        if _TTIR not in names:
            raise HostCompileUnavailable(
                f"the {target.backend} backend of Triton {api.version} has no "
                f"{_TTIR!r} stage to stop after (its stages: {', '.join(names)})"
            )
        env_vars = api.get_cache_invalidating_env_vars()
        key = api.get_cache_key(source, backend, options, env_vars)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        metadata = {
            "hash": digest,
            "target": target,
            **options.__dict__,
            **env_vars,
            "triton_version": api.version,
        }
        # Keep the context referenced until every module of it is gone.
        context = api.ir.context()
        api.ir.load_dialects(context)
        backend.load_dialects(context)
        codegen_fns = backend.get_codegen_implementation(options)
        module_map = backend.get_module_map()
        try:
            module = source.make_ir(target, options, codegen_fns, module_map, context)
        except Exception as exc:
            api.filter_traceback(exc)
            raise
        asm: dict[str, str | bytes] = {}
        for name in names[: names.index(_TTIR) + 1]:
            module = pipeline[name](module, metadata)
            asm[name] = module if isinstance(module, (str, bytes)) else str(module)
        del module
        # A later stage names the entry point; up to here it is the kernel's.
        metadata.setdefault("name", source.name)
        kernel_metadata = namedtuple(  # type: ignore[misc]
            "KernelMetadata", sorted(metadata)
        )(**metadata)
        del context
        return HostKernel(
            hash=digest,
            name=metadata["name"],
            asm=MappingProxyType(asm),
            metadata=kernel_metadata,
        )
