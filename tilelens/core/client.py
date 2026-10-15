from contextlib import contextmanager, nullcontext

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import ClassVar, Any, Literal
from collections.abc import Callable, Hashable, Mapping
import inspect
import operator
import threading
import warnings

from .data import Op, Launch
from .patch import (
    patch_op,
    unpatch_op,
    patch_for_loop,
    unpatch_for_loop,
    patch_calls,
    LoopIter,
    LoopSite,
)
from functools import wraps
from .callbacks import OpCallbacks, ForLoopCallbacks
from .patch import patch_lang, unpatch_lang
from .frontend.base import LANG_PATCH_SCOPES, get_frontend
from .config import config as cfg
from .host_compile import (
    HostCompiler,
    bind_failed,
    format_ir_target,
    resolve_ir_target,
)


LaunchPreference = Literal["skip", "indifferent"]
LAUNCH_PREFERENCES: tuple[LaunchPreference, ...] = ("skip", "indifferent")
# The compiler stages an IR client can read: the host compile runs Triton's
# front end and the backend's "ttir" passes, nothing after them.
IR_STAGE_NAMES = frozenset({"ttir"})

# (jit_fn, args, kwargs) -> (args, kwargs): the arguments a host compile of
# one JITFunction call must see, supplied by the trace.
RealArgs = Callable[[Any, tuple, dict], tuple[tuple, dict]]


@dataclass(frozen=True, eq=False)
class LaunchCall:
    """One traced launch as the caller made it, delivered to ``begin_launch``.

    Mechanism-only data. ``eq=False`` keeps identity comparison, since
    field-wise equality would compare tensors.
    """

    # The traced JITFunction; None when the trace has none (TRITON_INTERPRET /
    # InterpretedFunction runner, Gluon, NKI).
    jit_fn: Any
    args: tuple
    # The caller's keyword arguments, excluding ``grid`` and ``warmup``.
    # Config kwargs added by Autotuner/Heuristics layers appear only in
    # LaunchEvent.kwargs.
    kwargs: Mapping[str, Any]
    grid: Any
    # Whether IR clients receive compile events for this launch. False when
    # there is no JITFunction (``jit_fn`` is None).
    capture: bool


@dataclass(frozen=True, eq=False)
class LaunchEvent:
    """One call into a traced JITFunction's ``run``, as delivered to IR clients.

    Mechanism-only data: what was compiled, for which target, and how the
    call was bound. What the compiled kernel means is left to each client.
    ``eq=False`` keeps identity comparison, since field-wise equality would
    compare tensors.
    """

    jit_fn: Any
    # Positional arguments as passed to JITFunction.run.
    args: tuple
    # Keyword arguments, including Autotuner/Heuristics config kwargs,
    # excluding ``grid`` and ``warmup``.
    kwargs: Mapping[str, Any]
    # The grid as passed: a tuple, a callable, or None (e.g. for a warmup).
    grid: Any
    # ``grid`` canonicalized to three dims, a callable resolved against
    # ``bound_args`` as JITFunction.run does; None if it cannot be resolved.
    resolved_grid: tuple[Any, Any, Any] | None
    # Kernel parameter name -> value, defaults applied.
    bound_args: Mapping[str, Any]
    # The kernel compiled on the host for ``target`` through its TTIR
    # (tilelens.core.host_compile): ``.asm["ttir"]`` the text, ``.metadata``
    # the compile metadata, ``.hash`` the specialization. Never loaded or
    # launched. None in a compile_failed event.
    kernel: Any
    # Identity of the compiled specialization (``kernel.hash``: what
    # triton.compile names the kernel for ``target``); None when nothing
    # was compiled.
    specialization: Hashable
    # compile_failed only: the exception the host compile raised: the
    # kernel's compile error for ``target``, or, when the host compile could
    # not run at all, a HostCompileUnavailable or an error raised from one
    # (tilelens.core.host_compile's host_compile_unavailable tells them
    # apart; its target_queried marks an error after the front end had
    # asked the driver, in this compile or an earlier one of the kernel for
    # the target), or a LanguagePatchedError (no compile ran). Never the
    # call's own bind error (host_compile.bind_failed): the core raises
    # that as the untraced call does (see ClientManager.ir_capture).
    error: BaseException | None = None
    # The GPUTarget ``kernel`` was compiled for: the receiving clients'
    # (Client.ir_target). None only for an event built outside the core.
    target: Any = None


@dataclass
class CaptureWindow:
    """What one ``ClientManager.ir_capture`` window observed."""

    # Host compiles that produced a kernel (one per call).
    compiled: int = 0
    # Host compile exceptions, in call order (one per call).
    failures: list[BaseException] = field(default_factory=list)


class Client(ABC):
    NAME: ClassVar[str]
    # Whether the client consumes the interpreted run (op/loop callbacks,
    # pre/post_run votes, arg/grid callbacks). IR clients set this to False
    # and receive compiled kernels through before_launch/after_launch.
    NEEDS_INTERPRETER: ClassVar[bool] = True
    # Compiler stages (keys of the compiled kernel's asm) an IR client reads.
    # The core compiles each call on the host through the backend's "ttir"
    # passes, so "ttir" is the one stage there is; ClientManager.add_clients
    # refuses an IR client that declares another.
    IR_STAGES: ClassVar[frozenset[str]] = frozenset()
    # An IR client's real-launch preference. The core never runs the real
    # kernel of a traced launch that has IR clients: they get the kernels
    # compiled on the host, and the interpreted run stands in for the launch
    # when an interpreting client shares the trace. So an IR client must
    # declare "skip" (ClientManager.add_clients refuses one that does not);
    # an interpreting client's LAUNCH changes nothing.
    LAUNCH: ClassVar[LaunchPreference] = "indifferent"
    # The target an IR client's kernels are compiled for: a triton
    # GPUTarget or a spec such as "cuda:90" or "hip:gfx942" (see
    # tilelens.core.host_compile.parse_ir_target); None for the configured
    # default (tilelens.config.ir_target: TILELENS_IR_TARGET, else
    # "cuda:89"). A client may set it per instance. The IR clients of one
    # trace must compile for the same target (see ClientManager.ir_target).
    ir_target: Any = None

    def __init__(self) -> None:
        # Thread-local scratch space for per-thread callback state
        self._thread_local = threading.local()
        # Lock for serializing shared state where needed
        self._lock = threading.RLock()

    def _lock_context(self):
        if cfg.num_sms > 1:
            return self._lock
        return nullcontext()

    def lock_fn(self, fn: Callable) -> Callable:
        """Forces serial execution of the given function."""

        @wraps(fn)
        def wrapped(*args, **kwargs):
            with self._lock_context():
                return fn(*args, **kwargs)

        return wrapped

    @abstractmethod
    def pre_run_callback(self, fn: Callable) -> bool:
        """
        Returns True if the function should continue running, False if it should be skipped.
        """
        ...

    @abstractmethod
    def post_run_callback(self, fn: Callable) -> bool:
        """
        Returns True if the function should continue running, False if it should be skipped.
        """
        ...

    @abstractmethod
    def arg_callback(self, name, arg, arg_cvt):
        ...

    @abstractmethod
    def grid_callback(self, grid: tuple[int, ...]):
        ...

    @abstractmethod
    def grid_idx_callback(self, grid_idx: tuple[int, ...]):
        ...

    @abstractmethod
    def register_op_callback(
        self, op_type: type[Op], *args: Any, **kwargs: Any
    ) -> OpCallbacks:
        ...

    @abstractmethod
    def register_for_loop_callback(self) -> ForLoopCallbacks:
        ...

    @abstractmethod
    def finalize(self) -> list:
        ...

    @abstractmethod
    def pre_warmup_callback(self, jit_fn: Callable, *args, **kwargs) -> bool:
        """
        Returns True if the warmup should proceed, False to skip warmup.
        """
        ...

    @abstractmethod
    def post_warmup_callback(self, jit_fn: Callable, ret: Any) -> None:
        ...

    # Each begun launch ends in exactly one of finalize() or abort_launch().
    # A client whose begin_launch raised still gets abort_launch; clients
    # after it in the trace never begin that launch.

    def begin_launch(self, call: LaunchCall) -> None:
        """Called before every traced launch; reset per-launch state here."""

    def abort_launch(self, exc: BaseException) -> None:
        """Called when a traced launch raises before finalize; ``exc`` is
        re-raised afterwards and finalize() is not called for this launch."""

    def before_launch(self, event: LaunchEvent) -> None:
        """IR clients: ``event.kernel`` was compiled on the host for the
        client's target (``event.target``).

        Fires once per traced launch for each distinct (specialization,
        binding fingerprint), whichever call produced it: a compile-only
        warmup or a config of the launch's autotuning.
        The fingerprint summarizes how the call was bound, never tensor
        data: the resolved grid, and every argument and kwarg (config kwargs
        and compile options such as num_warps included) except those to
        tl.constexpr parameters, which the specialization already tells
        apart (so a heuristic handing out an equal but fresh constexpr
        object per call adds no binding). An int/bool/float/str/None value
        counts by type and value, a tuple item by item, a tensor by its
        data_ptr, shape, strides and dtype; any other value counts by
        identity, so an equal but distinct object is another binding. Two
        configs that compile to one kernel but differ in a runtime argument
        or the grid thus get an event each, while repeated identical calls
        share one.

        A TritonTrace compiles every config before the launch goes on, so
        each config is seen on every launch, whatever the autotune cache
        holds.
        """

    def after_launch(self, event: LaunchEvent) -> None:
        """IR clients: the call described by ``event`` has finished. Only a
        call that fired before_launch gets one; the real kernel never ran
        (see LAUNCH), so it follows before_launch right away."""

    def compile_failed(self, event: LaunchEvent) -> None:
        """IR clients: a call failed to compile on the host for the
        client's target (``event.target``); ``event.error`` is the
        exception, ``event.kernel`` is None.

        Fires once per traced launch for each distinct failing call (its
        arguments and kwargs, constexprs included). Nothing is loaded: a
        kernel the device could not run (e.g. too much shared memory) still
        compiles, and is delivered like any other. A failing host compile
        never fails the launch: the target is the IR client's choice, not
        the machine's, and the launch goes on without the config. Deciding
        what a failure means for the client's result is the client's call.

        A call that does not bind the kernel's parameters is no compile
        failure and never gets here: the launch raises the binder's error,
        as the untraced call does on any device (see
        ClientManager.ir_capture), and the clients get abort_launch.
        """

    def _set_thread_local(self, key: str, value: Any) -> None:
        setattr(self._thread_local, key, value)

    def _get_thread_local(self, key: str, default: Any = None) -> Any:
        return getattr(self._thread_local, key, default)

    @property
    def grid_idx(self) -> tuple[int, ...] | None:
        return self._get_thread_local("grid_idx", None)

    @grid_idx.setter
    def grid_idx(self, value: tuple[int, ...] | None) -> None:
        self._set_thread_local("grid_idx", value)


_MISSING = object()


@contextmanager
def _instance_attr(obj: Any, name: str, value: Any):
    """Install ``value`` as an instance attribute of ``obj`` for the scope, then
    restore exactly what was there (deleting it if the class provided it)."""
    previous = getattr(obj, "__dict__", {}).get(name, _MISSING)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if previous is _MISSING:
            obj.__dict__.pop(name, None)
        else:
            setattr(obj, name, previous)


def _bind_launch_args(
    jit_fn: Any, args: tuple, kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    """Parameter name -> value for one run() call, the way JITFunction's
    binder builds ``bound_args`` (bind, then apply defaults; non-parameter
    kwargs such as num_warps are compile options, not arguments)."""
    signature = getattr(jit_fn, "signature", None)
    if not isinstance(signature, inspect.Signature):
        return {}
    params = {k: v for k, v in kwargs.items() if k in signature.parameters}
    try:
        bound = signature.bind(*args, **params)
    except TypeError:
        return {}
    bound.apply_defaults()
    return dict(bound.arguments)


def _resolve_grid(grid: Any, bound_args: Mapping[str, Any]) -> tuple | None:
    """Canonicalize a launch grid to three dims; None if it cannot be resolved."""
    if grid is None:
        return None
    try:
        resolved = tuple(grid(dict(bound_args)) if callable(grid) else grid)
    except Exception:
        return None
    if not 1 <= len(resolved) <= 3:
        return None
    return resolved + (1,) * (3 - len(resolved))


def _specialization(kernel: Any) -> Hashable:
    specialization = getattr(kernel, "hash", None)
    return id(kernel) if specialization is None else specialization


def _fingerprint_value(value: Any, pinned: dict[int, Any]) -> Hashable:
    """A hashable summary of one call value that holds no user object (see
    Client.before_launch): a plain scalar by type and value, a tensor by
    data_ptr, shape, strides and dtype (never its data), a tuple item by
    item. Anything else is a type+id token; the object is put in ``pinned``
    so its id cannot be reused by another object while the tokens are
    compared, and a distinct object never shares a token."""
    if value is None:
        return None
    if isinstance(value, (bool, int)):
        return (type(value), int(value))
    if isinstance(value, float):
        # hex() tells -0.0 from 0.0 and makes NaN equal to itself.
        return (type(value), float.hex(value))
    if isinstance(value, str):
        return (type(value), str(value))
    if isinstance(value, tuple):
        return (type(value), tuple(_fingerprint_value(v, pinned) for v in value))
    if hasattr(value, "data_ptr"):
        try:
            return (
                "tensor",
                int(value.data_ptr()),
                tuple(int(size) for size in value.shape),
                tuple(int(stride) for stride in value.stride()),
                str(value.dtype),
            )
        except Exception:
            pass
    pinned[id(value)] = value
    return (type(value), id(value))


def _constexpr_params(jit_fn: Any) -> tuple[frozenset[int], frozenset[str]]:
    """The positions and names of ``jit_fn``'s tl.constexpr parameters."""
    constexprs = [
        (index, param.name)
        for index, param in enumerate(getattr(jit_fn, "params", None) or ())
        if getattr(param, "is_constexpr", False)
    ]
    return (
        frozenset(index for index, _ in constexprs),
        frozenset(name for _, name in constexprs),
    )


def _grid_fingerprint(resolved: tuple | None, pinned: dict[int, Any]) -> Hashable:
    if resolved is None:
        return None
    try:
        # The launcher reads each dim as an index, so e.g. a numpy int dim a
        # grid callable returns afresh per call is the same grid each time.
        return tuple(operator.index(dim) for dim in resolved)
    except Exception:
        return _fingerprint_value(resolved, pinned)


class LanguagePatchedError(RuntimeError):
    """A host compile refused to start while an interpreted traced launch
    has the language patched (see _refuse_patched_language): no compile
    ran, so it is no kernel's compile error."""


def _refuse_patched_language() -> None:
    """Raise LanguagePatchedError before a compile while an interpreted
    traced launch has triton.language patched (patch_lang is process-wide,
    so the code generator would run on the interpreter's builtins)."""
    patched = [name for name in ("triton", "gluon") if LANG_PATCH_SCOPES.get(name)]
    if patched:
        raise LanguagePatchedError(
            "a Triton compile cannot run while an interpreted traced "
            f"launch has the {'/'.join(patched)} language patched (e.g. on "
            "another host thread); concurrent traced launches that mix "
            "interpretation and host compiles are not supported."
        )


class ClientManager:
    def __init__(self, clients: list[Client] | None = None):
        self.clients: dict[str, Client] = {}
        if clients:
            self.add_clients(clients)
        self.launch = Launch()
        self._lock = threading.Lock()
        # Compiles every IR client's kernels on the host; its cache
        # lives as long as the trace.
        self.compiler = HostCompiler()
        # The host thread whose launch is in flight (begin_launch until
        # finalize or abort_launch), and the lock guarding it.
        self._launch_owner: int | None = None
        self._owner_lock = threading.Lock()
        self._clear_loop_hooks()
        self._reset_launch_state()

    def _reset_launch_state(self) -> None:
        # Per traced launch: which (target, specialization, binding
        # fingerprint) keys IR clients were already given, which (target,
        # call) compile failures, the objects their identity tokens stand
        # for, which parameters the fingerprint leaves out, and the grids of
        # the compiled kernels Launch.grid is settled from. Nothing here
        # refers to a caller's tensor once the launch has ended.
        self._delivered: set[tuple[Any, Hashable, Hashable]] = set()
        self._failed: set[tuple[Any, Hashable]] = set()
        self._pinned: dict[int, Any] = {}
        # id(jit_fn) -> (jit_fn, its constexpr positions, their names).
        self._constexprs: dict[int, tuple[Any, frozenset[int], frozenset[str]]] = {}
        self._compiled_grids: set[tuple] = set()
        self._finalize_started = False

    def _release_pinned(self) -> None:
        # The launch has ended: its fingerprints are compared no more, so
        # the objects kept alive for their identity tokens can go.
        self._pinned = {}

    def _lock_context(self):
        if cfg.num_sms > 1:
            return self._lock
        return nullcontext()

    def get_client(self, name: str) -> Client | None:
        return self.clients.get(name)

    def add_clients(self, new_clients_list: list[Client]) -> None:
        # Check every addition before inserting any, so a rejected batch
        # leaves the manager unchanged.
        additions: dict[str, Client] = {}
        for new_client in new_clients_list:
            duplicate = any(
                isinstance(existing_client, new_client.__class__)
                for existing_client in (*self.clients.values(), *additions.values())
            )
            if not duplicate:
                additions[new_client.NAME] = new_client
        for client in additions.values():
            self._check_declarations(client)
        self.clients.update(additions)

    @staticmethod
    def _check_declarations(client: Client) -> None:
        # Core only checks what it can deliver; what a launch means to a
        # client stays with the client.
        name = type(client).__name__
        if client.LAUNCH not in LAUNCH_PREFERENCES:
            raise ValueError(
                f"{name}.LAUNCH must be one of {LAUNCH_PREFERENCES}, got "
                f"{client.LAUNCH!r}"
            )
        if client.NEEDS_INTERPRETER:
            return
        if client.LAUNCH != "skip":
            raise ValueError(
                f"{name}.LAUNCH is {client.LAUNCH!r}, but a traced launch with "
                "IR clients never runs the real kernel: an IR client must "
                "declare LAUNCH = 'skip'"
            )
        unknown = set(client.IR_STAGES) - IR_STAGE_NAMES
        if unknown:
            raise ValueError(
                f"{name}.IR_STAGES: {sorted(unknown)} are no stage of a host "
                f"compile, which produces {sorted(IR_STAGE_NAMES)} only"
            )

    def interpreting_clients(self) -> list[Client]:
        return [c for c in self.clients.values() if c.NEEDS_INTERPRETER]

    def ir_clients(self) -> list[Client]:
        return [c for c in self.clients.values() if not c.NEEDS_INTERPRETER]

    def ir_target(self) -> Any:
        """The GPUTarget the IR clients' kernels are compiled for: their
        Client.ir_target, else the configured default; None without IR
        clients. Raises ValueError for a target spec that names no target,
        or for IR clients that name different targets: a trace compiles
        each call for one target."""
        targets: dict[Any, list[str]] = {}
        for client in self.ir_clients():
            target = resolve_ir_target(client.ir_target)
            targets.setdefault(target, []).append(client.NAME)
        if len(targets) > 1:
            named = "; ".join(
                f"{format_ir_target(target)}: {', '.join(names)}"
                for target, names in targets.items()
            )
            raise ValueError(
                "the IR clients of one trace must compile for one target, "
                f"these name several ({named}); trace the kernel once per "
                "target instead"
            )
        return next(iter(targets), None)

    def begin_launch(self, call: LaunchCall) -> None:
        """Start one traced launch: a fresh Launch and per-launch state, then
        every client's begin_launch.

        While a launch begun on another host thread is still in flight, this
        raises RuntimeError before changing anything or telling any client:
        concurrent launches of one trace are not supported. If a client's
        begin_launch raises, the clients whose begin_launch was called get
        abort_launch and the exception propagates; no launch is left open.
        """
        self._claim_launch()
        # Every launch gets its own Launch, so the entries TraceInterface
        # appends to `launches` stay distinct and tilelens.clear() releases
        # the tensors an interpreted run recorded in them (an IR-only launch
        # records none).
        self.launch = Launch()
        self._reset_launch_state()
        begun: list[Client] = []
        try:
            for client in self.clients.values():
                begun.append(client)
                client.begin_launch(call)
        except BaseException as exc:
            self._abort_clients(begun, exc)
            raise

    def _claim_launch(self) -> None:
        thread = threading.get_ident()
        with self._owner_lock:
            if self._launch_owner not in (None, thread):
                raise RuntimeError(
                    "this trace is already running a launch on another host "
                    "thread; concurrent launches of one traced kernel are not "
                    "supported."
                )
            self._launch_owner = thread

    def _release_launch(self) -> None:
        with self._owner_lock:
            if self._launch_owner == threading.get_ident():
                self._launch_owner = None

    def abort_launch(self, exc: BaseException) -> None:
        """Deliver ``exc`` to every client's abort_launch.

        Nothing is sent once finalize has started: every client is finalized
        by then, and each launch ends in either finalize or abort. Nor is
        anything sent while another host thread's launch is in flight (ours
        has ended already). A failing hook never replaces ``exc``, which the
        caller re-raises; the failure is attached to it as a note. Only a
        hook's KeyboardInterrupt or SystemExit is raised, after every client
        got the abort.
        """
        thread = threading.get_ident()
        with self._owner_lock:
            if self._launch_owner not in (None, thread):
                return
            if self._finalize_started:
                self._launch_owner = None
                return
            # Held until every client got the abort, so no other thread's
            # begin_launch resets the state in between.
            self._launch_owner = thread
        self._abort_clients(list(self.clients.values()), exc)

    def _abort_clients(self, clients: list[Client], exc: BaseException) -> None:
        interrupt: BaseException | None = None
        try:
            for client in clients:
                try:
                    client.abort_launch(exc)
                except Exception as hook_exc:
                    message = (
                        f"{type(client).__name__}.abort_launch raised "
                        f"{type(hook_exc).__name__}: {hook_exc}"
                    )
                    if hasattr(exc, "add_note"):
                        exc.add_note(message)
                    else:  # Python 3.10
                        warnings.warn(message, RuntimeWarning, stacklevel=3)
                except BaseException as hook_exc:
                    if interrupt is None:
                        interrupt = hook_exc
        finally:
            self._release_pinned()
            self._release_launch()
        if interrupt is not None:
            raise interrupt from exc

    @contextmanager
    def patch_warmup(self, jit_fn):
        if not hasattr(jit_fn, "warmup"):
            yield
            return

        def patcher(fn):
            @wraps(fn)
            def wrapped(*args, **kwargs):
                if all(
                    not client.pre_warmup_callback(jit_fn, *args, **kwargs)
                    for client in self.clients.values()
                ):
                    return None
                kwargs.pop("warmup", None)
                ret = fn(*args, **kwargs)
                for client in self.clients.values():
                    client.post_warmup_callback(jit_fn, ret)
                return ret

            return wrapped

        jit_fn.warmup = patcher(jit_fn.warmup)
        try:
            yield
        finally:
            jit_fn.warmup = jit_fn.warmup.__wrapped__

    @contextmanager
    def ir_capture(self, jit_fn, *, real_args: RealArgs | None = None):
        """Route every ``jit_fn.run`` call through a host compile, then
        IR-client dispatch. Nothing launches: a call returns the kernel
        compiled on the host, or None when it failed to compile.

        Only the traced JITFunction instance is touched (an instance attribute,
        restored on exit). Autotuner/Heuristics layers reach it through their
        ``fn.run`` calls, so every config they warm up goes through the
        capture; IR clients get one event per distinct (specialization,
        binding fingerprint) per traced launch (see Client.before_launch).
        The compile never enters the original ``run``: ``self.compiler``
        compiles the call on the host for the IR clients' target
        (``ir_target``), no driver or device involved, so the user's
        pre_run_hooks never fire. A callable grid is resolved once more per
        captured call, for the events and the fingerprint; one that raises
        (or gives no 1-3 dim grid) is recorded as an unknown grid
        (``resolved_grid`` None), never raised here. Compiles see the
        arguments ``real_args`` maps the call to; events and fingerprints
        describe the call as made.

        A failing host compile is delivered through compile_failed, never
        raised. A host compile that could not run at all raises (or is
        caused by) HostCompileUnavailable, which tells it from a kernel's
        compile error; a compile refused while the language is patched is
        delivered as a LanguagePatchedError. Yields the CaptureWindow. A
        target spec that names no target, or IR clients that name different
        targets, raise ValueError here, before anything is compiled.

        The one host compile error raised instead is a call that does not
        bind the kernel's parameters (host_compile.bind_failed: a missing,
        extra or misnamed argument, or a call the JIT cannot key, e.g. an
        unhashable constexpr value): the JIT's binder or its cache key
        raised it, as ``JITFunction.run`` does for the call on any device,
        before any target or compile had a say, so the call raises that very
        exception before any client hears of the call.

        On exit the window settles Launch.grid: the grid every kernel
        compiled during the launch shares (what the launch would have
        used), or None when configs disagree on it (a skipped autotuned
        launch picks no config). Each event keeps its own ``resolved_grid``.

        Calls from other host threads pass through untouched, and a second
        capture of the same jit_fn by another trace or thread is refused:
        concurrent traced launches sharing a JITFunction are unsupported.
        """
        window = CaptureWindow()
        current = getattr(jit_fn, "run", None)
        if current is None or not self.ir_clients():
            yield window
            return
        owner = getattr(current, "_tilelens_ir_capture", None)
        thread = threading.get_ident()
        if owner is not None:
            manager, owner_thread, outer = owner
            if manager is not self or owner_thread != thread:
                raise RuntimeError(
                    f"{jit_fn!r} is already being captured by another traced "
                    "launch; concurrent traced launches sharing one "
                    "JITFunction are not supported."
                )
            # Nested in our own capture: the outer window stays in charge.
            yield outer
            return
        target = self.ir_target()
        orig_run = current

        def run(*args, grid, warmup, **kwargs):
            if threading.get_ident() != thread:
                # Another host thread's launch (e.g. a peer trace's warmup
                # compile) is not ours to capture.
                return orig_run(*args, grid=grid, warmup=warmup, **kwargs)
            return self._captured_run(
                window, target, jit_fn, real_args, args, kwargs, grid
            )

        run._tilelens_ir_capture = (self, thread, window)  # type: ignore[attr-defined]
        with _instance_attr(jit_fn, "run", run):
            yield window
        self._settle_launch_grid()

    def _captured_run(self, window, target, jit_fn, real_args, args, kwargs, grid):
        if real_args is None:
            compile_args, compile_kwargs = args, kwargs
        else:
            compile_args, compile_kwargs = real_args(jit_fn, args, kwargs)
        bound_args = _bind_launch_args(jit_fn, args, kwargs)
        resolved_grid = _resolve_grid(grid, bound_args)
        try:
            _refuse_patched_language()
            kernel = self.compiler.compile(
                jit_fn, compile_args, compile_kwargs, target=target
            )
        except Exception as exc:
            if bind_failed(exc):
                raise
            window.failures.append(exc)
            self._compile_failed(
                target, jit_fn, args, kwargs, grid, exc, bound_args, resolved_grid
            )
            return None
        window.compiled += 1
        fingerprint = self._binding_fingerprint(jit_fn, args, kwargs, resolved_grid)
        key = (target, _specialization(kernel), fingerprint)
        if key not in self._delivered:
            self._delivered.add(key)
            event = self._launch_event(
                jit_fn,
                args,
                kwargs,
                grid,
                kernel,
                target=target,
                bound_args=bound_args,
                resolved_grid=resolved_grid,
            )
            self._record_compiled_grid(event)
            clients = self.ir_clients()
            self._dispatch_ir("before_launch", event, clients)
            self._dispatch_ir("after_launch", event, clients)
        return kernel

    def _binding_fingerprint(self, jit_fn, args, kwargs, resolved_grid) -> Hashable:
        """The binding part of the dedup key (see Client.before_launch).
        Arguments to tl.constexpr parameters are left out: Triton hashes
        each into the kernel, so the specialization already tells them
        apart."""
        pinned = self._pinned
        cached = self._constexprs.get(id(jit_fn))
        if cached is None or cached[0] is not jit_fn:
            cached = self._constexprs[id(jit_fn)] = (jit_fn, *_constexpr_params(jit_fn))
        _, positions, names = cached
        return (
            tuple(
                None if index in positions else _fingerprint_value(arg, pinned)
                for index, arg in enumerate(args)
            ),
            tuple(
                sorted(
                    (name, _fingerprint_value(value, pinned))
                    for name, value in kwargs.items()
                    if name not in names
                )
            ),
            _grid_fingerprint(resolved_grid, pinned),
        )

    def _compile_failed(
        self, target, jit_fn, args, kwargs, grid, error, bound_args, resolved_grid
    ):
        # Once per launch for each failing call (constexprs included, as no
        # specialization tells configs apart here): the same call compiled
        # again in the launch is not news.
        pinned = self._pinned
        call = (
            tuple(_fingerprint_value(arg, pinned) for arg in args),
            tuple(
                sorted(
                    (name, _fingerprint_value(value, pinned))
                    for name, value in kwargs.items()
                )
            ),
        )
        if (target, call) in self._failed:
            return
        self._failed.add((target, call))
        event = self._launch_event(
            jit_fn,
            args,
            kwargs,
            grid,
            None,
            error=error,
            target=target,
            bound_args=bound_args,
            resolved_grid=resolved_grid,
        )
        self._dispatch_ir("compile_failed", event, self.ir_clients())

    @staticmethod
    def _launch_event(
        jit_fn,
        args,
        kwargs,
        grid,
        kernel,
        error=None,
        *,
        target: Any = None,
        bound_args: dict[str, Any] | None = None,
        resolved_grid: Any = _MISSING,
    ) -> LaunchEvent:
        if bound_args is None:
            bound_args = _bind_launch_args(jit_fn, args, kwargs)
        if resolved_grid is _MISSING:
            resolved_grid = _resolve_grid(grid, bound_args)
        return LaunchEvent(
            jit_fn=jit_fn,
            args=tuple(args),
            kwargs=MappingProxyType(dict(kwargs)),
            grid=grid,
            resolved_grid=resolved_grid,
            bound_args=MappingProxyType(bound_args),
            kernel=kernel,
            specialization=None if kernel is None else _specialization(kernel),
            error=error,
            target=target,
        )

    def _record_compiled_grid(self, event: LaunchEvent) -> None:
        # Launch.tensors is not filled from the binding:
        # an interpreted run records (arg_callback) the host copies its eager
        # clients' records point into, and an IR-only launch records none, so
        # no device tensor outlives its launch in tilelens.launches. IR
        # clients keep the tensor facts they need in their own records.
        if event.resolved_grid is not None:
            with self._lock_context():
                self._compiled_grids.add(event.resolved_grid)

    def _settle_launch_grid(self) -> None:
        grids = self._compiled_grids
        resolved = next(iter(grids)) if len(grids) == 1 else None
        with self._lock_context():
            self.launch.grid = resolved

    @staticmethod
    def _dispatch_ir(hook: str, event: LaunchEvent, clients) -> None:
        # Runs on the launching host thread, never on interpreter workers.
        for client in clients:
            getattr(client, hook)(event)

    @contextmanager
    def patch_run(self, fn, frontend_name: str):
        frontend = get_frontend(frontend_name)
        namespaces = frontend.namespaces
        # IR clients take no part in op/loop registration: their empty
        # callbacks would otherwise replace an interpreting peer's patches.
        interpreting = self.interpreting_clients()
        with patch_calls(frontend_name):
            # Collect all for-loop callbacks from clients
            all_loop_callbacks = []
            for client in interpreting:
                for namespace, attrs in namespaces.items():  # patch ops
                    for attr, op in attrs.items():
                        callbacks = client.register_op_callback(op)
                        patch_op(
                            namespace,
                            attr,
                            callbacks,
                            frontend_name=frontend_name,
                        )
                all_loop_callbacks.append(client.register_for_loop_callback())

            self._populate_loop_hooks(all_loop_callbacks)
            patch_for_loop(frontend_name)
            patch_lang(fn, frontend_name, client_manager=self)
            try:
                yield
            finally:
                unpatch_lang(frontend_name)
                for namespace, attrs in namespaces.items():
                    for attr, op in attrs.items():
                        unpatch_op(namespace, attr, frontend_name)
                unpatch_for_loop(frontend_name)
                self._clear_loop_hooks()

    def pre_run_callback(self, fn: Callable) -> bool:
        with self._lock_context():
            rets = [c.pre_run_callback(fn) for c in self.interpreting_clients()]
            return all(rets) if rets else True

    def post_run_callback(self, fn: Callable) -> bool:
        with self._lock_context():
            rets = [c.post_run_callback(fn) for c in self.interpreting_clients()]
            # With no interpreting voter, keep running the whole grid.
            return any(rets) if rets else True

    def finalize(self) -> None:
        """Finalize every client into self.launch. This ends the launch:
        another host thread may begin the next one right after."""
        try:
            with self._lock_context():
                self._finalize_started = True
                self.launch.records = []
                # Finalize every client even if a peer raises (SystemExit
                # included), so none carries this launch's state into the
                # next; then re-raise the first failure.
                first_exc: BaseException | None = None
                for client in self.clients.values():
                    try:
                        # client may introduce tensors not declared in kernel args (e.g. tracer recording a tensor allocation)
                        self.launch.tensors.update(getattr(client, "tensors", []) or [])
                        self.launch.records += client.finalize()
                    except BaseException as exc:
                        if first_exc is None:
                            first_exc = exc
                if first_exc is not None:
                    raise first_exc
        finally:
            self._release_pinned()
            self._release_launch()

    def arg_callback(self, name, arg, arg_cvt):
        with self._lock_context():
            if hasattr(arg, "data_ptr"):
                self.launch.tensors.add(arg)
            for client in self.interpreting_clients():
                client.arg_callback(name, arg, arg_cvt)

    def grid_callback(self, grid: tuple[int]):
        with self._lock_context():
            self.launch.grid = grid
            for client in self.interpreting_clients():
                client.grid_callback(grid)

    def grid_idx_callback(self, grid_idx: tuple[int, ...]):
        with self._lock_context():
            for client in self.interpreting_clients():
                client.grid_idx_callback(grid_idx)

    # --- For-loop callback management ---

    def _clear_loop_hooks(self) -> None:
        self._range_type_hooks: list[Callable] = []
        self._before: list[Callable] = []
        self._iter_listeners: list[Callable] = []
        self._iter_overrider: Callable | None = None
        self._range_wrapper_factory: Callable | None = None
        self._after: list[Callable] = []

    def _populate_loop_hooks(self, callbacks_list: list[ForLoopCallbacks]) -> None:
        self._clear_loop_hooks()
        for cb in callbacks_list:
            if cb.range_type_callback is not None:
                self._range_type_hooks.append(cb.range_type_callback)
            if cb.before_loop_callback is not None:
                self._before.append(cb.before_loop_callback)
            if cb.loop_iter_listener is not None:
                self._iter_listeners.append(cb.loop_iter_listener)
            if cb.loop_iter_overrider is not None:
                if self._iter_overrider is not None:
                    raise RuntimeError("Only one loop_iter overrider allowed")
                self._iter_overrider = cb.loop_iter_overrider
            if cb.range_wrapper_factory is not None:
                if self._range_wrapper_factory is not None:
                    raise RuntimeError("Only one range_wrapper_factory allowed")
                self._range_wrapper_factory = cb.range_wrapper_factory
            if cb.after_loop_callback is not None:
                self._after.append(cb.after_loop_callback)

    def range_type(self, loop_site: LoopSite, range_type: str) -> None:
        for hook in self._range_type_hooks:
            hook(loop_site, range_type)

    def before_loop(self, loop_site: LoopSite, iterable: Any) -> None:
        for hook in self._before:
            hook(loop_site, iterable)

    def loop_iter(self, loop_site: LoopSite, idx: Any) -> Any:
        if self._iter_overrider is not None:
            new_idx = self._iter_overrider(loop_site, idx)
            if new_idx is not None:
                idx = new_idx

        for hook in self._iter_listeners:
            hook(loop_site, idx)

        return idx

    def after_loop(self, loop_site: LoopSite) -> None:
        for hook in self._after:
            hook(loop_site)

    def loop_iter_wrapper(
        self,
        iterable_callable: Callable,
        iter_args,
        iter_kwargs,
        loop_site: LoopSite,
        range_type: str,
    ) -> "LoopIter":
        args = tuple(iter_args) if iter_args is not None else ()
        kwargs = dict(iter_kwargs) if iter_kwargs is not None else {}

        if self._range_wrapper_factory is not None:
            wrapped = self._range_wrapper_factory(
                None, loop_site, range_type, args, kwargs, iterable_callable
            )
            if wrapped is not None:
                iterable = wrapped
            else:
                iterable = iterable_callable(*args, **kwargs)
        else:
            iterable = iterable_callable(*args, **kwargs)
        return LoopIter(self, iterable, loop_site, range_type)
