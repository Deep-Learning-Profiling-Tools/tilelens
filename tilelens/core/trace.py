import copy
import inspect
from contextlib import contextmanager
from collections.abc import Callable
from types import MappingProxyType
from typing import Any
from ..utils.traceback_utils import CODE_KEYS, get_code_key

from .config import config as cfg
from ..clients import Sanitizer, Profiler, RaceDetector, Tracer
from ..clients.race_detector.race_detector import NullRaceDetector
from .client import ClientManager, Client, LaunchCall, LanguagePatchedError
from .data import Launch
import types


launches: list[Launch] = []


def _without_warmup(kwargs: dict[str, Any]) -> dict[str, Any]:
    # Launch kwargs carry warmup=False; the warmup entry points set their own.
    return {k: v for k, v in kwargs.items() if k != "warmup"}


def _launch_call(
    jit_fn: Any, args: tuple, kwargs: dict[str, Any], *, capture: bool
) -> LaunchCall:
    return LaunchCall(
        jit_fn=jit_fn,
        args=tuple(args),
        kwargs=MappingProxyType(
            {k: v for k, v in kwargs.items() if k not in ("grid", "warmup")}
        ),
        grid=kwargs.get("grid"),
        capture=capture,
    )


class TraceInterface:
    def __init__(self, client: str | Client) -> None:
        self.client_manager = ClientManager()
        self.add_client(client)

    @staticmethod
    def _normalize_client(client: str | Client) -> Client:
        if isinstance(client, str):
            name = client.lower()
            if name == "sanitizer":
                return Sanitizer()
            if name == "profiler":
                return Profiler()
            if name == "race_detector":
                return RaceDetector()
            if name == "tracer":
                return Tracer()
            raise ValueError(f"Unknown client: {client}")
        elif isinstance(client, Client):
            return client
        else:
            raise TypeError(f"Expected str or Client, got {type(client)}")

    def add_client(self, new_client: str | Client) -> None:
        self.client_manager.add_clients([self._normalize_client(new_client)])

    def finalize(self):
        # Take the Launch first: once finalize ends the launch, another host
        # thread may begin the next one on this manager.
        launch = self.client_manager.launch
        self.client_manager.finalize()
        launches.append(launch)

    @contextmanager
    def _launch_scope(self, call: LaunchCall):
        """begin_launch, then abort_launch if the launch raises. A refused or
        failing begin_launch cleans up after itself and is not aborted: the
        refusal must not reach the clients of another thread's launch."""
        mgr = self.client_manager
        mgr.begin_launch(call)
        try:
            yield
        except BaseException as exc:
            mgr.abort_launch(exc)
            raise


class LaunchInterface:
    def warmup(self, *args, grid, **kwargs):
        from triton.runtime.jit import MockTensor

        return self.run(
            grid=grid,
            warmup=True,
            *map(MockTensor.wrap_dtype, args),
            **kwargs,
        )

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.run(
            grid=grid,
            warmup=False,
            *args,
            **kwargs,
        )


class KernelTraceSupport:
    @staticmethod
    def _is_autotuner(runner: Any) -> bool:
        from triton.runtime import Autotuner

        return isinstance(runner, Autotuner)

    @staticmethod
    def _is_heuristics(runner: Any) -> bool:
        from triton.runtime.autotuner import Heuristics

        return isinstance(runner, Heuristics)

    @staticmethod
    def dummy_benchmarker(fn, quantiles):
        fn()
        return (1.0, 1.0, 1.0)

    def _ir_runner(self, runner: Any, jit_fn: Any | None) -> Any | None:
        if jit_fn is None:
            return None
        return self._rebuild_runner(runner, _IRLeaf(jit_fn), ir=True)

    def _autotuned(self, runner: Any) -> bool:
        """Whether an Autotuner layer sits anywhere in ``runner``'s chain."""
        while self._is_autotuner(runner) or self._is_heuristics(runner):
            if self._is_autotuner(runner):
                return True
            runner = runner.fn
        return False

    def _drop_autotuner_args(self, runner: Any) -> None:
        """Clear the per-call tensors Autotuner layers in ``runner``'s chain
        of copies may still hold: ``nargs`` and ``restore_copies``.

        Autotuner.run and .warmup keep the call's arguments in ``nargs``
        until they return, and a benchmark call keeps its restore_value
        clones in ``restore_copies`` until its post_hook, which _bench skips
        for a KeyboardInterrupt. So a launch that raises would leave the
        caller's tensors, or device clones of them, on a copy that outlives
        the launch.
        """
        while self._is_autotuner(runner) or self._is_heuristics(runner):
            if self._is_autotuner(runner):
                runner.nargs = None
                if hasattr(runner, "restore_copies"):
                    runner.restore_copies = {}
            runner = runner.fn

    def _rebuild_runner(self, runner: Any, leaf: Any, *, ir: bool = False) -> Any:
        """Rebuild ``runner``'s Autotuner/Heuristics chain from shallow copies
        (a JITFunction holds an RLock, so no deepcopy), with ``leaf`` in place
        of its kernel, so the user's chain is never mutated. ``ir``: the chain
        whose warmup stands in for the launch's autotuning (the IR clients'
        compiles)."""
        if not (self._is_autotuner(runner) or self._is_heuristics(runner)):
            return leaf
        layer = copy.copy(runner)
        layer.fn = self._rebuild_runner(runner.fn, leaf, ir=ir)
        if self._is_autotuner(layer):
            # Read by Autotuner.run (the interpreted chain), not by .warmup.
            layer.cache = {}  # copy.copy shares the user's
            # check_disk_cache walks .fn down to a JITFunction, which the
            # interpreted chain lacks.
            layer.cache_results = False
            layer.do_bench = self.dummy_benchmarker
            if ir:
                layer.prune_configs = self._refusing_conflicts(layer)
        else:
            layer.warmup = self._heuristics_warmup(layer)
        return layer

    @staticmethod
    def _refusing_conflicts(layer: Any) -> Callable:
        # The launch's autotuning benchmarks each pruned config first, and
        # Autotuner._bench refuses a call that passes one of the config's
        # meta-parameters itself, with this ValueError (as Triton words it).
        # Autotuner.warmup, through which the IR clients' compiles
        # go instead, would pass the keyword twice (a TypeError naming the
        # IR leaf), so the IR chain's copy checks the pruned configs first.
        prune = layer.prune_configs

        def prune_configs(kwargs):
            pruned = prune(kwargs)
            for config in pruned:
                conflicts = kwargs.keys() & config.kwargs.keys()
                if conflicts:
                    raise ValueError(
                        f"Conflicting meta-parameters: {', '.join(conflicts)}."
                        " Make sure that you don't re-define auto-tuned symbols."
                    )
            return pruned

        return prune_configs

    @staticmethod
    def _heuristics_warmup(layer: Any) -> Callable:
        # KernelInterface.warmup would pass warmup= a second time and skip
        # fn.warmup, where the clients vote: fill in the heuristics, then call
        # fn.warmup.
        def warmup(*args, **kwargs):
            for name, heur in layer.values.items():
                kwargs[name] = heur({**dict(zip(layer.arg_names, args)), **kwargs})
            return layer.fn.warmup(*args, **kwargs)

        return warmup

    def _copy_callable_attrs(
        self,
        runner: Any,
        base_fn: Callable | None = None,
        *,
        src_fallback: Any = None,
    ) -> None:
        for attr in ("__name__", "__module__", "__doc__", "__qualname__"):
            if hasattr(runner, attr):
                setattr(self, attr, getattr(runner, attr))
            elif base_fn is not None and hasattr(base_fn, attr):
                setattr(self, attr, getattr(base_fn, attr))

        if not hasattr(self, "__name__"):
            self.__name__ = "<unknown>"

        if hasattr(runner, "src"):
            self.src = runner.src
        elif src_fallback is not None and hasattr(src_fallback, "src"):
            self.src = src_fallback.src


class TritonTrace(LaunchInterface, TraceInterface, KernelTraceSupport):
    def __init__(
        self,
        runner: Any,
        client: str | Client,
    ) -> None:
        from triton import JITFunction
        from triton.runtime.autotuner import Heuristics
        from triton.runtime.interpreter import InterpretedFunction

        self.jit_fn: Any | None = None
        self.base_fn: Callable | None = None
        self.interpreted_fn: Any | None = None

        def unpack_kernel(
            source: Any,
        ) -> tuple[Any | None, Callable | None, Any | None]:
            if isinstance(source, TritonTrace):
                return source.jit_fn, source.base_fn, source.interpreted_fn
            if isinstance(source, JITFunction):
                base_fn = source.fn
                return source, base_fn, InterpretedFunction(base_fn)
            if isinstance(source, InterpretedFunction):
                return None, source.fn, source
            if isinstance(source, Heuristics):
                # Heuristics wraps another kernel, recursively unpack it
                return unpack_kernel(source.fn)
            raise TypeError(f"Unsupported runner type: {type(source)}")

        if self._is_autotuner(runner):
            self.jit_fn, self.base_fn, self.interpreted_fn = unpack_kernel(runner.fn)
        elif self._is_heuristics(runner):
            self.jit_fn, self.base_fn, self.interpreted_fn = unpack_kernel(runner.fn)
        else:
            self.jit_fn, self.base_fn, self.interpreted_fn = unpack_kernel(runner)
        self.runner = self._rebuild_runner(runner, self.interpreted_fn)
        self.warmup_runner = (
            None if self.jit_fn is None else self._rebuild_runner(runner, self.jit_fn)
        )
        # The IR clients' host compiles go through a chain of their own,
        # which no warmup patch gates.
        self.ir_runner = self._ir_runner(runner, self.jit_fn)

        self.arg_names = runner.arg_names

        self.fn = runner

        TraceInterface.__init__(self, client)

        self._copy_callable_attrs(runner, self.base_fn, src_fallback=self.jit_fn)

    def run(self, *args, **kwargs):
        mgr = self.client_manager
        has_ir = bool(mgr.ir_clients())
        capture = has_ir and self.jit_fn is not None
        call = _launch_call(self.jit_fn, args, kwargs, capture=capture)
        with self._launch_scope(call):
            if not has_ir:
                return self._run_interpreted(*args, **kwargs)
            if mgr.interpreting_clients():
                if capture:
                    # Mixed trace: host-compile every config for the IR
                    # clients, then the interpreter produces the outputs (no
                    # real launch, no device). A call that does not bind the
                    # kernel's parameters raises here, as in an IR-only trace,
                    # before the interpreter runs: the interpreted run would
                    # fail on the same call (with Python's own TypeError for
                    # the kernel function), and the error raised is the one
                    # the untraced JIT raises.
                    self._compile_for_ir(args, kwargs)
                return self._run_interpreted(*args, **kwargs)
            # Only IR clients, none of which runs the real kernel (see
            # Client.LAUNCH). Without a JITFunction (TRITON_INTERPRET / an
            # InterpretedFunction runner; call.capture is False) there is no
            # compiled kernel for them either, so nothing runs.
            ret = self._run_compiled(*args, **kwargs) if capture else None
            self.finalize()
            return ret

    def _compile_for_ir(self, args, kwargs):
        """Compile every (pruned) config through the IR runner's warmup; the
        capture host-compiles each call for the IR clients' target and turns
        it into an IR event, or a compile_failed event, without launching or
        touching a device. Returns the warmup result and the capture window:
        for a plain or @heuristics kernel the host-compiled kernel (None if
        it failed to compile).
        """
        runner = self.ir_runner
        assert runner is not None  # built whenever jit_fn is set
        try:
            with self.client_manager.ir_capture(
                self.jit_fn, real_args=_untraced_call_args
            ) as window:
                ret = runner.warmup(*args, **_without_warmup(kwargs))
        finally:
            self._drop_autotuner_args(runner)
        return ret, window

    def _run_compiled(self, *args, **kwargs):
        """IR-only trace: no interpreter, no real launch.

        Returns what it can of the untraced return value: the host-compiled
        kernel when no Autotuner is involved (its only config; never loaded,
        it cannot launch; None if it failed to compile), None for an
        autotuned kernel (no config was picked). Only a compile refused
        while an interpreted traced launch has the language patched
        (LanguagePatchedError, no compile's outcome: concurrent traced
        launches that mix interpretation and host compiles are unsupported)
        fails the launch, and a call that does not bind the kernel's
        parameters, which raises the JIT binder's error as the untraced call
        does.
        """
        ret, window = self._compile_for_ir(args, kwargs)
        refused = [e for e in window.failures if isinstance(e, LanguagePatchedError)]
        if refused:
            raise refused[0]
        return None if self._autotuned(self.ir_runner) else ret

    def _run_interpreted(self, *args, **kwargs):
        self._voted_warmup(*args, **kwargs)

        with self.client_manager.patch_run(self.base_fn, frontend_name="triton"):
            kwargs.update({"client_manager": self.client_manager})
            kwargs.update({"jit_fn": self.jit_fn})
            try:
                ret = self.runner.run(*args, **kwargs)
            finally:
                self._drop_autotuner_args(self.runner)
            self.finalize()
            return ret

    def __call__(self, *args, **kwargs):
        # When a traced JIT function is called from within another JIT function,
        # we need to execute the underlying function directly

        # check that client sets match for calling and called functions
        from .frontend import triton as triton_frontend

        outer_client_manager = triton_frontend.frontend.current_client_manager()
        if outer_client_manager is not None:
            # Only interpreting clients take part in the interpreted run.
            outer_clients = {
                c.NAME for c in outer_client_manager.interpreting_clients()
            }
            inner_clients = {c.NAME for c in self.client_manager.interpreting_clients()}
            if outer_clients != inner_clients:
                raise RuntimeError(
                    "nested traced calls require matching clients; "
                    f"outer={outer_clients}, inner={inner_clients}"
                )

        return self.interpreted_fn(*args, **kwargs)

    def warmup(self, *args, **kwargs):
        self._voted_warmup(*args, **kwargs)

    def _voted_warmup(self, *args, **kwargs):
        with self.client_manager.patch_warmup(self.jit_fn):
            if self.warmup_runner:
                try:
                    self.warmup_runner.warmup(*args, **kwargs)
                finally:
                    self._drop_autotuner_args(self.warmup_runner)


class _IRLeaf:
    """The traced JITFunction at the bottom of the IR runner chain.

    Its warmup is the JITFunction class's, so no instance-level warmup patch
    (patch_warmup's vote, or anyone else's) decides whether an IR
    config compiles; everything else, ``run`` (where ir_capture sits, and
    host-compiles instead of entering JITFunction.run) included, is the
    JITFunction instance's. Its ``fn`` is the JITFunction too, as Triton's
    Autotuner expects when it follows ``.fn`` from its own down to the
    JITFunction it tunes.
    """

    def __init__(self, jit_fn: Any) -> None:
        self.jit_fn = jit_fn

    @property
    def fn(self) -> Any:
        return self.jit_fn

    def warmup(self, *args, **kwargs):
        return type(self.jit_fn).warmup(self.jit_fn, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name == "jit_fn":  # not set yet (e.g. mid-copy): no recursion
            raise AttributeError(name)
        return getattr(self.jit_fn, name)


def _untraced_call_args(
    jit_fn: Any, args: tuple, kwargs: dict[str, Any]
) -> tuple[tuple, dict[str, Any]]:
    """One JITFunction call's arguments as a host compile must see them:
    each TritonTrace passed as an argument (also inside a tuple), or bound
    as a parameter default, replaced by its JITFunction (the default passed
    explicitly). Triton's code generator treats any other callee as plain
    Python and would call TritonTrace.__call__.
    """

    def real(value: Any) -> Any:
        if isinstance(value, TritonTrace) and value.jit_fn is not None:
            return value.jit_fn
        if isinstance(value, tuple):
            items = [real(item) for item in value]
            if all(new is old for new, old in zip(items, value)):
                return value
            # A namedtuple is rebuilt from fields, a plain tuple from items.
            return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
        return value

    real_args = tuple(real(arg) for arg in args)
    real_kwargs = {name: real(value) for name, value in kwargs.items()}
    signature = getattr(jit_fn, "signature", None)
    if isinstance(signature, inspect.Signature):
        for index, (name, param) in enumerate(signature.parameters.items()):
            if index < len(real_args) or name in real_kwargs:
                continue
            if real(param.default) is not param.default:
                real_kwargs[name] = real(param.default)
    return real_args, real_kwargs


class NKITrace(LaunchInterface, TraceInterface):
    def __init__(self, kernel, client: str | Client, beta2: bool = True) -> None:
        nki_fn_cls: object = None
        if beta2:
            from .simulation.nki_beta2 import NKIBeta2InterpretedFunction

            nki_fn_cls = NKIBeta2InterpretedFunction
        else:
            from .simulation.nki import NKIInterpretedFunction

            nki_fn_cls = NKIInterpretedFunction

        self.frontend_name = "nki_beta2" if beta2 else "nki"
        if isinstance(kernel, nki_fn_cls):
            assert hasattr(kernel, "fn")
            self.interpreter_fn = kernel
            self.func = kernel.fn
        else:
            self.interpreter_fn = nki_fn_cls(kernel)
            self.func = kernel

        TraceInterface.__init__(self, client)

    def __getattr__(self, name):
        # Forward any missing attributes to the underlying runner
        # This allows Trace to transparently proxy attributes like 'src', 'hash', etc.
        # Use object.__getattribute__ to avoid infinite recursion
        try:
            fn = object.__getattribute__(self, "fn")
            if hasattr(fn, name):
                return getattr(fn, name)
        except AttributeError:
            pass

        try:
            jit_fn = object.__getattribute__(self, "jit_fn")
            if hasattr(jit_fn, name):
                return getattr(jit_fn, name)
        except AttributeError:
            pass

        try:
            base_fn = object.__getattribute__(self, "base_fn")
            if hasattr(base_fn, name):
                return getattr(base_fn, name)
        except AttributeError:
            pass

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __getitem__(self, *grid):
        return LaunchInterface.__getitem__(self, tuple(*grid))

    def __call__(self, *args, **kwargs):
        return self[(1,)](*args, **kwargs)

    def run(self, *args, pre_trace=True, platform_target="trn1", **kwargs):
        """
        pre_trace: determines whether to do an initial NKI Beta 2 trace to capture some semantic errors.
            pre_trace=False has fewer guarantees on interpreter parity with NKI compiler but must be set
            if you want full python flexibility inside kernels (e.g. importing modules inside a kernel).
            Does nothing if self.frontend_name == 'nki'.
        """
        with self._launch_scope(_launch_call(None, args, kwargs, capture=False)):
            if not self.client_manager.interpreting_clients():
                # Only IR clients: there is no compiled kernel here, and the
                # real kernel never runs for them, so nothing runs.
                self.finalize()
                return None
            if self.frontend_name == "nki_beta2" and pre_trace:
                import nki

                kwargs.pop("warmup", None)
                grid = kwargs.pop("grid", None)
                nki.trace(
                    self.func, grid=grid, platform_target=platform_target
                ).specialize(*args, **kwargs)
                kwargs["grid"] = grid
            with self.client_manager.patch_run(
                self.func,
                frontend_name=self.frontend_name,
            ):
                kwargs.update({"client_manager": self.client_manager})
                ret = self.interpreter_fn.run(*args, **kwargs)
                self.finalize()
                return ret


class GluonTrace(LaunchInterface, TraceInterface, KernelTraceSupport):
    def __init__(self, runner: Any, client: str | Client) -> None:
        from triton.experimental import gluon
        from triton.runtime.autotuner import Heuristics
        from .simulation.gluon import GluonInterpretedFunction

        # Gluon has exposed different JIT class names across Triton versions.
        # Treat an object that already has the JIT runner protocol as compiled,
        # and only call gluon.jit for raw Python functions.
        if not all(hasattr(runner, attr) for attr in ("fn", "run", "arg_names")):
            runner = gluon.jit(runner)

        def unpack_kernel(source: Any) -> tuple[Callable, Any]:
            if isinstance(source, GluonTrace):
                return source.base_fn, source.interpreted_fn
            if isinstance(source, Heuristics):
                return unpack_kernel(source.fn)
            if all(hasattr(source, attr) for attr in ("fn", "arg_names")):
                base_fn = source.fn
                return base_fn, GluonInterpretedFunction(base_fn, source.arg_names)
            raise TypeError(f"Unsupported runner type: {type(source)}")

        self.base_fn, self.interpreted_fn = unpack_kernel(
            runner.fn if self._is_autotuner(runner) else runner
        )
        self.fn = runner
        self.arg_names = runner.arg_names
        self.runner = self._rebuild_runner(runner, self.interpreted_fn)

        TraceInterface.__init__(self, client)
        self._copy_callable_attrs(runner, self.base_fn)

    def run(self, *args, **kwargs):
        grid = kwargs.get("grid")
        if grid is None:
            raise TypeError(
                "GluonTrace.run() missing required keyword argument: 'grid'"
            )

        with self._launch_scope(_launch_call(None, args, kwargs, capture=False)):
            if not self.client_manager.interpreting_clients():
                # Only IR clients: there is no compiled kernel here, and the
                # real kernel never runs for them, so nothing runs.
                self.finalize()
                return None
            with self.client_manager.patch_run(self.base_fn, frontend_name="gluon"):
                try:
                    ret = self.runner.run(
                        *args,
                        **kwargs,
                        client_manager=self.client_manager,
                    )
                finally:
                    self.client_manager.post_run_callback(self.base_fn)
                self.finalize()
                return ret

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def warmup(self, *args, **kwargs):
        return self.run(*args, warmup=True, **kwargs)


def trace_source(kernel):
    """
    Add the kernel code to be traceable within stack traces for clients
    (e.g. to capture source code to display with visualizer/client).
    You can also use this to decorate other functions that a kernel calls.
    """
    base_fn = kernel
    while not isinstance(base_fn, types.FunctionType):
        # base_fn may be a raw function but also a JITFunction, Autotuner, InterpretedFunction, ...
        # we want to strip away the wrappers until we get to the python function
        base_fn = base_fn.fn
    CODE_KEYS.add(get_code_key(base_fn))
    return kernel


def trace(client: str | Client | None = None, frontend: str = "triton"):
    """
    Create a trace object that can be used to run a kernel with instrumentation client(s).

    :param kernel: The kernel to run.
    :param client: A client to run with the kernel. Defaults to Tracer() if not specified.
    """
    if client is None:
        client = Tracer()

    if not isinstance(client, (str, Client)):
        raise TypeError(f"Expected str or Client, got {type(client)}")

    def _is_sanitizer_client(selected: str | Client) -> bool:
        if isinstance(selected, str):
            return selected.lower() == "sanitizer"
        return isinstance(selected, Sanitizer)

    def _is_race_detector_client(selected: str | Client) -> bool:
        if isinstance(selected, str):
            return selected.lower() == "race_detector"
        # A NullRaceDetector instance means the public RaceDetector(...) factory
        # was called while the flag was off — equivalent to the string-dispatch
        # flag-off case, so take the same fast path. Explicit SymbolicRaceDetector()
        # instances bypass the factory's __new__ and reflect a deliberate opt-in;
        # they are intentionally NOT matched here, preserving the "explicit
        # detector wins over flag" semantic guarded by
        # test_flag_off_does_not_swallow_explicit_instance.
        return isinstance(selected, NullRaceDetector)

    def decorator(kernel) -> TritonTrace | NKITrace | GluonTrace | Any:
        if cfg.cli_active and isinstance(kernel, TraceInterface):
            raise RuntimeError(
                "@tilelens.trace() decorator cannot be used together with "
                "CLI wrapper (e.g., tile-sanitizer / tile-profiler). "
                "Please remove the @tilelens.trace() decorator from your code "
                "when using CLI tools."
            )

        if _is_sanitizer_client(client) and not cfg.enable_sanitizer:
            # when dry-running tile-sanitizer CLI (i.e. wrap kernels with sanitizer
            # tracing but don't actually sanitize), don't actually trace the kernel
            return kernel

        if _is_race_detector_client(client) and not cfg.enable_race_detector:
            # Flag-off escape hatch: leave the kernel untraced so ENABLE_RACE_DETECTOR=0
            # truly has zero runtime impact on code that opts into race_detector tracing.
            return kernel

        # If the object is already initialized as a TraceInterface, just append the new client(s)
        if isinstance(kernel, TraceInterface):
            trace = kernel
            trace.add_client(client)
            return trace

        trace_source(kernel)

        # First-time wrapping
        if frontend in ("nki", "nki_beta2"):
            return NKITrace(kernel, client, beta2=("beta2" in frontend))
        elif frontend == "gluon":
            return GluonTrace(kernel, client)
        elif frontend == "triton":
            return TritonTrace(kernel, client)
        else:
            raise ValueError(f"Unknown frontend: {frontend}")

    return decorator


def clear() -> None:
    """
    Clear all traces.
    """
    launches.clear()
