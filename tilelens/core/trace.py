from copy import deepcopy
from collections.abc import Callable
from typing import Any
from ..utils.traceback_utils import CODE_KEYS, get_code_key

from .config import config as cfg
from ..clients import Sanitizer, Profiler, RaceDetector, Tracer
from ..clients.race_detector.race_detector import NullRaceDetector
from .client import ClientManager, Client
from .data import Launch
import types


launches: list[Launch] = []

# The clients trace() takes by name, each built with its defaults.
_NAMED_CLIENTS: dict[str, type[Client]] = {
    "sanitizer": Sanitizer,
    "profiler": Profiler,
    "race_detector": RaceDetector,
    "tracer": Tracer,
}


def _named_client_type(name: str) -> type[Client]:
    try:
        return _NAMED_CLIENTS[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown client: {name}") from None


class TraceInterface:
    def __init__(self, client: str | Client) -> None:
        self.client_manager = ClientManager()
        self.add_client(client)

    @staticmethod
    def _normalize_client(client: str | Client) -> Client:
        if isinstance(client, str):
            return _named_client_type(client)()
        elif isinstance(client, Client):
            return client
        else:
            raise TypeError(f"Expected str or Client, got {type(client)}")

    def add_client(self, new_client: str | Client) -> None:
        if isinstance(new_client, str):
            # A name asks for a default client, which one of its class already
            # in the trace serves: none of the caller's settings are lost.
            client_type = _named_client_type(new_client)
            existing = self.client_manager.get_client(client_type.NAME)
            if isinstance(existing, client_type):
                return
            new_client = client_type()
        self.client_manager.add_clients([self._normalize_client(new_client)])

    def finalize(self):
        self.client_manager.finalize()
        launches.append(self.client_manager.launch)


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

    def _interpreter_runner(self, runner: Any, interpreted_fn: Any) -> Any:
        if self._is_autotuner(runner):
            runner.fn = interpreted_fn
            # Kernel Cache: replace the benchmark with a dummy to skip performance testing.
            runner._do_bench = self.dummy_benchmarker
            return runner
        if self._is_heuristics(runner):
            runner.fn = interpreted_fn
            return runner
        return interpreted_fn

    def _warmup_runner(self, runner: Any, jit_fn: Any | None) -> Any | None:
        if not (self._is_autotuner(runner) or self._is_heuristics(runner)):
            return jit_fn
        if jit_fn is None:
            return None
        warmup_runner = deepcopy(runner)
        warmup_runner.fn = jit_fn
        return warmup_runner

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
        self.runner = self._interpreter_runner(runner, self.interpreted_fn)
        self.warmup_runner = self._warmup_runner(runner, self.jit_fn)

        self.arg_names = runner.arg_names

        self.fn = runner

        TraceInterface.__init__(self, client)

        self._copy_callable_attrs(runner, self.base_fn, src_fallback=self.jit_fn)

    def run(self, *args, **kwargs):
        with self.client_manager.patch_warmup(self.jit_fn):
            if self.warmup_runner:
                self.warmup_runner.warmup(*args, **kwargs)

        with self.client_manager.patch_run(self.base_fn, frontend_name="triton"):
            kwargs.update({"client_manager": self.client_manager})
            kwargs.update({"jit_fn": self.jit_fn})
            ret = self.runner.run(*args, **kwargs)
            self.finalize()
            return ret

    def __call__(self, *args, **kwargs):
        # When a traced JIT function is called from within another JIT function,
        # we need to execute the underlying function directly

        # check that client sets match for calling and called functions
        from .frontend import triton as triton_frontend

        outer_client_manager = triton_frontend.frontend.current_client_manager()
        if outer_client_manager is not None:
            outer_clients = set(outer_client_manager.clients)
            inner_clients = set(self.client_manager.clients)
            if outer_clients != inner_clients:
                raise RuntimeError(
                    "nested traced calls require matching clients; "
                    f"outer={outer_clients}, inner={inner_clients}"
                )

        return self.interpreted_fn(*args, **kwargs)

    def warmup(self, *args, **kwargs):
        with self.client_manager.patch_warmup(self.jit_fn):
            if self.warmup_runner:
                self.warmup_runner.warmup(*args, **kwargs)


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
        if self.frontend_name == "nki_beta2" and pre_trace:
            import nki

            kwargs.pop("warmup", None)
            grid = kwargs.pop("grid", None)
            nki.trace(self.func, grid=grid, platform_target=platform_target).specialize(
                *args, **kwargs
            )
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
        self.runner = self._interpreter_runner(runner, self.interpreted_fn)

        TraceInterface.__init__(self, client)
        self._copy_callable_attrs(runner, self.base_fn)

    def run(self, *args, **kwargs):
        grid = kwargs.get("grid")
        if grid is None:
            raise TypeError(
                "GluonTrace.run() missing required keyword argument: 'grid'"
            )

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
