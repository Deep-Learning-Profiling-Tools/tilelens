import copy
import inspect
from contextlib import contextmanager
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


def _without_warmup(kwargs: dict[str, Any]) -> dict[str, Any]:
    # Launch kwargs carry warmup=False; the warmup entry points set their own.
    return {k: v for k, v in kwargs.items() if k != "warmup"}


def _rebind_closure(fn: Any, old: Any, new: Any) -> Any:
    """Return ``fn`` with the closure cells that hold ``old`` pointing at ``new``."""
    closure = getattr(fn, "__closure__", None)
    if not closure:
        return fn
    cells = []
    for cell in closure:
        try:
            value = cell.cell_contents
        except ValueError:  # empty cell
            value = None
        cells.append(types.CellType(new) if value is old else cell)
    if all(a is b for a, b in zip(cells, closure)):
        return fn
    rebound = types.FunctionType(
        fn.__code__, fn.__globals__, fn.__name__, fn.__defaults__, tuple(cells)
    )
    rebound.__kwdefaults__ = fn.__kwdefaults__
    return rebound


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
        return self._rebuild_runner(runner, interpreted_fn, interpreted=True)

    def _warmup_runner(self, runner: Any, jit_fn: Any | None) -> Any | None:
        if jit_fn is None:
            return None
        return self._rebuild_runner(runner, jit_fn, interpreted=False)

    def _rebuild_runner(self, runner: Any, leaf: Any, *, interpreted: bool) -> Any:
        """Rebuild the Autotuner/Heuristics chain of ``runner`` on top of ``leaf``.

        Every layer is shallow-copied down to the kernel, which ``leaf``
        replaces, so the user's runner is never mutated. No deepcopy: a real
        JITFunction holds an RLock. A nested trace is looked through so the
        layers it wraps are kept.
        """
        if isinstance(runner, (TritonTrace, GluonTrace)):
            runner = runner.fn
        if not (self._is_autotuner(runner) or self._is_heuristics(runner)):
            return leaf
        layer = copy.copy(runner)
        layer.fn = self._rebuild_runner(runner.fn, leaf, interpreted=interpreted)
        if self._is_autotuner(layer):
            self._isolate_autotuner(runner, layer, interpreted=interpreted)
        elif not interpreted:
            layer.warmup = self._heuristics_warmup(layer)
        return layer

    def _isolate_autotuner(
        self, original: Any, layer: Any, *, interpreted: bool
    ) -> None:
        # copy.copy shares the cache dict Autotuner.run fills in; give the copy
        # its own, so a trace never changes what the user's autotuner picks.
        layer.cache = {}
        # Interpreter timings must never reach Triton's on-disk autotune cache.
        layer.cache_results = False
        # Triton's own reset_to_zero/restore_value hooks close over the
        # Autotuner they were built for (restore_copies lives on it); point
        # them at the copy.
        for name in ("pre_hook", "post_hook"):
            if not getattr(layer, f"user_defined_{name}", False):
                hook = _rebind_closure(getattr(layer, name), original, layer)
                setattr(layer, name, hook)
        if interpreted:
            # Kernel Cache: replace the benchmark with a dummy to skip performance testing.
            layer._do_bench = self.dummy_benchmarker
            # do_bench is a cached_property; drop a value the original cached.
            layer.__dict__.pop("do_bench", None)

    @staticmethod
    def _heuristics_warmup(layer: Any) -> Callable:
        # Heuristics inherits KernelInterface.warmup, which calls
        # run(warmup=True) and so bypasses fn.warmup, where patch_warmup
        # collects the warmup votes. Warm up like Autotuner.warmup does
        # instead: fill in the heuristic kwargs as Heuristics.run does, then
        # call fn.warmup.
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
        self.runner = self._interpreter_runner(runner, self.interpreted_fn)
        self.warmup_runner = self._warmup_runner(runner, self.jit_fn)

        self.arg_names = runner.arg_names

        self.fn = runner

        TraceInterface.__init__(self, client)

        self._copy_callable_attrs(runner, self.base_fn, src_fallback=self.jit_fn)

    def run(self, *args, **kwargs):
        self._voted_warmup(*args, **kwargs)

        with self.client_manager.patch_run(self.base_fn, frontend_name="triton"):
            kwargs.update({"client_manager": self.client_manager})
            kwargs.update({"jit_fn": self.jit_fn})
            ret = self.runner.run(*args, **kwargs)
            self.finalize()
            return ret

    def _real_compile_window(self):
        return _unwrapped_trace_globals(self.base_fn)

    def __call__(self, *args, **kwargs):
        # A traced JIT function called from inside a traced kernel's
        # interpreted run executes its interpreted function directly.
        from triton import knobs

        from .frontend import triton as triton_frontend

        outer_client_manager = triton_frontend.frontend.current_client_manager()
        if outer_client_manager is None:
            if knobs.runtime.interpret:
                # Triton's own interpreter (TRITON_INTERPRET=1), e.g. running
                # an untraced kernel: the helper runs untraced, as Triton's
                # InterpretedFunction would.
                return self.interpreted_fn(*args, **kwargs)
            # Outside any interpreter, the caller is a real compile that
            # reached the trace as a plain Python callable, through a path
            # _unwrapped_trace_globals and _untraced_call_args do not cover.
            # Running the interpreter here would patch triton.language for
            # every later compile.
            raise TypeError(
                f"{self.__name__} is a tilelens-traced Triton function called "
                "outside a traced launch's interpreter, with TRITON_INTERPRET "
                "off: e.g. by a real compile that reached it through a path "
                "tilelens does not unwrap; use its JITFunction "
                f"({self.__name__}.jit_fn) there instead."
            )
        # check that client sets match for calling and called functions
        outer_clients = set(outer_client_manager.clients)
        inner_clients = set(self.client_manager.clients)
        if outer_clients != inner_clients:
            raise RuntimeError(
                "nested traced calls require matching clients; "
                f"outer={outer_clients}, inner={inner_clients}"
            )

        return self.interpreted_fn(*args, **kwargs)

    def warmup(self, *args, **kwargs):
        return self._voted_warmup(*args, **kwargs)

    def _voted_warmup(self, *args, **kwargs):
        # The pre/post_warmup vote: a real compile only if some client asks.
        if not self.warmup_runner:
            return None
        with self.client_manager.patch_warmup(
            self.jit_fn,
            compile_context=self._real_compile_window,
            real_args=_untraced_call_args,
        ):
            return self.warmup_runner.warmup(*args, **_without_warmup(kwargs))


def _untraced_call_args(
    jit_fn: Any, args: tuple, kwargs: dict[str, Any]
) -> tuple[tuple, dict[str, Any]]:
    """One JITFunction warmup call's arguments as a real compile must see
    them: each TritonTrace passed as an argument (also inside a tuple), or
    bound as a parameter default, replaced by its JITFunction (the default
    passed explicitly). Triton's code generator treats any other callee as
    plain Python and would call TritonTrace.__call__.
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


def _code_names(code: types.CodeType) -> set[str]:
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _code_names(const)
    return names


def _is_triton_internal(value: Any) -> bool:
    # Triton's own modules and stdlib functions hold no user traces.
    module: Any = getattr(value, "__module__", None)
    if isinstance(value, types.ModuleType):
        module = value.__name__
    return isinstance(module, str) and module.split(".")[0] == "triton"


def _traced_references(
    base_fn: Callable | None,
) -> list[tuple[dict, str, "TritonTrace"]]:
    """Every (namespace, name, trace) binding a real compile of ``base_fn``
    resolves to a TritonTrace.

    Triton resolves a callee from the caller's globals and then through
    module attributes (``helpers.fn``, ``pkg.api.fn``). The walk follows the
    same paths, filtered by the names each function's code mentions
    (co_names), and continues into every referenced JIT function's own code,
    so helpers of helpers are found and unrelated bindings are left alone.
    Triton also resolves parameter default expressions in the globals; their
    names are not in co_names, so a global bound to a traced default counts
    too.
    """
    from triton import JITFunction

    found: list[tuple[dict, str, TritonTrace]] = []
    bindings: set[tuple[int, str]] = set()
    visited_fns: set[int] = set()
    pending: list[Any] = [base_fn]
    while pending:
        fn = pending.pop()
        code = getattr(fn, "__code__", None)
        fn_globals = getattr(fn, "__globals__", None)
        if not isinstance(code, types.CodeType) or not isinstance(fn_globals, dict):
            continue
        if id(fn) in visited_fns:
            continue
        visited_fns.add(id(fn))
        names = _code_names(code)
        traced_defaults = [
            d
            for d in getattr(fn, "__defaults__", None) or ()
            if isinstance(d, TritonTrace)
        ]
        if traced_defaults:
            names |= {
                name
                for name, value in fn_globals.items()
                if any(value is default for default in traced_defaults)
            }
        namespaces = [fn_globals]
        visited_namespaces = {id(fn_globals)}
        while namespaces:
            namespace = namespaces.pop()
            for name in names:
                value = namespace.get(name)
                if isinstance(value, TritonTrace):
                    if (
                        value.jit_fn is not None
                        and (id(namespace), name) not in bindings
                    ):
                        bindings.add((id(namespace), name))
                        found.append((namespace, name, value))
                    pending.append(value.base_fn)
                elif isinstance(value, JITFunction) and not _is_triton_internal(value):
                    pending.append(value.fn)
                elif isinstance(value, types.ModuleType) and not _is_triton_internal(
                    value
                ):
                    module_dict = getattr(value, "__dict__", None)
                    if (
                        isinstance(module_dict, dict)
                        and id(module_dict) not in visited_namespaces
                    ):
                        visited_namespaces.add(id(module_dict))
                        namespaces.append(module_dict)
    return found


@contextmanager
def _unwrapped_trace_globals(base_fn: Callable | None = None):
    """Temporarily bind each TritonTrace a real compile of ``base_fn`` would
    reach back to its JITFunction.

    Under the CLI wrappers every ``@triton.jit`` function, device functions
    included, becomes a TritonTrace. Triton's dependency walker and code
    generator only accept JITCallables as callees ("Unsupported function
    referenced"); the interpreter tolerates the wrapper through
    ``TritonTrace.__call__``, a real compile does not. Only the bindings the
    kernel's code can reach are swapped (see _traced_references), and each is
    restored on exit unless it was rebound in the meantime. Callees reached
    through closure variables are not covered (Triton's dependency walker
    rejects them); callees passed as arguments are mapped per call instead
    (_untraced_call_args).
    """
    swapped: list[tuple[dict, str, TritonTrace]] = []
    try:
        for namespace, name, trace in _traced_references(base_fn):
            if namespace.get(name) is trace:
                namespace[name] = trace.jit_fn
                swapped.append((namespace, name, trace))
        yield
    finally:
        for namespace, name, trace in reversed(swapped):
            if namespace.get(name) is trace.jit_fn:
                namespace[name] = trace


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
