from contextlib import AbstractContextManager, contextmanager, nullcontext

from abc import ABC, abstractmethod
from typing import ClassVar, Any
from collections.abc import Callable
import threading

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
from .frontend.base import get_frontend
from .config import config as cfg


# (jit_fn, args, kwargs) -> (args, kwargs): the arguments a real compile of
# one JITFunction call must see, supplied by the trace.
RealArgs = Callable[[Any, tuple, dict], tuple[tuple, dict]]


class Client(ABC):
    NAME: ClassVar[str]

    def __init__(self) -> None:
        # Whether this client needs ASM information from kernel warmup
        self.collect_asm: bool = False
        # Storage for ASM information if collected
        self.asm_info: dict | None = None
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


class _WarmupGate:
    """The warmup one ClientManager.patch_warmup scope puts on a jit_fn.

    It calls through to ``inner``, the warmup it found, and hides
    ``below``, the instance-level warmup it replaced (None: there was none,
    so the class's). Overlapping scopes on one jit_fn (e.g. on two host
    threads) stack their gates and may close in any order: ``remove`` takes
    a gate out wherever it sits, so once every scope has closed, jit_fn
    holds exactly the warmup it held before the first one opened.
    """

    def __init__(self, jit_fn: Any, vote: Callable[[Any, tuple, dict], Any]):
        self.inner = jit_fn.warmup
        self.below = vars(jit_fn).get("warmup")
        self._vote = vote

    def __call__(self, *args, **kwargs):
        return self._vote(self.inner, args, kwargs)

    def remove(self, jit_fn: Any) -> None:
        top = vars(jit_fn).get("warmup")
        if top is self:
            # Leave no bound method behind in the instance dict: it would
            # hide a later class-level change to JITFunction.warmup.
            if self.below is None:
                del jit_fn.warmup
            else:
                jit_fn.warmup = self.below
            return
        # A gate put on later sits above this one: route it past this one.
        while isinstance(top, _WarmupGate):
            if top.below is self:
                top.inner, top.below = self.inner, self.below
                return
            top = top.below


class ClientManager:
    def __init__(self, clients: list[Client] | None = None):
        self.clients: dict[str, Client] = {}
        if clients:
            self.add_clients(clients)
        self.launch = Launch()
        self._lock = threading.Lock()
        self._clear_loop_hooks()

    def _lock_context(self):
        if cfg.num_sms > 1:
            return self._lock
        return nullcontext()

    def get_client(self, name: str) -> Client | None:
        return self.clients.get(name)

    def add_clients(self, new_clients_list: list[Client]) -> None:
        for new_client in new_clients_list:
            duplicate = any(
                isinstance(existing_client, new_client.__class__)
                for existing_client in self.clients.values()
            )
            if not duplicate:
                self.clients[new_client.NAME] = new_client

    @contextmanager
    def patch_warmup(
        self,
        jit_fn,
        compile_context: Callable[[], AbstractContextManager] = nullcontext,
        real_args: RealArgs | None = None,
    ):
        """Gate ``jit_fn.warmup`` on this manager's warmup votes for the
        scope. The real compile, and only it, runs inside
        ``compile_context()``, on the arguments ``real_args`` maps the call
        to. On exit the scope takes its gate out again (see _WarmupGate).
        """
        if not hasattr(jit_fn, "warmup"):
            yield
            return

        def vote(warmup, args, kwargs):
            # Poll every client, also after a True vote, so each one sees
            # pre_warmup_callback before the post_warmup_callback all get.
            votes = [
                client.pre_warmup_callback(jit_fn, *args, **kwargs)
                for client in self.clients.values()
            ]
            if not any(votes):
                return None
            kwargs.pop("warmup", None)
            if real_args is not None:
                args, kwargs = real_args(jit_fn, args, kwargs)
            with compile_context():
                ret = warmup(*args, **kwargs)
            for client in self.clients.values():
                client.post_warmup_callback(jit_fn, ret)
            return ret

        gate = _WarmupGate(jit_fn, vote)
        jit_fn.warmup = gate
        try:
            yield
        finally:
            gate.remove(jit_fn)

    @contextmanager
    def patch_run(self, fn, frontend_name: str):
        frontend = get_frontend(frontend_name)
        namespaces = frontend.namespaces
        # Every launch gets its own Launch, so the entries TraceInterface
        # appends to `launches` stay distinct.
        self.launch = Launch()
        with patch_calls(frontend_name):
            lang_patched = False
            try:
                # Collect all for-loop callbacks from clients
                all_loop_callbacks = []
                for client in self.clients.values():
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
                lang_patched = True
                yield
            finally:
                if lang_patched:
                    unpatch_lang(frontend_name)
                for namespace, attrs in namespaces.items():
                    for attr, op in attrs.items():
                        unpatch_op(namespace, attr, frontend_name)
                unpatch_for_loop(frontend_name)
                self._clear_loop_hooks()

    def pre_run_callback(self, fn: Callable) -> bool:
        with self._lock_context():
            rets = [client.pre_run_callback(fn) for client in self.clients.values()]
            return all(rets) if rets else True

    def post_run_callback(self, fn: Callable) -> bool:
        with self._lock_context():
            rets = [client.post_run_callback(fn) for client in self.clients.values()]
            return any(rets)

    def finalize(self) -> None:
        with self._lock_context():
            self.launch.records = []
            # Finalize every client even if a peer raises (SystemExit
            # included), so none carries this launch's state into the next;
            # then re-raise the first failure.
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

    def arg_callback(self, name, arg, arg_cvt):
        with self._lock_context():
            if hasattr(arg, "data_ptr"):
                self.launch.tensors.add(arg)
            for client in self.clients.values():
                client.arg_callback(name, arg, arg_cvt)

    def grid_callback(self, grid: tuple[int]):
        with self._lock_context():
            self.launch.grid = grid
            for client in self.clients.values():
                client.grid_callback(grid)

    def grid_idx_callback(self, grid_idx: tuple[int, ...]):
        with self._lock_context():
            for client in self.clients.values():
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
