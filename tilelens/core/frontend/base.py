from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from importlib import import_module
from typing import Any, TYPE_CHECKING

from tilelens.core.data import Op

if TYPE_CHECKING:
    from tilelens.core.client import Client


_MISSING = object()


class _LangPatchScope:
    """Tracks patched attributes so they can be restored."""

    def __init__(self) -> None:
        self._changes: list[tuple[object, str, object]] = []

    def set_attr(self, obj: object, name: str, value: object) -> None:
        original = getattr(obj, name, _MISSING)
        self._changes.append((obj, name, original))
        setattr(obj, name, value)

    def save_attr(self, obj: object, name: str) -> None:
        original = getattr(obj, name, _MISSING)
        self._changes.append((obj, name, original))

    def restore(self) -> None:
        while self._changes:
            obj, name, original = self._changes.pop()
            if original is _MISSING:
                if hasattr(obj, name):
                    delattr(obj, name)
            else:
                setattr(obj, name, original)


@dataclass
class AdapterResult:
    """
    Standardized callback arguments derived from frontend-specific op signatures.
    """

    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


def _missing_op_stub(attr: str, namespace: Any) -> Callable:
    """Create a placeholder for a missing op that raises at runtime."""

    def stub(*args: Any, **kwargs: Any) -> Any:
        raise AttributeError(
            f"tilelens failed to intercept '{attr}': "
            f"'{type(namespace).__name__}' has no attribute '{attr}'. "
            "This operation is not supported by your frontend version."
        )

    return stub


def passthrough_adapter(*args: Any, **kwargs: Any) -> AdapterResult:
    """Return arguments unchanged for clients that expect the original signature."""
    return AdapterResult(*args, **kwargs)


@dataclass
class Frontend:
    """
    Defines one DSL frontend's patchable ops, adapters, and runtime patch hooks.
    """

    name: str
    builder: Any
    original_ops: dict[Any, dict[str, Callable]]
    # Default adapters keyed by operation type. These handle the common
    # signature for all callables that map to the same normalized Op.
    adapters: dict[type[Op], Callable[..., AdapterResult]]
    namespaces: dict[Any, dict[str, type[Op]]]

    @classmethod
    def from_namespaces(
        cls,
        *,
        name: str,
        builder: Any,
        namespaces: dict[Any, dict[str, type[Op]]],
        adapters: dict[type[Op], Callable[..., AdapterResult]],
    ) -> "Frontend":
        original_ops: dict[Any, dict[str, Callable]] = {
            namespace: {} for namespace in namespaces
        }
        for namespace, attrs in namespaces.items():
            for attr in attrs:
                if hasattr(namespace, attr):
                    original_ops[namespace][attr] = getattr(namespace, attr)
                else:
                    original_ops[namespace][attr] = _missing_op_stub(attr, namespace)

        for attrs in namespaces.values():
            for op_type in attrs.values():
                adapters.setdefault(op_type, passthrough_adapter)

        return cls(
            name=name,
            builder=builder,
            original_ops=original_ops,
            adapters=adapters,
            namespaces=namespaces,
        )

    def maybe_yield_for_multism(self) -> None:
        return None

    def wrap_kernel(self, kernel: Any, client: "str | Client") -> Any:
        """Create this frontend's traced callable, including CLI adaptation."""
        raise NotImplementedError(
            f"Frontend '{self.name}' does not implement kernel wrapping"
        )

    def run_op_overrider(
        self,
        op: Callable,
        op_type: type[Op],
        op_overrider: Callable,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ):
        return op_overrider(*args, **kwargs)

    @staticmethod
    def concrete_fn_for_op_type(
        namespace_ops: dict[str, Callable], op_type: type[Op], original_op: Callable
    ) -> Callable:
        return original_op

    @staticmethod
    def symbolic_ops_for_op_type(op_type: type[Op]) -> tuple[str, ...]:
        return ()

    def normalize_symbolic_value(self, value: Any) -> Any:
        """Convert frontend values/types into frontend-neutral symbolic metadata."""
        from tilelens.core.symbolic_metadata import normalize_symbolic_value

        return normalize_symbolic_value(value)

    def to_frontend_symbolic_value(self, value: Any) -> Any:
        """Convert neutral symbolic values/types back to frontend runtime objects."""
        return value

    def from_frontend_symbolic_value(self, value: Any) -> Any:
        """Normalize concrete replay returns from frontend objects to neutral values."""
        return self.normalize_symbolic_value(value)

    def wrap_symbolic_loop_index(self, expr: Any, dtype: Any) -> Any:
        """Wrap a symbolic loop index in the frontend's scalar tensor type if needed."""
        return expr

    def wrap_symbolic_concrete_fn(self, concrete_fn: Callable) -> Callable:
        """Wrap original ops that need frontend-specific symbolic replay conversion."""
        return concrete_fn

    def prepare_patched_op(
        self, namespace: Any, op_type: type[Op], original_op: Callable
    ) -> None:
        symbolic_ops = self.symbolic_ops_for_op_type(op_type)
        if not symbolic_ops:
            return

        from tilelens.clients.symbolic_engine import SymbolicExpr

        SymbolicExpr.set_frontend_hooks(
            self.normalize_symbolic_value,
            self.wrap_symbolic_loop_index,
        )
        concrete_fn = self.concrete_fn_for_op_type(
            self.original_ops[namespace], op_type, original_op
        )
        concrete_fn = self.wrap_symbolic_concrete_fn(concrete_fn)
        for symbolic_op in symbolic_ops:
            SymbolicExpr.set_concrete_fn(symbolic_op, concrete_fn)

    def patch_for_loop(self) -> None:
        return None

    def unpatch_for_loop(self) -> None:
        return None

    def patch_lang(self, fn: Callable, client_manager: Any = None) -> _LangPatchScope:
        return _LangPatchScope()

    def patch_calls(self):
        return nullcontext()


FRONTEND_REGISTRY: dict[str, Frontend] = {}
LANG_PATCH_SCOPES: dict[str, list[Any]] = {
    "triton": [],
    "nki": [],
    "nki_beta2": [],
}


def register_frontend(frontend: Frontend) -> Frontend:
    FRONTEND_REGISTRY[frontend.name] = frontend
    LANG_PATCH_SCOPES.setdefault(frontend.name, [])
    return frontend


def get_frontend(name: str) -> Frontend:
    if name not in FRONTEND_REGISTRY:
        module_name = f"tilelens.core.frontend.{name}"
        try:
            import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name:
                raise
            raise ValueError(f"Unknown frontend: {name}") from exc
    try:
        return FRONTEND_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown frontend: {name}") from exc
