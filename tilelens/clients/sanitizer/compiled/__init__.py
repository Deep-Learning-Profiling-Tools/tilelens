"""The compiled-mode sanitizer (``Sanitizer(compile=True)``): out-of-bounds,
integer-overflow and division-by-zero checks over a kernel's compiled TTIR,
instantiated per launch with its LaunchBinding.

``oob`` is the evaluator: an ``AccessGraph`` from ``tilelens.ir.ttir_reader``
and a launch's binding in, a ``CheckResult`` out. ``client`` is the trace
client, ``CompiledSanitizer``, which runs it on every config a launch
compiled.

Exports resolve on first access, so importing this package imports neither
Z3 nor Triton.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "CompiledSanitizer": (
        "tilelens.clients.sanitizer.compiled.client",
        "CompiledSanitizer",
    ),
    "CheckResult": ("tilelens.clients.sanitizer.compiled.oob", "CheckResult"),
    "Finding": ("tilelens.clients.sanitizer.compiled.oob", "Finding"),
    "SanitizerKind": ("tilelens.clients.sanitizer.compiled.oob", "SanitizerKind"),
    "check_graph": ("tilelens.clients.sanitizer.compiled.oob", "check_graph"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
