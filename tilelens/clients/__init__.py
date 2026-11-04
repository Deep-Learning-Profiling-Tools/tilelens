from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "Profiler": ("tilelens.clients.profiler.profiler", "Profiler"),
    "LoadStoreBytes": ("tilelens.clients.profiler.data", "LoadStoreBytes"),
    "OpTypeCounts": ("tilelens.clients.profiler.data", "OpTypeCounts"),
    "RaceDetector": ("tilelens.clients.race_detector.race_detector", "RaceDetector"),
    "Sanitizer": ("tilelens.clients.sanitizer.sanitizer", "Sanitizer"),
    "CompiledSanitizer": (
        "tilelens.clients.sanitizer.compiled.client",
        "CompiledSanitizer",
    ),
    "OutOfBoundsRecord": ("tilelens.clients.sanitizer.data", "OutOfBoundsRecord"),
    "CompiledSanitizerRecord": (
        "tilelens.clients.sanitizer.data",
        "CompiledSanitizerRecord",
    ),
    "SymbolicExpr": ("tilelens.clients.symbolic_engine", "SymbolicExpr"),
    "SymbolicClient": ("tilelens.clients.symbolic_engine", "SymbolicClient"),
    "RangeWrapper": ("tilelens.clients.symbolic_engine", "RangeWrapper"),
    "Tracer": ("tilelens.clients.tracer.tracer", "Tracer"),
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
