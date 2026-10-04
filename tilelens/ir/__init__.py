"""Compiled-IR layer: read Triton-compiled kernels (TTIR) for the IR-mode clients.

Exports resolve on first access, so importing ``tilelens.ir`` imports neither
Triton nor its MLIR bindings.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "IRClient": ("tilelens.ir.client", "IRClient"),
    "ArtifactLog": ("tilelens.ir.capture", "ArtifactLog"),
    "CompiledArtifacts": ("tilelens.ir.capture", "CompiledArtifacts"),
    "CompiledSpecialization": ("tilelens.ir.capture", "CompiledSpecialization"),
    "CompileFailure": ("tilelens.ir.capture", "CompileFailure"),
    "ParseCache": ("tilelens.ir.capture", "ParseCache"),
    "ParseOutcome": ("tilelens.ir.capture", "ParseOutcome"),
    "LaunchBinding": ("tilelens.ir.launch", "LaunchBinding"),
    "TensorFacts": ("tilelens.ir.launch", "TensorFacts"),
    "bind_launch": ("tilelens.ir.launch", "bind_launch"),
    "IRVerdict": ("tilelens.ir.verdict", "IRVerdict"),
    "ConfigVerdict": ("tilelens.ir.verdict", "ConfigVerdict"),
    "Refusal": ("tilelens.ir.verdict", "Refusal"),
    "SourceLocation": ("tilelens.ir.verdict", "SourceLocation"),
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
