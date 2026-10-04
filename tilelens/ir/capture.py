"""Per-launch compiled artifacts, and a content-addressed parse cache.

``ArtifactLog`` records what the core's IR hooks delivered during one traced
launch: per compiled specialization its declared IR stages and compile
metadata plus the LaunchBindings seen for it, and every compile failure.
``ParseCache`` runs a reader once per distinct text and keeps what it gave,
a graph or a typed refusal, so a refusal's kind survives cache hits.

Mechanism only: which specialization counts, what an absent stage means and
what a refusal or an error becomes are the client's calls. Neither class
raises for a bad kernel, text or reader.

Importing this module does not import Triton or the TTIR reader.
"""

from __future__ import annotations

import builtins
import hashlib
import sys
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from importlib import import_module
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from .launch import LaunchBinding, bind_launch, config_kwargs

if TYPE_CHECKING:
    from ..core.client import LaunchCall, LaunchEvent


@dataclass(frozen=True)
class CompiledArtifacts:
    """What one compiled specialization left in ``kernel.asm`` and
    ``kernel.metadata``."""

    # The declared stages the kernel holds: text, or bytes for a binary
    # stage. A declared stage the kernel lacks (e.g. under
    # TRITON_STORE_BINARY_ONLY) is absent.
    stages: Mapping[str, Any]
    # "backend", "arch", "num_warps", "num_stages", "shared", "name" from
    # the compile metadata (None where it has none, e.g. "shared" for a
    # kernel compiled only through TTIR), and "config": the config kwargs of
    # the call that first produced the specialization.
    meta: Mapping[str, Any]
    # What could not be read, "; "-joined; stages and meta hold the rest.
    error: str | None = None


@dataclass(frozen=True)
class CompiledSpecialization:
    """One specialization a traced launch compiled, and every binding it was
    delivered with (one per before_launch event)."""

    specialization: Hashable
    artifacts: CompiledArtifacts
    bindings: tuple[LaunchBinding, ...]

    @property
    def config(self) -> Mapping[str, Any]:
        return self.artifacts.meta["config"]


@dataclass(frozen=True)
class CompileFailure:
    """A call of the launch that failed to compile (compile_failed)."""

    # The exception the host compile raised, whole (a CompilationError
    # with its source excerpt and the errors it was raised from).
    error: BaseException | None
    config: Mapping[str, Any]
    # The GPUTarget the compile was for (LaunchEvent.target).
    target: Any = None
    # The JITFunction that failed to compile (LaunchEvent.jit_fn).
    jit_fn: Any = None


_METADATA_FIELDS = ("num_warps", "num_stages", "shared", "name")


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _read_artifacts(
    kernel: Any, stages: frozenset[str], config: Mapping[str, Any]
) -> CompiledArtifacts:
    texts: dict[str, Any] = {}
    meta: dict[str, Any] = dict.fromkeys(("backend", "arch", *_METADATA_FIELDS))
    meta["config"] = config
    errors: list[str] = []
    try:
        asm = kernel.asm
        for stage in sorted(stages):
            try:
                texts[stage] = asm[stage]
            except KeyError:
                pass
            except Exception as exc:  # e.g. "sass" needs cuobjdump
                errors.append(f"asm[{stage!r}]: {_describe(exc)}")
    except Exception as exc:
        errors.append(f"asm: {_describe(exc)}")
    try:
        metadata = kernel.metadata
        target = getattr(metadata, "target", None)
        meta["backend"] = getattr(target, "backend", None)
        meta["arch"] = getattr(target, "arch", None)
        for name in _METADATA_FIELDS:
            meta[name] = getattr(metadata, name, None)
    except Exception as exc:
        errors.append(f"metadata: {_describe(exc)}")
    return CompiledArtifacts(
        stages=MappingProxyType(texts),
        meta=MappingProxyType(meta),
        error="; ".join(errors) if errors else None,
    )


class ArtifactLog:
    """What one traced launch compiled, for an IR client that reads
    ``stages`` of each kernel.

    ``reset(call)`` starts a launch; ``record`` takes each before_launch
    event and ``record_failure`` each compile_failed event. Specializations
    and failures keep the order they were first seen in (the autotuner's
    config order).
    """

    def __init__(self, stages: Iterable[str]) -> None:
        self.stages = frozenset(stages)
        self.reset()

    def reset(self, call: LaunchCall | None = None) -> None:
        """Forget everything recorded; ``call`` is the launch about to start
        (it tells config kwargs from the caller's own, see config_kwargs)."""
        self.call = call
        self._compiled: dict[
            Hashable, tuple[CompiledArtifacts, list[LaunchBinding]]
        ] = {}
        self._failures: list[CompileFailure] = []

    def record(self, event: LaunchEvent) -> None:
        binding = bind_launch(event, self.call)
        entry = self._compiled.get(event.specialization)
        if entry is None:
            artifacts = _read_artifacts(event.kernel, self.stages, binding.config)
            self._compiled[event.specialization] = (artifacts, [binding])
        else:
            entry[1].append(binding)

    def record_failure(self, event: LaunchEvent) -> None:
        self._failures.append(
            CompileFailure(
                error=event.error,
                config=MappingProxyType(config_kwargs(event, self.call)),
                target=getattr(event, "target", None),
                jit_fn=getattr(event, "jit_fn", None),
            )
        )

    @property
    def specializations(self) -> tuple[CompiledSpecialization, ...]:
        return tuple(
            CompiledSpecialization(specialization, artifacts, tuple(bindings))
            for specialization, (artifacts, bindings) in self._compiled.items()
        )

    @property
    def failures(self) -> tuple[CompileFailure, ...]:
        return tuple(self._failures)


@dataclass(frozen=True)
class ParseOutcome:
    """A reader's result for one text: exactly one of the three is set,
    unless the reader itself returned None."""

    graph: Any = None
    # The reader's refusal exception (the TTIR reader's UnsupportedTTIR),
    # tracebacks dropped.
    refusal: BaseException | None = None
    # Any other exception the reader raised, as "Type: message".
    error: str | None = None


def content_key(text: str) -> str:
    """Stable SHA-256 of an IR text (a lone surrogate hashes as "?")."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _default_reader() -> Callable[..., Any]:
    # Resolved on every lookup, so a monkeypatched reader is picked up (and
    # keyed apart by its identity).
    return import_module(".ttir_reader", __package__).parse_ttir


def _default_refusal() -> type[BaseException] | None:
    try:
        return import_module(".ttir_reader", __package__).UnsupportedTTIR
    except Exception:  # no reader module: nothing can be its refusal
        return None


_EXCEPTION_GROUP = getattr(builtins, "BaseExceptionGroup", None)  # Python >= 3.11


def _without_frames(exc: BaseException, outer: BaseException | None) -> BaseException:
    # A cached refusal outlives its parse; its traceback, and those of every
    # exception chained to it, would keep the reader's frames alive. The
    # exception the caller was handling when it asked (``outer``, the chain's
    # implicit context) is the caller's: unlinked, never cleared.
    seen: set[int] = set()
    stack = [exc]
    while stack:
        link = stack.pop()
        if id(link) in seen:
            continue
        seen.add(id(link))
        link.__traceback__ = None
        if outer is not None and link.__context__ is outer:
            link.__context__ = None
        if outer is not None and link.__cause__ is outer:
            link.__cause__ = None
        stack.extend(x for x in (link.__cause__, link.__context__) if x is not None)
        if _EXCEPTION_GROUP is not None and isinstance(link, _EXCEPTION_GROUP):
            stack.extend(getattr(link, "exceptions", ()))
    return exc


class ParseCache:
    """Parse each distinct IR text once per reader and options.

    ``reader(text, **options)`` returns a graph or raises ``refusal`` (by
    default the TTIR reader's UnsupportedTTIR) to decline the text; any other
    exception is reported as an error and not cached, so a later lookup
    retries it. The default reader, ``tilelens.ir.ttir_reader.parse_ttir``,
    is imported at each lookup, not with this module; while that module
    cannot be imported, a lookup is an error outcome like any other reader
    error. Never raises (``Exception``s only; an interrupt still
    propagates); ``text`` is positional-only, so any option name reaches
    the reader.
    """

    def __init__(
        self,
        reader: Callable[..., Any] | None = None,
        *,
        refusal: type[BaseException] | None = None,
    ) -> None:
        self._reader = reader
        self._refusal = refusal
        self._outcomes: dict[Hashable, ParseOutcome] = {}

    def get(self, text: str, /, **options: Hashable) -> ParseOutcome:
        outer = sys.exc_info()[1]
        try:
            reader = self._reader if self._reader is not None else _default_reader()
            key = (
                content_key(text),
                reader,
                tuple(sorted(options.items())),
            )
            cached = self._outcomes.get(key)
        except Exception as exc:
            return ParseOutcome(error=_describe(exc))
        if cached is not None:
            return cached
        try:
            outcome = ParseOutcome(graph=reader(text, **options))
        except Exception as exc:
            refusal = self._refusal if self._refusal is not None else _default_refusal()
            if refusal is None or not isinstance(exc, refusal):
                return ParseOutcome(error=_describe(exc))
            outcome = ParseOutcome(refusal=_without_frames(exc, outer))
        self._outcomes[key] = outcome
        return outcome
