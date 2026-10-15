"""The structured record an IR client adds to ``Launch.records``.

Plain frozen records: statuses and scopes are strings in the reporting
client's own vocabulary, and nothing here aggregates, ranks or interprets
them. Fields hold only strings, ints, tuples, dicts and these records, so
verdicts are picklable and ``tilelens.save()`` holds them, provided
the config values are ones a trace can hold too.
``SourceLocation`` and ``Refusal`` are hashable; ``ConfigVerdict`` and
``IRVerdict`` compare by value but are not hashable (a config dict has no
hash).

Importing this module does not import Triton.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any


def _plain_str(value: Any, what: str) -> str:
    # A str subclass (e.g. the reader's TTIRKind enum) is saved as, and
    # loads back as, its plain string; holding that string keeps a record's
    # type, equality, hash and repr the same on both sides of a save.
    if not isinstance(value, str):
        raise TypeError(f"{what} must be a str, not {type(value).__name__}")
    return str.__str__(value)


@dataclass(frozen=True)
class SourceLocation:
    """A user-source location: file, 1-based line, and column if known."""

    file: str
    line: int
    col: int | None = None


@dataclass(frozen=True)
class Refusal:
    """Why (part of) a launch was not analyzed: a reader's UnsupportedTTIR,
    or a refusal a client defines itself."""

    kind: str
    message: str
    # The refused op's line in the IR text, and its user-source location.
    # ``loc`` also takes any object with ``file`` and ``line`` (and
    # optionally ``col``) attributes, such as the TTIR reader's source loc,
    # and holds it as a SourceLocation.
    line_no: int | None = None
    loc: SourceLocation | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _plain_str(self.kind, "Refusal.kind"))
        object.__setattr__(self, "message", _plain_str(self.message, "Refusal.message"))
        loc = self.loc
        if loc is None or isinstance(loc, SourceLocation):
            return
        if not (hasattr(loc, "file") and hasattr(loc, "line")):
            raise TypeError(
                "Refusal.loc must be a SourceLocation, None or an object with "
                f"file and line attributes, not {type(loc).__name__}"
            )
        object.__setattr__(
            self, "loc", SourceLocation(loc.file, loc.line, getattr(loc, "col", None))
        )

    @classmethod
    def from_exception(cls, exc: BaseException) -> Refusal:
        """Copy a refusal exception's structured fields (``kind``, and
        ``message`` / ``line_no`` / ``loc`` where it has them); the reader's
        source loc becomes a SourceLocation."""
        message = getattr(exc, "message", None)
        return cls(
            kind=getattr(exc, "kind"),
            message=str(exc) if message is None else message,
            line_no=getattr(exc, "line_no", None),
            loc=getattr(exc, "loc", None),
        )


@dataclass(frozen=True)
class ConfigVerdict:
    """One analyzed (or failed) config of a launch."""

    # The compiled specialization (kernel hash); None for a config that
    # produced no kernel.
    specialization: Hashable
    # The config kwargs the Autotuner/Heuristics layers added.
    config: Mapping[str, Any]
    status: str
    refusal: Refusal | None = None
    n_reports: int = 0

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", dict(self.config))


@dataclass(frozen=True)
class IRVerdict:
    """One IR client's result for one traced launch."""

    client: str  # the client's NAME
    status: str
    # What the status holds for (e.g. a proof's quantifier scope).
    scope: str | None = None
    refusal: Refusal | None = None
    per_config: tuple[ConfigVerdict, ...] = ()
    notes: tuple[str, ...] = ()

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        for name in ("per_config", "notes"):
            if isinstance(getattr(self, name), str):
                # tuple() would split it into characters
                raise TypeError(f"IRVerdict.{name} takes a sequence, not a str")
        per_config = tuple(self.per_config)
        for item in per_config:
            if not isinstance(item, ConfigVerdict):
                raise TypeError(
                    "IRVerdict.per_config items must be ConfigVerdicts, "
                    f"not {type(item).__name__}"
                )
        notes = tuple(_plain_str(note, "an IRVerdict note") for note in self.notes)
        object.__setattr__(self, "per_config", per_config)
        object.__setattr__(self, "notes", notes)
