"""What a config's failed host compile means for an IR client's launch.

A config that failed to compile never fails the launch: the target is the
client's choice, not the machine's. ``compiles_nowhere`` tells an error no
target compiles past (the config never launches anywhere: a note, see
``nowhere_note``) from one that may be the target's (the config was not
checked: ``failure_refusal``, kind ``COMPILE_FAILED``, naming how to check
it for another target); a host compile that could not run at all is
``HOST_COMPILE_UNAVAILABLE``. ``none_compiled`` is the refusal of a launch
whose every config was only noted, ``nothing_compiled`` the message of one
that compiled nothing and failed nothing.

Each refusal and note names where the kernel failed (``file:line`` of the
kernel's code the error points at, else of its ``def``) and the innermost
error in one line; the whole error stays on the CompileFailure. The
client's own words (how to name another target, which eager client checks
an uncompiled launch, how a config is shown) are passed in.

Importing this module does not import Triton.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..core.client import LanguagePatchedError
from ..core.host_compile import (
    bind_failed,
    format_ir_target,
    host_compile_unavailable,
    target_queried,
    unknown_options,
)
from .capture import ArtifactLog, CompileFailure
from .verdict import Refusal, SourceLocation

# A config failed to compile for the IR target with an error that need not
# hold for another target, its call does not bind the kernel's parameters,
# or no config compiled at all.
COMPILE_FAILED = "compile-failed"
# The host compile could not run for a config.
HOST_COMPILE_UNAVAILABLE = "host-compile-unavailable"


def nothing_compiled(log: ArtifactLog, eager_client: str) -> str:
    """Why a launch with neither a compiled config nor a failed one was not
    checked; ``eager_client`` (e.g. ``"Sanitizer()"``) checks a launch
    without a JITFunction."""
    if log.call is not None and not log.call.capture:
        return (
            "no compiled kernel to check: the launch has no JITFunction "
            "(TRITON_INTERPRET=1, an InterpretedFunction runner, Gluon or NKI); "
            f"the eager {eager_client} checks such launches"
        )
    return "no config of the launch was compiled, so nothing was checked"


def compiles_nowhere(error: BaseException | None) -> bool:
    """Whether a config's host compile error says the config compiles for
    no target, so it never launches anywhere (a note), rather than perhaps
    only not for the client's target (unsupported, compile-failed).

    The rule, by exception type: the error the kernel's code raised (below
    the CompilationErrors Triton's code generator wraps a called @jit
    helper's or builtin's error in, following ``__cause__``) is a failing
    ``tl.static_assert`` (CompileTimeAssertionFailure, which Triton's
    autotuner drops a config for too) or a Python construct the code
    generator never accepts (UnsupportedLanguageConstruct), and the compile
    never asked Triton's driver anything (tilelens.core.host_compile.
    target_queried: nor did an earlier compile of the kernel for the target):
    a static_assert on ``tl.target_info``, on a device query the kernel's
    code caught, or in a branch taken for the target only, is the target's.
    Every other error may be the target's: Triton raises a target's
    refusals (``num_ctas > 1`` below sm90, an fp8 type the target lacks,
    16-bit descriptor atomic min/max without native TMA, a dot shape the
    target's MMA lacks, a PTXAS failure) as plain ValueError /
    AssertionError / TypeError / RuntimeError, often wrapped in a
    CompilationError like an error of the kernel's own code, so none of
    them is taken to hold for every target. So is an unknown error (None).
    A call that does not bind (bind_failed) never launches either, but the
    untraced program raises for it (so does a traced launch: only an
    event built outside the core delivers one), so it is no note; a host
    compile that could not run (HostCompileUnavailable, LanguagePatchedError)
    is no compile error at all.
    """
    if error is None or host_compile_unavailable(error) is not None:
        return False
    from triton.compiler.errors import (
        CompilationError,
        CompileTimeAssertionFailure,
        UnsupportedLanguageConstruct,
    )

    target_free = (CompileTimeAssertionFailure, UnsupportedLanguageConstruct)
    seen: set[int] = set()
    link: BaseException = error
    while (
        isinstance(link, CompilationError)
        and not isinstance(link, target_free)
        and link.__cause__ is not None
        and id(link) not in seen
    ):
        seen.add(id(link))
        link = link.__cause__
    return isinstance(link, target_free) and not target_queried(error)


def _for_target(failure: CompileFailure) -> str:
    if failure.target is None:
        return ""
    return f" for {format_ir_target(failure.target)}"


def _error_chain(error: BaseException) -> list[BaseException]:
    """``error`` and the errors behind it, outermost first, as Triton's code
    generator nests them: a CompilationError raised from (``__cause__``),
    or ``from None`` while handling (the suppressed ``__context__``), the
    error of a called @jit helper, a builtin or the kernel's own code. Only
    CompilationErrors are followed."""
    from triton.compiler.errors import CompilationError

    chain = [error]
    link: BaseException | None = error
    while isinstance(link, CompilationError):
        link = link.__cause__ or (
            link.__context__ if link.__suppress_context__ else None
        )
        if link is None or any(link is seen for seen in chain):
            break
        chain.append(link)
    return chain


def _summary(error: BaseException) -> str:
    """One line for a compile error: the innermost error's type and the
    first line of its message (a CompilationError's own message, without
    the source excerpt it formats around it)."""
    from triton.compiler.errors import CompilationError

    inner = _error_chain(error)[-1]
    if isinstance(inner, CompilationError):
        text = str(getattr(inner, "error_message", None) or "")
    else:
        text = str(inner)
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return f"{type(inner).__name__}: {first}" if first else type(inner).__name__


def _kernel_site(jit_fn: Any, error: BaseException | None = None) -> Any:
    """Where in ``jit_fn``'s source file ``error`` was raised: the line of
    the kernel's own code the code generator names for it (for an error in
    a called @jit helper, the call), else the kernel's ``def`` line; None
    for a JITFunction without source (e.g. a stand-in)."""
    fn = getattr(jit_fn, "fn", None)
    code = getattr(fn, "__code__", None)
    start = getattr(jit_fn, "starting_line_number", None)
    raw_src = getattr(jit_fn, "raw_src", None)
    if code is None or not isinstance(start, int) or not raw_src:
        return None
    # The line of ``def`` after the decorators, as Triton counts it.
    offset = next(
        (i for i, line in enumerate(raw_src) if line.strip().startswith("def ")), 0
    )
    def_line = start + offset
    src = getattr(jit_fn, "src", None)
    for link in [] if error is None else _error_chain(error):
        lineno = getattr(getattr(link, "node", None), "lineno", None)
        if src is not None and getattr(link, "src", None) == src and lineno:
            # The code generator's lines count from the ``def`` line.
            return SourceLocation(code.co_filename, def_line + lineno - 1)
    return SourceLocation(code.co_filename, def_line)


def _at(site: Any) -> str:
    return "" if site is None else f"{site.file}:{site.line}: "


def failure_refusal(failure: CompileFailure, retarget: str) -> Refusal:
    """A failed config's refusal: why its host compile failed (see
    compiles_nowhere), where, and how to check it for another target where
    another target may compile it (``retarget``, e.g.
    ``"Sanitizer(compile=True, target=...), or TILELENS_IR_TARGET"``)."""
    error = failure.error
    site = _kernel_site(failure.jit_fn, error)
    unavailable = None if error is None else host_compile_unavailable(error)
    if unavailable is not None:
        return Refusal(
            HOST_COMPILE_UNAVAILABLE,
            f"{_at(site)}{unavailable}",
            loc=site,
        )
    if isinstance(error, LanguagePatchedError):
        # No compile ran, so it says nothing about the kernel or the target.
        return Refusal(HOST_COMPILE_UNAVAILABLE, f"{_at(site)}{error}", loc=site)
    if error is not None and bind_failed(error):
        # The call's own error: the untraced call raises it on any GPU, and
        # a traced launch does too; only an event built outside the
        # core gets here.
        return Refusal(
            COMPILE_FAILED,
            f"{_at(site)}the call does not bind to the kernel's parameters "
            f"({_summary(error)}), so it was not checked; Triton raises this "
            "error for the call whatever the GPU",
            loc=site,
        )
    why = "" if error is None else f" ({_summary(error)})"
    names = unknown_options(error)
    if names:
        # No kernel code failed: the call names options the target's backend
        # lacks. Only a target of a backend that has them can compile it.
        listed = ", ".join(repr(name) for name in names)
        return Refusal(
            COMPILE_FAILED,
            f"{_at(site)}it failed to compile{_for_target(failure)}{why}: the "
            f"call passes {listed}, neither a parameter of the kernel nor a "
            f"compile option{_for_target(failure)}, so it was not checked; "
            "Triton raises this for the call on every GPU whose backend lacks "
            "that option (a misspelled option: on every GPU); if it is another "
            "backend's option, name a target of that backend to check it "
            f"({retarget})",
            loc=site,
        )
    return Refusal(
        COMPILE_FAILED,
        f"{_at(site)}it failed to compile{_for_target(failure)}{why}, so it was "
        "not checked; a kernel can compile for one target and fail for another, "
        "so it may launch on a GPU of another kind: to check it, name a target "
        f"it compiles for ({retarget})",
        loc=site,
    )


def nowhere_note(failure: CompileFailure, config: Mapping[str, Any]) -> str:
    """The note for a config that compiles nowhere (see compiles_nowhere);
    ``config`` is the failed config's kwargs as the client shows them."""
    named = f"config {config}" if config else "the kernel"
    error = failure.error
    why = "" if error is None else f" ({_summary(error)})"
    return (
        f"{named} was not checked: {_at(_kernel_site(failure.jit_fn, error))}it "
        f"failed to compile{_for_target(failure)}{why}, an error of its own code "
        "whatever the target, so it never launches"
    )


def none_compiled(log: ArtifactLog) -> Refusal:
    """The refusal of a launch no config of which compiled, each only noted
    (see nowhere_note)."""
    targets = sorted(
        {format_ir_target(f.target) for f in log.failures if f.target is not None}
    )
    where = f" for {', '.join(targets)}" if targets else ""
    site = _kernel_site(log.failures[0].jit_fn) if log.failures else None
    return Refusal(
        COMPILE_FAILED,
        f"{_at(site)}no config of the launch compiled{where}, so nothing was "
        "checked: each failed with an error of its own code whatever the target "
        "(see the notes)",
        loc=site,
    )
