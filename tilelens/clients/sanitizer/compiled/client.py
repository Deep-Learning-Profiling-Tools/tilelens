"""The compiled sanitizer client (``Sanitizer(compile=True)``).

An ``IRClient`` that reads each compiled config's TTIR and checks the
launch against it with ``oob.check_graph``, without running the kernel
(``LAUNCH = "skip"``): output tensors are left untouched, and addresses
are the launch's tensor addresses. The TTIR is compiled on the host for the
client's target (``target=``, else ``TILELENS_IR_TARGET``, else
``cuda:89``), so no GPU is needed and a result never depends on the machine:
CPU tensors are checked like device ones.

Every config the launch compiled is checked, autotune benchmark configs
included, each against the bindings the core delivered for it, and
gets its own ``ConfigVerdict``: configs that compile to one kernel but bind
other runtime arguments or grids are checked and reported apart.

A config that failed to compile never fails the launch: the target is
this client's choice, not the machine's. What the failure means (see
_compiles_nowhere):

- an error no target compiles past (a failing ``tl.static_assert``, a Python
  construct Triton's code generator never accepts), from a compile that
  never asked Triton's driver anything (nor did an earlier compile of the
  kernel for the target, whose answer the kernel's code may keep): the
  config never launches anywhere (Triton's autotuner drops it too), so it
  is only noted;
- any other compile error may be the target's (``num_ctas > 1`` below
  cuda:90, an fp8 type the target lacks, 16-bit descriptor atomic min/max
  without native TMA, ...): the config may launch on the user's GPU and was
  not checked, so it is unsupported, kind ``"compile-failed"``, with a
  refusal naming the target and how to name another;
- a call that does not bind the kernel's parameters (a missing, extra or
  misnamed argument) is no compile failure: the core raises the JIT
  binder's error from the launch, as the untraced call raises it on any GPU,
  so it never reaches this client from a traced launch; one
  delivered anyway (an event built outside the core) is unsupported, kind
  ``"compile-failed"``, naming no other target, which would not help;
- a host compile that could not run at all (Triton's compile API, or a
  compile that asked for a device: ``HostCompileUnavailable``; or one
  refused while an interpreted traced launch has the language patched:
  ``LanguagePatchedError``) says nothing about the kernel: unsupported,
  kind ``"host-compile-unavailable"``.

Each refusal and note names where the kernel failed (``file:line`` of the
kernel's code the error points at, else of its ``def``) and the innermost
error in one line; the whole error stays on the ArtifactLog's
CompileFailure. A launch no config of which compiled is unsupported, with
the first failed config's refusal (``"compile-failed"`` when every config
was only noted; its notes are printed with it).
The launch's status is the union: ``"violations"`` if any config has a
finding, else ``"unsupported"`` if any config was not (fully) checked, else
``"ok"``. Nothing is ``"ok"`` before a check says so.

What ``"ok"`` covers (``IRVerdict.scope == "launch"``): this launch, with
the scalar arguments, grid and tensors it was called with, as compiled for
the target (a kernel's TTIR can differ between targets, e.g. for tensor
descriptors below sm90). Since no kernel
runs, its outputs are never written: a later launch whose arguments the
host computes from them (a count, an offset, a size read back) is checked
with the values the untraced program would not have used, so a program's
launches can each be "ok" while the untraced program goes out of bounds.
In a trace shared with an eager client that runs the interpreter, the
interpreted run happens before this analysis, unchecked by it.

A check is remembered: a later launch of the same compiled kernel with the
same scalar arguments, grid, and tensor shapes, strides and element sizes
(only the addresses differ, as in a training loop) reuses it, with the
findings' addresses moved to the new tensors.

Each finding becomes a ``CompiledSanitizerRecord``; the launch's
``IRVerdict`` follows them in ``Launch.records``. With
``abort_on_error`` every finding is printed as this client finalizes the
launch, which then raises ``SystemExit(1)``: the core still finalizes the
trace's other clients and exits after them, but the launch keeps none of
this client's records and is not added to ``tilelens.launches``. On a host
thread other than the main one the exit ends only that thread (the
process's exit status is unchanged). Without ``abort_on_error`` findings
are printed only under ``TILELENS_VERBOSE=1``. The parts of a launch that
were not checked are printed alongside, each op once per client and
compiled kernel (a kernel launched in a loop would repeat them).
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from collections.abc import Hashable, Mapping
from dataclasses import replace
from typing import Any, ClassVar

from ....core.client import LanguagePatchedError, LaunchCall
from ....core.config import config as cfg
from ....core.data import Load, Store
from ....core.host_compile import (
    bind_failed,
    format_ir_target,
    host_compile_unavailable,
    parse_ir_target,
    target_queried,
    unknown_options,
)
from ....ir.capture import (
    ArtifactLog,
    CompiledSpecialization,
    CompileFailure,
    ParseCache,
)
from ....ir.client import IRClient
from ....ir.launch import LaunchBinding, TensorFacts, tensor_facts
from ....ir.verdict import ConfigVerdict, IRVerdict, Refusal, SourceLocation
from ....utils.traceback_utils import location_to_traceback_info
from ..data import CompiledSanitizerRecord
from ..sanitizer import Sanitizer
from .oob import (
    CheckResult,
    Finding,
    SanitizerKind,
    check_graph,
    describe_site,
    launch_key,
    readdressed,
)

# Checks remembered per client (see the module docstring), least recently
# used dropped first.
_REMEMBERED_CHECKS = 256


class CompiledSanitizer(IRClient):
    """Out-of-bounds, integer-width and division-by-zero checks over a
    kernel's compiled TTIR.

    ``records`` (the last launch's findings), ``last_verdict`` and
    ``last_status`` (its status, None before a launch is finalized) are the
    compatibility view of the last launch.
    """

    NAME = "compiled_sanitizer"
    IR_STAGES: ClassVar[frozenset[str]] = frozenset({"ttir"})
    LAUNCH = "skip"

    def __init__(
        self,
        abort_on_error: bool = True,
        timeout_ms: int = 10_000,
        target: Any = None,
    ) -> None:
        """``target``: what the kernels are compiled for, e.g. ``"cuda:89"``,
        ``"cuda:90"``, ``"hip:gfx942"`` or a triton ``GPUTarget`` (see
        tilelens.core.host_compile.parse_ir_target); None for the configured
        default (``TILELENS_IR_TARGET``, else ``"cuda:89"``). A spec that
        names no target raises ValueError here."""
        super().__init__()
        # Parsed now, so a bad spec fails at construction, not at a launch.
        self.ir_target = None if target is None else parse_ir_target(target)
        if (
            isinstance(timeout_ms, bool)
            or not isinstance(timeout_ms, int)
            or timeout_ms <= 0
        ):
            # Z3 reads a timeout of 0 or below as none at all.
            raise ValueError(f"timeout_ms must be a positive int, not {timeout_ms!r}")
        self.abort_on_error = abort_on_error
        # Bounds each Z3 query; one that times out is "solver-unknown".
        self.timeout_ms = timeout_ms
        # Kept across launches: a TTIR text is read once, and a refusal keeps
        # its kind on every later hit.
        self.parses = ParseCache()
        self.records: list[CompiledSanitizerRecord] = []
        # Remembered checks: (id(graph), timeout, launch_key) -> (graph,
        # result); holding the graph keeps its id unique.
        self._checks: OrderedDict[Hashable, tuple[Any, CheckResult]] = OrderedDict()
        # The not-checked parts already printed (see print_unchecked).
        self._printed: set[Hashable] = set()

    @property
    def last_status(self) -> str | None:
        verdict = self.last_verdict
        return None if verdict is None else verdict.status

    # ── launch lifecycle ─────────────────────────────────────────────

    def begin_launch(self, call: LaunchCall) -> None:
        super().begin_launch(call)
        self.records = []

    def finalize(self) -> list:
        out = super().finalize()
        verdict = self.last_verdict
        assert verdict is not None
        if self.abort_on_error or cfg.verbose:
            for record in self.records:
                print_compiled_record(record)
            print_unchecked(verdict, self._printed)
        if self.abort_on_error and self.records:
            sys.exit(1)
        return out

    # ── the analysis ─────────────────────────────────────────────────

    def analyze_launch(self, log: ArtifactLog) -> tuple[list, IRVerdict]:
        failed: list[ConfigVerdict] = []
        notes: list[str] = []
        for failure in log.failures:
            if _compiles_nowhere(failure.error):
                notes.append(_nowhere_note(failure))
            else:
                failed.append(
                    ConfigVerdict(
                        None,
                        _saveable(failure.config),
                        "unsupported",
                        _failure_refusal(failure),
                    )
                )
        specializations = log.specializations
        if not specializations:
            if failed:
                refusal = failed[0].refusal
            elif log.failures:
                refusal = _none_compiled(log)
            else:
                refusal = Refusal(
                    SanitizerKind.NO_COMPILED_KERNEL, _nothing_compiled(log)
                )
            return [], IRVerdict(
                self.NAME,
                "unsupported",
                refusal=refusal,
                per_config=tuple(failed),
                notes=tuple(notes),
            )
        records: list[CompiledSanitizerRecord] = []
        per_config: list[ConfigVerdict] = []
        for spec in specializations:
            for found, verdict in self._check_specialization(spec):
                records += found
                per_config.append(verdict)
        per_config += failed
        statuses = {verdict.status for verdict in per_config}
        status = next(s for s in ("violations", "unsupported", "ok") if s in statuses)
        # The first refusal: under "violations" too, where it says the
        # findings may not be all there is.
        refusal = next((c.refusal for c in per_config if c.refusal is not None), None)
        self.records = records
        return records, IRVerdict(
            self.NAME,
            status,
            # A proof or a finding holds for this launch's arguments only.
            scope=None if status == "unsupported" else "launch",
            refusal=refusal,
            per_config=tuple(per_config),
            notes=tuple(notes),
        )

    def _check_specialization(
        self, spec: CompiledSpecialization
    ) -> list[tuple[list[CompiledSanitizerRecord], ConfigVerdict]]:
        """One (records, ConfigVerdict) per config among ``spec``'s
        bindings, in the order first seen (see _configs_of): each config is
        checked against its own bindings only."""
        configs = _configs_of(spec)
        try:
            graph = self._graph_of(spec)
        except Exception as exc:
            # A bug in this kernel's analysis; the other kernels still count.
            graph = Refusal(SanitizerKind.INTERNAL_ERROR, _describe(exc))
        if isinstance(graph, Refusal):
            return [
                ([], ConfigVerdict(spec.specialization, config, "unsupported", graph))
                for config, _ in configs
            ]
        return [
            self._check_config(spec.specialization, graph, config, bindings)
            for config, bindings in configs
        ]

    def _graph_of(self, spec: CompiledSpecialization) -> Any:
        """The access graph read from ``spec``'s TTIR, or the Refusal
        saying why there is none."""
        text = spec.artifacts.stages.get("ttir")
        if not isinstance(text, str):
            why = spec.artifacts.error or "the compiled kernel holds no TTIR"
            return Refusal(SanitizerKind.NO_TTIR, why)
        outcome = self.parses.get(text)
        if outcome.refusal is not None:
            # The reader's kind and fields, the message prefixed with the
            # location like the sanitizer's own refusals.
            refused = Refusal.from_exception(outcome.refusal)
            where = describe_site(refused.line_no, refused.loc)
            return replace(refused, message=f"{where}: {refused.message}")
        if outcome.error is not None:
            return Refusal(
                SanitizerKind.INTERNAL_ERROR,
                f"the TTIR reader failed: {outcome.error}",
            )
        return outcome.graph

    def _check_config(
        self,
        specialization: Hashable,
        graph: Any,
        config: dict[str, Any],
        bindings: list[LaunchBinding],
    ) -> tuple[list[CompiledSanitizerRecord], ConfigVerdict]:
        if not bindings:
            # Nothing to check against: never "ok" (the core delivers every
            # compiled kernel with a binding; another producer might not).
            return [], ConfigVerdict(
                specialization,
                config,
                "unsupported",
                Refusal(
                    SanitizerKind.INTERNAL_ERROR,
                    "no binding was delivered for this kernel, so nothing was checked",
                ),
            )
        try:
            records: list[CompiledSanitizerRecord] = []
            refusal: Refusal | None = None
            for binding in bindings:
                result = self._check(graph, binding)
                records += [
                    _record(finding, graph.kernel_name, binding, config)
                    for finding in result.findings
                ]
                if refusal is None:
                    refusal = result.refusal
        except Exception as exc:
            # A bug in this config's analysis; the other configs still count.
            refusal = Refusal(SanitizerKind.INTERNAL_ERROR, _describe(exc))
            return [], ConfigVerdict(specialization, config, "unsupported", refusal)
        if records:
            status = "violations"
        else:
            status = "ok" if refusal is None else "unsupported"
        return records, ConfigVerdict(
            specialization, config, status, refusal, n_reports=len(records)
        )

    def _check(self, graph: Any, binding: LaunchBinding) -> CheckResult:
        """check_graph, remembered (see the module docstring)."""
        key = (id(graph), self.timeout_ms, launch_key(binding))
        hit = self._checks.get(key)
        if hit is not None and hit[0] is graph:
            self._checks.move_to_end(key)
            return readdressed(hit[1], graph, binding)
        result = check_graph(graph, binding, timeout_ms=self.timeout_ms)
        self._checks[key] = (graph, result)
        if len(self._checks) > _REMEMBERED_CHECKS:
            self._checks.popitem(last=False)
        return result

    def on_analysis_error(self, exc: Exception) -> IRVerdict:
        self.records = []
        return IRVerdict(
            self.NAME,
            "unsupported",
            refusal=Refusal(SanitizerKind.INTERNAL_ERROR, _describe(exc)),
        )


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _nothing_compiled(log: ArtifactLog) -> str:
    if log.call is not None and not log.call.capture:
        return (
            "no compiled kernel to check: the launch has no JITFunction "
            "(TRITON_INTERPRET=1, an InterpretedFunction runner, Gluon or NKI); "
            "the eager Sanitizer() checks such launches"
        )
    return "no config of the launch was compiled, so nothing was checked"


# How to check a kernel for another target, as a refusal or note says it.
_NAME_A_TARGET = "Sanitizer(compile=True, target=...), or TILELENS_IR_TARGET"


def _compiles_nowhere(error: BaseException | None) -> bool:
    """Whether a config's host compile error says the config compiles for
    no target, so it never launches anywhere (a note), rather than perhaps
    only not for this client's target (unsupported, compile-failed).

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


def _failure_refusal(failure: CompileFailure) -> Refusal:
    """A failed config's refusal: why its host compile failed (see
    _compiles_nowhere), where, and how to check it for another target where
    another target may compile it."""
    error = failure.error
    site = _kernel_site(failure.jit_fn, error)
    unavailable = None if error is None else host_compile_unavailable(error)
    if unavailable is not None:
        return Refusal(
            SanitizerKind.HOST_COMPILE_UNAVAILABLE,
            f"{_at(site)}{unavailable}",
            loc=site,
        )
    if isinstance(error, LanguagePatchedError):
        # No compile ran, so it says nothing about the kernel or the target.
        return Refusal(
            SanitizerKind.HOST_COMPILE_UNAVAILABLE, f"{_at(site)}{error}", loc=site
        )
    if error is not None and bind_failed(error):
        # The call's own error: the untraced call raises it on any GPU, and
        # a traced launch does too; only an event built outside the
        # core gets here.
        return Refusal(
            SanitizerKind.COMPILE_FAILED,
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
            SanitizerKind.COMPILE_FAILED,
            f"{_at(site)}it failed to compile{_for_target(failure)}{why}: the "
            f"call passes {listed}, neither a parameter of the kernel nor a "
            f"compile option{_for_target(failure)}, so it was not checked; "
            "Triton raises this for the call on every GPU whose backend lacks "
            "that option (a misspelled option: on every GPU); if it is another "
            "backend's option, name a target of that backend to check it "
            f"({_NAME_A_TARGET})",
            loc=site,
        )
    return Refusal(
        SanitizerKind.COMPILE_FAILED,
        f"{_at(site)}it failed to compile{_for_target(failure)}{why}, so it was "
        "not checked; a kernel can compile for one target and fail for another, "
        "so it may launch on a GPU of another kind: to check it, name a target "
        f"it compiles for ({_NAME_A_TARGET})",
        loc=site,
    )


def _nowhere_note(failure: CompileFailure) -> str:
    config = f"config {_saveable(failure.config)}" if failure.config else "the kernel"
    error = failure.error
    why = "" if error is None else f" ({_summary(error)})"
    return (
        f"{config} was not checked: {_at(_kernel_site(failure.jit_fn, error))}it "
        f"failed to compile{_for_target(failure)}{why}, an error of its own code "
        "whatever the target, so it never launches"
    )


def _none_compiled(log: ArtifactLog) -> Refusal:
    targets = sorted(
        {format_ir_target(f.target) for f in log.failures if f.target is not None}
    )
    where = f" for {', '.join(targets)}" if targets else ""
    site = _kernel_site(log.failures[0].jit_fn) if log.failures else None
    return Refusal(
        SanitizerKind.COMPILE_FAILED,
        f"{_at(site)}no config of the launch compiled{where}, so nothing was "
        "checked: each failed with an error of its own code whatever the target "
        "(see the notes)",
        loc=site,
    )


_PLAIN = (str, int, float, bool, type(None))


def _tensor_of(value: Any) -> TensorFacts | None:
    """``value``'s tensor facts, None if it is no (readable) tensor."""
    if not hasattr(value, "data_ptr"):
        return None
    try:
        return tensor_facts(value)
    except Exception:
        return None


def _saveable(config: Mapping[str, Any]) -> dict[str, Any]:
    """The config kwargs, each value a saved trace cannot hold put as text:
    a tensor (e.g. a heuristic's view) described by its facts, never its
    data, anything else (e.g. a heuristic's tl.dtype) as its repr."""

    def describe(value: Any) -> Any:
        if isinstance(value, _PLAIN):
            return value
        facts = _tensor_of(value)
        if facts is None:
            return repr(value)
        return (
            f"<tensor data_ptr={facts.data_ptr:#x} shape={facts.shape} "
            f"strides={facts.strides} dtype={facts.dtype}>"
        )

    return {name: describe(value) for name, value in config.items()}


def _config_key(config: Mapping[str, Any]) -> Hashable:
    """What tells two bindings' config kwargs apart: a plain value by type
    and value, a tuple item by item, a tensor by its data_ptr, shape,
    strides and dtype (never its data), any other hashable value by type
    and equality, anything else by identity (the bindings hold the values
    while the keys are compared)."""

    def key(value: Any) -> Hashable:
        if isinstance(value, float):
            return (float, value.hex())  # -0.0 apart from 0.0, NaN equal
        if isinstance(value, _PLAIN):
            return (type(value), value)
        if isinstance(value, tuple):
            return (type(value), tuple(key(item) for item in value))
        facts = _tensor_of(value)
        if facts is not None:
            return ("tensor", facts.data_ptr, facts.shape, facts.strides, facts.dtype)
        try:
            hash(value)
        except Exception:
            return ("id", id(value))
        return (type(value), value)

    return tuple(sorted((name, key(value)) for name, value in config.items()))


def _configs_of(
    spec: CompiledSpecialization,
) -> list[tuple[dict[str, Any], list[LaunchBinding]]]:
    """``spec``'s bindings grouped by their config kwargs (see _config_key),
    each group with its saveable config, in the order first seen, so the
    first group's config is ``spec.config``. Configs that compile to one
    kernel (e.g. differing only in a runtime int kwarg) share a
    specialization but not their bindings."""
    groups: dict[Hashable, tuple[dict[str, Any], list[LaunchBinding]]] = {}
    for binding in spec.bindings:
        key = _config_key(binding.config)
        group = groups.get(key)
        if group is None:
            groups[key] = (_saveable(binding.config), [binding])
        else:
            group[1].append(binding)
    return list(groups.values()) or [(_saveable(spec.config), [])]


def _record(
    finding: Finding, kernel_name: str, binding: LaunchBinding, config: dict[str, Any]
) -> CompiledSanitizerRecord:
    loc = finding.loc
    tracebacks = (
        []
        if loc is None
        else [location_to_traceback_info((loc.file, loc.line, kernel_name))]
    )
    detail = finding.detail
    if loc is None and finding.line_no is not None:
        detail = f"{detail} (TTIR line {finding.line_no})"
    return CompiledSanitizerRecord(
        kind=finding.kind,
        # An atomic reads and writes; it is reported as the write.
        op_type=Load if finding.access_kind == "load" else Store,
        tensor_name=finding.base_param,
        tensor_facts=binding.tensors.get(finding.base_param),
        witness=dict(finding.witness),
        config=config,
        user_code_tracebacks=tracebacks,
        violation_offset=finding.violation_offset,
        violation_address=finding.violation_address,
        detail=detail,
    )


# ─────────────────────────── reporting ───────────────────────────

_TITLES = {
    "out-of-bounds": "Out-Of-Bounds Access Detected",
    "integer-overflow": "Integer Width Overflow Detected",
    "division-by-zero": "Division By Zero Detected",
}


def print_compiled_record(record: CompiledSanitizerRecord) -> None:
    """Print one compiled sanitizer finding, in the eager report's layout."""
    rule = "=" * 60
    print(rule)
    print(f"{_TITLES[record.kind]:^60}".rstrip())
    print(f"{'(compiled sanitizer)':^60}".rstrip())
    print(rule)
    print(f"Operation: {record.op_type.__name__}")
    print(f"Tensor Arg: {record.tensor_name}")
    facts = record.tensor_facts
    if facts is not None:
        print(
            f"Tensor Info: dtype={facts.dtype}, shape={facts.shape}, "
            f"strides={facts.strides}, contiguous={facts.contiguous}"
        )
        print(f"Tensor base memory address: {facts.data_ptr:#x}")
    if record.config:
        print(f"Config: {record.config}")
    for tb in record.user_code_tracebacks:
        print(f"File: {tb.filename}, Line: {tb.lineno}, in {tb.func_name}")
        print(f"  Code: {tb.line_of_code.strip()}")
    print("-" * 60)
    if record.violation_address is not None:
        print(
            f"Invalid access detected at address: {record.violation_address:#x} "
            f"(element offset {record.violation_offset})"
        )
    witness = ", ".join(f"{name}={value}" for name, value in record.witness.items())
    print(f"Witness: {witness}")
    if record.detail:
        print(f"Detail: {record.detail}")
    print(rule)


def print_unchecked(verdict: IRVerdict, printed: set[Hashable]) -> None:
    """Print one line per part of a launch the compiled sanitizer did not
    check (each refused config, else the launch's own refusal) that is not
    in ``printed`` yet, and add it there: once per specialization, kind and
    op, whatever launch-specific numbers the message holds; a config that
    compiled to nothing (a failed compile) once per config and message.
    The launch's own refusal is followed by its notes, each once: the
    refusal of a launch no config of which compiled says why in them."""
    refused = [
        (c.specialization, c.config, c.refusal)
        for c in verdict.per_config
        if c.refusal is not None
    ]
    notes: tuple[str, ...] = ()
    if not refused and verdict.refusal is not None:
        refused = [(None, {}, verdict.refusal)]
        notes = verdict.notes
    for specialization, config, refusal in refused:
        ident: Hashable = specialization
        if specialization is None:
            ident = (
                tuple(sorted((name, repr(value)) for name, value in config.items())),
                refusal.message,
            )
        key = (ident, refusal.kind, refusal.line_no, refusal.loc)
        if key in printed:
            continue
        printed.add(key)
        where = f" (config {config})" if config else ""
        print(
            f"[CompiledSanitizer] not checked{where}: "
            f"{refusal.kind}: {refusal.message}"
        )
    for note in notes:
        if ("note", note) in printed:
            continue
        printed.add(("note", note))
        print(f"[CompiledSanitizer] note: {note}")


# A sanitizer mode like the eager one: isinstance(..., Sanitizer) holds, so
# trace()'s ENABLE_SANITIZER=0 escape hatch covers a CompiledSanitizer() too.
Sanitizer.register(CompiledSanitizer)
