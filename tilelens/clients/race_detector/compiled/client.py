"""The compiled race detector client (``RaceDetector(compile=True)``).

An ``IRClient`` that reads each compiled config's TTIR (with the reader's
multipath and while-region facts) and decides, with ``analysis.analyze_graph``,
whether two program instances of the launch can race on global memory. The
kernel is not run (``LAUNCH = "skip"``), so output tensors are left untouched;
it is compiled on the host for the client's target (``target=``, else
``TILELENS_IR_TARGET``, else ``cuda:89``), so no GPU is needed.

Every config the launch compiled is decided, autotune benchmark configs
included, each against the bindings the core delivered for it. Configs whose
TTIR (minus its source locations), scalar arguments, tensor facts and
contents, and grid are the same (e.g. ones that differ only in ``num_warps``)
share one solve. A config's verdict is a ``RaceConfigVerdict``: "ok" with the
proof's rung and qualifiers, "races", "unsupported" (a refusal, a Z3 timeout
or unknown, or only possible races under over-approximation) or "vacuous".
The launch's ``RaceVerdict`` is the union: "races" if a config races, else
"unsupported", else "vacuous", else "ok". No launch-level time budget: each
solver has its own timeout (``T0_TIMEOUT_MS``, ``T1_TIMEOUT_MS``).

A config that failed to compile is classified by tilelens.ir.failures, as in
the compiled sanitizer: an error no target compiles past is only a note, any
other one makes the config unsupported (``"compile-failed"``), and a host
compile that could not run is ``"host-compile-unavailable"``.

A loaded integer value is read from the launch's pre-launch contents of its
tensor (a snapshot, taken in ``before_launch``): only for an integer or bool,
contiguous tensor of at most ``SNAPSHOT_MAX_ELEMENTS`` elements, and while the
launch's snapshots stay under ``SNAPSHOT_CAP_BYTES``. A verdict that read one
holds for those contents ("reads tensor contents").

What a verdict covers: this launch as compiled for the target. Since no
kernel runs, a later launch whose arguments the host computes from an earlier
one's outputs is analyzed with the values the untraced program would not
have used; likewise an autotune ``Config.pre_hook`` does not run, so a
verdict that reads tensor contents read them as the launch began, not as a
config's pre_hook would have left them.

Each race found becomes a ``CompiledRaceRecord`` (plain data: no Z3 terms, no
tensors), followed by the launch's ``RaceVerdict`` in ``Launch.records``;
``tilelens.save()`` keeps both. Every race is printed as the launch
finalizes, as the eager race detector prints them; under ``verbose`` or the
tile-race CLI, so is each part of a launch that was not decided (once per
specialization, kind and site) and a race-free launch's claim.

The analysis runs in Z3's main context, which is not thread-safe: solves are
serialized process-wide (``hb_common.MAIN_CONTEXT_LOCK``), so traced
launches on several host threads take turns.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, replace
from typing import Any, ClassVar

from ....core.client import LaunchCall, LaunchEvent
from ....core.config import config as cfg
from ....core.host_compile import parse_ir_target
from ....ir.capture import (
    ArtifactLog,
    CompiledSpecialization,
    ParseCache,
    configs_of,
    content_key,
    saveable_config,
)
from ....ir.client import IRClient
from ....ir.failures import (
    compiles_nowhere,
    failure_refusal,
    none_compiled,
    nothing_compiled,
    nowhere_note,
)
from ....ir.launch import LaunchBinding, TensorFacts
from ....ir.verdict import Refusal
from ..hb_common import MAIN_CONTEXT_LOCK
from ..data import (
    CompiledRaceRecord,
    RaceAccess,
    RaceConfigVerdict,
    RaceReport,
    RaceType,
    RaceVerdict,
)
from ..race_detector import RaceDetector, print_race_record
from .analysis import T0_TIMEOUT_MS, T1_TIMEOUT_MS, Outcome, analyze_graph
from .leaves import LaunchTensor, RaceKind, RaceRefusal

# How to analyze a kernel for another target, as a refusal says it.
_NAME_A_TARGET = "RaceDetector(compile=True, target=...), or TILELENS_IR_TARGET"

# What a proof at each rung holds for, weakest last.
_SCOPES = {
    "T0": "any-params-any-grid",
    "T1": "this-params-any-grid",
    "T1-launch": "this-params-this-grid",
}
_STATUS_ORDER = ("races", "unsupported", "vacuous", "ok")


@dataclass(frozen=True)
class _Solved:
    """One solve's result, shared by the configs whose key it has."""

    status: str
    rung: str | None = None
    qualifiers: tuple[str, ...] = ()
    refusal: Refusal | None = None
    races: tuple[CompiledRaceRecord, ...] = ()
    grid_fragile: tuple[CompiledRaceRecord, ...] = ()
    # Possible races under over-approximation, withheld beside the races.
    withheld: int = 0


class CompiledRaceDetector(IRClient):
    """Global-memory race detection over a kernel's compiled TTIR.

    ``last_reports`` (the last launch's races), ``last_verdict`` and
    ``last_status`` (its status, None before a launch is finalized) are the
    compatibility view of the last launch.
    """

    NAME = "compiled_race_detector"
    LOG_TAG: ClassVar[str] = "CompiledRaceDetector"
    IR_STAGES: ClassVar[frozenset[str]] = frozenset({"ttir"})
    LAUNCH = "skip"

    # Per-solver Z3 timeouts of the any-parameters rung and of the others.
    T0_TIMEOUT_MS: ClassVar[int] = T0_TIMEOUT_MS
    T1_TIMEOUT_MS: ClassVar[int] = T1_TIMEOUT_MS
    # The largest tensor whose contents are read, and the launch's total.
    SNAPSHOT_MAX_ELEMENTS: ClassVar[int] = 16384
    SNAPSHOT_CAP_BYTES: ClassVar[int] = 256 * 1024 * 1024

    def __init__(self, abort_on_error: bool = False, *, target: Any = None) -> None:
        """``target``: what the kernels are compiled for (see
        tilelens.core.host_compile.parse_ir_target); None for the configured
        default. ``abort_on_error`` is the RaceDetector factory's, unused."""
        super().__init__()
        self.ir_target = None if target is None else parse_ir_target(target)
        self.abort_on_error = abort_on_error
        self.parses = ParseCache()
        self.last_reports: list[CompiledRaceRecord] = []
        # id(binding) -> (binding, its tensors with their snapshots); holding
        # the binding keeps its id unique.
        self._contents: dict[int, tuple[LaunchBinding, dict[str, LaunchTensor]]] = {}
        self._kernel = "kernel"
        # What the launch's snapshots may still take (SNAPSHOT_CAP_BYTES).
        self._budget = self.SNAPSHOT_CAP_BYTES
        # The undecided parts already printed (see print_undecided).
        self._printed: set[Hashable] = set()

    @property
    def last_status(self) -> str | None:
        verdict = self.last_verdict
        return None if verdict is None else verdict.status

    # ── launch lifecycle ─────────────────────────────────────────────

    def begin_launch(self, call: LaunchCall) -> None:
        super().begin_launch(call)
        self.last_reports = []
        self._contents = {}
        self._budget = self.SNAPSHOT_CAP_BYTES
        self._kernel = getattr(call.jit_fn, "__name__", None) or "kernel"

    def abort_launch(self, exc: BaseException) -> None:
        super().abort_launch(exc)
        self._contents = {}

    def before_launch(self, event: LaunchEvent) -> None:
        super().before_launch(event)
        # The snapshots are taken for the binding just recorded.
        binding = self.artifacts.last_binding
        assert binding is not None
        self._contents[id(binding)] = (binding, self._snapshots(event, binding))

    def finalize(self) -> list:
        try:
            out = super().finalize()
        finally:
            self._contents = {}
        verdict = self.last_verdict
        assert isinstance(verdict, RaceVerdict)
        for record in self.last_reports:
            print_race_record(record)
        # The undecided parts and a proof print where someone asked to
        # see them: verbose, or tile-race, which reports every launch.
        if cfg.verbose or cfg.cli_active:
            print_undecided(verdict, self._printed)
            if verdict.status == "ok":
                print(f"[{self.LOG_TAG}] {self._kernel}: race-free{_claim(verdict)}")
        return out

    # ── snapshots ────────────────────────────────────────────────────

    def _snapshots(
        self, event: LaunchEvent, binding: LaunchBinding
    ) -> dict[str, LaunchTensor]:
        """Each tensor argument's facts and, where it can be read, its
        pre-launch contents (no kernel of the launch has run yet)."""
        out: dict[str, LaunchTensor] = {}
        for name, facts in binding.tensors.items():
            value = _described(event.bound_args.get(name))
            values: tuple[int, ...] | None = None
            nbytes = facts.numel * facts.elem_size
            try:
                why = _unreadable(value, facts, self.SNAPSHOT_MAX_ELEMENTS)
                if not why and nbytes > self._budget:
                    why = (
                        f"over the launch's {self.SNAPSHOT_CAP_BYTES}-byte snapshot cap"
                    )
                if not why:
                    values = tuple(
                        int(v) for v in value.detach().reshape(-1).cpu().tolist()
                    )
                    self._budget -= nbytes
            except Exception as exc:
                values, why = None, f"capture failed: {type(exc).__name__}"
            out[name] = LaunchTensor(facts, values, why)
        return out

    # ── the analysis ─────────────────────────────────────────────────

    def analyze_launch(self, log: ArtifactLog) -> tuple[list, RaceVerdict]:
        failed: list[RaceConfigVerdict] = []
        notes: list[str] = []
        for failure in log.failures:
            if compiles_nowhere(failure.error):
                notes.append(nowhere_note(failure, saveable_config(failure.config)))
            else:
                failed.append(
                    RaceConfigVerdict(
                        None,
                        saveable_config(failure.config),
                        "unsupported",
                        failure_refusal(failure, _NAME_A_TARGET),
                    )
                )
        specializations = log.specializations
        if not specializations:
            if failed:
                refusal = failed[0].refusal
            elif log.failures:
                refusal = none_compiled(log)
            else:
                refusal = Refusal(
                    RaceKind.NO_COMPILED_KERNEL,
                    nothing_compiled(log, "RaceDetector()"),
                )
            return [], RaceVerdict(
                self.NAME,
                "unsupported",
                refusal=refusal,
                per_config=tuple(failed),
                notes=tuple(notes),
            )
        solved: dict[Hashable, _Solved] = {}
        records: list[CompiledRaceRecord] = []
        per_config: list[RaceConfigVerdict] = []
        for spec in specializations:
            for found, verdict, withheld in self._analyze_specialization(spec, solved):
                records += found
                per_config.append(verdict)
                if withheld:
                    notes.append(
                        f"config {verdict.config}: {withheld} more possible "
                        "race(s) under over-approximation were withheld"
                    )
        per_config += failed
        status = _union(c.status for c in per_config)
        refusal = next((c.refusal for c in per_config if c.refusal is not None), None)
        rung = _weakest(c.rung for c in per_config)
        scope = _SCOPES[rung] if status == "ok" and rung is not None else None
        self.last_reports = records
        return records, RaceVerdict(
            self.NAME,
            status,
            scope=scope,
            refusal=refusal,
            per_config=tuple(per_config),
            notes=tuple(notes),
            qualifiers=_merged(c.qualifiers for c in per_config),
        )

    def _analyze_specialization(
        self, spec: CompiledSpecialization, solved: dict[Hashable, _Solved]
    ) -> list[tuple[list[CompiledRaceRecord], RaceConfigVerdict, int]]:
        configs = configs_of(spec)
        try:
            graph = self._graph_of(spec)
        except Exception as exc:
            graph = Refusal(RaceKind.INTERNAL_ERROR, _describe(exc))
        if isinstance(graph, Refusal):
            return [
                (
                    [],
                    RaceConfigVerdict(
                        spec.specialization, config, "unsupported", graph
                    ),
                    0,
                )
                for config, _ in configs
            ]
        text = _without_locs(spec.artifacts.stages["ttir"])
        return [
            self._analyze_config(
                spec.specialization, graph, text, config, bindings, solved
            )
            for config, bindings in configs
        ]

    def _graph_of(self, spec: CompiledSpecialization) -> Any:
        """The access graph read from ``spec``'s TTIR, or the Refusal saying
        why there is none."""
        text = spec.artifacts.stages.get("ttir")
        if not isinstance(text, str):
            why = spec.artifacts.error or "the compiled kernel holds no TTIR"
            return Refusal(RaceKind.NO_TTIR, why)
        outcome = self.parses.get(text, multipath=True, while_regions=True)
        if outcome.refusal is not None:
            refused = Refusal.from_exception(outcome.refusal)
            where = _site(refused.line_no, refused.loc)
            return replace(refused, message=f"{where}: {refused.message}")
        if outcome.error is not None:
            return Refusal(
                RaceKind.INTERNAL_ERROR, f"the TTIR reader failed: {outcome.error}"
            )
        return outcome.graph

    def _analyze_config(
        self,
        specialization: Hashable,
        graph: Any,
        text: str,
        config: dict[str, Any],
        bindings: list[LaunchBinding],
        solved: dict[Hashable, _Solved],
    ) -> tuple[list[CompiledRaceRecord], RaceConfigVerdict, int]:
        if not bindings:
            refusal = Refusal(
                RaceKind.INTERNAL_ERROR,
                "no binding was delivered for this kernel, so nothing was analyzed",
            )
            return (
                [],
                RaceConfigVerdict(specialization, config, "unsupported", refusal),
                0,
            )
        results: list[_Solved] = []
        for binding in bindings:
            contents = self._contents_of(binding)
            key = (text, _binding_key(binding, contents))
            hit = solved.get(key)
            if hit is None:
                hit = solved[key] = self._solve(graph, binding, contents)
            results.append(hit)
        status = _union(r.status for r in results)
        races = [replace(r, config=config) for s in results for r in s.races]
        verdict = RaceConfigVerdict(
            specialization,
            config,
            status,
            next((r.refusal for r in results if r.refusal is not None), None),
            n_reports=len(races),
            rung=_weakest(r.rung for r in results) if status == "ok" else None,
            qualifiers=_merged(r.qualifiers for r in results),
            grid_fragile=tuple(
                replace(r, config=config) for s in results for r in s.grid_fragile
            ),
        )
        return races, verdict, sum(r.withheld for r in results)

    def _contents_of(self, binding: LaunchBinding) -> dict[str, LaunchTensor]:
        entry = self._contents.get(id(binding))
        if entry is not None and entry[0] is binding:
            return entry[1]
        # A binding no before_launch of this client recorded: facts only.
        return {
            name: LaunchTensor(facts, None, "no snapshot was taken")
            for name, facts in binding.tensors.items()
        }

    def _solve(
        self, graph: Any, binding: LaunchBinding, contents: dict[str, LaunchTensor]
    ) -> _Solved:
        """One analysis, in Z3's main context: serialized with every other
        race-detector solve of the process, and plain data by the time the
        lock is released (no Z3 term outlives it, not even in a
        traceback)."""
        with MAIN_CONTEXT_LOCK:
            try:
                out = analyze_graph(
                    graph,
                    binding.params,
                    contents,
                    binding.grid,
                    t0_timeout_ms=self.T0_TIMEOUT_MS,
                    t1_timeout_ms=self.T1_TIMEOUT_MS,
                )
                try:
                    return _settle(out, binding)
                finally:
                    del out
            except Exception as exc:
                # A bug in this config's analysis; the other configs still
                # count.
                exc.__traceback__ = None
                return _Solved(
                    "unsupported",
                    refusal=Refusal(RaceKind.INTERNAL_ERROR, _describe(exc)),
                )

    def on_analysis_error(self, exc: Exception) -> RaceVerdict:
        self.last_reports = []
        return RaceVerdict(
            self.NAME,
            "unsupported",
            refusal=Refusal(RaceKind.INTERNAL_ERROR, _describe(exc)),
        )


# ─────────────────────────── helpers ───────────────────────────


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _site(line_no: int | None, loc: Any) -> str:
    if loc is not None:
        return f"{loc.file}:{loc.line}"
    return f"TTIR line {line_no}" if line_no is not None else "the kernel"


def _described(value: Any) -> Any:
    # A descriptor-style argument is bound as its .base tensor (see
    # tilelens.ir.launch.bind_launch).
    base = getattr(value, "base", None)
    if base is not None and hasattr(base, "data_ptr"):
        return base
    return value


def _unreadable(value: Any, facts: TensorFacts, max_elements: int) -> str:
    """Why ``value``'s contents are not read ("" when they are)."""
    dtype = getattr(value, "dtype", None)
    if dtype is None or not hasattr(value, "detach"):
        return "not a torch tensor"
    if getattr(dtype, "is_floating_point", True) or getattr(dtype, "is_complex", False):
        return f"float dtype {facts.dtype}"
    if any(getattr(value, flag, lambda: False)() for flag in ("is_conj", "is_neg")):
        return "lazy value transform"
    if not facts.contiguous:
        return "non-contiguous"
    if facts.numel == 0:
        return "empty"
    if facts.numel > max_elements:
        # "too large" is what the address refusal keys its kind on.
        return f"too large ({facts.numel} elements, bound {max_elements})"
    return ""


def _strip_locs(line: str) -> str:
    out: list[str] = []
    i = 0
    while True:
        j = line.find("loc(", i)
        if j < 0:
            out.append(line[i:])
            return "".join(out)
        if j > 0 and (line[j - 1].isalnum() or line[j - 1] in "_#.@%$"):
            out.append(line[i : j + 4])
            i = j + 4
            continue
        out.append(line[i:j].rstrip())
        depth, k, quoted = 0, j + 3, False
        while k < len(line):
            c = line[k]
            if quoted:
                if c == "\\":
                    k += 1
                elif c == '"':
                    quoted = False
            elif c == '"':
                quoted = True
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        i = k + 1


def _without_locs(text: str) -> str:
    """The content key of ``text`` without its source locations (configs of
    one kernel print the same ones)."""
    lines = (_strip_locs(line) for line in text.splitlines())
    return content_key("\n".join(ln for ln in lines if not ln.startswith("#loc")))


def _binding_key(
    binding: LaunchBinding, contents: Mapping[str, LaunchTensor]
) -> Hashable:
    """What the analysis reads of a binding."""
    return (
        tuple(sorted(binding.params.items())),
        tuple(sorted((n, t.facts, t.values, t.why) for n, t in contents.items())),
        binding.grid,
        binding.error,
    )


def _union(statuses: Any) -> str:
    seen = set(statuses)
    return next(s for s in _STATUS_ORDER if s in seen)


def _weakest(rungs: Any) -> str | None:
    order = list(_SCOPES)
    found = [r for r in rungs if r is not None]
    return max(found, key=order.index) if found else None


def _merged(groups: Any) -> tuple[str, ...]:
    out: list[str] = []
    for group in groups:
        out += [q for q in group if q not in out]
    return tuple(out)


def _qualifiers(out: Outcome) -> tuple[str, ...]:
    found: list[str] = []
    if out.overflow:
        widths = "/".join(f"i{bits}" for bits in out.overflow)
        found.append(f"assumes no {widths} overflow")
    if out.assumes_termination:
        found.append("assumes termination")
    if out.content:
        found.append("reads tensor contents")
    return tuple(found)


def _settle(out: Outcome, binding: LaunchBinding) -> _Solved:
    # A qualifier says what a verdict assumed: an undecided one has none.
    qualifiers = _qualifiers(out) if out.status in ("proved", "races") else ()
    if out.status == "proved":
        return _Solved(
            "ok",
            rung=out.rung,
            qualifiers=qualifiers,
            grid_fragile=tuple(_record(r, binding) for r in out.grid_fragile),
        )
    if out.status == "races":
        races = tuple(_record(r, binding) for r in out.reports)
        if races:
            return _Solved(
                "races", qualifiers=qualifiers, races=races, withheld=len(out.widened)
            )
        first = _record(out.widened[0], binding)
        return _Solved(
            "unsupported",
            refusal=Refusal(
                RaceKind.POTENTIAL_RACE,
                f"{len(out.widened)} possible race(s) under over-approximation "
                "(a dropped mask or a loaded value with no snapshot), not a "
                f"witness; the first: {_pair(first)}",
            ),
        )
    refusal = out.refusal or RaceRefusal(RaceKind.OTHER, "no verdict")
    return _Solved(
        out.status,
        refusal=Refusal.from_exception(refusal),
        grid_fragile=tuple(_record(r, binding) for r in out.grid_fragile),
    )


def _pair(record: CompiledRaceRecord) -> str:
    def side(access: RaceAccess) -> str:
        where = access.source_location
        line = f" (line {where[1]})" if where is not None else ""
        return f"{access.op} of {access.tensor_name}{line}"

    return f"{record.race_type} between {side(record.first)} and {side(record.second)}"


def _record(report: RaceReport, binding: LaunchBinding) -> CompiledRaceRecord:
    race_type = report.race_type
    writes = (
        race_type in (RaceType.WAW, RaceType.RAW),
        race_type in (RaceType.WAW, RaceType.WAR),
    )
    assert report.witness_addrs is not None
    accesses = []
    for event, event_writes, addr in zip(
        (report.first, report.second), writes, report.witness_addrs
    ):
        record = event.record
        facts = binding.tensors.get(record.tensor_name)
        element = (
            None
            if facts is None
            else (addr - facts.data_ptr) // max(1, record.elem_size)
        )
        accesses.append(
            RaceAccess(
                op=record.op_type.name,
                writes=event_writes,
                tensor_name=record.tensor_name,
                source_location=record.source_location,
                element=element,
            )
        )
    assert report.witness_grid_a is not None and report.witness_grid_b is not None
    return CompiledRaceRecord(
        race_type=race_type,
        first=accesses[0],
        second=accesses[1],
        programs=(report.witness_grid_a, report.witness_grid_b),
        reason=report.reason,
    )


def _claim(verdict: RaceVerdict) -> str:
    parts = [verdict.scope or "", *verdict.qualifiers]
    return f" ({'; '.join(p for p in parts if p)})"


# ─────────────────────────── reporting ───────────────────────────


def print_undecided(verdict: RaceVerdict, printed: set[Hashable]) -> None:
    """Print one line per part of a launch the compiled race detector did
    not decide (each refused config, else the launch's own refusal) that is
    not in ``printed`` yet, and add it there: once per specialization, kind
    and site; a config that compiled to nothing once per config and
    message. The launch's notes follow, each once."""
    refused = [
        (c.specialization, c.config, c.refusal)
        for c in verdict.per_config
        if c.refusal is not None and c.status != "races"
    ]
    if not refused and verdict.refusal is not None and verdict.status != "races":
        refused = [(None, {}, verdict.refusal)]
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
            f"[{CompiledRaceDetector.LOG_TAG}] not analyzed{where}: "
            f"{refusal.kind}: {refusal.message}"
        )
    for note in verdict.notes:
        if ("note", note) in printed:
            continue
        printed.add(("note", note))
        print(f"[{CompiledRaceDetector.LOG_TAG}] note: {note}")


# A race detector mode like the eager one: isinstance(..., RaceDetector)
# holds, so the factory's ENABLE_RACE_DETECTOR=0 rule reads it alike.
RaceDetector.register(CompiledRaceDetector)
