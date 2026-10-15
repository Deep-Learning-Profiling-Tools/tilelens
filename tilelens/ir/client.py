"""The IR client base: lifecycle only, no analysis defaults.

An ``IRClient`` takes no part in the interpreted run: its interpreter-path
methods are inert and it declares ``NEEDS_INTERPRETER = False``, so the core
hands it compiled kernels through ``before_launch`` / ``compile_failed``,
which fill its per-launch ``ArtifactLog``. ``finalize`` is a template:
``analyze_launch(log)`` returns the reports and the verdict, also for a
launch nothing could be captured for (see analyze_launch); an ``Exception``
from it goes to ``on_analysis_error``, which returns the verdict instead (an
interrupt or ``SystemExit`` propagates).

It returns the reports followed by the verdict, which ``ClientManager``
puts into ``Launch.records``, and keeps the verdict as ``last_verdict``
(None until a launch finalizes). A subclass declares ``NAME``,
``IR_STAGES`` and ``LAUNCH``, may set ``ir_target`` and implements the two
hooks; statuses, refusal meanings, caches and report printing are all its
own.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

from ..core.callbacks import ForLoopCallbacks, OpCallbacks
from ..core.client import Client, LaunchCall, LaunchEvent
from ..core.data import Op
from .capture import ArtifactLog
from .verdict import IRVerdict


class IRClient(Client):
    NEEDS_INTERPRETER: ClassVar[bool] = False

    def __init__(self) -> None:
        super().__init__()
        self.artifacts = ArtifactLog(self.IR_STAGES)
        self.last_verdict: IRVerdict | None = None

    # ── the client's analysis ────────────────────────────────────────

    @abstractmethod
    def analyze_launch(self, log: ArtifactLog) -> tuple[list, IRVerdict]:
        """Analyze one traced launch: the reports and the verdict.

        ``log.call`` is the launch's LaunchCall. When ``log.call.capture`` is
        False, nothing was compiled or recorded: the trace has no JITFunction
        (``log.call.jit_fn`` is None: TRITON_INTERPRET, an InterpretedFunction
        runner, Gluon, NKI). An empty log then says nothing about the
        kernel; what such a launch gets is the subclass's call.
        """

    @abstractmethod
    def on_analysis_error(self, exc: Exception) -> IRVerdict:
        """The verdict for a launch whose analysis raised ``exc``."""

    # ── launch lifecycle ─────────────────────────────────────────────
    # A subclass overriding one of these calls super().

    def begin_launch(self, call: LaunchCall) -> None:
        self.artifacts.reset(call)
        self.last_verdict = None

    def abort_launch(self, exc: BaseException) -> None:
        self.artifacts.reset()

    def before_launch(self, event: LaunchEvent) -> None:
        self.artifacts.record(event)

    def compile_failed(self, event: LaunchEvent) -> None:
        self.artifacts.record_failure(event)

    def finalize(self) -> list:
        try:
            reports, verdict = self._verdict()
        finally:
            # The log holds compile exceptions (and their frames); the next
            # launch starts from a fresh one anyway.
            self.artifacts.reset()
        self.last_verdict = verdict
        return [*reports, verdict]

    def _verdict(self) -> tuple[list, IRVerdict]:
        try:
            reports, verdict = self.analyze_launch(self.artifacts)
            return list(reports), verdict
        except Exception as exc:
            return [], self.on_analysis_error(exc)

    # ── inert interpreter path: the core calls none of these for an IR
    # client except the warmup vote, which declines (IR compiles go through
    # ir_capture) ──

    def pre_run_callback(self, fn: Callable) -> bool:
        return False

    def post_run_callback(self, fn: Callable) -> bool:
        return False

    def arg_callback(self, name: str, arg: Any, arg_cvt: Any) -> None:
        pass

    def grid_callback(self, grid: tuple[int, ...]) -> None:
        pass

    def grid_idx_callback(self, grid_idx: tuple[int, ...]) -> None:
        pass

    def register_op_callback(
        self, op_type: type[Op], *args: Any, **kwargs: Any
    ) -> OpCallbacks:
        return OpCallbacks()

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        return ForLoopCallbacks()

    def pre_warmup_callback(self, jit_fn: Callable, *args: Any, **kwargs: Any) -> bool:
        return False

    def post_warmup_callback(self, jit_fn: Callable, ret: Any) -> None:
        pass
