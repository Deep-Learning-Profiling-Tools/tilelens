"""tilelens.ir.failures: what a config's failed host compile means. The
compiled sanitizer's verdicts built on it are pinned in
tests/unit/sanitizer_compiled/test_client.py."""

from __future__ import annotations

import ast
from types import SimpleNamespace

from tilelens.core.host_compile import HostCompileUnavailable
from tilelens.ir.capture import CompileFailure
from tilelens.ir.failures import (
    compiles_nowhere,
    failure_refusal,
    none_compiled,
    nothing_compiled,
    nowhere_note,
)

HINT = "Client(target=...)"


def _cuda89():
    from triton.backends.compiler import GPUTarget

    return GPUTarget("cuda", 89, 32)


def test_an_error_no_target_compiles_past_is_a_note():
    from triton.compiler.errors import CompilationError, CompileTimeAssertionFailure

    error = CompilationError(None, ast.Pass(), None)
    error.__cause__ = CompileTimeAssertionFailure(None, ast.Pass(), "BLOCK <= 1024")
    failure = CompileFailure(error, {"BLOCK": 2048}, _cuda89())
    assert compiles_nowhere(error)
    note = nowhere_note(failure, {"BLOCK": 2048})
    assert note.startswith("config {'BLOCK': 2048} was not checked: it failed")
    assert "for cuda:89 (CompileTimeAssertionFailure: BLOCK <= 1024)" in note
    refusal = none_compiled(SimpleNamespace(failures=[failure]))
    assert refusal.kind == "compile-failed"
    assert "compiled for cuda:89" in refusal.message


def test_an_error_that_may_be_the_targets_names_another_target():
    failure = CompileFailure(RuntimeError("no MMA"), {}, _cuda89())
    assert not compiles_nowhere(failure.error)
    refusal = failure_refusal(failure, HINT)
    assert refusal.kind == "compile-failed"
    assert "for cuda:89 (RuntimeError: no MMA)" in refusal.message
    assert refusal.message.endswith(f"name a target it compiles for ({HINT})")


def test_a_host_compile_that_could_not_run_is_unavailable():
    error = HostCompileUnavailable("the installed Triton is 3.9.0")
    failure = CompileFailure(error, {}, _cuda89())
    assert not compiles_nowhere(error)
    refusal = failure_refusal(failure, HINT)
    assert refusal.kind == "host-compile-unavailable"
    assert refusal.message == "the installed Triton is 3.9.0"


def test_a_launch_without_a_jit_function_names_the_eager_client():
    log = SimpleNamespace(call=SimpleNamespace(capture=False))
    assert nothing_compiled(log, "Client()").endswith(
        "the eager Client() checks such launches"
    )
