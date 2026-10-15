import os
import subprocess
import sys
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch

import tilelens
from tilelens.core.config import config as cfg
from tilelens.core.trace import TraceInterface
from tilelens.clients.sanitizer.compiled import CompiledSanitizer
from tilelens.wrapper import (
    COMPILE_NOTE,
    create_patched_jit,
    create_patched_autotune,
    sanitizer_wrapper,
    compiled_sanitizer_wrapper,
    profiler_wrapper,
    apply_sanitizer,
    apply_profiler,
)

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def _isolate_cli_active():
    """Save and restore cfg.cli_active around every test."""
    saved = cfg.cli_active
    yield
    cfg.cli_active = saved


# ======== Wrapper Function Tests ===========


def test_sanitizer_wrapper_applies_trace():
    mock_kernel = MagicMock()
    mock_kernel.__name__ = "test_kernel"

    with patch("tilelens.wrapper.tilelens.trace") as mock_trace:
        mock_decorator = MagicMock()
        mock_trace.return_value = mock_decorator
        mock_decorator.return_value = "wrapped_kernel"

        result = sanitizer_wrapper(mock_kernel)

        mock_trace.assert_called_once()
        mock_decorator.assert_called_once_with(mock_kernel)
        assert result == "wrapped_kernel"


def test_sanitizer_wrapper_accepts_frontend():
    mock_kernel = MagicMock()
    mock_kernel.__name__ = "test_kernel"

    with patch("tilelens.wrapper.tilelens.trace") as mock_trace:
        mock_decorator = MagicMock()
        mock_trace.return_value = mock_decorator
        mock_decorator.return_value = "wrapped_kernel"

        result = sanitizer_wrapper(mock_kernel, frontend="gluon")

        assert mock_trace.call_args.kwargs["frontend"] == "gluon"
        mock_decorator.assert_called_once_with(mock_kernel)
        assert result == "wrapped_kernel"


def test_compiled_sanitizer_wrapper_traces_with_the_compiled_sanitizer(monkeypatch):
    monkeypatch.setattr(cfg, "enable_sanitizer", True)
    mock_kernel = MagicMock()
    mock_kernel.__name__ = "test_kernel"

    with patch("tilelens.wrapper.tilelens.trace") as mock_trace:
        mock_decorator = MagicMock(return_value="wrapped_kernel")
        mock_trace.return_value = mock_decorator

        result = compiled_sanitizer_wrapper(mock_kernel, frontend="gluon")

    client = mock_trace.call_args.kwargs["client"]
    assert isinstance(client, CompiledSanitizer) and client.abort_on_error
    assert mock_trace.call_args.kwargs["frontend"] == "gluon"
    mock_decorator.assert_called_once_with(mock_kernel)
    assert result == "wrapped_kernel"


def test_profiler_wrapper_applies_trace():
    mock_kernel = MagicMock()
    mock_kernel.__name__ = "test_kernel"

    with patch("tilelens.wrapper.tilelens.trace") as mock_trace:
        mock_decorator = MagicMock()
        mock_trace.return_value = mock_decorator
        mock_decorator.return_value = "wrapped_kernel"

        result = profiler_wrapper(mock_kernel)

        mock_trace.assert_called_once()
        mock_decorator.assert_called_once_with(mock_kernel)
        assert result == "wrapped_kernel"


# ======== create_patched_jit Tests ===========


def test_create_patched_jit_direct_decorator():
    """Test @triton.jit without parentheses"""
    mock_wrapper = MagicMock(return_value="final_kernel")
    mock_original_jit = MagicMock(return_value="jit_kernel")

    patched_jit = create_patched_jit(
        mock_wrapper,
        mock_original_jit,
        frontend="triton",
    )

    mock_fn = MagicMock()
    result = patched_jit(mock_fn)

    mock_original_jit.assert_called_once_with(mock_fn)
    mock_wrapper.assert_called_once_with("jit_kernel", frontend="triton")
    assert result == "final_kernel"


def test_create_patched_jit_with_kwargs():
    """Test @triton.jit(**opts) with parentheses"""
    mock_wrapper = MagicMock(return_value="final_kernel")
    mock_jit_decorator = MagicMock(return_value="jit_kernel")
    mock_original_jit = MagicMock(return_value=mock_jit_decorator)

    patched_jit = create_patched_jit(
        mock_wrapper,
        mock_original_jit,
        frontend="triton",
    )

    decorator = patched_jit(do_not_specialize=["n"])
    assert callable(decorator)

    mock_fn = MagicMock()
    result = decorator(mock_fn)

    mock_original_jit.assert_called_once_with(do_not_specialize=["n"])
    mock_jit_decorator.assert_called_once_with(mock_fn)
    mock_wrapper.assert_called_once_with("jit_kernel", frontend="triton")
    assert result == "final_kernel"


def test_create_patched_jit_direct_decorator_with_custom_original_jit():
    mock_wrapper = MagicMock(return_value="final_kernel")
    mock_original_jit = MagicMock(return_value="custom_jit_kernel")

    patched_jit = create_patched_jit(
        mock_wrapper,
        mock_original_jit,
        frontend="gluon",
    )

    mock_fn = MagicMock()
    result = patched_jit(mock_fn)

    mock_original_jit.assert_called_once_with(mock_fn)
    mock_wrapper.assert_called_once_with("custom_jit_kernel", frontend="gluon")
    assert result == "final_kernel"


def test_create_patched_jit_with_kwargs_and_custom_original_jit():
    mock_wrapper = MagicMock(return_value="final_kernel")
    mock_jit_decorator = MagicMock(return_value="custom_jit_kernel")
    mock_original_jit = MagicMock(return_value=mock_jit_decorator)

    patched_jit = create_patched_jit(
        mock_wrapper,
        mock_original_jit,
        frontend="gluon",
    )

    decorator = patched_jit(debug=True)
    assert callable(decorator)

    mock_fn = MagicMock()
    result = decorator(mock_fn)

    mock_original_jit.assert_called_once_with(debug=True)
    mock_jit_decorator.assert_called_once_with(mock_fn)
    mock_wrapper.assert_called_once_with("custom_jit_kernel", frontend="gluon")
    assert result == "final_kernel"


# ======== create_patched_autotune Tests ===========


def test_create_patched_autotune_with_kwargs():
    """Test @triton.autotune(**opts) with parentheses"""
    mock_wrapper = MagicMock(return_value="final_kernel")
    mock_autotune_decorator = MagicMock(return_value="autotune_kernel")
    mock_original_autotune = MagicMock(return_value=mock_autotune_decorator)

    with patch("tilelens.wrapper._original_autotune", mock_original_autotune):
        patched_autotune = create_patched_autotune(mock_wrapper)

        decorator = patched_autotune(configs=[], key=["n"])
        assert callable(decorator)

        mock_fn = MagicMock()
        result = decorator(mock_fn)

        mock_original_autotune.assert_called_once_with(configs=[], key=["n"])
        mock_autotune_decorator.assert_called_once_with(mock_fn)
        mock_wrapper.assert_called_once_with("autotune_kernel")
        assert result == "final_kernel"


def test_create_patched_autotune_direct_decorator():
    """Test @triton.autotune without parentheses (if fn is passed directly)"""
    mock_wrapper = MagicMock(return_value="final_kernel")
    mock_original_autotune = MagicMock(return_value="autotune_kernel")

    with patch("tilelens.wrapper._original_autotune", mock_original_autotune):
        patched_autotune = create_patched_autotune(mock_wrapper)

        mock_fn = MagicMock()
        result = patched_autotune(mock_fn)

        mock_original_autotune.assert_called_once_with(mock_fn)
        mock_wrapper.assert_called_once_with("autotune_kernel")
        assert result == "final_kernel"


# ======== CLI Active Guard Tests ===========


def test_trace_decorator_raises_when_cli_active(_isolate_cli_active):
    """trace() should raise RuntimeError on an already-wrapped kernel when CLI is active."""
    cfg.cli_active = True
    mock_kernel = MagicMock(spec=TraceInterface)
    decorator = tilelens.trace("tracer")
    with pytest.raises(RuntimeError, match="CLI wrapper"):
        decorator(mock_kernel)


def test_trace_decorator_allows_cli_own_wrapping(_isolate_cli_active):
    """trace() called by the CLI on a raw kernel should NOT raise, even when cli_active."""
    from triton.runtime.interpreter import InterpretedFunction

    cfg.cli_active = True
    # Simulate a raw kernel (not already wrapped by TraceInterface)
    mock_kernel = MagicMock(spec=InterpretedFunction)
    mock_kernel.fn = lambda: None
    mock_kernel.arg_names = []
    decorator = tilelens.trace("sanitizer")
    # Should not raise — this is the CLI's own first-time wrapping
    result = decorator(mock_kernel)
    assert isinstance(result, TraceInterface)


def test_trace_decorator_works_when_cli_inactive(_isolate_cli_active):
    """trace() should work normally when CLI is not active."""
    from triton.runtime.interpreter import InterpretedFunction

    cfg.cli_active = False
    mock_kernel = MagicMock(spec=InterpretedFunction)
    mock_kernel.fn = lambda: None
    mock_kernel.arg_names = []
    decorator = tilelens.trace("tracer")
    result = decorator(mock_kernel)
    assert isinstance(result, TraceInterface)


def test_apply_wrapper_rejects_non_cli_invocation():
    """apply_sanitizer/apply_profiler should raise when called from Python, not CLI."""
    with pytest.raises(RuntimeError, match="must be used as a CLI tool"):
        apply_sanitizer()
    with pytest.raises(RuntimeError, match="must be used as a CLI tool"):
        apply_profiler()


def test_wrapper_imports_without_pytest():
    """The CLIs import tilelens.wrapper; pytest is only a test extra."""
    code = "import sys; sys.modules['pytest'] = None; import tilelens.wrapper"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


# ======== tile-sanitizer --compile ===========

_CLIENTS_SCRIPT = """\
import sys
import triton


@triton.jit
def kernel(x_ptr):
    pass


print(sorted(kernel.client_manager.clients), sys.argv[1:])
"""


def _run_cli(argv):
    """Run apply_sanitizer in a subprocess (it patches triton.jit for good)
    as the command ``argv[0]`` with ``argv[1:]``."""
    code = (
        f"import sys; sys.argv = {argv!r}; "
        "from tilelens.wrapper import apply_sanitizer; apply_sanitizer()"
    )
    env = {k: v for k, v in os.environ.items() if k != "TRITON_INTERPRET"}
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, env=env
    )


@pytest.mark.parametrize(
    "command, before, after, clients",
    [
        ("tile-sanitizer", ["--compile"], [], ["compiled_sanitizer"]),
        ("triton-sanitizer", ["--compile"], ["--compile"], ["compiled_sanitizer"]),
        # After the script name the flag is the script's own argument.
        ("tile-sanitizer", [], ["--compile"], ["sanitizer"]),
    ],
)
def test_the_compile_flag_before_the_script_selects_the_compiled_sanitizer(
    tmp_path, command, before, after, clients
):
    script = tmp_path / "script.py"
    script.write_text(_CLIENTS_SCRIPT)
    proc = _run_cli([command, *before, str(script), *after])
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == f"{clients} {after}"
    # Under the flag the user is told, once, that kernels do not run.
    assert proc.stderr.count(COMPILE_NOTE) == (1 if before else 0)


def test_the_compile_flag_without_a_script_prints_the_usage():
    proc = _run_cli(["tile-sanitizer", "--compile"])
    assert proc.returncode == 1
    assert "Usage: tile-sanitizer [--compile] <script.py> [args...]" in proc.stdout
    assert "kernels are not run" in proc.stdout
