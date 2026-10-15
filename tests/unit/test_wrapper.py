import subprocess
import sys

import pytest
from unittest.mock import MagicMock, patch

import tilelens
import triton
from tilelens.core.config import config as cfg
from tilelens.core.frontend.base import FRONTEND_REGISTRY, Frontend
from tilelens.core.trace import TraceInterface
from tilelens.wrapper import (
    create_patched_jit,
    create_patched_autotune,
    sanitizer_wrapper,
    profiler_wrapper,
    apply_sanitizer,
    apply_profiler,
)


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


def test_patched_jit_preserves_device_function_dependencies(
    _isolate_cli_active, monkeypatch
):
    monkeypatch.setenv("TRITON_INTERPRET", "0")
    cfg.cli_active = True
    patched_jit = create_patched_jit(profiler_wrapper, triton.jit, frontend="triton")

    @patched_jit(noinline=True)
    def add_one(value):
        return value + 1

    @patched_jit
    def add_two(value):
        return add_one(add_one(value))

    # Hashing follows both global and closure references before compilation.
    assert isinstance(add_one, triton.JITFunction)
    assert add_one.noinline is True
    assert isinstance(add_two, triton.JITFunction)
    assert add_two.cache_key


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


def test_trace_supports_registered_frontend(monkeypatch):
    class PythonFrontend(Frontend):
        def wrap_kernel(self, kernel, client):
            return lambda value: kernel(value) + 1

    frontend = PythonFrontend("python", None, {}, {}, {})
    monkeypatch.setitem(FRONTEND_REGISTRY, frontend.name, frontend)

    @tilelens.trace(frontend="python")
    def double(value):
        return value * 2

    assert double(3) == 7


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
