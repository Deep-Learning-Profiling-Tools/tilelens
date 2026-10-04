import os
import shutil
import runpy
import sys
import triton
import tilelens
from triton.experimental import gluon as _gluon
from tilelens.clients import Profiler, RaceDetector, Sanitizer
from tilelens.core.config import config as cfg


# Command names
SANITIZER_COMMAND = "tile-sanitizer"
PROFILER_COMMAND = "tile-profiler"
RACE_DETECTOR_COMMAND = "tile-race"

# tile-sanitizer's flag for the compiled sanitizer, given before the script
# name.
COMPILE_FLAG = "--compile"
COMPILE_NOTE = (
    f"[{COMPILE_FLAG}] kernel launches are compiled and checked, not run: "
    "their outputs are never written, so the script sees them unchanged, and "
    "a launch whose arguments it computes from them is checked with those "
    "values"
)

# Former Triton-Viz command names, still installed as aliases.
LEGACY_COMMANDS = {
    SANITIZER_COMMAND: "triton-sanitizer",
    PROFILER_COMMAND: "triton-profiler",
    RACE_DETECTOR_COMMAND: "triton-race-detector",
}

# store the original triton.jit
_original_jit = triton.jit
_original_autotune = triton.autotune
_original_gluon_jit = _gluon.jit


def sanitizer_wrapper(kernel, *, frontend: str = "triton"):
    abort_on_error = True
    tracer = tilelens.trace(
        client=Sanitizer(abort_on_error=abort_on_error),
        frontend=frontend,
    )
    return tracer(kernel)


def compiled_sanitizer_wrapper(kernel, *, frontend: str = "triton"):
    tracer = tilelens.trace(
        client=Sanitizer(compile=True, abort_on_error=True),
        frontend=frontend,
    )
    return tracer(kernel)


def profiler_wrapper(kernel, *, frontend: str = "triton"):
    tracer = tilelens.trace(client=Profiler(), frontend=frontend)
    return tracer(kernel)


def race_detector_wrapper(kernel, *, frontend: str = "triton"):
    tracer = tilelens.trace(client=RaceDetector(), frontend=frontend)
    return tracer(kernel)


def create_patched_jit(wrapper_func, original_jit, *, frontend: str):
    def _patched_jit(fn=None, **jit_kw):
        if fn is None:  # @triton.jit(**opts)

            def _decorator(f):
                k = original_jit(**jit_kw)(f)
                return wrapper_func(k, frontend=frontend)

            return _decorator
        else:  # @triton.jit
            k = original_jit(fn)
            return wrapper_func(k, frontend=frontend)

    return _patched_jit


def create_patched_autotune(wrapper_func):
    def _patched_autotune(fn=None, **autotune_kw):
        if fn is None:

            def _decorator(f):
                k = _original_autotune(**autotune_kw)(f)
                return wrapper_func(k)

            return _decorator
        else:
            k = _original_autotune(fn)
            return wrapper_func(k)

    return _patched_autotune


def _apply_wrapper(wrapper_func, command_name, usage_msg, compile_wrapper=None):
    """
    Generic function to apply a wrapper to triton.jit and run the user script.
    A command with a ``compile_wrapper`` uses it instead when its first
    argument is COMPILE_FLAG.
    """
    legacy_command = LEGACY_COMMANDS[command_name]
    if os.path.basename(sys.argv[0]) not in (command_name, legacy_command):
        raise RuntimeError(
            f"{command_name} must be used as a CLI tool, not called from Python. "
            f"Usage: {usage_msg}"
        )

    cfg.cli_active = True

    if compile_wrapper is not None and sys.argv[1:2] == [COMPILE_FLAG]:
        del sys.argv[1]
        wrapper_func = compile_wrapper
        print(COMPILE_NOTE, file=sys.stderr)

    # Patch Triton kernels with the Triton frontend.
    _patched_jit = create_patched_jit(
        wrapper_func,
        _original_jit,
        frontend="triton",
    )
    _patched_autotune = create_patched_autotune(wrapper_func)

    triton.jit = _patched_jit
    triton.language.jit = _patched_jit
    import triton.runtime.interpreter as _interp

    _interp.jit = _patched_jit
    triton.autotune = _patched_autotune

    # Patch Gluon kernels with the Gluon frontend.
    _gluon.jit = create_patched_jit(
        wrapper_func,
        _original_gluon_jit,
        frontend="gluon",
    )

    # run user script
    if len(sys.argv) < 2:
        print(usage_msg)
        sys.exit(1)

    script = sys.argv[1]

    if os.path.isfile(script):
        # It's a Python script file, run it directly
        sys.argv = sys.argv[1:]
        runpy.run_path(script, run_name="__main__")
    else:
        # It might be a command like 'pytest', run it as a subprocess
        cmd = sys.argv[1:]

        # Check if it's an executable command
        if shutil.which(cmd[0]):
            if cmd[0] == "pytest":
                # pytest is only a test extra; import it just for this branch
                # so the CLIs work in a plain `pip install tilelens`.
                import pytest

                sys.exit(pytest.main(cmd[1:]))
            elif cmd[0] == "python":
                sys.argv = cmd[1:]
                try:
                    runpy.run_path(cmd[1], run_name="__main__")
                except SystemExit as e:
                    sys.exit(e.code)
                sys.exit(0)
        print(f"Error: '{script}' is neither a valid file nor a command")
        sys.exit(1)


def apply_sanitizer():
    """
    Apply the sanitizer wrapper to triton.jit and run the user script; with
    ``--compile`` before the script, the compiled sanitizer's.
    """
    _apply_wrapper(
        sanitizer_wrapper,
        SANITIZER_COMMAND,
        f"Usage: {SANITIZER_COMMAND} [{COMPILE_FLAG}] <script.py> [args...]\n"
        f"  {COMPILE_FLAG}  check each launch against the compiled kernel "
        "(Sanitizer(compile=True)) instead of interpreting it; kernels are not "
        "run, so their outputs are never written. An 'ok' holds for the "
        "arguments each launch was called with. Kernels are compiled on the "
        "host (no GPU needed) for TILELENS_IR_TARGET, by default cuda:89 "
        "(e.g. cuda:90, hip:gfx942); a config that fails to compile for it is "
        "reported as not checked, and the script goes on.",
        compile_wrapper=compiled_sanitizer_wrapper,
    )


def apply_profiler():
    """
    Apply the profiler wrapper to triton.jit and run the user script.
    """
    _apply_wrapper(
        profiler_wrapper,
        PROFILER_COMMAND,
        f"Usage: {PROFILER_COMMAND} <script.py> [args...]",
    )


def apply_race_detector():
    """
    Apply the race detector wrapper to triton.jit and run the user script.
    """
    _apply_wrapper(
        race_detector_wrapper,
        RACE_DETECTOR_COMMAND,
        f"Usage: {RACE_DETECTOR_COMMAND} <script.py> [args...]",
    )
