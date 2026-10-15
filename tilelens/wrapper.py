import os
import shutil
import runpy
import sys
from collections.abc import Callable
from typing import NoReturn
import triton
import tilelens
from triton.experimental import gluon as _gluon
from tilelens.clients import Profiler, RaceDetector, Sanitizer
from tilelens.clients.sanitizer.data import OutOfBoundsRecord
from tilelens.clients.sanitizer.sanitizer import SymbolicSanitizer
from tilelens.core.config import config as cfg


# Command names
SANITIZER_COMMAND = "tile-sanitizer"
PROFILER_COMMAND = "tile-profiler"
RACE_DETECTOR_COMMAND = "tile-race"

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

# Sanitizers created by tile-sanitizer; their findings set the exit status.
# One per traced kernel definition, kept for the life of the process.
_sanitizers: list[Sanitizer] = []


def sanitizer_wrapper(kernel, *, frontend: str = "triton"):
    sanitizer = Sanitizer()
    _sanitizers.append(sanitizer)
    tracer = tilelens.trace(client=sanitizer, frontend=frontend)
    return tracer(kernel)


def _oob_records() -> list[OutOfBoundsRecord]:
    return [
        record
        for sanitizer in _sanitizers
        if isinstance(sanitizer, SymbolicSanitizer)
        for record in sanitizer.records
        if isinstance(record, OutOfBoundsRecord)
    ]


def _oob_site(record: OutOfBoundsRecord) -> str:
    frames = record.user_code_tracebacks
    where = f"{frames[-1].filename}:{frames[-1].lineno}" if frames else "<unknown>"
    tensor = getattr(record, "tensor_name", None) or "<unknown tensor>"
    return f"{where}: {record.op_type.__name__} {tensor}"


def sanitizer_exit_status() -> int:
    """Summarize tile-sanitizer's findings on stderr; 1 if there are any."""
    records = _oob_records()
    if not records:
        return 0
    sites = dict.fromkeys(_oob_site(record) for record in records)
    sys.stdout.flush()  # keep the summary after the program's own output
    print(
        f"{SANITIZER_COMMAND}: {len(records)} out-of-bounds finding(s):",
        *(f"  {site}" for site in sites),
        sep="\n",
        file=sys.stderr,
    )
    return 1


def sanitizer_pytest_plugins() -> list[object]:
    """Fail each test whose call added findings, naming it as the old exit did."""
    import pytest

    class _FailTestsWithFindings:
        @pytest.hookimpl(wrapper=True)
        def pytest_runtest_call(self, item):
            before = len(_oob_records())
            result = yield
            if new := len(_oob_records()) - before:
                msg = f"{SANITIZER_COMMAND}: {new} out-of-bounds finding(s)"
                pytest.fail(msg, pytrace=False)
            return result

    return [_FailTestsWithFindings()]


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


def _run_script(path: str) -> str | int | None:
    """Run a user script as ``__main__`` and return its exit code."""
    try:
        runpy.run_path(path, run_name="__main__")
    except SystemExit as e:
        return e.code
    return 0


def _apply_wrapper(
    wrapper_func,
    command_name,
    usage_msg,
    findings_exit_status: Callable[[], int] = lambda: 0,
    pytest_plugins: Callable[[], list[object]] = list,
):
    """
    Generic function to apply a wrapper to triton.jit and run the user script.

    Exits with the program's own non-zero code if it has one, otherwise with
    ``findings_exit_status()``; an uncaught exception propagates as usual.
    ``pytest_plugins()`` are passed to ``pytest.main`` on the pytest path.
    """
    legacy_command = LEGACY_COMMANDS[command_name]
    if os.path.basename(sys.argv[0]) not in (command_name, legacy_command):
        raise RuntimeError(
            f"{command_name} must be used as a CLI tool, not called from Python. "
            f"Usage: {usage_msg}"
        )

    cfg.cli_active = True

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

    def _exit(code: str | int | None) -> NoReturn:
        # The program's own non-zero exit wins; otherwise findings decide.
        status = findings_exit_status()
        sys.exit(code if code not in (None, 0) else status)

    if os.path.isfile(script):
        # It's a Python script file, run it directly
        sys.argv = sys.argv[1:]
        _exit(_run_script(script))
    else:
        # It might be a command like 'pytest', run it as a subprocess
        cmd = sys.argv[1:]

        # Check if it's an executable command
        if shutil.which(cmd[0]):
            if cmd[0] == "pytest":
                # pytest is only a test extra; import it just for this branch
                # so the CLIs work in a plain `pip install tilelens`.
                import pytest

                _exit(pytest.main(cmd[1:], plugins=pytest_plugins()))
            elif cmd[0] == "python":
                sys.argv = cmd[1:]
                _exit(_run_script(cmd[1]))
        print(f"Error: '{script}' is neither a valid file nor a command")
        sys.exit(1)


def apply_sanitizer():
    """
    Apply the sanitizer wrapper to triton.jit and run the user script.
    """
    _apply_wrapper(
        sanitizer_wrapper,
        SANITIZER_COMMAND,
        f"Usage: {SANITIZER_COMMAND} <script.py> [args...]",
        sanitizer_exit_status,
        sanitizer_pytest_plugins,
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
