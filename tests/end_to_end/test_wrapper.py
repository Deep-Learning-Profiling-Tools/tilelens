import json
import os
import tempfile
import subprocess
from pathlib import Path
from string import Template
import shutil
import pytest
from tilelens.wrapper import (
    LEGACY_COMMANDS,
    SANITIZER_COMMAND,
    PROFILER_COMMAND,
    RACE_DETECTOR_COMMAND,
)


TEMPLATE_DIR = Path(__file__).parent / "templates"


def load_template(name: str, **kwargs) -> str:
    """Load a template file and substitute variables."""
    template_path = TEMPLATE_DIR / name
    template = Template(template_path.read_text())
    return template.substitute(**kwargs)


def test_tile_sanitizer_injects_trace_outermost(tmp_path: Path, monkeypatch):
    """
    Black-box verification:
    - Pure @triton.jit kernels have exactly one @tilelens.trace at the outermost layer;
    - @triton.autotune + @triton.jit kernels: trace is only added at the autotune outer
      layer, inner jit should not be traced again.
    Implementation approach:
      Use sitecustomize to monkey-patch tilelens.trace at subprocess startup,
      record each trace decorator application target (function name/type),
      write to JSON on exit.
    """
    # 1) Write sitecustomize: takes effect at interpreter startup
    trace_log = tmp_path / "trace_log.json"
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        load_template(
            "sitecustomize_spy_trace.py.template",
            log_path=json.dumps(str(trace_log)),
        ),
        encoding="utf-8",
    )

    # 2) Write a minimal test script: two kernels (k1: pure jit, k2: autotune + jit)
    my_program = tmp_path / "my_program.py"
    my_program.write_text(
        load_template("kernel_jit_autotune.py.template"),
        encoding="utf-8",
    )

    # 3) Assemble subprocess environment: ensure our sitecustomize is found first
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")

    # 4) Find tile-sanitizer executable - fail if not found
    exe = shutil.which(SANITIZER_COMMAND)
    assert exe is not None, (
        f"{SANITIZER_COMMAND} command not found. "
        "Please ensure the package is properly installed."
    )
    cmd = [exe, str(my_program)]

    # 5) Run subprocess
    proc = subprocess.run(
        cmd,
        cwd=str(tmp_path),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Optional debug output (easier to locate on failure)
    if proc.returncode != 0:
        print("STDOUT:\n", proc.stdout)
        print("STDERR:\n", proc.stderr)
    assert proc.returncode == 0, f"{SANITIZER_COMMAND} run failed"

    # 6) Read trace records
    assert (
        trace_log.exists()
    ), "trace log not generated (sitecustomize may not have taken effect)"
    records = json.loads(trace_log.read_text(encoding="utf-8"))

    # Build counter by function name
    from collections import Counter

    counts = Counter(rec["name"] for rec in records)

    # Assertion: the 2 kernel names defined in our script should each appear exactly once
    for expected in ["k1", "k2"]:
        assert counts[expected] == 1, (
            f"{expected} should be traced 1 time, actual count is {counts[expected]}; "
            "may have missing or duplicate injection"
        )


def test_tile_profiler_injects_trace_outermost(tmp_path: Path, monkeypatch):
    """
    Black-box verification for tile-profiler:
    - Pure @triton.jit kernels have exactly one @tilelens.trace at the outermost layer;
    - @triton.autotune + @triton.jit kernels: trace is only added at the autotune outer
      layer, inner jit should not be traced again.
    Implementation approach:
      Use sitecustomize to monkey-patch tilelens.trace at subprocess startup,
      record each trace decorator application target (function name/type),
      write to JSON on exit.
    """
    # 1) Write sitecustomize: takes effect at interpreter startup
    trace_log = tmp_path / "trace_log.json"
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        load_template(
            "sitecustomize_spy_trace.py.template",
            log_path=json.dumps(str(trace_log)),
        ),
        encoding="utf-8",
    )

    # 2) Write a minimal test script: two kernels (k1: pure jit, k2: autotune + jit)
    my_program = tmp_path / "my_program.py"
    my_program.write_text(
        load_template("kernel_jit_autotune.py.template"),
        encoding="utf-8",
    )

    # 3) Assemble subprocess environment: ensure our sitecustomize is found first
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")

    # 4) Find tile-profiler executable - fail if not found
    exe = shutil.which(PROFILER_COMMAND)
    assert exe is not None, (
        f"{PROFILER_COMMAND} command not found. "
        "Please ensure the package is properly installed."
    )
    cmd = [exe, str(my_program)]

    # 5) Run subprocess
    proc = subprocess.run(
        cmd,
        cwd=str(tmp_path),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Optional debug output (easier to locate on failure)
    if proc.returncode != 0:
        print("STDOUT:\n", proc.stdout)
        print("STDERR:\n", proc.stderr)
    assert proc.returncode == 0, f"{PROFILER_COMMAND} run failed"

    # 6) Read trace records
    assert (
        trace_log.exists()
    ), "trace log not generated (sitecustomize may not have taken effect)"
    records = json.loads(trace_log.read_text(encoding="utf-8"))

    # Build counter by function name
    from collections import Counter

    counts = Counter(rec["name"] for rec in records)

    # Assertion: the 2 kernel names defined in our script should each appear exactly once
    for expected in ["k1", "k2"]:
        assert counts[expected] == 1, (
            f"{expected} should be traced 1 time, actual count is {counts[expected]}; "
            "may have missing or duplicate injection"
        )


def test_tile_race_injects_trace_outermost(tmp_path: Path, monkeypatch):
    """
    Black-box verification for tile-race:
    - Pure @triton.jit kernels have exactly one @tilelens.trace at the outermost layer;
    - @triton.autotune + @triton.jit kernels: trace is only added at the autotune outer
      layer, inner jit should not be traced again.
    Implementation approach:
      Use sitecustomize to monkey-patch tilelens.trace at subprocess startup,
      record each trace decorator application target (function name/type),
      write to JSON on exit.
    """
    # 1) Write sitecustomize: takes effect at interpreter startup
    trace_log = tmp_path / "trace_log.json"
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        load_template(
            "sitecustomize_spy_trace.py.template",
            log_path=json.dumps(str(trace_log)),
        ),
        encoding="utf-8",
    )

    # 2) Write a minimal test script: two kernels (k1: pure jit, k2: autotune + jit)
    my_program = tmp_path / "my_program.py"
    my_program.write_text(
        load_template("kernel_jit_autotune.py.template"),
        encoding="utf-8",
    )

    # 3) Assemble subprocess environment: ensure our sitecustomize is found first
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")

    # 4) Find tile-race executable - fail if not found
    exe = shutil.which(RACE_DETECTOR_COMMAND)
    assert exe is not None, (
        f"{RACE_DETECTOR_COMMAND} command not found. "
        "Please ensure the package is properly installed."
    )
    cmd = [exe, str(my_program)]

    # 5) Run subprocess
    proc = subprocess.run(
        cmd,
        cwd=str(tmp_path),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Optional debug output (easier to locate on failure)
    if proc.returncode != 0:
        print("STDOUT:\n", proc.stdout)
        print("STDERR:\n", proc.stderr)
    assert proc.returncode == 0, f"{RACE_DETECTOR_COMMAND} run failed"

    # 6) Read trace records
    assert (
        trace_log.exists()
    ), "trace log not generated (sitecustomize may not have taken effect)"
    records = json.loads(trace_log.read_text(encoding="utf-8"))

    # Build counter by function name
    from collections import Counter

    counts = Counter(rec["name"] for rec in records)

    # Assertion: the 2 kernel names defined in our script should each appear exactly once
    for expected in ["k1", "k2"]:
        assert counts[expected] == 1, (
            f"{expected} should be traced 1 time, actual count is {counts[expected]}; "
            "may have missing or duplicate injection"
        )


def test_cli_invocation():
    """
    Simulate running:
        $ tile-sanitizer dummy_program.py
    and assert that tilelens.trace is invoked exactly once.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 1) Create a temporary Triton script
        (tmp_path / "dummy_program.py").write_text(
            load_template("kernel_dummy.py.template")
        )

        # 2) Patch tilelens.trace to count invocations
        (tmp_path / "sitecustomize.py").write_text(
            load_template("sitecustomize_fake_trace.py.template")
        )

        # 3) Start Subprocess to simulate CLI invocation
        env = os.environ.copy()
        env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")
        env["TRITON_INTERPRET"] = "1"

        # Run the dummy program using tile-sanitizer
        cmd = ["tile-sanitizer", str(tmp_path / "dummy_program.py")]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
        )

        # 4) Assertion: trace must be called exactly once
        assert proc.returncode == 0, f"CLI exited with {proc.returncode}\n{proc.stderr}"
        trace_count = proc.stdout.count("TRACE_CALLED")
        assert (
            trace_count == 1
        ), "tilelens.trace should be invoked exactly once via CLI path"


def test_cli_rejects_trace_decorator(tmp_path: Path):
    """
    When a user script contains @tilelens.trace() and is run via tile-sanitizer,
    the process should exit with a non-zero code and report a RuntimeError.
    """
    # 1) Write a script that has both @tilelens.trace and @triton.jit
    my_program = tmp_path / "my_program.py"
    my_program.write_text(
        load_template("kernel_with_trace_decorator.py.template"),
        encoding="utf-8",
    )

    # 2) Find tile-sanitizer executable
    exe = shutil.which(SANITIZER_COMMAND)
    assert exe is not None, (
        f"{SANITIZER_COMMAND} command not found. "
        "Please ensure the package is properly installed."
    )

    # 3) Run subprocess
    env = os.environ.copy()
    proc = subprocess.run(
        [exe, str(my_program)],
        cwd=str(tmp_path),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # 4) Assertion: should fail with RuntimeError about CLI wrapper conflict
    assert (
        proc.returncode != 0
    ), "Expected non-zero exit code when @tilelens.trace() is used with CLI wrapper"
    assert (
        "CLI wrapper" in proc.stderr
    ), f"Expected error message about CLI wrapper conflict in stderr, got:\n{proc.stderr}"


EXIT_STATUS_PROGRAM = Template(
    """
import threading, torch, triton, triton.language as tl

@triton.jit
def store_kernel(x_ptr, N: tl.constexpr):
    tl.store(x_ptr + tl.arange(0, N), tl.zeros((N,), tl.float32))

def test_store():
    store_kernel[(1,)](torch.zeros(8), N=$n)

if __name__ == "__main__":
    $main
"""
)
KEPT_RUNNING = "test_store(); print('kept running')"
IN_THREAD = "t = threading.Thread(target=test_store); t.start(); t.join()"
SAN, LEGACY = SANITIZER_COMMAND, LEGACY_COMMANDS[SANITIZER_COMMAND]
PYTEST = ["pytest", "-q", "-p", "no:cacheprovider"]


@pytest.mark.parametrize(
    "command, args, n, main, status, stdout",
    [
        (SAN, [], 16, KEPT_RUNNING, 1, "kept running"),
        (SAN, [], 8, KEPT_RUNNING, 0, "kept running"),
        (SAN, [], 16, "test_store(); raise SystemExit(3)", 3, ""),  # own code wins
        (SAN, [], 16, IN_THREAD, 1, ""),  # main: 0, the thread's exit was lost
        (LEGACY, ["python"], 16, KEPT_RUNNING, 1, "kept running"),
        (SAN, PYTEST, 16, "pass", 1, "FAILED"),  # the finding test fails by name
    ],
)
def test_tile_sanitizer_exit_status(
    tmp_path: Path, command, args, n, main, status, stdout
):
    script = tmp_path / "test_prog.py"
    script.write_text(EXIT_STATUS_PROGRAM.substitute(n=n, main=main))
    exe = shutil.which(command)
    assert exe is not None
    env = {**os.environ, "TRITON_INTERPRET": "1"}
    cmd = [exe, *args, str(script)]
    proc = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert proc.returncode == status, proc.stdout + proc.stderr
    assert stdout in proc.stdout
    assert ("ILLEGAL MEMORY ACCESS" in proc.stdout) == (n > 8)
    assert ("1 out-of-bounds finding(s)" in proc.stderr) == (n > 8)
