"""Tests for running the profiler on cuTile kernels."""
# ruff: noqa: E402

import importlib.util
import inspect
import os
from pathlib import Path
import shutil
import subprocess
import textwrap

import numpy as np
import pytest
import torch

ct = pytest.importorskip("cuda.tile")

import tilelens
from tilelens.clients import Profiler
from tilelens.clients.profiler.data import LoadStoreBytes
from tilelens.core.config import config as cfg
from tilelens.core.frontend.cutile import (
    cutile_skip_store,
    cutile_zero_dot,
    cutile_zero_load,
)
from tilelens.core.simulation.cutile import Array, Tile, cutile_builder
from tilelens.core.trace import launches
from tilelens.wrapper import PROFILER_COMMAND, SANITIZER_COMMAND

example_path = Path(__file__).resolve().parents[2] / "examples" / "cutile" / "matmul.py"
spec = importlib.util.spec_from_file_location("cutile_matmul_example", example_path)
assert spec is not None
assert spec.loader is not None
matmul_example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matmul_example)
matmul_kernel = matmul_example.matmul_kernel


@pytest.fixture(autouse=True)
def clear_traces():
    tilelens.clear()
    yield
    tilelens.clear()


@pytest.fixture
def _isolate_profiler_cfg():
    """Save and restore profiler-related config values around every test."""
    saved = (
        cfg.profiler_enable_block_sampling,
        cfg.profiler_enable_load_store_skipping,
        cfg.profiler_disable_buffer_load_check,
    )
    yield
    (
        cfg.profiler_enable_block_sampling,
        cfg.profiler_enable_load_store_skipping,
        cfg.profiler_disable_buffer_load_check,
    ) = saved


def copy_kernel(src, dst):
    block = ct.bid(0)
    tile = ct.load(src, index=(block,), shape=(4,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(dst, index=(block,), tile=tile)


def copy_pair(src, dst, index):
    for offset in range(2):
        tile = ct.load(
            src, index=(index + offset,), shape=(2,), padding_mode=ct.PaddingMode.ZERO
        )
        ct.store(dst, index=(index + offset,), tile=tile)


def looped_copy_kernel(src, dst):
    block = ct.bid(0)
    for step in range(3):
        copy_pair(src, dst, block * 6 + step * 2)
    # Like Triton, loops over other iterables are reported without a length;
    # comprehensions are not for statements.
    for _ in tuple(range(4)):
        pass
    _ = [i for i in range(4)]


def _bytes_records():
    records = [r for r in launches[-1].records if isinstance(r, LoadStoreBytes)]
    return {r.type: r for r in records}


def _line_of(fn, text):
    source, start = inspect.getsourcelines(fn)
    return start + next(i for i, line in enumerate(source) if text in line)


def test_profiler_counts_masked_load_store_bytes(_isolate_profiler_cfg):
    cfg.profiler_enable_block_sampling = False
    cfg.profiler_enable_load_store_skipping = False
    src = np.arange(1, 11, dtype=np.float32)
    dst = np.zeros(10, dtype=np.float32)
    profiler = Profiler()
    tilelens.trace(profiler, frontend="cutile")(copy_kernel)[(3,)](src, dst)

    np.testing.assert_array_equal(dst, src)
    records = _bytes_records()
    for kind in ("load", "store"):
        assert records[kind].total_bytes_attempted == 12 * 4
        assert records[kind].total_bytes_true == 10 * 4

    assert profiler.load_mask_total_count == 12
    assert profiler.load_mask_false_count == 2
    assert profiler.store_mask_false_count == 2
    load_stats = [s for s in profiler.mask_op_stats if s.op_type == "load"]
    assert load_stats and all("ct.load" in s.code_line for s in load_stats)


def test_profiler_counts_indexed_access_bytes(_isolate_profiler_cfg):
    cfg.profiler_enable_block_sampling = False
    cfg.profiler_enable_load_store_skipping = False

    @ct.kernel
    def indexed_kernel(src, dst):
        indices = ct.arange(4, dtype=ct.int32) - 1
        values = ct.gather(src, indices, padding_value=0)
        ct.scatter(dst, ct.arange(4, dtype=ct.int32), values)

    src = np.array([10, 20, 30, 40], np.int64)
    dst = np.zeros(4, np.int64)
    tilelens.trace("profiler", frontend="cutile")(indexed_kernel)[(1,)](src, dst)

    np.testing.assert_array_equal(dst, [0, 10, 20, 30])
    records = _bytes_records()
    assert records["load"].total_bytes_attempted == 4 * 8
    assert records["load"].total_bytes_true == 3 * 8
    assert records["store"].total_bytes_true == 4 * 8


def test_profiler_skips_memory_and_matmul(_isolate_profiler_cfg):
    cfg.profiler_enable_block_sampling = False
    cfg.profiler_enable_load_store_skipping = True
    lhs = np.ones((4, 8), dtype=np.float32)
    rhs = np.ones((8, 16), dtype=np.float32)
    result = np.full((4, 16), -1, dtype=np.float32)
    traced = tilelens.trace("profiler", frontend="cutile")(matmul_kernel)
    traced[(2, 2)](lhs, rhs, result)

    # Skipped stores leave the output untouched, but bytes are still counted.
    np.testing.assert_array_equal(result, -1)
    records = _bytes_records()
    assert records["load"].total_bytes_true == (4 * 8 + 8 * 16) * 2 * 4
    assert records["store"].total_bytes_true == 4 * 16 * 4


@pytest.mark.parametrize(
    "shape, index, load_shape, kwargs",
    [
        ((10,), (2,), (4,), {}),
        ((4, 6), (1, 1), (2, 4), {"order": "F"}),
        ((4, 8), (1, 0), (2, 8), {"traversal_steps": (1, 8)}),
    ],
)
def test_zero_load_matches_tiled_load(shape, index, load_shape, kwargs):
    array = Array(np.arange(np.prod(shape), dtype=np.float16).reshape(shape))
    expected = cutile_builder.load(
        array, index, load_shape, padding_mode=ct.PaddingMode.ZERO, **kwargs
    )
    zeros = cutile_zero_load(
        array, index, load_shape, padding_mode=ct.PaddingMode.ZERO, **kwargs
    )
    assert zeros.shape == expected.shape
    assert zeros.dtype == expected.dtype
    assert not zeros.data.any()


def test_zero_load_matches_gather():
    array = Array(np.arange(12, dtype=np.int32).reshape(3, 4))
    indices = (Tile(np.array([[0], [2]], np.int32)), Tile(np.arange(4, dtype=np.int32)))
    expected = cutile_builder.gather(array, indices)
    zeros = cutile_zero_load(array, indices, None, mask=None, check_bounds=True)
    assert zeros.shape == expected.shape == (2, 4)
    assert zeros.dtype == expected.dtype
    assert not zeros.data.any()
    assert cutile_skip_store(array, indices, zeros, _indexed=True) is None
    np.testing.assert_array_equal(array.data, np.arange(12).reshape(3, 4))


@pytest.mark.parametrize(
    "lhs, rhs",
    [
        (np.ones((2, 3), np.float32), np.ones((3, 4), np.float32)),
        (np.ones((2, 2, 3), np.float16), np.ones((2, 3, 5), np.float16)),
        (np.ones((2, 3), np.float16), np.ones((3, 4), np.float32)),
    ],
)
def test_zero_dot_matches_matmul(lhs, rhs):
    x, y = Tile(lhs), Tile(rhs)
    expected = cutile_builder.matmul(x, y)
    zeros = cutile_zero_dot(x, y)
    assert zeros.shape == expected.shape
    assert zeros.dtype == expected.dtype
    assert not zeros.data.any()


def test_zero_dot_matches_mma_accumulators():
    x = Tile(np.ones((2, 3), np.float16))
    y = Tile(np.ones((3, 4), np.float16))
    acc = Tile(np.full((2, 4), 7, np.float32))
    expected = cutile_builder.mma(x, y, acc)
    for zeros in (cutile_zero_dot(x, y, acc), cutile_zero_dot(x, y, acc=acc)):
        assert zeros.shape == expected.shape
        assert zeros.dtype == expected.dtype
        assert not zeros.data.any()

    # mma_scaled passes (x, x_scale, y, y_scale, acc); only acc decides the result.
    scale = Tile(np.ones((2, 1), np.float32))
    for zeros in (
        cutile_zero_dot(x, scale, y, scale, acc),
        cutile_zero_dot(x, scale, y, scale, acc=acc),
    ):
        assert zeros.shape == acc.shape
        assert zeros.dtype == ct.float32
        assert not zeros.data.any()


def test_profiler_reports_range_loops(_isolate_profiler_cfg):
    cfg.profiler_enable_block_sampling = False
    cfg.profiler_enable_load_store_skipping = False
    src = np.arange(1, 13, dtype=np.float32)
    dst = np.zeros(12, dtype=np.float32)
    profiler = Profiler()
    tilelens.trace(profiler, frontend="cutile")(looped_copy_kernel)[(1,)](src, dst)

    np.testing.assert_array_equal(dst, src)
    kernel_loop = _line_of(looped_copy_kernel, "range(3)")
    helper_loop = _line_of(copy_pair, "range(2)")
    tuple_loop = _line_of(looped_copy_kernel, "tuple(range(4))")
    assert set(profiler.loop_info) == {kernel_loop, helper_loop, tuple_loop}
    assert profiler.loop_info[kernel_loop].length == 3
    assert profiler.loop_info[helper_loop].length == 2
    assert profiler.loop_info[tuple_loop].length is None
    assert profiler.loop_info[kernel_loop].range_type == "python_range"
    assert profiler.loop_info[helper_loop].range_type == "python_range"
    assert profiler.loop_info[tuple_loop].range_type == "unknown"
    # Rewritten helpers are only swapped in during traced runs.
    assert "_tilelens_loop_iter_wrapper" not in inspect.signature(copy_pair).parameters


def test_profiler_reports_loops_in_closures_and_tile_functions(_isolate_profiler_cfg):
    cfg.profiler_enable_block_sampling = False
    cfg.profiler_enable_load_store_skipping = False
    steps = 2

    @ct.function
    def copy_steps(src, dst):
        for step in range(steps):
            ct.store(dst, (step,), ct.load(src, (step,), (2,)))

    # Reached only at call time, through cuTile's tile-function dispatch.
    helpers = {"copy": copy_steps}

    def kernel(src, dst):
        helpers["copy"](src, dst)

    src = np.arange(1, 5, dtype=np.float32)
    dst = np.zeros(4, dtype=np.float32)
    profiler = Profiler()
    tilelens.trace(profiler, frontend="cutile")(kernel)[(1,)](src, dst)

    np.testing.assert_array_equal(dst, src)
    helper_loop = _line_of(copy_steps, "range(steps)")
    assert set(profiler.loop_info) == {helper_loop}
    assert profiler.loop_info[helper_loop].length == 2


def test_profiler_runs_kernels_without_source(_isolate_profiler_cfg):
    cfg.profiler_enable_block_sampling = False
    cfg.profiler_enable_load_store_skipping = False
    namespace = {"ct": ct}
    exec(
        "def kernel(src, dst):\n"
        "    for i in range(2):\n"
        "        ct.store(dst, (i,), ct.load(src, (i,), (2,)))\n",
        namespace,
    )
    src = np.arange(1, 5, dtype=np.float32)
    dst = np.zeros(4, dtype=np.float32)
    profiler = Profiler()
    tilelens.trace(profiler, frontend="cutile")(namespace["kernel"])[(1,)](src, dst)

    # Without source there is nothing to rewrite, so loops go unreported.
    np.testing.assert_array_equal(dst, src)
    assert profiler.loop_info == {}


def nested_copy_kernel(src, dst):
    for row in range(2):
        for col in range(1, 4):
            tile = ct.load(src, index=(row, col - 1), shape=(2, 2))
            ct.store(dst, index=(row, col - 1), tile=tile)


def test_profiler_reports_nested_range_loops(_isolate_profiler_cfg):
    cfg.profiler_enable_block_sampling = False
    cfg.profiler_enable_load_store_skipping = False
    src = np.arange(24, dtype=np.float32).reshape(4, 6)
    dst = np.zeros((4, 6), dtype=np.float32)
    profiler = Profiler()
    tilelens.trace(profiler, frontend="cutile")(nested_copy_kernel)[(1,)](src, dst)

    np.testing.assert_array_equal(dst, src)
    outer_loop = _line_of(nested_copy_kernel, "range(2)")
    inner_loop = _line_of(nested_copy_kernel, "range(1, 4)")
    # The inner loop runs once per outer iteration but is one loop site.
    assert set(profiler.loop_info) == {outer_loop, inner_loop}
    assert profiler.loop_info[outer_loop].length == 2
    assert profiler.loop_info[inner_loop].length == 3
    # All 2 x 3 tile copies are counted.
    assert profiler.load_mask_total_count == 6 * 4


def test_profiler_block_sampling_skips_unsampled_blocks(_isolate_profiler_cfg):
    cfg.profiler_enable_block_sampling = True
    cfg.profiler_enable_load_store_skipping = False
    src = np.arange(1, 17, dtype=np.float32)
    dst = np.zeros(16, dtype=np.float32)
    profiler = Profiler(k=2)
    tilelens.trace(profiler, frontend="cutile")(copy_kernel)[(4,)](src, dst)

    written_blocks = {
        block for block in range(4) if np.any(dst[block * 4 : block * 4 + 4])
    }
    assert profiler.sampled_blocks is not None
    assert written_blocks == {idx[0] for idx in profiler.sampled_blocks}
    assert _bytes_records()["load"].total_bytes_true == 2 * 4 * 4


CLI_COPY_KERNEL = """
import cuda.tile as ct
import numpy as np


@ct.kernel
def copy_kernel(src, dst):
    for step in range(2):
        tile = ct.load(
            src, index=(step,), shape=(4,), padding_mode=ct.PaddingMode.ZERO
        )
        ct.store(dst, index=(step,), tile=tile)
"""


def _run_profiler_cli(tmp_path: Path, launch_code: str, command=PROFILER_COMMAND):
    exe = shutil.which(command)
    assert exe is not None, f"{command} command not found"
    script = tmp_path / "cutile_program.py"
    script.write_text(CLI_COPY_KERNEL + textwrap.dedent(launch_code), encoding="utf-8")
    env = os.environ.copy()
    env["PROFILER_ENABLE_LOAD_STORE_SKIPPING"] = "0"
    return subprocess.run(
        [exe, str(script)],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )


def test_tile_profiler_cli_runs_cutile_launches(tmp_path: Path):
    proc = _run_profiler_cli(
        tmp_path,
        """
        src = np.arange(1, 7, dtype=np.float32)
        dst = np.zeros(6, dtype=np.float32)
        ct.launch(None, (1,), copy_kernel, (src, dst))
        print("DST", dst.tolist())
        """,
    )

    assert proc.returncode == 0, proc.stderr
    assert "DST [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]" in proc.stdout
    assert "Total steps:    2" in proc.stdout
    assert "Masked percentage:       25.00%" in proc.stdout


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tile_profiler_cli_runs_device_tensor_launches(tmp_path: Path):
    proc = _run_profiler_cli(
        tmp_path,
        """
        import torch

        src = torch.arange(1, 7, dtype=torch.float32, device="cuda")
        dst = torch.zeros(6, dtype=torch.float32, device="cuda")
        ct.launch(torch.cuda.current_stream(), (1,), copy_kernel, (src, dst))
        print("DST", dst.tolist())
        """,
    )

    assert proc.returncode == 0, proc.stderr
    assert "DST [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]" in proc.stdout


def test_tile_profiler_cli_rejects_interpreted_launches(tmp_path: Path):
    proc = _run_profiler_cli(
        tmp_path,
        """
        from tilelens.core.frontend.cutile import interpreted_launches

        with interpreted_launches():
            pass
        """,
    )

    assert proc.returncode != 0
    assert "cannot be used together with CLI wrapper" in proc.stderr


def test_tile_profiler_cli_rejects_traced_kernels(tmp_path: Path):
    proc = _run_profiler_cli(
        tmp_path,
        """
        import tilelens

        traced = tilelens.trace("tracer", frontend="cutile")(copy_kernel)
        src = np.zeros(4, dtype=np.float32)
        ct.launch(None, (1,), traced, (src, src))
        """,
    )

    assert proc.returncode != 0
    assert "cannot be used together with CLI wrapper" in proc.stderr


def test_tile_profiler_cli_accumulates_across_launches(tmp_path: Path):
    proc = _run_profiler_cli(
        tmp_path,
        """
        src = np.arange(1, 7, dtype=np.float32)
        dst = np.zeros(6, dtype=np.float32)
        ct.launch(None, (1,), copy_kernel, (src, dst))
        ct.launch(None, (1,), copy_kernel, (src, dst))
        """,
    )

    assert proc.returncode == 0, proc.stderr
    # Like the Triton CLI, one profiler per kernel reports after every launch.
    assert proc.stdout.count("Profiler Issues Summary Ends") == 2
    assert "Total mask elements:     8\n" in proc.stdout
    assert "Total mask elements:     16\n" in proc.stdout


def test_other_clis_keep_nested_interpreted_launches(tmp_path: Path):
    proc = _run_profiler_cli(
        tmp_path,
        """
        from tilelens.core.frontend.cutile import interpreted_launches

        src = np.arange(1, 7, dtype=np.float32)
        dst = np.zeros(6, dtype=np.float32)
        with interpreted_launches(), interpreted_launches():
            ct.launch(None, (1,), copy_kernel, (src, dst))
        print("DST", dst.tolist())
        """,
        command=SANITIZER_COMMAND,
    )

    assert proc.returncode == 0, proc.stderr
    assert "DST [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]" in proc.stdout


def test_profiler_skipping_keeps_atomics(_isolate_profiler_cfg):
    cfg.profiler_enable_block_sampling = False
    cfg.profiler_enable_load_store_skipping = True

    @ct.kernel
    def atomic_kernel(counter, old):
        previous = ct.atomic_add(counter, ct.full((4,), 0, ct.int32), 1)
        ct.store(old, (0,), previous)

    counter = np.array([10], np.int32)
    old = np.full(4, -1, np.int32)
    tilelens.trace("profiler", frontend="cutile")(atomic_kernel)[(1,)](counter, old)

    # Like Triton, skipping drops plain stores but still applies atomics.
    np.testing.assert_array_equal(counter, [14])
    np.testing.assert_array_equal(old, -1)
