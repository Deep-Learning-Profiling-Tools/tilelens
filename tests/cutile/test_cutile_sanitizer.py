"""Tests for sanitizing cuTile accesses guarded by a user-written mask."""
# ruff: noqa: E402

import os
from pathlib import Path
import shutil
import subprocess
import textwrap

import numpy as np
import pytest

ct = pytest.importorskip("cuda.tile")

import tilelens
from tilelens.clients.sanitizer.cutile import CuTileSanitizer
from tilelens.core.data import Load, Store
from tilelens.wrapper import SANITIZER_COMMAND


@pytest.fixture(autouse=True)
def clear_traces():
    tilelens.clear()
    yield
    tilelens.clear()


class CountingSanitizer(CuTileSanitizer):
    """Collect reports instead of exiting, and record which blocks ran."""

    def __init__(self):
        super().__init__(abort_on_error=False)
        self.blocks: list[tuple[int, ...]] = []

    def grid_idx_callback(self, grid_idx):
        self.blocks.append(grid_idx)
        super().grid_idx_callback(grid_idx)


def _sanitize(kernel, grid, *args):
    sanitizer = CountingSanitizer()
    tilelens.trace(sanitizer, frontend="cutile")(kernel)[grid](*args)
    return sanitizer


def _elements(sanitizer, array):
    return [
        (record.violation_address - array.ctypes.data) // array.itemsize
        for record in sanitizer.records
    ]


def masked_gather_kernel(src, dst, bound):
    idx = ct.bid(0) * 4 + ct.arange(4, dtype=ct.int32)
    vals = ct.gather(src, idx, mask=idx < bound, check_bounds=False)
    ct.store(dst, (ct.bid(0),), vals)


def test_reports_wrong_mask_for_every_block_from_one_block():
    src = np.arange(10, dtype=np.float32)
    # The mask allows indices up to 11, but src has 10 elements.
    sanitizer = _sanitize(masked_gather_kernel, (3,), src, np.zeros(12, np.float32), 12)

    assert [record.op_type for record in sanitizer.records] == [Load]
    assert _elements(sanitizer, src) == [10]
    assert sanitizer.records[0].tensor_name == "src"
    # Z3 proves the last block goes out of bounds without running it.
    assert sanitizer.blocks == [(0, 0, 0)]


def test_correct_masks_are_not_reported():
    src = np.arange(10, dtype=np.float32)
    sanitizer = _sanitize(masked_gather_kernel, (3,), src, np.zeros(12, np.float32), 10)
    assert sanitizer.records == []

    def masked_2d_kernel(src, dst, rows):
        rows_idx = ct.arange(4, dtype=ct.int32)
        vals = ct.gather(src, (rows_idx, 1), mask=rows_idx < rows, check_bounds=False)
        ct.scatter(dst, rows_idx, vals, mask=rows_idx < rows, check_bounds=False)

    src_2d = np.arange(12, dtype=np.float32).reshape(3, 4)
    dst = np.zeros(3, dtype=np.float32)
    sanitizer = _sanitize(masked_2d_kernel, (1,), src_2d, dst, 3)
    assert sanitizer.records == []
    np.testing.assert_array_equal(dst, src_2d[:, 1])


def test_accesses_cutile_bounds_checks_are_not_reported():
    def checked_kernel(src, dst):
        idx = ct.bid(0) * 4 + ct.arange(4, dtype=ct.int32)
        # cuTile masks these out-of-bounds lanes itself.
        tile = ct.gather(src, idx)
        ct.scatter(dst, idx, tile)
        ct.store(dst, (ct.bid(0),), ct.load(src, (ct.bid(0),), (4,)))

    src = np.arange(10, dtype=np.float32)
    sanitizer = _sanitize(checked_kernel, (3,), src, np.zeros(10, np.float32))
    assert sanitizer.records == []


def test_reports_unchecked_scatter_in_loop_and_keeps_running():
    def looped_scatter_kernel(dst):
        for step in range(3):
            idx = step * 4 + ct.arange(4, dtype=ct.int32)
            ct.scatter(dst, idx, 1.0, check_bounds=False)

    dst = np.zeros(10, dtype=np.float32)
    sanitizer = _sanitize(looped_scatter_kernel, (1,), dst)

    assert [record.op_type for record in sanitizer.records] == [Store]
    assert _elements(sanitizer, dst) == [10]
    # After the report, only the out-of-bounds lanes are skipped.
    np.testing.assert_array_equal(dst, np.ones(10, np.float32))


def test_reports_unchecked_atomic():
    def atomic_kernel(counts):
        ct.atomic_add(counts, ct.arange(8, dtype=ct.int32), 1, check_bounds=False)

    counts = np.zeros(6, dtype=np.int32)
    sanitizer = _sanitize(atomic_kernel, (1,), counts)
    assert _elements(sanitizer, counts) == [6]
    np.testing.assert_array_equal(counts, np.ones(6, np.int32))


def test_loaded_indices_are_checked_in_every_block():
    def indirect_kernel(src, indices, dst):
        idx = ct.load(indices, (ct.bid(0),), (2,))
        ct.store(dst, (ct.bid(0),), ct.gather(src, idx, check_bounds=False))

    src = np.arange(8, dtype=np.float32)
    indices = np.array([0, 3, 1, 9], dtype=np.int32)
    sanitizer = _sanitize(indirect_kernel, (2,), src, indices, np.zeros(4, np.float32))

    # Indices from memory differ per block, so each block is checked concretely.
    assert _elements(sanitizer, src) == [9]
    assert len(sanitizer.blocks) == 2


def test_block_dependent_control_flow_runs_every_block():
    def last_block_kernel(src, dst):
        if ct.bid(0) == 2:
            idx = 8 + ct.arange(4, dtype=ct.int32)
            ct.store(dst, (0,), ct.gather(src, idx, check_bounds=False))

    src = np.arange(10, dtype=np.float32)
    sanitizer = _sanitize(last_block_kernel, (3,), src, np.zeros(4, np.float32))
    assert _elements(sanitizer, src) == [10]
    assert len(sanitizer.blocks) == 3


def test_block_dependent_loop_bound_runs_every_block():
    def growing_loop_kernel(dst):
        for step in range(ct.bid(0) + 1):
            idx = step * 4 + ct.arange(4, dtype=ct.int32)
            ct.scatter(dst, idx, 1.0, check_bounds=False)

    dst = np.zeros(10, dtype=np.float32)
    sanitizer = _sanitize(growing_loop_kernel, (3,), dst)
    assert _elements(sanitizer, dst) == [10]
    assert len(sanitizer.blocks) == 3


CLI_GATHER_KERNEL = """
import cuda.tile as ct
import numpy as np


@ct.kernel
def gather_kernel(src, dst, bound):
    idx = ct.bid(0) * 4 + ct.arange(4, dtype=ct.int32)
    vals = ct.gather(src, idx, mask=idx < bound, check_bounds=False)
    ct.store(dst, (ct.bid(0),), vals)


src = np.arange(10, dtype=np.float32)
dst = np.zeros(12, dtype=np.float32)
"""


def _run_sanitizer_cli(tmp_path: Path, launch_code: str):
    exe = shutil.which(SANITIZER_COMMAND)
    assert exe is not None, f"{SANITIZER_COMMAND} command not found"
    script = tmp_path / "cutile_program.py"
    script.write_text(
        CLI_GATHER_KERNEL + textwrap.dedent(launch_code), encoding="utf-8"
    )
    return subprocess.run(
        [exe, str(script)],
        cwd=str(tmp_path),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )


def test_tile_sanitizer_cli_reports_cutile_launch(tmp_path: Path):
    proc = _run_sanitizer_cli(
        tmp_path,
        """
        ct.launch(None, (3,), gather_kernel, (src, dst, 12))
        print("FINISHED")
        """,
    )

    assert proc.returncode == 1
    assert "ILLEGAL MEMORY ACCESS DETECTED" in proc.stdout
    assert "mask=idx < bound, check_bounds=False" in proc.stdout
    assert "FINISHED" not in proc.stdout


def test_tile_sanitizer_cli_passes_correct_cutile_launch(tmp_path: Path):
    proc = _run_sanitizer_cli(
        tmp_path,
        """
        ct.launch(None, (3,), gather_kernel, (src, dst, 10))
        print("FINISHED")
        """,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ILLEGAL MEMORY ACCESS" not in proc.stdout
    assert "FINISHED" in proc.stdout
