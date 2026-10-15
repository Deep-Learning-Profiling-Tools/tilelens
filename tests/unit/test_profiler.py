import numpy as np
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tilelens.core.config import config as cfg
from tilelens.clients.profiler.profiler import (
    Profiler,
    LoopInfo,
    MaskOpStats,
    AggregatedMaskOpStats,
)
from tilelens.clients.profiler.data import LoadStoreBytes


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


# ======== Profiler Init Tests =========


def test_profiler_init_defaults(_isolate_profiler_cfg):
    """Test Profiler default attribute initialization."""
    cfg.profiler_enable_block_sampling = False
    cfg.profiler_enable_load_store_skipping = False
    cfg.profiler_disable_buffer_load_check = False

    profiler = Profiler()

    # Verify load_bytes and store_bytes initialization
    assert isinstance(profiler.load_bytes, LoadStoreBytes)
    assert profiler.load_bytes.type == "load"
    assert profiler.load_bytes.total_bytes_true == 0
    assert profiler.load_bytes.total_bytes_attempted == 0

    assert isinstance(profiler.store_bytes, LoadStoreBytes)
    assert profiler.store_bytes.type == "store"
    assert profiler.store_bytes.total_bytes_true == 0
    assert profiler.store_bytes.total_bytes_attempted == 0

    # Verify loop_info initialization
    assert isinstance(profiler.loop_info, dict)
    assert len(profiler.loop_info) == 0

    # Verify mask statistics initialization
    assert profiler.load_mask_total_count == 0
    assert profiler.load_mask_false_count == 0
    assert profiler.store_mask_total_count == 0
    assert profiler.store_mask_false_count == 0

    # Verify default flags
    assert profiler.disable_for_loop_unroll_check is False
    assert profiler.disable_load_mask_percentage_check is False
    assert profiler.callpath is True


def test_profiler_init_with_params(_isolate_profiler_cfg):
    """Test Profiler initialization with custom parameters."""
    cfg.profiler_enable_block_sampling = True
    cfg.profiler_enable_load_store_skipping = False
    cfg.profiler_disable_buffer_load_check = False

    profiler = Profiler(
        callpath=False,
        disable_for_loop_unroll_check=True,
        disable_load_mask_percentage_check=True,
        k=5,
    )

    assert profiler.callpath is False
    assert profiler.disable_for_loop_unroll_check is True
    assert profiler.disable_load_mask_percentage_check is True
    assert profiler.k == 5
    assert profiler.block_sampling is True


# ======== DataClass Tests =========


def test_loop_info_dataclass():
    """Test LoopInfo dataclass creation and attributes."""
    # Test default values
    loop_info_default = LoopInfo()
    assert loop_info_default.length is None
    assert loop_info_default.range_type == "unknown"

    # Test with custom values
    loop_info = LoopInfo(length=10, range_type="python_range")
    assert loop_info.length == 10
    assert loop_info.range_type == "python_range"

    # Test with tl_range type
    loop_info_tl = LoopInfo(length=8, range_type="tl_range")
    assert loop_info_tl.length == 8
    assert loop_info_tl.range_type == "tl_range"


def test_mask_op_stats_dataclass():
    """Test MaskOpStats dataclass creation and attributes."""
    stats = MaskOpStats(
        op_type="load",
        lineno=42,
        filename="test_kernel.py",
        code_line="tl.load(ptr + offsets, mask=mask)",
        total_elements=128,
        false_elements=32,
    )

    assert stats.op_type == "load"
    assert stats.lineno == 42
    assert stats.filename == "test_kernel.py"
    assert stats.code_line == "tl.load(ptr + offsets, mask=mask)"
    assert stats.total_elements == 128
    assert stats.false_elements == 32


def test_aggregated_mask_op_stats_dataclass():
    """Test AggregatedMaskOpStats dataclass default values."""
    # Test default values
    agg_stats = AggregatedMaskOpStats()
    assert agg_stats.total == 0
    assert agg_stats.false == 0
    assert agg_stats.filename == ""
    assert agg_stats.code_line == ""
    assert agg_stats.op_type == ""

    # Test with custom values
    agg_stats_custom = AggregatedMaskOpStats(
        total=1000,
        false=250,
        filename="kernel.py",
        code_line="tl.store(...)",
        op_type="store",
    )
    assert agg_stats_custom.total == 1000
    assert agg_stats_custom.false == 250
    assert agg_stats_custom.filename == "kernel.py"
    assert agg_stats_custom.code_line == "tl.store(...)"
    assert agg_stats_custom.op_type == "store"


# ======== Block Sampling Grid Callback Tests =========


def test_block_sampling_grid_callback(_isolate_profiler_cfg):
    """Test grid_callback block sampling logic with mocked random permutation."""
    cfg.profiler_enable_block_sampling = True
    cfg.profiler_enable_load_store_skipping = False
    cfg.profiler_disable_buffer_load_check = True

    profiler = Profiler(k=3)

    # Mock np.random.permutation to return predictable values
    with patch.object(
        np.random, "permutation", return_value=np.array([0, 2, 4, 1, 3, 5, 6, 7])
    ):
        # Test with 1D grid of 8 blocks
        profiler.grid_callback((8,))

        # Should sample k=3 blocks
        assert profiler.sampled_blocks is not None
        assert len(profiler.sampled_blocks) == 3

        # The sampled indices should be the first k elements from permutation
        # which are 0, 2, 4 -> corresponding to (0, 0, 0), (2, 0, 0), (4, 0, 0)
        assert (0, 0, 0) in profiler.sampled_blocks
        assert (2, 0, 0) in profiler.sampled_blocks
        assert (4, 0, 0) in profiler.sampled_blocks


def test_block_sampling_grid_callback_2d(_isolate_profiler_cfg):
    """Test grid_callback with 2D grid."""
    cfg.profiler_enable_block_sampling = True
    cfg.profiler_enable_load_store_skipping = False
    cfg.profiler_disable_buffer_load_check = True

    profiler = Profiler(k=2)

    # 2x3 grid = 6 blocks total
    with patch.object(
        np.random, "permutation", return_value=np.array([0, 3, 1, 2, 4, 5])
    ):
        profiler.grid_callback((2, 3))

        assert profiler.sampled_blocks is not None
        assert len(profiler.sampled_blocks) == 2


def test_block_sampling_disabled(_isolate_profiler_cfg):
    """Test grid_callback when block sampling is disabled."""
    cfg.profiler_enable_block_sampling = False
    cfg.profiler_enable_load_store_skipping = False
    cfg.profiler_disable_buffer_load_check = True

    profiler = Profiler(k=3)

    profiler.grid_callback((8,))

    # When sampling is disabled, sampled_blocks should be None
    assert profiler.sampled_blocks is None


# ======== 32-bit Range Check Tests =========


def test_check_32bit_range_within_range(_isolate_profiler_cfg):
    """Test _check_32bit_range when offsets are within 32-bit range."""
    cfg.profiler_enable_block_sampling = False
    cfg.profiler_enable_load_store_skipping = False
    cfg.profiler_disable_buffer_load_check = False

    profiler = Profiler()
    profiler.has_buffer_load = True  # Simulate buffer_load detected

    # Offsets within 32-bit range
    byte_offset = np.array([0, 1000, 2**30, -(2**30)])
    offset_data = byte_offset // 4

    # Should not set potential_buffer_load_issue_found
    profiler._check_32bit_range(byte_offset, 4, offset_data)
    assert profiler.potential_buffer_load_issue_found is False


def test_check_32bit_range_outside_range(_isolate_profiler_cfg):
    """Test _check_32bit_range when offsets exceed 32-bit range."""
    cfg.profiler_enable_block_sampling = False
    cfg.profiler_enable_load_store_skipping = False
    cfg.profiler_disable_buffer_load_check = False

    profiler = Profiler()
    profiler.has_buffer_load = True  # Simulate buffer_load detected

    # Offsets outside 32-bit range (exceeds 2^31-1)
    byte_offset = np.array([0, 1000, 2**32])  # 2^32 exceeds 32-bit range
    offset_data = byte_offset // 4

    # Should not set issue found because there ARE offsets outside range
    profiler._check_32bit_range(byte_offset, 4, offset_data)
    # When some offsets are outside range, the issue is NOT flagged
    # (issue is only flagged when ALL offsets are within range but buffer_load is not used)
    assert profiler.potential_buffer_load_issue_found is False


def test_check_32bit_range_no_buffer_load(_isolate_profiler_cfg):
    """Test _check_32bit_range when buffer_load is not detected but should be used."""
    cfg.profiler_enable_block_sampling = False
    cfg.profiler_enable_load_store_skipping = False
    cfg.profiler_disable_buffer_load_check = False

    profiler = Profiler()
    profiler.has_buffer_load = False  # Simulate buffer_load NOT detected

    # Offsets within 32-bit range - should trigger warning
    byte_offset = np.array([0, 1000, 2000])
    offset_data = byte_offset // 4

    profiler._check_32bit_range(byte_offset, 4, offset_data)
    # Should flag issue because offsets are within range but buffer_load is not used
    assert profiler.potential_buffer_load_issue_found is True


# ======== Buffer Load Check: Skip Tests =========


def test_buffer_load_check_skipped_without_a_compile(_isolate_profiler_cfg, capsys):
    cfg.profiler_disable_buffer_load_check = False

    Profiler().finalize()

    assert "the kernel was not compiled" in capsys.readouterr().out


def test_buffer_load_check_skipped_on_a_non_amd_compile(_isolate_profiler_cfg, capsys):
    cfg.profiler_disable_buffer_load_check = False
    profiler = Profiler()

    profiler.post_warmup_callback(None, SimpleNamespace(asm={"ptx": "ld.global.f32"}))
    profiler.finalize()

    assert "not an AMD (amdgcn) kernel" in capsys.readouterr().out


# ======== Pre-run Callback Tests =========


def test_pre_run_callback_with_sampling(_isolate_profiler_cfg):
    """Test pre_run_callback behavior when block sampling is enabled."""
    cfg.profiler_enable_block_sampling = True
    cfg.profiler_enable_load_store_skipping = False
    cfg.profiler_disable_buffer_load_check = True

    profiler = Profiler(k=2)

    # Set up sampled blocks
    profiler.sampled_blocks = {(0, 0, 0), (2, 0, 0)}

    # Test with sampled block
    profiler.current_grid_idx = (0, 0, 0)
    assert profiler.pre_run_callback(lambda: None) is True

    # Test with non-sampled block
    profiler.current_grid_idx = (1, 0, 0)
    assert profiler.pre_run_callback(lambda: None) is False

    # Test with another sampled block
    profiler.current_grid_idx = (2, 0, 0)
    assert profiler.pre_run_callback(lambda: None) is True


def test_pre_run_callback_without_sampling(_isolate_profiler_cfg):
    """Test pre_run_callback behavior when block sampling is disabled."""
    cfg.profiler_enable_block_sampling = False
    cfg.profiler_enable_load_store_skipping = False
    cfg.profiler_disable_buffer_load_check = True

    profiler = Profiler()

    # When sampling is disabled, pre_run_callback should always return True
    profiler.current_grid_idx = (0, 0, 0)
    assert profiler.pre_run_callback(lambda: None) is True

    profiler.current_grid_idx = (5, 3, 2)
    assert profiler.pre_run_callback(lambda: None) is True
