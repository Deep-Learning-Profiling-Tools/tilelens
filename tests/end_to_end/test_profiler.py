import importlib

import torch
import pytest

import triton
import triton.language as tl

import tilelens
from tilelens.clients import Profiler
from tilelens.core.config import config as cfg


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


# ======== Case 2: Check if for loop can be unrolled ========
@triton.jit
def for_loop_test_kernel(
    in_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Test kernel with for-loops to verify loop statistics tracking.

    This kernel contains 4 for-loops:
    - First loop: Python range, iterates 10 times
    - Second loop: Python range, iterates 5 times
    - Third loop: tl.range, iterates 8 times
    - Fourth loop: tl.static_range, iterates 6 times
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load input
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)

    # First for-loop: Python range, 10 iterations
    result = x
    for i in range(10):
        result = result + 1.0

    # Second for-loop: Python range, 5 iterations
    for j in range(5):
        result = result * 2.0

    # Third for-loop: tl.range, 8 iterations
    for k in tl.range(2, 10):  # 2, 3, 4, 5, 6, 7, 8, 9
        result = result + 0.5

    # Fourth for-loop: tl.static_range, 6 iterations
    for m in tl.static_range(0, 12, 2):  # 0, 2, 4, 6, 8, 10
        result = result - 0.25

    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)


def test_for_loop_statistics(_isolate_profiler_cfg):
    """
    Test that the profiler correctly tracks for-loop statistics.
    """
    N = 100
    BLOCK_SIZE = 32
    num_blocks = triton.cdiv(N, BLOCK_SIZE)

    # Create input/output tensors
    x = torch.randn(N, dtype=torch.float32)
    y = torch.empty(N, dtype=torch.float32)

    # Create profiler in the test function
    # Disable block sampling to ensure all blocks are executed
    cfg.profiler_enable_block_sampling = False
    cfg.profiler_disable_buffer_load_check = True
    loop_profiler = Profiler(disable_load_mask_percentage_check=True)
    traced_kernel = tilelens.trace(loop_profiler)(for_loop_test_kernel)

    # Run the kernel
    grid = (num_blocks,)
    traced_kernel[grid](x, y, N, BLOCK_SIZE)

    # Expected loop statistics:
    # The kernel has 4 for-loops, but they are executed once per grid
    # Each loop should be recorded only once (when first encountered)
    # Loop 1: range(10) -> 10 steps, type: python_range
    # Loop 2: range(5) -> 5 steps, type: python_range
    # Loop 3: tl.range(2, 10) -> 8 steps, type: tl_range
    # Loop 4: tl.static_range(0, 12, 2) -> 6 steps, type: tl_static_range

    expected_num_loops = 4
    expected_loop_steps = [10, 5, 8, 6]
    expected_range_types = [
        "python_range",
        "python_range",
        "tl_range",
        "tl_static_range",
    ]

    # Verify the loop statistics from profiler
    assert (
        len(loop_profiler.loop_info) == expected_num_loops
    ), f"Expected {expected_num_loops} loops, got {len(loop_profiler.loop_info)}"

    for idx, (lineno, loop_info) in enumerate(loop_profiler.loop_info.items()):
        expected_steps = expected_loop_steps[idx]
        expected_type = expected_range_types[idx]

        assert (
            loop_info.length == expected_steps
        ), f"Loop #{idx + 1}: Expected {expected_steps} steps, got {loop_info.length}"
        assert (
            loop_info.range_type == expected_type
        ), f"Loop #{idx + 1}: Expected type {expected_type}, got {loop_info.range_type}"
        assert isinstance(
            lineno, int
        ), f"Loop #{idx + 1}: lineno should be int, got {type(lineno)}"
        assert lineno > 0, f"Loop #{idx + 1}: lineno should be positive, got {lineno}"


# ======== Case 3: Check masked element percentage for tuning BLOCK_SIZE ========
@triton.jit
def mask_percentage_test_kernel(
    in_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Test kernel with a known mix of masked and unmasked load/store operations.

    Expected operations per block:
    - 2 masked loads (with mask parameter)
    - 1 unmasked load (without mask parameter)
    - 2 masked stores (with mask parameter)
    - 1 unmasked store (without mask parameter)
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Masked load #1: load with mask
    a = tl.load(in_ptr + offsets, mask=mask, other=0.0)

    # Masked load #2: another load with mask
    b = tl.load(in_ptr + offsets, mask=mask, other=1.0)

    # Unmasked load: load without mask
    c = tl.load(in_ptr + tl.arange(0, BLOCK_SIZE))  # No mask - this is RawLoad

    # Compute result
    result = a + b + c

    # Masked store #1: store with mask
    tl.store(out_ptr + offsets, result, mask=mask)

    # Masked store #2: another store with mask
    tl.store(out_ptr + offsets + N, result * 2, mask=mask)

    # Unmasked store: store without mask
    tl.store(out_ptr + tl.arange(0, BLOCK_SIZE), result)  # No mask - this is RawStore


def test_mask_percentage(_isolate_profiler_cfg):
    """
    Test that the profiler correctly reports the mask element statistics.

    The profiler now tracks:
    - Total number of mask elements across all operations
    - Number of False elements in all masks
    """
    N = 100
    BLOCK_SIZE = 32
    num_blocks = triton.cdiv(N, BLOCK_SIZE)  # Should be 4 blocks

    # Create input/output tensors
    x = torch.randn(N, dtype=torch.float32)
    y = torch.empty(N * 2, dtype=torch.float32)

    # Create profiler in the test function
    # Disable block sampling to ensure all blocks are executed
    cfg.profiler_enable_block_sampling = False
    cfg.profiler_disable_buffer_load_check = True
    mask_profiler = Profiler()
    traced_kernel = tilelens.trace(mask_profiler)(mask_percentage_test_kernel)

    # Run the kernel
    grid = (num_blocks,)
    traced_kernel[grid](x, y, N, BLOCK_SIZE)

    # Expected mask statistics:
    # Note: In the triton interpreter, load/store operations without an explicit mask
    # are interpreted as having a mask where all elements are True.
    #
    # Block 0: offsets=[0..31], mask=[True]*32 (0 False)
    # Block 1: offsets=[32..63], mask=[True]*32 (0 False)
    # Block 2: offsets=[64..95], mask=[True]*32 (0 False)
    # Block 3: offsets=[96..127], mask=[True]*4 + [False]*28 (28 False)
    #
    # Each block has 3 loads total (2 with explicit masks + 1 without explicit mask):
    # - 2 loads with explicit mask: offsets < N
    # - 1 load without explicit mask: always all-True
    # - Total load mask elements = 3 loads * 32 elements * 4 blocks = 384
    # - False elements in loads = 2 masked loads * 28 False (only in block 3) = 56
    #
    # Similarly for stores (2 with explicit masks + 1 without):
    # - Total store mask elements = 3 stores * 32 elements * 4 blocks = 384
    # - False elements in stores = 2 masked stores * 28 False (only in block 3) = 56

    expected_load_mask_total = 384
    expected_load_mask_false = 56
    expected_store_mask_total = 384
    expected_store_mask_false = 56

    # Verify the statistics from profiler
    assert (
        mask_profiler.load_mask_total_count == expected_load_mask_total
    ), f"Expected {expected_load_mask_total} total load mask elements, got {mask_profiler.load_mask_total_count}"
    assert (
        mask_profiler.load_mask_false_count == expected_load_mask_false
    ), f"Expected {expected_load_mask_false} false load mask elements, got {mask_profiler.load_mask_false_count}"
    assert (
        mask_profiler.store_mask_total_count == expected_store_mask_total
    ), f"Expected {expected_store_mask_total} total store mask elements, got {mask_profiler.store_mask_total_count}"
    assert (
        mask_profiler.store_mask_false_count == expected_store_mask_false
    ), f"Expected {expected_store_mask_false} false store mask elements, got {mask_profiler.store_mask_false_count}"

    # Verify the masked percentage calculation
    expected_load_masked_percentage = (56 / 384) * 100  # ~14.58%
    expected_store_masked_percentage = (56 / 384) * 100  # ~14.58%

    actual_load_masked_percentage = (
        mask_profiler.load_mask_false_count / mask_profiler.load_mask_total_count
    ) * 100
    actual_store_masked_percentage = (
        mask_profiler.store_mask_false_count / mask_profiler.store_mask_total_count
    ) * 100

    assert (
        abs(actual_load_masked_percentage - expected_load_masked_percentage) < 0.01
    ), f"Expected load masked percentage {expected_load_masked_percentage:.2f}%, got {actual_load_masked_percentage:.2f}%"
    assert (
        abs(actual_store_masked_percentage - expected_store_masked_percentage) < 0.01
    ), f"Expected store masked percentage {expected_store_masked_percentage:.2f}%, got {actual_store_masked_percentage:.2f}%"


# ======== Block Sampling ========
@triton.jit
def block_sampling_test_kernel(
    x_ptr,
    y_ptr,
    counter_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Test kernel that increments a counter for each block executed.
    This allows us to verify that block sampling is working correctly.
    """
    pid = tl.program_id(0)

    # Atomically increment counter to count executions
    tl.atomic_add(counter_ptr, 1)

    # Normal kernel logic
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = x * 2
    tl.store(y_ptr + offsets, y, mask=mask)


@pytest.mark.parametrize(
    "test_name,enable_sampling,k_value,expected_executions",
    [
        ("No block sampling", False, None, 8),
        ("Block sampling k=1", True, 1, 1),
        ("Block sampling k=3", True, 3, 3),
        ("Block sampling k=5", True, 5, 5),
        ("Block sampling k=8 (all blocks)", True, 8, 8),
        ("Block sampling k=20 (exceeds total)", True, 20, 8),
    ],
    ids=lambda x: x if isinstance(x, str) else str(x),
)
def test_block_sampling(
    test_name, enable_sampling, k_value, expected_executions, _isolate_profiler_cfg
):
    """
    Test that the block sampling feature correctly samples k blocks from the grid.

    This test verifies:
    - Block sampling can be enabled/disabled
    - The k parameter correctly limits the number of blocks executed
    - When k > total blocks, all blocks are executed
    - The sampling works with different k values
    """
    # Test configuration
    n_elements = 1024
    BLOCK_SIZE = 128
    grid = (n_elements // BLOCK_SIZE,)  # Creates 8 blocks

    # Create test data
    x = torch.randn(n_elements, dtype=torch.float32)
    y = torch.zeros_like(x)
    counter = torch.zeros(1, dtype=torch.int32)

    # Configure block sampling via config - must be done BEFORE creating Profiler
    # Set the private attribute directly to ensure it takes effect
    cfg.profiler_enable_block_sampling = enable_sampling
    cfg.profiler_disable_buffer_load_check = True

    # Create profiler with k value
    profiler = Profiler(
        k=k_value,
        disable_load_mask_percentage_check=True,
    )

    # Apply trace decorator and run kernel
    traced_kernel = tilelens.trace(profiler)(block_sampling_test_kernel)
    traced_kernel[grid](x, y, counter, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    # Get actual execution count
    actual_executions = counter.item()

    # Verify the count matches expectation
    assert (
        actual_executions == expected_executions
    ), f"{test_name}: Expected {expected_executions} executions, got {actual_executions}"

    # Verify profiler settings
    if enable_sampling:
        assert (
            profiler.block_sampling == enable_sampling
        ), f"{test_name}: block_sampling should be {enable_sampling}, got {profiler.block_sampling}"
        assert (
            profiler.k == k_value
        ), f"{test_name}: k should be {k_value}, got {profiler.k}"


# ======== Skipping Load and Store ========
@triton.jit
def simple_kernel(in_ptr, out_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    data = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    result = data * 2.0
    tl.store(out_ptr + offsets, result, mask=mask)


def test_load_store_skip_disabled(_isolate_profiler_cfg):
    """Test that load/store operations work normally when skipping is disabled."""
    N = 128
    BLOCK_SIZE = 32

    # Create input data
    x = torch.ones(N, dtype=torch.float32) * 3.0  # All values are 3.0
    y = torch.zeros(N, dtype=torch.float32)  # Output

    # Set configuration - disable skipping for normal execution
    cfg.profiler_enable_load_store_skipping = False
    # Also disable block sampling to ensure all blocks are executed
    cfg.profiler_enable_block_sampling = False
    cfg.profiler_disable_buffer_load_check = True

    # Create profiler and traced kernel
    profiler = Profiler()
    traced_kernel = tilelens.trace(profiler)(simple_kernel)

    # Run kernel
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    traced_kernel[grid](x, y, N, BLOCK_SIZE)

    # Verify normal execution: output should be input * 2 = 6.0
    assert torch.allclose(
        y, torch.ones_like(y) * 6.0
    ), f"Normal execution should produce 6.0, got {y[:10]}"


def test_load_store_skip_enabled(_isolate_profiler_cfg):
    """Test that load/store operations are skipped when skipping is enabled."""
    N = 128
    BLOCK_SIZE = 32

    # Create input data
    x = torch.ones(N, dtype=torch.float32) * 3.0  # All values are 3.0
    y = torch.zeros(N, dtype=torch.float32)  # Output

    # Set configuration - enable skipping
    cfg.profiler_enable_load_store_skipping = True
    # Also disable block sampling to ensure all blocks are executed
    cfg.profiler_enable_block_sampling = False
    cfg.profiler_disable_buffer_load_check = True

    # Create profiler and traced kernel
    profiler = Profiler()
    traced_kernel = tilelens.trace(profiler)(simple_kernel)

    # Run kernel
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    traced_kernel[grid](x, y, N, BLOCK_SIZE)

    # Verify skipped execution: output should remain 0.0 (unchanged)
    assert torch.allclose(
        y, torch.zeros_like(y)
    ), f"Skipped execution should leave output unchanged, got {y[:10]}"


# ======== Buffer Load Check ========
def test_buffer_load_check_without_active_driver(
    _isolate_profiler_cfg, monkeypatch, capsys
):
    """The default buffer load check must not fail launches on driverless hosts."""
    # ``triton.runtime.driver`` resolves to the DriverConfig object, not the module.
    triton_driver = importlib.import_module("triton.runtime.driver")

    def _no_driver():
        raise RuntimeError("0 active drivers ([]). There should only be one.")

    # Simulate a CPU-only host even when a GPU driver is available.
    monkeypatch.setattr(triton_driver, "_create_driver", _no_driver)
    monkeypatch.setattr(triton_driver.driver, "_default", None)
    monkeypatch.setattr(triton_driver.driver, "_active", None)

    N = 128
    BLOCK_SIZE = 32
    x = torch.ones(N, dtype=torch.float32) * 3.0
    y = torch.zeros(N, dtype=torch.float32)

    cfg.profiler_enable_load_store_skipping = False
    cfg.profiler_enable_block_sampling = False
    cfg.profiler_disable_buffer_load_check = False

    profiler = Profiler()
    traced_kernel = tilelens.trace(profiler)(simple_kernel)
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    traced_kernel[grid](x, y, N, BLOCK_SIZE)

    assert torch.allclose(y, torch.ones_like(y) * 6.0)
    assert profiler.buffer_load_check_skipped_no_driver is True
    assert profiler.disable_buffer_load_check is True
    assert profiler.potential_buffer_load_issue_found is False
    assert "Skipped: no active Triton driver" in capsys.readouterr().out
