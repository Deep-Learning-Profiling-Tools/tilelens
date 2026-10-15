from collections.abc import Callable

from ...core.client import Client
from ...core.callbacks import OpCallbacks, ForLoopCallbacks
from ...core.data import Op, Load, Store, AddPtr, Dot
from ...core.config import config as cfg
from ...core.patch import LoopSite
from .data import LoadStoreBytes
from ...utils.traceback_utils import (
    extract_user_frames,
    extract_complete_statement_from_line,
)
from triton.runtime.interpreter import _get_np_dtype, TensorHandle
import numpy as np
from dataclasses import dataclass, replace


@dataclass(frozen=False)
class LoopInfo:
    length: int | None = None
    range_type: str = "unknown"


@dataclass
class MaskOpStats:
    """Statistics for a single mask operation."""

    op_type: str  # "load" or "store"
    lineno: int
    filename: str
    code_line: str
    total_elements: int
    false_elements: int


@dataclass
class AggregatedMaskOpStats:
    total: int = 0
    false: int = 0
    filename: str = ""
    code_line: str = ""
    op_type: str = ""


class Profiler(Client):
    NAME = "profiler"

    def __init__(
        self,
        callpath: bool = True,
        disable_for_loop_unroll_check: bool = False,
        disable_load_mask_percentage_check: bool = False,
        k: int | None = None,
    ):
        super().__init__()  # Initialize parent class
        self.callpath = callpath
        self.load_bytes = LoadStoreBytes("load", 0, 0)
        self.store_bytes = LoadStoreBytes("store", 0, 0)

        # Case 2: For-loop Unrolling Statistics
        self.disable_for_loop_unroll_check = disable_for_loop_unroll_check
        self.loop_info: dict[int, LoopInfo] = {}

        # Case 3: Mask Ratio Statistics
        self.disable_load_mask_percentage_check = disable_load_mask_percentage_check
        self.load_mask_total_count = (
            0  # Total number of mask elements in all load operations
        )
        self.load_mask_false_count = (
            0  # Total number of False elements in all load masks
        )
        self.store_mask_total_count = (
            0  # Total number of mask elements in all store operations
        )
        self.store_mask_false_count = (
            0  # Total number of False elements in all store masks
        )
        # Per-operation statistics for detailed analysis
        self.mask_op_stats: list[MaskOpStats] = []

        # Case 4: Buffer Load Check
        self.has_buffer_load = False
        self.disable_buffer_load_check = cfg.profiler_disable_buffer_load_check
        self.potential_buffer_load_issue_found = False

        # Block sampling
        self.block_sampling = cfg.profiler_enable_block_sampling
        self.k = k
        self.sampled_blocks: set[tuple[int, ...]] | None = None
        self.current_grid_idx: tuple[int, ...] | None = None

        # Load & Store Skipping
        # Config has enable_load_store_skipping, but profiler uses disable_load_store_skipping
        self.disable_load_store_skipping = not cfg.profiler_enable_load_store_skipping

    def pre_run_callback(self, fn: Callable) -> bool:
        # If block sampling is enabled, check if current block is selected
        if self.block_sampling and self.sampled_blocks is not None:
            return self.current_grid_idx in self.sampled_blocks
        return True

    def post_run_callback(self, fn: Callable) -> bool:
        return True

    def pre_warmup_callback(self, jit_fn, *args, **kwargs) -> bool:
        # Skip warmup if buffer load check is disabled
        return not self.disable_buffer_load_check

    def post_warmup_callback(self, jit_fn, ret) -> None:
        if not ret:
            return

        if (
            not self.disable_buffer_load_check
            and hasattr(ret, "asm")
            and "amdgcn" in ret.asm
        ):
            self.has_buffer_load = "buffer_load" in ret.asm["amdgcn"]
            if self.has_buffer_load:
                print("Detected buffer_load instruction in kernel ASM!")

    def arg_callback(self, name, arg, arg_cvt):
        pass

    def grid_idx_callback(self, grid_idx: tuple[int, ...]):
        # Store the current grid index
        self.current_grid_idx = grid_idx

    @property
    def current_grid_idx(self) -> tuple[int, ...] | None:
        return self._get_thread_local("current_grid_idx", None)

    @current_grid_idx.setter
    def current_grid_idx(self, grid_idx: tuple[int, ...] | None) -> None:
        self._set_thread_local("current_grid_idx", grid_idx)

    def grid_callback(self, grid: tuple[int, ...]):
        # If block sampling is enabled, determine which blocks to sample
        if self.block_sampling:
            # Generate all possible indices
            all_indices = []
            for x in range(grid[0] if len(grid) > 0 else 1):
                for y in range(grid[1] if len(grid) > 1 else 1):
                    for z in range(grid[2] if len(grid) > 2 else 1):
                        all_indices.append((x, y, z))

            # Sample k blocks randomly
            total_blocks = len(all_indices)
            k = min(self.k if self.k is not None else 1, total_blocks)

            # Randomly select k indices
            perm = np.random.permutation(total_blocks)[:k]
            sampled_indices = [all_indices[i] for i in perm]
            self.sampled_blocks = set(sampled_indices)
        else:
            # No sampling - all blocks will be executed
            self.sampled_blocks = None

    def _report_load_store_bytes(
        self, type, ptr: TensorHandle, mask: TensorHandle
    ):  # internal methods assumed to be called under the lock
        dtype_tt = ptr.get_element_ty()
        dtype_np: np.dtype = _get_np_dtype(dtype_tt)
        mask_true = np.count_nonzero(mask.data)
        mask_false = np.count_nonzero(np.logical_not(mask.data))
        total_bytes_true = mask_true * dtype_np.itemsize
        total_bytes_attempted = (mask_true + mask_false) * dtype_np.itemsize
        if type == "load":
            self.load_bytes.total_bytes_attempted += total_bytes_attempted
            self.load_bytes.total_bytes_true += total_bytes_true
        elif type == "store":
            self.store_bytes.total_bytes_attempted += total_bytes_attempted
            self.store_bytes.total_bytes_true += total_bytes_true

    def _check_32bit_range(
        self, byte_offset: np.ndarray, element_bytewidth: int, offset_data: np.ndarray
    ):
        """Check if byte offsets are within 32-bit signed integer range and print statistics."""
        assert isinstance(
            byte_offset, np.ndarray
        ), f"byte_offset must be np.ndarray, got {type(byte_offset)}"

        # Check if within 32-bit signed range (-2^31 to 2^31 - 1)
        INT32_MIN = -(2**31)
        INT32_MAX = 2**31 - 1

        within_32bit = np.logical_and(
            byte_offset >= INT32_MIN, byte_offset <= INT32_MAX
        )
        outside_32bit = np.logical_not(within_32bit)

        num_outside = np.count_nonzero(outside_32bit)

        # Check if ALL offsets are within 32-bit range
        if num_outside == 0:
            # All offsets are within 32-bit range
            # If we're on AMD GPU and buffer_load is NOT found, this is an error
            if self.has_buffer_load is False:
                # Buffer Load optimization should be used when offsets are within 32-bit range.
                self.potential_buffer_load_issue_found = True

    def register_op_callback(self, op_type: type[Op]) -> OpCallbacks:
        def _get_mask_stats(mask: TensorHandle) -> tuple[int, int]:
            """Get mask statistics: total count and false count.

            Args:
                mask: TensorHandle containing boolean mask data

            Returns:
                Tuple of (total_count, false_count)
            """
            total_count = mask.data.size
            false_count = np.count_nonzero(np.logical_not(mask.data))
            return total_count, false_count

        @self.lock_fn
        def pre_load_callback(ptr, mask, keys):
            self._report_load_store_bytes("load", ptr, mask)
            if not self.disable_load_mask_percentage_check:
                total_count, false_count = _get_mask_stats(mask)
                self.load_mask_total_count += total_count
                self.load_mask_false_count += false_count

                # Record per-operation statistics
                frames = extract_user_frames(num_frames=1)
                if frames:
                    _f = frames[-1]
                    lineno, filename = _f.lineno or 0, _f.filename
                    code_line = extract_complete_statement_from_line(filename, lineno)
                else:
                    lineno, filename, code_line = 0, "", ""
                if lineno > 0:
                    self.mask_op_stats.append(
                        MaskOpStats(
                            op_type="load",
                            lineno=lineno,
                            filename=filename,
                            code_line=code_line.strip() if code_line else "",
                            total_elements=total_count,
                            false_elements=false_count,
                        )
                    )

        @self.lock_fn
        def load_overrider(
            ptr, mask, other, cache_modifier, eviction_policy, is_volatile
        ):
            # Skip actual load, return zeros
            dtype_tt = ptr.get_element_ty()
            dtype_np = _get_np_dtype(dtype_tt)
            return TensorHandle(np.zeros_like(ptr.data, dtype=dtype_np), dtype_tt)

        @self.lock_fn
        def pre_store_callback(ptr, mask, keys):
            self._report_load_store_bytes("store", ptr, mask)
            if not self.disable_load_mask_percentage_check:
                total_count, false_count = _get_mask_stats(mask)
                self.store_mask_total_count += total_count
                self.store_mask_false_count += false_count

                # Record per-operation statistics
                frames = extract_user_frames(num_frames=1)
                if frames:
                    _f = frames[-1]
                    lineno, filename = _f.lineno or 0, _f.filename
                    code_line = extract_complete_statement_from_line(filename, lineno)
                else:
                    lineno, filename, code_line = 0, "", ""
                if lineno > 0:
                    self.mask_op_stats.append(
                        MaskOpStats(
                            op_type="store",
                            lineno=lineno,
                            filename=filename,
                            code_line=code_line.strip() if code_line else "",
                            total_elements=total_count,
                            false_elements=false_count,
                        )
                    )

        @self.lock_fn
        def store_overrider(ptr, value, mask, cache_modifier, eviction_policy):
            # Skip actual store
            pass

        @self.lock_fn
        def dot_overrider(a, b, d, input_precision, max_num_imprecise_acc):
            # Skip actual dot operation, return zeros with same shape as d
            # This replaces np.matmul(a_data, b_data, dtype=d.data.dtype) + d.data
            return TensorHandle(np.zeros_like(d.data), d.dtype.scalar)

        @self.lock_fn
        def pre_addptr_callback(ptr, offset):
            dtype_tt = ptr.get_element_ty()
            element_bitwidth = dtype_tt.primitive_bitwidth
            element_bytewidth = max(1, element_bitwidth // 8)

            # Get offset data
            offset_data = offset.data if isinstance(offset, TensorHandle) else offset

            # Calculate byte offset
            byte_offset = offset_data * element_bytewidth

            # Check if byte offsets are within 32-bit range
            if not self.disable_buffer_load_check:
                self._check_32bit_range(byte_offset, element_bytewidth, offset_data)

        if op_type is Load:
            if self.disable_load_store_skipping:
                return OpCallbacks(before_callback=pre_load_callback)
            else:
                return OpCallbacks(
                    before_callback=pre_load_callback, op_overrider=load_overrider
                )
        elif op_type is Store:
            if self.disable_load_store_skipping:
                return OpCallbacks(before_callback=pre_store_callback)
            else:
                return OpCallbacks(
                    before_callback=pre_store_callback, op_overrider=store_overrider
                )
        elif op_type is Dot:
            if self.disable_load_store_skipping:
                return OpCallbacks()
            else:
                return OpCallbacks(op_overrider=dot_overrider)
        elif op_type is AddPtr:
            return OpCallbacks(before_callback=pre_addptr_callback)

        return OpCallbacks()

    def register_for_loop_callback(self):
        @self.lock_fn
        def loop_hook_range_type(loop_site: LoopSite, range_type: str) -> None:
            lineno = loop_site.lineno
            cur = self.loop_info.get(lineno, LoopInfo())
            self.loop_info[lineno] = replace(cur, range_type=range_type)

        @self.lock_fn
        def loop_hook_before(loop_site: LoopSite, iterable):
            if self.disable_for_loop_unroll_check:
                return

            if not isinstance(iterable, range):
                return

            lineno = loop_site.lineno
            # Only record each unique loop (by line number) once
            # Different blocks will execute the same loop, so we deduplicate by lineno
            if self.loop_info[lineno].length is not None:
                return

            # Record loop information: line number and total steps
            length = len(iterable)
            # Update length in LoopInfo
            cur = self.loop_info.get(lineno, LoopInfo())
            self.loop_info[lineno] = replace(cur, length=length)

        @self.lock_fn
        def loop_hook_after(loop_site: LoopSite) -> None:
            # No action needed after loop for profiler
            pass

        return ForLoopCallbacks(
            range_type_callback=loop_hook_range_type,
            before_loop_callback=loop_hook_before,
            after_loop_callback=loop_hook_after,
        )

    def finalize(self) -> list:
        print("=" * 60, "Profiler Issues Summary", "=" * 60)
        if not self.disable_for_loop_unroll_check:
            print("\n" + "=" * 60)
            print(
                "-" * 10
                + " "
                + "Profiler: For-Loop Unrolling Statistics"
                + " "
                + "-" * 9
            )
            print("=" * 60)
            if self.loop_info:
                print(f"\nTotal for-loops detected: {len(self.loop_info)}\n")

                for idx, (lineno, loop_info) in enumerate(self.loop_info.items(), 1):
                    print(f"Loop #{idx}:")
                    print(f"  Line number:    {lineno}")
                    print(f"  Range type:     {loop_info.range_type}")
                    print(f"  Total steps:    {loop_info.length}")

                print("=" * 60)
            else:
                print("\nNo for-loops detected.\n")
                print("=" * 60)

        if not self.disable_load_mask_percentage_check:
            print("\n" + "=" * 60)
            print("-" * 10 + " " + "Profiler: Mask Ratio Statistics" + " " + "-" * 17)
            print("=" * 60)

            # Overall statistics
            if self.load_mask_total_count > 0:
                load_masked_percentage = (
                    self.load_mask_false_count / self.load_mask_total_count
                ) * 100
                print("\n" + "─" * 40)
                print("Overall Load Operations:")
                print(f"  Total mask elements:     {self.load_mask_total_count}")
                print(f"  False elements:          {self.load_mask_false_count}")
                print(f"  Masked percentage:       {load_masked_percentage:.2f}%")
            else:
                print("\nLoad Operations:")
                print("  No load operations detected")

            # Store statistics
            if self.store_mask_total_count > 0:
                store_masked_percentage = (
                    self.store_mask_false_count / self.store_mask_total_count
                ) * 100
                print("\nOverall Store Operations:")
                print(f"  Total mask elements:     {self.store_mask_total_count}")
                print(f"  False elements:          {self.store_mask_false_count}")
                print(f"  Masked percentage:       {store_masked_percentage:.2f}%")
            else:
                print("\nStore Operations:")
                print("  No store operations detected")

            # Detailed per-operation breakdown
            if self.mask_op_stats:
                print("\n" + "─" * 40)
                print("Per-Operation Breakdown:")

                # Aggregate stats by line number
                from collections import defaultdict

                aggregated_stats: defaultdict[
                    tuple[int, str], AggregatedMaskOpStats
                ] = defaultdict(lambda: AggregatedMaskOpStats())

                for stat in self.mask_op_stats:
                    key = (stat.lineno, stat.op_type)
                    aggregated_stats[key].total += stat.total_elements
                    aggregated_stats[key].false += stat.false_elements
                    aggregated_stats[key].filename = stat.filename
                    aggregated_stats[key].code_line = stat.code_line
                    aggregated_stats[key].op_type = stat.op_type

                # Sort by false elements (descending)
                sorted_stats = sorted(
                    [(k, v) for k, v in aggregated_stats.items()],
                    key=lambda x: x[1].false,
                    reverse=True,
                )

                # Display top contributors
                print("\nTop 5 Operations by False Elements:")
                print("─" * 40)

                import os

                for i, ((lineno, op_type), stats) in enumerate(sorted_stats[:5], 1):
                    percentage = (
                        (stats.false / stats.total * 100) if stats.total > 0 else 0
                    )
                    filename_short = os.path.basename(stats.filename)

                    print(f"\n#{i}. {op_type.upper()} at {filename_short}:{lineno}")
                    print(f"    Total elements: {stats.total:,}")
                    print(f"    False elements: {stats.false:,} ({percentage:.1f}%)")
                    if stats.code_line:
                        # Handle multi-line code
                        code_lines = stats.code_line.split("\n")
                        if len(code_lines) == 1:
                            print(f"    Code: {code_lines[0]}")
                        else:
                            print("    Code:")
                            for code_line in code_lines:
                                print(f"        {code_line}")

                # Summary table of all operations
                if len(sorted_stats) > 5:
                    print("\n" + "─" * 40)
                    print("All Operations Summary (sorted by false elements):")
                    print(
                        "\n{:<10} {:<40} {:>15} {:>15} {:>8}".format(
                            "Type", "Location", "Total Elems", "False Elems", "False%"
                        )
                    )
                    print("─" * 100)

                    for (lineno, op_type), stats in sorted_stats:
                        percentage = (
                            (stats.false / stats.total * 100) if stats.total > 0 else 0
                        )
                        filename_short = os.path.basename(stats.filename)
                        location = f"{filename_short}:{lineno}"
                        print(
                            "{:<10} {:<40} {:>15,} {:>15,} {:>7.1f}%".format(
                                op_type,
                                location[:40],
                                stats.total,
                                stats.false,
                                percentage,
                            )
                        )

            print("\n" + "=" * 60 + "\n")

        if not self.disable_buffer_load_check:
            print("\n" + "=" * 60)
            print(
                "-" * 10
                + " "
                + "Profiler: Buffer Load Issue Detection"
                + " "
                + "-" * 11
            )
            print("=" * 60)
            if self.potential_buffer_load_issue_found:
                print("\n>>>>>> Warning: Potential Buffer Load Issue Detected! <<<<<<")
                print(
                    "\nSome memory access offsets are within 32-bit range, "
                    "\nbut Buffer Load optimization was NOT used in the kernel."
                )
                print(
                    "\nThis may lead to suboptimal performance on AMD GPUs. "
                    "\nConsider enabling Buffer Load optimization."
                )
            else:
                print("No Buffer Load Issues Detected.")
                print(
                    "All memory access offsets are within 32-bit range, "
                    "and Buffer Load optimization was used appropriately."
                )
            print("=" * 60 + "\n")

        print("=" * 60, "Profiler Issues Summary Ends", "=" * 55)
        print("\n\n\n")

        return [self.load_bytes, self.store_bytes]
