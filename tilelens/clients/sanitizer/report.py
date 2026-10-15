from typing import TYPE_CHECKING

import numpy as np

from ...core.config import config as cfg
from ...core.data import Load, Store
from ...utils.traceback_utils import read_source_segment
from ..utils import (
    check_inner_stride_equal_to_one,
    check_storage_contiguous,
    get_physical_addr_from_tensor_slice,
    get_physical_addr_per_element,
)
from .data import (
    OutOfBoundsRecord,
    OutOfBoundsRecordBruteForce,
    OutOfBoundsRecordZ3,
    UndecidedAccessRecordZ3,
)


def _classify_layout_and_segments(tensor):
    """Return ``(layout_label, segments)`` mirroring SymbolicClient.arg_callback.

    Segments are ``(start, end)`` byte ranges with both endpoints inclusive.
    The layout label is one of:
      - ``contiguous`` — whole storage is one contiguous segment
      - ``dim-N-contiguous`` — dim N has stride 1; other dims index into
        independent contiguous runs
      - ``per-element`` — no dim has stride 1; each logical element is its
        own segment (precise view semantics)
      - ``empty`` — tensor has zero elements
    """
    if tensor.numel() == 0:
        return "empty", []
    if tensor.is_contiguous() or check_storage_contiguous(tensor):
        start = tensor.data_ptr()
        end = start + tensor.numel() * tensor.element_size() - 1
        return "contiguous", [(start, end)]
    if check_inner_stride_equal_to_one(tensor):
        inner_dim = min(
            (d for d in range(tensor.dim()) if tensor.stride(d) != 0),
            key=lambda d: tensor.stride(d),
        )
        return (
            f"dim-{inner_dim}-contiguous",
            get_physical_addr_from_tensor_slice(tensor),
        )
    return "per-element", get_physical_addr_per_element(tensor)


if TYPE_CHECKING:
    from ..symbolic_engine import SymbolicExpr


def print_oob_record(oob_record: OutOfBoundsRecord, max_display=10):
    """
    Print detailed logs for a given OOB record.

    Parameters
    ----------
    oob_record : OutOfBoundsRecord
        The record containing information about out-of-bounds accesses.
    max_display : int
        Maximum number of invalid accesses to display in detail.
    """
    if issubclass(oob_record.op_type, Store):
        op_type = "Store"
    elif issubclass(oob_record.op_type, Load):
        op_type = "Load"
    else:
        assert False, "Not supported op type: " + str(oob_record.op_type)

    # Read the tensor from the record
    tensor = oob_record.tensor

    # Basic info about the OOB event
    print("============================================================")
    print("                 Out-Of-Bounds Access Detected              ")
    print("============================================================")
    print(f"Operation: {op_type}")
    tensor_name = getattr(oob_record, "tensor_name", None)
    if tensor_name:
        print(f"Tensor Arg: {tensor_name}")
    print(
        f"Tensor Info: dtype={tensor.dtype}, shape={tensor.shape}, device={tensor.device}"
    )
    print(f"Tensor base memory address: {tensor.data_ptr()}")
    print(
        "Valid Access Range: [0, %d)" % (np.prod(tensor.shape) * tensor.element_size())
    )
    for traceback_info in oob_record.user_code_tracebacks:
        print(
            f"File: {traceback_info.filename}, Line: {traceback_info.lineno}, in {traceback_info.func_name}"
        )
        print(f"  Code: {traceback_info.line_of_code}")
    print("------------------------------------------------------------")

    if isinstance(oob_record, OutOfBoundsRecordBruteForce):
        # Convert memoryviews to NumPy arrays
        offsets_arr = np.array(oob_record.offsets)
        invalid_access_masks_arr = np.array(oob_record.invalid_access_masks)

        # Determine all invalid indices
        invalid_indices = np.where(invalid_access_masks_arr.flatten())[0]
        assert len(invalid_indices) != 0, "No invalid accesses found in this record."

        # Print OOB details
        print(f"Total invalid accesses: {len(invalid_indices)}")
        invalid_offsets = offsets_arr.flatten()[invalid_indices]
        print("Invalid offsets:")
        print(invalid_offsets)

    elif isinstance(oob_record, OutOfBoundsRecordZ3):
        # Read the violation index and constraints
        violation_address = oob_record.violation_address
        constraints = oob_record.constraints

        # Print OOB details
        print(f"Invalid access detected at address: {violation_address}")
        if constraints is not None:
            print("Constraints:")
            print(constraints)

    else:
        raise NotImplementedError(
            "Invalid OutOfBoundsRecord type: " + str(type(oob_record))
        )

    print("============================================================")
    print("            End of Out-Of-Bounds Record Details             ")
    print("============================================================")


def print_undecided_record(record: UndecidedAccessRecordZ3) -> None:
    """Print a warning for an access the sanitizer could not check."""
    op_type = "Store" if issubclass(record.op_type, Store) else "Load"
    print(
        f"\n\033[1m\033[93m⚠️  UNCHECKED MEMORY ACCESS\033[0m: Z3 could not decide "
        f"whether this {op_type} stays in bounds ({record.reason})"
    )
    if record.tensor_name:
        print(f"  Tensor Arg: {record.tensor_name}")
    for traceback_info in record.user_code_tracebacks:
        print(
            f"  File: {traceback_info.filename}, Line: {traceback_info.lineno}, "
            f"in {traceback_info.func_name}"
        )
        print(f"    Code: {traceback_info.line_of_code}")


def print_oob_record_pdb_style(
    oob_record: OutOfBoundsRecord, symbolic_expr: "SymbolicExpr | None" = None
):
    """
    Print a comprehensive diagnostic report for OOB errors in PDB-style format.

    Parameters
    ----------
    oob_record : OutOfBoundsRecord
        The record containing information about out-of-bounds accesses.
    symbolic_expr : SymbolicExpr | None
        The symbolic expression tree that led to the OOB access.
    """
    from pathlib import Path

    # Determine operation type
    if issubclass(oob_record.op_type, Store):
        op_color = "\033[91m"  # Red for store
    elif issubclass(oob_record.op_type, Load):
        op_color = "\033[93m"  # Yellow for load
    else:
        op_color = "\033[0m"

    reset_color = "\033[0m"
    bold = "\033[1m"
    cyan = "\033[96m"
    green = "\033[92m"
    magenta = "\033[95m"

    # ===================== Header =====================
    print(f"\n{bold}{op_color}🚨 ILLEGAL MEMORY ACCESS DETECTED 🚨{reset_color}")

    # ===================== PDB-Style Code Context =====================
    if oob_record.user_code_tracebacks:
        print(f"{bold}{cyan}━━━ Code Context ━━━{reset_color}")

        tb_info = oob_record.user_code_tracebacks[-1]  # Innermost user frame
        seg = read_source_segment(tb_info.filename, tb_info.lineno, context=3)
        if seg is not None:
            print(f"  {magenta}File:{reset_color} {tb_info.filename}")
            print(f"  {magenta}Function:{reset_color} {tb_info.func_name}")
            print(f"  {magenta}Line {tb_info.lineno}:{reset_color}")

            for line in seg["lines"]:
                line_num = line["no"]
                line_content = line["text"]
                if line_num == tb_info.lineno:
                    print(f"{op_color}→ {line_num:4d} │ {line_content}{reset_color}")
                else:
                    print(f"  {line_num:4d} │ {line_content}")
        else:
            # Fallback if we can't read the file
            print(f"  {magenta}File:{reset_color} {tb_info.filename}:{tb_info.lineno}")
            print(f"  {magenta}Function:{reset_color} {tb_info.func_name}")
            print(f"  {magenta}Code:{reset_color} {tb_info.line_of_code}")

    # ===================== Tensor Information =====================
    print(f"{bold}{cyan}━━━ Tensor Information ━━━{reset_color}")
    tensor = oob_record.tensor
    tensor_name = getattr(oob_record, "tensor_name", None)

    # Display two items per line to save vertical space
    # Calculate column widths for alignment
    col1_width = 12  # Width for first label
    col2_width = 25  # Width for first value
    col3_width = 12  # Width for second label

    if tensor_name:
        print(f"  {green}{'arg:':<{col1_width}}{reset_color} {tensor_name}")
    print(
        f"  {green}{'dtype:':<{col1_width}}{reset_color} {str(tensor.dtype):<{col2_width}} {green}{'shape:':<{col3_width}}{reset_color} {tensor.shape}"
    )
    print(
        f"  {green}{'strides:':<{col1_width}}{reset_color} {str(tensor.stride()):<{col2_width}} {green}{'device:':<{col3_width}}{reset_color} {tensor.device}"
    )
    print(
        f"  {green}{'contiguous:':<{col1_width}}{reset_color} {str(tensor.is_contiguous()):<{col2_width}} {green}{'base_ptr:':<{col3_width}}{reset_color} 0x{tensor.data_ptr():016x}"
    )

    total_bytes = int(np.prod(tensor.shape)) * tensor.element_size()
    layout, segments = _classify_layout_and_segments(tensor)

    def _seg_str(seg):
        s, e = seg
        return f"[0x{s:016x}, 0x{e + 1:016x})"

    if len(segments) == 1:
        # Single-segment: keep the original one-line format.
        size_str = f"{total_bytes} bytes"
        range_str = _seg_str(segments[0])
        print(
            f"  {green}{'size:':<{col1_width}}{reset_color} {size_str:<{col2_width}} {green}{'valid_range:':<{col3_width}}{reset_color} {range_str}"
        )
    elif len(segments) == 0:
        print(f"  {green}{'size:':<{col1_width}}{reset_color} 0 bytes (empty tensor)")
        print(f"  {green}{'valid_range:':<{col1_width}}{reset_color} (no segments)")
    else:
        span_start = segments[0][0]
        span_end = segments[-1][1] + 1
        size_str = f"{total_bytes} bytes (layout: {layout})"
        print(f"  {green}{'size:':<{col1_width}}{reset_color} {size_str}")
        range_header = (
            f"{len(segments)} segments spanning "
            f"[0x{span_start:016x}, 0x{span_end:016x})"
        )
        print(f"  {green}{'valid_range:':<{col1_width}}{reset_color} {range_header}")
        max_display = max(2, cfg.sanitizer_report_max_segments)
        if len(segments) <= max_display:
            for seg in segments:
                print(f"    {_seg_str(seg)}")
        else:
            head = tail = max_display // 2
            for seg in segments[:head]:
                print(f"    {_seg_str(seg)}")
            print(f"    ... ({len(segments) - head - tail} more)")
            for seg in segments[-tail:]:
                print(f"    {_seg_str(seg)}")

    # ===================== Call Stack =====================
    print(f"{bold}{cyan}━━━ Call Stack ━━━{reset_color}")

    if oob_record.user_code_tracebacks:
        for i, tb_info in enumerate(oob_record.user_code_tracebacks):
            frame_num = len(oob_record.user_code_tracebacks) - i
            file_name = Path(tb_info.filename).name
            print(f"  #{frame_num} {tb_info.func_name} at {file_name}:{tb_info.lineno}")
            if tb_info.line_of_code:
                print(f"     └─ {tb_info.line_of_code.strip()}")
    else:
        print("  (No traceback information available)")

    # ===================== Violation Details =====================
    print(f"{bold}{cyan}━━━ Violation Details ━━━{reset_color}")

    if isinstance(oob_record, OutOfBoundsRecordBruteForce):
        offsets_arr = np.array(oob_record.offsets)
        invalid_access_masks_arr = np.array(oob_record.invalid_access_masks)
        invalid_indices = np.where(invalid_access_masks_arr.flatten())[0]

        if len(invalid_indices) > 0:
            print(f"  {green}Total violations:{reset_color} {len(invalid_indices)}")
            invalid_offsets = offsets_arr.flatten()[invalid_indices]

            # Show first few invalid offsets
            display_count = min(10, len(invalid_offsets))
            print(f"  {green}Invalid offsets (first {display_count}):{reset_color}")
            for offset in invalid_offsets[:display_count]:
                print(f"    • 0x{offset:016x} (offset: {offset})")

            if len(invalid_offsets) > display_count:
                print(f"    ... and {len(invalid_offsets) - display_count} more")

    elif isinstance(oob_record, OutOfBoundsRecordZ3):
        if hasattr(oob_record, "violation_address"):
            print(
                f"  {green}Violation address:{reset_color} 0x{oob_record.violation_address:016x}"
            )

        constraints = getattr(oob_record, "constraints", None)
        if constraints is not None:
            print(f"  {green}SMT constraints:{reset_color}")
            print(f"    {constraints}")

    # ===================== Symbolic Expression Tree =====================
    if symbolic_expr is not None:
        print(f"{bold}{cyan}━━━ Symbolic Expression Tree ━━━{reset_color}")

        if hasattr(symbolic_expr, "to_tree_str"):
            tree_str = symbolic_expr.to_tree_str()
            # Indent the tree for better display
            for line in tree_str.split("\n"):
                if line:
                    print(f"  {line}")
        else:
            print(f"  {symbolic_expr}")

    print(f"{bold}End of IMA Diagnostic Report{reset_color}\n")
