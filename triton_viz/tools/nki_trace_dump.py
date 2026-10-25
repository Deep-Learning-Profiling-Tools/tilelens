"""Serialize NKI Triton-Viz trace records into a performance-model friendly JSONL.

This module intentionally stays lightweight: it consumes the normalized records that
Triton-Viz already records via the ``Tracer`` client and emits one JSON object per
operation.  The output is the first, stable seam for a future tile-level performance
simulator: later cost models can read this stream and map each operation to an
engine timeline event.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from triton_viz.core.data import (
    BinaryOp,
    Dot,
    Grid,
    Load,
    NkiCompute,
    ReduceSum,
    Store,
    TensorTranspose,
    Transfer,
)
from triton_viz.tools.nki_region_ir import build_region_ir

MAX_EXACT_BYTE_RANGES = 1024


def _jsonable(value: Any) -> Any:
    """Convert NumPy/dataclass values into JSON-serializable primitives."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if is_dataclass(value):
        return _jsonable(asdict(value))
    return value


def _shape(value: Any) -> list[int]:
    return [int(v) for v in tuple(value)]


def _dma_geometry(
    shape: list[int],
    nbytes: int,
    partition_axis: int | None = None,
    masks: Any | None = None,
) -> dict[str, int]:
    """Expose the SBUF partition geometry needed by the DMA cost model."""
    if len(shape) < 2:
        return {}
    axis = 0 if partition_axis is None else int(partition_axis)
    if not 0 <= axis < len(shape) or shape[axis] <= 0:
        return {}
    partitions = int(shape[axis])
    if masks is not None:
        mask = np.asarray(masks, dtype=bool)
        if list(mask.shape) == shape:
            reduce_axes = tuple(index for index in range(mask.ndim) if index != axis)
            active_partitions = np.any(mask, axis=reduce_axes) if reduce_axes else mask
            partitions = int(np.count_nonzero(active_partitions))
    if partitions <= 0:
        return {
            "partition_count": 0,
            "free_bytes_per_partition": 0,
            "partition_axis": axis,
        }
    return {
        "partition_count": partitions,
        "free_bytes_per_partition": int(nbytes // partitions),
        "partition_axis": axis,
    }


def _dot_flops(record: Dot) -> int | None:
    """Estimate FMA FLOPs for a normalized matmul record when shapes are 2-D."""
    if len(record.input_shape) != 2 or len(record.other_shape) != 2:
        return None
    m, k = (int(v) for v in record.input_shape)
    k_other, n = (int(v) for v in record.other_shape)
    if k != k_other:
        return None
    return 2 * m * n * k


def _transfer_engine(mem_src: str, mem_dst: str, shape: list[int]) -> str:
    """Return the likely NKI engine class for a transfer record."""
    memories = {mem_src.lower(), mem_dst.lower()}
    if "hbm" in memories:
        return "dma"
    if memories <= {"sbuf", "psum"}:
        # Fragmented on-chip transfers (e.g., free dimension <= 4) are typically
        # lowered to Static DMA by the compiler instead of VectorE, especially
        # during SBUF scatter/gather operations like transpose2d.
        free_dim = shape[-1] if len(shape) >= 2 else shape[0] if shape else 1
        if free_dim <= 4:
            return "static_dma"
        return "vector"
    return "unknown"


def _first_offset(offsets: Any) -> int | None:
    """Return the first logical element's byte offset without serializing a tile."""
    array = np.asarray(offsets)
    if not array.size:
        return None
    return int(array.reshape(-1)[0])


def _active_offsets(offsets: Any, masks: Any | None = None) -> np.ndarray:
    """Return flattened active byte offsets, excluding masked sentinel lanes."""
    array = np.asarray(offsets)
    if not array.size:
        return np.asarray([], dtype=np.int64)
    if masks is None:
        return array.reshape(-1)
    mask = np.asarray(masks, dtype=bool)
    if mask.shape != array.shape:
        return np.asarray([], dtype=np.int64)
    return array[mask].reshape(-1)


def _item_bytes(
    offsets: Any, nbytes: int, masks: Any | None = None, item_bytes: int | None = None
) -> int:
    if item_bytes is not None:
        return max(1, int(item_bytes))
    active = _active_offsets(offsets, masks)
    return max(1, int(nbytes // max(1, active.size)))


def _byte_ranges(
    offsets: Any,
    nbytes: int,
    masks: Any | None = None,
    item_bytes: int | None = None,
) -> list[list[int]]:
    """Coalesce active element offsets into exact half-open byte segments."""
    active = _active_offsets(offsets, masks)
    if not active.size:
        return []
    width = _item_bytes(offsets, nbytes, masks, item_bytes)
    ordered = np.unique(active.astype(np.int64, copy=False))
    if ordered.size == 1:
        value = int(ordered[0])
        return [[value, value + width]]
    breaks = np.flatnonzero(np.diff(ordered) > width)
    starts = np.concatenate((np.asarray([0]), breaks + 1))
    ends = np.concatenate((breaks, np.asarray([ordered.size - 1])))
    segment_count = int(starts.size)
    if segment_count > MAX_EXACT_BYTE_RANGES:
        # A fully interleaved tile can contain hundreds of thousands of
        # one-element segments. Emitting all of them makes JSONL enormous and
        # turns interval-history updates quadratic. Keep exact segments for
        # normal/tail accesses and use the conservative bounding span for these
        # very large affine scatters; access geometry still records stride,
        # density, and active count for the DMA calibration path.
        return []
    return [
        [int(ordered[start]), int(ordered[end]) + width]
        for start, end in zip(starts, ends)
    ]


def _byte_span(
    offsets: Any,
    nbytes: int,
    masks: Any | None = None,
    item_bytes: int | None = None,
) -> list[int] | None:
    """Return the bounding range of active lanes, ignoring masked sentinels."""
    active = _active_offsets(offsets, masks)
    if not active.size:
        return None
    flat = active.reshape(-1)
    lo = int(flat.min())
    hi = int(flat.max())
    return [lo, hi + _item_bytes(offsets, nbytes, masks, item_bytes)]


def _offset_geometry(offsets: Any, masks: Any, nbytes: int) -> dict[str, Any]:
    """Describe affine/strided HBM access geometry without dumping offsets.

    Offsets are byte offsets.  Masked padding must be excluded: Tilebench often
    allocates a 16K free tile while only the first 128--2048 columns are active.
    """
    array = np.asarray(offsets)
    mask = np.asarray(masks, dtype=bool)
    if not array.size or array.shape != mask.shape:
        return {"dma_pattern": "unknown"}
    active = array[mask]
    if not active.size:
        return {"dma_pattern": "empty", "active_access_count": 0}
    item_bytes = max(1, int(nbytes // active.size))

    def axis_stride(axis: int) -> int | None:
        if array.ndim <= axis or array.shape[axis] < 2:
            return None
        left = [slice(None)] * array.ndim
        right = [slice(None)] * array.ndim
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        valid = mask[tuple(left)] & mask[tuple(right)]
        diffs = (array[tuple(right)] - array[tuple(left)])[valid]
        if not diffs.size or np.any(diffs != diffs.reshape(-1)[0]):
            return None
        return int(diffs.reshape(-1)[0])

    partition_stride = axis_stride(0)
    free_stride = axis_stride(array.ndim - 1)
    free_stride_items = (
        free_stride // item_bytes
        if free_stride is not None and free_stride % item_bytes == 0
        else None
    )
    if free_stride_items == 1:
        pattern = "contiguous"
    elif free_stride_items is not None and free_stride_items > 1:
        pattern = "strided"
    elif free_stride_items is not None and free_stride_items < 0:
        pattern = "reverse"
    else:
        pattern = "irregular"
    span = int(active.max()) - int(active.min()) + item_bytes
    return {
        "dma_pattern": pattern,
        "active_access_count": int(active.size),
        "item_bytes": item_bytes,
        "free_stride_bytes": free_stride,
        "free_stride_items": free_stride_items,
        "partition_stride_bytes": partition_stride,
        "access_span_bytes": span,
        "access_density": float(nbytes / span) if span > 0 else 0.0,
    }


def _fusion_op_name(event: dict[str, Any]) -> str:
    """Return a stable lowering-pattern token for a source compute event."""
    op = str(event.get("op") or "unknown")
    if op == "compute":
        return str(event.get("api_op") or "compute").lower()
    if op == "binary":
        return str(event.get("binary_op") or "binary").lower()
    return op.lower()


def _annotate_fusion_signature(events: list[dict[str, Any]]) -> None:
    """Annotate contiguous source-compute regions with a lowering signature.

    A signature is deliberately a source-level *lookup key*, not a claim about
    compiler fusion.  Explorer calibration later maps the key to the actual
    per-engine lowering.  Grid and memory events bound regions, so operations
    from different program instances or separated by an explicit transfer are
    never grouped together.
    """
    compute_ops = {"binary", "compute", "dot", "reduce_sum", "tensor_transpose"}
    index = 0
    group_id = 0
    while index < len(events):
        if events[index].get("op") not in compute_ops:
            index += 1
            continue

        grid_idx = events[index].get("grid_idx")
        end = index + 1
        while (
            end < len(events)
            and events[end].get("op") in compute_ops
            and events[end].get("grid_idx") == grid_idx
        ):
            end += 1

        group = events[index:end]
        tokens = [_fusion_op_name(event) for event in group]
        signature = "_".join(tokens)
        reduction_tokens = {"reduce_sum", "max", "min", "mean"}
        pattern = (
            "reduce_broadcast_chain"
            if reduction_tokens.intersection(tokens)
            else "elementwise_chain"
        )
        identity = {
            "kernel": str(group[0].get("kernel_name") or "kernel"),
            "grid_idx": grid_idx,
            "ordinal": group_id,
            "signature": signature,
            "shapes": [
                event.get("output_shape") or event.get("input_shape") for event in group
            ],
            "dtypes": [
                event.get("output_dtype") or event.get("dtype") for event in group
            ],
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:12]
        source_region_id = f"{identity['kernel']}:r{group_id}:{digest}"
        left = index - 1
        while left >= 0 and events[left].get("op") not in compute_ops:
            left -= 1
        right = end
        while right < len(events) and events[right].get("op") not in compute_ops:
            right += 1
        # Only the adjacent memory window belongs to this region. Passing the
        # whole kernel confuses a later 1-partition weight load with an earlier
        # reduction region's layout.
        region_ir = build_region_ir(group, events[left + 1 : index] + events[end:right])
        for group_index, event in enumerate(group):
            event.update(
                {
                    "fusion_signature": signature,
                    "fusion_pattern": pattern,
                    "fusion_group": group_id,
                    "fusion_group_index": group_index,
                    "fusion_group_size": len(group),
                    "source_region_id": source_region_id,
                    "region_ir": region_ir,
                    "region_ir_key": region_ir["structural_key"],
                }
            )
        group_id += 1
        index = end

    # Compiler lowering can move setup work across an explicit transfer. Keep
    # the local grammar composable, but expose immediate region context so a
    # measured two-pass control can model that interaction without an
    # operator/full-signature key.
    leaders = [
        event
        for event in events
        if event.get("fusion_group_index") == 0 and event.get("region_ir")
    ]
    from triton_viz.tools.nki_region_ir import (
        region_ir_structural_key,
        structural_family,
    )

    for ordinal in range(1, len(leaders)):
        if leaders[ordinal - 1]["region_ir"].get("has_mask_or_tail"):
            leaders[ordinal]["region_ir"]["has_mask_or_tail"] = True
    bases = [structural_family(event["region_ir"]) for event in leaders]
    for ordinal, leader in enumerate(leaders):
        ir = leader["region_ir"]
        if ordinal:
            ir["previous_family"] = bases[ordinal - 1]
        if ordinal + 1 < len(bases):
            ir["next_family"] = bases[ordinal + 1]
        ir["structural_key"] = region_ir_structural_key(ir)
        group = leader["fusion_group"]
        for event in events:
            if event.get("fusion_group") == group:
                event["region_ir"] = ir
                event["region_ir_key"] = ir["structural_key"]


def _annotate_static_dma_groups(events: list[dict[str, Any]]) -> None:
    """Describe consecutive scalar SBUF scatter groups for calibrated costing.

    ``tensor_copy``-based transposes appear in the trace as one ``[p, 1]``
    transfer per scalar free-dimension element. The compiler batches these into
    Static DMA packets, so their cost is not the sum of independent VectorE
    operations. Keep the event stream intact, but attach one compact group
    geometry to every member so the cost model can charge the measured group
    latency exactly once across the group.
    """
    index = 0
    group_id = 0
    while index < len(events):
        event = events[index]
        if event.get("engine") != "static_dma":
            index += 1
            continue
        end = index + 1
        key = (
            event.get("src_ptr"),
            event.get("dst_ptr"),
            event.get("partition_count"),
            event.get("free_bytes_per_partition"),
        )
        while end < len(events):
            candidate = events[end]
            if candidate.get("engine") != "static_dma":
                break
            candidate_key = (
                candidate.get("src_ptr"),
                candidate.get("dst_ptr"),
                candidate.get("partition_count"),
                candidate.get("free_bytes_per_partition"),
            )
            if candidate_key != key:
                break
            end += 1

        group = events[index:end]
        copies = len(group)
        itemsize = int(event.get("free_bytes_per_partition") or 0)
        dst_offsets = [member.get("dst_offset_first") for member in group]
        x = y = 0
        if (
            itemsize > 0
            and copies > 0
            and all(offset is not None for offset in dst_offsets)
        ):
            normalized = [int(offset) // itemsize for offset in dst_offsets]
            if copies == 1:
                x = y = 1
            else:
                initial_stride = normalized[1] - normalized[0]
                if initial_stride > 0:
                    y = next(
                        (
                            position
                            for position in range(1, copies)
                            if normalized[position] - normalized[position - 1]
                            != initial_stride
                        ),
                        copies,
                    )
                    if copies % y == 0:
                        x = copies // y
        for group_index, member in enumerate(group):
            member.update(
                {
                    "static_dma_group": group_id,
                    "static_dma_group_index": group_index,
                    "static_dma_group_copies": copies,
                    "static_dma_group_x": x,
                    "static_dma_group_y": y,
                }
            )
        group_id += 1
        index = end


def record_to_event(
    record: Any, sequence: int, grid_idx: tuple[int, ...] | None
) -> dict[str, Any]:
    """Convert one Triton-Viz record into a compact JSON event dictionary."""
    base: dict[str, Any] = {
        "seq": int(sequence),
        "grid_idx": _jsonable(grid_idx),
        "record_type": type(record).__name__,
    }

    if isinstance(record, Grid):
        return {**base, "op": "grid", "grid_idx": _jsonable(record.idx)}

    if isinstance(record, Transfer):
        src_bytes = int(getattr(record, "src_bytes", record.bytes))
        dst_bytes = int(getattr(record, "dst_bytes", record.bytes))
        src_shape = _shape(record.src_offsets.shape)
        dst_shape = _shape(record.dst_offsets.shape)
        if record.mem_dst.lower() == "sbuf":
            shape = dst_shape
            partition_axis = record.dst_partition_axis
        else:
            shape = src_shape
            partition_axis = record.src_partition_axis
        dma_pattern = getattr(record, "dma_pattern", "copy")
        if (
            dma_pattern == "copy"
            and len(src_shape) == 2
            and src_shape == list(reversed(dst_shape))
            and src_shape != dst_shape
        ):
            dma_pattern = "transpose"
        event = {
            **base,
            "op": "transfer",
            "engine": _transfer_engine(record.mem_src, record.mem_dst, shape),
            "mem_src": record.mem_src,
            "mem_dst": record.mem_dst,
            "bytes": int(record.bytes),
            "src_ptr": int(record.src_ptr),
            "dst_ptr": int(record.dst_ptr),
            "src_offsets_shape": src_shape,
            "dst_offsets_shape": dst_shape,
            "src_offset_first": _first_offset(record.src_offsets),
            "dst_offset_first": _first_offset(record.dst_offsets),
            "dma_pattern": dma_pattern,
            "src_storage": int(record.src_ptr),
            "dst_storage": int(record.dst_ptr),
            "src_range": _byte_span(record.src_offsets, src_bytes),
            "src_ranges": _byte_ranges(record.src_offsets, src_bytes),
            "dst_range": _byte_span(record.dst_offsets, dst_bytes),
            "dst_ranges": _byte_ranges(record.dst_offsets, dst_bytes),
            **_dma_geometry(shape, int(record.bytes), partition_axis),
        }
        if src_bytes != dst_bytes:
            event.update(src_bytes=src_bytes, dst_bytes=dst_bytes)
        # Access geometry on the HBM side of the transfer. The legacy Load/
        # Store events always carry ``_offset_geometry``; beta2 Transfer
        # records carry concrete source/destination byte offsets instead, so
        # derive the same access features here. This is what the runtime
        # DMA-packet term and the strided-DMA surface consume.
        mem_src = str(record.mem_src or "").lower()
        mem_dst = str(record.mem_dst or "").lower()
        if "hbm" in mem_src:
            offsets = np.asarray(record.src_offsets)
        elif "hbm" in mem_dst:
            offsets = np.asarray(record.dst_offsets)
        else:
            offsets = np.asarray([])
        if offsets.size:
            masks = np.ones(offsets.shape, dtype=bool)
            geometry = _offset_geometry(offsets, masks, src_bytes if "hbm" in mem_src else dst_bytes)
            event.update(geometry)
        event["dma_pattern"] = dma_pattern
        return event

    if isinstance(record, Dot):
        return {
            **base,
            "op": "dot",
            "engine": "tensor",
            "input_shape": _shape(record.input_shape),
            "other_shape": _shape(record.other_shape),
            "output_shape": _shape(record.output_shape),
            "flops": _dot_flops(record),
            "input_ptrs": [int(ptr) for ptr in getattr(record, "input_ptrs", ())],
            "output_ptr": (
                int(record.output_ptr)
                if getattr(record, "output_ptr", None) is not None
                else None
            ),
            "input_storages": [int(value) for value in record.input_storages],
            "input_ranges": [list(value) for value in record.input_ranges],
            "input_versions": [int(value) for value in record.input_versions],
            "output_storage": record.output_storage,
            "output_range": (
                list(record.output_range) if record.output_range is not None else None
            ),
            "output_version": record.output_version,
            "input_dtypes": [str(value) for value in record.input_dtypes],
            "output_dtype": record.output_dtype,
        }

    if isinstance(record, BinaryOp):
        return {
            **base,
            "op": "binary",
            "binary_op": record.op,
            "engine": "vector",
            "input_shape": _shape(record.input_shape),
            "other_shape": _shape(record.other_shape),
            "output_shape": _shape(record.output_shape),
            "elements": int(np.prod(record.output_shape, dtype=np.int64)),
            "input_ptrs": [int(ptr) for ptr in record.input_ptrs],
            "output_ptr": (
                int(record.output_ptr) if record.output_ptr is not None else None
            ),
            "input_storages": [int(value) for value in record.input_storages],
            "input_ranges": [list(value) for value in record.input_ranges],
            "input_versions": [int(value) for value in record.input_versions],
            "output_storage": record.output_storage,
            "output_range": (
                list(record.output_range) if record.output_range is not None else None
            ),
            "output_version": record.output_version,
        }

    if isinstance(record, ReduceSum):
        return {
            **base,
            "op": "reduce_sum",
            "api_op": record.api_op,
            "engine": "vector_or_scalar",
            "input_shape": _shape(record.input_shape),
            "axis": record.index,
            "keep_dims": bool(record.keep_dims),
            "output_shape": _shape(record.output_shape),
            "input_ptrs": [int(ptr) for ptr in getattr(record, "input_ptrs", ())],
            "output_ptr": (
                int(record.output_ptr)
                if getattr(record, "output_ptr", None) is not None
                else None
            ),
            "input_storages": [int(value) for value in record.input_storages],
            "input_ranges": [list(value) for value in record.input_ranges],
            "input_versions": [int(value) for value in record.input_versions],
            "output_storage": record.output_storage,
            "output_range": (
                list(record.output_range) if record.output_range is not None else None
            ),
            "output_version": record.output_version,
            "input_dtypes": [str(value) for value in record.input_dtypes],
            "output_dtype": record.output_dtype,
        }

    if isinstance(record, NkiCompute) and record.attrs.get("memory_initializer"):
        shape = list(record.output_shapes[0]) if record.output_shapes else []
        return {
            **base,
            "op": "memory_fill",
            "api_op": record.api_op,
            "engine": record.engine,
            "input_storages": [],
            "input_ptrs": [],
            "output_ptr": int(record.output_ptrs[0]) if record.output_ptrs else None,
            "output_storage": int(record.output_storages[0]) if record.output_storages else None,
            "output_range": list(record.output_ranges[0]) if record.output_ranges else None,
            "output_version": int(record.output_versions[0]) if record.output_versions else None,
            "output_shape": shape,
            "output_dtype": record.output_dtype,
            "output_memory": record.attrs["output_memory"],
            "output_partition_axis": record.attrs["output_partition_axis"],
            "allocation_storage": record.attrs["allocation_storage"],
            "allocation_shape": record.attrs["allocation_shape"],
            "allocation_partition_axis": record.attrs["allocation_partition_axis"],
            "fill_value": record.attrs["fill_value"],
            "timing_status": "unpriced_initializer",
        }

    if isinstance(record, NkiCompute):
        out_shape = list(record.output_shapes[0]) if record.output_shapes else []
        in_shape = list(record.input_shapes[0]) if record.input_shapes else []
        free_dim = (
            int(out_shape[-1]) if out_shape else (int(in_shape[-1]) if in_shape else 0)
        )
        elements = int(np.prod(out_shape, dtype=np.int64)) if out_shape else 0
        return {
            **base,
            "op": "compute",
            "api_op": record.api_op,
            # ScalarE (activation) vs VectorE, as classified by the interpreter.
            "engine": record.engine,
            "input_shape": in_shape,
            "output_shape": out_shape,
            "free_dim": free_dim,
            "elements": elements,
            "input_ptrs": [int(ptr) for ptr in record.input_ptrs],
            "output_ptr": (int(record.output_ptrs[0]) if record.output_ptrs else None),
            "input_storages": [int(value) for value in record.input_storages],
            "input_ranges": [list(value) for value in record.input_ranges],
            "input_versions": [int(value) for value in record.input_versions],
            "output_storage": (
                int(record.output_storages[0]) if record.output_storages else None
            ),
            "output_range": (
                list(record.output_ranges[0]) if record.output_ranges else None
            ),
            "output_version": (
                int(record.output_versions[0]) if record.output_versions else None
            ),
            "input_dtypes": list(record.input_dtypes),
            "output_dtype": record.output_dtype,
            **({
                "output_ptrs": [int(x) for x in record.output_ptrs],
                "output_shapes": [list(x) for x in record.output_shapes],
                "output_storages": [int(x) for x in record.output_storages],
                "output_ranges": [list(x) for x in record.output_ranges],
                "output_versions": [int(x) for x in record.output_versions],
                "output_dtypes": record.attrs["output_dtypes"],
                "output_roles": record.attrs["output_roles"],
                "reduction_op": record.attrs.get("reduction_op"),
            } if len(record.output_ptrs) > 1 else {}),
            **({"scalar_state": record.attrs["scalar_state"]}
               if "scalar_state" in record.attrs else {}),
            "compute_mask_provided": bool(
                record.attrs.get("compute_mask_provided", False)
            ),
            **({"compute_mask_geometry": record.attrs["compute_mask_geometry"]}
               if "compute_mask_geometry" in record.attrs else {}),
        }

    if isinstance(record, TensorTranspose):
        input_shape = _shape(record.input_shape)
        output_shape = _shape(record.output_shape)
        # A TensorE PF-transpose is an identity matmul: stationary.T @ I.
        # For a source tile (P, F) the identity operand is (P, P), so the
        # equivalent FLOPs are 2 * F * P * P.
        flops = None
        if len(input_shape) == 2:
            par, free = input_shape
            flops = 2 * int(free) * int(par) * int(par)
        return {
            **base,
            "op": "tensor_transpose",
            "engine": "tensor",
            "input_shape": input_shape,
            "other_shape": [input_shape[0], input_shape[0]]
            if len(input_shape) == 2
            else [],
            "output_shape": output_shape,
            "flops": flops,
            "input_ptrs": [int(ptr) for ptr in record.input_ptrs],
            "output_ptr": (
                int(record.output_ptr)
                if record.output_ptr is not None
                else None
            ),
            "input_storages": [int(value) for value in record.input_storages],
            "input_ranges": [list(value) for value in record.input_ranges],
            "input_versions": [int(value) for value in record.input_versions],
            "output_storage": record.output_storage,
            "output_range": (
                list(record.output_range) if record.output_range is not None else None
            ),
            "output_version": record.output_version,
            "input_dtypes": [str(value) for value in record.input_dtypes],
            "output_dtype": record.output_dtype,
        }

    if isinstance(record, Load):
        active = int(np.count_nonzero(record.masks))
        shape = _shape(record.offsets.shape)
        return {
            **base,
            "op": "load",
            "engine": "dma_or_vector_load",
            "mem_src": record.mem_src,
            "mem_dst": record.mem_dst,
            "bytes": int(record.bytes),
            "active_lanes": active,
            "mask_provided": bool(getattr(record, "mask_provided", False)),
            "offsets_shape": shape,
            "src_ptr": int(record.ptr),
            "src_storage": int(record.src_storage or record.ptr),
            "src_range": _byte_span(
                record.offsets, int(record.bytes), record.masks
            ),
            "src_ranges": _byte_ranges(
                record.offsets, int(record.bytes), record.masks
            ),
            "src_version": record.src_version,
            "src_dtype": record.src_dtype,
            "dst_ptr": record.dst_ptr,
            "dst_storage": record.dst_storage,
            "dst_range": (
                list(record.dst_range) if record.dst_range is not None else None
            ),
            "dst_version": record.dst_version,
            **_offset_geometry(record.offsets, record.masks, int(record.bytes)),
            **_dma_geometry(shape, int(record.bytes), masks=record.masks),
        }

    if isinstance(record, Store):
        active = int(np.count_nonzero(record.masks))
        shape = _shape(record.offsets.shape)
        return {
            **base,
            "op": "store",
            "engine": "dma_or_vector_store",
            "mem_src": record.mem_src,
            "mem_dst": record.mem_dst,
            "bytes": int(record.bytes),
            "active_lanes": active,
            "mask_provided": bool(getattr(record, "mask_provided", False)),
            "offsets_shape": shape,
            "dst_ptr": int(record.ptr),
            "dst_storage": int(record.dst_storage or record.ptr),
            "dst_range": _byte_span(
                record.offsets, int(record.bytes), record.masks
            ),
            "dst_ranges": _byte_ranges(
                record.offsets, int(record.bytes), record.masks
            ),
            "dst_version": record.dst_version,
            "src_ptr": record.src_ptr,
            "src_storage": record.src_storage,
            "src_range": (
                list(record.src_range) if record.src_range is not None else None
            ),
            "src_version": record.src_version,
            "src_dtype": record.src_dtype,
            **_offset_geometry(record.offsets, record.masks, int(record.bytes)),
            **_dma_geometry(shape, int(record.bytes), masks=record.masks),
        }

    return {**base, "op": "unknown", "payload": _jsonable(record)}


def _source_site(record: Any) -> str | None:
    """Program point of a record: innermost user frame as ``file:line`` (None when not captured)."""
    frames = getattr(record, "call_path", None) or []
    if not frames:
        return None
    frame = frames[-1]
    filename = getattr(frame, "filename", None)
    lineno = getattr(frame, "lineno", None)
    if filename is None or lineno is None:
        return None
    return f"{Path(str(filename)).name}:{int(lineno)}"


def records_to_events(records: Iterable[Any]) -> list[dict[str, Any]]:
    """Convert a launch's records into ordered performance trace events."""
    events: list[dict[str, Any]] = []
    grid_idx: tuple[int, ...] | None = None
    for sequence, record in enumerate(records):
        if isinstance(record, Grid):
            grid_idx = tuple(record.idx)
        event = record_to_event(record, sequence, grid_idx)
        if getattr(record, "input_runs", None) is not None:
            def _lists(runs):
                return [None if r is None or r[1] is None else {"storage": r[0], "runs": [list(x) for x in r[1]]}
                        for r in runs]
            event["input_range_lists"] = _lists(record.input_runs)
            event["output_range_lists"] = _lists(record.output_runs)
        if getattr(record, "input_allocations", None) is not None:
            event["input_allocations"] = list(record.input_allocations)
            event["output_allocations"] = list(record.output_allocations)
        site = _source_site(record)
        if site is not None:
            event["source_site"] = site
        events.append(event)
    _annotate_static_dma_groups(events)
    _annotate_fusion_signature(events)
    return events


def write_jsonl(records: Iterable[Any], path: str | Path) -> list[dict[str, Any]]:
    """Write records to ``path`` as JSONL and return the emitted event list."""
    events = records_to_events(records)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(
                json.dumps(_jsonable(event), sort_keys=True, ensure_ascii=False) + "\n"
            )
    return events


def summarize_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return small aggregate counters useful in tests and quick CLI checks."""
    summary: dict[str, Any] = {
        "num_events": 0,
        "op_counts": {},
        "bytes_by_edge": {},
        "flops": 0,
    }
    for event in events:
        summary["num_events"] += 1
        op = event.get("op", "unknown")
        summary["op_counts"][op] = summary["op_counts"].get(op, 0) + 1
        if "bytes" in event:
            key = f"{event.get('mem_src', '?')}->{event.get('mem_dst', '?')}"
            summary["bytes_by_edge"][key] = summary["bytes_by_edge"].get(key, 0) + int(
                event["bytes"]
            )
        if event.get("flops") is not None:
            summary["flops"] += int(event["flops"])
    return summary
