"""Opt-in checks of the declared partition extent at NKI ISA operand use.

This is a partial ISA contract checker, not a compiler or allocation oracle. Large
free dimensions and HBM tensors are unrestricted by this particular check.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import inspect

_ENABLED: ContextVar[bool] = ContextVar("nki_partition_contract_checks", default=False)


class ISAContractError(ValueError):
    """A witnessed violation with machine-readable operand geometry."""

    def __init__(self, finding: dict):
        self.finding = finding
        if finding["kind"] == "matmul_partition_mismatch":
            message = (
                f"nc_matmul partition mismatch: stationary={finding['stationary_extent']} "
                f"!= moving={finding['moving_extent']}; "
                f"stationary shape={finding['stationary_shape']}, moving shape={finding['moving_shape']}"
            )
        else:
            message = (
                f"{finding['operation']} {finding['operand']} partition axis "
                f"{finding['partition_axis']} has extent {finding['partition_extent']}, "
                f"exceeding {finding['limit']}; shape={finding['shape']}"
            )
        super().__init__(message)


def checks_enabled() -> bool:
    return _ENABLED.get()


@contextmanager
def partition_contract_checks(enabled: bool = True):
    """Enable checks within a launch; nested and exceptional exits restore state."""
    token = _ENABLED.set(enabled)
    try:
        yield
    finally:
        _ENABLED.reset(token)



def partition_preserving_view(array, keys, partition_axis):
    """Basic SDK view semantics: preserve an indexed on-chip partition dimension.

    Convert integer selections to slices before squeezing free axes so even scalar
    HBM selections remain shared-storage arrays. None/fancy indexing is unvalidated.
    """
    import numpy as np

    keys = keys if isinstance(keys, tuple) else (keys,)
    if sum(k is Ellipsis for k in keys) > 1:
        raise IndexError("at most one ellipsis is supported")
    if any(k is Ellipsis for k in keys):
        index = next(i for i, k in enumerate(keys) if k is Ellipsis)
        keys = keys[:index] + (slice(None),) * (array.ndim - len(keys) + 1) + keys[index + 1:]
    if len(keys) > array.ndim:
        raise IndexError("too many indices")
    keys = keys + (slice(None),) * (array.ndim - len(keys))
    converted, dropped = [], []
    for axis, key in enumerate(keys):
        if isinstance(key, int):
            index = key + array.shape[axis] if key < 0 else key
            if not 0 <= index < array.shape[axis]:
                raise IndexError(f"index {key} out of bounds for axis {axis}")
            converted.append(slice(index, index + 1))
            if axis != partition_axis:
                dropped.append(axis)
        elif isinstance(key, slice):
            converted.append(key)
        else:
            raise NotImplementedError("partition view checks support basic integer/slice indexing only")
    value = array[tuple(converted)]
    if dropped:
        value = np.squeeze(value, axis=tuple(dropped))
    axis = partition_axis
    if axis is not None:
        axis -= sum(i < axis for i in dropped)
    minimum_rank = 2 if axis is not None else 1
    while value.ndim < minimum_rank:
        value = np.expand_dims(value, -1)
    return value, axis

def checked_operation(operation: str, function):
    """Check actual on-chip operands, including the explicit partition-axis marker."""
    signature = inspect.signature(function)

    @wraps(function)
    def checked(*args, **kwargs):
        if checks_enabled():
            from triton_viz.core.simulation.nki_beta2 import NDArray

            operands = signature.bind(*args, **kwargs).arguments
            for name, tile in operands.items():
                if not isinstance(tile, NDArray) or tile.buffer not in ("sbuf", "psum"):
                    continue
                # Bare nl.ndarray currently records None for an unmarked shape.
                # The ISA default is the leading axis; explicit par_dim survives
                # in partition_axis and must take precedence over this default.
                axis = tile.partition_axis if tile.partition_axis is not None else 0
                if not tile.shape:
                    continue
                extent = tile.shape[axis]
                if extent > 128:
                    raise ISAContractError(
                        {
                            "kind": "partition_extent_limit",
                            "operation": operation,
                            "operand": name,
                            "buffer": tile.buffer,
                            "shape": list(tile.shape),
                            "partition_axis": axis,
                            "partition_extent": extent,
                            "limit": 128,
                            "confidence": "witnessed_isa_constraint",
                            "guidance": (
                                "Keep the on-chip partition extent <=128. Increase contiguous "
                                "free-axis bytes or use partition-sized slices; do not place "
                                "extra rows on the partition axis. Check slicing and matmul "
                                "contraction semantics after changing the layout."
                            ),
                        }
                    )
            if operation == "nc_matmul":
                stationary, moving = operands.get("stationary"), operands.get("moving")
                if isinstance(stationary, NDArray) and isinstance(moving, NDArray):
                    sp = stationary.partition_axis if stationary.partition_axis is not None else 0
                    mp = moving.partition_axis if moving.partition_axis is not None else 0
                    if stationary.shape[sp] != moving.shape[mp]:
                        raise ISAContractError({
                            "kind": "matmul_partition_mismatch", "operation": operation,
                            "stationary_extent": stationary.shape[sp], "moving_extent": moving.shape[mp],
                            "stationary_shape": list(stationary.shape), "moving_shape": list(moving.shape),
                            "confidence": "witnessed_isa_constraint",
                            "guidance": "Both matmul operands need the same contraction partition extent. "
                                        "Integer selection of an on-chip partition axis retains a size-1 "
                                        "partition; it does not expose the next free axis as 128 partitions. "
                                        "Keep contraction rows on the partition axis and reuse/batching on free axes.",
                        })
        return function(*args, **kwargs)

    return checked
