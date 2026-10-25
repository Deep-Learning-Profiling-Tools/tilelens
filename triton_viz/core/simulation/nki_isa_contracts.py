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
        super().__init__(
            f"{finding['operation']} {finding['operand']} partition axis "
            f"{finding['partition_axis']} has extent {finding['partition_extent']}, "
            f"exceeding {finding['limit']}; shape={finding['shape']}"
        )


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
        return function(*args, **kwargs)

    return checked
