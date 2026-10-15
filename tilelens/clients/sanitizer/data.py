from ...core.data import Store, Load
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Any
import torch
import z3

from ...utils.traceback_utils import TracebackInfo


@dataclass
class OutOfBoundsRecord:
    op_type: type[Store | Load]
    tensor: torch.Tensor
    user_code_tracebacks: list[TracebackInfo]


@dataclass
class OutOfBoundsRecordBruteForce(OutOfBoundsRecord):
    offsets: NDArray[np.int_]
    masks: NDArray[np.bool_]
    valid_access_masks: NDArray[np.bool_]
    invalid_access_masks: NDArray[np.bool_]
    corrected_offsets: NDArray[np.int_]


@dataclass
class OutOfBoundsRecordZ3(OutOfBoundsRecord):
    """
    Attributes:
        constraints (z3.z3.BoolRef | None):
            A conjunction of Z3 constraint expressions defining valid ranges
            for memory access. Any address falling outside these ranges is considered invalid.

        violation_address (int):
            The exact address where an invalid memory access was detected.
            For example:
            Invalid access detected at index: 200

            A few simplified constraints might look like:
              And(x >= 50, x <= 60, x != 55)
              And(x >= 70, x <= 80)
              ...

            In this scenario, 200 is an out-of-bounds address because it
            falls outside the valid ranges described by these constraints.

        symbolic_expr (Any | None):
            The symbolic expression tree that led to the OOB access, if available.
    """

    constraints: z3.z3.BoolRef | None
    violation_address: int
    symbolic_expr: Any = None  # Optional symbolic expression tree
    tensor_name: str | None = None


@dataclass
class UndecidedAccessRecordZ3:
    """
    A memory access Z3 answered ``unknown`` for: it was neither proved in
    bounds nor shown out of bounds, so it went unchecked. Not a finding.

    Attributes:
        tensor (torch.Tensor | None):
            The tensor the pointer expression is based on, or None when it
            cannot be resolved (there is no witness address to match).

        reason (str):
            Z3's ``reason_unknown()`` for the query.
    """

    op_type: type[Store | Load]
    tensor: torch.Tensor | None
    user_code_tracebacks: list[TracebackInfo]
    reason: str
    symbolic_expr: Any = None
    tensor_name: str | None = None
