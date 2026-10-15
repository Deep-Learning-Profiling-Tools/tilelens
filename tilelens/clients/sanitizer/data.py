from ...core.data import Store, Load
import operator
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Any, Literal, get_args
import torch
import z3

from ...ir.launch import TensorFacts
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


CompiledFindingKind = Literal["out-of-bounds", "integer-overflow", "division-by-zero"]


@dataclass
class CompiledSanitizerRecord:
    """One finding of the compiled sanitizer (``Sanitizer(compile=True)``)
    on one access, with a witness: an out-of-bounds address (outside the
    view's footprint), an address/mask/path term that can overflow its
    declared integer width, or one that can divide by zero.

    Plain data: the tensor is described by its launch-time facts, never held,
    since records outlive the launch and must not pin device memory.
    """

    kind: CompiledFindingKind
    # The accessing op; an atomic (a read and a write) is reported as Store.
    op_type: type[Store | Load]
    # The kernel parameter of the accessed tensor, and its facts at launch;
    # None when the launch bound no tensor to it (a finding in the loop's
    # bounds is attributed to the loop's first access, whatever it reads).
    tensor_name: str
    tensor_facts: TensorFacts | None
    # The free variables' values in the witness (program ids, arange lanes,
    # loop iterations), by name.
    witness: dict[str, int]
    # The config kwargs of the config (binding) the finding is in.
    config: dict[str, Any]
    user_code_tracebacks: list[TracebackInfo]
    # For "out-of-bounds": the element offset from the view's data_ptr and
    # its byte address; None for the other kinds.
    violation_offset: int | None = None
    violation_address: int | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in get_args(CompiledFindingKind):
            raise ValueError(f"unknown compiled sanitizer finding: {self.kind!r}")
        if self.op_type not in (Load, Store):
            raise TypeError(f"op_type must be Load or Store, not {self.op_type!r}")
        # Ints and plain containers only, so a saved trace holds the record.
        self.witness = {
            name: operator.index(value) for name, value in self.witness.items()
        }
        self.config = dict(self.config)
        self.user_code_tracebacks = list(self.user_code_tracebacks)
        for name in ("violation_offset", "violation_address"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, operator.index(value))
