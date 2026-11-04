from dataclasses import dataclass, field
from typing import ClassVar
import traceback
import numpy.typing as npt
import numpy as np
import torch

from ..utils.traceback_utils import extract_user_frames


@dataclass
class Op:
    name: ClassVar[str] = "op"
    call_path: list[traceback.FrameSummary] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.call_path = extract_user_frames(num_frames=1)


@dataclass
class ProgramId(Op):
    name: ClassVar[str] = "program_id"


@dataclass
class Allocate(Op):
    name: ClassVar[str] = "allocate"
    ptr: int


@dataclass
class RawStore(Op):
    name: ClassVar[str] = "raw_store"


@dataclass
class Store(Op):
    name: ClassVar[str] = "store"
    ptr: int
    offsets: npt.NDArray[np.int_]
    masks: npt.NDArray[np.bool_]
    mem_src: str = "SBUF"
    mem_dst: str = "HBM"
    frontend: str = "nki"
    bytes: int = 0
    time_idx: int = 0


@dataclass
class RawLoad(Op):
    name: ClassVar[str] = "raw_load"


@dataclass
class Load(Op):
    name: ClassVar[str] = "load"
    ptr: int
    offsets: npt.NDArray[np.int_]
    masks: npt.NDArray[np.bool_]
    # buffer: str
    mem_src: str = "HBM"
    mem_dst: str = "SBUF"
    frontend: str = "nki"
    bytes: int = 0
    time_idx: int = 0


@dataclass
class Transfer(Op):
    name: ClassVar[str] = "transfer"
    src_ptr: int
    dst_ptr: int
    src_offsets: npt.NDArray[np.int_]
    dst_offsets: npt.NDArray[np.int_]
    mem_src: str
    mem_dst: str
    bytes: int = 0
    time_idx: int = 0


@dataclass
class UnaryOp(Op):
    name: ClassVar[str] = "unary_op"


@dataclass
class BinaryOp(Op):
    name: ClassVar[str] = "binary_op"
    op: str
    input_shape: tuple
    output_shape: tuple


@dataclass
class TernaryOp(Op):
    name: ClassVar[str] = "ternary_op"


@dataclass
class Fma(Op):
    name: ClassVar[str] = "fma"


@dataclass
class Dot(Op):
    name: ClassVar[str] = "dot"
    input_shape: tuple
    other_shape: tuple
    output_shape: tuple
    input_data: list[list[float]]
    other_data: list[list[float]]
    intermediate_results: dict[tuple[int, int], float] = field(
        default_factory=dict
    )  # Only storing the result now

    def update_intermediate(self, row: int, col: int, result: float):
        # Store only the result as a float
        self.intermediate_results[(row, col)] = result


@dataclass
class MakeRange(Op):
    name: ClassVar[str] = "make_range"
    start: int
    end: int


@dataclass
class AddPtr(Op):
    name: ClassVar[str] = "addptr"


@dataclass
class ExpandDims(Op):
    name: ClassVar[str] = "expand_dims"
    input_shape: tuple
    index: int
    output_shape: tuple


@dataclass
class Broadcast(Op):
    name: ClassVar[str] = "broadcast"


@dataclass
class Reduce(Op):
    name: ClassVar[str] = "reduce"
    input_shape: tuple
    index: int
    keep_dims: bool
    output_shape: tuple


@dataclass
class ReduceMin(Reduce):
    name: ClassVar[str] = "reduce_min"
    reduce_type: ClassVar[str] = "min"


@dataclass
class ReduceMax(Reduce):
    name: ClassVar[str] = "reduce_max"
    reduce_type: ClassVar[str] = "max"


@dataclass
class ReduceSum(Reduce):
    name: ClassVar[str] = "reduce_sum"
    reduce_type: ClassVar[str] = "sum"


@dataclass
class ReduceXor(Reduce):
    name: ClassVar[str] = "reduce_xor"
    reduce_type: ClassVar[str] = "xor_sum"


@dataclass
class ReduceOr(Reduce):
    name: ClassVar[str] = "reduce_or"
    reduce_type: ClassVar[str] = "reduce_or"


@dataclass
class Sort(Op):
    name: ClassVar[str] = "sort"


@dataclass
class Splat(Op):
    # Broadcasts a scalar to a tensor
    name: ClassVar[str] = "splat"


@dataclass
class Unsplat(Op):
    name: ClassVar[str] = "unsplat"


@dataclass
class MakeBlockPointer(Op):
    name: ClassVar[str] = "make_block_ptr"


@dataclass
class TensorPointerLoad(Op):
    name: ClassVar[str] = "tensor_pointer_load"


@dataclass
class TensorPointerStore(Op):
    name: ClassVar[str] = "tensor_pointer_store"


@dataclass
class Idiv(Op):
    name: ClassVar[str] = "idiv"


@dataclass
class Rsqrt(Op):
    name: ClassVar[str] = "rsqrt"


@dataclass
class CastImpl(Op):
    name: ClassVar[str] = "cast_impl"


@dataclass
class Reshape(Op):
    name: ClassVar[str] = "reshape"


@dataclass
class Join(Op):
    name: ClassVar[str] = "join"


@dataclass
class Split(Op):
    name: ClassVar[str] = "split"


@dataclass
class Fabs(Op):
    name: ClassVar[str] = "fabs"


@dataclass
class Ashr(Op):
    name: ClassVar[str] = "ashr"


@dataclass
class Advance(Op):
    name: ClassVar[str] = "advance"


@dataclass
class FpToFp(Op):
    name: ClassVar[str] = "fp_to_fp"


@dataclass
class Umulhi(Op):
    name: ClassVar[str] = "umulhi"


@dataclass
class Trans(Op):
    name: ClassVar[str] = "trans"


@dataclass
class CumSum(Op):
    name: ClassVar[str] = "cumsum"


@dataclass
class Bitcast(Op):
    name: ClassVar[str] = "bitcast"


@dataclass
class PtrToInt(Op):
    name: ClassVar[str] = "ptr_to_int"


@dataclass
class IntToPtr(Op):
    name: ClassVar[str] = "int_to_ptr"


@dataclass
class AtomicCas(Op):
    name: ClassVar[str] = "atomic_cas"


@dataclass
class AtomicRMW(Op):
    name: ClassVar[str] = "atomic_rmw"


@dataclass
class Barrier(Op):
    """``tl.debug_barrier``: no accesses of its own; it orders the accesses of
    one program before it against those after it."""

    name: ClassVar[str] = "barrier"


@dataclass
class InterpreterDivergence(Op):
    """The interpreted run just computed something compiled Triton does not.

    Delivered with ``(kind, source_location)``: ``source_location`` is the
    ``(filename, lineno, func_name)`` of the kernel code involved, and
    DESCRIPTIONS says what each kind means.
    """

    name: ClassVar[str] = "interpreter_divergence"

    DESCRIPTIONS: ClassVar[dict[str, str]] = {
        "block-truthiness": (
            "takes the truth value of a block tensor of more than one element "
            "(Python and/or/not/if/while or a conditional expression). The "
            "Triton interpreter evaluates this differently from compiled "
            "Triton: it treats any such tensor as True, where compiled Triton "
            "combines the lanes elementwise or rejects the code. Use &, |, ~ "
            "or tl.where for elementwise logic; if/while need a scalar. This "
            "is an upstream Triton interpreter bug, see "
            "https://github.com/triton-lang/triton/pull/12150"
        ),
    }

    @classmethod
    def describe(cls, kind: str, source_location: tuple[str, int, str]) -> str:
        filename, lineno, func_name = source_location
        return f"{func_name} at {filename}:{lineno} {cls.DESCRIPTIONS[kind]}"


@dataclass
class InlineAsm(Op):
    """``tl.inline_asm_elementwise``, which the interpreted run cannot execute:
    the Triton frontend approximates its result by its first argument
    converted to the result dtype. Delivered with no arguments, before the
    approximation runs."""

    name: ClassVar[str] = "inline_asm"


@dataclass
class Tensor:
    ptr: int
    dtype: str
    stride: tuple
    shape: tuple
    element_size: int
    data: torch.Tensor


@dataclass(frozen=True)
class TensorSnapshot:
    """
    Portable tensor metadata plus CPU data for saved trace archives.
    We can't just use the tilelens.core.data.Tensor class since some clients (e.g. sanitizer)
    assumes that the data is stored in a torch.Tensor, not a Tensor instance.
    """

    ptr: int
    dtype: str
    _stride: tuple
    shape: tuple
    _element_size: int
    data: torch.Tensor
    device: str
    _contiguous: bool

    @classmethod
    def from_tensor(cls, tensor) -> "TensorSnapshot":
        """Capture a tensor-like object into a CPU-backed snapshot."""
        data = tensor.detach().cpu()
        device = getattr(tensor, "device", "cpu")
        contiguous_fn = getattr(tensor, "is_contiguous", None)
        return cls(
            ptr=tensor.data_ptr(),
            dtype=str(tensor.dtype),
            _stride=tuple(tensor.stride()),
            shape=tuple(tensor.shape),
            _element_size=tensor.element_size(),
            data=data.clone()
            if isinstance(data, torch.Tensor)
            else torch.as_tensor(data.numpy() if hasattr(data, "numpy") else data),
            device=str(device),
            _contiguous=bool(contiguous_fn()) if callable(contiguous_fn) else True,
        )

    def data_ptr(self) -> int:
        """Mirror the tensor data_ptr() API used by trace consumers."""
        return self.ptr

    def stride(self) -> tuple:
        """Mirror the tensor stride() API used by trace consumers."""
        return self._stride

    def element_size(self) -> int:
        """Mirror the tensor element_size() API used by trace consumers."""
        return self._element_size

    def is_contiguous(self) -> bool:
        """Mirror the tensor is_contiguous() API used by trace consumers."""
        return self._contiguous


@dataclass
class Grid:
    idx: tuple


@dataclass
class Launch:
    grid: tuple | None = None
    tensors: set[Tensor] = field(default_factory=set)
    records: list = field(default_factory=list)
