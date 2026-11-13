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
    src_ptr: int | None = None
    src_storage: int | None = None
    src_range: tuple[int, int] | None = None
    src_version: int | None = None
    src_dtype: str | None = None
    dst_storage: int | None = None
    dst_version: int | None = None
    mask_provided: bool = False


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
    src_storage: int | None = None
    src_version: int | None = None
    src_dtype: str | None = None
    dst_ptr: int | None = None
    dst_storage: int | None = None
    dst_range: tuple[int, int] | None = None
    dst_version: int | None = None
    mask_provided: bool = False


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
    src_partition_axis: int | None = None
    dst_partition_axis: int | None = None
    dma_pattern: str = "copy"


class DmaTranspose(Transfer):
    """Normalized DMA-transpose operation."""

    name: ClassVar[str] = "dma_transpose"


@dataclass
class UnaryOp(Op):
    name: ClassVar[str] = "unary_op"


@dataclass
class BinaryOp(Op):
    name: ClassVar[str] = "binary_op"
    op: str
    input_shape: tuple
    output_shape: tuple
    other_shape: tuple = ()
    input_ptrs: tuple[int, ...] = ()
    output_ptr: int | None = None
    input_storages: tuple[int, ...] = ()
    input_ranges: tuple[tuple[int, int], ...] = ()
    input_versions: tuple[int, ...] = ()
    output_storage: int | None = None
    output_range: tuple[int, int] | None = None
    output_version: int | None = None


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
    # Base pointers of the two operands and the destination tile. These let a
    # downstream cost model resolve TensorE's true data dependency on the DMAs
    # that filled its SBUF inputs (and the store that drains its output) by
    # exact pointer matching, instead of a conservative "wait for every prior
    # transfer" heuristic. Defaulted so existing constructors stay valid.
    input_ptrs: tuple[int, ...] = ()
    output_ptr: int | None = None
    input_storages: tuple[int, ...] = ()
    input_ranges: tuple[tuple[int, int], ...] = ()
    input_versions: tuple[int, ...] = ()
    output_storage: int | None = None
    output_range: tuple[int, int] | None = None
    output_version: int | None = None
    input_dtypes: tuple[str, ...] = ()
    output_dtype: str | None = None
    accumulate: bool | None = None

    def update_intermediate(self, row: int, col: int, result: float):
        # Store only the result as a float
        self.intermediate_results[(row, col)] = result


@dataclass
class MakeRange(Op):
    name: ClassVar[str] = "make_range"
    start: int
    end: int


@dataclass
class NkiCompute(Op):
    """Catch-all record for NKI compute APIs (elementwise, reductions, etc.)."""
    name: ClassVar[str] = "nki_compute"
    api_op: str
    engine: str
    input_ptrs: tuple[int, ...]
    output_ptrs: tuple[int, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    output_shapes: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[str, ...]
    output_dtype: str
    attrs: dict
    input_storages: tuple[int, ...] = ()
    input_ranges: tuple[tuple[int, int], ...] = ()
    input_versions: tuple[int, ...] = ()
    output_storages: tuple[int, ...] = ()
    output_ranges: tuple[tuple[int, int], ...] = ()
    output_versions: tuple[int, ...] = ()


@dataclass
class TensorTranspose(Op):
    """On-chip TensorE PF-transpose (``nisa.nc_transpose``).

    A TensorE PF-transpose is implemented as a matrix multiply of the source
    tile against an identity matrix. The record keeps the source/destination
    storage identity so cross-engine dependencies resolve exactly, and the
    trace dumper derives the equivalent identity-matmul FLOPs from the source
    partition/free geometry.
    """

    name: ClassVar[str] = "tensor_transpose"
    input_shape: tuple
    output_shape: tuple
    input_ptrs: tuple[int, ...] = ()
    output_ptr: int | None = None
    input_storages: tuple[int, ...] = ()
    input_ranges: tuple[tuple[int, int], ...] = ()
    input_versions: tuple[int, ...] = ()
    output_storage: int | None = None
    output_range: tuple[int, int] | None = None
    output_version: int | None = None
    input_dtypes: tuple[str, ...] = ()
    output_dtype: str | None = None
    engine: str = "tensor"


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
    # Optional base-pointer linkage so a dependency-aware scheduler can resolve
    # cross-engine RAW/WAR/WAW edges through this reduction. Defaulted so all
    # existing positional constructors (triton/gluon/nki) remain valid.
    input_ptrs: tuple[int, ...] = ()
    output_ptr: int | None = None
    input_storages: tuple[int, ...] = ()
    input_ranges: tuple[tuple[int, int], ...] = ()
    input_versions: tuple[int, ...] = ()
    output_storage: int | None = None
    output_range: tuple[int, int] | None = None
    output_version: int | None = None
    api_op: str = "reduce_sum"
    # Preserve source operand precision separately from the reduction result.
    # NKI reductions may promote BF16 inputs to FP32; collapsing both roles
    # into one dtype makes source-only calibration unable to distinguish those
    # lowering domains.
    input_dtypes: tuple[str, ...] = ()
    output_dtype: str | None = None


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
    We can't just use the triton_viz.core.data.Tensor class since some clients (e.g. sanitizer)
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
