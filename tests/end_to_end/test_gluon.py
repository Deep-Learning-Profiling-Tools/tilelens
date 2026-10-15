import importlib.util
from pathlib import Path

import pytest
import torch
from triton import knobs
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import async_copy as amd_cdna4_cp
from triton.experimental.gluon.language.nvidia import blackwell, hopper
from triton.experimental.gluon.language.nvidia.blackwell import tma as blackwell_tma
from triton.experimental.gluon.language.nvidia.hopper import (
    mbarrier,
    tma,
)
from triton.experimental.gluon.nvidia.hopper import (
    TensorDescriptor,
    TensorDescriptorIm2Col,
)

import tilelens
from tilelens.clients.sanitizer.sanitizer import SymbolicSanitizer
from tilelens.core.data import Load
from tilelens.core.simulation.gluon import GluonInterpretedFunction, gluon_builder

try:
    from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp
except ImportError:
    cp = None

_HAS_AMPERE_ASYNC_COPY = (
    cp is not None and getattr(cp, "async_copy_global_to_local", None) is not None
)


def _run_gluon_on_cpu(fn, grid, *args, **kwargs):
    GluonInterpretedFunction(fn.fn).run(*args, grid=grid, **kwargs)


# CI can run against Gluon versions whose TensorMemoryLayout._to_ir uses older
# builder argument order; keep this CPU accuracy test on the current builder shape.
class _TensorMemoryLayoutForCpu(blackwell.TensorMemoryLayout):
    def __init__(self, block, col_stride, cga_layout=None, two_ctas=False):
        object.__setattr__(self, "block", tuple(block))
        object.__setattr__(self, "col_stride", int(col_stride))
        object.__setattr__(
            self,
            "cga_layout",
            [] if cga_layout is None else [list(basis) for basis in cga_layout],
        )
        object.__setattr__(self, "two_ctas", bool(two_ctas))
        object.__setattr__(self, "fp4_padded", False)
        object.__setattr__(self, "cta_split_num", None)

    def _to_ir(self, builder):
        return builder.get_tensor_memory_layout(
            self.block,
            self.col_stride,
            self.cga_layout,
            self.two_ctas,
            False,
        )

    def mangle(self):
        cga_layout_str = "_".join(
            "~".join(map(str, basis)) for basis in self.cga_layout
        )
        two_ctas_str = "2CT" if self.two_ctas else ""
        return f"TL{self.block[0]}x{self.block[1]}C{self.col_stride}{cga_layout_str}{two_ctas_str}TL"


@gluon.jit
def _copy_scalar_kernel(in_ptr, out_ptr):
    value = gl.load(in_ptr)
    gl.store(out_ptr, value)


@gluon.jit
def _oob_scalar_offset_kernel(in_ptr, out_ptr):
    value = gl.load(in_ptr + 1)
    gl.store(out_ptr, value)


@gluon.jit
def _masked_safe_kernel(
    in_ptr,
    out_ptr,
    n: gl.constexpr,
    BLOCK: gl.constexpr,
    layout: gl.constexpr,
):
    offs = gl.arange(0, BLOCK, layout=layout)
    mask = offs < n
    value = gl.load(in_ptr + offs, mask=mask, other=0.0)
    gl.store(out_ptr + offs, value, mask=mask)


@gluon.jit
def _range_memcpy_kernel(in_ptr, out_ptr, xnumel, BLOCK: gl.constexpr):
    pid = gl.program_id(0)
    start = pid * BLOCK
    end = min(start + BLOCK, xnumel)
    for i in range(start, end):
        value = gl.load(in_ptr + i)
        gl.store(out_ptr + i, value)


@gluon.jit
def _masked_1d_memcpy_kernel(
    in_ptr,
    out_ptr,
    xnumel,
    BLOCK: gl.constexpr,
    layout: gl.constexpr,
):
    pid = gl.program_id(0)
    offsets = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offsets < xnumel
    value = gl.load(in_ptr + offsets, mask=mask, other=0.0)
    gl.store(out_ptr + offsets, value, mask=mask)


@gluon.jit
def _masked_2d_memcpy_kernel(
    in_ptr,
    out_ptr,
    xnumel,
    ynumel,
    xstride_in,
    ystride_in,
    xstride_out,
    ystride_out,
    layout: gl.constexpr,
    XBLOCK: gl.constexpr,
    YBLOCK: gl.constexpr,
):
    pid_x = gl.program_id(0)
    pid_y = gl.program_id(1)
    start_x = pid_x * XBLOCK
    start_y = pid_y * YBLOCK
    offsets_x = start_x + gl.arange(
        0,
        XBLOCK,
        layout=gl.SliceLayout(dim=1, parent=layout),
    )
    offsets_y = start_y + gl.arange(
        0,
        YBLOCK,
        layout=gl.SliceLayout(dim=0, parent=layout),
    )
    in_offsets = xstride_in * offsets_x[:, None] + ystride_in * offsets_y[None, :]
    out_offsets = xstride_out * offsets_x[:, None] + ystride_out * offsets_y[None, :]
    mask = (offsets_x[:, None] < xnumel) & (offsets_y[None, :] < ynumel)

    value = gl.load(in_ptr + in_offsets, mask=mask, other=0.0)
    gl.store(out_ptr + out_offsets, value, mask=mask)


@gluon.jit
def _converted_layout_add_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    xnumel,
    ynumel,
    xstride_a,
    ystride_a,
    xstride_b,
    ystride_b,
    xstride_out,
    ystride_out,
    layout_in: gl.constexpr,
    layout_out: gl.constexpr,
    XBLOCK: gl.constexpr,
    YBLOCK: gl.constexpr,
):
    pid = gl.program_id(0)
    xoffs = pid * XBLOCK + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout_in))
    yoffs = gl.arange(0, YBLOCK, gl.SliceLayout(0, layout_in))
    mask = (xoffs[:, None] < xnumel) & (yoffs[None, :] < ynumel)
    a_offsets = xstride_a * xoffs[:, None] + ystride_a * yoffs[None, :]
    b_offsets = xstride_b * xoffs[:, None] + ystride_b * yoffs[None, :]
    out_offsets = xstride_out * xoffs[:, None] + ystride_out * yoffs[None, :]
    values = gl.load(a_ptr + a_offsets, mask=mask, other=0.0) + gl.load(
        b_ptr + b_offsets,
        mask=mask,
        other=0.0,
    )
    gl.store(out_ptr + out_offsets, gl.convert_layout(values, layout_out), mask=mask)


@gluon.jit
def _async_copy_1d_kernel(in_ptr, out_ptr, xnumel, BLOCK: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    offsets = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offsets < xnumel
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1,
        per_phase=1,
        max_phase=1,
        order=[0],
    )
    smem = gl.allocate_shared_memory(gl.float32, [BLOCK], layout=smem_layout)

    cp.async_copy_global_to_local(smem, in_ptr + offsets, mask=mask)
    cp.commit_group()
    cp.wait_group(0)
    gl.store(out_ptr + offsets, smem.load(layout), mask=mask)


@gluon.jit
def _amd_async_copy_other_kernel(in_ptr, out_ptr, xnumel, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    offsets = gl.arange(0, BLOCK, layout=layout)
    mask = offsets < xnumel
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1,
        per_phase=1,
        max_phase=1,
        order=[0],
    )
    smem = gl.allocate_shared_memory(gl.float32, [BLOCK], layout=smem_layout)

    amd_cdna4_cp.global_load_to_shared(
        smem,
        in_ptr + offsets,
        mask=mask,
        other=-7.0,
    )
    amd_cdna4_cp.wait_group(0)
    values = smem.load(layout)
    gl.store(out_ptr + offsets, values)


@gluon.jit
def _async_copy_elementwise_add_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    xnumel,
    ynumel,
    xstride_a,
    ystride_a,
    xstride_b,
    ystride_b,
    xstride_out,
    ystride_out,
    XBLOCK: gl.constexpr,
    YBLOCK: gl.constexpr,
    smem_layout: gl.constexpr,
):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    xoffs = pid * XBLOCK + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))
    yoffs = gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))
    mask = (xoffs < xnumel)[:, None] & (yoffs < ynumel)[None, :]
    a_smem = gl.allocate_shared_memory(gl.float32, [XBLOCK, YBLOCK], smem_layout)
    b_smem = gl.allocate_shared_memory(gl.float32, [XBLOCK, YBLOCK], smem_layout)

    cp.async_copy_global_to_local(
        a_smem,
        a_ptr + xstride_a * xoffs[:, None] + ystride_a * yoffs[None, :],
        mask=mask,
    )
    cp.async_copy_global_to_local(
        b_smem,
        b_ptr + xstride_b * xoffs[:, None] + ystride_b * yoffs[None, :],
        mask=mask,
    )
    cp.commit_group()
    cp.wait_group(0)
    values = a_smem.load(layout) + b_smem.load(layout)
    gl.store(
        out_ptr + xstride_out * xoffs[:, None] + ystride_out * yoffs[None, :],
        values,
        mask=mask,
    )


@gluon.jit
def _tma_copy_1d_kernel(in_desc, out_desc, BLOCK: gl.constexpr):
    pid = gl.program_id(0)
    smem = gl.allocate_shared_memory(in_desc.dtype, [BLOCK], in_desc.layout)
    barrier = gl.allocate_shared_memory(gl.int64, [1], in_desc.layout)

    mbarrier.init(barrier, count=1)
    mbarrier.expect(barrier, in_desc.block_type.nbytes)
    tma.async_load(in_desc, [pid * BLOCK], barrier, smem)
    mbarrier.wait(barrier, phase=0)
    mbarrier.invalidate(barrier)
    tma.async_store(out_desc, [pid * BLOCK], smem)
    tma.store_wait(0)


@gluon.jit
def _tma_elementwise_add_kernel(
    a_desc,
    b_desc,
    out_desc,
    xnumel,
    ynumel,
    XBLOCK: gl.constexpr,
    YBLOCK: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    a_smem = gl.allocate_shared_memory(a_desc.dtype, [XBLOCK, YBLOCK], a_desc.layout)
    b_smem = gl.allocate_shared_memory(b_desc.dtype, [XBLOCK, YBLOCK], b_desc.layout)
    out_smem = gl.allocate_shared_memory(
        out_desc.dtype,
        [XBLOCK, YBLOCK],
        out_desc.layout,
    )
    barrier = gl.allocate_shared_memory(gl.int64, [1], a_desc.layout)

    mbarrier.init(barrier, count=1)
    mbarrier.expect(barrier, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
    tma.async_load(a_desc, [0, 0], barrier, a_smem)
    tma.async_load(b_desc, [0, 0], barrier, b_smem)
    mbarrier.wait(barrier, phase=0)
    out_smem.store(a_smem.load(layout) + b_smem.load(layout))
    hopper.fence_async_shared()
    tma.async_store(out_desc, [0, 0], out_smem)
    tma.store_wait(0)


@gluon.jit
def _tma_atomic_float_kernel(
    add_desc,
    min_desc,
    max_desc,
    src_ptr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1],
        [1, 32],
        [1, gl.num_warps()],
        [1, 0],
    )
    offs_m = gl.arange(0, BLOCK_M, gl.SliceLayout(1, layout))
    offs_n = gl.arange(0, BLOCK_N, gl.SliceLayout(0, layout))
    src = gl.load(src_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :])
    smem = gl.allocate_shared_memory(src.dtype, [BLOCK_M, BLOCK_N], add_desc.layout)
    smem.store(src)
    hopper.fence_async_shared()
    tma.async_atomic_add(add_desc, [1, 2], smem)
    tma.async_atomic_min(min_desc, [1, 2], smem)
    tma.async_atomic_max(max_desc, [1, 2], smem)
    tma.store_wait(0)


@gluon.jit
def _blackwell_tma_gather_kernel(
    out_ptr,
    out_stride_x,
    out_stride_y,
    tensor_desc,
    x_offsets_ptr,
    y_offset,
    BLOCK_X: gl.constexpr,
):
    BLOCK_Y: gl.constexpr = tensor_desc.block_type.shape[1]
    offsets_layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    x_offsets = gl.load(x_offsets_ptr + gl.arange(0, BLOCK_X, offsets_layout))
    smem_dest = gl.allocate_shared_memory(
        tensor_desc.dtype,
        [BLOCK_X, BLOCK_Y],
        tensor_desc.layout,
    )
    barrier = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())

    mbarrier.init(barrier, count=1)
    mbarrier.expect(barrier, BLOCK_X * tensor_desc.block_type.nbytes)
    blackwell_tma.async_gather(
        tensor_desc,
        x_offsets,
        y_offset,
        barrier=barrier,
        result=smem_dest,
    )
    mbarrier.wait(barrier, phase=0)
    mbarrier.invalidate(barrier)

    output_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1],
        [1, 32],
        [1, gl.num_warps()],
        [1, 0],
    )
    out = smem_dest.load(output_layout)
    rows = gl.arange(0, BLOCK_X, gl.SliceLayout(1, output_layout))[:, None]
    cols = gl.arange(0, BLOCK_Y, gl.SliceLayout(0, output_layout))[None, :]
    gl.store(out_ptr + rows * out_stride_x + cols * out_stride_y, out)


@gluon.jit
def _blackwell_tma_scatter_kernel(
    tensor_desc,
    x_offsets_ptr,
    y_offset,
    src_ptr,
    src_stride_x,
    src_stride_y,
    BLOCK_X: gl.constexpr,
):
    BLOCK_Y: gl.constexpr = tensor_desc.block_type.shape[1]
    source_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1],
        [1, 32],
        [1, gl.num_warps()],
        [1, 0],
    )
    rows = gl.arange(0, BLOCK_X, gl.SliceLayout(1, source_layout))[:, None]
    cols = gl.arange(0, BLOCK_Y, gl.SliceLayout(0, source_layout))[None, :]
    src = gl.load(src_ptr + rows * src_stride_x + cols * src_stride_y)
    offsets_layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    x_offsets = gl.load(x_offsets_ptr + gl.arange(0, BLOCK_X, offsets_layout))
    smem_src = gl.allocate_shared_memory(
        tensor_desc.dtype,
        [BLOCK_X, BLOCK_Y],
        tensor_desc.layout,
    )
    smem_src.store(src)
    hopper.fence_async_shared()
    blackwell_tma.async_scatter(tensor_desc, x_offsets, y_offset, smem_src)
    blackwell_tma.store_wait(0)


@gluon.jit
def _blackwell_tma_bitwise_atomic_kernel(
    and_desc,
    or_desc,
    xor_desc,
    src_ptr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1],
        [1, 32],
        [1, gl.num_warps()],
        [1, 0],
    )
    offs_m = gl.arange(0, BLOCK_M, gl.SliceLayout(1, layout))
    offs_n = gl.arange(0, BLOCK_N, gl.SliceLayout(0, layout))
    src = gl.load(src_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :])
    smem = gl.allocate_shared_memory(src.dtype, [BLOCK_M, BLOCK_N], and_desc.layout)
    smem.store(src)
    hopper.fence_async_shared()
    blackwell_tma.async_atomic_and(and_desc, [1, 1], smem)
    blackwell_tma.async_atomic_or(or_desc, [1, 1], smem)
    blackwell_tma.async_atomic_xor(xor_desc, [1, 1], smem)
    blackwell_tma.store_wait(0)


@gluon.jit
def _tma_im2col_kernel(
    in_desc,
    out_desc,
    coord_n,
    coord_h,
    coord_w,
    coord_c,
    offset_h: gl.constexpr,
    offset_w: gl.constexpr,
):
    smem = gl.allocate_shared_memory(in_desc.dtype, in_desc.block_shape, in_desc.layout)
    barrier = gl.allocate_shared_memory(gl.int64, [1], in_desc.layout)
    mbarrier.init(barrier, count=1)
    mbarrier.expect(barrier, in_desc.block_type.nbytes)
    tma.async_load_im2col(
        in_desc,
        [coord_n, coord_h, coord_w, coord_c],
        [offset_h, offset_w],
        barrier,
        smem,
    )
    mbarrier.wait(barrier, phase=0)
    mbarrier.invalidate(barrier)
    tma.async_store(out_desc, [0, 0], smem)
    tma.store_wait(0)


def _run_im2col_case(
    inp,
    pixel_box_lower_corner,
    pixel_box_upper_corner,
    coord,
    offsets,
):
    out = torch.zeros((16, 32), dtype=torch.float32)
    layout = gl.NVMMASharedLayout(
        swizzle_byte_width=128,
        element_bitwidth=32,
        rank=2,
    )
    in_desc = TensorDescriptorIm2Col.from_tensor(
        inp,
        [16, 32],
        layout,
        padding="zero",
        element_strides=[1, 1, 1, 1],
        pixel_box_lower_corner=pixel_box_lower_corner,
        pixel_box_upper_corner=pixel_box_upper_corner,
    )
    out_desc = TensorDescriptor.from_tensor(out, [16, 32], layout)
    _run_gluon_on_cpu(
        _tma_im2col_kernel, (1,), in_desc, out_desc, *coord, *offsets, num_warps=1
    )

    return out


@gluon.constexpr_function
def _wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps):
    instr_m = 16
    instr_n = min(BLOCK_N, 64)
    while BLOCK_N % instr_n != 0:
        instr_n -= 8
    return gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=[num_warps, 1],
        instr_shape=[instr_m, instr_n, 256 // dtype.primitive_bitwidth],
    )


@gluon.jit
def _small_wgmma_kernel(
    a_desc,
    b_desc,
    c_desc,
    out_desc,
    LHS_IN_REG: gl.constexpr,
    num_warps: gl.constexpr,
):
    barrier = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(barrier, count=1)
    a_smem = gl.allocate_shared_memory(
        a_desc.dtype,
        a_desc.block_type.shape,
        a_desc.layout,
    )
    b_smem = gl.allocate_shared_memory(
        b_desc.dtype,
        b_desc.block_type.shape,
        b_desc.layout,
    )
    c_smem = gl.allocate_shared_memory(
        c_desc.dtype,
        c_desc.block_type.shape,
        c_desc.layout,
    )
    mbarrier.expect(
        barrier,
        a_desc.block_type.nbytes + b_desc.block_type.nbytes + c_desc.block_type.nbytes,
    )
    tma.async_load(a_desc, [0, 0], barrier, a_smem)
    tma.async_load(b_desc, [0, 0], barrier, b_smem)
    tma.async_load(c_desc, [0, 0], barrier, c_smem)
    mbarrier.wait(barrier, phase=0)
    mbarrier.invalidate(barrier)

    acc_layout: gl.constexpr = _wgmma_layout(
        a_desc.dtype,
        out_desc.block_type.shape[0],
        out_desc.block_type.shape[1],
        num_warps,
    )
    a_reg_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=acc_layout,
        k_width=32 // a_desc.dtype.primitive_bitwidth,
    )
    a = a_smem.load(a_reg_layout) if LHS_IN_REG else a_smem
    c = c_smem.load(acc_layout)
    out = hopper.warpgroup_mma(a, b_smem, c, is_async=True, use_acc=True)
    out = hopper.warpgroup_mma_wait(num_outstanding=0, deps=(out,))

    out_smem = gl.allocate_shared_memory(
        out_desc.dtype,
        out_desc.block_type.shape,
        out_desc.layout,
    )
    out_smem.store(out)
    hopper.fence_async_shared()
    tma.async_store(out_desc, [0, 0], out_smem)
    tma.store_wait(0)


def _run_small_wgmma_case(seed: int, lhs_in_reg: bool):
    torch.manual_seed(seed)
    a = torch.randn(64, 32, dtype=torch.float16) / 4
    b = torch.randn(32, 32, dtype=torch.float16) / 4
    c = torch.randn(64, 32, dtype=torch.float32) / 4
    out = torch.empty_like(c)
    a_layout = gl.NVMMASharedLayout.get_default_for(a.shape, gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for(b.shape, gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for(c.shape, gl.float32)
    a_desc = TensorDescriptor.from_tensor(a, list(a.shape), a_layout)
    b_desc = TensorDescriptor.from_tensor(b, list(b.shape), b_layout)
    c_desc = TensorDescriptor.from_tensor(c, list(c.shape), c_layout)
    out_desc = TensorDescriptor.from_tensor(out, list(out.shape), c_layout)
    _run_gluon_on_cpu(
        _small_wgmma_kernel,
        (1,),
        a_desc,
        b_desc,
        c_desc,
        out_desc,
        lhs_in_reg,
        num_warps=4,
    )

    torch.testing.assert_close(out, a.float() @ b.float() + c, atol=1e-2, rtol=1e-2)


@gluon.jit
def _tcgen05_small_mma_kernel(
    a_desc,
    b_desc,
    c_desc,
    out_desc,
    tmem_block: gl.constexpr,
    LHS_IN_TMEM: gl.constexpr,
    USE_COMMIT: gl.constexpr,
    num_warps: gl.constexpr,
):
    barrier = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(barrier, count=1)
    a_smem = gl.allocate_shared_memory(
        a_desc.dtype,
        a_desc.block_type.shape,
        a_desc.layout,
    )
    b_smem = gl.allocate_shared_memory(
        b_desc.dtype,
        b_desc.block_type.shape,
        b_desc.layout,
    )
    c_smem = gl.allocate_shared_memory(
        c_desc.dtype,
        c_desc.block_type.shape,
        c_desc.layout,
    )
    mbarrier.expect(
        barrier,
        a_desc.block_type.nbytes + b_desc.block_type.nbytes + c_desc.block_type.nbytes,
    )
    tma.async_load(a_desc, [0, 0], barrier, a_smem)
    tma.async_load(b_desc, [0, 0], barrier, b_smem)
    tma.async_load(c_desc, [0, 0], barrier, c_smem)
    mbarrier.wait(barrier, phase=0)
    mbarrier.invalidate(barrier)
    mbarrier.init(barrier, count=1)

    M: gl.constexpr = out_desc.block_type.shape[0]
    N: gl.constexpr = out_desc.block_type.shape[1]
    K: gl.constexpr = a_desc.block_type.shape[1]
    acc_tmem_layout: gl.constexpr = _TensorMemoryLayoutForCpu(
        tmem_block,
        col_stride=32 // out_desc.dtype.primitive_bitwidth,
        cga_layout=[[1, 0], [0, 1]],
    )
    acc_tmem = blackwell.allocate_tensor_memory(
        out_desc.dtype,
        [M, N],
        acc_tmem_layout,
    )
    acc_layout: gl.constexpr = _wgmma_layout(out_desc.dtype, M, N, num_warps)
    acc = c_smem.load(acc_layout)
    acc_tmem.store(acc)

    if LHS_IN_TMEM:
        lhs_tmem_layout: gl.constexpr = _TensorMemoryLayoutForCpu(
            tmem_block,
            col_stride=1,
            cga_layout=[[1, 0], [0, 1]],
        )
        lhs_tmem = blackwell.allocate_tensor_memory(
            a_desc.dtype,
            [M, K],
            lhs_tmem_layout,
        )
        lhs_layout: gl.constexpr = gl.DotOperandLayout(
            operand_index=0,
            parent=acc_layout,
            k_width=32 // a_desc.dtype.primitive_bitwidth,
        )
        lhs = a_smem.load(lhs_layout)
        lhs_tmem.store(lhs)
        a = lhs_tmem
    else:
        a = a_smem

    if USE_COMMIT:
        blackwell.tcgen05_mma(a, b_smem, acc_tmem)
        blackwell.tcgen05_commit(barrier)
    else:
        blackwell.tcgen05_mma(
            a,
            b_smem,
            acc_tmem,
            mbarriers=[barrier],
            mbarrier_preds=[True],
        )
    mbarrier.wait(barrier, phase=0)
    mbarrier.invalidate(barrier)

    out_smem = gl.allocate_shared_memory(
        out_desc.dtype,
        out_desc.block_type.shape,
        out_desc.layout,
    )
    out_smem.store(acc_tmem.load(acc_layout))
    hopper.fence_async_shared()
    tma.async_store(out_desc, [0, 0], out_smem)
    tma.store_wait(0)


def _run_tcgen05_small_mma_case(seed: int, lhs_in_tmem: bool, use_commit: bool):
    torch.manual_seed(seed)
    a = torch.randn(64, 32, dtype=torch.float16) / 4
    b = torch.randn(32, 64, dtype=torch.float16) / 4
    c = torch.randn(64, 64, dtype=torch.float32) / 4
    out = torch.empty_like(c)
    a_layout = gl.NVMMASharedLayout.get_default_for(a.shape, gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for(b.shape, gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for(c.shape, gl.float32)
    a_desc = TensorDescriptor.from_tensor(a, list(a.shape), a_layout)
    b_desc = TensorDescriptor.from_tensor(b, list(b.shape), b_layout)
    c_desc = TensorDescriptor.from_tensor(c, list(c.shape), c_layout)
    out_desc = TensorDescriptor.from_tensor(out, list(out.shape), c_layout)
    _run_gluon_on_cpu(
        _tcgen05_small_mma_kernel,
        (1,),
        a_desc,
        b_desc,
        c_desc,
        out_desc,
        (64, 64),
        lhs_in_tmem,
        use_commit,
        num_warps=4,
    )

    torch.testing.assert_close(out, a.float() @ b.float() + c, atol=1e-2, rtol=1e-2)


@gluon.jit
def _tensor_memory_roundtrip_kernel(
    in_ptr,
    out_ptr,
    M: gl.constexpr,
    N: gl.constexpr,
    num_warps: gl.constexpr,
):
    global_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1],
        [1, 32],
        [1, num_warps],
        [1, 0],
    )
    offs_m = gl.arange(0, M, gl.SliceLayout(1, global_layout))
    offs_n = gl.arange(0, N, gl.SliceLayout(0, global_layout))
    offsets = offs_m[:, None] * N + offs_n[None, :]
    value = gl.load(in_ptr + offsets)
    tmem_layout: gl.constexpr = _TensorMemoryLayoutForCpu(
        block=(64, 64),
        col_stride=32 // in_ptr.dtype.element_ty.primitive_bitwidth,
        cga_layout=[[1, 0], [0, 1]],
    )
    tmem = blackwell.allocate_tensor_memory(
        element_ty=in_ptr.dtype.element_ty,
        shape=[M, N],
        layout=tmem_layout,
    )
    tmem.store(value)
    out = tmem.load(global_layout)
    gl.store(out_ptr + offsets, out)


def test_gluon_trace_runs_copy_scalar_kernel():
    kernel = tilelens.trace("tracer", frontend="gluon")(_copy_scalar_kernel)

    inp = torch.tensor([42.0])
    out = torch.empty_like(inp)
    kernel[(1,)](inp, out, num_warps=1)

    torch.testing.assert_close(out, inp, atol=0, rtol=0)
    assert kernel.client_manager.launch.grid == (1, 1, 1)
    assert len(kernel.client_manager.launch.records) >= 1


def test_gluon_sanitizer_allows_in_bounds_kernel():
    sanitizer = SymbolicSanitizer(abort_on_error=False)
    kernel = tilelens.trace(
        client=sanitizer,
        frontend="gluon",
    )(_copy_scalar_kernel)

    inp = torch.tensor([42.0])
    out = torch.empty_like(inp)
    ret = kernel[(1,)](inp, out, num_warps=1)

    assert ret is None
    assert sanitizer.records == []


def test_gluon_sanitizer_reports_real_oob_load_kernel(monkeypatch):
    monkeypatch.setattr(knobs.compilation, "always_compile", True)
    sanitizer = SymbolicSanitizer(abort_on_error=False)
    kernel = tilelens.trace(
        client=sanitizer,
        frontend="gluon",
    )(_oob_scalar_offset_kernel)
    kernel.fn.device_caches.clear()

    inp = torch.tensor([42.0])
    out = torch.empty_like(inp)
    ret = kernel[(1,)](inp, out, num_warps=1)

    assert ret is None
    assert sanitizer.records
    assert sanitizer.records[0].op_type is Load
    assert (
        sanitizer.records[0].violation_address
        == sanitizer.records[0].tensor.data_ptr() + inp.element_size()
    )


def test_gluon_sanitizer_allows_masked_in_bounds_kernel():
    sanitizer = SymbolicSanitizer(abort_on_error=False)
    kernel = tilelens.trace(
        client=sanitizer,
        frontend="gluon",
    )(_masked_safe_kernel)

    inp = torch.arange(4, dtype=torch.float32)
    out = torch.empty((8,), dtype=torch.float32)
    layout = gl.BlockedLayout([1], [32], [1], [0])
    ret = kernel[(1,)](inp, out, 4, 8, layout, num_warps=1)

    assert ret is None
    assert sanitizer.records == []


def test_gluon_core_ops_run_scalar_range_memcpy_on_cpu():
    inp = torch.arange(40, dtype=torch.float32)
    out = torch.full_like(inp, -1)
    kernel = tilelens.trace("tracer", frontend="gluon")(_range_memcpy_kernel)

    kernel[(1,)](inp, out, inp.numel(), 64, num_warps=1)

    torch.testing.assert_close(out, inp, atol=0, rtol=0)
    assert kernel.client_manager.launch.grid == (1, 1, 1)


def test_gluon_core_ops_run_masked_1d_memcpy_on_cpu():
    inp = torch.arange(40, dtype=torch.float32)
    out = torch.full_like(inp, -1)
    layout = gl.BlockedLayout([1], [32], [1], [0])
    kernel = tilelens.trace("tracer", frontend="gluon")(_masked_1d_memcpy_kernel)

    kernel[(1,)](inp, out, inp.numel(), 64, layout, num_warps=1)

    torch.testing.assert_close(out, inp, atol=0, rtol=0)


def test_gluon_core_ops_run_masked_2d_memcpy_on_cpu():
    inp = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    out = torch.full_like(inp, -1)
    layout = gl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    kernel = tilelens.trace("tracer", frontend="gluon")(_masked_2d_memcpy_kernel)

    kernel[(1, 1)](
        inp,
        out,
        *inp.shape,
        *inp.stride(),
        *out.stride(),
        layout,
        8,
        8,
        num_warps=4,
    )

    torch.testing.assert_close(out, inp, atol=0, rtol=0)


def test_gluon_core_ops_run_converted_layout_elementwise_add_on_cpu():
    a = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    b = 10 + a
    out = torch.full_like(a, -1)
    layout_in = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    layout_out = gl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    kernel = tilelens.trace("tracer", frontend="gluon")(_converted_layout_add_kernel)

    kernel[(1,)](
        a,
        b,
        out,
        *a.shape,
        *a.stride(),
        *b.stride(),
        *out.stride(),
        layout_in,
        layout_out,
        8,
        8,
        num_warps=4,
    )

    torch.testing.assert_close(out, a + b, atol=0, rtol=0)


@pytest.mark.skipif(
    not _HAS_AMPERE_ASYNC_COPY,
    reason="Gluon Ampere async copy global-to-local builtin is unavailable",
)
def test_gluon_async_copy_runs_masked_1d_copy_on_cpu():
    inp = torch.arange(40, dtype=torch.float32)
    out = torch.full_like(inp, -1)
    kernel = tilelens.trace("tracer", frontend="gluon")(_async_copy_1d_kernel)

    kernel[(1,)](inp, out, inp.numel(), 64, num_warps=4)

    torch.testing.assert_close(out, inp, atol=0, rtol=0)


def test_gluon_amd_async_copy_preserves_masked_other_on_cpu():
    inp = torch.arange(40, dtype=torch.float32)
    out = torch.full((64,), -1, dtype=torch.float32)
    kernel = tilelens.trace("tracer", frontend="gluon")(_amd_async_copy_other_kernel)

    kernel[(1,)](inp, out, inp.numel(), out.numel(), num_warps=4)

    expected = torch.cat([inp, torch.full((24,), -7, dtype=torch.float32)])
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


@pytest.mark.skipif(
    not _HAS_AMPERE_ASYNC_COPY,
    reason="Gluon Ampere async copy global-to-local builtin is unavailable",
)
def test_gluon_async_copy_runs_staged_elementwise_add_on_cpu():
    a = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    b = 10 + a
    out = torch.full_like(a, -1)
    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    kernel = tilelens.trace("tracer", frontend="gluon")(
        _async_copy_elementwise_add_kernel
    )

    kernel[(1,)](
        a,
        b,
        out,
        *a.shape,
        *a.stride(),
        *b.stride(),
        *out.stride(),
        8,
        8,
        smem_layout,
        num_warps=4,
    )

    torch.testing.assert_close(out, a + b, atol=0, rtol=0)


def test_gluon_tma_runs_1d_copy_on_cpu():
    inp = torch.arange(40, dtype=torch.float32)
    out = torch.full_like(inp, -1)
    layout = gl.NVMMASharedLayout.get_default_for([64], gl.float32)
    in_desc = TensorDescriptor.from_tensor(inp, [64], layout)
    out_desc = TensorDescriptor.from_tensor(out, [64], layout)
    _run_gluon_on_cpu(_tma_copy_1d_kernel, (1,), in_desc, out_desc, 64, num_warps=1)

    torch.testing.assert_close(out, inp, atol=0, rtol=0)


def test_gluon_tma_runs_staged_elementwise_add_on_cpu():
    a = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    b = 10 + a
    out = torch.full_like(a, -1)
    layout = gl.NVMMASharedLayout.get_default_for([4, 8], gl.float32)
    a_desc = TensorDescriptor.from_tensor(a, [4, 8], layout)
    b_desc = TensorDescriptor.from_tensor(b, [4, 8], layout)
    out_desc = TensorDescriptor.from_tensor(out, [4, 8], layout)
    _run_gluon_on_cpu(
        _tma_elementwise_add_kernel,
        (1,),
        a_desc,
        b_desc,
        out_desc,
        *a.shape,
        4,
        8,
        num_warps=4,
    )

    torch.testing.assert_close(out, a + b, atol=0, rtol=0)


def test_gluon_tma_runs_float_atomics_on_cpu():
    block_m = 2
    block_n = 4
    src = torch.tensor(
        [[3.0, -2.0, 0.5, 8.0], [1.5, 7.0, -4.0, 0.25]],
        dtype=torch.float32,
    )
    add_dst = torch.arange(40, dtype=torch.float32).reshape(5, 8)
    min_dst = add_dst + 10.0
    max_dst = add_dst - 10.0
    layout = gl.NVMMASharedLayout.get_default_for([block_m, block_n], gl.float32)
    add_desc = TensorDescriptor.from_tensor(add_dst, [block_m, block_n], layout)
    min_desc = TensorDescriptor.from_tensor(min_dst, [block_m, block_n], layout)
    max_desc = TensorDescriptor.from_tensor(max_dst, [block_m, block_n], layout)
    expected_add = add_dst.clone()
    expected_min = min_dst.clone()
    expected_max = max_dst.clone()
    expected_add[1:3, 2:6] += src
    expected_min[1:3, 2:6] = torch.minimum(expected_min[1:3, 2:6], src)
    expected_max[1:3, 2:6] = torch.maximum(expected_max[1:3, 2:6], src)
    _run_gluon_on_cpu(
        _tma_atomic_float_kernel,
        (1,),
        add_desc,
        min_desc,
        max_desc,
        src,
        block_m,
        block_n,
        num_warps=4,
    )

    torch.testing.assert_close(add_dst, expected_add, atol=0, rtol=0)
    torch.testing.assert_close(min_dst, expected_min, atol=0, rtol=0)
    torch.testing.assert_close(max_dst, expected_max, atol=0, rtol=0)


def test_gluon_tma_im2col_runs_simple_tile_on_cpu():
    inp = torch.arange(1, 17, dtype=torch.float32).unsqueeze(1).repeat(1, 32)
    inp = inp.reshape(1, 4, 4, 32)

    out = _run_im2col_case(inp, [0, 0], [0, 0], [0, 0, 0, 0], [0, 0])

    torch.testing.assert_close(out, inp.reshape(16, 32), atol=0, rtol=0)


def test_gluon_tma_im2col_zero_fills_padded_pixels_on_cpu():
    inp = torch.arange(1, 17, dtype=torch.float32).unsqueeze(1).repeat(1, 32)
    inp = inp.reshape(1, 4, 4, 32)

    out = _run_im2col_case(inp, [-1, -1], [-1, -1], [0, -1, -1, 0], [0, 0])

    expected_first_channel = torch.tensor(
        [0, 0, 0, 0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11],
        dtype=torch.float32,
    )
    torch.testing.assert_close(out[:, 0], expected_first_channel, atol=0, rtol=0)


def test_gluon_tma_im2col_honors_runtime_offsets_on_cpu():
    inp = torch.arange(1, 17, dtype=torch.float32).unsqueeze(1).repeat(1, 32)
    inp = inp.reshape(1, 4, 4, 32)

    out = _run_im2col_case(inp, [-1, -1], [-1, -1], [0, -1, -1, 0], [1, 1])

    torch.testing.assert_close(out, inp.reshape(16, 32), atol=0, rtol=0)


def test_gluon_wgmma_runs_small_mma_on_cpu():
    _run_small_wgmma_case(seed=0, lhs_in_reg=False)


def test_gluon_wgmma_runs_small_mma_with_lhs_registers_on_cpu():
    _run_small_wgmma_case(seed=1, lhs_in_reg=True)


def test_gluon_blackwell_tcgen05_runs_small_mma_on_cpu():
    _run_tcgen05_small_mma_case(seed=5, lhs_in_tmem=False, use_commit=True)


def test_gluon_blackwell_tcgen05_runs_small_mma_with_lhs_tensor_memory_on_cpu():
    _run_tcgen05_small_mma_case(seed=6, lhs_in_tmem=True, use_commit=False)


def test_gluon_builder_preserves_tensor_memory_fp4_padding():
    layout = gluon_builder.get_tensor_memory_layout(
        (64, 64),
        1,
        [[1, 0], [0, 1]],
        False,
        True,
    )
    assert layout.fp4_padded is True


def test_gluon_blackwell_tensor_memory_roundtrips_tile_on_cpu():
    inp = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64)
    out = torch.empty_like(inp)
    _run_gluon_on_cpu(
        _tensor_memory_roundtrip_kernel, (1,), inp, out, *inp.shape, num_warps=4
    )

    torch.testing.assert_close(out, inp, atol=0, rtol=0)


def test_gluon_blackwell_tma_runs_gather_on_cpu():
    block_x = 8
    block_y = 8
    y_offset = -2
    inp = torch.arange(6 * 12, dtype=torch.float32).reshape(6, 12)
    x_offsets = torch.tensor([-1, 0, 4, 2, 6, 5, 1, 3], dtype=torch.int32)
    out = torch.full((block_x, block_y), -1.0)
    layout = gl.NVMMASharedLayout.get_default_for([block_x, block_y], gl.float32)
    desc = TensorDescriptor.from_tensor(inp, [1, block_y], layout)
    _run_gluon_on_cpu(
        _blackwell_tma_gather_kernel,
        (1,),
        out,
        *out.stride(),
        desc,
        x_offsets,
        y_offset,
        block_x,
        num_warps=4,
    )

    expected = torch.zeros_like(out)
    for out_row, src_row in enumerate(x_offsets.tolist()):
        for out_col in range(block_y):
            src_col = y_offset + out_col
            if 0 <= src_row < inp.shape[0] and 0 <= src_col < inp.shape[1]:
                expected[out_row, out_col] = inp[src_row, src_col]
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_gluon_blackwell_tma_runs_bitwise_atomics_on_cpu():
    block_m = 2
    block_n = 4
    src = torch.tensor(
        [[0x0F, 0x33, 0x55, 0xAA], [0xF0, 0xCC, 0x5A, 0xA5]],
        dtype=torch.int32,
    )
    base = torch.arange(40, dtype=torch.int32).reshape(5, 8) + 0x80
    and_dst = base.clone()
    or_dst = base.clone()
    xor_dst = base.clone()
    layout = gl.NVMMASharedLayout.get_default_for([block_m, block_n], gl.int32)
    and_desc = TensorDescriptor.from_tensor(and_dst, [block_m, block_n], layout)
    or_desc = TensorDescriptor.from_tensor(or_dst, [block_m, block_n], layout)
    xor_desc = TensorDescriptor.from_tensor(xor_dst, [block_m, block_n], layout)
    expected_and = and_dst.clone()
    expected_or = or_dst.clone()
    expected_xor = xor_dst.clone()
    expected_and[1:3, 1:5] = torch.bitwise_and(expected_and[1:3, 1:5], src)
    expected_or[1:3, 1:5] = torch.bitwise_or(expected_or[1:3, 1:5], src)
    expected_xor[1:3, 1:5] = torch.bitwise_xor(expected_xor[1:3, 1:5], src)
    _run_gluon_on_cpu(
        _blackwell_tma_bitwise_atomic_kernel,
        (1,),
        and_desc,
        or_desc,
        xor_desc,
        src,
        block_m,
        block_n,
        num_warps=4,
    )

    torch.testing.assert_close(and_dst, expected_and, atol=0, rtol=0)
    torch.testing.assert_close(or_dst, expected_or, atol=0, rtol=0)
    torch.testing.assert_close(xor_dst, expected_xor, atol=0, rtol=0)


def test_gluon_blackwell_tma_runs_scatter_on_cpu():
    block_x = 8
    block_y = 8
    y_offset = 6
    out = torch.full((6, 12), -1.0)
    x_offsets = torch.tensor([0, 5, 6, 3, 2, 8, 1, 4], dtype=torch.int32)
    src = torch.arange(block_x * block_y, dtype=torch.float32).reshape(
        block_x,
        block_y,
    )
    layout = gl.NVMMASharedLayout.get_default_for([block_x, block_y], gl.float32)
    desc = TensorDescriptor.from_tensor(out, [1, block_y], layout)
    _run_gluon_on_cpu(
        _blackwell_tma_scatter_kernel,
        (1,),
        desc,
        x_offsets,
        y_offset,
        src,
        *src.stride(),
        block_x,
        num_warps=4,
    )

    expected = torch.full_like(out, -1.0)
    for src_row, dst_row in enumerate(x_offsets.tolist()):
        for src_col in range(block_y):
            dst_col = y_offset + src_col
            if 0 <= dst_row < out.shape[0] and 0 <= dst_col < out.shape[1]:
                expected[dst_row, dst_col] = src[src_row, src_col]
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_gluon_tma_oob_example_reports(capsys):
    example_path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "sanitizer"
        / "gluon_tma_oob.py"
    )
    spec = importlib.util.spec_from_file_location("gluon_tma_oob_example", example_path)
    assert spec is not None
    assert spec.loader is not None
    gluon_tma_oob = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gluon_tma_oob)

    with pytest.raises(SystemExit):
        gluon_tma_oob.run()

    output = capsys.readouterr().out
    assert "ILLEGAL MEMORY ACCESS" in output
    assert "gluon_tma_oob.py" in output
    assert "descriptor_access" in output
