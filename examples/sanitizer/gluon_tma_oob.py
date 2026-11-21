import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma

import tilelens
from tilelens.clients.sanitizer.sanitizer import Sanitizer


BLOCK_M = 16
BLOCK_N = 16


@gluon.jit
def gluon_tma_oob_kernel(
    x,
    m: gl.constexpr,
    n: gl.constexpr,
    block_m: gl.constexpr,
    block_n: gl.constexpr,
    layout: gl.constexpr,
):
    desc = tma.make_tensor_descriptor(
        x,
        [m, n],
        [n, 1],
        [block_m, block_n],
        layout,
    )
    smem = gl.allocate_shared_memory(gl.float32, [block_m, block_n], layout)
    bar = mbarrier.allocate_mbarrier()
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, desc.nbytes_per_cta)
    if hasattr(tma, "async_load"):
        tma.async_load(desc, [m, 0], bar, smem)  # OOB: row coordinate starts past x.
    else:
        tma.async_copy_global_to_shared(desc, [m, 0], bar, smem)


def run():
    x = torch.zeros((BLOCK_M, BLOCK_N), dtype=torch.float32)
    layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float32)
    kernel = tilelens.trace(client=Sanitizer(), frontend="gluon")(gluon_tma_oob_kernel)
    return kernel[(1,)](
        x,
        BLOCK_M,
        BLOCK_N,
        BLOCK_M,
        BLOCK_N,
        layout,
        num_warps=4,
    )


if __name__ == "__main__":
    run()
