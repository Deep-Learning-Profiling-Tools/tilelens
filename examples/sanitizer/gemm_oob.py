import torch

import triton
import triton.language as tl

import tilelens


@tilelens.trace("sanitizer")
@triton.jit
def gemm_kernel(
    A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, TILE_SIZE: tl.constexpr
):
    m_block = tl.program_id(0)
    n_block = tl.program_id(1)
    range_m = tl.arange(0, TILE_SIZE)
    range_n = tl.arange(0, TILE_SIZE)
    range_k = tl.arange(0, TILE_SIZE)
    range_m_block = TILE_SIZE * m_block + range_m[:, None]
    range_n_block = TILE_SIZE * n_block + range_n[None, :]
    accum = tl.zeros((TILE_SIZE, TILE_SIZE), dtype=tl.float32)
    for k_block in range(K // TILE_SIZE):
        range_k_block = TILE_SIZE * k_block + range_k
        A_off = K * range_m_block + range_k_block[None, :]
        A_tile = tl.load(A + A_off + 1)  # Out-Of-Bounds Access HERE!

        B_off = N * range_k_block[:, None] + range_n_block
        B_tile = tl.load(B + B_off)

        accum += tl.dot(A_tile, B_tile, allow_tf32=False)
    C_off = N * range_m_block + range_n_block
    tl.store(C + C_off, accum)


def test_gemm():
    M, N, K = 32, 32, 32
    A = torch.randn((M, K))
    B = torch.randn((K, N))
    C = torch.empty((M, N))
    tile_size = 16

    gemm_kernel[(M // tile_size, N // tile_size)](A, B, C, M, N, K, tile_size)
    print("GEMM finished")


test_gemm()
