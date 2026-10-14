import torch
import triton
import triton.language as tl

import tilelens
from tilelens.clients import Tracer


# Simple matmul kernel producing a C = A @ B (fp32, small sizes for demo)
@tilelens.trace(client=Tracer())
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0
        )
        b = tl.load(
            b_ptrs, mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0
        )
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def run_demo():
    torch.manual_seed(0)
    M, N, K = 32, 32, 64
    a = torch.randn((M, K), dtype=torch.float32)
    b = torch.randn((K, N), dtype=torch.float32)
    c = torch.empty((M, N), dtype=torch.float32)

    BLOCK_M, BLOCK_N, BLOCK_K = 16, 16, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # Verify correctness
    ref = a @ b
    assert torch.allclose(c, ref, atol=1e-3), "matmul result mismatch"

    # Launch the viz UI locally at http://localhost:5001 (blocks until Ctrl+C)
    tilelens.launch(share=False, port=5001)


if __name__ == "__main__":
    run_demo()
