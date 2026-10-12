"""Trace a tiled cuTile matmul with optional visualization and trace export."""

import argparse
import math

import cuda.tile as ct
import numpy as np

import tilelens


TILE_M = 2
TILE_K = 4
TILE_N = 8


@ct.kernel
def matmul_kernel(lhs, rhs, result):
    """Compute matrix multiplication one output tile at a time.

    Args:
        lhs: Left-hand input with shape [M, K].
        rhs: Right-hand input with shape [K, N].
        result: Output with shape [M, N].
    """
    _, K = lhs.shape

    # Compute one output tile per block.
    m, n = ct.bid(0), ct.bid(1)
    result_tile = ct.full((TILE_M, TILE_N), 0, ct.float32)

    # Accumulate partial sums.
    for k in range(ct.cdiv(K, TILE_K)):
        lhs_tile = ct.load(
            lhs,
            (m, k),
            (TILE_M, TILE_K),
            padding_mode=ct.PaddingMode.ZERO,
        )
        rhs_tile = ct.load(
            rhs,
            (k, n),
            (TILE_K, TILE_N),
            padding_mode=ct.PaddingMode.ZERO,
        )
        result_tile = result_tile + ct.matmul(lhs_tile, rhs_tile)

    ct.store(result, (m, n), result_tile)


def _run_demo():
    parser = argparse.ArgumentParser(description="Trace a cuTile matmul on the CPU")
    parser.add_argument(
        "--visualize", action="store_true", help="Open the local visualizer"
    )
    parser.add_argument("--save", metavar="PATH", help="Save the trace as a .tvz file")
    args = parser.parse_args()
    M = 4
    K = 8
    N = 16

    lhs = np.arange(M * K, dtype=np.float32).reshape(M, K)
    rhs = np.arange(K * N, dtype=np.float32).reshape(K, N)
    result = np.empty((M, N), dtype=np.float32)
    kernel_grid = (math.ceil(M / TILE_M), math.ceil(N / TILE_N))

    print("Executing matmul_kernel with the cuTile NumPy interpreter...")
    traced_kernel = tilelens.trace("tracer", frontend="cutile")(matmul_kernel)
    traced_kernel[kernel_grid](lhs, rhs, result)

    expected = lhs @ rhs
    print(np.max(np.abs(expected - result)))
    assert np.allclose(expected, result)
    if args.save:
        tilelens.save(args.save)
    if args.visualize:
        tilelens.launch(share=False)


if __name__ == "__main__":
    _run_demo()
