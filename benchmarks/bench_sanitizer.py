#!/usr/bin/env python
"""Sanitizer performance benchmark for CI.

Runs a set of Triton kernels through the SymbolicSanitizer and records
wall-clock timing.  Four modes:

  Run (default)          → execute benchmarks, write JSON
  Compare                → compare two JSON files, emit markdown table
  Compare-single         → emit markdown table from a single JSON file
  Triton-kernels pytest  → run selected Triton-kernels E2E benchmarks

Usage:
  python benchmarks/bench_sanitizer.py --output results.json
  python benchmarks/bench_sanitizer.py --compare base.json pr.json
  python benchmarks/bench_sanitizer.py --compare-single pr.json
  python benchmarks/bench_sanitizer.py --triton-kernels-output-prefix /tmp/bench-pr
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl

import tilelens
from tilelens.clients.sanitizer.sanitizer import (
    SymbolicSanitizer,
    _fn_symbolic_cache_set,
)

# ---------------------------------------------------------------------------
# Benchmark timeout helper
# ---------------------------------------------------------------------------

BENCH_TIMEOUT = 180  # seconds per benchmark


class BenchmarkTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise BenchmarkTimeout("Benchmark exceeded timeout")


# ---------------------------------------------------------------------------
# Sanitizer instances (one per kernel, following test_sanitizer.py pattern)
# ---------------------------------------------------------------------------

gemm_sanitizer = SymbolicSanitizer()
gemm_oob_sanitizer = SymbolicSanitizer()
indirect_sanitizer = SymbolicSanitizer()
nested_sanitizer = SymbolicSanitizer()
block_ptr_sanitizer = SymbolicSanitizer()
jsd_sanitizer = SymbolicSanitizer()
element_mul_sanitizer = SymbolicSanitizer()
flaggems_sanitizer = SymbolicSanitizer()
swiglu_fwd_sanitizer = SymbolicSanitizer()
swiglu_bwd_sanitizer = SymbolicSanitizer()
cross_entropy_sanitizer = SymbolicSanitizer()

# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@tilelens.trace(client=gemm_sanitizer)
@triton.jit
def gemm_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TILE_SIZE: tl.constexpr,
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
        A_tile = tl.load(A + A_off)
        B_off = N * range_k_block[:, None] + range_n_block
        B_tile = tl.load(B + B_off)
        accum += tl.dot(A_tile, B_tile, allow_tf32=False)
    C_off = N * range_m_block + range_n_block
    tl.store(C + C_off, accum)


@tilelens.trace(client=gemm_oob_sanitizer)
@triton.jit
def gemm_oob_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TILE_SIZE: tl.constexpr,
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
        A_tile = tl.load(A + A_off + 1)  # intentional OOB
        B_off = N * range_k_block[:, None] + range_n_block
        B_tile = tl.load(B + B_off)
        accum += tl.dot(A_tile, B_tile, allow_tf32=False)
    C_off = N * range_m_block + range_n_block
    tl.store(C + C_off, accum)


@tilelens.trace(client=indirect_sanitizer)
@triton.jit
def indirect_load_kernel(idx_ptr, src_ptr, dst_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    indices = tl.load(idx_ptr + offs)
    vals = tl.load(src_ptr + indices)
    tl.store(dst_ptr + offs, vals)


@tilelens.trace(client=nested_sanitizer)
@triton.jit
def nested_loop_kernel(out_ptr, OUTER: tl.constexpr, INNER: tl.constexpr):
    for i in range(0, OUTER):
        for j in range(0, INNER):
            idx = i * INNER + j
            tl.store(out_ptr + idx, idx)


@tilelens.trace(client=block_ptr_sanitizer)
@triton.jit
def block_pointer_loop_advance_kernel(ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(N,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    for _ in range(N // BLOCK):
        tl.load(block_ptr, boundary_check=(0,))
        block_ptr = tl.advance(block_ptr, (BLOCK,))


# ---------------------------------------------------------------------------
# SwiGLU kernels (from Liger-Kernel ops/swiglu.py)
# ---------------------------------------------------------------------------


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@tilelens.trace(client=swiglu_fwd_sanitizer)
@triton.jit
def swiglu_forward_kernel(
    a_ptr, b_ptr, c_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """SwiGLU forward: c = silu(a) * b, one program per row."""
    program_id = tl.program_id(0).to(tl.int64)

    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    c_row = silu(a_row) * b_row
    tl.store(c_ptr + col_offsets, c_row, mask=mask)


@tilelens.trace(client=swiglu_bwd_sanitizer)
@triton.jit
def swiglu_backward_kernel(
    dc_ptr, a_ptr, b_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """SwiGLU backward: compute da and db in-place, one program per row."""
    program_id = tl.program_id(0).to(tl.int64)

    dc_ptr += program_id * stride
    a_ptr += program_id * stride
    b_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dc_row = tl.load(dc_ptr + col_offsets, mask=mask, other=0)
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)

    # Recomputation to save memory
    sig_a = tl.sigmoid(a_row)
    silu_a = a_row * sig_a
    db_row = dc_row * silu_a
    da_row = dc_row * (silu_a * (1 - sig_a) + sig_a) * b_row

    tl.store(a_ptr + col_offsets, da_row, mask=mask)
    tl.store(b_ptr + col_offsets, db_row, mask=mask)


# ---------------------------------------------------------------------------
# Cross Entropy kernel (from Liger-Kernel ops/cross_entropy.py)
# ---------------------------------------------------------------------------


@tilelens.trace(client=cross_entropy_sanitizer)
@triton.jit
def liger_cross_entropy_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    loss_ptr,
    loss_stride,
    n_cols,
    n_non_ignore,
    ignore_index,
    label_smoothing: tl.constexpr,
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused cross entropy: computes loss and stores gradient in-place in X_ptr.
    Uses online softmax (Algorithm 3, https://arxiv.org/pdf/1805.02867).
    One program per row (batch * seq_len).
    """
    program_id = tl.program_id(0).to(tl.int64)

    # 1. Load target; skip if ignore_index
    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    X_ptr += program_id * X_stride

    if y == ignore_index:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return

    loss_ptr += program_id * loss_stride

    # 2. Online softmax first pass: find max + sum
    m = float("-inf")
    d = 0.0
    ori_X_y = tl.load(X_ptr + y)

    scaled_x_sum = 0.0
    eps = label_smoothing / n_cols

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf")
        )
        block_max = tl.max(X_block)
        if label_smoothing > 0:
            scaled_x_sum += tl.sum(tl.where(X_offsets < n_cols, -eps * X_block, 0.0))
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    # 3. Second pass: compute gradients (softmax - label) in-place
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf")
        )
        if reduction == "mean":
            X_block = (tl.exp(X_block - m) / d - eps) / (n_non_ignore)
        else:
            X_block = tl.exp(X_block - m) / d - eps
        tl.store(X_ptr + X_offsets, X_block, mask=X_offsets < n_cols)

    tl.debug_barrier()

    # 4. Calculate loss
    loss = -(ori_X_y - m - tl.log(d))

    if label_smoothing > 0:
        smooth_loss = scaled_x_sum + label_smoothing * (m + tl.log(d))
        loss = loss * (1 - label_smoothing) + smooth_loss

    if reduction == "mean":
        loss = loss / n_non_ignore

    # 5. Special handling for the y-th element gradient
    X_y = tl.load(X_ptr + y)
    if reduction == "mean":
        X_y += -(1 - label_smoothing) / (n_non_ignore)
    else:
        X_y += -(1 - label_smoothing)

    tl.store(loss_ptr, loss)
    tl.store(X_ptr + y, X_y)


@tilelens.trace(client=jsd_sanitizer)
@triton.jit
def jsd_kernel(
    X_ptr,  # input in logspace, X = log Q
    X_stride,
    Y_ptr,  # ground truth in logspace, Y = log P
    Y_stride,
    loss_ptr,
    loss_stride,
    dX_ptr,
    dX_stride,
    label_ptr,
    beta,
    n_non_ignore: int,
    ignore_index: tl.constexpr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    HAS_LABEL: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    X_ptr += pid * X_stride
    dX_ptr += pid * dX_stride
    Y_ptr += pid * Y_stride
    loss_ptr += pid * loss_stride
    label_ptr += pid

    if HAS_LABEL:
        label = tl.load(label_ptr)
        if label == ignore_index:
            for i in range(0, n_cols, BLOCK_SIZE):
                offsets = i + tl.arange(0, BLOCK_SIZE)
                tl.store(dX_ptr + offsets, 0.0, mask=offsets < n_cols)
            return

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        X = tl.load(X_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)
        Y = tl.load(Y_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)

        Q = tl.exp(X)
        P = tl.exp(Y)
        M = beta * P + (1 - beta) * Q
        log_M = tl.log(M)

        loss = beta * P * Y + (1 - beta) * Q * X - M * log_M
        loss = loss / n_non_ignore
        tl.store(loss_ptr + offsets, loss, mask=mask)

        dX = (1 - beta) * Q * (X - log_M) / n_non_ignore
        tl.store(dX_ptr + offsets, dX, mask=mask)


@tilelens.trace(client=element_mul_sanitizer)
@triton.jit
def element_mul_kernel(
    X_ptr,
    X_stride,
    grad_output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    X_ptr += program_id * X_stride

    grad_output = tl.load(grad_output_ptr)

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols)
        tl.store(X_ptr + X_offsets, X_block * grad_output, mask=X_offsets < n_cols)


# ---------------------------------------------------------------------------
# FlagGems LayerNorm kernels
# Adapted from FlagGems/src/flag_gems/ops/layernorm.py
# ---------------------------------------------------------------------------


@tilelens.trace(client=flaggems_sanitizer)
@triton.jit
def flaggems_ln_persistent_kernel(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,
    out_rstd_ptr,
    M,
    N,
    eps,
    TILE_N: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    n_offsets = tl.arange(0, TILE_N)
    mask = n_offsets < N

    x = tl.load(in_ptr + pid * N + n_offsets, mask, other=0.0).to(tl.float32)
    m = tl.sum(x) / N
    d = x - m
    s = tl.where(mask, d * d, 0)
    sum_square = tl.sum(s)
    var = sum_square / N
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(out_mean_ptr + pid, m)
    tl.store(out_rstd_ptr + pid, rstd)

    if weight_ptr is None:
        w = 1
    else:
        w = tl.load(weight_ptr + n_offsets, mask=mask)
    if bias_ptr is None:
        b = 0
    else:
        b = tl.load(bias_ptr + n_offsets, mask=mask)
    out = (x - m) * rstd * w + b
    tl.store(out_ptr + pid * N + n_offsets, out, mask=mask)


@tilelens.trace(client=flaggems_sanitizer)
@triton.jit
def flaggems_ln_multiline_kernel(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,
    out_rstd_ptr,
    M,
    N,
    eps,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    m_offsets = pid * TILE_M + tl.arange(0, TILE_M)
    m_mask = m_offsets < M

    n_offsets = tl.arange(0, TILE_N)[None, :]
    n_mask = n_offsets < N
    mask = m_mask[:, None] & n_mask

    x = tl.load(in_ptr + m_offsets[:, None] * N + n_offsets, mask, other=0.0).to(
        tl.float32
    )
    m = tl.sum(x, axis=1) / N
    d = x - m[:, None]
    s = tl.where(mask, d * d, 0)
    sum_square = tl.sum(s, axis=1)
    var = sum_square / N
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(out_mean_ptr + m_offsets, m, mask=m_mask)
    tl.store(out_rstd_ptr + m_offsets, rstd, mask=m_mask)

    if weight_ptr is None:
        w = 1
    else:
        w = tl.load(weight_ptr + n_offsets, mask=n_mask)
    if bias_ptr is None:
        b = 0
    else:
        b = tl.load(bias_ptr + n_offsets, mask=n_mask)
    out = (x - m[:, None]) * rstd[:, None] * w + b
    tl.store(out_ptr + m_offsets[:, None] * N + n_offsets, out, mask=mask)


@tilelens.trace(client=flaggems_sanitizer)
@triton.jit
def flaggems_ln_loop_kernel(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,
    out_rstd_ptr,
    M,
    N,
    eps,
    TILE_N: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)

    # Welford online mean/variance
    m = tl.zeros((TILE_N,), dtype=tl.float32)
    s = tl.zeros((TILE_N,), dtype=tl.float32)
    cnt = tl.zeros((TILE_N,), dtype=tl.int32)
    num_steps = tl.cdiv(N, TILE_N)
    for step in range(0, num_steps - 1, 1):
        start_n = step * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * N + n_offsets).to(tl.float32)
        new_m = m + (x - m) / (step + 1)
        new_s = s + (x - new_m) * (x - m)
        cnt += 1
        m = new_m
        s = new_s

    # last step with masking
    for step in range(num_steps - 1, num_steps, 1):
        start_n = step * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(in_ptr + pid * N + n_offsets, mask=mask).to(tl.float32)
        new_m = tl.where(mask, m + (x - m) / (step + 1), m)
        new_s = tl.where(mask, s + (x - new_m) * (x - m), s)
        cnt += mask.to(tl.int32)
        m = new_m
        s = new_s

    final_m = tl.sum(m * cnt) / N
    var = tl.sum(s + cnt * (m - final_m) * (m - final_m)) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    m = final_m
    tl.store(out_mean_ptr + pid, m)
    tl.store(out_rstd_ptr + pid, rstd)

    # Normalize - reverse sweep (inlined prev_multiple_of)
    prev_multiple = tl.cdiv(N, TILE_N) * TILE_N - TILE_N
    # first step with masking
    for start_n in range(0, TILE_N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(in_ptr + pid * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
        if weight_ptr is None:
            w = 1
        else:
            w = tl.load(weight_ptr + n_offsets, mask=mask)
        if bias_ptr is None:
            b = 0
        else:
            b = tl.load(bias_ptr + n_offsets, mask=mask)
        out = w * (x - m) * rstd + b
        tl.store(out_ptr + pid * N + n_offsets, out, mask=mask)

    for start_n in range(TILE_N, N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * N + n_offsets).to(tl.float32)
        if weight_ptr is None:
            w = 1
        else:
            w = tl.load(weight_ptr + n_offsets)
        if bias_ptr is None:
            b = 0
        else:
            b = tl.load(bias_ptr + n_offsets)
        out = w * (x - m) * rstd + b
        tl.store(out_ptr + pid * N + n_offsets, out)


@tilelens.trace(client=flaggems_sanitizer)
@triton.jit
def flaggems_ln_backward_kernel(
    dY,
    X,
    W,
    Mean,
    Rstd,
    dX,
    M,
    N,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    pid = (
        tl.program_id(0).to(tl.int64) * BLOCK_ROW_SIZE
        + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    )
    row_mask = pid < M
    dY += pid * N
    X += pid * N
    dX += pid * N
    Mean += pid
    Rstd += pid

    mean = tl.load(Mean).to(tl.float32)
    rstd = tl.load(Rstd).to(tl.float32)

    dx_part2 = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    dx_part3 = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)

    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)
        col_mask = cols[None, :] < N
        mask = row_mask & col_mask
        dy = tl.load(dY + cols[None, :], mask).to(tl.float32)
        x = tl.load(X + cols[None, :], mask).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        x_hat = x * rstd
        if W is None:
            w = 1
        else:
            w = tl.load(W + cols, mask=cols < N).to(tl.float32)
        dx_hat = dy * w
        dx_part2 += dx_hat
        dx_part3 += dx_hat * x_hat

    dx_2 = tl.sum(dx_part2, axis=1)[:, None]
    dx_3 = tl.sum(dx_part3, axis=1)[:, None]

    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)
        col_mask = cols[None, :] < N
        mask = row_mask & col_mask
        dy = tl.load(dY + cols[None, :], mask).to(tl.float32)
        x = tl.load(X + cols[None, :], mask).to(tl.float32)
        if W is None:
            w = 1
        else:
            w = tl.load(W + cols, mask=cols < N).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        x_hat = x * rstd
        dx_hat = dy * w
        dx = rstd * (dx_hat - (dx_2 + x_hat * dx_3) / N)
        tl.store(dX + cols, dx, mask=mask)


@tilelens.trace(client=flaggems_sanitizer)
@triton.jit
def flaggems_ln_wb_backward_kernel(
    dY,
    X,
    Mean,
    Rstd,
    dW,
    dB,
    M,
    N,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    pid = (
        tl.program_id(0).to(tl.int64) * BLOCK_COL_SIZE
        + tl.arange(0, BLOCK_COL_SIZE)[None, :]
    )
    col_mask = pid < N
    dY += pid
    X += pid
    accW = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    accB = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    for off in range(0, M, BLOCK_ROW_SIZE):
        rows = off + tl.arange(0, BLOCK_ROW_SIZE)
        row_mask = rows[:, None] < M
        mask = row_mask & col_mask
        dy = tl.load(dY + rows[:, None] * N, mask).to(tl.float32)
        x = tl.load(X + rows[:, None] * N, mask).to(tl.float32)
        mean = tl.load(Mean + rows, mask=rows < M)[:, None].to(tl.float32)
        rstd = tl.load(Rstd + rows, mask=rows < M)[:, None].to(tl.float32)
        x = tl.where(col_mask, x - mean, 0.0)
        x_hat = x * rstd
        accW += dy * x_hat
        accB += dy
    dw = tl.sum(accW, axis=0)
    tl.store(dW + pid, dw[None, :], mask=col_mask)
    db = tl.sum(accB, axis=0)
    tl.store(dB + pid, db[None, :], mask=col_mask)


# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------


def _reset_sanitizer(san: SymbolicSanitizer):
    """Clear sanitizer state and the global cache so re-runs are not skipped."""
    _fn_symbolic_cache_set.clear()
    san.records.clear()


# ---------------------------------------------------------------------------
# GEMM benchmarks: multiple (M, N, K, TILE_SIZE) constexpr combos
# Each unique combo is a separate symbolic cache entry.
# ---------------------------------------------------------------------------

_GEMM_CONFIGS = [
    # (M, N, K, TILE_SIZE)
    (64, 64, 64, 16),
    (128, 128, 128, 16),
    (64, 128, 64, 16),
    (128, 64, 128, 16),
    (64, 64, 64, 32),
    (128, 128, 128, 32),
    (64, 128, 128, 32),
]


def _gemm_setup_all():
    return [
        {
            "A": torch.randn(M, K, dtype=torch.float32),
            "B": torch.randn(K, N, dtype=torch.float32),
            "C": torch.empty(M, N, dtype=torch.float32),
            "M": M,
            "N": N,
            "K": K,
            "TILE": TILE,
        }
        for M, N, K, TILE in _GEMM_CONFIGS
    ]


def _gemm_run_all(configs):
    for d in configs:
        grid = (d["M"] // d["TILE"], d["N"] // d["TILE"])
        gemm_kernel[grid](
            d["A"],
            d["B"],
            d["C"],
            M=d["M"],
            N=d["N"],
            K=d["K"],
            TILE_SIZE=d["TILE"],
        )


def _gemm_oob_setup_all():
    return [
        {
            "A": torch.randn(M, K, dtype=torch.float32),
            "B": torch.randn(K, N, dtype=torch.float32),
            "C": torch.empty(M, N, dtype=torch.float32),
            "M": M,
            "N": N,
            "K": K,
            "TILE": TILE,
        }
        for M, N, K, TILE in _GEMM_CONFIGS
    ]


def _gemm_oob_run_all(configs):
    for d in configs:
        grid = (d["M"] // d["TILE"], d["N"] // d["TILE"])
        gemm_oob_kernel[grid](
            d["A"],
            d["B"],
            d["C"],
            M=d["M"],
            N=d["N"],
            K=d["K"],
            TILE_SIZE=d["TILE"],
        )


# ---------------------------------------------------------------------------
# Nested loop benchmarks: multiple (OUTER, INNER) constexpr combos
# ---------------------------------------------------------------------------

_NESTED_CONFIGS = [
    # (OUTER, INNER)
    (4, 8),
    (8, 4),
    (4, 16),
    (16, 4),
    (8, 8),
    (8, 16),
    (16, 8),
]


def _nested_setup_all():
    max_elems = max(o * i for o, i in _NESTED_CONFIGS)
    return {"out": torch.empty(max_elems, dtype=torch.int32)}


def _nested_run_all(data):
    for outer, inner in _NESTED_CONFIGS:
        nested_loop_kernel[(1,)](data["out"], OUTER=outer, INNER=inner)


BENCHMARKS: dict[str, dict[str, Any]] = {
    "gemm": {
        "setup": _gemm_setup_all,
        "run": _gemm_run_all,
        "sanitizer": gemm_sanitizer,
    },
    "gemm_oob": {
        "setup": _gemm_oob_setup_all,
        "run": _gemm_oob_run_all,
        "sanitizer": gemm_oob_sanitizer,
    },
    "indirect_load": {
        "setup": lambda: {
            "idx": torch.arange(1024, dtype=torch.int32),
            "src": torch.randn(1024, dtype=torch.float32),
            "dst": torch.empty(1024, dtype=torch.float32),
        },
        "run": lambda d: indirect_load_kernel[(16,)](
            d["idx"], d["src"], d["dst"], BLOCK=64
        ),
        "sanitizer": indirect_sanitizer,
    },
    "nested_loop": {
        "setup": _nested_setup_all,
        "run": _nested_run_all,
        "sanitizer": nested_sanitizer,
    },
    "block_pointer_loop_advance": {
        "setup": lambda: {
            "data": torch.randn(4096, dtype=torch.float32),
        },
        "run": lambda d: block_pointer_loop_advance_kernel[(16,)](
            d["data"], N=4096, BLOCK=64
        ),
        "sanitizer": block_ptr_sanitizer,
    },
}

# ---------------------------------------------------------------------------
# Liger-Kernel fused_linear_jsd: 12 parameter combinations (grouped)
# Matches test_correctness_functional parameter space exactly
# ---------------------------------------------------------------------------

_LIGER_SHAPES = [
    # Grid = B * T; keep moderate to avoid sanitizer timeout.
    (2, 256, 512, 1600),
    (1, 512, 256, 2048),
    (1, 512, 128, 1024),
]
_LIGER_DTYPES = [
    (0.5, torch.bfloat16),
    (0.5, torch.float32),
]
_LIGER_PARAMS = [
    (1.0, 0.5, -100),  # default
    (2.0, 0.1, 42),  # custom
]

MAX_FUSED_SIZE = 65536 // 2


def _make_liger_setup(BT, H, V, scalar, dtype, beta, ignore_index):
    def setup():
        BLOCK_SIZE_JSD = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
        student_log_probs = torch.log_softmax(
            torch.randn(BT, V, dtype=dtype) * scalar, dim=-1
        ).contiguous()
        teacher_log_probs = torch.log_softmax(
            torch.randn(BT, V, dtype=dtype) * scalar, dim=-1
        ).contiguous()
        loss = torch.zeros(BT, V, dtype=torch.float32)
        dX = torch.empty(BT, V, dtype=dtype)
        labels = torch.randint(0, V, (BT,), dtype=torch.long)
        labels[: max(1, BT // 2)] = ignore_index
        n_non_ignore = int((labels != ignore_index).sum().item())

        H_student = H // 2
        BLOCK_SIZE_MUL = min(MAX_FUSED_SIZE, triton.next_power_of_2(H_student))
        grad_input = torch.randn(BT, H_student, dtype=dtype).contiguous()
        grad_output = torch.tensor(0.5, dtype=dtype)

        return {
            "BT": BT,
            "V": V,
            "student_log_probs": student_log_probs,
            "teacher_log_probs": teacher_log_probs,
            "loss": loss,
            "dX": dX,
            "labels": labels,
            "BLOCK_SIZE_JSD": BLOCK_SIZE_JSD,
            "n_non_ignore": n_non_ignore,
            "beta": beta,
            "ignore_index": ignore_index,
            "grad_input": grad_input,
            "grad_output": grad_output,
            "H_student": H_student,
            "BLOCK_SIZE_MUL": BLOCK_SIZE_MUL,
        }

    return setup


def _liger_run(d):
    jsd_kernel[(d["BT"],)](
        X_ptr=d["student_log_probs"],
        X_stride=d["student_log_probs"].stride(0),
        Y_ptr=d["teacher_log_probs"],
        Y_stride=d["teacher_log_probs"].stride(0),
        loss_ptr=d["loss"],
        loss_stride=d["loss"].stride(0),
        dX_ptr=d["dX"],
        dX_stride=d["dX"].stride(0),
        label_ptr=d["labels"],
        beta=d["beta"],
        n_non_ignore=d["n_non_ignore"],
        ignore_index=d["ignore_index"],
        n_cols=d["V"],
        BLOCK_SIZE=d["BLOCK_SIZE_JSD"],
        HAS_LABEL=True,
    )
    element_mul_kernel[(d["BT"],)](
        d["grad_input"],
        d["grad_input"].stride(0),
        d["grad_output"],
        d["H_student"],
        BLOCK_SIZE=d["BLOCK_SIZE_MUL"],
    )


_LIGER_CONFIGS = [
    (_B * _T, _H, _V, _scalar, _dtype, _beta, _ignore_index)
    for _B, _T, _H, _V in _LIGER_SHAPES
    for _scalar, _dtype in _LIGER_DTYPES
    for _temperature, _beta, _ignore_index in _LIGER_PARAMS
]


def _liger_setup_all():
    return [
        _make_liger_setup(BT, H, V, scalar, dtype, beta, ignore_index)()
        for BT, H, V, scalar, dtype, beta, ignore_index in _LIGER_CONFIGS
    ]


def _liger_run_all(configs):
    for d in configs:
        _liger_run(d)


BENCHMARKS["liger_jsd"] = {
    "setup": _liger_setup_all,
    "run": _liger_run_all,
    "sanitizer": [jsd_sanitizer, element_mul_sanitizer],
}

# ---------------------------------------------------------------------------
# FlagGems LayerNorm: 15 parameter combinations (grouped)
# Matches test_accuracy_layernorm parameter space exactly
# ---------------------------------------------------------------------------

_FLAGGEMS_LN_SHAPES = [
    # M determines grid size; keep moderate to avoid sanitizer timeout.
    # N values chosen to cover all three kernel variants (multiline/persistent/loop).
    (64, 36),
    (1, 40999),
    (64, 256),
]
_FLAGGEMS_LN_DTYPES = [torch.float32]
_FLAGGEMS_LN_WB = [True]  # wb_none=True only (with weight/bias errors)

_LN_BWD_BLOCK_ROW = 4
_LN_BWD_BLOCK_COL = 1024
_LN_WB_BWD_BLOCK_ROW = 8
_LN_WB_BWD_BLOCK_COL = 1024
_LN_LOOP_TILE_N = 1024


def _make_flaggems_ln_setup(M, N, dtype, wb_none):
    def setup():
        inp = torch.randn(M, N, dtype=dtype)
        out = torch.empty_like(inp)
        mean = torch.empty(M, dtype=torch.float32)
        rstd = torch.empty(M, dtype=torch.float32)
        weight = None if wb_none else torch.randn(N, dtype=dtype)
        bias = None if wb_none else torch.randn(N, dtype=dtype)

        # Backward tensors
        out_grad = torch.randn(M, N, dtype=dtype)
        in_grad = torch.empty_like(inp)
        weight_grad = None if wb_none else torch.empty(N, dtype=dtype)
        bias_grad = None if wb_none else torch.empty(N, dtype=dtype)

        # Forward grid and tile sizes
        if N <= 128:
            TILE_N = triton.next_power_of_2(N)
            TILE_M = triton.cdiv(1024, TILE_N)
            fwd_grid = (triton.cdiv(M, TILE_M),)
            kernel_variant = "multiline"
        elif N <= 4096:
            TILE_N = triton.next_power_of_2(N)
            TILE_M = 0
            fwd_grid = (M,)
            kernel_variant = "persistent"
        else:
            TILE_N = _LN_LOOP_TILE_N
            TILE_M = 0
            fwd_grid = (M,)
            kernel_variant = "loop"

        bwd_grid = (triton.cdiv(M, _LN_BWD_BLOCK_ROW),)
        wb_bwd_grid = (triton.cdiv(N, _LN_WB_BWD_BLOCK_COL),)

        return {
            "M": M,
            "N": N,
            "inp": inp,
            "out": out,
            "mean": mean,
            "rstd": rstd,
            "weight": weight,
            "bias": bias,
            "out_grad": out_grad,
            "in_grad": in_grad,
            "weight_grad": weight_grad,
            "bias_grad": bias_grad,
            "kernel_variant": kernel_variant,
            "TILE_N": TILE_N,
            "TILE_M": TILE_M,
            "fwd_grid": fwd_grid,
            "bwd_grid": bwd_grid,
            "wb_bwd_grid": wb_bwd_grid,
            "wb_none": wb_none,
        }

    return setup


def _flaggems_ln_run(d):
    M, N = d["M"], d["N"]
    eps = 1e-5

    # Forward
    if d["kernel_variant"] == "multiline":
        flaggems_ln_multiline_kernel[d["fwd_grid"]](
            d["inp"],
            d["out"],
            d["weight"],
            d["bias"],
            d["mean"],
            d["rstd"],
            M,
            N,
            eps,
            d["TILE_M"],
            d["TILE_N"],
        )
    elif d["kernel_variant"] == "persistent":
        flaggems_ln_persistent_kernel[d["fwd_grid"]](
            d["inp"],
            d["out"],
            d["weight"],
            d["bias"],
            d["mean"],
            d["rstd"],
            M,
            N,
            eps,
            d["TILE_N"],
        )
    else:
        flaggems_ln_loop_kernel[d["fwd_grid"]](
            d["inp"],
            d["out"],
            d["weight"],
            d["bias"],
            d["mean"],
            d["rstd"],
            M,
            N,
            eps,
            d["TILE_N"],
        )

    # Backward
    flaggems_ln_backward_kernel[d["bwd_grid"]](
        d["out_grad"],
        d["inp"],
        d["weight"],
        d["mean"],
        d["rstd"],
        d["in_grad"],
        M,
        N,
        BLOCK_ROW_SIZE=_LN_BWD_BLOCK_ROW,
        BLOCK_COL_SIZE=_LN_BWD_BLOCK_COL,
    )
    if not d["wb_none"]:
        flaggems_ln_wb_backward_kernel[d["wb_bwd_grid"]](
            d["out_grad"],
            d["inp"],
            d["mean"],
            d["rstd"],
            d["weight_grad"],
            d["bias_grad"],
            M,
            N,
            BLOCK_ROW_SIZE=_LN_WB_BWD_BLOCK_ROW,
            BLOCK_COL_SIZE=_LN_WB_BWD_BLOCK_COL,
        )


def _flaggems_layernorm_setup_all():
    return [
        _make_flaggems_ln_setup(M, N, dtype, wb_none)()
        for M, N in _FLAGGEMS_LN_SHAPES
        for dtype in _FLAGGEMS_LN_DTYPES
        for wb_none in _FLAGGEMS_LN_WB
    ]


def _flaggems_layernorm_run_all(configs):
    for d in configs:
        _flaggems_ln_run(d)


BENCHMARKS["flaggems_layernorm"] = {
    "setup": _flaggems_layernorm_setup_all,
    "run": _flaggems_layernorm_run_all,
    "sanitizer": flaggems_sanitizer,
}

# ---------------------------------------------------------------------------
# SwiGLU benchmarks: 8 parameter combinations (grouped, template pattern)
# Matches test_correctness_llamamlp parameter space.
# Tensors pre-allocated once; backward tensors cloned per-run (kernel writes
# gradients in-place into a_ptr/b_ptr).
# ---------------------------------------------------------------------------

_SWIGLU_SHAPES = [
    # (bsz, seq_len, hidden_size, intermediate_size)
    # Grid = bsz * seq_len; keep moderate to avoid sanitizer timeout.
    (1, 4, 256, 512),
    (1, 4, 256, 1024),
    (1, 4, 256, 2048),
    (1, 4, 256, 4096),
    (1, 4, 256, 8192),
    (1, 4, 128, 768),
    (1, 4, 128, 1536),
    (1, 4, 128, 3072),
    (1, 4, 128, 6144),
    (1, 4, 64, 256),
    (1, 4, 64, 640),
]
_SWIGLU_DTYPES = [torch.float32]


def _swiglu_setup_all():
    templates = []
    for bsz, seq_len, _hidden_size, intermediate_size in _SWIGLU_SHAPES:
        N_ROWS = bsz * seq_len
        N_COLS = intermediate_size
        BLOCK_SIZE = triton.next_power_of_2(N_COLS)
        for dtype in _SWIGLU_DTYPES:
            templates.append(
                {
                    "N_ROWS": N_ROWS,
                    "N_COLS": N_COLS,
                    "BLOCK_SIZE": BLOCK_SIZE,
                    # Forward: a, b are read-only; c is output
                    "a": torch.randn(N_ROWS, N_COLS, dtype=dtype),
                    "b": torch.randn(N_ROWS, N_COLS, dtype=dtype),
                    "c": torch.empty(N_ROWS, N_COLS, dtype=dtype),
                    # Backward templates: dc is read-only; a_bwd/b_bwd are
                    # overwritten in-place so we clone them per-run
                    "dc": torch.randn(N_ROWS, N_COLS, dtype=dtype),
                    "a_bwd_template": torch.randn(N_ROWS, N_COLS, dtype=dtype),
                    "b_bwd_template": torch.randn(N_ROWS, N_COLS, dtype=dtype),
                }
            )
    return templates


def _swiglu_run_all(templates):
    for t in templates:
        N_ROWS, N_COLS, BLOCK_SIZE = t["N_ROWS"], t["N_COLS"], t["BLOCK_SIZE"]
        swiglu_forward_kernel[(N_ROWS,)](
            t["a"],
            t["b"],
            t["c"],
            t["c"].stride(0),
            n_cols=N_COLS,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        a_bwd = t["a_bwd_template"].clone()
        b_bwd = t["b_bwd_template"].clone()
        swiglu_backward_kernel[(N_ROWS,)](
            t["dc"],
            a_bwd,
            b_bwd,
            t["dc"].stride(0),
            n_cols=N_COLS,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        del a_bwd, b_bwd


BENCHMARKS["swiglu"] = {
    "setup": _swiglu_setup_all,
    "run": _swiglu_run_all,
    "sanitizer": [swiglu_fwd_sanitizer, swiglu_bwd_sanitizer],
}

# ---------------------------------------------------------------------------
# Cross Entropy benchmarks: 12 parameter combinations (grouped, streaming)
# Matches test_correctness shape/reduction/dtype space (scalar dropped — does
# not affect symbolic analysis).
#
# Memory strategy: setup stores only lightweight metadata.  run() allocates
# one (BT, V, dtype) template at a time, runs both reductions with cloned X
# (kernel overwrites X in-place), then frees the template before moving to
# the next.  Peak ≈ one template + one clone ≈ 4.2 GB (for V=128256 f32),
# well within CI's ~7 GB limit.
#
# ---------------------------------------------------------------------------

_CE_SHAPES = [
    # (B, T, V) → (BT=B*T, V)  — deduplicated
    # Grid = BT; keep moderate to avoid sanitizer timeout.
    (1, 4, 512),
    (1, 4, 1024),
    (1, 4, 2048),
    (1, 4, 4096),
    (1, 4, 8192),
    (1, 4, 16384),
    (1, 4, 32000),
]
_CE_REDUCTIONS = ["sum", "mean"]
_CE_LABEL_SMOOTHINGS = [0.0, 0.1]
_CE_DTYPES = [torch.float32]


def _ce_setup_all():
    # Store only metadata — no large tensors.
    configs = []
    for B, T, V in _CE_SHAPES:
        BT = B * T
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
        for dtype in _CE_DTYPES:
            configs.append({"BT": BT, "V": V, "BLOCK_SIZE": BLOCK_SIZE, "dtype": dtype})
    return configs


def _ce_run_all(configs):
    for cfg in configs:
        BT, V, BLOCK_SIZE, dtype = (
            cfg["BT"],
            cfg["V"],
            cfg["BLOCK_SIZE"],
            cfg["dtype"],
        )
        X_template = torch.randn(BT, V, dtype=dtype)
        Y = torch.randint(0, V, (BT,), dtype=torch.long)
        loss = torch.zeros(BT, dtype=torch.float32)

        for reduction in _CE_REDUCTIONS:
            for label_smoothing in _CE_LABEL_SMOOTHINGS:
                X = X_template.clone()
                liger_cross_entropy_kernel[(BT,)](
                    X_ptr=X,
                    X_stride=X.stride(0),
                    Y_ptr=Y,
                    Y_stride=Y.stride(0),
                    loss_ptr=loss,
                    loss_stride=loss.stride(0),
                    n_cols=V,
                    n_non_ignore=BT,
                    ignore_index=-100,
                    label_smoothing=label_smoothing,
                    reduction=reduction,
                    BLOCK_SIZE=BLOCK_SIZE,
                )
                del X
        del X_template, Y, loss


BENCHMARKS["cross_entropy"] = {
    "setup": _ce_setup_all,
    "run": _ce_run_all,
    "sanitizer": cross_entropy_sanitizer,
}

# ---------------------------------------------------------------------------
# Fused Linear JSD benchmarks: 12 parameter combinations (grouped, template pattern)
# Matches test_correctness parameter space.
# Reuses existing jsd_kernel + element_mul_kernel.
# Tensors pre-allocated once; student_log_probs/dX/grad_input cloned per-run
# (jsd_kernel writes dX in-place; element_mul overwrites grad_input).
# ---------------------------------------------------------------------------

_FLJSD_SHAPES = [
    # (B, T, H, V)
    # Grid = B * T; keep moderate to avoid sanitizer timeout.
    (1, 4, 512, 800),
    (1, 4, 512, 1600),
    (1, 4, 512, 3200),
    (1, 4, 256, 1024),
    (1, 4, 256, 2048),
    (1, 4, 256, 4096),
    (1, 4, 128, 512),
    (1, 4, 128, 1024),
    (1, 4, 128, 2048),
]
_FLJSD_DTYPES = [torch.float32]
_FLJSD_PARAMS = [
    (1.0, 0.5),  # (temperature, beta)
    (2.0, 0.1),
]


def _fljsd_setup_all():
    templates = []
    for B, T, H, V in _FLJSD_SHAPES:
        BT = B * T
        BLOCK_SIZE_JSD = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
        H_student = H // 2
        BLOCK_SIZE_MUL = min(MAX_FUSED_SIZE, triton.next_power_of_2(H_student))
        for dtype in _FLJSD_DTYPES:
            student_log_probs = torch.log_softmax(
                torch.randn(BT, V, dtype=dtype), dim=-1
            ).contiguous()
            teacher_log_probs = torch.log_softmax(
                torch.randn(BT, V, dtype=dtype), dim=-1
            ).contiguous()
            labels = torch.randint(0, V, (BT,), dtype=torch.long)
            labels[: max(1, BT // 2)] = -100
            n_non_ignore = int((labels != -100).sum().item())

            for _temperature, beta in _FLJSD_PARAMS:
                templates.append(
                    {
                        "BT": BT,
                        "V": V,
                        "BLOCK_SIZE_JSD": BLOCK_SIZE_JSD,
                        "H_student": H_student,
                        "BLOCK_SIZE_MUL": BLOCK_SIZE_MUL,
                        "student_template": student_log_probs,
                        "teacher_log_probs": teacher_log_probs,
                        "loss": torch.zeros(BT, V, dtype=torch.float32),
                        "dX_template": torch.empty(BT, V, dtype=dtype),
                        "labels": labels,
                        "n_non_ignore": n_non_ignore,
                        "beta": beta,
                        "grad_template": torch.randn(
                            BT, H_student, dtype=dtype
                        ).contiguous(),
                        "grad_output": torch.tensor(0.5, dtype=dtype),
                    }
                )
    return templates


def _fljsd_run_all(templates):
    for t in templates:
        student = t["student_template"].clone()
        dX = t["dX_template"].clone()
        grad_input = t["grad_template"].clone()
        jsd_kernel[(t["BT"],)](
            X_ptr=student,
            X_stride=student.stride(0),
            Y_ptr=t["teacher_log_probs"],
            Y_stride=t["teacher_log_probs"].stride(0),
            loss_ptr=t["loss"],
            loss_stride=t["loss"].stride(0),
            dX_ptr=dX,
            dX_stride=dX.stride(0),
            label_ptr=t["labels"],
            beta=t["beta"],
            n_non_ignore=t["n_non_ignore"],
            ignore_index=-100,
            n_cols=t["V"],
            BLOCK_SIZE=t["BLOCK_SIZE_JSD"],
            HAS_LABEL=True,
        )
        element_mul_kernel[(t["BT"],)](
            grad_input,
            grad_input.stride(0),
            t["grad_output"],
            t["H_student"],
            BLOCK_SIZE=t["BLOCK_SIZE_MUL"],
        )
        del student, dX, grad_input


BENCHMARKS["fused_linear_jsd"] = {
    "setup": _fljsd_setup_all,
    "run": _fljsd_run_all,
    "sanitizer": [jsd_sanitizer, element_mul_sanitizer],
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


TRITON_KERNEL_BENCHMARKS = [
    (
        "triton_kernels._masked_compaction",
        "masked-compaction",
        "test_tilelens_sanitizer_masked_compaction",
    ),
    (
        "triton_kernels._ragged_tensor_metadata_memset_compute",
        "ragged-metadata-make",
        "test_tilelens_sanitizer_make_ragged_tensor_metadata",
    ),
    (
        "triton_kernels._remap_ragged_tensor_metadata",
        "ragged-metadata-remap",
        "test_tilelens_sanitizer_remap_ragged_tensor_metadata",
    ),
    (
        "triton_kernels._topk_forward",
        "topk-forward",
        "test_tilelens_sanitizer_topk_forward",
    ),
    (
        "triton_kernels._topk_backward",
        "topk-backward",
        "test_tilelens_sanitizer_topk_backward",
    ),
    ("triton_kernels._swiglu", "swiglu", "test_tilelens_sanitizer_swiglu"),
    (
        "triton_kernels.upcast_mxfp4_tile",
        "mxfp4-tile-upcast",
        "test_tilelens_sanitizer_mxfp4_tile_upcast",
    ),
]


def _result_entry(times: list[float], error: str | None = None) -> dict[str, Any]:
    if error is not None:
        return {
            "times": [],
            "mean": None,
            "median": None,
            "min": None,
            "stddev": None,
            "error": error,
        }
    return {
        "times": times,
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "stddev": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


def _pytest_benchmark_result(
    command: list[str], cwd: Path, iterations: int
) -> dict[str, Any]:
    times: list[float] = []
    for i in range(iterations):
        print(f"  iteration {i + 1}/{iterations}: {' '.join(command)}", flush=True)
        t0 = time.perf_counter()
        completed = subprocess.run(command, cwd=cwd, text=True)
        elapsed = time.perf_counter() - t0
        if completed.returncode != 0:
            return _result_entry(
                times,
                error=f"Command exited with status {completed.returncode}",
            )
        times.append(elapsed)
    return _result_entry(times)


def _write_single_benchmark_result(
    name: str, result: dict[str, Any], output: Path, iterations: int
) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "iterations": iterations,
        "warmup_iterations": 0,
        "benchmarks": {name: result},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results written to {output}")


def run_triton_kernel_benchmarks(
    cwd: str,
    output_prefix: str,
    round_id: str,
    iterations: int = 1,
    device: str = "cpu",
) -> None:
    cwd_path = Path(cwd).resolve()
    test_file = cwd_path / "tests/end_to_end/test_triton_kernels.py"
    if not test_file.exists():
        print(f"Skipping Triton-kernels benchmarks; missing {test_file}")
        return

    pytest_command = (
        ["uv", "run", "pytest"]
        if shutil.which("uv")
        else [sys.executable, "-m", "pytest"]
    )
    for name, slug, test_name in TRITON_KERNEL_BENCHMARKS:
        nodeid = f"tests/end_to_end/test_triton_kernels.py::{test_name}"
        command = [
            *pytest_command,
            nodeid,
            "-q",
            f"--triton-kernels-device={device}",
        ]
        output = Path(f"{output_prefix}-{slug}-{round_id}.json")
        result = _pytest_benchmark_result(command, cwd_path, iterations)
        _write_single_benchmark_result(name, result, output, iterations)


def run_benchmarks(
    warmup: int = 1,
    iterations: int = 20,
) -> dict[str, Any]:
    results: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "iterations": iterations,
        "warmup_iterations": warmup,
        "benchmarks": {},
    }

    for name, bench in BENCHMARKS.items():
        print(f"  [{name}] ", end="", flush=True)

        data = bench["setup"]()
        sanitizers = bench["sanitizer"]
        if not isinstance(sanitizers, list):
            sanitizers = [sanitizers]

        def _reset_all():
            for san in sanitizers:
                _reset_sanitizer(san)

        # Set per-benchmark timeout (Unix only)
        has_alarm = hasattr(signal, "SIGALRM")
        if has_alarm:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(BENCH_TIMEOUT)

        try:
            # Warmup
            for _ in range(warmup):
                _reset_all()
                bench["run"](data)

            # Measured iterations
            times: list[float] = []
            for _ in range(iterations):
                _reset_all()
                t0 = time.perf_counter()
                bench["run"](data)
                t1 = time.perf_counter()
                times.append(t1 - t0)

            results["benchmarks"][name] = {
                "times": times,
                "mean": statistics.mean(times),
                "median": statistics.median(times),
                "min": min(times),
                "stddev": statistics.stdev(times) if len(times) > 1 else 0.0,
            }
            min_str = f"{min(times):.3f}s"
            print(f"{min_str}")

        except BenchmarkTimeout:
            print(f"TIMEOUT (>{BENCH_TIMEOUT}s)")
            results["benchmarks"][name] = {
                "times": [],
                "mean": None,
                "median": None,
                "min": None,
                "stddev": None,
                "error": f"Timeout after {BENCH_TIMEOUT}s",
            }

        except Exception as e:
            print(f"ERROR: {e}")
            results["benchmarks"][name] = {
                "times": [],
                "mean": None,
                "median": None,
                "min": None,
                "stddev": None,
                "error": str(e),
            }

        finally:
            if has_alarm:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

    return results


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

REGRESSION_THRESHOLD = 0.05  # 5%


def _fmt_time(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val:.3f}s"


def _fmt_change(base_val: float | None, pr_val: float | None) -> str:
    if base_val is None or pr_val is None:
        return "N/A"
    if base_val == 0:
        return "N/A"
    pct = (pr_val - base_val) / base_val
    sign = "+" if pct >= 0 else ""
    flag = " :warning:" if pct > REGRESSION_THRESHOLD else ""
    return f"{sign}{pct:.1%}{flag}"


def _fmt_samples(bench: dict) -> str:
    times = bench.get("times")
    if not isinstance(times, list):
        return "N/A"
    return str(len(times))


def _has_regression(base_val: float | None, pr_val: float | None) -> bool:
    if base_val is None or pr_val is None or base_val == 0:
        return False
    return (pr_val - base_val) / base_val > REGRESSION_THRESHOLD


def generate_comparison(base: dict, pr: dict) -> str:
    lines = [
        "## Performance Benchmark",
        "",
        "| Benchmark | main (min) | PR (min) | Change | Samples |",
        "|-----------|------------|----------|--------|---------|",
    ]

    all_names = list(pr.get("benchmarks", {}).keys())
    base_total = 0.0
    pr_total = 0.0
    base_total_valid = True
    pr_total_valid = True
    any_regression = False

    for name in all_names:
        pr_bench = pr["benchmarks"].get(name, {})
        base_bench = base.get("benchmarks", {}).get(name, {})
        pr_val = pr_bench.get("min")
        base_val = base_bench.get("min")

        if _has_regression(base_val, pr_val):
            any_regression = True

        if pr_val is not None:
            pr_total += pr_val
        else:
            pr_total_valid = False
        if base_val is not None:
            base_total += base_val
        else:
            base_total_valid = False

        lines.append(
            f"| {name} | {_fmt_time(base_val)} | {_fmt_time(pr_val)} "
            f"| {_fmt_change(base_val, pr_val)} "
            f"| {_fmt_samples(base_bench)} / {_fmt_samples(pr_bench)} |"
        )

    # Total row
    bt = base_total if base_total_valid else None
    pt = pr_total if pr_total_valid else None
    lines.append(
        f"| **Total** | **{_fmt_time(bt)}** | **{_fmt_time(pt)}** "
        f"| {_fmt_change(bt, pt)} | N/A |"
    )

    lines.append("")
    if any_regression:
        lines.append(
            f"_Threshold: >{REGRESSION_THRESHOLD:.0%} regression flagged with :warning:_"
        )
    lines.append(
        f"_Iterations: {pr.get('warmup_iterations', '?')} warmup + {pr.get('iterations', '?')} measured_"
    )
    lines.append(
        "_Samples are shown as main / PR; long pytest benchmarks may use fewer samples._"
    )
    return "\n".join(lines)


def generate_single(pr: dict) -> str:
    lines = [
        "## Performance Benchmark",
        "",
        "_No baseline available (benchmark script not present on main)._",
        "",
        "| Benchmark | PR (min) | Samples |",
        "|-----------|----------|---------|",
    ]

    pr_total = 0.0
    pr_total_valid = True

    for name, bench in pr.get("benchmarks", {}).items():
        val = bench.get("min")
        if val is not None:
            pr_total += val
        else:
            pr_total_valid = False
        lines.append(f"| {name} | {_fmt_time(val)} | {_fmt_samples(bench)} |")

    pt = pr_total if pr_total_valid else None
    lines.append(f"| **Total** | **{_fmt_time(pt)}** | N/A |")

    lines.append("")
    lines.append(
        f"_Iterations: {pr.get('warmup_iterations', '?')} warmup + {pr.get('iterations', '?')} measured_"
    )
    lines.append("_Long pytest benchmarks may use fewer samples._")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def merge_results(file_paths: list[str]) -> dict[str, Any]:
    """Merge multiple benchmark JSON files by concatenating times arrays."""
    with open(file_paths[0]) as f:
        merged = json.load(f)

    for path in file_paths[1:]:
        with open(path) as f:
            data = json.load(f)
        for name, bench in data.get("benchmarks", {}).items():
            if name in merged.get("benchmarks", {}):
                merged["benchmarks"][name]["times"].extend(bench.get("times", []))
            else:
                merged["benchmarks"][name] = bench

    # Recompute stats from merged times
    total_iters = 0
    for bench in merged["benchmarks"].values():
        times = bench.get("times", [])
        if times:
            bench["mean"] = statistics.mean(times)
            bench["median"] = statistics.median(times)
            bench["min"] = min(times)
            bench["stddev"] = statistics.stdev(times) if len(times) > 1 else 0.0
            total_iters = max(total_iters, len(times))

    merged["iterations"] = total_iters
    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Sanitizer performance benchmark")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations")
    parser.add_argument(
        "--iterations", type=int, default=20, help="Measured iterations"
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASE_JSON", "PR_JSON"),
        help="Compare two result files and print markdown",
    )
    parser.add_argument(
        "--compare-single",
        metavar="PR_JSON",
        help="Print markdown table from a single result file (no baseline)",
    )
    parser.add_argument(
        "--merge",
        nargs="+",
        metavar="JSON_FILE",
        help="Merge multiple result JSON files into one (use with --output)",
    )
    parser.add_argument(
        "--triton-kernels-output-prefix",
        help="Run Triton-kernels pytest benchmarks and write per-kernel JSON files",
    )
    parser.add_argument(
        "--triton-kernels-cwd",
        default=".",
        help="Repository checkout where Triton-kernels pytest benchmarks run",
    )
    parser.add_argument(
        "--triton-kernels-round",
        default="1",
        help="Round suffix for Triton-kernels benchmark output files",
    )
    parser.add_argument(
        "--triton-kernels-device",
        choices=("auto", "cuda", "cpu"),
        default="cpu",
        help="Device mode for Triton-kernels pytest benchmarks",
    )
    parser.add_argument(
        "--triton-kernels-iterations",
        type=int,
        default=1,
        help="Measured iterations for each Triton-kernels pytest benchmark",
    )

    args = parser.parse_args()

    if args.triton_kernels_output_prefix:
        if args.triton_kernels_iterations < 1:
            parser.error("--triton-kernels-iterations must be >= 1")
        run_triton_kernel_benchmarks(
            cwd=args.triton_kernels_cwd,
            output_prefix=args.triton_kernels_output_prefix,
            round_id=args.triton_kernels_round,
            iterations=args.triton_kernels_iterations,
            device=args.triton_kernels_device,
        )
        return

    # Compare mode
    if args.compare:
        with open(args.compare[0]) as f:
            base = json.load(f)
        with open(args.compare[1]) as f:
            pr = json.load(f)
        print(generate_comparison(base, pr))
        return

    # Single mode
    if args.compare_single:
        with open(args.compare_single) as f:
            pr = json.load(f)
        print(generate_single(pr))
        return

    # Merge mode
    if args.merge:
        result = merge_results(args.merge)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Merged {len(args.merge)} files → {args.output}")
        else:
            print(json.dumps(result, indent=2))
        return

    # Run mode
    print("Running sanitizer benchmarks...")
    results = run_benchmarks(warmup=args.warmup, iterations=args.iterations)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.output}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
