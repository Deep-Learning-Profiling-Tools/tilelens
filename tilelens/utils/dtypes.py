"""Dtype aliases and canonical storage dtypes for custom interpreters."""

from typing import TypeAlias

import ml_dtypes
import numpy as np

DTypeLike: TypeAlias = (
    str | type | np.dtype | None
)  # e.g. np.float32, "bfloat16", None (default)

STORAGE_DTYPES: dict[DTypeLike, np.dtype | None] = {
    None: None,
    int: np.dtype(np.int64),
    float: np.dtype(np.float64),
    "int8": np.dtype(np.int8),
    "int16": np.dtype(np.int16),
    "int32": np.dtype(np.int32),
    "uint8": np.dtype(np.uint8),
    "uint16": np.dtype(np.uint16),
    "uint32": np.dtype(np.uint32),
    "float16": np.dtype(np.float16),
    "half": np.dtype(np.float16),
    "float32": np.dtype(np.float32),
    "float": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
    "double": np.dtype(np.float64),
    "bfloat16": np.dtype(ml_dtypes.bfloat16),
    "float8_e4m3": np.dtype(ml_dtypes.float8_e4m3),
    "float8_e5m2": np.dtype(ml_dtypes.float8_e5m2),
    "float8_e4m3fn": np.dtype(ml_dtypes.float8_e4m3fn),
    "float4_e2m1fn": np.dtype(ml_dtypes.float4_e2m1fn),
}
STORAGE_DTYPES |= {dtype: dtype for dtype in STORAGE_DTYPES.values()}
