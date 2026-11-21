"""Kernel fixture for the cross-file LoopSite regression test.

The for-loop below must stay at function-relative line 2 (the line right
after ``def``), matching the inner loop in ``loop_site_cross_file_helper.py``.
With bare-lineno loop identity both loops would map to the same symbolic
iterator variable.
"""

import triton
import triton.language as tl  # noqa: F401  (the interpreter requires tl in the kernel's globals)

import tilelens
from tilelens.clients.sanitizer.sanitizer import SymbolicSanitizer

from .loop_site_cross_file_helper import inner_store_helper

cross_file_loop_sanitizer = SymbolicSanitizer()


@tilelens.trace(client=cross_file_loop_sanitizer)
@triton.jit
def cross_file_outer_loop_kernel(out_ptr):
    for i in range(0, 8):
        inner_store_helper(out_ptr, i)
