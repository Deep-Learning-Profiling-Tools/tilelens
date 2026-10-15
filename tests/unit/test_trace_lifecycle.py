"""CPU-only tests of the traced-launch lifecycle."""

import pytest
import torch
import triton
import triton.language as tl

import tilelens


@pytest.fixture(autouse=True)
def _real_jit():
    # tests/unit/test_multithreading.py sets TRITON_INTERPRET=1 at import
    # time; these tests need @triton.jit to build real JITFunctions.
    from triton import knobs

    with knobs.runtime.scope():
        knobs.runtime.interpret = False
        yield


def test_autotune_over_heuristics_runs_on_copies_of_the_users_chain():
    @triton.autotune(
        configs=[triton.Config({"BLOCK": 4}), triton.Config({"BLOCK": 8})],
        key=["n"],
        cache_results=True,
    )
    @triton.heuristics({"EVEN": lambda args: args["n"] % args["BLOCK"] == 0})
    @triton.jit
    def user(x_ptr, out_ptr, n, BLOCK: tl.constexpr, EVEN: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1, mask=mask)

    # A benchmarker the user's autotuner cached (do_bench is a cached_property).
    user.__dict__["do_bench"] = None
    before = dict(vars(user))
    x, out = torch.arange(8, dtype=torch.float32), torch.zeros(8)

    tilelens.trace("tracer")(user)[(2,)](x, out, 8)

    torch.testing.assert_close(out, x + 1)
    assert vars(user) == before and user.cache == {}
