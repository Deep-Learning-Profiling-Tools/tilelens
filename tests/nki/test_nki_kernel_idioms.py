"""Kernels written in the style of the official NKI samples, run by TileLens and by
NKI's own CPU simulator (``nki.simulate_kernel``); both must give the same outputs."""

import math

import numpy as np
import pytest

try:
    from neuronxcc import nki
    import neuronxcc.nki.language as nl
except ModuleNotFoundError:
    pytest.skip(
        "NeuronX dependencies are missing. Install tilelens[nki] to run these tests.",
        allow_module_level=True,
    )

import tilelens
from tilelens.clients import Tracer

pytestmark = pytest.mark.nki  # only run at "pytest -m nki"


def softmax_kernel(a_tensor):
    # The output is allocated in shared HBM and returned; rows are tiled by 128 and
    # the tail is masked with a [128, 1] mask that broadcasts over the tile.
    out_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    ix = nl.arange(128)[:, None]
    iy = nl.arange(a_tensor.shape[1])[None, :]
    num_rows = a_tensor.shape[0]
    for i in nl.affine_range(math.ceil(num_rows / 128)):
        row_mask = i * 128 + ix < num_rows
        a_tile = nl.load(a_tensor[i * 128 + ix, iy], mask=row_mask)
        max_vals = nl.max(a_tile, axis=[1], keepdims=True, mask=row_mask)
        shifted = nl.subtract(a_tile, max_vals, mask=row_mask)
        numerator = nl.exp(shifted)
        denominator = nl.sum(numerator, axis=[1])  # [128, 1], as in NKI
        nl.store(
            out_tensor[i * 128 + ix, iy], value=numerator / denominator, mask=row_mask
        )
    return out_tensor


def copy_kernel(in_tensor):
    # Whole-tensor load and store, and a load nested in a store.
    out_tensor = nl.ndarray(
        in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.shared_hbm
    )
    tmp = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    nl.store(tmp, value=nl.load(in_tensor))
    ix = nl.arange(in_tensor.shape[0])[:, None]
    iy = nl.arange(in_tensor.shape[1])[None, :]
    nl.store(out_tensor[ix, iy], value=nl.copy(nl.load(tmp[ix, iy])))
    return out_tensor


def spmd_add_kernel(a_input, b_input):
    # One [128, 256] tile per program on a 2-D grid.
    c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)
    ix = nl.program_id(0) * 128 + nl.arange(128)[:, None]
    iy = nl.program_id(1) * 256 + nl.arange(256)[None, :]
    nl.store(
        c_output[ix, iy],
        value=nl.add(nl.load(a_input[ix, iy]), nl.load(b_input[ix, iy])),
    )
    return c_output


def loops_and_casts_kernel(x):
    # static_range / sequential_range loops, par_dim in a shape, static_cast,
    # maximum / minimum / divide, and a reduction over two free axes.
    out = nl.ndarray((x.shape[0], 1), dtype=nl.float32, buffer=nl.shared_hbm)
    acc = nl.zeros((nl.par_dim(x.shape[0]), 1), dtype=nl.float32)
    for k in nl.static_range(2):
        t = nl.load(x[:, k])
        clipped = nl.minimum(nl.maximum(t, -1.0), 1.0)
        acc[...] = nl.add(acc, nl.sum(clipped, axis=[1, 2]))
    for _ in nl.sequential_range(1):
        acc[...] = nl.divide(acc, 2.0)
    nl.store(out, value=nl.static_cast(acc, nl.float32))
    return out


def _tilelens(kernel, grid, *args):
    return tilelens.trace(client=Tracer(), frontend="nki")(kernel)[grid](*args)


def _reference(kernel, grid, *args):
    jitted = nki.jit(kernel)
    return nki.simulate_kernel(jitted[grid] if grid != (1,) else jitted, *args)


rng = np.random.default_rng(0)


@pytest.mark.parametrize(
    "kernel, grid, args",
    [
        (softmax_kernel, (1,), (rng.standard_normal((200, 64)).astype(np.float32),)),
        (copy_kernel, (1,), (rng.standard_normal((16, 32)).astype(np.float32),)),
        (
            spmd_add_kernel,
            (2, 2),
            (
                rng.standard_normal((256, 512)).astype(np.float32),
                rng.standard_normal((256, 512)).astype(np.float32),
            ),
        ),
        (
            loops_and_casts_kernel,
            (1,),
            (rng.standard_normal((8, 2, 4, 3)).astype(np.float32),),
        ),
    ],
    ids=["softmax", "copy", "spmd_add", "loops_and_casts"],
)
def test_kernel_idioms_match_nki_simulator(kernel, grid, args):
    tilelens.clear()
    got = _tilelens(kernel, grid, *args)
    want = _reference(kernel, grid, *args)
    assert isinstance(got, np.ndarray)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
