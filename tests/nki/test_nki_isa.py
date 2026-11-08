"""NKI ISA instructions and language helpers in the interpreter, checked against NKI's own
CPU simulator (``nki.simulate_kernel``) on small kernels."""

import numpy as np
import pytest

try:
    from neuronxcc import nki
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
except ModuleNotFoundError:
    pytest.skip(
        "NeuronX dependencies are missing. Install tilelens[nki] to run these tests.",
        allow_module_level=True,
    )

import tilelens
from tilelens.clients import Tracer

pytestmark = pytest.mark.nki  # only run at "pytest -m nki"


def matmul_kernel(lhsT, rhs):
    # dma_copy between HBM and SBUF, and a Tensor Engine matmul: lhsT.T @ rhs.
    out = nl.ndarray(
        (lhsT.shape[1], rhs.shape[1]), dtype=nl.float32, buffer=nl.shared_hbm
    )
    a = nl.ndarray(lhsT.shape, dtype=lhsT.dtype, buffer=nl.sbuf)
    b = nl.ndarray(rhs.shape, dtype=rhs.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a, src=lhsT)
    nisa.dma_copy(dst=b, src=rhs)
    res = nisa.nc_matmul(a, b)
    nl.store(out, value=nl.copy(res, dtype=nl.float32))
    return out


def vector_kernel(x):
    # activation, tensor_scalar, tensor_tensor, tensor_reduce, affine_select,
    # activation_reduce, tensor_copy, memset, and nc_transpose.
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    sums = nl.ndarray((x.shape[0], 1), dtype=x.dtype, buffer=nl.shared_hbm)
    rows = nl.ndarray((x.shape[0], 1), dtype=x.dtype, buffer=nl.shared_hbm)
    tr = nl.ndarray((x.shape[1], x.shape[0]), dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.load(x)
    bias = nisa.memset((x.shape[0], 1), 0.25, nl.float32)
    a = nisa.activation(op=nl.exp, data=t, bias=bias, scale=0.5)
    s = nisa.tensor_scalar(a, np.multiply, 2.0, op1=np.add, operand1=1.0)
    u = nisa.tensor_tensor(s, t, op=np.subtract)
    nl.store(rows, value=nisa.tensor_reduce(np.add, u, axis=[1]))
    i = nl.arange(x.shape[0])[:, None]
    j = nl.arange(x.shape[1])[None, :]
    lower = nisa.affine_select(i - j >= 0, u, 0.0)
    acc = nl.ndarray((x.shape[0], 1), dtype=nl.float32, buffer=nl.sbuf)
    act = nisa.activation_reduce(
        op=nl.copy, data=lower, reduce_op=np.add, reduce_res=acc
    )
    nl.store(out, value=nisa.tensor_copy(act))
    nl.store(sums, value=acc)
    nl.store(tr, value=nisa.nc_transpose(t))
    return out, sums, rows, tr


def grid_kernel(x):
    # nl.ds slices from program_id, num_programs, full, and ones on a (2,) grid.
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    p = nl.program_id(0)
    n = nl.num_programs(0)
    t = nl.load(x[nl.ds(p * 128, 128), :])
    c = nl.full((128, x.shape[1]), 2.0, dtype=nl.float32)
    o = nl.ones((128, x.shape[1]), dtype=nl.float32)
    nl.store(
        out[nl.ds(p * 128, 128), :], value=nl.add(nl.multiply(t, c), nl.multiply(o, n))
    )
    return out


def dma_slices_kernel(x):
    # dma_copy with subscripted source and destination, HBM to HBM through SBUF.
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    for k in nl.affine_range(x.shape[0] // 128):
        buf = nl.ndarray((128, x.shape[1]), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=buf, src=x[nl.ds(k * 128, 128), :])
        nisa.dma_copy(dst=out[nl.ds(k * 128, 128), :], src=buf)
    return out


def misc_kernel(x, y):
    # scalar_tensor_tensor, floor, mod, iota, dma_transpose of an HBM slice,
    # shared_identity_matrix, and program_ndim.
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    idx = nl.ndarray(x.shape, dtype=nl.int32, buffer=nl.shared_hbm)
    tr = nl.ndarray((y.shape[1], y.shape[0]), dtype=y.dtype, buffer=nl.shared_hbm)
    t = nl.load(x)
    s = nisa.scalar_tensor_tensor(
        data=t, op0=np.multiply, operand0=2.0, op1=np.add, operand1=t
    )
    eye = nl.load(nl.shared_identity_matrix(x.shape[1], dtype=nl.float32))
    nl.store(
        out, value=nl.add(nl.mod(nl.floor(s), 3.0), nl.program_ndim() + eye[0:1, :])
    )
    i = nl.arange(x.shape[0])[:, None]
    j = nl.arange(x.shape[1])[None, :]
    nl.store(idx, value=nisa.iota(i * x.shape[1] + j, dtype=nl.int32))
    nl.store(tr, value=nisa.dma_transpose(y[:, :]))
    return out, idx, tr


from neuronxcc.nki.language import par_dim  # noqa: E402 - kernels import it directly


def partition_kernel(x, scale):
    # par_dim imported directly, nl.mgrid's .p / .x, a 1-D (partition) scale vector,
    # tensor_tensor_scan, nisa.reciprocal, and a [P, 1] result assigned to a [P] slot.
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    tot = nl.ndarray((x.shape[0], 2), dtype=x.dtype, buffer=nl.shared_hbm)
    i = nl.mgrid[0 : x.shape[0], 0 : x.shape[1]]
    t = nl.ndarray((par_dim(x.shape[0]), x.shape[1]), dtype=x.dtype)
    t[i.p, i.x] = nl.load(x[i.p, i.x])
    s = nl.load(scale[0 : x.shape[0], 0])  # 1-D: one value per partition
    a = nisa.activation(op=nl.exp, data=t, scale=s)
    scan = nisa.tensor_tensor_scan(a, t, initial=0, op0=np.multiply, op1=np.add)
    rows = nl.ndarray((par_dim(x.shape[0]), 2), dtype=x.dtype)
    rows[:, 0] = nisa.tensor_reduce(np.add, scan, axis=[1])  # [P, 1] into a [P] column
    rows[:, 1] = nisa.tensor_reduce(np.add, t, axis=[1])
    nl.store(
        out,
        value=nisa.tensor_scalar(
            scan, np.multiply, nisa.reciprocal(nisa.tensor_reduce(np.add, t, axis=[1]))
        ),
    )
    nl.store(tot, value=rows)
    return out, tot


rng = np.random.default_rng(0)


@pytest.mark.parametrize(
    "kernel, grid, args",
    [
        (
            matmul_kernel,
            (1,),
            (
                rng.standard_normal((128, 64)).astype(np.float32),
                rng.standard_normal((128, 512)).astype(np.float32),
            ),
        ),
        (vector_kernel, (1,), (rng.standard_normal((128, 64)).astype(np.float32),)),
        (grid_kernel, (2,), (rng.standard_normal((256, 32)).astype(np.float32),)),
        (dma_slices_kernel, (1,), (rng.standard_normal((256, 32)).astype(np.float32),)),
        (
            misc_kernel,
            (1,),
            (
                (4 * rng.standard_normal((128, 64))).astype(np.float32),
                rng.standard_normal((32, 128)).astype(nl.bfloat16),
            ),
        ),
        (
            partition_kernel,
            (1,),
            (
                rng.random((128, 16)).astype(np.float32) + 0.5,
                (0.1 * rng.standard_normal((128, 2))).astype(np.float32),
            ),
        ),
    ],
    ids=["matmul", "vector", "grid", "dma_slices", "misc", "partition"],
)
def test_isa_kernels_match_nki_simulator(kernel, grid, args):
    tilelens.clear()
    got = tilelens.trace(client=Tracer(), frontend="nki")(kernel)[grid](*args)
    jitted = nki.jit(kernel)
    want = nki.simulate_kernel(jitted[grid], *args)
    got = got if isinstance(got, tuple) else (got,)
    want = want if isinstance(want, tuple) else (want,)
    assert len(got) == len(want)
    for g, w in zip(got, want):
        np.testing.assert_allclose(
            g.astype(np.float32), w.astype(np.float32), rtol=1e-4, atol=1e-4
        )


def test_dma_copy_is_traced_as_a_load_and_a_store():
    tilelens.clear()
    from tilelens.core.data import Load, Store
    from tilelens.core.trace import launches

    x = rng.standard_normal((256, 32)).astype(np.float32)
    tilelens.trace(client=Tracer(), frontend="nki")(dma_slices_kernel)[(1,)](x)
    records = launches[-1].records
    loads = [r for r in records if isinstance(r, Load)]
    stores = [r for r in records if isinstance(r, Store)]
    # Two iterations, each a DMA from HBM to SBUF and one from SBUF to HBM.
    assert len(loads) == 4 and len(stores) == 4


def oob_slice_kernel(x):
    out = nl.ndarray((128, x.shape[1]), dtype=x.dtype, buffer=nl.shared_hbm)
    nl.store(
        out, value=nl.load(x[0:128, 1 : x.shape[1] + 1])
    )  # one column past the end
    return out


def test_out_of_range_slice_raises_like_nki():
    # numpy would truncate the slice to the tensor; NKI keeps the tile and reports the access.
    tilelens.clear()
    x = rng.standard_normal((128, 32)).astype(np.float32)
    with pytest.raises(IndexError):
        nki.simulate_kernel(nki.jit(oob_slice_kernel)[(1,)], x)
    with pytest.raises(IndexError):
        tilelens.trace(client=Tracer(), frontend="nki")(oob_slice_kernel)[(1,)](x)
