"""Selection, statistics, loop and partition reductions, predicated copies, HBM DMA, and
dropout in the NKI interpreter, checked against NKI's own CPU simulator
(``nki.simulate_kernel``) on small kernels."""

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


def selection_kernel(x, y, pred):
    # nl.where, nl.equal, nisa.select_reduce with the accumulator carried from one call to
    # the next, and nisa.tensor_copy_predicated.
    sel = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    eq = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.shared_hbm)
    picked = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    maxes = nl.ndarray((x.shape[0], 2), dtype=x.dtype, buffer=nl.shared_hbm)
    copied = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    a = nl.load(x)
    b = nl.load(y)
    p = nl.load(pred)
    nl.store(sel, value=nl.where(a > b, a, -5.0))
    nl.store(eq, value=nl.equal(a, b))
    dst = nl.ndarray(x.shape, dtype=x.dtype)
    res = nl.ndarray((x.shape[0], 2), dtype=x.dtype)
    i_p = nl.arange(x.shape[0])[:, None]
    nisa.select_reduce(
        dst=dst,
        predicate=p,
        on_true=a,
        on_false=-9.0,
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
        reduce_res=res[:, 0],
        reduce_op=np.max,
    )
    nisa.select_reduce(
        dst=dst,
        predicate=p,
        on_true=b,
        on_false=-9.0,
        reverse_pred=True,
        reduce_cmd=nisa.reduce_cmd.reduce,
        reduce_res=res[i_p, 1],  # an indexed slice: a copy in numpy, written through
        reduce_op=np.max,
    )
    nl.store(picked, value=dst)
    nl.store(maxes, value=res)
    c = nl.full(x.shape, 7.0, dtype=x.dtype)
    nisa.tensor_copy_predicated(src=a, dst=c, predicate=p)
    nl.store(copied, value=c)
    return sel, eq, picked, maxes, copied


def stats_kernel(x):
    # nisa.bn_stats over two chunks (the second masked), nisa.bn_aggr, and
    # nisa.tensor_partition_reduce.
    stats = nl.ndarray((x.shape[0], 12), dtype=nl.float32, buffer=nl.shared_hbm)
    mean_var = nl.ndarray((x.shape[0], 2), dtype=nl.float32, buffer=nl.shared_hbm)
    col_sum = nl.ndarray((1, x.shape[1]), dtype=x.dtype, buffer=nl.shared_hbm)
    col_max = nl.ndarray((1, x.shape[1]), dtype=x.dtype, buffer=nl.shared_hbm)
    a = nl.load(x)
    s = nl.ndarray((x.shape[0], 12), dtype=nl.float32)
    i_p = nl.arange(x.shape[0])[:, None]
    i_s = nl.arange(6)[None, :]
    i_f = nl.arange(64)[None, :]
    s[i_p, i_s] = nisa.bn_stats(a[i_p, i_f], dtype=nl.float32)
    s[i_p, 6 + i_s] = nisa.bn_stats(a[i_p, 64 + i_f], mask=(i_f < 37), dtype=nl.float32)
    nl.store(stats, value=s)
    nl.store(mean_var, value=nisa.bn_aggr(s))
    nl.store(col_sum, value=nisa.tensor_partition_reduce(np.add, a))
    nl.store(col_max, value=nisa.tensor_partition_reduce(np.max, a))
    return stats, mean_var, col_sum, col_max


def loop_reduce_kernel(x):
    # nl.loop_reduce into a subscripted destination: dst = op(dst, x), also into one slice
    # of a larger tile per outer iteration.
    total = nl.ndarray((128, x.shape[2]), dtype=nl.float32, buffer=nl.shared_hbm)
    per_j = nl.ndarray((2, 128, x.shape[2]), dtype=nl.float32, buffer=nl.shared_hbm)
    acc = nl.full((128, x.shape[2]), 1.0, dtype=nl.float32, buffer=nl.sbuf)
    for i in nl.affine_range(x.shape[0]):
        acc[:, :] = nl.loop_reduce(nl.load(x[i]), op=np.add, loop_indices=[i])
    nl.store(total, value=acc)
    for j in nl.affine_range(2):
        tile = nl.zeros((128, 2 * x.shape[2]), dtype=nl.float32, buffer=nl.sbuf)
        for i in nl.affine_range(x.shape[0]):
            tile[:, j * x.shape[2] : (j + 1) * x.shape[2]] = nl.loop_reduce(
                nl.load(x[i]), op=np.add, loop_indices=[i], dtype=nl.float32
            )
        nl.store(per_j[j], value=tile[:, j * x.shape[2] : (j + 1) * x.shape[2]])
    return total, per_j


def hbm_dma_kernel(x):
    # nisa._tiled_offloaded_memcpy and nisa._tiled_offloaded_fma between HBM slices.
    copy = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    fma = nl.ndarray((x.shape[0], x.shape[1] // 2), dtype=x.dtype, buffer=nl.shared_hbm)
    h = x.shape[1] // 2
    nisa._tiled_offloaded_memcpy(dst=copy[:, 0:h], src=x[:, 0:h])
    nisa._tiled_offloaded_memcpy(dst=copy[:, h : 2 * h], src=x[:, h : 2 * h])
    nisa._tiled_offloaded_fma(x[:, 0:h], x[:, h : 2 * h], dst=fma, scales=[1.0, 2.0])
    return copy, fma


rng = np.random.default_rng(0)
_x = rng.standard_normal((128, 64)).astype(np.float32)
_y = np.where(rng.random((128, 64)) < 0.5, _x, rng.standard_normal((128, 64))).astype(
    np.float32
)


@pytest.mark.parametrize(
    "kernel, args",
    [
        (selection_kernel, (_x, _y, (rng.random((128, 64)) < 0.5).astype(np.uint8))),
        (stats_kernel, (rng.standard_normal((128, 128)).astype(np.float32),)),
        (loop_reduce_kernel, (rng.standard_normal((4, 128, 64)).astype(np.float32),)),
        (hbm_dma_kernel, (rng.standard_normal((128, 64)).astype(np.float32),)),
    ],
    ids=["selection", "stats", "loop_reduce", "hbm_dma"],
)
def test_kernels_match_nki_simulator(kernel, args):
    tilelens.clear()
    got = tilelens.trace(client=Tracer(), frontend="nki")(kernel)[(1,)](*args)
    want = nki.simulate_kernel(nki.jit(kernel)[(1,)], *args)
    got = got if isinstance(got, tuple) else (got,)
    want = want if isinstance(want, tuple) else (want,)
    assert len(got) == len(want)
    for g, w in zip(got, want):
        np.testing.assert_allclose(
            g.astype(np.float32), w.astype(np.float32), rtol=1e-4, atol=1e-4
        )


def test_hbm_dma_is_traced_as_loads_and_stores():
    tilelens.clear()
    from tilelens.core.data import Load, Store
    from tilelens.core.trace import launches

    x = rng.standard_normal((128, 64)).astype(np.float32)
    tilelens.trace(client=Tracer(), frontend="nki")(hbm_dma_kernel)[(1,)](x)
    records = launches[-1].records
    # Two memcpys (a load and a store each) and an fma of two sources (two loads, a store).
    assert len([r for r in records if isinstance(r, Load)]) == 4
    assert len([r for r in records if isinstance(r, Store)]) == 3


def dropout_kernel(x, rate):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    nl.random_seed(seed=nl.full((1, 1), 3, dtype=nl.int32))
    nl.store(out, value=nl.dropout(nl.load(x), rate=rate))
    return out


def test_dropout_zeroes_without_rescaling():
    # NKI's random numbers differ between its simulator and the hardware; only the
    # distribution is checked.
    x = rng.random((128, 512)).astype(np.float32) + 1.0
    tilelens.clear()
    kept = tilelens.trace(client=Tracer(), frontend="nki")(dropout_kernel)[(1,)](x, 0.0)
    np.testing.assert_array_equal(kept, x)
    tilelens.clear()
    half = tilelens.trace(client=Tracer(), frontend="nki")(dropout_kernel)[(1,)](x, 0.5)
    assert 0.45 < (half == 0).mean() < 0.55
    np.testing.assert_array_equal(half[half != 0], x[half != 0])
    tilelens.clear()
    again = tilelens.trace(client=Tracer(), frontend="nki")(dropout_kernel)[(1,)](
        x, 0.5
    )
    np.testing.assert_array_equal(again, half)  # same seed, same zeros


def all_reduce_kernel(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    shard_id = nl.program_id(0)
    nl.store(out, value=nl.all_reduce(nl.load(x), op=np.add, program_axes=[shard_id]))
    return out


def test_all_reduce_across_programs_is_reported_unsupported():
    tilelens.clear()
    x = rng.standard_normal((128, 64)).astype(np.float32)
    with pytest.raises(NotImplementedError, match="all_reduce"):
        tilelens.trace(client=Tracer(), frontend="nki")(all_reduce_kernel)[(2,)](x)


def gather_kernel(x):
    # `a = t[idx, j]` then `nl.load(a, mask=m)`: an HBM access pattern with computed indices,
    # out of range on lanes the mask turns off.
    out = nl.ndarray((256, x.shape[1]), dtype=x.dtype, buffer=nl.shared_hbm)
    for t in nl.affine_range(2):
        i_p = nl.arange(128)[:, None]
        i_f = nl.arange(x.shape[1])[None, :]
        pos = t * 128 + i_p
        src = nisa.iota(pos, dtype=nl.int32)
        valid = pos < x.shape[0]
        rows = x[src, i_f]
        nl.store(out[pos, i_f], value=nl.load(rows, mask=valid), mask=valid)
    return out


def test_masked_gather_through_an_access_pattern():
    x = rng.standard_normal((200, 64)).astype(np.float32)
    tilelens.clear()
    got = tilelens.trace(client=Tracer(), frontend="nki")(gather_kernel)[(1,)](x)
    want = nki.simulate_kernel(nki.jit(gather_kernel)[(1,)], x)
    np.testing.assert_array_equal(got[:200], want[:200])  # rows 200.. are never written


@nki.jit(mode="trace")
def _scaled_rows(t_hbm, rows, scale):
    # a helper kernel, called by name: TileLens rewrites it as it rewrites the kernel
    return nl.multiply(nl.load(t_hbm[rows, :]), scale)


@nki.jit
def kernel_with_helper(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    for t in nl.affine_range(x.shape[0] // 128):
        rows = nl.ds(t * 128, 128)
        nl.store(out[rows, :], value=_scaled_rows(x, rows, 3.0))
    return out


def test_jit_kernel_and_its_helper_kernels_run_as_written():
    x = rng.standard_normal((256, 32)).astype(np.float32)
    tilelens.clear()
    got = tilelens.trace(client=Tracer(), frontend="nki")(kernel_with_helper)[(1,)](x)
    want = nki.simulate_kernel(kernel_with_helper[(1,)], x)
    np.testing.assert_allclose(got, want, rtol=1e-6)


import neuronxcc.nki.compiler as ncc  # noqa: E402


def tile_idioms_kernel(x, *, scale):
    # a tensor passed by keyword; `t.reshape(shape=...)` of an HBM tensor; a [P] vector
    # indexed as `v[:, 0]`; an activation of a [P] vector with a [P, 1] bias; a predicate
    # subscript `t[mask]`; a range directive; and a bit view of a tile.
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    col = nl.ndarray((x.shape[0], 1), dtype=x.dtype, buffer=nl.shared_hbm)
    bits = nl.ndarray((x.shape[0], 1), dtype=nl.int32, buffer=nl.shared_hbm)
    flat = x.reshape(shape=(x.shape[0], x.shape[1] * x.shape[2]))
    t = nl.load(flat)
    s = nl.load(scale)  # [P, 1]
    rows = nl.ndarray((x.shape[0], 2), dtype=nl.float32)
    v = rows[:, 0]  # a [P] vector
    v[:, 0] = nisa.tensor_reduce(np.add, t, axis=[1])
    e = nisa.activation(op=nl.copy, data=v, bias=s, scale=2.0)
    i_f = nl.arange(t.shape[1])[None, :]
    keep = (
        i_f < t.shape[1]
    )  # every lane on: the predicate only must not change the tile
    for k in nl.sequential_range(2, directives=ncc.multi_buffer(2)):
        nl.store(out.reshape(shape=flat.shape), value=nl.multiply(t[keep], 2.0))
    nl.store(col, value=e)
    nl.store(bits, value=nl.copy(s).view(nl.int32))
    return out, col, bits


def test_tile_idioms_match_nki_simulator():
    x = rng.standard_normal((128, 4, 8)).astype(np.float32)
    scale = rng.standard_normal((128, 1)).astype(np.float32)
    tilelens.clear()
    got = tilelens.trace(client=Tracer(), frontend="nki")(tile_idioms_kernel)[(1,)](
        x, scale=scale
    )
    want = nki.simulate_kernel(nki.jit(tile_idioms_kernel)[(1,)], x, scale=scale)
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, w, rtol=1e-5, atol=1e-5)


def abs_kernel(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    amax = nl.ndarray((x.shape[0], 1), dtype=x.dtype, buffer=nl.shared_hbm)
    acc = nl.ndarray((x.shape[0], 1), dtype=x.dtype)
    t = nisa.tensor_scalar_reduce(
        data=nl.load(x), op0=nl.abs, operand0=0.0, reduce_op=nl.max, reduce_res=acc
    )
    nl.store(out, value=t)
    nl.store(amax, value=acc)
    return out, amax


def test_trn2_target_abs_operator_and_nc_version(monkeypatch):
    from tilelens.core.simulation.nki import nki_builder

    assert nki_builder.get_nc_version() == nisa.nc_version.gen2
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    assert nki_builder.get_nc_version() == nisa.nc_version.gen3
    x = rng.standard_normal((128, 64)).astype(np.float32)
    tilelens.clear()
    got = tilelens.trace(client=Tracer(), frontend="nki")(abs_kernel)[(1,)](x)
    want = nki.simulate_kernel(nki.jit(abs_kernel)[(1,)], x)
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, w)


def rows_kernel(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    rows = nl.ds(nl.program_id(0) * 128, 128)
    nl.store(out[rows, :], value=nl.load(x[rows, :]))
    return out


def test_a_program_a_client_skips_does_not_stop_the_grid():
    # The profiler samples programs: a skipped program must not end the grid.
    class SkipFirstProgram(Tracer):
        def grid_idx_callback(self, grid_idx):
            super().grid_idx_callback(grid_idx)
            self.current = grid_idx

        def pre_run_callback(self, fn):
            return self.current != (0, 0, 0)

    x = rng.standard_normal((384, 16)).astype(np.float32)
    tilelens.clear()
    got = tilelens.trace(client=SkipFirstProgram(), frontend="nki")(rows_kernel)[(3,)](
        x
    )
    np.testing.assert_array_equal(got[128:], x[128:])


def double_row_kernel(lhsT, rhs):
    # nc_matmul with perf_mode="double_row_gen3" (Trn2): [K, 2, M] x [K, 2, N] -> [M, N]
    out = nl.ndarray(
        (lhsT.shape[2], rhs.shape[2]), dtype=nl.float32, buffer=nl.shared_hbm
    )
    res = nl.zeros((lhsT.shape[2], rhs.shape[2]), dtype=nl.float32, buffer=nl.psum)
    res[...] += nisa.nc_matmul(nl.load(lhsT), nl.load(rhs), perf_mode="double_row_gen3")
    nl.store(out, value=res)
    return out


def test_double_row_matmul_matches_nki_simulator(monkeypatch):
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    lhsT = rng.random((128, 2, 128)).astype(nl.float8_e4m3)
    rhs = rng.random((128, 2, 512)).astype(nl.float8_e4m3)
    tilelens.clear()
    got = tilelens.trace(client=Tracer(), frontend="nki")(double_row_kernel)[(1,)](
        lhsT, rhs
    )
    want = nki.simulate_kernel(nki.jit(double_row_kernel)[(1,)], lhsT, rhs)
    np.testing.assert_allclose(got, want, rtol=1e-5)


def double_buffered_kernel(x):
    # Four logical tiles on two physical tiles (ncc.sbuf.mod_alloc), and a tile declared in
    # the loop at a fixed address: as NKI's simulator and the device compute it.
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    bufs = nl.ndarray(
        (4, nl.par_dim(128), x.shape[1]),
        dtype=x.dtype,
        buffer=ncc.sbuf.mod_alloc(base_addr=0, num_free_tiles=(2,)),
    )
    for i in nl.affine_range(4):
        rows = nl.ds(i * 128, 128)
        bufs[i] = nl.load(x[rows, :])
        res = nl.ndarray(
            (nl.par_dim(128), x.shape[1]),
            dtype=x.dtype,
            buffer=ncc.sbuf.mod_alloc(base_addr=2 * x.shape[1] * 4),
        )
        res[...] = nl.exp(bufs[i])
        nl.store(out[rows, :], value=res)
    return out


def test_address_placed_tiles_match_nki_simulator():
    x = rng.standard_normal((512, 64)).astype(np.float32)
    tilelens.clear()
    got = tilelens.trace(client=Tracer(), frontend="nki")(double_buffered_kernel)[(1,)](
        x
    )
    want = nki.simulate_kernel(nki.jit(double_buffered_kernel)[(1,)], x)
    np.testing.assert_allclose(got, want, rtol=1e-6)


def shared_address_kernel(x):
    out = nl.ndarray((2, 128, x.shape[1]), dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(
        (2, nl.par_dim(128), x.shape[1]),
        dtype=x.dtype,
        buffer=ncc.sbuf.mod_alloc(base_addr=0, num_free_tiles=(1,)),
    )
    u = nl.ndarray(
        (nl.par_dim(128), x.shape[1]),
        dtype=x.dtype,
        buffer=ncc.sbuf.mod_alloc(base_addr=0),
    )
    t[0] = nl.load(x)
    nl.store(out[0], value=nl.multiply(t[1], 2.0))  # logical tile 1 is tile 0's memory
    nl.store(out[1], value=nl.multiply(u, 3.0))  # so is another tensor at address 0
    return out


def test_tiles_at_one_address_share_their_values():
    # The device semantics of direct allocation (NKI's simulator gives each tensor its own
    # storage, so it is not the reference here).
    from tilelens.core.simulation.nki import AddressedTiles

    x = rng.standard_normal((128, 64)).astype(np.float32)
    tilelens.clear()
    got = tilelens.trace(client=Tracer(), frontend="nki")(shared_address_kernel)[(1,)](
        x
    )
    np.testing.assert_allclose(got[0], 2 * x, rtol=1e-6)
    np.testing.assert_allclose(got[1], 3 * x, rtol=1e-6)
    from tilelens.core.simulation.nki import nki_builder

    t = nki_builder.ndarray(
        (4, nl.par_dim(128), 8),
        np.float32,
        buffer=ncc.sbuf.mod_alloc(base_addr=0, num_free_tiles=(2,)),
    )
    assert (
        isinstance(t, AddressedTiles)
        and t.shape == (4, 128, 8)
        and t.data.shape == (2, 128, 8)
    )


def psum_reuse_kernel(a, b):
    out = nl.ndarray((2, 128, b.shape[1]), dtype=nl.float32, buffer=nl.shared_hbm)
    x = nl.load(a)
    y = nl.load(b)
    for i in nl.affine_range(2):
        p = nl.ndarray(
            (128, b.shape[1]), dtype=nl.float32, buffer=ncc.psum.mod_alloc(base_bank=3)
        )
        p[...] += nisa.nc_matmul(x, y)  # starts from zero, though the bank was used
        nl.store(out[i], value=nl.copy(p))
    return out


def test_psum_tensors_start_from_zero():
    a = rng.standard_normal((128, 128)).astype(np.float32)
    b = rng.standard_normal((128, 512)).astype(np.float32)
    tilelens.clear()
    got = tilelens.trace(client=Tracer(), frontend="nki")(psum_reuse_kernel)[(1,)](a, b)
    want = nki.simulate_kernel(nki.jit(psum_reuse_kernel)[(1,)], a, b)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
