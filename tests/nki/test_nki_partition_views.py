"""Compare the opt-in view semantics with the installed SDK's actual tensor views."""
import itertools
import numpy as np
import pytest
import nki.language as nl
from triton_viz.core.simulation import nki_beta2 as b2
from triton_viz.core.simulation.nki_isa_contracts import (
    ISAContractError, checked_operation, partition_contract_checks,
)

sdk_tensor = pytest.importorskip("nki.language.tensor").NkiTensor
SDK_BUFFERS = {"sbuf": nl.sbuf, "psum": nl.psum, "hbm": nl.hbm}
pytestmark = pytest.mark.nki


def cases():
    for shape in ((2, 3), (2, 3, 4), (2, 3, 4, 5), (128, 2, 3)):
        keys = list(itertools.product((0, -1, slice(None)), repeat=len(shape)))
        keys += [0, -1, (Ellipsis, 0), (0, Ellipsis),
                 (slice(None),) * (len(shape) - 1) + (slice(None, None, 2),)]
        for buffer, key in itertools.product(SDK_BUFFERS, keys):
            yield shape, buffer, key


@pytest.mark.parametrize("shape,buffer,key", list(cases()))
def test_view_matches_sdk_shape_offset_values_and_storage(shape, buffer, key):
    expected = sdk_tensor(shape, nl.float32, None, SDK_BUFFERS[buffer])[key]
    value = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    with partition_contract_checks():
        if buffer in ("sbuf", "psum"):
            tile = b2.ndarray(shape, nl.float32, buffer=getattr(b2, buffer))
            tile.data[...] = value
        else:
            tile = b2.NDArray(value=value, buffer="hbm")
        actual = tile[key]
    assert actual.shape == expected.shape
    assert np.shares_memory(actual.data, tile.data)
    assert (actual.data_ptr() - tile.data_ptr()) // 4 == expected.offset
    coordinates = np.indices(expected.shape)
    flat_indices = expected.offset + sum(c * s for c, s in zip(coordinates, expected.strides))
    np.testing.assert_array_equal(actual.data, value.reshape(-1)[flat_indices])
    # Unit-axis strides are observationally irrelevant and NumPy canonicalizes them.
    for extent, sdk_stride, actual_stride in zip(expected.shape, expected.strides, actual.stride()):
        if extent > 1:
            assert actual_stride == sdk_stride


def test_matmul_reports_retained_single_partition_instead_of_numpy_rank_drop():
    with partition_contract_checks():
        tiles = b2.ndarray((2, 128, 128), nl.float32, buffer=b2.sbuf)
        stationary = tiles[0]
        moving = b2.ndarray((128, 128), nl.float32, buffer=b2.sbuf)
        dst = b2.ndarray((128, 128), nl.float32, buffer=b2.psum)
        def multiply(dst, stationary, moving):
            return dst
        checked = checked_operation("nc_matmul", multiply)
        with pytest.raises(ISAContractError) as caught:
            checked(dst, stationary, moving)
        finding = caught.value.finding
        assert finding["kind"] == "matmul_partition_mismatch"
        assert finding["stationary_extent"] == 1
        assert finding["moving_extent"] == 128
        assert finding["stationary_shape"] == [1, 128, 128]
    # Historical interpretation remains unchanged outside the opt-in scope.
    legacy = b2.ndarray((2, 128, 128), nl.float32, buffer=b2.sbuf)
    assert legacy[0].shape == (128, 128)


def test_opt_in_dma_preserves_elements_across_layouts_before_matmul_check():
    source = b2.NDArray(value=np.arange(128 * 128, dtype=np.float32).reshape(128, 128), buffer="hbm")
    from triton_viz.core.frontend.base import _LangPatchScope
    import nki.isa as nisa

    scope = _LangPatchScope()
    try:
        with partition_contract_checks():
            b2.nki_patch_lang(scope)
            tiles = b2.ndarray((2, 128, 128), nl.float32, buffer=b2.sbuf)
            view = tiles[0]
            assert view.shape == (1, 128, 128)
            nisa.dma_copy(dst=view, src=source)
            np.testing.assert_array_equal(view.data.reshape(128, 128), source.data)
            moving = b2.NDArray(value=np.ones((128, 128), np.float32), buffer="sbuf")
            dst = b2.NDArray(value=np.zeros((128, 128), np.float32), buffer="psum")
            with pytest.raises(ISAContractError) as caught:
                nisa.nc_matmul(dst=dst, stationary=view, moving=moving)
            assert caught.value.finding["stationary_extent"] == 1
    finally:
        scope.restore()


def test_sdk_dma_python_validation_accepts_equal_count_layouts(monkeypatch):
    """Exercise actual SDK validation; only backend materialization is stubbed."""
    import nki.isa._copy as sdk_copy
    import nki.isa._enums as sdk_enums
    from types import SimpleNamespace

    calls = []
    monkeypatch.setattr(sdk_enums, "get_backend", lambda: SimpleNamespace(get_nc_version=lambda: 2))
    monkeypatch.setattr(sdk_copy, "get_backend", lambda: SimpleNamespace(dma_copy=lambda **kw: calls.append(kw)))
    monkeypatch.setattr(sdk_copy, "tensor_ap_view", lambda tensor: tensor.shape)
    source = sdk_tensor((128, 128), nl.float32, None, SDK_BUFFERS["hbm"])
    target = sdk_tensor((2, 128, 128), nl.float32, None, SDK_BUFFERS["sbuf"])[0]
    # __wrapped__ omits only public API stack-capture bookkeeping.
    sdk_copy.dma_copy.__wrapped__(dst=target, src=source)
    assert len(calls) == 1
    assert calls[0]["src"] == (128, 128)
    assert calls[0]["dst"] == (1, 128, 128)
    bad = sdk_tensor((128, 127), nl.float32, None, SDK_BUFFERS["hbm"])
    with pytest.raises(AssertionError, match="same number of elements"):
        sdk_copy.dma_copy.__wrapped__(dst=target, src=bad)


def test_legal_single_partition_matmul_runs_instead_of_rejecting_all_selection():
    with partition_contract_checks():
        source = b2.NDArray(value=np.arange(6, dtype=np.float32).reshape(2, 3), buffer="sbuf")
        stationary = source[0]
        moving = b2.NDArray(value=np.ones((1, 4), np.float32), buffer="sbuf")
        dst = b2.NDArray(value=np.zeros((3, 4), np.float32), buffer="psum")
        checked_operation("nc_matmul", b2.nc_matmul)(dst, stationary, moving)
        assert stationary.shape == (1, 3)
        np.testing.assert_array_equal(dst.data, source.data[0:1].T @ moving.data)
