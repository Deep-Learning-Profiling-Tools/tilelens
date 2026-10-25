import numpy as np
import pytest

from triton_viz.core.frontend.base import _LangPatchScope
from triton_viz.core.simulation import nki_beta2 as b2
from triton_viz.core.simulation.nki_isa_contracts import (
    ISAContractError, checked_operation, checks_enabled, partition_contract_checks,
)
import nki.isa as nisa

pytestmark = pytest.mark.nki


def tile(shape, buffer="sbuf", partition_axis=0):
    return b2.NDArray(value=np.zeros(shape, np.float32), buffer=buffer,
                      partition_axis=partition_axis)


def test_boundaries_free_axis_and_hbm_are_distinct():
    def op(dst, src):
        return dst
    checked = checked_operation("dma_copy", op)
    large_hbm = tile((1024, 512), buffer="hbm", partition_axis=None)
    with partition_contract_checks():
        legal = tile((128, 4096))
        assert checked(legal, large_hbm) is legal
        marked = tile((1024, 128), partition_axis=1)
        assert checked(marked, large_hbm) is marked
        bad = tile((129, 4))
        with pytest.raises(ISAContractError) as caught:
            checked(bad, large_hbm)
        assert caught.value.finding["shape"] == [129, 4]
        assert caught.value.finding["operand"] == "dst"
        with pytest.raises(ISAContractError):
            checked(tile((4, 129), partition_axis=1), large_hbm)
    assert checked(bad, large_hbm) is bad


def test_context_restores_after_exceptions_and_nested_disable():
    assert not checks_enabled()
    with pytest.raises(RuntimeError):
        with partition_contract_checks():
            assert checks_enabled()
            with partition_contract_checks(False):
                assert not checks_enabled()
            assert checks_enabled()
            raise RuntimeError("exit")
    assert not checks_enabled()


def test_frontend_patch_rejects_before_copy_and_restores_default():
    src = tile((256, 4), buffer="hbm", partition_axis=None)
    dst = tile((256, 4))
    scope = _LangPatchScope()
    try:
        with partition_contract_checks():
            b2.nki_patch_lang(scope)
            with pytest.raises(ISAContractError):
                nisa.dma_copy(dst=dst, src=src)
    finally:
        scope.restore()
    # Same source remains accepted by the historical, non-strict interpreter.
    scope = _LangPatchScope()
    try:
        b2.nki_patch_lang(scope)
        nisa.dma_copy(dst=dst, src=src)
    finally:
        scope.restore()


def test_real_trace_checks_unmarked_ndarray_and_preserves_legal_free_extent():
    import triton_viz
    from triton_viz.clients import Tracer
    import nki.language as nl

    def kernel(x, out):
        scratch = nl.ndarray(x.shape, nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=scratch, src=x)
        nisa.dma_copy(dst=out, src=scratch)
        return out

    def run(shape, strict):
        x = np.ones(shape, np.float32)
        out = np.zeros(shape, np.float32)
        triton_viz.clear()
        with partition_contract_checks(strict):
            triton_viz.trace(client=Tracer(), frontend="nki_beta2")(kernel)[(1,)](x, out, pre_trace=False)
        np.testing.assert_array_equal(out, x)

    run((256, 4), False)
    with pytest.raises(ISAContractError):
        run((256, 4), True)
    run((128, 1024), True)
