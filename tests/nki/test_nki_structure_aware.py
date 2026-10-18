"""Opt-in structure-aware completion (carried serialization) of the NKI cost model."""

import pytest

from triton_viz.tools.nki_cost_model import GlobalCompletionCalibration, carried_load_fraction


def _load(storage, nbytes=1024):
    return {"op": "transfer", "mem_src": "hbm", "mem_dst": "sbuf", "dst_storage": storage, "bytes": nbytes}


def test_carried_fraction_counts_reused_destinations():
    fresh = [_load(i) for i in range(4)]
    assert carried_load_fraction(fresh) == 0.0
    hoisted = [_load("x") for _ in range(4)]
    assert carried_load_fraction(hoisted) == 1.0
    mixed = [_load("x"), _load("x"), _load("y"), _load("z")]
    assert carried_load_fraction(mixed) == pytest.approx(0.5)
    stores = [{"op": "transfer", "mem_src": "sbuf", "mem_dst": "hbm", "dst_storage": "o", "bytes": 9}] * 3
    assert carried_load_fraction(stores) == 0.0


def test_default_is_unchanged_and_opt_in_serializes():
    busy = {"dma": 100.0, "vector": 60.0}
    base = GlobalCompletionCalibration(overlap_fraction=0.33, completion_offset_ns=5.0)
    assert base.predict_ns(busy, carried_fraction=1.0) == base.predict_ns(busy)
    aware = GlobalCompletionCalibration(overlap_fraction=0.33, completion_offset_ns=5.0,
                                        carried_serialization=True)
    assert aware.predict_ns(busy, carried_fraction=0.0) == pytest.approx(base.predict_ns(busy))
    assert aware.predict_ns(busy, carried_fraction=1.0) == pytest.approx(100.0 + 60.0 + 5.0)
    assert aware.predict_ns(busy, carried_fraction=0.5) < aware.predict_ns(busy, carried_fraction=1.0)
