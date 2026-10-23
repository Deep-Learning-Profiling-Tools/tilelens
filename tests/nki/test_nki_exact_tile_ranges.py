"""Opt-in exact tile footprints (CostModel.exact_tile_ranges) and consumer-ordered loads."""

from dataclasses import replace

import numpy as np
import pytest

from triton_viz.core.simulation.nki import array_byte_runs
from triton_viz.tools.nki_cost_model import CostModel, _exact_runs, simulate
from triton_viz.tools.nki_structure_schedule import StructureSchedule, need_order, rename


def test_slice_of_partition_major_tile_is_one_run_per_partition():
    root = np.zeros((4, 8, 16), np.float32)
    view = root[:, 3, :]
    runs = array_byte_runs(view, int(view.ctypes.data) - int(root.ctypes.data))
    assert runs == tuple((p * 512 + 192, p * 512 + 256) for p in range(4))
    assert array_byte_runs(root, 0) == ((0, root.nbytes),)
    assert array_byte_runs(root[:, :, ::-1], 0) is None             # negative stride: bounding interval kept
    assert array_byte_runs(np.zeros((1024, 4), np.float32)[:, :2], 0, max_runs=16) is None


def test_exact_runs_replace_storage_and_interval():
    accesses = [(7, None, 7, 0.0, 100.0)]
    out = _exact_runs(accesses, [{"storage": 3, "runs": [[0, 4], [16, 20]]}])
    assert out == [(3, None, 7, 0.0, 4.0), (3, None, 7, 16.0, 20.0)]
    assert _exact_runs(accesses, [None]) == accesses


def load(seq, dst, lo, hi):
    return dict(op="transfer", engine="dma", seq=seq, mem_src="hbm", mem_dst="sbuf", bytes=hi - lo,
                src_storage=99, src_range=[lo, hi], dst_storage=dst, dst_range=[lo, hi], partition_count=128)


def mm(seq, store, lo, hi, out):
    return dict(op="dot", engine="tensor", seq=seq, input_shape=[128, 128], other_shape=[128, 128],
                output_shape=[128, 128], flops=2 * 128 ** 3, input_dtypes=["float32", "float32"],
                input_storages=[store], input_ranges=[[0, 1 << 20]],
                input_range_lists=[{"storage": store, "runs": [[lo, hi]]}],
                output_storage=out, output_range=[0, 65536])


def test_consumer_waits_only_for_the_slice_it_reads():
    events = [load(0, 5, 0, 1000), load(1, 5, 1000, 2000), mm(2, 5, 0, 1000, 50), mm(3, 5, 1000, 2000, 51)]
    base = CostModel(tensor_startup_ns=0.0)
    exact = replace(base, exact_tile_ranges=True)
    t_base = {e.seq: e.start for e in simulate(events, base).timeline["tensor"]}
    t_exact = {e.seq: e.start for e in simulate(events, exact).timeline["tensor"]}
    first_load_end = simulate(events, exact).timeline["dma"][0].end
    assert t_exact[2] == pytest.approx(first_load_end)               # starts after its own slice
    assert t_base[2] > t_exact[2]                                    # bounding range waits for both


def test_need_order_issues_the_first_needed_load_first():
    a, b = load(0, 5, 0, 1000), load(1, 6, 0, 1000)
    consumer_b = dict(mm(2, 6, 0, 1000, 50))
    consumer_a = dict(mm(3, 5, 0, 1000, 51))
    out = need_order([a, b, consumer_b, consumer_a])
    assert [e["seq"] for e in out] == [1, 0, 2, 3]
    same_dst = [load(0, 5, 0, 1000), load(1, 5, 0, 1000), mm(2, 5, 0, 1000, 50)]
    assert [e["seq"] for e in need_order(same_dst)] == [0, 1, 2]        # overlapping destinations keep order


def test_renaming_remaps_exact_lists():
    events = [load(0, 5, 0, 1000), mm(1, 5, 0, 1000, 50)]
    for e in events:
        e["source_site"] = f"s{e['seq']}"
    out = rename(events, 2)
    assert out[1]["input_range_lists"][0]["storage"] == out[0]["dst_storage"]
    assert StructureSchedule().need_order is False and StructureSchedule().exact_ranges is False


def test_pipelined_queue_hides_fixed_cost_only_when_busy():
    from types import SimpleNamespace

    class Surface:
        def split(self, partitions):
            return 1000.0, 1.0                      # 1 us fixed, 1 byte/ns

        def lookup(self, partitions, free_bytes):
            return SimpleNamespace(bandwidth_gbps=0.5, match="exact", log_distance=0.0,
                                   lookup_partitions=partitions, lookup_free_bytes=free_bytes)

    model = replace(CostModel(), dma_pipelined_fixed_cost=True)
    model._dma_surface = lambda event: Surface()     # noqa: E731 - test double
    model._dma_lookup_free_bytes = lambda event: 4   # noqa: E731
    back_to_back = [dict(load(0, 5, 0, 1000), bytes=1000), dict(load(1, 6, 0, 1000), bytes=1000)]
    tl = simulate(back_to_back, model).timeline["dma"]
    assert tl[0].end - tl[0].start == pytest.approx(2000.0)   # idle queue: fixed + stream
    assert tl[1].end - tl[1].start == pytest.approx(1000.0)   # busy queue: stream only
    assert CostModel().dma_pipelined_fixed_cost is False


def test_multiqueue_dma_shares_bandwidth_and_hides_fixed_cost_when_concurrent():
    from types import SimpleNamespace

    class Surface:
        def split(self, partitions):
            return 1000.0, 1.0

        def lookup(self, partitions, free_bytes):
            return SimpleNamespace(bandwidth_gbps=0.5, match="exact", log_distance=0.0,
                                   lookup_partitions=partitions, lookup_free_bytes=free_bytes)

    model = replace(CostModel(), dma_queues=2, dma_aggregate_gbps=1.0)
    model._dma_surface = lambda event: Surface()     # noqa: E731 - test double
    model._dma_lookup_free_bytes = lambda event: 4   # noqa: E731
    two = [dict(load(0, 5, 0, 1000), bytes=1000), dict(load(1, 6, 0, 1000), bytes=1000)]
    tl = sorted(simulate(two, model).timeline["dma"], key=lambda e: e.start)
    assert tl[0].end - tl[0].start == pytest.approx(2000.0)   # alone: fixed + bytes / single bandwidth
    assert tl[1].start == pytest.approx(0.0)                  # second queue starts at once
    assert tl[1].end - tl[1].start == pytest.approx(2000.0)   # concurrent: bytes * 2 / aggregate, no fixed
    assert CostModel().dma_queues == 0
