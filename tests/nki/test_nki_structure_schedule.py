"""Opt-in structure-aware issue order (bounded renaming + dependency-ordered loads)."""

from dataclasses import replace

from triton_viz.tools.nki_cost_model import CostModel, simulate
from triton_viz.tools.nki_structure_schedule import (
    StructureSchedule,
    apply_structure_schedule,
    buffering_depth,
)

F = 512
TILE = 128 * F * 4
HBM_IN, HBM_OUT = 1 << 40, 1 << 41


def load(seq, i, dst, site="k.py:10"):
    return dict(seq=seq, op="transfer", engine="dma", mem_src="hbm", mem_dst="sbuf", bytes=TILE,
                src_storage=HBM_IN, src_range=[i * TILE, (i + 1) * TILE], dst_storage=dst, dst_range=[0, TILE],
                partition_count=128, free_bytes_per_partition=F * 4, source_site=site)


def compute(seq, src, dst, site="k.py:11"):
    return dict(seq=seq, op="compute", engine="vector", api_op="exp", input_shape=[128, F], output_shape=[128, F],
                free_dim=F, input_dtypes=["float32"], input_storages=[src], input_ranges=[[0, TILE]],
                output_storage=dst, output_range=[0, TILE], source_site=site)


def store(seq, i, src, site="k.py:12"):
    return dict(seq=seq, op="transfer", engine="dma", mem_src="sbuf", mem_dst="hbm", bytes=TILE,
                src_storage=src, src_range=[0, TILE], dst_storage=HBM_OUT, dst_range=[i * TILE, (i + 1) * TILE],
                partition_count=128, free_bytes_per_partition=F * 4, source_site=site)


def loop(n, local=True):
    """n iterations of load -> compute -> store; loop-local tiles get fresh storages per iteration."""
    events, seq = [], 0
    for i in range(n):
        x, y = (100 + 2 * i, 101 + 2 * i) if local else (100, 101)
        for e in (load(seq, i, x), compute(seq + 1, x, y), store(seq + 2, i, y)):
            events.append(e)
        seq += 3
    return events


def dma_order(events, config):
    out, _ = apply_structure_schedule(events, config)
    return [(e["mem_src"], e["src_range"][0] // TILE if e["mem_src"] == "hbm" else e["dst_range"][0] // TILE)
            for e in out if e["op"] == "transfer"]


def test_default_model_is_unchanged():
    events = loop(4)
    assert CostModel().structure_schedule is None
    a = simulate(events, CostModel())
    b = simulate(events, replace(CostModel(), structure_schedule=None))
    assert a.predicted_latency_ns == b.predicted_latency_ns
    assert "structure_buffer_depth" not in a.components_ns


def test_loop_local_tiles_are_prefetched_up_to_the_depth():
    events = loop(4)
    order = dma_order(events, StructureSchedule(depth=8))
    assert order[:4] == [("hbm", 0), ("hbm", 1), ("hbm", 2), ("hbm", 3)]      # every load hoisted, FIFO
    assert_load_waits_for_reader(events, depth=2)
    base = simulate(events, CostModel()).components_ns["makespan_only_ns"]
    deep = simulate(events, replace(CostModel(), structure_schedule=StructureSchedule(depth=8)))
    assert deep.components_ns["makespan_only_ns"] <= base
    assert deep.components_ns["structure_buffer_depth"] == 8


def assert_load_waits_for_reader(events, depth):
    """With `depth` buffers, the load of tile i follows the compute that last read its buffer (tile i - depth)
    but may precede that tile's store, which reads a different buffer."""
    out, _ = apply_structure_schedule(events, StructureSchedule(depth=depth))
    pos = {(e["op"], e["seq"] // 3, e.get("mem_src")): k for k, e in enumerate(out)}
    for i in range(depth, 4):
        assert pos[("compute", i - depth, None)] < pos[("transfer", i, "hbm")] < pos[("transfer", i - depth, "sbuf")]


def test_depth_one_waits_for_the_reader_of_the_single_buffer():
    assert_load_waits_for_reader(loop(4), depth=1)


def test_hoisted_buffer_serializes_iterations_like_a_carried_dependency():
    events = loop(4, local=False)
    assert dma_order(events, StructureSchedule(depth=8)) == dma_order(events, StructureSchedule(depth=1))


def test_load_never_passes_a_store_to_the_same_hbm_range():
    events = [store(0, 0, 7, site="k.py:5"), load(1, 0, 9, site="k.py:6")]
    events[0]["dst_storage"] = HBM_IN                                         # store writes the region the load reads
    out, _ = apply_structure_schedule(events, StructureSchedule(depth=8))
    assert [e["seq"] for e in out] == [0, 1]


def test_capacity_derived_depth_and_signature_fallback():
    events = loop(4)
    assert buffering_depth(events, StructureSchedule()) == (176 * 1024) // (2 * F * 4)
    no_sites = [{k: v for k, v in e.items() if k != "source_site"} for e in events]
    assert dma_order(no_sites, StructureSchedule(depth=2)) == dma_order(events, StructureSchedule(depth=2))


def test_schedule_completion_replaces_the_global_term():
    events = loop(4)
    model = replace(CostModel(), structure_schedule=StructureSchedule(schedule_completion=True))
    r = simulate(events, model)
    assert r.predicted_latency_ns == r.components_ns["makespan_only_ns"]


def test_manual_ping_pong_buffers_are_never_merged():
    """Two persistent buffers alternated by the source keep their identity even at depth 1."""
    events, seq = [], 0
    for i in range(4):
        x, y = (100 + i % 2, 200 + i % 2)
        events += [load(seq, i, x), compute(seq + 1, x, y), store(seq + 2, i, y)]
        seq += 3
    out, _ = apply_structure_schedule(events, StructureSchedule(depth=1))
    pos = {(e["op"], e["seq"] // 3, e.get("mem_src")): k for k, e in enumerate(out)}
    assert pos[("transfer", 1, "hbm")] < pos[("compute", 0, None)]           # tile 1 uses the other buffer
    assert pos[("compute", 0, None)] < pos[("transfer", 2, "hbm")]           # tile 2 reuses buffer 0


def test_persistent_bytes_reduce_capacity():
    events = loop(4, local=False)
    assert buffering_depth(events, StructureSchedule()) == 1                  # nothing fresh to rename
    fresh = loop(4)
    big = [dict(e) for e in fresh] + [dict(load(99, 0, 7, site="k.py:3"), free_bytes_per_partition=150 * 1024),
                                      dict(load(100, 1, 7, site="k.py:3"), free_bytes_per_partition=150 * 1024)]
    assert buffering_depth(big, StructureSchedule()) == (176 * 1024 - 150 * 1024) // (2 * F * 4)


def test_dma_result_latency_delays_only_dependent_consumers():
    from triton_viz.tools.nki_structure_schedule import StructureSchedule as _S  # noqa: F401
    base = CostModel()
    lat = replace(base, dma_result_latency_ns=1500.0)
    chain = loop(1)                                                           # load -> compute -> store
    a, b = simulate(chain, base), simulate(chain, lat)
    ca = [t.start for t in a.timeline["vector"]][0]
    cb = [t.start for t in b.timeline["vector"]][0]
    assert abs((cb - ca) - 1500.0) < 1e-6                                     # compute waits for the data
    stream = [load(i, i, 100 + i) for i in range(3)]                          # independent loads
    sa, sb = simulate(stream, base), simulate(stream, lat)
    assert [t.start for t in sa.timeline["dma"]] == [t.start for t in sb.timeline["dma"]]   # queue throughput unchanged
    assert simulate(chain, base).predicted_latency_ns == simulate(chain, replace(base, dma_result_latency_ns=None)).predicted_latency_ns


def test_schedule_completion_keeps_the_calibrated_offset():
    from triton_viz.tools.nki_cost_model import GlobalCompletionCalibration
    cal = GlobalCompletionCalibration(overlap_fraction=0.33, completion_offset_ns=5000.0)
    model = replace(CostModel(), global_completion_calibration=cal,
                    structure_schedule=StructureSchedule(schedule_completion=True))
    r = simulate(loop(4), model)
    assert abs(r.predicted_latency_ns - (r.components_ns["makespan_only_ns"] + 5000.0)) < 1e-6


def test_residual_offset_and_global_floor():
    from triton_viz.tools.nki_cost_model import GlobalCompletionCalibration
    cal = GlobalCompletionCalibration(overlap_fraction=1.0, completion_offset_ns=50000.0)
    events = loop(4)
    plain = simulate(events, replace(CostModel(), global_completion_calibration=cal,
                                     structure_schedule=StructureSchedule(schedule_completion=True, residual_offset_ns=800.0)))
    assert abs(plain.predicted_latency_ns - (plain.components_ns["makespan_only_ns"] + 800.0)) < 1e-6
    floored = simulate(events, replace(CostModel(), global_completion_calibration=cal,
                                       structure_schedule=StructureSchedule(schedule_completion=True, residual_offset_ns=800.0,
                                                                            global_floor=True)))
    assert floored.predicted_latency_ns == floored.components_ns["global_completion_ns"]


def test_parse_spec():
    from triton_viz.tools.nki_structure_schedule import parse_spec
    assert parse_spec(None) is None
    s = parse_spec("capacity+completion+floor+offset=790")
    assert s.depth is None and s.schedule_completion and s.global_floor and s.residual_offset_ns == 790.0
    assert parse_spec("2").depth == 2 and not parse_spec("2").schedule_completion


def test_dma_latency_log2_partition_law():
    base = replace(CostModel(), dma_result_latency_ns=1400.0, dma_result_latency_law="log2_partitions")
    one = dict(load(0, 0, 100), partition_count=1)
    full = load(0, 0, 100)
    assert base.result_ready_ns(one, "dma", 10.0) == 10.0
    assert abs(base.result_ready_ns(full, "dma", 10.0) - 1410.0) < 1e-9


def test_surface_split_recovers_fixed_cost_and_bandwidth():
    from triton_viz.tools.nki_cost_model import DmaCalibrationSurface
    pts = {}
    for p in (16, 128):
        fixed, bw = (200.0, 100.0) if p == 16 else (600.0, 250.0)
        for f in (256, 1024, 4096, 16384):
            nbytes = p * f
            pts[(p, f)] = nbytes / (fixed + nbytes / bw)        # aggregate GB/s = bytes / ns
    surf = DmaCalibrationSurface(pts)
    for p, (fixed, bw) in ((16, (200.0, 100.0)), (128, (600.0, 250.0))):
        got = surf.split(p)
        assert abs(got[0] - fixed) < 1e-6 and abs(got[1] - bw) < 1e-6
    mid = surf.split(45)                                        # interpolated in log2(p)
    assert 200.0 < mid[0] < 600.0 and 100.0 < mid[1] < 250.0


def test_capacity_excludes_reserved_sbuf_space():
    event = dict(load(0, 0, 100), free_bytes_per_partition=64 * 1024)
    # Three buffers fit physical capacity, but not the SDK's usable allocation
    # space. Explicit capacity overrides remain available for other devices.
    assert buffering_depth([event], StructureSchedule()) == 2
    assert buffering_depth([event], StructureSchedule(sbuf_partition_bytes=192 * 1024)) == 3
