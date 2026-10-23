"""Opt-in TensorE per-instruction law (TensorInstructionLaw): time set by geometry, not FLOPs."""

from dataclasses import replace

import pytest

from triton_viz.tools.nki_cost_model import CostModel, TensorInstructionLaw, simulate

LAW = TensorInstructionLaw({"float32": (0.0, 100.0, 1.0, 0.5)})


def dot(seq, m, k, n, dst, dtype="float32", stationary=1, offset=0):
    return dict(op="dot", engine="tensor", seq=seq, input_shape=[m, k], other_shape=[k, n], output_shape=[m, n],
                flops=2 * m * k * n, input_dtypes=[dtype, dtype], output_dtype="float32",
                input_storages=[stationary, 2], input_ranges=[[offset, offset + 4 * m * k], [0, 4 * k * n]],
                output_storage=dst, output_range=[0, 4 * m * n])


def tensor_busy(events, model):
    return sum(t.end - t.start for t in simulate(events, model).timeline["tensor"])


def test_default_is_unchanged():
    assert CostModel().tensor_instruction_law is None
    events = [dot(i, 128, 128, 512, 10 + i) for i in range(4)]
    a = simulate(events, CostModel())
    b = simulate(events, replace(CostModel(), tensor_instruction_law=None))
    assert a.predicted_latency_ns == b.predicted_latency_ns


def test_instruction_time_does_not_depend_on_contraction_depth():
    assert LAW.instruction_ns("float32", 128, 32, 512) == pytest.approx(100 + 512 + 64)   # reload_ns 0
    assert LAW.instruction_ns("float32", 128, 128, 512) == LAW.instruction_ns("float32", 128, 32, 512)
    assert LAW.instruction_ns("float32", 64, 128, 512) == pytest.approx(100 + 512 + 32)


def test_shapes_beyond_one_instruction_are_tiled():
    one = LAW.instruction_ns("float32", 128, 128, 512)
    assert LAW.instruction_ns("float32", 256, 256, 1024) == pytest.approx(8 * one)
    assert LAW.instruction_ns("bfloat16", 128, 128, 512) == one       # falls back to float32


def test_splitting_the_contraction_costs_more_with_the_law_not_with_flops():
    full = [dot(0, 128, 128, 512, 10)]
    split = [dot(i, 128, 32, 512, 10) for i in range(4)]            # same FLOPs, four instructions
    base = CostModel(tensor_startup_ns=0.0)                          # FLOP-priced
    assert tensor_busy(split, base) == pytest.approx(tensor_busy(full, base), rel=0.05)
    law = replace(base, tensor_instruction_law=LAW)
    assert tensor_busy(split, law) == pytest.approx(4 * tensor_busy(full, law))


def test_from_csv(tmp_path):
    path = tmp_path / "law.csv"
    path.write_text("dtype,startup_ns,base_ns,ns_per_n,ns_per_m,reload_ns\nfp32,2000,50,0.9,0.2,100\n")
    law = TensorInstructionLaw.from_csv(path)
    assert law.startup_ns("float32") == 2000
    assert law.instruction_ns("float32", 128, 64, 256) == pytest.approx(50 + 0.9 * 256 + 0.2 * 128 + 100)



def test_a_partial_row_stationary_exposes_the_weight_load():
    law = TensorInstructionLaw({"float32": (0.0, 100.0, 1.0, 0.0, 200.0)})
    assert law.instruction_ns("float32", 128, 32, 512) == pytest.approx(100 + 512 + 200)
    assert law.instruction_ns("float32", 128, 128, 512) == pytest.approx(100 + 512)
    assert law.instruction_ns("float32", 128, 160, 512) == pytest.approx(2 * (100 + 512) + 200)   # 128 + 32
