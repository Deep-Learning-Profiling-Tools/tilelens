"""Replay cost-model predictions on saved operator traces without recompiling hardware."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from triton_viz.tools.nki_cost_model import (
    AttentionPipelineCalibration,
    DmaElapsedCalibration,
    OnChipTransferCalibration,
    GlobalCompletionCalibration,
    ComputeCalibration,
    CostModel,
    DmaCalibrationSurface,
    StructuralStaticDmaCalibration,
    StridedDmaCalibration,
    StructuredControlCalibration,
    WholeProgramRoutingCalibration,
    TensorCalibrationSurface,
    TensorDotCountCalibration,
    _canonical_engine,
    _compute_value_dtype,
    _expand_lowering_groups,
    _free_dim,
    _input_stream_count,
    eliminate_redundant_hbm_loads,
    simulate,
    InstructionLatencyCalibration,
)

FIELDS = [
    "case",
    "op",
    "rows",
    "cols",
    "dtype",
    "compiler_elided_load_count",
    "compiler_elided_load_bytes",
    "predicted_dynamic_dma_us",
    "predicted_static_dma_us",
    "predicted_total_dma_us",
    "predicted_vector_us",
    "predicted_vector_payload_us",
    "predicted_scalar_us",
    "predicted_scalar_payload_us",
    "predicted_gpsimd_us",
    "predicted_gpsimd_payload_us",
    "predicted_tensor_us",
    "predicted_total_us",
    "predicted_compute_only_us",
    "predicted_compute_dma_us",
    "predicted_resource_overlap_us",
    "whole_program_routing_covered",
    "whole_program_routing_clamped",
    "predicted_makespan_only_us",
    "predicted_dma_queue_floor_us",
    "dma_descriptor_count",
    "dma_fragmented_descriptor_count",
    "dma_queue_floor_ood",
    "onchip_transfer_count",
    "onchip_transfer_ood",
    "predicted_global_completion_us",
    "global_completion_activated",
    "attention_pipeline_covered",
    "attention_pipeline_ood",
    "compute_only_error_pct",
    "compute_dma_error_pct",
    "resource_overlap_error_pct",
    "makespan_only_error_pct",
    "hardware_nc_p50_us",
    "nc_error_pct",
    "hardware_tensor_active_us",
    "hardware_tensor_reference",
    "tensor_error_pct",
    "hardware_vector_active_us",
    "hardware_vector_payload_us",
    "hardware_vector_payload_reference",
    "vector_payload_evaluable",
    "vector_payload_error_pct",
    "vector_error_pct",
    "hardware_scalar_active_us",
    "hardware_scalar_payload_us",
    "hardware_scalar_payload_reference",
    "scalar_payload_evaluable",
    "scalar_payload_error_pct",
    "scalar_error_pct",
    "hardware_gpsimd_active_us",
    "hardware_gpsimd_reference",
    "hardware_gpsimd_payload_us",
    "hardware_gpsimd_payload_reference",
    "gpsimd_payload_evaluable",
    "gpsimd_payload_error_pct",
    "gpsimd_error_pct",
    "gpsimd_static_opcode_match",
    "hardware_dynamic_dma_us",
    "hardware_dynamic_dma_reference",
    "dynamic_dma_error_pct",
    "hardware_static_dma_us",
    "static_dma_error_pct",
    "static_dma_packet_match",
    "structural_static_dma_padded_exact_count",
    "structural_static_dma_structural_key_count",
    "structural_static_dma_rule_sequence_count",
    "structural_static_dma_none_count",
    "hardware_total_dma_us",
    "dma_error_pct",
    "calibration_match",
    "dma_calibration_path",
    "dma_surface_match",
    "dma_surface_exact_count",
    "dma_surface_interpolated_count",
    "dma_surface_ood_count",
    "dma_surface_max_log_distance",
    "tensor_flops_domain_ood_count",
    "tensor_source_geometry_ood_count",
    "tensor_static_matmul_instruction_count",
    "tensor_instruction_calibration_match",
    "level_a_exact_key_count",
    "level_a_legacy_family_count",
    "level_a_family_prefix_count",
    "level_a_compositional_count",
    "level_a_signature_count",
    "level_a_none_count",
    "micro_dag_vector_covered",
    "micro_dag_scalar_covered",
    "micro_dag_gpsimd_covered",
    "micro_dag_tensor_covered",
    "micro_dag_static_dma_covered",
    "micro_dag_unsupported_engine_events",
    "micro_dag_timing_exact_count",
    "micro_dag_timing_interpolated_count",
    "micro_dag_timing_aggregate_count",
    "micro_dag_source_region_count",
    "micro_dag_exact_region_count",
    "micro_dag_all_regions_covered",
]

PAYLOAD_RESOLUTION_US = 0.010


def _structure_schedule(spec: str | None):
    from triton_viz.tools.nki_structure_schedule import parse_spec
    return parse_spec(spec)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--dma-read-surface-csv", type=Path, required=True)
    parser.add_argument("--dma-write-surface-csv", type=Path, required=True)
    parser.add_argument("--dma-transpose-surface-csv", type=Path)
    parser.add_argument("--compute-calibration-csv", type=Path, required=True)
    parser.add_argument("--instruction-latency-csv", type=Path, default=None,
                        help="opt-in consumer-visible instruction latency (InstructionLatencyCalibration)")
    parser.add_argument("--structure-schedule", default=None,
                        help="opt-in structure-aware issue order: 'capacity' or a depth, optionally '+completion'")
    parser.add_argument("--dma-result-latency-ns", type=float, default=None,
                        help="opt-in fixed DMA result latency beyond bandwidth occupancy")
    parser.add_argument("--dma-result-latency-law", default="constant", choices=("constant", "log2_partitions"))
    parser.add_argument("--structured-control-csv", type=Path, required=True)
    parser.add_argument("--whole-program-control-root", type=Path)
    parser.add_argument("--trace-filename", default="trace.jsonl")
    parser.add_argument("--tensor-calibration-csv", type=Path)
    parser.add_argument("--tensor-source-geometry-csv", type=Path)
    parser.add_argument("--attention-pipeline-calibration-csv", type=Path)
    parser.add_argument("--global-completion-csv", type=Path)
    parser.add_argument("--dma-elapsed-csv", type=Path)
    parser.add_argument("--onchip-transfer-csv", type=Path)
    parser.add_argument("--structural-static-dma-csv", type=Path, required=True)
    parser.add_argument("--strided-dma-csv", type=Path)
    parser.add_argument("--strict-calibration", action="store_true")
    parser.add_argument(
        "--target-aggregate-labels-only",
        action="store_true",
        help="Do not open target hardware summaries; score only saved aggregate CSV labels.",
    )
    parser.add_argument(
        "--gpsimd-reference-csv",
        type=Path,
        action="append",
        default=[],
        help="Saved aggregate-only GpSimd labels; repeated cases use their median.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    gpsimd_references: dict[str, list[float]] = {}
    for reference_path in args.gpsimd_reference_csv:
        with reference_path.open(encoding="utf-8", newline="") as file:
            for reference_row in csv.DictReader(file):
                value = reference_row.get("hardware_gpsimd_active_us")
                if value not in (None, ""):
                    gpsimd_references.setdefault(reference_row["case"], []).append(
                        float(value)
                    )
    source_rows = list(csv.DictReader((args.root / "operator_results.csv").open()))
    out = []
    models: dict[str, CostModel] = {}
    for source in source_rows:
        if source.get("status") != "ok" or not source.get("hardware_dma_active_us"):
            continue
        dtype = source["dtype"]
        if dtype not in models:
            models[dtype] = CostModel(
                dma_calibration=(
                    DmaCalibrationSurface.from_csv(
                        args.dma_read_surface_csv,
                        bandwidth_column="derived.read_gbps_dynamic_dma_active",
                        dtype_name=dtype,
                        duplicate_policy="median",
                    )
                    if args.dma_read_surface_csv else None
                ),
                dma_write_calibration=(
                    DmaCalibrationSurface.from_csv(
                        args.dma_write_surface_csv,
                        "dma_write_partition_surface",
                        "derived.write_gbps_dynamic_dma_active",
                        dtype,
                        required_repeat=16,
                    )
                    if args.dma_write_surface_csv else None
                ),
                dma_transpose_calibration=(
                    DmaCalibrationSurface.from_csv(
                        args.dma_transpose_surface_csv,
                        "dma_transpose_surface",
                        "derived.read_gbps_dynamic_dma_active",
                        dtype,
                    )
                    if args.dma_transpose_surface_csv else None
                ),
                compute_calibration=ComputeCalibration.from_csv(
                    args.compute_calibration_csv
                ),
                structured_control_lowering=StructuredControlCalibration.from_csv(
                    args.structured_control_csv
                ),
                whole_program_routing=(
                    WholeProgramRoutingCalibration.from_control_root(
                        args.whole_program_control_root
                    )
                    if args.whole_program_control_root else None
                ),
                tensor_calibration=(
                    TensorCalibrationSurface.from_csv(
                        args.tensor_calibration_csv,
                        benchmark_name="tensor_matmul_tiled",
                    )
                    if args.tensor_calibration_csv
                    else None
                ),
                tensor_dot_count_calibration=(
                    TensorDotCountCalibration.from_csv(args.tensor_source_geometry_csv)
                    if args.tensor_source_geometry_csv else None
                ),
                attention_pipeline_calibration=(
                    AttentionPipelineCalibration.from_csv(
                        args.attention_pipeline_calibration_csv
                    )
                    if args.attention_pipeline_calibration_csv else None
                ),
                dma_elapsed_calibration=(
                    DmaElapsedCalibration.from_csv(args.dma_elapsed_csv)
                    if args.dma_elapsed_csv else None
                ),
                onchip_transfer_calibration=(
                    OnChipTransferCalibration.from_csv(args.onchip_transfer_csv)
                    if args.onchip_transfer_csv else None
                ),
                instruction_latency=(
                    InstructionLatencyCalibration.from_csv(args.instruction_latency_csv)
                    if args.instruction_latency_csv else None
                ),
                structure_schedule=_structure_schedule(args.structure_schedule),
                dma_result_latency_ns=args.dma_result_latency_ns,
                dma_result_latency_law=args.dma_result_latency_law,
                global_completion_calibration=(
                    GlobalCompletionCalibration.from_csv(args.global_completion_csv)
                    if args.global_completion_csv else None
                ),
                structural_static_dma=StructuralStaticDmaCalibration.from_csv(
                    args.structural_static_dma_csv
                ),
                strided_dma_calibration=(
                    StridedDmaCalibration.from_csv(args.strided_dma_csv)
                    if args.strided_dma_csv
                    else None
                ),
                strict_calibration=args.strict_calibration,
            )
        case_name = (
            f"{source['op']}__r{source['rows']}__c{source['cols']}__{source['dtype']}"
        )
        trace = args.root / case_name / args.trace_filename
        events = [json.loads(line) for line in trace.read_text().splitlines() if line]
        tensor_dots = [event for event in events if event.get("op") == "dot"]
        attention_signature = (
            len(tensor_dots) == 2
            and sum(event.get("op") == "tensor_transpose" for event in events) >= 3
            and sum(event.get("op") == "reduce_sum" for event in events) >= 2
        )
        if attention_signature and args.attention_pipeline_calibration_csv:
            for event in tensor_dots:
                event["attention_pipeline_value_width"] = int(source["cols"])
        if tensor_dots and args.tensor_source_geometry_csv:
            def unique_tiles(index: int, storage_key: str, ranges_key: str) -> int:
                values = set()
                for event in tensor_dots:
                    storages = event.get(storage_key) or []
                    ranges = event.get(ranges_key) or []
                    if index < len(storages) and index < len(ranges):
                        values.add((int(storages[index]), tuple(ranges[index])))
                return len(values)
            lhs_tiles = unique_tiles(0, "input_storages", "input_ranges")
            rhs_tiles = unique_tiles(1, "input_storages", "input_ranges")
            output_tiles = len({
                (event.get("output_storage"), tuple(event.get("output_range") or ()))
                for event in tensor_dots
            })
            for event in tensor_dots:
                event["tensor_source_dot_count"] = len(tensor_dots)
                event["tensor_source_lhs_tile_count"] = lhs_tiles
                event["tensor_source_rhs_tile_count"] = rhs_tiles
                event["tensor_source_output_tile_count"] = output_tiles
        # Source-only prediction boundary: no target post-compile artifact may
        # influence model events or engine busy time.
        static_matmul_count = 0
        tensor_instruction_match = "not_applicable"
        model_events, cse = eliminate_redundant_hbm_loads(events)
        result = simulate(
            model_events,
            models[dtype],
            routing_source_events=events,
            routing_dtype=dtype,
        )
        gpsimd_static_opcode_match = "disabled"
        static_dma_packet_match = "disabled"
        # Audit the lowered compute events that CostModel actually prices.
        # Raw source primitives may have been replaced by structured lowering.
        matches = set()
        audit_events = _expand_lowering_groups(model_events, models[dtype])
        for event in audit_events:
            if event.get("op") not in {"binary", "compute", "reduce_sum"}:
                continue
            engine = _canonical_engine(
                str(event.get("engine") or ""), str(event.get("op") or "")
            )
            streams = _input_stream_count(event)
            free_dim = _free_dim(event)
            if free_dim is None:
                matches.add("missing_geometry")
                continue
            _, match = models[dtype].compute_calibration.instruction_lookup(
                engine,
                _compute_value_dtype(event),
                streams,
                free_dim,
                strict_dtype=args.strict_calibration,
            )
            matches.add(match)
        dynamic_us = result.engine_busy_ns.get("dma", 0.0) / 1000.0
        static_us = result.engine_busy_ns.get("static_dma", 0.0) / 1000.0
        predicted_us = dynamic_us + static_us
        hardware_us = float(source["hardware_dma_active_us"])
        predicted_total_us = result.predicted_latency_ns / 1000.0
        predicted_tensor_us = result.engine_busy_ns.get("tensor", 0.0) / 1000.0
        predicted_vector_us = result.engine_busy_ns.get("vector", 0.0) / 1000.0
        predicted_scalar_us = result.engine_busy_ns.get("scalar", 0.0) / 1000.0
        predicted_gpsimd_us = result.engine_busy_ns.get("gpsimd", 0.0) / 1000.0
        components = result.components_ns
        predicted_vector_payload_us = max(
            0.0,
            predicted_vector_us - components["vector_runtime_baseline_ns"] / 1000.0,
        )
        predicted_scalar_payload_us = max(
            0.0,
            predicted_scalar_us - components["scalar_runtime_baseline_ns"] / 1000.0,
        )
        predicted_gpsimd_payload_us = max(
            0.0,
            predicted_gpsimd_us - components["gpsimd_runtime_baseline_ns"] / 1000.0,
        )
        hardware_nc_us = float(source["hardware_nc_p50_us"])
        hardware_tensor_us = float(source.get("hardware_tensor_active_us") or 0.0)
        hardware_tensor_reference = "operator_results"
        hardware_vector_us = float(source.get("hardware_vector_active_us") or 0.0)
        hardware_scalar_us = float(source.get("hardware_scalar_active_us") or 0.0)
        # Target post-compile artifacts are never opened. All scoring labels
        # must already be present in the saved aggregate operator CSV (or an
        # explicitly supplied aggregate-only GpSimd reference CSV).
        profile = {}
        if case_name in gpsimd_references:
            hardware_gpsimd_us = statistics.median(gpsimd_references[case_name])
            hardware_gpsimd_reference = (
                f"median_of_{len(gpsimd_references[case_name])}_saved_aggregate_labels"
            )
        else:
            hardware_gpsimd_us = float(
                profile.get("gpsimd_engine_active_time") or 0.0
            ) * 1e6
            hardware_gpsimd_reference = (
                "missing_aggregate_reference"
            )
        counter_dynamic_dma_us = (
            float(profile.get("software_dynamic_dma_active_time") or 0.0)
            + float(profile.get("hardware_dynamic_dma_active_time") or 0.0)
        ) * 1e6
        hardware_static_dma_us = float(
            profile.get("static_dma_active_time") or 0.0
        ) * 1e6
        # Explorer exposes both explicit dynamic counters and the aggregate
        # DMA decomposition. Use the larger physically reconciled value:
        # older/compiler-specific profiles can under-report one of the dynamic
        # counter families, while dma_active - static_dma is the corresponding
        # aggregate residual. Preserve the choice in the CSV audit.
        residual_dynamic_dma_us = max(0.0, hardware_us - hardware_static_dma_us)
        hardware_dynamic_dma_us = max(
            counter_dynamic_dma_us, residual_dynamic_dma_us
        )
        hardware_dynamic_dma_reference = (
            "explicit_dynamic_counters"
            if counter_dynamic_dma_us >= residual_dynamic_dma_us
            else "dma_active_minus_static"
        )
        # Strict source-only payload decomposition.  Never open target compiler
        # instruction mappings here: payload is the aggregate Explorer ACTIVE
        # counter minus the independently measured runtime-control baseline,
        # exactly matching the prediction-side decomposition above.
        hardware_vector_payload_us = max(
            0.0,
            hardware_vector_us - components["vector_runtime_baseline_ns"] / 1000.0,
        )
        hardware_scalar_payload_us = max(
            0.0,
            hardware_scalar_us - components["scalar_runtime_baseline_ns"] / 1000.0,
        )
        hardware_gpsimd_payload_us = max(
            0.0,
            hardware_gpsimd_us - components["gpsimd_runtime_baseline_ns"] / 1000.0,
        )
        payload_reference = "saved_aggregate_active_minus_independent_runtime_control"
        has_tensor_events = any(
            event.get("op") in {"dot", "tensor_transpose"}
            for event in model_events
        )
        dma_paths = set()
        dma_matches = set()
        for event in model_events:
            if event.get("op") not in {"load", "store", "transfer"}:
                continue
            lookup = models[dtype].dma_lookup(event)
            dma_paths.add(str(lookup["path"]))
            dma_matches.add(str(lookup["match"]))
        out.append(
            {
                "case": case_name,
                "op": source["op"],
                "rows": source["rows"],
                "cols": source["cols"],
                "dtype": dtype,
                "compiler_elided_load_count": cse["eliminated_load_count"],
                "compiler_elided_load_bytes": cse["eliminated_load_bytes"],
                "predicted_dynamic_dma_us": dynamic_us,
                "predicted_static_dma_us": static_us,
                "predicted_total_dma_us": predicted_us,
                "predicted_vector_us": predicted_vector_us,
                "predicted_vector_payload_us": predicted_vector_payload_us,
                "predicted_scalar_us": predicted_scalar_us,
                "predicted_scalar_payload_us": predicted_scalar_payload_us,
                "predicted_gpsimd_us": predicted_gpsimd_us,
                "predicted_gpsimd_payload_us": predicted_gpsimd_payload_us,
                "predicted_tensor_us": predicted_tensor_us,
                "predicted_total_us": predicted_total_us,
                "predicted_compute_only_us": components["compute_only"] / 1000.0,
                "predicted_compute_dma_us": components["compute_plus_dma"] / 1000.0,
                "predicted_resource_overlap_us": components["resource_overlap_makespan"]
                / 1000.0,
                "whole_program_routing_covered": int(
                    components.get("whole_program_routing_covered", 0.0)
                ),
                "whole_program_routing_clamped": int(
                    components.get("whole_program_routing_clamped", 0.0)
                ),
                "predicted_makespan_only_us": components["makespan_only_ns"] / 1000.0,
                "predicted_dma_queue_floor_us": components["dma_queue_floor_ns"]
                / 1000.0,
                "dma_descriptor_count": int(components["dma_descriptor_count"]),
                "dma_fragmented_descriptor_count": int(
                    components["dma_fragmented_descriptor_count"]
                ),
                "dma_queue_floor_ood": int(components["dma_queue_floor_ood"]),
                "onchip_transfer_count": int(components["onchip_transfer_count"]),
                "onchip_transfer_ood": int(components["onchip_transfer_ood"]),
                "predicted_global_completion_us": components["global_completion_ns"]
                / 1000.0,
                "global_completion_activated": int(
                    components["global_completion_activated"]
                ),
                "attention_pipeline_covered": int(
                    components["attention_pipeline_covered"]
                ),
                "attention_pipeline_ood": int(
                    components["attention_pipeline_ood"]
                ),
                "compute_only_error_pct": (components["compute_only"] / 1000.0 - hardware_nc_us)
                / hardware_nc_us
                * 100,
                "compute_dma_error_pct": (components["compute_plus_dma"] / 1000.0 - hardware_nc_us)
                / hardware_nc_us
                * 100,
                "resource_overlap_error_pct": (
                    components["resource_overlap_makespan"] / 1000.0 - hardware_nc_us
                )
                / hardware_nc_us
                * 100,
                "makespan_only_error_pct": (
                    components["makespan_only_ns"] / 1000.0 - hardware_nc_us
                )
                / hardware_nc_us
                * 100,
                "hardware_nc_p50_us": hardware_nc_us,
                "nc_error_pct": (predicted_total_us - hardware_nc_us)
                / hardware_nc_us
                * 100,
                "hardware_tensor_active_us": (
                    hardware_tensor_us if has_tensor_events else ""
                ),
                "hardware_tensor_reference": (
                    hardware_tensor_reference if has_tensor_events else ""
                ),
                "tensor_error_pct": (
                    (predicted_tensor_us - hardware_tensor_us)
                    / hardware_tensor_us
                    * 100
                    if hardware_tensor_us and has_tensor_events
                    else ""
                ),
                "hardware_vector_active_us": hardware_vector_us,
                "hardware_vector_payload_us": hardware_vector_payload_us,
                "hardware_vector_payload_reference": payload_reference,
                "vector_payload_evaluable": int(
                    hardware_vector_payload_us > PAYLOAD_RESOLUTION_US
                ),
                "vector_payload_error_pct": (
                    (predicted_vector_payload_us - hardware_vector_payload_us)
                    / hardware_vector_payload_us
                    * 100
                    if hardware_vector_payload_us > PAYLOAD_RESOLUTION_US
                    else ""
                ),
                "vector_error_pct": (
                    (predicted_vector_us - hardware_vector_us)
                    / hardware_vector_us
                    * 100
                    if hardware_vector_us
                    else ""
                ),
                "hardware_scalar_active_us": hardware_scalar_us,
                "hardware_scalar_payload_us": hardware_scalar_payload_us,
                "hardware_scalar_payload_reference": payload_reference,
                "scalar_payload_evaluable": int(
                    hardware_scalar_payload_us > PAYLOAD_RESOLUTION_US
                ),
                "scalar_payload_error_pct": (
                    (predicted_scalar_payload_us - hardware_scalar_payload_us)
                    / hardware_scalar_payload_us
                    * 100
                    if hardware_scalar_payload_us > PAYLOAD_RESOLUTION_US
                    else ""
                ),
                "scalar_error_pct": (
                    (predicted_scalar_us - hardware_scalar_us)
                    / hardware_scalar_us
                    * 100
                    if hardware_scalar_us
                    else ""
                ),
                "hardware_gpsimd_active_us": hardware_gpsimd_us,
                "hardware_gpsimd_reference": hardware_gpsimd_reference,
                "hardware_gpsimd_payload_us": hardware_gpsimd_payload_us,
                "hardware_gpsimd_payload_reference": payload_reference,
                "gpsimd_payload_evaluable": int(
                    hardware_gpsimd_payload_us > PAYLOAD_RESOLUTION_US
                ),
                "gpsimd_payload_error_pct": (
                    (predicted_gpsimd_payload_us - hardware_gpsimd_payload_us)
                    / hardware_gpsimd_payload_us
                    * 100
                    if hardware_gpsimd_payload_us > PAYLOAD_RESOLUTION_US
                    else ""
                ),
                "gpsimd_error_pct": (
                    (predicted_gpsimd_us - hardware_gpsimd_us)
                    / hardware_gpsimd_us
                    * 100
                    if hardware_gpsimd_us
                    else ""
                ),
                "gpsimd_static_opcode_match": gpsimd_static_opcode_match,
                "hardware_dynamic_dma_us": hardware_dynamic_dma_us,
                "hardware_dynamic_dma_reference": hardware_dynamic_dma_reference,
                "dynamic_dma_error_pct": (
                    (dynamic_us - hardware_dynamic_dma_us)
                    / hardware_dynamic_dma_us
                    * 100
                    if hardware_dynamic_dma_us
                    else ""
                ),
                "hardware_static_dma_us": hardware_static_dma_us,
                "static_dma_error_pct": (
                    (static_us - hardware_static_dma_us)
                    / hardware_static_dma_us
                    * 100
                    if hardware_static_dma_us
                    else ""
                ),
                "static_dma_packet_match": static_dma_packet_match,
                "hardware_total_dma_us": hardware_us,
                "dma_error_pct": (predicted_us - hardware_us) / hardware_us * 100,
                "calibration_match": ";".join(sorted(matches)) or "not_applicable",
                "dma_calibration_path": ";".join(sorted(dma_paths))
                or "not_applicable",
                "dma_surface_match": ";".join(sorted(dma_matches))
                or "not_applicable",
                "dma_surface_ood_count": int(
                    components.get("dma_surface_ood_count", 0)
                ),
                "dma_surface_exact_count": int(
                    components.get("dma_surface_exact_count", 0)
                ),
                "dma_surface_interpolated_count": int(
                    components.get("dma_surface_interpolated_count", 0)
                ),
                "dma_surface_max_log_distance": components.get(
                    "dma_surface_max_log_distance", 0.0
                ),
                "tensor_flops_domain_ood_count": int(
                    components.get("tensor_flops_domain_ood_count", 0)
                ),
                "tensor_source_geometry_ood_count": int(
                    components.get("tensor_source_geometry_ood_count", 0)
                ),
                "tensor_static_matmul_instruction_count": static_matmul_count,
                "tensor_instruction_calibration_match": tensor_instruction_match,
                **{
                    f"structural_static_dma_{match}_count": int(
                        components.get(
                            f"structural_static_dma_{match}_count", 0
                        )
                    )
                    for match in (
                        "padded_exact",
                        "structural_key",
                        "rule_sequence",
                        "none",
                    )
                },
                **{
                    f"level_a_{match}_count": int(
                        components.get(f"level_a_{match}_count", 0)
                    )
                    for match in (
                        "exact_key",
                        "legacy_family",
                        "family_prefix",
                        "compositional",
                        "signature",
                        "none",
                    )
                },
                "micro_dag_vector_covered": int(
                    components.get("micro_dag_vector_covered", 0)
                ),
                "micro_dag_scalar_covered": int(
                    components.get("micro_dag_scalar_covered", 0)
                ),
                "micro_dag_gpsimd_covered": int(
                    components.get("micro_dag_gpsimd_covered", 0)
                ),
                "micro_dag_tensor_covered": int(
                    components.get("micro_dag_tensor_covered", 0)
                ),
                "micro_dag_static_dma_covered": int(
                    components.get("micro_dag_static_dma_covered", 0)
                ),
                "micro_dag_unsupported_engine_events": int(
                    components.get("micro_dag_unsupported_engine_events", 0)
                ),
                "micro_dag_timing_exact_count": int(
                    components.get("micro_dag_timing_exact_count", 0)
                ),
                "micro_dag_timing_interpolated_count": int(
                    components.get("micro_dag_timing_interpolated_count", 0)
                ),
                "micro_dag_timing_aggregate_count": int(
                    components.get("micro_dag_timing_aggregate_count", 0)
                ),
                "micro_dag_source_region_count": int(
                    components.get("micro_dag_source_region_count", 0)
                ),
                "micro_dag_exact_region_count": int(
                    components.get("micro_dag_exact_region_count", 0)
                ),
                "micro_dag_all_regions_covered": int(
                    components.get("micro_dag_all_regions_covered", 0)
                ),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out)
    mape = statistics.mean(abs(float(row["dma_error_pct"])) for row in out)
    tensor_rows = [row for row in out if row.get("hardware_tensor_active_us")]
    tensor_mape = (
        statistics.mean(abs(float(row["tensor_error_pct"])) for row in tensor_rows)
        if tensor_rows
        else ""
    )
    print(
        f"Replayed {len(out)} cases; combined DMA MAPE={mape:.6f}%; "
        f"TensorE busy MAPE={tensor_mape if tensor_mape != '' else 'n/a'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
