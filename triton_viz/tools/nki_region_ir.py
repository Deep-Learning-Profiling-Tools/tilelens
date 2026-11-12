"""Structured, composable lowering features for one contiguous NKI region."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_REDUCTIONS = {"reduce_sum", "max", "min", "mean"}
_TRANSCENDENTAL = {"exp", "rsqrt", "sqrt", "log", "sin", "cos", "tanh", "sigmoid"}
_ONE_INPUT = {"exp", "rsqrt", "sqrt", "log", "sin", "cos", "tanh", "sigmoid", "relu"}
_IGNORED_FAMILY_OPS = {"where"}
REGION_IR_SCHEMA_NAME = "triton-viz.nki-region-ir"
REGION_IR_SCHEMA_VERSION = 3
SUPPORTED_REGION_IR_SCHEMA_VERSIONS = frozenset({1, 2, REGION_IR_SCHEMA_VERSION})
_KNOWN_FAMILY_OPS = (
    _REDUCTIONS
    | _TRANSCENDENTAL
    | {
        "add",
        "broadcast_to",
        "divide",
        "maximum",
        "minimum",
        "multiply",
        "relu",
        "subtract",
        "where",
    }
)


@dataclass(frozen=True)
class GrammarRule:
    """One auditable region-family classification rule.

    Predicates at the same priority must be mutually exclusive.  Keeping rule
    metadata beside the predicate makes the production classifier inspectable
    without changing the stable family strings used by calibration CSVs.
    """

    rule_id: str
    priority: int
    predicate: Callable[[dict[str, Any]], bool]
    family: str | Callable[[dict[str, Any]], str]
    condition: str
    rationale: str
    evidence: tuple[str, ...] = ()

    def render_family(self, facts: dict[str, Any]) -> str:
        return self.family(facts) if callable(self.family) else self.family


@dataclass(frozen=True)
class GrammarMatch:
    """Classification result with evidence and explicit OOD diagnostics."""

    family: str
    rule_id: str
    rationale: str
    evidence: tuple[str, ...]
    ood_reasons: tuple[str, ...]
    consumed_features: tuple[str, ...]


def _token(event: dict[str, Any]) -> str:
    return str(
        event.get("api_op") or event.get("binary_op") or event.get("op") or "unknown"
    ).lower()


def _shape(event: dict[str, Any], key: str) -> list[int]:
    value = event.get(key) or []
    return [int(item) for item in value] if isinstance(value, (list, tuple)) else []


def _value_dtypes(values: Any) -> list[str]:
    """Return canonical non-predicate dtype strings in source order."""
    if values is None:
        return []
    if not isinstance(values, (list, tuple)):
        values = [values]
    result = []
    for value in values:
        if not value:
            continue
        dtype = str(value)
        if dtype.lower() in {"bool", "boolean"}:
            continue
        result.append(dtype)
    return result


def _dominant_dtype(values: list[str], default: str = "float32") -> str:
    """Choose deterministically while preserving the first value on a tie."""
    return Counter(values).most_common(1)[0][0] if values else default


def completion_calibration_dtype(region: dict[str, Any]) -> str:
    """Return the source-input precision that selects completion controls.

    Completion is a whole-kernel observation.  For reductions, the source
    input precision and accumulator precision are distinct compiler-visible
    facts (for example BF16 -> FP32), so using the accumulator/majority dtype
    silently aliases two different control domains.  Older Region IR remains
    readable through the legacy ``dtype`` fallback.
    """
    return str(region.get("input_dtype") or region.get("dtype") or "float32")


def region_ir_structural_key(region: dict[str, Any]) -> str:
    """Hash only compiler-relevant structure, excluding run provenance."""
    canonical = json.dumps(
        {
            key: value
            for key, value in region.items()
            if key not in {"provenance", "structural_key"}
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def build_region_ir(
    members: list[dict[str, Any]], context: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Return stable structural features; never include allocation addresses."""
    tokens = [_token(event) for event in members]
    shapes = [
        _shape(event, "input_shape") or _shape(event, "output_shape")
        for event in members
    ]
    free_dim = max((shape[-1] for shape in shapes if shape), default=1)
    partition_count = max((shape[0] for shape in shapes if len(shape) > 1), default=1)
    reductions = [token for token in tokens if token in _REDUCTIONS]
    input_arity = [
        len(event.get("input_ptrs") or event.get("input_dtypes") or [])
        for event in members
    ]
    one_input = sum(
        token in _ONE_INPUT or arity == 1 for token, arity in zip(tokens, input_arity)
    )
    two_input = sum(
        token not in _REDUCTIONS and arity >= 2
        for token, arity in zip(tokens, input_arity)
    )
    producer: dict[int, int] = {}
    producer_shapes: dict[int, list[int]] = {}
    edges: list[list[int]] = []
    broadcast_edges = 0
    for index, event in enumerate(members):
        for pointer in event.get("input_ptrs") or []:
            if pointer in producer:
                edges.append([producer[pointer], index])
                source_shape = producer_shapes.get(pointer, [])
                target_shape = _shape(event, "output_shape")
                if (
                    source_shape
                    and target_shape
                    and source_shape[-1] == 1 < target_shape[-1]
                ):
                    broadcast_edges += 1
        output = event.get("output_ptr")
        if output is not None:
            producer[output] = index
            producer_shapes[output] = _shape(event, "output_shape")

    indegree = Counter(target for _, target in edges)
    outdegree = Counter(source for source, _ in edges)
    # Each produced value stays live from its source index through the index
    # before its last consumer. A single edge scan plus a prefix sum gives the
    # exact original peak without rescanning all edges and values per node.
    last_consumer: dict[int, int] = {}
    for source, target in edges:
        last_consumer[source] = max(last_consumer.get(source, target), target)
    live_changes = [0] * (len(members) + 1)
    for source, last in last_consumer.items():
        if source < last:
            live_changes[source] += 1
            live_changes[last] -= 1
    live = 0
    max_live_values = 0
    for change in live_changes[:-1]:
        live += change
        max_live_values = max(max_live_values, live)
    depths = [0] * len(members)
    for source, target in edges:
        depths[target] = max(depths[target], depths[source] + 1)

    context = context or []
    loads = [event for event in context if event.get("op") == "load"]
    stores = [event for event in context if event.get("op") == "store"]
    logical_free = max(
        (
            int(event.get("active_lanes", 0))
            // max(1, int(event.get("partition_count", 1)))
            for event in loads
        ),
        default=free_dim,
    )
    logical_free = min(free_dim, logical_free) if logical_free else free_dim
    logical_active_partitions = max(
        (
            int(event.get("active_lanes", 0)) // max(1, logical_free)
            for event in loads
            if int(event.get("active_lanes", 0)) >= logical_free
            and int(event.get("active_lanes", 0)) % max(1, logical_free) == 0
        ),
        default=partition_count,
    )
    logical_active_partitions = min(partition_count, logical_active_partitions)
    tail = logical_free < free_dim
    has_explicit_mask = any(
        bool(event.get("mask_provided")) for event in loads + stores
    )
    # Precision roles must not be collapsed.  The declared source trace can
    # contain BF16 inputs followed by FP32 reduction/epilogue intermediates;
    # using the majority dtype then loses the original input precision.
    produced_ptrs: set[Any] = set()
    external_input_dtypes: list[str] = []
    all_input_dtypes: list[str] = []
    output_dtypes: list[str] = []
    reduction_output_dtypes: list[str] = []
    for event, token in zip(members, tokens):
        event_input_dtypes = _value_dtypes(event.get("input_dtypes"))
        all_input_dtypes.extend(event_input_dtypes)
        input_ptrs = list(event.get("input_ptrs") or [])
        for index, input_dtype in enumerate(event_input_dtypes):
            pointer = input_ptrs[index] if index < len(input_ptrs) else None
            if pointer is None or pointer not in produced_ptrs:
                external_input_dtypes.append(input_dtype)
        event_output_dtypes = _value_dtypes(event.get("output_dtype"))
        output_dtypes.extend(event_output_dtypes)
        if token in _REDUCTIONS:
            reduction_output_dtypes.extend(event_output_dtypes)
        output_ptr = event.get("output_ptr")
        if output_ptr is not None:
            produced_ptrs.add(output_ptr)

    # HBM->SBUF loads are the strongest source-visible evidence for program
    # input precision, and avoid scalar FP32 literals outvoting a BF16 tile.
    loaded_input_dtypes = [
        str(event.get("dst_dtype") or event.get("src_dtype"))
        for event in loads
        if event.get("dst_dtype") or event.get("src_dtype")
    ]
    input_dtype_evidence = (
        loaded_input_dtypes or external_input_dtypes or all_input_dtypes
    )
    input_dtype = _dominant_dtype(input_dtype_evidence)
    accumulator_dtype = _dominant_dtype(
        reduction_output_dtypes or output_dtypes or all_input_dtypes,
        default=input_dtype,
    )
    dtypes = output_dtypes + all_input_dtypes
    dtype = _dominant_dtype(dtypes, default=input_dtype)
    item_bytes = 2 if dtype.lower() in {"float16", "fp16", "bfloat16", "bf16"} else 4
    block_elems = max(1, 8192 // item_bytes)
    result = {
        "schema_name": REGION_IR_SCHEMA_NAME,
        "schema_version": REGION_IR_SCHEMA_VERSION,
        "provenance": {"extractor": "triton_viz.tools.nki_region_ir.build_region_ir"},
        "tokens": tokens,
        "op_histogram": dict(sorted(Counter(tokens).items())),
        "reduction_kind": reductions[0]
        if len(set(reductions)) == 1 and reductions
        else ("mixed" if reductions else "none"),
        "reduction_count": len(reductions),
        "broadcast_edge_count": broadcast_edges
        + sum(
            1
            for event in members
            if _shape(event, "input_shape")
            and _shape(event, "output_shape")
            and _shape(event, "input_shape")[-1]
            == 1
            < _shape(event, "output_shape")[-1]
        ),
        "partition_broadcast_input_count": sum(
            1
            for event in loads
            if int(event.get("partition_count", 1)) == 1 and partition_count > 1
        ),
        "one_input_elementwise_count": one_input,
        "two_input_elementwise_count": two_input,
        "transcendental_count": sum(token in _TRANSCENDENTAL for token in tokens),
        "compute_mask_count": sum(
            bool(event.get("compute_mask_provided")) for event in members
        ),
        "dtype": dtype,
        "input_dtype": input_dtype,
        "accumulator_dtype": accumulator_dtype,
        "input_dtypes": sorted(set(input_dtype_evidence)),
        "output_dtypes": sorted(set(output_dtypes)),
        "has_mixed_precision": input_dtype.lower() != accumulator_dtype.lower(),
        "partition_count": partition_count,
        "logical_active_partition_count": logical_active_partitions,
        "free_dim": free_dim,
        "logical_free_dim": logical_free,
        "free_block_count": math.ceil(logical_free / block_elems),
        "has_mask_or_tail": tail
        or has_explicit_mask
        or "where" in tokens
        or any(
            int(event.get("active_lanes", 0))
            < math.prod(event.get("offsets_shape") or [0])
            for event in loads
        ),
        "memory_spaces": sorted(
            {str(event.get("mem_src")) for event in loads + stores}
            | {str(event.get("mem_dst")) for event in loads + stores}
        ),
        "uses_sbuf": any(
            "SBUF" in (str(event.get("mem_src")), str(event.get("mem_dst")))
            for event in loads + stores
        ),
        "uses_psum": any(
            "PSUM" in (str(event.get("mem_src")), str(event.get("mem_dst")))
            for event in loads + stores
        ),
        "dag_edges": edges,
        "dag_branch_value_count": sum(value > 1 for value in outdegree.values()),
        "dag_join_node_count": sum(value > 1 for value in indegree.values()),
        "dag_max_fanout": max(outdegree.values(), default=0),
        "dag_max_fanin": max(indegree.values(), default=0),
        "dag_max_live_values": max_live_values,
        "dag_critical_path_length": max(depths, default=0),
        "has_explicit_mask": has_explicit_mask,
        "has_tail": tail,
    }
    result["structural_key"] = region_ir_structural_key(result)
    return result


def compositional_features(region: dict[str, Any]) -> dict[str, float]:
    """Numeric features used by an interpretable additive Level-A model."""
    names = [
        "reduction_count",
        "broadcast_edge_count",
        "one_input_elementwise_count",
        "two_input_elementwise_count",
        "transcendental_count",
        "free_block_count",
        "dag_branch_value_count",
        "dag_join_node_count",
        "dag_max_fanout",
        "dag_max_fanin",
        "dag_max_live_values",
        "dag_critical_path_length",
    ]
    features = {name: float(region.get(name, 0)) for name in names}
    features["compute_mask_count"] = float(region.get("compute_mask_count", 0))
    features["intercept"] = 1.0
    features["log2_free_dim"] = math.log2(
        max(1, int(region.get("logical_free_dim", 1)))
    )
    features["mask_or_tail"] = float(bool(region.get("has_mask_or_tail")))
    features["two_reduction_interaction"] = float(
        int(region.get("reduction_count", 0)) >= 2
    )
    features["two_reduction_interaction_x_free"] = (
        features["two_reduction_interaction"]
        * float(max(1, int(region.get("logical_free_dim", 1))))
    )
    features["rsqrt_newton_interaction"] = float(
        region.get("op_histogram", {}).get("rsqrt", 0) > 0
        and region.get("op_histogram", {}).get("multiply", 0) >= 3
    )
    features["two_reduction_rsqrt_interaction"] = (
        features["two_reduction_interaction"] * features["rsqrt_newton_interaction"]
    )
    log_f = features["log2_free_dim"]
    for name in (
        "reduction_count",
        "one_input_elementwise_count",
        "two_input_elementwise_count",
        "transcendental_count",
    ):
        features[f"{name}_x_log2_free"] = features[name] * log_f
    total_elementwise = (
        features["one_input_elementwise_count"]
        + features["two_input_elementwise_count"]
    )
    features["elementwise_only"] = float(
        features["reduction_count"] == 0 and total_elementwise > 0
    )
    features["single_elementwise"] = float(
        features["reduction_count"] == 0 and total_elementwise == 1
    )
    free = float(max(1, int(region.get("logical_free_dim", 1))))
    features["free_dim_linear"] = free
    allocation_free = float(max(1, int(region.get("free_dim", free))))
    features["allocation_free_dim"] = allocation_free
    features["allocation_to_logical_ratio"] = allocation_free / free
    features["wide_allocation"] = float(allocation_free >= 8192)
    features["tile2k_masked"] = float(
        allocation_free == 2048 and features["mask_or_tail"] > 0
    )
    features["two_input_elementwise_count_x_free"] = (
        features["two_input_elementwise_count"] * free
    )
    features["one_input_elementwise_count_x_free"] = (
        features["one_input_elementwise_count"] * free
    )
    features["transcendental_count_x_free"] = (
        features["transcendental_count"] * free
    )
    features["reduction_count_x_free"] = features["reduction_count"] * free
    features["has_transcendental"] = float(features["transcendental_count"] > 0)
    features["has_reduction"] = float(features["reduction_count"] > 0)
    features["arithmetic_only"] = float(
        features["transcendental_count"] == 0
        and features["reduction_count"] == 0
    )
    primitive_count = (
        features["one_input_elementwise_count"]
        + features["two_input_elementwise_count"]
        + features["reduction_count"]
    )
    features["multi_primitive"] = float(primitive_count > 1)
    features["one_input_arity"] = float(
        features["one_input_elementwise_count"] > 0
        and features["two_input_elementwise_count"] == 0
        and features["reduction_count"] == 0
    )
    features["two_input_arity"] = float(
        features["two_input_elementwise_count"] > 0
        and features["one_input_elementwise_count"] == 0
        and features["reduction_count"] == 0
    )
    features["free_dim_linear_x_multi"] = free * features["multi_primitive"]
    features["has_mask_or_tail_feature"] = float(
        bool(region.get("has_mask_or_tail"))
    )
    features["has_compute_mask"] = float(features["compute_mask_count"] > 0)
    features["compute_mask_count_x_free"] = features["compute_mask_count"] * free
    features["has_mask_or_tail_feature_x_multi"] = (
        features["has_mask_or_tail_feature"] * features["multi_primitive"]
    )
    for name in (
        "one_input_elementwise_count",
        "one_input_elementwise_count_x_free",
        "two_input_elementwise_count",
        "two_input_elementwise_count_x_free",
        "transcendental_count",
        "transcendental_count_x_free",
        "reduction_count",
        "reduction_count_x_free",
        "has_transcendental",
        "has_reduction",
        "arithmetic_only",
        "two_reduction_interaction",
        "two_reduction_interaction_x_free",
    ):
        features[f"{name}_x_multi"] = features[name] * features["multi_primitive"]
    partition = int(region.get("partition_count") or 1)
    nearest_partition = min((1, 16, 128), key=lambda value: abs(value - partition))
    for value in (1, 16, 128):
        features[f"partition_p{value}"] = float(nearest_partition == value)
    logical_partition = int(
        region.get("logical_active_partition_count") or partition
    )
    features["logical_active_partition_count"] = float(logical_partition)
    features["log2_logical_active_partition_count"] = math.log2(
        max(1, logical_partition)
    )
    nearest_logical_partition = min(
        (1, 16, 128), key=lambda value: abs(value - logical_partition)
    )
    for value in (1, 16, 128):
        features[f"logical_partition_p{value}"] = float(
            nearest_logical_partition == value
        )
    for name in (
        "two_input_elementwise_count",
        "two_input_elementwise_count_x_free",
        "transcendental_count",
        "transcendental_count_x_free",
        "reduction_count",
        "reduction_count_x_free",
        "has_transcendental",
        "has_reduction",
        "arithmetic_only",
        "free_dim_linear",
        "two_reduction_interaction",
        "two_reduction_interaction_x_free",
    ):
        features[f"{name}_x_partition_p128"] = (
            features[name] * features["partition_p128"]
        )
        features[f"{name}_x_partition_p128_x_multi"] = (
            features[f"{name}_x_partition_p128"] * features["multi_primitive"]
        )
    for name in (
        "two_input_elementwise_count",
        "one_input_elementwise_count",
        "transcendental_count",
        "reduction_count",
        "has_transcendental",
        "has_reduction",
        "arithmetic_only",
        "two_reduction_interaction",
    ):
        features[f"{name}_x_wide_allocation"] = (
            features[name] * features["wide_allocation"]
        )
        features[f"{name}_x_wide_allocation_x_multi"] = (
            features[f"{name}_x_wide_allocation"] * features["multi_primitive"]
        )
    for token, count in region.get("op_histogram", {}).items():
        features[f"op_{token}"] = float(count)
        features[f"op_{token}_x_free"] = float(count) * free
        features[f"op_{token}_x_multi"] = (
            float(count) * features["multi_primitive"]
        )
        features[f"op_{token}_x_free_x_multi"] = (
            float(count) * free * features["multi_primitive"]
        )
        features[f"op_{token}_x_mask"] = (
            float(count) * features["has_mask_or_tail_feature"]
        )
        features[f"op_{token}_x_free_x_mask"] = (
            float(count) * free * features["has_mask_or_tail_feature"]
        )
        features[f"op_{token}_x_allocation_free"] = float(count) * allocation_free
        features[f"op_{token}_x_compute_mask"] = (
            float(count) * features["has_compute_mask"]
        )
        features[f"op_{token}_x_free_x_compute_mask"] = (
            float(count) * free * features["has_compute_mask"]
        )
        features[f"op_{token}_x_sqrt_free_x_compute_mask"] = (
            float(count) * math.sqrt(max(1.0, free)) * features["has_compute_mask"]
        )
        features[f"op_{token}_x_log2_free_x_compute_mask"] = (
            float(count) * math.log2(max(1.0, free)) * features["has_compute_mask"]
        )
        features[f"op_{token}_x_wide_allocation"] = (
            float(count) * features["wide_allocation"]
        )
        features[f"op_{token}_x_wide_allocation_x_multi"] = (
            features[f"op_{token}_x_wide_allocation"]
            * features["multi_primitive"]
        )
        for arity in ("one_input_arity", "two_input_arity"):
            features[f"op_{token}_x_{arity}"] = float(count) * features[arity]
            features[f"op_{token}_x_free_x_{arity}"] = (
                float(count) * free * features[arity]
            )
        features[f"op_{token}_x_mask_x_multi"] = (
            features[f"op_{token}_x_mask"] * features["multi_primitive"]
        )
        features[f"op_{token}_x_free_x_mask_x_multi"] = (
            features[f"op_{token}_x_free_x_mask"] * features["multi_primitive"]
        )
    ordered_tokens = list(region.get("tokens", []))
    features["long_mixed_tile2k_applicable"] = float(
        features["tile2k_masked"] > 0
        and features["partition_p128"] > 0
        and len(ordered_tokens) >= 12
        and (
            int(region.get("reduction_count", 0)) > 0
            or int(region.get("transcendental_count", 0)) > 0
        )
    )
    features["atomic_wide_masked_applicable"] = float(
        features["partition_p128"] > 0
        and features["wide_allocation"] > 0
        and features["has_compute_mask"] > 0
        and len(ordered_tokens) == 1
    )
    if ordered_tokens:
        first_name = f"first_op_{ordered_tokens[0]}"
        features[first_name] = 1.0
        features[f"{first_name}_x_multi"] = features["multi_primitive"]
        features[f"{first_name}_x_tile2k_masked"] = features["tile2k_masked"]
        features[f"{first_name}_x_tile2k_masked_x_multi"] = (
            features["tile2k_masked"] * features["multi_primitive"]
        )
    for lhs, rhs in zip(ordered_tokens, ordered_tokens[1:]):
        name = f"bigram_{lhs}__{rhs}"
        features[name] = features.get(name, 0.0) + 1.0
    for first, second, third in zip(
        ordered_tokens, ordered_tokens[1:], ordered_tokens[2:]
    ):
        name = f"trigram_{first}__{second}__{third}"
        features[name] = features.get(name, 0.0) + 1.0
    for index, token in enumerate(ordered_tokens[:6]):
        features[f"prefix_pos{index}_{token}"] = 1.0
    for index, token in enumerate(ordered_tokens):
        features[f"absolute_pos{index}_{token}"] = 1.0
    for index, token in enumerate(reversed(ordered_tokens[-6:]), start=1):
        features[f"suffix_pos{index}_{token}"] = 1.0
    if ordered_tokens:
        runs: list[tuple[str, int]] = []
        for token in ordered_tokens:
            if runs and runs[-1][0] == token:
                runs[-1] = (token, runs[-1][1] + 1)
            else:
                runs.append((token, 1))
        features["token_run_count"] = float(len(runs))
        features["token_change_count"] = float(max(0, len(runs) - 1))
        for token in set(ordered_tokens):
            token_runs = [length for name, length in runs if name == token]
            features[f"run_count_{token}"] = float(len(token_runs))
            features[f"max_run_length_{token}"] = float(max(token_runs))
        special_positions = [
            index for index, token in enumerate(ordered_tokens)
            if token in _TRANSCENDENTAL or token in _REDUCTIONS
        ]
        if special_positions:
            first_special, last_special = special_positions[0], special_positions[-1]
            features["first_special_position"] = float(first_special)
            features["last_special_position"] = float(last_special)
            features["special_span"] = float(last_special - first_special)
            for token in set(ordered_tokens):
                features[f"before_special_count_{token}"] = float(
                    ordered_tokens[:first_special].count(token)
                )
                features[f"after_special_count_{token}"] = float(
                    ordered_tokens[last_special + 1:].count(token)
                )
        barriers = _TRANSCENDENTAL | _REDUCTIONS
        segments: list[list[str]] = [[]]
        for token in ordered_tokens:
            if token in barriers:
                segments.append([])
            else:
                segments[-1].append(token)
        nonempty_segments = [segment for segment in segments if segment]
        features["affine_segment_count"] = float(len(nonempty_segments))
        features["affine_segment_total_unique_ops"] = float(
            sum(len(set(segment)) for segment in nonempty_segments)
        )
        features["affine_segment_max_length"] = float(
            max((len(segment) for segment in nonempty_segments), default=0)
        )
        features["affine_segment_internal_changes"] = float(sum(
            sum(lhs != rhs for lhs, rhs in zip(segment, segment[1:]))
            for segment in nonempty_segments
        ))
        for token in set(ordered_tokens) - barriers:
            features[f"affine_segments_with_{token}"] = float(sum(
                token in segment for segment in nonempty_segments
            ))
        features["affine_segments_with_additive_and_multiply"] = float(sum(
            "multiply" in segment
            and ("add" in segment or "subtract" in segment)
            for segment in nonempty_segments
        ))
        for index, segment in enumerate(segments[:5]):
            features[f"affine_segment{index}_length"] = float(len(segment))
            features[f"affine_segment{index}_unique_ops"] = float(len(set(segment)))
    for name in [key for key in features if key.startswith("bigram_")]:
        features[f"{name}_x_multi"] = features[name] * features["multi_primitive"]
        features[f"{name}_x_tile2k_masked"] = features[name] * features["tile2k_masked"]
        features[f"{name}_x_tile2k_masked_x_multi"] = (
            features[name] * features["tile2k_masked"] * features["multi_primitive"]
        )
    return features


def _family_facts(region: dict[str, Any]) -> dict[str, Any]:
    reductions = int(region.get("reduction_count", 0))
    histogram = dict(region.get("op_histogram", {}))
    has_rsqrt = int(histogram.get("rsqrt", 0)) > 0
    ops = {
        name
        for name, count in histogram.items()
        if count and name not in _IGNORED_FAMILY_OPS
    }
    count = int(region.get("one_input_elementwise_count", 0)) + int(
        region.get("two_input_elementwise_count", 0)
    )
    return {
        "reductions": reductions,
        "has_rsqrt": has_rsqrt,
        "has_transcendental": int(region.get("transcendental_count", 0)) > 0,
        "has_partition_broadcast": int(region.get("partition_broadcast_input_count", 0))
        > 0,
        "ops": ops,
        "elementwise_count": count,
        "has_two_input": int(region.get("two_input_elementwise_count", 0)) > 0,
    }


GRAMMAR_RULES: tuple[GrammarRule, ...] = (
    GrammarRule(
        "reduction.two_with_rsqrt",
        100,
        lambda f: f["reductions"] >= 2 and f["has_rsqrt"],
        "two_reduction_rsqrt",
        "reductions >= 2 and has_rsqrt",
        "Two-or-more reductions interact with an rsqrt/Newton lowering.",
        ("two_reduction_rsqrt", "two_pass_reduce_affine"),
    ),
    GrammarRule(
        "reduction.one_with_rsqrt",
        100,
        lambda f: f["reductions"] == 1 and f["has_rsqrt"],
        "reduction_rsqrt",
        "reductions == 1 and has_rsqrt",
        "A single reduction feeds an rsqrt lowering.",
        ("reduce_rsqrt",),
    ),
    GrammarRule(
        "reduction.two_or_more",
        100,
        lambda f: (
            f["reductions"] >= 2 and not f["has_rsqrt"] and not f["has_transcendental"]
        ),
        "two_reduction",
        "reductions >= 2 and not has_rsqrt and not has_transcendental",
        "Two-or-more reductions lower as a distinct reduction grammar.",
        ("two_reduction",),
    ),
    GrammarRule(
        "reduction.with_transcendental",
        100,
        lambda f: (
            f["reductions"] >= 1 and not f["has_rsqrt"] and f["has_transcendental"]
        ),
        "reduction_transcendental",
        "reductions >= 1 and not has_rsqrt and has_transcendental",
        "A reduction is composed with a non-rsqrt transcendental operation.",
        ("softmax",),
    ),
    GrammarRule(
        "reduction.broadcast",
        100,
        lambda f: f["reductions"] == 1 and not f["has_transcendental"],
        "reduction_broadcast",
        "reductions == 1 and not has_transcendental",
        "A single reduction without a transcendental uses the reduction/broadcast grammar.",
        ("reduce_broadcast",),
    ),
    GrammarRule(
        "elementwise.partition_broadcast_multiply",
        100,
        lambda f: (
            f["reductions"] == 0
            and f["has_partition_broadcast"]
            and f["ops"] <= {"broadcast_to", "multiply"}
        ),
        "elementwise_broadcast_multiply",
        "no reductions and partition broadcast and ops subset of {broadcast_to,multiply}",
        "A partition-broadcast input feeds a multiply-only epilogue.",
        ("partition_broadcast_multiply",),
    ),
    GrammarRule(
        "elementwise.partition_broadcast_affine",
        100,
        lambda f: (
            f["reductions"] == 0
            and f["has_partition_broadcast"]
            and not f["ops"] <= {"broadcast_to", "multiply"}
        ),
        "elementwise_broadcast_affine",
        "no reductions and partition broadcast and non-multiply-only ops",
        "A partition-broadcast input feeds a non-multiply-only epilogue.",
        ("partition_broadcast_affine",),
    ),
    GrammarRule(
        "elementwise.multiply_chain",
        100,
        lambda f: (
            f["reductions"] == 0
            and not f["has_partition_broadcast"]
            and f["ops"] == {"multiply"}
        ),
        lambda f: f"elementwise_multiply_n{f['elementwise_count']}",
        "no reductions, no partition broadcast, and ops == {multiply}",
        "A multiply-only elementwise chain is instruction-selected separately.",
        ("elementwise_two",),
    ),
    GrammarRule(
        "elementwise.mixed",
        100,
        lambda f: (
            f["reductions"] == 0
            and not f["has_partition_broadcast"]
            and len(f["ops"]) >= 3
        ),
        "elementwise_mixed",
        "no reductions, no partition broadcast, and at least three distinct ops",
        "Three-or-more distinct elementwise operations form the mixed grammar.",
        ("mixed_epilogue",),
    ),
    GrammarRule(
        "elementwise.arity_chain",
        100,
        lambda f: (
            f["reductions"] == 0
            and not f["has_partition_broadcast"]
            and f["ops"] != {"multiply"}
            and len(f["ops"]) < 3
        ),
        lambda f: (
            f"elementwise_{'two' if f['has_two_input'] else 'one'}_n{f['elementwise_count']}"
        ),
        "no reductions, no partition broadcast, non-multiply-only, and fewer than three ops",
        "A non-multiply elementwise chain is keyed by input arity and operation count.",
        ("elementwise_one", "elementwise_two"),
    ),
)


def match_structural_family(
    region: dict[str, Any],
    *,
    rules: tuple[GrammarRule, ...] = GRAMMAR_RULES,
    strict: bool = False,
) -> GrammarMatch:
    """Classify a region and return the rule explanation and OOD diagnostics."""
    facts = _family_facts(region)
    matches = [rule for rule in rules if rule.predicate(facts)]
    if not matches:
        raise ValueError(f"No grammar rule matched region facts: {facts}")
    priority = max(rule.priority for rule in matches)
    winners = [rule for rule in matches if rule.priority == priority]
    if len(winners) != 1:
        ids = ", ".join(rule.rule_id for rule in winners)
        raise ValueError(f"Ambiguous grammar rules at priority {priority}: {ids}")

    unknown_ops = tuple(sorted(facts["ops"] - _KNOWN_FAMILY_OPS))
    ood_reasons = []
    if unknown_ops:
        ood_reasons.append("unknown_ops:" + ",".join(unknown_ops))
    if not facts["ops"] and facts["reductions"] == 0:
        ood_reasons.append("empty_elementwise_grammar")
    schema_version = int(region.get("schema_version", 1))
    if schema_version not in SUPPORTED_REGION_IR_SCHEMA_VERSIONS:
        ood_reasons.append(f"unsupported_schema:{schema_version}")
    if strict and ood_reasons:
        raise ValueError("Out-of-distribution region: " + "; ".join(ood_reasons))

    rule = winners[0]
    suffix = "_masked" if region.get("has_mask_or_tail") else ""
    context_suffix = ""
    if region.get("previous_family"):
        context_suffix += "__after_" + str(region["previous_family"])
    if region.get("next_family"):
        context_suffix += "__before_" + str(region["next_family"])
    suffix += context_suffix
    return GrammarMatch(
        family=rule.render_family(facts) + suffix,
        rule_id=rule.rule_id,
        rationale=rule.rationale,
        evidence=rule.evidence,
        ood_reasons=tuple(ood_reasons),
        consumed_features=(
            "reduction_count",
            "op_histogram",
            "transcendental_count",
            "partition_broadcast_input_count",
            "one_input_elementwise_count",
            "two_input_elementwise_count",
            "has_mask_or_tail",
            "previous_family",
            "next_family",
        ),
    )


def structural_family(region: dict[str, Any]) -> str:
    """Return the stable family string used by existing calibration tables."""
    return match_structural_family(region).family


def structural_calibration_key(region: dict[str, Any]) -> str:
    """Return an operator-name-free key for compiler instruction selection.

    A grammar family describes topology, but it intentionally does not fully
    distinguish primitives that lower to different NeuronCore engines.  The
    normalized source-op multiset supplies that missing compiler-facing
    evidence without introducing benchmark/operator names into the model.
    """
    match = match_structural_family(region)
    histogram = region.get("op_histogram") or {}
    ops = ",".join(
        f"{str(op).lower()}:{int(count)}"
        for op, count in sorted(histogram.items())
        if str(op).lower() not in _IGNORED_FAMILY_OPS
    )
    one = int(region.get("one_input_elementwise_count", 0))
    two = int(region.get("two_input_elementwise_count", 0))
    previous = str(region.get("previous_family") or "none")
    following = str(region.get("next_family") or "none")
    partition_count = max(
        1,
        int(
            region.get("logical_active_partition_count")
            or region.get("partition_count")
            or 1
        ),
    )
    partition_bucket = 1 << (partition_count.bit_length() - 1)
    partition_broadcast_count = max(
        0, int(region.get("partition_broadcast_input_count") or 0)
    )
    # Instruction selection distinguishes no broadcast, a single broadcast,
    # and a multi-broadcast grammar.  The exact number of broadcast operands
    # within the latter is source-DAG bookkeeping and otherwise prevents the
    # independent controls from matching equivalent compiler paths.
    partition_broadcast_bucket = min(partition_broadcast_count, 2)
    return (
        f"{match.rule_id}|ops={ops or 'none'}|arity={one}:{two}"
        f"|mask={int(bool(region.get('has_mask_or_tail')))}"
        f"|context={previous}>{following}"
        f"|blocks={max(1, int(region.get('free_block_count') or 1))}"
        f"|pbcast={partition_broadcast_bucket}"
        f"|p={partition_count}|pbucket={partition_bucket}"
    )
