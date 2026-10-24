"""Analytical cost model + timeline scheduler for NKI tile trace events.

This module consumes the JSONL event stream produced by
``triton_viz.tools.nki_trace_dump`` and turns it into a *predicted* per-engine
timeline.  It closes the loop that the trace dumper only starts:

    NKI kernel
      -> triton_viz.trace(client=Tracer(), frontend="nki_beta2")
      -> launches[-1].records
      -> nki_trace_dump.records_to_events        (WHAT ran: op/engine/memory/bytes)
      -> nki_cost_model.simulate                  (HOW LONG it takes + overlap)
      -> per-engine timeline + predicted latency  (this file)

The DMA path supports both a geometry-aware analytical fallback and an Inf2
hardware calibration surface loaded from the microbenchmark CSV. Compute
constants remain provisional until their per-engine calibration is integrated.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from triton_viz.tools.nki_features import AccessPattern, ComputeRegion

# ---------------------------------------------------------------------------
# Engine model
# ---------------------------------------------------------------------------
# NeuronCore exposes several independent engines plus DMA queues.  We model each
# as a single serial resource: instructions issued to the same engine run one
# after another, while different engines run in parallel.  This is deliberately
# coarse; it is enough to demonstrate compute/DMA overlap in the timeline.
ENGINE_DMA = "dma"
ENGINE_TENSOR = "tensor"
ENGINE_VECTOR = "vector"
ENGINE_SCALAR = "scalar"
ENGINE_GPSIMD = "gpsimd"
ENGINE_STATIC_DMA = "static_dma"
ENGINE_SYNC = "sync"


@dataclass
class DmaCalibrationSurface:
    """Measured aggregate GB/s indexed by (partitions, free bytes/partition)."""

    points: dict[tuple[int, int], float]

    @dataclass(frozen=True)
    class Lookup:
        bandwidth_gbps: float
        match: str
        requested_partitions: int
        requested_free_bytes: int
        lookup_partitions: int
        lookup_free_bytes: int
        log_distance: float

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        benchmark_name: str = "dma_partition_surface",
        bandwidth_column: str = "derived.read_gbps_dynamic_dma_active",
        dtype_name: str | None = None,
        required_repeat: int | None = None,
        duplicate_policy: str = "error",
    ) -> DmaCalibrationSurface:
        samples: dict[tuple[int, int], list[float]] = {}
        with Path(path).open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                if row.get("row_type") != "benchmark" or row.get("status") != "ok":
                    continue
                if row.get("spec.name") != benchmark_name:
                    continue
                if required_repeat is not None:
                    try:
                        if int(float(row.get("spec.repeat", 0))) != required_repeat:
                            continue
                    except (TypeError, ValueError):
                        continue
                if dtype_name is not None:
                    expected = {
                        "bf16": "bfloat16",
                        "fp32": "float32",
                        "fp16": "float16",
                    }.get(dtype_name, dtype_name)
                    observed = {
                        "bf16": "bfloat16",
                        "fp32": "float32",
                        "fp16": "float16",
                    }.get(
                        str(row.get("spec.dtype", "")), str(row.get("spec.dtype", ""))
                    )
                    if observed != expected:
                        continue
                try:
                    key = (
                        int(float(row["work.partition_count"])),
                        int(float(row["work.free_bytes_per_partition"])),
                    )
                    bandwidth = float(row[bandwidth_column])
                except (KeyError, TypeError, ValueError):
                    continue
                if (
                    key[0] <= 0
                    or key[1] <= 0
                    or not math.isfinite(bandwidth)
                    or bandwidth <= 0
                ):
                    continue
                samples.setdefault(key, []).append(bandwidth)
        if duplicate_policy not in {"error", "median"}:
            raise ValueError(f"Unknown duplicate policy: {duplicate_policy}")
        points: dict[tuple[int, int], float] = {}
        for key, values in samples.items():
            if duplicate_policy == "error" and any(
                not math.isclose(values[0], value, rel_tol=1e-9, abs_tol=1e-9)
                for value in values[1:]
            ):
                raise ValueError(
                    f"Conflicting calibration rows for {key}: {values}"
                )
            points[key] = (
                statistics.median(values)
                if duplicate_policy == "median"
                else values[0]
            )
        if not points:
            raise ValueError(
                f"No {benchmark_name} calibration rows with {bandwidth_column}"
                f" for dtype={dtype_name or '*'} in {path}"
            )
        return cls(points)

    def split(self, partitions: int) -> tuple[float, float] | None:
        """(fixed_ns, asymptotic GB/s) of a transfer over ``partitions``: the surface's own points at that
        partition count fitted as ``time = fixed + bytes / bandwidth``; between measured partition counts
        both are interpolated in log2(partitions), outside they are clamped. None when < 3 points."""
        cache = self.__dict__.setdefault("_split_cache", {})
        if partitions in cache:
            return cache[partitions]
        fits = {}
        for p in sorted({key[0] for key in self.points}):
            xs = [p * f for (pp, f) in self.points if pp == p]
            ys = [p * f / self.points[(pp, f)] for (pp, f) in self.points if pp == p]
            if len(xs) < 3:
                continue
            n = len(xs)
            mx, my = sum(xs) / n, sum(ys) / n
            var = sum((x - mx) ** 2 for x in xs)
            if var <= 0:
                continue
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var
            if slope <= 0:
                continue
            fits[p] = (max(0.0, my - slope * mx), 1.0 / slope)
        if not fits:
            cache[partitions] = None
            return None
        ps = sorted(fits)
        q = min(max(partitions, ps[0]), ps[-1])
        lo = max(p for p in ps if p <= q)
        hi = min(p for p in ps if p >= q)
        if lo == hi:
            result = fits[lo]
        else:
            w = (math.log2(q) - math.log2(lo)) / (math.log2(hi) - math.log2(lo))
            fixed = fits[lo][0] + w * (fits[hi][0] - fits[lo][0])
            inv = 1 / fits[lo][1] + w * (1 / fits[hi][1] - 1 / fits[lo][1])
            result = (fixed, 1.0 / inv)
        cache[partitions] = result
        return result

    def lookup(self, partitions: int, free_bytes: int) -> Lookup:
        """Return bandwidth plus explicit exact/interpolated/OOD provenance."""
        exact = self.points.get((partitions, free_bytes))
        if exact is not None:
            return self.Lookup(
                exact,
                "exact",
                partitions,
                free_bytes,
                partitions,
                free_bytes,
                0.0,
            )
        if partitions <= 0 or free_bytes <= 0:
            raise ValueError(
                f"DMA geometry must be positive, received "
                f"partitions={partitions}, free_bytes={free_bytes}"
            )
        measured_p = [point[0] for point in self.points]
        measured_f = [point[1] for point in self.points]
        clamped_p = min(max(partitions, min(measured_p)), max(measured_p))
        clamped_f = min(max(free_bytes, min(measured_f)), max(measured_f))
        target_p = math.log2(clamped_p)
        target_f = math.log2(clamped_f)
        nearest = sorted(
            self.points.items(),
            key=lambda item: (
                (math.log2(item[0][0]) - target_p) ** 2
                + (math.log2(item[0][1]) - target_f) ** 2
            ),
        )[:4]
        weighted = 0.0
        weight_sum = 0.0
        for (p, f), bandwidth in nearest:
            distance = math.hypot(math.log2(p) - target_p, math.log2(f) - target_f)
            weight = 1.0 / max(distance, 1e-9)
            weighted += weight * bandwidth
            weight_sum += weight
        requested_distance = math.hypot(
            math.log2(partitions) - target_p,
            math.log2(free_bytes) - target_f,
        )
        return self.Lookup(
            weighted / weight_sum,
            "interpolated" if requested_distance == 0 else "ood_clamped",
            partitions,
            free_bytes,
            clamped_p,
            clamped_f,
            requested_distance,
        )

    def bandwidth_gbps(self, partitions: int, free_bytes: int) -> float:
        """Backward-compatible numeric lookup."""
        return self.lookup(partitions, free_bytes).bandwidth_gbps

    def in_domain(self, partitions: int, free_bytes: int) -> bool:
        return self.lookup(partitions, free_bytes).match != "ood_clamped"


@dataclass
class StructuralStaticDmaCalibration:
    """Compiler-generated Static DMA busy time keyed by structural grammar."""

    points: dict[tuple[str, int, int, int], float]
    padded_points: dict[tuple[int, int, int], float] = field(default_factory=dict)

    @classmethod
    def from_csv(cls, path: str | Path) -> StructuralStaticDmaCalibration:
        samples: dict[tuple[str, int, int, int], list[float]] = {}
        padded_samples: dict[tuple[int, int, int], list[float]] = {}
        with Path(path).open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                if row.get("calibration_mode") == "padded_partition_shape":
                    try:
                        padded_key = (
                            int(row["element_bytes"]),
                            int(row["logical_partition_count"]),
                            int(row["logical_free_dim"]),
                        )
                        padded_value = float(row["static_dma_ns"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if min(padded_key) > 0 and padded_value >= 0:
                        padded_samples.setdefault(padded_key, []).append(padded_value)
                    continue
                try:
                    sequences = {
                        row["structural_rule_sequence"],
                        row.get("structural_calibration_sequence") or "",
                    }
                    element_bytes = int(row["element_bytes"])
                    partitions = int(row.get("partition_count") or 0)
                    free_dim = int(row["logical_free_dim"])
                    value = float(row["static_dma_ns"])
                except (KeyError, TypeError, ValueError):
                    continue
                for sequence in sequences:
                    key = (sequence, element_bytes, partitions, free_dim)
                    if sequence and element_bytes > 0 and free_dim > 0 and value >= 0:
                        samples.setdefault(key, []).append(value)
        points = {
            key: statistics.median(values) for key, values in samples.items()
        }
        padded_points = {
            key: statistics.median(values)
            for key, values in padded_samples.items()
        }
        if not points and not padded_points:
            raise ValueError(f"No structural Static DMA calibration rows in {path}")
        return cls(points, padded_points)

    def predict_ns_with_provenance(
        self, events: Iterable[dict[str, Any]]
    ) -> tuple[float, str]:
        """Predict Static-DMA time and expose the source-only lookup path."""
        regions: dict[int, dict[str, Any]] = {}
        element_bytes = 0
        for event in events:
            if event.get("region_ir") is not None:
                regions[int(event["fusion_group"])] = event["region_ir"]
            if not element_bytes and event.get("op") in {"load", "store"}:
                lanes = int(event.get("active_lanes") or 0)
                nbytes = int(event.get("bytes") or 0)
                if lanes > 0 and nbytes > 0 and nbytes % lanes == 0:
                    element_bytes = nbytes // lanes
        if not regions or element_bytes <= 0:
            return 0.0, "none"
        from triton_viz.tools.nki_region_ir import (
            match_structural_family,
            structural_calibration_key,
        )

        calibration_sequence = ";".join(
            structural_calibration_key(regions[group]) for group in sorted(regions)
        )
        rule_sequence = ";".join(
            match_structural_family(regions[group]).rule_id for group in sorted(regions)
        )
        free_dim = max(
            int(region.get("logical_free_dim") or 0) for region in regions.values()
        )
        partition_count = max(
            int(region.get("partition_count") or 1) for region in regions.values()
        )
        # PMAX-padded kernels expose their physical 128-row tile and mask in
        # the source trace.  Recover logical rows from active lanes; this is an
        # independently calibrated source geometry, not a target case key.
        if self.padded_points and partition_count == 128 and free_dim > 0:
            logical_partitions = max(
                (
                    int(event.get("active_lanes") or 0) // free_dim
                    for event in events
                    if event.get("op") in {"load", "store"}
                    and int(event.get("active_lanes") or 0) >= free_dim
                    and int(event.get("active_lanes") or 0) % free_dim == 0
                ),
                default=0,
            )
            padded = self.padded_points.get(
                (element_bytes, logical_partitions, free_dim)
            )
            if padded is not None:
                return padded, "padded_exact"
        candidates = []
        match = "none"
        for sequence, candidate_match in (
            (calibration_sequence, "structural_key"),
            (rule_sequence, "rule_sequence"),
        ):
            candidates = [
                (point_free_dim, value)
                for (
                    point_sequence,
                    point_bytes,
                    point_partitions,
                    point_free_dim,
                ), value in self.points.items()
                if point_sequence == sequence and point_bytes == element_bytes
                and point_partitions in {0, partition_count}
            ]
            if candidates:
                match = candidate_match
                break
        if not candidates or free_dim <= 0:
            return 0.0, "none"
        value = min(
            candidates,
            key=lambda item: abs(math.log2(item[0]) - math.log2(free_dim)),
        )[1]
        return value, match

    def predict_ns(self, events: Iterable[dict[str, Any]]) -> float:
        """Backward-compatible numeric Static-DMA lookup."""
        return self.predict_ns_with_provenance(events)[0]


@dataclass
class TensorCalibrationSurface:
    """Control-only TensorE throughput calibration keyed by operand dtype.

    TensorE active time is fit as ``startup + flops / throughput`` across every
    measured FLOP value of one dtype. The calibration has no per-source-Dot
    table and no tile-shape key: an unseen Dot (for example a small attention
    tile) is priced by its FLOPs through the fitted line, and its FLOP-domain
    position is exposed so replay can flag below/above-domain extrapolation
    instead of silently claiming an in-domain match. All samples come from
    independent control microbenchmarks; operator traces are never accepted.
    """

    points: dict[str, tuple[float, float]]
    flops_domain: dict[str, tuple[float, float]]
    active_time_points: dict[str, list[tuple[float, float]]] = field(
        default_factory=dict
    )

    @staticmethod
    def _normalize_dtype(dtype: str) -> str:
        value = str(dtype or "").strip().lower()
        aliases = {
            "float32": "float32",
            "fp32": "float32",
            "float16": "float16",
            "fp16": "float16",
            "bfloat16": "bfloat16",
            "bf16": "bfloat16",
            "float8_e5m2": "float8_e5m2",
            "fp8_e5m2": "float8_e5m2",
            "float8_e4m3fn": "float8_e4m3fn",
            "fp8_e4m3fn": "float8_e4m3fn",
        }
        return aliases.get(value, value)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        benchmark_name: str = "tensor_matmul",
        duplicate_policy: str = "median",
    ) -> "TensorCalibrationSurface":
        samples: dict[str, dict[int, list[float]]] = {}
        with Path(path).open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                if row.get("row_type") not in (None, "benchmark"):
                    continue
                if row.get("status") != "ok":
                    continue
                if row.get("kind") not in {
                    benchmark_name,
                    "tensor_matmul_small",
                }:
                    continue
                dtype = cls._normalize_dtype(row.get("spec.dtype") or "")
                if not dtype:
                    continue
                try:
                    flops = int(float(row["work.matmul_flops"]))
                    active_ns = (
                        float(row["profile.tensor_engine_active_time"]) * 1e9
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                if (
                    flops <= 0
                    or not math.isfinite(active_ns)
                    or active_ns <= 0
                ):
                    continue
                samples.setdefault(dtype, {}).setdefault(flops, []).append(
                    active_ns
                )
        if duplicate_policy not in {"error", "median"}:
            raise ValueError(f"Unknown duplicate policy: {duplicate_policy}")
        points: dict[str, tuple[float, float]] = {}
        flops_domain: dict[str, tuple[float, float]] = {}
        for dtype, by_flops in samples.items():
            measured_flops = sorted(by_flops)
            active_times = [
                statistics.median(by_flops[flops])
                if duplicate_policy == "median"
                else by_flops[flops][0]
                for flops in measured_flops
            ]
            if len(measured_flops) >= 2:
                slope, intercept = statistics.linear_regression(
                    measured_flops, active_times
                )
                if slope > 0 and math.isfinite(slope):
                    points[dtype] = (1.0 / slope, max(0.0, intercept))
                    flops_domain[dtype] = (measured_flops[0], measured_flops[-1])
                    continue
            points[dtype] = (measured_flops[0] / active_times[0], 0.0)
            flops_domain[dtype] = (measured_flops[0], measured_flops[-1])
        if not points:
            raise ValueError(f"No tensor_matmul calibration rows in {path}")
        active_time_points = {
            dtype: [
                (float(flops), float(statistics.median(by_flops[flops])))
                for flops in sorted(by_flops)
            ]
            for dtype, by_flops in samples.items()
        }
        return cls(points, flops_domain, active_time_points)

    def _lookup(self, dtype: str, *, strict: bool) -> tuple[float, float]:
        dtype = self._normalize_dtype(dtype)
        exact = self.points.get(dtype)
        if exact is not None:
            return exact
        if strict:
            raise ValueError(f"Missing exact TensorE calibration for dtype={dtype}")
        fallback = self.points.get("float32")
        if fallback is None and self.points:
            fallback = next(iter(self.points.values()))
        if fallback is None:
            return 90000.0, 0.0
        return fallback

    def flops_per_ns(self, dtype: str, *, strict: bool = False) -> float:
        """Return calibrated steady-state TensorE throughput in FLOPs/ns."""
        return self._lookup(dtype, strict=strict)[0]

    def startup_ns(self, dtype: str, *, strict: bool = False) -> float:
        """Return the once-per-kernel TensorE startup intercept."""
        return self._lookup(dtype, strict=strict)[1]

    def active_ns(self, dtype: str, flops: float, *, strict: bool = False) -> float:
        """Interpolate total TensorE active time on the control FLOP surface."""
        normalized = self._normalize_dtype(dtype)
        rows = sorted(self.active_time_points.get(normalized, ()))
        if not rows:
            throughput, startup = self._lookup(dtype, strict=strict)
            return startup + max(0.0, flops) / throughput
        if len(rows) == 1:
            return rows[0][1]
        exact = [active for measured, active in rows if measured == flops]
        if exact:
            return statistics.median(exact)
        if flops <= rows[0][0]:
            lower, upper = rows[0], rows[1]
        elif flops >= rows[-1][0]:
            lower, upper = rows[-2], rows[-1]
        else:
            lower = max(row for row in rows if row[0] <= flops)
            upper = min(row for row in rows if row[0] >= flops)
        weight = (flops - lower[0]) / (upper[0] - lower[0])
        return max(0.0, lower[1] + weight * (upper[1] - lower[1]))

    def domain_match(self, dtype: str, flops: float) -> str:
        """Classify one Dot's FLOPs against the fitted control domain."""
        dtype = self._normalize_dtype(dtype)
        domain = self.flops_domain.get(dtype) or self.flops_domain.get("float32")
        if domain is None:
            return "missing_domain"
        low, high = domain
        if flops < low:
            return "below_domain"
        if flops > high:
            return "above_domain"
        return "in_domain"


@dataclass
class TensorDotCountCalibration:
    """Control-only TensorE surface keyed by source-visible tiled-Dot geometry."""

    points: dict[str, tuple[float, float, float, float, float]]
    revisit_ns: dict[str, float] = field(default_factory=dict)
    geometry_domains: dict[str, list[list[float]]] = field(default_factory=dict)

    @classmethod
    def from_csv(cls, path: str | Path) -> "TensorDotCountCalibration":
        points, revisit_ns, domains = {}, {}, {}
        with Path(path).open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                dtype = TensorCalibrationSurface._normalize_dtype(row["dtype"])
                version = row.get("geometry_model")
                if version not in (None, "", "tiled-revisit-v1"):
                    raise ValueError(f"Unknown Tensor geometry model: {version}")
                if not version and row.get("revisit_ns"):
                    raise ValueError("Tensor revisit coefficient requires a model version")
                if version == "tiled-revisit-v1":
                    revisit_ns[dtype] = float(row["revisit_ns"])
                    if not math.isfinite(revisit_ns[dtype]) or revisit_ns[dtype] < 0:
                        raise ValueError("Invalid Tensor revisit coefficient")
                    domain = json.loads(row["geometry_domain"])
                    if len(domain) != 5 or any(
                        len(bounds) != 2 or not all(math.isfinite(v) for v in bounds)
                        or bounds[0] > bounds[1] for bounds in domain
                    ):
                        raise ValueError("Invalid Tensor geometry domain")
                    domains[dtype] = domain
                points[dtype] = (
                    float(row["startup_ns"]), float(row["dot_ns"]),
                    float(row.get("lhs_tile_ns") or 0.0),
                    float(row.get("rhs_tile_ns") or 0.0),
                    float(row.get("output_tile_ns") or 0.0),
                )
        if not points:
            raise ValueError(f"No source-Dot Tensor calibration rows in {path}")
        return cls(points, revisit_ns, domains)

    def active_ns(self, dtype: str, dot_count: int, lhs_tiles: int = 0,
                  rhs_tiles: int = 0, output_tiles: int = 0) -> tuple[float, str]:
        dtype = TensorCalibrationSurface._normalize_dtype(dtype)
        point = self.points.get(dtype)
        if point is None or dot_count <= 0:
            return 0.0, "missing"
        startup, dot_ns, lhs_ns, rhs_ns, output_ns = point
        total = (startup + dot_ns * dot_count + lhs_ns * lhs_tiles
                 + rhs_ns * rhs_tiles + output_ns * output_tiles)
        if dtype not in self.revisit_ns:
            return total, "source_geometry"
        # For tiled matmul this is mt * (nt - 1) * kt**2. It describes
        # repeated K-tile work across N tiles using source geometry alone.
        revisit = lhs_tiles * rhs_tiles * (1.0 - lhs_tiles / dot_count)
        if dtype != "bfloat16":
            revisit = 0.0
        geometry = [dot_count, lhs_tiles, rhs_tiles, output_tiles, revisit]
        ood = any(value < low or value > high for value, (low, high)
                  in zip(geometry, self.geometry_domains[dtype]))
        total += self.revisit_ns[dtype] * revisit
        return max(0.0, total), ("source_geometry_revisit_ood" if ood
                                else "source_geometry_revisit")


@dataclass
class AttentionPipelineCalibration:
    """Control-only TensorE busy surface for QK-normalize-PV Dot DAGs.

    This is an engine occupancy model, not a completion floor: it supplies the
    TensorE active time that source geometry alone cannot identify for a
    two-Dot normalise-and-reuse pipeline.
    """

    points: dict[str, list[tuple[int, float]]]

    @classmethod
    def from_csv(cls, path: str | Path) -> "AttentionPipelineCalibration":
        points: dict[str, list[tuple[int, float]]] = {}
        with Path(path).open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                dtype = TensorCalibrationSurface._normalize_dtype(row["dtype"])
                points.setdefault(dtype, []).append(
                    (int(row["value_width"]), float(row["tensor_active_ns"]))
                )
        for values in points.values():
            values.sort()
        if not points:
            raise ValueError(f"No attention-pipeline calibration rows in {path}")
        return cls(points)

    def predict_ns(self, dtype: str, value_width: int) -> tuple[float, str]:
        values = self.points.get(TensorCalibrationSurface._normalize_dtype(dtype), [])
        if len(values) < 2 or value_width <= 0:
            return 0.0, "missing"
        match = "interpolated"
        if value_width <= values[0][0]:
            lower, upper = values[0], values[1]
            match = "ood_extrapolated"
        elif value_width >= values[-1][0]:
            lower, upper = values[-2], values[-1]
            match = "ood_extrapolated" if value_width > values[-1][0] else "exact"
        else:
            upper_index = next(i for i, row in enumerate(values) if row[0] >= value_width)
            lower, upper = values[upper_index - 1], values[upper_index]
            if upper[0] == value_width:
                return upper[1], "exact"
        weight = (value_width - lower[0]) / (upper[0] - lower[0])
        return lower[1] + weight * (upper[1] - lower[1]), match


@dataclass
class DmaElapsedCalibration:
    """Descriptor-issue bound on DMA *elapsed* time.

    A DMA transfer's bandwidth model prices the bytes it moves, but the queue
    is occupied per *descriptor*, not per byte.  A contiguous run costs one
    descriptor; a fragmented free axis costs one descriptor per element, and
    the independent strided-store controls show the resulting elapsed time can
    exceed measured DMA active time by more than 10x.  The physical bound is

        ``elapsed_ns >= descriptor_count * ns_per_descriptor``

    where ``descriptor_count`` comes from the access geometry alone
    (partition count and free-axis stride) and ``ns_per_descriptor`` is one
    global measured issue interval.  It never lowers the bandwidth estimate.
    """

    ns_per_descriptor: float
    measured_min_descriptors: float = 0.0
    measured_max_descriptors: float = float("inf")

    @classmethod
    def from_csv(cls, path: str | Path) -> "DmaElapsedCalibration":
        with Path(path).open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        if len(rows) != 1:
            raise ValueError(
                f"DMA elapsed calibration must be a single global row: {path}"
            )
        return cls(
            ns_per_descriptor=float(rows[0]["ns_per_descriptor"]),
            measured_min_descriptors=float(rows[0].get("measured_min_descriptors") or 0.0),
            measured_max_descriptors=float(
                rows[0].get("measured_max_descriptors") or float("inf")
            ),
        )

    @staticmethod
    def serial_descriptor_count(pattern: AccessPattern, engines: int) -> float:
        """Descriptors that must issue back to back on one queue.

        A contiguous (or stride-0 broadcast) free axis coalesces into one bulk
        descriptor per partition, and those descriptors are striped across the
        DMA engines, so their issue cost is ``ceil(partitions / engines)`` and
        is negligible beside the bandwidth term.  A fragmented free axis
        instead produces one descriptor per active element on a single queue,
        which is the regime the independent strided controls measure.
        """
        if pattern.active_access_count <= 0:
            return 0.0
        if pattern.free_stride_items in (0, 1):
            partitions = max(1, pattern.partition_count)
            return math.ceil(partitions / max(1, engines))
        return float(pattern.active_access_count)

    def queue_floor_ns(
        self, events: Iterable[dict[str, Any]], engines: int
    ) -> tuple[float, float, float]:
        """Least DMA-resource occupancy implied by descriptor issue.

        Descriptors from every transfer share the same issue path, so the bound
        is on their total, not on any single transfer.  It is a floor on DMA
        engine busy time, never an addition to the bandwidth estimate.

        Returns the floor, the total descriptor count, and the count coming
        from fragmented transfers, which is the quantity the controls measured
        and therefore the one whose domain must be reported.
        """
        descriptors = 0.0
        fragmented = 0.0
        for event in events:
            pattern = AccessPattern.from_event(event)
            if pattern is None:
                continue
            count = self.serial_descriptor_count(pattern, engines)
            descriptors += count
            if pattern.free_stride_items not in (0, 1):
                fragmented += count
        return descriptors * self.ns_per_descriptor, descriptors, fragmented

    def fragmented_out_of_domain(self, fragmented_descriptors: float) -> bool:
        """Whether a fragmented transfer total leaves the measured domain."""
        if fragmented_descriptors <= 0:
            return False
        return not (
            self.measured_min_descriptors
            <= fragmented_descriptors
            <= self.measured_max_descriptors
        )


@dataclass
class OnChipTransferCalibration:
    """Measured PSUM/SBUF on-chip copy latency, keyed by generic geometry.

    An on-chip transfer is not an HBM transfer: no bandwidth surface applies to
    it, and the engine that performs it is occupied for a latency that grows
    with the free width of the tile being moved.  Independent ``onchip_copy``
    controls isolate that latency by differencing repeats, so the fit contains
    no setup or store cost.  The model is ``startup + ns_per_free_elem * free``
    keyed by ``(engine, dtype)`` with an explicit measured free-width domain;
    it replaces a hardcoded placeholder and is not keyed by program structure.
    """

    points: dict[tuple[str, str], tuple[float, float, int, int]]

    @classmethod
    def from_csv(cls, path: str | Path) -> "OnChipTransferCalibration":
        points: dict[tuple[str, str], tuple[float, float, int, int]] = {}
        with Path(path).open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                key = (
                    str(row["engine"]).strip().lower(),
                    ComputeCalibration._norm_dtype(row["dtype"]),
                )
                points[key] = (
                    float(row["startup_ns"]),
                    float(row["ns_per_free_elem"]),
                    int(float(row["domain_min_free"])),
                    int(float(row["domain_max_free"])),
                )
        if not points:
            raise ValueError(f"No on-chip transfer calibration rows in {path}")
        return cls(points)

    def latency_ns(
        self, engine: str, dtype: str, free_elements: int
    ) -> tuple[float, str]:
        key = (str(engine).strip().lower(), ComputeCalibration._norm_dtype(dtype))
        point = self.points.get(key)
        if point is None:
            # The controls measured the Vector copy path; other engines reuse
            # that latency rather than inventing a second unmeasured surface.
            point = next(
                (
                    value
                    for (candidate, candidate_dtype), value in self.points.items()
                    if candidate_dtype == key[1]
                ),
                None,
            )
            if point is None:
                return 0.0, "missing"
        startup, per_elem, low, high = point
        free = max(1, int(free_elements))
        match = "in_domain" if low <= free <= high else "ood_extrapolated"
        return max(0.0, startup + per_elem * free), match


@dataclass
class GlobalCompletionCalibration:
    """One global, structure-free NC completion model.

    Independent control programs show that end-to-end NC latency is not the
    dependency makespan: a fixed fraction of the non-critical engines' work
    fails to overlap with the critical engine, and every kernel pays a fixed
    launch/drain cost.  Both quantities are single global numbers measured as
    ``NC-p50 minus measured engine active time`` over independent controls and
    gated by leave-one-entire-free-dimension-out CV.  There is deliberately no
    structural, operator or grammar key: this term is applied identically to
    every program, which is what distinguishes it from the per-structure
    completion lookup tables it replaces.

    A pipeline-depth refinement of ``overlap_fraction`` (making the residue of
    each engine depend on how many independent work items the source DAG gives
    it) was fitted and cross-validated on a control set spanning both a serial
    whole-program family and a software-pipelined tiled-Dot family.  It won the
    control CV decisively and still failed to improve the authoritative target
    metric, so it was rejected; see the 2026-08-30 status entry.
    """

    overlap_fraction: float
    completion_offset_ns: float
    # How much the non-overlapped fraction grows with engine-load imbalance.
    # When one engine dominates, the others hide under it; when several carry
    # comparable work they contend for issue slots and serialise.  The measure
    # is ``(sum(busy) - max(busy)) / max(busy)``, computed from the model's own
    # predicted busy vector -- no structural, operator or grammar key.  Zero
    # reduces this to the single-constant form.
    overlap_imbalance_slope: float = 0.0
    # Launch/drain cost that scales with how many SBUF partitions the program
    # activates.  Partition setup and the cross-partition sync that drains a
    # kernel are tree-structured, so any growth should be logarithmic, not
    # linear.  ``partitions`` is source-visible geometry, not a structural key.
    #
    # NOTE: this term fits to **0.0** on the full 19-kind control set.  It was
    # first measured at 100 ns/step on a control set that was missing seven
    # region-control kinds, which left ``reduction.two_or_more`` with 5
    # calibration points instead of 53.  Once that Level-A conditioning defect
    # is repaired the partition term is no longer supported: it had been
    # absorbing the defect, not modelling partition setup.  The mechanism is
    # retained because a different control mix could support it and the fitter
    # decides per calibration, but on the current production controls it is
    # inert and the model reduces to the imbalance form.
    completion_offset_ns_per_log2_partition: float = 0.0
    # Opt-in structure-aware completion ("carried serialization").  The
    # structure-free term above cannot distinguish rewrites that keep every
    # engine's work but change how iterations overlap.  When enabled, the share
    # ``c`` of HBM->SBUF load bytes whose destination buffer is reused across
    # loads (hoisted allocations, hand-written ping-pong buffers, unrolled
    # loops) serialises:  f_eff = f + (1 - f) * c.  Loop-local tiles (a fresh
    # buffer per iteration, which the compiler multi-buffers) keep c = 0, so
    # such programs are priced exactly as before.  No parameter is fitted.
    carried_serialization: bool = False

    @classmethod
    def from_csv(cls, path: str | Path) -> "GlobalCompletionCalibration":
        with Path(path).open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        if len(rows) != 1:
            raise ValueError(
                f"Global completion calibration must be a single global row: {path}"
            )
        return cls(
            overlap_fraction=float(rows[0]["overlap_fraction"]),
            completion_offset_ns=float(rows[0]["completion_offset_ns"]),
            overlap_imbalance_slope=float(
                rows[0].get("overlap_imbalance_slope") or 0.0
            ),
            completion_offset_ns_per_log2_partition=float(
                rows[0].get("completion_offset_ns_per_log2_partition") or 0.0
            ),
        )

    def effective_overlap_fraction(self, critical: float, residue: float) -> float:
        """Non-overlapped fraction of the residue, clamped to [0, 1]."""
        if critical <= 0.0:
            return self.overlap_fraction
        raw = self.overlap_fraction + self.overlap_imbalance_slope * (
            residue / critical
        )
        return min(1.0, max(0.0, raw))

    def offset_ns(self, partitions: int = 1) -> float:
        """Launch/drain cost for a program activating ``partitions`` lanes."""
        return self.completion_offset_ns + (
            self.completion_offset_ns_per_log2_partition
            * math.log2(max(1, partitions))
        )

    def predict_ns(
        self,
        engine_busy_ns: dict[str, float],
        partitions: int = 1,
        carried_fraction: float = 0.0,
    ) -> float:
        if not engine_busy_ns:
            return max(0.0, self.offset_ns(partitions))
        busy = list(engine_busy_ns.values())
        critical = max(busy)
        fraction = self.effective_overlap_fraction(critical, sum(busy) - critical)
        if self.carried_serialization:
            carried = min(1.0, max(0.0, carried_fraction))
            fraction = min(1.0, fraction + (1.0 - fraction) * carried)
        return max(
            0.0,
            critical + fraction * (sum(busy) - critical) + self.offset_ns(partitions),
        )


def carried_load_fraction(events: Iterable[dict[str, Any]]) -> float:
    """Share of HBM->SBUF load bytes whose destination buffer receives more than one load."""
    per: dict[Any, list[int]] = {}
    for event in events:
        if (
            event.get("op") == "transfer"
            and event.get("mem_src") == "hbm"
            and event.get("mem_dst") == "sbuf"
        ):
            key = event.get("dst_storage") or event.get("dst_ptr")
            row = per.setdefault(key, [0, 0])
            row[0] += 1
            row[1] += int(event.get("bytes") or 0)
    total = sum(nbytes for _, nbytes in per.values())
    if total <= 0:
        return 0.0
    return sum(nbytes for count, nbytes in per.values() if count >= 2) / total


@dataclass
class ComputeCalibration:
    """Measured per-*instruction* cost for VectorE/ScalarE compute engines.

    Level B of the lowering-aware compute model (Status.md goal 1). Each hardware
    compute instruction over a tile with free dimension ``F`` costs
    ``startup_ns + F * ns_per_free_elem`` (partition-parallel: cost tracks the
    free axis, not total elements). Keyed by ``(engine, dtype, input_streams)``
    so one-input vs two-input and FP32 vs BF16 can differ. This is the cost of a
    *single* lowered instruction. An optional CSV ``issue_floor_ns`` column
    supplies one shared minimum issue interval per engine/dtype, giving
    ``max(issue_floor, startup + F * ns_per_free_elem)``. Missing floors preserve
    the legacy affine path exactly. How many instructions a source ``nl.*`` op
    expands into is supplied separately by the Level-A expansion table.
    """

    points: dict
    # Optional enqueue/resource floor shared by all stream counts of an engine
    # and dtype. Empty for every legacy CSV and direct constructor.
    issue_floors_ns: dict = field(default_factory=dict)

    @classmethod
    def from_csv(cls, path):
        points = {}
        floors = {}
        with Path(path).open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                try:
                    key = (
                        str(row["engine"]).strip().lower(),
                        str(row["dtype"]).strip().lower(),
                        int(float(row["input_stream_count"])),
                    )
                    value = (float(row["startup_ns"]), float(row["ns_per_free_elem"]))
                except (KeyError, TypeError, ValueError):
                    continue
                if value[0] < 0 or value[1] < 0:
                    continue
                raw_floor = row.get("issue_floor_ns")
                if raw_floor is not None and str(raw_floor).strip():
                    floor = float(raw_floor)
                    if not math.isfinite(floor) or floor < 0 or not all(math.isfinite(v) for v in value):
                        raise ValueError(f"Invalid compute issue floor/calibration for {key}")
                    floor_key = (key[0], cls._norm_dtype(key[1]))
                    if floor_key in floors and floors[floor_key] != floor:
                        raise ValueError(f"Conflicting engine/dtype issue floors for {floor_key}")
                    floors[floor_key] = floor
                points[key] = value
        if not points:
            raise ValueError(f"No compute calibration rows in {path}")
        return cls(points, floors)

    @staticmethod
    def _norm_dtype(dtype):
        d = (dtype or "").lower()
        if d in ("float32", "fp32", "f32"):
            return "float32"
        if d in ("float16", "fp16", "f16"):
            return "float16"
        if d in ("bfloat16", "bf16"):
            return "bfloat16"
        return d or "float32"

    def instruction_lookup(
        self, engine, dtype, input_streams, free_dim, *, strict_dtype=False
    ):
        """Return ``(nanoseconds, match_kind)`` for one lowered instruction."""
        engine = (engine or "").lower()
        dtype = self._norm_dtype(dtype)
        streams = max(1, int(input_streams))
        stream_class = 2 if streams >= 2 else 1
        candidates = [
            ((engine, dtype, streams), "exact"),
            ((engine, dtype, stream_class), "streams_fallback"),
            (("derived", engine, dtype), "derived_streams"),
        ]
        if not strict_dtype:
            candidates.extend(
                [
                    ((engine, "float32", streams), "dtype_fallback"),
                    ((engine, "float32", stream_class), "dtype_streams_fallback"),
                ]
            )
        seen = set()
        for key, match in candidates:
            if key in seen:
                continue
            seen.add(key)
            if match == "derived_streams":
                hit = (
                    self._derived_two_input(engine, dtype)
                    if stream_class == 2
                    else None
                )
            else:
                hit = self.points.get(key)
            if hit is not None:
                startup, per_elem = hit
                duration = startup + max(0, int(free_dim)) * per_elem
                calibrated_dtype = key[2] if match == "derived_streams" else key[1]
                floor = self.issue_floors_ns.get((engine, calibrated_dtype))
                if floor is None:
                    return duration, match
                if not math.isfinite(floor) or floor < 0 or not math.isfinite(duration):
                    raise ValueError("Invalid compute issue floor or affine duration")
                return max(floor, duration), match
        return None, "missing"

    def _derived_two_input(self, engine, dtype):
        """Two-input cost derived from the same engine's one-input fit.

        Some engine/dtype pairs are only calibrated for one input stream (e.g.
        two-input ScalarE instructions, which appear in LayerNorm-style
        kernels). The startup cost is kept and the per-element cost is scaled
        by VectorE's measured two-input / one-input ratio for the same dtype.
        Returns ``None`` when any of the three required fits is missing.
        """
        one = self.points.get((engine, dtype, 1))
        vector_one = self.points.get(("vector", dtype, 1))
        vector_two = self.points.get(("vector", dtype, 2))
        if one is None or vector_one is None or vector_two is None:
            return None
        if vector_one[1] <= 0:
            return None
        return one[0], one[1] * vector_two[1] / vector_one[1]

    def instruction_ns(
        self, engine, dtype, input_streams, free_dim, *, strict_dtype=False
    ):
        """Return one lowered instruction's cost, or None if uncalibrated."""
        return self.instruction_lookup(
            engine,
            dtype,
            input_streams,
            free_dim,
            strict_dtype=strict_dtype,
        )[0]


@dataclass
class InstructionLatencyCalibration:
    """Consumer-visible latency of one compute instruction (opt-in).

    Level-B (`ComputeCalibration`) was fitted on *independent* instruction streams and so measures
    the initiation interval: how soon the engine can start the next instruction. A consumer that
    reads the instruction's result must instead wait for its latency, which dependent-chain controls
    measure separately (`microbench/inf2_nki/configs/structure_latency_v1.json`, mode
    `dependent_chain`). The scheduler keeps the engine busy for the initiation interval and makes the
    result visible after ``max(initiation interval, latency)``, so dependent chains serialize on
    latency while independent streams keep their throughput. One linear law per
    ``(engine, dtype, input_streams)``: ``latency = startup_ns + ns_per_free_elem * free``. The law is
    applied only within the free sizes the controls covered (optional CSV column ``max_free_elem``);
    outside that range no latency is modelled and the result is visible at the calibrated duration.
    """

    points: dict[tuple[str, str, int], tuple[float, float, float]]

    @classmethod
    def from_csv(cls, path: str | Path) -> "InstructionLatencyCalibration":
        points: dict[tuple[str, str, int], tuple[float, float, float]] = {}
        with Path(path).open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                limit = float(row["max_free_elem"]) if row.get("max_free_elem") else math.inf
                points[(row["engine"], row["dtype"], int(row["input_stream_count"]))] = (
                    float(row["startup_ns"]), float(row["ns_per_free_elem"]), limit)
        return cls(points)

    def latency_ns(self, engine: str, dtype: str, input_streams: int, free_dim: int) -> float | None:
        point = self.points.get((engine, dtype, int(input_streams)))
        if point is None:
            point = self.points.get((engine, dtype, 1))
        if point is None:
            return None
        startup, slope, *limit = point
        free = max(1, int(free_dim))
        if limit and free > limit[0]:
            return None
        return max(0.0, startup + slope * free)


@dataclass
class TensorInstructionLaw:
    """TensorE time per matmul instruction from its geometry (opt-in).

    The PE array is 128 x 128. A matmul instruction loads the stationary operand (K x M, LDWEIGHTS)
    and streams the moving operand's N columns through the array (MATMUL), so its time depends on M
    and N and not on the contraction depth K. The FLOP-priced surfaces (``TensorCalibrationSurface``)
    were fitted on full 128 x 128 x 512 tiles and charge an instruction 2*M*K*N, which makes a K = 32
    instruction four times too cheap. Controls ``bench/controls_pe`` (same product, only K, M, N
    varied) measure the steady-state issue interval per instruction; one linear law per dtype:
    ``interval = base_ns + ns_per_n * N + ns_per_m * M + [K < 128] * reload_ns``, plus a once-per-kernel
    ``startup_ns``. A full-row stationary's weight load is hidden behind the previous instruction's
    stream; a partial-row one (K < 128) is exposed. Shapes beyond one instruction (M, K > 128 or
    N > 512) are tiled into instructions.
    """

    points: dict[str, tuple[float, ...]]

    MAX_M, MAX_K, MAX_N = 128, 128, 512

    @classmethod
    def from_csv(cls, path: str | Path) -> "TensorInstructionLaw":
        points: dict[str, tuple[float, ...]] = {}
        with Path(path).open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                points[TensorCalibrationSurface._normalize_dtype(row["dtype"])] = (
                    float(row["startup_ns"]), float(row["base_ns"]), float(row["ns_per_n"]),
                    float(row["ns_per_m"]), float(row.get("reload_ns") or 0.0))
        return cls(points)

    def _point(self, dtype: str) -> tuple[float, ...] | None:
        return self.points.get(TensorCalibrationSurface._normalize_dtype(dtype)) or self.points.get("float32")

    def startup_ns(self, dtype: str) -> float:
        point = self._point(dtype)
        return point[0] if point else 0.0

    def instruction_ns(self, dtype: str, m: int, k: int, n: int) -> float | None:
        point = self._point(dtype)
        if point is None or m <= 0 or k <= 0 or n <= 0:
            return None
        base, per_n, per_m = point[1], point[2], point[3]
        reload_ns = point[4] if len(point) > 4 else 0.0
        total = 0.0
        for k0 in range(0, k, self.MAX_K):
            exposed = reload_ns if min(self.MAX_K, k - k0) < self.MAX_K else 0.0
            for m0 in range(0, m, self.MAX_M):
                mt = min(self.MAX_M, m - m0)
                for n0 in range(0, n, self.MAX_N):
                    total += base + per_n * min(self.MAX_N, n - n0) + per_m * mt + exposed
        return total


@dataclass
class StructuredControlCalibration:
    """Interpolated points keyed by reusable structural grammar families."""

    points: dict[tuple[str, str, str], list[tuple[int, float, int, float]]]
    micro_dags: dict[tuple[str, str, int], dict[str, Any]] = field(
        default_factory=dict
    )
    opcode_timing_points: dict[
        tuple[str, str, str], list[tuple[int, float]]
    ] = field(default_factory=dict)

    @classmethod
    def from_csv(cls, path: str | Path) -> StructuredControlCalibration:
        points = {}
        micro_dags = {}
        opcode_timing_samples: dict[
            tuple[str, str, str, int, str], list[float]
        ] = {}
        with Path(path).open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                key = (
                    row.get("calibration_key") or row["family"],
                    row["engine"],
                    ComputeCalibration._norm_dtype(row["dtype"]),
                )
                points.setdefault(key, []).append(
                    (
                        int(row["free_dim"]),
                        float(row["effective_count"]),
                        int(row["instruction_count"]),
                        float(row["fixed_ns"]),
                    )
                )
                raw_dag = row.get("micro_dag_json") or ""
                if raw_dag:
                    dag_key = (key[0], key[2], int(row["free_dim"]))
                    dag = json.loads(raw_dag)
                    previous = micro_dags.get(dag_key)
                    if previous is not None and previous != dag:
                        raise ValueError(
                            f"Conflicting micro-DAG rows for {dag_key}"
                        )
                    micro_dags[dag_key] = dag
                    for node in dag.get("nodes", []):
                        if node.get("is_sync"):
                            continue
                        engine = _canonical_engine(
                            str(node.get("engine") or ""), "compute"
                        )
                        opcode = str(node.get("opcode_family") or "")
                        timing = node.get("timing") or {}
                        duration = float(
                            timing.get("completion_latency_ns") or 0.0
                        )
                        if duration <= 0:
                            continue
                        sample_key = (
                            engine,
                            key[2],
                            opcode,
                            int(row["free_dim"]),
                            str(row.get("case") or ""),
                        )
                        opcode_timing_samples.setdefault(sample_key, []).append(
                            duration
                        )
        opcode_timing_points: dict[
            tuple[str, str, str], list[tuple[int, float]]
        ] = {}
        for (engine, dtype, opcode, free_dim, _case), values in (
            opcode_timing_samples.items()
        ):
            opcode_timing_points.setdefault(
                (engine, dtype, opcode), []
            ).append((free_dim, statistics.median(values)))
        return cls(points, micro_dags, opcode_timing_points)

    def micro_dag_lookup(
        self, region_ir: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, str]:
        """Return an exact control-backed compiler Flow micro-DAG."""
        from triton_viz.tools.nki_region_ir import structural_calibration_key

        key = (
            structural_calibration_key(region_ir),
            ComputeCalibration._norm_dtype(region_ir.get("dtype")),
            int(region_ir.get("logical_free_dim") or region_ir.get("free_dim") or 1),
        )
        dag = self.micro_dags.get(key)
        return (dag, "exact") if dag is not None else (None, "missing")

    def opcode_timing_lookup(
        self,
        engine: str,
        dtype: str,
        opcode_family: str,
        free_dim: int,
    ) -> tuple[float, str]:
        """Interpolate operator-name-free instruction timing controls."""
        rows = sorted(
            self.opcode_timing_points.get(
                (
                    _canonical_engine(engine, "compute"),
                    ComputeCalibration._norm_dtype(dtype),
                    opcode_family,
                ),
                [],
            )
        )
        if not rows:
            return 0.0, "ood"
        exact = [value for size, value in rows if size == free_dim]
        if exact:
            return statistics.median(exact), "exact"
        sizes = [size for size, _value in rows]
        if free_dim < min(sizes) or free_dim > max(sizes):
            return 0.0, "ood"
        lower = max(row for row in rows if row[0] <= free_dim)
        upper = min(row for row in rows if row[0] >= free_dim)
        if lower[0] == upper[0]:
            return lower[1], "interpolated"
        weight = (math.log2(free_dim) - math.log2(lower[0])) / (
            math.log2(upper[0]) - math.log2(lower[0])
        )
        return lower[1] + weight * (upper[1] - lower[1]), "interpolated"

    def predict_points(
        self, region_ir: dict[str, Any]
    ) -> dict[str, tuple[float, int, float]]:
        """Return effective count, real ISA count and fixed time per engine."""
        return self.predict_points_with_provenance(region_ir)[0]

    def predict_points_with_provenance(
        self, region_ir: dict[str, Any]
    ) -> tuple[dict[str, tuple[float, int, float]], dict[str, str]]:
        """Return Level-A points and the source-only lookup path per engine."""
        from triton_viz.tools.nki_region_ir import (
            structural_calibration_key,
            structural_family,
        )

        family = structural_family(region_ir)
        calibration_key, dtype = (
            structural_calibration_key(region_ir),
            ComputeCalibration._norm_dtype(region_ir.get("dtype")),
        )
        # Masked/padded kernels lower for the active logical width. Using the
        # allocation tile width here spuriously extrapolates (Tilebench uses a
        # 16K backing tile even for 128 active columns).
        free = int(region_ir.get("logical_free_dim") or region_ir.get("free_dim") or 1)
        result = {}
        provenance = {}
        for engine, streams in (("vector", 2), ("scalar", 1)):
            rows = sorted(self.points.get((calibration_key, engine, dtype), []))
            match = "exact_key" if rows else "none"
            # Backward compatibility for pre-key tables. New tables must not
            # silently cross primitive-specific instruction-selection paths.
            if not rows and (family, engine, dtype) in self.points:
                rows = sorted(self.points[(family, engine, dtype)])
                match = "legacy_family"
            if not rows and "__" in family:
                rows = sorted(
                    self.points.get((family.split("__", 1)[0], engine, dtype), [])
                )
                if rows:
                    match = "family_prefix"
            if not rows:
                continue
            exact = [row for row in rows if row[0] == free]
            if exact:
                row = (
                    free,
                    statistics.median(value[1] for value in exact),
                    round(statistics.median(value[2] for value in exact)),
                    statistics.median(value[3] for value in exact),
                )
            else:
                lower = max(
                    (row for row in rows if row[0] <= free),
                    default=rows[0],
                    key=lambda x: x[0],
                )
                upper = min(
                    (row for row in rows if row[0] >= free),
                    default=rows[-1],
                    key=lambda x: x[0],
                )
                if lower[0] == upper[0]:
                    row = lower
                else:
                    weight = (math.log2(free) - math.log2(lower[0])) / (
                        math.log2(upper[0]) - math.log2(lower[0])
                    )
                    row = (
                        free,
                        lower[1] + weight * (upper[1] - lower[1]),
                        round(lower[2] + weight * (upper[2] - lower[2])),
                        lower[3] + weight * (upper[3] - lower[3]),
                    )
            if row[1] > 0:
                result[engine] = (row[1], row[2], row[3])
                provenance[engine] = match
        return result, provenance

    def predict(self, region_ir: dict[str, Any]) -> dict[str, tuple[float, int, float]]:
        points = self.predict_points(region_ir)
        return {
            engine: (value[0], 2 if engine == "vector" else 1, value[2])
            for engine, value in points.items()
        }


@dataclass
class StridedDmaCalibration:
    """Access-geometry DMA busy-time calibration for strided stores.

    This is an engine occupancy surface only.  The independent strided controls
    also measured an end-to-end packet/queue tail, but that column was a
    per-geometry completion floor and has been removed along with every other
    per-structure completion table.
    """

    points: dict[tuple[str, int, int], list[tuple[int, float, float]]]

    @classmethod
    def from_csv(cls, path: str | Path) -> StridedDmaCalibration:
        points: dict[tuple[str, int, int], list[tuple[int, float, float]]] = {}
        with Path(path).open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                key = (
                    ComputeCalibration._norm_dtype(row["dtype"]),
                    int(row["stride_items"]),
                    int(row["partition_count"]),
                )
                points.setdefault(key, []).append(
                    (
                        int(row["free_dim"]),
                        float(
                            row.get("dynamic_dma_active_ns")
                            or row["dma_active_ns"]
                        ),
                        float(row.get("static_dma_active_ns") or 0.0),
                    )
                )
        return cls(points)

    def predict_components(
        self, events: Iterable[dict[str, Any]]
    ) -> tuple[float, float] | None:
        patterns = [
            (event, pattern)
            for event in events
            if (pattern := AccessPattern.from_event(event)) is not None
            and pattern.dst_space == "hbm"
            and pattern.layout_family == "strided_positive"
        ]
        if not patterns:
            return None
        first, first_pattern = patterns[0]
        stride = first_pattern.free_stride_items
        partitions = first_pattern.partition_count
        active = sum(pattern.active_access_count for _, pattern in patterns)
        free = max(1, active // max(1, partitions * len(patterns)))
        dtype = ComputeCalibration._norm_dtype(
            first.get("dtype")
            or first.get("src_dtype")
            or first.get("dst_dtype")
            or ("float16" if first_pattern.item_bytes == 2 else "float32")
        )
        rows = self.points.get((dtype, stride, partitions), [])
        if not rows:
            return None
        rows = sorted(rows)
        lower = max((row for row in rows if row[0] <= free), default=rows[0])
        upper = min((row for row in rows if row[0] >= free), default=rows[-1])
        if lower[0] == upper[0]:
            return lower[1], lower[2]
        # Interpolate only between independent control sizes. Values outside
        # their measured range remain clamped and are reported as OOD by the
        # experiment layer rather than silently extrapolated.
        weight = (free - lower[0]) / (upper[0] - lower[0])
        return (
            lower[1] + weight * (upper[1] - lower[1]),
            lower[2] + weight * (upper[2] - lower[2]),
        )

    @staticmethod
    def matched_indices(events: Iterable[dict[str, Any]]) -> list[int]:
        """Return only the transfer events described by this calibration."""
        return [
            index
            for index, event in enumerate(events)
            if (pattern := AccessPattern.from_event(event)) is not None
            and pattern.dst_space == "hbm"
            and pattern.layout_family == "strided_positive"
        ]


def _canonical_engine(raw_engine: str, op: str) -> str:
    """Map a coarse dumper engine tag onto a concrete NeuronCore engine.

    The trace dumper only knows rough categories (e.g. it cannot yet tell a
    PSUM/SBUF copy apart into VectorE vs ScalarE).  We resolve those here so the
    scheduler works on a fixed set of resources.  This mapping is the natural
    place to refine once the dumper carries the explicit ``engine`` argument.
    """
    if raw_engine == ENGINE_DMA or raw_engine == "dma_or_vector_load":
        return ENGINE_DMA
    if raw_engine == "dma_or_vector_store":
        return ENGINE_DMA
    if raw_engine == ENGINE_TENSOR:
        return ENGINE_TENSOR
    if raw_engine == ENGINE_STATIC_DMA:
        return ENGINE_STATIC_DMA
    if raw_engine == ENGINE_SCALAR:
        return ENGINE_SCALAR
    if raw_engine == ENGINE_VECTOR:
        return ENGINE_VECTOR
    if raw_engine == ENGINE_GPSIMD:
        return ENGINE_GPSIMD
    if raw_engine == ENGINE_SYNC:
        return ENGINE_SYNC
    if raw_engine == "tensor_or_vector_copy":
        # On-chip PSUM<->SBUF copies are commonly issued on a vector/scalar
        # engine; assume VectorE until the dumper records the real engine token.
        return ENGINE_VECTOR
    if raw_engine == "vector_or_scalar":
        return ENGINE_VECTOR
    # Fallback: keep unknown work off the compute engines so it does not hide
    # real bottlenecks.  Treat it as a generic vector op.
    return ENGINE_VECTOR


DMA_STREAM_SATURATION_BYTES = 4096
DMA_SATURATED_LOOKUP_BYTES = 32768


def _dma_contiguous_run_bytes(event: dict[str, Any]) -> int | None:
    """Median contiguous byte run on the HBM side of a DMA, or ``None``.

    Reads come from ``src_ranges`` and writes go to ``dst_ranges``; each range
    is a ``[start, end)`` byte interval recorded by the tracer. Rows that are
    adjacent in HBM appear as one merged range, so the run of a full-row block
    load can exceed the bytes a single partition moves; callers take the min
    with the free size.
    """
    src = str(event.get("mem_src", "")).lower()
    dst = str(event.get("mem_dst", "")).lower()
    if src == "hbm":
        ranges = event.get("src_ranges")
    elif dst == "hbm":
        ranges = event.get("dst_ranges")
    else:
        return None
    runs = []
    for item in ranges or ():
        try:
            lo, hi = int(item[0]), int(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        if hi > lo:
            runs.append(hi - lo)
    if not runs:
        return None
    runs.sort()
    return runs[len(runs) // 2]


def _free_dim(event: dict[str, Any]) -> int | None:
    """Return the free size (elements per partition) for a compute op.

    VectorE/ScalarE are partition-parallel: the partition axis (axis 0) maps
    onto lanes, so op latency tracks the elements each partition streams, i.e.
    the product of all free axes of the *processed* tile, not the total element
    count. A ``(P, A, B)`` tile streams ``A * B`` elements per partition; taking
    only the last axis would price it ``A`` times too cheap (e.g. casts over
    coalesced ``(128, 8, 1024)`` operands or 4-D rotate-half views). For a
    reduction the input free size is larger than the output's (which collapses
    to 1), and the engine still streams the whole input, so we take the max
    across the input and output shapes. A 1-D shape is treated as its own free
    size. Returns ``None`` when no shape is available so the caller can fall
    back to the element-count model.
    """
    free = None
    for key in ("output_shape", "input_shape", "other_shape"):
        shape = event.get(key)
        if isinstance(shape, (list, tuple)) and len(shape) >= 1:
            try:
                dims = [int(dim) for dim in shape]
            except (TypeError, ValueError):
                continue
            candidate = math.prod(dims[1:]) if len(dims) >= 2 else dims[0]
            free = candidate if free is None else max(free, candidate)
    return free


def _compute_value_dtype(event: dict[str, Any]) -> str:
    """Return the calibrated value dtype, not a predicate result dtype."""
    output = str(event.get("output_dtype") or "").lower()
    if output not in {"bool", "boolean"}:
        return str(event.get("output_dtype") or "float32")
    value_dtypes = [
        str(value)
        for value in event.get("input_dtypes") or ()
        if value and str(value).lower() not in {"bool", "boolean"}
    ]
    region_dtype = (event.get("region_ir") or {}).get("dtype")
    if region_dtype and str(region_dtype).lower() not in {"bool", "boolean"}:
        value_dtypes.append(str(region_dtype))
    return Counter(value_dtypes).most_common(1)[0][0] if value_dtypes else output


def _dot_geometry(event: dict[str, Any]) -> tuple[int, int, int] | None:
    """(M, K, N) of a normalized Dot event: input (M, K) @ other (K, N)."""
    a, b = event.get("input_shape") or (), event.get("other_shape") or ()
    if len(a) != 2 or len(b) != 2 or int(a[1]) != int(b[0]):
        return None
    return int(a[0]), int(a[1]), int(b[1])


def _tensor_dtype(event: dict[str, Any]) -> str:
    """Return the TensorE operand dtype, ignoring bool predicates/accumulators."""
    input_dtypes = [
        str(value)
        for value in event.get("input_dtypes") or ()
        if value and str(value).lower() not in {"bool", "boolean"}
    ]
    return input_dtypes[0] if input_dtypes else _compute_value_dtype(event)


def _input_stream_count(event):
    """Number of distinct tile inputs a compute op streams (1-input vs 2-input).

    Two-input ops (tensor_tensor add/mul/sub/div/max between two tiles) read two
    SBUF streams and cost more per element than one-input ops (activations,
    scalar ops, reductions). Inferred from recorded input pointers, then the op
    name, defaulting to 1.
    """
    explicit = event.get("input_stream_count")
    if explicit is not None:
        return max(1, int(explicit))
    api = str(event.get("api_op") or "")
    if api in {"exp", "rsqrt", "sqrt", "log", "sin", "cos", "tanh", "sigmoid", "relu"}:
        # Activation instructions stream one data tile; bias/scale are scalar
        # or per-partition epilogue operands, not extra input streams. This
        # matches the Level-B scalar one-input calibration.
        return 1
    ptrs = event.get("input_ptrs")
    if isinstance(ptrs, (list, tuple)) and len(ptrs) >= 1:
        return 2 if len(ptrs) >= 2 else 1
    if event.get("op") == "binary":
        return 2
    two_input = {
        "add",
        "subtract",
        "multiply",
        "divide",
        "maximum",
        "minimum",
        "greater",
        "less",
        "tensor_tensor",
        "where",
    }
    return 2 if api in two_input else 1


@dataclass
class WholeProgramRoutingCalibration:
    """Control-only engine occupancy indexed by source-visible program grammar."""

    samples: list[dict[str, Any]]

    @classmethod
    def from_control_root(cls, root: str | Path) -> WholeProgramRoutingCalibration:
        from triton_viz.tools.nki_evaluate_whole_program_regime import _source_sample

        root = Path(root)
        with (root / "operator_results.csv").open(encoding="utf-8", newline="") as file:
            return cls([
                _source_sample(root, row)
                for row in csv.DictReader(file)
                if row.get("status") == "ok"
            ])

    def predict_with_provenance(
        self, events: list[dict[str, Any]], dtype: str
    ) -> tuple[dict[str, float], str] | None:
        """Interpolate control engine occupancy at the program's transfer size.

        A partition-parallel engine's active time is ``startup + free_width *
        per_element``, so within one program grammar it grows linearly with the
        program's total source transfer bytes.  Reading the control surface by
        nearest neighbour therefore errs systematically between control points;
        interpolating between the two bracketing controls is the physically
        correct read of the same measurements.  On the 288 independent
        whole-program controls under leave-one-entire-free-dimension-out CV this
        moves Vector WAPE from 5.65% to 3.85% and Scalar from 5.96% to 3.70%.
        Outside the measured range the value is clamped and reported.
        """
        from triton_viz.tools.nki_evaluate_whole_program_regime import (
            source_descriptor_from_events,
        )

        descriptor = source_descriptor_from_events(events, dtype)
        candidates = [
            sample
            for sample in self.samples
            if sample["key"] == descriptor["key"]
        ]
        if not candidates:
            return None
        engines = (ENGINE_VECTOR, ENGINE_SCALAR, ENGINE_GPSIMD)
        by_distance: dict[float, list[dict[str, Any]]] = {}
        for sample in candidates:
            by_distance.setdefault(float(sample["distance_feature"]), []).append(sample)
        knots = sorted(by_distance)

        def at(knot: float) -> dict[str, float]:
            group = by_distance[knot]
            # Explorer active-time labels are microseconds; the simulator uses ns.
            return {
                engine: statistics.mean(item["actual"][engine] for item in group) * 1000.0
                for engine in engines
            }

        distance = float(descriptor["distance_feature"])
        if distance in by_distance:
            return at(distance), "exact"
        if len(knots) < 2 or distance < knots[0] or distance > knots[-1]:
            nearest = min(knots, key=lambda knot: abs(knot - distance))
            return at(nearest), "clamped"
        upper = next(knot for knot in knots if knot > distance)
        lower = max(knot for knot in knots if knot < distance)
        weight = (distance - lower) / (upper - lower)
        low, high = at(lower), at(upper)
        return (
            {engine: low[engine] + weight * (high[engine] - low[engine]) for engine in engines},
            "interpolated",
        )

    def predict_ns(self, events: list[dict[str, Any]], dtype: str) -> dict[str, float] | None:
        prediction = self.predict_with_provenance(events, dtype)
        return None if prediction is None else prediction[0]


@dataclass
class CostModel:
    """Placeholder analytical cost model (units: nanoseconds).

    All constants are illustrative defaults, NOT measured hardware values.
    Replace them with neuron-profile-calibrated numbers per NeuronCore version.
    """

    # Inf2/NCv2 DMA geometry. Each engine owns up to eight SBUF partitions.
    # The analytical fallback is startup + bytes / active-engine peak; a
    # measured surface can replace it for small/partial tiles.
    dma_startup_ns: float = 300.0
    dma_engine_bytes_per_ns: float = 17.0
    dma_max_engines: int = 16
    dma_bytes_per_ns: float | None = None  # legacy explicit aggregate override
    # Opt-in: look DMA surfaces up by the contiguous bytes each partition moves
    # in one run, as seen inside a stream of transfers (see
    # ``_dma_lookup_free_bytes``). Off by default because the surfaces and the
    # published benchmark error were fit without it.
    dma_contiguity_aware: bool = False
    dma_calibration: DmaCalibrationSurface | None = None
    dma_write_calibration: DmaCalibrationSurface | None = None
    dma_transpose_calibration: DmaCalibrationSurface | None = None
    structural_static_dma: StructuralStaticDmaCalibration | None = None
    # Level-B per-instruction compute cost (VectorE/ScalarE). When present,
    # NkiCompute/binary/reduce events cost their lowered-instruction count times
    # the measured single-instruction cost instead of the hardcoded VectorE fit.
    compute_calibration: ComputeCalibration | None = None
    # Opt-in consumer-visible instruction latency (dependent chains); None keeps the
    # initiation-interval-only behaviour byte-identical.
    instruction_latency: InstructionLatencyCalibration | None = None
    # Opt-in structure-aware issue order (bounded renaming + dependency-ordered loads); see
    # triton_viz.tools.nki_structure_schedule. Typed loosely to avoid an import cycle.
    structure_schedule: Any = None
    # Opt-in fixed DMA result latency (ns) beyond bandwidth occupancy; None keeps the default model.
    dma_result_latency_ns: float | None = None
    # "constant" or "log2_partitions" (latency scales with log2 of the partitions a transfer spans).
    dma_result_latency_law: str = "constant"
    # Opt-in: split each surface-priced transfer into bandwidth occupancy and a fixed result latency
    # fitted from the surface's own points (DmaCalibrationSurface.split).
    dma_split_fixed_cost: bool = False
    # Opt-in pipelined DMA queue: a transfer that starts while its queue is still busy hides the surface's
    # fixed cost behind the previous transfer's stream and occupies only bytes / asymptotic bandwidth;
    # a transfer that starts on an idle queue pays the fixed cost as occupancy.
    dma_pipelined_fixed_cost: bool = False
    # Opt-in multi-queue DMA: ``dma_queues`` independent FIFO queues; a transfer starting with n transfers in
    # flight (itself included) streams at min(single-transfer surface bandwidth, dma_aggregate_gbps / n) and
    # pays the surface's fixed cost only when it starts alone. 0 keeps the resource-slot model.
    dma_queues: int = 0
    dma_aggregate_gbps: float = 300.0
    structured_control_lowering: StructuredControlCalibration | None = None
    whole_program_routing: WholeProgramRoutingCalibration | None = None
    strided_dma_calibration: StridedDmaCalibration | None = None
    tensor_calibration: TensorCalibrationSurface | None = None
    tensor_dot_count_calibration: TensorDotCountCalibration | None = None
    # Opt-in per-instruction TensorE law (geometry, not FLOPs); None keeps the FLOP surfaces.
    tensor_instruction_law: TensorInstructionLaw | None = None
    # Opt-in: compute operands' hazards use their exact byte runs (a slice t[:, j, :] touches one run per
    # partition) instead of the bounding interval, so a consumer waits only for the transfer it reads.
    exact_tile_ranges: bool = False
    attention_pipeline_calibration: AttentionPipelineCalibration | None = None
    dma_elapsed_calibration: DmaElapsedCalibration | None = None
    onchip_transfer_calibration: OnChipTransferCalibration | None = None
    global_completion_calibration: GlobalCompletionCalibration | None = None
    strict_calibration: bool = False

    # On-chip copy (VectorE/ScalarE moving PSUM<->SBUF): cheaper per byte.
    onchip_startup_ns: float = 100.0
    onchip_bytes_per_ns: float = 1000.0

    # TensorE: modeled via a throughput (initiation-interval) style estimate.
    # NOT a single-instruction end-to-end latency, because TensorE is deeply
    # pipelined.  cost ~= startup + flops / peak_flops_per_ns.
    tensor_startup_ns: float = 200.0
    tensor_flops_per_ns: float = 90000.0  # placeholder peak MAC throughput

    # VectorE processes all SBUF partitions in parallel across its lanes, so its
    # latency scales with the *free* dimension (elements per partition), NOT the
    # total element count. The primary model is therefore
    #   cost = vector_free_startup_ns + free_dim / vector_free_per_ns
    # with defaults fit to the Inf2 tensor_add sweep (two-input fp32
    # ``tensor_tensor``): active time = 0.954 us + 2.585 ns * free_dim across
    # p = 1..128, f = 32..1024 (max abs residual 0.32 us). Because it is
    # partition-parallel this fit is flat in p, which is why the old
    # element-count model was ~30x low at small p.
    vector_free_startup_ns: float = 953.7
    # Fit slope is 2.5854 ns per free-element, i.e. ~0.387 free-elems/ns.
    vector_free_per_ns: float = 1.0 / 2.5854  # ~= 0.387 free-elems/ns
    # Legacy element-count fallback, used only when an event carries no output
    # shape to derive the free dimension from (older traces / synthetic events).
    vector_startup_ns: float = 100.0
    vector_elements_per_ns: float = 128.0 * 1.12 / 2.0

    # Optional end-to-end launch/sequencer overhead. This is deliberately
    # separate from engine busy times and defaults to zero for backward
    # compatibility. Inf2 workload validation currently observes an ~8 us
    # residual between the source-operation timeline and NC p50 latency.
    kernel_overhead_ns: float = 0.0

    # Cross-engine handoff latency. When a consumer op depends on a value
    # produced on a *different* engine, the NeuronCore must synchronize the two
    # engines (semaphore set/wait) before the consumer can start. This models
    # that fixed per-dependency cost; it is only charged when the dependency
    # crosses an engine boundary, never within the same serial engine (that is
    # already captured by program-order queueing). Defaults to zero so existing
    # callers and tests are unaffected until a value is calibrated.
    cross_engine_sync_ns: float = 0.0

    # Number of independent DMA queues. NeuronCore-v2 issues DMA on several
    # queues in parallel, so *independent* transfers (no data hazard between
    # them, e.g. loading two distinct operands) overlap instead of serializing.
    # Modeling the DMA engine as this many parallel slots removes a systematic
    # over-count of DMA busy time for multi-input kernels. Data-dependent
    # transfers still serialize through the RAW/WAR/WAW hazard logic. Compute
    # engines stay single-slot. Default 1 preserves prior behavior/tests.
    dma_queue_count: int = 1
    # NCv2 exposes 16 DMA engines. A transfer consumes min(partitions, 16)
    # tokens for its lifetime; only disjoint remaining tokens may issue other
    # transfers concurrently. ``dma_queue_count`` remains a compatibility
    # fallback when this is set to zero.
    dma_resource_count: int = 16
    # Aggregate HBM cap shared by all DMA engines. The analytical path applies
    # it to per-transfer bandwidth; calibrated aggregate slopes already include
    # the cap and are therefore not rescaled.
    hbm_bandwidth_bytes_per_ns: float = 272.0

    def _dma_surface(
        self, event: dict[str, Any]
    ) -> DmaCalibrationSurface | None:
        memories = {
            str(event.get("mem_src", "")).lower(),
            str(event.get("mem_dst", "")).lower(),
        }
        if memories and "" not in memories and "hbm" not in memories:
            # On-chip PSUM<->SBUF copies are priced by the vector/static-DMA
            # path, not by the HBM bandwidth surfaces.
            return None
        if event.get("dma_pattern") == "transpose":
            return self.dma_transpose_calibration
        if (
            str(event.get("mem_src", "")).lower() == "sbuf"
            and str(event.get("mem_dst", "")).lower() == "hbm"
        ):
            return self.dma_write_calibration
        return self.dma_calibration

    def _dma_lookup_free_bytes(self, event: dict[str, Any]) -> int:
        """Free bytes per partition used to look up the DMA bandwidth surface.

        By default this is the transfer's ``free_bytes_per_partition``. With
        ``dma_contiguity_aware`` two effects measured on Inf2 are applied:

        * A strided view (e.g. a column slice of several rows per partition)
          moves several separate runs per partition, so its granularity is the
          median contiguous run on the HBM side, not the free size.
        * Inside a stream of DMAs, runs of at least
          ``DMA_STREAM_SATURATION_BYTES`` contiguous bytes per partition already
          reach the bandwidth ceiling: a 4 MiB copy issued as 16 DMAs of 4 KiB
          per partition is as fast as one 32 KiB-per-partition DMA, whereas the
          single-transfer surface prices it about 17% slower. Such runs are
          looked up at ``DMA_SATURATED_LOOKUP_BYTES``.
        """
        free_bytes = int(event.get("free_bytes_per_partition") or 0)
        if not self.dma_contiguity_aware or free_bytes <= 0:
            return free_bytes
        run = _dma_contiguous_run_bytes(event)
        if run is None:
            return free_bytes
        effective = min(free_bytes, run)
        if effective >= DMA_STREAM_SATURATION_BYTES:
            effective = max(free_bytes, DMA_SATURATED_LOOKUP_BYTES)
        return effective

    def dma_lookup(self, event: dict[str, Any]) -> dict[str, Any]:
        """Describe the selected DMA calibration path without changing cost."""
        partitions = int(event.get("partition_count") or 0)
        free_bytes = self._dma_lookup_free_bytes(event)
        surface = self._dma_surface(event)
        if partitions > 0 and free_bytes > 0 and surface is not None:
            point = surface.lookup(partitions, free_bytes)
            return {
                "path": "surface",
                "match": point.match,
                "bandwidth_gbps": point.bandwidth_gbps,
                "log_distance": point.log_distance,
                "lookup_partitions": point.lookup_partitions,
                "lookup_free_bytes": point.lookup_free_bytes,
            }
        return {"path": "analytical_fallback", "match": "fallback"}

    def _dma_cost_ns(self, event: dict[str, Any], nbytes: int) -> float:
        partitions = int(event.get("partition_count") or 0)
        free_bytes = self._dma_lookup_free_bytes(event)
        surface = self._dma_surface(event)
        if partitions > 0 and free_bytes > 0 and surface is not None:
            if self.dma_split_fixed_cost:
                split = surface.split(partitions)
                if split is not None:
                    # queue occupancy is the bandwidth term; the fixed cost becomes result latency
                    return nbytes / split[1]
            bandwidth = surface.lookup(partitions, free_bytes).bandwidth_gbps
            return nbytes / bandwidth
        if self.dma_bytes_per_ns is not None:
            bandwidth = self.dma_bytes_per_ns
        elif partitions > 0:
            # Inf2 parquet shows contiguous partitions striped by engine index:
            # p<=16 activates p engines; p=128 gives eight partitions/engine.
            engines = min(self.dma_max_engines, partitions)
            bandwidth = min(
                engines * self.dma_engine_bytes_per_ns,
                self.hbm_bandwidth_bytes_per_ns,
            )
        else:
            bandwidth = min(
                self.dma_max_engines * self.dma_engine_bytes_per_ns,
                self.hbm_bandwidth_bytes_per_ns,
            )
        return self.dma_startup_ns + nbytes / bandwidth

    def dma_split_parts(self, event: dict[str, Any]) -> tuple[float, float] | None:
        """(fixed_ns, bandwidth occupancy ns) of a surface-priced transfer, or None."""
        partitions = int(event.get("partition_count") or 0)
        surface = self._dma_surface(event)
        if partitions <= 0 or surface is None or self._dma_lookup_free_bytes(event) <= 0:
            return None
        split = surface.split(partitions)
        if split is None:
            return None
        return split[0], int(event.get("bytes") or 0) / split[1]

    def result_ready_ns(self, event: dict[str, Any], engine: str, duration: float) -> float:
        """Time from issue until the op's result is visible to a consumer.

        Compute: ``max(duration, latency)`` under ``instruction_latency``. DMA: ``duration +
        dma_result_latency_ns`` (opt-in): the queue is released after the transfer's bandwidth
        occupancy, while its data lands a fixed latency later, which only dependent consumers see.
        """
        if event.get("op") in ("transfer", "load", "store"):
            fixed = 0.0
            if self.dma_split_fixed_cost:
                surface = self._dma_surface(event)
                parts = int(event.get("partition_count") or 0)
                split = surface.split(parts) if surface is not None and parts > 0 else None
                fixed = split[0] if split is not None else 0.0
            latency = (self.dma_result_latency_ns or 0.0)
            if latency and self.dma_result_latency_law == "log2_partitions":
                # Completion waits for every queue a transfer is spread over: 0 for one partition,
                # the full latency for 128 partitions.
                parts = max(1, min(128, int(event.get("partition_count") or 1)))
                latency *= math.log2(parts) / 7.0
            return duration + fixed + latency
        if self.instruction_latency is None or event.get("op") not in ("binary", "compute", "reduce_sum"):
            return duration
        free_dim = _free_dim(event)
        if free_dim is None:
            return duration
        latency = self.instruction_latency.latency_ns(
            engine, _compute_value_dtype(event), _input_stream_count(event), free_dim)
        return duration if latency is None else max(duration, latency)

    def cost_ns(self, event: dict[str, Any]) -> float:
        """Return the estimated duration in ns for one trace event."""
        op = event.get("op")
        engine = _canonical_engine(event.get("engine", ""), op or "")
        if op == "dot":
            flops = event.get("flops") or 0
            source_dot_count = int(event.get("tensor_source_dot_count") or 0)
            if self.tensor_dot_count_calibration is not None and source_dot_count > 0:
                total_ns, match = self.tensor_dot_count_calibration.active_ns(
                    _tensor_dtype(event), source_dot_count,
                    int(event.get("tensor_source_lhs_tile_count") or 0),
                    int(event.get("tensor_source_rhs_tile_count") or 0),
                    int(event.get("tensor_source_output_tile_count") or 0),
                )
                event["tensor_dot_count_calibration_match"] = match
                return total_ns / source_dot_count
            if self.tensor_calibration is not None:
                dtype = _tensor_dtype(event)
                flops_per_ns = self.tensor_calibration.flops_per_ns(
                    dtype, strict=self.strict_calibration
                )
                return flops / flops_per_ns
            return self.tensor_startup_ns + flops / self.tensor_flops_per_ns
        if op == "tensor_transpose":
            # Transpose FLOPs are an accounting proxy, not matmul arithmetic;
            # do not feed them through the independently fitted Dot surface.
            return (event.get("flops") or 0) / self.tensor_flops_per_ns
        if op in ("binary", "compute", "reduce_sum"):
            free_dim = _free_dim(event)
            # Level-B calibrated path: cost one lowered instruction from the
            # measured per-instruction surface, times the Level-A expansion
            # factor (how many hardware instructions this source op lowers to on
            # this engine). Default expansion is 1 (one source op -> one
            # instruction) until a richer expansion table is supplied.
            if free_dim is not None and self.compute_calibration is not None:
                dtype = _compute_value_dtype(event)
                streams = _input_stream_count(event)
                per_instr = self.compute_calibration.instruction_ns(
                    engine,
                    dtype,
                    streams,
                    free_dim,
                    strict_dtype=self.strict_calibration,
                )
                if per_instr is None and self.strict_calibration:
                    raise ValueError(
                        "Missing exact compute calibration for "
                        f"engine={engine}, dtype={dtype}, streams={streams}"
                    )
                if per_instr is not None:
                    expansion = float(event.get("lowering_expansion") or 1.0)
                    fixed_ns = float(event.get("lowering_fixed_ns") or 0.0)
                    return per_instr * max(1.0, expansion) + fixed_ns
            if free_dim is not None:
                # Compute events may run on VectorE or ScalarE. For now we apply
                # the same fit (since activation latency bounds are likely
                # similar on the scalar unit).
                # TODO: add scalar-specific calibration surface.
                return self.vector_free_startup_ns + free_dim / self.vector_free_per_ns
            # Fallback: no shape info to derive the free dimension.
            elements = int(event.get("elements", 0))
            return self.vector_startup_ns + elements / self.vector_elements_per_ns
        if op in ("transfer", "load", "store"):
            nbytes = int(event.get("bytes", 0))
            if engine == ENGINE_DMA:
                return self._dma_cost_ns(event, nbytes)
            if self.onchip_transfer_calibration is not None:
                item_bytes = int(event.get("item_bytes") or 0)
                if item_bytes <= 0:
                    item_bytes = 2 if "16" in _compute_value_dtype(event) else 4
                free_bytes = int(event.get("free_bytes_per_partition") or 0)
                free_elements = (
                    free_bytes // item_bytes
                    if free_bytes > 0
                    else max(1, nbytes // max(1, int(event.get("partition_count") or 1)) // item_bytes)
                )
                latency_ns, match = self.onchip_transfer_calibration.latency_ns(
                    engine, _compute_value_dtype(event), free_elements
                )
                if match != "missing":
                    event["onchip_transfer_match"] = match
                    return latency_ns
            return self.onchip_startup_ns + nbytes / self.onchip_bytes_per_ns
        # grid markers and unknown ops take no engine time.
        return 0.0


@dataclass
class TimelineEntry:
    seq: int
    op: str
    engine: str
    start: float
    end: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "op": self.op,
            "engine": self.engine,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.end - self.start, 3),
        }


@dataclass
class SimulationResult:
    predicted_latency_ns: float
    timeline: dict[str, list[TimelineEntry]] = field(default_factory=dict)
    engine_busy_ns: dict[str, float] = field(default_factory=dict)
    components_ns: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "predicted_latency_ns": round(self.predicted_latency_ns, 3),
            "engine_busy_ns": {k: round(v, 3) for k, v in self.engine_busy_ns.items()},
            "components_ns": {k: round(v, 3) for k, v in self.components_ns.items()},
            "engine_utilization": {
                k: (
                    round(v / self.predicted_latency_ns, 4)
                    if self.predicted_latency_ns
                    else 0.0
                )
                for k, v in self.engine_busy_ns.items()
            },
            "timeline": {
                engine: [entry.as_dict() for entry in entries]
                for engine, entries in self.timeline.items()
            },
        }


_RANGE_INF = float("inf")


def _access_range(event: dict[str, Any], range_key: str) -> tuple[float, float]:
    """Return the ``[lo, hi)`` byte range for one side of a memory access.

    When the event carries no explicit range we fall back to the whole buffer
    ``[0, inf)``. Two whole-buffer accesses to the same storage therefore always
    overlap, which preserves the original base-pointer-only hazard behavior for
    events (and synthetic tests) that do not supply ranges.
    """
    rng = event.get(range_key)
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        lo, hi = float(rng[0]), float(rng[1])
        if hi > lo:
            return lo, hi
    return 0.0, _RANGE_INF


def _access_ranges(
    event: dict[str, Any], range_key: str
) -> list[tuple[float, float]]:
    """Return exact segments when present, otherwise one conservative span."""
    plural_key = f"{range_key}s"
    ranges = event.get(plural_key)
    if isinstance(ranges, (list, tuple)):
        valid = []
        for rng in ranges:
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                lo, hi = float(rng[0]), float(rng[1])
                if hi > lo:
                    valid.append((lo, hi))
        if valid:
            return valid
    return [_access_range(event, range_key)]


def _read_accesses(
    event: dict[str, Any],
) -> list[tuple[int, int | None, int, float, float]]:
    """Return ``(storage, version, ptr, lo, hi)`` read accesses.

    ``storage_key`` identifies the underlying allocation for hazard matching. It
    defaults to the base pointer, but an explicit ``storage`` field lets a view
    that records a distinct pointer still alias its parent allocation.
    """
    accesses: list[tuple[int, int | None, int, float, float]] = []
    src = event.get("src_ptr")
    src_storage = event.get("src_storage")
    if src is not None or src_storage is not None:
        key = int(src_storage if src_storage is not None else src)
        ptr = int(src if src is not None else key)
        version = event.get("src_version")
        for lo, hi in _access_ranges(event, "src_range"):
            accesses.append(
                (key, int(version) if version is not None else None, ptr, lo, hi)
            )
    storages = event.get("input_storages") or ()
    pointers = event.get("input_ptrs") or ()
    ranges = event.get("input_ranges") or ()
    versions = event.get("input_versions") or ()
    if storages:
        for index, storage in enumerate(storages):
            ptr = pointers[index] if index < len(pointers) else storage
            rng = ranges[index] if index < len(ranges) else None
            lo, hi = _access_range({"range": rng}, "range")
            version = versions[index] if index < len(versions) else None
            accesses.append((int(storage), int(version) if version is not None else None, int(ptr), lo, hi))
        return accesses
    for ptr in pointers:
        if ptr is not None:
            accesses.append((int(ptr), None, int(ptr), 0.0, _RANGE_INF))
    return accesses


def _write_accesses(
    event: dict[str, Any],
) -> list[tuple[int, int | None, int, float, float]]:
    """Return ``(storage, version, ptr, lo, hi)`` write accesses."""
    accesses: list[tuple[int, int | None, int, float, float]] = []
    dst = event.get("dst_ptr")
    dst_storage = event.get("dst_storage")
    if dst is not None or dst_storage is not None:
        key = int(dst_storage if dst_storage is not None else dst)
        ptr = int(dst if dst is not None else key)
        version = event.get("dst_version")
        for lo, hi in _access_ranges(event, "dst_range"):
            accesses.append(
                (key, int(version) if version is not None else None, ptr, lo, hi)
            )
    storages = event.get("output_storages") or ()
    pointers = event.get("output_ptrs") or ()
    ranges = event.get("output_ranges") or ()
    versions = event.get("output_versions") or ()
    if storages:
        for index, output_storage in enumerate(storages):
            ptr = pointers[index] if index < len(pointers) else output_storage
            rng = ranges[index] if index < len(ranges) else None
            lo, hi = _access_range({"range": rng}, "range")
            version = versions[index] if index < len(versions) else None
            accesses.append(
                (
                    int(output_storage),
                    int(version) if version is not None else None,
                    int(ptr),
                    lo,
                    hi,
                )
            )
        return accesses
    storage = event.get("output_storage")
    if storage is not None:
        ptr = event.get("output_ptr") or storage
        lo, hi = _access_range(event, "output_range")
        version = event.get("output_version")
        accesses.append((int(storage), int(version) if version is not None else None, int(ptr), lo, hi))
        return accesses
    out = event.get("output_ptr")
    if out is not None:
        accesses.append((int(out), None, int(out), 0.0, _RANGE_INF))
    return accesses


def _exact_runs(accesses: list[tuple], runs: list | None) -> list[tuple]:
    """Replace each operand's (storage, bounding interval) with its root allocation and exact byte runs."""
    if not runs or len(runs) != len(accesses):
        return accesses
    out = []
    for access, operand in zip(accesses, runs):
        if not operand or not operand.get("runs"):
            out.append(access)
            continue
        _key, version, ptr, _lo, _hi = access
        key = int(operand["storage"])
        out.extend((key, version, ptr, float(lo), float(hi)) for lo, hi in operand["runs"])
    return out


def _ranges_overlap(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> bool:
    """True when two half-open byte ranges intersect."""
    return a_lo < b_hi and b_lo < a_hi


def _subtract_interval(
    entry: tuple,
    cut_lo: float,
    cut_hi: float,
) -> list[tuple]:
    """Subtract ``[cut_lo, cut_hi)`` while preserving uncovered history.

    Entries are ``(lo, hi, end, engine, version, *extra)``; extra fields are carried unchanged.
    """
    lo, hi, *rest = entry
    if not _ranges_overlap(lo, hi, cut_lo, cut_hi):
        return [entry]
    result = []
    if lo < cut_lo:
        result.append((lo, min(hi, cut_lo), *rest))
    if cut_hi < hi:
        result.append((max(lo, cut_hi), hi, *rest))
    return result


def eliminate_redundant_hbm_loads(
    events: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Model exact compiler load CSE without using operator identities.

    A later HBM load is redundant only when it reads the same storage, exact
    byte range, shape, and byte count within the same grid program. Any HBM
    store overlapping the cached range invalidates it. This intentionally does
    not guess at partial overlap, eviction, or pointer aliasing.
    """
    retained: list[dict[str, Any]] = []
    cached: dict[tuple[Any, ...], tuple[float, float]] = {}
    eliminated_count = 0
    eliminated_bytes = 0
    for event in events:
        op = event.get("op")
        if op == "store" and str(event.get("mem_dst", "")).lower() == "hbm":
            storage = event.get("dst_storage", event.get("dst_ptr"))
            span = event.get("dst_range") or ()
            if storage is not None and len(span) == 2:
                lo, hi = float(span[0]), float(span[1])
                cached = {
                    key: cached_span
                    for key, cached_span in cached.items()
                    if key[1] != storage
                    or not _ranges_overlap(lo, hi, cached_span[0], cached_span[1])
                }
        if op == "load" and str(event.get("mem_src", "")).lower() == "hbm":
            storage = event.get("src_storage", event.get("src_ptr"))
            span = event.get("src_range") or ()
            if storage is not None and len(span) == 2:
                key = (
                    tuple(event.get("grid_idx") or ()),
                    storage,
                    tuple(span),
                    tuple(event.get("offsets_shape") or ()),
                    int(event.get("bytes") or 0),
                )
                if key in cached:
                    eliminated_count += 1
                    eliminated_bytes += int(event.get("bytes") or 0)
                    continue
                cached[key] = (float(span[0]), float(span[1]))
        retained.append(event)
    return retained, {
        "eliminated_load_count": eliminated_count,
        "eliminated_load_bytes": eliminated_bytes,
    }


def _expand_lowering_groups(
    events: Iterable[dict[str, Any]], model: CostModel
) -> list[dict[str, Any]]:
    """Replace calibrated source fusion groups with per-engine Level-A work."""
    source = list(events)
    if model.structured_control_lowering is None or model.compute_calibration is None:
        return source
    expanded: list[dict[str, Any]] = []
    index = 0
    while index < len(source):
        event = source[index]
        group = event.get("fusion_group")
        signature = event.get("fusion_signature")
        if group is None or not signature:
            expanded.append(event)
            index += 1
            continue
        end = index + 1
        while end < len(source) and source[end].get("fusion_group") == group:
            end += 1
        members = source[index:end]
        free_dim = max((_free_dim(member) or 0) for member in members)
        dtype = str(
            next(
                (
                    member.get("output_dtype")
                    for member in members
                    if member.get("output_dtype")
                ),
                "float32",
            )
        )
        targets: dict[str, Any] = {}
        structured = {}
        lowering_free_dim = free_dim
        region_ir = dict(event["region_ir"]) if event.get("region_ir") else {}
        if region_ir:
            compute_region = ComputeRegion.from_event(event)
            if compute_region is not None:
                region_ir["partition_count"] = compute_region.partition_count
                region_ir["logical_free_dim"] = compute_region.logical_free_dim
                lowering_free_dim = compute_region.logical_free_dim
        if (
            not targets
            and model.structured_control_lowering is not None
            and event.get("region_ir")
        ):
            # Normalize the lowering input through the shared, operator-agnostic
            # schema. This keeps partition/free geometry independent of an
            # operator name or structural calibration key.
            compute_region = ComputeRegion.from_event(event)
            if compute_region is not None:
                region_ir["partition_count"] = compute_region.partition_count
                region_ir["logical_free_dim"] = compute_region.logical_free_dim
            if str(region_ir.get("dtype", "")).lower() in {"bool", "boolean"}:
                value_dtypes = [
                    str(value)
                    for member in members
                    for value in (
                        [member.get("output_dtype")]
                        + list(member.get("input_dtypes") or [])
                    )
                    if value and str(value).lower() not in {"bool", "boolean"}
                ]
                if value_dtypes:
                    region_ir["dtype"] = Counter(value_dtypes).most_common(1)[0][0]
            points, structured_provenance = (
                model.structured_control_lowering.predict_points_with_provenance(
                    region_ir
                )
            )
            structured = {
                engine: (value[0], 2 if engine == "vector" else 1, value[2])
                for engine, value in points.items()
            }
            lowering_free_dim = int(
                region_ir.get("logical_free_dim")
                or region_ir.get("free_dim")
                or free_dim
            )
            targets = {
                engine: (value[0], value[1]) for engine, value in structured.items()
            }
        else:
            structured_provenance = {}
        if not targets:
            for member in members:
                retained = dict(member)
                retained["level_a_match"] = "none"
                retained["micro_dag_match"] = "missing"
                retained["micro_dag_region"] = str(
                    member.get("source_region_id") or group
                )
                expanded.append(retained)
            index = end
            continue

        first_inputs = list(members[0].get("input_ptrs") or ())
        first_input_storages = list(members[0].get("input_storages") or ())
        first_input_ranges = list(members[0].get("input_ranges") or ())
        first_input_versions = list(members[0].get("input_versions") or ())
        if any(member.get("input_storages") for member in members):
            external: dict[tuple[int, int | None], tuple[int, Any, int | None]] = {}
            produced: set[tuple[int, int | None]] = set()
            for member in members:
                storages = member.get("input_storages") or ()
                ptrs = member.get("input_ptrs") or ()
                ranges = member.get("input_ranges") or ()
                versions = member.get("input_versions") or ()
                for item_index, storage in enumerate(storages):
                    version = versions[item_index] if item_index < len(versions) else None
                    identity = (int(storage), version)
                    if identity not in produced:
                        ptr = ptrs[item_index] if item_index < len(ptrs) else storage
                        rng = ranges[item_index] if item_index < len(ranges) else None
                        external.setdefault(identity, (int(ptr), rng, version))
                output_storage = member.get("output_storage")
                if output_storage is not None:
                    produced.add((int(output_storage), member.get("output_version")))
            first_input_storages = [identity[0] for identity in external]
            first_inputs = [value[0] for value in external.values()]
            first_input_ranges = [value[1] for value in external.values()]
            first_input_versions = [value[2] for value in external.values()]
        final_output = members[-1].get("output_ptr")
        final_output_storage = members[-1].get("output_storage")
        final_output_range = members[-1].get("output_range")
        final_output_version = members[-1].get("output_version")
        ordered_targets = sorted(targets.items())
        dag_match = "missing"
        dag = None
        if (
            structured
            and model.structured_control_lowering is not None
            and event.get("region_ir")
        ):
            dag, dag_match = model.structured_control_lowering.micro_dag_lookup(
                region_ir
            )
            if dag is not None:
                engine_first_start = {}
                for node in dag.get("nodes", []):
                    engine_name = _canonical_engine(
                        str(node.get("engine") or ""), "compute"
                    )
                    engine_first_start.setdefault(
                        engine_name, len(engine_first_start)
                    )
                ordered_targets = sorted(
                    targets.items(),
                    key=lambda item: (
                        engine_first_start.get(item[0], len(engine_first_start)),
                        item[0],
                    ),
                )
        dag_engines = (
            {
                _canonical_engine(str(node.get("engine") or ""), "compute")
                for node in dag.get("nodes", [])
                if not node.get("is_sync")
            }
            if dag is not None
            else set()
        )
        if (
            dag_match == "exact"
            and dag is not None
            and not dag.get("unsupported_unmapped_payload", True)
            and set(targets).issubset(dag_engines)
        ):
            dag_nodes = {str(node["id"]): node for node in dag.get("nodes", [])}
            predecessors: dict[str, list[str]] = {node_id: [] for node_id in dag_nodes}
            successors: dict[str, list[str]] = {node_id: [] for node_id in dag_nodes}
            for source_id, target_id in dag.get("edges", []):
                source_id, target_id = str(source_id), str(target_id)
                if source_id in dag_nodes and target_id in dag_nodes:
                    predecessors[target_id].append(source_id)
                    successors[source_id].append(target_id)
            roots = {node_id for node_id, deps in predecessors.items() if not deps}
            sinks = {node_id for node_id, deps in successors.items() if not deps}

            # Flow supplies topology only.  Do not copy control instruction
            # durations into holdouts: that would turn the micro-DAG into an
            # exact timing lookup.  Instead preserve the independently
            # calibrated per-engine aggregate work and distribute it over the
            # exact control-backed nodes in proportion to their observed
            # instruction durations. Unsupported engines remain explicit
            # zero-duration nodes and are reported as unsupported rather than
            # being silently priced from holdout/control wall time.
            predicted_engine_ns: dict[str, float] = {}
            for engine, (count, streams) in targets.items():
                aggregate = dict(members[0])
                aggregate.update(
                    {
                        "op": "compute",
                        "api_op": "lowered_fusion",
                        "engine": engine,
                        "input_shape": [1, lowering_free_dim],
                        "output_shape": [1, lowering_free_dim],
                        "free_dim": lowering_free_dim,
                        "output_dtype": dtype,
                        "input_stream_count": streams,
                        "lowering_expansion": count,
                        "lowering_fixed_ns": (
                            structured.get(engine, (0.0, streams, 0.0))[2]
                            if int(group) == 0
                            else 0.0
                        ),
                    }
                )
                predicted_engine_ns[engine] = model.cost_ns(aggregate)
            observed_by_engine: dict[str, float] = Counter()
            node_engine: dict[str, str] = {}
            for node_id, node in dag_nodes.items():
                engine = _canonical_engine(
                    str(node.get("engine") or ""), "compute"
                )
                node_engine[node_id] = engine
                if engine in predicted_engine_ns:
                    timing = node.get("timing") or {}
                    observed_by_engine[engine] += max(
                        0.0, float(timing.get("completion_latency_ns") or 0.0)
                    )
            unsupported_engines = sorted(dag_engines - set(targets) - {ENGINE_SYNC})
            opcode_timing: dict[str, tuple[float, str]] = {}
            for node_id, node in dag_nodes.items():
                if node.get("is_sync"):
                    continue
                opcode_timing[node_id] = (
                    model.structured_control_lowering.opcode_timing_lookup(
                        node_engine[node_id],
                        dtype,
                        str(node.get("opcode_family") or ""),
                        lowering_free_dim,
                    )
                )
            use_opcode_timing = (
                bool(opcode_timing)
                and all(match != "ood" for _value, match in opcode_timing.values())
            )
            opcode_weight_by_engine: dict[str, float] = Counter()
            if use_opcode_timing:
                for node_id, (value, _match) in opcode_timing.items():
                    opcode_weight_by_engine[node_engine[node_id]] += max(0.0, value)
            normalize_opcode_totals = int(region_ir.get("reduction_count") or 0) > 0
            for node_index, node in enumerate(dag.get("nodes", [])):
                node_id = str(node["id"])
                timing = node.get("timing") or {}
                engine = node_engine[node_id]
                observed_ns = max(
                    0.0, float(timing.get("completion_latency_ns") or 0.0)
                )
                duration_ns = 0.0
                timing_match = "not_applicable"
                if node_id in opcode_timing and use_opcode_timing:
                    opcode_ns, timing_match = opcode_timing[node_id]
                    denominator = opcode_weight_by_engine.get(engine, 0.0)
                    if (
                        normalize_opcode_totals
                        and engine in predicted_engine_ns
                        and denominator > 0
                    ):
                        # Reduction lowering is context-sensitive: opcode
                        # durations define relative placement while Level-A/B
                        # supplies the independently calibrated region total.
                        duration_ns = (
                            predicted_engine_ns[engine]
                            * max(0.0, opcode_ns)
                            / denominator
                        )
                    else:
                        # Straight-line opcodes have an independent timing
                        # surface and may execute on engines (notably GpSimdE)
                        # that have no aggregate Level-B instruction model.
                        duration_ns = max(0.0, opcode_ns)
                elif engine in predicted_engine_ns:
                    denominator = observed_by_engine.get(engine, 0.0)
                    duration_ns = (
                        predicted_engine_ns[engine] * observed_ns / denominator
                        if denominator > 0
                        else predicted_engine_ns[engine]
                    )
                    timing_match = "aggregate_calibration"
                lowered = dict(members[0])
                lowered.update(
                    {
                        "op": "micro_event",
                        "api_op": "lowered_micro_event",
                        "engine": engine,
                        "opcode_family": str(node.get("opcode_family") or ""),
                        "scheduler_duration_override_ns": duration_ns,
                        "micro_event_id": node_id,
                        "micro_event_predecessors": predecessors[node_id],
                        "micro_dag_match": "exact",
                        "level_a_match": structured_provenance.get(
                            engine, "structured"
                        ),
                        "micro_dag_region": str(
                            event.get("source_region_id") or group
                        ),
                        "micro_dag_order": node_index,
                        "micro_dag_timing_source": (
                            "opcode_control_surface"
                            if use_opcode_timing
                            else "calibrated_engine_work"
                        ),
                        "micro_dag_timing_match": timing_match,
                        "micro_dag_unsupported_engines": unsupported_engines,
                        "input_ptrs": first_inputs if node_id in roots else [],
                        "input_storages": (
                            first_input_storages if node_id in roots else []
                        ),
                        "input_ranges": first_input_ranges if node_id in roots else [],
                        "input_versions": (
                            first_input_versions if node_id in roots else []
                        ),
                        "output_ptr": None,
                        "output_storage": None,
                        "output_range": None,
                        "output_version": None,
                    }
                )
                expanded.append(lowered)
            publish = dict(members[-1])
            publish.update(
                {
                    "op": "micro_event",
                    "api_op": "lowered_micro_publish",
                    "engine": ENGINE_VECTOR,
                    "opcode_family": "PUBLISH",
                    "scheduler_duration_override_ns": 0.0,
                    "micro_event_id": f"publish:{group}",
                    "micro_event_predecessors": sorted(sinks),
                    "micro_dag_match": "exact",
                    "micro_dag_region": str(
                        event.get("source_region_id") or group
                    ),
                    "input_ptrs": [],
                    "input_storages": [],
                    "input_ranges": [],
                    "input_versions": [],
                    "output_ptr": final_output,
                    "output_storage": final_output_storage,
                    "output_range": final_output_range,
                    "output_version": final_output_version,
                }
            )
            expanded.append(publish)
            index = end
            continue
        for target_index, (engine, (count, streams)) in enumerate(ordered_targets):
            lowered = dict(members[0])
            lowered.update(
                {
                    "op": "compute",
                    "api_op": "lowered_fusion",
                    "engine": engine,
                    "input_shape": [1, lowering_free_dim],
                    "output_shape": [1, lowering_free_dim],
                    "free_dim": lowering_free_dim,
                    "output_dtype": dtype,
                    "input_ptrs": first_inputs,
                    "input_storages": first_input_storages,
                    "input_ranges": first_input_ranges,
                    "input_versions": first_input_versions,
                    "input_stream_count": streams,
                    "lowering_expansion": count,
                    "lowering_fixed_ns": (
                        structured.get(engine, (0.0, streams, 0.0))[2]
                        if int(group) == 0
                        else 0.0
                    ),
                    "lowered_from_signature": signature,
                    "level_a_match": (
                        structured_provenance.get(engine, "structured")
                        if model.structured_control_lowering is not None
                        else "signature"
                    ),
                    "micro_dag_match": dag_match,
                    "micro_dag_region": str(
                        event.get("source_region_id") or group
                    ),
                    "micro_dag_order": target_index,
                    # Publish the logical result only once; otherwise parallel
                    # target engines would create artificial WAW hazards.
                    "output_ptr": final_output
                    if target_index == len(ordered_targets) - 1
                    else None,
                    "output_storage": final_output_storage
                    if target_index == len(ordered_targets) - 1
                    else None,
                    "output_range": final_output_range
                    if target_index == len(ordered_targets) - 1
                    else None,
                    "output_version": final_output_version
                    if target_index == len(ordered_targets) - 1
                    else None,
                }
            )
            expanded.append(lowered)
        index = end
    return expanded


def _scale_to_routed_occupancy(events, routing_events, routing_dtype, model):
    """Make the schedule consistent with control-routed occupancy (structure-schedule mode only).

    Whole-program routing replaces Vector/Scalar/GpSimd busy time with control-learned occupancy after
    scheduling, but the per-event durations on the timeline stay unrouted. When the schedule itself sets
    the completion, each routed engine's event durations are scaled so that they sum to its routed busy.
    """
    value_dtypes = [str(v) for e in events for v in ([e.get("output_dtype")] + list(e.get("input_dtypes") or []))
                    if v and str(v).lower() not in {"bool", "boolean"}]
    dtype = ComputeCalibration._norm_dtype(
        routing_dtype or (Counter(value_dtypes).most_common(1)[0][0] if value_dtypes else "float32"))
    routed = model.whole_program_routing.predict_with_provenance(list(routing_events), dtype)
    if routed is None or routed[0] is None:
        return events
    raw: dict[str, float] = defaultdict(float)
    costs = []
    for e in events:
        engine = _canonical_engine(e.get("engine", ""), e.get("op") or "")
        c = float(e.get("scheduler_duration_override_ns", model.cost_ns(e))) if engine in routed[0] else 0.0
        costs.append((engine, c))
        raw[engine] += c
    out = []
    for e, (engine, c) in zip(events, costs):
        if engine in routed[0] and raw[engine] > 0 and c > 0:
            e = dict(e)
            e["scheduler_duration_override_ns"] = c * routed[0][engine] / raw[engine]
        out.append(e)
    return out


def simulate(
    events: Iterable[dict[str, Any]],
    cost_model: CostModel | None = None,
    *,
    routing_source_events: Iterable[dict[str, Any]] | None = None,
    routing_dtype: str | None = None,
) -> SimulationResult:
    """Schedule trace events onto per-engine timelines with data dependencies.

    Scheduling policy (models both parallelism and true data hazards):
      * Each engine is a serial resource; its next op cannot start until its
        previous op finishes (program order within an engine).
      * Full memory dependencies through base pointers:
          - RAW (read-after-write): an op waits for the last writer of every
            buffer it reads;
          - WAW (write-after-write): an op that writes a buffer waits for the
            previous writer of that buffer;
          - WAR (write-after-read): an op that overwrites a buffer waits for all
            prior readers of that buffer to finish (buffer-reuse anti-hazard).
        WAR/WAW matter on NeuronCore because SBUF tiles are reused across DMA
        and compute engines; ignoring them lets an overwrite float before the
        readers/writers it must follow, under-counting latency.
      * Cross-engine sync: when a resolved dependency crosses an engine boundary
        the consumer additionally waits ``cross_engine_sync_ns`` (semaphore
        handoff). Same-engine dependencies pay nothing beyond queueing.
      * Parallel DMA queues: the DMA engine is modeled as ``dma_queue_count``
        independent slots, so transfers with no data hazard between them overlap
        (each transfer runs on the slot that frees earliest). Compute engines
        are single serial slots. Data-dependent transfers still serialize
        through the hazard logic below.
      * ``grid`` markers are ignored for timing.

    Conservative fallback: a compute op that carries *no* resolvable input
    pointers (older traces) still waits for every memory transfer issued before
    it, so the timeline stays physically plausible. With pointer linkage present
    (see ``Dot.input_ptrs``) the precise RAW edge is used instead.

    The result is a predicted end-to-end latency plus per-engine timelines.
    """
    model = cost_model or CostModel()
    source_events = list(events)
    routing_events = (
        list(routing_source_events)
        if routing_source_events is not None
        else source_events
    )
    structural_static_ns, structural_static_match = (
        model.structural_static_dma.predict_ns_with_provenance(source_events)
        if model.structural_static_dma is not None
        else (0.0, "none")
    )
    strided_dma_prediction = (
        model.strided_dma_calibration.predict_components(source_events)
        if model.strided_dma_calibration is not None
        else None
    )
    if strided_dma_prediction is not None:
        strided_dynamic_dma_ns, strided_static_dma_ns = strided_dma_prediction
        matched_indices = model.strided_dma_calibration.matched_indices(source_events)
        unmatched_indices = [
            index
            for index, event in enumerate(source_events)
            if event.get("op") in {"load", "store", "transfer"}
            and index not in matched_indices
        ]
        unmatched_total = sum(
            model.cost_ns(source_events[index]) for index in unmatched_indices
        )
        matched_costs = [
            model.cost_ns(source_events[index]) for index in matched_indices
        ]
        matched_raw_total = sum(matched_costs)
        matched_target_total = max(0.0, strided_dynamic_dma_ns - unmatched_total)
        if matched_raw_total > 0:
            scale = matched_target_total / matched_raw_total
            source_events = [dict(event) for event in source_events]
            for index, raw_cost in zip(matched_indices, matched_costs):
                source_events[index]["scheduler_duration_override_ns"] = (
                    raw_cost * scale
                )
        if strided_static_dma_ns > 0:
            structural_static_ns = max(structural_static_ns, strided_static_dma_ns)
    events = _expand_lowering_groups(source_events, model)
    structure_depth = 0
    if model.structure_schedule is not None:
        from triton_viz.tools.nki_structure_schedule import apply_structure_schedule
        import dataclasses as _dc
        events, structure_depth = apply_structure_schedule(
            events, _dc.replace(model.structure_schedule, exact_ranges=model.exact_tile_ranges))
        if model.whole_program_routing is not None:
            events = _scale_to_routed_occupancy(events, routing_events, routing_dtype, model)
    level_a_matches = Counter(
        match
        for _region, _engine, match in {
            (
                str(event.get("micro_dag_region") or event.get("source_region_id") or event.get("fusion_group")),
                str(event.get("engine") or ""),
                str(event.get("level_a_match")),
            )
            for event in events
            if event.get("level_a_match")
            and event.get("api_op") != "lowered_micro_publish"
        }
    )
    if structural_static_ns > 0:
        events.append(
            {
                "seq": -1,
                "op": "micro_event",
                "api_op": "compiler_static_dma",
                "engine": ENGINE_STATIC_DMA,
                "opcode_family": "STATIC_DMA_PACKET_TRAIN",
                "scheduler_duration_override_ns": structural_static_ns,
                "static_dma_dependency_unknown": True,
            }
        )
    tensor_events = [
        event
        for event in events
        if _canonical_engine(event.get("engine", ""), event.get("op"))
        == ENGINE_TENSOR
    ]
    tensor_startup_ns = 0.0
    attention_pipeline_match = "disabled"
    tensor_domain_ood = 0
    tensor_geometry_ood = 0
    micro_dag_engine_coverage: set[str] = set()
    source_compute_regions = {
        str(event.get("source_region_id") or event.get("fusion_group"))
        for event in source_events
        if event.get("region_ir") is not None
    }
    exact_micro_dag_regions = {
        str(event.get("micro_dag_region"))
        for event in events
        if event.get("micro_dag_match") == "exact"
        and event.get("micro_dag_region") is not None
    }
    complete_micro_dag_coverage = (
        bool(source_compute_regions)
        and source_compute_regions.issubset(exact_micro_dag_regions)
    )
    micro_dag_unsupported_engine_events = 0
    micro_dag_timing_matches: Counter[str] = Counter()
    for event in events:
        if event.get("micro_dag_match") != "exact":
            continue
        engine = _canonical_engine(event.get("engine", ""), event.get("op", ""))
        if (
            complete_micro_dag_coverage
            and float(event.get("scheduler_duration_override_ns") or 0.0) > 0
        ):
            micro_dag_engine_coverage.add(engine)
        micro_dag_timing_matches[str(event.get("micro_dag_timing_match") or "")] += 1
        micro_dag_unsupported_engine_events += len(
            event.get("micro_dag_unsupported_engines") or ()
        )
    law_dot_events = (
        [event for event in tensor_events if event.get("op") == "dot"]
        if model.tensor_instruction_law is not None
        else []
    )
    if (
        model.attention_pipeline_calibration is not None
        and len(law_dot_events) == 2
        and int(law_dot_events[0].get("attention_pipeline_value_width") or 0) > 0
    ):
        # The law replaces the FLOP surface only: the two-Dot attention pipeline keeps the surface
        # calibrated on its own controls.
        law_dot_events = []
    law_costs = [
        model.tensor_instruction_law.instruction_ns(_tensor_dtype(event), *_dot_geometry(event))
        if _dot_geometry(event) is not None
        else None
        for event in law_dot_events
    ]
    if law_dot_events and all(cost is not None for cost in law_costs):
        # Opt-in TensorE instruction law: each Dot is charged its geometry's issue interval.
        dtype = Counter(_tensor_dtype(event) for event in law_dot_events).most_common(1)[0][0]
        tensor_startup_ns = max(0.0, model.tensor_instruction_law.startup_ns(dtype))
        for event, cost in zip(law_dot_events, law_costs):
            event["scheduler_duration_override_ns"] = cost
            event["tensor_instruction_law"] = True
    elif model.tensor_calibration is not None:
        calibrated_dot_events = [
            event for event in tensor_events if event.get("op") == "dot"
        ]
        if calibrated_dot_events:
            dtype = Counter(
                _tensor_dtype(event) for event in calibrated_dot_events
            ).most_common(1)[0][0]
            total_tensor_flops = sum(
                max(0, int(event.get("flops") or 0))
                for event in calibrated_dot_events
            )
            calibrated_tensor_active_ns = 0.0
            attention_value_width = int(
                calibrated_dot_events[0].get("attention_pipeline_value_width") or 0
            )
            source_dot_count = int(
                calibrated_dot_events[0].get("tensor_source_dot_count") or 0
            )
            source_dot_surface_used = False
            if (
                model.attention_pipeline_calibration is not None
                and len(calibrated_dot_events) == 2
                and attention_value_width > 0
            ):
                (
                    calibrated_tensor_active_ns,
                    attention_pipeline_match,
                ) = model.attention_pipeline_calibration.predict_ns(
                    dtype, attention_value_width
                )
                source_dot_surface_used = calibrated_tensor_active_ns > 0
            if (
                model.tensor_dot_count_calibration is not None
                and not source_dot_surface_used
                and source_dot_count == len(calibrated_dot_events)
            ):
                calibrated_tensor_active_ns, _match = (
                    model.tensor_dot_count_calibration.active_ns(
                        dtype, source_dot_count,
                        int(calibrated_dot_events[0].get("tensor_source_lhs_tile_count") or 0),
                        int(calibrated_dot_events[0].get("tensor_source_rhs_tile_count") or 0),
                        int(calibrated_dot_events[0].get("tensor_source_output_tile_count") or 0),
                    )
                )
                source_dot_surface_used = calibrated_tensor_active_ns > 0
                if source_dot_surface_used and _match.endswith("_ood"):
                    tensor_geometry_ood = len(calibrated_dot_events)
            if source_dot_surface_used:
                tensor_startup_ns = 0.0
                # The fitted target is Explorer's whole TensorE active union,
                # including compiler-created transpose/load-weight work. Once
                # that total is assigned across Dot events, source transpose
                # proxies must not be charged a second time.
                for tensor_event in tensor_events:
                    if tensor_event.get("op") != "dot":
                        tensor_event["scheduler_duration_override_ns"] = 0.0
            else:
                tensor_startup_ns = max(
                    0.0,
                    model.tensor_calibration.startup_ns(
                        dtype, strict=model.strict_calibration
                    ),
                )
                calibrated_tensor_active_ns = model.tensor_calibration.active_ns(
                    dtype,
                    total_tensor_flops,
                    strict=model.strict_calibration,
                )
                tensor_startup_ns = min(
                    tensor_startup_ns, calibrated_tensor_active_ns
                )
            tensor_work_ns = max(
                0.0, calibrated_tensor_active_ns - tensor_startup_ns
            )
            if total_tensor_flops > 0:
                for event in calibrated_dot_events:
                    event["scheduler_duration_override_ns"] = (
                        tensor_work_ns
                        * max(0, int(event.get("flops") or 0))
                        / total_tensor_flops
                    )
        tensor_domain_ood = sum(
            1
            for event in calibrated_dot_events
            if (flops := int(event.get("flops") or 0)) > 0
            and model.tensor_calibration.domain_match(
                _tensor_dtype(event), flops
            )
            != "in_domain"
        )
    sync_ns = max(0.0, model.cross_engine_sync_ns)
    dma_queue_count = max(1, int(model.dma_queue_count))
    dma_resource_count = max(0, int(model.dma_resource_count))
    # Each engine owns a list of per-slot free times. Most engines are a single
    # serial slot; the DMA engine owns ``dma_queue_count`` parallel slots.
    engine_slots: dict[str, list[float]] = {}
    if tensor_startup_ns > 0:
        engine_slots[ENGINE_TENSOR] = [tensor_startup_ns]
    # Per-storage hazard tracking keyed by storage id. Each writer/reader entry
    # is (lo, hi, end_time, engine) so we can test *address-range* overlap, not
    # just base-pointer equality: disjoint tiles of one allocation run in
    # parallel, while overlapping ranges (even via different view pointers that
    # share a storage id) still serialize with the correct RAW/WAR/WAW edge.
    writers: dict[int, list[tuple[float, float, float, str, int | None, float]]] = {}
    readers: dict[int, list[tuple[float, float, float, str, int | None]]] = {}
    timeline: dict[str, list[TimelineEntry]] = {}
    engine_busy: dict[str, float] = {}
    # Runtime/setup instructions are part of Explorer's engine ACTIVE counters,
    # but not of the source-mapped payload labels.  Preserve the independently
    # calibrated contribution so replay can compare like with like.
    engine_runtime_baseline_ns: dict[str, float] = {}
    dma_surface_matches: Counter[str] = Counter()
    dma_calibration_paths: Counter[str] = Counter()
    dma_surface_max_log_distance = 0.0
    if tensor_startup_ns > 0:
        engine_busy[ENGINE_TENSOR] = tensor_startup_ns
    makespan = tensor_startup_ns
    makespan_dynamic = tensor_startup_ns
    cross_engine_edges = 0
    cross_engine_edge_keys: set[tuple[Any, ...]] = set()
    micro_event_end: dict[str, tuple[float, str]] = {}
    # Running high-water mark of all memory-transfer completions, used as a
    # conservative dependency floor for compute ops that lack pointer linkage.
    prior_transfer_end = 0.0

    def _dep(
        earliest: float,
        producer_end: float,
        producer_engine: str,
        consumer_engine: str,
        edge_key: tuple[Any, ...] | None = None,
    ) -> float:
        """Fold one dependency edge in, adding sync cost if it crosses engines."""
        nonlocal cross_engine_edges
        ready = producer_end
        if producer_engine != consumer_engine:
            if edge_key is None or edge_key not in cross_engine_edge_keys:
                cross_engine_edges += 1
                if edge_key is not None:
                    cross_engine_edge_keys.add(edge_key)
            ready += sync_ns
        return max(earliest, ready)

    def _slots_for(engine: str) -> list[float]:
        if engine not in engine_slots:
            width = (
                dma_resource_count or dma_queue_count if engine == ENGINE_DMA else 1
            )
            engine_slots[engine] = [0.0] * width
        return engine_slots[engine]

    for event in events:
        op = event.get("op")
        if op in (None, "grid", "unknown"):
            continue
        engine = _canonical_engine(event.get("engine", ""), op)
        if op in ("transfer", "load", "store") and engine == ENGINE_DMA:
            lookup = model.dma_lookup(event)
            dma_calibration_paths[str(lookup["path"])] += 1
            dma_surface_matches[str(lookup["match"])] += 1
            dma_surface_max_log_distance = max(
                dma_surface_max_log_distance,
                float(lookup.get("log_distance") or 0.0),
            )
        duration = float(
            event.get("scheduler_duration_override_ns", model.cost_ns(event))
        )

        if (event.get("op") == "dot"
                and "scheduler_duration_override_ns" not in event
                and str(event.get("tensor_dot_count_calibration_match", "")).endswith("_ood")):
            tensor_geometry_ood += 1

        reads = _read_accesses(event)
        writes = _write_accesses(event)
        if model.exact_tile_ranges and event.get("op") not in ("transfer", "load", "store"):
            reads = _exact_runs(reads, event.get("input_range_lists"))
            writes = _exact_runs(writes, event.get("output_range_lists"))

        # Earliest start from engine availability. For a multi-slot engine the
        # op takes the slot that frees earliest, letting independent transfers
        # overlap; a single-slot engine keeps strict program order.
        multiqueue = (model.dma_queues > 0 and engine == ENGINE_DMA and op in ("transfer", "load", "store")
                      and "scheduler_duration_override_ns" not in event)
        if multiqueue:
            slots = engine_slots.setdefault("dma_mq", [0.0] * int(model.dma_queues))
        else:
            slots = _slots_for(engine)
        queue_free = min(slots) if slots else 0.0
        if multiqueue:
            slot_indices = [min(range(len(slots)), key=lambda i: slots[i])]
            earliest = slots[slot_indices[0]]
        elif engine == ENGINE_DMA and dma_resource_count:
            partitions = max(1, int(event.get("partition_count") or 1))
            demand = min(dma_resource_count, partitions)
            slot_indices = sorted(range(len(slots)), key=lambda i: slots[i])[:demand]
            earliest = max(slots[index] for index in slot_indices)
        else:
            slot_indices = [min(range(len(slots)), key=lambda i: slots[i])]
            earliest = slots[slot_indices[0]]

        # Compiler Flow predecessors inside an exact control-backed micro-DAG.
        # These edges are explicit evidence, independent of reconstructed
        # storage aliases.
        for predecessor in event.get("micro_event_predecessors") or ():
            producer = micro_event_end.get(str(predecessor))
            if producer is not None:
                earliest = _dep(
                    earliest,
                    producer[0],
                    producer[1],
                    engine,
                    ("micro", str(predecessor), str(event.get("micro_event_id"))),
                )

        # RAW: wait for prior writers whose range overlaps a buffer we read; a
        # reader needs the producer's result, i.e. its result-ready time.
        for key, version, _ptr, lo, hi in reads:
            for w_lo, w_hi, _w_end, w_eng, w_version, w_ready in writers.get(key, ()):
                version_matches = (
                    version is None or w_version is None or version == w_version
                )
                if version_matches and _ranges_overlap(lo, hi, w_lo, w_hi):
                    earliest = _dep(earliest, w_ready, w_eng, engine)

        # WAW + WAR: a write waits for prior overlapping writers and readers of
        # that storage (buffer-reuse hazards across engines). Ordering two
        # writes needs only the earlier one to have been issued and released by
        # its engine, not its result latency.
        for key, _version, _ptr, lo, hi in writes:
            for w_lo, w_hi, w_end, w_eng, _w_version, _w_ready in writers.get(key, ()):
                if _ranges_overlap(lo, hi, w_lo, w_hi):
                    earliest = _dep(earliest, w_end, w_eng, engine)
            for r_lo, r_hi, r_end, r_eng, _r_version in readers.get(key, ()):
                if _ranges_overlap(lo, hi, r_lo, r_hi):
                    earliest = _dep(earliest, r_end, r_eng, engine)

        if op in ("dot", "tensor_transpose") and not reads:
            # Compute op without pointer linkage: conservatively depend on all
            # memory transfers issued so far (older traces without Dot pointers).
            earliest = max(earliest, prior_transfer_end + (sync_ns if sync_ns else 0.0))

        start = earliest
        share = 1.0
        if multiqueue:
            parts = model.dma_split_parts(event)
            if parts is not None:
                fixed, stream = parts                    # stream = bytes / single-transfer asymptotic bandwidth
                in_flight = 1 + sum(1 for i, t in enumerate(slots) if i != slot_indices[0] and t > start)
                nbytes = float(event.get("bytes") or 0)
                aggregate = model.dma_aggregate_gbps     # bytes per ns
                duration = max(stream, nbytes * in_flight / aggregate) + (fixed if in_flight == 1 else 0.0)
                share = 1.0 / in_flight
        pipelined = None
        if (model.dma_pipelined_fixed_cost and not multiqueue and engine == ENGINE_DMA
                and op in ("transfer", "load", "store") and "scheduler_duration_override_ns" not in event):
            pipelined = model.dma_split_parts(event)
        if pipelined is not None:
            fixed, stream = pipelined
            # an idle queue pays the fixed cost; a busy one hides it behind the previous stream
            busy = queue_free > 0.0 and start <= queue_free + 1.0
            duration = stream + (0.0 if busy else fixed)
        end = start + duration
        # Result visibility (>= engine release); equal to `end` unless instruction latency is modelled.
        ready = (end + (model.dma_result_latency_ns or 0.0) if pipelined is not None
                 else start + model.result_ready_ns(event, engine, duration))

        for slot_index in slot_indices:
            slots[slot_index] = end

        # Publish this op's effects for downstream dependency resolution.
        for key, version, _ptr, lo, hi in reads:
            readers.setdefault(key, []).append((lo, hi, end, engine, version))
        for key, version, _ptr, lo, hi in writes:
            # A new write to a range supersedes prior writers/readers that it
            # fully covers; keep only the non-overlapping history so later
            # disjoint accesses are not spuriously serialized against stale
            # entries, while overlapping history is replaced by this write.
            writers[key] = [
                remainder
                for entry in writers.get(key, ())
                for remainder in _subtract_interval(entry, lo, hi)
            ]
            writers[key].append((lo, hi, end, engine, version, ready))
            readers[key] = [
                remainder
                for entry in readers.get(key, ())
                for remainder in _subtract_interval(entry, lo, hi)
            ]
        if op in ("transfer", "load", "store"):
            prior_transfer_end = max(prior_transfer_end, end)

        timeline.setdefault(engine, []).append(
            TimelineEntry(
                seq=int(event.get("seq", -1)),
                op=op,
                engine=engine,
                start=start,
                end=end,
            )
        )
        engine_busy[engine] = engine_busy.get(engine, 0.0) + duration * share
        makespan = max(makespan, ready)
        if engine != ENGINE_STATIC_DMA:
            makespan_dynamic = max(makespan_dynamic, ready)
        micro_event_id = event.get("micro_event_id")
        if micro_event_id is not None:
            micro_event_end[str(micro_event_id)] = (ready, engine)

    # Compose control-learned whole-program routing into the resource model.
    # The source DAG still determines ordering/overlap; calibrated aggregate
    # occupancy replaces only Vector/Scalar/GpSimd work and provides a physical
    # makespan lower bound.  This avoids inventing target compiler Flow edges.
    whole_program_routing_match = "disabled"
    value_dtypes = [
        str(value)
        for event in source_events
        for value in ([event.get("output_dtype")] + list(event.get("input_dtypes") or []))
        if value and str(value).lower() not in {"bool", "boolean"}
    ]
    source_dtype = ComputeCalibration._norm_dtype(
        routing_dtype
        or (Counter(value_dtypes).most_common(1)[0][0] if value_dtypes else "float32")
    )
    if model.whole_program_routing is not None:
        routed = model.whole_program_routing.predict_with_provenance(
            routing_events, source_dtype
        )
        routed_busy = None if routed is None else routed[0]
        if routed_busy is not None:
            whole_program_routing_match = routed[1]
            engine_busy.update(routed_busy)
            makespan = max(makespan, max(routed_busy.values(), default=0.0))
        else:
            whole_program_routing_match = "ood"

    # DMA descriptor-issue floor.  Every fragmented transfer element occupies
    # the shared DMA issue path for a measured interval, so the DMA resource
    # cannot finish sooner than the total descriptor count times that interval.
    # This is a physical occupancy bound on engine busy time computed from the
    # access geometry; it is not keyed by program structure.
    dma_queue_floor_ns = 0.0
    dma_descriptor_count = 0.0
    dma_fragmented_descriptors = 0.0
    if model.dma_elapsed_calibration is not None:
        (
            dma_queue_floor_ns,
            dma_descriptor_count,
            dma_fragmented_descriptors,
        ) = model.dma_elapsed_calibration.queue_floor_ns(
            source_events, max(1, int(model.dma_max_engines))
        )
        dma_busy_before_floor = engine_busy.get(ENGINE_DMA, 0.0) + engine_busy.get(
            ENGINE_STATIC_DMA, 0.0
        )
        if dma_queue_floor_ns > dma_busy_before_floor:
            engine_busy[ENGINE_DMA] = (
                engine_busy.get(ENGINE_DMA, 0.0)
                + dma_queue_floor_ns
                - dma_busy_before_floor
            )
            makespan = max(makespan, dma_queue_floor_ns)

    # There is exactly one global overhead model: the completion term below.
    # An earlier parametric runtime-overhead floor estimated the same physical
    # quantity and was combined with a max, which is biased upward and double
    # counts the launch cost the completion offset already contains.  An A/B on
    # the 288 independent whole-program controls confirmed it: the floor moved
    # control NC-p50 MAPE from 7.9585% to 10.9133% and the median signed error
    # from -1.40% to +2.26%.  It was therefore removed, not retuned.
    max_partitions = max(
        (int(event.get("partition_count") or 1) for event in source_events),
        default=1,
    )
    compute_critical_ns = max(
        engine_busy.get(ENGINE_VECTOR, 0.0),
        engine_busy.get(ENGINE_SCALAR, 0.0),
        engine_busy.get(ENGINE_TENSOR, 0.0),
    )
    dma_busy_ns = engine_busy.get(ENGINE_DMA, 0.0) + engine_busy.get(
        ENGINE_STATIC_DMA, 0.0
    )
    phase_critical_ns = dma_busy_ns + compute_critical_ns
    final_ns = makespan + max(0.0, model.kernel_overhead_ns)
    # One global completion term replaces every per-structure completion
    # lookup table.  It is applied identically to every program: a single
    # measured non-overlap fraction of the non-critical engines plus a single
    # measured launch/drain constant, both frozen from independent controls
    # under leave-one-free-dimension-out CV.  There is no structural key, so
    # no program can be given a bespoke floor.
    makespan_only_ns = final_ns
    completion_partitions = max(
        [int(event.get("partition_count") or 0) for event in source_events] + [1]
    )
    calibration = model.global_completion_calibration
    carried_fraction = (
        carried_load_fraction(source_events)
        if calibration is not None and calibration.carried_serialization
        else 0.0
    )
    global_completion_ns = (
        model.global_completion_calibration.predict_ns(
            engine_busy, completion_partitions, carried_fraction
        )
        if model.global_completion_calibration is not None
        else 0.0
    )
    final_ns = max(makespan_only_ns, global_completion_ns)
    # Programs priced by a dedicated control-calibrated path (attention pipeline) keep that result.
    if (model.structure_schedule is not None and model.structure_schedule.schedule_completion
            and attention_pipeline_match not in {"exact", "interpolated"}):
        # Schedule-derived completion: the structure-aware schedule replaces the structure-free
        # non-overlap term; the controls-calibrated launch/drain offset (which no schedule contains)
        # is kept as the single global residual.
        schedule = model.structure_schedule
        offset = (schedule.residual_offset_ns if schedule.residual_offset_ns is not None
                  else calibration.offset_ns(completion_partitions) if calibration is not None else 0.0)
        # The compiler's static DMA packet train runs during launch: it overlaps the launch/drain offset
        # rather than adding to it.
        final_ns = max(makespan_dynamic + max(0.0, model.kernel_overhead_ns) + offset, structural_static_ns)
        if schedule.global_floor:
            final_ns = max(final_ns, global_completion_ns)
    result = SimulationResult(
        predicted_latency_ns=final_ns,
        timeline=timeline,
        engine_busy_ns=engine_busy,
        components_ns={
            "compute_only": compute_critical_ns,
            "compute_plus_dma": phase_critical_ns,
            "resource_overlap_makespan": makespan,
            "whole_program_routing_covered": float(
                whole_program_routing_match in {"exact", "interpolated", "clamped"}
            ),
            "whole_program_routing_clamped": float(
                whole_program_routing_match == "clamped"
            ),
            "runtime_critical_path_extension": final_ns - makespan,
            "makespan_only_ns": makespan_only_ns,
            "dma_queue_floor_ns": dma_queue_floor_ns,
            "dma_descriptor_count": dma_descriptor_count,
            "dma_fragmented_descriptor_count": dma_fragmented_descriptors,
            "onchip_transfer_count": float(
                sum(
                    1
                    for event in source_events
                    if event.get("onchip_transfer_match") is not None
                )
            ),
            "onchip_transfer_ood": float(
                sum(
                    1
                    for event in source_events
                    if event.get("onchip_transfer_match") == "ood_extrapolated"
                )
            ),
            "dma_queue_floor_ood": float(
                model.dma_elapsed_calibration is not None
                and model.dma_elapsed_calibration.fragmented_out_of_domain(
                    dma_fragmented_descriptors
                )
            ),
            "global_completion_ns": global_completion_ns,
            "carried_load_fraction": carried_fraction,
            "global_completion_activated": float(
                global_completion_ns > makespan_only_ns
            ),
            "attention_pipeline_covered": float(
                attention_pipeline_match in {"exact", "interpolated"}
            ),
            "attention_pipeline_ood": float(
                attention_pipeline_match == "ood_extrapolated"
            ),
            **{
                f"level_a_{match}_count": float(level_a_matches.get(match, 0))
                for match in (
                    "exact_key",
                    "legacy_family",
                    "family_prefix",
                    "compositional",
                    "signature",
                    "none",
                )
            },
            "vector_runtime_baseline_ns": engine_runtime_baseline_ns.get(
                ENGINE_VECTOR, 0.0
            ),
            "scalar_runtime_baseline_ns": engine_runtime_baseline_ns.get(
                ENGINE_SCALAR, 0.0
            ),
            "gpsimd_runtime_baseline_ns": engine_runtime_baseline_ns.get(
                ENGINE_GPSIMD, 0.0
            ),
            "dma_surface_ood_count": float(
                dma_surface_matches.get("ood_clamped", 0)
            ),
            "dma_surface_interpolated_count": float(
                dma_surface_matches.get("interpolated", 0)
            ),
            "dma_surface_exact_count": float(
                dma_surface_matches.get("exact", 0)
            ),
            "dma_surface_max_log_distance": dma_surface_max_log_distance,
            "tensor_flops_domain_ood_count": float(tensor_domain_ood),
            "tensor_source_geometry_ood_count": float(tensor_geometry_ood),
            "micro_dag_vector_covered": float(
                ENGINE_VECTOR in micro_dag_engine_coverage
            ),
            "micro_dag_scalar_covered": float(
                ENGINE_SCALAR in micro_dag_engine_coverage
            ),
            "micro_dag_gpsimd_covered": float(
                ENGINE_GPSIMD in micro_dag_engine_coverage
            ),
            "micro_dag_tensor_covered": float(
                ENGINE_TENSOR in micro_dag_engine_coverage
            ),
            "micro_dag_static_dma_covered": float(
                ENGINE_STATIC_DMA in micro_dag_engine_coverage
            ),
            "micro_dag_unsupported_engine_events": float(
                micro_dag_unsupported_engine_events
            ),
            "micro_dag_source_region_count": float(len(source_compute_regions)),
            "micro_dag_exact_region_count": float(len(exact_micro_dag_regions)),
            "micro_dag_all_regions_covered": float(complete_micro_dag_coverage),
            "micro_dag_timing_exact_count": float(
                micro_dag_timing_matches.get("exact", 0)
            ),
            "micro_dag_timing_interpolated_count": float(
                micro_dag_timing_matches.get("interpolated", 0)
            ),
            "micro_dag_timing_aggregate_count": float(
                micro_dag_timing_matches.get("aggregate_calibration", 0)
            ),
            "static_dma_dependency_unknown": float(structural_static_ns > 0),
            **{
                f"structural_static_dma_{match}_count": float(
                    structural_static_match == match
                )
                for match in (
                    "padded_exact",
                    "structural_key",
                    "rule_sequence",
                    "none",
                )
            },
            "final": final_ns,
        },
    )
    if model.structure_schedule is not None:
        result.components_ns["structure_buffer_depth"] = float(structure_depth)
    return result


def simulate_jsonl(
    path: str | Path, cost_model: CostModel | None = None
) -> SimulationResult:
    """Load a JSONL trace file and simulate it."""
    events = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return simulate(events, cost_model=cost_model)


def main() -> None:  # pragma: no cover - thin CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trace", help="Path to a JSONL trace produced by nki_trace_dump"
    )
    parser.add_argument("--dma-calibration-csv", type=Path, default=None)
    parser.add_argument("--dma-write-calibration-csv", type=Path, default=None)
    parser.add_argument("--dma-transpose-calibration-csv", type=Path, default=None)
    parser.add_argument("--compute-calibration-csv", type=Path, default=None)
    parser.add_argument(
        "--kernel-overhead-us",
        type=float,
        default=0.0,
        help="Optional fixed end-to-end kernel overhead; engine busy times are unchanged.",
    )
    args = parser.parse_args()
    calibration = (
        DmaCalibrationSurface.from_csv(args.dma_calibration_csv)
        if args.dma_calibration_csv
        else None
    )
    write_calibration = (
        DmaCalibrationSurface.from_csv(
            args.dma_write_calibration_csv,
            "dma_write_partition_surface",
            "derived.write_gbps_dynamic_dma_active",
            required_repeat=16,
        )
        if args.dma_write_calibration_csv
        else None
    )
    transpose_calibration = (
        DmaCalibrationSurface.from_csv(
            args.dma_transpose_calibration_csv,
            "dma_transpose_surface",
            "derived.read_gbps_dynamic_dma_active",
        )
        if args.dma_transpose_calibration_csv
        else None
    )
    compute_calibration = (
        ComputeCalibration.from_csv(args.compute_calibration_csv)
        if args.compute_calibration_csv
        else None
    )
    result = simulate_jsonl(
        args.trace,
        cost_model=CostModel(
            dma_calibration=calibration,
            dma_write_calibration=write_calibration,
            dma_transpose_calibration=transpose_calibration,
            compute_calibration=compute_calibration,
            kernel_overhead_ns=max(0.0, args.kernel_overhead_us * 1000.0),
        ),
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
