"""Structure-aware issue order for the NKI cost model (opt-in, ``CostModel.structure_schedule``).

The interpreter trace has two properties the compiled program does not have:

* **Unlimited renaming.** A tile allocated inside a loop body is a fresh array in every iteration, so
  the scheduler sees no hazard between iterations. The compiler rotates a bounded number of
  physical buffers per allocation site.
* **Program-order DMA issue.** The scheduler places events in trace order, so the load of iteration
  i+1 waits behind the store of iteration i. The compiler issues a load as soon as its dependencies
  and its destination buffer allow (prefetch).

This pass rewrites the event list before scheduling with two general rules, and no per-pattern keys:

1. **Bounded renaming.** Every fresh on-chip storage maps to ``(allocation site, k mod depth)``. The site is
   the program point of the storage's first writer (``source_site``, or an operation signature when
   the trace has no source locations), and k is the order of that site's storages. Versions of
   renamed storages are dropped, so hazards follow the physical buffer. A storage that one site writes
   more than once is a persistent buffer allocated outside the loop (a carried buffer, or one half of a
   manual ping-pong). It keeps its identity, and its bytes are subtracted from the capacity.
2. **Dependency-ordered loads.** Each HBM->on-chip load moves before preceding events that neither
   write its source range nor read or write its destination range. Loads keep their FIFO order.

``depth`` is a fixed integer, or ``None`` to derive it from SBUF capacity:
``floor(sbuf_partition_bytes / live bytes per partition of one iteration)``, where an iteration's live
bytes are the per-partition bytes of one buffer per on-chip site. No parameter is fitted.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

SBUF_PARTITION_BYTES = 192 * 1024


@dataclass(frozen=True)
class StructureSchedule:
    depth: int | None = None                    # None: derived from SBUF capacity
    sbuf_partition_bytes: int = SBUF_PARTITION_BYTES
    hoist_loads: bool = True
    schedule_completion: bool = False           # final latency = makespan + calibrated launch/drain offset
    residual_offset_ns: float | None = None     # offset jointly calibrated with the DMA latency (None: the global one)
    global_floor: bool = False                  # keep the structure-free global completion as a lower bound


def _site(event: dict[str, Any]) -> Any:
    site = event.get("source_site")
    if site is not None:
        return site
    shape = event.get("dst_offsets_shape") if event.get("op") == "transfer" else event.get("output_shape")
    return ("sig", event.get("op"), event.get("api_op"), event.get("mem_src"), tuple(shape or ()))


def _written(event: dict[str, Any]) -> Any:
    return event.get("dst_storage") if event.get("op") == "transfer" else event.get("output_storage")


def _bytes_per_partition(event: dict[str, Any]) -> float:
    if event.get("op") == "transfer":
        return float(event.get("free_bytes_per_partition") or 0.0)
    rng = event.get("output_range")
    shape = event.get("output_shape") or []
    if not rng or not shape:
        return 0.0
    return (float(rng[1]) - float(rng[0])) / max(1, int(shape[0]))


def _hbm_storages(events: list[dict[str, Any]]) -> set[int]:
    hbm: set[int] = set()
    for e in events:
        if e.get("op") != "transfer":
            continue
        if e.get("mem_src") == "hbm" and e.get("src_storage") is not None:
            hbm.add(int(e["src_storage"]))
        if e.get("mem_dst") == "hbm" and e.get("dst_storage") is not None:
            hbm.add(int(e["dst_storage"]))
    return hbm


def _persistent(events: list[dict[str, Any]], hbm: set[int]) -> set[int]:
    writes: dict[tuple[int, Any], int] = defaultdict(int)
    for e in events:
        st = _written(e)
        if st is not None and int(st) not in hbm:
            writes[(int(st), _site(e))] += 1
    return {st for (st, _site_key), n in writes.items() if n >= 2}


def buffering_depth(events: list[dict[str, Any]], config: StructureSchedule) -> int:
    if config.depth is not None:
        return max(1, int(config.depth))
    hbm = _hbm_storages(events)
    persistent = _persistent(events, hbm)
    per_site: dict[Any, float] = {}
    fixed: dict[int, float] = {}
    for e in events:
        st = _written(e)
        if st is None or int(st) in hbm:
            continue
        if int(st) in persistent:
            fixed[int(st)] = max(fixed.get(int(st), 0.0), _bytes_per_partition(e))
            continue
        site = _site(e)
        per_site[site] = max(per_site.get(site, 0.0), _bytes_per_partition(e))
    live = sum(per_site.values())
    capacity = max(0.0, config.sbuf_partition_bytes - sum(fixed.values()))
    return max(1, int(capacity // live)) if live > 0 else 1


def rename(events: list[dict[str, Any]], depth: int) -> list[dict[str, Any]]:
    hbm = _hbm_storages(events)
    persistent = _persistent(events, hbm)
    order: dict[Any, int] = defaultdict(int)
    physical: dict[int, int] = {}
    for e in events:
        st = _written(e)
        if st is None or int(st) in hbm or int(st) in physical or int(st) in persistent:
            continue
        site = _site(e)
        k = order[site]
        order[site] += 1
        physical[int(st)] = (hash(("structure_schedule", site, k % depth)) & 0x3FFFFFFFFFFF) | (1 << 46)
    out = []
    for e in events:
        e = dict(e)
        for key, version in (("src_storage", "src_version"), ("dst_storage", "dst_version")):
            if e.get(key) is not None and int(e[key]) in physical:
                e[key] = physical[int(e[key])]
                e.pop(version, None)
        storages = e.get("input_storages")
        if storages:
            versions = list(e.get("input_versions") or [None] * len(storages))
            e["input_storages"] = [physical.get(int(s), s) for s in storages]
            e["input_versions"] = [None if int(s) in physical else v for s, v in zip(storages, versions)]
        if e.get("output_storage") is not None and int(e["output_storage"]) in physical:
            e["output_storage"] = physical[int(e["output_storage"])]
            e["output_version"] = None
        out.append(e)
    return out


def _accesses(event: dict[str, Any], kind: str) -> list[tuple[Any, list]]:
    if event.get("op") == "transfer":
        storage = event.get("src_storage" if kind == "read" else "dst_storage")
        ranges = event.get("src_ranges" if kind == "read" else "dst_ranges") or \
            [event.get("src_range" if kind == "read" else "dst_range")]
        return [(storage, r) for r in ranges if r]
    if kind == "read":
        return [(s, r) for s, r in zip(event.get("input_storages") or [], event.get("input_ranges") or []) if r]
    rng = event.get("output_range")
    return [(event.get("output_storage"), rng)] if rng else []


def _overlap(a: list, b: list) -> bool:
    return any(sa is not None and sa == sb and ra[0] < rb[1] and rb[0] < ra[1] for sa, ra in a for sb, rb in b)


def _is_load(event: dict[str, Any]) -> bool:
    return event.get("op") == "transfer" and event.get("mem_src") == "hbm" and event.get("mem_dst") != "hbm"


def hoist_loads(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in events:
        if not _is_load(e):
            out.append(e)
            continue
        reads, writes = _accesses(e, "read"), _accesses(e, "write")
        pos = len(out)
        while pos > 0:
            prev = out[pos - 1]
            if prev.get("op") == "grid" or _is_load(prev):
                break
            prev_writes = _accesses(prev, "write")
            if _overlap(reads, prev_writes) or _overlap(writes, prev_writes) or _overlap(writes, _accesses(prev, "read")):
                break
            pos -= 1
        out.insert(pos, e)
    return out


def apply_structure_schedule(events: list[dict[str, Any]], config: StructureSchedule) -> tuple[list[dict[str, Any]], int]:
    """Return the rewritten event list and the buffering depth used."""
    depth = buffering_depth(events, config)
    out = rename(events, depth)
    if config.hoist_loads:
        out = hoist_loads(out)
    return out, depth


def parse_spec(spec: str | None) -> StructureSchedule | None:
    """``depth|capacity[+completion][+floor][+offset=NS]`` -> StructureSchedule (None for an empty spec)."""
    if not spec:
        return None
    head, *flags = spec.split("+")
    offset = next((float(f.split("=", 1)[1]) for f in flags if f.startswith("offset=")), None)
    return StructureSchedule(depth=None if head == "capacity" else int(head),
                             schedule_completion="completion" in flags, global_floor="floor" in flags,
                             residual_offset_ns=offset)
