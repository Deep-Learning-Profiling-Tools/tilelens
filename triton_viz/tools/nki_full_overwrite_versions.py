"""What-if schedule with two versions of fully overwritten SBUF tiles.

This is a source-trace scenario, not a prediction of compiler allocation. It
requires complete allocation metadata, dense FP32 SBUF roots, whole-root reads
and writes, and no in-place operation. It refuses unsupported traces rather than
inventing dependencies. Its capacity calculation assumes all two-version SBUF
roots coexist; compiler scratch and real lifetimes still need device checks.
"""
from copy import deepcopy

CAPACITY_PER_PARTITION = 176 * 1024


def apply(events, depth=2):
    from triton_viz.tools.nki_cost_model import _read_accesses, _write_accesses

    if depth != 2:
        raise ValueError('version scenario supports exactly two slots')
    events = list(events)
    allocations = {}
    grids = 0
    for event in events:
        op = event.get('op')
        if op == 'grid':
            grids += 1
            continue
        if op not in ('memory_fill', 'transfer', 'binary'):
            raise ValueError('version scenario has unsupported source operation')
        inputs = event.get('input_allocations')
        outputs = event.get('output_allocations')
        if not isinstance(inputs, list) or not isinstance(outputs, list):
            raise ValueError('version scenario requires allocation metadata')
        for alloc in inputs + outputs:
            if not isinstance(alloc, dict) or type(alloc.get('storage')) is not int:
                raise ValueError('version scenario has incomplete allocation metadata')
            root = alloc['storage']
            memory = alloc.get('memory')
            if memory not in ('sbuf', 'hbm'):
                raise ValueError('version scenario has unsupported memory')
            if memory == 'sbuf' and alloc.get('view_storage') != root:
                raise ValueError('version scenario does not support SBUF aliases')
            if root in allocations:
                # HBM may be addressed through several views of one root.
                left = {k: v for k, v in allocations[root].items() if k != 'view_storage'}
                right = {k: v for k, v in alloc.items() if k != 'view_storage'}
                if left != right:
                    raise ValueError('version scenario has inconsistent allocation metadata')
            else:
                allocations[root] = alloc
    if grids != 1:
        raise ValueError('version scenario requires one grid execution')
    if not any(event.get('op') == 'memory_fill' for event in events):
        raise ValueError('version scenario requires an explicit initializer')
    onchip = {root: a for root, a in allocations.items() if a['memory'] == 'sbuf'}
    if not onchip:
        raise ValueError('version scenario has no SBUF allocation')
    for root, alloc in onchip.items():
        if alloc.get('view_storage') != root:
            raise ValueError('version scenario does not support SBUF aliases')
        shape = alloc.get('shape')
        if (not isinstance(shape, list) or len(shape) != 2
                or any(type(x) is not int or x <= 0 for x in shape)
                or shape[0] > 128 or alloc.get('partition_axis') != 0
                or alloc.get('dtype') != 'float32'):
            raise ValueError('version scenario requires dense 2-D FP32 SBUF roots')
    budget = sum(2 * alloc['shape'][1] * 4 for alloc in onchip.values())
    if budget > CAPACITY_PER_PARTITION:
        raise ValueError('version scenario exceeds assumed SBUF capacity')
    base = 1 << 62
    physical = {(root, slot): base + index * depth + slot
                for index, root in enumerate(sorted(onchip)) for slot in range(depth)}
    if set(allocations) & set(physical.values()):
        raise ValueError('version scenario storage identifier collision')

    versions = {}
    counts = {root: 0 for root in onchip}
    output = []
    for event in events:
        if event.get('op') == 'grid':
            output.append(dict(event))
            continue
        if 'scheduler_duration_override_ns' in event:
            raise ValueError('version scenario does not support duration overrides')
        reads = _read_accesses(event)
        writes = _write_accesses(event)
        for key, accesses in (('input_range_lists', reads), ('output_range_lists', writes)):
            entries = event.get(key)
            if not isinstance(entries, list):
                raise ValueError('version scenario requires exact access runs')
            for storage, _version, _ptr, lo, hi in accesses:
                if storage in onchip:
                    matching = [entry for entry in entries if isinstance(entry, dict)
                                and entry.get('storage') == storage]
                    if not matching or any(entry.get('runs') != [[lo, hi]] for entry in matching):
                        raise ValueError('version scenario requires dense whole-root access runs')
        if (event.get('input_storages') is not None
                and len(event.get('input_ptrs') or []) != len(event['input_storages'])):
            raise ValueError('version scenario has mismatched input pointers')
        for storage, _version, ptr, lo, hi in reads + writes:
            if storage not in allocations:
                raise ValueError('version scenario access is not allocation-bound')
            if storage in onchip:
                if ptr != storage:
                    raise ValueError('version scenario does not support SBUF pointer aliases')
                shape = onchip[storage]['shape']
                size = shape[0] * shape[1] * 4
                if lo != 0 or hi != size:
                    raise ValueError('version scenario requires whole-root accesses')
        read_roots = {s for s, _, _, _, _ in reads if s in onchip}
        write_roots = {s for s, _, _, _, _ in writes if s in onchip}
        if read_roots & write_roots:
            raise ValueError('version scenario cannot rename in-place operations')
        if any(root not in versions for root in read_roots):
            raise ValueError('version scenario read before write')
        read_map = {root: versions[root] for root in read_roots}
        write_map = {}
        for root in write_roots:
            write_map[root] = physical[root, counts[root] % depth]
            counts[root] += 1

        result = deepcopy(event)
        for prefix, mapping in (('src', read_map), ('dst', write_map)):
            root = event.get(prefix + '_storage')
            if root in mapping:
                result[prefix + '_storage'] = mapping[root]
                result[prefix + '_ptr'] = mapping[root]
                result[prefix + '_version'] = None
        roots = event.get('input_storages') or []
        if roots:
            result['input_storages'] = [read_map.get(root, root) for root in roots]
            result['input_ptrs'] = [read_map.get(root, ptr) for root, ptr in zip(roots, event['input_ptrs'])]
            result['input_versions'] = [None] * len(roots)
        root = event.get('output_storage')
        if root in write_map:
            result['output_storage'] = write_map[root]
            result['output_ptr'] = write_map[root]
            result['output_version'] = None
            if result.get('allocation_storage') == root:
                result['allocation_storage'] = write_map[root]
        for key, mapping in (('input_range_lists', read_map), ('output_range_lists', write_map)):
            for entry in result.get(key) or []:
                if entry and entry.get('storage') in mapping:
                    entry['storage'] = mapping[entry['storage']]
        for key, mapping in (('input_allocations', read_map), ('output_allocations', write_map)):
            for alloc in result.get(key) or []:
                root = alloc['storage']
                if root in mapping:
                    alloc['storage'] = mapping[root]
                    alloc['view_storage'] = mapping[root]
        output.append(result)
        versions.update(write_map)
    return output, {
        'depth': depth,
        'sbuf_roots': len(onchip),
        'assumed_bytes_per_partition': budget,
        'writes_by_root': list(counts.values()),
        'scope': 'hypothetical full-overwrite renaming; not a compiler or device prediction',
    }
