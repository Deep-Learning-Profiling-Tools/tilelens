"""Compact source mask facts; declared tile shapes and pricing stay separate."""
import numpy as np


def compute_mask_geometry(mask, shape):
    """Describe exact activity and its box without treating a box as a dense mask."""
    try:
        active = np.broadcast_to(np.asarray(mask, dtype=bool), tuple(shape))
    except (ValueError, TypeError):
        return {"status": "unsupported_broadcast", "shape": list(shape)}
    count = int(np.count_nonzero(active))
    if not count:
        return {"status": "empty", "shape": list(shape), "active_elements": 0}
    bounds = []
    for axis in range(active.ndim):
        other = tuple(i for i in range(active.ndim) if i != axis)
        positions = np.flatnonzero(np.any(active, axis=other) if other else active)
        bounds.append([int(positions[0]), int(positions[-1]) + 1])
    volume = int(np.prod([hi-lo for lo,hi in bounds], dtype=np.int64))
    return {"status": "known", "shape": list(shape), "active_elements": count,
            "bounds": bounds, "rectangular": volume == count}
