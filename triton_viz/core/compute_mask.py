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


# Semantic classification, not a kernel- or structure-specific rule. Masking a
# reduction or a cross-element operation cannot be priced as a smaller pointwise tile.
POINTWISE_OPS = frozenset({'add', 'subtract', 'multiply', 'divide', 'maximum', 'minimum',
    'exp', 'log', 'sqrt', 'rsqrt', 'reciprocal', 'square', 'abs', 'negative',
    'tanh', 'sigmoid', 'relu', 'sin', 'cos'})


def masked_free_dim(event):
    """Return a supported rectangular pointwise extent, otherwise keep declared pricing."""
    geometry = event.get('compute_mask_geometry', {})
    if not isinstance(geometry, dict):
        return None
    if (event.get('op') != 'compute' or event.get('api_op') not in POINTWISE_OPS
            or not event.get('compute_mask_provided') or geometry.get('status') != 'known'
            or geometry.get('rectangular') is not True):
        return None
    shape, bounds = geometry.get('shape'), geometry.get('bounds')
    if (not isinstance(shape, list) or len(shape) < 2 or not isinstance(bounds, list)
            or len(bounds) != len(shape) or event.get('input_shape') != shape
            or event.get('output_shape') != shape):
        return None
    if any(not isinstance(b, list) or len(b) != 2 or any(type(x) is not int for x in b)
           or type(size) is not int or not 0 <= b[0] < b[1] <= size for b,size in zip(bounds,shape)):
        return None
    widths = [hi-lo for lo,hi in bounds]
    if int(np.prod(widths, dtype=np.int64)) != geometry.get('active_elements'):
        return None
    return int(np.prod(widths[1:], dtype=np.int64))
