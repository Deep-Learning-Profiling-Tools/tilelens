import numpy as np

get = lambda x, default: default if x is None else x


def normalize_slice(ndarray: np.ndarray, keys: tuple) -> tuple[tuple, tuple]:
    """Separate singleton dims (None) and add bounds to slices like :2, 2:, or :"""
    singleton_dims = []
    dim_idx = 0
    new_keys = []
    for key_dim, k in enumerate(keys):
        if k is None:
            singleton_dims.append(key_dim)
            continue

        arr_dim = ndarray.shape[dim_idx]
        if isinstance(k, slice):
            # add bounds to [:, :N, N:]-type slices
            start = get(k.start, 0)
            stop = get(k.stop, arr_dim)
            step = get(k.step, 1)
            if start < 0:
                start = max(0, arr_dim + start)
            if stop < 0:
                stop = max(0, arr_dim + stop)
            k = slice(start, stop, step)

        new_keys.append(k)
        dim_idx += 1
    return tuple(new_keys), tuple(singleton_dims)


def _calculate_target_shape(
    keys: tuple, original_shape: tuple[int, ...]
) -> tuple[int, ...]:
    """Calculate the shape needed for the padded array to handle indexing."""
    target_shape = list(original_shape)
    for i, (arr_dim, key) in enumerate(zip(original_shape, keys)):
        if isinstance(key, slice):
            target_shape[i] = max(arr_dim, key.stop)
        elif isinstance(key, (int, np.integer)):
            target_shape[i] = max(arr_dim, int(key) + 1)
        elif isinstance(key, (list, np.ndarray)):
            target_shape[i] = max(arr_dim, np.max(key) + 1)
    return tuple(target_shape)


def _get_valid_indices(
    keys: tuple, original_shape: tuple[int, ...], result_shape: tuple[int, ...]
) -> np.ndarray:
    """Get a boolean mask indicating which result indices are within original array bounds."""
    coords = np.mgrid[[slice(0, s) for s in result_shape]]
    valid_mask = np.ones(result_shape, dtype=bool)

    for key_idx, (arr_dim, key) in enumerate(zip(original_shape, keys)):
        if isinstance(key, slice):
            start = get(key.start, 0)
            step = get(key.step, 1)
            original_coords = coords[key_idx] * step + start
            valid_mask &= (0 <= original_coords) & (original_coords < arr_dim)
        elif isinstance(key, (list, np.ndarray)):
            valid_indices = np.array(key)
            valid_mask &= (0 <= valid_indices) & (valid_indices < arr_dim)
        elif isinstance(key, (int, np.integer)):
            valid_mask &= 0 <= int(key) < arr_dim

    return valid_mask


def tile_shape(ndarray: np.ndarray, keys) -> tuple:
    """Shape of ``ndarray[keys]``, also when some integer-array keys are out of bounds."""
    if not isinstance(keys, tuple):
        keys = (keys,)
    clipped = []
    for dim, key in enumerate(keys):
        if (
            isinstance(key, np.ndarray)
            and key.dtype.kind in "iu"
            and dim < ndarray.ndim
        ):
            key = np.clip(key, 0, max(ndarray.shape[dim] - 1, 0))
        clipped.append(key)
    return ndarray[tuple(clipped)].shape


def tile_mask(ndarray: np.ndarray, keys, mask) -> np.ndarray | None:
    """``mask`` broadcast to the shape of ``ndarray[keys]``, as NKI broadcasts masks;
    None stays None."""
    if mask is None:
        return None
    return np.broadcast_to(np.asarray(mask, dtype=bool), tile_shape(ndarray, keys))


def masked_load(
    ndarray: np.ndarray, keys: tuple, mask: np.ndarray = None
) -> np.ndarray:
    """
    Load array elements with masking for out-of-bounds errors.

    Args:
        ndarray: Input numpy array
        keys: Indexing keys (tuple of slices, integers, arrays, etc.)
        mask: Boolean mask array. If None, returns direct indexing.
              Where mask is False at OOB indices, garbage values are kept.
              Mask shape should match the shape of the result, not the input array.

    Returns:
        Indexed array with masked error handling
    """
    try:  # fast path case - if keys aren't OOB, just go with that
        out = ndarray[keys].copy()
        if mask is None:
            return out
        out[~mask] = np.iinfo(ndarray.dtype).max
        return out
    except Exception:
        pass

    # Convert keys to tuple if it's not already
    if not isinstance(keys, tuple):
        keys = (keys,)

    # normalize (remove Nones in slices)
    keys, singleton_dims = normalize_slice(ndarray, keys)
    target_shape = _calculate_target_shape(keys, ndarray.shape)
    padded_array = np.empty(target_shape, dtype=ndarray.dtype)

    # Calculate the region where we can safely copy the original array
    copy_slices = []
    for i in range(len(ndarray.shape)):
        copy_slices.append(slice(0, ndarray.shape[i]))
    padded_array[tuple(copy_slices)] = ndarray[tuple(copy_slices)]

    result = padded_array[keys]
    if mask is None:  # every lane is active, so an out-of-bounds one is an error
        mask = np.ones(np.expand_dims(result, singleton_dims).shape, dtype=bool)
    assert np.expand_dims(result, singleton_dims).shape == mask.shape

    mask = np.squeeze(mask, singleton_dims)

    # Determine which indices are actually within original array bounds
    in_bounds_mask = _get_valid_indices(keys, ndarray.shape, result.shape)

    # Check if there are any OOB indices where mask=True
    oob_with_true_mask = (~in_bounds_mask) & mask
    oob_coords = np.where(oob_with_true_mask)
    if len(oob_coords[0]) > 0:
        # Get the first OOB coordinate
        oob_idx = tuple(coord[0] for coord in oob_coords)
        raise IndexError(
            f"index {oob_idx} is out of bounds for array of size {ndarray.shape}"
        )

    valid_mask = mask & in_bounds_mask
    # Use appropriate max value based on dtype
    if np.issubdtype(ndarray.dtype, np.integer):
        result[~valid_mask] = np.iinfo(ndarray.dtype).max
    elif np.issubdtype(ndarray.dtype, np.floating):
        result[~valid_mask] = np.finfo(ndarray.dtype).max
    else:
        result[~valid_mask] = 0  # false

    return np.expand_dims(result, singleton_dims)


def masked_store(
    ndarray: np.ndarray, keys: tuple, value: np.ndarray, mask: np.ndarray = None
) -> None:
    """
    Store array elements with masking for out-of-bounds errors.
    General idea of this procedure is to get all the indices that the slice + mask needs for a single advanced indexing assign.
    - to handle OOB accesses we infer the min possible array size where array[keys] wouldn't have any OOBs
    - we can get the indices for each dim by selecting from an mgrid

    Args:
        ndarray: Input numpy array to store values into
        keys: Indexing keys (tuple of slices, integers, arrays, etc.)
        value: Values to store
        mask: Boolean mask array. If None, performs direct indexing.
              Where mask is False at OOB indices, values are not stored.
              Mask shape should match the shape of the result, not the input array.

    Returns:
        None (modifies ndarray in-place)
    """
    # Handle mask=None case
    if mask is None:
        ndarray[keys] = value
        return

    assert value.shape == mask.shape

    # Convert keys to tuple if it's not already
    if not isinstance(keys, tuple):
        keys = (keys,)

    flat_mask = mask.ravel()  # can only index with bool tensors if they're 1d

    # normalize (remove Nones in slices)
    keys, singleton_dims = normalize_slice(ndarray, keys)
    target_shape = _calculate_target_shape(keys, ndarray.shape)
    mgrid = np.mgrid[tuple(slice(0, dim, 1) for dim in target_shape)]
    idxs = []
    for mgrid_dim in mgrid:
        # get the section that would be sliced if the array was big enough
        sliced_section = mgrid_dim[keys]

        # get just the section where mask=True
        dim_idxs = sliced_section.ravel()[flat_mask]
        idxs.append(dim_idxs)
    ndarray[tuple(idxs)] = value.ravel()[flat_mask]
