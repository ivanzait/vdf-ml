import numpy as np


def iter_array_batches(X, indices=None, batch_size=64):
    """Yield (batch_indices, batch) over X, batch_size samples at a time."""

    if indices is None:
        indices = np.arange(X.shape[0])

    for batch_indices in iter_index_batches(indices, batch_size):
        yield batch_indices, get_array_batch(X, batch_indices)


def iter_index_batches(indices, batch_size):
    """Yield indices in chunks of at most batch_size."""

    indices = np.asarray(indices)
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


def get_array_batch(X, batch_indices):
    """Select batch_indices from X, using a slice (no copy) when they're contiguous and increasing."""

    batch_slice = create_contiguous_slice(batch_indices)
    if batch_slice is not None:
        return X[batch_slice]

    return X[batch_indices]


def create_contiguous_slice(indices):
    """A slice equivalent to indices, or None when indices aren't contiguous and increasing."""

    indices = np.asarray(indices)
    if len(indices) == 0:
        return slice(0, 0)

    if not np.issubdtype(indices.dtype, np.integer):
        return None

    start = int(indices[0])
    stop = start + len(indices)

    if start < 0 or int(indices[-1]) != stop - 1:
        return None

    if np.all(indices == np.arange(start, stop)):
        return slice(start, stop)

    return None
