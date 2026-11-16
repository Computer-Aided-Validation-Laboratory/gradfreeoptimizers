"""Helpers shared by the example problems: accept one point or a batch."""

import numpy as np

SINGLE_POINT_NDIM = 1
BATCH_NDIM = 2


def as_batch(x) -> tuple[np.ndarray, bool]:
    """Return ``x`` as an ``(n, d)`` float array, and whether it was one point.

    Parameters
    ----------
    x : array_like
        One point of shape ``(d,)`` or a batch of shape ``(n, d)``.

    Returns
    -------
    (np.ndarray, bool)
        The ``(n, d)`` array and True if ``x`` was a single point.
    """
    array = np.asarray(x, dtype=float)
    if array.ndim == SINGLE_POINT_NDIM:
        return array[np.newaxis, :], True
    if array.ndim == BATCH_NDIM:
        return array, False
    raise ValueError(f"Expected x with shape (d,) or X with shape (n, d), got {array.shape}")


def finish_batch(values: np.ndarray, single: bool) -> np.ndarray | float:
    """Return a float for a single point, the ``(n,)`` array otherwise."""
    if single:
        return float(values[0])
    return values
