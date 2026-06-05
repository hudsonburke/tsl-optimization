"""Internal validators for array inputs."""

from typing import Iterable
import numpy as np


def as_1d_float_array(
    values: float | Iterable[float],
    *,
    name: str,
    positive: bool = False,
) -> np.ndarray:
    """
    Convert scalar or iterable values to a 1D float array with validation.

    Parameters:
    - values: Scalar or iterable of values to convert
    - name: Parameter name for error messages
    - positive: If True, enforce that all values are > 0

    Returns:
    - 1D numpy array of float values

    Raises:
    - ValueError: If array is not 1D, empty, contains non-finite values, or (if positive=True) non-positive values
    """
    array = np.atleast_1d(np.asarray(values, dtype=float))
    if array.ndim != 1:
        raise ValueError(
            f"{name} must be a scalar or 1D array, got shape {array.shape}"
        )
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if positive and np.any(array <= 0):
        raise ValueError(f"{name} must contain positive values, got {array}")
    return array
