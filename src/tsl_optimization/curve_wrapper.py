from collections.abc import Iterable
from typing import Any, Protocol, cast
import weakref

import numpy as np
from scipy.interpolate import interp1d

from ._validators import as_1d_float_array

class FunctionLike(Protocol):
    def calcValue(self, x: float) -> float: ...


type CurveType = FunctionLike | np.ndarray

_WRAPPER_CACHE: dict[int, tuple[weakref.ReferenceType[Any], "CurveWrapper"]] = {}


def _make_interp(
    x_data: np.ndarray,
    y_data: np.ndarray,
    *,
    kind: str,
    fill_value: Any,
) -> interp1d:
    interp_ctor = cast(Any, interp1d)
    return cast(
        interp1d,
        interp_ctor(
            x_data,
            y_data,
            kind=kind,
            bounds_error=False,
            fill_value=fill_value,
        ),
    )


def _extract_increasing_segment(
    x_data: np.ndarray, y_data: np.ndarray, *, tolerance: float = 1e-12
) -> tuple[np.ndarray, np.ndarray] | None:
    dy = np.diff(y_data)
    increasing = dy > tolerance
    if not np.any(increasing):
        return None

    first_inc = int(np.argmax(increasing))
    x_mono = [float(x_data[first_inc])]
    y_mono = [float(y_data[first_inc])]
    for index in range(first_inc + 1, len(x_data)):
        if y_data[index] > y_mono[-1] + tolerance:
            x_mono.append(float(x_data[index]))
            y_mono.append(float(y_data[index]))

    if len(x_mono) < 2:
        return None

    return np.asarray(x_mono), np.asarray(y_mono)


def _build_inverse_interp(
    x_data: np.ndarray, y_data: np.ndarray
) -> tuple[interp1d, float, float] | None:
    y_diff = np.diff(y_data)
    if np.all(y_diff > 0):
        y_inverse = y_data
        x_inverse = x_data
    elif np.all(y_diff < 0):
        y_inverse = y_data[::-1]
        x_inverse = x_data[::-1]
    else:
        return None

    return (
        _make_interp(
            y_inverse,
            x_inverse,
            kind="linear",
            fill_value=(float(x_inverse[0]), float(x_inverse[-1])),
        ),
        float(np.min(y_data)),
        float(np.max(y_data)),
    )


def _build_inverse_data(
    x_data: np.ndarray, y_data: np.ndarray
) -> tuple[interp1d, float, float] | None:
    inverse_data = _build_inverse_interp(x_data, y_data)
    if inverse_data is not None:
        return inverse_data

    y_diff = np.diff(y_data)
    if not (np.all(y_diff >= 0) and np.any(y_diff > 0)):
        return None

    inverse_segment = _extract_increasing_segment(x_data, y_data)
    if inverse_segment is None:
        return None

    x_mono, y_mono = inverse_segment
    return (
        _make_interp(
            y_mono,
            x_mono,
            kind="linear",
            fill_value=(float(x_mono[0]), float(x_mono[-1])),
        ),
        float(y_mono[0]),
        float(y_mono[-1]),
    )


def _validate_numpy_curve(curve: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if curve.ndim != 2 or curve.shape[0] != 2:
        raise ValueError(f"Expected numpy curve with shape (2, n), got {curve.shape}")
    if curve.shape[1] < 2:
        raise ValueError("Curve must contain at least two sample points")
    if not np.all(np.isfinite(curve)):
        raise ValueError("Curve samples must contain only finite values")

    x_data = np.array(curve[0, :], dtype=float, copy=True)
    y_data = np.array(curve[1, :], dtype=float, copy=True)
    if not np.all(np.diff(x_data) > 0):
        raise ValueError("Curve x-values must be strictly increasing")

    return x_data, y_data


def _get_cached_wrapper(curve: CurveType) -> "CurveWrapper":
    cache_key = id(curve)
    cached = _WRAPPER_CACHE.get(cache_key)
    if cached is not None:
        curve_ref, wrapper = cached
        if curve_ref() is curve:
            return wrapper
        _WRAPPER_CACHE.pop(cache_key, None)

    wrapper = CurveWrapper(curve)
    try:
        curve_ref = weakref.ref(
            curve, lambda _ref, key=cache_key: _WRAPPER_CACHE.pop(key, None)
        )
    except TypeError:
        return wrapper

    _WRAPPER_CACHE[cache_key] = (curve_ref, wrapper)
    return wrapper


class CurveWrapper:
    """
    Wrapper class for OpenSim Functions and numpy array curves.
    Provides unified interface with fast interpolation-based inverse.

    For OpenSim curves, pre-samples the curve densely on construction
    to build scipy interpolants for both forward and inverse evaluation.
    This avoids expensive per-point calcValue calls during optimization.
    """

    def __init__(
        self,
        curve: CurveType,
        interp_type: str = "linear",
        x_range: tuple[float, float] = (-0.5, 3.0),
        n_samples: int = 5000,
    ):
        if n_samples < 2:
            raise ValueError(f"n_samples must be at least 2, got {n_samples}")
        if x_range[0] >= x_range[1]:
            raise ValueError(f"x_range must be increasing, got {x_range}")

        try:
            self.curve = weakref.proxy(curve)
        except TypeError:
            self.curve = curve
        self._inverse_interp: interp1d | None = None
        self._y_min = float("nan")
        self._y_max = float("nan")

        calc_value = getattr(cast(Any, curve), "calcValue", None)
        self.is_opensim = callable(calc_value)

        if self.is_opensim:
            x_data = np.linspace(x_range[0], x_range[1], n_samples)
            opensim_curve = cast(Any, curve)
            y_data = np.asarray(
                [opensim_curve.calcValue(float(x)) for x in x_data], dtype=float
            )
            if not np.all(np.isfinite(y_data)):
                raise ValueError("OpenSim curve sampling produced non-finite values")

            self._x_min = float(x_data[0])
            self._x_max = float(x_data[-1])
            self._forward_interp = _make_interp(
                x_data,
                y_data,
                kind=interp_type,
                fill_value=(float(y_data[0]), float(y_data[-1])),
            )

            inverse_data = _build_inverse_data(x_data, y_data)
            if inverse_data is not None:
                self._inverse_interp, self._y_min, self._y_max = inverse_data
        else:
            if not isinstance(curve, np.ndarray):
                raise TypeError(
                    "curve must be an OpenSim Function-like object or a numpy array"
                )

            x_data, y_data = _validate_numpy_curve(curve)
            self._x_min = float(x_data[0])
            self._x_max = float(x_data[-1])
            self._forward_interp = _make_interp(
                x_data,
                y_data,
                kind=interp_type,
                fill_value=(float(y_data[0]), float(y_data[-1])),
            )

            inverse_data = _build_inverse_data(x_data, y_data)
            if inverse_data is not None:
                self._inverse_interp, self._y_min, self._y_max = inverse_data

    def evaluate(self, points: float | Iterable[float]) -> np.ndarray:
        """Evaluate curve at given points."""
        points_array = as_1d_float_array(points, name="points")
        points_array = np.clip(points_array, self._x_min, self._x_max)
        return np.asarray(self._forward_interp(points_array), dtype=float)

    def evaluate_inverse(self, values: float | Iterable[float]) -> np.ndarray:
        """Evaluate inverse of curve at given values."""
        if self._inverse_interp is None:
            raise ValueError(
                "Curve inverse is undefined for non-monotonic or flat curves"
            )

        values_array = as_1d_float_array(values, name="values")
        values_array = np.clip(values_array, self._y_min, self._y_max)
        return np.asarray(self._inverse_interp(values_array), dtype=float)


def evaluate_curve(
    curve: CurveType | CurveWrapper, points: float | np.ndarray, inverse: bool = False
) -> np.ndarray:
    """
    Evaluate a curve at given points with optional caching via CurveWrapper.

    Args:
        curve: OpenSim Function, numpy array [2 x n], or CurveWrapper
        points: Points at which to evaluate the curve
        inverse: Whether to evaluate inverse of the curve
    """
    if isinstance(curve, CurveWrapper):
        return curve.evaluate_inverse(points) if inverse else curve.evaluate(points)

    wrapper = _get_cached_wrapper(curve)
    return wrapper.evaluate_inverse(points) if inverse else wrapper.evaluate(points)
