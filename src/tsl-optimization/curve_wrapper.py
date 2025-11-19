import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from pyopensim.common import Function
from typing import Iterable

type CurveType = Function | np.ndarray

class CurveWrapper:
    """
    Wrapper class for OpenSim Functions and numpy array curves with caching.
    Provides unified interface and caches expensive inverse calculations.
    """

    def __init__(self, curve: CurveType, interp_type: str = 'cubic'):
        self.curve = curve
        self.is_opensim = isinstance(curve, Function)
        self._inverse_cache: dict[float, float] = {}
        
        if not self.is_opensim:
            # Pre-compute interpolation functions for numpy arrays
            # At this point we know curve is np.ndarray, not Function
            assert isinstance(curve, np.ndarray), "Expected numpy array"
            x_data, y_data = curve[0, :], curve[1, :]
            self._forward_interp = interp1d(x_data, y_data, kind=interp_type,
                                          bounds_error=False, fill_value=np.nan)
            self._inverse_interp = interp1d(y_data, x_data, kind=interp_type,
                                          bounds_error=False, fill_value=np.nan)

    def evaluate(self, points: float | Iterable[float]) -> np.ndarray:
        """Evaluate curve at given points."""
        points = np.asarray(points)
        points = np.atleast_1d(points)
        
        if self.is_opensim:
            opensim_curve = self.curve
            result = np.array([opensim_curve.calcValue(float(x)) for x in points])  # type: ignore
        else:
            result = self._forward_interp(points)

        return result

    def evaluate_inverse(self, values: float | Iterable[float]) -> np.ndarray:
        """Evaluate inverse of curve at given values with caching."""
        values = np.asarray(values)
        values = np.atleast_1d(values)
        
        if self.is_opensim:
            # Use caching for expensive OpenSim inverse calculations
            result = np.array([self._cached_inverse(float(v)) for v in values])
        else:
            result = self._inverse_interp(values)
        
        return result
    
    def _cached_inverse(self, value: float, tolerance: float = 1e-6) -> float:
        """Cached inverse calculation for OpenSim functions."""
        # Check cache with tolerance
        for cached_val, cached_result in self._inverse_cache.items():
            if abs(cached_val - value) < tolerance:
                return cached_result
        
        # Calculate new inverse
        opensim_curve = self.curve
        def objective(x):
            return (opensim_curve.calcValue(float(x)) - value) ** 2  # type: ignore
        
        result = minimize(objective, x0=0, method='BFGS')
        inverse_val = float(result.x[0])
        
        # Cache the result
        self._inverse_cache[value] = inverse_val
        
        # Limit cache size to prevent memory issues
        if len(self._inverse_cache) > 1000:
            # Remove oldest entries (simple FIFO)
            old_keys = list(self._inverse_cache.keys())[:100]
            for key in old_keys:
                del self._inverse_cache[key]
        
        return inverse_val

def evaluate_curve(curve: CurveType | CurveWrapper, 
                 points: float | np.ndarray, 
                 inverse: bool = False) -> np.ndarray:
    """
    Evaluate a curve at given points with optional caching via CurveWrapper.
    
    Args:
        curve: OpenSim Function, numpy array [2 x n], or CurveWrapper
        points: Points at which to evaluate the curve
        inverse: Whether to evaluate inverse of the curve
    """
    if isinstance(curve, CurveWrapper):
        # If already a CurveWrapper, use its methods directly
        return curve.evaluate_inverse(points) if inverse else curve.evaluate(points)
    # Backward compatibility - create temporary wrapper
    wrapper = CurveWrapper(curve)
    return wrapper.evaluate_inverse(points) if inverse else wrapper.evaluate(points)