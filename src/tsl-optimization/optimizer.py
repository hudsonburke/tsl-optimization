import numpy as np
from scipy.optimize import minimize
from typing import Iterable, Callable

from .curve_wrapper import CurveType
from .muscle_parameters import calc_tsl

def ssdp(data: Iterable) -> float:
    # Calculate error equal to the sum of squared differences between every element of the vector
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D array-like structure")
    diff_matrix = np.tile(data, (len(data), 1))
    difference = diff_matrix - diff_matrix.T
    err = np.sum(difference**2)
    return float(err)

def ssd(data: Iterable) -> float:
    # Sum of squared differences between slack lengths and the mean slack length
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D array-like structure")
    return float(np.sum((data - np.mean(data))**2))

def optimize_fiber_length(
    lmt: Iterable[float],
    lm_opt: float,
    alpha_opt: float,
    afl: CurveType,
    pfl: CurveType,
    tfl: CurveType,
    lm_norm_range: tuple[float, float] = (0.5, 1.5),
    method: str | None = None, # 'SLSQP',
    objective: Callable[[Iterable], float] = ssdp
    ) -> np.ndarray:
    
    """
    Optimize the fiber length of a muscle using the Manal 2004 model. 
    This can be used to find the tendon slack length (TSL) that minimizes the error between the calculated and expected muscle-tendon lengths. 

    Parameters:
    - lmt: Muscle-tendon lengths.
    - lm_opt: Optimal muscle fiber length.
    - alpha_opt: Optimal pennation angle.
    - afl: Active force length curve.
    - pfl: Passive force length curve.
    - tfl: Tendon force length curve.
    - lm_norm_range: Range for normalized muscle fiber lengths (default is (0.5, 1.5)).
    - method: Optimization method to use (optional).
    - objective: Objective function to minimize (default is sum of squared differences between pairs).
    
    Returns:
    - Optimized fiber lengths as a numpy array.
    """
    lmt = np.asarray(lmt)
    # Initial guess for fiber lengths
    lm0 = np.linspace(lm_norm_range[0], lm_norm_range[1], len(lmt)) * lm_opt
    lb = np.full_like(lm0, lm_opt*lm_norm_range[0], dtype=float)
    ub = np.full_like(lm0, lm_opt*lm_norm_range[1], dtype=float)
    
    # Minimization function
    # func = lambda lm: objective(calc_tsl(lmt, lm, lm_opt, alpha_opt, afl, pfl, tfl))

    result = minimize(lambda lm: objective(calc_tsl(lmt, lm, lm_opt, alpha_opt, afl, pfl, tfl)), lm0, bounds=list(zip(lb, ub)), method=method)
    if not result.success:
        raise RuntimeError("Optimization failed")
    
    return result.x  # Return the optimized fiber lengths