import numpy as np
from typing import Iterable

from .curve_wrapper import CurveType, evaluate_curve

def calc_pennation(
    lm_norm: Iterable[float] | float, 
    lm_opt: float, 
    alpha_opt: float
    ) -> np.ndarray:
    """
    Calculate pennation angle with vectorized operations.
    
    Parameters:
    - lm_norm: Normalized muscle fiber lengths (can be a single value or an iterable).
    - lm_opt: Optimal muscle fiber length.
    - alpha_opt: Optimal pennation angle (in radians).
    Returns:
    - Pennation angles in radians, with values clipped between 0 and pi/2.
    """
    # Validate inputs
    if not (0 <= alpha_opt < np.pi/2):
        raise ValueError(f"alpha_opt must be between 0 and pi/2, got {alpha_opt}")
    if lm_opt <= 0:
        raise ValueError(f"lm_opt must be positive, got {lm_opt}")

    lm_norm = np.atleast_1d(np.asarray(lm_norm))
    if np.any(lm_norm <= 0):
        raise ValueError(f"All values in lm_norm must be positive, got {lm_norm}")

    # Calculate the muscle fiber lengths
    lm = lm_norm * lm_opt
    
    # Handle zero/near-zero cases - return pi/2 for very small values
    near_zero = lm < 1e-6
    
    # Calculate pennation angles
    l = np.clip(lm_opt * np.sin(alpha_opt) / np.where(near_zero, 1, lm), 0, 1) # sin(alpha) = width / fiber_length
    alpha = np.clip(np.arcsin(l), 0, np.pi/2)
    
    # Set near-zero cases to pi/2
    alpha[near_zero] = np.pi/2
    
    return alpha

def calc_tsl(
    lmt: Iterable[float] | float, 
    lm: Iterable[float] | float,
    lm_opt: float, 
    alpha_opt: float, 
    afl: CurveType, 
    pfl: CurveType, 
    tfl: CurveType
    ) -> np.ndarray:
    """
    Calculate the slack length of a muscle
    
    Parameters:
    - lmt: Muscle-tendon lengths.
    - lm: Muscle fiber lengths.
    - lm_opt: Optimal muscle fiber length.
    - alpha_opt: Optimal pennation angle.
    - afl: Active force length curve.
    - pfl: Passive force length curve.
    - tfl: Tendon force length curve.
    """
    if lm_opt <= 0:
        raise ValueError(f"lm_opt must be positive, got {lm_opt}")
    if alpha_opt < 0 or alpha_opt >= np.pi/2:
        raise ValueError(f"alpha_opt must be between 0 and pi/2, got {alpha_opt}")
    lmt = np.asarray(lmt)
    if np.any(lmt <= 0):
        raise ValueError(f"lmt must contain positive values, got {lmt}")
    lm = np.asarray(lm)
    if np.any(lm <= 0):
        raise ValueError(f"lm must contain positive values, got {lm}")
    
    # Ensure lmt and lm have the same length
    if len(lmt) != len(lm):
        raise ValueError(f"lmt and lm must have the same length, got lmt: {len(lmt)}, lm: {len(lm)}")

    # Calculate the normalized muscle fiber lengths
    lm_norm = lm / lm_opt

    # Calculate the pennation angle
    alpha = calc_pennation(lm_norm, lm_opt, alpha_opt)
    
    # Evaluate force length curves
    forces = np.zeros((2, len(lm_norm)))
    forces[0, :] = evaluate_curve(afl, lm_norm) # Active force length curve
    forces[1, :] = evaluate_curve(pfl, lm_norm) # Passive force length curve

    # Calculate total muscle fiber forces from sum of active and passive forces
    Fm_norm = np.sum(forces, axis=0)
    
    # Calculate normalized tendon forces assuming equilibrium with muscle fiber forces
    Ft_norm = Fm_norm * np.cos(alpha)

    # Calculate normalized tendon lengths from tendon forces
    lt_norm = evaluate_curve(tfl, Ft_norm, inverse=True)

    # Calculate slack length
    fiber_proj = lm_opt * lm_norm * np.cos(alpha)
    
    # Handle zero/near-zero lt_norm values to avoid division by zero
    mask = np.abs(lt_norm) < 1e-6
    lt_s = np.zeros_like(lmt)
    lt_s[~mask] = (lmt[~mask] - fiber_proj[~mask]) / lt_norm[~mask] # Should this be (1+lt_norm) or just lt_norm?
    lt_s[mask] = lmt[mask] - fiber_proj[mask]  # When lt_norm ≈ 0, assume lt_s = lmt - fiber_proj
    lt_s = np.clip(lt_s, 0, lmt)  # Maintain physical bounds
    
    return lt_s

def calc_fiber_length(
    lmt: Iterable[float] | float, 
    tsl: float,
    lm_opt: float, 
    alpha_opt: float, 
    afl: CurveType, 
    pfl: CurveType, 
    tfl: CurveType
    ) -> np.ndarray:
    """
    Calculate the muscle fiber length from the tendon slack length.
    
    Parameters:
    - lmt: Muscle-tendon lengths.
    - tsl: Tendon slack lengths.
    - lm_opt: Optimal muscle fiber length.
    - alpha_opt: Optimal pennation angle.
    - afl: Active force length curve.
    - pfl: Passive force length curve.
    - tfl: Tendon force length curve.
    """
    raise NotImplementedError("This function is not implemented yet.")
    if lm_opt <= 0:
        raise ValueError(f"lm_opt must be positive, got {lm_opt}")
    if alpha_opt < 0 or alpha_opt >= np.pi/2:
        raise ValueError(f"alpha_opt must be between 0 and pi/2, got {alpha_opt}")
    lmt = np.asarray(lmt)
    if np.any(lmt <= 0):
        raise ValueError(f"lmt must contain positive values, got {lmt}")
    if tsl <= 0:
        raise ValueError(f"tsl must be positive, got {tsl}")

    return np.array([])

def calc_tendon_force(
    lmt: Iterable[float] | float, 
    lm: Iterable[float] | float,
    tsl: float,
    lm_opt: float, 
    alpha_opt: float, 
    afl: CurveType, 
    pfl: CurveType, 
    tfl: CurveType
    ) -> np.ndarray:
    """
    Calculate the tendon force from the tendon slack length.
    
    Parameters:
    - lmt: Muscle-tendon lengths.
    - lm: Muscle fiber lengths.
    - tsl: Tendon slack length.
    - lm_opt: Optimal muscle fiber length.
    - alpha_opt: Optimal pennation angle.
    - afl: Active force length curve.
    - pfl: Passive force length curve.
    - tfl: Tendon force length curve.
    """
    raise NotImplementedError("This function is not implemented yet.")

def calc_fiber_force(
    lmt: Iterable[float] | float, 
    lm: Iterable[float] | float,
    tsl: float,
    lm_opt: float, 
    alpha_opt: float, 
    afl: CurveType, 
    pfl: CurveType, 
    tfl: CurveType
    ) -> np.ndarray:
    """
    Calculate the fiber force from the tendon slack length.
    
    Parameters:
    - lmt: Muscle-tendon lengths.
    - lm: Muscle fiber lengths.
    - tsl: Tendon slack length.
    - lm_opt: Optimal muscle fiber length.
    - alpha_opt: Optimal pennation angle.
    - afl: Active force length curve.
    - pfl: Passive force length curve.
    - tfl: Tendon force length curve.
    """
    raise NotImplementedError("This function is not implemented yet.")