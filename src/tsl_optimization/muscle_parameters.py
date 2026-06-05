import numpy as np
from typing import Iterable
from .curve_wrapper import CurveType, CurveWrapper, evaluate_curve
from ._validators import as_1d_float_array


def calc_pennation(
    lm_norm: Iterable[float] | float, lm_opt: float, alpha_opt: float
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
    if not np.isfinite(alpha_opt) or not (0 <= alpha_opt < np.pi / 2):
        raise ValueError(
            f"alpha_opt must be finite and between 0 and pi/2, got {alpha_opt}"
        )
    if not np.isfinite(lm_opt) or lm_opt <= 0:
        raise ValueError(f"lm_opt must be finite and positive, got {lm_opt}")

    lm_norm = as_1d_float_array(lm_norm, name="lm_norm", positive=True)
    lm = lm_norm * lm_opt
    near_zero = lm < 1e-6
    width = lm_opt * np.sin(alpha_opt)

    l = np.clip(width / np.where(near_zero, 1.0, lm), 0.0, 1.0)
    alpha = np.clip(np.arcsin(l), 0.0, np.pi / 2)
    if width > 0:
        alpha[near_zero] = np.pi / 2

    return alpha


def calc_tsl(
    lmt: Iterable[float] | float,
    lm: Iterable[float] | float,
    lm_opt: float,
    alpha_opt: float,
    afl: CurveType | CurveWrapper,
    pfl: CurveType | CurveWrapper,
    tfl: CurveType | CurveWrapper,
    strict: bool = False,
    clip: bool = True,
) -> np.ndarray:
    """
    Calculate the slack length of a muscle.

    Parameters:
    - lmt: Muscle-tendon lengths.
    - lm: Muscle fiber lengths.
    - lm_opt: Optimal muscle fiber length.
    - alpha_opt: Optimal pennation angle.
    - afl: Active force length curve.
    - pfl: Passive force length curve.
    - tfl: Tendon force length curve.
    - strict: Raise instead of sanitizing nonphysical tendon states.
    - clip: Clamp slack lengths into physical bounds when strict=False.
    """
    if not np.isfinite(lm_opt) or lm_opt <= 0:
        raise ValueError(f"lm_opt must be finite and positive, got {lm_opt}")
    if not np.isfinite(alpha_opt) or alpha_opt < 0 or alpha_opt >= np.pi / 2:
        raise ValueError(
            f"alpha_opt must be finite and between 0 and pi/2, got {alpha_opt}"
        )
    lmt = as_1d_float_array(lmt, name="lmt", positive=True)
    lm = as_1d_float_array(lm, name="lm", positive=True)

    if lmt.shape != lm.shape:
        raise ValueError(
            f"lmt and lm must have the same shape, got lmt: {lmt.shape}, lm: {lm.shape}"
        )

    lm_norm = lm / lm_opt
    alpha = calc_pennation(lm_norm, lm_opt, alpha_opt)
    Fm_norm = evaluate_curve(afl, lm_norm) + evaluate_curve(pfl, lm_norm)
    Ft_norm = np.clip(Fm_norm * np.cos(alpha), 0.0, None)
    lt_norm = evaluate_curve(tfl, Ft_norm, inverse=True)
    lt_norm = np.where(Ft_norm < 1e-6, 1.0, lt_norm)

    invalid_lt_norm = ~np.isfinite(lt_norm) | (lt_norm < 1.0 - 1e-12)
    if strict and np.any(invalid_lt_norm):
        raise ValueError(
            "Calculated tendon lengths must be finite and at or above slack length"
        )
    lt_norm = np.where(invalid_lt_norm, 1.0, lt_norm)

    fiber_proj = lm * np.cos(alpha)
    lt_s = (lmt - fiber_proj) / lt_norm
    invalid_lt_s = ~np.isfinite(lt_s) | (lt_s < 0) | (lt_s > lmt)
    if strict and np.any(invalid_lt_s):
        raise ValueError(
            "Calculated tendon slack lengths fall outside physical bounds"
        )
    if clip:
        lt_s = np.clip(lt_s, 0, lmt)

    return lt_s
