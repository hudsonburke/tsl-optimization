import numpy as np
from scipy.optimize import minimize
from typing import Callable, Iterable

from .curve_wrapper import CurveType, CurveWrapper, evaluate_curve
from .muscle_parameters import calc_pennation
from ._validators import as_1d_float_array


def _validate_lm_norm_range(lm_norm_range: tuple[float, float]) -> tuple[float, float]:
    lower = float(lm_norm_range[0])
    upper = float(lm_norm_range[1])
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("lm_norm_range values must be finite")
    if lower <= 0 or upper <= 0 or lower >= upper:
        raise ValueError(
            f"lm_norm_range must be positive and increasing, got {lm_norm_range}"
        )
    return lower, upper


_CONSTRAINED_METHODS = {"SLSQP", "trust-constr"}


def _validate_constrained_method(method: str | None) -> str:
    chosen = "SLSQP" if method is None else str(method)
    if chosen not in _CONSTRAINED_METHODS:
        supported = ", ".join(sorted(_CONSTRAINED_METHODS))
        raise ValueError(
            f"method must support nonlinear constraints, got {chosen!r}; use one of {supported}"
        )
    return chosen


def _validate_tendon_margin(tendon_margin: float) -> float:
    margin = float(tendon_margin)
    if not np.isfinite(margin) or margin < 0:
        raise ValueError(
            f"tendon_margin must be finite and nonnegative, got {tendon_margin}"
        )
    return margin


def _geometric_tendon_length(
    lm: np.ndarray, lmt: np.ndarray, lm_opt: float, alpha_opt: float
) -> np.ndarray:
    alpha = calc_pennation(lm / lm_opt, lm_opt, alpha_opt)
    return np.asarray(lmt - lm * np.cos(alpha), dtype=float)


def _tendon_model_terms(
    lm: np.ndarray,
    lmt: np.ndarray,
    lm_opt: float,
    alpha_opt: float,
    afl: CurveType | CurveWrapper,
    pfl: CurveType | CurveWrapper,
    tfl: CurveType | CurveWrapper,
) -> tuple[np.ndarray, np.ndarray]:
    lm_norm = lm / lm_opt
    alpha = calc_pennation(lm_norm, lm_opt, alpha_opt)
    fm_norm = evaluate_curve(afl, lm_norm) + evaluate_curve(pfl, lm_norm)
    ft_norm = np.clip(fm_norm * np.cos(alpha), 0.0, None)
    lt_norm = evaluate_curve(tfl, ft_norm, inverse=True)
    lt_norm = np.where(ft_norm < 1e-6, 1.0, lt_norm)

    invalid_lt_norm = ~np.isfinite(lt_norm) | (lt_norm < 1.0 - 1e-12)
    if np.any(invalid_lt_norm):
        raise ValueError(
            "Calculated tendon lengths must be finite and at or above slack length"
        )

    lt_geom = _geometric_tendon_length(lm, lmt, lm_opt, alpha_opt)
    if not np.all(np.isfinite(lt_geom)):
        raise ValueError("Calculated geometric tendon lengths must be finite")

    return lt_geom, lt_norm


def _ensure_constraint_satisfied(
    values: np.ndarray, *, name: str, tolerance: float = 1.0e-8
) -> None:
    min_value = float(np.min(values))
    if min_value < -tolerance:
        raise RuntimeError(
            f"Optimization returned a solution that violates the {name} constraint (min={min_value:.3e})"
        )


def ssdp(data: Iterable) -> float:
    # Calculate error equal to the sum of squared differences between every element of the vector
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D array-like structure")
    n = len(data)
    sum_sq = np.sum(data**2)
    sum_val = np.sum(data)
    err = n * sum_sq - sum_val**2
    return float(err)


def ssd(data: Iterable) -> float:
    # Sum of squared differences between slack lengths and the mean slack length
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D array-like structure")
    return float(np.sum((data - np.mean(data)) ** 2))


def optimize_fiber_length(
    lmt: Iterable[float],
    lm_opt: float,
    alpha_opt: float,
    afl: CurveType,
    pfl: CurveType,
    tfl: CurveType,
    lm_norm_range: tuple[float, float] = (0.5, 1.5),
    method: str | None = "SLSQP",
    objective: Callable[[Iterable], float] = ssdp,
    max_evaluations: int = 200000,
) -> np.ndarray:
    """
    Optimize the fiber length of a muscle using the Manal 2004 model.

    A hard no-buckling constraint is enforced across the full range of motion,
    so geometric tendon length must remain nonnegative at every sampled pose.
    """
    if not np.isfinite(lm_opt) or lm_opt <= 0:
        raise ValueError(f"lm_opt must be finite and positive, got {lm_opt}")
    if not np.isfinite(alpha_opt) or alpha_opt < 0 or alpha_opt >= np.pi / 2:
        raise ValueError(
            f"alpha_opt must be finite and between 0 and pi/2, got {alpha_opt}"
        )
    if max_evaluations <= 0:
        raise ValueError(f"max_evaluations must be positive, got {max_evaluations}")
    if not callable(objective):
        raise TypeError("objective must be callable")

    method_name = _validate_constrained_method(method)
    lmt = as_1d_float_array(lmt, name="lmt", positive=True)
    lm_norm_lower, lm_norm_upper = _validate_lm_norm_range(lm_norm_range)

    afl_w = CurveWrapper(afl) if not isinstance(afl, CurveWrapper) else afl
    pfl_w = CurveWrapper(pfl) if not isinstance(pfl, CurveWrapper) else pfl
    tfl_w = CurveWrapper(tfl) if not isinstance(tfl, CurveWrapper) else tfl

    lm0 = np.linspace(lm_norm_lower, lm_norm_upper, len(lmt)) * lm_opt
    lb = np.full_like(lm0, lm_opt * lm_norm_lower, dtype=float)
    ub = np.full_like(lm0, lm_opt * lm_norm_upper, dtype=float)

    def objective_wrapper(lm: np.ndarray) -> float:
        try:
            lt_geom, lt_norm = _tendon_model_terms(
                np.asarray(lm, dtype=float), lmt, lm_opt, alpha_opt, afl_w, pfl_w, tfl_w
            )
        except ValueError:
            return 1.0e20

        tsl = lt_geom / lt_norm
        if not np.all(np.isfinite(tsl)):
            return 1.0e20

        val = objective(tsl)
        if not np.isfinite(val):
            return 1.0e20
        return float(val)

    def no_buckle_constraint(lm: np.ndarray) -> np.ndarray:
        return _geometric_tendon_length(np.asarray(lm, dtype=float), lmt, lm_opt, alpha_opt)

    result = minimize(
        objective_wrapper,
        lm0,
        bounds=list(zip(lb, ub)),
        constraints=[{"type": "ineq", "fun": no_buckle_constraint}],
        method=method_name,
        options={
            "maxiter": max_evaluations,
            "ftol": 1e-10,
        },
    )

    if not result.success:
        result_method = getattr(result, "method", method_name)
        raise RuntimeError(
            "Optimization failed "
            f"(method={result_method}, status={result.status}, nfev={result.nfev}): "
            f"{result.message}"
        )

    lm = np.asarray(result.x, dtype=float)
    _ensure_constraint_satisfied(no_buckle_constraint(lm), name="no-buckling")
    return lm



def optimize_fiber_length_and_tsl(
    lmt: Iterable[float],
    lm_opt: float,
    alpha_opt: float,
    afl: CurveType,
    pfl: CurveType,
    tfl: CurveType,
    lm_norm_range: tuple[float, float] = (0.5, 1.5),
    tendon_margin: float = 0.0,
    method: str = "SLSQP",
    max_iterations: int = 2000,
) -> tuple[np.ndarray, float]:
    """
    Jointly optimize fiber lengths and a shared tendon slack length with a
    hard no-buckling inequality constraint.

    The optimized variables are lm[0..n-1] and a shared lts. The objective
    minimizes the mismatch between geometric tendon length and tendon length
    implied by force equilibrium, while enforcing:

        lmt - lm * cos(alpha) >= (1 + tendon_margin) * lts

    at every sampled pose.
    """
    if not np.isfinite(lm_opt) or lm_opt <= 0:
        raise ValueError(f"lm_opt must be finite and positive, got {lm_opt}")
    if not np.isfinite(alpha_opt) or alpha_opt < 0 or alpha_opt >= np.pi / 2:
        raise ValueError(
            f"alpha_opt must be finite and between 0 and pi/2, got {alpha_opt}"
        )
    if max_iterations <= 0:
        raise ValueError(f"max_iterations must be positive, got {max_iterations}")

    method_name = _validate_constrained_method(method)
    lmt = as_1d_float_array(lmt, name="lmt", positive=True)
    lm_norm_lower, lm_norm_upper = _validate_lm_norm_range(lm_norm_range)
    tendon_margin = _validate_tendon_margin(tendon_margin)

    afl_w = CurveWrapper(afl) if not isinstance(afl, CurveWrapper) else afl
    pfl_w = CurveWrapper(pfl) if not isinstance(pfl, CurveWrapper) else pfl
    tfl_w = CurveWrapper(tfl) if not isinstance(tfl, CurveWrapper) else tfl

    lm0 = np.linspace(lm_norm_lower, lm_norm_upper, len(lmt)) * lm_opt
    try:
        lt_geom0, lt_norm0 = _tendon_model_terms(lm0, lmt, lm_opt, alpha_opt, afl_w, pfl_w, tfl_w)
        tsl0_candidates = lt_geom0 / lt_norm0
    except ValueError:
        tsl0_candidates = np.asarray([np.nan], dtype=float)

    finite_tsl0 = tsl0_candidates[np.isfinite(tsl0_candidates)]
    tsl_upper = float(np.min(lmt) / (1.0 + tendon_margin))
    tsl0 = float(np.median(finite_tsl0)) if finite_tsl0.size else float("nan")
    if not np.isfinite(tsl0) or tsl0 <= 0:
        tsl0 = min(tsl_upper, float(np.min(lmt) * 0.5))
    tsl0 = float(np.clip(tsl0, 1.0e-8, tsl_upper))

    x0 = np.concatenate([lm0, np.asarray([tsl0], dtype=float)])
    bounds = [
        (lm_opt * lm_norm_lower, lm_opt * lm_norm_upper) for _ in range(len(lmt))
    ]
    bounds.append((1.0e-8, tsl_upper))

    def objective_wrapper(x: np.ndarray) -> float:
        lm = np.asarray(x[:-1], dtype=float)
        tsl = float(x[-1])
        if not np.isfinite(tsl) or tsl <= 0:
            return 1.0e20

        try:
            lt_geom, lt_norm = _tendon_model_terms(
                lm, lmt, lm_opt, alpha_opt, afl_w, pfl_w, tfl_w
            )
        except ValueError:
            return 1.0e20

        residual = lt_geom - tsl * lt_norm
        if not np.all(np.isfinite(residual)):
            return 1.0e20
        return float(np.sum(residual**2))

    def no_buckle_constraint(x: np.ndarray) -> np.ndarray:
        lm = np.asarray(x[:-1], dtype=float)
        tsl = float(x[-1])
        lt_geom = _geometric_tendon_length(lm, lmt, lm_opt, alpha_opt)
        return np.asarray(lt_geom - (1.0 + tendon_margin) * tsl, dtype=float)

    result = minimize(
        objective_wrapper,
        x0,
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": no_buckle_constraint}],
        method=method_name,
        options={
            "maxiter": max_iterations,
            "ftol": 1e-10,
        },
    )

    if not result.success:
        result_method = getattr(result, "method", method_name)
        raise RuntimeError(
            "Constrained optimization failed "
            f"(method={result_method}, status={result.status}, nfev={result.nfev}): "
            f"{result.message}"
        )

    lm = np.asarray(result.x[:-1], dtype=float)
    tsl = float(result.x[-1])
    _ensure_constraint_satisfied(no_buckle_constraint(result.x), name="no-buckling")
    return lm, tsl
