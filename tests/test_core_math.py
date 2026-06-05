import gc
import os
import subprocess
import sys
import types
import weakref
from pathlib import Path

import numpy as np
import pytest

opensim_stub = types.ModuleType("opensim")
opensim_stub.Function = object
sys.modules.setdefault("opensim", opensim_stub)

scipy_stub = types.ModuleType("scipy")
scipy_interpolate_stub = types.ModuleType("scipy.interpolate")
scipy_optimize_stub = types.ModuleType("scipy.optimize")


class _Interp1D:
    def __init__(self, x, y, kind="linear", bounds_error=False, fill_value=None):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.fill_value = fill_value

    def __call__(self, values):
        values = np.asarray(values, dtype=float)
        if isinstance(self.fill_value, tuple):
            left, right = self.fill_value
        else:
            left = self.fill_value
            right = self.fill_value
        return np.interp(values, self.x, self.y, left=left, right=right)


def _unused_minimize(*args, **kwargs):
    raise RuntimeError("minimize stub should not be called in these tests")


class _FunctionCurve:
    def __init__(self, func):
        self._func = func

    def calcValue(self, x):
        return float(self._func(float(x)))


scipy_interpolate_stub.interp1d = _Interp1D
scipy_optimize_stub.minimize = _unused_minimize
scipy_stub.interpolate = scipy_interpolate_stub
scipy_stub.optimize = scipy_optimize_stub
sys.modules.setdefault("scipy", scipy_stub)
sys.modules.setdefault("scipy.interpolate", scipy_interpolate_stub)
sys.modules.setdefault("scipy.optimize", scipy_optimize_stub)

from tsl_optimization.curve_wrapper import CurveWrapper, _WRAPPER_CACHE, evaluate_curve
from tsl_optimization.muscle_parameters import calc_pennation, calc_tsl
from tsl_optimization.optimizer import optimize_fiber_length, optimize_fiber_length_and_tsl
import tsl_optimization.optimizer as optimizer_module


def test_curve_wrapper_rejects_invalid_numpy_shape():
    with pytest.raises(ValueError, match="shape"):
        CurveWrapper(np.array([0.0, 1.0, 2.0]))


def test_curve_wrapper_rejects_unsorted_x_values():
    curve = np.array([[0.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
    with pytest.raises(ValueError, match="strictly increasing"):
        CurveWrapper(curve)


def test_curve_wrapper_inverse_requires_monotonic_curve():
    curve = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 0.5]])
    wrapper = CurveWrapper(curve)
    with pytest.raises(ValueError, match="inverse is undefined"):
        wrapper.evaluate_inverse([0.25, 0.75])


def test_curve_wrapper_inverse_accepts_leading_plateau():
    curve = np.array([[0.8, 1.0, 1.2, 1.4], [0.0, 0.0, 0.2, 0.8]])
    wrapper = CurveWrapper(curve)
    values = wrapper.evaluate_inverse([0.0, 0.2, 0.8])
    assert np.allclose(values, [1.0, 1.2, 1.4])


def test_curve_wrapper_opensim_inverse_accepts_plateau_then_increase():
    curve = _FunctionCurve(lambda x: 0.0 if x <= 1.0 else x - 1.0)
    wrapper = CurveWrapper(curve, x_range=(0.0, 2.0), n_samples=1001)

    values = wrapper.evaluate_inverse([0.0, 0.25, 0.5])

    assert np.allclose(values, [1.0, 1.25, 1.5], atol=1e-3)


def test_curve_wrapper_opensim_inverse_requires_monotonic_curve():
    curve = _FunctionCurve(lambda x: 1.0 - (x - 1.0) ** 2)
    wrapper = CurveWrapper(curve, x_range=(0.0, 2.0), n_samples=1001)

    with pytest.raises(ValueError, match="inverse is undefined"):
        wrapper.evaluate_inverse([0.25, 0.75])

def test_package_import_does_not_require_opensim():
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")

    result = subprocess.run(
        [sys.executable, "-c", "import tsl_optimization"],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_evaluate_curve_cache_does_not_keep_numpy_curve_alive():
    _WRAPPER_CACHE.clear()
    curve = np.array([[0.0, 1.0], [0.0, 2.0]])
    curve_ref = weakref.ref(curve)

    evaluate_curve(curve, [0.5])
    assert len(_WRAPPER_CACHE) == 1

    del curve
    gc.collect()

    assert curve_ref() is None
    assert len(_WRAPPER_CACHE) == 0


def test_calc_pennation_accepts_scalar_input():
    alpha = calc_pennation(1.0, 0.1, 0.2)
    assert alpha.shape == (1,)
    assert np.isclose(alpha[0], 0.2)


def test_calc_pennation_rejects_nan_input():
    with pytest.raises(ValueError, match="finite"):
        calc_pennation([np.nan], 1.0, 0.2)


def test_calc_pennation_rejects_non_1d_input():
    with pytest.raises(ValueError, match="1D"):
        calc_pennation([[1.0, 1.1]], 1.0, 0.2)


def test_calc_pennation_rejects_nonfinite_lm_opt():
    with pytest.raises(ValueError, match="finite and positive"):
        calc_pennation([1.0], np.nan, 0.2)


def test_calc_pennation_zero_width_stays_zero_near_zero_length():
    alpha = calc_pennation([1.0e-8], 1.0, 0.0)

    assert np.isclose(alpha[0], 0.0)


def test_calc_tsl_accepts_scalar_inputs():
    afl = np.array([[0.5, 1.0, 1.5], [1.0, 1.0, 1.0]])
    pfl = np.array([[0.5, 1.0, 1.5], [0.0, 0.0, 0.0]])
    tfl = np.array([[1.0, 2.0], [0.0, 1.0]])

    tsl = calc_tsl(2.0, 1.0, 1.0, 0.0, afl, pfl, tfl)

    assert tsl.shape == (1,)
    assert np.isclose(tsl[0], 0.5)



def test_calc_tsl_accepts_plateau_tfl_curve():
    afl = np.array([[0.5, 1.0, 1.5], [1.0, 1.0, 1.0]])
    pfl = np.array([[0.5, 1.0, 1.5], [0.0, 0.0, 0.0]])
    tfl = np.array([[0.8, 1.0, 2.0], [0.0, 0.0, 1.0]])

    tsl = calc_tsl(2.0, 1.0, 1.0, 0.0, afl, pfl, tfl)

    assert tsl.shape == (1,)
    assert np.isclose(tsl[0], 0.5)


def test_calc_tsl_strict_mode_raises_for_nonphysical_slack_lengths():
    afl = np.array([[0.5, 1.0, 1.5], [1.0, 1.0, 1.0]])
    pfl = np.array([[0.5, 1.0, 1.5], [0.0, 0.0, 0.0]])
    tfl = np.array([[1.0, 2.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="physical bounds"):
        calc_tsl(0.5, 1.0, 1.0, 0.0, afl, pfl, tfl, strict=True)



def test_calc_tsl_rejects_nonfinite_lm_opt():
    afl = np.array([[0.5, 1.0, 1.5], [1.0, 1.0, 1.0]])
    pfl = np.array([[0.5, 1.0, 1.5], [0.0, 0.0, 0.0]])
    tfl = np.array([[1.0, 2.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="finite and positive"):
        calc_tsl(2.0, 1.0, np.nan, 0.0, afl, pfl, tfl)



def test_calc_tsl_rejects_nonfinite_alpha_opt():
    afl = np.array([[0.5, 1.0, 1.5], [1.0, 1.0, 1.0]])
    pfl = np.array([[0.5, 1.0, 1.5], [0.0, 0.0, 0.0]])
    tfl = np.array([[1.0, 2.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="finite and between"):
        calc_tsl(2.0, 1.0, 1.0, np.nan, afl, pfl, tfl)


def test_evaluate_curve_reuses_cached_wrapper_for_same_curve():
    curve = np.array([[0.0, 1.0], [0.0, 2.0]])
    first = evaluate_curve(curve, [0.25, 0.75])
    second = evaluate_curve(curve, [0.25, 0.75])
    assert np.allclose(first, second)


def test_optimize_fiber_length_validates_inputs():
    afl = np.array([[0.5, 1.0, 1.5], [1.0, 1.0, 1.0]])
    pfl = np.array([[0.5, 1.0, 1.5], [0.0, 0.0, 0.0]])
    tfl = np.array([[1.0, 2.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="lm_norm_range"):
        optimize_fiber_length([2.0, 2.1], 1.0, 0.0, afl, pfl, tfl, (1.0, 1.0))



def test_optimize_fiber_length_rejects_nonfinite_lm_opt():
    afl = np.array([[0.5, 1.0, 1.5], [1.0, 1.0, 1.0]])
    pfl = np.array([[0.5, 1.0, 1.5], [0.0, 0.0, 0.0]])
    tfl = np.array([[1.0, 2.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="finite and positive"):
        optimize_fiber_length([2.0, 2.1], np.nan, 0.0, afl, pfl, tfl)



def test_optimize_fiber_length_rejects_nonfinite_alpha_opt():
    afl = np.array([[0.5, 1.0, 1.5], [1.0, 1.0, 1.0]])
    pfl = np.array([[0.5, 1.0, 1.5], [0.0, 0.0, 0.0]])
    tfl = np.array([[1.0, 2.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="finite and between"):
        optimize_fiber_length([2.0, 2.1], 1.0, np.nan, afl, pfl, tfl)


def test_optimize_fiber_length_rejects_unconstrained_method():
    afl = np.array([[0.5, 1.0, 1.5], [1.0, 1.0, 1.0]])
    pfl = np.array([[0.5, 1.0, 1.5], [0.0, 0.0, 0.0]])
    tfl = np.array([[1.0, 2.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="nonlinear constraints"):
        optimize_fiber_length([2.0, 2.1], 1.0, 0.0, afl, pfl, tfl, method="L-BFGS-B")


def test_optimize_fiber_length_enforces_no_buckling_constraint(monkeypatch):
    afl = np.array([[0.5, 1.0, 1.5], [1.0, 1.0, 1.0]])
    pfl = np.array([[0.5, 1.0, 1.5], [0.0, 0.0, 0.0]])
    tfl = np.array([[1.0, 2.0], [0.0, 1.0]])

    def fake_minimize(fun, x0, bounds=None, constraints=(), method=None, options=None):
        assert method == "SLSQP"
        assert len(x0) == 2
        assert len(bounds) == 2
        assert len(constraints) == 1
        objective = fun(x0)
        assert np.isfinite(objective)
        constraint_value = constraints[0]["fun"](x0)
        assert constraint_value.shape == (2,)
        assert np.all(constraint_value >= 0)
        return types.SimpleNamespace(
            success=True,
            x=np.asarray(x0, dtype=float),
            status=0,
            nfev=1,
            message="ok",
        )

    monkeypatch.setattr(optimizer_module, "minimize", fake_minimize)

    lm = optimize_fiber_length([2.0, 2.1], 1.0, 0.0, afl, pfl, tfl)

    assert lm.shape == (2,)



def test_optimize_fiber_length_and_tsl_validates_tendon_margin():
    afl = np.array([[0.5, 1.0, 1.5], [1.0, 1.0, 1.0]])
    pfl = np.array([[0.5, 1.0, 1.5], [0.0, 0.0, 0.0]])
    tfl = np.array([[1.0, 2.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="tendon_margin"):
        optimize_fiber_length_and_tsl(
            [2.0, 2.1], 1.0, 0.0, afl, pfl, tfl, tendon_margin=-0.01
        )



def test_optimize_fiber_length_and_tsl_returns_fiber_lengths_and_slack_length(
    monkeypatch,
):
    afl = np.array([[0.5, 1.0, 1.5], [1.0, 1.0, 1.0]])
    pfl = np.array([[0.5, 1.0, 1.5], [0.0, 0.0, 0.0]])
    tfl = np.array([[1.0, 2.0], [0.0, 1.0]])

    def fake_minimize(fun, x0, bounds=None, constraints=(), method=None, options=None):
        assert method == "SLSQP"
        assert len(x0) == 3
        assert len(bounds) == 3
        assert len(constraints) == 1
        objective = fun(x0)
        assert np.isfinite(objective)
        constraint_value = constraints[0]["fun"](x0)
        assert constraint_value.shape == (2,)
        return types.SimpleNamespace(
            success=True,
            x=np.asarray(x0, dtype=float),
            status=0,
            nfev=1,
            message="ok",
        )

    monkeypatch.setattr(optimizer_module, "minimize", fake_minimize)

    lm, tsl = optimize_fiber_length_and_tsl(
        [2.0, 2.1], 1.0, 0.0, afl, pfl, tfl, tendon_margin=0.01
    )

    assert lm.shape == (2,)
    assert np.isfinite(tsl)
    assert tsl > 0


def test_optimize_fiber_length_and_tsl_rejects_successful_buckled_solution(
    monkeypatch,
):
    afl = np.array([[0.5, 1.0, 1.5], [1.0, 1.0, 1.0]])
    pfl = np.array([[0.5, 1.0, 1.5], [0.0, 0.0, 0.0]])
    tfl = np.array([[1.0, 2.0], [0.0, 1.0]])

    def fake_minimize(fun, x0, bounds=None, constraints=(), method=None, options=None):
        return types.SimpleNamespace(
            success=True,
            x=np.asarray([1.9, 2.0, 0.5], dtype=float),
            status=0,
            nfev=1,
            message="ok",
        )

    monkeypatch.setattr(optimizer_module, "minimize", fake_minimize)

    with pytest.raises(RuntimeError, match="no-buckling"):
        optimize_fiber_length_and_tsl([2.0, 2.1], 1.0, 0.0, afl, pfl, tfl)
