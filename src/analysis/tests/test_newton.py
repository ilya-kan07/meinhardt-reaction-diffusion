import numpy as np
import pytest
from src.analysis.config.parameters import SYSTEM_PARAMS
from src.analysis.core.newton import newton_method

def _system_params() -> dict:
    return {name: value for name, value in SYSTEM_PARAMS}


def test_newton_method():
    """
    Проверяет что метод Ньютона при заданных параметрах сходится к известному равновесию
    x = [0.512934   0.14034087 1.0065681 ]
    """
    params = _system_params()
    x0 = (0.5, 0.5, 0.5)

    iterations, solution, f1, f2, f3, residual_norm = newton_method(
        c=params["с"], mu=params["μ"], C0=params["С₀"], V=params["V"],
        eps=params["ɛ"], d=params["d"], e=params["e"], f=params["f"], eta=params["η"],
        x0=x0,
        max_iter=100,
        tol=1e-8
    )

    expected_solution = np.array([0.512934, 0.14034087, 1.0065681])
    expected_residuals = np.array(
        [-1.866071519529555e-09, 1.8765882181526194e-09, -6.994405055138486e-14])
    expected_norm = 2.646470529101115e-09

    # проверка сходимости
    assert iterations < 100, f"Слишком много итераций: {iterations}"
    assert residual_norm < 1e-8, f"Невязка слишком большая: {residual_norm}"

    # проверка решения
    assert np.allclose(solution, expected_solution, atol=1e-5), \
        f"Решение не совпадает: {solution} ≠ {expected_solution}"

    # проверка невязок
    assert np.allclose([f1, f2, f3], expected_residuals, atol=1e-8), \
        f"Невязки не совпадают: [{f1:.2e}, {f2:.2e}, {f3:.2e}] ≠ {expected_residuals}"

    # проверка нормы
    assert np.isclose(residual_norm, expected_norm, atol=1e-12), \
        f"Норма невязки не совпадает: {residual_norm:.2e} ≠ {expected_norm:.2e}"
