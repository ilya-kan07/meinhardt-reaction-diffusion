"""
Решение нелинейной системы уравнений Мейнхардта методом Ньютона.
"""
import numpy as np
from typing import Tuple


def newton_method(
    c: float, mu: float, C0: float, V: float, eps: float,
    d: float, e: float, f: float, eta: float,
    x0: Tuple[float, float, float],
    max_iter: int = 100,
    tol: float = 1e-8
) -> Tuple[int, np.ndarray, float, float, float, float]:
    """
    Решает систему F(x) = 0 методом Ньютона.

    Returns:
        iterations, solution, f1, f2, f3, residual_norm
    """
    x = np.array(x0, dtype=float)
    a, s, y = x

    for iteration in range(max_iter):
        # Остаточная функция
        f1 = c * a**2 * s - mu * a
        f2 = C0 - c * a**2 * s - V * s - eps * s * y
        f3 = d * a - e * y + (eta * y**2) / (1 + f * y**2)
        F = np.array([f1, f2, f3])

        if np.linalg.norm(F) < tol:
            residual_norm = np.linalg.norm(F)
            return iteration + 1, x, f1, f2, f3, residual_norm

        # Якобиан
        J11 = 2 * c * a * s - mu
        J12 = c * a**2
        J13 = 0
        J21 = -2 * c * a * s
        J22 = -c * a**2 - V - eps * y
        J23 = -eps * s
        J31 = d
        J32 = 0
        J33 = -e + (2 * eta * y * (1 + f * y**2) - 2 *
                    f * y**3 * eta) / (1 + f * y**2)**2

        J = np.array([[J11, J12, J13], [J21, J22, J23], [J31, J32, J33]])

        try:
            delta = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            break

        x += delta
        a, s, y = x

    # Финальная невязка
    f1 = c * a**2 * s - mu * a
    f2 = C0 - c * a**2 * s - V * s - eps * s * y
    f3 = d * a - e * y + (eta * y**2) / (1 + f * y**2)
    residual_norm = np.linalg.norm([f1, f2, f3])

    return max_iter, x, f1, f2, f3, residual_norm
