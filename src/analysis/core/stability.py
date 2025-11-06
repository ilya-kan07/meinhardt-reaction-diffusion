"""
Анализ устойчивости равновесия в модели Мейнхардта.
"""
import math
import numpy as np
from typing import Dict, List


def calculate_stability_matrix(
    c: float, mu: float, C0: float, V: float, eps: float,
    d: float, e: float, f: float, eta: float,
    Da: float, Ds: float, Dy: float,
    a_eq: float, s_eq: float, y_eq: float,
    k: float, L: float
) -> Dict:
    """
    Вычисляет матрицу Якоби, коэффициенты характеристического полинома и устойчивость.
    Args:
        c, mu, C0, V, eps, d, e, f, eta, Da, Ds, Dy: Параметры системы
        a, s, y: Параметры равновесия
        k, L: Параметры пространственной области
    Returns:
        dict: Результаты вычислений (матрицы, коэффициенты, устойчивость, max_real_parts)
    """
    kappa2 = ((math.pi * k) / L) ** 2

    J11 = 2 * c * a_eq * s_eq - mu
    J12 = c * a_eq ** 2
    J13 = 0
    J21 = -2 * c * a_eq * s_eq
    J22 = -c * (a_eq ** 2) - V - eps * y_eq
    J23 = -eps * s_eq
    J31 = d
    J32 = 0
    J33 = -e + (2 * eta * y_eq) / ((1 + f * y_eq ** 2) ** 2)

    b0 = 1
    b1 = -(J11 + J22 + J33)
    b2 = (J11 * J22 + J11 * J33 + J22 * J33 - J12 * J21)
    b3 = -(J11 * J22 * J33 - J12 * J21 * J33 + J12 * J23 * J31)
    b = b1 * b2 - b3
    print(b0, b1, b2, b3, b)
    stability = b1 > 0 and b3 > 0 and b > 0

    Jk11 = J11 - kappa2 * Da
    Jk12 = J12
    Jk13 = J13
    Jk21 = J21
    Jk22 = J22 - kappa2 * Ds
    Jk23 = J23
    Jk31 = J31
    Jk32 = J32
    Jk33 = J33 - kappa2 * Dy

    bk0 = 1
    bk1 = -(Jk11 + Jk22 + Jk33)
    bk2 = (Jk11 * Jk22 + Jk11 * Jk33 + Jk22 * Jk33 - Jk12 * Jk21)
    bk3 = -(Jk11 * Jk22 * Jk33 - Jk12 * Jk21 * Jk33 + Jk12 * Jk23 * Jk31)
    bk = bk1 * bk2 - bk3
    print(bk0, bk1, bk2, bk3, bk)
    stability_k = bk1 > 0 and bk3 > 0 and bk > 0

    k_values = np.arange(0, 11)
    max_real_parts = []

    for k_val in k_values:
        kappa2_val = ((math.pi * k_val) / L) ** 2
        Jk11_val = J11 - kappa2_val * Da
        Jk22_val = J22 - kappa2_val * Ds
        Jk33_val = J33 - kappa2_val * Dy

        Jk = np.array([
            [Jk11_val, Jk12, Jk13],
            [Jk21, Jk22_val, Jk23],
            [Jk31, Jk32, Jk33_val]
        ])

        eigenvalues = np.roots([
            1,
            -(Jk11_val + Jk22_val + Jk33_val),
            (Jk11_val * Jk22_val + Jk11_val * Jk33_val +
             Jk22_val * Jk33_val - Jk12 * Jk21),
            -(Jk11_val * Jk22_val * Jk33_val - Jk12 *
              Jk21 * Jk33_val + Jk12 * Jk23 * Jk31)
        ])

        max_real = max(e.real for e in eigenvalues)
        max_real_parts.append(max_real)

    return {
        'J': [[J11, J12, J13], [J21, J22, J23], [J31, J32, J33]],
        'b0': b0, 'b1': b1, 'b2': b2, 'b3': b3, 'b': b,
        'stability': stability,
        'kappa2': kappa2,
        'Jk': [[Jk11, Jk12, Jk13], [Jk21, Jk22, Jk23], [Jk31, Jk32, Jk33]],
        'bk0': bk0, 'bk1': bk1, 'bk2': bk2, 'bk3': bk3, 'bk': bk,
        'stability_k': stability_k,
        'k_values': k_values,
        'max_real_parts': max_real_parts
    }
