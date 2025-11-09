from typing import Dict
import numpy as np
from numpy.typing import NDArray

FINE_GRID_POINTS: int = 1000


def compute_initial_conditions(
    a0: NDArray, a1: NDArray, a2: NDArray, a3: NDArray,
    b1: int, b2: int, b3: int, n: int
) -> Dict[str, Dict[str, NDArray]]:
    """
    Вычисляет начальные условия на грубой и тонкой сетке.
    Args:
        a0, a1, a2, a3: массивы [3] — базовые значения
        b1, b2, b3: целые — частоты косинусов
        n: количество узлов на грубой сетке (включая границы)

    Returns:
        dict с 'coarse' и 'fine' сетками
    """
    x_coarse = np.linspace(0, 1, n)
    x_fine = np.linspace(0, 1, FINE_GRID_POINTS)

    def u(x: NDArray, a0_i: float, a1_i: float, a2_i: float, a3_i: float) -> NDArray:
        return (
            a0_i
            + a1_i * np.cos(b1 * np.pi * x)
            + a2_i * np.cos(b2 * np.pi * x)
            + a3_i * np.cos(b3 * np.pi * x)
        )

    # Грубая сетка
    a_coarse = u(x_coarse, a0[0], a1[0], a2[0], a3[0])
    s_coarse = u(x_coarse, a0[1], a1[1], a2[1], a3[1])
    y_coarse = u(x_coarse, a0[2], a1[2], a2[2], a3[2])

    # Тонкая сетка
    a_fine = u(x_fine, a0[0], a1[0], a2[0], a3[0])
    s_fine = u(x_fine, a0[1], a1[1], a2[1], a3[1])
    y_fine = u(x_fine, a0[2], a1[2], a2[2], a3[2])

    return {
        'coarse': {
            'x': x_coarse,
            'a': a_coarse,
            's': s_coarse,
            'y': y_coarse
        },
        'fine': {
            'x': x_fine,
            'a': a_fine,
            's': s_fine,
            'y': y_fine
        }
    }
