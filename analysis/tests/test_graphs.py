import numpy as np
import pytest
from analysis.src.core.graphs import find_graphs

# Ожидаемые координаты точек пересечения
EXPECTED: list[tuple[float]] = [
    (0.021620378678209963, 3.329525534720379, 0.0003013793161065249),
    (0.04665752160326191, 1.542850978916794, 0.11055292689404808),
    (0.4236873178085874, 0.16990266135826518, 1.0050379605663258),
    (0.5129315070024422, 0.14034155027852746, 1.0065680408504034),
    (0.8972186742622621, 0.08023194896006178, 0.09389833458911935),
    (0.9151708854424055, 0.07865809984178564, 0.014784909643395206)
]

SYSTEM_PARAMS: tuple[float] = (
    16.67, 1.2, 1.128, 0.33, 3.3, 0.023,
    1.67, 9.0, 16.67,
)


def test_find_graphs():
    """
    При параметрах для точки бифуркации ф-я должна вернуть 6 точек пересечения
    с известными координатами.
    Допуск - 1e-5.
    """
    a, s_a, y, aplus_y, aminus_y, a_y, intersections = find_graphs(*SYSTEM_PARAMS)

    assert len(intersections) == len(EXPECTED), \
        f"Ожидалось {len(EXPECTED)} пересечений, получено {len(intersections)}"

    intersections = sorted(intersections, key=lambda x: x[0])
    expected = sorted(EXPECTED, key=lambda x: x[0])

    tol = 1e-5
    for (a_int, s_int, y_int), (a_exp, s_exp, y_exp) in zip(intersections, expected):
        assert np.isclose(
            a_int, a_exp, atol=tol), f"a: {a_int:.6f} ≠ {a_exp:.6f}"
        assert np.isclose(
            s_int, s_exp, atol=tol), f"s: {s_int:.6f} ≠ {s_exp:.6f}"
        assert np.isclose(
            y_int, y_exp, atol=tol), f"y: {y_int:.6f} ≠ {y_exp:.6f}"
