import numpy as np
import pytest
from analysis.src.core.stability import calculate_stability_matrix
from analysis.src.config.parameters import (
    SYSTEM_PARAMS, EQUILIBRIUM_PARAMS, SPATIAL_PARAMS
)


def _params_to_dict(params):
    return {name: value for name, value in params}


def test_calculate_stability_matrix():
    """
    Проверяет, что при параметрах системы для точки бифуркации:
        - Коэффициенты полинома при k=0 и k≠0 совпадают с известными
        - Устойчивость: True для обеих гармоник
    """
    system = _params_to_dict(SYSTEM_PARAMS)
    eq = _params_to_dict(EQUILIBRIUM_PARAMS)
    spatial = _params_to_dict(SPATIAL_PARAMS)

    result = calculate_stability_matrix(
        c=system["с"], mu=system["μ"], C0=system["С₀"], V=system["V"],
        eps=system["ɛ"], d=system["d"], e=system["e"], f=system["f"], eta=system["η"],
        Da=system["Da"], Ds=system["Ds"], Dy=system["Dy"],
        a_eq=eq["a"], s_eq=eq["s"], y_eq=eq["y"],
        k=spatial["k"], L=spatial["L"]
    )

    # проверка коэффициентов при k=0
    expected_k0 = {
        'b0': 1.0,
        'b1': 8.179802429115968,
        'b2': 10.058665429231464,
        'b3': 1.2293057144566213,
        'b': 81.04859019723573
    }

    for key, expected in expected_k0.items():
        assert np.isclose(result[key], expected, atol=1e-10), \
            f"{key} (k=0): {result[key]:.12f} ≠ {expected:.12f}"

    # проверка коэффициентов при k≠0
    expected_knz = {
        'bk0': 1.0,
        'bk1': 28.53096250911307,
        'bk2': 194.17504049892042,
        'bk3': 7.518172022435987e-09,
        'bk': 5540.000800672693
    }

    for key, expected in expected_knz.items():
        assert np.isclose(result[key], expected, atol=1e-8), \
            f"{key} (k≠0): {result[key]:.12f} ≠ {expected:.12f}"

    # проверка устойчивости
    assert result['stability'] is True, \
        "Равновесие должно быть устойчиво по нулевой гармонике"
    assert result['stability_k'] is True, \
        "Равновесие должно быть устойчиво по k-ой гармонике"

    # проверка макс. действ. частей > 0
    assert all(m < 1e-10 for m in result['max_real_parts']), \
        "Все max(Re(λ)) должны быть < 0 (устойчивость)"
