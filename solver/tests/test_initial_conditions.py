import numpy as np
from solver.src.core.initial_conditions import compute_initial_conditions
from solver.src.config.presets import PRESETS


def test_initial_conditions_preset_1():
    preset_name = "Набор 1 (стационарное однородное)"
    data = PRESETS[preset_name]

    a0 = data["a_0"]
    a1 = data["a_1"]
    a2 = data["a_2"]
    a3 = data["a_3"]
    b1 = data["b_1"]
    b2 = data["b_2"]
    b3 = data["b_3"]
    n = data["n"] + 1

    result = compute_initial_conditions(a0, a1, a2, a3, b1, b2, b3, n)

    x_coarse = result['coarse']['x']
    x_fine = result['fine']['x']

    def expected_u(x, a0_i, a1_i, a2_i, a3_i):
        return (
            a0_i
            + a1_i * np.cos(b1 * np.pi * x)
            + a2_i * np.cos(b2 * np.pi * x)
            + a3_i * np.cos(b3 * np.pi * x)
        )

    expected_a_coarse = expected_u(x_coarse, a0[0], a1[0], a2[0], a3[0])
    expected_s_coarse = expected_u(x_coarse, a0[1], a1[1], a2[1], a3[1])
    expected_y_coarse = expected_u(x_coarse, a0[2], a1[2], a2[2], a3[2])

    # проверка базовой сетки
    assert np.allclose(result['coarse']['a'], expected_a_coarse, atol=1e-12)
    assert np.allclose(result['coarse']['s'], expected_s_coarse, atol=1e-12)
    assert np.allclose(result['coarse']['y'], expected_y_coarse, atol=1e-12)

    # проверка контрольной сетки
    expected_a_fine = expected_u(x_fine, a0[0], a1[0], a2[0], a3[0])
    expected_s_fine = expected_u(x_fine, a0[1], a1[1], a2[1], a3[1])
    expected_y_fine = expected_u(x_fine, a0[2], a1[2], a2[2], a3[2])

    assert np.allclose(result['fine']['a'], expected_a_fine, atol=1e-12)
    assert np.allclose(result['fine']['s'], expected_s_fine, atol=1e-12)
    assert np.allclose(result['fine']['y'], expected_y_fine, atol=1e-12)

    # доп: проверка размеров
    assert len(result['coarse']['x']) == n
    assert len(result['fine']['x']) == 1000
    assert result['fine']['x'][0] == 0.0
    assert result['fine']['x'][-1] == 1.0


def test_initial_conditions_preset_2():
    preset_name = "Набор 2 (стационарное неоднородное)"
    data = PRESETS[preset_name]

    a0 = data["a_0"]
    a1 = data["a_1"]
    a2 = data["a_2"]
    a3 = data["a_3"]
    b1 = data["b_1"]
    b2 = data["b_2"]
    b3 = data["b_3"]
    n = data["n"] + 1

    result = compute_initial_conditions(a0, a1, a2, a3, b1, b2, b3, n)

    x_coarse = result['coarse']['x']
    x_fine = result['fine']['x']

    def expected_u(x, a0_i, a1_i, a2_i, a3_i):
        return (
            a0_i
            + a1_i * np.cos(b1 * np.pi * x)
            + a2_i * np.cos(b2 * np.pi * x)
            + a3_i * np.cos(b3 * np.pi * x)
        )

    expected_a_coarse = expected_u(x_coarse, a0[0], a1[0], a2[0], a3[0])
    expected_s_coarse = expected_u(x_coarse, a0[1], a1[1], a2[1], a3[1])
    expected_y_coarse = expected_u(x_coarse, a0[2], a1[2], a2[2], a3[2])

    assert np.allclose(result['coarse']['a'], expected_a_coarse, atol=1e-12)
    assert np.allclose(result['coarse']['s'], expected_s_coarse, atol=1e-12)
    assert np.allclose(result['coarse']['y'], expected_y_coarse, atol=1e-12)

    expected_a_fine = expected_u(x_fine, a0[0], a1[0], a2[0], a3[0])
    expected_s_fine = expected_u(x_fine, a0[1], a1[1], a2[1], a3[1])
    expected_y_fine = expected_u(x_fine, a0[2], a1[2], a2[2], a3[2])

    assert np.allclose(result['fine']['a'], expected_a_fine, atol=1e-12)
    assert np.allclose(result['fine']['s'], expected_s_fine, atol=1e-12)
    assert np.allclose(result['fine']['y'], expected_y_fine, atol=1e-12)

    assert len(result['coarse']['x']) == n
    assert len(result['fine']['x']) == 1000
    assert result['fine']['x'][0] == 0.0
    assert result['fine']['x'][-1] == 1.0
