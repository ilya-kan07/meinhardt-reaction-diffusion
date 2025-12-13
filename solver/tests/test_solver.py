import numpy as np
import copy
from solver.src.core.solver import MeinhardtSolver
from solver.src.config.presets import PRESETS

ETALON_PRESET_1 = {
    "final_a": np.array([1.040017, 1.040016, 1.040015, 1.040014, 1.040014, 1.040015, 1.040016, 1.040016, 1.040016, 1.040015, 1.040014]),
    "final_s": np.array([0.054657, 0.054657, 0.054657, 0.054658, 0.054657, 0.054657, 0.054657, 0.054657, 0.054657, 0.054657, 0.054658]),
    "final_y": np.array([0.573469, 0.573466, 0.573459, 0.573454, 0.573455, 0.573461, 0.573467, 0.573468, 0.573464, 0.573457, 0.573454]),
    "max_a_diff": 0.0299,
    "max_s_diff": 0.0304,
    "max_y_diff": 0.0034
}

ETALON_PRESET_2 = {
    "final_a": np.array([0.804268, 0.791470, 0.753399, 0.691792, 0.611437, 0.520993, 0.431592, 0.353768, 0.294905, 0.258762, 0.246635]),
    "final_s": np.array([0.101419, 0.103099, 0.108032, 0.115854, 0.125871, 0.137050, 0.148154, 0.157989, 0.165609, 0.170395, 0.172023]),
    "final_y": np.array([1.007328, 1.007299, 1.007217, 1.007088, 1.006924, 1.006743, 1.006563, 1.006401, 1.006273, 1.006192, 1.006164]),
    "max_a_diff": 0.0913,
    "max_s_diff": 0.0294,
    "max_y_diff": 1.9382
}


def run_solver_with_preset(preset_name: str,
                           override_m: int = None,
                           override_T: float = None):

    data = copy.deepcopy(PRESETS[preset_name])
    system = data["system_params"]

    if override_m is not None:
        data["m"] = override_m
    if override_T is not None:
        data["T"] = override_T

    x_coarse = np.linspace(0, 1, data["n"] + 1)

    def u(x, idx):
        return (
            data["a_0"][idx]
            + data["a_1"][idx] * np.cos(data["b_1"] * np.pi * x)
            + data["a_2"][idx] * np.cos(data["b_2"] * np.pi * x)
            + data["a_3"][idx] * np.cos(data["b_3"] * np.pi * x)
        )

    a_coarse = u(x_coarse, 0)
    s_coarse = u(x_coarse, 1)
    y_coarse = u(x_coarse, 2)

    initial_coarse = {
        'x': x_coarse,
        'a': a_coarse,
        's': s_coarse,
        'y': y_coarse
    }

    solver = MeinhardtSolver(
        c=system["c"], mu=system["μ"], c_0=system["c₀"], nu=system["ν"],
        epsilon=system["ε"], d=system["d"], e=system["e"], f=system["f"], eta=system["η"],
        D_a=system["Dₐ"], D_s=system["Dₛ"], D_y=system["Dᵧ"],
        n_coarse=data["n"] + 1,
        m_coarse=data["m"],
        T=data["T"],
        initial_coarse=initial_coarse,
        a0=data["a_0"], a1=data["a_1"], a2=data["a_2"], a3=data["a_3"],
        b1=data["b_1"], b2=data["b_2"], b3=data["b_3"]
    )

    (
        base_data,
        control_data,
        _base_db,            # прореженные — не нужны в тесте
        _control_db,         # прореженные — не нужны в тесте
        max_a, max_s, max_y
    ) = solver.solve()
    return base_data[-1], (max_a, max_s, max_y)


def test_solver_preset_1():
    final_layer, (max_a, max_s, max_y) = run_solver_with_preset(
        "Набор 1 (стационарное однородное)",
        override_m=50,
        override_T=1.0)

    assert np.allclose(final_layer["a"], ETALON_PRESET_1["final_a"], atol=1e-3)
    assert np.allclose(final_layer["s"], ETALON_PRESET_1["final_s"], atol=1e-3)
    assert np.allclose(final_layer["y"], ETALON_PRESET_1["final_y"], atol=1e-3)

    assert abs(max_a - ETALON_PRESET_1["max_a_diff"]) < 1e-4
    assert abs(max_s - ETALON_PRESET_1["max_s_diff"]) < 1e-4
    assert abs(max_y - ETALON_PRESET_1["max_y_diff"]) < 1e-4


def test_solver_preset_2():
    final_layer, (max_a, max_s, max_y) = run_solver_with_preset(
        "Набор 2 (стационарное неоднородное)",
        override_m=10500,
        override_T=50.0)

    assert np.allclose(final_layer["a"], ETALON_PRESET_2["final_a"], atol=1e-3)
    assert np.allclose(final_layer["s"], ETALON_PRESET_2["final_s"], atol=1e-3)
    assert np.allclose(final_layer["y"], ETALON_PRESET_2["final_y"], atol=1e-3)

    assert abs(max_a - ETALON_PRESET_2["max_a_diff"]) < 1e-4
    assert abs(max_s - ETALON_PRESET_2["max_s_diff"]) < 1e-4
    assert abs(max_y - ETALON_PRESET_2["max_y_diff"]) < 1e-4
