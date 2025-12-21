import numpy as np
import copy
from solver.src.core.solver import MeinhardtSolver
from solver.src.config.presets import PRESETS

ETALON_PRESET_1 = {
    "final_a": np.array([0.5492002, 0.5492002, 0.5492002, 0.5492002, 0.5492002, 0.5492002, 0.5492002, 0.5492002, 0.5492002, 0.5492002, 0.5492002]),
    "final_s": np.array([0.12969786, 0.12969786, 0.12969786, 0.12969786, 0.12969786, 0.12969786, 0.12969786, 0.12969786, 0.12969786, 0.12969786, 0.12969786]),
    "final_y": np.array([1.00729677, 1.00729677, 1.00729677, 1.00729677, 1.00729677, 1.00729677, 1.00729677, 1.00729677, 1.00729677, 1.00729677, 1.00729677]),
    "max_a_diff": 0.029871963382933187,
    "max_s_diff": 0.030374353852220773,
    "max_y_diff": 0.0034096139563413175
}

ETALON_PRESET_2 = {
    "final_a": np.array([0.80530864, 0.79250808, 0.75441616, 0.69274104, 0.61224638, 0.52159151, 0.43194266, 0.35388116, 0.29483176, 0.2585734, 0.24640782]),
    "final_s": np.array([0.1012293, 0.10291075, 0.10784736, 0.11567671, 0.12570609, 0.13690234, 0.14802745, 0.15788201, 0.16551706, 0.17031313, 0.17194465]),
    "final_y": np.array([1.00733986, 1.00731142, 1.00722865, 1.00709922, 1.00693555, 1.00675398, 1.00657291, 1.00641054, 1.00628263, 1.00620105, 1.00617305]),
    "max_a_diff": 0.006307319678447998,
    "max_s_diff": 0.01351539746852673,
    "max_y_diff": 0.005447510937574784
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
        override_m=500,
        override_T=10.0)

    assert np.allclose(final_layer["a"], ETALON_PRESET_1["final_a"], atol=1e-6)
    assert np.allclose(final_layer["s"], ETALON_PRESET_1["final_s"], atol=1e-6)
    assert np.allclose(final_layer["y"], ETALON_PRESET_1["final_y"], atol=1e-6)

    assert abs(max_a - ETALON_PRESET_1["max_a_diff"]) < 1e-12
    assert abs(max_s - ETALON_PRESET_1["max_s_diff"]) < 1e-12
    assert abs(max_y - ETALON_PRESET_1["max_y_diff"]) < 1e-12


def test_solver_preset_2():
    final_layer, (max_a, max_s, max_y) = run_solver_with_preset(
        "Набор 2 (стационарное неоднородное)",
        override_m=150000,
        override_T=600.0)

    assert np.allclose(final_layer["a"], ETALON_PRESET_2["final_a"], atol=1e-6)
    assert np.allclose(final_layer["s"], ETALON_PRESET_2["final_s"], atol=1e-6)
    assert np.allclose(final_layer["y"], ETALON_PRESET_2["final_y"], atol=1e-6)

    assert abs(max_a - ETALON_PRESET_2["max_a_diff"]) < 1e-12
    assert abs(max_s - ETALON_PRESET_2["max_s_diff"]) < 1e-12
    assert abs(max_y - ETALON_PRESET_2["max_y_diff"]) < 1e-12
