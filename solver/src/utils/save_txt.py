# solver/src/utils/save_txt.py

from pathlib import Path
import numpy as np
from typing import List, Dict


def save_results_to_txt(
    name: str,
    calc_id: int,
    parameter_app,
    numerical_app,
    base_data: List[Dict],
    control_data: List[Dict],
    max_diffs: tuple,
    note: str | None = None
) -> Path:
    """
    Сохраняет все результаты в один текстовый файл.
    """
    from solver.src.utils.paths import get_results_dir

    results_dir = get_results_dir()
    safe_name = "".join(c if c.isalnum() or c in " _-()" else "_" for c in name)
    filename = results_dir / f"Результаты_{safe_name}_ID{calc_id}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"ЭКСПЕРИМЕНТ: {name}\n")
        f.write(f"ID в базе: {calc_id}\n")
        if note:
            f.write(f"Заметка: {note}\n")
        f.write(f"Дата сохранения: {np.datetime64('now')}\n")
        f.write("=" * 80 + "\n\n")

        # Параметры системы
        f.write("ПАРАМЕТРЫ СИСТЕМЫ:\n")
        sys_params = {
            "c": numerical_app.entries["c"].get(),
            "μ": numerical_app.entries["μ"].get(),
            "c₀": numerical_app.entries["c₀"].get(),
            "ν": numerical_app.entries["ν"].get(),
            "ε": numerical_app.entries["ε"].get(),
            "d": numerical_app.entries["d"].get(),
            "e": numerical_app.entries["e"].get(),
            "f": numerical_app.entries["f"].get(),
            "η": numerical_app.entries["η"].get(),
            "Dₐ": numerical_app.entries["Dₐ"].get(),
            "Dₛ": numerical_app.entries["Dₛ"].get(),
            "Dᵧ": numerical_app.entries["Dᵧ"].get(),
        }
        for k, v in sys_params.items():
            f.write(f"{k:>4} = {float(v):.10g}\n")
        f.write("\n")

        # Параметры сетки
        n_val = int(parameter_app.entries["n (сетка по x)"].get()) + 1
        m_val = int(parameter_app.entries["m (сетка по t)"].get())
        T_val = float(parameter_app.entries["T (время)"].get())
        f.write("ПАРАМЕТРЫ СЕТКИ:\n")
        f.write(f"n (узлов по x) = {n_val}\n")
        f.write(f"m (шагов по t) = {m_val}\n")
        f.write(f"T (время)      = {T_val:.10g}\n")
        f.write("\n")

        # Начальные условия
        f.write("НАЧАЛЬНЫЕ УСЛОВИЯ:\n")
        a0 = [float(e.get()) for e in parameter_app.entries["a_0"]]
        a1 = [float(e.get()) for e in parameter_app.entries["a_1"]]
        a2 = [float(e.get()) for e in parameter_app.entries["a_2"]]
        a3 = [float(e.get()) for e in parameter_app.entries["a_3"]]
        b1 = int(parameter_app.entries["b_1"].get())
        b2 = int(parameter_app.entries["b_2"].get())
        b3 = int(parameter_app.entries["b_3"].get())

        f.write("Коэффициенты косинусов:\n")
        f.write(f"{'':<4}  {'a(x)':>14} {'s(x)':>14} {'y(x)':>14}\n")
        for label, vals in zip(["A₀", "A₁", "A₂", "A₃"], [a0, a1, a2, a3]):
            f.write(f"{label:<4}  {vals[0]:14.8f} {vals[1]:14.8f} {vals[2]:14.8f}\n")
        f.write(f"b₁ = {b1}, b₂ = {b2}, b₃ = {b3}\n\n")

        # Погрешности
        max_a, max_s, max_y = max_diffs
        f.write("МАКСИМАЛЬНЫЕ РАЗНОСТИ:\n")
        f.write(f"max |a - a*|  = {max_a:.10e}\n")
        f.write(f"max |s - s*|  = {max_s:.10e}\n")
        f.write(f"max |y - y*|  = {max_y:.10e}\n\n")

        # Заголовок таблицы данных
        header = (
            f"{'x':>10} "
            f"{'a_баз':>14} {'a_контр':>14} {'|Δa|':>12} "
            f"{'s_баз':>14} {'s_контр':>14} {'|Δs|':>12} "
            f"{'y_баз':>14} {'y_контр':>14} {'|Δy|':>12}\n"
        )
        separator = "-" * 140 + "\n"

        # Словарь для быстрого доступа к контрольным слоям
        control_by_layer = {layer["layer"]: layer for layer in control_data}

        for base_layer in base_data:
            t_base = base_layer["layer"]
            t_control = t_base * 4
            control_layer = control_by_layer.get(t_control)

            if control_layer is None:
                continue

            x_base = base_layer["x"]
            a_base = base_layer["a"]
            s_base = base_layer["s"]
            y_base = base_layer["y"]

            a_control = np.interp(x_base, control_layer["x"], control_layer["a"])
            s_control = np.interp(x_base, control_layer["x"], control_layer["s"])
            y_control = np.interp(x_base, control_layer["x"], control_layer["y"])

            da = np.abs(a_base - a_control)
            ds = np.abs(s_base - s_control)
            dy = np.abs(y_base - y_control)

            f.write(f"\nСЛОЙ t = {t_base} (базовая сетка) / t = {t_control} (контрольная сетка)\n")
            f.write(header)
            f.write(separator)

            for i in range(len(x_base)):
                f.write(
                    f"{x_base[i]:10.6f} "
                    f"{a_base[i]:14.8f} {a_control[i]:14.8f} {da[i]:12.3e} "
                    f"{s_base[i]:14.8f} {s_control[i]:14.8f} {ds[i]:12.3e} "
                    f"{y_base[i]:14.8f} {y_control[i]:14.8f} {dy[i]:12.3e}\n"
                )
            f.write("\n")

    return filename
