import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import threading
import queue
import time
import math
import os
import re
import sys
from tkinter import simpledialog
from src.utils.paths import get_resource_path, get_results_dir
from src.config.presets import PRESETS
from src.core.initial_conditions import compute_initial_conditions


class ParameterApp:
    def __init__(self, parent, numerical_app_callback):
        self.main_frame = ttk.Frame(parent)
        self.main_frame.pack(fill="both", expand=True)

        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side="left", fill="y", padx=10, pady=10)

        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side="right", fill="both",
                              expand=True, padx=10, pady=10)

        self.numerical_app_callback = numerical_app_callback
        self.add_conditions_image()
        self.entries = {}
        self.initial_conditions = {}
        self.create_parameter_inputs()
        self.create_axis_limits_inputs()
        self.create_preset_selector()
        self.create_buttons()
        self.load_default_parameters()

        self.fig = plt.figure(figsize=(12, 4))
        self.ax1 = self.fig.add_subplot(1, 3, 1)
        self.ax2 = self.fig.add_subplot(1, 3, 2)
        self.ax3 = self.fig.add_subplot(1, 3, 3)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.table_frame = ttk.Frame(self.right_frame)
        self.table_frame.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(self.table_frame, columns=(
            "x", "a(x)", "s(x)", "y(x)"), show="headings")
        self.tree.heading("x", text="x")
        self.tree.heading("a(x)", text="a(x)")
        self.tree.heading("s(x)", text="s(x)")
        self.tree.heading("y(x)", text="y(x)")

        self.tree.column("x", width=100, anchor="center")
        self.tree.column("a(x)", width=100, anchor="center")
        self.tree.column("s(x)", width=100, anchor="center")
        self.tree.column("y(x)", width=100, anchor="center")

        scrollbar = ttk.Scrollbar(
            self.table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def add_conditions_image(self):
        image_path = get_resource_path("conditions.png")

        if not os.path.exists(image_path):
            messagebox.showerror("Ошибка", f"Файл не найден: {image_path}")
            return
        try:
            image = Image.open(image_path)
            new_width = 400
            image = image.resize((new_width, int(
                new_width * image.height / image.width)), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image)
            image_label = ttk.Label(self.left_frame, image=self.photo)
            image_label.pack(pady=10)
        except Exception as e:
            messagebox.showerror(
                "Ошибка", f"Не удалось загрузить изображение 'conditions.png': {e}")

    def create_parameter_inputs(self):
        params_frame = ttk.LabelFrame(
            self.left_frame, text="Введите параметры")
        params_frame.pack(fill="x", pady=(0, 10))

        header_frame = ttk.Frame(params_frame)
        header_frame.pack(fill="x", pady=(5, 2))
        ttk.Label(header_frame, text="", width=10).pack(side="left")
        ttk.Label(header_frame, text="a", width=8).pack(side="left", padx=2)
        ttk.Label(header_frame, text="s", width=8).pack(side="left", padx=2)
        ttk.Label(header_frame, text="y", width=8).pack(side="left", padx=2)
        ttk.Label(header_frame, text="", width=10).pack(side="left")

        a0_frame = ttk.Frame(params_frame)
        a0_frame.pack(fill="x", pady=2)
        ttk.Label(a0_frame, text="A0:", width=5).pack(side="left", padx=(0, 5))
        entry_a0_a = ttk.Entry(a0_frame, width=8)
        entry_a0_s = ttk.Entry(a0_frame, width=8)
        entry_a0_y = ttk.Entry(a0_frame, width=8)
        entry_a0_a.pack(side="left", padx=2)
        entry_a0_s.pack(side="left", padx=2)
        entry_a0_y.pack(side="left", padx=2)
        ttk.Label(a0_frame, text="", width=10).pack(side="left")
        self.entries["a_0"] = (entry_a0_a, entry_a0_s, entry_a0_y)

        for i in range(1, 4):
            frame = ttk.Frame(params_frame)
            frame.pack(fill="x", pady=2)
            ttk.Label(frame, text=f"A{i}:", width=5).pack(
                side="left", padx=(0, 5))
            entry_a = ttk.Entry(frame, width=8)
            entry_s = ttk.Entry(frame, width=8)
            entry_y = ttk.Entry(frame, width=8)
            entry_a.pack(side="left", padx=2)
            entry_s.pack(side="left", padx=2)
            entry_y.pack(side="left", padx=2)
            self.entries[f"a_{i}"] = (entry_a, entry_s, entry_y)
            ttk.Label(frame, text=f"b{i}:", width=5).pack(
                side="left", padx=(0, 5))
            entry_b = ttk.Entry(frame, width=5)
            entry_b.pack(side="left", padx=2)
            self.entries[f"b_{i}"] = entry_b

        grid_frame = ttk.LabelFrame(self.left_frame, text="Параметры сетки")
        grid_frame.pack(fill="x", pady=10)
        for param, default in [("n (сетка по x)", 10), ("m (сетка по t)", 50), ("T (время)", 1.0)]:
            frame = ttk.Frame(grid_frame)
            frame.pack(fill="x", pady=2)
            ttk.Label(frame, text=f"{param}:", width=15).pack(
                side="left", padx=(0, 5))
            entry = ttk.Entry(frame, width=10)
            entry.insert(0, str(default))
            entry.pack(side="left", padx=2)
            self.entries[param] = entry
            entry.bind('<KeyRelease>', lambda e: self.update_time_constraint())

        self.time_constraint_label = ttk.Label(
            grid_frame, text="Ограничение tau < h²/(2*max(D_a, D_s, D_y)): ещё не задано")
        self.time_constraint_label.pack(pady=5)
        self.tau_display_label = ttk.Label(
            grid_frame, text="Текущий tau: ещё не задано", foreground="blue")
        self.tau_display_label.pack(pady=5)

    def save_initial_conditions_plot(self, save_dir):
        if not self.initial_conditions:
            messagebox.showerror(
                "Ошибка", "Нет начальных условий для сохранения!")
            return

        limits = self.get_axis_limits()
        if not limits:
            return

        fig = plt.figure(figsize=(12, 4))
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        x_fine = self.initial_conditions['fine']['x']
        a_x_fine = self.initial_conditions['fine']['a']
        s_x_fine = self.initial_conditions['fine']['s']
        y_x_fine = self.initial_conditions['fine']['y']

        ax1.plot(x_fine, a_x_fine, label='a(x)', color='blue')
        ax1.set_xlabel('x')
        ax1.set_ylabel('a(x)')
        ax1.set_ylim(limits['a_min'], limits['a_max'])
        ax1.grid(True)
        ax1.legend()
        ax1.set_title('Начальное условие: a(x)')

        ax2.plot(x_fine, s_x_fine, label='s(x)', color='green')
        ax2.set_xlabel('x')
        ax2.set_ylabel('s(x)')
        ax2.set_ylim(limits['s_min'], limits['s_max'])
        ax2.grid(True)
        ax2.legend()
        ax2.set_title('Начальное условие: s(x)')

        ax3.plot(x_fine, y_x_fine, label='y(x)', color='red')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y(x)')
        ax3.set_ylim(limits['y_min'], limits['y_max'])
        ax3.grid(True)
        ax3.legend()
        ax3.set_title('Начальное условие: y(x)')

        fig.tight_layout()
        fig.savefig(save_dir / "initial_conditions.png",
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

    def validate_parameters(self):
        try:
            for key in ["a_0", "a_1", "a_2", "a_3"]:
                for entry in self.entries[key]:
                    value = float(entry.get())
                    if value < 0:
                        raise ValueError(
                            f"Значение {key} должно быть неотрицательным")

            for i in range(1, 4):
                b_value = int(self.entries[f"b_{i}"].get())
                if b_value < 0:
                    raise ValueError(
                        f"Значение b_{i} должно быть неотрицательным целым числом")

            n = int(self.entries["n (сетка по x)"].get()) + 1
            m = int(self.entries["m (сетка по t)"].get())
            T = float(self.entries["T (время)"].get())
            if n <= 2:
                raise ValueError("n (сетка по x) должно быть >= 2")
            if m <= 0:
                raise ValueError("m (сетка по t) должно быть положительным")
            if T <= 0:
                raise ValueError("T (время) должно быть положительным")

            for func in ["a", "s", "y"]:
                min_val = float(self.entries[f"{func}_min"].get())
                max_val = float(self.entries[f"{func}_max"].get())
                if min_val >= max_val:
                    raise ValueError(
                        f"Для {func}: минимальное значение должно быть меньше максимального")

            return True
        except ValueError as e:
            messagebox.showerror(
                "Ошибка ввода", f"Некорректные параметры: {str(e)}")
            return False

    def update_time_constraint(self, event=None):
        if not self.validate_parameters():
            self.time_constraint_label.config(
                text="Ограничение tau < h²/(2*max(D_a, D_s, D_y)): введите корректные значения", foreground="red"
            )
            self.tau_display_label.config(
                text="Текущий tau: введите корректные значения", foreground="blue"
            )
            return

        try:
            n = int(self.entries["n (сетка по x)"].get()) + 1
            m = float(self.entries["m (сетка по t)"].get())
            T = float(self.entries["T (время)"].get())

            D_a = float(
                self.numerical_app_callback.__self__.entries["Dₐ"].get())
            D_s = float(
                self.numerical_app_callback.__self__.entries["Dₛ"].get())
            D_y = float(
                self.numerical_app_callback.__self__.entries["Dᵧ"].get())

            if D_a < 0 or D_s < 0 or D_y < 0:
                raise ValueError(
                    "Коэффициенты диффузии должны быть неотрицательными")

            L = 1.0
            h = L / (n - 1)
            max_D = max(D_a, D_s, D_y)
            max_tau = h**2 / (2 * max_D)
            current_tau = T / m

            self.time_constraint_label.config(
                text=f"Ограничение: tau < {max_tau:.6f}"
            )
            self.tau_display_label.config(
                text=f"Текущий tau = {current_tau:.6f}", foreground="blue"
            )

            if current_tau < max_tau:
                self.time_constraint_label.config(foreground="green")
            else:
                self.time_constraint_label.config(
                    text=f"Ограничение: tau < {max_tau:.6f} не выполнено!", foreground="red"
                )
        except ValueError as e:
            self.time_constraint_label.config(
                text=f"Ограничение tau < h²/(2*max(D_a, D_s, D_y)): {str(e)}", foreground="red"
            )
            self.tau_display_label.config(
                text="Текущий tau: введите корректные значения", foreground="blue"
            )

    def create_axis_limits_inputs(self):
        limits_frame = ttk.LabelFrame(self.left_frame, text="Пределы осей Y")
        limits_frame.pack(fill="x", pady=10)

        for func, defaults in [
            ("a", (-1.0, 2.0)),
            ("s", (-1.0, 2.0)),
            ("y", (-1.0, 2.0))
        ]:
            frame = ttk.Frame(limits_frame)
            frame.pack(fill="x", pady=2)
            ttk.Label(frame, text=f"{func} min:", width=8).pack(
                side="left", padx=(0, 5))
            entry_min = ttk.Entry(frame, width=8)
            entry_min.insert(0, str(defaults[0]))
            entry_min.pack(side="left", padx=2)
            self.entries[f"{func}_min"] = entry_min

            ttk.Label(frame, text=f"{func} max:", width=8).pack(
                side="left", padx=(0, 5))
            entry_max = ttk.Entry(frame, width=8)
            entry_max.insert(0, str(defaults[1]))
            entry_max.pack(side="left", padx=2)
            self.entries[f"{func}_max"] = entry_max

    def create_preset_selector(self):
        preset_frame = ttk.LabelFrame(
            self.left_frame, text="Наборы параметров")
        preset_frame.pack(fill="x", pady=10)

        self.preset_var = tk.StringVar(
            value="Набор 1 (стационарное однородное)")
        self.preset_selector = ttk.Combobox(
            preset_frame, textvariable=self.preset_var, state="readonly")
        self.preset_selector['values'] = (
            "Набор 1 (стационарное однородное)", "Набор 2 (стационарное неоднородное)")
        self.preset_selector.pack(fill="x", padx=5, pady=5)

        button_frame = ttk.Frame(preset_frame)
        button_frame.pack(fill="x", pady=5)
        ttk.Button(button_frame, text="Задать параметры из файла",
                   command=self.load_from_file).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Задать параметры",
                   command=self.apply_preset).pack(side="left", padx=5)

    def create_buttons(self):
        ttk.Button(self.left_frame, text="Показать графики",
                   command=self.plot_graphs).pack(pady=5)

    def load_default_parameters(self):
        self.apply_preset()
        self.update_time_constraint()

    def apply_preset(self):
        preset_name = self.preset_var.get()
        if preset_name not in PRESETS:
            return

        preset_data = PRESETS[preset_name]

        try:
            # Заполнение a_0
            for i, val in enumerate(preset_data["a_0"]):
                self.entries["a_0"][i].delete(0, tk.END)
                self.entries["a_0"][i].insert(0, str(val))

            # Заполнение a_1, a_2, a_3 и b_1, b_2, b_3
            for i in range(1, 4):
                a_i = preset_data[f"a_{i}"]
                b_i = preset_data[f"b_{i}"]
                for j, val in enumerate(a_i):
                    self.entries[f"a_{i}"][j].delete(0, tk.END)
                    self.entries[f"a_{i}"][j].insert(0, str(val))
                self.entries[f"b_{i}"].delete(0, tk.END)
                self.entries[f"b_{i}"].insert(0, str(b_i))

            # n, m, T
            self.entries["n (сетка по x)"].delete(0, tk.END)
            self.entries["m (сетка по t)"].delete(0, tk.END)
            self.entries["T (время)"].delete(0, tk.END)
            self.entries["n (сетка по x)"].insert(0, str(preset_data["n"]))
            self.entries["m (сетка по t)"].insert(0, str(preset_data["m"]))
            self.entries["T (время)"].insert(0, str(preset_data["T"]))

            # Системные параметры
            self.numerical_app_callback(preset_data["system_params"])

            self.update_time_constraint()

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить параметры: {e}")

            self.entries["a_0"][0].delete(0, tk.END)
            self.entries["a_0"][1].delete(0, tk.END)
            self.entries["a_0"][2].delete(0, tk.END)
            self.entries["a_0"][0].insert(0, str(preset_data["a_0"][0]))
            self.entries["a_0"][1].insert(0, str(preset_data["a_0"][1]))
            self.entries["a_0"][2].insert(0, str(preset_data["a_0"][2]))

            for i in range(1, 4):
                a_i = preset_data[f"a_{i}"]
                b_i = preset_data[f"b_{i}"]
                self.entries[f"a_{i}"][0].delete(0, tk.END)
                self.entries[f"a_{i}"][1].delete(0, tk.END)
                self.entries[f"a_{i}"][2].delete(0, tk.END)
                self.entries[f"a_{i}"][0].insert(0, str(a_i[0]))
                self.entries[f"a_{i}"][1].insert(0, str(a_i[1]))
                self.entries[f"a_{i}"][2].insert(0, str(a_i[2]))
                self.entries[f"b_{i}"].delete(0, tk.END)
                self.entries[f"b_{i}"].insert(0, str(b_i))

            self.entries["n (сетка по x)"].delete(0, tk.END)
            self.entries["m (сетка по t)"].delete(0, tk.END)
            self.entries["T (время)"].delete(0, tk.END)
            self.entries["n (сетка по x)"].insert(0, str(preset_data["n"]))
            self.entries["m (сетка по t)"].insert(0, str(preset_data["m"]))
            self.entries["T (время)"].insert(0, str(preset_data["T"]))

            self.numerical_app_callback(preset_data["system_params"])

            self.update_time_constraint()

    def compute_initial_conditions(self) -> bool:
        if not self.validate_parameters():
            return False

        try:
            a0 = np.array([float(e.get()) for e in self.entries["a_0"]])
            a1 = np.array([float(e.get()) for e in self.entries["a_1"]])
            a2 = np.array([float(e.get()) for e in self.entries["a_2"]])
            a3 = np.array([float(e.get()) for e in self.entries["a_3"]])
            b1 = int(self.entries["b_1"].get())
            b2 = int(self.entries["b_2"].get())
            b3 = int(self.entries["b_3"].get())
            n = int(self.entries["n (сетка по x)"].get()) + 1

            self.initial_conditions = compute_initial_conditions(
                a0, a1, a2, a3, b1, b2, b3, n
            )
            return True

        except ValueError as e:
            messagebox.showerror("Ошибка", f"Некорректные данные: {e}")
            return False

    def get_axis_limits(self):
        try:
            limits = {
                'a_min': float(self.entries['a_min'].get()),
                'a_max': float(self.entries['a_max'].get()),
                's_min': float(self.entries['s_min'].get()),
                's_max': float(self.entries['s_max'].get()),
                'y_min': float(self.entries['y_min'].get()),
                'y_max': float(self.entries['y_max'].get())
            }
            if limits['a_min'] >= limits['a_max'] or limits['s_min'] >= limits['s_max'] or limits['y_min'] >= limits['y_max']:
                raise ValueError(
                    "Минимальное значение должно быть меньше максимального!")
            return limits
        except ValueError as e:
            messagebox.showerror("Ошибка", f"Ошибка в пределах осей: {e}")
            return None

    def plot_graphs(self):
        if not self.compute_initial_conditions():
            return

        limits = self.get_axis_limits()
        if not limits:
            return

        x_fine = self.initial_conditions['fine']['x']
        a_x_fine = self.initial_conditions['fine']['a']
        s_x_fine = self.initial_conditions['fine']['s']
        y_x_fine = self.initial_conditions['fine']['y']
        x_coarse = self.initial_conditions['coarse']['x']
        a_x_coarse = self.initial_conditions['coarse']['a']
        s_x_coarse = self.initial_conditions['coarse']['s']
        y_x_coarse = self.initial_conditions['coarse']['y']

        n = int(self.entries["n (сетка по x)"].get())
        x_precision = math.ceil(math.log10(n)) + 1 if n > 2 else 1

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        self.ax1.plot(x_fine, a_x_fine, label='a(x)', color='blue')
        self.ax1.set_xlabel('x')
        self.ax1.set_ylabel('a(x)')
        self.ax1.set_ylim(limits['a_min'], limits['a_max'])
        self.ax1.grid(True)
        self.ax1.legend()

        self.ax2.plot(x_fine, s_x_fine, label='s(x)', color='green')
        self.ax2.set_xlabel('x')
        self.ax2.set_ylabel('s(x)')
        self.ax2.set_ylim(limits['s_min'], limits['s_max'])
        self.ax2.grid(True)
        self.ax2.legend()

        self.ax3.plot(x_fine, y_x_fine, label='y(x)', color='red')
        self.ax3.set_xlabel('x')
        self.ax3.set_ylabel('y(x)')
        self.ax3.set_ylim(limits['y_min'], limits['y_max'])
        self.ax3.grid(True)
        self.ax3.legend()

        self.fig.tight_layout()
        self.canvas.draw()

        for item in self.tree.get_children():
            self.tree.delete(item)

        for i in range(len(x_coarse)):
            self.tree.insert("", "end", values=(
                f"{x_coarse[i]:.{x_precision}f}", f"{a_x_coarse[i]:.6f}", f"{s_x_coarse[i]:.6f}", f"{y_x_coarse[i]:.6f}"))

    def load_from_file(self):
        file_path = filedialog.askopenfilename(
            initialdir=os.path.dirname(__file__),
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not file_path:
            return
        try:
            params = read_parameters(file_path)
            # Начальные условия
            for key in ["a_0", "a_1", "a_2", "a_3"]:
                if key in params:
                    self.entries[key][0].delete(0, tk.END)
                    self.entries[key][1].delete(0, tk.END)
                    self.entries[key][2].delete(0, tk.END)
                    self.entries[key][0].insert(0, str(params[key][0]))
                    self.entries[key][1].insert(0, str(params[key][1]))
                    self.entries[key][2].insert(0, str(params[key][2]))
            for i in range(1, 4):
                b_key = f"b_{i}"
                if b_key in params:
                    self.entries[b_key].delete(0, tk.END)
                    self.entries[b_key].insert(0, str(params[b_key]))
            # Параметры сетки
            for param in ["n (сетка по x)", "m (сетка по t)", "T (время)"]:
                if param in params:
                    self.entries[param].delete(0, tk.END)
                    self.entries[param].insert(0, str(params[param]))
            # Параметры системы
            system_params = {k: params[k] for k in params if k in [
                "c", "μ", "c₀", "ν", "ε", "d", "e", "f", "η", "Dₐ", "Dₛ", "Dᵧ"]}
            self.numerical_app_callback(system_params)
            self.update_time_constraint()
            messagebox.showinfo(
                "Успех", "Параметры успешно загружены из файла!")
        except Exception as e:
            messagebox.showerror(
                "Ошибка", f"Не удалось загрузить параметры: {str(e)}")


class NumericalSolutionApp:
    def __init__(self, parent, main_app):
        self.main_app = main_app
        self.parameter_app = None
        self.main_frame = ttk.Frame(parent)
        self.main_frame.pack(fill="both", expand=True)

        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side="left", fill="y", padx=5, pady=5)

        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side="right", fill="both",
                              expand=True, padx=5, pady=5)

        self.entries = {}
        self.add_conditions_image()

        self.create_parameter_inputs()
        self.create_buttons()
        self.create_layer_selector()

        self.radio_frame = ttk.LabelFrame(
            self.left_frame, text="Выбор таблицы")
        self.radio_frame.pack(fill="x", pady=2)
        self.table_mode = tk.StringVar(value="a")
        ttk.Radiobutton(self.radio_frame, text="Функция a", variable=self.table_mode,
                        value="a", command=self.switch_table).pack(anchor="w", padx=5, pady=2)
        ttk.Radiobutton(self.radio_frame, text="Функция s", variable=self.table_mode,
                        value="s", command=self.switch_table).pack(anchor="w", padx=5, pady=2)
        ttk.Radiobutton(self.radio_frame, text="Функция y", variable=self.table_mode,
                        value="y", command=self.switch_table).pack(anchor="w", padx=5, pady=2)

        self.create_max_diff_frame()
        self.create_progress_and_controls()

        self.fig = plt.figure(figsize=(12, 4))
        self.ax1 = self.fig.add_subplot(1, 3, 1)
        self.ax2 = self.fig.add_subplot(1, 3, 2)
        self.ax3 = self.fig.add_subplot(1, 3, 3)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.table_frame = ttk.Frame(self.right_frame)
        self.table_frame.pack(fill="both", expand=True)

        self.a_tree = ttk.Treeview(self.table_frame, columns=(
            "Layer", "x", "a", "a*", "a_diff"), show="headings")
        self.a_tree.heading("Layer", text="Слой")
        self.a_tree.heading("x", text="x")
        self.a_tree.heading("a", text="a(x,t)")
        self.a_tree.heading("a*", text="a*(x,t)")
        self.a_tree.heading("a_diff", text="|a - a*|")

        self.a_tree.column("Layer", width=60, anchor="center")
        self.a_tree.column("x", width=60, anchor="center")
        self.a_tree.column("a", width=80, anchor="center")
        self.a_tree.column("a*", width=80, anchor="center")
        self.a_tree.column("a_diff", width=100, anchor="center")

        self.s_tree = ttk.Treeview(self.table_frame, columns=(
            "Layer", "x", "s", "s*", "s_diff"), show="headings")
        self.s_tree.heading("Layer", text="Слой")
        self.s_tree.heading("x", text="x")
        self.s_tree.heading("s", text="s(x,t)")
        self.s_tree.heading("s*", text="s*(x,t)")
        self.s_tree.heading("s_diff", text="|s - s*|")

        self.s_tree.column("Layer", width=60, anchor="center")
        self.s_tree.column("x", width=60, anchor="center")
        self.s_tree.column("s", width=80, anchor="center")
        self.s_tree.column("s*", width=80, anchor="center")
        self.s_tree.column("s_diff", width=100, anchor="center")

        self.y_tree = ttk.Treeview(self.table_frame, columns=(
            "Layer", "x", "y", "y*", "y_diff"), show="headings")
        self.y_tree.heading("Layer", text="Слой")
        self.y_tree.heading("x", text="x")
        self.y_tree.heading("y", text="y(x,t)")
        self.y_tree.heading("y*", text="y*(x,t)")
        self.y_tree.heading("y_diff", text="|y - y*|")

        self.y_tree.column("Layer", width=60, anchor="center")
        self.y_tree.column("x", width=60, anchor="center")
        self.y_tree.column("y", width=80, anchor="center")
        self.y_tree.column("y*", width=80, anchor="center")
        self.y_tree.column("y_diff", width=100, anchor="center")

        self.scrollbar = ttk.Scrollbar(self.table_frame, orient="vertical")
        self.a_tree.configure(yscrollcommand=self.scrollbar.set)
        self.s_tree.configure(yscrollcommand=self.scrollbar.set)
        self.y_tree.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")

        self.base_data = []
        self.control_data = []
        self.is_computing = False
        self.is_paused = False
        self.current_step = 0
        self.current_a1 = None
        self.current_s1 = None
        self.current_y1 = None
        self.current_a2 = None
        self.current_s2 = None
        self.current_y2 = None
        self.max_a_diff = 0.0
        self.max_s_diff = 0.0
        self.max_y_diff = 0.0
        self.switch_table()
        self.progress_queue = queue.Queue()

    def add_conditions_image(self):
        image_path = get_resource_path("main_conditions.png")
        if not os.path.exists(image_path):
            messagebox.showerror("Ошибка", f"Файл не найден: {image_path}")
            return
        try:
            image = Image.open(image_path)
            new_width = 300
            image = image.resize((new_width, int(
                new_width * image.height / image.width)), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image)
            image_label = ttk.Label(self.left_frame, image=self.photo)
            image_label.pack(anchor="nw", pady=2)
        except Exception as e:
            messagebox.showerror(
                "Ошибка", f"Не удалось загрузить изображение 'main_conditions.png': {e}")

    def create_parameter_inputs(self):
        params_frame = ttk.LabelFrame(
            self.left_frame, text="Параметры системы")
        params_frame.pack(fill="x", pady=2)
        columns_frame = ttk.Frame(params_frame)
        columns_frame.pack(fill="x", padx=5, pady=5)
        left_column = ttk.Frame(columns_frame)
        left_column.pack(side="left", fill="y", padx=(0, 10))
        right_column = ttk.Frame(columns_frame)
        right_column.pack(side="left", fill="y")

        params = [
            ("c", 1.0), ("\u03BC", 1.0), ("c\u2080", 1.0), ("\u03BD", 1.0),
            ("\u03B5", 1.0), ("d", 1.0), ("e", 1.0), ("f", 1.0),
            ("\u03B7", 1.0), ("D\u2090", 0.1), ("D\u209B", 0.1), ("D\u1D67", 0.1)
        ]
        mid = len(params) // 2
        for param, default in params[:mid]:
            frame = ttk.Frame(left_column)
            frame.pack(fill="x", pady=2)
            ttk.Label(frame, text=f"{param}:", width=5, font=(
                "Arial", 12)).pack(side="left", padx=(0, 2))
            entry = ttk.Entry(frame, width=10)
            entry.insert(0, str(default))
            entry.pack(side="left", padx=2)
            self.entries[param] = entry
            if param in ["D\u2090", "D\u209B", "D\u1D67"]:
                entry.bind(
                    '<KeyRelease>', lambda e: self.parameter_app.update_time_constraint())
        for param, default in params[mid:]:
            frame = ttk.Frame(right_column)
            frame.pack(fill="x", pady=2)
            ttk.Label(frame, text=f"{param}:", width=5, font=(
                "Arial", 12)).pack(side="left", padx=(0, 2))
            entry = ttk.Entry(frame, width=10)
            entry.insert(0, str(default))
            entry.pack(side="left", padx=2)
            self.entries[param] = entry
            if param in ["D\u2090", "D\u209B", "D\u1D67"]:
                entry.bind(
                    '<KeyRelease>', lambda e: self.parameter_app.update_time_constraint())

    def update_system_parameters(self, system_params):
        for param, value in system_params.items():
            self.entries[param].delete(0, tk.END)
            self.entries[param].insert(0, str(value))

    def create_buttons(self):
        self.compute_button = ttk.Button(
            self.left_frame, text="Рассчитать", command=self.start_computation)
        self.compute_button.pack(pady=2)
        self.plot_base_button = ttk.Button(
            self.left_frame, text="Показать график на базовой сетке", command=self.plot_base_grid)
        self.plot_base_button.pack(pady=2)
        self.plot_control_button = ttk.Button(
            self.left_frame, text="Показать график на контрольной сетке", command=self.plot_control_grid)
        self.plot_control_button.pack(pady=2)

    def create_max_diff_frame(self):
        self.max_diff_frame = ttk.LabelFrame(
            self.left_frame, text="Максимальные разности")
        self.max_diff_frame.pack(fill="x", pady=2)
        self.max_a_diff_label = ttk.Label(
            self.max_diff_frame, text="Max |a - a*| = Н/Д")
        self.max_a_diff_label.pack(anchor="w", pady=2)
        self.max_s_diff_label = ttk.Label(
            self.max_diff_frame, text="Max |s - s*| = Н/Д")
        self.max_s_diff_label.pack(anchor="w", pady=2)
        self.max_y_diff_label = ttk.Label(
            self.max_diff_frame, text="Max |y - y*| = Н/Д")
        self.max_y_diff_label.pack(anchor="w", pady=2)

    def create_progress_and_controls(self):
        progress_frame = ttk.LabelFrame(
            self.left_frame, text="Управление расчетом")
        progress_frame.pack(fill="x", pady=2)

        self.progressbar = ttk.Progressbar(
            progress_frame, orient="horizontal", mode="determinate", maximum=100)
        self.progressbar.pack(fill="x", padx=5, pady=2)
        self.progress_label = ttk.Label(
            progress_frame, text="Прогресс: 0% (0/0)")
        self.progress_label.pack(anchor="w", padx=5, pady=2)

        self.time_label = ttk.Label(progress_frame, text="Время расчета: Н/Д")
        self.time_label.pack(anchor="w", padx=5, pady=2)

        button_frame = ttk.Frame(progress_frame)
        button_frame.pack(fill="x", pady=2)
        self.stop_button = ttk.Button(
            button_frame, text="Остановить", command=self.stop_computation, state="disabled")
        self.stop_button.pack(side="left", padx=2)
        self.continue_button = ttk.Button(
            button_frame, text="Продолжить", command=self.continue_computation, state="disabled")
        self.continue_button.pack(side="left", padx=2)
        self.finish_button = ttk.Button(
            button_frame, text="Завершить", command=self.finish_computation, state="disabled")
        self.finish_button.pack(side="left", padx=2)

        self.save_button = ttk.Button(
            self.left_frame, text="Сохранить результаты", command=self.main_app.save_results)
        self.save_button.pack(pady=5)

    def create_layer_selector(self):
        self.layer_frame = ttk.LabelFrame(
            self.left_frame, text="Режим отображения графиков")
        self.layer_frame.pack(fill="x", pady=2)
        self.layer_mode = tk.StringVar(value="Последний слой")
        self.layer_selector = ttk.Combobox(
            self.layer_frame, textvariable=self.layer_mode, state="readonly")
        self.layer_selector['values'] = ("Последний слой", "Несколько слоев")
        self.layer_selector.pack(fill="x", padx=5, pady=5)
        self.layer_selector.bind(
            "<<ComboboxSelected>>", self.on_layer_mode_change)

    def on_layer_mode_change(self, event):
        if self.layer_mode.get() == "Несколько слоев" and (self.is_computing or self.is_paused):
            messagebox.showwarning(
                "Предупреждение", "Режим 'Несколько слоев' доступен только после полного завершения расчета!")
            self.layer_mode.set("Последний слой")
            self.layer_selector.set("Последний слой")
        self.update_table()
        if hasattr(self, 'last_plotted_grid'):
            if self.last_plotted_grid == "base":
                self.plot_base_grid()
            elif self.last_plotted_grid == "control":
                self.plot_control_grid()
        else:
            self.plot_base_grid()

    def switch_table(self):
        self.a_tree.pack_forget()
        self.s_tree.pack_forget()
        self.y_tree.pack_forget()
        mode = self.table_mode.get()
        if mode == "a":
            self.a_tree.pack(side="left", fill="both", expand=True)
            self.scrollbar.config(command=self.a_tree.yview)
        elif mode == "s":
            self.s_tree.pack(side="left", fill="both", expand=True)
            self.scrollbar.config(command=self.s_tree.yview)
        else:
            self.y_tree.pack(side="left", fill="both", expand=True)
            self.scrollbar.config(command=self.y_tree.yview)
        self.update_table()

    def show_completion_notification(self, message="Расчёт окончен"):
        toast = tk.Toplevel(self.left_frame)
        toast.overrideredirect(True)
        toast.attributes('-alpha', 0.9)

        frame = ttk.Frame(toast, style='Toast.TFrame')
        frame.pack()

        style = ttk.Style()
        style.configure('Toast.TFrame', background='#4CAF50')
        style.configure('Toast.TLabel', background='#4CAF50',
                        foreground='white', font=('Arial', 10))

        ttk.Label(frame, text=f"✔ {message}",
                  style='Toast.TLabel').pack(padx=10, pady=5)

        left_frame_coords = self.left_frame.winfo_rootx(), self.left_frame.winfo_rooty()
        left_frame_height = self.left_frame.winfo_height()
        toast_width = 150
        toast_height = 40
        x_pos = left_frame_coords[0] + 10
        y_pos = left_frame_coords[1] + left_frame_height - toast_height - 10
        toast.geometry(f"{toast_width}x{toast_height}+{x_pos}+{y_pos}")

        def fade_out(alpha=0.9, step=0.1):
            alpha -= step
            if alpha <= 0:
                toast.destroy()
            else:
                toast.attributes('-alpha', alpha)
                toast.after(50, fade_out, alpha, step)

        toast.after(2000, fade_out)

    def update_button_states(self):
        if self.is_computing:
            self.compute_button["state"] = "disabled"
            self.stop_button["state"] = "normal"
            self.continue_button["state"] = "disabled"
            self.finish_button["state"] = "disabled"
            self.plot_base_button["state"] = "disabled"
            self.plot_control_button["state"] = "disabled"
        elif self.is_paused:
            self.compute_button["state"] = "disabled"
            self.stop_button["state"] = "disabled"
            self.continue_button["state"] = "normal"
            self.finish_button["state"] = "normal"
            self.plot_base_button["state"] = "normal" if self.base_data else "disabled"
            self.plot_control_button["state"] = "normal" if self.control_data else "disabled"
        else:
            self.compute_button["state"] = "normal"
            self.stop_button["state"] = "disabled"
            self.continue_button["state"] = "disabled"
            self.finish_button["state"] = "disabled"
            self.plot_base_button["state"] = "normal" if self.base_data else "disabled"
            self.plot_control_button["state"] = "normal" if self.control_data else "disabled"

    def check_progress_queue(self):
        try:
            while True:
                progress_data = self.progress_queue.get_nowait()
                progress, step, m1 = progress_data
                self.progressbar["value"] = progress
                self.progress_label.config(
                    text=f"Прогресс: {progress:.1f}% ({step}/{m1})")
        except queue.Empty:
            pass
        if self.is_computing:
            self.main_frame.after(100, self.check_progress_queue)
        self.main_frame.update_idletasks()

    def start_computation(self):
        if self.is_computing or self.is_paused:
            return
        self.is_computing = True
        self.is_paused = False
        self.current_step = 0
        self.base_data = []
        self.control_data = []
        self.max_a_diff = 0.0
        self.max_s_diff = 0.0
        self.max_y_diff = 0.0
        self.layer_mode.set("Последний слой")
        self.layer_selector.set("Последний слой")
        self.update_button_states()
        while not self.progress_queue.empty():
            self.progress_queue.get()
        computation_thread = threading.Thread(target=self.compute_solution)
        computation_thread.daemon = True
        computation_thread.start()
        self.main_frame.after(100, self.check_progress_queue)

    def stop_computation(self):
        if not self.is_computing:
            return
        self.is_computing = False
        self.is_paused = True
        self.update_button_states()

        def update_ui_on_stop():
            if self.current_step > 0 and self.current_step not in [data["layer"] for data in self.base_data]:
                self.base_data.append({
                    "layer": self.current_step,
                    "x": self.base_data[0]["x"].copy(),
                    "a": self.current_a1.copy(),
                    "s": self.current_s1.copy(),
                    "y": self.current_y1.copy()
                })
                control_step = self.current_step * 4
                self.control_data.append({
                    "layer": control_step,
                    "x": self.control_data[0]["x"].copy(),
                    "a": self.current_a2.copy(),
                    "s": self.current_s2.copy(),
                    "y": self.current_y2.copy()
                })
            self.update_table()
            self.plot_base_grid()
            end_time = time.time()
            elapsed_time = end_time - self.start_time
            self.time_label.config(
                text=f"Время расчета: {elapsed_time:.2f} сек")
            self.show_completion_notification("Расчёт приостановлен")
        self.main_frame.after(0, update_ui_on_stop)

    def continue_computation(self):
        if not self.is_paused:
            return
        self.is_computing = True
        self.is_paused = False
        self.update_button_states()
        while not self.progress_queue.empty():
            self.progress_queue.get()
        computation_thread = threading.Thread(
            target=self.compute_solution, args=(self.current_step,))
        computation_thread.daemon = True
        computation_thread.start()
        self.main_frame.after(100, self.check_progress_queue)

    def finish_computation(self):
        if not self.is_paused:
            return
        self.is_computing = False
        self.is_paused = False
        self.update_button_states()
        self.show_completion_notification("Расчёт завершен")
        self.update_table()
        self.plot_base_grid()

    def validate_system_parameters(self):
        try:
            params = ["c", "\u03BC", "c\u2080", "\u03BD", "\u03B5",
                      "d", "e", "f", "\u03B7", "D\u2090", "D\u209B", "D\u1D67"]
            for param in params:
                value = float(self.entries[param].get())
                if value < 0 and param in ["D\u2090", "D\u209B", "D\u1D67"]:
                    raise ValueError(
                        f"Коэффициент диффузии {param} должен быть неотрицательным")
            return True
        except ValueError as e:
            messagebox.showerror(
                "Ошибка ввода", f"Некорректные параметры системы: {str(e)}")
            return False

    def compute_solution(self, continue_from_step=0):
        try:
            if not self.parameter_app.initial_conditions:
                def update_ui_on_error():
                    self.is_computing = False
                    self.is_paused = False
                    self.update_button_states()
                    messagebox.showerror(
                        "Ошибка", "Сначала задайте начальные условия на первой вкладке!")
                self.main_frame.after(0, update_ui_on_error)
                return

            if not self.validate_system_parameters() or not self.parameter_app.validate_parameters():
                def update_ui_on_error():
                    self.is_computing = False
                    self.is_paused = False
                    self.update_button_states()
                self.main_frame.after(0, update_ui_on_error)
                return

            # Параметры системы
            c = float(self.entries["c"].get())
            mu = float(self.entries["\u03BC"].get())
            c_0 = float(self.entries["c\u2080"].get())
            nu = float(self.entries["\u03BD"].get())
            epsilon = float(self.entries["\u03B5"].get())
            d = float(self.entries["d"].get())
            e = float(self.entries["e"].get())
            f = float(self.entries["f"].get())
            D_a = float(self.entries["D\u2090"].get())
            D_s = float(self.entries["D\u209B"].get())
            D_y = float(self.entries["D\u1D67"].get())
            eta = float(self.entries["\u03B7"].get())

            n1 = int(self.parameter_app.entries["n (сетка по x)"].get()) + 1
            T = float(self.parameter_app.entries["T (время)"].get())
            m1 = int(self.parameter_app.entries["m (сетка по t)"].get())
            n2 = (2 * n1) - 1
            m2 = 4 * m1

            if continue_from_step == 0:
                a_0 = np.array([float(self.parameter_app.entries["a_0"][0].get()),
                                float(
                                    self.parameter_app.entries["a_0"][1].get()),
                                float(self.parameter_app.entries["a_0"][2].get())])
                a_1 = np.array([float(self.parameter_app.entries["a_1"][0].get()),
                                float(
                                    self.parameter_app.entries["a_1"][1].get()),
                                float(self.parameter_app.entries["a_1"][2].get())])
                a_2 = np.array([float(self.parameter_app.entries["a_2"][0].get()),
                                float(
                                    self.parameter_app.entries["a_2"][1].get()),
                                float(self.parameter_app.entries["a_2"][2].get())])
                a_3 = np.array([float(self.parameter_app.entries["a_3"][0].get()),
                                float(
                                    self.parameter_app.entries["a_3"][1].get()),
                                float(self.parameter_app.entries["a_3"][2].get())])
                b_1 = int(self.parameter_app.entries["b_1"].get())
                b_2 = int(self.parameter_app.entries["b_2"].get())
                b_3 = int(self.parameter_app.entries["b_3"].get())

                x1 = self.parameter_app.initial_conditions['coarse']['x']
                a1 = self.parameter_app.initial_conditions['coarse']['a'].copy(
                )
                s1 = self.parameter_app.initial_conditions['coarse']['s'].copy(
                )
                y1 = self.parameter_app.initial_conditions['coarse']['y'].copy(
                )

                x2 = np.array([i * 1.0 / (n2-1) for i in range(n2)])
                a2 = a_0[0] + a_1[0]*np.cos(b_1*np.pi*x2) + a_2[0] * \
                    np.cos(b_2*np.pi*x2) + a_3[0]*np.cos(b_3*np.pi*x2)
                s2 = a_0[1] + a_1[1]*np.cos(b_1*np.pi*x2) + a_2[1] * \
                    np.cos(b_2*np.pi*x2) + a_3[1]*np.cos(b_3*np.pi*x2)
                y2 = a_0[2] + a_1[2]*np.cos(b_1*np.pi*x2) + a_2[2] * \
                    np.cos(b_2*np.pi*x2) + a_3[2]*np.cos(b_3*np.pi*x2)
            else:
                x1 = self.base_data[0]["x"]
                a1 = self.current_a1.copy()
                s1 = self.current_s1.copy()
                y1 = self.current_y1.copy()
                x2 = self.control_data[0]["x"]
                a2 = self.current_a2.copy()
                s2 = self.current_s2.copy()
                y2 = self.current_y2.copy()

            L = 1.0
            h1 = L / (n1 - 1)
            tau1 = T / m1
            h2 = L / (n2 - 1)
            tau2 = T / m2

            max_D = max(D_a, D_s, D_y)
            # Используем явную схему с контролем шага по времени (условие Куранта)
            if tau1 >= h1**2 / (2 * max_D) or tau2 >= h2**2 / (2 * max_D):
                def show_warning():
                    self.is_computing = False
                    self.is_paused = False
                    self.update_button_states()
                    messagebox.showwarning(
                        "Предупреждение", "Шаг по времени слишком велик! Возможна неустойчивость.")
                self.main_frame.after(0, show_warning)
                return

            a1_new = np.zeros(n1)
            s1_new = np.zeros(n1)
            y1_new = np.zeros(n1)
            a2_new = np.zeros(n2)
            s2_new = np.zeros(n2)
            y2_new = np.zeros(n2)

            layers_to_save1 = [0] + [m1 // 5 * i for i in range(1, 6)]
            layers_to_save2 = [0] + [m2 // 5 * i for i in range(1, 6)]
            if continue_from_step == 0:
                self.base_data = []
                self.control_data = []

            last_progress = -1

            self.start_time = time.time()
            start_time = time.time()

            for step in range(continue_from_step, m1 + 1):
                if not self.is_computing:
                    self.current_a1 = a1.copy()
                    self.current_s1 = s1.copy()
                    self.current_y1 = y1.copy()
                    self.current_a2 = a2.copy()
                    self.current_s2 = s2.copy()
                    self.current_y2 = y2.copy()
                    self.current_step = step
                    # Сохраняем последний слой, если он еще не сохранен
                    if step not in [data["layer"] for data in self.base_data]:
                        self.base_data.append({
                            "layer": step,
                            "x": x1.copy(),
                            "a": a1.copy(),
                            "s": s1.copy(),
                            "y": y1.copy()
                        })
                        control_step = step * 4
                        self.control_data.append({
                            "layer": control_step,
                            "x": x2.copy(),
                            "a": a2.copy(),
                            "s": s2.copy(),
                            "y": y2.copy()
                        })

                    def update_ui_on_stop():
                        self.update_table()
                        self.plot_base_grid()
                        self.update_button_states()
                        self.show_completion_notification(
                            "Расчёт приостановлен")
                    self.main_frame.after(0, update_ui_on_stop)
                    return

                if step in layers_to_save1:
                    self.base_data.append({"layer": step, "x": x1.copy(),
                                           "a": a1.copy(), "s": s1.copy(), "y": y1.copy()})
                control_step = step * 4
                if control_step in layers_to_save2:
                    self.control_data.append({"layer": control_step, "x": x2.copy(),
                                              "a": a2.copy(), "s": s2.copy(), "y": y2.copy()})

                if step > 0:
                    a_control = a2[::2]
                    s_control = s2[::2]
                    y_control = y2[::2]
                    a_diff = np.abs(a1 - a_control)
                    s_diff = np.abs(s1 - s_control)
                    y_diff = np.abs(y1 - y_control)
                    self.max_a_diff = max(self.max_a_diff, np.max(a_diff))
                    self.max_s_diff = max(self.max_s_diff, np.max(s_diff))
                    self.max_y_diff = max(self.max_y_diff, np.max(y_diff))

                if step < m1:
                    # Явная конечно-разностная схема для внутренних узлов
                    # du/dt = c*u²*v - μ*u + D∇²u
                    a1_new[1:-1] = a1[1:-1] + tau1 * (
                        c * a1[1:-1]**2 * s1[1:-1] - mu * a1[1:-1] +
                        D_a * (a1[2:] - 2*a1[1:-1] + a1[:-2]) / h1**2
                    )
                    s1_new[1:-1] = s1[1:-1] + tau1 * (
                        c_0 - c * a1[1:-1]**2 * s1[1:-1] - nu * s1[1:-1] -
                        epsilon * s1[1:-1] * y1[1:-1] +
                        D_s * (s1[2:] - 2*s1[1:-1] + s1[:-2]) / h1**2
                    )
                    y1_new[1:-1] = y1[1:-1] + tau1 * (
                        d * a1[1:-1] - e * y1[1:-1] +
                        (eta * y1[1:-1]**2) / (1 + f * y1[1:-1]**2) +
                        D_y * (y1[2:] - 2*y1[1:-1] + y1[:-2]) / h1**2
                    )

                    a1_new[0] = a1[0] + tau1 * (
                        c * a1[0]**2 * s1[0] - mu * a1[0] +
                        D_a * 2 * (a1[1] - a1[0]) / h1**2
                    )
                    s1_new[0] = s1[0] + tau1 * (
                        c_0 - c * a1[0]**2 * s1[0] - nu * s1[0] -
                        epsilon * s1[0] * y1[0] +
                        D_s * 2 * (s1[1] - s1[0]) / h1**2
                    )
                    y1_new[0] = y1[0] + tau1 * (
                        d * a1[0] - e * y1[0] +
                        (eta * y1[0]**2) / (1 + f * y1[0]**2) +
                        D_y * 2 * (y1[1] - y1[0]) / h1**2
                    )
                    a1_new[n1-1] = a1[n1-1] + tau1 * (
                        c * a1[n1-1]**2 * s1[n1-1] - mu * a1[n1-1] +
                        D_a * 2 * (a1[n1-2] - a1[n1-1]) / h1**2
                    )
                    s1_new[n1-1] = s1[n1-1] + tau1 * (
                        c_0 - c * a1[n1-1]**2 * s1[n1-1] - nu * s1[n1-1] -
                        epsilon * s1[n1-1] * y1[n1-1] +
                        D_s * 2 * (s1[n1-2] - s1[n1-1]) / h1**2
                    )
                    y1_new[n1-1] = y1[n1-1] + tau1 * (
                        d * a1[n1-1] - e * y1[n1-1] +
                        (eta * y1[n1-1]**2) / (1 + f * y1[n1-1]**2) +
                        D_y * 2 * (y1[n1-2] - y1[n1-1]) / h1**2
                    )

                    a1[:] = a1_new[:]
                    s1[:] = s1_new[:]
                    y1[:] = y1_new[:]

                for substep in range(4 if step < m1 else 0):
                    a2_new[1:-1] = a2[1:-1] + tau2 * (
                        c * a2[1:-1]**2 * s2[1:-1] - mu * a2[1:-1] +
                        D_a * (a2[2:] - 2*a2[1:-1] + a2[:-2]) / h2**2
                    )
                    s2_new[1:-1] = s2[1:-1] + tau2 * (
                        c_0 - c * a2[1:-1]**2 * s2[1:-1] - nu * s2[1:-1] -
                        epsilon * s2[1:-1] * y2[1:-1] +
                        D_s * (s2[2:] - 2*s2[1:-1] + s2[:-2]) / h2**2
                    )
                    y2_new[1:-1] = y2[1:-1] + tau2 * (
                        d * a2[1:-1] - e * y2[1:-1] +
                        (eta * y2[1:-1]**2) / (1 + f * y2[1:-1]**2) +
                        D_y * (y2[2:] - 2*y2[1:-1] + y2[:-2]) / h2**2
                    )

                    a2_new[0] = a2[0] + tau2 * (
                        c * a2[0]**2 * s2[0] - mu * a2[0] +
                        D_a * 2 * (a2[1] - a2[0]) / h2**2
                    )
                    s2_new[0] = s2[0] + tau2 * (
                        c_0 - c * a2[0]**2 * s2[0] - nu * s2[0] -
                        epsilon * s2[0] * y2[0] +
                        D_s * 2 * (s2[1] - s2[0]) / h2**2
                    )
                    y2_new[0] = y2[0] + tau2 * (
                        d * a2[0] - e * y2[0] +
                        (eta * y2[0]**2) / (1 + f * y2[0]**2) +
                        D_y * 2 * (y2[1] - y2[0]) / h2**2
                    )
                    a2_new[n2-1] = a2[n2-1] + tau2 * (
                        c * a2[n2-1]**2 * s2[n2-1] - mu * a2[n2-1] +
                        D_a * 2 * (a2[n2-2] - a2[n2-1]) / h2**2
                    )
                    s2_new[n2-1] = s2[n2-1] + tau2 * (
                        c_0 - c * a2[n2-1]**2 * s2[n2-1] - nu * s2[n2-1] -
                        epsilon * s2[n2-1] * y2[n2-1] +
                        D_s * 2 * (s2[n2-2] - s2[n2-1]) / h2**2
                    )
                    y2_new[n2-1] = y2[n2-1] + tau2 * (
                        d * (eta * y2[n2-1]**2) / (1 + f * y2[n2-1]**2) +
                        D_y * 2 * (y2[n2-2] - y2[n2-1]) / h2**2
                    )

                    a2[:] = a2_new[:]
                    s2[:] = s2_new[:]
                    y2[:] = y2_new[:]

                # Обновление прогрессбара каждые 1% прогресса или на последнем шаге
                progress = (step / m1) * 100
                if int(progress) > last_progress or step == m1:
                    last_progress = int(progress)
                    self.progress_queue.put((progress, step, m1))
                    time.sleep(0.01)  # Небольшая задержка для освобождения GIL

            def finalize_computation():
                self.is_computing = False
                self.is_paused = False
                self.update_button_states()
                self.show_completion_notification("Расчёт завершён")
                self.max_a_diff_label.config(
                    text=f"Max |a - a*|: {self.max_a_diff:.4f}")
                self.max_s_diff_label.config(
                    text=f"Max |s - s*|: {self.max_s_diff:.4f}")
                self.max_y_diff_label.config(
                    text=f"Max |y - y*|: {self.max_y_diff:.4f}")
                end_time = time.time()
                elapsed_time = end_time - start_time
                self.time_label.config(
                    text=f"Время расчета: {elapsed_time:.2f} сек")
                self.update_table()
                self.plot_base_grid()
            self.main_frame.after(0, finalize_computation)

        except ValueError as e:
            def handle_value_error():
                self.progressbar["value"] = 0
                self.progress_label.config(text=f"Прогресс: 0% (0/{m1})")
                self.is_computing = False
                self.is_paused = False
                self.update_button_states()
                end_time = time.time()
                elapsed_time = end_time - self.start_time
                self.time_label.config(
                    text=f"Время расчета: {elapsed_time:.2f} сек")
                messagebox.showerror(
                    "Проверьте введенные параметры: все значения должны быть числами!")
            self.main_frame.after(0, handle_value_error)
        except Exception as e:
            def handle_general_error():
                self.progressbar["value"] = 0
                self.progress_label.config(text=f"Прогресс: 0% (0/{m1})")
                self.is_computing = False
                self.is_paused = False
                self.update_button_states()
                end_time = time.time()
                elapsed_time = end_time - self.start_time
                self.time_label.config(
                    text=f"Время расчета: {elapsed_time:.2f} сек")
                messagebox.showerror("Ошибка", f"Произошла ошибка: {str(e)}")
                self.main_frame.after(0, handle_general_error)

    def plot_base_grid(self):
        self.last_plotted_grid = "base"
        self.plot_grid(self.base_data, "Базовая сетка")

    def plot_control_grid(self):
        self.last_plotted_grid = "control"
        self.plot_grid(self.control_data, "Контрольная сетка")

    def plot_grid(self, data, title_prefix):
        limits = self.parameter_app.get_axis_limits()
        if not limits:
            return

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'black']
        mode = self.layer_mode.get()

        if mode == "Последний слой":
            layer_data = data[-1]
            x_vals = layer_data["x"]
            a_vals = layer_data["a"]
            s_vals = layer_data["s"]
            y_vals = layer_data["y"]
            layer = layer_data["layer"]

            self.ax1.plot(x_vals, a_vals,
                          label=f'Слой {layer}', color=colors[1])
            self.ax2.plot(x_vals, s_vals,
                          label=f'Слой {layer}', color=colors[2])
            self.ax3.plot(x_vals, y_vals,
                          label=f'Слой {layer}', color=colors[0])
        else:
            for idx, layer_data in enumerate(data):
                x_vals = layer_data["x"]
                a_vals = layer_data["a"]
                s_vals = layer_data["s"]
                y_vals = layer_data["y"]
                layer = layer_data["layer"]

                self.ax1.plot(
                    x_vals, a_vals, label=f'Слой {layer}', color=colors[idx % len(colors)])
                self.ax2.plot(
                    x_vals, s_vals, label=f'Слой {layer}', color=colors[idx % len(colors)])
                self.ax3.plot(
                    x_vals, y_vals, label=f'Слой {layer}', color=colors[idx % len(colors)])

        self.ax1.set_xlabel('x')
        self.ax1.set_ylabel('a(x, t)')
        self.ax1.set_ylim(limits['a_min'], limits['a_max'])
        self.ax1.grid(True)
        self.ax1.legend()
        self.ax1.set_title(f"{title_prefix}: a(x, t)")

        self.ax2.set_xlabel('x')
        self.ax2.set_ylabel('s(x, t)')
        self.ax2.set_ylim(limits['s_min'], limits['s_max'])
        self.ax2.grid(True)
        self.ax2.legend()
        self.ax2.set_title(f"{title_prefix}: s(x, t)")

        self.ax3.set_xlabel('x')
        self.ax3.set_ylabel('y(x, t)')
        self.ax3.set_ylim(limits['y_min'], limits['y_max'])
        self.ax3.grid(True)
        self.ax3.legend()
        self.ax3.set_title(f"{title_prefix}: y(x, t)")

        self.fig.tight_layout()
        self.canvas.draw()

    def save_solution_plots(self, save_dir):
        if not self.base_data:
            messagebox.showerror(
                "Ошибка", "Нет данных численного решения для сохранения!")
            return

        limits = self.parameter_app.get_axis_limits()
        if not limits:
            return

        fig = plt.figure(figsize=(12, 4))
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        layer_data = self.base_data[-1]
        x_vals = layer_data["x"]
        a_vals = layer_data["a"]
        s_vals = layer_data["s"]
        y_vals = layer_data["y"]
        layer = layer_data["layer"]

        ax1.plot(x_vals, a_vals, label=f'Слой {layer}', color='blue')
        ax1.set_xlabel('x')
        ax1.set_ylabel('a(x, t)')
        ax1.set_ylim(limits['a_min'], limits['a_max'])
        ax1.grid(True)
        ax1.legend()
        ax1.set_title(f"Базовая сетка: a(x, t)")

        ax2.plot(x_vals, s_vals, label=f'Слой {layer}', color='green')
        ax2.set_xlabel('x')
        ax2.set_ylabel('s(x, t)')
        ax2.set_ylim(limits['s_min'], limits['s_max'])
        ax2.grid(True)
        ax2.legend()
        ax2.set_title(f"Базовая сетка: s(x, t)")

        ax3.plot(x_vals, y_vals, label=f'Слой {layer}', color='red')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y(x, t)')
        ax3.set_ylim(limits['y_min'], limits['y_max'])
        ax3.grid(True)
        ax3.legend()
        ax3.set_title(f"Базовая сетка: y(x, t)")

        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "base_grid_last_layer.png"),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=(12, 4))
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        colors = ['red', 'blue', 'green', 'orange', 'purple', 'black']
        for idx, layer_data in enumerate(self.base_data):
            x_vals = layer_data["x"]
            a_vals = layer_data["a"]
            s_vals = layer_data["s"]
            y_vals = layer_data["y"]
            layer = layer_data["layer"]

            ax1.plot(x_vals, a_vals,
                     label=f'Слой {layer}', color=colors[idx % len(colors)])
            ax2.plot(x_vals, s_vals,
                     label=f'Слой {layer}', color=colors[idx % len(colors)])
            ax3.plot(x_vals, y_vals,
                     label=f'Слой {layer}', color=colors[idx % len(colors)])

        ax1.set_xlabel('x')
        ax1.set_ylabel('a(x, t)')
        ax1.set_ylim(limits['a_min'], limits['a_max'])
        ax1.grid(True)
        ax1.legend()
        ax1.set_title(f"Базовая сетка: a(x, t)")

        ax2.set_xlabel('x')
        ax2.set_ylabel('s(x, t)')
        ax2.set_ylim(limits['s_min'], limits['s_max'])
        ax2.grid(True)
        ax2.legend()
        ax2.set_title(f"Базовая сетка: s(x, t)")

        ax3.set_xlabel('x')
        ax3.set_ylabel('y(x, t)')
        ax3.set_ylim(limits['y_min'], limits['y_max'])
        ax3.grid(True)
        ax3.legend()
        ax3.set_title(f"Базовая сетка: y(x, t)")

        fig.tight_layout()
        fig.savefig(os.path.join(
            save_dir, "base_grid_multiple_layers.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

    def update_table(self):
        mode = self.table_mode.get()
        if not self.base_data or not self.control_data:
            return

        display_last_layer = self.layer_mode.get() == "Последний слой"
        x_base = self.base_data[0]["x"]

        n = int(self.parameter_app.entries["n (сетка по x)"].get())
        x_precision = math.ceil(math.log10(n)) + 1 if n > 2 else 1

        if mode == "a":
            for item in self.a_tree.get_children():
                self.a_tree.delete(item)
            if display_last_layer:
                base_layer = self.base_data[-1]
                control_layer = self.control_data[-1]
                layer_display = f"{base_layer['layer']}/{control_layer['layer']}"
                a_control = control_layer["a"][::2]
                a_diff = np.abs(base_layer["a"] - a_control)
                for i in range(len(x_base)):
                    self.a_tree.insert("", "end", values=(
                        layer_display, f"{x_base[i]:.{x_precision}f}",
                        f"{base_layer['a'][i]:.6f}", f"{a_control[i]:.6f}", f"{a_diff[i]:.6f}"))
            else:
                for base_layer, control_layer in zip(self.base_data, self.control_data):
                    layer_display = f"{base_layer['layer']}/{control_layer['layer']}"
                    a_control = control_layer["a"][::2]
                    a_diff = np.abs(base_layer["a"] - a_control)
                    for i in range(len(x_base)):
                        self.a_tree.insert("", "end", values=(
                            layer_display, f"{x_base[i]:.{x_precision}f}",
                            f"{base_layer['a'][i]:.6f}", f"{a_control[i]:.6f}", f"{a_diff[i]:.6f}"))

        elif mode == "s":
            for item in self.s_tree.get_children():
                self.s_tree.delete(item)
            if display_last_layer:
                base_layer = self.base_data[-1]
                control_layer = self.control_data[-1]
                layer_display = f"{base_layer['layer']}/{control_layer['layer']}"
                s_control = control_layer["s"][::2]
                s_diff = np.abs(base_layer["s"] - s_control)
                for i in range(len(x_base)):
                    self.s_tree.insert("", "end", values=(
                        layer_display, f"{x_base[i]:.{x_precision}f}",
                        f"{base_layer['s'][i]:.6f}", f"{s_control[i]:.6f}", f"{s_diff[i]:.6f}"))
            else:
                for base_layer, control_layer in zip(self.base_data, self.control_data):
                    layer_display = f"{base_layer['layer']}/{control_layer['layer']}"
                    s_control = control_layer["s"][::2]
                    s_diff = np.abs(base_layer["s"] - s_control)
                    for i in range(len(x_base)):
                        self.s_tree.insert("", "end", values=(
                            layer_display, f"{x_base[i]:.{x_precision}f}",
                            f"{base_layer['s'][i]:.6f}", f"{s_control[i]:.6f}", f"{s_diff[i]:.6f}"))

        else:
            for item in self.y_tree.get_children():
                self.y_tree.delete(item)
            if display_last_layer:
                base_layer = self.base_data[-1]
                control_layer = self.control_data[-1]
                layer_display = f"{base_layer['layer']}/{control_layer['layer']}"
                y_control = control_layer["y"][::2]
                y_diff = np.abs(base_layer["y"] - y_control)
                for i in range(len(x_base)):
                    self.y_tree.insert("", "end", values=(
                        layer_display, f"{x_base[i]:.{x_precision}f}",
                        f"{base_layer['y'][i]:.6f}", f"{y_control[i]:.6f}", f"{y_diff[i]:.6f}"))
            else:
                for base_layer, control_layer in zip(self.base_data, self.control_data):
                    layer_display = f"{base_layer['layer']}/{control_layer['layer']}"
                    y_control = control_layer["y"][::2]
                    y_diff = np.abs(base_layer["y"] - y_control)
                    for i in range(len(x_base)):
                        self.y_tree.insert("", "end", values=(
                            layer_display, f"{x_base[i]:.{x_precision}f}",
                            f"{base_layer['y'][i]:.6f}", f"{y_control[i]:.6f}", f"{y_diff[i]:.6f}"))


class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title(
            "Численное решение системы реакции диффузии Майнхардта")

        def on_closing():
            plt.close('all')
            self.root.destroy()

        self.root.protocol("WM_DELETE_WINDOW", on_closing)

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)

        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="Начальные условия")

        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="Численное решение")

        self.numerical_app = NumericalSolutionApp(self.tab2, self)
        self.parameter_app = ParameterApp(
            self.tab1, self.numerical_app.update_system_parameters)
        self.numerical_app.parameter_app = self.parameter_app

    def save_results(self):
        if not self.numerical_app.base_data:
            messagebox.showerror(
                "Ошибка", "Нет данных для сохранения. Сначала выполните расчет!")
            return

        folder_name = simpledialog.askstring(
            "Сохранение результатов", "Введите название результата:", parent=self.root)
        if not folder_name:
            messagebox.showwarning(
                "Предупреждение", "Имя папки не указано. Сохранение отменено.")
            return

        save_dir = get_results_dir() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        self.parameter_app.save_initial_conditions_plot(save_dir)
        self.numerical_app.save_solution_plots(save_dir)

        # Сохранение параметров в текстовый файл
        try:
            with open(os.path.join(save_dir, "parameters.txt"), 'w', encoding='utf-8') as f:
                f.write("Начальные условия:\n")
                for key in ["a_0", "a_1", "a_2", "a_3"]:
                    a, s, y = self.parameter_app.entries[key]
                    f.write(f"{key}: a={a.get()}, s={s.get()}, y={y.get()}\n")
                for i in range(1, 4):
                    b = self.parameter_app.entries[f"b_{i}"].get()
                    f.write(f"b_{i}: {b}\n")

                f.write("\nПараметры сетки:\n")
                for param in ["n (сетка по x)", "m (сетка по t)", "T (время)"]:
                    value = self.parameter_app.entries[param].get()
                    f.write(f"{param}: {value}\n")

                f.write("\nПараметры системы:\n")
                for param in ["c", "\u03BC", "c\u2080", "\u03BD", "\u03B5", "d", "e", "f", "\u03B7", "D\u2090", "D\u209B", "D\u1D67"]:
                    value = self.numerical_app.entries[param].get()
                    f.write(f"{param}: {value}\n")

            messagebox.showinfo(
                "Успех", f"Результаты и параметры сохранены в {save_dir}")
        except Exception as e:
            messagebox.showerror(
                "Ошибка", f"Не удалось сохранить параметры: {str(e)}")


def read_parameters(filename):
    """
    Читает параметры из файла parameters.txt, созданного при сохранении результатов.
    Возвращает словарь с начальными условиями, параметрами сетки и системы.
    """
    params = {}
    current_section = None
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line in ["Начальные условия:", "Параметры сетки:", "Параметры системы:"]:
                current_section = line
                continue
            if current_section == "Начальные условия:":
                if line.startswith("a_"):
                    # Пример: a_0: a=1.0, s=0.5, y=0.2
                    match = re.match(
                        r"(a_\d+)\:\s*a=([\d.]+),\s*s=([\d.]+),\s*y=([\d.]+)", line)
                    if match:
                        key = match.group(1)
                        a_val = float(match.group(2))
                        s_val = float(match.group(3))
                        y_val = float(match.group(4))
                        params[key] = [a_val, s_val, y_val]
                    else:
                        raise ValueError(f"Некорректный формат строки: {line}")
                elif line.startswith("b_"):
                    # Пример: b_1: 1
                    key, value = line.split(":")
                    params[key.strip()] = int(value.strip())
            elif current_section == "Параметры сетки:":
                # Пример: n (сетка по x): 10
                key, value = line.split(":")
                key = key.strip()
                value = value.strip()
                params[key] = int(value) if key in [
                    "n (сетка по x)", "m (сетка по t)"] else float(value)
            elif current_section == "Параметры системы:":
                # Пример: c: 16.67
                key, value = line.split(":")
                params[key.strip()] = float(value.strip())
    # Проверка наличия всех необходимых параметров
    required_initial = ["a_0", "a_1", "a_2", "a_3", "b_1", "b_2", "b_3"]
    required_grid = ["n (сетка по x)", "m (сетка по t)", "T (время)"]
    required_system = ["c", "μ", "c₀", "ν", "ε",
                       "d", "e", "f", "η", "Dₐ", "Dₛ", "Dᵧ"]
    missing = []
    missing.extend([key for key in required_initial if key not in params])
    missing.extend([key for key in required_grid if key not in params])
    missing.extend([key for key in required_system if key not in params])
    if missing:
        raise ValueError(f"Отсутствуют необходимые параметры: {missing}")
    return params


if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
