import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from .config.parameters import SYSTEM_PARAMS, EQUILIBRIUM_PARAMS, SPATIAL_PARAMS, INITIAL_PARAMS
from .core.stability import calculate_stability_matrix
from .core.newton import newton_method
from .core.graphs import find_graphs

# Создаем главное окно
root = tk.Tk()
root.title("Исследование системы реакции диффузии Мейнхардта | Кандрушин И.Б.")
root.geometry("1280x900")


def on_closing():
    plt.close('all')
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_closing)

# Создаем вкладки
notebook = ttk.Notebook(root)
notebook.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

# Функция для создания вкладки "Подбор параметров"
def create_parameters_tab(tab_name):
    tab = ttk.Frame(notebook)
    notebook.add(tab, text=tab_name)

    button_frame = ttk.Frame(tab)
    button_frame.grid(row=0, column=0, pady=5, sticky="ew")

    calc_button = ttk.Button(button_frame, text="Рассчитать")
    calc_button.grid(row=0, column=0, padx=5)

    graph_button = ttk.Button(button_frame, text="Показать график")
    graph_button.grid(row=0, column=1, padx=5)

    left_frame = ttk.LabelFrame(tab, text="Параметры системы", padding=10)
    left_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

    entries = {}
    for idx, (param, value) in enumerate(SYSTEM_PARAMS):
        col = idx // 6
        row = idx % 6
        label = ttk.Label(left_frame, text=f"{param}:")
        label.grid(row=row, column=col * 2, padx=5, pady=5, sticky="e")
        entry = ttk.Entry(left_frame, width=12)
        entry.insert(0, str(value))
        entry.grid(row=row, column=col * 2 + 1, padx=5, pady=5)
        entries[param] = entry

    equilibrium_frame = ttk.LabelFrame(tab, text="Равновесие", padding=10)
    equilibrium_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

    equilibrium_entries = {}
    for idx, (param, value) in enumerate(EQUILIBRIUM_PARAMS):
        label = ttk.Label(equilibrium_frame, text=f"{param}:")
        label.grid(row=idx, column=0, padx=5, pady=5, sticky="e")
        entry = ttk.Entry(equilibrium_frame, width=10)
        entry.insert(0, str(value))
        entry.grid(row=idx, column=1, padx=5, pady=5)
        equilibrium_entries[param] = entry

    spatial_frame = ttk.LabelFrame(
        tab, text="Характеристики пространственной области и гармоники", padding=10)
    spatial_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")

    spatial_entries = {}
    for idx, (param, value) in enumerate(SPATIAL_PARAMS):
        label = ttk.Label(spatial_frame, text=f"{param}:")
        label.grid(row=idx, column=0, padx=5, pady=5, sticky="e")
        entry = ttk.Entry(spatial_frame, width=10)
        entry.insert(0, str(value))
        entry.grid(row=idx, column=1, padx=5, pady=5)
        spatial_entries[param] = entry

    right_frame = ttk.LabelFrame(tab, text="Вывод данных", padding=10)
    right_frame.grid(row=0, column=1, rowspan=4,
                     padx=10, pady=10, sticky="nsew")

    output_text = tk.Text(right_frame, height=50, width=150)
    output_text.grid(row=0, column=0, padx=5, pady=5)

    # Функция для расчета матрицы
    def calculate_matrix():
        try:
            params = {
                'c': float(entries["с"].get()),
                'mu': float(entries["μ"].get()),
                'C0': float(entries["С₀"].get()),
                'V': float(entries["V"].get()),
                'eps': float(entries["ɛ"].get()),
                'd': float(entries["d"].get()),
                'e': float(entries["e"].get()),
                'f': float(entries["f"].get()),
                'eta': float(entries["η"].get()),
                'Da': float(entries["Da"].get()),
                'Ds': float(entries["Ds"].get()),
                'Dy': float(entries["Dy"].get()),
                'a': float(equilibrium_entries["a"].get()),
                's': float(equilibrium_entries["s"].get()),
                'y': float(equilibrium_entries["y"].get()),
                'k': float(spatial_entries["k"].get()),
                'L': float(spatial_entries["L"].get())
            }

            # Вызываем вычислительную функцию
            result = calculate_stability_matrix(**params)

            if 'error' in result:
                output_text.delete(1.0, tk.END)
                output_text.insert(tk.END, result['error'])
                return

            output_text.delete(1.0, tk.END)

            output_text.insert(tk.END, "Матрица Якоби:\n")
            for row in result['J']:
                output_text.insert(
                    tk.END, f"[{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f}]\n")
            output_text.insert(tk.END, "\n")

            output_text.insert(
                tk.END, "Коэффициенты характеристического полинома при k = 0:\n")
            output_text.insert(tk.END, f"b0 = {result['b0']:8.4f}\n")
            output_text.insert(tk.END, f"b1 = {result['b1']:8.4f}\n")
            output_text.insert(tk.END, f"b2 = {result['b2']:8.4f}\n")
            output_text.insert(tk.END, f"b3 = {result['b3']:8.4f}\n")
            output_text.insert(tk.END, f"b1*b2 - b3 = {result['b']:8.4f}\n\n")

            output_text.insert(tk.END, "Равновесие устойчиво по нулевой гармонике\n\n" if result['stability']
                            else "Равновесие не устойчиво по нулевой гармонике\n\n")

            output_text.insert(tk.END, f"kappa2 = {result['kappa2']:8.4f}\n\n")

            output_text.insert(tk.END, "Матрица J_k:\n")
            for row in result['Jk']:
                output_text.insert(
                    tk.END, f"[{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f}]\n")
            output_text.insert(tk.END, "\n")

            output_text.insert(
                tk.END, "Коэффициенты характеристического полинома при k != 0:\n")
            output_text.insert(tk.END, f"bk0 = {result['bk0']:8.8f}\n")
            output_text.insert(tk.END, f"bk1 = {result['bk1']:8.8f}\n")
            output_text.insert(tk.END, f"bk2 = {result['bk2']:8.8f}\n")
            output_text.insert(tk.END, f"bk3 = {result['bk3']:8.8f}\n")
            output_text.insert(tk.END, f"bk1*bk2 - bk3 = {result['bk']:8.8f}\n\n")

            output_text.insert(tk.END, "Равновесие устойчиво по k-ой гармонике\n\n" if result['stability_k']
                            else "Равновесие не устойчиво по k-ой гармонике\n\n")

            output_text.insert(
                tk.END, "Максимальные действительные части собственных чисел:\n")
            for k_val, max_real in zip(result['k_values'], result['max_real_parts']):
                output_text.insert(tk.END, f"k = {k_val}: {max_real:.6f}\n")

            calculate_matrix.max_real_parts = result['max_real_parts']
            calculate_matrix.k_values = result['k_values']

        except ValueError:
            output_text.delete(1.0, tk.END)
            output_text.insert(
                tk.END, "Ошибка: введите корректные числовые значения")

    # Функция для построения графика
    def show_graph():
        try:
            if not hasattr(calculate_matrix, 'max_real_parts') or not hasattr(calculate_matrix, 'k_values'):
                output_text.delete(1.0, tk.END)
                output_text.insert(tk.END, "Ошибка: сначала выполните расчет")
                return

            fig = plt.Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            ax.plot(calculate_matrix.k_values,
                    calculate_matrix.max_real_parts, marker='o')
            ax.set_title("Максимальная действительная часть собственных чисел")
            ax.set_xlabel("k")
            ax.set_ylabel("max(Re(λ))")
            ax.grid(True)
            ax.axhline(y=0, color='gray', linestyle='--')

            graph_window = tk.Toplevel(tab)
            graph_window.title("График max(Re(λ)) от k")
            canvas = FigureCanvasTkAgg(fig, master=graph_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            output_text.delete(1.0, tk.END)
            output_text.insert(
                tk.END, f"Ошибка при построении графика: {str(e)}")

    calc_button.config(command=calculate_matrix)
    graph_button.config(command=show_graph)

    tab.columnconfigure(0, weight=1)
    tab.columnconfigure(1, weight=3)
    tab.rowconfigure(0, weight=0)
    tab.rowconfigure(1, weight=1)
    tab.rowconfigure(2, weight=1)
    tab.rowconfigure(3, weight=1)


# Функция для создания вкладки "Метод Ньютона"
def create_newton_tab():
    tab = ttk.Frame(notebook)
    notebook.add(tab, text="Метод Ньютона")

    newton_calc_button = ttk.Button(tab, text="Рассчитать")
    newton_calc_button.grid(row=0, column=0, pady=5)

    newton_left_frame = ttk.LabelFrame(
        tab, text="Параметры системы", padding=10)
    newton_left_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

    newton_entries = {}
    for idx, (param, value) in enumerate(SYSTEM_PARAMS):
        col = idx // 6
        row = idx % 6
        label = ttk.Label(newton_left_frame, text=f"{param}:")
        label.grid(row=row, column=col * 2, padx=5, pady=5, sticky="e")
        entry = ttk.Entry(newton_left_frame, width=12)
        entry.insert(0, str(value))
        entry.grid(row=row, column=col * 2 + 1, padx=5, pady=5)
        newton_entries[param] = entry

    newton_initial_frame = ttk.LabelFrame(
        tab, text="Начальное приближение", padding=10)
    newton_initial_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

    newton_initial_entries = {}
    for idx, (param, value) in enumerate(INITIAL_PARAMS):
        label = ttk.Label(newton_initial_frame, text=f"{param}:")
        label.grid(row=idx, column=0, padx=5, pady=5, sticky="e")
        entry = ttk.Entry(newton_initial_frame, width=10)
        entry.insert(0, str(value))
        entry.grid(row=idx, column=1, padx=5, pady=5)
        newton_initial_entries[param] = entry

    newton_right_frame = ttk.LabelFrame(tab, text="Вывод данных", padding=10)
    newton_right_frame.grid(row=0, column=1, rowspan=3,
                            padx=10, pady=10, sticky="nsew")

    newton_output_text = tk.Text(newton_right_frame, height=50, width=150)
    newton_output_text.grid(row=0, column=0, padx=5, pady=5)

    # Функция для решения системы методом Ньютона
    def solve_newton():
        try:
            max_iterations = 100

            iteration, x, f1, f2, f3, residual_norm = newton_method(
                c=float(newton_entries["с"].get()),
                mu=float(newton_entries["μ"].get()),
                C0=float(newton_entries["С₀"].get()),
                V=float(newton_entries["V"].get()),
                eps=float(newton_entries["ɛ"].get()),
                d=float(newton_entries["d"].get()),
                e=float(newton_entries["e"].get()),
                f=float(newton_entries["f"].get()),
                eta=float(newton_entries["η"].get()),
                a=float(newton_initial_entries["a"].get()),
                s=float(newton_initial_entries["s"].get()),
                y=float(newton_initial_entries["y"].get()),
                max_iter=max_iterations
            )

            newton_output_text.delete(1.0, tk.END)

            if iteration < max_iterations:
                newton_output_text.insert(
                    tk.END, f"Решение найдено за {iteration} итераций:\n")
                newton_output_text.insert(tk.END, f"a = {x[0]:.6f}\n")
                newton_output_text.insert(tk.END, f"s = {x[1]:.6f}\n")
                newton_output_text.insert(tk.END, f"y = {x[2]:.6f}\n")
                newton_output_text.insert(
                    tk.END, "\nНевязка системы уравнений:\n")
                newton_output_text.insert(tk.END, f"f1 = {f1:.6e}\n")
                newton_output_text.insert(tk.END, f"f2 = {f2:.6e}\n")
                newton_output_text.insert(tk.END, f"f3 = {f3:.6e}\n")
                newton_output_text.insert(
                    tk.END, f"Норма невязки ||F|| = {residual_norm:.6e}\n")
            else:
                newton_output_text.insert(
                    tk.END, "Метод не сошелся за максимальное число итераций.\n")
                newton_output_text.insert(
                    tk.END, "Попробуйте изменить начальное приближение.\n")

        except ValueError:
            newton_output_text.delete(1.0, tk.END)
            newton_output_text.insert(
                tk.END, "Ошибка: введите корректные числовые значения.\n")
        except np.linalg.LinAlgError:
            newton_output_text.delete(1.0, tk.END)
            newton_output_text.insert(
                tk.END, "Ошибка: матрица Якоби вырождена.\nПопробуйте изменить начальное приближение.\n")

    newton_calc_button.config(command=solve_newton)

    tab.columnconfigure(0, weight=1)
    tab.columnconfigure(1, weight=3)
    tab.rowconfigure(0, weight=0)
    tab.rowconfigure(1, weight=1)
    tab.rowconfigure(2, weight=1)


def create_solutions_tab():
    tab = ttk.Frame(notebook)
    notebook.add(tab, text="Расчет состояний равновесия")

    calc_button = ttk.Button(tab, text="Показать графики")
    calc_button.grid(row=0, column=0, pady=5)

    left_frame = ttk.LabelFrame(tab, text="Параметры системы", padding=10)
    left_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

    entries = {}
    for idx, (param, value) in enumerate(SYSTEM_PARAMS):
        col = idx // 6
        row = idx % 6
        label = ttk.Label(left_frame, text=f"{param}:")
        label.grid(row=row, column=col * 2, padx=5, pady=5, sticky="e")
        entry = ttk.Entry(left_frame, width=12)
        entry.insert(0, str(value))
        entry.grid(row=row, column=col * 2 + 1, padx=5, pady=5)
        entries[param] = entry

    intersections_frame = ttk.LabelFrame(
        tab, text="Точки пересечения", padding=10)
    intersections_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

    intersections_text = tk.Text(intersections_frame, height=20, width=40)
    intersections_text.grid(row=0, column=0, padx=5, pady=5)

    axis_limits_frame = ttk.LabelFrame(
        tab, text="Пределы оси Y для графика a(y)", padding=10)
    axis_limits_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")

    ttk.Label(axis_limits_frame, text="Y min:").grid(
        row=0, column=0, padx=5, pady=5, sticky="e")
    y_min_entry = ttk.Entry(axis_limits_frame, width=10)
    y_min_entry.insert(0, "-1")
    y_min_entry.grid(row=0, column=1, padx=5, pady=5)

    ttk.Label(axis_limits_frame, text="Y max:").grid(
        row=1, column=0, padx=5, pady=5, sticky="e")
    y_max_entry = ttk.Entry(axis_limits_frame, width=10)
    y_max_entry.insert(0, "1.5")
    y_max_entry.grid(row=1, column=1, padx=5, pady=5)

    graph_frame = ttk.LabelFrame(tab, text="Графики функций a,s,y", padding=10)
    graph_frame.grid(row=0, column=1, rowspan=4,
                     padx=10, pady=10, sticky="nsew")

    fig = plt.figure(figsize=(15, 10))

    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], wspace=0.3, hspace=0.4)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.get_tk_widget().pack()

    # Функция для построения графиков и поиска пересечений
    def plot_graphs():
        try:
            y_min = float(y_min_entry.get())
            y_max = float(y_max_entry.get())

            a, s_a, y, aplus_y, aminus_y, a_y, intersections = find_graphs(
                c=float(entries["с"].get()),
                mu=float(entries["μ"].get()),
                C0=float(entries["С₀"].get()),
                V=float(entries["V"].get()),
                eps=float(entries["ɛ"].get()),
                d=float(entries["d"].get()),
                e=float(entries["e"].get()),
                f=float(entries["f"].get()),
                eta=float(entries["η"].get())
            )

            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()

            # График s(a)
            ax1.plot(a, s_a, label="s(a)", color="blue")
            ax1.set_title("График s(a)")
            ax1.set_xlabel("a")
            ax1.set_ylabel("s(a)")
            ax1.legend()
            ax1.grid(True)

            # График a(y) для aplus_y и aminus_y
            ax2.plot(y, aplus_y, label="a_plus(y)", color="red")
            ax2.plot(y, aminus_y, label="a_minus(y)", color="blue")
            ax2.set_title("График a(y)")
            ax2.set_xlabel("y")
            ax2.set_ylabel("a(y)")
            ax2.legend()
            ax2.grid(True)

            # График a(y)
            ax3.plot(y, a_y, label="a(y)", color="green")
            ax3.set_title("График a(y)")
            ax3.set_xlabel("y")
            ax3.set_ylabel("a(y)")
            ax3.legend()
            ax3.grid(True)

            # График aplus_y, aminus_y и a_y вместе с точками пересечения
            ax4.plot(y, aplus_y, label="a_plus(y)", color="red")
            ax4.plot(y, aminus_y, label="a_minus(y)", color="blue")
            ax4.plot(y, a_y, label="a(y)", color="green")
            if intersections:
                a_inter, s_inter, y_inter = zip(*intersections)
                ax4.scatter(y_inter, a_inter, color="black",
                            zorder=5, label="Пересечения")
            ax4.set_title("Графики a(y) и точки пересечения")
            ax4.set_xlabel("y")
            ax4.set_ylabel("a")
            ax4.set_ylim(y_min, y_max)
            ax4.legend()
            ax4.grid(True)

            # Выводим точки пересечения в текстовое поле
            intersections_text.delete(1.0, tk.END)
            intersections_text.insert(
                tk.END, f"Количество пересечений: {len(intersections)}\n\n")
            if intersections:
                intersections_text.insert(
                    tk.END, "Координаты точек пересечения (a, s, y):\n\n")
                for i, (a_val, s_val, y_val) in enumerate(intersections, 1):
                    intersections_text.insert(
                        tk.END, f"Точка {i}: ({a_val:.6f}, {s_val:.6f}, {y_val:.6f})\n")
            else:
                intersections_text.insert(tk.END, "Пересечения не найдены.\n")

            canvas.draw()

        except (ValueError, RuntimeWarning):
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()
            ax1.text(0.5, 0.5, "Ошибка: введите корректные числовые значения",
                     ha="center", va="center")
            ax2.text(0.5, 0.5, "Ошибка: введите корректные числовые значения",
                     ha="center", va="center")
            ax3.text(0.5, 0.5, "Ошибка: введите корректные числовые значения",
                     ha="center", va="center")
            ax4.text(0.5, 0.5, "Ошибка: введите корректные числовые значения",
                     ha="center", va="center")
            intersections_text.delete(1.0, tk.END)
            intersections_text.insert(
                tk.END, "Ошибка: введите корректные числовые значения")
            canvas.draw()

    calc_button.config(command=plot_graphs)

    tab.columnconfigure(0, weight=1)
    tab.columnconfigure(1, weight=3)
    tab.rowconfigure(0, weight=0)
    tab.rowconfigure(1, weight=1)
    tab.rowconfigure(2, weight=1)
    tab.rowconfigure(3, weight=1)


create_parameters_tab("Подбор параметров")
create_newton_tab()
create_solutions_tab()

root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

root.mainloop()
