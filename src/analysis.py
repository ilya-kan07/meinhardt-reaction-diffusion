import tkinter as tk
from tkinter import ttk
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Создаем главное окно
root = tk.Tk()
root.title("Исследование системы реакции диффузии Мейнхардта | Кандрушин И.Б.")
root.geometry("1280x900")


# Обработчик закрытия окна
def on_closing():
    plt.close('all')  # Закрываем все фигуры matplotlib
    root.destroy()  # Завершаем приложение


root.protocol("WM_DELETE_WINDOW", on_closing)

# Список параметров и их начальных значений (глобальный)
parameters = [
    ("с", 16.67),
    ("μ", 1.2),
    ("С₀", 1.128),
    ("V", 0.33),
    ("ɛ", 3.3),
    ("d", 0.023),
    ("e", 1.67),
    ("f", 9.0),
    ("η", 16.67),
    ("Da", 0.0620036278),
    ("Ds", 1.0),
    ("Dy", 1.0)
]

# Создаем вкладки
notebook = ttk.Notebook(root)
notebook.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

# Функция для создания вкладки "Подбор параметров"


def create_parameters_tab(tab_name):
    tab = ttk.Frame(notebook)
    notebook.add(tab, text=tab_name)

    # Фрейм для кнопок
    button_frame = ttk.Frame(tab)
    button_frame.grid(row=0, column=0, pady=5, sticky="ew")

    # Кнопка для расчета
    calc_button = ttk.Button(button_frame, text="Рассчитать")
    calc_button.grid(row=0, column=0, padx=5)

    # Кнопка для показа графика
    graph_button = ttk.Button(button_frame, text="Показать график")
    graph_button.grid(row=0, column=1, padx=5)

    # Создаем фрейм для левой части (параметры системы)
    left_frame = ttk.LabelFrame(tab, text="Параметры системы", padding=10)
    left_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

    # Создаем поля для ввода параметров в два столбца
    entries = {}
    for idx, (param, value) in enumerate(parameters):
        col = idx // 6
        row = idx % 6
        label = ttk.Label(left_frame, text=f"{param}:")
        label.grid(row=row, column=col * 2, padx=5, pady=5, sticky="e")
        entry = ttk.Entry(left_frame, width=12)
        entry.insert(0, str(value))
        entry.grid(row=row, column=col * 2 + 1, padx=5, pady=5)
        entries[param] = entry

    # Создаем фрейм для параметров равновесия
    equilibrium_frame = ttk.LabelFrame(tab, text="Равновесие", padding=10)
    equilibrium_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

    # Параметры равновесия и их начальные значения
    equilibrium_params = [
        ("a", 0.512934),
        ("s", 0.140341),
        ("y", 1.006568)
    ]

    # Создаем поля для ввода параметров равновесия
    equilibrium_entries = {}
    for idx, (param, value) in enumerate(equilibrium_params):
        label = ttk.Label(equilibrium_frame, text=f"{param}:")
        label.grid(row=idx, column=0, padx=5, pady=5, sticky="e")
        entry = ttk.Entry(equilibrium_frame, width=10)
        entry.insert(0, str(value))
        entry.grid(row=idx, column=1, padx=5, pady=5)
        equilibrium_entries[param] = entry

    # Создаем фрейм для характеристик пространственной области и гармоники
    spatial_frame = ttk.LabelFrame(
        tab, text="Характеристики пространственной области и гармоники", padding=10)
    spatial_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")

    # Параметры пространственной области и их начальные значения
    spatial_params = [
        ("k", 1.0),
        ("L", 1.0)
    ]

    # Создаем поля для ввода параметров пространственной области
    spatial_entries = {}
    for idx, (param, value) in enumerate(spatial_params):
        label = ttk.Label(spatial_frame, text=f"{param}:")
        label.grid(row=idx, column=0, padx=5, pady=5, sticky="e")
        entry = ttk.Entry(spatial_frame, width=10)
        entry.insert(0, str(value))
        entry.grid(row=idx, column=1, padx=5, pady=5)
        spatial_entries[param] = entry

    # Создаем фрейм для правой части (вывод данных)
    right_frame = ttk.LabelFrame(tab, text="Вывод данных", padding=10)
    right_frame.grid(row=0, column=1, rowspan=4,
                     padx=10, pady=10, sticky="nsew")

    # Поле для вывода
    output_text = tk.Text(right_frame, height=50, width=150)
    output_text.grid(row=0, column=0, padx=5, pady=5)

    # Функция для расчета матрицы
    def calculate_matrix():
        try:
            # Получаем значения из полей
            c = float(entries["с"].get())
            mu = float(entries["μ"].get())
            C0 = float(entries["С₀"].get())
            V = float(entries["V"].get())
            eps = float(entries["ɛ"].get())
            d = float(entries["d"].get())
            e = float(entries["e"].get())
            f = float(entries["f"].get())
            eta = float(entries["η"].get())
            Da = float(entries["Da"].get())
            Ds = float(entries["Ds"].get())
            Dy = float(entries["Dy"].get())
            a = float(equilibrium_entries["a"].get())
            s = float(equilibrium_entries["s"].get())
            y = float(equilibrium_entries["y"].get())
            k = float(spatial_entries["k"].get())
            L = float(spatial_entries["L"].get())

            # Вычисляем kappa2
            kappa2 = ((math.pi * k) / L) ** 2

            # Вычисляем коэффициенты матрицы Якоби
            J11 = 2 * c * a * s - mu
            J12 = c * a ** 2
            J13 = 0
            J21 = -2 * c * a * s
            J22 = -c * (a ** 2) - V - eps * y
            J23 = -eps * s
            J31 = d
            J32 = 0
            J33 = -e + (2 * eta * y) / ((1 + f * y ** 2) ** 2)

            # Коэффициенты характеристического полинома при k = 0
            b0 = 1
            b1 = -(J11 + J22 + J33)
            b2 = (J11 * J22 + J11 * J33 + J22 * J33 - J12 * J21)
            b3 = -(J11 * J22 * J33 - J12 * J21 * J33 + J12 * J23 * J31)

            b = b1 * b2 - b3

            stability = b1 > 0 and b3 > 0 and b > 0

            # Вычисляем матрицу J_k
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
            bk3 = -(Jk11 * Jk22 * Jk33 - Jk12 *
                    Jk21 * Jk33 + Jk12 * Jk23 * Jk31)

            bk = bk1 * bk2 - bk3

            stability_k = bk1 > 0 and bk3 > 0 and bk > 0

            # Собственные числа для k от 0 до 10
            k_values = np.arange(0, 11)
            max_real_parts = []

            for k_val in k_values:
                kappa2_val = ((math.pi * k_val) / L) ** 2
                Jk11_val = J11 - kappa2_val * Da
                Jk22_val = J22 - kappa2_val * Ds
                Jk33_val = J33 - kappa2_val * Dy

                # Матрица J_k
                Jk = np.array([
                    [Jk11_val, Jk12, Jk13],
                    [Jk21, Jk22_val, Jk23],
                    [Jk31, Jk32, Jk33_val]
                ])

                # Собственные числа
                eigenvalues = np.roots([1, -(Jk11_val + Jk22_val + Jk33_val),
                                       (Jk11_val * Jk22_val + Jk11_val * Jk33_val +
                                        Jk22_val * Jk33_val - Jk12 * Jk21),
                                       -(Jk11_val * Jk22_val * Jk33_val - Jk12 * Jk21 * Jk33_val + Jk12 * Jk23 * Jk31)])

                # Находим максимальную действительную часть
                max_real = max(e.real for e in eigenvalues)
                max_real_parts.append(max_real)

            # Очищаем поле вывода
            output_text.delete(1.0, tk.END)

            # Выводим матрицу Якоби
            output_text.insert(tk.END, "Матрица Якоби:\n")
            output_text.insert(tk.END, f"[{J11:8.4f} {J12:8.4f} {J13:8.4f}]\n")
            output_text.insert(tk.END, f"[{J21:8.4f} {J22:8.4f} {J23:8.4f}]\n")
            output_text.insert(
                tk.END, f"[{J31:8.4f} {J32:8.4f} {J33:8.4f}]\n\n")

            # Коэффициенты характеристического полинома при k = 0
            output_text.insert(
                tk.END, "Коэффициенты характеристического полинома при k = 0:\n")
            output_text.insert(tk.END, f"b0 = {b0:8.4f}\n")
            output_text.insert(tk.END, f"b1 = {b1:8.4f}\n")
            output_text.insert(tk.END, f"b2 = {b2:8.4f}\n")
            output_text.insert(tk.END, f"b3 = {b3:8.4f}\n")
            output_text.insert(tk.END, f"b1*b2 - b3 = {b:8.4f}\n\n")

            output_text.insert(tk.END, "Равновесие устойчиво по нулевой гармонике\n\n" if stability
                               else "Равновесие не устойчиво по нулевой гармонике\n\n")

            # Выводим kappa2
            output_text.insert(tk.END, f"kappa2 = {kappa2:8.4f}\n\n")

            # Выводим матрицу J_k
            output_text.insert(tk.END, "Матрица J_k:\n")
            output_text.insert(
                tk.END, f"[{Jk11:8.4f} {Jk12:8.4f} {Jk13:8.4f}]\n")
            output_text.insert(
                tk.END, f"[{Jk21:8.4f} {Jk22:8.4f} {Jk23:8.4f}]\n")
            output_text.insert(
                tk.END, f"[{Jk31:8.4f} {Jk32:8.4f} {Jk33:8.4f}]\n\n")

            # Коэффициенты характеристического полинома при k != 0
            output_text.insert(
                tk.END, "Коэффициенты характеристического полинома при k != 0:\n")
            output_text.insert(tk.END, f"bk0 = {bk0:8.8f}\n")
            output_text.insert(tk.END, f"bk1 = {bk1:8.8f}\n")
            output_text.insert(tk.END, f"bk2 = {bk2:8.8f}\n")
            output_text.insert(tk.END, f"bk3 = {bk3:8.8f}\n")
            output_text.insert(tk.END, f"bk1*bk2 - bk3 = {bk:8.8f}\n\n")

            output_text.insert(tk.END, "Равновесие устойчиво по k-ой гармонике\n\n" if stability_k
                               else "Равновесие не устойчиво по k-ой гармонике\n\n")

            # Вывод максимальных действительных частей собственных чисел
            output_text.insert(
                tk.END, "Максимальные действительные части собственных чисел:\n")
            for k_val, max_real in zip(k_values, max_real_parts):
                output_text.insert(tk.END, f"k = {k_val}: {max_real:.6f}\n")

            # Сохраняем данные для графика
            calculate_matrix.max_real_parts = max_real_parts
            calculate_matrix.k_values = k_values

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

            # Создаем новое окно для графика
            fig = plt.Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            ax.plot(calculate_matrix.k_values,
                    calculate_matrix.max_real_parts, marker='o')
            ax.set_title("Максимальная действительная часть собственных чисел")
            ax.set_xlabel("k")
            ax.set_ylabel("max(Re(λ))")
            ax.grid(True)
            ax.axhline(y=0, color='gray', linestyle='--')

            # Встраиваем график в интерфейс
            graph_window = tk.Toplevel(tab)
            graph_window.title("График max(Re(λ)) от k")
            canvas = FigureCanvasTkAgg(fig, master=graph_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            output_text.delete(1.0, tk.END)
            output_text.insert(
                tk.END, f"Ошибка при построении графика: {str(e)}")

    # Привязываем функции к кнопкам
    calc_button.config(command=calculate_matrix)
    graph_button.config(command=show_graph)

    # Настраиваем растяжку столбцов и строк
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

    # Кнопка для расчета
    newton_calc_button = ttk.Button(tab, text="Рассчитать")
    newton_calc_button.grid(row=0, column=0, pady=5)

    # Создаем фрейм для параметров системы в вкладке "Метод Ньютона"
    newton_left_frame = ttk.LabelFrame(
        tab, text="Параметры системы", padding=10)
    newton_left_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

    # Создаем поля для ввода параметров в два столбца
    newton_entries = {}
    for idx, (param, value) in enumerate(parameters):
        col = idx // 6
        row = idx % 6
        label = ttk.Label(newton_left_frame, text=f"{param}:")
        label.grid(row=row, column=col * 2, padx=5, pady=5, sticky="e")
        entry = ttk.Entry(newton_left_frame, width=12)
        entry.insert(0, str(value))
        entry.grid(row=row, column=col * 2 + 1, padx=5, pady=5)
        newton_entries[param] = entry

    # Создаем фрейм для начального приближения
    newton_initial_frame = ttk.LabelFrame(
        tab, text="Начальное приближение", padding=10)
    newton_initial_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

    # Параметры начального приближения и их значения
    initial_params = [
        ("a", 0.5),
        ("s", 0.5),
        ("y", 0.5)
    ]

    # Создаем поля для ввода начального приближения
    newton_initial_entries = {}
    for idx, (param, value) in enumerate(initial_params):
        label = ttk.Label(newton_initial_frame, text=f"{param}:")
        label.grid(row=idx, column=0, padx=5, pady=5, sticky="e")
        entry = ttk.Entry(newton_initial_frame, width=10)
        entry.insert(0, str(value))
        entry.grid(row=idx, column=1, padx=5, pady=5)
        newton_initial_entries[param] = entry

    # Создаем фрейм для вывода данных в вкладке "Метод Ньютона"
    newton_right_frame = ttk.LabelFrame(tab, text="Вывод данных", padding=10)
    newton_right_frame.grid(row=0, column=1, rowspan=3,
                            padx=10, pady=10, sticky="nsew")

    # Поле для вывода
    newton_output_text = tk.Text(newton_right_frame, height=50, width=150)
    newton_output_text.grid(row=0, column=0, padx=5, pady=5)

    # Функция для решения системы методом Ньютона
    def solve_newton():
        try:
            # Получаем параметры системы
            c = float(newton_entries["с"].get())
            mu = float(newton_entries["μ"].get())
            C0 = float(newton_entries["С₀"].get())
            V = float(newton_entries["V"].get())
            eps = float(newton_entries["ɛ"].get())
            d = float(newton_entries["d"].get())
            e = float(newton_entries["e"].get())
            f = float(newton_entries["f"].get())
            eta = float(newton_entries["η"].get())

            # Получаем начальное приближение
            a = float(newton_initial_entries["a"].get())
            s = float(newton_initial_entries["s"].get())
            y = float(newton_initial_entries["y"].get())

            # Начальное приближение в виде вектора
            x = np.array([a, s, y])

            # Максимальное количество итераций и точность
            max_iterations = 100
            tolerance = 1e-6

            # Счетчик итераций
            iteration = 0

            while iteration < max_iterations:
                # Вычисляем значения системы уравнений
                f1 = c * x[0]**2 * x[1] - mu * x[0]
                f2 = C0 - c * x[0]**2 * x[1] - V * x[1] - eps * x[1] * x[2]
                f3 = d * x[0] - e * x[2] + (eta * x[2]**2) / (1 + f * x[2]**2)
                F = np.array([f1, f2, f3])

                # Проверяем норму F для условия сходимости
                if np.linalg.norm(F) < tolerance:
                    break

                # Матрица Якоби (частные производные)
                J11 = 2 * c * x[0] * x[1] - mu  # df1/da
                J12 = c * x[0]**2               # df1/ds
                J13 = 0                         # df1/dy
                J21 = -2 * c * x[0] * x[1]      # df2/da
                J22 = -c * x[0]**2 - V - eps * x[2]  # df2/ds
                J23 = -eps * x[1]               # df2/dy
                J31 = d                         # df3/da
                J32 = 0                         # df3/ds
                J33 = -e + (2 * eta * x[2] * (1 + f * x[2]**2) - 2 *
                            f * x[2]**3 * eta) / (1 + f * x[2]**2)**2  # df3/dy

                J = np.array([
                    [J11, J12, J13],
                    [J21, J22, J23],
                    [J31, J32, J33]
                ])

                # Решаем систему J * delta_x = -F
                delta_x = np.linalg.solve(J, -F)

                # Обновляем приближение
                x = x + delta_x

                iteration += 1

            # Очищаем поле вывода
            newton_output_text.delete(1.0, tk.END)

            # Проверяем сходимость
            if iteration < max_iterations:
                # Вычисляем невязку в точке решения
                f1 = c * x[0]**2 * x[1] - mu * x[0]
                f2 = C0 - c * x[0]**2 * x[1] - V * x[1] - eps * x[1] * x[2]
                f3 = d * x[0] - e * x[2] + (eta * x[2]**2) / (1 + f * x[2]**2)
                F = np.array([f1, f2, f3])
                residual_norm = np.linalg.norm(F)

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

    # Привязываем функцию к кнопке
    newton_calc_button.config(command=solve_newton)

    # Настраиваем растяжку столбцов и строк для вкладки "Метод Ньютона"
    tab.columnconfigure(0, weight=1)
    tab.columnconfigure(1, weight=3)
    tab.rowconfigure(0, weight=0)
    tab.rowconfigure(1, weight=1)
    tab.rowconfigure(2, weight=1)


def create_solutions_tab():
    tab = ttk.Frame(notebook)
    notebook.add(tab, text="Расчет состояний равновесия")

    # Кнопка для построения графиков
    calc_button = ttk.Button(tab, text="Показать графики")
    calc_button.grid(row=0, column=0, pady=5)

    left_frame = ttk.LabelFrame(tab, text="Параметры системы", padding=10)
    left_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

    parameters = [
        ("с", 16.67),
        ("μ", 1.2),
        ("С₀", 1.128),
        ("V", 0.33),
        ("ɛ", 3.3),
        ("d", 0.023),
        ("e", 1.67),
        ("f", 9.0),
        ("η", 16.67),
        ("Da", 0.0620036278),
        ("Ds", 1.0),
        ("Dy", 1.0)
    ]

    # Создаем поля для ввода параметров в два столбца
    entries = {}
    for idx, (param, value) in enumerate(parameters):
        col = idx // 6
        row = idx % 6
        label = ttk.Label(left_frame, text=f"{param}:")
        label.grid(row=row, column=col * 2, padx=5, pady=5, sticky="e")
        entry = ttk.Entry(left_frame, width=12)
        entry.insert(0, str(value))
        entry.grid(row=row, column=col * 2 + 1, padx=5, pady=5)
        entries[param] = entry

    # Создаем фрейм для вывода точек пересечения
    intersections_frame = ttk.LabelFrame(
        tab, text="Точки пересечения", padding=10)
    intersections_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

    # Поле для вывода точек пересечения (увеличено до height=20)
    intersections_text = tk.Text(intersections_frame, height=20, width=40)
    intersections_text.grid(row=0, column=0, padx=5, pady=5)

    # Создаем фрейм для пределов осей
    axis_limits_frame = ttk.LabelFrame(
        tab, text="Пределы оси Y для графика a(y)", padding=10)
    axis_limits_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")

    # Поля для ввода пределов оси Y
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

    # Создаем фрейм для графиков
    graph_frame = ttk.LabelFrame(tab, text="Графики функций a,s,y", padding=10)
    graph_frame.grid(row=0, column=1, rowspan=4,
                     padx=10, pady=10, sticky="nsew")

    # Создаем фигуру с четырьмя подграфиками с увеличенным размером и настройкой расстояния
    fig = plt.figure(figsize=(15, 10))  # Увеличиваем размер фигуры
    # Добавляем отступы между подграфиками
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], wspace=0.3, hspace=0.4)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.get_tk_widget().pack()
    # fig.tight_layout()

    # Функция для построения графиков и поиска пересечений
    def plot_graphs():
        try:
            # Получаем значения из полей
            c = float(entries["с"].get())
            mu = float(entries["μ"].get())
            C0 = float(entries["С₀"].get())
            V = float(entries["V"].get())
            eps = float(entries["ɛ"].get())
            d = float(entries["d"].get())
            e = float(entries["e"].get())
            f = float(entries["f"].get())
            eta = float(entries["η"].get())
            Da = float(entries["Da"].get())
            Ds = float(entries["Ds"].get())
            Dy = float(entries["Dy"].get())
            y_min = float(y_min_entry.get())
            y_max = float(y_max_entry.get())

            # Генерируем данные
            a = np.linspace(0.001, 1, 100)
            s_a = mu / (c * a)
            # y_a = (((C0 * c * a) / mu) - c * a * a - V) / eps

            y = np.linspace(0, 1.2, 1000)  # Ограничиваем y >= 0
            a_y = (y * (e + e * f * y * y - eta * y)) / (d * (1 + f * y * y))

            discriminant = C0 * C0 * c * c - 4 * c * mu * mu * (eps * y + V)
            aplus_y = np.where(
                discriminant >= 0, (C0 * c + np.sqrt(np.maximum(discriminant, 0))) / (2 * c * mu), np.nan)
            aminus_y = np.where(
                discriminant >= 0, (C0 * c - np.sqrt(np.maximum(discriminant, 0))) / (2 * c * mu), np.nan)

            # Поиск точек пересечения
            def find_intersections(y, func1, func2):
                intersections = []
                diff = func1 - func2
                # Находим индексы, где разность меняет знак
                sign_change = np.where(np.diff(np.sign(diff)))[0]
                for idx in sign_change:
                    # Интерполяция для более точного нахождения точки пересечения
                    y1, y2 = y[idx], y[idx + 1]
                    f1, f2 = diff[idx], diff[idx + 1]
                    # Линейная интерполяция
                    y_inter = y1 - f1 * (y2 - y1) / (f2 - f1)
                    # Значение a в точке пересечения
                    a_inter = np.interp(y_inter, y, func1)

                    s_inter = mu / (c * a_inter)

                    if 0 <= a_inter:
                        intersections.append((a_inter, s_inter, y_inter))
                return intersections

            # Находим пересечения aplus_y и a_y
            intersections_plus = find_intersections(y, aplus_y, a_y)
            # Находим пересечения aminus_y и a_y
            intersections_minus = find_intersections(y, aminus_y, a_y)
            # Объединяем и сортируем точки пересечения
            intersections = intersections_plus + intersections_minus
            intersections.sort(key=lambda x: x[0])  # Сортировка по y

            # Очищаем подграфики
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
            ax4.set_ylim(y_min, y_max)  # Используем введенные пределы
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

            # Обновляем холст
            canvas.draw()

        except (ValueError, RuntimeWarning):
            # Очищаем подграфики в случае ошибки
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

    # Привязываем функцию к кнопке
    calc_button.config(command=plot_graphs)

    # Настраиваем растяжку столбцов и строк
    tab.columnconfigure(0, weight=1)
    tab.columnconfigure(1, weight=3)
    tab.rowconfigure(0, weight=0)
    tab.rowconfigure(1, weight=1)
    tab.rowconfigure(2, weight=1)
    tab.rowconfigure(3, weight=1)


# Создаем вкладки
create_parameters_tab("Подбор параметров")
create_newton_tab()
create_solutions_tab()

# Настраиваем растяжку для главного окна
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# Запускаем приложение
root.mainloop()
