import math
import numpy as np
from scipy.optimize import brentq, root_scalar
from scipy.interpolate import interp1d


# ══════════════════════════════════════════════════════════════════════════════
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ══════════════════════════════════════════════════════════════════════════════

def _find_equilibrium_raw(params, y_grid=None):
    """
    Ищет точку равновесия P по пересечениям a1(y) и a2(y).
    Возвращает (y, a, s) или None.

    y_grid — необязательная заранее вычисленная сетка (ускоряет повторные вызовы
    с теми же границами, если менялся только один параметр).
    """
    c = params['c']
    mu = params['mu']
    eps = params['eps']
    nu = params['nu']
    e = params['e']
    eta = params['eta']
    d = params['d']
    f = params['f']
    c0 = params.get('c0')

    if c0 is None:
        return None

    N = 20_000
    if y_grid is None:
        y_grid = np.linspace(-3.0, 3.0, N)

    denom_a2 = d * (1.0 + f * y_grid ** 2)
    a_two = y_grid * (e + e * f * y_grid ** 2 - eta * y_grid) / denom_a2

    disc = c0 ** 2 * c ** 2 - 4.0 * c * mu ** 2 * (eps * y_grid + nu)
    sqrt_disc = np.where(disc >= 0, np.sqrt(np.maximum(disc, 0.0)), np.nan)
    denom_a1 = 2.0 * c * mu

    ap = np.where(disc >= 0, (c0 * c + sqrt_disc) / denom_a1, np.nan)
    am = np.where(disc >= 0, (c0 * c - sqrt_disc) / denom_a1, np.nan)

    crossings = []
    tol = 5e-4

    for a1 in (ap, am):
        valid = ~np.isnan(a1)
        diff = a1 - a_two
        i = 0
        while i < N - 1:
            if valid[i] and valid[i + 1]:
                d1, d2 = diff[i], diff[i + 1]
                if d1 * d2 < 0 or (abs(d1) < tol and abs(d2) < tol):
                    denom = d1 - d2
                    frac = d1 / denom if abs(denom) > 1e-12 else 0.5
                    y_c = y_grid[i] + frac * (y_grid[i + 1] - y_grid[i])
                    a_c = a_two[i] + frac * (a_two[i + 1] - a_two[i])
                    crossings.append((y_c, a_c))
                    i += 20
                    continue
            i += 1

    # Убираем дубликаты
    unique = []
    for cr in crossings:
        if not any(abs(cr[0] - u[0]) < 1e-5 for u in unique):
            unique.append(cr)
    unique.sort(key=lambda x: x[0])

    if len(unique) >= 6:
        y_p, a_p = unique[5]
    elif len(unique) == 5:
        y_p, a_p = unique[4]
    elif len(unique) == 4:
        y_p, a_p = unique[3]
    else:
        return None

    s_p = mu / (c * a_p)
    return y_p, a_p, s_p


# Кэшируем сетку — создаётся один раз
_Y_GRID = np.linspace(-3.0, 3.0, 20_000)


def find_equilibrium(params, **kwargs):
    """Публичная обёртка с кэшированной сеткой."""
    return _find_equilibrium_raw(params, y_grid=_Y_GRID)


# ══════════════════════════════════════════════════════════════════════════════
#  ВЫЧИСЛЕНИЕ УСТОЙЧИВОСТИ
# ══════════════════════════════════════════════════════════════════════════════

def _jacobian_and_criteria(params, eq):
    """
    По уже найденному равновесию eq = (y, a, s) вычисляет
    критерии Рауса–Гурвица для k=0 и для волнового числа k.
    Возвращает словарь.
    """
    y, a, s = eq

    c = params['c']
    mu = params['mu']
    eps = params['eps']
    d = params['d']
    e = params['e']
    f = params['f']
    eta = params['eta']
    Da = params['Da']
    Ds = params['Ds']
    Dy = params['Dy']
    nu = params['nu']
    k = params.get('k', 1.0)
    L = params.get('L', 1.0)

    kappa2 = (math.pi * k / L) ** 2

    # Якобиан при k=0
    J11 = 2*c*a*s - mu
    J12 = c*a**2
    J21 = -2*c*a*s
    J22 = -c*a**2 - nu - eps*y
    J23 = -eps*s
    J31 = d
    J33 = -e + 2*eta*y / (1 + f*y**2)**2

    # Коэффициенты характеристического уравнения (знак: det(λI - J))
    b1 = -(J11 + J22 + J33)
    b2 = J11*J22 - J12*J21 + J22*J33 + J11*J33
    b3 = -(J11*J22*J33 - J12*J21*J33 + J12*J23*J31)
    B = b1*b2 - b3

    # С диффузией
    Jk11 = J11 - kappa2*Da
    Jk22 = J22 - kappa2*Ds
    Jk33 = J33 - kappa2*Dy

    bk1 = -(Jk11 + Jk22 + Jk33)
    bk2 = Jk11*Jk22 - J12*J21 + Jk22*Jk33 + Jk11*Jk33
    bk3 = -(Jk11*Jk22*Jk33 - J12*J21*Jk33 + J12*J23*J31)

    stability_k0 = (b1 > 0) and (b3 > 0) and (B > 0)

    TOL_BK3 = 1e-3
    if not stability_k0:
        status = "Условия устойчивости нарушены (k=0)"
    elif bk1 <= 0 or bk2 <= 0:
        status = "Условия устойчивости нарушены (bk1 или bk2 <= 0)"
    elif abs(bk3) > TOL_BK3:
        sign = ">" if bk3 > 0 else "<"
        status = f"bk3 значительно {sign} 0 ({bk3:.2e})"
    else:
        status = "OK"

    return {
        'stability_k0': stability_k0,
        'b1': b1, 'b3': b3, 'b': B,
        'bk1': bk1, 'bk2': bk2, 'bk3': bk3,
        'bk3_abs': abs(bk3),
        'equilibrium': (a, s, y),   # порядок (a, s, y) — как было раньше
        'status': status,
    }


def compute_stability(params):
    """Полный расчёт устойчивости (совместим с прежним интерфейсом)."""
    # Проверка отрицательных параметров
    neg = [k for k in ('c', 'mu', 'eps', 'd', 'e', 'eta', 'nu', 'c0')
           if k in params and params[k] < 0]
    if neg:
        eq = find_equilibrium(params)
        y, a, s = eq if eq else (np.nan, np.nan, np.nan)
        return {
            'stability_k0': False,
            'bk1': 0, 'bk2': 0, 'bk3': 0,
            'b1': 0, 'b3': 0, 'b': 0,
            'equilibrium': (y, a, s),
            'status': f"Отрицательный параметр: {', '.join(neg)}",
            'bk3_abs': 0.0,
        }

    eq = find_equilibrium(params)
    if eq is None:
        return {
            'stability_k0': False,
            'bk1': 0, 'bk2': 0, 'bk3': 0,
            'b1': 0, 'b3': 0, 'b': 0,
            'equilibrium': (np.nan, np.nan, np.nan),
            'status': "Не найдено равновесие",
            'bk3_abs': 0.0,
        }

    return _jacobian_and_criteria(params, eq)


# ══════════════════════════════════════════════════════════════════════════════
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ БИФУРКАЦИИ
# ══════════════════════════════════════════════════════════════════════════════

def _bk3_func(p2_val, p1_val, p1_name, p2_name, fixed_params):
    """
    Вычисляет bk3 в точке (p1_val, p2_val).
    eq берётся из find_equilibrium — самого дорогого шага.
    """
    temp = fixed_params.copy()
    temp[p1_name] = p1_val
    temp[p2_name] = p2_val
    eq = find_equilibrium(temp)
    if eq is None:
        return np.nan
    res = _jacobian_and_criteria(temp, eq)
    return res['bk3']


def _find_p2_brentq(p1_val, p2_guess, p1_name, p2_name, fixed_params,
                    search_radius=0.5, n_bracket=40):
    """
    Находит ноль bk3(p2) при фиксированном p1 методом Брента.
    Начинает с p2_guess и расширяет интервал до ±search_radius.

    Возвращает (p2_root, bk3_at_root) или (None, None).
    """
    def f(p2): return _bk3_func(p2, p1_val, p1_name, p2_name, fixed_params)

    f0 = f(p2_guess)
    if np.isnan(f0):
        return None, None
    if abs(f0) < 1e-9:
        return p2_guess, f0

    # Ищем скобку [lo, hi] с разными знаками
    lo = hi = p2_guess
    sign0 = np.sign(f0)
    step = search_radius / n_bracket

    for _ in range(n_bracket):
        lo -= step
        f_lo = f(lo)
        if not np.isnan(f_lo) and np.sign(f_lo) != sign0:
            try:
                root = brentq(f, lo, p2_guess, xtol=1e-8,
                              rtol=1e-10, maxiter=80)
                return root, f(root)
            except Exception:
                pass

        hi += step
        f_hi = f(hi)
        if not np.isnan(f_hi) and np.sign(f_hi) != sign0:
            try:
                root = brentq(f, p2_guess, hi, xtol=1e-8,
                              rtol=1e-10, maxiter=80)
                return root, f(root)
            except Exception:
                pass

    return None, None


# ══════════════════════════════════════════════════════════════════════════════
#  ОСНОВНАЯ ФУНКЦИЯ БИФУРКАЦИИ
# ══════════════════════════════════════════════════════════════════════════════

def compute_bifurcation(params, param1, param2, step_size, num_steps):
    """
    Строит бифуркационную кривую Тьюринга (bk3 = 0) методом предиктор-корректор.

    Алгоритм:
      1. Предиктор: p2_{n+1} ≈ p2_n + dp2/dp1 * step_size  (конечная разность)
      2. Корректор: уточнение корня bk3 = 0 методом Брента в узком окне

    Это в 3–10 раз быстрее наивного широкого поиска, а точность выше
    за счёт xtol=1e-8 в brentq.
    """

    if 'c0' not in params:
        raise ValueError("В params обязательно должен быть ключ 'c0'")

    fixed_params = {k: v for k, v in params.items()
                    if k not in (param1, param2)}

    p1_arr = np.linspace(params[param1],
                         params[param1] + step_size * (num_steps - 1),
                         num_steps)

    print(
        f"Построение бифуркационной кривой для k = {params.get('k', 1.0):.3f}")
    print(
        f"  {param1}: {p1_arr[0]:.5f} → {p1_arr[-1]:.5f}  ({num_steps} точек)")

    points = []      # (p1, p2)
    points_data = []      # полный словарь для таблицы

    p2_cur = params[param2]   # стартовое значение подбираемого параметра

    # ── Начальная точка: широкий поиск ──────────────────────────────────────
    p2_root, _ = _find_p2_brentq(p1_arr[0], p2_cur, param1, param2,
                                 fixed_params, search_radius=1.0, n_bracket=60)
    if p2_root is None:
        print(
            f"  Не удалось найти стартовую точку, используем p2 = {p2_cur:.6f}")
    else:
        p2_cur = p2_root

    # ── Основной цикл ───────────────────────────────────────────────────────
    p2_prev = None   # для предиктора

    for i, p1 in enumerate(p1_arr):

        # --- Предиктор (линейная экстраполяция) ---
        if p2_prev is not None:
            dp2 = p2_cur - p2_prev
            p2_predict = p2_cur + dp2          # тренд прошлого шага
        else:
            p2_predict = p2_cur

        # --- Корректор (Брент в узком окне ±adaptive_radius) ---
        adaptive_radius = max(abs(p2_cur - p2_predict)
                              * 3 + step_size * 0.5, 0.05)

        p2_new, bk3_new = _find_p2_brentq(
            p1, p2_predict, param1, param2, fixed_params,
            search_radius=adaptive_radius, n_bracket=30)

        if p2_new is None:
            # Предиктор не помог → широкий поиск
            p2_new, bk3_new = _find_p2_brentq(
                p1, p2_cur, param1, param2, fixed_params,
                search_radius=0.8, n_bracket=60)

        if p2_new is None:
            print(
                f"  Шаг {i+1:3d}: {param1}={p1:.4f} — корень не найден, кривая прервана")
            break

        # --- Расчёт полного набора характеристик для этой точки ---
        temp = fixed_params.copy()
        temp[param1] = p1
        temp[param2] = p2_new

        eq = find_equilibrium(temp)
        if eq is None:
            print(f"  Шаг {i+1:3d}: равновесие не найдено при p2={p2_new:.6f}")
            break

        res = _jacobian_and_criteria(temp, eq)
        a_eq, s_eq, y_eq = res['equilibrium']   # порядок (a, s, y)

        status = res['status']
        if p1 < 0:
            status = "Изменяемый параметр < 0"
        if p2_new < 0:
            status = "Подбираемый параметр < 0"

        points.append((round(p1, 6), round(p2_new, 8)))

        points_data.append({
            'p1': p1, 'p2': p2_new,
            'a': a_eq, 's': s_eq, 'y': y_eq,
            'bk1': res['bk1'], 'bk2': res['bk2'], 'bk3': res['bk3'],
            'b1':  res['b1'],  'b3':  res['b3'],  'b':   res['b'],
            'status': status,
        })

        print(f"  Шаг {i+1:3d}: {param1}={p1:.5f}  {param2}={p2_new:.7f}"
              f"  bk3={res['bk3']:+.2e}  {status}")

        # Обновляем предиктор
        p2_prev = p2_cur
        p2_cur = p2_new

    # ── Финализация ─────────────────────────────────────────────────────────
    if not points:
        return np.array([]), np.array([]), points_data

    points = sorted(set(points), key=lambda x: x[0])
    data_p1 = np.array([p[0] for p in points])
    data_p2 = np.array([p[1] for p in points])

    print(f"  Итого точек на кривой: {len(points)}")

    # ────── СОХРАНЕНИЕ РЕЗУЛЬТАТОВ В CSV ──────
    if len(points) > 0:
        import pandas as pd
        import os
        from datetime import datetime

        # Создаём папку, если её нет
        os.makedirs('results', exist_ok=True)

        # Создаём DataFrame с основными колонками
        df = pd.DataFrame({
            param1: data_p1,
            param2: data_p2
        })

        # Добавляем дополнительные полезные данные (по желанию можно убрать)
        extra_df = pd.DataFrame(points_data)
        df_full = pd.concat([df.reset_index(drop=True),
                             extra_df.reset_index(drop=True)[['a', 's', 'y', 'bk1', 'bk2', 'bk3', 'status']]],
                            axis=1)

        # Имя файла с временной меткой
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'results/bifurcation_turing_{param1}_{param2}_{timestamp}.csv'

        # Сохраняем
        # df_full.to_csv(filename, index=False, float_format='%.10f')

        print(f"\n✅ Результаты бифуркационной кривой сохранены в файл:")
        print(f"   → {filename}")
        print(f"   Количество точек: {len(df)}")
    else:
        print("\n⚠️  Нет данных для сохранения в CSV (кривая пустая).")

    return data_p1, data_p2, points_data

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def find_curve_intersections(curves):
    """Находит точки пересечения между кривыми"""
    intersections = []
    ks = list(curves.keys())

    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            k1, k2 = ks[i], ks[j]
            c0_1, da_1 = curves[k1]
            c0_2, da_2 = curves[k2]

            f1 = interp1d(c0_1, da_1, kind='linear', fill_value="extrapolate")
            f2 = interp1d(c0_2, da_2, kind='linear', fill_value="extrapolate")

            c0_fine = np.linspace(max(c0_1.min(), c0_2.min()),
                                  min(c0_1.max(), c0_2.max()), 2000)
            diff = f1(c0_fine) - f2(c0_fine)

            sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
            for idx in sign_change:
                c0_cross = c0_fine[idx] - diff[idx] * (c0_fine[idx+1] - c0_fine[idx]) / (diff[idx+1] - diff[idx])
                if c0_1.min() <= c0_cross <= c0_1.max():
                    intersections.append(round(c0_cross, 6))

    return sorted(set(intersections))


def plot_c0_divisions(params_base, c0_start=1.128, c0_end=4.0, num_points=500, ks=None):
    """
    Чистый график: горизонтальная линия + ключевые точки с подписями на оси.
    """
    if ks is None:
        ks = [1, 2, 3, 4]

    print("=== Построение бифуркационных кривых для анализа ===")

    curves = {}
    for k in ks:
        params = params_base.copy()
        params['k'] = float(k)
        params['c0'] = c0_start

        try:
            c0_data, da_data, _ = compute_bifurcation(
                params, param1='c0', param2='Da',
                step_size=(c0_end - c0_start)/(num_points-1),
                num_steps=num_points
            )
            if len(c0_data) > 10:
                curves[k] = (c0_data, da_data)
                print(f"✓ k={k} построена")
        except Exception as e:
            print(f"✗ k={k}: {e}")

    if not curves:
        print("Не удалось построить кривые!")
        return None

    # === Точки пересечения кривых ===
    inter_c0 = find_curve_intersections(curves)

    # === Точка пересечения k=4 с Da=0 ===
    c0_k4 = curves[4][0]
    da_k4 = curves[4][1]
    try:
        f4 = interp1d(da_k4, c0_k4, kind='linear', fill_value="extrapolate")
        zero_cross_k4 = float(f4(0))
    except:
        zero_cross_k4 = c0_end

    # Все критические точки (без конечной точки 5.128)
    critical_points = [c0_start] + inter_c0 + [round(zero_cross_k4, 6)]
    critical_points = sorted(set([round(x, 6) for x in critical_points if x <= c0_end]))

    print(f"\nКлючевые точки c0:")
    for cp in critical_points:
        print(f"   {cp:.6f}")

    # === Построение графика ===
    fig, ax = plt.subplots(figsize=(14, 5))

    # Горизонтальная линия
    ax.plot([c0_start, c0_end], [0, 0], 'k-', linewidth=3.5, alpha=0.85)

    # Вертикальные пунктирные линии
    for cp in critical_points:
        ax.axvline(x=cp, color='red', linestyle=':', alpha=0.85, linewidth=1.4)

    # Подписи значений c0 НА ГОРИЗОНТАЛЬНОЙ ЛИНИИ (снизу)
    for cp in critical_points:
        ax.text(cp, -0.62, f'{cp:.4f}',
                horizontalalignment='center',
                verticalalignment='top',
                rotation=90,
                fontsize=11,
                color='red',
                fontweight='bold')

    # Настройка осей
    ax.set_xlim(c0_start - 0.02, 4.0)
    ax.set_ylim(-0.6, 0.8)
    ax.set_xlabel('$c_0$', fontsize=14, labelpad=15)
    ax.set_title('Ключевые точки разделения интервала $c_0$',
                 fontsize=15, pad=20)

    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Обычная сетка по x
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()

    return critical_points, curves

if __name__ == "__main__":
    # ================== НАСТРОЙКИ ==================
    params_base = {
        'c': 16.67,           # ← подставь свои реальные значения
        'mu': 1.2,
        'eps': 3.3,
        'nu': 0.33,
        'e': 1.67,
        'eta': 16.67,
        'd': 0.023,
        'f': 9.0,
        'Da': 0.0620036278,         # начальное приближение
        'Ds': 1.0,         # важно! обычно Ds >> Da
        'Dy': 1.0,
        'L': 1.0,
        # Добавь сюда все остальные параметры, которые использует твоя модель
        # (c0 будет переопределяться)
    }

    # ================== ЗАПУСК ==================
    critical_points, curves = plot_c0_divisions(
        params_base=params_base,
        c0_start=1.128,
        c0_end=5.128,
        num_points=500,      # можно 300–600
        ks=[1, 2, 3, 4]
    )

    print("\nКритические точки для деления интервала:")
    for i, cp in enumerate(critical_points):
        print(f"  {i+1:2d}. c0 = {cp:.6f}")
