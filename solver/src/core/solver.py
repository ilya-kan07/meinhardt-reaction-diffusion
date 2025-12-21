from typing import List, Dict, Optional, Callable
import numpy as np
from numpy.typing import NDArray

LayerData = Dict[str, NDArray]
GridData = List[LayerData]


class MeinhardtSolver:
    """
    Решатель уравнений Мейнхардта с двойной сеткой.
    Содержит ТОЛЬКО вычислительную логику.
    """

    def __init__(
        self,
        c: float, mu: float, c_0: float, nu: float,
        epsilon: float, d: float, e: float, f: float, eta: float,
        D_a: float, D_s: float, D_y: float,
        n_coarse: int, m_coarse: int, T: float,
        initial_coarse: Dict[str, NDArray],
        a0: NDArray, a1: NDArray, a2: NDArray, a3: NDArray,
        b1: int, b2: int, b3: int,
        progress_callback: Optional[Callable[[float, int, int], None]] = None
    ):
        self.c = c
        self.mu = mu
        self.c_0 = c_0
        self.nu = nu
        self.epsilon = epsilon
        self.d = d
        self.e = e
        self.f = f
        self.eta = eta
        self.D_a = D_a
        self.D_s = D_s
        self.D_y = D_y

        self.n1 = n_coarse
        self.m1 = m_coarse
        self.T = T

        self.n2 = (2 * self.n1) - 1
        self.m2 = 4 * self.m1

        self.h1 = 1.0 / (self.n1 - 1)
        self.tau1 = self.T / self.m1
        self.h2 = 1.0 / (self.n2 - 1)
        self.tau2 = self.T / self.m2

        self.x1 = initial_coarse['x']
        self.a1 = initial_coarse['a'].copy()
        self.s1 = initial_coarse['s'].copy()
        self.y1 = initial_coarse['y'].copy()

        # Инициализация тонкой сетки
        self.x2 = np.linspace(0, 1, self.n2)
        self.a2 = (a0[0] + a1[0]*np.cos(b1*np.pi*self.x2) +
                   a2[0]*np.cos(b2*np.pi*self.x2) + a3[0]*np.cos(b3*np.pi*self.x2))
        self.s2 = (a0[1] + a1[1]*np.cos(b1*np.pi*self.x2) +
                   a2[1]*np.cos(b2*np.pi*self.x2) + a3[1]*np.cos(b3*np.pi*self.x2))
        self.y2 = (a0[2] + a1[2]*np.cos(b1*np.pi*self.x2) +
                   a2[2]*np.cos(b2*np.pi*self.x2) + a3[2]*np.cos(b3*np.pi*self.x2))

        self.progress_callback = progress_callback

        # Для хранения промежуточных слоёв
        self.base_data: GridData = []
        self.control_data: GridData = []

        # Отдельные списки для сохранения в БД (прореженные при больших m)
        self.base_data_for_db: GridData = []
        self.control_data_for_db: GridData = []

        # Разница между сетками
        self.max_a_diff = 0.0
        self.max_s_diff = 0.0
        self.max_y_diff = 0.0

        # Сохраняем ВСЕ слои (каждый шаг по времени)
        self.layers_to_save1 = list(range(self.m1 + 1))        # 0, 1, 2, ..., m1
        self.layers_to_save2 = [i * 4 for i in range(self.m1 + 1)]  # 0, 4, 8, ..., m1*4

    def _save_if_needed(self, step: int):
        """Сохраняет слой, если он в списке layers_to_save"""
        if step in self.layers_to_save1:
            self.base_data.append({
                "layer": step,
                "x": self.x1.copy(),
                "a": self.a1.copy(),
                "s": self.s1.copy(),
                "y": self.y1.copy()
            })

        control_step = step * 4
        if control_step in self.layers_to_save2:
            self.control_data.append({
                "layer": control_step,
                "x": self.x2.copy(),
                "a": self.a2.copy(),
                "s": self.s2.copy(),
                "y": self.y2.copy()
            })

    def _save_to_db_lists(self, step: int):
        """Сохраняет текущий слой в прореженные списки для БД (только при больших m)"""
        if self.m1 <= 10000:
            # Если m маленький — сохраняем всё в оба списка
            self.base_data_for_db.append({
                "layer": step,
                "x": self.x1.copy(),
                "a": self.a1.copy(),
                "s": self.s1.copy(),
                "y": self.y1.copy()
            })
            control_step = step * 4
            if control_step <= self.m2:
                self.control_data_for_db.append({
                    "layer": control_step,
                    "x": self.x2.copy(),
                    "a": self.a2.copy(),
                    "s": self.s2.copy(),
                    "y": self.y2.copy()
                })
            return

        # === Логика прореживания для больших m ===
        total_steps = self.m1
        TARGET_TOTAL = 5000
        EARLY = min(1500, total_steps // 100)
        LATE = EARLY
        MIDDLE = max(0, TARGET_TOTAL - EARLY - LATE)

        save_base = False

        if step < EARLY:
            save_base = True
        elif step >= total_steps - LATE + 1:
            save_base = True
        elif MIDDLE > 0:
            start_middle = EARLY
            end_middle = total_steps - LATE
            middle_length = end_middle - start_middle + 1
            if middle_length > 0:
                step_in_middle = (step - start_middle) % max(1, (middle_length + MIDDLE - 1) // MIDDLE)
                if step_in_middle == 0:
                    save_base = True

        if save_base:
            self.base_data_for_db.append({
                "layer": step,
                "x": self.x1.copy(),
                "a": self.a1.copy(),
                "s": self.s1.copy(),
                "y": self.y1.copy()
            })

            control_step = step * 4
            if control_step <= self.m2:
                self.control_data_for_db.append({
                    "layer": control_step,
                    "x": self.x2.copy(),
                    "a": self.a2.copy(),
                    "s": self.s2.copy(),
                    "y": self.y2.copy()
                })

    def _update_control_diff(self, step: int):
        """Обновляет максимальную разницу между грубой и тонкой сеткой"""
        if step > 0:
            a_control = self.a2[::2]
            s_control = self.s2[::2]
            y_control = self.y2[::2]

            a_diff = np.abs(self.a1 - a_control)
            s_diff = np.abs(self.s1 - s_control)
            y_diff = np.abs(self.y1 - y_control)

            self.max_a_diff = max(self.max_a_diff, float(np.max(a_diff)))
            self.max_s_diff = max(self.max_s_diff, float(np.max(s_diff)))
            self.max_y_diff = max(self.max_y_diff, float(np.max(y_diff)))

    def solve(self) -> tuple[GridData, GridData, float, float, float]:
        """
        Основной цикл решения.
        Возвращает: (base_data, control_data, max_a_diff, max_s_diff, max_y_diff)
        """
        a1_new = np.zeros(self.n1)
        s1_new = np.zeros(self.n1)
        y1_new = np.zeros(self.n1)
        a2_new = np.zeros(self.n2)
        s2_new = np.zeros(self.n2)
        y2_new = np.zeros(self.n2)

        last_progress = -1

        for step in range(self.m1 + 1):
            # === Сохраняем ВСЕ слои ===
            self.base_data.append({
                "layer": step,
                "x": self.x1.copy(),
                "a": self.a1.copy(),
                "s": self.s1.copy(),
                "y": self.y1.copy()
            })

            # Контрольная сетка — сохраняем соответствующий момент
            control_step = step * 4
            if control_step <= self.m2:
                self.control_data.append({
                    "layer": control_step,
                    "x": self.x2.copy(),
                    "a": self.a2.copy(),
                    "s": self.s2.copy(),
                    "y": self.y2.copy()
                })

            self._save_to_db_lists(step)

            # Прогресс
            if self.progress_callback:
                progress = (step / self.m1) * 100
                if int(progress) > last_progress or step == self.m1:
                    last_progress = int(progress)
                    self.progress_callback(progress, step, self.m1)

            # Обновляем максимальные разности
            self._update_control_diff(step)

            # Явная схема на грубой сетке
            if step < self.m1:
                # Внутренние узлы
                a1_new[1:-1] = self.a1[1:-1] + self.tau1 * (
                    self.c * self.a1[1:-1]**2 * self.s1[1:-1] - self.mu * self.a1[1:-1] +
                    self.D_a * (self.a1[2:] - 2*self.a1[1:-
                                1] + self.a1[:-2]) / self.h1**2
                )
                s1_new[1:-1] = self.s1[1:-1] + self.tau1 * (
                    self.c_0 - self.c * self.a1[1:-1]**2 * self.s1[1:-1] - self.nu * self.s1[1:-1] -
                    self.epsilon * self.s1[1:-1] * self.y1[1:-1] +
                    self.D_s * (self.s1[2:] - 2*self.s1[1:-
                                1] + self.s1[:-2]) / self.h1**2
                )
                y1_new[1:-1] = self.y1[1:-1] + self.tau1 * (
                    self.d * self.a1[1:-1] - self.e * self.y1[1:-1] +
                    (self.eta * self.y1[1:-1]**2) / (1 + self.f * self.y1[1:-1]**2) +
                    self.D_y * (self.y1[2:] - 2*self.y1[1:-
                                1] + self.y1[:-2]) / self.h1**2
                )

                # Граничные узлы (односторонняя аппроксимация)
                for var, D, new_var in [
                    (self.a1, self.D_a, a1_new),
                    (self.s1, self.D_s, s1_new),
                    (self.y1, self.D_y, y1_new)
                ]:
                    new_var[0] = var[0] + self.tau1 * \
                        (2 * D * (var[1] - var[0]) / self.h1**2)
                    new_var[-1] = var[-1] + self.tau1 * \
                        (2 * D * (var[-2] - var[-1]) / self.h1**2)

                # Специально для a и s (источники)
                a1_new[0] += self.tau1 * \
                    (self.c * self.a1[0]**2 *
                     self.s1[0] - self.mu * self.a1[0])
                s1_new[0] += self.tau1 * (self.c_0 - self.c * self.a1[0]**2 * self.s1[0] -
                                          self.nu * self.s1[0] - self.epsilon * self.s1[0] * self.y1[0])
                y1_new[0] += self.tau1 * (self.d * self.a1[0] - self.e * self.y1[0] + (
                    self.eta * self.y1[0]**2) / (1 + self.f * self.y1[0]**2))

                a1_new[-1] += self.tau1 * \
                    (self.c * self.a1[-1]**2 *
                     self.s1[-1] - self.mu * self.a1[-1])
                s1_new[-1] += self.tau1 * (self.c_0 - self.c * self.a1[-1]**2 * self.s1[-1] -
                                           self.nu * self.s1[-1] - self.epsilon * self.s1[-1] * self.y1[-1])
                y1_new[-1] += self.tau1 * (self.d * self.a1[-1] - self.e * self.y1[-1] + (
                    self.eta * self.y1[-1]**2) / (1 + self.f * self.y1[-1]**2))

                # Обновление
                self.a1, self.s1, self.y1 = a1_new.copy(), s1_new.copy(), y1_new.copy()

            # Тонкая сетка — 4 подшага
            for _ in range(4 if step < self.m1 else 0):
                a2_new[1:-1] = self.a2[1:-1] + self.tau2 * (
                    self.c * self.a2[1:-1]**2 * self.s2[1:-1] - self.mu * self.a2[1:-1] +
                    self.D_a * (self.a2[2:] - 2*self.a2[1:-
                                1] + self.a2[:-2]) / self.h2**2
                )
                s2_new[1:-1] = self.s2[1:-1] + self.tau2 * (
                    self.c_0 - self.c * self.a2[1:-1]**2 * self.s2[1:-1] - self.nu * self.s2[1:-1] -
                    self.epsilon * self.s2[1:-1] * self.y2[1:-1] +
                    self.D_s * (self.s2[2:] - 2*self.s2[1:-
                                1] + self.s2[:-2]) / self.h2**2
                )
                y2_new[1:-1] = self.y2[1:-1] + self.tau2 * (
                    self.d * self.a2[1:-1] - self.e * self.y2[1:-1] +
                    (self.eta * self.y2[1:-1]**2) / (1 + self.f * self.y2[1:-1]**2) +
                    self.D_y * (self.y2[2:] - 2*self.y2[1:-
                                1] + self.y2[:-2]) / self.h2**2
                )

                # Граничные узлы
                for var, D, new_var in [(self.a2, self.D_a, a2_new), (self.s2, self.D_s, s2_new), (self.y2, self.D_y, y2_new)]:
                    new_var[0] = var[0] + self.tau2 * \
                        (2 * D * (var[1] - var[0]) / self.h2**2)
                    new_var[-1] = var[-1] + self.tau2 * \
                        (2 * D * (var[-2] - var[-1]) / self.h2**2)

                a2_new[0] += self.tau2 * \
                    (self.c * self.a2[0]**2 *
                     self.s2[0] - self.mu * self.a2[0])
                s2_new[0] += self.tau2 * (self.c_0 - self.c * self.a2[0]**2 * self.s2[0] -
                                          self.nu * self.s2[0] - self.epsilon * self.s2[0] * self.y2[0])
                y2_new[0] += self.tau2 * (self.d * self.a2[0] - self.e * self.y2[0] + (
                    self.eta * self.y2[0]**2) / (1 + self.f * self.y2[0]**2))

                a2_new[-1] += self.tau2 * \
                    (self.c * self.a2[-1]**2 *
                     self.s2[-1] - self.mu * self.a2[-1])
                s2_new[-1] += self.tau2 * (self.c_0 - self.c * self.a2[-1]**2 * self.s2[-1] -
                                           self.nu * self.s2[-1] - self.epsilon * self.s2[-1] * self.y2[-1])
                y2_new[-1] += self.tau2 * (self.d * self.a2[-1] - self.e * self.y2[-1] + (
                    self.eta * self.y2[-1]**2) / (1 + self.f * self.y2[-1]**2))

                self.a2, self.s2, self.y2 = a2_new.copy(), s2_new.copy(), y2_new.copy()

        return (
            self.base_data,
            self.control_data,
            self.base_data_for_db,    # прореженные — для БД
            self.control_data_for_db, # прореженные — для БД
            self.max_a_diff,
            self.max_s_diff,
            self.max_y_diff
        )
