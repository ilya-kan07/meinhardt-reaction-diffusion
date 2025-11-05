import numpy as np


def find_graphs(c, mu, C0, V, eps, d, e, f, eta):
    a = np.linspace(0.001, 1, 100)
    s_a = mu / (c * a)

    y = np.linspace(0, 1.2, 1000)
    a_y = (y * (e + e * f * y * y - eta * y)) / (d * (1 + f * y * y))

    discriminant = C0 * C0 * c * c - 4 * c * mu * mu * (eps * y + V)
    aplus_y = np.where(
        discriminant >= 0, (C0 * c + np.sqrt(np.maximum(discriminant, 0))) / (2 * c * mu), np.nan)
    aminus_y = np.where(
        discriminant >= 0, (C0 * c - np.sqrt(np.maximum(discriminant, 0))) / (2 * c * mu), np.nan)

    def find_intersections(y, func1, func2):
        intersections = []
        diff = func1 - func2
        sign_change = np.where(np.diff(np.sign(diff)))[0]
        for idx in sign_change:
            y1, y2 = y[idx], y[idx + 1]
            f1, f2 = diff[idx], diff[idx + 1]
            y_inter = y1 - f1 * (y2 - y1) / (f2 - f1)
            a_inter = np.interp(y_inter, y, func1)

            s_inter = mu / (c * a_inter)

            if 0 <= a_inter:
                intersections.append((a_inter, s_inter, y_inter))
        return intersections

    intersections_plus = find_intersections(y, aplus_y, a_y)
    intersections_minus = find_intersections(y, aminus_y, a_y)
    intersections = intersections_plus + intersections_minus
    intersections.sort(key=lambda x: x[0])

    return a, s_a, y, aplus_y, aminus_y, a_y, intersections
