import numpy as np


def newton_method(c, mu, C0, V, eps, d, e, f, eta, a, s, y, max_iter):
    x = np.array([a, s, y])

    max_iterations = max_iter
    tolerance = 1e-6

    iteration = 0

    while iteration < max_iterations:
        f1 = c * x[0]**2 * x[1] - mu * x[0]
        f2 = C0 - c * x[0]**2 * x[1] - V * x[1] - eps * x[1] * x[2]
        f3 = d * x[0] - e * x[2] + (eta * x[2]**2) / (1 + f * x[2]**2)
        F = np.array([f1, f2, f3])

        if np.linalg.norm(F) < tolerance:
            break

        J11 = 2 * c * x[0] * x[1] - mu
        J12 = c * x[0]**2
        J13 = 0
        J21 = -2 * c * x[0] * x[1]
        J22 = -c * x[0]**2 - V - eps * x[2]
        J23 = -eps * x[1]
        J31 = d
        J32 = 0
        J33 = -e + (2 * eta * x[2] * (1 + f * x[2]**2) - 2 *
                    f * x[2]**3 * eta) / (1 + f * x[2]**2)**2

        J = np.array([
            [J11, J12, J13],
            [J21, J22, J23],
            [J31, J32, J33]
        ])

        delta_x = np.linalg.solve(J, -F)

        x = x + delta_x

        iteration += 1

    f1 = c * x[0]**2 * x[1] - mu * x[0]
    f2 = C0 - c * x[0]**2 * x[1] - V * x[1] - eps * x[1] * x[2]
    f3 = d * x[0] - e * x[2] + (eta * x[2]**2) / (1 + f * x[2]**2)
    F = np.array([f1, f2, f3])
    residual_norm = np.linalg.norm(F)

    return iteration, x, f1, f2, f3, residual_norm
