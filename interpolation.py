import matplotlib.pyplot as plt
import math

def f(z):
    return math.exp(z)


def chebyshev_points(a, b, n):
    return [((a + b) / 2) + ((b - a) / 2) * math.cos((2 * i + 1) * math.pi / (2 * n)) for i in range(n)]


def piecewise_linear_interpolation(points, values, z):
    for i in range(len(points) - 1):
        if points[i] <= z <= points[i + 1]:
            x0, x1 = points[i], points[i + 1]
            y0, y1 = values[i], values[i + 1]
            return y0 + (y1 - y0) * (z - x0) / (x1 - x0)
    return None


def piecewise_polynomial_interpolation(points, values, z):
    for i in range(len(points) - 2):
        if points[i] <= z <= points[i + 2]:
            x0, x1, x2 = points[i], points[i + 1], points[i + 2]
            y0, y1, y2 = values[i], values[i + 1], values[i + 2]
            # Quadratic interpolation formula
            L0 = ((z - x1) * (z - x2)) / ((x0 - x1) * (x0 - x2))
            L1 = ((z - x0) * (z - x2)) / ((x1 - x0) * (x1 - x2))
            L2 = ((z - x0) * (z - x1)) / ((x2 - x0) * (x2 - x1))
            return y0 * L0 + y1 * L1 + y2 * L2
    return None


def cubic_spline_interpolation(points, values, z):
    n = len(points) - 1
    h = [(points[i+1] - points[i]) for i in range(n)]
    alpha = [0] * (n + 1)
    for i in range(1, n):
        alpha[i] = (3 / h[i]) * (values[i+1] - values[i]) - (3 / h[i-1]) * (values[i] - values[i-1])


    l = [1] + [0] * n
    mu = [0] * (n + 1)
    z_vals = [0] * (n + 1)
    for i in range(1, n):
        l[i] = 2 * (points[i+1] - points[i-1]) - h[i-1] * mu[i-1]
        mu[i] = h[i] / l[i]
        z_vals[i] = (alpha[i] - h[i-1] * z_vals[i-1]) / l[i]
    l[n] = 1


    c = [0] * (n + 1)
    b = [0] * n
    d = [0] * n
    for j in range(n - 1, -1, -1):
        c[j] = z_vals[j] - mu[j] * c[j+1]
        b[j] = (values[j+1] - values[j]) / h[j] - h[j] * (c[j+1] + 2 * c[j]) / 3
        d[j] = (c[j+1] - c[j]) / (3 * h[j])


    for i in range(n):
        if points[i] <= z <= points[i+1]:
            dx = z - points[i]
            return values[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
    return None



test_points = [a + i * (b - a) / 100 for i in range(101)]
actual_values = [f(z) for z in test_points]


linear_values = [piecewise_linear_interpolation(chebyshev_nodes, f_values, z) for z in test_points]
poly_values = [piecewise_polynomial_interpolation(chebyshev_nodes, f_values, z) for z in test_points]
spline_values = [cubic_spline_interpolation(chebyshev_nodes, f_values, z) for z in test_points]


def mean_squared_error(interpolated, actual):
    return sum((i - j) ** 2 for i, j in zip(interpolated, actual)) / len(actual)