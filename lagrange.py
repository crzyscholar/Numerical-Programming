import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.sin(x)


x_points = np.array([0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])
y_points = f(x_points)


def lagrange_interpolation(x, x_points, y_points):
    n = len(x_points)
    L_x = 0
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if j != i:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        L_x += term
    return L_x


x_values = np.linspace(0, np.pi / 2, 100)


f_values = f(x_values)


L_values = []
error_values = []

for x in x_values:
    L_x = lagrange_interpolation(x, x_points, y_points)
    L_values.append(L_x)
    error_values.append(abs(f(x) - L_x))


L_values = np.array(L_values)
error_values = np.array(error_values)


fig, ax1 = plt.subplots(figsize=(10, 6))


ax1.plot(x_values, f_values, label="f(x) = sin(x)", color="black")
ax1.plot(x_values, L_values, label="L(x) - Lagrange Polynomial", color="red", linestyle="--")
ax1.scatter(x_points, y_points, color="red", zorder=5, label="Interpolation Points")
for i in range(len(x_points)):
    ax1.plot([x_points[i], x_points[i]], [0, y_points[i]], color="lightgray", linestyle="--")


ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title("Function f(x), Lagrange Polynomial L(x), and Interpolation Error |f(x) - L(x)|")
ax1.set_xlim(0, np.pi / 2)
ax1.legend(loc="upper left")


ax2 = ax1.twinx()
ax2.plot(x_values, error_values, label="Error(x) = |f(x) - L(x)|", color="green", linestyle=":")
ax2.set_ylabel("Error y")
ax2.tick_params(axis='y', labelcolor="black")


ax1.grid()
ax1.set_ylim(0, 1.1)
ax2.set_ylim(0, np.max(error_values) * 1.1)


plt.show()
