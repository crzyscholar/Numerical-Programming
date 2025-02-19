import numpy as np
import matplotlib.pyplot as plt

def fixed_point_iteration(g, x0, tol, max_iter=1000):
    iterations = [x0]
    for _ in range(max_iter):
        x1 = g(x0)
        iterations.append(x1)
        if abs(x1 - x0) < tol:
            break
        x0 = x1
    return iterations

def g(x):
    return np.cbrt(4 * x + 9)

def f(x):
    return x**3 - 4 * x - 9

x0 = 2
error_tolerance = 1e-6

iterations = fixed_point_iteration(g, x0, error_tolerance)

x_values = np.linspace(1.5, 3, 500)
y_values = f(x_values)

plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label="f(x) = x^3 - 4x - 9", color="blue")
plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
plt.scatter(iterations, [f(x) for x in iterations], color="red", label="Iterations")
plt.plot(iterations, [f(x) for x in iterations], linestyle="--", color="red", alpha=0.6)

for i, x in enumerate(iterations):
    plt.text(x, f(x), f"x{i}", fontsize=9, verticalalignment="bottom")

plt.title("Fixed Point Iteration and Convergence")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid()
plt.show()

print("Iterations:", iterations)
print("Approximate root:", iterations[-1])

