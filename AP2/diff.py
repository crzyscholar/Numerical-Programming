import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.sin(x)

def exact_derivative(x):
    return np.cos(x)


def forward_difference(x, h):
    return (f(x + h) - f(x)) / h

def backward_difference(x, h):
    return (f(x) - f(x - h)) / h

def central_difference(x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


x = np.pi / 4


h_values = [10**-i for i in range(1, 10)]


errors_forward = []
errors_backward = []
errors_central = []
for h in h_values:
    error_forward = abs(forward_difference(x, h) - exact_derivative(x))
    error_backward = abs(backward_difference(x, h) - exact_derivative(x))
    error_central = abs(central_difference(x, h) - exact_derivative(x))
    errors_forward.append(error_forward)
    errors_backward.append(error_backward)
    errors_central.append(error_central)


plt.figure(figsize=(10, 6))
plt.loglog(h_values, errors_forward, label="Forward Difference Error", marker="o")
plt.loglog(h_values, errors_backward, label="Backward Difference Error", marker="^")
plt.loglog(h_values, errors_central, label="Central Difference Error", marker="s")
plt.xlabel("Step size h")
plt.ylabel("Error")
plt.title("Error in Finite Difference Approximations as h → 0")
plt.legend()
plt.grid(True, which="both")
plt.show()


h_example = 0.1
x_vals = np.linspace(x - 0.5, x + 0.5, 100)
tangent_line = exact_derivative(x) * (x_vals - x) + f(x)
approx_forward = forward_difference(x, h_example) * (x_vals - x) + f(x)
approx_backward = backward_difference(x, h_example) * (x_vals - x) + f(x)
approx_central = central_difference(x, h_example) * (x_vals - x) + f(x)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, f(x_vals), label="f(x) = sin(x)", color="blue")
plt.plot(x_vals, tangent_line, label="Exact Tangent (cos(x))", linestyle="--", color="black")
plt.plot(x_vals, approx_forward, label="Forward Difference Approx.", linestyle="--", color="orange")
plt.plot(x_vals, approx_backward, label="Backward Difference Approx.", linestyle="--", color="purple")
plt.plot(x_vals, approx_central, label="Central Difference Approx.", linestyle="--", color="green")
plt.scatter(x, f(x), color="red", zorder=5, label="Point of Tangency")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Visualization of Tangent Line and Finite Difference Approximations")
plt.legend()
plt.show()
