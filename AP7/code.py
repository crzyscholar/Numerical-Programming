import numpy as np
import matplotlib.pyplot as plt


a, b, c, d = 0.5, 0.02, 0.5, 0.01


def f(t, y):
    y1, y2 = y
    dy1 = a * y1 - b * y1 * y2
    dy2 = -c * y2 + d * y1 * y2
    return np.array([dy1, dy2])


def adams_bashforth_2(f, t0, y0, h, steps):
    t = np.linspace(t0, t0 + steps * h, steps + 1)
    y = np.zeros((steps + 1, len(y0)))
    y[0] = y0


    y[1] = y[0] + h * f(t[0], y[0])


    for n in range(1, steps):
        y[n+1] = y[n] + h * (3/2 * f(t[n], y[n]) - 1/2 * f(t[n-1], y[n-1]))
    return t, y


t0, y0 = 0, [40, 9]
h = 0.1
steps = 500


t, y = adams_bashforth_2(f, t0, y0, h, steps)


plt.figure(figsize=(10, 6))
plt.plot(t, y[:, 0], label="Prey (y1)", color="blue")
plt.plot(t, y[:, 1], label="Predator (y2)", color="red")
plt.xlabel("Time (t)")
plt.ylabel("Population")
plt.title("Predator-Prey System (Adams-Bashforth 2-Step Method)")
plt.legend()
plt.grid()
plt.show()
