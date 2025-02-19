import numpy as np
import matplotlib.pyplot as plt

class PredatorPreyModel:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def derivatives(self, z, t):
        x, y = z 
        dxdt = x * (self.a - self.b * y)
        dydt = -y * (self.c - self.d * x)
        return np.array([dxdt, dydt])

    def solve(self, z0, t):
        z = np.zeros((len(t), len(z0)))
        z[0] = z0

        for i in range(1, len(t)):
            dt = t[i] - t[i - 1]
            z[i] = self._rk4_step(z[i - 1], t[i - 1], dt)

        return z

    def _rk4_step(self, z, t, dt):
        k1 = self.derivatives(z, t)
        k2 = self.derivatives(z + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self.derivatives(z + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self.derivatives(z + dt * k3, t + dt)
        return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

a = 0.8
b = 0.1
c = 1.0
d = 0.2
#predator and oprey initial population
x0 = 40 
y0 = 4  
z0 = [x0, y0]
t_span = (0, 50) 
num_points = 1000
t = np.linspace(t_span[0], t_span[1], num_points)

model = PredatorPreyModel(a, b, c, d)
solution = model.solve(z0, t)
x = solution[:, 0]  
y = solution[:, 1] 

plt.figure(figsize=(12, 6))

plt.plot(t, x, label="Prey Population (x(t))", color="blue")
plt.plot(t, y, label="Predator Population (y(t))", color="red")
plt.title("Predator-Prey Model Dynamics")
plt.xlabel("Time")
plt.ylabel("Population")
plt.legend()
plt.grid()
plt.show()

print("Stability Analysis:")
print("The equilibrium points of the system are at (0, 0) and at (c/d, a/b).")
equilibrium_point = (c/d, a/b)
print(f"For the given parameters, the non-trivial equilibrium point is at {equilibrium_point}.")
print("The long-term behavior of the system depends on the initial conditions and parameters.\n")


# equilibrium points are (0,0) and (c/d, a/b). first when both populations are zero and second where the population is non-zero and the two species coexist steadily.
# long term behavior: the prey population increases -> predator population increases -> prey population decreases -> predator population decreases -> the prey population increases -> ......
