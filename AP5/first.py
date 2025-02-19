import numpy as np
import matplotlib.pyplot as plt

def ode_system(t, state, params):
    x, y, z, p = state
    a, K, b, c, d, e, f, g, h, i, j, k = params

    dx_dt = a * x * (1 - x / K) - b * x * y
    dy_dt = c * x * y - d * y - e * y * z
    dz_dt = f * y * z - g * z
    dp_dt = h * x + i * y + j * z - k * p
    
    return np.array([dx_dt, dy_dt, dz_dt, dp_dt])

def rk4_step(func, t, state, dt, params):
    k1 = func(t, state, params)
    k2 = func(t + dt/2, state + dt/2 * k1, params)
    k3 = func(t + dt/2, state + dt/2 * k2, params)
    k4 = func(t + dt, state + dt * k3, params)
    return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

def simulate(ode_system, state0, params, t_span, dt):
    t_values = np.arange(t_span[0], t_span[1], dt)
    states = np.zeros((len(t_values), len(state0)))

    states[0] = state0
    for i in range(1, len(t_values)):
        states[i] = rk4_step(ode_system, t_values[i-1], states[i-1], dt, params)

    return t_values, states

params = [0.5, 100, 0.02, 0.01, 0.1, 0.01, 0.02, 0.1, 0.05, 0.02, 0.03, 0.1]
state0 = [50, 5, 2, 0]  
t_span = [0, 50]  
dt = 0.1  

t_values, states = simulate(ode_system, state0, params, t_span, dt)

x_values, y_values, z_values, p_values = states.T

plt.figure(figsize=(12, 8))
plt.plot(t_values, x_values, label='Prey (x)')
plt.plot(t_values, y_values, label='Predator (y)')
plt.plot(t_values, z_values, label='Scavenger (z)')
plt.plot(t_values, p_values, label='Pollution (p)')
plt.xlabel('Time')
plt.ylabel('Population / Level')
plt.title('Predator-Prey-Scavenger Model with Environmental Effects')
plt.legend()
plt.grid()
plt.show()
