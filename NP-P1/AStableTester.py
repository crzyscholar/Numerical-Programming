import numpy as np
import matplotlib.pyplot as plt

def tester(k, h, t):
    N = int(t/h)
    t = np.linspace(0, t, N+1)
    y = np.zeros(N+1, dtype=complex)
    y[0] = 1
    
    for n in range(N):
        y[n+1] = y[n] + h*k*y[n]
    
    return t, y

def implicit_euler(k, h, t_end):
    """
    Applies the implicit Euler method to the test equation y' = ky.

    Args:
        k: The complex constant in the test equation.
        h: The step size.
        t_end: The end time for the simulation.

    Returns:
        A tuple containing the time array and the solution array.
        Returns None if h is too large and division by zero is imminent.
    """
    N = int(t_end / h)
    t = np.linspace(0, t_end, N + 1)
    y = np.zeros(N + 1, dtype=complex)
    y[0] = 1

    # Check if (1 - h*k) is close to zero to prevent division errors.
    if abs(1 - h*k) < 1e-10:  # Adjust tolerance as needed
      print("Warning: Step size h is too large, resulting in near division by zero. Returning None.")
      return None
    
    for n in range(N):
        y[n + 1] = y[n] / (1 - h * k)

    return t, y


def rk4_step(k, y, h):
    """
    Take a single RK4 step for the test equation y' = ky
    
    Parameters:
    k: complex - coefficient in test equation
    y: complex - current solution value
    h: float - step size
    
    Returns:
    complex - next solution value
    """
    # For test equation y' = ky, the stages simplify to:
    k1 = k * y
    k2 = k * (y + h/2 * k1)
    k3 = k * (y + h/2 * k2)
    k4 = k * (y + h * k3)
    
    return y + h * (k1 + 2*k2 + 2*k3 + k4) / 6

def solve_rk4(k, h, t):
    """
    Solve y' = ky with y(0) = 1 using RK4
    
    Parameters:
    k: complex - coefficient in test equation
    T: float - end time
    h: float - step size
    
    Returns:
    t: array - time points
    y: array - solution values
    """
    N = int(T/h)
    t = np.linspace(0, T, N+1)
    y = np.zeros(N+1, dtype=complex)
    y[0] = 1  # initial condition
    
    for n in range(N):
        y[n+1] = rk4_step(k, y[n], h)
    
    return t, y

def exact_solution(k, t):
    return np.exp(k*t)

h = 0.001  # step size
T = 100    # end time

k1 = -10 + 20j
t1, y1 = implicit_euler(k1, h, T)
exact1 = exact_solution(k1, t1)

plt.plot(t1, np.real(y1), 'b-', label='Implicit Euler')
plt.plot(t1, np.real(exact1), 'r-', label='Exact')
#plt.plot(t1, np.abs(exact1-y1), "g-", label="Error")
plt.title(f'Solution for k={k1}')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.show()