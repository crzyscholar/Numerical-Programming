import time
from helpers import *
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

km=0.1
g=9.81

def acceleration(vx, vy, km, g):
    v = np.sqrt(vx**2 + vy**2)
    if v == 0:
        return 0, g
    
    ax = - km * vx * v
    ay = (- g) - km * (vy * v)
    
    return ax, ay

def simulate_trajectory(v0, angle, dt=0.001):
    theta = np.radians(angle)

    # Initial conditions
    x = 0.0
    y = 0.0
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)

    xs = [x]
    ys = [y]

    t = 0
    max_time = 3
    
    while y >= 0 and t < max_time:
        ax, ay = acceleration(vx, vy, km, g)
        
        x += vx * dt
        y += vy * dt
        vx += ax * dt
        vy += ay * dt
        
        xs.append(x)
        ys.append(y)
        t += dt
    
    return (x, y), (np.array(xs), np.array(ys))

def simulate_trajectory_rk4(v0, angle, dt=0.001):
    """Simulate the trajectory using Runge-Kutta method"""
    theta = np.radians(angle)

    # Initial conditions
    x = 0.0
    y = 0.0
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)

    xs = [x]
    ys = [y]

    def derivatives(x, y, vx, vy):
        ax, ay = acceleration(vx, vy, km, g)
        return vx, vy, ax, ay

    t = 0
    max_time = 3  # Maximum simulation time

    while y >= 0 and t < max_time:
        # RK4 integration steps
        k1x, k1y, k1vx, k1vy = derivatives(x, y, vx, vy)
        k2x, k2y, k2vx, k2vy = derivatives(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, vx + 0.5 * dt * k1vx, vy + 0.5 * dt * k1vy)
        k3x, k3y, k3vx, k3vy = derivatives(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, vx + 0.5 * dt * k2vx, vy + 0.5 * dt * k2vy)
        k4x, k4y, k4vx, k4vy = derivatives(x + dt * k3x, y + dt * k3y, vx + dt * k3vx, vy + dt * k3vy)

        # Update position and velocity
        x += (dt / 6) * (k1x + 2 * k2x + 2 * k3x + k4x)
        y += (dt / 6) * (k1y + 2 * k2y + 2 * k3y + k4y)
        vx += (dt / 6) * (k1vx + 2 * k2vx + 2 * k3vx + k4vx)
        vy += (dt / 6) * (k1vy + 2 * k2vy + 2 * k3vy + k4vy)

        xs.append(x)
        ys.append(y)
        t += dt

    return (x, y), (np.array(xs), np.array(ys))

def implicit_system(vars, x_curr, y_curr, vx_curr, vy_curr, km, g, dt):
    # vars = [x_next, y_next, vx_next, vy_next]
    x_next, y_next, vx_next, vy_next = vars
    
    # Calculate acceleration at the next time step
    ax_next, ay_next = acceleration(vx_next, vy_next, km, g)
    
    # Implicit Euler equations
    eq1 = x_next - (x_curr + vx_next * dt)
    eq2 = y_next - (y_curr + vy_next * dt)
    eq3 = vx_next - (vx_curr + ax_next * dt)
    eq4 = vy_next - (vy_curr + ay_next * dt)
    
    return [eq1, eq2, eq3, eq4]

def simulate_trajectory_implicit(v0, angle, dt=0.001):
    theta = np.radians(angle)

    # Initial conditions
    x = 0.0
    y = 0.0
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)

    xs = [x]
    ys = [y]
    
    t = 0
    max_time = 3
    
    while y >= 0 and t < max_time:
        # Initial guess for the next state (using explicit Euler)
        ax_curr, ay_curr = acceleration(vx, vy, km, g)
        x_guess = x + vx * dt
        y_guess = y + vy * dt
        vx_guess = vx + ax_curr * dt
        vy_guess = vy + ay_curr * dt
        
        # Solve the implicit system
        guess = [x_guess, y_guess, vx_guess, vy_guess]
        solution = fsolve(implicit_system, guess, args=(x, y, vx, vy, km, g, dt))
        
        # Update states
        x, y, vx, vy = solution
        
        xs.append(x)
        ys.append(y)
        t += dt
    
    return (x, y), (np.array(xs), np.array(ys))

def main():
    v = 100
    angle = 45

    startRK = time.perf_counter()
    (_, _), (rkxs, rkys) = simulate_trajectory_rk4(v, angle)
    endRK = time.perf_counter()

    startEI = time.perf_counter()
    (_, _), (eixs, eiys) = simulate_trajectory_implicit(v, angle)
    endEI = time.perf_counter()

    print(f"RK: {endRK - startRK}")
    print(f"Euler (Implicit): {endEI - startEI}")

    plt.plot(rkxs, rkys, 'g-', label="RK")
    plt.plot(eixs, eiys, 'r-', label="EI")
    plt.plot(np.linspace(0, rkxs.max(), rkxs.shape[0]), np.abs(rkxs-rkxs), 'r-', label="Error")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()