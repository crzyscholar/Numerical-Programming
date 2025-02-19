import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from helpers import *
from scipy.optimize import fsolve

def acceleration(vx, vy, km, g):
    v = np.sqrt(vx**2 + vy**2)
    if v == 0:
        return 0, g
    
    ax = - km * vx * v
    ay = (- g) - km * (vy * v)
    
    return ax, ay

def simulate_trajectory_rk4(vx0, vy0, ft, dt, km, g, tx, ty, x0 = 0.0, y0 = 0.0):
        x = x0
        y = y0
        vx = vx0
        vy = vy0

        xs = [x]
        ys = [y]

        def derivatives(x, y, vx, vy):
            ax, ay = acceleration(vx, vy, km, g)
            return vx, vy, ax, ay

        t = 0

        while not within_tolerance(x, tx, y, ty) and t < ft:
            k1x, k1y, k1vx, k1vy = derivatives(x, y, vx, vy)
            k2x, k2y, k2vx, k2vy = derivatives(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, vx + 0.5 * dt * k1vx, vy + 0.5 * dt * k1vy)
            k3x, k3y, k3vx, k3vy = derivatives(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, vx + 0.5 * dt * k2vx, vy + 0.5 * dt * k2vy)
            k4x, k4y, k4vx, k4vy = derivatives(x + dt * k3x, y + dt * k3y, vx + dt * k3vx, vy + dt * k3vy)

            x += (dt / 6) * (k1x + 2 * k2x + 2 * k3x + k4x)
            y += (dt / 6) * (k1y + 2 * k2y + 2 * k3y + k4y)
            vx += (dt / 6) * (k1vx + 2 * k2vx + 2 * k3vx + k4vx)
            vy += (dt / 6) * (k1vy + 2 * k2vy + 2 * k3vy + k4vy)

            xs.append(x)
            ys.append(y)
            t += dt

        return (x, y), (np.array(xs), np.array(ys))

def acceleration(vx, vy, km, g):
    v = np.sqrt(vx**2 + vy**2)
    ax = -km * v * vx
    ay = -km * v * vy - g
    return ax, ay

def implicit_system(vars, x_curr, y_curr, vx_curr, vy_curr, km, g, dt):
    x_next, y_next, vx_next, vy_next = vars
    
    ax_next, ay_next = acceleration(vx_next, vy_next, km, g)
    
    eq1 = x_next - (x_curr + vx_next * dt)
    eq2 = y_next - (y_curr + vy_next * dt)
    eq3 = vx_next - (vx_curr + ax_next * dt)
    eq4 = vy_next - (vy_curr + ay_next * dt)
    
    return [eq1, eq2, eq3, eq4]

def simulate_trajectory_implicit(vx0, vy0, ft, dt, km, g, tx, ty, x0=0.0, y0=0.0):
    x = x0
    y = y0
    vx = vx0
    vy = vy0
    
    xs = [x]
    ys = [y]
    
    t = 0
    
    while not within_tolerance(x, tx, y, ty) and t < ft:
        # Initial guess (chveulebrivi Euler-it)
        ax_curr, ay_curr = acceleration(vx, vy, km, g)
        x_guess = x + vx * dt
        y_guess = y + vy * dt
        vx_guess = vx + ax_curr * dt
        vy_guess = vy + ay_curr * dt
        
        guess = [x_guess, y_guess, vx_guess, vy_guess]
        solution = fsolve(implicit_system, guess, args=(x, y, vx, vy, km, g, dt))
        
        x, y, vx, vy = solution
        
        xs.append(x)
        ys.append(y)
        t += dt
    
    return (x, y), (np.array(xs), np.array(ys))

def get_bullet_params(tx, ty, km, g, ft, dt, initial_guess=(10, 10)):
    max_iterations = 100
    tolerance = 0.1  # meters
    
    params = np.array(initial_guess, dtype=float)
    
    for _ in range(max_iterations):
        (x, y), _ = simulate_trajectory_implicit(params[0], params[1], ft, dt, km, g, tx, ty)
        F = np.array([x - tx, y - ty])
        error = np.sqrt(np.sum(F**2))
        
        if error < tolerance:
            return params, error
        
        # jacobian
        J = np.zeros((2, 2))
        
        # vx-is mimart
        (x_dvx, y_dvx), _ = simulate_trajectory_implicit(params[0] + dt, params[1], ft, dt, km, g, tx, ty)
        J[0,0] = (x_dvx - x) / dt
        J[1,0] = (y_dvx - y) / dt
        
        # vy-is mimart
        (x_dvy, y_dvy), _ = simulate_trajectory_implicit(params[0], params[1] + dt, ft, dt, km, g, tx, ty)
        J[0,1] = (x_dvy - x) / dt
        J[1,1] = (y_dvy - y) / dt
        
        # Update parameters using Newton's method
        # http://www.ohiouniversityfaculty.com/youngt/IntNumMeth/lecture13.pdf
        # https://en.wikipedia.org/wiki/Newton%27s_method#Multidimensional_formulations
        try:
            delta = np.linalg.solve(J, -F)
            params += delta
        except np.linalg.LinAlgError:
            # determinant = 0
            print("Singular matrix")
            params = np.array([np.random.uniform(10, 100), np.random.uniform(10, 100)])
    
    return params, error

def main():
    path = 'target.mp4'
    t = 0.4
    dt = 0.001

    km, g, vs, pos = get_velocities_from_video(path)

    # targets next location and predicted trajectory
    (tx, ty), (txs, tys) = simulate_trajectory_implicit(vs[-1][0], vs[-1][1], t, dt, km, g, 0, 0, pos[-1][0], pos[-1][1])

    #bullet predicted params
    bullet_params, _ = get_bullet_params(tx, ty, km, g, t, dt)

    #bullet trajectory (calculated with predicted params)
    _, (bxs, bys) = simulate_trajectory_implicit(bullet_params[0], bullet_params[1], t, dt, km, g, tx, ty)

    psx = [x for [x, _] in pos]
    psy = [y for [_, y] in pos]

    fig, ax = plt.subplots()
    ax.set_xlabel('Distance')
    ax.set_ylabel('Height')
    ax.grid(True)

    ax.plot(tx, ty, "ro") # target predicted location
    ax.plot(psx, psy) # target trajectory from video

    tline, = ax.plot([], [], 'b-', label='Predited target trajectory')
    bline, = ax.plot([], [], 'r-', label='Bullet Trajetory')

    print(f"{km}, {g}, {bullet_params}")

    def init():
        tline.set_data([], [])
        bline.set_data([], [])
        return (tline, bline)

    def animate(frame):
        bx = bxs[:frame]
        by = bys[:frame]

        tx = txs[:frame]
        ty = tys[:frame]

        tline.set_data(tx, ty)
        bline.set_data(bx, by)

        return (tline, bline)

    ani = animation.FuncAnimation(fig=fig, func=animate, init_func=init, frames=int(t/dt), interval=1, blit=True)
    plt.axis('equal')
    plt.tight_layout()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
