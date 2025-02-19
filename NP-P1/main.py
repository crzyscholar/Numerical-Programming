import numpy as np
import matplotlib.pyplot as plt
from helpers import *
import matplotlib.animation as animation
from scipy.optimize import fsolve

class BallisticCalculator:
    def __init__(self):
        self.g = 9.81  # gravitacia
        self.k = 0.001  # air drag
        self.m = 1  # masa
        self.dt = 0.001
        
    def acceleration(self, vx, vy):
        v = np.sqrt(vx**2 + vy**2)
        if v == 0:
            return 0, -self.g
        
        ax = - (self.k/self.m) * vx * v
        ay = (- self.g) - (self.k/self.m) * vy * v
        
        return ax, ay

    def implicit_system(self, vars, x_curr, y_curr, vx_curr, vy_curr):
        x_next, y_next, vx_next, vy_next = vars
        
        ax_next, ay_next = self.acceleration(vx_next, vy_next)
        
        eq1 = x_next - (x_curr + vx_next * self.dt)
        eq2 = y_next - (y_curr + vy_next * self.dt)
        eq3 = vx_next - (vx_curr + ax_next * self.dt)
        eq4 = vy_next - (vy_curr + ay_next * self.dt)
        
        return [eq1, eq2, eq3, eq4]

    def simulate_trajectory_implicit(self, vx0, vy0):
        x = 0
        y = 0
        vx = vx0
        vy = vy0
        
        xs = [x]
        ys = [y]
        
        t = 0
        max_time = 4 # flight time
        
        while y >= 0 and t < max_time:
            # Initial guess (chveulebrivi Euler-it)
            ax_curr, ay_curr = self.acceleration(vx, vy)
            x_guess = x + vx * self.dt
            y_guess = y + vy * self.dt
            vx_guess = vx + ax_curr * self.dt
            vy_guess = vy + ay_curr * self.dt
            
            guess = [x_guess, y_guess, vx_guess, vy_guess]
            solution = fsolve(self.implicit_system, guess, args=(x, y, vx, vy))
            
            x, y, vx, vy = solution
            
            xs.append(x)
            ys.append(y)
            t += self.dt
        
        return (x, y), (np.array(xs), np.array(ys))

    def simulate_trajectory_rk4(self, vy0, vx0):
        # Initial values
        x = 0.0
        y = 0.0
        vx = vx0
        vy = vy0

        xs = [x]
        ys = [y]

        def derivatives(x, y, vx, vy):
            ax, ay = self.acceleration(vx, vy)
            return vx, vy, ax, ay

        t = 0
        max_time = 2  # flight time

        while y >= 0 and t < max_time:
            k1x, k1y, k1vx, k1vy = derivatives(x, y, vx, vy)
            k2x, k2y, k2vx, k2vy = derivatives(x + 0.5 * self.dt * k1x, y + 0.5 * self.dt * k1y, vx + 0.5 * self.dt * k1vx, vy + 0.5 * self.dt * k1vy)
            k3x, k3y, k3vx, k3vy = derivatives(x + 0.5 * self.dt * k2x, y + 0.5 * self.dt * k2y, vx + 0.5 * self.dt * k2vx, vy + 0.5 * self.dt * k2vy)
            k4x, k4y, k4vx, k4vy = derivatives(x + self.dt * k3x, y + self.dt * k3y, vx + self.dt * k3vx, vy + self.dt * k3vy)

            x += (self.dt / 6) * (k1x + 2 * k2x + 2 * k3x + k4x)
            y += (self.dt / 6) * (k1y + 2 * k2y + 2 * k3y + k4y)
            vx += (self.dt / 6) * (k1vx + 2 * k2vx + 2 * k3vx + k4vx)
            vy += (self.dt / 6) * (k1vy + 2 * k2vy + 2 * k3vy + k4vy)

            xs.append(x)
            ys.append(y)
            t += self.dt

        return (x, y), (np.array(xs), np.array(ys))

    def shooting_method(self, target_x, target_y, initial_guess=(10, 10)):
        max_iterations = 100
        tolerance = 0.1
        
        # initial guess
        params = np.array(initial_guess, dtype=float)
        
        for _ in range(max_iterations):
            (x, y), _ = self.simulate_trajectory_implicit(params[0], params[1])
            F = np.array([x - target_x, y - target_y])
            error = np.sqrt(np.sum(F**2))
            
            if error < tolerance:
                return params, error
            
            # jacobian
            J = np.zeros((2, 2))
            
            # vx-is mimart
            (x_dvx, y_dvx), _ = self.simulate_trajectory_implicit(params[0] + self.dt, params[1])
            J[0,0] = (x_dvx - x) / self.dt
            J[1,0] = (y_dvx - y) / self.dt
            
            # vy-is mimart
            (x_dvy, y_dvy), _ = self.simulate_trajectory_implicit(params[0], params[1] + self.dt)
            J[0,1] = (x_dvy - x) / self.dt
            J[1,1] = (y_dvy - y) / self.dt
            
            # http://www.ohiouniversityfaculty.com/youngt/IntNumMeth/lecture13.pdf
            # https://en.wikipedia.org/wiki/Newton%27s_method#Multidimensional_formulations
            try:
                delta = np.linalg.solve(J, -F)
                params += delta
            except np.linalg.LinAlgError:
                # determinant = 0
                print("Singular matrix")
                params = np.array([np.random.uniform(10, 200), np.random.uniform(10, 200)])
        
        return params, error

    def get_lines(self, v0, angle):
        _, (x, y) = self.simulate_trajectory_implicit(v0, angle)
        return x, y

def main():
    calc = BallisticCalculator()
    path = 'test2.png'

    print("Getting targets")
    centers = get_target_coords(path)
    print("Targets acquired")

    solutions = []
    trajectories = []

    print("Calculations speeds and angles")
    for center in centers:
        x = center[0]
        y = center[1]

        (vx, vy), _ = calc.shooting_method(x, y)
        solutions.append([vx, vy])
        print(f"{vx} {vy}")

        x_traj, y_traj = calc.get_lines(vx, vy)
        trajectories.append((x_traj, y_traj))

    print("Finished calculating speeds and angles")

    fig, ax = plt.subplots()
    ax.set_xlabel('Distance')
    ax.set_ylabel('Height')
    ax.grid(True)

    for center in centers:
        ax.plot(center[0], center[1], 'ro', markersize=15)

    ax.plot(0, 0, 'go', markersize=10, label='Launch point')

    lines = []
    for _ in range(len(trajectories)):
        line, = ax.plot([], [], 'b-', label='Trajectory')
        lines.append(line)

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def animate(frame):
        traj_index = frame // 100
        if traj_index >= len(trajectories):
            return lines
        
        point_index = (frame % 100) + 1
        
        for i in range(traj_index):
            x, y = trajectories[i]
            lines[i].set_data(x, y)
        
        x, y = trajectories[traj_index]
        points = min(int(len(x) * (point_index/100)), len(x))
        lines[traj_index].set_data(x[:points], y[:points])
        
        return lines

    n_frames = len(trajectories) * 100 # 100 frame-i tito traeqtoriaze
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=n_frames, interval=20, blit=True)
    
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()