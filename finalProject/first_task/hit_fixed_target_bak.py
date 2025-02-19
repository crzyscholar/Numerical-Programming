import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import fsolve


# CP1-dan

def dbscan(X, eps, min_points):
    X = np.asarray(X)
    n_samples = X.shape[0]
    visited = np.zeros(n_samples, dtype=bool)
    clustered = np.zeros(n_samples, dtype=bool)
    clusters = []
    
    def find_neighbors(point_idx):
        distances = np.sqrt(np.sum((X - X[point_idx]) ** 2, axis=1))
        return np.where(distances <= eps)[0]
    
    # mtavari pointebi
    for point_idx in range(n_samples):
        if visited[point_idx]:
            continue
            
        visited[point_idx] = True
        neighbors = find_neighbors(point_idx)
        
        # daskipe ara mtavari pointebi
        if len(neighbors) < min_points:
            continue
            
        current_cluster = [point_idx]
        clustered[point_idx] = True
        
        # mezoblebis gachekva
        seed_points = list(neighbors) 
        
        while seed_points:
            current_point = seed_points.pop(0)
            
            if not visited[current_point]:
                visited[current_point] = True
                point_neighbors = find_neighbors(current_point)
                
                if len(point_neighbors) >= min_points:
                    seed_points.extend([p for p in point_neighbors if p not in seed_points])
            
            if not clustered[current_point]:
                current_cluster.append(current_point)
                clustered[current_point] = True
        
        clusters.append(X[current_cluster])
    
    # datasets - clusters = noise
    noise = X[~clustered]
    
    return clusters, noise

# https://en.wikipedia.org/wiki/Canny_edge_detector#Process
def canny_edge_detection(image, low_threshold=20, high_threshold=40, sigma=1):
    # https://en.wikipedia.org/wiki/Gaussian_blur
    def gaussian_kernel(size, sigma): # https://gist.github.com/FienSoP/49b61a7afdcb992fef0c8ef1b2b3364c
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        return g
    
    def non_maximum_suppression(magnitude, direction):
        height, width = magnitude.shape
        suppressed = np.zeros_like(magnitude)
        
        angle = np.rad2deg(direction) % 180
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                # daxrilobis mixedvit mezoblebis shemowmeba
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180): # horizontaluri
                    neighbors = [magnitude[i,j-1], magnitude[i,j+1]]
                elif 22.5 <= angle[i,j] < 67.5: # diagnali
                    neighbors = [magnitude[i-1,j-1], magnitude[i+1,j+1]]
                elif 67.5 <= angle[i,j] < 112.5: # vertikaluri
                    neighbors = [magnitude[i-1,j], magnitude[i+1,j]]
                else: # meore diagonali
                    neighbors = [magnitude[i-1,j+1], magnitude[i+1,j-1]]
                
                if magnitude[i,j] >= max(neighbors):
                    suppressed[i,j] = magnitude[i,j]
                    
        return suppressed
    
    def hysteresis_thresholding(img, low, high):
        strong = (img > high)
        weak = (img > low) & (img <= high)
        output = np.zeros_like(img)
        output[strong] = 1
        
        # low-s da high-s shoris rac moxvda magati mezoblebis gachekva
        height, width = img.shape
        for i in range(1, height-1):
            for j in range(1, width-1):
                if weak[i,j]:
                    if np.any(output[i-1:i+2, j-1:j+2] == 1):
                        output[i,j] = 1
                        
        return output
    
    # Gausiani
    kernel_size = int(6*sigma + 1) if int(6*sigma + 1) % 2 == 1 else int(6*sigma + 2)
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed = np.zeros_like(image, dtype=float)
    
    padded = np.pad(image, ((kernel_size//2, kernel_size//2), (kernel_size//2, kernel_size//2)), mode='edge')
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            smoothed[i,j] = np.sum(window * kernel)

    # https://en.wikipedia.org/wiki/Sobel_operator
    # sobeli
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    gradient_x = np.zeros_like(smoothed)
    gradient_y = np.zeros_like(smoothed)
    
    padded = np.pad(smoothed, ((1, 1), (1, 1)), mode='edge')
    
    for i in range(smoothed.shape[0]):
        for j in range(smoothed.shape[1]):
            window = padded[i:i+3, j:j+3]
            gradient_x[i,j] = np.sum(window * sobel_x)
            gradient_y[i,j] = np.sum(window * sobel_y)
    
    # https://en.wikipedia.org/wiki/Sobel_operator
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.atan2(gradient_y, gradient_x) # aq romeli jobia ar vici (atan2 tu arctan2)
    
    suppressed = non_maximum_suppression(gradient_magnitude, gradient_direction)
    
    edges = hysteresis_thresholding(suppressed, low_threshold, high_threshold)
    
    return edges

def get_target_coords(path, low_threshold=30, high_threshold=90, sigma=1.0, eps=30, min_points=5):

    def calculate_center(cluster):
        return np.mean(cluster, axis=0)

    frame = cv2.imread(path)
    #frame = cv2.resize(frame, (200, 200))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = canny_edge_detection(gray, low_threshold, high_threshold, sigma)

    edge_pixels = np.column_stack(np.where(edges > 0))

    clusters, _ = dbscan(edge_pixels, eps, min_points)

    centers = []
    for cluster in clusters:
        center = calculate_center(cluster)
        x = center[1]
        y = frame.shape[0] - center[0]
        centers.append([x, y])

    centers.sort(key=lambda c: c[0])

    return centers



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np


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
    path = 'test4.png'

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