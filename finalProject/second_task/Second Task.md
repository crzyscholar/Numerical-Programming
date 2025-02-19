### Problem Formulation
#### task description

develop a simulation in which a ball is thrown at a moving ball in the provided video intercepting it.

- Input: part of a video of a moving ball

- Task: Throw a ball and intercept moving ball

- Output: Animation corresponding to the task description


### Numerical Methods

#### Shooting Method
In numerical analysis, the **shooting method** is a method for solving a boundary value problem by reducing it to an initial value problem. It involves finding solutions to the initial value problem for different initial conditions until one finds the solution that also satisfies the boundary conditions of the boundary value problem. 

- Code uses Newton's method for finding correct initial velocity and angle
- Adjusts parameters to hit target's coordinates
- Convergence criterion is *error* < 0.1


#### Video Processing 
1. Extract video frames, convert them to grayscale, and apply Gaussian blur to             reduce noise.
1. Apply Canny edge detection and contour detection to track the ball's position.




#### **Velocity Calculation**

Central difference method for velocity estimation:

$v_x = \frac{x_{i+1} - x_{i-1}}{2\Delta t}$

$v_y = \frac{y_{i+1} - y_{i-1}}{2\Delta t}$


#### **Parameter Estimation**

- Uses consecutive velocity measurements to estimate $k/m$ and $g$

- Applies averaging to reduce measurement noise

- Accounts for frame rate in temporal calculations


#### Stability
In my project, I utilized the RK4 method and the implicit Euler method to solve ordinary differential equations (ODEs). While RK4 is not A-stable, the implicit Euler method is. However, in the scenarios considered, both methods yield comparable results.

**A-stability** refers to a numerical method's ability to handle stiff equations without the solution becoming unstable, regardless of the step size. A method is A-stable if its stability region covers the entire left half of the complex plane, meaning it remains stable for all eigenvalues with negative real parts.

- **RK4 Method:** A fourth-order method with a local truncation error of O(h5)O(h5) and a global error of O(h4)O(h4). Although RK4 is highly accurate, it is **not A-stable** because its stability region does not cover the entire left half-plane. This makes it less suitable for stiff problems at large step sizes.
- **Implicit Euler Method:** A first-order method with a local truncation error of O(h2)O(h2). It **is A-stable** because its stability region includes the entire left half-plane, making it stable for stiff equations even with large step sizes.


### Algorithm
2. process video
	- extract frames, convert them to grayscale and apply gaussian blur to reduce noise
	- use canny for edge detection
	- track the position of the ball with contour detection
3. **Velocity Calculation**

	Central difference method for velocity estimation:

	$v_x = \frac{x_{i+1} - x_{i-1}}{2\Delta t}$

	$v_y = \frac{y_{i+1} - y_{i-1}}{2\Delta t}$


4. make initial guess for launch velocities ($v_x$, $v_y$)
	some initial velocities for x and y axis are given. I took (10,10).
	
5. calculate trajectory using RK4/Implicit Euler
	use the chosen numerical method (in my case RK4 and Implicit Euler) to calculate the ball's trajectory based on the initial guess of the velocities (solve the equation of motion).
	
6. compare final position with target's position
	after simulating the trajectory, compare the final and target position. the error is how far we are from the correct guess.
	
7. adjust initial velocities using Newton's method
	Apply Newton's method to refine the initial velocity estimates in order to minimize the error. This involves approximating the Jacobian matrix, which describes how the final position varies with respect to the initial velocities. The Jacobian is computed numerically by using a small perturbation. The Jacobian matrix is approximated numerically using the backwards differentiation formula.
	
$$J = \begin{pmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{pmatrix}$$

	The shooting method aims to determine the initial velocity components (v_$x_0$,v_$y_0$) required for the projectile to reach the target located at (x<sub>target</sub>, y<sub>target</sub>). This involves solving the following system of equations:
	$$
F(v_{x_0}, v_{y_0}) = 
\begin{pmatrix}
x_{\text{final}}(v_{x_0}, v_{y_0}) - x_{\text{target}} \\
y_{\text{final}}(v_{x_0}, v_{y_0}) - y_{\text{target}}
\end{pmatrix}
$$

8. repeat until convergence 
	Keep adjusting the initial velocity estimates until the error falls below a specified tolerance. Once the error is sufficiently small, the method is considered to have converged, and the current estimates represent the optimal initial velocities.
- non-convergence:
	If the method fails to converge within the maximum number of allowed iterations, it will report a failure. This could suggest that the initial guess was too inaccurate or that the tolerance level was set too low.

### Numerical Experiments
I compared implicit Euler's method and RK. the results are almost the same(dt=0.001), but time it takes to calculate differs.



### parameters
- Air Drag $k = 0.001$
- Air Drag $k = 0.001$
- Time step: $dt = 0.001s$
- Maximum simulation time: $4s$
- Shooting method tolerance: $0.1m$
- Maximum iterations: $100$


## Test Results

![[Screenshot 2025-02-03 at 20.02.57.png]]


### Numerical Expreiments
I compared implicit Euler's method and RK. the results are almost the same(dt=0.001), but time it takes to calculate differs.

![[Screenshot 2025-02-03 at 19.56.13.png]]

![[Screenshot 2025-02-03 at 19.58.24.png]]



### Sources:
- https://en.wikipedia.org/wiki/Shooting_method
- https://en.wikipedia.org/wiki/Runge–Kutta_methods
- https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions
- https://en.wikipedia.org/wiki/Newton%27s_method#Multidimensional_formulations
- https://en.wikipedia.org/wiki/Backward_Euler_method