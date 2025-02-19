### Problem Formulation
#### task description

develop a simulation which throws a ball at the objects of different size and shape in the given picture.

- Input: Image of randomly scattered polygons, circles, ellipses each with and without holes. 

- Task: Throw a ball and hit targets on the image one after another

- Output: Animation corresponding to the task description


### Numerical Methods

####  Shooting Method
In numerical analysis, the **shooting method** is a method for solving a boundary value problem by reducing it to an initial value problem. It involves finding solutions to the initial value problem for different initial conditions until one finds the solution that also satisfies the boundary conditions of the boundary value problem. 

- Code uses Newton's method for finding correct initial velocity and angle
- Adjusts parameters to hit target's coordinates
- Convergence criterion is *error* < 0.1

in our case, 
- starting point is (0,0)
- target point is (x<sub>target</sub>, y<sub>target</sub>)
- we need to find velocities ($v_x$, $v_y$) that will allow us to hit the target

#### Stability
In my project, I utilized the RK4 method and the implicit Euler method to solve ordinary differential equations (ODEs). While RK4 is not A-stable, the implicit Euler method is. However, in the scenarios considered, both methods yield comparable results.

**A-stability** refers to a numerical method's ability to handle stiff equations without the solution becoming unstable, regardless of the step size. A method is A-stable if its stability region covers the entire left half of the complex plane, meaning it remains stable for all eigenvalues with negative real parts.

- **RK4 Method:** A fourth-order method with a local truncation error of O(h5)O(h5) and a global error of O(h4)O(h4). Although RK4 is highly accurate, it is **not A-stable** because its stability region does not cover the entire left half-plane. This makes it less suitable for stiff problems at large step sizes.
- **Implicit Euler Method:** A first-order method with a local truncation error of O(h2)O(h2). It **is A-stable** because its stability region includes the entire left half-plane, making it stable for stiff equations even with large step sizes.


### Algorithm
1. make initial guess for launch velocities ($v_x$, $v_y$)
	some initial velocities for x and y axis are given. I took (10,10).
	
2. calculate trajectory using RK4/Implicit Euler
	use the chosen numerical method (in my case RK4 and Implicit Euler) to calculate the ball's trajectory based on the initial guess of the velocities (solve the equation of motion).
	
3. compare final position with target's position
	after simulating the trajectory, compare the final and target position. the error is how far we are from the correct guess.
	
4. adjust initial velocities using Newton's method
	Apply Newton's method to refine the initial velocity estimates in order to minimize the error. This involves approximating the Jacobian matrix, which describes how the final position varies with respect to the initial velocities. The Jacobian is computed numerically by using a small perturbation. The Jacobian matrix is approximated numerically using the backwards differentiation formula.
5. repeat until convergence 
	Keep adjusting the initial velocity estimates until the error falls below a specified tolerance. Once the error is sufficiently small, the method is considered to have converged, and the current estimates represent the optimal initial velocities.
- non-convergence:
	If the method fails to converge within the maximum number of allowed iterations, it will report a failure. This could suggest that the initial guess was too inaccurate or that the tolerance level was set too low.

### Numerical Experiments
I compared implicit Euler's method and RK. the results are almost the same(dt=0.001), but time it takes to calculate differs.

![[Screenshot 2025-02-03 at 16.25.00.png]]

![[Screenshot 2025-02-03 at 16.26.53.png]]
### parameters
- Air Drag $k = 0.001$
- Air Drag $k = 0.001$
- Time step: $dt = 0.001s$
- Maximum simulation time: $4s$
- Shooting method tolerance: $0.1m$
- Maximum iterations: $100$


## Test
I took the following image to test the implementation:

![[test4.png]]

the output:

![[Screenshot 2025-02-03 at 16.36.11.png]]

![[Screenshot 2025-02-03 at 16.37.05.png]]











### Sources:
- https://en.wikipedia.org/wiki/Shooting_method
- https://en.wikipedia.org/wiki/Runge–Kutta_methods
- https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions
- https://en.wikipedia.org/wiki/Newton%27s_method#Multidimensional_formulations
- https://en.wikipedia.org/wiki/Backward_Euler_method