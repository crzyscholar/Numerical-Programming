### Explanation of the Algorithm


### **videos**
video2.mp4 works fine. video.mp4 doesn't. in video.mp4 the ball
moves too fast which is one of the reasons it doesn't work(correctly).


#### **Formulation of the Algorithm**
1. **ODE Definition**:
   - The algorithm models the motion of a ball under the influence of gravity and air drag using the ODE:
     \[
     \frac{dx}{dt} = v_x, \quad \frac{dy}{dt} = v_y
     \]
     \[
     \frac{dv_x}{dt} = -\frac{k}{m} v_x v, \quad \frac{dv_y}{dt} = -g -\frac{k}{m} v_y v
     \]
     where \(v = \sqrt{v_x^2 + v_y^2}\).

2. **Video Tracking**:
   - The positions of the ball in each frame are detected using color segmentation in the HSV color space.
   - The ball's centroid is computed from the largest contour in the detected mask.

3. **Optimization**:
   - The algorithm estimates parameters (\(k, m, v_{x0}, v_{y0}\)) by minimizing the error between the observed and simulated trajectories.
   - The least squares error is computed as the Euclidean distance between observed and simulated positions.

4. **Numerical Solver**:
   - `solve_ivp` (Runge-Kutta method) is used to solve the ODE for given parameters.

---

#### **Properties of Numerical Methods**
- **Runge-Kutta Method (RK45)**:
  - Adaptive step size ensures accurate solutions for complex dynamics.
  - Handles stiff systems better than simple Euler methods.
  - Computationally efficient for small- to medium-scale problems.

- **Minimization Using `scipy.optimize.minimize`**:
  - Bounds constrain parameters to realistic ranges.
  - Relies on gradient-free methods for robust optimization.
