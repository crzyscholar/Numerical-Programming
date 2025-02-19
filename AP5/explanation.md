
# Solving a System of ODEs: Predator-Prey-Scavenger Model

---

## **Problem Formulation**

I selected a **predator-prey-scavenger model** with an additional environmental factor:

1. **Prey (\(x\))**: Grows logistically but decreases due to predators.
2. **Predator (\(y\))**: Grows by consuming prey, decreases naturally and due to scavengers.
3. **Scavenger (\(z\))**: Feeds on predators but decreases naturally.
4. **Pollution (\(p\))**: Accumulates due to populations and dissipates over time.

The system of ODEs is:

$$
\begin{aligned}
    \frac{dx}{dt} &= a x \left(1 - \frac{x}{K}\right) - b x y \\
    \frac{dy}{dt} &= c x y - d y - e y z \\
    \frac{dz}{dt} &= f y z - g z \\
    \frac{dp}{dt} &= h x + i y + j z - k p
\end{aligned}
$$

### **Parameters**
- \(a, K, b, c, d, e, f, g, h, i, j, k\): Interaction coefficients, growth rates, and decay constants.
- Example values: \(a = 0.5, K = 100, b = 0.02, c = 0.01, d = 0.1, e = 0.01, f = 0.02, g = 0.1, h = 0.05, i = 0.02, j = 0.03, k = 0.1\).

### **Initial Conditions**
- \(x_0 = 50\), \(y_0 = 5\), \(z_0 = 2\), \(p_0 = 0\).

---

## **Numerical Method Selection**

**Chosen Method**: 4th-Order Runge-Kutta (RK4)

### **Justification**
1. **Accuracy**: RK4 provides reliable approximations for nonlinear ODEs.
2. **Stability**: Handles oscillatory dynamics effectively.
3. **Efficiency**: Balances computational cost and precision better than simpler methods.

---

## **Numerical Experiments**

### **Experiment 1: Stable Ecosystem**
- **Initial Conditions**: \(x_0 = 50, y_0 = 5, z_0 = 2, p_0 = 0\).
- **Parameters**: Moderate predator-prey interaction, fast pollution decay (\(k = 0.1\)).

**Observations**:
- Prey stabilizes near the carrying capacity (\(K\)).
- Predator and scavenger populations oscillate but reach equilibrium.
- Pollution dissipates over time.

---

### **Experiment 2: Unstable Ecosystem**
- **Initial Conditions**: \(x_0 = 20, y_0 = 15, z_0 = 5, p_0 = 10\).
- **Parameters**: High predator-prey interaction, slow pollution decay (\(k = 0.01\)).

**Observations**:
- Predators grow quickly, depleting prey.
- Scavenger population collapses due to predator scarcity.
- Pollution remains high, destabilizing the system.

---

## **Visualization**

Key results:
- Prey (\(x\)) approaches a steady state near \(K\) unless overwhelmed by predators.
- Predator (\(y\)) and scavenger (\(z\)) populations oscillate but stabilize under suitable conditions.
- Pollution (\(p\)) decays or destabilizes the system depending on parameters.

---

## **Conclusions**

- RK4 is effective for solving nonlinear systems like predator-prey models due to its stability and accuracy.
- Environmental factors like pollution significantly affect population dynamics.
- Numerical experiments highlight the importance of parameter selection for ecosystem stability.
