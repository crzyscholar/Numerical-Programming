# Solving a System of ODEs with Linear Multistep Method

## Problem Statement
### 1. System of ODEs (Predator-Prey Model - Lotka-Volterra Equations):
- Prey: \( \frac{dy_1}{dt} = a y_1 - b y_1 y_2 \)
- Predator: \( \frac{dy_2}{dt} = -c y_2 + d y_1 y_2 \)

where \( a, b, c, d > 0 \).

## 2. Numerical Method
We use **Adams-Bashforth 2-Step Method** (a Linear Multistep Method):

\[
y_{n+1} = y_n + h \left( \frac{3}{2} f(t_n, y_n) - \frac{1}{2} f(t_{n-1}, y_{n-1}) \righ)

## 3. Numerical Experiments and Visualization
The Adams-Bashforth 2-Step method successfully models the oscillatory dynamics of the predator-prey system.

- **Prey Population** \( y_1 \): Increases when predators are fewer.
- **Predator Population** \( y_2 \): Grows as prey population increases.

## 4. Explanation
- System of ODEs: Lotka-Volterra Model
- Numerical Method: Adams-Bashforth 2-Step (Explicit Linear Multistep)
- Step 1: Use Euler to bootstrap initial value.
- Step 2: Iteratively compute using Adams-Bashforth formula.
- Visualization: Shows oscillatory predator-prey behavior.
