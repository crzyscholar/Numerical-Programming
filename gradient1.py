# implement the specified gradient descent technique in code. your code should calculate at least three iterations, showing all steps. print the values of the parameters and fcuntions output after each iteration.
# mini-batch gradient descent (batch size = 2)
# given the following dataset with points:
# (x1,x2,x3,x4) = (1, -1, 0.5, 2), (0.5, -1.5, 1, 0.5), (-1, 1, -0.5, 1.5)
# and functions:
# f(x1,x2,x3,x4) = x1^2 + 2*x2^2 + 3*x3^2 + 4*x4^2 - x1*x2 + 2*x2*x3 + -3*x3*x4 + 5
# 1.implement **mini-batch gradient descent** with a batch size of 2. 
# 2. start at (1, -1, 0.5, 2) with a learning rate miu = 0.02.
# 3. for each iteration, select a new batch of two points and update the parameters. print parameter values after each update.

import numpy as np

# Define the function f and its gradient calculation
def f(x1, x2, x3, x4):
    return (
        x1**2 + 2*x2**2 + 3*x3**2 + 4*x4**2
        - x1*x2 + 2*x2*x3 - 3*x3*x4 + 5
    )

def gradient_f(x1, x2, x3, x4):
    df_dx1 = 2 * x1 - x2
    df_dx2 = 4 * x2 - x1 + 2 * x3
    df_dx3 = 6 * x3 + 2 * x2 - 3 * x4
    df_dx4 = 8 * x4 - 3 * x3
    return np.array([df_dx1, df_dx2, df_dx3, df_dx4])

# Dataset points
dataset = [
    [1, -1, 0.5, 2],
    [0.5, -1.5, 1, 0.5],
    [-1, 1, -0.5, 1.5]
]

# Parameters and settings
learning_rate = 0.02
iterations = 3
batch_size = 2
parameters = np.array([1, -1, 0.5, 2])

# Mini-batch gradient descent
for iteration in range(iterations):
    # Select a mini-batch of points
    batch_indices = np.random.choice(len(dataset), batch_size, replace=False)
    batch = [dataset[i] for i in batch_indices]
    
    # Compute the gradient over the batch
    batch_gradients = []
    for point in batch:
        gradient = gradient_f(*point)
        batch_gradients.append(gradient)
    batch_gradient = np.mean(batch_gradients, axis=0)
    
    # Update parameters
    parameters -= learning_rate * batch_gradient
    
    # Print iteration results
    print(f"Iteration {iteration + 1}:")
    print(f"Selected batch points: {batch}")
    print(f"Batch gradient: {batch_gradient}")
    print(f"Updated parameters: {parameters}")
    print(f"Function output: {f(*parameters)}\n")
