import numpy as np

dataset = [[1, -1, 0.5, 2], [0.5, -1.5, 1, 0.5], [-1, 1, -0.5, 1.5]]

parameters = np.array([1, -1, 0.5, 2])
learning_rate = 0.02

def f(x1, x2, x3, x4): 
    return (x1**2 + 2*x2**2 + 3*x3**2 + 4*x4**2 - x1*x2 + 2*x2*x3 - 3*x3*x4 + 5)

def g(x1, x2, x3, x4): 
    return [2*x1 - x2, 4*x2 - x1 + 2*x3, 6*x3 + 2*x2 - 3*x4, 8*x4 - 3*x3]

for i in range(3):
    batch = [dataset[j] for j in np.random.choice(len(dataset), 2, False)]
    batch_gradients = [g(p[0], p[1], p[2], p[3]) for p in batch]
    parameters -= learning_rate * np.mean(batch_gradients, 0)
    print("iteration", i + 1, ":\nselected batch points:", batch, "\nbatch gradient:", np.mean(batch_gradients, 0), "\nupdated parameters:", parameters, "\nfunction output:", f(parameters[0], parameters[1], parameters[2], parameters[3]), "\n")
