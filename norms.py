import numpy as np

def first_norm(vector):
    return np.sum(np.abs(vector))

def second_norm(vector):
    return np.sqrt(np.sum(np.square(vector)))

def infinity_norm(vector):
    return np.max(np.abs(vector))

# Example usage:
vector = np.array([3, -4, 2])

print(f"First norm (L1): {first_norm(vector)}")
print(f"Second norm (L2): {second_norm(vector)}")
print(f"Infinity norm (L*infinity*): {infinity_norm(vector)}")

def p_norm(vector, p):
    return np.sum(np.abs(vector) ** p) ** (1 / p)

p = 3  # You can change p to any positive value
print(f"P-norm (p={p}): {p_norm(vector, p)}")

