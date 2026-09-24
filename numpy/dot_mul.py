import numpy as np

u = np.array([2, 3, 4])
v = np.array([1, 2, 3])

# Dot function
print(np.dot(u, v))

# Dot symbol
print(u @ v)

# Multiplication 
A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

print(A @ B)

A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

print(A * B)


x = np.array([2, 3])
w = np.array([4, 5])
print(x @ w)

print(x.shape == (100, 5))
