import numpy as np

x = np.array([5, 10, 15, 20, 25, 30])

# Slicing
print(x[1:4])
print(x[2:])

# Filtering
print(x[x > 15])

X = np.array([
    [10, 100],
    [20, 200],
    [30, 300],
    [40, 400]
])

print(X[:, 0])

print(X[X[:, 0] > 20])