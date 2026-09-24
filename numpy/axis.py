import numpy as np

X = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])
# Shape
print(X.shape)

# Mean across row
print(X.mean(axis=0))

# Mean across column
print(X.mean(axis=1))


