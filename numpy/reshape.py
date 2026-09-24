import numpy as np 

x = np.array([1, 2, 3, 4, 5, 6])
print(x)
print(x.reshape(1, 6))

x = np.array([10, 20, 30, 40, 50, 60])
print(x.reshape(3, 2))

a = np.array([
    1, 2, 3
])

print(a.reshape(1, 3))