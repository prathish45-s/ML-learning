import numpy as np
import pandas as pd

data = {
    "Name": ["Arun", "Bala", "Charan", "Divya", "Esha"],
    "Age": [20, 22, 21, 23, 25],
    "Marks": [75, 82, 68, 90, 88]
}

df = pd.DataFrame(data)

df.info()

print(df.describe())

data = {
    "Name": ["Arun", "Bala", "Charan", "Divya", "Esha"],
    "Age": [20, 22, None, 23, 25],
    "Marks": [75, None, 68, 90, 88]
}

df = pd.DataFrame(data)

# To find Null values
print(df.isnull())

# Sum of missing values
print(df.isnull().sum())

# One_hot_encoding

