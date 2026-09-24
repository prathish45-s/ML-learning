import pandas as pd

data = {
    "Name": ["Appa", "Amma", "Prathish", "Darshana"],
    "Age": [45, 40, 23, 20],
    "Marks": [100, 90, 80, 70]
}

df = pd.DataFrame(data)
print(df)

# Head
print(df.head(2))  

# Shape
print(df.shape)

# Specific Column
print(df["Age"])

# Multiple columns
print(df[["Age", "Name"]])